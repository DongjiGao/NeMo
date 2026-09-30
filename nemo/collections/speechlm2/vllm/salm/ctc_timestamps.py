# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep CTC timestamp inputs per request inside vLLM and align finished transcripts.

The encoder returns ``CTCTimestampInputs`` (ASR states, Sortformer speaker
probabilities, 10 ms diarization labels) from the same forward that feeds the LLM,
and runs the CTC head only when a finished transcript is aligned, in
``ParallelExpertEncoder.generate_ctc_timestamps``. This module holds those inputs
between the two points. vLLM makes that awkward in two ways:

* ``embed_multimodal(**mm_kwargs_batch)`` receives only tensors, so the inputs have
  no request id or ``mm_hash`` to be keyed by when they are produced.
* The scheduler only runs the encoder for audio whose ``mm_hash`` is absent from the
  encoder cache, so a request served from that cache produces no inputs of its own.

Both are solved by keying on ``mm_hash``, the identity vLLM already uses. Inputs are
stored under positional placeholders during the forward and renamed when the runner
announces each item's hash, and every new request is mapped to its hashes, so a cache
hit finds the inputs its audio produced earlier. The hook is ``_execute_mm_encoder``,
around one call to ``_batch_mm_inputs_from_scheduler``, whose signatures match in vLLM
0.23 and 0.28.

The inputs move to pinned host memory without blocking the engine. An ASR state is
1,280 values per 80 ms frame, far less than the vocabulary-wide CTC output, so the
copy overlaps with compute instead of stalling it. Once a copy lands, each row is
compacted to its own valid frames, so retaining it does not keep its whole padded
batch alive.

vLLM's own encoder cache drops a request's claim once prefill has consumed the
embeddings; a capture is read later, after the request has finished, by a caller vLLM
does not know about. So each capture records the requests that own it (repeated audio
shares one through the encoder cache). A capture is deleted once the last of them is
aligned with ``release=True`` or released through :func:`ctc_release`, and vLLM has
evicted its audio from the encoder cache: until then a new request with that audio is
served from the cache and captures nothing, so it needs the old capture. A request can
also opt out of capture with ``mm_processor_kwargs={"capture_ctc_timestamps": False}``,
which the server's prompt hook sets for responses it never aligns. Captures nobody
releases are bounded only by ``NEMO_CTC_TIMESTAMP_RETAIN_GB`` of host memory
(default 8), least recently used first.

Alignment runs in the engine process, where the inputs live. Offline callers use
:func:`ctc_timestamps` (or :func:`ctc_word_timestamps` for words only); a server
reaches the same code through
``collective_rpc`` by the names in ``WORKER_ALIGN_METHOD``,
``WORKER_ALIGN_BATCH_METHOD`` and ``WORKER_RELEASE_METHOD``, passing the external
request id, which vLLM's scheduler knows with a random suffix appended.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import torch

from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

# Placeholder ids the inputs are stored under before the real hash is known.
_PENDING_PREFIX = "__nemo_ctc_pending_"

# Guards against captures nobody aligns or releases; an hour of audio keeps on the
# order of 0.1-0.2 GB of encoder states.
_RETENTION_GB_ENV = "NEMO_CTC_TIMESTAMP_RETAIN_GB"
_DEFAULT_RETENTION_GB = 8.0

# Requests nobody aligns or releases would otherwise stay mapped forever.
_MAX_TRACKED_REQUESTS = 1 << 16

# Worker method names reached through collective_rpc: one request (a server's
# transcription hooks), many (offline LLM callers), or releasing without aligning.
WORKER_ALIGN_METHOD = "nemo_ctc_align_request"
WORKER_ALIGN_BATCH_METHOD = "nemo_ctc_align_requests"
WORKER_RELEASE_METHOD = "nemo_ctc_release_requests"

# vLLM forms the scheduler's request id as f"{external_id}-{random_uuid():.8}".
_INTERNAL_ID_SUFFIX_LEN = 8

# Requests decoded together in one deferred-head call, which materializes a
# (batch, frames, vocabulary) log-prob tensor on the device.
_ALIGN_BATCH = 16

# Weight of the Sortformer speaker-activity prior in CTC alignment, overridable per
# checkpoint as ctc_timestamps.speaker_logprob_weight. The aligner's own default is
# 0.0, which lets words in overlapped speech drift out of their speaker's turns.
DEFAULT_SPEAKER_PRIOR_WEIGHT = 0.25

# CTCTimestampInputs tensor fields: the time axis to trim and pad along, and the
# lengths field marking its valid frames (None for the lengths fields themselves).
_FIELDS = (
    ("asr_encoded", 2, "asr_encoded_lengths"),
    ("asr_encoded_lengths", None, None),
    ("sortformer_sigmoids", 1, "sortformer_lengths"),
    ("sortformer_lengths", None, None),
    ("diarization_labels", 2, "diarization_lengths"),
    ("diarization_lengths", None, None),
)

# Placeholder bookkeeping stays thread-local: it is produced and consumed within a
# single _execute_mm_encoder call on the engine thread.
_state = threading.local()

# Process-wide, unlike _state, and guarded by _lock. With an in-process engine the
# alignment call runs on the caller's thread while the encoder runs on the engine
# thread, so anything shared between them must not be thread-local.
_registry: dict[str, Any] = {}
_store: dict[str, dict] = {}
_request_hashes: dict[str, list[str]] = {}
# mm_hash -> the requests that still own its capture.
_hash_owners: dict[str, set[str]] = {}
# Hashes whose audio vLLM's encoder cache holds. A new request with that audio is
# served from the cache and captures nothing, so it needs the existing capture.
_engine_cached: set[str] = set()
# External request id -> the internal id the scheduler reported for it.
_external_ids: dict[str, str] = {}
# Stored rows that still view their forward's padded host buffers, oldest first.
_uncompacted: deque[dict] = deque()
# Reentrant because the alignment lookup resolves request ids while holding it.
_lock = threading.RLock()


def register_encoder(encoder: Any) -> None:
    """Publish the live encoder, which also turns input capture on.

    The transcription hooks are classmethods on the vLLM model interface and never
    receive the model instance. One engine hosts one model per process, so a
    module-level reference is enough to bridge that.

    Args:
        encoder (Any): The perception encoder; it holds the loaded aligner.
    """
    _registry["encoder"] = encoder


def active_encoder() -> Any:
    """Return the live encoder, or ``None`` when timestamps are not enabled."""
    return _registry.get("encoder")


def require_v1_model_runner(vllm_config: Any) -> None:
    """Refuse CTC timestamps on vLLM's Model Runner V2.

    Captured inputs are renamed, trimmed and compacted by a hook on the V1 runner's
    ``GPUModelRunner._execute_mm_encoder``. Under V2, the default from vLLM 0.30 on,
    that hook never runs: nothing would be aligned, and the inputs would pile up in
    host memory.

    Args:
        vllm_config (Any): The engine's ``VllmConfig``; without ``use_v2_model_runner``
            (older vLLM) the V1 runner is assumed.
    """
    if getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError(
            "CTC timestamps need vLLM's V1 GPU model runner, but this engine uses Model Runner V2; "
            "set VLLM_USE_V2_MODEL_RUNNER=0."
        )


def read_speaker_prior_weight(ctc_config: Any) -> float:
    """Read the Sortformer prior weight from a checkpoint's ``ctc_timestamps`` block.

    The config key is ``speaker_logprob_weight``, after the aligner's constructor
    parameter. The aligner validates it only in that constructor, and the plugin sets
    it on an aligner that is already built, so it is validated here instead.

    Args:
        ctc_config (Any): The ``ctc_timestamps`` block, as a dict or an attribute object.

    Returns:
        float: The configured weight, or ``DEFAULT_SPEAKER_PRIOR_WEIGHT`` when unset.
    """
    value = (
        ctc_config.get("speaker_logprob_weight")
        if isinstance(ctc_config, dict)
        else getattr(ctc_config, "speaker_logprob_weight", None)
    )
    speaker_prior_weight = float(DEFAULT_SPEAKER_PRIOR_WEIGHT if value is None else value)
    if speaker_prior_weight < 0:
        raise ValueError(f"ctc_timestamps.speaker_logprob_weight must be non-negative; got {speaker_prior_weight}.")
    return speaker_prior_weight


def pending_row_ids(count: int) -> list[str]:
    """Return placeholder ids for the items of one encoder forward.

    Args:
        count (int): Number of multimodal items in the forward.

    Returns:
        list[str]: Placeholder ids, in item order.
    """
    start = getattr(_state, "pending_next", 0)
    ids = [f"{_PENDING_PREFIX}{start + i}" for i in range(count)]
    _state.pending_next = start + count
    _state.pending_queue = getattr(_state, "pending_queue", []) + ids
    return ids


def store_timestamp_inputs(
    row_ids: Sequence[str | None], inputs: Any, audio_durations: torch.Tensor | Sequence[float]
) -> None:
    """Keep one forward's ``CTCTimestampInputs`` per item, in host memory.

    Args:
        row_ids (Sequence[str | None]): One placeholder id per batch row, from
            :func:`pending_row_ids`; ``None`` skips the row.
        inputs (CTCTimestampInputs): The encoder's timestamp inputs for the batch.
        audio_durations (torch.Tensor | Sequence[float]): Audio duration per row, in
            seconds. A device tensor is copied along with the inputs, so the forward
            never waits to read it.
    """
    if not isinstance(audio_durations, torch.Tensor):
        audio_durations = torch.tensor(audio_durations, dtype=torch.float64)
    host = {name: _to_host(getattr(inputs, name)) for name, _, _ in _FIELDS}
    durations = _to_host(audio_durations)
    on_device = audio_durations.is_cuda or any(
        getattr(inputs, name) is not None and getattr(inputs, name).is_cuda for name, _, _ in _FIELDS
    )
    # Rows are read only after this event, so the copies never block the forward.
    ready = torch.cuda.current_stream().record_event() if on_device else None
    with _lock:
        for row, row_id in enumerate(row_ids):
            if row_id is None:
                continue
            entry = {name: None if tensor is None else tensor[row : row + 1] for name, tensor in host.items()}
            entry.update(
                diarization_frame_seconds=inputs.diarization_frame_seconds,
                duration=durations[row : row + 1],
                ready=ready,
            )
            entry["nbytes"] = _entry_nbytes(entry)
            _store[row_id] = entry
            _uncompacted.append(entry)


def _to_host(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy a tensor to host memory; pinned and non-blocking when it is on the GPU."""
    if tensor is None:
        return None
    if not tensor.is_cuda:
        return tensor.detach().clone()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor.detach(), non_blocking=True)
    return host


def _entry_nbytes(entry: dict) -> int:
    """Host bytes of an entry's tensors, each counted at its own (row) size."""
    return sum(entry[name].numel() * entry[name].element_size() for name, _, _ in _FIELDS if entry[name] is not None)


def _resolve_request_id(request_id: str) -> str | None:
    """Return the recorded key for a request id, accepting the external form.

    Callers holding a ``RequestOutput`` see the external id, while the scheduler
    reports the internal one; exact matches win so ids without a suffix (older vLLM,
    in-process callers) resolve unchanged. Must hold ``_lock``.
    """
    if request_id in _request_hashes:
        return request_id
    internal = _external_ids.get(request_id)
    return internal if internal in _request_hashes else None


def _external_id(internal_id: str) -> str | None:
    """Strip vLLM's ``-{8 chars}`` request id suffix, or ``None`` when there is none."""
    cut = len(internal_id) - _INTERNAL_ID_SUFFIX_LEN - 1
    return internal_id[:cut] if cut > 0 and internal_id[cut] == "-" else None


def mm_hashes_for_request(request_id: str) -> list[str]:
    """Return the multimodal hashes seen for a request, in arrival order.

    Args:
        request_id (str): vLLM request id, internal or external.

    Returns:
        list[str]: Hashes whose inputs belong to this request.
    """
    with _lock:
        key = _resolve_request_id(request_id)
        return list(_request_hashes[key]) if key is not None else []


def _record_request_hashes(pairs) -> None:
    """Associate a request with the hashes whose inputs belong to it, and make it an owner."""
    with _lock:
        for req_id, mm_hash in pairs:
            hashes = _request_hashes.setdefault(req_id, [])
            if mm_hash not in hashes:
                hashes.append(mm_hash)
            _hash_owners.setdefault(mm_hash, set()).add(req_id)
            external = _external_id(req_id)
            if external is not None:
                _external_ids[external] = req_id
        while len(_request_hashes) > _MAX_TRACKED_REQUESTS:
            _forget_request(next(iter(_request_hashes)))


def _drop_capture(mm_hash: str) -> None:
    """Delete a capture and tell compaction to skip it. Must hold ``_lock``."""
    entry = _store.pop(mm_hash, None)
    if entry is not None:
        entry["dropped"] = True


def _forget_request(req_id: str) -> None:
    """Drop a request's claims and delete the captures nobody needs anymore. Must hold ``_lock``."""
    for mm_hash in _request_hashes.pop(req_id, ()):
        owners = _hash_owners.get(mm_hash)
        if owners is None:
            continue
        owners.discard(req_id)
        if not owners:
            del _hash_owners[mm_hash]
            if mm_hash not in _engine_cached:
                _drop_capture(mm_hash)
    external = _external_id(req_id)
    if external is not None and _external_ids.get(external) == req_id:
        del _external_ids[external]


def _follow_engine_cache(freed: Sequence[str] = (), encoded: Sequence[str] = ()) -> None:
    """Track which audio vLLM's encoder cache holds, deleting unowned captures it evicted."""
    with _lock:
        for mm_hash in freed:
            _engine_cached.discard(mm_hash)
            if mm_hash not in _hash_owners:
                _drop_capture(mm_hash)
        _engine_cached.update(encoded)


def _release_requests(request_ids: Sequence[str]) -> None:
    """Drop these requests' claims on their captures, accepting external ids."""
    with _lock:
        for request_id in request_ids:
            key = _resolve_request_id(request_id)
            if key is not None:
                _forget_request(key)


def _byte_limit() -> int:
    """How many bytes of stored inputs to hold before evicting the least recently used."""
    try:
        gigabytes = float(os.environ.get(_RETENTION_GB_ENV, _DEFAULT_RETENTION_GB))
    except ValueError:
        gigabytes = _DEFAULT_RETENTION_GB
    return int(max(gigabytes, 0.0) * 1e9)


def _rename_pending(mm_hashes: list[str]) -> int:
    """Rename this forward's placeholder entries to their real hashes."""
    queue = getattr(_state, "pending_queue", None) or []
    if len(queue) != len(mm_hashes):
        # Pairing is positional, so unequal counts would attach inputs to another
        # item's audio; no timestamps are better than wrong ones.
        logging.warning(
            "[NeMoSpeechLM] The encoder captured CTC timestamp inputs for %d items but vLLM scheduled %d; "
            "discarding them.",
            len(queue),
            len(mm_hashes),
        )
        _discard_pending()
        return 0
    renamed = 0
    # vLLM accumulates encoder outputs in item order and pairs them with mm_hashes
    # positionally, so the queue head corresponds to the first hash.
    with _lock:
        for placeholder, mm_hash in zip(queue, mm_hashes):
            entry = _store.pop(placeholder, None)
            if entry is not None:
                # Re-encoded audio replaces its older entry and becomes the most recent,
                # which assigning to the existing key would not do.
                replaced = _store.pop(mm_hash, None)
                if replaced is not None:
                    replaced["dropped"] = True
                _store[mm_hash] = entry
                renamed += 1
    _state.pending_queue = []
    return renamed


def _discard_pending() -> int:
    """Drop queued placeholder entries that no scheduled item claims."""
    queue = getattr(_state, "pending_queue", None)
    if not queue:
        return 0
    with _lock:
        for placeholder in queue:
            entry = _store.pop(placeholder, None)
            if entry is not None:
                entry["dropped"] = True
    _state.pending_queue = []
    return len(queue)


def _trim_store() -> int:
    """Evict the least recently used captures while stored inputs exceed the byte budget.

    Captures normally go when their last owner is aligned or released, so this only
    catches captures nobody releases: outputs an offline caller drops, and in a server
    aborted requests and ``verbose_json`` requests without word timestamps. Python
    dicts preserve insertion order and alignment re-inserts what it keeps, so the first
    keys are the least recently used. The newest capture is kept even when it alone
    exceeds the budget.
    """
    byte_limit = _byte_limit()
    evicted = 0
    with _lock:
        stored = sum(entry["nbytes"] for entry in _store.values())
        while stored > byte_limit and len(_store) > 1:
            entry = _store.pop(next(iter(_store)))
            entry["dropped"] = True
            stored -= entry["nbytes"]
            evicted += 1
    if evicted:
        # A long-running server can reach it through requests it never aligns;
        # alignment reports each capture it then misses.
        logging.warning(
            "[NeMoSpeechLM] Evicting CTC timestamp captures that were never aligned or released to stay under "
            "%.1f GB (logged once). Offline, align or ctc_release() outputs sooner, or raise %s.",
            byte_limit / 1e9,
            _RETENTION_GB_ENV,
            mode=LogMode.ONCE,
        )
    return evicted


def _compact_ready() -> int:
    """Compact stored rows whose host copies have landed, oldest first, without waiting.

    A row starts as a view of its forward's padded host buffers, which would stay
    alive for as long as any row of that forward is retained. Copies complete in
    stream order, so the first row still in flight ends the pass.
    """
    compacted = 0
    with _lock:
        while _uncompacted:
            entry = _uncompacted[0]
            if entry["ready"] is not None and not entry["ready"].query():
                break
            _uncompacted.popleft()
            if not entry.get("dropped"):
                _compact(entry)
                compacted += 1
    return compacted


def _compact(entry: dict) -> None:
    """Replace an entry's views of its forward's buffers with copies of its valid frames."""
    for name, time_axis, lengths_name in _FIELDS:
        tensor = entry[name]
        if tensor is None:
            continue
        if time_axis is not None and entry[lengths_name] is not None:
            valid = min(int(entry[lengths_name][0]), tensor.shape[time_axis])
            tensor = tensor.narrow(time_axis, 0, valid)
        entry[name] = tensor.clone()
    entry["duration"] = float(entry["duration"])
    entry["ready"] = None
    entry["nbytes"] = _entry_nbytes(entry)


def _empty_result() -> dict:
    return {"words": [], "diarization": [], "speaker_tag_to_diarization_speaker": {}}


def align_requests(items: Sequence[tuple[str, str]], release: bool = True) -> list[dict]:
    """Align finished transcripts against their requests' stored inputs.

    Args:
        items (Sequence[tuple[str, str]]): ``(request_id, transcript)`` pairs; the
            transcript is aligned exactly as given, speaker tags included.
        release (bool): Afterwards drop each request's claim on its capture, so the
            capture can be deleted. Pass ``False`` to align the same requests again.

    Returns:
        list[dict]: Per item, ``words`` (``word``/``start``/``end``/``speaker`` in
        start-time order, speaker being the transcript's ``<spk:N>`` tag),
        ``diarization`` (10 ms Sortformer activity segments, ``speaker`` being the
        Sortformer speaker index; the output to score diarization with), and
        ``speaker_tag_to_diarization_speaker`` linking the two. Empty lists when
        timestamps are not enabled, when no capture is recorded for the request
        (no audio, or already released), when its capture was evicted, when it has
        more than one audio item, when the transcript is empty, or when the aligner
        cannot align it.

    Raises:
        Exception: Any aligner error other than ``ValueError``, which is how the
            aligner rejects a transcript it cannot align.
    """
    results: list[dict] = [_empty_result() for _ in items]
    encoder = active_encoder()
    if encoder is None:
        return results
    pending, no_audio, multi_audio, evicted = [], [], [], []
    with _lock:
        for index, (request_id, text) in enumerate(items):
            if not text.strip():
                continue
            hashes = mm_hashes_for_request(request_id)
            if not hashes:
                no_audio.append(request_id)
                continue
            if len(hashes) > 1:
                # The transcript spans all of the request's audio, but each item has its
                # own inputs and timeline, so aligning it to any one item would be wrong.
                multi_audio.append(request_id)
                continue
            key = next((h for h in hashes if h in _store), None)
            if key is None:
                evicted.append(request_id)
                continue
            # Re-insert to mark it recently used; releasing happens after alignment.
            entry = _store.pop(key)
            _store[key] = entry
            # A snapshot, so compaction on the engine thread cannot swap its tensors
            # while they are being collated.
            pending.append((index, text, dict(entry)))
    if multi_audio:
        logging.warning(
            "[NeMoSpeechLM] CTC timestamps need one audio item per request; skipped %d requests with more "
            "(first: %s).",
            len(multi_audio),
            multi_audio[0],
        )
    if no_audio:
        logging.warning(
            "[NeMoSpeechLM] No capture is recorded for %d of %d requests (first: %s): they carried no audio, "
            "were already aligned or released with release=True, or the request id is unknown.",
            len(no_audio),
            len(items),
            no_audio[0],
        )
    if evicted:
        logging.warning(
            "[NeMoSpeechLM] The captures of %d of %d requests (first: %s) were evicted to stay under %.1f GB; "
            "align or ctc_release() requests sooner, or raise %s.",
            len(evicted),
            len(items),
            evicted[0],
            _byte_limit() / 1e9,
            _RETENTION_GB_ENV,
        )
    # t-SOT output always opens with a tag, so a transcript without any lost them in decoding.
    untagged = [items[index][0] for index, text, _ in pending if "<spk:" not in text]
    if untagged:
        logging.warning(
            "[NeMoSpeechLM] %d of %d transcripts have no <spk:N> speaker tags (first: %s) and are aligned as "
            "one speaker; decode with skip_special_tokens=False to keep them.",
            len(untagged),
            len(pending),
            untagged[0],
        )

    device = next(encoder.parameters()).device
    for start in range(0, len(pending), _ALIGN_BATCH):
        chunk = pending[start : start + _ALIGN_BATCH]
        for (index, _, _), result in zip(chunk, _align_chunk(encoder, chunk, device)):
            results[index] = result
    if release:
        _release_requests([request_id for request_id, _ in items])
    return results


def align_request(request_id: str, text: str, release: bool = True) -> dict:
    """Align one finished transcript; see :func:`align_requests`."""
    return align_requests([(request_id, text)], release=release)[0]


def _align_chunk(encoder: Any, chunk: list, device: torch.device) -> list[dict]:
    """Run the deferred head and alignment for up to ``_ALIGN_BATCH`` requests."""
    entries = [entry for _, _, entry in chunk]
    try:
        timestamp_inputs = _collate(entries, device)
        results = encoder.generate_ctc_timestamps(
            timestamp_inputs=timestamp_inputs,
            sot_transcripts=[text for _, text, _ in chunk],
            # Read only after _collate has waited for the host copies.
            audio_durations=[float(entry["duration"]) for entry in entries],
        )
    except Exception as error:  # noqa: BLE001
        if len(chunk) > 1:
            # Retry one by one, so that a single bad transcript, or a batch too large
            # for device memory, does not cost the rest their timestamps.
            return [_align_chunk(encoder, [item], device)[0] for item in chunk]
        if not isinstance(error, ValueError):
            raise
        logging.warning("[NeMoSpeechLM] CTC alignment failed: %s", error)
        return [_empty_result()]
    return [_public_result(result) for result in results]


def _collate(entries: list[dict], device: torch.device) -> Any:
    """Batch stored single-row inputs, padding each time axis to the longest row."""
    from nemo.collections.asr.modules.parallel_expert_encoder import CTCTimestampInputs

    for entry in entries:
        if entry["ready"] is not None:
            entry["ready"].synchronize()
    fields = {}
    for name, time_axis, _ in _FIELDS:
        tensors = [entry[name] for entry in entries]
        fields[name] = None if any(t is None for t in tensors) else _collate_field(tensors, time_axis, device)
    return CTCTimestampInputs(**fields, diarization_frame_seconds=entries[0]["diarization_frame_seconds"])


def _collate_field(tensors: list[torch.Tensor], time_axis: int | None, device: torch.device) -> torch.Tensor:
    """Stack single-row tensors, zero-padding ``time_axis``; lengths mark the valid frames.

    The batch is assembled in pinned memory when it is bound for the GPU, so the copy
    to the device does not block.
    """
    shape = [len(tensors), *tensors[0].shape[1:]]
    if time_axis is not None:
        shape[time_axis] = max(tensor.shape[time_axis] for tensor in tensors)
    batch = torch.zeros(shape, dtype=tensors[0].dtype, pin_memory=device.type == "cuda")
    for row, tensor in enumerate(tensors):
        target = batch[row : row + 1]
        if time_axis is not None:
            target = target.narrow(time_axis, 0, tensor.shape[time_axis])
        target.copy_(tensor)
    return batch.to(device=device, non_blocking=True)


def _public_result(result: dict) -> dict:
    """Plain, serializable view of one aligner result (see :func:`align_requests`).

    Times are rounded to milliseconds; further digits are float noise from frame
    arithmetic.
    """
    words = [
        {
            "word": word["word"],
            "start": _seconds(word["start"]),
            "end": _seconds(word["end"]),
            "speaker": str(word["speaker"]),
        }
        for speaker_words in (result.get("speaker_word_timestamps") or {}).values()
        for word in speaker_words
    ]
    words.sort(key=lambda w: (w["start"], w["end"]))
    diarization = [
        {"speaker": int(segment["speaker"]), "start": _seconds(segment["start"]), "end": _seconds(segment["end"])}
        for segment in result.get("diarization_timestamps") or ()
    ]
    mapping = {
        str(tag): None if column is None else int(column)
        for tag, column in (result.get("speaker_tag_to_sortformer_column") or {}).items()
    }
    return {"words": words, "diarization": diarization, "speaker_tag_to_diarization_speaker": mapping}


def _seconds(value: Any) -> float:
    return round(float(value), 3)


def _new_request_hashes(scheduler_output: Any):
    """Yield ``(req_id, mm_hash)`` for each multimodal item of newly scheduled requests."""
    for new_req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
        features = getattr(new_req, "mm_features", None)
        if features is not None:
            hashes = [getattr(feature, "identifier", None) for feature in features]
        else:
            hashes = list(getattr(new_req, "mm_hashes", None) or ())
        for mm_hash in hashes:
            if mm_hash:
                yield new_req.req_id, mm_hash


def _aligns_here() -> bool:
    """Whether this worker aligns.

    Every tensor-parallel rank of the first pipeline stage runs the encoder and holds
    the same inputs, so only its rank 0 aligns; ``collective_rpc`` lists that worker's
    reply first. Outside an initialized engine, as in unit tests, every caller aligns.
    """
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    except ImportError:  # pragma: no cover - vLLM absent
        return True
    try:
        return get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank
    except (AssertionError, AttributeError):
        return True


def _worker_align_request(worker: Any, request_id: str, text: str) -> dict:
    """``collective_rpc`` entry point: align on the worker that holds the inputs.

    A server aligns each request once, so the request's capture is released after it.
    """
    del worker
    if not _aligns_here():
        _release_requests([request_id])
        return _empty_result()
    return align_request(request_id, text)


def _worker_align_requests(worker: Any, items: list, release: bool = True) -> list[dict]:
    """``collective_rpc`` entry point: align many finished requests in one call.

    Only :func:`ctc_timestamps` calls this, so a model loaded without timestamps is an
    error here; the single-request entry point, which a server calls for every
    timestamped request, keeps answering with empty results instead. Every rank keeps
    its own captures, so ranks that do not align still release theirs.
    """
    del worker
    if not _aligns_here():
        if release:
            _release_requests([request_id for request_id, _ in items])
        return [_empty_result() for _ in items]
    if active_encoder() is None:
        raise RuntimeError(
            "CTC timestamps are not enabled for this model: its config has no ctc_timestamps.adapter_path "
            "(check the hf_overrides key)."
        )
    return align_requests([(request_id, text) for request_id, text in items], release=release)


def _worker_release_requests(worker: Any, request_ids: list) -> None:
    """``collective_rpc`` entry point: release captures that will not be aligned, on every rank."""
    del worker
    _release_requests(list(request_ids))


def install_worker_align_method() -> None:
    """Expose alignment and release to callers outside the engine as worker methods.

    ``collective_rpc`` resolves a method name on the worker, so the entry points are
    attached to vLLM's GPU worker class under ``WORKER_ALIGN_METHOD``,
    ``WORKER_ALIGN_BATCH_METHOD`` and ``WORKER_RELEASE_METHOD``.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:  # pragma: no cover - vLLM absent
        return
    for name, method in (
        (WORKER_ALIGN_METHOD, _worker_align_request),
        (WORKER_ALIGN_BATCH_METHOD, _worker_align_requests),
        (WORKER_RELEASE_METHOD, _worker_release_requests),
    ):
        if getattr(Worker, name, None) is not method:
            setattr(Worker, name, method)


def ctc_timestamps(
    llm: Any,
    outputs: Sequence[Any],
    texts: Sequence[str] | None = None,
    chunk_size: int = 256,
    release: bool = True,
) -> list[dict]:
    """CTC word timestamps and diarization for finished outputs of an offline vLLM ``LLM``.

    Alignment runs in the engine process, where the stored inputs live, with one RPC
    per ``chunk_size`` outputs. Call it after generation: the RPC runs on the engine
    thread, so aligning while other requests decode would stall them.

    Each output's capture is deleted once it has been aligned with ``release=True``.
    Pass ``release=False`` to align the same outputs again, e.g. with ``texts``, and
    release outputs that will never be aligned with :func:`ctc_release`; captures
    nobody releases are bounded only by ``NEMO_CTC_TIMESTAMP_RETAIN_GB`` (default 8).

    Args:
        llm (Any): The ``vllm.LLM`` that produced ``outputs``, with CTC timestamps enabled.
        outputs (Sequence[Any]): Its ``RequestOutput`` objects, one audio item each.
            Decode speaker-tagged prompts with ``skip_special_tokens=False`` so
            ``<spk:N>`` reaches the aligner.
        texts (Sequence[str] | None): Transcripts to align instead of the generated
            ones, one per output, e.g. a reference or corrected transcript.
        chunk_size (int): Outputs aligned per RPC.
        release (bool): Delete the outputs' captures after aligning them.

    Returns:
        list[dict]: Per output, the result described in :func:`align_requests`.

    Raises:
        RuntimeError: When ``llm`` was loaded without CTC timestamps, e.g. because the
            ``ctc_timestamps`` key in ``hf_overrides`` is misspelled.
    """
    if texts is not None and len(texts) != len(outputs):
        raise ValueError(f"Got {len(texts)} texts for {len(outputs)} outputs.")
    items = [
        (out.request_id, out.outputs[0].text if texts is None else texts[index]) for index, out in enumerate(outputs)
    ]
    results: list[dict] = []
    for start in range(0, len(items), chunk_size):
        batch = items[start : start + chunk_size]
        # One reply per worker, in rank order; only the first worker aligns.
        results.extend(llm.collective_rpc(WORKER_ALIGN_BATCH_METHOD, args=(batch, release))[0])
    return results


def ctc_word_timestamps(
    llm: Any, outputs: Sequence[Any], chunk_size: int = 256, release: bool = True
) -> list[list[dict]]:
    """Just the word timestamps of :func:`ctc_timestamps`, one list per output."""
    return [result["words"] for result in ctc_timestamps(llm, outputs, chunk_size=chunk_size, release=release)]


def ctc_release(llm: Any, outputs: Sequence[Any]) -> None:
    """Delete the stored CTC timestamp inputs of outputs that will not be aligned.

    Args:
        llm (Any): The ``vllm.LLM`` that produced ``outputs``.
        outputs (Sequence[Any]): Its ``RequestOutput`` objects.
    """
    llm.collective_rpc(WORKER_RELEASE_METHOD, args=([out.request_id for out in outputs],))


def install_encoder_cache_binding() -> None:
    """Rename stored inputs to ``mm_hash`` and record the request-to-hash map.

    The hook passes straight through while no encoder is registered.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:  # pragma: no cover - vLLM absent
        return

    original = GPUModelRunner._execute_mm_encoder
    if getattr(original, "_nemo_ctc_timestamp_bound", False):
        return

    def _execute_mm_encoder(self, scheduler_output, *args, **kwargs):
        if active_encoder() is None:
            return original(self, scheduler_output, *args, **kwargs)

        # New requests resolve to their hashes here, including those whose audio is
        # an encoder-cache hit and therefore never reaches the batch below; without
        # this, repeated audio would find no inputs.
        _record_request_hashes(_new_request_hashes(scheduler_output))
        # vLLM evicting audio only ends cache hits on its capture; owners that have
        # not been aligned yet keep it.
        _follow_engine_cache(freed=getattr(scheduler_output, "free_encoder_mm_hashes", ()))

        # Encoder runs outside this hook, such as the startup profiling pass on dummy
        # audio, queue placeholders that no scheduled item claims. Left queued they
        # would pair with this batch's hashes and shift every entry onto the next
        # request.
        _discard_pending()

        mm_hashes, _, mm_lora_refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
        outputs = original(self, scheduler_output, *args, **kwargs)

        _rename_pending(mm_hashes)
        _follow_engine_cache(encoded=mm_hashes)
        _record_request_hashes((req_id, mm_hash) for mm_hash, (req_id, _position) in zip(mm_hashes, mm_lora_refs))
        _trim_store()
        _compact_ready()
        return outputs

    _execute_mm_encoder._nemo_ctc_timestamp_bound = True
    GPUModelRunner._execute_mm_encoder = _execute_mm_encoder
    logging.info("[NeMoSpeechLM] CTC timestamp inputs bound to vLLM encoder-cache identity.")

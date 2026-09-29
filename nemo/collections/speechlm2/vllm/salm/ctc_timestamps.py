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
copy overlaps with compute instead of stalling it. Retention is local and bounded by
``NEMO_CTC_TIMESTAMP_RETAIN`` entries (default: twice the engine's ``max_num_seqs``),
least recently used first: vLLM frees its encoder-cache entry when a request
finishes, which is before alignment reads it.

Alignment runs in the engine process, where the inputs live. Offline callers use
:func:`ctc_timestamps` (or :func:`ctc_word_timestamps` for words only); a server
reaches the same code through
``collective_rpc`` by the names in ``WORKER_ALIGN_METHOD`` and
``WORKER_ALIGN_BATCH_METHOD``, passing the external request id, which vLLM's
scheduler knows with a random suffix appended.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

import torch

from nemo.utils import logging

# Placeholder ids the inputs are stored under before the real hash is known.
_PENDING_PREFIX = "__nemo_ctc_pending_"

_RETENTION_ENV = "NEMO_CTC_TIMESTAMP_RETAIN"
# Used until set_default_retention sizes retention from the engine, and as its floor.
_DEFAULT_RETENTION = 64

# Worker method names reached through collective_rpc: one request (a server's
# transcription hooks) or many (offline LLM callers).
WORKER_ALIGN_METHOD = "nemo_ctc_align_request"
WORKER_ALIGN_BATCH_METHOD = "nemo_ctc_align_requests"

# vLLM forms the scheduler's request id as f"{external_id}-{random_uuid():.8}".
_INTERNAL_ID_SUFFIX_LEN = 8

# Requests decoded together in one deferred-head call, which materializes a
# (batch, frames, vocabulary) log-prob tensor on the device.
_ALIGN_BATCH = 16

# CTCTimestampInputs tensor fields, with the time axis to pad along when collating
# requests from different forwards (None for per-row lengths).
_FIELDS = (
    ("asr_encoded", 2),
    ("asr_encoded_lengths", None),
    ("sortformer_sigmoids", 1),
    ("sortformer_lengths", None),
    ("diarization_labels", 2),
    ("diarization_lengths", None),
)

# Placeholder bookkeeping stays thread-local: it is produced and consumed within a
# single _execute_mm_encoder call on the engine thread.
_state = threading.local()

# Process-wide, unlike _state. With an in-process engine the alignment call runs on
# the caller's thread while the encoder runs on the engine thread, so anything shared
# between them must not be thread-local.
_registry: dict[str, Any] = {}
_store: dict[str, dict] = {}
_request_hashes: dict[str, list[str]] = {}
_request_lock = threading.Lock()


def register_encoder(get_encoder) -> None:
    """Publish the live encoder, which also turns input capture on.

    The transcription hooks are classmethods on the vLLM model interface and never
    receive the model instance. One engine hosts one model per process, so a
    module-level getter is enough to bridge that.
    """
    _registry["get_encoder"] = get_encoder


def active_encoder() -> Any:
    """Return the live encoder, or ``None`` when timestamps are not enabled."""
    getter = _registry.get("get_encoder")
    return getter() if getter is not None else None


def set_default_retention(max_num_seqs: int) -> None:
    """Size retention from the engine's concurrency; ``NEMO_CTC_TIMESTAMP_RETAIN`` still wins.

    Twice ``max_num_seqs`` holds every request that can be in flight plus as many
    finished ones awaiting alignment, so a batch up to that size can be generated in
    one call and aligned afterwards.

    Args:
        max_num_seqs (int): The engine's ``scheduler_config.max_num_seqs``.
    """
    _registry["retention"] = max(_DEFAULT_RETENTION, 2 * int(max_num_seqs))


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


def store_timestamp_inputs(row_ids: Sequence[str], inputs: Any, audio_durations: Sequence[float]) -> None:
    """Keep one forward's ``CTCTimestampInputs`` per item, in host memory.

    Args:
        row_ids (Sequence[str]): One placeholder id per batch row, from :func:`pending_row_ids`.
        inputs (CTCTimestampInputs): The encoder's timestamp inputs for the batch.
        audio_durations (Sequence[float]): Audio duration per row, in seconds.
    """
    host = {name: _to_host(getattr(inputs, name)) for name, _ in _FIELDS}
    on_device = any(getattr(inputs, name) is not None and getattr(inputs, name).is_cuda for name, _ in _FIELDS)
    # Rows are read only after this event, so the copies never block the forward.
    ready = torch.cuda.current_stream().record_event() if on_device else None
    for row, (row_id, duration) in enumerate(zip(row_ids, audio_durations)):
        entry = {name: None if tensor is None else tensor[row : row + 1] for name, tensor in host.items()}
        entry.update(
            diarization_frame_seconds=inputs.diarization_frame_seconds,
            duration=float(duration),
            ready=ready,
        )
        _store[row_id] = entry


def _to_host(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy a tensor to host memory; pinned and non-blocking when it is on the GPU."""
    if tensor is None:
        return None
    if not tensor.is_cuda:
        return tensor.detach().clone()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor.detach(), non_blocking=True)
    return host


def _resolve_request_id(request_id: str) -> str | None:
    """Return the recorded key for a request id, accepting the external form.

    Callers holding a ``RequestOutput`` see the external id, while the scheduler
    reports the internal one; exact matches win so ids without a suffix (older vLLM,
    in-process callers) resolve unchanged. Must hold ``_request_lock``.
    """
    if request_id in _request_hashes:
        return request_id
    prefix = f"{request_id}-"
    for recorded in _request_hashes:
        if recorded.startswith(prefix) and len(recorded) == len(prefix) + _INTERNAL_ID_SUFFIX_LEN:
            return recorded
    return None


def mm_hashes_for_request(request_id: str) -> list[str]:
    """Return the multimodal hashes seen for a request, in arrival order.

    Args:
        request_id (str): vLLM request id, internal or external.

    Returns:
        list[str]: Hashes whose inputs belong to this request.
    """
    with _request_lock:
        key = _resolve_request_id(request_id)
        return list(_request_hashes[key]) if key is not None else []


def _record_request_hashes(pairs) -> None:
    """Associate a request with the hashes whose inputs belong to it."""
    with _request_lock:
        for req_id, mm_hash in pairs:
            hashes = _request_hashes.setdefault(req_id, [])
            if mm_hash not in hashes:
                hashes.append(mm_hash)
        # Entries are tiny but never removed on use, so cap the map on the same
        # order as the input store.
        limit = _retention_limit() * 4
        while len(_request_hashes) > limit:
            _request_hashes.pop(next(iter(_request_hashes)), None)


def _retention_limit() -> int:
    """How many stored entries to hold before evicting the least recently used."""
    default = _registry.get("retention", _DEFAULT_RETENTION)
    try:
        value = int(os.environ.get(_RETENTION_ENV, default))
    except ValueError:
        return default
    return max(1, value)


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
    for placeholder, mm_hash in zip(queue, mm_hashes):
        entry = _store.pop(placeholder, None)
        if entry is not None:
            _store[mm_hash] = entry
            renamed += 1
    _state.pending_queue = []
    return renamed


def _discard_pending() -> int:
    """Drop queued placeholder entries that no scheduled item claims."""
    queue = getattr(_state, "pending_queue", None)
    if not queue:
        return 0
    for placeholder in queue:
        _store.pop(placeholder, None)
    _state.pending_queue = []
    return len(queue)


def _trim_store() -> int:
    """Evict least recently used entries so retention stays bounded.

    Nothing else removes entries: alignment reads without taking so that repeated
    audio, served from vLLM's encoder cache, still finds them. Python dicts preserve
    insertion order and alignment re-inserts what it reads, so the first keys are the
    least recently used.
    """
    limit = _retention_limit()
    evicted = 0
    while len(_store) > limit:
        _store.pop(next(iter(_store)), None)
        evicted += 1
    if evicted:
        logging.debug(
            "[NeMoSpeechLM] Evicted %d CTC timestamp entries (limit %d). "
            "Raise %s if timestamped requests are being dropped.",
            evicted,
            limit,
            _RETENTION_ENV,
        )
    return evicted


def _empty_result() -> dict:
    return {"words": [], "diarization": [], "speaker_tag_to_diarization_speaker": {}}


def align_requests(items: Sequence[tuple[str, str]]) -> list[dict]:
    """Align finished transcripts against their requests' stored inputs.

    Args:
        items (Sequence[tuple[str, str]]): ``(request_id, transcript)`` pairs; the
            transcript is aligned exactly as given, speaker tags included.

    Returns:
        list[dict]: Per item, ``words`` (``word``/``start``/``end``/``speaker`` in
        start-time order, speaker being the transcript's ``<spk:N>`` tag),
        ``diarization`` (10 ms Sortformer activity segments, ``speaker`` being the
        Sortformer speaker index; the output to score diarization with), and
        ``speaker_tag_to_diarization_speaker`` linking the two. Empty lists when
        nothing is stored for that request, when it has more than one audio item,
        when the transcript is empty, or when its alignment fails.
    """
    results: list[dict] = [_empty_result() for _ in items]
    encoder = active_encoder()
    if encoder is None:
        return results
    pending, multi_audio, missing = [], [], []
    for index, (request_id, text) in enumerate(items):
        if not text.strip():
            continue
        hashes = mm_hashes_for_request(request_id)
        if len(hashes) > 1:
            # The transcript spans all of the request's audio, but each item has its
            # own inputs and timeline, so aligning it to any one item would be wrong.
            multi_audio.append(request_id)
            continue
        key = next((h for h in hashes if h in _store), None)
        if key is None:
            missing.append(request_id)
            continue
        # Re-insert rather than take: a repeated call for this request, or another
        # request whose audio hit vLLM's encoder cache, must still find the inputs.
        entry = _store.pop(key)
        _store[key] = entry
        pending.append((index, text, entry))
    if multi_audio:
        logging.warning(
            "[NeMoSpeechLM] CTC timestamps need one audio item per request; skipped %d requests with more "
            "(first: %s).",
            len(multi_audio),
            multi_audio[0],
        )
    if missing:
        logging.warning(
            "[NeMoSpeechLM] No CTC timestamp inputs for %d of %d requests (first: %s): evicted or never "
            "captured. Only the latest %d captures are kept; align more often or raise %s.",
            len(missing),
            len(items),
            missing[0],
            _retention_limit(),
            _RETENTION_ENV,
        )

    device = next(encoder.parameters()).device
    for start in range(0, len(pending), _ALIGN_BATCH):
        chunk = pending[start : start + _ALIGN_BATCH]
        for (index, _, _), result in zip(chunk, _align_chunk(encoder, chunk, device)):
            results[index] = result
    return results


def align_request(request_id: str, text: str) -> dict:
    """Align one finished transcript; see :func:`align_requests`."""
    return align_requests([(request_id, text)])[0]


def _align_chunk(encoder: Any, chunk: list, device: torch.device) -> list[dict]:
    """Run the deferred head and alignment for up to ``_ALIGN_BATCH`` requests."""
    try:
        results = encoder.generate_ctc_timestamps(
            timestamp_inputs=_collate([entry for _, _, entry in chunk], device),
            sot_transcripts=[text for _, text, _ in chunk],
            audio_durations=[entry["duration"] for _, _, entry in chunk],
        )
    except Exception as error:  # noqa: BLE001
        if len(chunk) > 1:
            # One transcript the tokenizer cannot split consistently should not
            # cost the rest of the batch its timestamps.
            return [_align_chunk(encoder, [item], device)[0] for item in chunk]
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
    for name, time_axis in _FIELDS:
        tensors = [entry[name] for entry in entries]
        if any(tensor is None for tensor in tensors):
            fields[name] = None
            continue
        if time_axis is not None:
            width = max(tensor.shape[time_axis] for tensor in tensors)
            tensors = [_pad_to(tensor, time_axis, width) for tensor in tensors]
        fields[name] = torch.cat(tensors, dim=0).to(device=device, non_blocking=True)
    return CTCTimestampInputs(**fields, diarization_frame_seconds=entries[0]["diarization_frame_seconds"])


def _pad_to(tensor: torch.Tensor, axis: int, width: int) -> torch.Tensor:
    """Zero-pad ``tensor`` along ``axis`` to ``width``; lengths mark the valid frames."""
    if tensor.shape[axis] == width:
        return tensor
    shape = list(tensor.shape)
    shape[axis] = width
    padded = tensor.new_zeros(shape)
    padded.narrow(axis, 0, tensor.shape[axis]).copy_(tensor)
    return padded


def _public_result(result: dict) -> dict:
    """Plain, serializable view of one aligner result (see :func:`align_requests`)."""
    words = [
        {
            "word": word["word"],
            "start": float(word["start"]),
            "end": float(word["end"]),
            "speaker": str(word["speaker"]),
        }
        for speaker_words in (result.get("speaker_word_timestamps") or {}).values()
        for word in speaker_words
    ]
    words.sort(key=lambda w: (w["start"], w["end"]))
    diarization = [
        {"speaker": int(segment["speaker"]), "start": float(segment["start"]), "end": float(segment["end"])}
        for segment in result.get("diarization_timestamps") or ()
    ]
    mapping = {
        str(tag): None if column is None else int(column)
        for tag, column in (result.get("speaker_tag_to_sortformer_column") or {}).items()
    }
    return {"words": words, "diarization": diarization, "speaker_tag_to_diarization_speaker": mapping}


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


def _worker_align_request(worker: Any, request_id: str, text: str) -> dict:
    """``collective_rpc`` entry point: align on the worker that holds the inputs."""
    del worker
    return align_request(request_id, text)


def _worker_align_requests(worker: Any, items: list) -> list[dict]:
    """``collective_rpc`` entry point: align many finished requests in one call."""
    del worker
    return align_requests([(request_id, text) for request_id, text in items])


def install_worker_align_method() -> None:
    """Expose alignment to callers outside the engine as worker methods.

    ``collective_rpc`` resolves a method name on the worker, so the alignment entry
    points are attached to vLLM's GPU worker class under ``WORKER_ALIGN_METHOD`` and
    ``WORKER_ALIGN_BATCH_METHOD``.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:  # pragma: no cover - vLLM absent
        return
    for name, method in (
        (WORKER_ALIGN_METHOD, _worker_align_request),
        (WORKER_ALIGN_BATCH_METHOD, _worker_align_requests),
    ):
        if getattr(Worker, name, None) is not method:
            setattr(Worker, name, method)


def ctc_timestamps(
    llm: Any, outputs: Sequence[Any], texts: Sequence[str] | None = None, chunk_size: int = 256
) -> list[dict]:
    """CTC word timestamps and diarization for finished outputs of an offline vLLM ``LLM``.

    Alignment runs in the engine process, where the stored inputs live, with one RPC
    per ``chunk_size`` outputs. Call it after generation: the RPC runs on the engine
    thread, so aligning while other requests decode would stall them.

    Inputs are held only for the most recent ``NEMO_CTC_TIMESTAMP_RETAIN`` captures
    (default: twice the engine's ``max_num_seqs``), so align at least that often,
    e.g. by generating in chunks no larger than the limit, or raise the limit to
    cover a whole batch.

    Args:
        llm (Any): The ``vllm.LLM`` that produced ``outputs``, with CTC timestamps enabled.
        outputs (Sequence[Any]): Its ``RequestOutput`` objects, one audio item each.
            Decode speaker-tagged prompts with ``skip_special_tokens=False`` so
            ``<spk:N>`` reaches the aligner.
        texts (Sequence[str] | None): Transcripts to align instead of the generated
            ones, one per output, e.g. a reference or corrected transcript.
        chunk_size (int): Outputs aligned per RPC.

    Returns:
        list[dict]: Per output, the result described in :func:`align_requests`.
    """
    if texts is not None and len(texts) != len(outputs):
        raise ValueError(f"Got {len(texts)} texts for {len(outputs)} outputs.")
    items = [
        (out.request_id, out.outputs[0].text if texts is None else texts[index]) for index, out in enumerate(outputs)
    ]
    results: list[dict] = []
    for start in range(0, len(items), chunk_size):
        batch = items[start : start + chunk_size]
        results.extend(llm.collective_rpc(WORKER_ALIGN_BATCH_METHOD, args=(batch,))[0])
    return results


def ctc_word_timestamps(llm: Any, outputs: Sequence[Any], chunk_size: int = 256) -> list[list[dict]]:
    """Just the word timestamps of :func:`ctc_timestamps`, one list per output."""
    return [result["words"] for result in ctc_timestamps(llm, outputs, chunk_size=chunk_size)]


def install_encoder_cache_binding(get_encoder) -> None:
    """Rename stored inputs to ``mm_hash`` and record the request-to-hash map.

    Args:
        get_encoder (Callable[[], Any]): Returns the encoder, or ``None`` when
            timestamps are not enabled.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:  # pragma: no cover - vLLM absent
        return

    original = GPUModelRunner._execute_mm_encoder
    if getattr(original, "_nemo_ctc_timestamp_bound", False):
        return

    def _execute_mm_encoder(self, scheduler_output, *args, **kwargs):
        if get_encoder() is None:
            return original(self, scheduler_output, *args, **kwargs)

        # New requests resolve to their hashes here, including those whose audio is
        # an encoder-cache hit and therefore never reaches the batch below; without
        # this, repeated audio would find no inputs.
        _record_request_hashes(_new_request_hashes(scheduler_output))

        # Encoder runs outside this hook, such as the startup profiling pass on dummy
        # audio, queue placeholders that no scheduled item claims. Left queued they
        # would pair with this batch's hashes and shift every entry onto the next
        # request.
        _discard_pending()

        mm_hashes, _, mm_lora_refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
        outputs = original(self, scheduler_output, *args, **kwargs)

        _rename_pending(mm_hashes)
        _record_request_hashes((req_id, mm_hash) for mm_hash, (req_id, _position) in zip(mm_hashes, mm_lora_refs))
        # Deliberately NOT mirroring scheduler_output.free_encoder_mm_hashes: vLLM
        # frees its encoder-cache entry when the request finishes, which is before
        # alignment reads the inputs. Bound retention here instead.
        _trim_store()
        return outputs

    _execute_mm_encoder._nemo_ctc_timestamp_bound = True
    GPUModelRunner._execute_mm_encoder = _execute_mm_encoder
    logging.info("[NeMoSpeechLM] CTC timestamp inputs bound to vLLM encoder-cache identity.")

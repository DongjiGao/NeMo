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

"""Bind captured CTC timestamp rows to vLLM's multimodal cache identity.

The encoder captures CTC rows as a side effect of the forward that also feeds the
LLM, so timestamps cost no second encoder pass. Two things make that side effect
hard to address in a server:

* ``embed_multimodal(**mm_kwargs_batch)`` receives only tensors. No request id and
  no ``mm_hash`` reach the model, so the capture has nothing to key on.
* The scheduler only schedules encoder inputs whose ``mm_hash`` is absent from the
  encoder cache. On a cache hit the encoder never runs, so a request that needs
  timestamps would find nothing captured.

Both are solved by keying on ``mm_hash``, the identity vLLM already uses: a cache
hit then finds rows sitting under the key the request resolves to.

The encoder captures under positional placeholders, because the hash is unknown at
that point, and this renames them afterwards. The hook is ``_execute_mm_encoder``,
wrapped around one call to ``_batch_mm_inputs_from_scheduler``, which returns

    mm_hashes, mm_kwargs, mm_lora_refs

in item order, with ``mm_lora_refs`` carrying ``(req_id, placeholder_range)``. That
gives both the rename and a request-to-hash map, so a caller can later ask for its
own timestamps by request id.

Both of those methods have identical signatures in vLLM 0.23 and 0.28, which is why
they are hooked instead of ``_cache_encoder_output`` -- that one is cleaner to pair
against but does not exist in 0.23, and CTC work needs to iterate on whichever
stack is healthy.

Lifetime is deliberately decoupled from vLLM's. The rows share vLLM's *identity*
but not its retention: vLLM releases an encoder-cache entry once the request
finishes, whereas the serving hooks that consume the rows run during response
assembly, at or after that point. Following ``free_encoder_mm_hashes`` would
therefore discard rows immediately before they are read, so retention is instead
bounded locally by ``NEMO_CTC_TIMESTAMP_RETAIN`` entries, least recently used
evicted first.

Under ``vllm serve`` the transcription hooks run in the API server process while
the rows live in the engine process next to the model, so the server cannot read
them directly. It asks the engine to align through ``collective_rpc`` by the name
in ``WORKER_ALIGN_METHOD``, passing the external request id; vLLM's scheduler
knows the request by that id plus a random suffix, which the lookup resolves.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from nemo.utils import logging

# Placeholder ids the encoder captures under before the real hash is known.
_PENDING_PREFIX = "__nemo_ctc_pending_"

# Captured rows must outlive vLLM's encoder-cache entry for the same hash.
# vLLM frees that entry when the request finishes, but the serving hooks that
# consume the rows (parse_diarized_transcript / get_word_timestamps) run during
# response assembly, i.e. at or after that point. Mirroring vLLM's eviction
# therefore drops rows just before they are needed, so retention is bounded
# here instead: oldest-first, with a cap, since nothing else limits growth.
_RETENTION_ENV = "NEMO_CTC_TIMESTAMP_RETAIN"
_DEFAULT_RETENTION = 64

# Worker method name the API server calls through collective_rpc.
WORKER_ALIGN_METHOD = "nemo_ctc_align_request"

# vLLM forms the scheduler's request id as f"{external_id}-{random_uuid():.8}".
_INTERNAL_ID_SUFFIX_LEN = 8

# Placeholder bookkeeping stays thread-local: it is produced and consumed within
# a single _execute_mm_encoder call on the engine thread, so keeping it per-thread
# avoids interleaving if that ever runs concurrently.
_state = threading.local()

# Process-wide, unlike _state. With an in-process engine the hooks run on the
# caller's thread while the capture happens on the engine thread, so anything
# shared between them must not be thread-local or the hooks would observe an
# empty map.
_registry: dict[str, Any] = {}
_request_hashes: dict[str, list[str]] = {}
_request_lock = threading.Lock()


def register_encoder(get_encoder) -> None:
    """Publish the live ASR encoder for the model's transcription classmethods.

    ``parse_diarized_transcript`` and ``get_word_timestamps`` are classmethods on
    the vLLM model interface, so they never receive the model instance and cannot
    reach ``self.perception.encoder``. One engine hosts one model per process, so
    a module-level getter is sufficient to bridge that.
    """
    _registry["get_encoder"] = get_encoder


def active_encoder() -> Any:
    """Return the live ASR encoder, or ``None`` when timestamps are not enabled."""
    getter = _registry.get("get_encoder")
    return getter() if getter is not None else None


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


def _resolve_request_id(request_id: str) -> str | None:
    """Return the recorded key for a request id, accepting the external form.

    Callers holding a ``RequestOutput`` see the external id, while the scheduler
    reports the internal one; exact matches win so ids without a suffix (older
    vLLM, in-process callers) resolve unchanged. Must hold ``_request_lock``.
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
        list[str]: Hashes whose rows belong to this request.
    """
    with _request_lock:
        key = _resolve_request_id(request_id)
        return list(_request_hashes[key]) if key is not None else []


def forget_request(request_id: str) -> None:
    """Drop the request-to-hash mapping for a finished request."""
    with _request_lock:
        key = _resolve_request_id(request_id)
        if key is not None:
            _request_hashes.pop(key, None)


def _record_request_hashes(pairs) -> None:
    """Associate a request with the hashes whose rows belong to it."""
    with _request_lock:
        for req_id, mm_hash in pairs:
            hashes = _request_hashes.setdefault(req_id, [])
            if mm_hash not in hashes:
                hashes.append(mm_hash)
        # The map is tiny per entry but still unbounded if callers never take
        # their timestamps, so cap it on the same order as the row store.
        limit = _retention_limit() * 4
        while len(_request_hashes) > limit:
            _request_hashes.pop(next(iter(_request_hashes)), None)


def _retention_limit() -> int:
    """How many captured entries to hold before evicting the oldest."""
    try:
        value = int(os.environ.get(_RETENTION_ENV, _DEFAULT_RETENTION))
    except ValueError:
        return _DEFAULT_RETENTION
    return max(1, value)


def _rename_pending(encoder: Any, mm_hashes: list[str]) -> int:
    """Rename this forward's placeholder entries to their real hashes."""
    store = encoder.__dict__.get("_ctc_timestamp_request_store")
    queue = getattr(_state, "pending_queue", None)
    if not store or not queue:
        return 0
    renamed = 0
    # vLLM accumulates encoder outputs in item order and pairs them with
    # mm_hashes positionally, so the queue head corresponds to the first hash.
    for mm_hash in mm_hashes:
        if not queue:
            break
        placeholder = queue.pop(0)
        if placeholder in store:
            store[mm_hash] = store.pop(placeholder)
            renamed += 1
    return renamed


def _trim_store(encoder: Any) -> int:
    """Evict least recently used entries so retention stays bounded.

    Nothing else removes rows: alignment reads without taking so that repeated
    audio, served from vLLM's encoder cache, still finds them. Python dicts
    preserve insertion order and ``align_request`` re-inserts what it reads, so
    the first keys are the least recently used.
    """
    store = encoder.__dict__.get("_ctc_timestamp_request_store")
    if not store:
        return 0
    limit = _retention_limit()
    evicted = 0
    while len(store) > limit:
        oldest = next(iter(store))
        store.pop(oldest, None)
        evicted += 1
    if evicted:
        logging.debug(
            "[NeMoSpeechLM] Evicted %d un-taken CTC timestamp entries (limit %d). "
            "Raise %s if timestamped requests are being dropped.",
            evicted,
            limit,
            _RETENTION_ENV,
        )
    return evicted


def align_request(request_id: str, text: str) -> list[dict]:
    """Align a finished transcript against that request's captured CTC rows.

    Shared by both transcription hooks so a request is aligned once regardless of
    whether the client asked for word timings, speaker segments, or both.

    Args:
        request_id (str): vLLM request id of the finished request.
        text (str): The transcript to align, as returned to the client.

    Returns:
        list[dict]: Word entries with ``word``, ``start``, ``end`` and
        ``speaker``, ordered by start time. Empty when nothing was captured for
        the request, which is the expected result if timestamps were not enabled.
    """
    encoder = active_encoder()
    if encoder is None or not text.strip():
        return []

    store = encoder.__dict__.get("_ctc_timestamp_request_store") or {}
    hashes = mm_hashes_for_request(request_id)
    key = next((h for h in hashes if h in store), None)
    if key is None:
        logging.warning(
            "[NeMoSpeechLM] No CTC rows for request %s (hashes=%s, store=%d entries). "
            "The rows may have been evicted; raise %s.",
            request_id,
            hashes,
            len(store),
            _RETENTION_ENV,
        )
        return []

    cached = encoder.__dict__.get("_ctc_timestamp_extractor_cache")
    if cached is None:
        logging.warning("[NeMoSpeechLM] CTC extractor is not loaded; cannot align timestamps.")
        return []
    extractor = cached[1]

    # Re-insert rather than take: another request with the same audio resolves to
    # this hash through an encoder-cache hit, and _trim_store bounds retention.
    outputs = store.pop(key)
    store[key] = outputs
    forget_request(request_id)

    # The aligner derives its frame grid as duration / frame_count, so a duration
    # synthesized from the frame count reproduces the encoder's own grid exactly.
    # Using the request's wall-clock duration instead would shift every frame by
    # the rounding the encoder already applied when it padded to whole frames.
    frames = int(outputs["ctc_lengths"][0].item())
    frame_seconds = _frame_seconds(encoder)
    duration = frames * frame_seconds

    try:
        aligned = extractor.extract_from_outputs_batch(
            sot_transcripts=[text],
            audio_durations=[duration],
            ctc_log_probs=outputs["ctc_log_probs"],
            ctc_lengths=outputs["ctc_lengths"],
            sortformer_sigmoids=outputs["sortformer_sigmoids"],
            sortformer_lengths=outputs["sortformer_lengths"],
        )
    except Exception as error:  # noqa: BLE001
        # A tokenizer disagreement on one utterance should degrade to "no
        # timestamps" rather than failing the whole transcription request.
        logging.warning("[NeMoSpeechLM] CTC alignment failed for request %s: %s", request_id, error)
        return []

    if not aligned:
        return []
    words: list[dict] = []
    for speaker, speaker_words in (aligned[0].get("speaker_word_timestamps") or {}).items():
        for word in speaker_words:
            words.append(
                {
                    "word": word.get("word", ""),
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                    "speaker": str(word.get("speaker_tag", speaker)),
                }
            )
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


def _frame_seconds(encoder: Any) -> float:
    """Seconds of audio per CTC frame for this encoder."""
    shift = float(getattr(encoder, "frame_shift_seconds", 0.01) or 0.01)
    asr_encoder = getattr(encoder, "asr_encoder", None)
    subsampling = int(getattr(asr_encoder, "subsampling_factor", 8) or 8)
    return shift * subsampling


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


def _worker_align_request(worker: Any, request_id: str, text: str) -> list[dict]:
    """``collective_rpc`` entry point: align on the worker that holds the rows."""
    del worker
    return align_request(request_id, text)


def install_worker_align_method() -> None:
    """Expose :func:`align_request` to the API server as a worker method.

    ``collective_rpc`` resolves a method name on the worker, so the alignment
    entry point is attached to vLLM's GPU worker class under
    ``WORKER_ALIGN_METHOD``.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:  # pragma: no cover - vLLM absent
        return
    if getattr(Worker, WORKER_ALIGN_METHOD, None) is not _worker_align_request:
        setattr(Worker, WORKER_ALIGN_METHOD, _worker_align_request)


def install_encoder_cache_binding(get_encoder) -> None:
    """Rename captured rows to ``mm_hash`` and record the request-to-hash map.

    Args:
        get_encoder (Callable[[], Any]): Returns the ASR encoder holding the capture
            store, or ``None`` when timestamps are not enabled.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:  # pragma: no cover - vLLM absent
        return

    original = GPUModelRunner._execute_mm_encoder
    if getattr(original, "_nemo_ctc_timestamp_bound", False):
        return

    def _execute_mm_encoder(self, scheduler_output, *args, **kwargs):
        encoder = get_encoder()
        if encoder is None:
            return original(self, scheduler_output, *args, **kwargs)

        # New requests resolve to their hashes here, including those whose
        # encoder input is a cache hit and therefore never reaches the batch
        # below; without this, repeated audio would find no rows.
        _record_request_hashes(_new_request_hashes(scheduler_output))

        mm_hashes, _, mm_lora_refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
        outputs = original(self, scheduler_output, *args, **kwargs)

        _rename_pending(encoder, mm_hashes)

        _record_request_hashes(
            (req_id, mm_hash)
            for mm_hash, (req_id, _position) in zip(mm_hashes, mm_lora_refs)
        )

        # Deliberately NOT mirroring scheduler_output.free_encoder_mm_hashes.
        # vLLM frees its encoder-cache entry when the request finishes, which is
        # at or before the point the serving hooks read the rows, so following
        # that signal would discard them just before use. Bound retention here
        # instead.
        _trim_store(encoder)
        return outputs

    _execute_mm_encoder._nemo_ctc_timestamp_bound = True
    GPUModelRunner._execute_mm_encoder = _execute_mm_encoder
    logging.info("[NeMoSpeechLM] CTC timestamp rows bound to vLLM encoder-cache identity.")

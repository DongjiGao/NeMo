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
"""

from __future__ import annotations

import threading
from typing import Any

from nemo.utils import logging

# Placeholder ids the encoder captures under before the real hash is known.
_PENDING_PREFIX = "__nemo_ctc_pending_"

# The encode and the rename happen in separate calls, so the counter and the
# request map have to survive between them.
_state = threading.local()


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


def mm_hashes_for_request(request_id: str) -> list[str]:
    """Return the multimodal hashes seen for a request, in arrival order.

    Args:
        request_id (str): vLLM request id.

    Returns:
        list[str]: Hashes whose rows belong to this request.
    """
    return list(getattr(_state, "request_hashes", {}).get(request_id, ()))


def forget_request(request_id: str) -> None:
    """Drop the request-to-hash mapping for a finished request."""
    getattr(_state, "request_hashes", {}).pop(request_id, None)


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

        mm_hashes, _, mm_lora_refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
        outputs = original(self, scheduler_output, *args, **kwargs)

        _rename_pending(encoder, mm_hashes)

        request_hashes = getattr(_state, "request_hashes", None)
        if request_hashes is None:
            request_hashes = _state.request_hashes = {}
        for mm_hash, (req_id, _position) in zip(mm_hashes, mm_lora_refs):
            request_hashes.setdefault(req_id, []).append(mm_hash)

        # Our rows must not outlive vLLM's entry for the same hash, or a later
        # cache hit resolves to rows we already dropped.
        for freed in getattr(scheduler_output, "free_encoder_mm_hashes", None) or ():
            encoder.discard_ctc_timestamps(freed)
        return outputs

    _execute_mm_encoder._nemo_ctc_timestamp_bound = True
    GPUModelRunner._execute_mm_encoder = _execute_mm_encoder
    logging.info("[NeMoSpeechLM] CTC timestamp rows bound to vLLM encoder-cache identity.")

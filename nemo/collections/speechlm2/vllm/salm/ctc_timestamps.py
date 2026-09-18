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

Rather than re-deriving the item-to-hash mapping, this reuses vLLM's own. The
runner pairs them positionally and then announces each pair:

    for mm_hash, output in zip(mm_hashes, encoder_outputs):
        self._cache_encoder_output(mm_hash, output, ...)

So the encoder captures under positional placeholders, and wrapping
``_cache_encoder_output`` renames each placeholder to its real ``mm_hash`` using
vLLM's pairing instead of a parallel reimplementation that could drift from it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from nemo.utils import logging

if TYPE_CHECKING:
    import torch

# Placeholder ids the encoder captures under before the real hash is known.
_PENDING_PREFIX = "__nemo_ctc_pending_"

# One model runner per process, but the encode and the caching announcements are
# separate calls, so the counter needs to survive between them.
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


def _rekey(encoder: Any, old_id: str, new_id: str) -> bool:
    """Move a captured entry from a placeholder id to its ``mm_hash``."""
    store = encoder.__dict__.get("_ctc_timestamp_request_store")
    if not store or old_id not in store:
        return False
    store[new_id] = store.pop(old_id)
    return True


def install_encoder_cache_binding(get_encoder) -> None:
    """Rename captured rows to ``mm_hash`` as vLLM caches each encoder output.

    Args:
        get_encoder (Callable[[], Any] | None): Returns the ASR encoder holding the
            capture store, or ``None`` when timestamps are not enabled.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:  # pragma: no cover - vLLM absent
        return

    original = GPUModelRunner._cache_encoder_output
    if getattr(original, "_nemo_ctc_timestamp_bound", False):
        return

    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: "torch.Tensor",
        ec_manager_metadata: Any = None,
        free_encoder_mm_hashes: Any = None,
    ) -> None:
        encoder = get_encoder()
        if encoder is not None:
            queue = getattr(_state, "pending_queue", None)
            if queue:
                # vLLM announces pairs in the same order the items were encoded,
                # so the head of the queue is this hash's row.
                _rekey(encoder, queue.pop(0), mm_hash)
            # Our rows must not outlive vLLM's entry for the same hash, or a later
            # cache hit resolves to rows we already dropped.
            for freed in free_encoder_mm_hashes or ():
                encoder.discard_ctc_timestamps(freed)
        return original(self, mm_hash, output, ec_manager_metadata, free_encoder_mm_hashes)

    _cache_encoder_output._nemo_ctc_timestamp_bound = True
    GPUModelRunner._cache_encoder_output = _cache_encoder_output
    logging.info("[NeMoSpeechLM] CTC timestamp rows bound to vLLM encoder-cache identity.")

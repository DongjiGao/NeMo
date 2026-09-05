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

"""
Attention-backend selection for the NeMo ``TransformerEncoder``.

Two backends are available. ``"flex"`` is the default BF16 PyTorch FlexAttention path. ``"fp8_flex"``
is an opt-in inference-only path that casts Q/K/V to ``float8_e4m3fn`` and runs FlexAttention's
Triton lowering behind a custom operator; it is implemented in
:mod:`~nemo.collections.asr.parts.utils.sortformer_fp8_flex_attention`.

Backend selection is independent of quantization: a BF16 run and an NVFP4 run may both select
``fp8_flex``. Apply the backend before ``torch.compile`` so the compiled graph contains the selected
attention path.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from nemo.collections.asr.parts.utils.sortformer_fp8_flex_attention import (
    FP8_FLEX_BACKEND,
    fp8_flex_backend_info,
)

if TYPE_CHECKING:
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

__all__ = [
    "FLEX_BACKEND",
    "SUPPORTED_ATTENTION_BACKENDS",
    "attention_backend_cache_identity",
    "attention_backend_info",
    "configure_attention_backend",
    "validate_attention_backend",
]

FLEX_BACKEND = "flex"
SUPPORTED_ATTENTION_BACKENDS = (FLEX_BACKEND, FP8_FLEX_BACKEND)


def validate_attention_backend(backend: Optional[str]) -> str:
    """Normalize and validate an attention-backend selection.

    Args:
        backend: Requested backend name; ``None`` is accepted as a YAML-friendly alias for the default.

    Returns:
        backend (str): One of :data:`SUPPORTED_ATTENTION_BACKENDS`.

    Raises:
        ValueError: If the backend is not a string or is not supported.
    """
    if backend is None:
        return FLEX_BACKEND
    if not isinstance(backend, str):
        raise ValueError(f"attention_backend must be a string, got {type(backend).__name__}.")
    if backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            f"attention_backend='{backend}' is not supported. Supported backends: {SUPPORTED_ATTENTION_BACKENDS}."
        )
    return backend


def configure_attention_backend(diar_model: "SortformerEncLabelModel", backend: Optional[str]) -> Dict[str, str]:
    """Apply an attention backend to every ``TransformerEncoder`` present on a Sortformer model.

    Call this before ``torch.compile`` so the compiled graph already contains the selected attention path.

    Args:
        diar_model (SortformerEncLabelModel): Restored diarization model.
        backend: Requested backend name, see :func:`validate_attention_backend`.

    Returns:
        applied (Dict[str, str]): Attribute name -> backend, for every module the backend was applied to.
        Modules that are not attention-backend selectable (e.g. a Conformer front end) are skipped.

    Raises:
        ValueError: If a non-default backend was requested but no module accepts it.
    """
    backend = validate_attention_backend(backend)
    applied: Dict[str, str] = {}
    for name in ("encoder", "transformer_encoder"):
        module = getattr(diar_model, name, None)
        setter = getattr(module, "set_attention_backend", None) if module is not None else None
        if setter is None:
            continue
        setter(backend)
        applied[name] = backend
    if backend != FLEX_BACKEND and not applied:
        raise ValueError(
            f"attention_backend='{backend}' was requested but the model has no attention-backend selectable "
            "TransformerEncoder to apply it to."
        )
    return applied


def attention_backend_cache_identity(backend: Optional[str]) -> Optional[str]:
    """Describe the attention backend for prediction-cache metadata.

    A non-default backend is numerically close to, but not identical with, the default FlexAttention path, so
    their predictions must not share a cache entry. The default backend maps to ``None`` so that caches
    written before this option existed keep matching a default run.

    Args:
        backend: Requested backend name.

    Returns:
        identity (Optional[str]): Backend name, or ``None`` for the default FlexAttention backend.
    """
    backend = validate_attention_backend(backend)
    return None if backend == FLEX_BACKEND else backend


def attention_backend_info(backend: Optional[str], device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Collect the reproducibility facts of whichever attention backend was selected.

    Each backend reports its own facts, so the FP8 path is described only when it is the one selected.
    Never raises.

    Args:
        backend: Requested backend name, see :func:`validate_attention_backend`.
        device: CUDA device to report on. Defaults to the current device when CUDA is available.

    Returns:
        info (Dict[str, Any]): Backend-specific facts; always contains ``"backend"``.
    """
    backend = validate_attention_backend(backend)
    if backend == FP8_FLEX_BACKEND:
        return fp8_flex_backend_info(device)
    return {
        "backend": backend,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

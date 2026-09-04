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
Opt-in FlashAttention-4 (``flash_attn.cute``) inference backend for the NeMo ``TransformerEncoder``.

The default encoder backend stays FlexAttention. Selecting ``"fa4_cute"`` replaces the per-layer
FlexAttention call with a single ``flash_attn.cute.interface.flash_attn_varlen_func`` invocation on the
*dense* padded ``(B, T, H, D)`` Q/K/V views, using post-subsampling per-sample key lengths as ``seqused_k``.
That is exactly the semantics of the encoder's current padding mask (``kv_idx < lengths[b]``: keys are
masked, queries are not), so no token packing/unpacking is introduced and no block mask is built.

The backend is one common implementation for the Blackwell compute-capability families 10.x, 11.x and 12.x
(SM100/SM110/SM120), matching what the installed FA4 wrapper declares for BF16. There is no device-name
branch and no architecture-specific tuning. Everything else -- other devices, other dtypes, relative-position
score modification, causal/block-sparse modes, autograd/training use, malformed lengths -- fails closed with
an actionable error. This path never silently falls back to FlexAttention.

``torch.compile`` compatibility: calling the FA4 python wrapper directly from a ``fullgraph=True`` region
fails, so the call is wrapped in a ``torch.library.custom_op`` with a registered fake implementation. Dynamo
then sees one opaque operation and the fixed-shape graph is preserved. No autograd formula is registered:
this is an inference-only path.

``flash_attn`` is imported lazily, on the first executed FA4 attention call only, so importing NeMo (or this
module) on a CPU-only or minimal install neither imports nor requires it.

This module also hosts the shared backend-selection surface (:data:`SUPPORTED_ATTENTION_BACKENDS`,
:func:`validate_attention_backend`, :func:`configure_attention_backend`, :func:`attention_backend_info`,
:func:`attention_backend_cache_identity`) for every opt-in backend, including the FP8 FlexAttention path
implemented in ``sortformer_fp8_flex_attention.py``.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from nemo.collections.asr.parts.utils.sortformer_fp8_flex_attention import (
    FP8_FLEX_BACKEND,
    fp8_flex_backend_info,
)

if TYPE_CHECKING:
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

FLEX_BACKEND = "flex"
FA4_CUTE_BACKEND = "fa4_cute"
SUPPORTED_ATTENTION_BACKENDS = (FLEX_BACKEND, FA4_CUTE_BACKEND, FP8_FLEX_BACKEND)

# Blackwell families declared BF16-supported by the installed ``flash_attn.cute`` interface
# (SM100 / SM110 / SM120). FA4's FP8 path is SM100-only and is deliberately not used here.
SUPPORTED_CAPABILITY_MAJORS = (10, 11, 12)
SUPPORTED_DTYPES = (torch.bfloat16,)
# FA4 tile shapes: head dim must be a multiple of 8 and fit the largest supported tile.
MAX_HEAD_DIM = 128
HEAD_DIM_MULTIPLE = 8

FA4_CUSTOM_OP_NAME = "nemo_sortformer::fa4_cute_attention"

# Resolved by :func:`_resolve_flash_attn_varlen_func` on first executed FA4 call.
_flash_attn_varlen_func = None


def validate_attention_backend(backend: Optional[str]) -> str:
    """Normalize and validate an attention-backend selection.

    Args:
        backend: Requested backend name; ``None`` is accepted as a YAML-friendly alias for the default.

    Returns:
        backend (str): One of :data:`SUPPORTED_ATTENTION_BACKENDS`.
    """
    if backend is None:
        return FLEX_BACKEND
    if not isinstance(backend, str):
        raise ValueError(f"attention_backend must be a string, got {type(backend).__name__}.")
    if backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            f"attention_backend='{backend}' is not supported. " f"Supported backends: {SUPPORTED_ATTENTION_BACKENDS}."
        )
    return backend


def validate_fa4_cute_attention_config(attn_mode: str, self_attention_model: str) -> None:
    """Reject encoder configurations the FA4 backend cannot serve.

    FA4 is used here as plain dense bidirectional attention with an additive-free score: only the
    key-padding contract is expressed, through ``seqused_k``. A ``score_mod`` (Transformer-XL relative
    position) or a causal/block-sparse pattern cannot be expressed that way, so those configurations raise
    instead of falling back to FlexAttention.

    Args:
        attn_mode: Encoder attention pattern, e.g. ``"full"`` or ``"causal"``.
        self_attention_model: Positional-encoding / scoring scheme, e.g. ``"rope"``.
    """
    if attn_mode != "full":
        raise ValueError(
            f"attention_backend='{FA4_CUTE_BACKEND}' supports only dense attn_mode='full', got "
            f"attn_mode='{attn_mode}'. Use attention_backend='{FLEX_BACKEND}' for causal or block-sparse "
            "attention."
        )
    if self_attention_model == "rel_pos":
        raise ValueError(
            f"attention_backend='{FA4_CUTE_BACKEND}' cannot apply the Transformer-XL relative-position "
            "score modification. Use attention_backend='flex', or a self_attention_model without a "
            "score_mod such as 'rope', 'abs_pos' or 'no_pos'."
        )


def validate_fa4_cute_inference_mode(training: bool) -> None:
    """Reject a module that is still in training mode.

    The grad check in :func:`fa4_cute_attention` only catches gradient-bearing inputs, so a module left in
    training mode and executed under ``torch.no_grad()``/``torch.inference_mode()`` would otherwise reach
    the kernel. This path has no autograd formula and is only measured for inference, so training mode
    fails closed here rather than producing a silently non-trainable forward.

    Args:
        training: ``nn.Module.training`` of the layer about to run FA4.
    """
    if training:
        raise RuntimeError(
            f"attention_backend='{FA4_CUTE_BACKEND}' is inference-only and no autograd formula is "
            f"registered for {FA4_CUSTOM_OP_NAME}, but the module is in training mode. Call .eval() before "
            f"running FA4 attention, or use attention_backend='{FLEX_BACKEND}' for training."
        )


def prepare_fa4_seqused_k(length: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Convert post-subsampling lengths into the ``seqused_k`` tensor FA4 expects.

    Called once per encoder forward (not once per layer): every layer of a stack shares the same key
    lengths, so the int64 -> int32 conversion and the contiguity copy are paid a single time.

    The value-range check needs a device-to-host read, so it is skipped while tracing under
    ``torch.compile``; it is a static structural property of the caller (post-subsampling lengths never
    exceed the padded time axis) rather than something a traced graph can branch on.

    Args:
        length: ``(B,)`` per-sample valid key counts, any integer dtype, on the encoder's device.
        seq_len: Padded time dimension ``T`` of the Q/K/V tensors. A ``torch.SymInt`` is accepted so that a
            dynamic-shape ``torch.compile`` region can call this without specializing the time axis.

    Returns:
        seqused_k (torch.Tensor): Contiguous ``(B,)`` int32 tensor on ``length``'s device.
    """
    if not isinstance(length, torch.Tensor):
        raise ValueError(f"length must be a torch.Tensor, got {type(length).__name__}.")
    if length.dim() != 1:
        raise ValueError(f"length must be rank 1 (batch,), got shape {tuple(length.shape)}.")
    if length.is_floating_point():
        raise ValueError(f"length must be an integer tensor, got dtype {length.dtype}.")
    if not isinstance(seq_len, (int, torch.SymInt)):
        raise ValueError(f"seq_len must be a positive int, got {type(seq_len).__name__}.")

    seqused_k = length.to(dtype=torch.int32).contiguous()
    if not torch.compiler.is_compiling():
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive int, got {seq_len}.")
        if seqused_k.numel() > 0:
            minimum = int(seqused_k.min())
            maximum = int(seqused_k.max())
            if minimum < 0 or maximum > seq_len:
                raise ValueError(
                    f"seqused_k values must lie in [0, {seq_len}] (the padded time dimension), got range "
                    f"[{minimum}, {maximum}]."
                )
    return seqused_k


def fa4_cute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Dense BF16 FA4 attention over padded ``(B, T, H, D)`` tensors with per-sample key lengths.

    All inputs are validated here, in python, so the failure is actionable in eager mode and every check is
    a static property that Dynamo can fold away while tracing. The FA4 kernel itself is reached through the
    registered custom operator, which stays opaque to Dynamo.

    Args:
        query: ``(B, T, H, D)`` BF16 CUDA tensor; only the last dimension needs to be contiguous.
        key: ``(B, T, H, D)``, same dtype/device/shape as ``query``.
        value: ``(B, T, H, D)``, same dtype/device/shape as ``query``.
        seqused_k: Contiguous ``(B,)`` int32 tensor of valid key counts, see :func:`prepare_fa4_seqused_k`.
        softmax_scale: Score scale; defaults to ``D ** -0.5``, matching FlexAttention.

    Returns:
        out (torch.Tensor): Contiguous ``(B, T, H, D)`` BF16 tensor. Rows of fully padded samples
        (``seqused_k == 0``) are finite zeros.
    """
    _validate_fa4_inputs(query, key, value, seqused_k)
    head_dim = query.shape[-1]
    scale = float(head_dim**-0.5) if softmax_scale is None else float(softmax_scale)
    return torch.ops.nemo_sortformer.fa4_cute_attention(query, key, value, seqused_k, scale)


def configure_attention_backend(diar_model: "SortformerEncLabelModel", backend: Optional[str]) -> Dict[str, str]:
    """Apply an attention backend to every ``TransformerEncoder`` present on a Sortformer model.

    Selection is orthogonal to the quantization recipe: a BF16 run and an NVFP4 run may both select
    ``fa4_cute``. Call this before quantization and before ``torch.compile`` so the compiled graph already
    contains the selected attention path.

    Args:
        diar_model (SortformerEncLabelModel): Restored diarization model.
        backend: Requested backend name, see :func:`validate_attention_backend`.

    Returns:
        applied (Dict[str, str]): Attribute name -> backend, for every module the backend was applied to.
        Modules that are not attention-backend selectable (e.g. a Conformer front end) are skipped; a
        non-default backend with nothing to apply to raises instead.
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

    Every backend is numerically close to, but not identical with, the default FlexAttention path, so their
    predictions must not share a cache entry. The default backend maps to ``None`` so that caches written
    before this option existed keep matching a default run, mirroring the quantization cache-identity
    convention.

    Args:
        backend: Requested backend name.

    Returns:
        identity (Optional[str]): Backend name, or ``None`` for the default FlexAttention backend.
    """
    backend = validate_attention_backend(backend)
    return None if backend == FLEX_BACKEND else backend


def attention_backend_info(backend: Optional[str], device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Collect the reproducibility facts of whichever attention backend was selected.

    Each backend reports its own facts -- FA4 must never be described for an FP8 FlexAttention run, and the
    FP8 path must not import ``flash_attn`` just to be logged. Never raises.

    Args:
        backend: Requested backend name, see :func:`validate_attention_backend`.
        device: CUDA device to report on. Defaults to the current device when CUDA is available.

    Returns:
        info (Dict[str, Any]): Backend-specific facts; always contains ``"backend"``.
    """
    backend = validate_attention_backend(backend)
    if backend == FA4_CUTE_BACKEND:
        return fa4_cute_backend_info(device)
    if backend == FP8_FLEX_BACKEND:
        return fp8_flex_backend_info(device)
    return {
        "backend": backend,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def fa4_cute_backend_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Collect the backend facts worth logging for reproducibility.

    Never raises and never imports ``flash_attn`` as a side effect beyond the version probe; unavailable
    facts are reported as ``None`` so this can be logged before any FA4 call has run.

    Args:
        device: CUDA device to report on. Defaults to the current device when CUDA is available.

    Returns:
        info (Dict[str, Any]): Torch/CUDA/device/FA4 facts.
    """
    info: Dict[str, Any] = {
        "backend": FA4_CUTE_BACKEND,
        "custom_op": FA4_CUSTOM_OP_NAME,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "compute_capability": None,
        "flash_attn_version": None,
    }
    if torch.cuda.is_available():
        try:
            index = device.index if isinstance(device, torch.device) and device.index is not None else None
            info["device_name"] = torch.cuda.get_device_name(index)
            info["compute_capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability(index))
        except RuntimeError:  # pragma: no cover - depends on the runtime CUDA state
            pass
    try:
        import flash_attn

        info["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        pass
    return info


@torch.library.custom_op(FA4_CUSTOM_OP_NAME, mutates_args=())
def _fa4_cute_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Runtime implementation: one direct ``flash_attn.cute`` call on the dense padded views.

    Registered as a custom operator so that Dynamo treats the whole FA4 invocation as a single opaque
    operation instead of tracing into the wrapper (which skips ``active_fake_mode`` and therefore breaks
    ``fullgraph=True``).
    """
    _validate_fa4_device(query)
    varlen_func = _resolve_flash_attn_varlen_func()
    out = varlen_func(query, key, value, seqused_k=seqused_k, softmax_scale=softmax_scale, causal=False)
    # The installed wrapper may return ``(out, lse)`` even when the LSE was not requested.
    if isinstance(out, (tuple, list)):
        out = out[0]
    # Keep the returned layout equal to the fake implementation's contiguous allocation.
    return out.contiguous()


@_fa4_cute_attention_op.register_fake
def _fa4_cute_attention_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Shape/dtype/layout of the real output: a contiguous ``(B, T, H, D)`` tensor like ``query``."""
    return query.new_empty(query.shape)


def _validate_fa4_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seqused_k: torch.Tensor,
) -> None:
    """Fail-closed contract check for :func:`fa4_cute_attention`."""
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be rank 4 (batch, time, heads, head_dim), got {tuple(tensor.shape)}.")
        if tensor.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"{FA4_CUTE_BACKEND} attention supports {SUPPORTED_DTYPES} only, got {name} dtype "
                f"{tensor.dtype}. Run the model in bfloat16 or use attention_backend='{FLEX_BACKEND}'."
            )
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous along its last dimension, got strides {tensor.stride()}.")
        if tensor.shape != query.shape:
            raise ValueError(
                f"query, key and value must share one shape; got {name} {tuple(tensor.shape)} against query "
                f"{tuple(query.shape)}."
            )
        if tensor.device != query.device:
            raise ValueError(f"{name} is on device {tensor.device} but query is on {query.device}.")

    head_dim = query.shape[-1]
    if head_dim > MAX_HEAD_DIM or head_dim % HEAD_DIM_MULTIPLE != 0:
        raise ValueError(
            f"{FA4_CUTE_BACKEND} attention requires a head dim that is a multiple of {HEAD_DIM_MULTIPLE} and at "
            f"most {MAX_HEAD_DIM}, got {head_dim}."
        )

    if not isinstance(seqused_k, torch.Tensor):
        raise ValueError(f"seqused_k must be a torch.Tensor, got {type(seqused_k).__name__}.")
    if seqused_k.dtype != torch.int32:
        raise ValueError(f"seqused_k must be int32, got {seqused_k.dtype}. Use prepare_fa4_seqused_k().")
    if seqused_k.dim() != 1 or seqused_k.shape[0] != query.shape[0]:
        raise ValueError(
            f"seqused_k must have shape ({query.shape[0]},) matching the query batch, got "
            f"{tuple(seqused_k.shape)}."
        )
    if not seqused_k.is_contiguous():
        raise ValueError("seqused_k must be contiguous. Use prepare_fa4_seqused_k().")
    if seqused_k.device != query.device:
        raise ValueError(f"seqused_k is on device {seqused_k.device} but query is on {query.device}.")

    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        raise RuntimeError(
            f"{FA4_CUTE_BACKEND} attention is inference-only: no autograd formula is registered for "
            f"{FA4_CUSTOM_OP_NAME}. Run inside torch.no_grad()/torch.inference_mode(), or use "
            f"attention_backend='{FLEX_BACKEND}' for training."
        )

    # Checked last so that the structural contract above is reportable on any device.
    _validate_fa4_device(query)


def _validate_fa4_device(query: torch.Tensor) -> None:
    """Reject non-CUDA tensors and non-Blackwell devices, naming the measured support envelope."""
    if not query.is_cuda:
        raise RuntimeError(
            f"{FA4_CUTE_BACKEND} attention requires CUDA tensors, got device {query.device}. Use "
            f"attention_backend='{FLEX_BACKEND}' on CPU."
        )
    major, minor = torch.cuda.get_device_capability(query.device)
    if major not in SUPPORTED_CAPABILITY_MAJORS:
        raise RuntimeError(
            f"{FA4_CUTE_BACKEND} attention requires a Blackwell device with compute capability major in "
            f"{SUPPORTED_CAPABILITY_MAJORS}, got {major}.{minor} on "
            f"'{torch.cuda.get_device_name(query.device)}'. Use attention_backend='{FLEX_BACKEND}'."
        )


def _resolve_flash_attn_varlen_func():
    """Import the FA4 wrapper on first use; keeps NeMo importable without ``flash_attn`` installed."""
    global _flash_attn_varlen_func
    if _flash_attn_varlen_func is None:
        try:
            from flash_attn.cute.interface import flash_attn_varlen_func
        except ImportError as err:  # pragma: no cover - depends on the install
            raise RuntimeError(
                f"attention_backend='{FA4_CUTE_BACKEND}' requires the flash_attn package providing "
                "flash_attn.cute.interface.flash_attn_varlen_func, which is not installed. Install "
                f"FlashAttention 4 or use attention_backend='{FLEX_BACKEND}'."
            ) from err
        _flash_attn_varlen_func = flash_attn_varlen_func
    return _flash_attn_varlen_func

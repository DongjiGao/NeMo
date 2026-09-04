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
Opt-in FP8 FlexAttention (Triton) inference backend for the NeMo ``TransformerEncoder``.

Selecting ``"fp8_flex"`` keeps the encoder's FlexAttention semantics -- the same per-sample key-padding
block mask (``kv_idx < lengths[b]``) built once by ``TransformerEncoder.forward_internal`` and shared by
every layer -- and only changes the numeric format the kernel runs in: the post-QK-norm, post-RoPE BF16
Q/K/V tensors are cast to ``torch.float8_e4m3fn`` in the layouts PyTorch's FP8 FlexAttention kernel requires
(Q/K row-major, V column-major along the time axis), FlexAttention is invoked with its Triton backend
explicitly selected, and the result is converted back to BF16 before the unchanged output projection.

The attention kernel is PyTorch's own FlexAttention/Triton lowering. This module adds no Triton, CUDA,
CUTLASS or FlashAttention kernel of its own and no new dependency; the only numeric step is a direct E4M3
cast with FlexAttention's default ``1/sqrt(D)`` score scale -- no dynamic amax pass and no static
calibration contract. Final DER is the accuracy decision for that choice, so no scaling scheme is smuggled
in here.

What this module *does* add is a compilation boundary. The cast/layout work plus the FlexAttention call are
compiled once, on their own, as the private root :func:`_fp8_flex_attention_boundary`
(``fullgraph=True, dynamic=False``), and are reached from the encoder through a single
``torch.library.custom_op`` (:data:`FP8_FLEX_CUSTOM_OP_NAME`). The operator is opaque to the *outer* encoder
compiler, so the 31 encoder layers no longer inline 31 copies of the FP8 attention region into the outer
graph; it is not opaque to PyTorch, which still lowers the attention itself. Because a custom operator takes
only tensors and scalars, every piece of per-batch state crosses that boundary explicitly: the eight
``BlockMask`` tensor fields, the two sequence lengths, the two block sizes, and the ``(B,)`` per-sample valid
key lengths. The ``BlockMask`` python wrapper -- including the ``kv_idx < valid_lengths[b]`` padding
``mask_mod``, which keeps a non-block-aligned final valid key exact -- is rebuilt inside the compiled root
from exactly those arguments. No global mask, no thread-local state, no closure-identity trick, no cache
keyed on a python object, and no data-dependent device-to-host read in the hot path.

The backend is one common implementation for the Blackwell compute-capability families 10.x, 11.x and 12.x
(SM100/SM103, SM110, SM120). There is no device-name check, no minor-version dispatch and no
architecture-specific tuning branch. Everything else -- other devices, other dtypes, relative-position score
modification, causal/block-sparse modes, autograd/training use, a missing or malformed block mask, malformed
lengths -- fails closed with an actionable error, and a compiler or runtime failure on an admitted device
surfaces to the caller instead of silently falling back to BF16 FlexAttention.

``FLASH`` is never selected as the FlexAttention backend for this path: on SM120 the FP8 FLASH lowering
fails an output-dtype assertion, so the only accepted option contract is the explicit
``{"BACKEND": "TRITON"}`` selection.
"""

from typing import Any, Dict, Optional

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

FP8_FLEX_BACKEND = "fp8_flex"

FP8_DTYPE = torch.float8_e4m3fn
# Only the BF16 residual stream this encoder runs inference in is admitted; the cast target is fixed.
SUPPORTED_INPUT_DTYPES = (torch.bfloat16,)
# Blackwell families this one implementation targets (SM100/SM103, SM110, SM120).
SUPPORTED_CAPABILITY_MAJORS = (10, 11, 12)
# The FP8 Triton dot path needs a head dim that tiles into 16-element K blocks and fits the largest tile.
MIN_HEAD_DIM = 16
MAX_HEAD_DIM = 128
HEAD_DIM_MULTIPLE = 16
# The only verified kernel-option contract on the pinned stack. No speculative tuning flags.
FLEX_KERNEL_OPTIONS = {"BACKEND": "TRITON"}

FP8_FLEX_CUSTOM_OP_NAME = "nemo_sortformer::fp8_flex_attention"

# The ``BlockMask`` tensor fields carried across the operator boundary, in ``BlockMask``'s own order.
BLOCK_MASK_TENSOR_FIELDS = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)
# Only the first pair is required to evaluate the forward pass; the remaining three pairs are optional
# ``BlockMask`` optimizations/back-pass tensors and are forwarded verbatim when present.
REQUIRED_BLOCK_MASK_TENSOR_FIELDS = ("kv_num_blocks", "kv_indices")
# Pairs that ``BlockMask`` itself requires to be both present or both absent.
_PAIRED_BLOCK_MASK_TENSOR_FIELDS = (
    ("full_kv_num_blocks", "full_kv_indices"),
    ("q_num_blocks", "q_indices"),
    ("full_q_num_blocks", "full_q_indices"),
)

_MASK_SOURCE_HINT = (
    "The encoder must pass the create_block_mask() result built by TransformerEncoder.forward_internal "
    "into every attention layer."
)


def validate_fp8_flex_attention_config(attn_mode: str, self_attention_model: str) -> None:
    """Reject encoder configurations the FP8 FlexAttention backend cannot serve.

    Only dense bidirectional attention with an unmodified score is verified in FP8: the key-padding contract
    is carried by the caller's block mask, and nothing else is claimed. A ``score_mod`` (Transformer-XL
    relative position) mixes a BF16 bias into FP8 scores, and causal/block-sparse patterns are unmeasured, so
    both raise instead of falling back to FlexAttention in BF16.

    Args:
        attn_mode: Encoder attention pattern, e.g. ``"full"`` or ``"causal"``.
        self_attention_model: Positional-encoding / scoring scheme, e.g. ``"rope"``.
    """
    if attn_mode != "full":
        raise ValueError(
            f"attention_backend='{FP8_FLEX_BACKEND}' supports only dense attn_mode='full', got "
            f"attn_mode='{attn_mode}'. Use attention_backend='flex' for causal or block-sparse attention."
        )
    if self_attention_model == "rel_pos":
        raise ValueError(
            f"attention_backend='{FP8_FLEX_BACKEND}' cannot apply the Transformer-XL relative-position "
            "score modification. Use attention_backend='flex', or a self_attention_model without a "
            "score_mod such as 'rope', 'abs_pos' or 'no_pos'."
        )


def validate_fp8_flex_inference_mode(training: bool) -> None:
    """Reject a module that is still in training mode.

    The grad check in :func:`fp8_flex_attention` only catches gradient-bearing inputs, so a module left in
    training mode and executed under ``torch.no_grad()``/``torch.inference_mode()`` would otherwise reach the
    kernel and silently train through a lossy E4M3 cast. This path is only measured for inference and no
    autograd formula is registered for :data:`FP8_FLEX_CUSTOM_OP_NAME`, so training mode fails closed here.

    Args:
        training: ``nn.Module.training`` of the layer about to run FP8 FlexAttention.
    """
    if training:
        raise RuntimeError(
            f"attention_backend='{FP8_FLEX_BACKEND}' is inference-only: attention runs on a lossy "
            "float8_e4m3fn cast of Q/K/V and no training use is verified, but the module is in training "
            "mode. Call .eval() before running FP8 FlexAttention, or use attention_backend='flex' for "
            "training."
        )


def prepare_fp8_flex_valid_lengths(length: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Convert post-subsampling lengths into the per-sample valid-key tensor this backend expects.

    Called once per encoder forward (not once per layer): every layer of a stack shares the same key
    lengths, so the int64 -> int32 conversion and the contiguity copy are paid a single time, and every
    layer's operator call then receives the identical tensor.

    The value-range check needs a device-to-host read, so it is skipped while tracing under
    ``torch.compile``; it is a static structural property of the caller (post-subsampling lengths never
    exceed the padded time axis) rather than something a traced graph can branch on.

    Args:
        length: ``(B,)`` per-sample valid key counts, any integer dtype, on the encoder's device.
        seq_len: Padded time dimension ``T`` of the Q/K/V tensors. A ``torch.SymInt`` is accepted so that a
            dynamic-shape ``torch.compile`` region can call this without specializing the time axis.

    Returns:
        valid_lengths (torch.Tensor): Contiguous ``(B,)`` int32 tensor on ``length``'s device.
    """
    if not isinstance(length, torch.Tensor):
        raise ValueError(f"length must be a torch.Tensor, got {type(length).__name__}.")
    if length.dim() != 1:
        raise ValueError(f"length must be rank 1 (batch,), got shape {tuple(length.shape)}.")
    if length.is_floating_point():
        raise ValueError(f"length must be an integer tensor, got dtype {length.dtype}.")
    if not isinstance(seq_len, (int, torch.SymInt)):
        raise ValueError(f"seq_len must be a positive int, got {type(seq_len).__name__}.")

    valid_lengths = length.to(dtype=torch.int32).contiguous()
    if not torch.compiler.is_compiling():
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive int, got {seq_len}.")
        if valid_lengths.numel() > 0:
            minimum = int(valid_lengths.min())
            maximum = int(valid_lengths.max())
            if minimum < 0 or maximum > seq_len:
                raise ValueError(
                    f"valid length values must lie in [0, {seq_len}] (the padded time dimension), got range "
                    f"[{minimum}, {maximum}]."
                )
    return valid_lengths


def fp8_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Optional[BlockMask],
    valid_lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    """FP8 FlexAttention over ``(B, H, T, D)`` BF16 Q/K/V with the encoder's exact block mask.

    Consumes the post-QK-norm, post-RoPE tensors the default FlexAttention path would consume, and returns
    the same ``(B, H, T, D)`` BF16 layout, so the caller's layout conversion and output projection stay
    unchanged. All inputs are validated in python, so failures are actionable in eager mode, and every check
    is a static property (shape, dtype, stride, device, block-mask structure) that Dynamo folds away while
    tracing -- there is no data-dependent host read in the hot path.

    The mask is *decomposed* here and rebuilt inside the compiled boundary: only tensors and ints cross the
    ``torch.library.custom_op`` seam, so nothing depends on the identity of the caller's ``BlockMask`` object
    or of the calling layer.

    Args:
        query: ``(B, H, T, D)`` BF16 CUDA tensor; only the head dimension needs to be contiguous.
        key: ``(B, H, T, D)``, same dtype/device/shape as ``query``.
        value: ``(B, H, T, D)``, same dtype/device/shape as ``query``.
        block_mask: The ``BlockMask`` built by the encoder for this batch. Required: the key-padding contract
            is never approximated by a dense/no-mask call.
        valid_lengths: Contiguous ``(B,)`` int32 tensor of per-sample valid key counts describing the very
            same padding as ``block_mask``, see :func:`prepare_fp8_flex_valid_lengths`. Required: the
            operator rebuilds the mask's ``mask_mod`` from it.

    Returns:
        out (torch.Tensor): Contiguous ``(B, H, T, D)`` tensor in ``query``'s dtype.
    """
    _validate_fp8_flex_inputs(query, key, value, block_mask, valid_lengths)

    q_len, kv_len = block_mask.seq_lengths[0], block_mask.seq_lengths[1]
    q_block_size, kv_block_size = block_mask.BLOCK_SIZE[0], block_mask.BLOCK_SIZE[1]
    return torch.ops.nemo_sortformer.fp8_flex_attention(
        query,
        key,
        value,
        valid_lengths,
        getattr(block_mask, "kv_num_blocks", None),
        getattr(block_mask, "kv_indices", None),
        getattr(block_mask, "full_kv_num_blocks", None),
        getattr(block_mask, "full_kv_indices", None),
        getattr(block_mask, "q_num_blocks", None),
        getattr(block_mask, "q_indices", None),
        getattr(block_mask, "full_q_num_blocks", None),
        getattr(block_mask, "full_q_indices", None),
        q_len,
        kv_len,
        q_block_size,
        kv_block_size,
    )


def fp8_flex_backend_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Collect the backend facts worth logging for reproducibility.

    Never raises and never imports an optional dependency; unavailable facts are reported as ``None`` so this
    can be logged before any FP8 attention call has run.

    Args:
        device: CUDA device to report on. Defaults to the current device when CUDA is available.

    Returns:
        info (Dict[str, Any]): Torch/CUDA/device facts plus the FP8 format, the kernel-selection contract and
        the custom-operator boundary. ``attention_kernel`` names PyTorch's own FlexAttention/Triton lowering:
        the operator is a compilation boundary, not a hand-written GPU kernel.
    """
    info: Dict[str, Any] = {
        "backend": FP8_FLEX_BACKEND,
        "fp8_dtype": str(FP8_DTYPE),
        "flex_kernel_options": dict(FLEX_KERNEL_OPTIONS),
        "supported_input_dtypes": [str(dtype) for dtype in SUPPORTED_INPUT_DTYPES],
        "supported_capability_majors": list(SUPPORTED_CAPABILITY_MAJORS),
        "inference_only": True,
        "custom_op": FP8_FLEX_CUSTOM_OP_NAME,
        "custom_op_role": "compilation boundary opaque to the outer encoder compiler",
        "attention_kernel": "torch.nn.attention.flex_attention (Triton backend)",
        "custom_attention_kernel": False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "compute_capability": None,
    }
    if torch.cuda.is_available():
        try:
            index = device.index if isinstance(device, torch.device) and device.index is not None else None
            info["device_name"] = torch.cuda.get_device_name(index)
            info["compute_capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability(index))
        except RuntimeError:  # pragma: no cover - depends on the runtime CUDA state
            pass
    return info


@torch.library.custom_op(FP8_FLEX_CUSTOM_OP_NAME, mutates_args=())
def _fp8_flex_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_lengths: torch.Tensor,
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: Optional[torch.Tensor],
    full_kv_indices: Optional[torch.Tensor],
    q_num_blocks: Optional[torch.Tensor],
    q_indices: Optional[torch.Tensor],
    full_q_num_blocks: Optional[torch.Tensor],
    full_q_indices: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    q_block_size: int,
    kv_block_size: int,
) -> torch.Tensor:
    """Runtime implementation: one call into the separately compiled FP8 FlexAttention boundary.

    Registered as a custom operator so the outer encoder graph sees a single opaque node per layer instead
    of inlining the cast/layout/attention region 31 times. Dispatch itself costs about 0.06 ms per layer at
    the production shape, and the operator's output is bit-identical to calling the compiled root directly.
    """
    _validate_fp8_flex_device(query)
    return _fp8_flex_attention_compiled(
        query,
        key,
        value,
        valid_lengths,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        q_len,
        kv_len,
        q_block_size,
        kv_block_size,
    )


@_fp8_flex_attention_op.register_fake
def _fp8_flex_attention_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_lengths: torch.Tensor,
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: Optional[torch.Tensor],
    full_kv_indices: Optional[torch.Tensor],
    q_num_blocks: Optional[torch.Tensor],
    q_indices: Optional[torch.Tensor],
    full_q_num_blocks: Optional[torch.Tensor],
    full_q_indices: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    q_block_size: int,
    kv_block_size: int,
) -> torch.Tensor:
    """Shape/dtype/device/layout of the real output: a contiguous ``(B, H, T, D)`` tensor like ``query``.

    ``query`` is BF16 by contract and the boundary converts the FP8 accumulation back to that dtype, so the
    real output metadata is exactly ``query``'s. No autograd formula is registered: this is inference-only.
    """
    return query.new_empty(query.shape)


def _fp8_flex_attention_boundary(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_lengths: torch.Tensor,
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: Optional[torch.Tensor],
    full_kv_indices: Optional[torch.Tensor],
    q_num_blocks: Optional[torch.Tensor],
    q_indices: Optional[torch.Tensor],
    full_q_num_blocks: Optional[torch.Tensor],
    full_q_indices: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    q_block_size: int,
    kv_block_size: int,
) -> torch.Tensor:
    """The single region this backend compiles: cast, layout, mask reconstruction and one FlexAttention call.

    Everything it needs arrives as a tensor or an int, because that is all a ``torch.library.custom_op`` can
    carry. The ``BlockMask`` python wrapper is rebuilt here from the caller's own eight tensor fields, its
    sequence lengths and its block sizes; only the ``mask_mod`` closure has to be recreated, and it is
    recreated as exactly the encoder's key-padding predicate ``kv_idx < valid_lengths[b]``, which stays exact
    for a final valid key that does not land on a block boundary.

    Deliberately a dedicated module-level function rather than ``torch.compile(flex_attention)``: Dynamo keys
    its compiled-code cache on the code object of the compiled frame, so wrapping ``flex_attention`` itself
    would share one cache -- and one recompile budget -- with the encoder's default
    ``flex_attention_compiled`` and every other FlexAttention wrapper in the process. Once that budget is
    exhausted the frame silently falls back to eager, where ``kernel_options`` (and therefore the mandatory
    Triton backend selection) is ignored.
    """
    # Q/K row-major: contiguous over ``(B, H, T, D)``.
    query_fp8 = query.to(FP8_DTYPE).contiguous()
    key_fp8 = key.to(FP8_DTYPE).contiguous()
    # V in the column-major attention layout the FP8 kernel reads: shape ``(B, H, T, D)`` with stride 1 along
    # time and stride T along the head dim, built by the usual transpose-contiguous-transpose.
    value_fp8 = value.to(FP8_DTYPE).transpose(-2, -1).contiguous().transpose(-2, -1)

    block_mask = BlockMask(
        seq_lengths=(q_len, kv_len),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=(q_block_size, kv_block_size),
        mask_mod=_make_key_padding_mask_mod(valid_lengths),
    )
    out = flex_attention(
        query_fp8,
        key_fp8,
        value_fp8,
        block_mask=block_mask,
        kernel_options=dict(FLEX_KERNEL_OPTIONS),
    )
    # ``.contiguous()`` keeps the returned layout equal to the fake implementation's allocation.
    return out.to(query.dtype).contiguous()


# ``kernel_options`` -- and therefore the explicit Triton backend selection -- is only honoured on
# FlexAttention's compiled path, so the boundary is compiled here. ``fullgraph=True`` because a graph break
# inside this region would drop back to eager and silently discard the backend selection; ``dynamic=False``
# because this backend is inference-only and measured at one fixed shape. This compile is entered from the
# custom operator's runtime implementation, i.e. from outside whatever graph the encoder is compiled into.
_fp8_flex_attention_compiled = torch.compile(_fp8_flex_attention_boundary, fullgraph=True, dynamic=False)


def _make_key_padding_mask_mod(valid_lengths: torch.Tensor):
    """Rebuild the encoder's ``kv_idx < lengths[b]`` key-padding predicate from an explicit tensor."""

    def key_padding_mask(b, h, q_idx, kv_idx):
        return kv_idx < valid_lengths[b]

    return key_padding_mask


def _validate_fp8_flex_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Optional[BlockMask],
    valid_lengths: Optional[torch.Tensor],
) -> None:
    """Fail-closed contract check for :func:`fp8_flex_attention`."""
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be rank 4 (batch, heads, time, head_dim), got {tuple(tensor.shape)}.")
        if tensor.dtype not in SUPPORTED_INPUT_DTYPES:
            raise ValueError(
                f"{FP8_FLEX_BACKEND} attention supports {SUPPORTED_INPUT_DTYPES} inputs only, got {name} dtype "
                f"{tensor.dtype}. Run the model in bfloat16 or use attention_backend='flex'."
            )
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous along its head dimension, got strides {tensor.stride()}.")
        if tensor.shape != query.shape:
            raise ValueError(
                f"query, key and value must share one shape; got {name} {tuple(tensor.shape)} against query "
                f"{tuple(query.shape)}."
            )
        if tensor.device != query.device:
            raise ValueError(f"{name} is on device {tensor.device} but query is on {query.device}.")

    head_dim = query.shape[-1]
    if head_dim < MIN_HEAD_DIM or head_dim > MAX_HEAD_DIM or head_dim % HEAD_DIM_MULTIPLE != 0:
        raise ValueError(
            f"{FP8_FLEX_BACKEND} attention requires a head dim that is a multiple of {HEAD_DIM_MULTIPLE} and "
            f"lies in [{MIN_HEAD_DIM}, {MAX_HEAD_DIM}], got {head_dim}."
        )

    _validate_fp8_flex_block_mask(block_mask, query)
    _validate_fp8_flex_valid_lengths(valid_lengths, query)

    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        raise RuntimeError(
            f"{FP8_FLEX_BACKEND} attention is inference-only: gradients through the float8_e4m3fn cast are "
            f"not verified and no autograd formula is registered for {FP8_FLEX_CUSTOM_OP_NAME}. Run inside "
            "torch.no_grad()/torch.inference_mode(), or use attention_backend='flex' for training."
        )

    # Checked last so that the structural contract above is reportable on any device.
    _validate_fp8_flex_device(query)


def _validate_fp8_flex_block_mask(block_mask: Optional[BlockMask], query: torch.Tensor) -> None:
    """Check every ``BlockMask`` field this backend decomposes across the operator boundary.

    Structural only: dtype, shape, device and presence. Nothing here reads a mask value, so the whole check
    folds away under ``torch.compile``.
    """
    if block_mask is None:
        raise ValueError(
            f"attention_backend='{FP8_FLEX_BACKEND}' requires the encoder's key-padding block mask; padding "
            f"is never approximated by a dense attention call. {_MASK_SOURCE_HINT}"
        )

    fields = {name: getattr(block_mask, name, None) for name in BLOCK_MASK_TENSOR_FIELDS}
    for name in REQUIRED_BLOCK_MASK_TENSOR_FIELDS:
        if not isinstance(fields[name], torch.Tensor):
            raise ValueError(
                f"block_mask.{name} must be a torch.Tensor for {FP8_FLEX_BACKEND} attention, got "
                f"{type(fields[name]).__name__}. {_MASK_SOURCE_HINT}"
            )
    for first, second in _PAIRED_BLOCK_MASK_TENSOR_FIELDS:
        if (fields[first] is None) != (fields[second] is None):
            raise ValueError(
                f"block_mask.{first} and block_mask.{second} must be both present or both absent, got "
                f"{type(fields[first]).__name__} and {type(fields[second]).__name__}. {_MASK_SOURCE_HINT}"
            )
    for name, tensor in fields.items():
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"block_mask.{name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.is_floating_point():
            raise ValueError(f"block_mask.{name} must be an integer tensor, got dtype {tensor.dtype}.")
        if tensor.device != query.device:
            raise ValueError(f"block_mask.{name} is on device {tensor.device} but query is on {query.device}.")

    kv_indices = fields["kv_indices"]
    if kv_indices.dim() < 2:
        raise ValueError(f"block_mask.kv_indices must have at least 2 dimensions, got {tuple(kv_indices.shape)}.")
    batch = query.shape[0]
    mask_batch = kv_indices.shape[0] if kv_indices.dim() > 2 else 1
    if mask_batch not in (1, batch):
        raise ValueError(
            f"block_mask must broadcast over the query batch {batch}, but its batch dimension is {mask_batch}."
        )

    seq_lengths = _validate_int_pair(getattr(block_mask, "seq_lengths", None), "seq_lengths")
    if seq_lengths != (query.shape[-2], query.shape[-2]):
        raise ValueError(
            f"block_mask.seq_lengths {seq_lengths} must equal the (Q_LEN, KV_LEN) of the attention inputs "
            f"{(query.shape[-2], query.shape[-2])}; this backend never re-adjusts or crops the caller's mask."
        )
    block_size = _validate_int_pair(getattr(block_mask, "BLOCK_SIZE", None), "BLOCK_SIZE")
    if block_size[0] <= 0 or block_size[1] <= 0:
        raise ValueError(f"block_mask.BLOCK_SIZE must be two positive ints, got {block_size}.")


def _validate_fp8_flex_valid_lengths(valid_lengths: Optional[torch.Tensor], query: torch.Tensor) -> None:
    """Check the per-sample valid-key tensor the operator rebuilds the padding ``mask_mod`` from."""
    if valid_lengths is None:
        raise ValueError(
            f"attention_backend='{FP8_FLEX_BACKEND}' requires per-sample valid key lengths alongside the "
            "block mask; the operator rebuilds the key-padding mask_mod from them. The encoder must pass "
            "them into every attention layer, see prepare_fp8_flex_valid_lengths()."
        )
    if not isinstance(valid_lengths, torch.Tensor):
        raise ValueError(f"valid_lengths must be a torch.Tensor, got {type(valid_lengths).__name__}.")
    if valid_lengths.dtype != torch.int32:
        raise ValueError(
            f"valid_lengths must be int32, got {valid_lengths.dtype}. Use prepare_fp8_flex_valid_lengths()."
        )
    if valid_lengths.dim() != 1 or valid_lengths.shape[0] != query.shape[0]:
        raise ValueError(
            f"valid_lengths must have shape ({query.shape[0]},) matching the query batch, got "
            f"{tuple(valid_lengths.shape)}."
        )
    if not valid_lengths.is_contiguous():
        raise ValueError("valid_lengths must be contiguous. Use prepare_fp8_flex_valid_lengths().")
    if valid_lengths.device != query.device:
        raise ValueError(f"valid_lengths is on device {valid_lengths.device} but query is on {query.device}.")
    # A value check needs a device-to-host read, so it is only done in eager mode: under torch.compile it
    # would be a data-dependent read in the hot path.
    if not torch.compiler.is_compiling() and valid_lengths.numel() > 0:
        seq_len = query.shape[-2]
        minimum = int(valid_lengths.min())
        maximum = int(valid_lengths.max())
        if minimum < 0 or maximum > seq_len:
            raise ValueError(
                f"valid_lengths values must lie in [0, {seq_len}] (the padded time dimension), got range "
                f"[{minimum}, {maximum}]."
            )


def _validate_int_pair(value: Any, name: str) -> tuple:
    """Normalize a ``BlockMask`` ``(int, int)`` context field, rejecting anything else."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"block_mask.{name} must be a pair of ints, got {value!r}. {_MASK_SOURCE_HINT}")
    for part in value:
        if isinstance(part, bool) or not isinstance(part, (int, torch.SymInt)):
            raise ValueError(f"block_mask.{name} must be a pair of ints, got {value!r}. {_MASK_SOURCE_HINT}")
    return (value[0], value[1])


def _validate_fp8_flex_device(query: torch.Tensor) -> None:
    """Reject non-CUDA tensors and non-Blackwell devices, naming the measured support envelope."""
    if not query.is_cuda:
        raise RuntimeError(
            f"{FP8_FLEX_BACKEND} attention requires CUDA tensors, got device {query.device}. Use "
            "attention_backend='flex' on CPU."
        )
    major, minor = torch.cuda.get_device_capability(query.device)
    if major not in SUPPORTED_CAPABILITY_MAJORS:
        raise RuntimeError(
            f"{FP8_FLEX_BACKEND} attention requires a Blackwell device with compute capability major in "
            f"{SUPPORTED_CAPABILITY_MAJORS}, got {major}.{minor} on "
            f"'{torch.cuda.get_device_name(query.device)}'. Use attention_backend='flex'."
        )

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
Fused QKV output boundary of a static-NVFP4 Sortformer attention block.

This module is a *prototype primitive*, not a wired-in runtime path. It fuses exactly the work that currently
follows the native FP4 QKV GEMM:

    raw BF16 FP4-GEMM output -> global scalar rescale -> optional bias -> Q/K/V layout -> RoPE on Q/K

``raw_qkv`` is the unscaled BF16 result of the FP4 GEMM with shape ``(B, T, 3 * H * D)``; ``global_scale`` is the
scalar dequant multiplier that TorchAO applies after ``_scaled_mm``. Both entry points return contiguous
``(B, H, T, D)`` Q, K and V.

Two entry points are provided and implement identical semantics:

* :func:`fused_qkv_rope_boundary_reference` -- pure PyTorch, any device, used as the numerical contract.
* :func:`fused_qkv_rope_boundary_triton` -- one Triton kernel launch on CUDA, BF16 only.

The Triton path never silently falls back to the reference path: every unsupported input raises. Triton is
imported lazily so that importing this module on a CPU-only or minimal install neither imports Triton nor fails.

RoPE semantics match :class:`~nemo.collections.asr.parts.submodules.multi_head_attention.RotaryPositionalEncoding`
with ``cos``/``sin`` of shape ``(T', rotary_dim)`` whose frequencies are already duplicated across the two halves.
Only the no-KV-cache case is covered: position ``t`` of Q and K both read row ``t`` of ``cos``/``sin``.
"""

from typing import Optional, Tuple, Union

import torch

# Rows (i.e. ``(batch, time, head)`` triples) handled by a single Triton program.
_ROWS_PER_BLOCK = 4
_NUM_WARPS = 4

# Populated by :func:`_resolve_triton`; ``_qkv_rope_boundary_kernel`` resolves ``tl`` through module globals at
# JIT-compile time, so these must stay module-level names.
triton = None
tl = None
_compiled_kernel = None


def fused_qkv_rope_boundary_reference(
    raw_qkv: torch.Tensor,
    global_scale: Union[float, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the static-NVFP4 QKV output boundary.

    Args:
        raw_qkv: unscaled GEMM output, shape ``(B, T, 3 * num_heads * head_dim)``, contiguous.
        global_scale: scalar dequant multiplier (python scalar or 1-element tensor).
        cos: RoPE cosine table, shape ``(T' >= T, rotary_dim)``, contiguous.
        sin: RoPE sine table, same shape/dtype/device as ``cos``.
        num_heads: number of attention heads ``H``.
        head_dim: per-head feature dim ``D``.
        bias: optional QKV bias of shape ``(3 * num_heads * head_dim,)``, applied before RoPE.

    Returns:
        Contiguous ``(B, H, T, D)`` tensors ``(q, k, v)`` in ``raw_qkv``'s dtype.
    """
    batch, seq_len, _, rotary_dim = _validate_boundary_inputs(
        raw_qkv, global_scale, cos, sin, num_heads, head_dim, bias
    )

    # Cast a scale tensor to the input dtype: a fp32 scale would otherwise promote Q/K/V to fp32,
    # whereas the Triton path folds the scale in as a fp32 scalar and stores back in the input dtype.
    if isinstance(global_scale, torch.Tensor):
        global_scale = global_scale.to(raw_qkv.dtype)
    qkv = raw_qkv * global_scale
    if bias is not None:
        qkv = qkv + bias
    qkv = qkv.view(batch, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    cos_t = cos[:seq_len].reshape(1, 1, seq_len, rotary_dim).to(q.dtype)
    sin_t = sin[:seq_len].reshape(1, 1, seq_len, rotary_dim).to(q.dtype)
    q = _apply_rotary(q, cos_t, sin_t, rotary_dim)
    k = _apply_rotary(k, cos_t, sin_t, rotary_dim)
    return q.contiguous(), k.contiguous(), v.contiguous()


def fused_qkv_rope_boundary_triton(
    raw_qkv: torch.Tensor,
    global_scale: Union[float, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton implementation of :func:`fused_qkv_rope_boundary_reference` (CUDA, BF16 only).

    The transformation runs in a single kernel launch that maps raw QKV storage straight into three contiguous
    ``(B, H, T, D)`` buffers; no intermediate scaled QKV tensor is materialized. Unsupported devices or dtypes
    raise instead of falling back to the reference path.

    A 1-element CUDA ``global_scale`` tensor is read on the host, which synchronizes; the intended static-scale
    usage passes a python float.
    """
    batch, seq_len, _, rotary_dim = _validate_boundary_inputs(
        raw_qkv, global_scale, cos, sin, num_heads, head_dim, bias
    )
    if not raw_qkv.is_cuda:
        raise RuntimeError(
            f"fused_qkv_rope_boundary_triton requires CUDA tensors, got raw_qkv on device {raw_qkv.device}."
        )
    if raw_qkv.dtype != torch.bfloat16:
        raise RuntimeError(
            f"fused_qkv_rope_boundary_triton only supports torch.bfloat16 inputs, got {raw_qkv.dtype}. "
            "Use fused_qkv_rope_boundary_reference for other dtypes."
        )

    triton_mod, kernel = _resolve_triton()

    scale = float(global_scale.item()) if isinstance(global_scale, torch.Tensor) else float(global_scale)
    out_shape = (batch, num_heads, seq_len, head_dim)
    q = torch.empty(out_shape, dtype=raw_qkv.dtype, device=raw_qkv.device)
    k = torch.empty(out_shape, dtype=raw_qkv.dtype, device=raw_qkv.device)
    v = torch.empty(out_shape, dtype=raw_qkv.dtype, device=raw_qkv.device)

    num_rows = batch * seq_len * num_heads
    grid = (triton_mod.cdiv(num_rows, _ROWS_PER_BLOCK),)
    kernel[grid](
        raw_qkv,
        bias if bias is not None else raw_qkv,  # unused pointer when HAS_BIAS is False
        cos,
        sin,
        q,
        k,
        v,
        scale,
        num_rows,
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        HAS_BIAS=bias is not None,
        BLOCK_D=triton_mod.next_power_of_2(head_dim),
        ROWS_PER_BLOCK=_ROWS_PER_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return q, k, v


def _apply_rotary(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` features of ``t``; the remaining features pass through unchanged."""
    if rotary_dim == t.shape[-1]:
        return t * cos + _rotate_half(t) * sin
    t_rot = t[..., :rotary_dim]
    t_pass = t[..., rotary_dim:]
    t_rot = t_rot * cos + _rotate_half(t_rot) * sin
    return torch.cat((t_rot, t_pass), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim of ``x`` in half and rotate: ``(x1, x2) -> (-x2, x1)``."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _validate_boundary_inputs(
    raw_qkv: torch.Tensor,
    global_scale: Union[float, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor],
) -> Tuple[int, int, int, int]:
    """Fail-closed contract check shared by both entry points.

    Returns:
        ``(batch, seq_len, hidden, rotary_dim)`` where ``hidden == 3 * num_heads * head_dim``.
    """
    if not isinstance(raw_qkv, torch.Tensor):
        raise ValueError(f"raw_qkv must be a torch.Tensor, got {type(raw_qkv).__name__}.")
    if raw_qkv.dim() != 3:
        raise ValueError(
            f"raw_qkv must be rank 3 (batch, time, 3 * heads * head_dim), got shape {tuple(raw_qkv.shape)}."
        )
    if not raw_qkv.is_floating_point():
        raise ValueError(f"raw_qkv must be a floating point tensor, got dtype {raw_qkv.dtype}.")
    if not raw_qkv.is_contiguous():
        raise ValueError("raw_qkv must be contiguous; call .contiguous() on the GEMM output first.")
    if not isinstance(num_heads, int) or not isinstance(head_dim, int) or num_heads <= 0 or head_dim <= 0:
        raise ValueError(f"num_heads and head_dim must be positive ints, got {num_heads} and {head_dim}.")

    batch, seq_len, hidden = raw_qkv.shape
    expected_hidden = 3 * num_heads * head_dim
    if hidden != expected_hidden:
        raise ValueError(
            f"raw_qkv last dim must be 3 * num_heads * head_dim = {expected_hidden}, got {hidden} "
            f"(num_heads={num_heads}, head_dim={head_dim})."
        )

    if isinstance(global_scale, torch.Tensor):
        if global_scale.numel() != 1:
            raise ValueError(f"global_scale must be a scalar, got a tensor with {global_scale.numel()} elements.")
        if not global_scale.is_floating_point():
            raise ValueError(f"global_scale tensor must be floating point, got dtype {global_scale.dtype}.")
    elif not isinstance(global_scale, (float, int)) or isinstance(global_scale, bool):
        raise ValueError(f"global_scale must be a float or a 1-element tensor, got {type(global_scale).__name__}.")

    for name, table in (("cos", cos), ("sin", sin)):
        if not isinstance(table, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(table).__name__}.")
        if table.dim() != 2:
            raise ValueError(f"{name} must be rank 2 (time, rotary_dim), got shape {tuple(table.shape)}.")
        if not table.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if table.dtype != raw_qkv.dtype:
            raise ValueError(f"{name} dtype {table.dtype} must match raw_qkv dtype {raw_qkv.dtype}.")
        if table.device != raw_qkv.device:
            raise ValueError(f"{name} is on device {table.device} but raw_qkv is on {raw_qkv.device}.")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have the same shape, got {tuple(cos.shape)} and {tuple(sin.shape)}.")
    if cos.shape[0] < seq_len:
        raise ValueError(f"cos/sin cover {cos.shape[0]} positions but raw_qkv has {seq_len} time steps.")

    rotary_dim = cos.shape[1]
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary dim (cos.shape[-1]) must be a positive even number, got {rotary_dim}.")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary dim {rotary_dim} must not exceed head_dim {head_dim}.")

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise ValueError(f"bias must be a torch.Tensor or None, got {type(bias).__name__}.")
        if bias.dim() != 1 or bias.shape[0] != expected_hidden:
            raise ValueError(f"bias must have shape ({expected_hidden},), got {tuple(bias.shape)}.")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous.")
        if bias.dtype != raw_qkv.dtype:
            raise ValueError(f"bias dtype {bias.dtype} must match raw_qkv dtype {raw_qkv.dtype}.")
        if bias.device != raw_qkv.device:
            raise ValueError(f"bias is on device {bias.device} but raw_qkv is on {raw_qkv.device}.")

    return batch, seq_len, hidden, rotary_dim


def _resolve_triton():
    """Import Triton on first use and JIT-wrap the boundary kernel.

    Keeping this out of module import time is what lets a CPU-only install import this module.
    """
    global triton, tl, _compiled_kernel
    if _compiled_kernel is not None:
        return triton, _compiled_kernel
    try:
        import triton as triton_module
        import triton.language as triton_language
    except ImportError as err:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "fused_qkv_rope_boundary_triton requires Triton, which is not installed. "
            "Install Triton or use fused_qkv_rope_boundary_reference."
        ) from err
    triton = triton_module
    tl = triton_language
    _compiled_kernel = triton_module.jit(_qkv_rope_boundary_kernel)
    return triton, _compiled_kernel


# Deliberately left undecorated: ``triton.jit`` is applied in :func:`_resolve_triton` so that importing this
# module does not import Triton. For the same reason the ``constexpr`` annotations are written as strings --
# python evaluates parameter annotations eagerly, and Triton accepts the string form.
def _qkv_rope_boundary_kernel(
    raw_ptr,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    global_scale,
    num_rows,
    seq_len,
    num_heads,
    head_dim,
    rotary_dim,
    HAS_BIAS: "tl.constexpr",
    BLOCK_D: "tl.constexpr",
    ROWS_PER_BLOCK: "tl.constexpr",
):
    """Rescale, bias, re-lay-out and rotate one tile of ``(batch, time, head)`` rows.

    Each program owns ``ROWS_PER_BLOCK`` rows and the full ``head_dim`` feature axis, so the rotate-half partner
    of every feature lives in the same program and no intermediate scaled QKV tensor is needed.
    """
    pid = tl.program_id(0)
    rows = (pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)).to(tl.int64)
    row_mask = rows < num_rows

    # raw_qkv is viewed as (B, T, 3, H, D); outputs are contiguous (B, H, T, D).
    head = rows % num_heads
    bt = rows // num_heads
    pos = bt % seq_len
    batch = bt // seq_len

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    rot_mask = d_offs < rotary_dim
    half = rotary_dim // 2
    # _rotate_half: feature d < half pairs with d + half (negated), d in [half, rotary_dim) with d - half.
    partner = tl.where(d_offs < half, d_offs + half, d_offs - half)
    sign = tl.where(d_offs < half, -1.0, 1.0)

    mask = row_mask[:, None] & d_mask[None, :]
    rot_tile_mask = row_mask[:, None] & rot_mask[None, :]

    pe_offs = pos[:, None] * rotary_dim + d_offs[None, :]
    cos_v = tl.load(cos_ptr + pe_offs, mask=rot_tile_mask, other=0.0).to(tl.float32)
    sin_v = tl.load(sin_ptr + pe_offs, mask=rot_tile_mask, other=0.0).to(tl.float32)
    # Non-rotated features pass through: cos = 1, sin = 0.
    cos_v = tl.where(rot_mask[None, :], cos_v, 1.0)

    head_hidden = num_heads * head_dim
    raw_base = bt * (3 * head_hidden) + head * head_dim
    out_base = (batch * num_heads + head) * seq_len * head_dim + pos * head_dim
    main_offs = raw_base[:, None] + d_offs[None, :]
    partner_offs = raw_base[:, None] + partner[None, :]
    out_offs = out_base[:, None] + d_offs[None, :]
    bias_main = head[:, None] * head_dim + d_offs[None, :]
    bias_partner = head[:, None] * head_dim + partner[None, :]

    # Q (j = 0)
    q_main = tl.load(raw_ptr + main_offs, mask=mask, other=0.0).to(tl.float32) * global_scale
    q_partner = tl.load(raw_ptr + partner_offs, mask=rot_tile_mask, other=0.0).to(tl.float32) * global_scale
    if HAS_BIAS:
        q_main += tl.load(bias_ptr + bias_main, mask=mask, other=0.0).to(tl.float32)
        q_partner += tl.load(bias_ptr + bias_partner, mask=rot_tile_mask, other=0.0).to(tl.float32)
    q_out = q_main * cos_v + sign[None, :] * q_partner * sin_v
    tl.store(q_ptr + out_offs, q_out.to(q_ptr.dtype.element_ty), mask=mask)

    # K (j = 1)
    k_main = tl.load(raw_ptr + main_offs + head_hidden, mask=mask, other=0.0).to(tl.float32) * global_scale
    k_partner = tl.load(raw_ptr + partner_offs + head_hidden, mask=rot_tile_mask, other=0.0).to(tl.float32)
    k_partner = k_partner * global_scale
    if HAS_BIAS:
        k_main += tl.load(bias_ptr + bias_main + head_hidden, mask=mask, other=0.0).to(tl.float32)
        k_partner += tl.load(bias_ptr + bias_partner + head_hidden, mask=rot_tile_mask, other=0.0).to(tl.float32)
    k_out = k_main * cos_v + sign[None, :] * k_partner * sin_v
    tl.store(k_ptr + out_offs, k_out.to(k_ptr.dtype.element_ty), mask=mask)

    # V (j = 2): rescaled and re-laid-out only, never rotated.
    v_out = tl.load(raw_ptr + main_offs + 2 * head_hidden, mask=mask, other=0.0).to(tl.float32) * global_scale
    if HAS_BIAS:
        v_out += tl.load(bias_ptr + bias_main + 2 * head_hidden, mask=mask, other=0.0).to(tl.float32)
    tl.store(v_ptr + out_offs, v_out.to(v_ptr.dtype.element_ty), mask=mask)

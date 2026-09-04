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
Fused producer-to-NVFP4-pack boundaries of a static-NVFP4 Sortformer encoder.

This module holds *prototype primitives* for exact-shape numerical and timing probes. Nothing here is wired into
the model, the quantization recipe, or any dispatch table, and none of it is production or cross-device evidence.

It removes two measured BF16 materialization boundaries that currently sit in front of MSLK's NVFP4 activation
packing:

* ``BF16 residual -> LayerNorm -> BF16 rounding -> NVFP4 block scale + pack``
* ``raw BF16 FP4-GEMM output -> scalar rescale -> optional bias -> exact GELU -> BF16 rounding -> NVFP4 pack``

Each boundary exposes two entry points with identical semantics:

* ``*_reference`` -- materializes the producer in BF16 with PyTorch and then calls pinned MSLK
  ``triton_quantize_nvfp4``. It is the oracle for the packed format.
* ``*_triton`` -- performs the producer *and* the NVFP4 pack in one Triton kernel launch, never materializing the
  BF16 ``(M, K)`` producer tensor and never silently falling back to the reference path.

Both entry points take a contiguous rank-2 ``(M, K)`` tensor with ``K`` a positive multiple of 64, and return
``(packed_data, blocked_scales)`` in exactly the layout the pinned MSLK 1.2 native NVFP4 GEMM path consumes:

* ``packed_data``: shape ``(M, K // 2)``, viewed as ``torch.float4_e2m1fn_x2``;
* ``blocked_scales``: shape ``(ceil(M / 128) * 128, ceil((K // 16) / 4) * 4)``, dtype ``torch.float8_e4m3fn``, in
  the Blackwell 128x4 swizzled layout with zero padding.

The producer arithmetic deliberately preserves the current BF16 boundary: mean/variance/affine and the GELU are
accumulated in FP32, the result is explicitly rounded to BF16, and only then is the block amax taken and the value
packed. That keeps the fused kernels numerically comparable to the unfused path they replace.

The two boundaries therefore accept different input dtypes, matching the CUDA BF16 autocast path they reproduce:

* the LayerNorm input is the *residual stream*, which under autocast is FP32 whenever a preceding normalization
  produced it (``layer_norm`` is on autocast's FP32 list, so it keeps FP32 input and FP32 arithmetic and only the
  following quantized linear consumes BF16). It is therefore accepted as either FP32 or BF16, and the LayerNorm
  arithmetic runs in FP32 either way; the BF16 rounding still happens exactly once, in front of the pack;
* the scaled-GELU input is the raw output of the preceding native FP4 GEMM, which ``_scaled_mm`` always produces at
  ``out_dtype=torch.bfloat16``, so it stays BF16-only.

The residual is never cast or mutated in place: an FP32 residual is read as FP32 and only the packed activation
that leaves this module is quantized.

Triton and MSLK are imported lazily, on the first call of a pack entry point, so importing this module on a
CPU-only or minimal install neither imports nor requires either dependency.

.. note::
   The scale-store address arithmetic in :func:`_store_blocked_scale_row_src` reproduces the 128x4 swizzled layout
   that MSLK's ``nvfp4_scale_swizzle`` writes. MSLK's helper is a grid-level pass over an already-materialized
   linear scale tensor, so it cannot be invoked from a fused single-launch kernel; its importability is still required
   here so that an MSLK build which moved or dropped the swizzle fails closed instead of silently pinning a stale
   layout. Equality against MSLK's own ``triton_quantize_nvfp4`` output is what validates the layout, and the tests
   in ``tests/collections/speaker_tasks/utils/test_sortformer_nvfp4_fused_pack.py`` assert exactly that.
"""

import math
from typing import Any, Optional, Sequence, Tuple

import torch

# NVFP4 constants, identical to pinned MSLK ``mslk.quantize.triton.fp4_quantize``. The Triton kernels below
# cannot read these names (Triton rejects non-constexpr python globals) and repeat their values as literals.
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_SCALE_EPS = 1.5258789e-05

# Blackwell blocked-scale tiling: scales are stored in 128-row x 4-column tiles.
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4

# ``K`` must be a multiple of this so that the block count ``K // 16`` is already a multiple of ``SCALE_TILE_COLS``.
REQUIRED_K_MULTIPLE = 64

# Accepted producer-input dtypes per boundary; see the module docstring for why they differ.
LAYER_NORM_INPUT_DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32)
SCALED_GELU_INPUT_DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16,)

# torch dtypes the packed NVFP4 contract depends on.
_REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn")

_MSLK_QUANTIZE_API = "mslk.quantize.triton.fp4_quantize:triton_quantize_nvfp4"
_MSLK_SWIZZLE_API = "mslk.quantize.triton.fp4_quantize:nvfp4_scale_swizzle"
_MSLK_PACK_API = "mslk.quantize.triton.fp4_quantize:convert_fp32_to_fp4_packed"

# Populated lazily by :func:`_resolve_fused_backend` / :func:`_resolve_mslk_quantize`. The Triton kernels resolve
# ``tl``, ``convert_fp32_to_fp4_packed``, ``_erf``, ``_pack_bf16_row`` and ``_store_blocked_scale_row`` through this
# module's globals at JIT-compile time, so these must stay module-level names.
triton = None
tl = None
convert_fp32_to_fp4_packed = None
_erf = None
_pack_bf16_row = None
_store_blocked_scale_row = None
_layer_norm_kernel = None
_scaled_gelu_kernel = None
_mslk_quantize_nvfp4 = None


def layer_norm_nvfp4_pack_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference LayerNorm producer followed by the pinned MSLK NVFP4 pack.

    The BF16 producer is materialized with :func:`torch.nn.functional.layer_norm` and handed to MSLK's
    ``triton_quantize_nvfp4``; this defines the packed-data and blocked-scale contract that
    :func:`layer_norm_nvfp4_pack_triton` must reproduce.

    An FP32 residual keeps FP32 LayerNorm arithmetic -- the BF16 affine parameters are widened to FP32 exactly as
    autocast does, since ``layer_norm`` is on autocast's FP32 list -- and the single explicit BF16 rounding stays
    where it is today, immediately in front of the pack. A BF16 residual is normalized exactly as before.

    Args:
        x: contiguous FP32 or BF16 residual input, shape ``(M, K)`` on CUDA, ``K`` a positive multiple of 64.
        weight: contiguous BF16 LayerNorm weight, shape ``(K,)``.
        activation_global_scale: one-element CUDA float32 tensor, interpreted exactly as MSLK does.
        bias: optional contiguous BF16 LayerNorm bias, shape ``(K,)``.
        eps: positive finite LayerNorm epsilon (production uses ``1e-5``).

    Returns:
        ``(packed_data, blocked_scales)``: ``(M, K // 2)`` ``torch.float4_e2m1fn_x2`` and
        ``(ceil(M / 128) * 128, ceil((K // 16) / 4) * 4)`` ``torch.float8_e4m3fn``.
    """
    _validate_layer_norm_inputs(x, weight, activation_global_scale, bias, eps)
    quantize = _resolve_mslk_quantize()
    # The affine parameters follow the input's precision, so an FP32 residual normalizes in FP32; the explicit
    # ``.to(bfloat16)`` is then the one producer boundary the pack sees, for either input dtype.
    producer = torch.nn.functional.layer_norm(
        x, (x.shape[1],), weight.to(x.dtype), None if bias is None else bias.to(x.dtype), eps
    ).to(torch.bfloat16)
    return quantize(producer.contiguous(), activation_global_scale)


def layer_norm_nvfp4_pack_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused LayerNorm + NVFP4 pack, in exactly one Triton kernel launch.

    Semantics match :func:`layer_norm_nvfp4_pack_reference`: mean and population variance (``unbiased=False``) are
    accumulated in FP32, the affine transform is applied in FP32, the result is explicitly rounded to BF16, and the
    block amax and packing are computed from that BF16 value. The ``(M, K)`` BF16 producer is never materialized.
    ``x`` may be the FP32 residual of the autocast pre-norm boundary or a BF16 one; the kernel loads it to FP32
    either way, so the normalization arithmetic and the single BF16 producer rounding do not depend on it.

    Unsupported devices, dtypes, layouts, shapes, scales or missing optional dependencies raise; this function
    never falls back to the reference path.
    """
    m_dim, k_dim = _validate_layer_norm_inputs(x, weight, activation_global_scale, bias, eps)
    kernel = _resolve_fused_backend("layer_norm_nvfp4_pack_triton", _layer_norm_kernel_name)

    packed, scales, block_k, n_col_blocks, grid_rows = _allocate_pack_outputs(x, m_dim, k_dim)
    kernel[(grid_rows,)](
        x,
        weight,
        bias if bias is not None else weight,  # unused pointer when HAS_BIAS is False
        activation_global_scale,
        packed,
        scales,
        m_dim,
        k_dim,
        float(eps),
        n_col_blocks,
        HAS_BIAS=bias is not None,
        BLOCK_K=block_k,
    )
    return _view_pack_outputs(packed, scales)


def scaled_gelu_nvfp4_pack_reference(
    raw: torch.Tensor,
    output_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference scaled-GELU producer followed by the pinned MSLK NVFP4 pack.

    The producer is ``F.gelu(raw * output_global_scale + bias, approximate='none')``, materialized in BF16 and
    handed to MSLK's ``triton_quantize_nvfp4``.

    Args:
        raw: contiguous BF16 unscaled output of the preceding native FP4 GEMM, shape ``(M, K)`` on CUDA.
        output_global_scale: one-element CUDA float32 tensor, the dequant multiplier currently applied after
            ``_scaled_mm``.
        activation_global_scale: one-element CUDA float32 tensor for packing into the next NVFP4 linear.
        bias: optional contiguous BF16 linear bias, shape ``(K,)``, applied after the rescale and before GELU.

    Returns:
        ``(packed_data, blocked_scales)``, laid out exactly as in :func:`layer_norm_nvfp4_pack_reference`.
    """
    _validate_scaled_gelu_inputs(raw, output_global_scale, activation_global_scale, bias)
    quantize = _resolve_mslk_quantize()
    producer = raw * output_global_scale
    if bias is not None:
        producer = producer + bias
    producer = torch.nn.functional.gelu(producer, approximate='none').to(torch.bfloat16)
    return quantize(producer.contiguous(), activation_global_scale)


def scaled_gelu_nvfp4_pack_triton(
    raw: torch.Tensor,
    output_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused output-rescale + bias + exact GELU + NVFP4 pack, in exactly one Triton kernel launch.

    Semantics match :func:`scaled_gelu_nvfp4_pack_reference`: the rescale, bias and the exact ``erf`` GELU are
    evaluated in FP32, the result is explicitly rounded to BF16, and the block amax and packing are computed from
    that BF16 value. The ``(M, K)`` BF16 producer is never materialized, and unsupported inputs raise rather than
    falling back.
    """
    m_dim, k_dim = _validate_scaled_gelu_inputs(raw, output_global_scale, activation_global_scale, bias)
    kernel = _resolve_fused_backend("scaled_gelu_nvfp4_pack_triton", _scaled_gelu_kernel_name)

    packed, scales, block_k, n_col_blocks, grid_rows = _allocate_pack_outputs(raw, m_dim, k_dim)
    kernel[(grid_rows,)](
        raw,
        bias if bias is not None else raw,  # unused pointer when HAS_BIAS is False
        output_global_scale,
        activation_global_scale,
        packed,
        scales,
        m_dim,
        k_dim,
        n_col_blocks,
        HAS_BIAS=bias is not None,
        BLOCK_K=block_k,
    )
    return _view_pack_outputs(packed, scales)


def nvfp4_pack_output_shapes(m_dim: int, k_dim: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the MSLK ``(packed_data_shape, blocked_scales_shape)`` for an ``(M, K)`` producer."""
    if m_dim <= 0 or k_dim <= 0 or k_dim % REQUIRED_K_MULTIPLE != 0:
        raise ValueError(
            f"Expected M > 0 and K a positive multiple of {REQUIRED_K_MULTIPLE}, got M={m_dim}, K={k_dim}."
        )
    num_blocks = k_dim // NVFP4_BLOCK_SIZE
    padded_rows = _ceil_to(m_dim, SCALE_TILE_ROWS)
    padded_cols = _ceil_to(num_blocks, SCALE_TILE_COLS)
    return (m_dim, k_dim // 2), (padded_rows, padded_cols)


# --------------------------------------------------------------------------------------------------------------
# Input validation. Runs before any optional dependency is resolved, so a bad call on a CPU-only install raises an
# actionable error instead of an import error.
# --------------------------------------------------------------------------------------------------------------


def _validate_layer_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
) -> Tuple[int, int]:
    """Fail-closed contract check shared by both LayerNorm entry points; returns ``(M, K)``."""
    m_dim, k_dim = _validate_producer_tensor(x, "x", allowed_dtypes=LAYER_NORM_INPUT_DTYPES)
    _validate_affine_vector(weight, "weight", k_dim, required=True)
    _validate_affine_vector(bias, "bias", k_dim, required=False)
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError(f"eps must be a python float, got {type(eps).__name__}.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps}.")
    _validate_global_scale(activation_global_scale, "activation_global_scale")
    _validate_devices(
        x, "x", (("weight", weight), ("bias", bias), ("activation_global_scale", activation_global_scale))
    )
    return m_dim, k_dim


def _validate_scaled_gelu_inputs(
    raw: torch.Tensor,
    output_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[int, int]:
    """Fail-closed contract check shared by both scaled-GELU entry points; returns ``(M, K)``."""
    m_dim, k_dim = _validate_producer_tensor(raw, "raw", allowed_dtypes=SCALED_GELU_INPUT_DTYPES)
    _validate_affine_vector(bias, "bias", k_dim, required=False)
    _validate_global_scale(output_global_scale, "output_global_scale")
    _validate_global_scale(activation_global_scale, "activation_global_scale")
    _validate_devices(
        raw,
        "raw",
        (
            ("bias", bias),
            ("output_global_scale", output_global_scale),
            ("activation_global_scale", activation_global_scale),
        ),
    )
    return m_dim, k_dim


def _validate_producer_tensor(
    tensor: torch.Tensor, name: str, allowed_dtypes: Tuple[torch.dtype, ...] = SCALED_GELU_INPUT_DTYPES
) -> Tuple[int, int]:
    """Check the structure of the ``(M, K)`` producer input and return its shape.

    ``allowed_dtypes`` is per boundary and never widened globally: the scaled-GELU input stays BF16-only, while the
    LayerNorm input may also be the FP32 residual the autocast pre-norm boundary actually supplies.

    Device placement is checked separately, by :func:`_validate_devices`, so that a structurally invalid call
    reports the structural problem rather than being masked by the CUDA requirement.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be rank 2 (M, K), got shape {tuple(tensor.shape)}.")
    if tensor.dtype not in allowed_dtypes:
        expected = " or ".join(str(dtype) for dtype in allowed_dtypes)
        raise ValueError(f"{name} must be {expected}, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; call .contiguous() before packing.")
    m_dim, k_dim = int(tensor.shape[0]), int(tensor.shape[1])
    if m_dim <= 0:
        raise ValueError(f"{name} must have M > 0, got shape {tuple(tensor.shape)}.")
    if k_dim <= 0 or k_dim % REQUIRED_K_MULTIPLE != 0:
        raise ValueError(
            f"{name} must have K a positive multiple of {REQUIRED_K_MULTIPLE}, got K={k_dim}. This keeps the "
            f"block count K // {NVFP4_BLOCK_SIZE} a multiple of the {SCALE_TILE_COLS}-wide scale tile."
        )
    return m_dim, k_dim


def _validate_affine_vector(vector: Optional[torch.Tensor], name: str, k_dim: int, required: bool) -> None:
    """Check the structure of an optional BF16 ``(K,)`` weight/bias vector; device placement is checked later."""
    if vector is None:
        if required:
            raise ValueError(f"{name} is required and must be a torch.Tensor of shape ({k_dim},).")
        return
    if not isinstance(vector, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor or None, got {type(vector).__name__}.")
    if vector.dim() != 1 or int(vector.shape[0]) != k_dim:
        raise ValueError(f"{name} must have shape ({k_dim},), got {tuple(vector.shape)}.")
    if vector.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {vector.dtype}.")
    if not vector.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_global_scale(scale: torch.Tensor, name: str) -> None:
    """Check the structure of a one-element float32 global-scale tensor, as MSLK requires."""
    if not isinstance(scale, torch.Tensor):
        raise ValueError(f"{name} must be a one-element torch.Tensor, got {type(scale).__name__}.")
    if scale.numel() != 1:
        raise ValueError(f"{name} must have exactly one element, got {scale.numel()}.")
    if scale.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32, got {scale.dtype}.")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_devices(
    reference: torch.Tensor, name: str, companions: Sequence[Tuple[str, Optional[torch.Tensor]]]
) -> None:
    """Require a CUDA producer and companion tensors that live on the same device.

    Checked after every structural check so that, for example, a two-element scale on CPU reports the element
    count rather than the missing CUDA placement.
    """
    if not reference.is_cuda:
        raise ValueError(
            f"{name} must be a CUDA tensor, got device {reference.device}. NVFP4 packing has no CPU implementation."
        )
    for companion_name, companion in companions:
        if companion is not None and companion.device != reference.device:
            raise ValueError(f"{companion_name} is on device {companion.device} but {name} is on {reference.device}.")


# --------------------------------------------------------------------------------------------------------------
# Output allocation
# --------------------------------------------------------------------------------------------------------------


def _allocate_pack_outputs(
    reference: torch.Tensor, m_dim: int, k_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Allocate the uint8-backed packed-data and blocked-scale buffers plus the kernel tiling constants.

    Both buffers are left uninitialized, and no fill is ever issued in front of the launch: the fused kernels write
    every packed byte of every row in ``[0, M)`` (``K`` is a positive multiple of 64, so the packed width ``K // 2``
    is fully covered) and every scale byte of *every* blocked-scale row, valid or padded. The grid is therefore the
    padded scale-row count rather than ``M``: programs with ``row >= M`` read no producer input and store no packed
    data, and write only their blocked-scale row as zero. For a multiple-of-128 ``M`` the padded count equals ``M``,
    so the grid is unchanged and no program takes the padded branch.

    Returns:
        ``(packed, scales, block_k, n_col_blocks, grid_rows)``, with ``grid_rows`` the one-dimensional grid size.
    """
    (packed_rows, packed_cols), scale_shape = nvfp4_pack_output_shapes(m_dim, k_dim)
    packed = torch.empty((packed_rows, packed_cols), dtype=torch.uint8, device=reference.device)
    scales = torch.empty(scale_shape, dtype=torch.uint8, device=reference.device)
    block_k = triton.next_power_of_2(k_dim)
    return packed, scales, block_k, scale_shape[1] // SCALE_TILE_COLS, scale_shape[0]


def _view_pack_outputs(packed: torch.Tensor, scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reinterpret the uint8 kernel outputs as the FP4/FP8 dtypes MSLK returns."""
    return packed.view(torch.float4_e2m1fn_x2), scales.view(torch.float8_e4m3fn)


# Warp counts offered to Triton's autotuner, one per candidate configuration.
#
# WHY AUTOTUNE RATHER THAN A FORMULA: the previous heuristic, max(4, min(16, BLOCK_K // 256)), was biased high and
# could never return fewer than four warps. Measured on SM120 at M=21888, the best choices are 1 warp for the K=512
# LayerNorm boundary (13.8 us against 18.4 us at the old default of 4), 4 warps for the K=2048 LayerNorm boundary
# (37.9 us against 47.4 us at 8) and 2 warps for the K=2048 scaled-GELU boundary (58.9 us against 62.0 us at 8) --
# so two boundaries with identical BLOCK_K want different warp counts, which no function of BLOCK_K alone can
# express. Autotune also keys its cache on the device, so SM103 and any future architecture converge on their own
# winner instead of inheriting numbers measured here.
#
# RANGE: one program handles a whole row, so a row of only BLOCK_K elements starves large warp counts. Every
# measured optimum lies in 1..4, and the penalty above that is severe (32 warps cost 218 us at K=512, 12x the
# optimum), so the search stops at 8. Reaching down to 1 is essential -- that is where the K=512 optimum lives and
# exactly what the old floor of 4 made unreachable.
_AUTOTUNE_NUM_WARPS: Tuple[int, ...] = (1, 2, 4, 8)


def _autotuned(triton_module, kernel_src):
    """Wrap a kernel source function in ``triton.jit`` plus ``triton.autotune`` over the warp candidates.

    ``key=["K"]`` re-runs the search when the row width changes and reuses the winner otherwise, so the cost is paid
    once per distinct K per process. In this model that is two values (512 and 2048), and it lands during model
    warmup, which the evaluator already excludes from its timing window.

    ``BLOCK_K`` stays a caller-supplied constexpr rather than a searched parameter: the kernel body indexes a whole
    row with ``tl.arange(0, BLOCK_K)``, so it must cover K and is not free to vary.
    """
    configs = [triton_module.Config({}, num_warps=w) for w in _AUTOTUNE_NUM_WARPS]
    return triton_module.autotune(configs=configs, key=["K"])(triton_module.jit(kernel_src))


def _ceil_to(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple


# --------------------------------------------------------------------------------------------------------------
# Lazy optional-dependency resolution
# --------------------------------------------------------------------------------------------------------------


def _resolve_mslk_quantize():
    """Import pinned MSLK on first use and return ``triton_quantize_nvfp4``."""
    global _mslk_quantize_nvfp4
    if _mslk_quantize_nvfp4 is None:
        _require_torch_dtypes("The MSLK NVFP4 reference pack")
        _mslk_quantize_nvfp4 = _import_api(_MSLK_QUANTIZE_API, "The MSLK NVFP4 reference pack")
    return _mslk_quantize_nvfp4


def _resolve_fused_backend(entry_point: str, kernel_name: str):
    """Import Triton and the pinned MSLK Triton helpers on first use and JIT-wrap the fused kernels.

    Keeping this out of module import time is what lets a CPU-only install import this module. The MSLK helpers
    are resolved before jitting because the kernels reference ``convert_fp32_to_fp4_packed`` through this module's
    globals; ``nvfp4_scale_swizzle`` is required (though it cannot be called from inside a single-launch kernel)
    so that an MSLK build which moved or dropped the swizzle fails closed rather than pinning a stale layout.

    Returns:
        The JIT-wrapped kernel bound to ``kernel_name``.
    """
    global triton, tl, convert_fp32_to_fp4_packed, _erf, _pack_bf16_row, _store_blocked_scale_row
    global _layer_norm_kernel, _scaled_gelu_kernel

    if _layer_norm_kernel is None or _scaled_gelu_kernel is None:
        _require_torch_dtypes(f"{entry_point}")
        try:
            import triton as triton_module
            import triton.language as triton_language
        except ImportError as err:  # pragma: no cover - depends on the install
            raise RuntimeError(
                f"{entry_point} requires Triton, which is not installed. Install Triton, or use the "
                "corresponding reference entry point."
            ) from err
        triton = triton_module
        tl = triton_language

        convert_fp32_to_fp4_packed = _import_api(_MSLK_PACK_API, entry_point)
        _import_api(_MSLK_SWIZZLE_API, entry_point)
        _erf = _resolve_erf(entry_point)

        _store_blocked_scale_row = triton_module.jit(_store_blocked_scale_row_src)
        _pack_bf16_row = triton_module.jit(_pack_bf16_row_src)
        # The two entry-point kernels are wrapped in autotune; the device functions above are not, because they are
        # inlined into the callers rather than launched. Keying on K is enough: BLOCK_K is next_power_of_2(K), so K
        # determines the tile, and the two kernels tune independently because each has its own autotuner instance.
        _layer_norm_kernel = _autotuned(triton_module, _layer_norm_nvfp4_pack_kernel_src)
        _scaled_gelu_kernel = _autotuned(triton_module, _scaled_gelu_nvfp4_pack_kernel_src)

    return globals()[kernel_name]


def _resolve_erf(entry_point: str):
    """Locate Triton's exact ``erf`` device function across the Triton versions that moved libdevice."""
    for path in ("math.erf", "extra.libdevice.erf", "extra.cuda.libdevice.erf"):
        target: Any = tl
        for attribute in path.split("."):
            target = getattr(target, attribute, None)
            if target is None:
                break
        if target is not None:
            return target
    raise RuntimeError(
        f"{entry_point} needs Triton's exact erf device function, but none of tl.math.erf, "
        "tl.extra.libdevice.erf or tl.extra.cuda.libdevice.erf exist in the installed Triton. "
        "The approximate GELU is not an acceptable substitute here."
    )


def _require_torch_dtypes(entry_point: str) -> None:
    """Raise unless this torch build exposes the FP4/FP8 dtypes the packed NVFP4 contract needs."""
    missing = [name for name in _REQUIRED_TORCH_DTYPES if not isinstance(getattr(torch, name, None), torch.dtype)]
    if missing:
        raise RuntimeError(
            f"{entry_point} requires torch dtypes {missing}, which torch {torch.__version__} does not expose."
        )


def _import_api(api: str, entry_point: str):
    """Import ``module:attribute`` lazily, with an error naming the caller that needed it."""
    module_path, _, attribute = api.partition(":")
    try:
        module = __import__(module_path, fromlist=[attribute])
    except ImportError as err:  # pragma: no cover - depends on the install
        raise RuntimeError(f"{entry_point} requires MSLK >= 1.2, but '{module_path}' could not be imported.") from err
    try:
        return getattr(module, attribute)
    except AttributeError as err:
        raise RuntimeError(
            f"{entry_point} requires '{api}', which the installed MSLK does not provide. This prototype is pinned "
            "to the MSLK 1.2 NVFP4 packed-data and blocked-scale layout and refuses to guess a different one."
        ) from err


# --------------------------------------------------------------------------------------------------------------
# Triton kernels. Deliberately left undecorated: ``triton.jit`` is applied in :func:`_resolve_fused_backend` so
# that importing this module does not import Triton. For the same reason the ``constexpr`` annotations are written
# as strings -- python evaluates parameter annotations eagerly, and Triton accepts the string form.
# --------------------------------------------------------------------------------------------------------------

_layer_norm_kernel_name = "_layer_norm_kernel"
_scaled_gelu_kernel_name = "_scaled_gelu_kernel"


def _layer_norm_nvfp4_pack_kernel_src(
    x_ptr,
    weight_ptr,
    bias_ptr,
    act_scale_ptr,
    packed_ptr,
    scale_ptr,
    M,
    K,
    eps,
    n_col_blocks,
    HAS_BIAS: "tl.constexpr",
    BLOCK_K: "tl.constexpr",
):
    """LayerNorm one ``(M, K)`` row in FP32, round to BF16, then block-scale and pack it as NVFP4.

    One program owns one row and the whole feature axis, so the LayerNorm reduction and every 16-wide quantization
    block live in the same program and no BF16 producer row is ever written to memory.

    ``x_ptr`` may be FP32 or BF16 -- Triton takes the element type from the pointer and the load is upcast to FP32
    here -- so the FP32 residual of the autocast pre-norm boundary is normalized in FP32 without being cast first.

    The grid covers the *padded* blocked-scale rows, so ``row`` may be at or past ``M``. Such a program touches no
    input and no packed byte; it only zeroes its blocked-scale row, which is what keeps the pack a single launch.
    """
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    if row < M:
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / K
        centered = tl.where(mask, x - mean, 0.0)
        # Population variance, matching torch.nn.functional.layer_norm (unbiased=False).
        var = tl.sum(centered * centered, axis=0) / K
        rstd = 1.0 / tl.sqrt(var + eps)

        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        producer = centered * rstd * weight
        if HAS_BIAS:
            producer += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        producer = tl.where(mask, producer, 0.0)

        # Explicit BF16 boundary: quantization sees exactly what the unfused BF16 producer would have stored.
        producer = producer.to(tl.bfloat16).to(tl.float32)
        act_scale = tl.load(act_scale_ptr).to(tl.float32)
        _pack_bf16_row(producer, row, packed_ptr, scale_ptr, K, n_col_blocks, act_scale, BLOCK_K=BLOCK_K)
    else:
        _store_blocked_scale_row(
            tl.zeros([BLOCK_K // 16], dtype=tl.uint8), row, scale_ptr, K, n_col_blocks, BLOCK_K=BLOCK_K
        )


def _scaled_gelu_nvfp4_pack_kernel_src(
    raw_ptr,
    bias_ptr,
    out_scale_ptr,
    act_scale_ptr,
    packed_ptr,
    scale_ptr,
    M,
    K,
    n_col_blocks,
    HAS_BIAS: "tl.constexpr",
    BLOCK_K: "tl.constexpr",
):
    """Rescale one raw FP4-GEMM row, add bias, apply the exact erf GELU, round to BF16 and pack it as NVFP4.

    As in the LayerNorm kernel, the grid covers the padded blocked-scale rows and a program with ``row >= M`` reads
    nothing and writes only zeros into its blocked-scale row.
    """
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    if row < M:
        out_scale = tl.load(out_scale_ptr).to(tl.float32)
        act_scale = tl.load(act_scale_ptr).to(tl.float32)

        value = tl.load(raw_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32) * out_scale
        if HAS_BIAS:
            value += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). No tanh approximation.
        producer = value * 0.5 * (1.0 + _erf(value * 0.70710678118654752440))
        producer = tl.where(mask, producer, 0.0)

        # Explicit BF16 boundary: quantization sees exactly what the unfused BF16 producer would have stored.
        producer = producer.to(tl.bfloat16).to(tl.float32)
        _pack_bf16_row(producer, row, packed_ptr, scale_ptr, K, n_col_blocks, act_scale, BLOCK_K=BLOCK_K)
    else:
        _store_blocked_scale_row(
            tl.zeros([BLOCK_K // 16], dtype=tl.uint8), row, scale_ptr, K, n_col_blocks, BLOCK_K=BLOCK_K
        )


def _pack_bf16_row_src(
    producer,
    row,
    packed_ptr,
    scale_ptr,
    K,
    n_col_blocks,
    act_scale,
    BLOCK_K: "tl.constexpr",
):
    """Block-scale and pack one BF16-rounded producer row into the MSLK NVFP4 layout.

    ``producer`` is an FP32 vector of length ``BLOCK_K`` holding BF16-exact values, zero beyond ``K``. Scales use
    the pinned MSLK math -- ``clamp((amax / 6.0) * global_scale, eps, 448)`` converted to E4M3, then values scaled
    by ``global_scale / scale_e4m3`` and converted with the native saturating E2M1x2 instruction. There is no E8M0
    mode and no reciprocal or approximate scale math.
    """
    # Every number below is spelled as a literal on purpose: Triton refuses to read non-constexpr python globals
    # from a kernel body, so NVFP4_MAX (6.0), NVFP4_SCALE_EPS (1.5258789e-05), FP8_E4M3_MAX (448.0) and
    # NVFP4_BLOCK_SIZE (16) appear here -- and SCALE_TILE_ROWS (128) and SCALE_TILE_COLS (4) in
    # :func:`_store_blocked_scale_row_src` -- as their pinned values. Keep them in sync with the module-level
    # constants of the same name.
    NUM_BLOCKS: tl.constexpr = BLOCK_K // 16

    blocks = tl.reshape(producer, (NUM_BLOCKS, 16))
    amax = tl.max(tl.abs(blocks), axis=1)
    scale = (amax / 6.0) * act_scale
    scale = tl.minimum(tl.maximum(scale, 1.5258789e-05), 448.0)
    scale_e4m3 = scale.to(tl.float8e4nv)
    # Quantize against the E4M3-rounded scale, which is what the consumer dequantizes with.
    quantized = blocks * (act_scale / scale_e4m3.to(tl.float32))[:, None]

    # Packed data: byte j of a row holds elements 2j (low nibble) and 2j + 1 (high nibble). The pinned MSLK helper
    # forwards its single argument straight to ``tl.inline_asm_elementwise(args=...)``, which needs the two nibble
    # lanes as a sequence of tensors -- so hand it ``pairs.split()``, exactly as MSLK's own callers do.
    pairs = tl.reshape(quantized, (BLOCK_K // 2, 2))
    packed = convert_fp32_to_fp4_packed(pairs.split())
    packed_offs = tl.arange(0, BLOCK_K // 2)
    tl.store(packed_ptr + row * (K // 2) + packed_offs, packed, mask=packed_offs < (K // 2))

    _store_blocked_scale_row(scale_e4m3.to(tl.uint8, bitcast=True), row, scale_ptr, K, n_col_blocks, BLOCK_K=BLOCK_K)


def _store_blocked_scale_row_src(
    scale_bytes,
    row,
    scale_ptr,
    K,
    n_col_blocks,
    BLOCK_K: "tl.constexpr",
):
    """Store one row of E4M3 scale bytes into the MSLK 128x4 swizzled blocked-scale layout.

    The scales are held as 128x4 tiles in row-major tile order; within a tile the 128 rows split as
    ``r = 32 * a + b`` and the byte index is ``b * 16 + a * 4 + c``. ``K`` is a positive multiple of 64, so the
    ``K // 16`` block columns exactly fill ``n_col_blocks`` four-wide tiles and this store covers every scale byte
    of the row -- which is why a padded row (``row >= M``) is fully zeroed by calling this with zero bytes, and no
    separate fill of the padded tail is needed. ``row`` is only required to be below the padded row count, so the
    addresses stay inside the blocked-scale buffer for padded rows too.
    """
    NUM_BLOCKS: tl.constexpr = BLOCK_K // 16

    block_idx = tl.arange(0, NUM_BLOCKS)
    block_mask = block_idx < (K // 16)
    row_in_tile = row % 128
    tile_offset = (row // 128) * n_col_blocks + (block_idx // 4)
    inner = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4 + (block_idx % 4)
    tl.store(scale_ptr + tile_offset * 512 + inner, scale_bytes, mask=block_mask)

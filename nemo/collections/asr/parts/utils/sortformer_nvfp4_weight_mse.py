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
MSE-optimal and activation-weighted NVFP4 block-scale selection for a single Sortformer weight.

This module is a *standalone* PTQ prototype, in the spirit of NVIDIA ModelOpt's block-scale search. It takes an
original high-precision weight plus an already-converted TorchAO ``NVFP4Tensor`` and returns a new ``NVFP4Tensor``
holding the *same* weight repacked with better-chosen block scales.

The ordinary NVFP4 conversion picks each 16-weight block's E4M3 scale from the block amax alone
(``scale_fp8 = amax / (6 * per_tensor_scale)``, rounded to E4M3). That is optimal only for the block's largest
element. :func:`repack_nvfp4_weight_mse` instead evaluates *every* one of the 126 positive finite E4M3 encodings for
every block and keeps the encoding with the lowest local reconstruction MSE, which trades a little clipping of the
block maximum for a better fit of the remaining fifteen weights whenever that lowers the block error.

:func:`repack_nvfp4_weight_local_hessian` runs the *same* exhaustive search over the *same* 126 encodings, but
weights each weight's squared reconstruction error by that input channel's activation second moment
``h[j] = E[x_j^2]``, damped by 1% of the vector's mean. That is the diagonal approximation of the local layer-output
error ``E||X (W - Q(W))||^2``: a weight column the model never excites much may be reconstructed loosely, and a
column that carries large activations may not. Unweighted MSE (``h`` constant) is exactly the special case, and the
two entry points then select byte-identical scales.

Two properties are load-bearing and deliberately not negotiable for the two exhaustive searches:

* **The global per-tensor scale is fixed.** It is read from the template, used unchanged, and handed to the new
  tensor as the very same object. Only the per-block E4M3 scales and the FP4 payload are recomputed. (The
  Four-Over-Six packer below is the single documented exception: its reference arithmetic *defines* a different
  global normalization, which it derives from the template's own scale.)
* **The wire format does not change.** The block size, the packed ``qdata`` layout and the Blackwell 128x4 swizzled
  E4M3 scale layout are exactly what the template carried, so the result stays consumable by the existing runtime
  without any kernel or dispatch change.

The quantization arithmetic reproduces pinned TorchAO 0.17 exactly -- the reciprocal
``(1 / per_tensor_scale) / scale_fp8``, the ``[-6, 6]`` clamp, ``f32_to_f4_unpacked``/``pack_uint4`` for the payload,
``f4_unpacked_to_f32(q) * (per_tensor_scale * scale_fp8)`` for the reconstruction, and ``to_blocked`` for the scale
swizzle -- by calling TorchAO's own kernels rather than reimplementing them.

:func:`repack_nvfp4_weight_four_over_six` is a third, deliberately *non*-exhaustive selection: it reproduces NVIDIA
ModelOpt 0.46.0's Four-Over-Six weight arithmetic. Each block is written either with its amax mapped onto FP4's
largest magnitude 6 (``M=6``) or onto 4 (``M=4``, the same scale multiplied by 1.5), and the representation with the
lower plain squared reconstruction error wins, ties going to ``M=6``. Because the ``M=4`` scale is 1.5x larger than
the ordinary one, the *global* scale is renormalized against :data:`NVFP4_FOUR_OVER_SIX_FP8_MAX` (256) rather than
448, which is the one place where this method does not keep the template's global scale fixed.

:func:`repack_nvfp4_weight_awq_clip` is a fourth selection and the only one that does not derive its scales from the
weight alone. It adapts NVIDIA ModelOpt 0.46.0's AWQ clipping: a *clipped* block scale is the ordinary one applied to
a shrunken block maximum ``block_amax * ratio``, with ``ratio`` one of the ten clipping ratios
``NVFP4_AWQ_CLIP_RATIOS[:10]``. Which ratio each output-row/input-block pair takes is decided **offline**, against
quantized activations, by :func:`select_nvfp4_ratio_codes_awq_clip`; the runtime only receives the resulting uint8
codes and reproduces the selected scales exactly. The global per-tensor scale, the packed layout and every wrapper
attribute are the template's, unchanged.

Code :data:`NVFP4_AWQ_CLIP_UNCLIPPED_CODE` -- ratio ``1.00`` -- is *not* recomputed at all. It means "leave this
block exactly as the ordinary conversion wrote it", and both the repack and the offline search take that block's
bytes and reconstruction from the supplied template rather than from any formula. That is the only way the unclipped
code can be byte-identical to the deployed ordinary template in both supported construction modes: pinned TorchAO's
non-Triton conversion floors its scaled block scale at ``finfo(float8_e4m3fn).tiny`` (``2 ** -6``) while the
accelerated MSLK kernel uses a pre-cast epsilon and stores an E4M3 zero for an all-zero block, and their packed
payloads differ on ordinary blocks too. Reproducing either by formula would store weights the other backend never
executes, so the template itself is the reference and :data:`AWQ_CLIP_TEMPLATE_ARITHMETICS` names which construction
an artifact was selected against.

:func:`select_nvfp4_gptq_payload` is a fifth method and the only one that does not touch a scale at all. It adapts
NVIDIA ModelOpt 0.46.0's GPTQ: the block scales and the global scale stay exactly the ones the *ordinary* template of
the original weight carries, and only the packed FP4 payload changes. Input columns are visited in order; each column
is written with the payload its fixed template scales quantize the current *working* value into, and the residual
``(w_col - q_col) / h_inv[i, i]`` is propagated to the columns that have not been written yet, over 128-column update
blocks, under the full input Hessian :func:`nvfp4_gptq_hessian` builds from quantized activation rows. Because the
scales, the swizzled scale buffer, the global scale and the wire format are the template's own,
:func:`repack_nvfp4_weight_gptq` replaces nothing but ``qdata``.

The search is exhaustive but memory-bounded: candidates and blocks are both processed in chunks, so the transient
``[block_chunk_size, candidate_chunk_size, 16]`` working set never grows with the size of the linear. Ties are
resolved deterministically towards the first candidate in ascending E4M3 bit order.

TorchAO is imported lazily, on the first call of a public entry point, so importing this module on a CPU-only or
minimal install neither imports nor requires TorchAO.
"""

import math
from typing import Any, NamedTuple, Optional, Sequence, Tuple

import torch

# NVFP4 / E4M3 constants, identical to pinned TorchAO ``torchao.prototype.mx_formats``.
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = 6.0
FP8_E4M3_MAX = 448.0

# 0..127 as E4M3 bit patterns is +0.0, 126 positive finite values, and NaN.
NVFP4_SCALE_CANDIDATE_COUNT = 126

# Four-Over-Six normalization maximum. The weight global scale is ``global_amax / (NVFP4_MAX * 256)`` instead of the
# ordinary ``global_amax / (NVFP4_MAX * 448)``, which is exactly the headroom the 1.5x ``M=4`` candidate needs: the
# largest scale this method can store is ``384 * block_amax / global_amax``, still inside E4M3's finite range.
NVFP4_FOUR_OVER_SIX_FP8_MAX = 256.0

# The two representations compared per 16-weight block, in evaluation order. The block amax is mapped onto FP4's
# largest magnitude 6 (``M=6``, scale multiplier 1.0) or onto 4 (``M=4``, scale multiplier 1.5). An exact tie keeps
# the first entry, ``M=6``.
NVFP4_FOUR_OVER_SIX_MAGNITUDES: Tuple[int, int] = (6, 4)

# The stored per-block scale is clamped into this range before it is rounded to E4M3, exactly as the reference
# static export does. The lower bound is the smallest positive (subnormal) E4M3 encoding.
NVFP4_FOUR_OVER_SIX_SCALE_MIN = 2.0**-9
NVFP4_FOUR_OVER_SIX_SCALE_MAX = FP8_E4M3_MAX

# NVIDIA ModelOpt 0.46.0 AWQ-clip candidate ratios, in the reference's exact insertion order. Each ratio shrinks the
# block's own absolute maximum before the ordinary NVFP4 block-scale arithmetic is applied to it, so the last entry
# -- ratio 1.00, code 10 -- is exactly the ordinary amax rule. The order *is* part of the algorithm's identity: a
# stored code indexes into this tuple, and an exact tie keeps the earliest ratio, 0.50.
NVFP4_AWQ_CLIP_RATIOS: Tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
NVFP4_AWQ_CLIP_RATIO_COUNT = len(NVFP4_AWQ_CLIP_RATIOS)

# Code of the unclipped ratio 1.00, i.e. of the ordinary amax rule. It is the baseline every other candidate is
# compared against and the one code that must reproduce an ordinary TorchAO conversion byte for byte.
NVFP4_AWQ_CLIP_UNCLIPPED_CODE = NVFP4_AWQ_CLIP_RATIO_COUNT - 1

# The AWQ-clip *clipped* block scale is clamped into this range before it is rounded to E4M3 -- the same clamp the
# Four-Over-Six export above applies, repeated under its own name because it belongs to this algorithm's recorded
# identity too. It applies to codes 0..9 only: the unclipped code never goes through this formula at all, because its
# stored scale is the template's own byte.
NVFP4_AWQ_CLIP_SCALE_MIN = 2.0**-9
NVFP4_AWQ_CLIP_SCALE_MAX = FP8_E4M3_MAX

# The two ways an ordinary NVFP4 template can be constructed, and the exact arithmetic the unclipped code therefore
# reproduces. They are *not* interchangeable: the non-Triton conversion and the accelerated MSLK kernel disagree on
# the E4M3 scale floor of a near-zero block and on the packed payload of ordinary blocks, so an artifact selected
# against one of them must never be deployed on the other. An artifact records which one it was built against and
# the runtime refuses the mismatch.
AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE = "torchao_non_triton"
AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED = "mslk_triton"
AWQ_CLIP_TEMPLATE_ARITHMETICS: Tuple[str, ...] = (
    AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE,
    AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED,
)

# NVIDIA ModelOpt 0.46.0 GPTQ constants. Unlike every method above, GPTQ does not select a scale at all: it keeps
# the ordinary template's scales fixed and changes only which FP4 payload each input column is written with, so that
# the rounding error of an already-quantized column is compensated by the columns that follow it. Each of these three
# values *is* the algorithm rather than a knob -- a different damping fraction, a different update block or a
# different Hessian normalization would propagate a different residual -- so all three are written into every
# artifact and re-checked on load.
NVFP4_GPTQ_UPDATE_BLOCK_SIZE = 128
NVFP4_GPTQ_PERC_DAMP = 0.01
# The factor of two in ``scaled = sqrt(2 / N_g) * Xq_g.T`` is the reference's own normalization. A global positive
# factor cancels in the GPTQ update -- the residual divides by a diagonal of the same Hessian's inverse factor -- but
# it does not cancel in the reported quadratic objective, and it is part of the recorded identity either way.
NVFP4_GPTQ_HESSIAN_FACTOR = 2.0

# GPTQ writes the payload of the *ordinary* template, whose two constructions do not produce the same bytes, so it
# inherits exactly the AWQ-clip template-arithmetic distinction and its fail-closed backend binding.
NVFP4_GPTQ_TEMPLATE_ARITHMETICS: Tuple[str, ...] = AWQ_CLIP_TEMPLATE_ARITHMETICS

# Search chunking defaults. They only bound peak memory; the selected scales do not depend on them.
DEFAULT_CANDIDATE_CHUNK_SIZE = 32
DEFAULT_BLOCK_CHUNK_SIZE = 8192

# Chunking defaults of the AWQ-clip ratio search. The transient working set is
# ``[block_chunk, row_chunk * 11, AWQ_CLIP_ACTIVATION_ROW_BATCH, 16]`` FP32 products, so both knobs bound peak
# memory only; the ratio codes and the reported objectives do not depend on them. The output-row default mirrors
# the reference's own output-row batching.
DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE = 64
DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE = 4

# Activation rows whose dot products are formed at once inside the AWQ-clip objective. This is a *fixed constant of
# the arithmetic* and deliberately not a caller knob: the FP64 accumulation of the per-row squares walks the rows in
# batches of exactly this size, so the reduction a given (output row, input block, candidate) triple goes through is
# the same however the caller chunks output rows or input blocks.
AWQ_CLIP_ACTIVATION_ROW_BATCH = 64

# High-precision weight dtypes this packer accepts.
SUPPORTED_WEIGHT_DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32)

# Fixed diagonal damping of the activation second moments, as a fraction of their own mean:
# ``h_damped = h + 0.01 * mean(h)``. It is a constant of the algorithm, not a tunable: a channel the samples never
# excited still gets a floor, so a block is never selected purely on channels that happened to be observed. Changing
# it would define a different algorithm, which is why the runtime artifact records it and the loader rejects
# anything else.
NVFP4_HESSIAN_DAMPING = 0.01

_TORCHAO_TENSOR_MODULE = "torchao.prototype.mx_formats.nvfp4_tensor"
_TORCHAO_KERNELS_MODULE = "torchao.prototype.mx_formats.kernels"
_TORCHAO_UTILS_MODULE = "torchao.prototype.mx_formats.utils"

# Populated lazily by :func:`_resolve_torchao`.
_torchao_backend = None

# Populated lazily by :func:`_resolve_awq_clip_readback`, which only the offline AWQ-clip candidate needs.
_awq_clip_readback = None


class TorchAOUnavailableError(ImportError, ValueError):
    """Raised when the pinned TorchAO NVFP4 API is absent or incompatible.

    It derives from both ``ImportError`` and ``ValueError`` so that callers which treat a missing optional backend as
    an import problem and callers which treat it as one more fail-closed contract violation both catch it.
    """


class FourOverSixSelection(NamedTuple):
    """The Four-Over-Six block scales plus the deterministic evidence of which representation won each block."""

    # ``(M, K // 16)`` linear (un-swizzled) E4M3 block scales.
    scale_fp8: torch.Tensor
    # ``(M, K // 16)`` int64 index into :data:`NVFP4_FOUR_OVER_SIX_MAGNITUDES`: 0 is ``M=6``, 1 is ``M=4``.
    magnitude_index: torch.Tensor
    m6_block_count: int
    m4_block_count: int

    @property
    def block_count(self) -> int:
        """Total number of 16-weight blocks; every block is counted under exactly one representation."""
        return self.m6_block_count + self.m4_block_count


class FourOverSixRepack(NamedTuple):
    """A Four-Over-Six repacked weight together with the per-representation block counts it was built from."""

    # The new ``type(template)`` instance holding the repacked weight.
    weight: Any
    block_count: int
    m6_block_count: int
    m4_block_count: int


class AWQClipSelection(NamedTuple):
    """The AWQ-clip ratio codes of one weight plus the deterministic offline evidence of how they were chosen."""

    # ``(M, K // 16)`` uint8 codes indexing :data:`NVFP4_AWQ_CLIP_RATIOS`; every entry is in ``[0, 10]``.
    ratio_codes: torch.Tensor
    # Mean over every ``(output row, input block)`` pair of the selected candidate's loss, reduced in FP64.
    selected_objective: float
    # The same mean under the unclipped ratio ``1.00``, i.e. under the ordinary amax rule, for comparison.
    unclipped_objective: float
    # Number of blocks that selected each ratio, in candidate order; the entries sum to the block count.
    code_counts: Tuple[int, ...]

    @property
    def block_count(self) -> int:
        """Total number of 16-weight blocks; every block selected exactly one ratio."""
        return int(sum(self.code_counts))


class NVFP4TemplateIdentity(NamedTuple):
    """The buffers of an ordinary NVFP4 template that a GPTQ payload replacement must leave byte-identical.

    ``scale`` and ``global_scale`` are the template's own tensors, not copies: a caller digests them to bind an
    artifact to the template the runtime actually produced, and the repack hands the very same objects to the new
    wrapper. ``qdata`` is the only buffer GPTQ ever replaces.
    """

    rows: int
    columns: int
    # The swizzled (Blackwell 128x4) E4M3 block-scale buffer, including its padding.
    scale: torch.Tensor
    # The per-tensor scale attribute, exactly as the template carries it.
    global_scale: torch.Tensor
    # The packed ``(M, K // 2)`` FP4 payload.
    qdata: torch.Tensor


class NVFP4GPTQHessian(NamedTuple):
    """A damped GPTQ Hessian together with the deterministic evidence of how it was prepared."""

    # ``(K, K)`` FP32 damped Hessian, ready for the Cholesky inverse.
    matrix: torch.Tensor
    # Columns whose original weight is exactly zero in every output row, zeroed and unit-diagonal before damping.
    dead_columns: int
    # The absolute damping added to the diagonal, i.e. ``0.01 * mean(diag(H))`` after the dead-column handling.
    damping: float
    diagonal_min: float
    diagonal_max: float
    diagonal_mean: float


class NVFP4GPTQSelection(NamedTuple):
    """The packed payload GPTQ selected for one weight, plus the values it decodes to."""

    # ``(M, K // 2)`` ``torch.uint8`` packed FP4 payload, ready to replace a template's ``qdata``.
    qdata: torch.Tensor
    # ``(M, K)`` FP32 values that payload decodes to under the template's own fixed scales.
    values: torch.Tensor


def repack_nvfp4_weight_mse(
    weight: torch.Tensor,
    template: Any,
    candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> Any:
    """Repack ``weight`` into a new ``NVFP4Tensor`` whose block scales minimize local reconstruction MSE.

    Every contiguous 16-weight block of ``weight`` is quantized once per positive finite E4M3 encoding, and the
    encoding whose dequantized block is closest to the original block in squared error is kept. The template's
    global per-tensor scale is used unchanged, and every wrapper/runtime attribute of ``template`` -- block size,
    original dtype, per-tensor and activation per-tensor scales, swizzled-scale flag, Triton flag and activation
    quantization kwargs -- is carried over as is, so the result is a drop-in replacement for ``template``.

    Neither ``weight`` nor ``template`` (nor any tensor owned by ``template``) is mutated.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, with ``K`` a multiple of 16,
            on the same device and of the same shape as ``template``.
        template: an already-converted TorchAO ``NVFP4Tensor`` for the same weight, with block size 16 and swizzled
            scales.
        candidate_chunk_size: number of E4M3 candidates evaluated at once; bounds peak memory only.
        block_chunk_size: number of 16-weight blocks evaluated at once; bounds peak memory only.

    Returns:
        A new instance of ``type(template)`` with re-selected block scales and the matching repacked ``qdata``.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor`` or ``weight`` is not a tensor.
        ValueError: for any violated shape, dtype, device, finiteness, metadata or chunk-size contract.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    metadata = _validate_template(template, backend)
    _validate_weight(weight, metadata.rows, metadata.columns, metadata.device)
    _validate_chunk_sizes(candidate_chunk_size, block_chunk_size)

    weight_f32 = weight.detach().to(torch.float32)
    scale_fp8 = _select_block_scales(weight_f32, metadata.per_tensor_scale, candidate_chunk_size, block_chunk_size)
    qdata = _pack_qdata(weight_f32, scale_fp8, metadata.per_tensor_scale, backend)

    return _build_nvfp4_tensor(template, metadata, qdata, scale_fp8, backend)


def repack_nvfp4_weight_local_hessian(
    weight: torch.Tensor,
    template: Any,
    second_moments: torch.Tensor,
    candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> Any:
    """Repack ``weight`` into a new ``NVFP4Tensor`` whose block scales minimize the diagonal local-Hessian error.

    This is :func:`repack_nvfp4_weight_mse` with one change: the squared error of column ``j`` is weighted by the
    damped input-channel second moment ``h_damped[j]``, so the objective of block ``b`` of output row ``r`` is

        ``sum_{j in b} h_damped[j] * (W[r, j] - Q_scale(W[r, j])) ** 2``.

    Everything else -- the 126 searched encodings, TorchAO's reconstruction arithmetic, the fixed global per-tensor
    scale, the payload packing, the scale swizzle, the chunk bounds and the smallest-scale tie break -- is shared
    with the MSE repacker, so the produced wire format and every carried wrapper attribute are identical.

    Neither ``weight``, ``second_moments`` nor ``template`` (nor any tensor owned by ``template``) is mutated.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, as for
            :func:`repack_nvfp4_weight_mse`.
        template: an already-converted TorchAO ``NVFP4Tensor`` for the same weight.
        second_moments: a rank-1 finite non-negative ``(K,)`` float tensor of input-channel second moments
            ``E[x_j^2]``, on the same device as ``weight``, with a finite positive mean.
        candidate_chunk_size: number of E4M3 candidates evaluated at once; bounds peak memory only.
        block_chunk_size: number of 16-weight blocks evaluated at once; bounds peak memory only.

    Returns:
        A new instance of ``type(template)`` with re-selected block scales and the matching repacked ``qdata``.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor``, or ``weight``/``second_moments`` is not a tensor.
        ValueError: for any violated shape, dtype, device, finiteness, metadata or chunk-size contract.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    metadata = _validate_template(template, backend)
    _validate_weight(weight, metadata.rows, metadata.columns, metadata.device)
    _validate_chunk_sizes(candidate_chunk_size, block_chunk_size)
    moments = _search_weights(damped_second_moments(second_moments, metadata.columns, metadata.device))

    weight_f32 = weight.detach().to(torch.float32)
    scale_fp8 = _select_block_scales(
        weight_f32, metadata.per_tensor_scale, candidate_chunk_size, block_chunk_size, moments
    )
    qdata = _pack_qdata(weight_f32, scale_fp8, metadata.per_tensor_scale, backend)

    return _build_nvfp4_tensor(template, metadata, qdata, scale_fp8, backend)


def repack_nvfp4_weight_four_over_six(
    weight: torch.Tensor,
    template: Any,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> FourOverSixRepack:
    """Repack ``weight`` into a new ``NVFP4Tensor`` with NVIDIA ModelOpt 0.46.0's Four-Over-Six weight arithmetic.

    Exactly two candidates are evaluated per contiguous 16-weight block, in this order:

    * ``M=6`` -- the block amax mapped onto FP4's largest magnitude 6, i.e. the ordinary rule in the renormalized
      basis, ``scale = (amax / 6) / per_tensor_scale``;
    * ``M=4`` -- the block amax mapped onto 4 instead, ``scale = (amax / 4) / per_tensor_scale``, which is the same
      scale multiplied by 1.5 and spends the block's range on FP4's denser codes at the cost of no clipping headroom.

    Each candidate is clamped into ``[2 ** -9, 448]`` and rounded to E4M3 *before* it is scored, and the score is the
    plain FP32 squared reconstruction error of the block after the full E4M3 -> E2M1 -> dequantize round trip. The
    lower error wins; an exact tie keeps ``M=6``. A block whose unnormalized scale is zero -- an all-zero block --
    takes the reference's substitution of ``1.0`` for that unnormalized scale before normalization and clamping.

    Unlike the two exhaustive searches in this module, this method does *not* keep the template's global scale: its
    reference arithmetic normalizes the weight global scale against :data:`NVFP4_FOUR_OVER_SIX_FP8_MAX` instead of
    448, so the new scale is exactly the template's own scale times ``448 / 256`` (see
    :func:`four_over_six_global_scale`). Everything else -- logical shape, block size, packed ``qdata`` layout,
    swizzled E4M3 scale layout, original dtype, activation per-tensor scale, swizzle flag, Triton flag and activation
    quantization kwargs -- is the template's, unchanged.

    Neither ``weight`` nor ``template`` (nor any tensor owned by ``template``) is mutated.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, with ``K`` a multiple of 16,
            on the same device and of the same shape as ``template``.
        template: an already-converted TorchAO ``NVFP4Tensor`` for the same weight, carrying the *ordinary*
            (448-normalized) global per-tensor scale, with block size 16 and swizzled scales.
        block_chunk_size: number of 16-weight blocks evaluated at once; bounds peak memory only.

    Returns:
        A :class:`FourOverSixRepack` holding a new instance of ``type(template)`` and the ``M=6``/``M=4`` block
        counts, so a caller can report them without re-quantizing the weight.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor`` or ``weight`` is not a tensor.
        ValueError: for any violated shape, dtype, device, finiteness, metadata or chunk-size contract, including a
            renormalized global scale that is not finite and positive.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    metadata = _validate_template(template, backend)
    _validate_weight(weight, metadata.rows, metadata.columns, metadata.device)
    _validate_positive_int(block_chunk_size, "block_chunk_size")

    # A new tensor of the template's own shape, dtype and device; the template's scale is never mutated.
    per_tensor_scale_attr = four_over_six_global_scale(metadata.per_tensor_scale_attr)
    per_tensor_scale = _as_scalar_scale(per_tensor_scale_attr, metadata.device)

    weight_f32 = weight.detach().to(torch.float32)
    selection = _select_four_over_six_block_scales(weight_f32, per_tensor_scale, block_chunk_size)
    qdata = _pack_qdata(weight_f32, selection.scale_fp8, per_tensor_scale, backend)
    repacked = _build_nvfp4_tensor(
        template,
        metadata._replace(per_tensor_scale_attr=per_tensor_scale_attr, per_tensor_scale=per_tensor_scale),
        qdata,
        selection.scale_fp8,
        backend,
    )
    return FourOverSixRepack(
        weight=repacked,
        block_count=selection.block_count,
        m6_block_count=selection.m6_block_count,
        m4_block_count=selection.m4_block_count,
    )


def repack_nvfp4_weight_awq_clip(
    weight: torch.Tensor,
    template: Any,
    ratio_codes: torch.Tensor,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> Any:
    """Repack ``weight`` into a new ``NVFP4Tensor`` whose block scales are the AWQ-clip codes' scales.

    ``ratio_codes`` is the ``(M, K // 16)`` uint8 matrix an offline AWQ-clip artifact carries: entry ``[r, b]`` is
    the index into :data:`NVFP4_AWQ_CLIP_RATIOS` that was selected for output row ``r`` and input block ``b``. This
    function performs no search at all. A *clipped* code ``0..9`` is reconstructed as exactly the candidate it names,

        ``scale_fp8 = E4M3(clamp(((block_amax * ratio) / 6) / per_tensor_scale, 2 ** -9, 448))``

    packed into the matching FP4 payload with TorchAO's own kernels. The unclipped code
    :data:`NVFP4_AWQ_CLIP_UNCLIPPED_CODE` is not reconstructed at all: that block's stored E4M3 scale byte and its
    eight packed payload bytes are **copied out of the supplied template**, so an all-``10`` code matrix produces a
    new wrapper whose complete ``qdata`` and scale buffers are byte-identical to the template's, whichever of
    :data:`AWQ_CLIP_TEMPLATE_ARITHMETICS` constructed it. Reconstructing that code by formula instead could only ever
    match one of the two constructions; see this module's docstring.

    Copying is exact per block because both buffers are block-aligned: block ``b`` of output row ``r`` owns the eight
    contiguous ``qdata`` bytes ``[8 * b, 8 * b + 8)`` of that row -- 16 nibbles, so no packed pair ever straddles two
    blocks -- and exactly one E4M3 scale byte, which the same ``to_blocked`` swizzle places for the codes and for the
    marker that decides which byte to keep.

    The template's global per-tensor scale is used unchanged and handed to the new tensor as the very same object,
    and every wrapper/runtime attribute of ``template`` -- block size, original dtype, activation per-tensor scale,
    swizzled-scale flag, Triton flag and activation quantization kwargs -- is carried over as is. The result is
    always a *new* tensor; the template is never returned and never mutated.

    Neither ``weight``, ``ratio_codes`` nor ``template`` (nor any tensor owned by ``template``) is mutated.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, with ``K`` a multiple of 16,
            on the same device and of the same shape as ``template``.
        template: an already-converted TorchAO ``NVFP4Tensor`` for the same weight, with block size 16 and swizzled
            scales.
        ratio_codes: a contiguous ``(M, K // 16)`` ``torch.uint8`` tensor on the template's device, every entry in
            ``[0, 10]``.
        block_chunk_size: number of 16-weight blocks whose scales are rebuilt at once; bounds peak memory only.

    Returns:
        A new instance of ``type(template)`` holding the selected block scales and the matching repacked ``qdata``.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor``, or ``weight``/``ratio_codes`` is not a tensor.
        ValueError: for any violated shape, dtype, device, range, finiteness, metadata or chunk-size contract.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    metadata = _validate_template(template, backend)
    _validate_weight(weight, metadata.rows, metadata.columns, metadata.device)
    _validate_positive_int(block_chunk_size, "block_chunk_size")
    num_blocks = metadata.columns // NVFP4_BLOCK_SIZE
    codes = _validate_ratio_codes(ratio_codes, metadata.rows, num_blocks, metadata.device)

    # ``True`` exactly where the block is *clipped*, i.e. where this repack overwrites the template. Everything the
    # marker leaves ``False`` -- every unclipped block and every swizzle padding byte -- keeps the template's byte.
    clipped = codes != NVFP4_AWQ_CLIP_UNCLIPPED_CODE
    weight_f32 = weight.detach().to(torch.float32)
    # The unclipped blocks' entries of these two buffers are placeholders that the merges below discard; they are
    # still computed for the whole weight because doing so keeps the packing one bounded batched pass.
    scale_fp8 = _awq_clip_block_scales(weight_f32, codes, metadata.per_tensor_scale, block_chunk_size)
    qdata = _awq_clip_merged_qdata(
        _pack_qdata(weight_f32, scale_fp8, metadata.per_tensor_scale, backend),
        metadata.qdata,
        clipped,
        metadata.rows,
        metadata.columns,
    )
    blocked_scale = _awq_clip_merged_scale(scale_fp8, metadata.scale, clipped, backend)

    return _build_nvfp4_tensor(template, metadata, qdata, None, backend, blocked_scale=blocked_scale)


def select_nvfp4_ratio_codes_awq_clip(
    weight: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    group_activations: Sequence[torch.Tensor],
    template_reconstruction: torch.Tensor,
    row_chunk_size: int = DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE,
) -> AWQClipSelection:
    """Choose one AWQ-clip ratio code per output row and input block, against quantized activations.

    This is the *offline* half of the method: it is what an artifact builder runs, never the runtime. For output
    row ``r``, contiguous 16-column input block ``b`` and candidate ``c``, with
    ``delta = W[r, b] - Q_c(W[r, b])`` the block's reconstruction error, the loss is the block-local output error

        ``mean_g(mean_rows((sum_j Xq_g[row, j] * delta[j]) ** 2))``

    over the equally weighted groups ``g`` of ``group_activations``. Every group counts once however many rows it
    contributed. This is a *full-covariance* block-local objective: it keeps the within-block cross terms that the
    diagonal local-Hessian objective drops, and it scores no interaction between different blocks.

    ``Q_c`` is exactly what the runtime will deploy for that code, which is why the two halves cannot drift apart:
    the ten clipped candidates are :func:`nvfp4_awq_clip_candidate_scales`' ratios ``0.50..0.95`` reconstructed with
    TorchAO's arithmetic, and the eleventh -- the unclipped code -- is ``template_reconstruction``, the FP32 decode
    of the already-converted ordinary template's own stored bytes. :func:`repack_nvfp4_weight_awq_clip` copies that
    block's bytes straight out of the template, so scoring anything else here would rank a reconstruction nobody
    executes; decoding them in FP32 is what puts the eleventh delta on the same footing as the other ten.

    The dot products multiply FP32 activations by FP32 weight deltas and are themselves reduced in FP32 over the
    fixed 16-value block, which is the precision the deployed kernel would compute them in. Only their *results* are
    widened to FP64, before they are squared and before any row or group is accumulated, so a long reduction cannot
    lose a winner and neither the order of the groups nor either chunk size can move one. Candidates are compared in
    the fixed order of :data:`NVFP4_AWQ_CLIP_RATIOS` and a
    candidate only replaces the incumbent on a strictly smaller loss, so an exact tie -- an all-zero block, whose
    eleven candidates all reconstruct it identically -- keeps the earliest ratio, code ``0``.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, ``K`` a multiple of 16.
        per_tensor_scale: the weight's global per-tensor scale, as :func:`nvfp4_weight_global_scale` derives it.
        group_activations: one non-empty FP32 ``(N_g, K)`` tensor per equally weighted source group, on ``weight``'s
            device, already passed through :func:`nvfp4_awq_clip_activation_qdq`.
        template_reconstruction: the ``(M, K)`` float values the ordinary NVFP4 template of this weight decodes to,
            on ``weight``'s device, as :func:`nvfp4_awq_clip_template_reconstruction` produces them for the
            deployment's own construction mode. This is the unclipped candidate.
        row_chunk_size: number of output rows scored at once; bounds peak memory only.
        block_chunk_size: number of input blocks scored at once; bounds peak memory only.

    Returns:
        An :class:`AWQClipSelection` with the uint8 codes, the selected and unclipped objectives, and the per-code
        block counts.

    Raises:
        TypeError: if ``weight``, ``per_tensor_scale``, ``template_reconstruction`` or a group is not a tensor.
        ValueError: for any violated shape, dtype, device, finiteness or chunk-size contract, or if a candidate
            loss is not finite.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    _validate_positive_int(row_chunk_size, "row_chunk_size")
    _validate_positive_int(block_chunk_size, "block_chunk_size")
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    num_blocks = columns // NVFP4_BLOCK_SIZE
    groups = _validate_group_activations(group_activations, columns, weight.device)
    template_blocks = _validate_template_reconstruction(template_reconstruction, weight).reshape(
        rows, num_blocks, NVFP4_BLOCK_SIZE
    )

    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    blocks = weight.detach().to(torch.float32).reshape(rows, num_blocks, NVFP4_BLOCK_SIZE)
    # Only the ten *clipped* ratios are reconstructed by formula; the eleventh candidate is the template itself.
    ratios = torch.tensor(
        NVFP4_AWQ_CLIP_RATIOS[:NVFP4_AWQ_CLIP_UNCLIPPED_CODE], dtype=torch.float32, device=weight.device
    )
    codes = torch.zeros(rows, num_blocks, dtype=torch.uint8, device=weight.device)
    # Filled block by block and reduced once at the end, so the reported objectives are as chunk-independent as
    # the codes are.
    selected_loss = torch.empty(rows, num_blocks, dtype=torch.float64, device=weight.device)
    unclipped_loss = torch.empty(rows, num_blocks, dtype=torch.float64, device=weight.device)

    for block_start in range(0, num_blocks, block_chunk_size):
        block_stop = min(block_start + block_chunk_size, num_blocks)
        # ``[B, N_g, 16]`` FP32 activation operands of this block chunk, built once per chunk and reused by every
        # output-row chunk so the retained rows are read once per chunk instead of once per row batch.
        group_blocks = []
        for group in groups:
            group_rows = group.reshape(int(group.shape[0]), num_blocks, NVFP4_BLOCK_SIZE)
            block_rows = group_rows[:, block_start:block_stop].permute(1, 0, 2)
            group_blocks.append(block_rows.contiguous())
        for row_start in range(0, rows, row_chunk_size):
            row_stop = min(row_start + row_chunk_size, rows)
            chunk = blocks[row_start:row_stop, block_start:block_stop]
            candidates = _awq_clip_candidate_scales(chunk, ratios, scale)
            # [R, B, C, 16] is the whole transient weight-side working set of the selection. The unclipped
            # candidate is appended last, in the position its code names, and is the template's own decoded bytes.
            reconstruction = torch.cat(
                (
                    _reconstruct(chunk[:, :, None, :], candidates[..., None], scale, backend),
                    template_blocks[row_start:row_stop, block_start:block_stop, None, :],
                ),
                dim=2,
            )
            loss = _awq_clip_losses(chunk[:, :, None, :] - reconstruction, group_blocks)
            if not bool(torch.isfinite(loss).all()):
                raise ValueError(
                    "The AWQ-clip objective is not finite for every candidate of output rows "
                    f"[{row_start}, {row_stop}) and input blocks [{block_start}, {block_stop}); the quantized "
                    "activations or the weight cannot support a selection."
                )
            best, best_index = _awq_clip_winners(loss)
            codes[row_start:row_stop, block_start:block_stop] = best_index.to(torch.uint8)
            selected_loss[row_start:row_stop, block_start:block_stop] = best
            unclipped_loss[row_start:row_stop, block_start:block_stop] = loss[..., NVFP4_AWQ_CLIP_UNCLIPPED_CODE]

    counts = torch.bincount(codes.reshape(-1).to(torch.long), minlength=NVFP4_AWQ_CLIP_RATIO_COUNT)
    block_count = float(rows * num_blocks)
    return AWQClipSelection(
        ratio_codes=codes,
        selected_objective=float(selected_loss.reshape(-1).sum() / block_count),
        unclipped_objective=float(unclipped_loss.reshape(-1).sum() / block_count),
        code_counts=tuple(int(value) for value in counts.tolist()),
    )


def nvfp4_awq_clip_template_reconstruction(
    weight: torch.Tensor, per_tensor_scale: torch.Tensor, template_arithmetic: str
) -> torch.Tensor:
    """Return the FP32 values an *ordinary* NVFP4 template of ``weight`` decodes to, in one construction mode.

    This is the offline half's unclipped candidate and nothing else: the builder converts the weight exactly as the
    deployment will -- through TorchAO's own ``NVFP4Tensor.to_nvfp4`` with the ordinary global scale, the swizzled
    Blackwell layout and the mode ``template_arithmetic`` names -- and then decodes *that template's own stored
    bytes*: its packed FP4 payload through TorchAO's ``unpack_uint4``/``f4_unpacked_to_f32``, and its blocked E4M3
    scales un-swizzled through TorchAO's ``from_blocked``, multiplied by the global scale in FP32. Because
    :func:`repack_nvfp4_weight_awq_clip` copies an unclipped block's bytes out of the runtime template rather than
    recomputing them, decoding exactly those bytes is what makes the offline objective rank the reconstruction the
    runtime actually deploys.

    The decode is deliberately *not* the wrapper's own ``dequantize``: pinned TorchAO's ``get_hp_scales`` casts the
    stored E4M3 block scale through the template's ``orig_dtype`` -- BF16 for these weights -- before the global
    scale is applied, so a wrapper readback is a BF16-influenced view of the deployed bytes. Every other candidate's
    delta is FP32, and the eleventh has to be measured on the same footing or the eleven losses would not be
    comparable. The bytes decoded here are the very bytes the wrapper holds; only the scale arithmetic is FP32
    throughout, exactly as :func:`_reconstruct` computes it for the ten clipped candidates.

    The two modes are not interchangeable and neither is derivable from the other, which is why the caller must say
    which one it is building for and why the choice is recorded in the artifact rather than assumed.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, ``K`` a multiple of 16.
        per_tensor_scale: the ordinary global per-tensor scale, as :func:`nvfp4_weight_global_scale` derives it.
        template_arithmetic: one of :data:`AWQ_CLIP_TEMPLATE_ARITHMETICS`.

    Returns:
        A new contiguous FP32 ``(M, K)`` tensor on ``weight``'s device.

    Raises:
        TypeError: if ``weight`` or ``per_tensor_scale`` is not a tensor.
        ValueError: for an unknown mode, a violated weight/scale contract, or a template this build cannot decode
            as a finite tensor of the weight's own shape.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported, does not expose ``to_nvfp4``
            with the pinned keywords, or does not expose the pinned ``unpack_uint4``/``from_blocked`` helpers.
    """
    backend = _resolve_torchao()
    readback = _resolve_awq_clip_readback()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    if template_arithmetic not in AWQ_CLIP_TEMPLATE_ARITHMETICS:
        raise ValueError(
            f"template_arithmetic must be one of {list(AWQ_CLIP_TEMPLATE_ARITHMETICS)}, got "
            f"{template_arithmetic!r}; the unclipped candidate is the template that mode constructs and there is "
            "no default."
        )
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    template = _ordinary_nvfp4_template(weight.detach(), scale, template_arithmetic, backend)
    # Validated like any other template this module reads, so a wrapper whose buffers cannot be addressed as the
    # deployed bytes is refused rather than decoded on a guess.
    metadata = _validate_template(template, backend)
    if (metadata.rows, metadata.columns) != (int(weight.shape[0]), int(weight.shape[1])):
        raise ValueError(
            f"The ordinary NVFP4 template of a {tuple(weight.shape)} weight covers "
            f"({metadata.rows}, {metadata.columns}); the installed torchao does not convert as this builder "
            "requires."
        )
    values = _decode_nvfp4_template_values(metadata, backend, readback)
    if tuple(values.shape) != tuple(weight.shape):
        raise ValueError(
            f"The ordinary NVFP4 template of a {tuple(weight.shape)} weight decoded to {tuple(values.shape)}; "
            "the installed torchao does not store its payload as this builder requires."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError(
            "The ordinary NVFP4 template decoded to non-finite values; refusing to score an AWQ-clip objective "
            "against a template the runtime could not execute."
        )
    return values.contiguous()


def select_nvfp4_block_scales_awq_clip(
    weight: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    ratio_codes: torch.Tensor,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> torch.Tensor:
    """Return the E4M3 *clipped-candidate* block scales the given AWQ-clip codes name, without an ``NVFP4Tensor``.

    This is the formula :func:`repack_nvfp4_weight_awq_clip` packs the clipped codes ``0..9`` with, exposed for
    diagnostics and tests. ``weight`` is a contiguous rank-2 ``(M, K)`` BF16 or FP32 tensor with ``K`` a multiple of
    16 and ``ratio_codes`` a ``(M, K // 16)`` uint8 tensor on its device; the result is an ``(M, K // 16)``
    ``torch.float8_e4m3fn`` tensor of *linear* (un-swizzled) block scales.

    An entry for the unclipped code :data:`NVFP4_AWQ_CLIP_UNCLIPPED_CODE` is the ratio-``1.00`` value of that same
    formula and is **not** what the repack stores: that block keeps the template's own scale byte, which the
    formula cannot reproduce in both construction modes. Read such an entry as "the amax rule in this basis", never
    as the deployed scale.
    """
    _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    _validate_positive_int(block_chunk_size, "block_chunk_size")
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    codes = _validate_ratio_codes(ratio_codes, rows, columns // NVFP4_BLOCK_SIZE, weight.device)
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    return _awq_clip_block_scales(weight.detach().to(torch.float32), codes, scale, block_chunk_size)


def nvfp4_awq_clip_candidate_scales(weight: torch.Tensor, per_tensor_scale: torch.Tensor) -> torch.Tensor:
    """Return every block's eleven AWQ-clip candidate scales, as FP32, in :data:`NVFP4_AWQ_CLIP_RATIOS` order.

    The result has shape ``(M, K // 16, 11)`` and holds exactly the values
    ``E4M3(clamp(((block_amax * ratio) / 6) / per_tensor_scale, 2 ** -9, 448))``, in that order of operations: the
    clipped block maximum is mapped onto FP4's largest magnitude, normalized by the global scale, clamped and only
    then rounded to E4M3, because any other order would round a different value. Every returned value is therefore
    an exact E4M3 encoding, and the last column -- ratio ``1.00`` -- is the ordinary amax rule *in this formula's
    basis*, which is the amax scale of any block the ``2 ** -9`` floor does not decide. Neither the search nor the
    repack uses that last column: the unclipped code takes the supplied template's own scale byte instead.
    """
    _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    ratios = torch.tensor(NVFP4_AWQ_CLIP_RATIOS, dtype=torch.float32, device=weight.device)
    blocks = weight.detach().to(torch.float32).reshape(rows, columns // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    return _awq_clip_candidate_scales(blocks, ratios, scale)


def nvfp4_awq_clip_activation_qdq(activations: torch.Tensor, activation_amax: float) -> torch.Tensor:
    """Quantize and dequantize activation rows exactly as the static NVFP4 runtime will.

    ``activation_amax`` is the calibrated per-tensor maximum *after* the configured scale margin, from which
    TorchAO 0.17 derives the activation global scale ``activation_amax / (448 * 6)``. Every contiguous 16-value
    block then takes its own dynamic E4M3 scale, and the payload is produced and read back through TorchAO's own
    ``f32_to_f4_unpacked``/``f4_unpacked_to_f32`` kernels, with the same reciprocal ordering and ``[-6, 6]`` clamp
    the deployed kernel uses. The rows an AWQ-clip search scores are therefore the rows the quantized model will
    actually multiply, not their BF16 originals.

    Args:
        activations: a non-empty rank-2 ``(N, K)`` float tensor with ``K`` a multiple of 16 and finite values.
        activation_amax: the finite positive calibrated activation maximum of this module.

    Returns:
        A new FP32 ``(N, K)`` tensor of reconstructed activations, on ``activations``' device.

    Raises:
        TypeError: if ``activations`` is not a tensor.
        ValueError: if the rows, the amax or the reconstruction are unusable.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    if not isinstance(activations, torch.Tensor):
        raise TypeError(f"activations must be a torch.Tensor, got {type(activations).__name__}.")
    if activations.dim() != 2 or int(activations.shape[0]) == 0 or int(activations.shape[1]) == 0:
        raise ValueError(f"activations must be a non-empty rank-2 (N, K) tensor, got {tuple(activations.shape)}.")
    if not activations.is_floating_point():
        raise ValueError(f"activations must be a float tensor, got {activations.dtype}.")
    if int(activations.shape[1]) % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"activations must have K a multiple of {NVFP4_BLOCK_SIZE}, got K={int(activations.shape[1])}."
        )
    if isinstance(activation_amax, bool) or not isinstance(activation_amax, (int, float)):
        raise ValueError(f"activation_amax must be a number, got {activation_amax!r}.")
    amax = float(activation_amax)
    if not math.isfinite(amax) or amax <= 0.0:
        raise ValueError(f"activation_amax must be finite and positive, got {activation_amax!r}.")

    values = activations.detach().to(torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("activations contain non-finite values; NVFP4 activation quantization is undefined here.")

    rows, columns = int(values.shape[0]), int(values.shape[1])
    calibrated = torch.tensor(amax, dtype=torch.float32, device=values.device).reshape(())
    global_scale = calibrated / (NVFP4_MAX * FP8_E4M3_MAX)
    _validate_per_tensor_scale(global_scale)
    blocks = values.reshape(rows, columns // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = torch.clamp(
        (block_amax / NVFP4_MAX) / global_scale, NVFP4_AWQ_CLIP_SCALE_MIN, NVFP4_AWQ_CLIP_SCALE_MAX
    )
    block_scale = block_scale.to(torch.float8_e4m3fn).to(torch.float32)
    reconstruction = _reconstruct(blocks, block_scale, global_scale, backend).reshape(rows, columns)
    if not bool(torch.isfinite(reconstruction).all()):
        raise ValueError(
            "The NVFP4 activation quantize/dequantize produced non-finite rows; refusing to score an AWQ-clip "
            "objective against them."
        )
    return reconstruction.contiguous()


def nvfp4_weight_global_scale(weight: torch.Tensor) -> torch.Tensor:
    """Return the ordinary TorchAO weight per-tensor scale ``global_amax / (448 * 6)``, as an FP32 scalar.

    This reproduces the dynamic per-tensor scale the NVFP4 conversion derives from a weight, so that an offline
    search scores exactly the scale basis the runtime template will carry. The runtime never uses this value: it
    keeps the template's own scale object, which this only has to agree with.

    Raises:
        TypeError: if ``weight`` is not a tensor.
        ValueError: if the weight is empty or its derived scale is not positive and finite.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}.")
    if weight.numel() == 0:
        raise ValueError("weight is empty, so it has no global amax to derive a per-tensor scale from.")
    scale = weight.detach().abs().max().to(torch.float32).reshape(()) / (NVFP4_MAX * FP8_E4M3_MAX)
    _validate_per_tensor_scale(scale)
    return scale


def four_over_six_global_scale(per_tensor_scale: torch.Tensor) -> torch.Tensor:
    """Return the Four-Over-Six weight global scale: the ordinary template scale times ``448 / 256``.

    The ordinary TorchAO conversion derives its global scale as ``global_amax / (6 * 448)``; the reference
    Four-Over-Six export derives it as ``global_amax / (6 * 256)``. Multiplying the template's own scale by
    ``448 / 256`` is therefore the exact reference normalization for a weight the ordinary recipe already converted,
    and it keeps the runtime's global-scale convention explicit instead of re-deriving a global amax here.

    The result is a *new* tensor with the shape, dtype and device of ``per_tensor_scale``; the input is never
    mutated. It is re-validated, because a low-precision scale dtype could overflow on the way.

    Raises:
        TypeError: if ``per_tensor_scale`` is not a tensor.
        ValueError: if it is not a positive finite scalar, or the rescaled value is not.
    """
    _validate_per_tensor_scale(per_tensor_scale)
    rescaled = per_tensor_scale.detach() * (FP8_E4M3_MAX / NVFP4_FOUR_OVER_SIX_FP8_MAX)
    _validate_per_tensor_scale(rescaled)
    return rescaled


def select_nvfp4_block_scales_four_over_six(
    weight: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> FourOverSixSelection:
    """Return the Four-Over-Six E4M3 block scales for ``weight``, without building an ``NVFP4Tensor``.

    This is the two-candidate selection itself, exposed for diagnostics and tests. ``weight`` is a contiguous rank-2
    ``(M, K)`` BF16 or FP32 tensor with ``K`` a multiple of 16, and ``per_tensor_scale`` is the *Four-Over-Six*
    global scale -- the one :func:`four_over_six_global_scale` produces, not the template's ordinary one.
    """
    _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    _validate_positive_int(block_chunk_size, "block_chunk_size")
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    return _select_four_over_six_block_scales(weight.detach().to(torch.float32), scale, block_chunk_size)


def select_nvfp4_block_scales(
    weight: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> torch.Tensor:
    """Return the MSE-optimal E4M3 block scales for ``weight``, without building an ``NVFP4Tensor``.

    This is the search itself, exposed for diagnostics and tests. ``weight`` is a contiguous rank-2 ``(M, K)`` BF16
    or FP32 tensor with ``K`` a multiple of 16; the result is an ``(M, K // 16)`` ``torch.float8_e4m3fn`` tensor of
    *linear* (un-swizzled) block scales, on ``weight``'s device.
    """
    _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    _validate_chunk_sizes(candidate_chunk_size, block_chunk_size)
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    return _select_block_scales(weight.detach().to(torch.float32), scale, candidate_chunk_size, block_chunk_size)


def select_nvfp4_block_scales_local_hessian(
    weight: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    second_moments: torch.Tensor,
    candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_BLOCK_CHUNK_SIZE,
) -> torch.Tensor:
    """Return the activation-weighted E4M3 block scales for ``weight``, without building an ``NVFP4Tensor``.

    This is the weighted search itself, exposed for diagnostics and tests. ``weight`` is a contiguous rank-2
    ``(M, K)`` BF16 or FP32 tensor with ``K`` a multiple of 16, ``second_moments`` is the ``(K,)`` second-moment
    vector on the same device, and the result is an ``(M, K // 16)`` ``torch.float8_e4m3fn`` tensor of *linear*
    (un-swizzled) block scales.

    With a constant ``second_moments`` the search weights of :func:`_search_weights` are exactly ``1.0`` and this
    returns the very same scales as :func:`select_nvfp4_block_scales`.
    """
    _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    _validate_chunk_sizes(candidate_chunk_size, block_chunk_size)
    moments = _search_weights(damped_second_moments(second_moments, int(weight.shape[1]), weight.device))
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    return _select_block_scales(
        weight.detach().to(torch.float32), scale, candidate_chunk_size, block_chunk_size, moments
    )


def nvfp4_blockwise_local_hessian_objective(
    weight: torch.Tensor,
    scale_fp8: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    second_moments: torch.Tensor,
) -> torch.Tensor:
    """Return the per-block activation-weighted reconstruction error under the given E4M3 block scales.

    This is :func:`nvfp4_blockwise_mse` with the exact damped second moments folded in: each block's value is the
    mean over its 16 weights of ``h_damped[j] * (W[r, j] - Q(W[r, j])) ** 2``, i.e. the objective
    ``sum_{j in b} h_damped[j] * (W[r, j] - Q(W[r, j])) ** 2`` that
    :func:`select_nvfp4_block_scales_local_hessian` minimizes, divided by the block size 16 and by nothing else.

    The values are therefore *absolute*, in the units the second moments carry: multiplying ``second_moments`` by a
    positive constant multiplies them by exactly that constant, and two blocks -- or two layers built on the same
    statistics -- are directly comparable. The search's own internal max-normalization (:func:`_search_weights`)
    cannot move an argmin but would rescale each layer separately, so it is deliberately not applied here.
    """
    backend = _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    num_blocks = columns // NVFP4_BLOCK_SIZE
    if tuple(scale_fp8.shape) != (rows, num_blocks):
        raise ValueError(f"scale_fp8 must have shape ({rows}, {num_blocks}), got {tuple(scale_fp8.shape)}.")
    moments = damped_second_moments(second_moments, columns, weight.device)

    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    blocks = weight.detach().to(torch.float32).reshape(rows * num_blocks, NVFP4_BLOCK_SIZE)
    scales = scale_fp8.reshape(rows * num_blocks, 1).float()
    reconstruction = _reconstruct(blocks, scales, scale, backend)
    weighted = ((reconstruction - blocks) ** 2) * moments.reshape(1, num_blocks, NVFP4_BLOCK_SIZE).expand(
        rows, num_blocks, NVFP4_BLOCK_SIZE
    ).reshape(rows * num_blocks, NVFP4_BLOCK_SIZE)
    return weighted.mean(dim=-1).reshape(rows, num_blocks)


def damped_second_moments(
    second_moments: torch.Tensor, in_features: Optional[int] = None, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Validate an input-channel second-moment vector and return the exact damped vector, as FP32.

    The returned vector is ``h_damped = h + 0.01 * mean(h)`` and nothing else: it is the objective's own weight
    vector, in the units the second moments carry, so an error evaluated with it is the absolute
    ``sum_j h_damped[j] * (W - Q(W)) ** 2``. The damping is :data:`NVFP4_HESSIAN_DAMPING` and is part of the
    algorithm's identity.

    The search does not weight with this vector directly: :func:`_search_weights` rescales it to a maximum of
    ``1.0`` first, which cannot move an argmin and is documented there. Reporting is deliberately not rescaled,
    because a per-layer rescaling would make two layers' objectives incomparable.

    Args:
        second_moments: rank-1 float tensor of non-negative finite second moments ``E[x_j^2]``.
        in_features: expected length, or ``None`` when unconstrained.
        device: device the vector must already live on, or ``None`` when unconstrained. The vector is never moved
            here: a caller that hands over a host vector for a device weight is making a mistake this cannot fix.

    Returns:
        The damped FP32 vector, on ``second_moments``' device.

    Raises:
        TypeError: if ``second_moments`` is not a tensor.
        ValueError: if it is not a rank-1 float vector of the expected length on the expected device, holds a
            non-finite or negative value, has a non-finite or non-positive mean, or damps to a vector whose
            maximum is not finite and positive.
    """
    if not isinstance(second_moments, torch.Tensor):
        raise TypeError(f"second_moments must be a torch.Tensor, got {type(second_moments).__name__}.")
    if second_moments.dim() != 1:
        raise ValueError(f"second_moments must be rank 1 (K,), got shape {tuple(second_moments.shape)}.")
    if not second_moments.is_floating_point():
        raise ValueError(f"second_moments must be a float tensor, got {second_moments.dtype}.")
    if in_features is not None and int(second_moments.shape[0]) != in_features:
        raise ValueError(
            f"second_moments must cover the weight's {in_features} input channels, got "
            f"{int(second_moments.shape[0])}."
        )
    if device is not None and second_moments.device != device:
        raise ValueError(f"second_moments is on device {second_moments.device} but the weight is on {device}.")

    moments = second_moments.detach().to(torch.float32)
    if not bool(torch.isfinite(moments).all()):
        raise ValueError("second_moments contains non-finite values; the local-Hessian objective is undefined.")
    if bool((moments < 0.0).any()):
        raise ValueError("second_moments contains negative values; a second moment E[x^2] can never be negative.")
    mean = moments.mean()
    if not bool(torch.isfinite(mean)) or float(mean) <= 0.0:
        raise ValueError(
            f"second_moments must have a finite positive mean, got {float(mean)}. An all-zero vector describes a "
            "layer whose inputs are identically zero, which no activation sample can support."
        )
    damped = moments + NVFP4_HESSIAN_DAMPING * mean
    # The maximum is the largest damped value, so checking it finite and positive is what rules out an overflowed
    # element; :func:`_search_weights` divides by exactly this value.
    maximum = damped.max()
    if not bool(torch.isfinite(maximum)) or float(maximum) <= 0.0:
        raise ValueError(f"The damped second moments have a non-positive or non-finite maximum ({float(maximum)}).")
    return damped


def nvfp4_blockwise_mse(
    weight: torch.Tensor,
    scale_fp8: torch.Tensor,
    per_tensor_scale: torch.Tensor,
) -> torch.Tensor:
    """Return the per-block reconstruction MSE of ``weight`` under the given E4M3 block scales.

    ``scale_fp8`` holds linear ``(M, K // 16)`` E4M3 block scales -- the ones :func:`select_nvfp4_block_scales`
    returns, or any other candidate set such as the ordinary amax-derived ones. The reconstruction is TorchAO's,
    so this is the objective the search minimizes, averaged over each block's 16 weights.
    """
    backend = _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    num_blocks = columns // NVFP4_BLOCK_SIZE
    if tuple(scale_fp8.shape) != (rows, num_blocks):
        raise ValueError(f"scale_fp8 must have shape ({rows}, {num_blocks}), got {tuple(scale_fp8.shape)}.")

    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    blocks = weight.detach().to(torch.float32).reshape(rows * num_blocks, NVFP4_BLOCK_SIZE)
    scales = scale_fp8.reshape(rows * num_blocks, 1).float()
    reconstruction = _reconstruct(blocks, scales, scale, backend)
    return ((reconstruction - blocks) ** 2).mean(dim=-1).reshape(rows, num_blocks)


def nvfp4_scale_candidates(device: Optional[torch.device] = None) -> torch.Tensor:
    """Return the 126 positive finite E4M3 scale encodings, as FP32, in ascending bit-pattern order.

    The candidates are the *exact* decoded values of the uint8 bit patterns ``0..127`` viewed as
    ``torch.float8_e4m3fn``, minus ``+0.0`` (pattern ``0x00``) and NaN (pattern ``0x7F``). No approximate float grid
    is used, so every candidate round-trips through E4M3 exactly and the search never selects an unrepresentable
    scale. The order is the bit order, which is also ascending in value, and it is what makes tie-breaking towards
    the smallest encoding deterministic.
    """
    _require_float8_e4m3("The NVFP4 block-scale search")
    patterns = torch.arange(128, dtype=torch.uint8, device=device)
    values = patterns.view(torch.float8_e4m3fn).to(torch.float32)
    candidates = values[torch.isfinite(values) & (values > 0.0)]
    if candidates.numel() != NVFP4_SCALE_CANDIDATE_COUNT:
        raise TorchAOUnavailableError(
            f"Expected {NVFP4_SCALE_CANDIDATE_COUNT} positive finite E4M3 encodings, got {candidates.numel()}; "
            f"torch {torch.__version__} does not decode torch.float8_e4m3fn as this packer requires."
        )
    return candidates


def nvfp4_gptq_hessian(group_activations: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the full ``(K, K)`` input Hessian of one module, balanced over equally weighted source groups.

    For each group ``g`` of ``N_g`` quantized activation rows ``Xq_g``, the reference forms

        ``scaled = sqrt(2 / N_g) * Xq_g.T`` and ``H_g = scaled @ scaled.T``

    in FP32, and the balanced Hessian is the arithmetic mean of the ``H_g``. Every group therefore counts exactly
    once however many rows it retained, which is what keeps a high-volume corpus from outvoting a small stratum.

    The groups are consumed in the order the caller passes them -- the builder passes them in sorted label order --
    and are summed in that order and divided by their count, all in FP32. That fixed order is part of the recorded
    identity: FP32 addition is not associative, so a different order would define a different matrix.

    ``group_activations`` must already have been through :func:`nvfp4_awq_clip_activation_qdq`, i.e. these are the
    rows the quantized runtime will actually multiply, not their BF16 originals.

    Args:
        group_activations: one non-empty FP32 ``(N_g, K)`` tensor per equally weighted source group, all of the same
            width and on the same device.

    Returns:
        A new contiguous FP32 ``(K, K)`` tensor on the groups' device.

    Raises:
        TypeError: if the sequence holds anything but tensors.
        ValueError: if the sequence is empty, a group is malformed, or the reduction is not finite.
    """
    if not isinstance(group_activations, (list, tuple)) or not group_activations:
        raise ValueError(
            "group_activations must be a non-empty sequence of quantized activation row tensors, one per equally "
            "weighted source group."
        )
    head = group_activations[0]
    if not isinstance(head, torch.Tensor):
        raise TypeError(f"group_activations[0] must be a torch.Tensor, got {type(head).__name__}.")
    if head.dim() != 2 or int(head.shape[1]) == 0:
        raise ValueError(f"group_activations[0] must be a non-empty rank-2 (N, K) tensor, got {tuple(head.shape)}.")
    groups = _validate_group_activations(group_activations, int(head.shape[1]), head.device)

    total: Optional[torch.Tensor] = None
    for group in groups:
        # ``[K, N_g]``: the contiguous copy only fixes the layout the matmul sees, and multiplying before or after
        # the transpose is the same element-wise product either way.
        scaled = (math.sqrt(NVFP4_GPTQ_HESSIAN_FACTOR / float(int(group.shape[0]))) * group.t()).contiguous()
        moment = scaled @ scaled.t()
        total = moment if total is None else total + moment
    hessian = total / float(len(groups))
    if not bool(torch.isfinite(hessian).all()):
        raise ValueError(
            "The group-balanced GPTQ Hessian is not finite; the quantized activation rows cannot support a payload "
            "selection."
        )
    return hessian.contiguous()


def nvfp4_gptq_damped_hessian(hessian: torch.Tensor, weight: torch.Tensor) -> NVFP4GPTQHessian:
    """Apply the reference's dead-column handling and its ``perc_damp`` damping to a GPTQ Hessian.

    A *dead* column is one whose original weight is exactly zero in every output row: it contributes nothing to the
    layer output, so the reference zeroes its Hessian row and column and sets its diagonal entry to ``1`` before
    damping, which keeps the matrix factorizable without letting that column steer any other column's payload.

    The damping is then the reference's ``perc_damp = 0.01`` applied to the *post*-dead-column diagonal:
    ``damping = 0.01 * mean(diag(H))``, added to every diagonal entry. It is computed from the FP32 diagonal mean,
    rounded once to a single FP32 scalar, and both added and reported as that scalar, so the value an artifact
    records is exactly the value the FP32 diagonal received. Rounding it once here rather than adding a wider
    python float is what keeps ``diagonal_min >= damping`` true of the recorded evidence even for an input channel
    whose own second moment is exactly zero -- which the activation quantize/dequantize can produce for a channel
    that always underflows its block -- and what makes the sum independent of how a backend widens a scalar.

    Neither ``hessian`` nor ``weight`` is mutated; the returned matrix is a new tensor.

    Args:
        hessian: the ``(K, K)`` FP32 Hessian :func:`nvfp4_gptq_hessian` produced, on ``weight``'s device.
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, ``K`` a multiple of 16.

    Returns:
        An :class:`NVFP4GPTQHessian` with the damped matrix and the evidence an artifact records.

    Raises:
        TypeError: if either argument is not a tensor.
        ValueError: for any violated shape, dtype, device or finiteness contract, or a diagonal whose mean is not
            finite and positive.
    """
    _validate_weight(weight, None, None, None)
    columns = int(weight.shape[1])
    matrix = _validate_gptq_hessian_matrix(hessian, columns, weight.device).clone()

    dead = (weight.detach().to(torch.float32) == 0.0).all(dim=0)
    dead_columns = int(dead.sum())
    if dead_columns:
        matrix[dead, :] = 0.0
        matrix[:, dead] = 0.0
    # A view onto the matrix's own diagonal, so both writes below land in ``matrix``.
    diagonal = torch.diagonal(matrix)
    if dead_columns:
        diagonal[dead] = 1.0
    mean = diagonal.mean()
    if not bool(torch.isfinite(mean)) or float(mean) <= 0.0:
        raise ValueError(
            f"The GPTQ Hessian diagonal has a non-positive or non-finite mean ({float(mean)}); no damping can make "
            "it factorizable."
        )
    added = torch.full((), NVFP4_GPTQ_PERC_DAMP * float(mean), dtype=torch.float32, device=matrix.device)
    damping = float(added)
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError(f"The GPTQ damping {damping!r} is not finite and positive.")
    diagonal += added
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("The damped GPTQ Hessian is not finite; refusing to invert it.")
    return NVFP4GPTQHessian(
        matrix=matrix,
        dead_columns=dead_columns,
        damping=damping,
        diagonal_min=float(diagonal.min()),
        diagonal_max=float(diagonal.max()),
        diagonal_mean=float(diagonal.mean()),
    )


def select_nvfp4_gptq_payload(
    weight: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scale: torch.Tensor,
    hessian: torch.Tensor,
) -> NVFP4GPTQSelection:
    """Select one weight's packed FP4 payload with NVIDIA ModelOpt 0.46.0's column-wise GPTQ update.

    Nothing here is searched and no scale is ever recomputed: ``per_tensor_scale`` and ``block_scale`` are the
    *ordinary* template's own fixed scales, and every column is written with the payload those scales quantize the
    current working value into. What GPTQ changes is the working value. For each 128-column update block, and for
    each column ``i`` of it in input order:

    * ``q_col`` is the fixed-scale quantize/dequantize of the working column ``w_col``;
    * the residual ``err = (w_col - q_col) / h_inv[i, i]`` is subtracted from the *later* columns of the same block
      through ``addr_(err, h_inv[i, i:], alpha=-1)``, so a column that has not been written yet absorbs part of the
      error the already-written columns made;
    * after the block, the accumulated residuals are propagated to every column past the block through
      ``addmm_(errors, h_inv[block, block_end:], alpha=-1)``.

    ``h_inv`` is ``torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)`` of the
    *damped* Hessian. The reference falls back to the identity when a factorization fails; this production path does
    not: a failed factorization, a zero or non-finite divisor, a non-finite working value, residual or decoded value
    are each a hard error, because a fallback would silently deploy a payload nobody selected.

    Everything is FP32, which is the precision the reference runs the update in and the precision the deployed
    reconstruction is defined in.

    Neither ``weight``, ``block_scale`` nor ``hessian`` is mutated.

    Args:
        weight: the original contiguous rank-2 ``(M, K)`` BF16 or FP32 weight, finite, ``K`` a multiple of 16.
        per_tensor_scale: the template's unchanged global per-tensor scale.
        block_scale: the ``(M, K // 16)`` FP32 linear E4M3 block scales decoded off the template, all finite and
            non-negative, on ``weight``'s device. A scale of exactly zero is the template's own encoding of an
            all-zero or underflowed block; those rows are written as FP4 code zero and read back as ``0.0``.
        hessian: the ``(K, K)`` FP32 damped Hessian of :func:`nvfp4_gptq_damped_hessian`, on ``weight``'s device.

    Returns:
        An :class:`NVFP4GPTQSelection` with the packed ``(M, K // 2)`` uint8 payload and the FP32 values it decodes
        to under exactly those fixed scales.

    Raises:
        TypeError: if an argument is not a tensor.
        ValueError: for any violated shape, dtype, device, finiteness or positivity contract, a Hessian this build
            cannot factorize, or a non-finite intermediate.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    scales = _validate_gptq_block_scales(block_scale, rows, columns // NVFP4_BLOCK_SIZE, weight.device)
    matrix = _validate_gptq_hessian_matrix(hessian, columns, weight.device)
    inverse = _gptq_inverse_cholesky(matrix)

    working = weight.detach().to(torch.float32).clone()
    codes = torch.zeros(rows, columns, dtype=torch.uint8, device=weight.device)
    values = torch.zeros(rows, columns, dtype=torch.float32, device=weight.device)
    for block_start in range(0, columns, NVFP4_GPTQ_UPDATE_BLOCK_SIZE):
        block_stop = min(block_start + NVFP4_GPTQ_UPDATE_BLOCK_SIZE, columns)
        block_inverse = inverse[block_start:block_stop, block_start:block_stop]
        # The reference's per-block clone: the within-block residuals accumulate here, while ``working`` keeps the
        # written payload's values and receives the cross-block update below.
        block_working = working.clone()
        errors = torch.zeros(rows, block_stop - block_start, dtype=torch.float32, device=weight.device)
        for offset in range(block_stop - block_start):
            column = block_start + offset
            current = block_working[:, column]
            code, quantized = _gptq_quantize_column(current, scale, scales[:, column // NVFP4_BLOCK_SIZE], backend)
            error = (current - quantized) / block_inverse[offset, offset]
            codes[:, column] = code
            values[:, column] = quantized
            working[:, column] = quantized
            block_working[:, column:block_stop].addr_(error, block_inverse[offset, offset:], alpha=-1)
            errors[:, offset] = error
        # Checked once per update block rather than once per column: nothing is written outside this function, so
        # failing here is still failing before any artifact or module could see the payload.
        _require_finite_gptq_block(errors, values[:, block_start:block_stop], block_start, block_stop)
        if block_stop < columns:
            working[:, block_stop:].addmm_(errors, inverse[block_start:block_stop, block_stop:], alpha=-1)
            if not bool(torch.isfinite(working[:, block_stop:]).all()):
                raise ValueError(
                    f"The GPTQ update of input columns [{block_start}, {block_stop}) propagated a non-finite "
                    "working value into the columns that follow it; refusing to select a payload from it."
                )

    return NVFP4GPTQSelection(qdata=backend.pack_uint4(codes.contiguous()), values=values)


def nvfp4_gptq_objective(reconstruction: torch.Tensor, weight: torch.Tensor, hessian: torch.Tensor) -> float:
    """Return the damped Hessian quadratic form of one reconstruction, per weight element.

    With ``delta = W - Q(W)`` the reconstruction error of the whole weight, this is

        ``sum_r delta[r] @ H @ delta[r] / (M * K)``,

    i.e. the layer-output error ``E||X (W - Q(W))||^2`` the balanced Hessian describes, divided by the weight's own
    element count so that two layers -- and the element-count-weighted aggregate over them -- are in the same units.
    It is reduced in FP64 so that a long accumulation cannot lose the comparison between two payloads of the same
    layer, and it is *offline* evidence: it says nothing about DER.

    Raises:
        TypeError: if an argument is not a tensor.
        ValueError: for a violated shape/device contract or a non-finite objective.
    """
    _validate_weight(weight, None, None, None)
    if not isinstance(reconstruction, torch.Tensor):
        raise TypeError(f"reconstruction must be a torch.Tensor, got {type(reconstruction).__name__}.")
    if tuple(reconstruction.shape) != tuple(weight.shape):
        raise ValueError(
            f"reconstruction has shape {tuple(reconstruction.shape)} but the weight is {tuple(weight.shape)}."
        )
    if reconstruction.device != weight.device:
        raise ValueError(f"reconstruction is on {reconstruction.device} but the weight is on {weight.device}.")
    matrix = _validate_gptq_hessian_matrix(hessian, int(weight.shape[1]), weight.device)
    delta = (weight.detach().to(torch.float32) - reconstruction.detach().to(torch.float32)).to(torch.float64)
    objective = float(((delta @ matrix.to(torch.float64)) * delta).sum() / float(delta.numel()))
    if not math.isfinite(objective):
        raise ValueError(f"The GPTQ Hessian quadratic objective is not finite ({objective!r}).")
    return objective


def nvfp4_template_identity(template: Any) -> NVFP4TemplateIdentity:
    """Return the buffers of an NVFP4 template that a GPTQ payload replacement must not disturb.

    This deliberately does *not* import TorchAO and does not check the wrapper's type: it is the runtime binding
    check, it runs on whatever wrapper the conversion left behind, and its caller has already established that the
    weight is an NVFP4 tensor. What it does check is everything a byte-exact binding needs -- that the payload and
    the swizzled scale buffer are contiguous single-byte-element tensors of the right sizes and that the global
    scale is a positive finite scalar -- so a wrapper this build cannot address byte by byte is refused rather than
    digested on a guess.

    The returned tensors are the template's *own* objects, not copies.

    Raises:
        TypeError: if ``template`` is not a tensor.
        ValueError: for any violated rank, shape, contiguity, storage or scale contract.
    """
    if not isinstance(template, torch.Tensor):
        raise TypeError(f"template must be a torch.Tensor, got {type(template).__name__}.")
    if template.dim() != 2:
        raise ValueError(f"template must be rank 2 (M, K), got shape {tuple(template.shape)}.")
    rows, columns = int(template.shape[0]), int(template.shape[1])
    if columns % NVFP4_BLOCK_SIZE != 0 or rows <= 0 or columns <= 0:
        raise ValueError(f"template must be a non-empty (M, K) tensor with K a multiple of {NVFP4_BLOCK_SIZE}.")

    qdata = _template_attr(template, ("qdata", "_data"), "packed qdata")
    scale = _template_attr(template, ("_scale_e4m3", "scale", "_scale"), "block scales")
    global_scale = _template_attr(template, ("_per_tensor_scale", "per_tensor_scale"), "per-tensor scale")
    for name, buffer in (("qdata", qdata), ("scale", scale)):
        if not isinstance(buffer, torch.Tensor):
            raise ValueError(f"template {name} must be a torch.Tensor, got {type(buffer).__name__}.")
        if not buffer.is_contiguous():
            raise ValueError(f"template {name} must be contiguous; its bytes cannot be bound to an artifact.")
        if buffer.element_size() != 1:
            raise ValueError(
                f"template {name} holds {buffer.element_size()}-byte elements; a GPTQ artifact binds this buffer "
                "byte by byte and refuses a storage it cannot address as bytes."
            )
    if qdata.numel() != rows * columns // 2:
        raise ValueError(
            f"template qdata holds {qdata.numel()} packed byte(s), expected {rows * columns // 2} for a "
            f"({rows}, {columns}) NVFP4 weight."
        )
    if scale.numel() < rows * (columns // NVFP4_BLOCK_SIZE):
        raise ValueError(
            f"template block scales hold {scale.numel()} byte(s), which is fewer than the "
            f"{rows * (columns // NVFP4_BLOCK_SIZE)} block(s) of a ({rows}, {columns}) NVFP4 weight."
        )
    _validate_per_tensor_scale(global_scale)
    return NVFP4TemplateIdentity(rows=rows, columns=columns, scale=scale, global_scale=global_scale, qdata=qdata)


def nvfp4_ordinary_template(weight: torch.Tensor, per_tensor_scale: torch.Tensor, template_arithmetic: str) -> Any:
    """Convert ``weight`` exactly the way the deployment's ordinary conversion does, in the named construction.

    This is the template a GPTQ build takes its fixed scales and its baseline payload from: the same
    ``NVFP4Tensor.to_nvfp4`` call, the same pinned keywords and the same Triton flag the runtime's own conversion
    would use for that backend. The two constructions are not interchangeable, which is why the caller has to name
    one and why the artifact records it.

    Raises:
        TypeError: if ``weight`` or ``per_tensor_scale`` is not a tensor.
        ValueError: for an unknown construction or a violated weight/scale contract.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 conversion cannot be imported or does not accept the
            pinned keywords.
    """
    backend = _resolve_torchao()
    _validate_per_tensor_scale(per_tensor_scale)
    _validate_weight(weight, None, None, None)
    if template_arithmetic not in NVFP4_GPTQ_TEMPLATE_ARITHMETICS:
        raise ValueError(
            f"template_arithmetic must be one of {list(NVFP4_GPTQ_TEMPLATE_ARITHMETICS)}, got "
            f"{template_arithmetic!r}; the GPTQ payload is written under the scales that construction produces and "
            "there is no default."
        )
    scale = _as_scalar_scale(per_tensor_scale, weight.device)
    return _ordinary_nvfp4_template(weight.detach(), scale, template_arithmetic, backend)


def nvfp4_template_block_scales(template: Any) -> torch.Tensor:
    """Return a template's ``(M, K // 16)`` *linear* E4M3 block scales, in FP32, off its own stored bytes.

    The blocked Blackwell layout is un-swizzled with pinned TorchAO's own ``from_blocked``, so these are exactly the
    scales the deployed kernel reads. They are returned on their own rather than multiplied into the global scale,
    because :func:`select_nvfp4_gptq_payload` applies the two in the very order TorchAO's reconstruction does --
    ``(1 / per_tensor_scale) / scale_fp8`` on the way in and ``per_tensor_scale * scale_fp8`` on the way out -- and
    a pre-multiplied product would round differently.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor``.
        ValueError: if the template cannot be un-swizzled into finite scales of the expected shape.
        TorchAOUnavailableError: if the pinned TorchAO API cannot be imported.
    """
    backend = _resolve_torchao()
    readback = _resolve_awq_clip_readback()
    metadata = _validate_template(template, backend)
    num_blocks = metadata.columns // NVFP4_BLOCK_SIZE
    scales = readback.from_blocked(metadata.scale, metadata.rows, num_blocks)
    if scales.numel() != metadata.rows * num_blocks:
        raise ValueError(
            f"The template's blocked scales un-swizzled to {scales.numel()} value(s), but a "
            f"({metadata.rows}, {metadata.columns}) NVFP4 weight has {metadata.rows * num_blocks} block(s); the "
            "installed torchao does not swizzle as this build requires."
        )
    linear = scales.to(torch.float32).reshape(metadata.rows, num_blocks)
    if not bool(torch.isfinite(linear).all()):
        raise ValueError("The template's block scales un-swizzled to non-finite values; refusing to write under them.")
    return linear.contiguous()


def nvfp4_template_values(template: Any) -> torch.Tensor:
    """Return the FP32 values a template's own stored bytes decode to, with the FP32 scale arithmetic.

    This is :func:`nvfp4_awq_clip_template_reconstruction`'s decode step applied to an *already converted* template
    rather than to a weight, so a GPTQ build can score the ordinary payload it is replacing on exactly the same
    footing as the payload it selected.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor``.
        ValueError: if the template cannot be decoded into finite values of its own logical shape.
        TorchAOUnavailableError: if the pinned TorchAO API cannot be imported.
    """
    backend = _resolve_torchao()
    readback = _resolve_awq_clip_readback()
    metadata = _validate_template(template, backend)
    values = _decode_nvfp4_template_values(metadata, backend, readback)
    if tuple(values.shape) != (metadata.rows, metadata.columns):
        raise ValueError(
            f"The template's payload decoded to {tuple(values.shape)} but it covers "
            f"({metadata.rows}, {metadata.columns}); the installed torchao does not pack as this build requires."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("The template decoded to non-finite values; refusing to score a GPTQ objective against it.")
    return values.contiguous()


def repack_nvfp4_weight_gptq(template: Any, qdata: torch.Tensor) -> Any:
    """Return a new ``NVFP4Tensor`` that is ``template`` with *only* its packed payload replaced.

    The template's swizzled E4M3 scale buffer is handed to the new wrapper unchanged -- the same values in the same
    layout, padding included -- as is its global per-tensor scale object and every other wrapper attribute: block
    size, original dtype, activation per-tensor scale, swizzle flag, Triton flag and activation quantization
    kwargs. Nothing about the wire format, the scale layout or the kernel path changes, so the result is a drop-in
    replacement that differs from ``template`` in its FP4 payload bytes and in nothing else.

    Neither ``template`` (nor any tensor it owns) nor ``qdata`` is mutated; the result is always a new tensor.

    Args:
        template: an already-converted TorchAO ``NVFP4Tensor``, with block size 16 and swizzled scales.
        qdata: a contiguous ``torch.uint8`` tensor of exactly ``M * K / 2`` packed bytes on the template's device.

    Returns:
        A new instance of ``type(template)`` holding ``qdata`` and the template's own scales.

    Raises:
        TypeError: if ``template`` is not an ``NVFP4Tensor`` or ``qdata`` is not a tensor.
        ValueError: for any violated shape, dtype, device or metadata contract.
        TorchAOUnavailableError: if the pinned TorchAO NVFP4 API cannot be imported.
    """
    backend = _resolve_torchao()
    metadata = _validate_template(template, backend)
    if not isinstance(qdata, torch.Tensor):
        raise TypeError(f"qdata must be a torch.Tensor, got {type(qdata).__name__}.")
    if qdata.dtype != torch.uint8:
        raise ValueError(f"qdata must be {torch.uint8} packed bytes, got {qdata.dtype}.")
    if not qdata.is_contiguous():
        raise ValueError("qdata must be contiguous; call .contiguous() before replacing a template's payload.")
    if qdata.numel() != metadata.rows * metadata.columns // 2:
        raise ValueError(
            f"qdata holds {qdata.numel()} packed byte(s), but a ({metadata.rows}, {metadata.columns}) NVFP4 weight "
            f"needs {metadata.rows * metadata.columns // 2}."
        )
    if qdata.device != metadata.device:
        raise ValueError(f"qdata is on device {qdata.device} but the template is on {metadata.device}.")
    # ``blocked_scale`` is the template's own buffer, so the new wrapper carries exactly those bytes and this
    # function never goes near ``to_blocked``.
    return _build_nvfp4_tensor(template, metadata, qdata, None, backend, blocked_scale=metadata.scale)


# ------------------------------------------------------------------------------------------------------------------
# Search
# ------------------------------------------------------------------------------------------------------------------


def _search_weights(damped: torch.Tensor) -> torch.Tensor:
    """Rescale the damped second moments to a maximum of ``1.0``: the weights the *search* reduces with.

    Multiplying the complete weight vector by a positive constant scales every candidate's objective equally and
    therefore cannot move the argmin, so this is numerical hygiene only and never a change of algorithm. Dividing
    by the *maximum* rather than by the mean or the sum is deliberate: a maximum is a selection, not an
    accumulation, so a constant input vector rescales to exactly ``1.0`` and the weighted search then reproduces
    the unweighted MSE search bit for bit.

    ``damped`` must come from :func:`damped_second_moments`, which is where its maximum is checked finite and
    positive. Reported objectives use the damped vector itself, not this rescaling, because the divisor differs
    from layer to layer.
    """
    return damped / damped.max()


def _select_block_scales(
    weight_f32: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    candidate_chunk_size: int,
    block_chunk_size: int,
    moments: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exhaustively pick each block's E4M3 scale by reconstruction error, in bounded-memory chunks.

    ``weight_f32`` is the FP32 view of the ``(M, K)`` weight; the returned tensor is ``(M, K // 16)`` E4M3. With
    ``moments`` -- the ``(K,)`` vector :func:`_search_weights` produced from the damped second moments -- the
    per-column squared errors are weighted, which is the diagonal local-Hessian objective up to the positive
    constant that rescaling divided out; without it the reduction is the plain unweighted sum, and the two differ
    in nothing else.

    Both loops are plain chunking of one flat argmin, so the result never depends on the chunk sizes: within a
    candidate chunk the smallest *index* attaining the chunk minimum is taken, and a later chunk only wins on a
    strictly smaller error. Since candidates are visited in ascending bit order, the first minimum always wins.
    """
    backend = _resolve_torchao()
    rows, columns = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    blocks = weight_f32.reshape(-1, NVFP4_BLOCK_SIZE)
    total_blocks = blocks.shape[0]
    num_blocks = columns // NVFP4_BLOCK_SIZE
    # ``blocks`` runs row-major, so block ``i`` covers columns of the block ``i % num_blocks`` of its output row;
    # the per-chunk gather below is what keeps the weighting bounded by the chunk instead of the whole weight.
    moment_blocks = None if moments is None else moments.reshape(num_blocks, NVFP4_BLOCK_SIZE)

    candidates = nvfp4_scale_candidates(weight_f32.device)
    selected = torch.empty(total_blocks, dtype=torch.long, device=weight_f32.device)

    for start in range(0, total_blocks, block_chunk_size):
        chunk = blocks[start : start + block_chunk_size]
        best_error = torch.full((chunk.shape[0],), float("inf"), dtype=torch.float32, device=chunk.device)
        best_index = torch.zeros(chunk.shape[0], dtype=torch.long, device=chunk.device)
        chunk_moments = None
        if moment_blocks is not None:
            positions = torch.arange(start, start + chunk.shape[0], device=chunk.device) % num_blocks
            chunk_moments = moment_blocks[positions][:, None, :]

        for offset in range(0, candidates.numel(), candidate_chunk_size):
            window = candidates[offset : offset + candidate_chunk_size]
            # [chunk_blocks, window_candidates, 16] is the whole transient working set of the search.
            reconstruction = _reconstruct(chunk[:, None, :], window[None, :, None], per_tensor_scale, backend)
            squared = (reconstruction - chunk[:, None, :]) ** 2
            error = (squared if chunk_moments is None else squared * chunk_moments).sum(dim=-1)

            window_error, _ = error.min(dim=1)
            # ``argmin`` does not promise the first minimum, so take the smallest tying index explicitly.
            positions = torch.arange(window.numel(), device=chunk.device).expand_as(error)
            window_index = torch.where(error == window_error[:, None], positions, positions.new_full((), 2**31))
            window_index = window_index.min(dim=1).values + offset

            improved = window_error < best_error
            best_index = torch.where(improved, window_index, best_index)
            best_error = torch.where(improved, window_error, best_error)

        selected[start : start + chunk.shape[0]] = best_index

    return candidates[selected].reshape(rows, columns // NVFP4_BLOCK_SIZE).to(torch.float8_e4m3fn)


def _select_four_over_six_block_scales(
    weight_f32: torch.Tensor, per_tensor_scale: torch.Tensor, block_chunk_size: int
) -> FourOverSixSelection:
    """Pick each block's representation from exactly the two Four-Over-Six candidates, in bounded-memory chunks.

    ``weight_f32`` is the FP32 view of the ``(M, K)`` weight and ``per_tensor_scale`` is the renormalized
    Four-Over-Six global scale. The two candidates are scored by the plain FP32 sum of squared reconstruction
    errors of the block, after the same E4M3 rounding and E2M1 conversion the runtime will execute, and ``M=4`` is
    only taken when it is *strictly* better, which is what resolves an exact tie towards ``M=6``.

    The chunking bounds the transient ``[block_chunk_size, 2, 16]`` working set and cannot change the selection: no
    reduction crosses a chunk boundary.
    """
    backend = _resolve_torchao()
    rows, columns = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    blocks = weight_f32.reshape(-1, NVFP4_BLOCK_SIZE)
    total_blocks = blocks.shape[0]
    magnitudes = torch.tensor(
        [float(magnitude) for magnitude in NVFP4_FOUR_OVER_SIX_MAGNITUDES],
        dtype=torch.float32,
        device=weight_f32.device,
    )
    selected_scale = torch.empty(total_blocks, dtype=torch.float32, device=weight_f32.device)
    selected_index = torch.empty(total_blocks, dtype=torch.long, device=weight_f32.device)

    for start in range(0, total_blocks, block_chunk_size):
        chunk = blocks[start : start + block_chunk_size]
        candidates = _four_over_six_candidate_scales(chunk, magnitudes, per_tensor_scale)
        # [chunk_blocks, 2, 16] is the whole transient working set of the selection.
        reconstruction = _reconstruct(chunk[:, None, :], candidates[:, :, None], per_tensor_scale, backend)
        error = ((reconstruction - chunk[:, None, :]) ** 2).sum(dim=-1)
        index = (error[:, 1] < error[:, 0]).long()
        selected_index[start : start + chunk.shape[0]] = index
        selected_scale[start : start + chunk.shape[0]] = candidates.gather(1, index[:, None]).reshape(-1)

    magnitude_index = selected_index.reshape(rows, columns // NVFP4_BLOCK_SIZE)
    m4_block_count = int(magnitude_index.sum())
    return FourOverSixSelection(
        # The values are already exact E4M3 encodings, so this cast rounds nothing.
        scale_fp8=selected_scale.reshape(rows, columns // NVFP4_BLOCK_SIZE).to(torch.float8_e4m3fn),
        magnitude_index=magnitude_index,
        m6_block_count=magnitude_index.numel() - m4_block_count,
        m4_block_count=m4_block_count,
    )


def _four_over_six_candidate_scales(
    blocks: torch.Tensor, magnitudes: torch.Tensor, per_tensor_scale: torch.Tensor
) -> torch.Tensor:
    """Return the ``[blocks, 2]`` E4M3-rounded candidate scales, in the reference's exact order of operations.

    The reference static export computes the unnormalized block scale ``amax / magnitude``, substitutes ``1.0`` for
    a zero one, divides by the global scale, clamps into ``[2 ** -9, 448]`` and only then rounds to E4M3. Each of
    those steps is reproduced here, in that order, because a different order would round a different value.
    """
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    unnormalized = amax / magnitudes[None, :]
    unnormalized = torch.where(unnormalized == 0.0, torch.ones_like(unnormalized), unnormalized)
    scales = torch.clamp(unnormalized / per_tensor_scale, NVFP4_FOUR_OVER_SIX_SCALE_MIN, NVFP4_FOUR_OVER_SIX_SCALE_MAX)
    return scales.to(torch.float8_e4m3fn).to(torch.float32)


def _awq_clip_candidate_scales(
    blocks: torch.Tensor, ratios: torch.Tensor, per_tensor_scale: torch.Tensor
) -> torch.Tensor:
    """Return the E4M3-rounded AWQ-clip scales of ``blocks`` for every ratio, in the exact order of operations.

    ``blocks`` is any FP32 tensor whose last dimension is the 16-weight block and ``ratios`` broadcasts against the
    resulting ``[..., 1]`` amax, so the same code serves the eleven-candidate search and the single-code repack.
    The clipped maximum is mapped onto FP4's largest magnitude, normalized, clamped into ``[2 ** -9, 448]`` and only
    then rounded to E4M3; a different order would round a different value.

    An all-zero block has amax ``0`` under every ratio, so all eleven candidates clamp to the same smallest scale
    and reconstruct the block identically -- which is exactly why such a block deterministically keeps code ``0``.
    """
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    unnormalized = (amax * ratios) / NVFP4_MAX
    scales = torch.clamp(unnormalized / per_tensor_scale, NVFP4_AWQ_CLIP_SCALE_MIN, NVFP4_AWQ_CLIP_SCALE_MAX)
    return scales.to(torch.float8_e4m3fn).to(torch.float32)


def _awq_clip_block_scales(
    weight_f32: torch.Tensor, codes: torch.Tensor, per_tensor_scale: torch.Tensor, block_chunk_size: int
) -> torch.Tensor:
    """Rebuild the ``(M, K // 16)`` E4M3 scale of exactly the ratio each block's code names, in bounded chunks.

    Every chunk goes through :func:`_awq_clip_candidate_scales` with that chunk's own per-block ratio, so a stored
    code reproduces bit for bit the candidate the offline search compared, and the chunking cannot change a scale.
    """
    rows, columns = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    blocks = weight_f32.reshape(-1, NVFP4_BLOCK_SIZE)
    ratios = torch.tensor(NVFP4_AWQ_CLIP_RATIOS, dtype=torch.float32, device=weight_f32.device)
    selected = ratios[codes.reshape(-1).to(torch.long)]
    scales = torch.empty(blocks.shape[0], dtype=torch.float32, device=weight_f32.device)

    for start in range(0, blocks.shape[0], block_chunk_size):
        chunk = blocks[start : start + block_chunk_size]
        # [chunk_blocks, 1] is the whole transient working set of the reconstruction.
        chunk_ratios = selected[start : start + chunk.shape[0], None]
        candidates = _awq_clip_candidate_scales(chunk, chunk_ratios, per_tensor_scale)
        scales[start : start + chunk.shape[0]] = candidates.reshape(-1)

    # The values are already exact E4M3 encodings, so this cast rounds nothing.
    return scales.reshape(rows, columns // NVFP4_BLOCK_SIZE).to(torch.float8_e4m3fn)


def _awq_clip_merged_qdata(
    produced: torch.Tensor, template_qdata: torch.Tensor, clipped: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    """Keep the newly packed payload of the clipped blocks and the template's own bytes everywhere else.

    16 FP4 values pack into 8 whole bytes, so block ``b`` of output row ``r`` owns exactly the contiguous byte range
    ``[8 * b, 8 * b + 8)`` of that row and no packed nibble pair ever straddles a block boundary. That is what makes
    a per-block choice between the two payloads a plain byte selection rather than a nibble rewrite.
    """
    bytes_per_block = NVFP4_BLOCK_SIZE // 2
    num_blocks = columns // NVFP4_BLOCK_SIZE
    for name, buffer in (("repacked qdata", produced), ("template qdata", template_qdata)):
        if buffer.element_size() != 1:
            raise ValueError(
                f"{name} holds {buffer.element_size()}-byte elements; the AWQ-clip repack preserves the template's "
                "unclipped blocks byte by byte and refuses a payload it cannot address as bytes."
            )
    candidate = produced.reshape(-1).contiguous().view(torch.uint8)
    reference = template_qdata.reshape(-1).contiguous().view(torch.uint8)
    if candidate.numel() != rows * columns // 2 or reference.numel() != candidate.numel():
        raise ValueError(
            f"The repacked payload holds {candidate.numel()} byte(s) and the template's {reference.numel()}, but a "
            f"({rows}, {columns}) NVFP4 weight needs {rows * columns // 2}; the template's storage layout cannot be "
            "preserved safely."
        )
    take = clipped[:, :, None].expand(rows, num_blocks, bytes_per_block).reshape(-1)
    return torch.where(take, candidate, reference).reshape(rows, columns // 2)


def _awq_clip_merged_scale(
    scale_fp8: torch.Tensor, template_scale: torch.Tensor, clipped: torch.Tensor, backend: "_TorchAOBackend"
) -> torch.Tensor:
    """Swizzle the clipped blocks' new scales into the template's scale buffer, keeping every other byte.

    ``to_blocked`` places a ``(M, K // 16)`` grid into the Blackwell 128x4 layout and pads it; it is a pure
    placement, so swizzling a marker grid with the very same call marks exactly the bytes of the clipped blocks.
    The marker is ``True`` for *clipped* rather than for kept blocks on purpose: padding is unmarked, so it keeps
    the template's own padding bytes and an all-unclipped code matrix reproduces the template's scale buffer byte
    for byte without depending on what value the swizzle pads with.
    """
    blocked = backend.to_blocked(scale_fp8).flatten()
    marker = clipped.to(torch.uint8).contiguous().view(torch.float8_e4m3fn)
    take = backend.to_blocked(marker).flatten().contiguous().view(torch.uint8) != 0
    candidate = blocked.reshape(-1).contiguous().view(torch.uint8)
    reference = template_scale.reshape(-1).contiguous().view(torch.uint8)
    if candidate.numel() != reference.numel() or take.numel() != candidate.numel():
        raise ValueError(
            f"The swizzled block scales hold {candidate.numel()} byte(s) but the template's hold "
            f"{reference.numel()}; the template's NVFP4 scale layout cannot be preserved safely."
        )
    return torch.where(take, candidate, reference).view(torch.float8_e4m3fn)


def _awq_clip_losses(delta: torch.Tensor, group_blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the ``[R, B, C]`` FP64 AWQ-clip loss of one output-row/input-block chunk over the balanced groups.

    ``delta`` is the ``[R, B, C, 16]`` FP32 weight reconstruction error and ``group_blocks`` holds one ``[B, N, 16]``
    FP32 activation operand per equally weighted group.

    The dot product itself stays in FP32 -- that is the arithmetic the deployed kernel performs, and scoring it in a
    wider precision would score a layer output nobody executes. Its *result* is widened to FP64 before it is squared
    and before a single row or group is accumulated, so the long reductions cannot lose a winner to rounding.

    The dot's reduced dimension is always the fixed 16-value block and the per-row squares are accumulated over
    fixed :data:`AWQ_CLIP_ACTIVATION_ROW_BATCH` batches of the group's own rows, so every candidate goes through the
    same sequence of operations however the caller chunks output rows or input blocks. Groups are added in the
    caller's canonical order and divided by their count, which is what weights every group equally regardless of how
    many rows it retained.
    """
    rows, blocks, candidates = int(delta.shape[0]), int(delta.shape[1]), int(delta.shape[2])
    # ``[B, R * C, 1, 16]``, broadcast below against one ``[B, 1, n, 16]`` batch of that group's activation rows.
    operand = delta.permute(1, 0, 2, 3).reshape(blocks, rows * candidates, NVFP4_BLOCK_SIZE).unsqueeze(-2)
    total: Optional[torch.Tensor] = None
    for group in group_blocks:
        count = int(group.shape[1])
        accumulated: Optional[torch.Tensor] = None
        for start in range(0, count, AWQ_CLIP_ACTIVATION_ROW_BATCH):
            batch = group[:, start : start + AWQ_CLIP_ACTIVATION_ROW_BATCH, :].unsqueeze(-3)
            # ``[B, R * C, n]`` FP32 block-local dot products, widened only once they are complete.
            dots = (operand * batch).sum(dim=-1).to(torch.float64)
            partial = (dots * dots).sum(dim=-1)
            accumulated = partial if accumulated is None else accumulated + partial
        per_group = accumulated / float(count)
        total = per_group if total is None else total + per_group
    return (total / float(len(group_blocks))).reshape(blocks, rows, candidates).permute(1, 0, 2)


def _awq_clip_winners(loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the winning ``[R, B]`` loss and candidate index, comparing candidates in their fixed order.

    A candidate replaces the incumbent only on a *strictly* smaller loss, which is what makes an exact tie keep the
    earliest ratio and makes the whole selection independent of how the candidates happen to be laid out.
    """
    best = loss[..., 0].clone()
    best_index = torch.zeros(loss.shape[:-1], dtype=torch.long, device=loss.device)
    for index in range(1, NVFP4_AWQ_CLIP_RATIO_COUNT):
        improved = loss[..., index] < best
        best = torch.where(improved, loss[..., index], best)
        best_index = torch.where(improved, torch.full_like(best_index, index), best_index)
    return best, best_index


def _gptq_quantize_column(
    column: torch.Tensor, per_tensor_scale: torch.Tensor, block_scale: torch.Tensor, backend: "_TorchAOBackend"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize one input column under its block's *fixed* template scale, returning the codes and their values.

    For every output row whose block scale is nonzero this is exactly :func:`_reconstruct`'s arithmetic -- the
    ``(1 / per_tensor_scale) / scale_fp8`` reciprocal, the ``[-6, 6]`` clamp, TorchAO's ``f32_to_f4_unpacked`` and
    the ``per_tensor_scale * scale_fp8`` dequantization -- split so that the payload codes can be packed later and
    the decoded values can drive the GPTQ residual now.

    A row whose block scale is exactly zero is the fixed template's own representation of an all-zero or underflowed
    block: it has no reciprocal, and every payload it could carry decodes to zero. Those rows are therefore written
    as FP4 code zero and read back as ``0.0`` without any division, rather than being given a substituted scale that
    the deployed template does not carry. The substitution below only feeds the division for rows that are discarded
    immediately afterwards, so the bits of every nonzero row are TorchAO's own.
    """
    usable = block_scale > 0.0
    divisor = torch.where(usable, block_scale, torch.ones_like(block_scale))
    reciprocal = (1.0 / per_tensor_scale) / divisor
    scaled = torch.clamp(column * reciprocal, -NVFP4_MAX, NVFP4_MAX)
    codes = backend.f32_to_f4_unpacked(scaled)
    values = backend.f4_unpacked_to_f32(codes) * (per_tensor_scale * block_scale)
    return (
        torch.where(usable, codes, torch.zeros((), dtype=codes.dtype, device=codes.device)),
        torch.where(usable, values, torch.zeros((), dtype=values.dtype, device=values.device)),
    )


def _gptq_inverse_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Return the reference's upper Cholesky factor of the damped Hessian's inverse, or fail closed.

    The reference computes ``cholesky(cholesky_inverse(cholesky(H)), upper=True)`` and substitutes the identity when
    a factorization raises. This production path has no such fallback: a Hessian this build cannot factorize would
    make the update propagate residuals nobody selected, so it is a hard error before an artifact is written or a
    module is mutated. The diagonal is checked once here rather than once per column, which is the same condition:
    every divisor the update uses is one of these entries.
    """
    try:
        factor = torch.linalg.cholesky(matrix)
        inverse = torch.cholesky_inverse(factor)
        upper = torch.linalg.cholesky(inverse, upper=True)
    except RuntimeError as error:
        raise ValueError(
            f"The damped GPTQ Hessian could not be factorized ({error}); this build has no identity fallback, "
            "because propagating residuals through a substituted inverse would deploy a payload nobody selected."
        ) from error
    if not bool(torch.isfinite(upper).all()):
        raise ValueError("The inverse Cholesky factor of the damped GPTQ Hessian is not finite.")
    diagonal = torch.diagonal(upper)
    if bool((diagonal == 0.0).any()):
        raise ValueError(
            "The inverse Cholesky factor of the damped GPTQ Hessian has a zero on its diagonal, so a GPTQ residual "
            "would divide by zero."
        )
    return upper


def _require_finite_gptq_block(errors: torch.Tensor, values: torch.Tensor, block_start: int, block_stop: int) -> None:
    """Fail closed on a non-finite residual or decoded value anywhere in one 128-column GPTQ update block."""
    if not bool(torch.isfinite(errors).all()):
        raise ValueError(
            f"The GPTQ update of input columns [{block_start}, {block_stop}) produced a non-finite residual; "
            "refusing to select a payload from it."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError(
            f"The GPTQ payload of input columns [{block_start}, {block_stop}) decodes to non-finite values; "
            "refusing to select it."
        )


def _reconstruct(
    blocks: torch.Tensor, scales: torch.Tensor, per_tensor_scale: torch.Tensor, backend: "_TorchAOBackend"
) -> torch.Tensor:
    """Dequantize ``blocks`` under ``scales`` with pinned TorchAO's exact two-level NVFP4 arithmetic.

    ``blocks`` is any FP32 tensor whose last dimension is the 16-weight block, and ``scales`` is an FP32 tensor of
    E4M3-representable block scales broadcastable against it. The reciprocal, the ``[-6, 6]`` clamp, the E2M1
    conversion and the dequantization multiplier are TorchAO's own, so the reconstruction the search minimizes is
    exactly the one the runtime will compute.
    """
    reciprocal = (1.0 / per_tensor_scale) / scales
    data_scaled = torch.clamp(blocks * reciprocal, -NVFP4_MAX, NVFP4_MAX)
    codes = backend.f32_to_f4_unpacked(data_scaled)
    return backend.f4_unpacked_to_f32(codes) * (per_tensor_scale * scales)


def _pack_qdata(
    weight_f32: torch.Tensor, scale_fp8: torch.Tensor, per_tensor_scale: torch.Tensor, backend: "_TorchAOBackend"
) -> torch.Tensor:
    """Quantize the whole weight once under the selected scales and pack it to ``(M, K // 2)`` uint4 pairs."""
    rows, columns = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    reciprocal = (1.0 / per_tensor_scale) / scale_fp8.float()
    data_scaled = weight_f32.reshape(rows, columns // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE) * reciprocal[..., None]
    data_scaled = torch.clamp(data_scaled, -NVFP4_MAX, NVFP4_MAX).reshape(rows, columns)
    codes = backend.f32_to_f4_unpacked(data_scaled.contiguous())
    return backend.pack_uint4(codes)


def _build_nvfp4_tensor(
    template: Any,
    metadata: "_TemplateMetadata",
    qdata: torch.Tensor,
    scale_fp8: Optional[torch.Tensor],
    backend: "_TorchAOBackend",
    blocked_scale: Optional[torch.Tensor] = None,
) -> Any:
    """Assemble the new tensor, preserving the template's storage layout and every wrapper attribute.

    The linear E4M3 scales go through TorchAO's ``to_blocked`` swizzle and are then viewed as the template's scale
    shape, which is what keeps the Blackwell 128x4 layout the runtime kernels expect. Any layout the swizzle cannot
    reproduce byte for byte is refused rather than silently reshaped.

    ``blocked_scale`` is for the one caller that has already assembled the swizzled buffer itself -- the AWQ-clip
    repack, which merges newly swizzled bytes with the template's own -- and replaces ``scale_fp8`` when given.
    """
    qdata = _match_storage(qdata, metadata.qdata, "qdata")
    blocked = backend.to_blocked(scale_fp8).flatten() if blocked_scale is None else blocked_scale.flatten()
    scale = _match_storage(blocked, metadata.scale, "scale")

    return type(template)(
        qdata,
        scale,
        metadata.block_size,
        metadata.orig_dtype,
        per_tensor_scale=metadata.per_tensor_scale_attr,
        act_per_tensor_scale=metadata.act_per_tensor_scale,
        is_swizzled_scales=metadata.is_swizzled_scales,
        use_triton_kernel=metadata.use_triton_kernel,
        act_quant_kwargs=metadata.act_quant_kwargs,
    )


def _match_storage(produced: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Reshape/reinterpret ``produced`` into ``reference``'s exact shape, dtype and device, or fail closed."""
    if produced.numel() != reference.numel():
        raise ValueError(
            f"Repacked {name} has {produced.numel()} elements but the template's has {reference.numel()}; "
            "the template's NVFP4 storage layout cannot be preserved safely."
        )
    if produced.dtype != reference.dtype:
        if produced.element_size() != reference.element_size():
            raise ValueError(
                f"Repacked {name} is {produced.dtype} but the template's is {reference.dtype}, and the two are not "
                "byte-compatible; the template's NVFP4 storage layout cannot be preserved safely."
            )
        produced = produced.view(reference.dtype)
    if produced.device != reference.device:
        raise ValueError(f"Repacked {name} is on {produced.device} but the template's is on {reference.device}.")
    return produced.contiguous().reshape(reference.shape)


# ------------------------------------------------------------------------------------------------------------------
# Validation. Every check below is fail-closed: this packer never falls back to the template's own scales.
# ------------------------------------------------------------------------------------------------------------------


class _TemplateMetadata(NamedTuple):
    """Everything read off the template: its storage buffers, its logical shape and its wrapper attributes."""

    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    per_tensor_scale_attr: torch.Tensor
    per_tensor_scale: torch.Tensor
    act_per_tensor_scale: Any
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: Any
    rows: int
    columns: int
    device: torch.device


def _validate_template(template: Any, backend: "_TorchAOBackend") -> _TemplateMetadata:
    """Check that ``template`` is a compatible NVFP4 wrapper and return everything the repack must preserve."""
    if not isinstance(template, backend.NVFP4Tensor):
        raise TypeError(
            f"template must be a torchao {backend.NVFP4Tensor.__name__}, got {type(template).__name__}. This packer "
            "repacks an already-converted NVFP4 weight and does not convert one itself."
        )
    if template.dim() != 2:
        raise ValueError(f"template must be rank 2 (M, K), got shape {tuple(template.shape)}.")

    qdata = _template_attr(template, ("qdata", "_data"), "packed qdata")
    scale = _template_attr(template, ("_scale_e4m3", "scale", "_scale"), "block scales")
    block_size = int(_template_attr(template, ("_block_size", "block_size"), "block size"))
    orig_dtype = _template_attr(template, ("_orig_dtype", "orig_dtype"), "original dtype")
    per_tensor_scale = _template_attr(template, ("_per_tensor_scale", "per_tensor_scale"), "per-tensor scale")
    act_per_tensor_scale = _template_attr(
        template, ("_act_per_tensor_scale", "act_per_tensor_scale"), "activation per-tensor scale"
    )
    is_swizzled_scales = _template_attr(template, ("_is_swizzled_scales", "is_swizzled_scales"), "swizzle flag")
    use_triton_kernel = _template_attr(template, ("use_triton_kernel", "_use_triton_kernel"), "Triton flag")
    act_quant_kwargs = _template_attr(template, ("act_quant_kwargs", "_act_quant_kwargs"), "activation quant kwargs")

    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError(f"template must use block size {NVFP4_BLOCK_SIZE}, got {block_size}.")
    if not bool(is_swizzled_scales):
        raise ValueError(
            "template must carry swizzled (Blackwell 128x4) scales; this packer preserves the runtime scale layout "
            "and refuses to convert a linear-scale tensor into a swizzled one."
        )
    if not isinstance(orig_dtype, torch.dtype) or orig_dtype not in SUPPORTED_WEIGHT_DTYPES:
        raise ValueError(f"template.orig_dtype must be one of {SUPPORTED_WEIGHT_DTYPES}, got {orig_dtype}.")
    for name, buffer in (("qdata", qdata), ("scale", scale)):
        if not isinstance(buffer, torch.Tensor):
            raise ValueError(f"template {name} must be a torch.Tensor, got {type(buffer).__name__}.")
        if not buffer.is_contiguous():
            raise ValueError(f"template {name} must be contiguous; its storage layout cannot be preserved safely.")

    rows, columns = int(template.shape[0]), int(template.shape[1])
    if columns % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"template K must be a multiple of {NVFP4_BLOCK_SIZE}, got K={columns}.")
    if qdata.numel() != rows * columns // 2:
        raise ValueError(
            f"template qdata holds {qdata.numel()} packed elements, expected {rows * columns // 2} for a "
            f"({rows}, {columns}) NVFP4 weight; its storage layout cannot be preserved safely."
        )

    _validate_per_tensor_scale(per_tensor_scale)
    if per_tensor_scale.device != template.device:
        raise ValueError(
            f"template per-tensor scale is on {per_tensor_scale.device} but the template is on {template.device}."
        )

    return _TemplateMetadata(
        qdata=qdata,
        scale=scale,
        block_size=block_size,
        orig_dtype=orig_dtype,
        per_tensor_scale_attr=per_tensor_scale,
        per_tensor_scale=_as_scalar_scale(per_tensor_scale, template.device),
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
        rows=rows,
        columns=columns,
        device=template.device,
    )


def _validate_weight(
    weight: torch.Tensor, rows: Optional[int], columns: Optional[int], device: Optional[torch.device]
) -> None:
    """Check the original high-precision weight; ``rows``/``columns``/``device`` are ``None`` when unconstrained."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}.")
    if weight.dim() != 2:
        raise ValueError(f"weight must be rank 2 (M, K), got shape {tuple(weight.shape)}.")
    if weight.dtype not in SUPPORTED_WEIGHT_DTYPES:
        expected = " or ".join(str(dtype) for dtype in SUPPORTED_WEIGHT_DTYPES)
        raise ValueError(f"weight must be {expected}, got {weight.dtype}.")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous; call .contiguous() before repacking.")
    if int(weight.shape[1]) % NVFP4_BLOCK_SIZE != 0 or weight.numel() == 0:
        raise ValueError(
            f"weight must be non-empty with K a positive multiple of {NVFP4_BLOCK_SIZE}, got shape "
            f"{tuple(weight.shape)}."
        )
    if rows is not None and (int(weight.shape[0]), int(weight.shape[1])) != (rows, columns):
        raise ValueError(f"weight has shape {tuple(weight.shape)} but the template covers ({rows}, {columns}).")
    if device is not None and weight.device != device:
        raise ValueError(f"weight is on device {weight.device} but the template is on {device}.")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains non-finite values; NVFP4 repacking has no defined result for them.")


def _validate_ratio_codes(
    ratio_codes: Any, rows: int, num_blocks: int, device: Optional[torch.device]
) -> torch.Tensor:
    """Check an AWQ-clip code matrix and return it detached and contiguous; nothing here is coerced or clipped."""
    if not isinstance(ratio_codes, torch.Tensor):
        raise TypeError(f"ratio_codes must be a torch.Tensor, got {type(ratio_codes).__name__}.")
    if ratio_codes.dtype != torch.uint8:
        raise ValueError(
            f"ratio_codes must be {torch.uint8}, got {ratio_codes.dtype}; the artifact stores one byte per block."
        )
    if ratio_codes.dim() != 2 or (int(ratio_codes.shape[0]), int(ratio_codes.shape[1])) != (rows, num_blocks):
        raise ValueError(
            f"ratio_codes must have shape ({rows}, {num_blocks}), got {tuple(ratio_codes.shape)}; the codes do not "
            "describe this weight."
        )
    if device is not None and ratio_codes.device != device:
        raise ValueError(f"ratio_codes is on device {ratio_codes.device} but the weight is on {device}.")
    # uint8 cannot be negative, so the upper bound is the only reachable violation.
    if int(ratio_codes.max()) >= NVFP4_AWQ_CLIP_RATIO_COUNT:
        raise ValueError(
            f"ratio_codes holds {int(ratio_codes.max())}, but the {NVFP4_AWQ_CLIP_RATIO_COUNT} AWQ-clip ratios are "
            f"indexed by 0..{NVFP4_AWQ_CLIP_RATIO_COUNT - 1}."
        )
    return ratio_codes.detach().contiguous()


def _validate_group_activations(
    group_activations: Any, columns: int, device: torch.device
) -> Tuple[torch.Tensor, ...]:
    """Check the quantized activation rows of every equally weighted source group of an AWQ-clip search."""
    if not isinstance(group_activations, (list, tuple)) or not group_activations:
        raise ValueError(
            "group_activations must be a non-empty sequence of quantized activation row tensors, one per equally "
            "weighted source group."
        )
    groups = []
    for index, rows in enumerate(group_activations):
        described = f"group_activations[{index}]"
        if not isinstance(rows, torch.Tensor):
            raise TypeError(f"{described} must be a torch.Tensor, got {type(rows).__name__}.")
        if rows.dim() != 2 or int(rows.shape[0]) == 0:
            raise ValueError(f"{described} must be a non-empty rank-2 (N, K) tensor, got {tuple(rows.shape)}.")
        if int(rows.shape[1]) != columns:
            raise ValueError(
                f"{described} holds {int(rows.shape[1])}-wide rows, but the weight has {columns} input channel(s)."
            )
        if rows.dtype != torch.float32:
            raise ValueError(
                f"{described} must be {torch.float32}; the search scores the FP32 rows that "
                "nvfp4_awq_clip_activation_qdq produced, never raw low-precision ones."
            )
        if rows.device != device:
            raise ValueError(f"{described} is on device {rows.device} but the weight is on {device}.")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError(f"{described} contains non-finite values; the AWQ-clip objective is undefined for them.")
        groups.append(rows.detach().contiguous())
    return tuple(groups)


def _validate_template_reconstruction(template_reconstruction: Any, weight: torch.Tensor) -> torch.Tensor:
    """Check the unclipped candidate: the values the ordinary template of exactly *this* weight reads back."""
    if not isinstance(template_reconstruction, torch.Tensor):
        raise TypeError(
            f"template_reconstruction must be a torch.Tensor, got {type(template_reconstruction).__name__}."
        )
    if not template_reconstruction.is_floating_point():
        raise ValueError(f"template_reconstruction must be a float tensor, got {template_reconstruction.dtype}.")
    if tuple(template_reconstruction.shape) != tuple(weight.shape):
        raise ValueError(
            f"template_reconstruction has shape {tuple(template_reconstruction.shape)} but the weight is "
            f"{tuple(weight.shape)}; it must be the ordinary NVFP4 template's readback of this very weight."
        )
    if template_reconstruction.device != weight.device:
        raise ValueError(
            f"template_reconstruction is on device {template_reconstruction.device} but the weight is on "
            f"{weight.device}."
        )
    values = template_reconstruction.detach().to(torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError(
            "template_reconstruction contains non-finite values; the AWQ-clip objective is undefined for them."
        )
    return values.contiguous()


def _validate_gptq_hessian_matrix(hessian: Any, columns: int, device: torch.device) -> torch.Tensor:
    """Check a GPTQ Hessian and return it as FP32; nothing here is symmetrized, damped or repaired."""
    if not isinstance(hessian, torch.Tensor):
        raise TypeError(f"hessian must be a torch.Tensor, got {type(hessian).__name__}.")
    if hessian.dim() != 2 or (int(hessian.shape[0]), int(hessian.shape[1])) != (columns, columns):
        raise ValueError(
            f"hessian must have shape ({columns}, {columns}) to describe this weight's input channels, got "
            f"{tuple(hessian.shape)}."
        )
    if not hessian.is_floating_point():
        raise ValueError(f"hessian must be a float tensor, got {hessian.dtype}.")
    if hessian.device != device:
        raise ValueError(f"hessian is on device {hessian.device} but the weight is on {device}.")
    matrix = hessian.detach().to(torch.float32)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("hessian contains non-finite values; the GPTQ update is undefined for them.")
    return matrix


def _validate_gptq_block_scales(block_scale: Any, rows: int, num_blocks: int, device: torch.device) -> torch.Tensor:
    """Check the template block scales a GPTQ payload is written under; they are fixed and never recomputed.

    An exactly zero scale is part of the fixed template rather than a violation: the ordinary conversion encodes an
    all-zero -- or underflowed -- 16-weight block as a blocked scale of zero, and every value of such a block is
    zero however it is coded. :func:`_gptq_quantize_column` writes exactly those rows as FP4 code zero without ever
    taking a reciprocal, so no floor and no substitute scale is invented here. A negative or non-finite scale is
    refused, because no template construction produces one and nothing could be written under it.
    """
    if not isinstance(block_scale, torch.Tensor):
        raise TypeError(f"block_scale must be a torch.Tensor, got {type(block_scale).__name__}.")
    if block_scale.dim() != 2 or (int(block_scale.shape[0]), int(block_scale.shape[1])) != (rows, num_blocks):
        raise ValueError(
            f"block_scale must have shape ({rows}, {num_blocks}), got {tuple(block_scale.shape)}; the scales do not "
            "describe this weight."
        )
    if not block_scale.is_floating_point():
        raise ValueError(f"block_scale must be a float tensor, got {block_scale.dtype}.")
    if block_scale.device != device:
        raise ValueError(f"block_scale is on device {block_scale.device} but the weight is on {device}.")
    scales = block_scale.detach().to(torch.float32)
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0.0).any()):
        raise ValueError(
            "block_scale must be finite and non-negative: these are the ordinary template's own fixed scales, and a "
            "negative or non-finite one is not a scale that template could have produced."
        )
    return scales


def _validate_per_tensor_scale(per_tensor_scale: Any) -> None:
    """Check the fixed global scale. It is used as is and is never re-optimized, so it must be usable as given."""
    if per_tensor_scale is None:
        raise ValueError(
            "per_tensor_scale is missing; this packer keeps the global scale fixed and cannot supply one itself."
        )
    if not isinstance(per_tensor_scale, torch.Tensor):
        raise TypeError(f"per_tensor_scale must be a torch.Tensor, got {type(per_tensor_scale).__name__}.")
    if per_tensor_scale.numel() != 1:
        raise ValueError(f"per_tensor_scale must be a scalar tensor, got {per_tensor_scale.numel()} elements.")
    value = per_tensor_scale.detach().reshape(()).to(torch.float32)
    if not bool(torch.isfinite(value)) or float(value) <= 0.0:
        raise ValueError(f"per_tensor_scale must be positive and finite, got {float(value)}.")


def _validate_chunk_sizes(candidate_chunk_size: Any, block_chunk_size: Any) -> None:
    """Check the two bounded-search knobs; they may change peak memory but never the selected scales."""
    for name, value in (("candidate_chunk_size", candidate_chunk_size), ("block_chunk_size", block_chunk_size)):
        _validate_positive_int(value, name)


def _validate_positive_int(value: Any, name: str) -> None:
    """Check one bounded-search knob. ``True`` is an int in python but never a meaningful chunk size."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}.")


def _as_scalar_scale(per_tensor_scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the global scale as a detached 0-dim FP32 tensor, without touching the template's own storage."""
    return per_tensor_scale.detach().reshape(()).to(device=device, dtype=torch.float32)


def _template_attr(template: Any, names: Tuple[str, ...], description: str) -> Any:
    """Read the first attribute of ``names`` present on the template, or fail closed naming what was expected."""
    for name in names:
        if hasattr(template, name):
            return getattr(template, name)
    raise ValueError(
        f"template does not expose its {description} under any of {names}; the installed torchao NVFP4 wrapper is "
        "not the pinned one, and this packer refuses to guess which attribute holds it."
    )


# ------------------------------------------------------------------------------------------------------------------
# Lazy optional-dependency resolution
# ------------------------------------------------------------------------------------------------------------------


class _TorchAOBackend(NamedTuple):
    """The pinned TorchAO NVFP4 entry points this packer is written against."""

    NVFP4Tensor: type
    f32_to_f4_unpacked: Any
    f4_unpacked_to_f32: Any
    pack_uint4: Any
    to_blocked: Any


def _resolve_torchao() -> _TorchAOBackend:
    """Import the pinned TorchAO NVFP4 API on first use.

    Keeping this out of module import time is what lets a CPU-only or TorchAO-less install import this module.
    """
    global _torchao_backend
    if _torchao_backend is None:
        _require_float8_e4m3("The NVFP4 MSE weight packer")
        tensor_module = _import_module(_TORCHAO_TENSOR_MODULE)
        kernels = _import_module(_TORCHAO_KERNELS_MODULE)
        utils = _import_module(_TORCHAO_UTILS_MODULE)
        _torchao_backend = _TorchAOBackend(
            NVFP4Tensor=_module_attr(tensor_module, _TORCHAO_TENSOR_MODULE, "NVFP4Tensor"),
            f32_to_f4_unpacked=_module_attr(kernels, _TORCHAO_KERNELS_MODULE, "f32_to_f4_unpacked"),
            f4_unpacked_to_f32=_module_attr(kernels, _TORCHAO_KERNELS_MODULE, "f4_unpacked_to_f32"),
            pack_uint4=_module_attr(kernels, _TORCHAO_KERNELS_MODULE, "pack_uint4"),
            to_blocked=_module_attr(utils, _TORCHAO_UTILS_MODULE, "to_blocked"),
        )
    return _torchao_backend


class _AWQClipReadback(NamedTuple):
    """The pinned TorchAO entry points the offline unclipped candidate decodes a template's stored bytes with."""

    unpack_uint4: Any
    from_blocked: Any


def _resolve_awq_clip_readback() -> _AWQClipReadback:
    """Import the pinned TorchAO un-packing and un-swizzling helpers on first use.

    They are resolved separately from :func:`_resolve_torchao` on purpose: only the *offline* AWQ-clip unclipped
    candidate decodes a template's stored bytes, so an install that moved either helper fails that builder alone
    and never the runtime repacks, which do not need them. Both are re-exported by the pinned NVFP4 tensor module
    -- it decodes with them itself -- so either spelling of the import is accepted and neither is guessed at.
    """
    global _awq_clip_readback
    if _awq_clip_readback is None:
        _require_float8_e4m3("The NVFP4 AWQ-clip unclipped candidate")
        _awq_clip_readback = _AWQClipReadback(
            unpack_uint4=_pinned_attr((_TORCHAO_KERNELS_MODULE, _TORCHAO_TENSOR_MODULE), "unpack_uint4"),
            from_blocked=_pinned_attr((_TORCHAO_UTILS_MODULE, _TORCHAO_TENSOR_MODULE), "from_blocked"),
        )
    return _awq_clip_readback


def _ordinary_nvfp4_template(
    weight: torch.Tensor, per_tensor_scale: torch.Tensor, template_arithmetic: str, backend: "_TorchAOBackend"
) -> Any:
    """Convert ``weight`` the way the deployment's ordinary conversion does, in the named construction mode.

    The keywords are the pinned ones the runtime's own conversion is configured with -- block size 16, the ordinary
    global per-tensor scale, the swizzled Blackwell scale layout, and the Triton flag the resolved backend sets --
    so the template this returns is the one that run would have converted the weight into. An installed torchao
    that does not accept exactly those keywords is refused rather than called with a guessed subset.
    """
    to_nvfp4 = getattr(backend.NVFP4Tensor, "to_nvfp4", None)
    if to_nvfp4 is None:
        raise TorchAOUnavailableError(
            f"The AWQ-clip unclipped candidate requires {backend.NVFP4Tensor.__name__}.to_nvfp4, which the "
            "installed torchao does not provide."
        )
    try:
        return to_nvfp4(
            weight,
            block_size=NVFP4_BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=template_arithmetic == AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED,
        )
    except TypeError as err:
        raise TorchAOUnavailableError(
            f"{backend.NVFP4Tensor.__name__}.to_nvfp4 does not accept the pinned keywords (block_size, "
            "per_tensor_scale, is_swizzled_scales, use_triton_kernel); this prototype is pinned to the torchao "
            "0.17 NVFP4 conversion and refuses to construct the unclipped candidate any other way."
        ) from err


def _decode_nvfp4_template_values(
    metadata: "_TemplateMetadata", backend: "_TorchAOBackend", readback: "_AWQClipReadback"
) -> torch.Tensor:
    """Decode a template's exact stored bytes into FP32, with the same scale arithmetic every candidate uses.

    The payload is un-packed and mapped off the E2M1 grid with TorchAO's own kernels, and the blocked E4M3 scales
    are un-swizzled with TorchAO's own ``from_blocked``, so the values are those of the bytes the runtime deploys
    for an unclipped block. The two scales are then multiplied in FP32 and applied in the very order
    :func:`_reconstruct` applies them to a clipped candidate, ``payload * (per_tensor_scale * block_scale)``, which
    is what makes the eleven candidate deltas comparable. Nothing here casts through the template's ``orig_dtype``.
    """
    rows, columns = metadata.rows, metadata.columns
    num_blocks = columns // NVFP4_BLOCK_SIZE
    if metadata.qdata.element_size() != 1:
        raise ValueError(
            f"template qdata holds {metadata.qdata.element_size()}-byte elements; the AWQ-clip unclipped candidate "
            "decodes the deployed payload byte by byte and refuses a storage it cannot address as bytes."
        )
    payload = backend.f4_unpacked_to_f32(readback.unpack_uint4(metadata.qdata.contiguous().view(torch.uint8)))
    if payload.numel() != rows * columns:
        raise ValueError(
            f"The template's packed payload decoded to {payload.numel()} value(s), but a ({rows}, {columns}) NVFP4 "
            f"weight holds {rows * columns}; the installed torchao does not pack as this builder requires."
        )
    scales = readback.from_blocked(metadata.scale, rows, num_blocks)
    if scales.numel() != rows * num_blocks:
        raise ValueError(
            f"The template's blocked scales un-swizzled to {scales.numel()} value(s), but a ({rows}, {columns}) "
            f"NVFP4 weight has {rows * num_blocks} block(s); the installed torchao does not swizzle as this builder "
            "requires."
        )
    payload = payload.to(torch.float32).reshape(rows, num_blocks, NVFP4_BLOCK_SIZE)
    scales = scales.to(torch.float32).reshape(rows, num_blocks, 1)
    return (payload * (metadata.per_tensor_scale * scales)).reshape(rows, columns)


def _import_module(module_path: str):
    """Import a pinned TorchAO module, with an error that names the requirement rather than the traceback."""
    try:
        return __import__(module_path, fromlist=["__name__"])
    except ImportError as err:  # pragma: no cover - depends on the install
        raise TorchAOUnavailableError(
            f"The NVFP4 MSE weight packer requires torchao 0.17, but '{module_path}' could not be imported."
        ) from err


def _pinned_attr(module_paths: Tuple[str, ...], attribute: str) -> Any:
    """Read a pinned TorchAO attribute from the first of ``module_paths`` that defines it, or fail closed."""
    for module_path in module_paths:
        module = _import_module(module_path)
        if hasattr(module, attribute):
            return getattr(module, attribute)
    raise TorchAOUnavailableError(
        f"The NVFP4 AWQ-clip unclipped candidate requires '{attribute}' from one of {list(module_paths)}, which the "
        "installed torchao does not provide. This prototype is pinned to the torchao 0.17 NVFP4 quantization "
        "arithmetic and scale layout."
    )


def _module_attr(module: Any, module_path: str, attribute: str) -> Any:
    """Read a pinned TorchAO attribute, refusing to guess a replacement if the installed version moved it."""
    try:
        return getattr(module, attribute)
    except AttributeError as err:
        raise TorchAOUnavailableError(
            f"The NVFP4 MSE weight packer requires '{module_path}.{attribute}', which the installed torchao does not "
            "provide. This prototype is pinned to the torchao 0.17 NVFP4 quantization arithmetic and scale layout."
        ) from err


def _require_float8_e4m3(entry_point: str) -> None:
    """Raise unless this torch build exposes the E4M3 dtype the block scales are encoded in."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        raise TorchAOUnavailableError(
            f"{entry_point} requires torch.float8_e4m3fn, which torch {torch.__version__} does not expose."
        )

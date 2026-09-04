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
Controlled NVFP4 quantization of Sortformer transformer-encoder linear layers.

The utilities in this module are deliberately narrow: they only ever touch the four transformer-block linear
families listed in :data:`QUANTIZATION_TARGET_SUFFIXES`, they never mutate the model definition, and they never
silently degrade a requested recipe. TorchAO and MSLK are imported lazily so that a disabled recipe, or a CPU-only
import of this module, does not pull in optional dependencies.

An opt-in execution variant lives in :mod:`~nemo.collections.asr.parts.utils.sortformer_nvfp4_scale_fold`: with
``fold_global_scales`` the calibrated global-scale product is carried by the ``_scaled_mm`` block-scale operands
instead of a separate post-GEMM rescale. It is disabled by default and leaves the ordinary path untouched.

A second opt-in execution variant lives in
:mod:`~nemo.collections.asr.parts.utils.sortformer_nvfp4_producer_fusion`: with ``fuse_producer_packing`` the
LayerNorm and GELU producers in front of ``attn.w_qkv``, ``ffn.net.0`` and ``ffn.net.3`` are evaluated inside the
NVFP4 activation pack instead of being materialized in BF16 first. It is disabled by default, mutually exclusive
with folding, and leaves the ordinary path untouched.

A third opt-in variant is file-backed BF16 restoration: on top of ``nvfp4_weight_only`` or ``nvfp4_all``,
``bf16_override_path`` names a JSON file listing exact target FQNs that stay in BF16 while every unlisted target
is quantized as usual — NVFP4 weight-only under the first recipe, NVFP4 W4A4 under the second. It exists for
controlled family/layer sensitivity experiments, is rejected for every other recipe, and is read strictly by
:func:`load_bf16_override`, so a malformed or stale file fails instead of quietly changing which layers an
experiment measured. Static calibration is validated and applied only for the FQNs that stay W4A4; entries for
restored FQNs may remain in a common calibration artifact and are reported as unused.

A fourth opt-in variant selects *how* each NVFP4 weight's block scales are chosen: ``weight_scale_method='mse'``
converts every selected weight with the ordinary TorchAO recipe and then repacks it with the exhaustive
per-16-weight-block E4M3 search of
:mod:`~nemo.collections.asr.parts.utils.sortformer_nvfp4_weight_mse`, keeping TorchAO's global per-tensor scale and
wire format. It is a weight-side PTQ technique only: no kernel, packed format or recipe changes, and the default
``'amax'`` runs the ordinary batched TorchAO conversion calls unchanged. It is rejected together with global-scale
folding, which would re-round the searched block scales, and the summary reports the per-FQN and aggregate
reconstruction MSE it achieved.

``weight_scale_method='local_hessian'`` is the same search against a different objective: each block scale
minimizes ``sum_j h_damped[j] * (W - Q(W))^2``, the diagonal approximation of ``E||X (W - Q(W))||^2``, where ``h``
is one non-negative input-channel second-moment vector per FQN read from a strict, checkpoint-bound
:data:`HESSIAN_SCHEMA` artifact and damped by :data:`WEIGHT_SCALE_HESSIAN_DAMPING` of its own mean. It requires
``recipe='nvfp4_all'`` and ``weight_scale_hessian_path``; the artifact is loaded, validated against the live target
set and verified against every original weight's canonical digest *before* anything is converted, and it is
rejected together with global-scale folding and BF16 restoration. Nothing about the packed layout, the kernels or
the execution path changes, and no fallback exists: a failed requested method is a failed run.

``weight_scale_method='four_over_six'`` is the third opt-in method and the only one that is not an exhaustive
search: it reproduces NVIDIA ModelOpt 0.46.0's Four-Over-Six weight arithmetic inside this runtime's packed format.
Every converted NVFP4 weight is repacked against a weight global scale renormalized to
:data:`WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX` (the template's own scale times ``448 / 256``), and each 16-weight block
is written with whichever of :data:`WEIGHT_SCALE_FOUR_OVER_SIX_MAGNITUDES` -- the block amax mapped onto FP4's
largest magnitude 6, or onto 4 -- reconstructs it with the lower plain squared error, ties keeping ``M=6``. It
applies to every NVFP4 weight of any enabled NVFP4 recipe, never to an FP8 or BF16-restored one, and it is rejected
together with global-scale folding and with a Hessian artifact path. Unlike the two searches it carries no
guarantee of beating the ordinary conversion, so the summary reports both reconstruction MSEs and the per-FQN
``M=6``/``M=4`` counts and rejects nothing merely for being worse.

``weight_scale_method='awq_clip'`` is the fourth opt-in method and the only one whose selection is made entirely
offline. It adapts NVIDIA ModelOpt 0.46.0's AWQ clipping to this runtime's packed format: for every output row and
contiguous 16-weight input block, an offline builder evaluates the eleven fixed clipping ratios
:data:`WEIGHT_SCALE_AWQ_CLIP_RATIOS` and keeps the one minimizing the *block-local output* error against
runtime-matched quantized activations, ties keeping the earliest ratio. The runtime consumes only the resulting
uint8 ratio codes from a strict, checkpoint-bound :data:`AWQ_CLIP_SCHEMA` artifact, reconstructs exactly those block
scales with the template's own global scale, and changes nothing about the packed layout, the activation behavior or
the kernels. The eleventh, unclipped ratio ``1.00`` is not a formula at all: it means "leave this block exactly as
the ordinary conversion wrote it", so those blocks keep the template's own scale and payload bytes. That is only
well defined against one ordinary-template construction, so the artifact records which of
:data:`WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS` it was scored against and a run whose backend would construct
the other one is refused before anything is converted. It requires ``recipe='nvfp4_all'`` with ``scale_mode='static'``, ``scale_margin`` exactly
:data:`WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN`, a readable static calibration file and
``weight_scale_awq_clip_path``; the artifact is bound to the exact bytes of that calibration file, because the
activations it scored were quantized with those values. Like Four-Over-Six it carries no guarantee of a lower plain
reconstruction MSE -- it optimizes output error, not weight error -- so both MSEs are reported and neither rejects a
valid repack. It is rejected together with global-scale folding, BF16 restoration and a Hessian artifact path.

``weight_scale_method='gptq'`` is the fifth opt-in method and the only one that changes no scale at all. It adapts
NVIDIA ModelOpt 0.46.0's GPTQ to this runtime's packed format: an offline builder forms the full input Hessian of
every target from runtime-matched quantized activation rows, balanced equally over labelled source groups, damps it
by :data:`WEIGHT_SCALE_GPTQ_PERC_DAMP` of its own diagonal mean after the reference's dead-column handling, and then
writes each input column's FP4 payload under the *ordinary* template's own fixed block and global scales, propagating
each column's rounding residual to the columns that follow it over
:data:`WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE`-column blocks. The runtime consumes only the resulting packed payload
bytes from a strict, checkpoint-bound :data:`GPTQ_SCHEMA` artifact: it converts each target with the ordinary recipe,
verifies that the template's scale-buffer bytes and global scale are exactly the ones the payload was written under,
and replaces nothing but ``qdata``. The block scales, the swizzled scale layout, the activation behavior and the
kernels are untouched. Like ``awq_clip`` it requires ``recipe='nvfp4_all'`` with ``scale_mode='static'``,
``scale_margin`` exactly :data:`WEIGHT_SCALE_GPTQ_SCALE_MARGIN`, a readable static calibration file the artifact is
bound to by digest, and ``weight_scale_gptq_path``; it is rejected together with global-scale folding, BF16
restoration and any other method's artifact path, and a backend that would construct the other ordinary template is
refused before anything is converted. It optimizes output error, so both reconstruction MSEs are reported and neither
rejects a valid replacement.

Self-contained quantized ``.nemo`` export is not supported here; a quantized model produced by
:func:`quantize_sortformer_model` is an in-process evaluation artifact only.

Two claims are deliberately kept conditional rather than asserted. Accelerated NVFP4 packing depends on the
runtime token count (``M % 128 == 0``), which is unknown at quantization time and cannot be enforced afterwards
without being traced away by ``torch.compile`` or bypassed by the module swap, so the summary reports the
configured packing path together with an explicit "conditional, unverified" acceleration status. Likewise, the
accepted Blackwell compute capabilities are a capability *policy*: the summary separates policy acceptance from
per-architecture test evidence, which callers supply through :class:`CapabilityFacts`.
"""

import base64
import binascii
import hashlib
import importlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nemo.collections.asr.parts.utils.sortformer_nvfp4_producer_fusion import (
    PRODUCER_FUSION_REQUIRED_CONSUMERS,
    apply_producer_fusion,
    disabled_producer_fusion_summary,
)
from nemo.collections.asr.parts.utils.sortformer_nvfp4_scale_fold import (
    apply_global_scale_folding,
    disabled_folding_summary,
    validate_fold_exponent,
)
from nemo.collections.asr.parts.utils.sortformer_nvfp4_weight_mse import (
    AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED,
    AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE,
    AWQ_CLIP_TEMPLATE_ARITHMETICS,
    FP8_E4M3_MAX,
    NVFP4_AWQ_CLIP_RATIO_COUNT,
    NVFP4_AWQ_CLIP_RATIOS,
    NVFP4_AWQ_CLIP_SCALE_MIN,
    NVFP4_AWQ_CLIP_UNCLIPPED_CODE,
    NVFP4_BLOCK_SIZE,
    NVFP4_FOUR_OVER_SIX_FP8_MAX,
    NVFP4_FOUR_OVER_SIX_MAGNITUDES,
    NVFP4_GPTQ_PERC_DAMP,
    NVFP4_GPTQ_UPDATE_BLOCK_SIZE,
    NVFP4_HESSIAN_DAMPING,
    NVFP4_MAX,
    FourOverSixRepack,
    damped_second_moments,
    nvfp4_template_identity,
    repack_nvfp4_weight_awq_clip,
    repack_nvfp4_weight_four_over_six,
    repack_nvfp4_weight_gptq,
    repack_nvfp4_weight_local_hessian,
    repack_nvfp4_weight_mse,
)

# Exact transformer-block linear families that may be quantized. Matching is anchored to a full FQN suffix so
# that model-level projections (e.g. the encoder ``out_proj``), the pre-encoder, norms, and the diarization head
# can never be selected by accident.
QUANTIZATION_TARGET_SUFFIXES: Tuple[str, ...] = (
    "attn.w_qkv",
    "attn.out_proj",
    "ffn.net.0",
    "ffn.net.3",
)

QUANTIZATION_RECIPES: Tuple[str, ...] = (
    "disabled",
    "nvfp4_all",
    "nvfp4_qkv_only",
    "nvfp4_qkv_fp8_rest",
    "nvfp4_weight_only",
)

SCALE_MODES: Tuple[str, ...] = ("dynamic", "static")

# How each NVFP4 weight's per-16-element block scales are chosen. ``amax`` is TorchAO's ordinary rule (the block
# amax mapped onto FP4's largest magnitude, rounded to E4M3) and is the default; ``mse`` converts with that same
# recipe and then repacks every selected weight with an exhaustive search over the 126 positive finite E4M3
# encodings, keeping TorchAO's global per-tensor scale and wire format. The two produce numerically different
# weights, so they are separated in the prediction-cache identity.
WEIGHT_SCALE_METHOD_AMAX = "amax"
WEIGHT_SCALE_METHOD_MSE = "mse"
# ``local_hessian`` runs the same exhaustive search but minimizes the *activation-weighted* block error
# ``sum_j h_damped[j] * (W - Q(W))^2``, where ``h`` is the diagonal input-channel second moment of a strict,
# checkpoint-bound offline artifact. It is the diagonal approximation of ``E||X (W - Q(W))||^2``, i.e. of the error
# the layer actually makes on this checkpoint's activations, which raw weight MSE is not aligned with.
WEIGHT_SCALE_METHOD_LOCAL_HESSIAN = "local_hessian"
# ``four_over_six`` is neither of the two exhaustive searches: it reproduces NVIDIA ModelOpt 0.46.0's Four-Over-Six
# weight arithmetic, which compares exactly two representations of every 16-weight block -- the block amax mapped
# onto FP4's largest magnitude 6, or onto 4 -- and keeps the one with the lower plain squared reconstruction error.
# It is the one method that does not keep TorchAO's global per-tensor scale: the reference normalizes the weight
# global scale against 256 instead of 448, which is exactly the headroom the 1.5x 'M=4' scale needs.
WEIGHT_SCALE_METHOD_FOUR_OVER_SIX = "four_over_six"
# ``awq_clip`` adapts NVIDIA ModelOpt 0.46.0's AWQ clipping: an offline builder decides, per output row and 16-weight
# input block, which of eleven fixed clipping ratios of the block's own amax minimizes that block's contribution to
# the layer's *output* error on runtime-matched quantized activations, and the runtime only reconstructs the scales
# those codes name. It is the one method whose selection cannot be recomputed at conversion time, because the
# activation rows it scored are deliberately absent from the artifact.
WEIGHT_SCALE_METHOD_AWQ_CLIP = "awq_clip"
# ``gptq`` is the one method that selects no scale at all. It keeps the ordinary template's block scales, its global
# scale and its scale bytes exactly, and changes only which FP4 payload each input column is written with, so that
# the rounding error of an already-written column is compensated by the columns that follow it under the layer's
# full input Hessian. Like ``awq_clip`` the selection happens offline and cannot be recomputed at conversion time.
WEIGHT_SCALE_METHOD_GPTQ = "gptq"
WEIGHT_SCALE_METHODS: Tuple[str, ...] = (
    WEIGHT_SCALE_METHOD_AMAX,
    WEIGHT_SCALE_METHOD_MSE,
    WEIGHT_SCALE_METHOD_LOCAL_HESSIAN,
    WEIGHT_SCALE_METHOD_FOUR_OVER_SIX,
    WEIGHT_SCALE_METHOD_AWQ_CLIP,
    WEIGHT_SCALE_METHOD_GPTQ,
)

# Stable identity of the searched repack, recorded in the summary and -- only when the search is on -- in the
# prediction-cache identity, so a searched run can never reuse amax predictions and a future search version can
# never reuse this one's.
WEIGHT_SCALE_MSE_ALGORITHM = "nvfp4_block_e4m3_exhaustive_mse"
WEIGHT_SCALE_MSE_ALGORITHM_VERSION = 1

# The same, for the activation-weighted search. The damping is part of the identity, not a knob: it is the packer's
# own constant, is written into every artifact, and is rejected on load when it differs.
WEIGHT_SCALE_HESSIAN_ALGORITHM = "nvfp4_block_e4m3_exhaustive_diagonal_hessian"
WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION = 1
WEIGHT_SCALE_HESSIAN_DAMPING = NVFP4_HESSIAN_DAMPING

# ``local_hessian`` is a W4A4 technique for the complete target set: the moments are collected on, and the artifact
# is bound to, exactly the 124 modules of this recipe.
WEIGHT_SCALE_HESSIAN_RECIPES: Tuple[str, ...] = ("nvfp4_all",)

# Stable identity of the Four-Over-Six repack. The normalization maximum and the two candidate magnitudes are part
# of the identity rather than knobs: they are what the reference preset fixes, so a run that used different ones
# would be a different algorithm and must never reuse this one's predictions.
WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM = "nvfp4_block_e4m3_modelopt_four_over_six"
WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION = 1
WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX = NVFP4_FOUR_OVER_SIX_FP8_MAX
WEIGHT_SCALE_FOUR_OVER_SIX_MAGNITUDES: Tuple[int, int] = NVFP4_FOUR_OVER_SIX_MAGNITUDES

# Stable identity of the AWQ-clip repack. The ratio list, its order, the block size, the tie rule, the candidate
# scale rule and the activation-QDQ arithmetic are all part of that identity rather than knobs: a build that changed
# any of them would store different weights under the same method name, so every one of them is written into the
# artifact, re-checked on load, and recorded in the prediction-cache identity.
WEIGHT_SCALE_AWQ_CLIP_ALGORITHM = "nvfp4_block_e4m3_modelopt_awq_clip"
WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION = 1
WEIGHT_SCALE_AWQ_CLIP_RATIOS: Tuple[float, ...] = NVFP4_AWQ_CLIP_RATIOS
WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT = NVFP4_AWQ_CLIP_RATIO_COUNT
WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE = NVFP4_BLOCK_SIZE
WEIGHT_SCALE_AWQ_CLIP_SCALE_MIN = NVFP4_AWQ_CLIP_SCALE_MIN
WEIGHT_SCALE_AWQ_CLIP_UNCLIPPED_CODE = NVFP4_AWQ_CLIP_UNCLIPPED_CODE

# Which ordinary-template construction the codes were selected against. The unclipped code means "leave this block
# as the ordinary conversion wrote it", and the two constructions do not write it the same way -- pinned TorchAO's
# non-Triton path floors a near-zero block scale at ``2 ** -6`` where the accelerated MSLK kernel stores an E4M3
# zero, and their packed payloads differ on ordinary blocks too. An artifact therefore records the one it was scored
# against, and a run whose backend would construct the other is refused before anything is converted.
WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS: Tuple[str, ...] = AWQ_CLIP_TEMPLATE_ARITHMETICS
WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE = AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE
WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED = AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED

# ``awq_clip`` is a W4A4 technique for the complete target set with calibrated static activation scales: the codes
# were selected against exactly those quantized activations, so the artifact is bound to all 124 modules and to the
# exact calibration file, and the runtime margin must be the 1.0 that the frozen calibration's baked-in headroom
# already accounts for.
WEIGHT_SCALE_AWQ_CLIP_RECIPES: Tuple[str, ...] = ("nvfp4_all",)
WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE = "static"
WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN = 1.0

# The reference implementation this method was adapted from, recorded so an artifact says which arithmetic it
# reproduces. ModelOpt is never imported and is not a runtime dependency; only its identity is carried.
MODELOPT_REFERENCE_VERSION = "0.46.0"
MODELOPT_REFERENCE_WHEEL_SHA256 = "1864b4e9921e287b065be3861ab48345144e673273ebb2b94bd9a6119a9eba8e"

# Both searches are exhaustive over exactly the encodings the amax rule also rounds into, so the searched objective
# can never exceed the converted one except by float reduction noise. These two tolerances cover that noise and
# nothing else: a genuinely worse repack is a defect and is raised rather than accepted.
WEIGHT_SCALE_MSE_RELATIVE_TOLERANCE = 1e-6
WEIGHT_SCALE_MSE_ABSOLUTE_TOLERANCE = 1e-12

# Schema of the runtime diagonal-Hessian artifact produced by
# ``scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_hessian_stats.py``. It carries no activation
# row, no weight, no quantized payload, no label and no task metric: one non-negative second-moment vector per FQN,
# one canonical digest per original weight, one digest each over the moments and over the provenance, the checkpoint
# it was built on, the algorithm identity with its damping, and the provenance that says exactly which bounded
# sample files produced it. Every key set below is closed,
# because an artifact carrying an unknown key was not written by that builder and cannot be read as if it were.
HESSIAN_SCHEMA = "sortformer_nvfp4_diagonal_hessian"
HESSIAN_SCHEMA_VERSION = 1
HESSIAN_ARTIFACT_KEYS: frozenset = frozenset(
    {
        "schema",
        "version",
        "checkpoint_sha256",
        "algorithm",
        "algorithm_version",
        "damping",
        "weight_digest_method",
        "weight_sha256",
        "moment_sha256",
        "provenance_sha256",
        "diagonal_hessian",
        "provenance",
    }
)
HESSIAN_PROVENANCE_KEYS: frozenset = frozenset(
    {
        "method",
        "method_version",
        "objective",
        "group_reduction",
        "targets",
        "target_module_count",
        "target_fqns",
        "sources",
        "aggregate",
    }
)
HESSIAN_SOURCE_KEYS: frozenset = frozenset(
    {
        "label",
        "name",
        "sha256",
        "size_bytes",
        "seed",
        "max_rows",
        "sampled_row_count",
        "finite_row_count",
        "nonfinite_row_count",
        "metadata",
    }
)
HESSIAN_AGGREGATE_KEYS: frozenset = frozenset(
    {"module_count", "source_count", "source_labels", "moment_count", "moment_min", "moment_max"}
)

# Exact construction and objective recorded in every artifact. Both are constants of this implementation: an
# artifact that states a different construction was produced by a different builder, and one that states a
# different objective describes multipliers this runtime does not implement.
HESSIAN_CONSTRUCTION_METHOD = "group_balanced_mean_of_per_source_mean_squared_activation_rows"
HESSIAN_CONSTRUCTION_METHOD_VERSION = 1
HESSIAN_GROUP_REDUCTION = "mean over labelled source groups, each group weighted equally regardless of its rows"
HESSIAN_OBJECTIVE = (
    "per output row and contiguous 16-weight input block, the positive finite E4M3 scale minimizing "
    "sum_j h_damped[j] * (W[r, j] - Q_scale(W[r, j])) ** 2, with h_damped = h + 0.01 * mean(h)"
)

# Schema of the runtime AWQ-clip artifact produced by
# ``scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_awq_clip.py``. It carries no activation row, no
# weight, no quantized payload, no label and no task metric: one compact uint8 ratio-code matrix per FQN, the exact
# arithmetic identity of the selection, the identity of the activation-calibration file whose values quantized the
# scored activations, one canonical digest per original weight, one digest each over the codes and over the
# provenance, and the provenance that says exactly which bounded sample files produced it. Every key set below is
# closed, because an artifact carrying an unknown key was not written by that builder.
AWQ_CLIP_SCHEMA = "sortformer_nvfp4_awq_clip"
AWQ_CLIP_SCHEMA_VERSION = 1
AWQ_CLIP_ARTIFACT_KEYS: frozenset = frozenset(
    {
        "schema",
        "version",
        "checkpoint_sha256",
        "algorithm",
        "algorithm_version",
        "arithmetic",
        "activation_calibration",
        "weight_digest_method",
        "weight_sha256",
        "code_encoding",
        "ratio_codes",
        "ratio_code_sha256",
        "provenance_sha256",
        "provenance",
    }
)
AWQ_CLIP_ARITHMETIC_KEYS: frozenset = frozenset(
    {
        "block_size",
        "clip_ratios",
        "tie_rule",
        "objective",
        "group_reduction",
        "candidate_scale_rule",
        "template_arithmetic",
        "activation_qdq",
        "fp4_max",
        "fp8_e4m3_max",
        "scale_min",
        "modelopt_reference_version",
        "modelopt_reference_wheel_sha256",
    }
)
AWQ_CLIP_CALIBRATION_KEYS: frozenset = frozenset(
    {
        "name",
        "sha256",
        "size_bytes",
        "checkpoint_sha256",
        "version",
        "scale_mode",
        "scale_margin",
        "target_module_count",
        "headroom",
        "headroom_baked_in",
        "runtime_scale_margin",
        "merge_method",
        "merge_method_version",
    }
)
AWQ_CLIP_CODE_KEYS: frozenset = frozenset({"shape", "codes"})
AWQ_CLIP_PROVENANCE_KEYS: frozenset = frozenset(
    {
        "method",
        "method_version",
        "objective",
        "group_reduction",
        "targets",
        "target_module_count",
        "target_fqns",
        "sources",
        "modules",
        "aggregate",
    }
)
AWQ_CLIP_SOURCE_KEYS: frozenset = frozenset(
    {
        "label",
        "name",
        "sha256",
        "size_bytes",
        "seed",
        "max_rows",
        "sampled_row_count",
        "finite_row_count",
        "nonfinite_row_count",
        "metadata",
    }
)
AWQ_CLIP_MODULE_KEYS: frozenset = frozenset(
    {"block_count", "ratio_histogram", "selected_objective", "unclipped_objective"}
)
AWQ_CLIP_AGGREGATE_KEYS: frozenset = frozenset(
    {
        "module_count",
        "source_count",
        "source_labels",
        "block_count",
        "ratio_histogram",
        "selected_objective",
        "unclipped_objective",
    }
)

# Exact construction, objective and arithmetic recorded in every AWQ-clip artifact. All of them are constants of
# this implementation: an artifact stating anything else describes a selection this runtime does not implement, and
# is refused rather than read as if it were this one's.
AWQ_CLIP_CONSTRUCTION_METHOD = "group_balanced_mean_of_per_source_mean_squared_block_output_error"
AWQ_CLIP_CONSTRUCTION_METHOD_VERSION = 1
AWQ_CLIP_GROUP_REDUCTION = "mean over labelled source groups, each group weighted equally regardless of its rows"
AWQ_CLIP_OBJECTIVE = (
    "per output row r and contiguous 16-weight input block b, the clipping ratio minimizing "
    "mean_g(mean_rows((sum_j Xq_g[row, j] * (W[r, j] - Q_ratio(W[r, j]))) ** 2)), where Xq_g are the group's "
    "activation rows after the runtime-matched NVFP4 activation quantize/dequantize and Q_ratio is the candidate "
    "the runtime deploys for that code: the clipped reconstruction for codes 0..9 and, for code 10, the FP32 decode "
    "of the ordinary template's own stored bytes, f4_unpacked_to_f32(payload) * (weight_global_scale * scale_fp8) "
    "with both scales in FP32; every delta is therefore FP32, the block-local full covariance is kept, and no "
    "interaction between different blocks is scored"
)
AWQ_CLIP_TIE_RULE = (
    "candidates are compared in the fixed clip-ratio order and one replaces the incumbent only on a strictly "
    "smaller loss, so an exact tie keeps the earliest ratio 0.5"
)
AWQ_CLIP_CANDIDATE_SCALE_RULE = (
    "codes 0..9 store scale_fp8 = E4M3(clamp(((block_amax * ratio) / 6) / weight_global_scale, 2 ** -9, 448)), "
    "with weight_global_scale = global_amax / (448 * 6); code 10 stores nothing of its own and keeps the ordinary "
    "template's exact scale byte and packed payload bytes for that block"
)
AWQ_CLIP_ACTIVATION_QDQ = (
    "torchao 0.17 NVFP4: activation global scale = calibrated amax / (448 * 6); per contiguous 16-value block "
    "scale_fp8 = E4M3(clamp((block_amax / 6) / global_scale, 2 ** -9, 448)); payload = "
    "f32_to_f4_unpacked(clamp(x * ((1 / global_scale) / scale_fp8), -6, 6)); reconstruction = "
    "f4_unpacked_to_f32(payload) * (global_scale * scale_fp8)"
)
AWQ_CLIP_CODE_ENCODING = "base64(contiguous row-major uint8 codes indexing the fixed clip-ratio list)"

# Stable identity of the GPTQ payload selection. Every constant below *is* the algorithm: the damping fraction, the
# Hessian normalization and group reduction, the dead-column rule, the inverse, the 128-column update block, the
# 16-value NVFP4 block, the fixed-original-template-scale rule and the activation QDQ each change which payload a
# column is written with, so all of them are written into every artifact, re-checked on load, and recorded in the
# prediction-cache identity.
WEIGHT_SCALE_GPTQ_ALGORITHM = "nvfp4_qdata_modelopt_gptq"
WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION = 1
WEIGHT_SCALE_GPTQ_PERC_DAMP = NVFP4_GPTQ_PERC_DAMP
WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE = NVFP4_GPTQ_UPDATE_BLOCK_SIZE
WEIGHT_SCALE_GPTQ_BLOCK_SIZE = NVFP4_BLOCK_SIZE

# GPTQ writes the payload of the *ordinary* template, whose two constructions do not produce the same scale bytes or
# the same baseline payload, so it inherits exactly the AWQ-clip construction distinction and its backend binding.
WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS: Tuple[str, ...] = AWQ_CLIP_TEMPLATE_ARITHMETICS
WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_REFERENCE = AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE
WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED = AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED

# ``gptq`` is a W4A4 technique for the complete target set with calibrated static activation scales: the Hessians
# were formed from exactly those quantized activations, so the artifact is bound to all 124 modules and to the exact
# calibration file, and the runtime margin must be the 1.0 the frozen calibration's baked-in headroom accounts for.
WEIGHT_SCALE_GPTQ_RECIPES: Tuple[str, ...] = ("nvfp4_all",)
WEIGHT_SCALE_GPTQ_SCALE_MODE = "static"
WEIGHT_SCALE_GPTQ_SCALE_MARGIN = 1.0

# Schema of the runtime GPTQ artifact produced by
# ``scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_gptq.py``. Unlike every other artifact in this
# module it carries a *payload*: the final packed FP4 bytes of every target weight. It still carries no activation
# row, no high-precision weight, no Hessian matrix, no label and no task metric -- only those bytes, compact
# per-module statistics, the identities everything is bound to, and the provenance of the construction. Every key
# set below is closed, because an artifact carrying an unknown key was not written by that builder.
GPTQ_SCHEMA = "sortformer_nvfp4_gptq"
GPTQ_SCHEMA_VERSION = 1
GPTQ_ARTIFACT_KEYS: frozenset = frozenset(
    {
        "schema",
        "version",
        "checkpoint_sha256",
        "algorithm",
        "algorithm_version",
        "arithmetic",
        "activation_calibration",
        "weight_digest_method",
        "section_digest_method",
        "weight_sha256",
        "payload_encoding",
        "qdata",
        "template_scale",
        "hessian",
        "qdata_sha256",
        "hessian_sha256",
        "provenance_sha256",
        "provenance",
    }
)
GPTQ_ARITHMETIC_KEYS: frozenset = frozenset(
    {
        "perc_damp",
        "hessian_rule",
        "group_reduction",
        "dead_column_rule",
        "inverse_rule",
        "update_block_size",
        "block_size",
        "template_scale_rule",
        "activation_qdq",
        "objective",
        "template_arithmetic",
        "fp4_max",
        "fp8_e4m3_max",
        "hessian_digest_method",
        "modelopt_reference_version",
        "modelopt_reference_wheel_sha256",
    }
)
# The calibration identity is the very object ``nvfp4_awq_clip_calibration_identity`` produces, plus the margin the
# codes were selected at, so both methods bind to a static calibration file in exactly the same way.
GPTQ_CALIBRATION_KEYS: frozenset = AWQ_CLIP_CALIBRATION_KEYS
GPTQ_QDATA_KEYS: frozenset = frozenset({"shape", "dtype", "byte_length", "payload", "sha256"})
GPTQ_TEMPLATE_SCALE_KEYS: frozenset = frozenset({"shape", "dtype", "byte_length", "sha256", "global_scale_sha256"})
GPTQ_HESSIAN_KEYS: frozenset = frozenset(
    {
        "sha256",
        "input_features",
        "sampled_row_count",
        "dead_column_count",
        "damping",
        "diagonal_min",
        "diagonal_max",
        "diagonal_mean",
    }
)
GPTQ_PROVENANCE_KEYS: frozenset = frozenset(
    {
        "method",
        "method_version",
        "objective",
        "group_reduction",
        "targets",
        "target_module_count",
        "target_fqns",
        "sources",
        "modules",
        "aggregate",
    }
)
GPTQ_SOURCE_KEYS: frozenset = AWQ_CLIP_SOURCE_KEYS
GPTQ_MODULE_KEYS: frozenset = frozenset(
    {
        "shape",
        "weight_count",
        "block_count",
        "qdata_byte_length",
        "template_mse",
        "selected_mse",
        "template_objective",
        "selected_objective",
    }
)
GPTQ_AGGREGATE_KEYS: frozenset = frozenset(
    {
        "module_count",
        "source_count",
        "source_labels",
        "block_count",
        "weight_count",
        "qdata_byte_length",
        "dead_column_count",
        "template_mse",
        "selected_mse",
        "template_objective",
        "selected_objective",
    }
)

# Exact construction, objective and arithmetic recorded in every GPTQ artifact. All of them are constants of this
# implementation: an artifact stating anything else describes a payload selection this runtime does not implement.
GPTQ_CONSTRUCTION_METHOD = "group_balanced_mean_input_hessian_column_wise_gptq_payload_under_fixed_template_scales"
GPTQ_CONSTRUCTION_METHOD_VERSION = 1
GPTQ_GROUP_REDUCTION = "mean over labelled source groups, each group weighted equally regardless of its rows"
GPTQ_HESSIAN_RULE = (
    "per labelled source group g of N_g bounded activation rows Xq_g, taken after the runtime-matched static NVFP4 "
    "activation quantize/dequantize: scaled = sqrt(2 / N_g) * Xq_g.T in FP32 and H_g = scaled @ scaled.T; the "
    "groups are visited in sorted label order, summed in that order and divided by the group count, all in FP32"
)
GPTQ_DEAD_COLUMN_RULE = (
    "a column whose original weight is exactly zero in every output row has its Hessian row and column zeroed and "
    "its diagonal entry set to 1, before the damping is computed"
)
GPTQ_INVERSE_RULE = (
    "h_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H_damped)), upper=True); a failed "
    "factorization, a zero diagonal or a non-finite entry is a hard error and never an identity fallback"
)
GPTQ_TEMPLATE_SCALE_RULE = (
    "every payload decision uses the ordinary original-weight template's own fixed scales -- its blocked E4M3 "
    "buffer decoded with torchao's from_blocked and its unchanged global per-tensor scale "
    "global_amax / (448 * 6) -- applied as f32_to_f4_unpacked(clamp(w * ((1 / global_scale) / scale_fp8), -6, 6)) "
    "and f4_unpacked_to_f32(payload) * (global_scale * scale_fp8); no scale is ever recomputed from the "
    "GPTQ-updated working weight, and only the packed payload bytes are stored"
)
GPTQ_OBJECTIVE = (
    "input columns are visited in order and written with the payload their fixed template scales quantize the "
    "current working column into; the residual (w_col - q_col) / h_inv[i, i] is subtracted from the later columns "
    "of the same 128-column update block through addr_(err, h_inv[i, i:], alpha=-1) and, after the block, from "
    "every following column through addmm_(errors, h_inv[block, block_end:], alpha=-1); the reported offline "
    "objective is the damped Hessian quadratic form sum_r delta_r^T H delta_r divided by the weight's element "
    "count, with delta = W - Q(W)"
)
GPTQ_ACTIVATION_QDQ = AWQ_CLIP_ACTIVATION_QDQ
GPTQ_PAYLOAD_ENCODING = (
    "base64(contiguous row-major bytes of the torchao pack_uint4 NVFP4 payload, two FP4 codes per byte in "
    "torchao's own nibble order)"
)
GPTQ_HESSIAN_DIGEST_METHOD = (
    "sha256(contiguous row-major little-endian float32 bytes of the damped Hessian); the matrix itself is never "
    "stored"
)

# Canonical per-weight digest, shared by the offline builder and the runtime binding check. Hashing the dtype and
# the shape alongside the raw little-endian bytes makes a reshaped or re-dtyped weight a different weight, which is
# what a binding between offline statistics and a runtime weight has to mean.
WEIGHT_DIGEST_METHOD = "sha256(str(dtype) + '|' + str(tuple(shape)) + '|' + contiguous_cpu_raw_bytes)"

# Canonical digest of one whole artifact section, shared by the offline builder and the runtime loader. The section
# is re-serialized as compact, sorted, non-NaN UTF-8 JSON before hashing, so the digest is a property of the values
# the section carries and not of the file's indentation or key order, and it survives the write/read round trip.
# The artifact records one such digest for its moments and one for its provenance, which makes each of those
# independently verifiable instead of merely implied by the digest of the whole file.
SECTION_DIGEST_METHOD = (
    "sha256(json.dumps(section, sort_keys=True, separators=(',', ':'), ensure_ascii=False, "
    "allow_nan=False).encode('utf-8'))"
)

# Precision labels used in the structured summary and in the internal FQN -> precision map.
PRECISION_NVFP4_W4A4 = "nvfp4_w4a4"
PRECISION_NVFP4_WEIGHT_ONLY = "nvfp4_weight_only"
PRECISION_FP8_DYNAMIC = "fp8_dynamic"
PRECISION_BF16 = "bf16"

# Backend labels: the packing path that was configured and verified as available. ``mslk_accelerated`` states
# that MSLK-backed packing was requested and MSLK is importable and new enough; it does not by itself claim that
# every matmul executed on the accelerated kernel, which is what ``acceleration_status`` qualifies below.
BACKEND_MSLK_ACCELERATED = "mslk_accelerated"
BACKEND_REFERENCE_UNACCELERATED = "reference_unaccelerated"
BACKEND_WEIGHT_ONLY = "weight_only"
BACKEND_DISABLED = "disabled"

# Whether the configured backend actually accelerates execution. ``M`` is the runtime token count of each matmul,
# is not known at quantization time, and cannot be enforced afterwards without either being traced away by
# ``torch.compile`` or being bypassed by the quantized module swap. Accelerated packing is therefore reported as
# conditional and unverified rather than as an accomplished fact.
ACCELERATION_CONDITIONAL = "conditional_unverified"
ACCELERATION_UNACCELERATED = "unaccelerated"
ACCELERATION_NOT_APPLICABLE = "not_applicable"

# Blackwell compute capabilities accepted by policy. Acceptance is a capability policy decision, not a
# qualification statement; no architecture in this list has been qualified by this module, and SM103 in
# particular remains unverified. Callers holding real evidence inject it through
# ``CapabilityFacts.qualified_compute_capabilities`` instead of this module hard-coding a favoured family.
SUPPORTED_COMPUTE_CAPABILITIES: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (11, 0), (12, 0), (12, 1))
QUALIFIED_COMPUTE_CAPABILITIES: Tuple[Tuple[int, int], ...] = ()
ARCHITECTURE_BY_COMPUTE_CAPABILITY: Dict[Tuple[int, int], str] = {
    (10, 0): "sm100",
    (10, 3): "sm103",
    (11, 0): "sm110",
    (12, 0): "sm120",
    (12, 1): "sm121",
}

QUALIFICATION_TESTED = "tested_on_this_architecture"
QUALIFICATION_POLICY_ONLY = "policy_accepted_untested"

MINIMUM_MSLK_VERSION: Tuple[int, int] = (1, 2)

# TorchAO entry points, addressed as ``module:attribute`` so that availability can be probed and mocked.
TORCHAO_QUANTIZE_API = "torchao.quantization:quantize_"
TORCHAO_NVFP4_DYNAMIC_CONFIG_API = "torchao.prototype.mx_formats:NVFP4DynamicActivationNVFP4WeightConfig"
TORCHAO_NVFP4_WEIGHT_ONLY_CONFIG_API = "torchao.prototype.mx_formats:NVFP4WeightOnlyConfig"
TORCHAO_NVFP4_OBSERVED_LINEAR_API = "torchao.prototype.mx_formats:NVFP4ObservedLinear"
TORCHAO_FP8_CONFIG_API = "torchao.quantization:Float8DynamicActivationFloat8WeightConfig"

REQUIRED_TORCH_DTYPES: Tuple[str, ...] = ("float4_e2m1fn_x2", "float8_e4m3fn")

# TorchAO documents these shape constraints for accelerated NVFP4 dynamic packing. ``K`` is a static weight
# dimension and is enforced here; ``M`` is the runtime token count and can only be reported as a requirement.
ACCELERATED_PACKING_M_MULTIPLE = 128
ACCELERATED_PACKING_K_MULTIPLE = 64

CALIBRATION_SCHEMA_VERSION = 1

# Schema of the optional BF16 restoration file. It is the only supported way to keep part of a quantized target
# set in BF16, and it is deliberately narrow: an exact list of canonical target FQNs, nothing else. The file is
# read strictly (no unknown keys, no duplicates, no unknown FQNs), because a silently ignored entry would make a
# family/layer sensitivity experiment report the wrong precision assignment. Only the two recipes that quantize
# all four families uniformly accept it, so the restored set is exactly what the file names: ``nvfp4_weight_only``
# for a storage/accuracy comparator and ``nvfp4_all`` for a W4A4 sensitivity experiment.
BF16_OVERRIDE_SCHEMA_VERSION = 1
BF16_OVERRIDE_FQNS_FIELD = "bf16_fqns"
BF16_OVERRIDE_KEYS: frozenset = frozenset({"version", BF16_OVERRIDE_FQNS_FIELD})
BF16_OVERRIDE_RECIPES: Tuple[str, ...] = ("nvfp4_all", "nvfp4_weight_only")

# Optional schema-v1 field carrying one activation maximum per hook invocation, in invocation order. Runtime
# loading ignores it, so a file that carries it stays consumable by :func:`load_calibration`; the merger uses it
# to compute per-source percentiles instead of a single global maximum.
CALIBRATION_SAMPLES_FIELD = "activation_amax_samples"

# Optional schema-v1 field counting, per module, the hook invocations in which at least one non-finite activation
# element was filtered out. A shard that saw any non-finite activation is not a healthy statistics source, so the
# merger rejects it instead of silently merging the maxima of the finite remainder.
CALIBRATION_NONFINITE_FIELD = "activation_nonfinite_counts"

# Optional schema-v1 field naming the checkpoint the statistics were collected on, as a 64-character lowercase
# hexadecimal SHA-256. Activation scales are only valid for the exact weights that produced them, so the merger
# refuses any input whose declared checkpoint differs from the one the merge is finalized against.
CALIBRATION_CHECKPOINT_FIELD = "checkpoint_sha256"

# Provenance markers for an input that predates the fields above and therefore cannot testify about itself.
CALIBRATION_PROVENANCE_CLEAN = "clean"
CALIBRATION_PROVENANCE_UNKNOWN_LEGACY = "unknown_legacy"

# Merge method recorded in the provenance of a merged artifact: per group and FQN a conservative nearest-rank
# percentile, then the maximum across groups (so a small domain is not averaged away by a high-volume one),
# then a baked-in headroom multiplier.
CALIBRATION_MERGE_METHOD = "source_balanced_group_max_percentile"
CALIBRATION_MERGE_METHOD_VERSION = 1


@dataclass
class SortformerQuantizationConfig:
    """User-facing quantization options for Sortformer evaluation."""

    recipe: str = "disabled"
    scale_mode: str = "dynamic"
    calibration_path: Optional[str] = None
    calibration_output: Optional[str] = None
    scale_margin: float = 1.0
    accelerated_packing: bool = True
    allow_reference_kernels: bool = False
    overwrite_calibration: bool = False
    # Opt-in: carry the calibrated global-scale product inside the two _scaled_mm block-scale operands instead of
    # rescaling the GEMM output. Disabled by default, so the ordinary TorchAO execution path is untouched.
    fold_global_scales: bool = False
    fold_activation_exponent: int = -10
    # Opt-in: fuse the LayerNorm/GELU producers into the NVFP4 activation packs of the complete transformer
    # blocks. Disabled by default, so the ordinary TorchAO execution path is untouched.
    fuse_producer_packing: bool = False
    # Opt-in: JSON file restoring an explicit proper subset of the quantization targets to BF16. Only meaningful
    # on top of the recipes in :data:`BF16_OVERRIDE_RECIPES`; every unlisted target keeps the recipe's precision.
    bf16_override_path: Optional[str] = None
    # Opt-in: how the NVFP4 weight block scales are chosen. ``'amax'`` is TorchAO's ordinary rule and the default;
    # ``'mse'`` repacks every converted NVFP4 weight with the exhaustive per-block E4M3 search; ``'local_hessian'``
    # runs the same search against the activation-weighted diagonal objective of a strict offline artifact;
    # ``'four_over_six'`` repacks with ModelOpt's two-candidate M=6/M=4 comparison and its 256-normalized weight
    # global scale.
    weight_scale_method: str = WEIGHT_SCALE_METHOD_AMAX
    # Required by, and only accepted with, ``weight_scale_method='local_hessian'``: the diagonal-Hessian artifact
    # holding one second-moment vector per target FQN.
    weight_scale_hessian_path: Optional[str] = None
    # Required by, and only accepted with, ``weight_scale_method='awq_clip'``: the AWQ-clip artifact holding one
    # uint8 clipping-ratio code per output row and 16-weight input block of every target FQN.
    weight_scale_awq_clip_path: Optional[str] = None
    # Required by, and only accepted with, ``weight_scale_method='gptq'``: the GPTQ artifact holding the final
    # packed FP4 payload of every target FQN, bound to that FQN's exact ordinary-template scale buffer.
    weight_scale_gptq_path: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Whether a quantized execution recipe was requested."""
        return self.recipe != "disabled"

    @property
    def uses_activation_quantization(self) -> bool:
        """Whether the recipe quantizes activations to NVFP4 (W4A4)."""
        return self.recipe in ("nvfp4_all", "nvfp4_qkv_only", "nvfp4_qkv_fp8_rest")

    @property
    def has_bf16_override(self) -> bool:
        """Whether a BF16 restoration file was requested."""
        return self.bf16_override_path is not None

    @property
    def uses_mse_weight_scales(self) -> bool:
        """Whether the NVFP4 weight block scales are re-selected by the exhaustive per-block MSE search."""
        return self.weight_scale_method == WEIGHT_SCALE_METHOD_MSE

    @property
    def uses_local_hessian_weight_scales(self) -> bool:
        """Whether the NVFP4 weight block scales are re-selected by the activation-weighted diagonal search."""
        return self.weight_scale_method == WEIGHT_SCALE_METHOD_LOCAL_HESSIAN

    @property
    def uses_four_over_six_weight_scales(self) -> bool:
        """Whether the NVFP4 weight block scales are re-selected by the ModelOpt Four-Over-Six comparison."""
        return self.weight_scale_method == WEIGHT_SCALE_METHOD_FOUR_OVER_SIX

    @property
    def uses_awq_clip_weight_scales(self) -> bool:
        """Whether the NVFP4 weight block scales come from an offline AWQ-clip artifact's ratio codes."""
        return self.weight_scale_method == WEIGHT_SCALE_METHOD_AWQ_CLIP

    @property
    def uses_gptq_weight_scales(self) -> bool:
        """Whether the NVFP4 weight payloads come from an offline GPTQ artifact, under unchanged template scales."""
        return self.weight_scale_method == WEIGHT_SCALE_METHOD_GPTQ

    @property
    def uses_searched_weight_scales(self) -> bool:
        """Whether any per-block selection runs, i.e. whether conversion happens one FQN at a time with a repack."""
        return (
            self.uses_mse_weight_scales
            or self.uses_local_hessian_weight_scales
            or self.uses_four_over_six_weight_scales
            or self.uses_awq_clip_weight_scales
            or self.uses_gptq_weight_scales
        )

    def validate(self) -> None:
        """
        Validate option values and their combinations.

        Raises:
            ValueError: If any option is unknown, out of range, or combined with an incompatible option.
        """
        if self.recipe not in QUANTIZATION_RECIPES:
            raise ValueError(f"quantization recipe must be one of {list(QUANTIZATION_RECIPES)}, got '{self.recipe}'")
        if self.scale_mode not in SCALE_MODES:
            raise ValueError(f"quantization scale_mode must be one of {list(SCALE_MODES)}, got '{self.scale_mode}'")
        if self.weight_scale_method not in WEIGHT_SCALE_METHODS:
            raise ValueError(
                f"quantization weight_scale_method must be one of {list(WEIGHT_SCALE_METHODS)}, got "
                f"'{self.weight_scale_method}'"
            )
        _positive_finite(self.scale_margin, "quantization scale_margin")

        if self.uses_mse_weight_scales:
            # The search repacks already-converted NVFP4 weights, so it needs a recipe that produces some.
            if not self.enabled:
                raise ValueError(
                    "quantization_weight_scale_method='mse' re-selects the block scales of converted NVFP4 "
                    "weights, but recipe='disabled' quantizes nothing. Select an NVFP4 recipe or leave the method "
                    f"at '{WEIGHT_SCALE_METHOD_AMAX}'."
                )
            if self.fold_global_scales:
                # Folding rebases and re-rounds every block scale, so a folded run would no longer measure the
                # searched scales. Keeping the two apart is what makes this an isolated technique.
                raise ValueError(
                    "quantization_weight_scale_method='mse' and quantization_fold_global_scales are mutually "
                    "exclusive: folding re-rounds the searched block scales into a different basis, so the run "
                    "would no longer execute the scales the search selected. Enable only one."
                )

        if self.uses_four_over_six_weight_scales:
            # The repack rewrites already-converted NVFP4 weights, so it needs a recipe that produces some.
            if not self.enabled:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_FOUR_OVER_SIX}' re-selects the block "
                    "scales of converted NVFP4 weights, but recipe='disabled' quantizes nothing. Select an NVFP4 "
                    f"recipe or leave the method at '{WEIGHT_SCALE_METHOD_AMAX}'."
                )
            if self.fold_global_scales:
                # Four-Over-Six already renormalizes the weight global scale against 256; folding would rebase and
                # re-round exactly those block scales, so the run would no longer execute the selected ones.
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_FOUR_OVER_SIX}' and "
                    "quantization_fold_global_scales are mutually exclusive: Four-Over-Six renormalizes the weight "
                    "global scale itself, and folding would re-round the selected block scales into a third basis. "
                    "Enable only one."
                )

        if self.uses_local_hessian_weight_scales:
            # The moments describe the inputs of exactly the 124 W4A4 modules of 'nvfp4_all'; under any other
            # recipe the artifact could not cover the converted set, and a partially covered artifact is refused
            # rather than applied to whatever happens to match.
            if self.recipe not in WEIGHT_SCALE_HESSIAN_RECIPES:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_LOCAL_HESSIAN}' selects the block "
                    "scales of every NVFP4 W4A4 target from its activation second moments and requires one of "
                    f"recipe={list(WEIGHT_SCALE_HESSIAN_RECIPES)}, got recipe='{self.recipe}'."
                )
            if not str(self.weight_scale_hessian_path or "").strip():
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_LOCAL_HESSIAN}' requires "
                    "quantization_weight_scale_hessian_path to point at a diagonal-Hessian artifact JSON; there is "
                    "no default and no fallback to another method."
                )
            path = Path(str(self.weight_scale_hessian_path).strip()).expanduser()
            if not path.is_file() or not os.access(path, os.R_OK):
                raise ValueError(
                    f"quantization_weight_scale_hessian_path {path} is not a readable file. The artifact is read "
                    "before anything is converted, so an unreadable path is a failed run, not a fallback."
                )
            if self.fold_global_scales:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_LOCAL_HESSIAN}' and "
                    "quantization_fold_global_scales are mutually exclusive: folding re-rounds the searched block "
                    "scales into a different basis, so the run would no longer execute the selected scales."
                )
            if self.has_bf16_override:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_LOCAL_HESSIAN}' is bound to the "
                    "complete 'nvfp4_all' target set and cannot be combined with "
                    "quantization_bf16_override_path, which leaves part of that set in BF16."
                )
        elif self.weight_scale_hessian_path is not None:
            # Silently ignoring the path would let a run that meant to be activation weighted report itself as one
            # while executing amax or unweighted-MSE scales.
            raise ValueError(
                "quantization_weight_scale_hessian_path is only used with "
                f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_LOCAL_HESSIAN}', but the method is "
                f"'{self.weight_scale_method}'."
            )

        if self.uses_awq_clip_weight_scales:
            # The codes describe every 16-weight block of exactly the 124 W4A4 modules of 'nvfp4_all', selected
            # against the activations *that recipe* quantizes; no other recipe converts the set they cover.
            if self.recipe not in WEIGHT_SCALE_AWQ_CLIP_RECIPES:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' selects the block scales of "
                    "every NVFP4 W4A4 target from an offline artifact and requires one of "
                    f"recipe={list(WEIGHT_SCALE_AWQ_CLIP_RECIPES)}, got recipe='{self.recipe}'."
                )
            if self.scale_mode != WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' requires "
                    f"scale_mode='{WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE}', got scale_mode='{self.scale_mode}'; the "
                    "ratio codes were chosen against activations quantized with the calibrated static scales, and "
                    "a dynamic run would execute different ones."
                )
            calibration = str(self.calibration_path or "").strip()
            if not calibration or not _is_readable_file(calibration):
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' requires "
                    "quantization_calibration_path to point at the readable static activation-calibration JSON the "
                    "artifact was built against; the artifact is bound to that file's exact bytes."
                )
            if float(self.scale_margin) != float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN):
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' requires "
                    f"quantization_scale_margin={WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN}, got {self.scale_margin!r}; "
                    "the calibration artifact already bakes its headroom in, and any other margin would quantize "
                    "the activations differently from the ones the codes were selected against."
                )
            if not str(self.weight_scale_awq_clip_path or "").strip():
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' requires "
                    "quantization_weight_scale_awq_clip_path to point at an AWQ-clip artifact JSON; there is no "
                    "default and no fallback to another method."
                )
            awq_path = Path(str(self.weight_scale_awq_clip_path).strip()).expanduser()
            if not _is_readable_file(awq_path):
                raise ValueError(
                    f"quantization_weight_scale_awq_clip_path {awq_path} is not a readable file. The artifact is "
                    "read before anything is converted, so an unreadable path is a failed run, not a fallback."
                )
            if self.fold_global_scales:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' and "
                    "quantization_fold_global_scales are mutually exclusive: folding re-rounds the selected block "
                    "scales into a different basis, so the run would no longer execute the selected scales."
                )
            if self.has_bf16_override:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' is bound to the complete "
                    "'nvfp4_all' target set and cannot be combined with quantization_bf16_override_path, which "
                    "leaves part of that set in BF16."
                )
        elif self.weight_scale_awq_clip_path is not None:
            # Silently ignoring the path would let a run that meant to execute the AWQ-clip codes report itself as
            # one while executing amax, MSE, Four-Over-Six or activation-weighted scales.
            raise ValueError(
                "quantization_weight_scale_awq_clip_path is only used with "
                f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}', but the method is "
                f"'{self.weight_scale_method}'."
            )

        if self.uses_gptq_weight_scales:
            # The payloads describe every 16-weight block of exactly the 124 W4A4 modules of 'nvfp4_all', selected
            # against the activations *that recipe* quantizes; no other recipe converts the set they cover.
            if self.recipe not in WEIGHT_SCALE_GPTQ_RECIPES:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' replaces the packed payload of "
                    "every NVFP4 W4A4 target from an offline artifact and requires one of "
                    f"recipe={list(WEIGHT_SCALE_GPTQ_RECIPES)}, got recipe='{self.recipe}'."
                )
            if self.scale_mode != WEIGHT_SCALE_GPTQ_SCALE_MODE:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' requires "
                    f"scale_mode='{WEIGHT_SCALE_GPTQ_SCALE_MODE}', got scale_mode='{self.scale_mode}'; the payloads "
                    "were selected under Hessians formed from activations quantized with the calibrated static "
                    "scales, and a dynamic run would execute different ones."
                )
            calibration = str(self.calibration_path or "").strip()
            if not calibration or not _is_readable_file(calibration):
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' requires "
                    "quantization_calibration_path to point at the readable static activation-calibration JSON the "
                    "artifact was built against; the artifact is bound to that file's exact bytes."
                )
            if float(self.scale_margin) != float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN):
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' requires "
                    f"quantization_scale_margin={WEIGHT_SCALE_GPTQ_SCALE_MARGIN}, got {self.scale_margin!r}; the "
                    "calibration artifact already bakes its headroom in, and any other margin would quantize the "
                    "activations differently from the ones the Hessians were formed from."
                )
            if not str(self.weight_scale_gptq_path or "").strip():
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' requires "
                    "quantization_weight_scale_gptq_path to point at a GPTQ artifact JSON; there is no default and "
                    "no fallback to another method."
                )
            gptq_path = Path(str(self.weight_scale_gptq_path).strip()).expanduser()
            if not _is_readable_file(gptq_path):
                raise ValueError(
                    f"quantization_weight_scale_gptq_path {gptq_path} is not a readable file. The artifact is read "
                    "before anything is converted, so an unreadable path is a failed run, not a fallback."
                )
            if self.fold_global_scales:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' and "
                    "quantization_fold_global_scales are mutually exclusive: folding rebases the very block scales "
                    "the payload was written under, so the run would no longer execute the selected payload."
                )
            if self.has_bf16_override:
                raise ValueError(
                    f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' is bound to the complete "
                    "'nvfp4_all' target set and cannot be combined with quantization_bf16_override_path, which "
                    "leaves part of that set in BF16."
                )
        elif self.weight_scale_gptq_path is not None:
            # Silently ignoring the path would let a run that meant to execute the GPTQ payload report itself as one
            # while executing amax, MSE, Four-Over-Six, AWQ-clip or activation-weighted scales.
            raise ValueError(
                "quantization_weight_scale_gptq_path is only used with "
                f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}', but the method is "
                f"'{self.weight_scale_method}'."
            )

        if self.enabled and self.calibration_output is not None:
            raise ValueError(
                "Activation calibration collection and quantized execution cannot happen in one invocation. "
                f"Run calibration with recipe='disabled', then run recipe='{self.recipe}' with "
                "quantization_calibration_path pointing at the produced JSON file."
            )
        if not self.enabled and self.calibration_path is not None:
            raise ValueError(
                "quantization_calibration_path was given but the quantization recipe is 'disabled'; "
                "select a W4A4 recipe with scale_mode='static' to consume a calibration file."
            )
        if self.scale_mode == "static":
            if not self.uses_activation_quantization:
                raise ValueError(
                    f"scale_mode='static' requires a W4A4 recipe, but recipe='{self.recipe}' does not "
                    "quantize activations."
                )
            if not self.calibration_path:
                raise ValueError("scale_mode='static' requires quantization_calibration_path to a calibration JSON.")
        elif self.calibration_path is not None:
            raise ValueError("quantization_calibration_path is only used with scale_mode='static'.")

        if self.uses_activation_quantization and not self.accelerated_packing and not self.allow_reference_kernels:
            raise ValueError(
                "quantization_accelerated_packing=False runs unaccelerated reference NVFP4 kernels. "
                "Set quantization_allow_reference_kernels=True to acknowledge this explicitly."
            )

        if self.fold_global_scales:
            if not self.uses_activation_quantization:
                raise ValueError(
                    "quantization_fold_global_scales folds the calibrated activation/weight global-scale product "
                    f"into the NVFP4 block scales, but recipe='{self.recipe}' does not quantize activations."
                )
            if self.scale_mode != "static":
                raise ValueError(
                    "quantization_fold_global_scales requires scale_mode='static'; a dynamic activation scale is "
                    "not known at conversion time and cannot be folded into the weight block scales."
                )
            if not self.accelerated_packing:
                raise ValueError(
                    "quantization_fold_global_scales requires quantization_accelerated_packing=True; it targets "
                    "the native Blackwell _scaled_mm path only."
                )
            if self.allow_reference_kernels:
                raise ValueError(
                    "quantization_fold_global_scales and quantization_allow_reference_kernels request different "
                    "execution paths; folding replaces the reference NVFP4 kernels with the native GEMM."
                )
            validate_fold_exponent(self.fold_activation_exponent)

        if self.has_bf16_override:
            # BF16 restoration only subtracts from a uniformly quantized target set; it has no meaning for a recipe
            # that already leaves families in BF16 or splits them across precisions, because there the file could
            # not express which precision an unlisted module ends up with.
            if self.recipe not in BF16_OVERRIDE_RECIPES:
                raise ValueError(
                    "quantization_bf16_override_path restores a subset of the NVFP4 targets to BF16 and requires "
                    f"one of recipe={list(BF16_OVERRIDE_RECIPES)}, got recipe='{self.recipe}'."
                )
            if not str(self.bf16_override_path).strip():
                raise ValueError(
                    "quantization_bf16_override_path is empty; give the path of a JSON file "
                    f"{{'version': {BF16_OVERRIDE_SCHEMA_VERSION}, '{BF16_OVERRIDE_FQNS_FIELD}': [...]}} or leave "
                    "the option unset."
                )

        if self.fuse_producer_packing:
            # Producer fusion rewrites complete transformer blocks, so every one of the four target families must
            # be converted to static W4A4 on the native MSLK path; each incompatible option is named separately.
            if self.recipe != "nvfp4_all":
                raise ValueError(
                    "quantization_fuse_producer_packing fuses complete transformer blocks and requires "
                    f"recipe='nvfp4_all', got recipe='{self.recipe}'."
                )
            if self.scale_mode != "static":
                raise ValueError(
                    "quantization_fuse_producer_packing requires scale_mode='static'; the fused packs consume the "
                    "calibrated activation scale, which a dynamic recipe only knows at runtime."
                )
            if not self.accelerated_packing:
                raise ValueError(
                    "quantization_fuse_producer_packing requires quantization_accelerated_packing=True; it targets "
                    "the native Blackwell _scaled_mm path only."
                )
            if self.allow_reference_kernels:
                raise ValueError(
                    "quantization_fuse_producer_packing and quantization_allow_reference_kernels request different "
                    "execution paths; the fused packs replace the reference NVFP4 kernels with Triton kernels."
                )
            if self.fold_global_scales:
                raise ValueError(
                    "quantization_fuse_producer_packing and quantization_fold_global_scales are mutually "
                    "exclusive: folding rebases the block scales the fused packs produce. Enable only one."
                )
            # A bf16_override is compatible with fusion as long as it leaves the three fused consumers
            # (attn.w_qkv, ffn.net.0, ffn.net.3) quantized: those are the only linears a fused block packs into
            # and dispatches itself. attn.out_proj is reached solely through attn.forward_from_qkv, so restoring
            # it changes nothing fusion depends on. Restoring a genuine consumer still has no pack to fuse into
            # and is rejected -- but that needs the override file's FQN list, which validate() cannot see because
            # only the path is known here, so it is enforced in quantize_sortformer_model immediately after
            # load_bf16_override, before the first module is touched.
            pass


def quantization_config_from_eval_cfg(cfg) -> SortformerQuantizationConfig:
    """
    Build and validate a quantization config from the evaluator's ``DiarizationConfig``.

    Args:
        cfg (DiarizationConfig): Evaluator configuration carrying the ``quantization_*`` fields.

    Returns:
        config (SortformerQuantizationConfig): Validated quantization options.
    """
    config = SortformerQuantizationConfig(
        recipe=str(cfg.quantization_recipe),
        scale_mode=str(cfg.quantization_scale_mode),
        calibration_path=cfg.quantization_calibration_path,
        calibration_output=cfg.quantization_calibration_output,
        scale_margin=float(cfg.quantization_scale_margin),
        accelerated_packing=bool(cfg.quantization_accelerated_packing),
        allow_reference_kernels=bool(cfg.quantization_allow_reference_kernels),
        overwrite_calibration=bool(cfg.quantization_overwrite_calibration),
        fold_global_scales=bool(cfg.quantization_fold_global_scales),
        # Deliberately not coerced with int(): validate() rejects a non-integer exponent instead of rounding it.
        fold_activation_exponent=cfg.quantization_fold_activation_exponent,
        fuse_producer_packing=bool(cfg.quantization_fuse_producer_packing),
        bf16_override_path=cfg.quantization_bf16_override_path,
        weight_scale_method=str(cfg.quantization_weight_scale_method),
        weight_scale_hessian_path=cfg.quantization_weight_scale_hessian_path,
        weight_scale_awq_clip_path=cfg.quantization_weight_scale_awq_clip_path,
        weight_scale_gptq_path=cfg.quantization_weight_scale_gptq_path,
    )
    config.validate()
    return config


@dataclass
class CapabilityFacts:
    """Injectable snapshot of the runtime facts that gate NVFP4 execution."""

    device_type: str
    compute_capability: Optional[Tuple[int, int]] = None
    torch_version: Optional[str] = None
    torchao_version: Optional[str] = None
    mslk_version: Optional[str] = None
    available_apis: Tuple[str, ...] = ()
    available_dtypes: Tuple[str, ...] = ()
    # Compute capabilities for which the caller has actual test evidence. Empty by default: policy acceptance is
    # not qualification.
    qualified_compute_capabilities: Tuple[Tuple[int, int], ...] = QUALIFIED_COMPUTE_CAPABILITIES

    @property
    def architecture(self) -> Optional[str]:
        """Short architecture name for the compute capability, if it is a known one."""
        if self.compute_capability is None:
            return None
        return ARCHITECTURE_BY_COMPUTE_CAPABILITY.get(tuple(self.compute_capability))

    @property
    def qualification(self) -> str:
        """Whether this compute capability is merely accepted by policy or actually qualified by evidence."""
        if self.compute_capability is None:
            return QUALIFICATION_POLICY_ONLY
        qualified = {tuple(capability) for capability in self.qualified_compute_capabilities}
        return QUALIFICATION_TESTED if tuple(self.compute_capability) in qualified else QUALIFICATION_POLICY_ONLY


@dataclass
class QuantizationSelection:
    """Result of matching the exact target families against a model."""

    precision_by_fqn: Dict[str, str] = field(default_factory=dict)
    fqns_by_suffix: Dict[str, List[str]] = field(default_factory=dict)

    def fqns_for_precision(self, precision: str) -> List[str]:
        """Sorted FQNs assigned to the given precision label."""
        return sorted(fqn for fqn, value in self.precision_by_fqn.items() if value == precision)

    @property
    def counts_by_precision(self) -> Dict[str, int]:
        """Number of selected modules per precision label."""
        counts: Dict[str, int] = {}
        for precision in self.precision_by_fqn.values():
            counts[precision] = counts.get(precision, 0) + 1
        return dict(sorted(counts.items()))


def select_quantization_targets(
    model: torch.nn.Module, recipe: str, bf16_fqns: Optional[Sequence[str]] = None
) -> QuantizationSelection:
    """
    Match the exact Sortformer target families and assign a precision to each matched module.

    Args:
        model (torch.nn.Module): Model to search. Only anchored suffix matches of
            :data:`QUANTIZATION_TARGET_SUFFIXES` are considered.
        recipe (str): One of :data:`QUANTIZATION_RECIPES`.
        bf16_fqns (Optional[Sequence[str]]): Exact matched target FQNs to restore to :data:`PRECISION_BF16`
            instead of the recipe's precision, as validated by :func:`load_bf16_override`. Every unlisted target
            keeps the precision the recipe assigns it.

    Returns:
        selection (QuantizationSelection): FQN -> precision map and the FQNs grouped by target family.

    Raises:
        ValueError: If the recipe is unknown, an expected target family is missing from the model, a matched
            object is not a ``torch.nn.Linear``, or a requested BF16 FQN is not a matched target.
    """
    if recipe not in QUANTIZATION_RECIPES:
        raise ValueError(f"quantization recipe must be one of {list(QUANTIZATION_RECIPES)}, got '{recipe}'")

    fqns_by_suffix = _match_target_modules(model)
    precision_by_suffix = _precision_by_suffix(recipe)
    precision_by_fqn = {}
    for suffix, fqns in fqns_by_suffix.items():
        for fqn in fqns:
            precision_by_fqn[fqn] = precision_by_suffix[suffix]

    if bf16_fqns:
        unknown = sorted(set(bf16_fqns) - set(precision_by_fqn))
        if unknown:
            raise ValueError(
                f"BF16 override lists {len(unknown)} name(s) that are not Sortformer quantization targets in this "
                f"model: {unknown[:8]}{'...' if len(unknown) > 8 else ''}."
            )
        for fqn in bf16_fqns:
            precision_by_fqn[fqn] = PRECISION_BF16
    return QuantizationSelection(precision_by_fqn=precision_by_fqn, fqns_by_suffix=fqns_by_suffix)


def collect_capability_facts(device: torch.device) -> CapabilityFacts:
    """
    Gather the runtime facts required to decide whether NVFP4 execution is permitted.

    TorchAO and MSLK are imported here, so this must only be called on an enabled quantization path.

    Args:
        device (torch.device): Device the model will run on.

    Returns:
        facts (CapabilityFacts): Device, compute capability, dependency versions, and available APIs.
    """
    device = torch.device(device)
    compute_capability = None
    if device.type == "cuda" and torch.cuda.is_available():
        compute_capability = tuple(torch.cuda.get_device_capability(device))

    available_apis = tuple(api for api in _ALL_TORCHAO_APIS if _resolve_api(api) is not None)
    available_dtypes = tuple(name for name in REQUIRED_TORCH_DTYPES if getattr(torch, name, None) is not None)
    return CapabilityFacts(
        device_type=device.type,
        compute_capability=compute_capability,
        torch_version=torch.__version__,
        torchao_version=_module_version("torchao"),
        mslk_version=_module_version("mslk"),
        available_apis=available_apis,
        available_dtypes=available_dtypes,
    )


def check_nvfp4_capability(config: SortformerQuantizationConfig, facts: CapabilityFacts) -> str:
    """
    Verify that the requested recipe can run as requested, and resolve the execution backend.

    Args:
        config (SortformerQuantizationConfig): Validated quantization options.
        facts (CapabilityFacts): Runtime facts, either collected or injected for testing.

    Returns:
        backend (str): One of :data:`BACKEND_MSLK_ACCELERATED`, :data:`BACKEND_REFERENCE_UNACCELERATED`,
            :data:`BACKEND_WEIGHT_ONLY`, or :data:`BACKEND_DISABLED`.

    Raises:
        RuntimeError: If the device, compute capability, TorchAO APIs, torch dtypes, or MSLK version cannot
            support the requested recipe. The request is never downgraded to make it succeed.
    """
    if not config.enabled:
        return BACKEND_DISABLED

    if facts.device_type != "cuda":
        raise RuntimeError(
            f"NVFP4 quantization requires a CUDA device, but the model is on '{facts.device_type}'. "
            "Set recipe='disabled' for CPU runs."
        )
    capability = None if facts.compute_capability is None else tuple(facts.compute_capability)
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            f"NVFP4 quantization is only accepted on compute capabilities "
            f"{[list(cc) for cc in SUPPORTED_COMPUTE_CAPABILITIES]}, but this device reports {capability}."
        )

    missing_dtypes = [name for name in REQUIRED_TORCH_DTYPES if name not in facts.available_dtypes]
    if missing_dtypes:
        raise RuntimeError(
            f"torch {facts.torch_version} does not expose the dtypes required for NVFP4: {missing_dtypes}."
        )
    missing_apis = [api for api in _required_apis(config) if api not in facts.available_apis]
    if missing_apis:
        raise RuntimeError(
            f"TorchAO {facts.torchao_version} does not provide the APIs required by recipe '{config.recipe}' "
            f"(scale_mode='{config.scale_mode}'): {missing_apis}. Install a TorchAO build that exports them."
        )

    if config.recipe == "nvfp4_weight_only":
        return BACKEND_WEIGHT_ONLY
    if not config.accelerated_packing:
        # ``validate()`` already required an explicit acknowledgement to reach this point.
        logging.warning(
            "NVFP4 is running with UNACCELERATED reference kernels (quantization_accelerated_packing=False). "
            "Numerics are representative; throughput is not."
        )
        return BACKEND_REFERENCE_UNACCELERATED

    mslk_version = _parse_version(facts.mslk_version)
    if mslk_version is None:
        raise RuntimeError(
            "Accelerated NVFP4 packing requires an importable MSLK "
            f">= {'.'.join(str(part) for part in MINIMUM_MSLK_VERSION)}, but MSLK is not available. "
            "Install MSLK, or set quantization_accelerated_packing=False together with "
            "quantization_allow_reference_kernels=True to run unaccelerated reference kernels."
        )
    if mslk_version[: len(MINIMUM_MSLK_VERSION)] < MINIMUM_MSLK_VERSION:
        raise RuntimeError(
            f"Accelerated NVFP4 packing requires MSLK >= {'.'.join(str(part) for part in MINIMUM_MSLK_VERSION)}, "
            f"but MSLK {facts.mslk_version} is installed."
        )
    return BACKEND_MSLK_ACCELERATED


def load_calibration(
    calibration_path: str,
    selection: QuantizationSelection,
    scale_margin: float = 1.0,
) -> Dict[str, Any]:
    """
    Load and validate a static-activation calibration file against the current selection.

    Args:
        calibration_path (str): Path to a calibration JSON produced by :class:`ActivationAmaxCollector`.
        selection (QuantizationSelection): Selection whose NVFP4 W4A4 FQNs must be covered. Coverage is required
            only for the FQNs that remain W4A4, so a common calibration artifact stays usable when a BF16 override
            restored part of the target set; its entries are reported in ``unused_fqns``.
        scale_margin (float): Positive multiplier applied to every activation amax.

    Returns:
        calibration (Dict[str, Any]): ``activation_amax`` (margin applied, sorted), ``raw_activation_amax``,
            ``unused_fqns``, ``version``, ``scale_margin``, and ``path``.

    Raises:
        ValueError: If the file is malformed, versioned incorrectly, contains duplicate, non-finite, or
            non-positive entries, references FQNs that are not target modules, or misses a selected FQN.
    """
    margin = _positive_finite(scale_margin, "scale_margin")

    path = Path(calibration_path).expanduser()
    with open(path, "r", encoding="utf-8") as calibration_file:
        payload = json.load(calibration_file, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file {path} must contain a JSON object")
    version = payload.get("version")
    if version != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Calibration file {path} has version {version!r}, but version {CALIBRATION_SCHEMA_VERSION} is required."
        )
    raw_amax = payload.get("activation_amax")
    if not isinstance(raw_amax, dict) or not raw_amax:
        raise ValueError(f"Calibration file {path} must contain a non-empty 'activation_amax' object")

    known_fqns = {fqn for fqns in selection.fqns_by_suffix.values() for fqn in fqns}
    unknown = sorted(set(raw_amax) - known_fqns)
    if unknown:
        raise ValueError(
            f"Calibration file {path} contains entries that are not Sortformer quantization targets in this "
            f"model: {unknown}. Re-run calibration against the same checkpoint."
        )
    for fqn, value in sorted(raw_amax.items()):
        _positive_finite(value, f"Calibration entry '{fqn}'")

    required_fqns = set(selection.fqns_for_precision(PRECISION_NVFP4_W4A4))
    missing = sorted(required_fqns - set(raw_amax))
    if missing:
        raise ValueError(
            f"Calibration file {path} is missing activation amax values for {len(missing)} selected NVFP4 "
            f"modules: {missing[:8]}{'...' if len(missing) > 8 else ''}."
        )

    raw_normalized = {fqn: float(raw_amax[fqn]) for fqn in sorted(raw_amax)}
    return {
        "path": str(path),
        "version": int(version),
        "scale_margin": margin,
        "raw_activation_amax": raw_normalized,
        "activation_amax": {fqn: float(raw_normalized[fqn]) * margin for fqn in sorted(required_fqns)},
        "unused_fqns": sorted(set(raw_normalized) - required_fqns),
    }


def load_bf16_override(override_path: str, selection: QuantizationSelection) -> Dict[str, Any]:
    """
    Load and strictly validate a BF16 restoration file against the live matched target set.

    The file is UTF-8 JSON with exactly the schema
    ``{"version": 1, "bf16_fqns": ["exact.model.fqn", ...]}``. Every entry must be an exact canonical target FQN of
    *this* model, the list must be non-empty, and it must not cover every target: an override that restores all of
    them is a BF16 run, which ``recipe='disabled'`` already expresses honestly. Nothing in the file is ever
    ignored, so an entry that this model does not have is an error rather than a silently smaller experiment.

    Args:
        override_path (str): Path to the override JSON file.
        selection (QuantizationSelection): Selection whose matched target FQNs the file is validated against.

    Returns:
        override (Dict[str, Any]): ``enabled``, the expanded ``path``, the ``sha256`` of the exact file bytes,
            the restored ``count``, and the sorted ``fqns``.

    Raises:
        ValueError: If the file is not readable UTF-8 JSON, is not an object, has duplicate keys, unknown or
            missing top-level keys, a different version, a non-list FQN field, a non-string or empty entry,
            duplicate entries, an entry that is not a matched target, an empty list, or the complete target set.
        OSError: If the file cannot be read.
    """
    path = Path(override_path).expanduser()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_bf16_override_keys)
    except UnicodeDecodeError as error:
        raise ValueError(f"BF16 override file {path} must be UTF-8 encoded: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"BF16 override file {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"BF16 override file {path} must contain a JSON object with keys {sorted(BF16_OVERRIDE_KEYS)}, got "
            f"{type(payload).__name__}"
        )

    unknown_keys = sorted(set(payload) - BF16_OVERRIDE_KEYS)
    missing_keys = sorted(BF16_OVERRIDE_KEYS - set(payload))
    if unknown_keys or missing_keys:
        raise ValueError(
            f"BF16 override file {path} must contain exactly the keys {sorted(BF16_OVERRIDE_KEYS)}; unknown keys "
            f"{unknown_keys}, missing keys {missing_keys}."
        )
    version = payload["version"]
    if isinstance(version, bool) or version != BF16_OVERRIDE_SCHEMA_VERSION:
        raise ValueError(
            f"BF16 override file {path} has version {version!r}, but version {BF16_OVERRIDE_SCHEMA_VERSION} is "
            "required."
        )

    raw_fqns = payload[BF16_OVERRIDE_FQNS_FIELD]
    if not isinstance(raw_fqns, list):
        raise ValueError(
            f"BF16 override file {path} must give '{BF16_OVERRIDE_FQNS_FIELD}' as a list of module FQN strings, "
            f"got {type(raw_fqns).__name__}"
        )
    for index, entry in enumerate(raw_fqns):
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"Entry {index} of '{BF16_OVERRIDE_FQNS_FIELD}' in {path} must be a non-empty module FQN string, "
                f"got {entry!r}"
            )
    duplicates = sorted({entry for entry in raw_fqns if raw_fqns.count(entry) > 1})
    if duplicates:
        raise ValueError(
            f"BF16 override file {path} lists {len(duplicates)} FQN(s) more than once: {duplicates[:8]}"
            f"{'...' if len(duplicates) > 8 else ''}. Every module may be restored at most once."
        )
    if not raw_fqns:
        raise ValueError(
            f"BF16 override file {path} restores no module at all; omit quantization_bf16_override_path to run "
            f"one of recipe={list(BF16_OVERRIDE_RECIPES)} unchanged."
        )

    target_fqns = set(selection.precision_by_fqn)
    unknown_fqns = sorted(set(raw_fqns) - target_fqns)
    if unknown_fqns:
        raise ValueError(
            f"BF16 override file {path} lists {len(unknown_fqns)} name(s) that are not Sortformer quantization "
            f"targets in this model: {unknown_fqns[:8]}{'...' if len(unknown_fqns) > 8 else ''}. Entries must be "
            f"exact FQNs of the {len(target_fqns)} matched target modules."
        )
    if set(raw_fqns) == target_fqns:
        raise ValueError(
            f"BF16 override file {path} restores all {len(target_fqns)} target modules to BF16, which leaves "
            "nothing quantized; run recipe='disabled' instead of a quantized recipe that quantizes nothing."
        )

    return {
        "enabled": True,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(raw_fqns),
        "fqns": sorted(raw_fqns),
    }


def load_diagonal_hessian(
    hessian_path: str, model: torch.nn.Module, selection: QuantizationSelection
) -> Dict[str, Any]:
    """
    Load and strictly validate a diagonal-Hessian artifact against the live model and the current selection.

    The artifact is the only thing standing between an offline statistics run and the weights a quantized run
    executes, so every one of its claims is checked before a single module is converted: the closed top-level and
    provenance key sets, the schema and version, the algorithm identity and its exact damping, the canonical
    weight-digest method, the exact coverage of the selected NVFP4 W4A4 FQNs in canonical form, the width of every
    moment vector against the live ``in_features``, the finiteness, non-negativity and positive mean of every
    vector *as the FP32 vector the search will weight with*, the shape of every source record, the recorded
    :data:`SECTION_DIGEST_METHOD` digests of the moments and of the provenance, the agreement of every aggregate
    field with the sections it summarizes, and finally the digest of every live original weight against the one the
    artifact was built on. Nothing here falls back: a violated claim is a failed run, and every one of these checks
    runs before a single module is converted.

    Args:
        hessian_path (str): Path to the artifact JSON.
        model (torch.nn.Module): Live, still-unconverted model whose weights and ``in_features`` are checked.
        selection (QuantizationSelection): Selection whose NVFP4 W4A4 FQNs the artifact must cover exactly.

    Returns:
        hessian (Dict[str, Any]): ``path``, ``sha256`` of the exact file bytes, ``checkpoint_sha256``, ``damping``,
            ``algorithm``, ``algorithm_version``, the sorted ``fqns``, the per-FQN ``second_moments`` lists, the
            per-FQN ``weight_sha256``, the verified ``moment_sha256`` and ``provenance_sha256``, and the artifact's
            ``provenance``.

    Raises:
        ValueError: For any violated claim above.
        OSError: If the file cannot be read.
    """
    path = Path(hessian_path).expanduser()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_hessian_keys)
    except UnicodeDecodeError as error:
        raise ValueError(f"Diagonal-Hessian artifact {path} must be UTF-8 encoded: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Diagonal-Hessian artifact {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Diagonal-Hessian artifact {path} must contain a JSON object, got {type(payload).__name__}")
    _require_exact_keys(payload, HESSIAN_ARTIFACT_KEYS, f"Diagonal-Hessian artifact {path}", "top-level")

    if payload["schema"] != HESSIAN_SCHEMA:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} declares schema {payload['schema']!r}, but {HESSIAN_SCHEMA!r} is "
            "required."
        )
    if isinstance(payload["version"], bool) or payload["version"] != HESSIAN_SCHEMA_VERSION:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} has version {payload['version']!r}, but version "
            f"{HESSIAN_SCHEMA_VERSION} is required."
        )
    if payload["algorithm"] != WEIGHT_SCALE_HESSIAN_ALGORITHM or (
        isinstance(payload["algorithm_version"], bool)
        or payload["algorithm_version"] != WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION
    ):
        raise ValueError(
            f"Diagonal-Hessian artifact {path} was built for algorithm {payload['algorithm']!r} "
            f"v{payload['algorithm_version']!r}, but this build only consumes "
            f"{WEIGHT_SCALE_HESSIAN_ALGORITHM!r} v{WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION}."
        )
    damping = payload["damping"]
    numeric_damping = not isinstance(damping, bool) and isinstance(damping, (int, float))
    if not numeric_damping or float(damping) != float(WEIGHT_SCALE_HESSIAN_DAMPING):
        raise ValueError(
            f"Diagonal-Hessian artifact {path} records damping {damping!r}, but this build implements exactly "
            f"{WEIGHT_SCALE_HESSIAN_DAMPING}; a different damping is a different algorithm."
        )
    if payload["weight_digest_method"] != WEIGHT_DIGEST_METHOD:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} records the weight-digest method "
            f"{payload['weight_digest_method']!r}, but this build verifies weights with {WEIGHT_DIGEST_METHOD!r}."
        )
    checkpoint_sha256 = _validate_checkpoint_sha256(payload["checkpoint_sha256"], f"'checkpoint_sha256' in {path}")

    required = selection.fqns_for_precision(PRECISION_NVFP4_W4A4)
    provenance = _validate_hessian_provenance(payload["provenance"], required, path)
    modules = dict(model.named_modules())
    second_moments = _validate_hessian_moments(payload["diagonal_hessian"], required, modules, path)
    # The two component digests are recomputed from the sections the file actually carries, so an edited moment or
    # a rewritten source record is refused even when the result is still structurally valid JSON. They are checked
    # after the structural validation so that a malformed section is named for what it is rather than as a digest
    # mismatch, and before anything is converted.
    moment_sha256 = _verify_hessian_section_digest(payload, "moment_sha256", "diagonal_hessian", path)
    provenance_sha256 = _verify_hessian_section_digest(payload, "provenance_sha256", "provenance", path)
    # A digest only proves that nobody edited a section *after* the digest was taken; it says nothing about whether
    # the summary the provenance states is the summary of the data the artifact carries. The aggregate is therefore
    # recomputed from the validated sections and compared, which a fabricated artifact cannot pass by recomputing
    # its own provenance digest.
    _validate_hessian_aggregate(provenance, second_moments, required, path)
    weight_sha256 = _validate_hessian_weight_digests(payload["weight_sha256"], required, modules, path)

    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "algorithm": WEIGHT_SCALE_HESSIAN_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION,
        "damping": float(WEIGHT_SCALE_HESSIAN_DAMPING),
        "fqns": list(required),
        "second_moments": second_moments,
        "weight_sha256": weight_sha256,
        "moment_sha256": moment_sha256,
        "provenance_sha256": provenance_sha256,
        "provenance": provenance,
    }


def load_awq_clip_artifact(
    awq_clip_path: str,
    model: torch.nn.Module,
    selection: QuantizationSelection,
    calibration_path: str,
    scale_margin: float = WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN,
) -> Dict[str, Any]:
    """
    Load and strictly validate an AWQ-clip artifact against the live model and the exact calibration file.

    The artifact is the only thing standing between an offline ratio search and the weights a quantized run
    executes, and the search's own evidence -- the activation rows -- is deliberately absent from it, so every one
    of its claims is checked before a single module is converted: the closed top-level, arithmetic, calibration,
    code, provenance, source, module and aggregate key sets; the schema, version and algorithm identity; every
    arithmetic constant, including the exact ratio list and its order, the block size, the tie rule, the objective,
    the group reduction, the candidate scale rule, the recorded ordinary-template construction, the activation-QDQ
    identity, the FP4/FP8 maxima, the scale floor
    and the ModelOpt reference version and wheel digest; the identity of the configured activation-calibration file,
    recomputed from its actual bytes; the exact coverage of the selected NVFP4 W4A4 FQNs in canonical form; every
    code matrix's shape against the live ``out_features``/``in_features`` and every code's range; the recorded
    :data:`SECTION_DIGEST_METHOD` digests of the codes and of the provenance; the agreement of every per-module and
    aggregate claim with the codes the artifact actually carries; and finally the digest of every live original
    weight against the one the artifact was built on. Nothing here falls back: a violated claim is a failed run.

    Args:
        awq_clip_path (str): Path to the artifact JSON.
        model (torch.nn.Module): Live, still-unconverted model whose weights and shapes are checked.
        selection (QuantizationSelection): Selection whose NVFP4 W4A4 FQNs the artifact must cover exactly.
        calibration_path (str): Path of the static activation-calibration file the run will consume; the artifact
            must be bound to exactly its bytes.
        scale_margin (float): The run's activation scale margin, which must be the one the builder used.

    Returns:
        awq_clip (Dict[str, Any]): ``path``, ``sha256`` of the exact file bytes, ``checkpoint_sha256``,
            ``algorithm``, ``algorithm_version``, ``clip_ratios``, ``block_size``, ``tie_rule``,
            ``template_arithmetic``,
            ``calibration_path``, the verified ``calibration`` identity, the sorted ``fqns``, the decoded per-FQN
            ``ratio_codes`` as raw bytes with their ``code_shapes``, the per-FQN ``ratio_histogram``, the per-FQN
            ``weight_sha256``, the verified ``ratio_code_sha256`` and ``provenance_sha256``, and the ``provenance``.

    Raises:
        ValueError: For any violated claim above.
        OSError: If the artifact or the calibration file cannot be read.
    """
    path = Path(awq_clip_path).expanduser()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_awq_clip_keys)
    except UnicodeDecodeError as error:
        raise ValueError(f"AWQ-clip artifact {path} must be UTF-8 encoded: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"AWQ-clip artifact {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"AWQ-clip artifact {path} must contain a JSON object, got {type(payload).__name__}")
    _require_exact_keys(payload, AWQ_CLIP_ARTIFACT_KEYS, f"AWQ-clip artifact {path}", "top-level")

    if payload["schema"] != AWQ_CLIP_SCHEMA:
        raise ValueError(
            f"AWQ-clip artifact {path} declares schema {payload['schema']!r}, but {AWQ_CLIP_SCHEMA!r} is required."
        )
    if isinstance(payload["version"], bool) or payload["version"] != AWQ_CLIP_SCHEMA_VERSION:
        raise ValueError(
            f"AWQ-clip artifact {path} has version {payload['version']!r}, but version {AWQ_CLIP_SCHEMA_VERSION} "
            "is required."
        )
    if payload["algorithm"] != WEIGHT_SCALE_AWQ_CLIP_ALGORITHM or (
        isinstance(payload["algorithm_version"], bool)
        or payload["algorithm_version"] != WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION
    ):
        raise ValueError(
            f"AWQ-clip artifact {path} was built for algorithm {payload['algorithm']!r} "
            f"v{payload['algorithm_version']!r}, but this build only consumes "
            f"{WEIGHT_SCALE_AWQ_CLIP_ALGORITHM!r} v{WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION}."
        )
    if payload["weight_digest_method"] != WEIGHT_DIGEST_METHOD:
        raise ValueError(
            f"AWQ-clip artifact {path} records the weight-digest method {payload['weight_digest_method']!r}, but "
            f"this build verifies weights with {WEIGHT_DIGEST_METHOD!r}."
        )
    if payload["code_encoding"] != AWQ_CLIP_CODE_ENCODING:
        raise ValueError(
            f"AWQ-clip artifact {path} records the code encoding {payload['code_encoding']!r}, but this build "
            f"decodes {AWQ_CLIP_CODE_ENCODING!r}."
        )
    checkpoint_sha256 = _validate_checkpoint_sha256(payload["checkpoint_sha256"], f"'checkpoint_sha256' in {path}")
    template_arithmetic = _validate_awq_clip_arithmetic(payload["arithmetic"], path)
    calibration = _validate_awq_clip_calibration(
        payload["activation_calibration"], calibration_path, scale_margin, checkpoint_sha256, path
    )

    required = selection.fqns_for_precision(PRECISION_NVFP4_W4A4)
    modules = dict(model.named_modules())
    decoded = _validate_awq_clip_codes(payload["ratio_codes"], required, modules, path)
    provenance = _validate_awq_clip_provenance(payload["provenance"], required, path)
    # The two component digests are recomputed from the sections the file actually carries, so an edited code
    # payload or a rewritten source record is refused even when the result is still structurally valid JSON. They
    # are checked after the structural validation so a malformed section is named for what it is, and before
    # anything is converted.
    ratio_code_sha256 = _verify_awq_clip_section_digest(payload, "ratio_code_sha256", "ratio_codes", path)
    provenance_sha256 = _verify_awq_clip_section_digest(payload, "provenance_sha256", "provenance", path)
    # A digest only proves that nobody edited a section after it was taken. The per-module and aggregate claims are
    # therefore recomputed from the decoded codes themselves, which a fabricated artifact cannot pass by recomputing
    # its own provenance digest.
    _validate_awq_clip_aggregate(provenance, decoded["ratio_histogram"], required, path)
    weight_sha256 = _validate_awq_clip_weight_digests(payload["weight_sha256"], required, modules, path)

    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "algorithm": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION,
        "clip_ratios": [float(ratio) for ratio in WEIGHT_SCALE_AWQ_CLIP_RATIOS],
        "block_size": int(WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE),
        "tie_rule": AWQ_CLIP_TIE_RULE,
        "template_arithmetic": template_arithmetic,
        "calibration_path": str(Path(calibration_path).expanduser()),
        "calibration": calibration,
        "fqns": list(required),
        # Kept as raw bytes rather than as Python integers: one uint8 per block is a few megabytes of ``bytes`` for
        # the whole model, and materializing them as lists of ints would cost two orders of magnitude more.
        "ratio_codes": decoded["ratio_codes"],
        "code_shapes": decoded["code_shapes"],
        "ratio_histogram": decoded["ratio_histogram"],
        "weight_sha256": weight_sha256,
        "ratio_code_sha256": ratio_code_sha256,
        "provenance_sha256": provenance_sha256,
        "provenance": provenance,
    }


def awq_clip_template_arithmetic_for_backend(backend: str) -> str:
    """
    The ordinary-template construction a run on ``backend`` will convert its weights with.

    ``_apply_nvfp4_activation_quantization`` sets TorchAO's Triton-kernel flag from exactly this distinction, so the
    accelerated backend converts through the MSLK kernel and every other backend through the reference non-Triton
    path. The unclipped AWQ-clip code means "leave this block as the ordinary conversion wrote it", and those two
    conversions do not write it identically, so this is what an artifact's recorded construction must match.

    Args:
        backend (str): Resolved backend label.

    Returns:
        template_arithmetic (str): One of :data:`WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS`.
    """
    if backend == BACKEND_MSLK_ACCELERATED:
        return WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED
    return WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE


def require_awq_clip_template_arithmetic(awq_clip: Dict[str, Any], backend: str) -> str:
    """
    Refuse an AWQ-clip artifact whose unclipped code was scored against the other ordinary-template construction.

    This runs before a single module is converted, because the artifact's ratio-10 blocks are only meaningful as
    "the bytes *that* conversion produces": deploying them under the other backend would keep bytes the offline
    search never scored, and would silently execute a different weight than the one that was selected.

    Args:
        awq_clip (Dict[str, Any]): Validated artifact from :func:`load_awq_clip_artifact`.
        backend (str): Resolved backend label of this run.

    Returns:
        template_arithmetic (str): The agreed construction.

    Raises:
        ValueError: If the artifact was built against the other construction.
    """
    expected = awq_clip_template_arithmetic_for_backend(backend)
    recorded = str(awq_clip["template_arithmetic"])
    if recorded != expected:
        raise ValueError(
            f"AWQ-clip artifact {awq_clip['path']} selected its unclipped blocks against the "
            f"'{recorded}' ordinary-template construction, but backend '{backend}' converts with "
            f"'{expected}'. The two do not produce the same block scales or the same packed payload, so the codes "
            "would deploy blocks nobody scored. Rebuild the codes for this backend."
        )
    return expected


def load_gptq_artifact(
    gptq_path: str,
    model: torch.nn.Module,
    selection: QuantizationSelection,
    calibration_path: str,
    scale_margin: float = WEIGHT_SCALE_GPTQ_SCALE_MARGIN,
) -> Dict[str, Any]:
    """
    Load and strictly validate a GPTQ artifact against the live model and the exact calibration file.

    This artifact is the only thing standing between an offline payload selection and the bytes a quantized run
    executes, and the evidence it was selected from -- the activation rows and the Hessians -- is deliberately
    absent from it, so every one of its claims is checked before a single module is converted: the closed
    top-level, arithmetic, calibration, payload, template-scale, Hessian, provenance, source, module and aggregate
    key sets; the schema, version and algorithm identity; every arithmetic constant, including the damping, the
    Hessian and group rules, the dead-column rule, the inverse identity, the 128-column update block, the 16-value
    NVFP4 block, the fixed-template-scale rule, the activation-QDQ identity, the FP4/FP8 maxima, the recorded
    ordinary-template construction and the ModelOpt reference version and wheel digest; the identity of the
    configured activation-calibration file, recomputed from its actual bytes; the exact coverage of the selected
    NVFP4 W4A4 FQNs in canonical form; every payload's declared shape against the live
    ``out_features``/``in_features``, its declared byte length and its content digest; every recorded
    template-scale and Hessian claim; the three recorded :data:`SECTION_DIGEST_METHOD` digests; the agreement of
    every per-module and aggregate claim with the payloads the artifact actually carries; and finally the digest of
    every live original weight against the one the artifact was built on.

    What this cannot check yet is the *template* each payload was written under, because TorchAO has not converted
    anything at this point. That binding -- the exact scale-buffer bytes and the unchanged global scale -- is
    re-checked per FQN, against the template the conversion actually produced, immediately before the payload
    replaces it. Nothing here falls back: a violated claim is a failed run.

    Args:
        gptq_path (str): Path to the artifact JSON.
        model (torch.nn.Module): Live, still-unconverted model whose weights and shapes are checked.
        selection (QuantizationSelection): Selection whose NVFP4 W4A4 FQNs the artifact must cover exactly.
        calibration_path (str): Path of the static activation-calibration file the run will consume; the artifact
            must be bound to exactly its bytes.
        scale_margin (float): The run's activation scale margin, which must be the one the builder used.

    Returns:
        gptq (Dict[str, Any]): ``path``, ``sha256`` of the exact file bytes, ``checkpoint_sha256``, ``algorithm``,
            ``algorithm_version``, ``perc_damp``, ``update_block_size``, ``block_size``, ``template_arithmetic``,
            ``calibration_path``, the verified ``calibration`` identity, the sorted ``fqns``, the decoded per-FQN
            ``qdata`` as raw bytes with their ``qdata_shapes``, the per-FQN ``template_scale`` and ``hessian``
            claims, the per-FQN ``weight_sha256``, the three verified section digests, and the ``provenance``.

    Raises:
        ValueError: For any violated claim above.
        OSError: If the artifact or the calibration file cannot be read.
    """
    path = Path(gptq_path).expanduser()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_gptq_keys)
    except UnicodeDecodeError as error:
        raise ValueError(f"GPTQ artifact {path} must be UTF-8 encoded: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"GPTQ artifact {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"GPTQ artifact {path} must contain a JSON object, got {type(payload).__name__}")
    _require_exact_keys(payload, GPTQ_ARTIFACT_KEYS, f"GPTQ artifact {path}", "top-level")

    if payload["schema"] != GPTQ_SCHEMA:
        raise ValueError(
            f"GPTQ artifact {path} declares schema {payload['schema']!r}, but {GPTQ_SCHEMA!r} is required."
        )
    if isinstance(payload["version"], bool) or payload["version"] != GPTQ_SCHEMA_VERSION:
        raise ValueError(
            f"GPTQ artifact {path} has version {payload['version']!r}, but version {GPTQ_SCHEMA_VERSION} is "
            "required."
        )
    if payload["algorithm"] != WEIGHT_SCALE_GPTQ_ALGORITHM or (
        isinstance(payload["algorithm_version"], bool)
        or payload["algorithm_version"] != WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION
    ):
        raise ValueError(
            f"GPTQ artifact {path} was built for algorithm {payload['algorithm']!r} "
            f"v{payload['algorithm_version']!r}, but this build only consumes {WEIGHT_SCALE_GPTQ_ALGORITHM!r} "
            f"v{WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION}."
        )
    for field, expected in (
        ("weight_digest_method", WEIGHT_DIGEST_METHOD),
        ("section_digest_method", SECTION_DIGEST_METHOD),
        ("payload_encoding", GPTQ_PAYLOAD_ENCODING),
    ):
        if payload[field] != expected:
            raise ValueError(
                f"GPTQ artifact {path} records '{field}' as {payload[field]!r}, but this build uses {expected!r}."
            )
    checkpoint_sha256 = _validate_checkpoint_sha256(payload["checkpoint_sha256"], f"'checkpoint_sha256' in {path}")
    template_arithmetic = _validate_gptq_arithmetic(payload["arithmetic"], path)
    calibration = _validate_gptq_calibration(
        payload["activation_calibration"], calibration_path, scale_margin, checkpoint_sha256, path
    )

    required = selection.fqns_for_precision(PRECISION_NVFP4_W4A4)
    modules = dict(model.named_modules())
    decoded = _validate_gptq_qdata(payload["qdata"], required, modules, path)
    template_scale = _validate_gptq_template_scale(payload["template_scale"], required, modules, path)
    hessian = _validate_gptq_hessian_section(payload["hessian"], required, modules, path)
    provenance = _validate_gptq_provenance(payload["provenance"], required, path)
    # The three component digests are recomputed from the sections the file actually carries, so an edited payload
    # byte, a rewritten Hessian claim or a rewritten source record is refused even when the result is still
    # structurally valid JSON. They are checked after the structural validation so a malformed section is named for
    # what it is, and before anything is converted.
    qdata_sha256 = _verify_gptq_section_digest(payload, "qdata_sha256", "qdata", path)
    hessian_sha256 = _verify_gptq_section_digest(payload, "hessian_sha256", "hessian", path)
    provenance_sha256 = _verify_gptq_section_digest(payload, "provenance_sha256", "provenance", path)
    # A digest only proves that nobody edited a section after it was taken. Every per-module and aggregate claim is
    # therefore recomputed from the decoded payloads and the validated sections themselves.
    _validate_gptq_aggregate(provenance, decoded, hessian, required, path)
    weight_sha256 = _validate_gptq_weight_digests(payload["weight_sha256"], required, modules, path)

    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "algorithm": WEIGHT_SCALE_GPTQ_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION,
        "perc_damp": float(WEIGHT_SCALE_GPTQ_PERC_DAMP),
        "update_block_size": int(WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE),
        "block_size": int(WEIGHT_SCALE_GPTQ_BLOCK_SIZE),
        "template_arithmetic": template_arithmetic,
        "calibration_path": str(Path(calibration_path).expanduser()),
        "calibration": calibration,
        "fqns": list(required),
        # Kept as raw bytes rather than as Python integers: the payload of the whole model is a few megabytes of
        # ``bytes``, and materializing it as lists of ints would cost two orders of magnitude more. Exactly one
        # FQN's bytes are ever turned into a device tensor, immediately before that module's payload is replaced.
        "qdata": decoded["qdata"],
        "qdata_shapes": decoded["qdata_shapes"],
        "qdata_digests": decoded["qdata_digests"],
        "template_scale": template_scale,
        "hessian": hessian,
        "weight_sha256": weight_sha256,
        "qdata_sha256": qdata_sha256,
        "hessian_sha256": hessian_sha256,
        "provenance_sha256": provenance_sha256,
        "provenance": provenance,
    }


def require_gptq_template_arithmetic(gptq: Dict[str, Any], backend: str) -> str:
    """
    Refuse a GPTQ artifact whose payload was written under the other ordinary-template construction.

    This runs before a single module is converted. A GPTQ payload is only meaningful under the exact block scales
    it was written for, and the two constructions do not produce the same ones, so deploying it on the other
    backend would decode every block against a scale nobody selected it under.

    Args:
        gptq (Dict[str, Any]): Validated artifact from :func:`load_gptq_artifact`.
        backend (str): Resolved backend label of this run.

    Returns:
        template_arithmetic (str): The agreed construction.

    Raises:
        ValueError: If the artifact was built against the other construction.
    """
    # The backend -> construction mapping is exactly the AWQ-clip one: it is a property of the conversion TorchAO
    # performs for a backend, not of either method.
    expected = awq_clip_template_arithmetic_for_backend(backend)
    recorded = str(gptq["template_arithmetic"])
    if recorded != expected:
        raise ValueError(
            f"GPTQ artifact {gptq['path']} wrote its payload under the '{recorded}' ordinary-template "
            f"construction, but backend '{backend}' converts with '{expected}'. The two do not produce the same "
            "block scales, so the stored payload would decode against scales nobody selected it under. Rebuild the "
            "payload for this backend."
        )
    return expected


def nvfp4_gptq_hessian_digest(hessian: torch.Tensor) -> str:
    """
    Canonical digest of one damped GPTQ Hessian, shared by the offline builder and the runtime evidence check.

    The digest is :data:`GPTQ_HESSIAN_DIGEST_METHOD`: the SHA-256 of the matrix's contiguous row-major float32
    bytes on the host. The matrix itself is never stored -- a ``(K, K)`` FP32 matrix per module would dwarf the
    payload and would carry activation statistics into a runtime artifact -- so this digest is the only identity a
    run has of the Hessian its payload was selected under, and two builds on the same rows produce the same value.

    Args:
        hessian (torch.Tensor): The damped ``(K, K)`` Hessian.

    Returns:
        digest (str): 64-character lowercase hexadecimal SHA-256.
    """
    data = hessian.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def nvfp4_gptq_weighted_objective(modules: Dict[str, Any], fqns: Sequence[str], field: str) -> float:
    """
    Weight-count-weighted mean of one per-module GPTQ statistic, reduced in the canonical FQN order.

    The offline builder records exactly what this returns and the runtime loader recomputes it from the artifact's
    own per-module numbers, so the two are compared exactly rather than approximately: python float arithmetic is
    deterministic for a fixed order, and a JSON round trip reproduces a double exactly.

    Args:
        modules (Dict[str, Any]): Per-FQN evidence objects carrying ``weight_count`` and ``field``.
        fqns (Sequence[str]): The FQNs to reduce over, in the canonical (sorted) order both sides use.
        field (str): Name of the per-module statistic, e.g. ``'selected_objective'``.

    Returns:
        value (float): The weighted mean.

    Raises:
        ValueError: If the modules describe no weight at all, which no artifact may claim.
    """
    total = 0.0
    total_weights = 0
    for fqn in fqns:
        count = int(modules[fqn]["weight_count"])
        total += float(modules[fqn][field]) * count
        total_weights += count
    if total_weights <= 0:
        raise ValueError(f"Cannot reduce '{field}' over {len(list(fqns))} module(s) describing no weight.")
    return total / total_weights


def nvfp4_awq_clip_calibration_identity(calibration_path: str) -> Dict[str, Any]:
    """
    Canonical identity of the activation-calibration file an AWQ-clip artifact is bound to.

    The offline builder records exactly this object and the runtime loader recomputes it from the file the run was
    configured with, so a run can only proceed when the codes were selected against the very activations this run
    will quantize. Beyond the file's exact bytes and size it carries the claims that decide those activations: the
    schema version, the static scale mode, the checkpoint the maxima were collected on, how many modules they
    cover, the baked-in headroom, the runtime margin that headroom presumes, and the merge method identity.

    :func:`merge_calibrations` writes the checkpoint digest inside ``metadata``, which is where a production
    calibration carries it; a top-level :data:`CALIBRATION_CHECKPOINT_FIELD` is accepted as well, for older
    single-collector artifacts. When both are present both must be valid and *identical*, because two conflicting
    claims cannot both describe the checkpoint the maxima were collected on. When neither is present the file is
    refused outright: AWQ-clip ratio codes are only meaningful bound to a checkpoint, and an unbound calibration
    cannot supply that binding.

    The baked-headroom contract is enforced here too, because it is what makes
    :data:`WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN` the only valid runtime margin: the file must state a finite positive
    ``headroom``, must state ``headroom_baked_in`` as exactly boolean ``true``, and must presume exactly that
    runtime margin. A file that merely omits those claims, or states them falsely, would have its headroom applied
    twice or not at all, so it is refused rather than read optimistically.

    Args:
        calibration_path (str): Path to a schema-v1 calibration JSON file.

    Returns:
        identity (Dict[str, Any]): The closed identity object described above.

    Raises:
        ValueError: If the file is not readable UTF-8 JSON, is not an object, has duplicate keys, has another
            schema version, carries no activation maxima, declares a malformed or conflicting checkpoint digest,
            declares no checkpoint at all, or violates the baked-headroom contract.
        OSError: If the file cannot be read.
    """
    path = Path(calibration_path).expanduser()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as error:
        raise ValueError(f"Calibration file {path} must be UTF-8 encoded: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Calibration file {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file {path} must contain a JSON object, got {type(payload).__name__}")
    version = payload.get("version")
    if isinstance(version, bool) or version != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Calibration file {path} has version {version!r}, but version {CALIBRATION_SCHEMA_VERSION} is required."
        )
    activation_amax = payload.get("activation_amax")
    if not isinstance(activation_amax, dict) or not activation_amax:
        raise ValueError(f"Calibration file {path} must contain a non-empty 'activation_amax' object")
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    scale_mode = payload.get("scale_mode")
    checkpoint = _resolve_calibration_checkpoint(payload, metadata, path)
    headroom = _validated_finite_number(metadata.get("headroom"), f"'metadata.headroom' in {path}")
    if headroom <= 0.0:
        raise ValueError(f"'metadata.headroom' in {path} must be positive, got {headroom!r}")
    baked_in = metadata.get("headroom_baked_in")
    if baked_in is not True:
        raise ValueError(
            f"'metadata.headroom_baked_in' in {path} is {baked_in!r}, but an AWQ-clip run consumes exactly a "
            "calibration whose headroom is already baked into its values. A file that does not state this as "
            "boolean true would have its headroom applied twice or not at all at margin "
            f"{float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN)!r}."
        )
    runtime_margin = _validated_finite_number(
        metadata.get("runtime_scale_margin"), f"'metadata.runtime_scale_margin' in {path}"
    )
    if runtime_margin != float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN):
        raise ValueError(
            f"Calibration file {path} presumes a runtime scale margin of {runtime_margin!r}, but AWQ-clip codes "
            f"are selected and executed at exactly {float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN)!r}."
        )
    method = metadata.get("method")
    if method is not None and not isinstance(method, str):
        raise ValueError(f"'metadata.method' in {path} must be a string, got {method!r}")
    method_version = metadata.get("method_version")
    if method_version is not None and (isinstance(method_version, bool) or not isinstance(method_version, int)):
        raise ValueError(f"'metadata.method_version' in {path} must be an integer, got {method_version!r}")
    return {
        "name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "checkpoint_sha256": checkpoint,
        "version": int(version),
        "scale_mode": None if scale_mode is None else str(scale_mode),
        "target_module_count": len(activation_amax),
        "headroom": headroom,
        "headroom_baked_in": True,
        "runtime_scale_margin": runtime_margin,
        "merge_method": None if method is None else str(method),
        "merge_method_version": None if method_version is None else int(method_version),
    }


def nvfp4_weight_digest(weight: torch.Tensor) -> str:
    """
    Canonical digest of one unconverted weight, shared by the offline builder and the runtime binding check.

    The digest is :data:`WEIGHT_DIGEST_METHOD`: the SHA-256 of ``str(dtype)``, ``str(tuple(shape))`` and the raw
    contiguous host-order bytes of the tensor on CPU, separated by ``|``. Hashing dtype and shape alongside the
    bytes means a weight that was reshaped or re-dtyped is a different weight, which is what a binding between
    offline statistics and a runtime weight has to mean.

    Args:
        weight (torch.Tensor): Unconverted weight of any dtype and shape.

    Returns:
        digest (str): 64-character lowercase hexadecimal SHA-256.
    """
    data = weight.detach().to("cpu").contiguous()
    header = f"{data.dtype}|{tuple(int(size) for size in data.shape)}|".encode("utf-8")
    # Viewing as uint8 keeps this exact for bfloat16 and float8 storage, which NumPy has no dtype for.
    raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()


def nvfp4_section_digest(section: Any) -> str:
    """
    Canonical digest of one artifact section, shared by the offline builder and the runtime loader.

    The digest is :data:`SECTION_DIGEST_METHOD`: the section is re-serialized as compact, sorted, non-NaN UTF-8
    JSON and hashed. Canonicalizing before hashing is what makes the digest survive the artifact's own pretty
    printed write and its later parse, so the builder's recorded value and the runtime's recomputed one agree
    exactly for the same data and differ for any edited value.

    Args:
        section (Any): JSON-safe section, e.g. the ``diagonal_hessian`` mapping or the ``provenance`` object.

    Returns:
        digest (str): 64-character lowercase hexadecimal SHA-256.

    Raises:
        ValueError: If the section cannot be canonicalized, which includes a non-finite number: a section that
            cannot be serialized deterministically cannot be bound to a digest either.
    """
    try:
        canonical = json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Section cannot be canonicalized for digesting: {error}") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def nvfp4_awq_clip_weighted_objective(modules: Dict[str, Any], fqns: Sequence[str], field: str) -> float:
    """
    Block-count-weighted mean of one per-module AWQ-clip objective, reduced in the canonical FQN order.

    The offline builder records exactly what this returns and the runtime loader recomputes it from the artifact's
    own per-module numbers, so the two can be compared exactly rather than approximately: python float arithmetic
    is deterministic for a fixed order, and a JSON round trip reproduces a double exactly.

    Args:
        modules (Dict[str, Any]): Per-FQN evidence objects carrying ``block_count`` and ``field``.
        fqns (Sequence[str]): The FQNs to reduce over, in the canonical (sorted) order both sides use.
        field (str): Name of the per-module objective, e.g. ``'selected_objective'``.

    Returns:
        objective (float): The weighted mean.

    Raises:
        ValueError: If the modules describe no block at all, which no artifact may claim.
    """
    total = 0.0
    total_blocks = 0
    for fqn in fqns:
        blocks = int(modules[fqn]["block_count"])
        total += float(modules[fqn][field]) * blocks
        total_blocks += blocks
    if total_blocks <= 0:
        raise ValueError(f"Cannot reduce '{field}' over {len(list(fqns))} module(s) describing no 16-weight block.")
    return total / total_blocks


def validate_sha256_digest(value: Any, description: str) -> str:
    """Validate a digest as 64 lowercase hexadecimal characters, returning the normalized value."""
    return _validate_checkpoint_sha256(value, description)


def disabled_bf16_override_summary() -> Dict[str, Any]:
    """Summary section reported when no BF16 restoration file was given; same schema as the enabled one."""
    return {"enabled": False, "path": None, "sha256": None, "count": 0, "fqns": []}


def disabled_weight_scale_mse_summary(method: str = WEIGHT_SCALE_METHOD_AMAX) -> Dict[str, Any]:
    """
    Summary section reported when the block-MSE weight-scale search is off; same schema as the enabled one.

    A disabled section names the method that actually selected the block scales, so a summary can never be
    mistaken for evidence that this search ran. Under the default ``'amax'`` that is TorchAO's ordinary rule;
    under another searched method it is that method, because claiming amax there would be plainly false.

    Args:
        method (str): The ``weight_scale_method`` the run actually used. Defaults to
            :data:`WEIGHT_SCALE_METHOD_AMAX`.
    """
    notes = ["NVFP4 weight block scales are the ordinary TorchAO amax-derived ones; the per-block MSE search is OFF."]
    if method != WEIGHT_SCALE_METHOD_AMAX:
        notes = [
            "The per-block MSE search is OFF; the NVFP4 weight block scales of this run were selected by "
            f"'{method}'."
        ]
    return {
        "enabled": False,
        "method": method,
        "algorithm": None,
        "algorithm_version": None,
        "target_count": 0,
        "total_weight_count": 0,
        "layers": [],
        "aggregate": _weight_scale_mse_aggregate([]),
        "notes": notes,
    }


def disabled_weight_scale_hessian_summary(method: str = WEIGHT_SCALE_METHOD_AMAX) -> Dict[str, Any]:
    """
    Summary section reported when the activation-weighted search is off; same schema as the enabled one.

    A disabled section names the method that actually ran and carries no artifact identity, so a summary can never
    be mistaken for evidence that activation statistics selected the block scales.

    Args:
        method (str): The ``weight_scale_method`` the run actually used. Defaults to
            :data:`WEIGHT_SCALE_METHOD_AMAX`.
    """
    return {
        "enabled": False,
        "method": method,
        "algorithm": None,
        "algorithm_version": None,
        "damping": None,
        "artifact_path": None,
        "artifact_sha256": None,
        "moment_sha256": None,
        "provenance_sha256": None,
        "checkpoint_sha256": None,
        "target_count": 0,
        "target_fqns": [],
        "total_weight_count": 0,
        "layers": [],
        "aggregate": _weight_scale_hessian_aggregate([]),
        "notes": [
            "NVFP4 weight block scales were not selected by the activation-weighted diagonal-Hessian search; it "
            "is OFF."
        ],
    }


def quantize_sortformer_model(
    model: torch.nn.Module,
    config: SortformerQuantizationConfig,
    facts: Optional[CapabilityFacts] = None,
) -> Dict[str, Any]:
    """
    Apply the requested quantization recipe in place and return a JSON/OmegaConf-safe summary.

    Args:
        model (torch.nn.Module): Restored Sortformer model, already moved to its device and set to eval mode.
        config (SortformerQuantizationConfig): Quantization options; validated here if not already.
        facts (Optional[CapabilityFacts]): Injected runtime facts. Collected from the live runtime when ``None``.

    Returns:
        summary (Dict[str, Any]): Recipe, scale mode, backend, the conditional acceleration report, architecture,
            compute capability and its qualification status, selected FQNs and counts by precision,
            rejected/skipped data, scale margin, the weight-scale method with its per-FQN and aggregate block-MSE
            evidence, and dependency versions.

    Raises:
        RuntimeError: If the runtime cannot support the request, or if a selected module cannot be quantized as
            requested. Kernel failures are never swallowed to continue in BF16.
        ValueError: If options, targets, or calibration data are invalid.
    """
    config.validate()
    selection = select_quantization_targets(model, config.recipe)
    bf16_override = disabled_bf16_override_summary()
    if config.has_bf16_override:
        # Validated against the live matched targets before any module is touched, so a malformed or stale
        # override file fails on a still-unmodified model.
        bf16_override = load_bf16_override(config.bf16_override_path, selection)
        if config.fuse_producer_packing:
            # Producer fusion packs into and dispatches exactly attn.w_qkv, ffn.net.0 and ffn.net.3; restoring
            # one of those leaves a fused kernel with nothing to write into. attn.out_proj is reached only via
            # attn.forward_from_qkv, so restoring it is compatible. Checked here, before conversion, so the run
            # fails on an unmodified model rather than after TorchAO has rewritten the weights.
            restored_consumers = sorted(
                fqn
                for fqn in bf16_override["fqns"]
                if any(fqn.endswith("." + consumer) for consumer in PRODUCER_FUSION_REQUIRED_CONSUMERS)
            )
            if restored_consumers:
                raise ValueError(
                    "quantization_fuse_producer_packing cannot be combined with a bf16_override that restores a "
                    f"fused consumer {list(PRODUCER_FUSION_REQUIRED_CONSUMERS)}; there would be no NVFP4 pack to "
                    f"fuse into. Offending entries: {restored_consumers[:8]}"
                    f"{'...' if len(restored_consumers) > 8 else ''}. Restoring attn.out_proj is supported."
                )
        selection = select_quantization_targets(model, config.recipe, bf16_fqns=bf16_override["fqns"])
    device = _model_device(model)
    if facts is None:
        # Optional dependencies are only probed on an enabled path, so a disabled recipe never imports them.
        facts = (
            collect_capability_facts(device)
            if config.enabled
            else CapabilityFacts(device_type=device.type, torch_version=torch.__version__)
        )
    backend = check_nvfp4_capability(config, facts)

    nvfp4_fqns = selection.fqns_for_precision(PRECISION_NVFP4_W4A4)
    weight_only_fqns = selection.fqns_for_precision(PRECISION_NVFP4_WEIGHT_ONLY)
    fp8_fqns = selection.fqns_for_precision(PRECISION_FP8_DYNAMIC)
    acceleration = _acceleration_report(model, nvfp4_fqns, backend)
    if config.uses_searched_weight_scales and not (nvfp4_fqns or weight_only_fqns):
        # Checked before anything is converted: a recipe that selects no NVFP4 weight has nothing to repack, and
        # silently reporting an empty search would look like a run that measured something.
        raise ValueError(
            f"quantization_weight_scale_method='{config.weight_scale_method}' repacks converted NVFP4 weights, but "
            f"recipe '{config.recipe}' selects no NVFP4 weight in this model."
        )

    calibration = None
    if config.scale_mode == "static":
        calibration = load_calibration(config.calibration_path, selection, config.scale_margin)

    hessian = None
    if config.uses_local_hessian_weight_scales:
        # Loaded, validated and bound to the live weights while the model is still unconverted, so a stale or
        # foreign artifact fails on an unmodified model instead of halfway through the conversion.
        hessian = load_diagonal_hessian(config.weight_scale_hessian_path, model, selection)

    awq_clip = None
    if config.uses_awq_clip_weight_scales:
        # Also loaded, validated and bound -- to the live weights *and* to the exact calibration file whose values
        # quantized the activations the codes were selected against -- while the model is still unconverted.
        awq_clip = load_awq_clip_artifact(
            config.weight_scale_awq_clip_path, model, selection, config.calibration_path, config.scale_margin
        )
        # And bound to the ordinary-template construction this backend will convert with: the unclipped code keeps
        # that conversion's own bytes, which the other backend does not produce.
        require_awq_clip_template_arithmetic(awq_clip, backend)

    gptq = None
    if config.uses_gptq_weight_scales:
        # Same discipline as AWQ-clip: the artifact, its calibration binding and every live weight digest are
        # validated while the model is still unconverted, and the per-FQN template-scale binding is re-checked
        # against each template the conversion produces before that module's payload is replaced.
        gptq = load_gptq_artifact(
            config.weight_scale_gptq_path, model, selection, config.calibration_path, config.scale_margin
        )
        require_gptq_template_arithmetic(gptq, backend)

    global_scale_folding = disabled_folding_summary()
    producer_fusion = disabled_producer_fusion_summary()
    # Both sections name the method that actually ran, so the one that is OFF never claims the scales came from
    # somewhere they did not.
    weight_scale_mse = disabled_weight_scale_mse_summary(config.weight_scale_method)
    weight_scale_hessian = disabled_weight_scale_hessian_summary(config.weight_scale_method)
    weight_scale_four_over_six = None
    weight_scale_awq_clip = None
    weight_scale_gptq = None
    if config.enabled:
        quantize_ = _require_api(TORCHAO_QUANTIZE_API)
        # ``None`` is the ordinary batched amax path; otherwise exactly one repacker is named, so no branch can
        # ever infer which method runs from the presence or absence of another method's data.
        repack_plan = _weight_repack_plan(config, hessian, awq_clip, gptq)
        search_entries: List[Dict[str, Any]] = []
        if nvfp4_fqns:
            search_entries += _apply_nvfp4_activation_quantization(
                model, nvfp4_fqns, backend, calibration, quantize_, repack_plan
            )
            if config.fold_global_scales:
                # Runs strictly after TorchAO has converted exactly these FQNs, and touches no other module.
                global_scale_folding = apply_global_scale_folding(model, nvfp4_fqns, config.fold_activation_exponent)
            if config.fuse_producer_packing:
                # Also strictly after conversion and after every weight repack, and before the caller compiles the
                # encoder. Fusion requires recipe='nvfp4_all', so no weight-only repack can follow it.
                producer_fusion = apply_producer_fusion(model, nvfp4_fqns)
        if weight_only_fqns:
            weight_only_config = _require_api(TORCHAO_NVFP4_WEIGHT_ONLY_CONFIG_API)(use_dynamic_per_tensor_scale=True)
            search_entries += _apply_nvfp4_weight_only_quantization(
                model, weight_only_fqns, weight_only_config, quantize_, repack_plan
            )
        if fp8_fqns:
            fp8_config = _require_api(TORCHAO_FP8_CONFIG_API)()
            quantize_(model, fp8_config, filter_fn=_fqn_filter(fp8_fqns))
        if config.uses_mse_weight_scales:
            weight_scale_mse = _weight_scale_mse_summary(search_entries)
        if config.uses_local_hessian_weight_scales:
            weight_scale_hessian = _weight_scale_hessian_summary(search_entries, hessian)
        if config.uses_four_over_six_weight_scales:
            weight_scale_four_over_six = _weight_scale_four_over_six_summary(search_entries)
        if config.uses_awq_clip_weight_scales:
            weight_scale_awq_clip = _weight_scale_awq_clip_summary(search_entries, awq_clip)
        if config.uses_gptq_weight_scales:
            weight_scale_gptq = _weight_scale_gptq_summary(search_entries, gptq)

    summary: Dict[str, Any] = {
        "recipe": config.recipe,
        "scale_mode": config.scale_mode if config.uses_activation_quantization else "not_applicable",
        "backend": backend,
        "accelerated_packing": bool(config.accelerated_packing),
        "explicit_reference_kernels": backend == BACKEND_REFERENCE_UNACCELERATED,
        "architecture": facts.architecture,
        "compute_capability": None if facts.compute_capability is None else list(facts.compute_capability),
        "architecture_qualification": {
            "accepted_by_policy": config.enabled,
            "status": facts.qualification,
            "policy_accepted_architectures": [
                ARCHITECTURE_BY_COMPUTE_CAPABILITY[capability] for capability in SUPPORTED_COMPUTE_CAPABILITIES
            ],
            "qualified_architectures": [
                ARCHITECTURE_BY_COMPUTE_CAPABILITY.get(tuple(capability), str(tuple(capability)))
                for capability in facts.qualified_compute_capabilities
            ],
        },
        "device": str(device),
        "selected_fqns": {
            precision: selection.fqns_for_precision(precision)
            for precision in (
                PRECISION_NVFP4_W4A4,
                PRECISION_NVFP4_WEIGHT_ONLY,
                PRECISION_FP8_DYNAMIC,
                PRECISION_BF16,
            )
        },
        "counts_by_precision": selection.counts_by_precision,
        "fqn_counts_by_target": {suffix: len(fqns) for suffix, fqns in sorted(selection.fqns_by_suffix.items())},
        # Rejections (missing or non-Linear targets, unusable calibration entries) raise instead of being
        # tolerated, so this list is empty whenever a summary is produced at all; it is kept for schema stability.
        "rejected_fqns": [],
        "skipped_fqns": selection.fqns_for_precision(PRECISION_BF16),
        "scale_margin": float(config.scale_margin) if config.scale_mode == "static" else None,
        "calibration_path": None if calibration is None else calibration["path"],
        "unused_calibration_fqns": [] if calibration is None else calibration["unused_fqns"],
        "acceleration": acceleration,
        "bf16_override": bf16_override,
        "global_scale_folding": global_scale_folding,
        "producer_fusion": producer_fusion,
        "weight_scale_method": config.weight_scale_method,
        "weight_scale_mse": weight_scale_mse,
        "weight_scale_hessian": weight_scale_hessian,
        # Present only for an enabled Four-Over-Six run, so every amax, MSE and local-Hessian summary keeps exactly
        # the keys it had before this method existed.
        **({} if weight_scale_four_over_six is None else {"weight_scale_four_over_six": weight_scale_four_over_six}),
        # Present only for an enabled AWQ-clip run, for exactly the same reason.
        **({} if weight_scale_awq_clip is None else {"weight_scale_awq_clip": weight_scale_awq_clip}),
        # Present only for an enabled GPTQ run, for exactly the same reason.
        **({} if weight_scale_gptq is None else {"weight_scale_gptq": weight_scale_gptq}),
        "versions": {
            "torch": facts.torch_version,
            "torchao": facts.torchao_version,
            "mslk": facts.mslk_version,
        },
        "notes": _summary_notes(config, backend, facts),
    }
    return _normalize_json(summary)


class ActivationAmaxCollector:
    """
    Collect per-module activation maxima on the exact Sortformer quantization targets.

    Removable forward pre-hooks record the detached, finite absolute maximum of each module input. The collector
    is install/remove idempotent so that an evaluator ``finally`` block can always call :meth:`remove`.

    Maxima are recorded under :func:`_canonical_fqn` names, so a calibration file collected on a
    ``torch.compile``-wrapped model still matches the uncompiled module tree that quantization runs against.

    Besides the running maximum, every accepted per-invocation maximum is retained in invocation order in
    :attr:`activation_amax_samples`, which :func:`merge_calibrations` turns into per-source percentiles. Only one
    Python float per target and model invocation is kept; no activation tensor, tensor element, or GPU tensor is
    retained.

    Non-finite elements are filtered out of the maximum exactly as before, but the invocations in which that
    happened are counted in :attr:`nonfinite_observations` and written to the calibration file, so a shard whose
    activations were partly NaN or infinite cannot later be merged as if it were healthy.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model (torch.nn.Module): Model whose target linears are observed.

        Raises:
            ValueError: If an expected target family is missing or a matched object is not a ``torch.nn.Linear``.
        """
        self.model = model
        self.fqns_by_suffix = _match_target_modules(model)
        self.target_fqns: List[str] = sorted(fqn for fqns in self.fqns_by_suffix.values() for fqn in fqns)
        self.activation_amax: Dict[str, float] = {}
        self.activation_amax_samples: Dict[str, List[float]] = {}
        self.nonfinite_observations: Dict[str, int] = {}
        self._handles: List[Any] = []

    @property
    def installed(self) -> bool:
        """Whether forward pre-hooks are currently installed."""
        return bool(self._handles)

    def install(self) -> None:
        """Install forward pre-hooks on every target module; repeated calls are no-ops."""
        if self._handles:
            return
        modules = dict(self.model.named_modules())
        for fqn in self.target_fqns:
            self._handles.append(modules[fqn].register_forward_pre_hook(self._make_hook(_canonical_fqn(fqn))))

    def remove(self) -> None:
        """Remove every installed hook; repeated calls are no-ops."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "ActivationAmaxCollector":
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()

    def save(
        self,
        calibration_path: str,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_sha256: Optional[str] = None,
    ) -> str:
        """
        Atomically write the collected maxima as a deterministic, versioned JSON file.

        The payload keeps the schema-v1 fields that :func:`load_calibration` consumes and adds the per-invocation
        sample histories under :data:`CALIBRATION_SAMPLES_FIELD`, the per-module non-finite invocation counts
        under :data:`CALIBRATION_NONFINITE_FIELD`, and, when given, the checkpoint identity under
        :data:`CALIBRATION_CHECKPOINT_FIELD`. The runtime loader ignores all three.

        Args:
            calibration_path (str): Destination path.
            overwrite (bool): Replace an existing file. Defaults to ``False``.
            metadata (Optional[Dict[str, Any]]): Extra JSON-safe metadata recorded alongside the maxima.
            checkpoint_sha256 (Optional[str]): SHA-256 of the checkpoint these statistics were collected on, as
                64 hexadecimal characters. Recording it lets :func:`merge_calibrations` refuse a merge across
                checkpoints; omitting it leaves this shard's checkpoint identity unverifiable.

        Returns:
            calibration_path (str): The path written.

        Raises:
            RuntimeError: If no activation was observed, or a target observed only non-finite activations.
            ValueError: If ``checkpoint_sha256`` is given but is not a 64-character hexadecimal digest.
            FileExistsError: If the destination exists and ``overwrite`` is ``False``.
        """
        if not self.activation_amax:
            raise RuntimeError(
                "No activations were observed, so no calibration file was written. The collector must be "
                "installed around a real forward pass."
            )
        missing = sorted({_canonical_fqn(fqn) for fqn in self.target_fqns} - set(self.activation_amax))
        if missing:
            logging.warning(
                f"Activation calibration observed {len(self.activation_amax)} of {len(self.target_fqns)} target "
                f"modules; {len(missing)} were never invoked, e.g. {missing[:4]}."
            )
        corrupt = sorted(set(self.nonfinite_observations) - set(self.activation_amax))
        if corrupt:
            raise RuntimeError(
                f"{len(corrupt)} target module(s) observed only non-finite activations, e.g. {corrupt[:4]}; no "
                "calibration file was written because their statistics cannot be recorded honestly."
            )
        payload = {
            "version": CALIBRATION_SCHEMA_VERSION,
            "recipe": "disabled",
            "scale_mode": "static",
            "targets": list(QUANTIZATION_TARGET_SUFFIXES),
            "metadata": _normalize_json(metadata or {}),
            "activation_amax": {fqn: float(self.activation_amax[fqn]) for fqn in sorted(self.activation_amax)},
            CALIBRATION_SAMPLES_FIELD: {
                fqn: [float(sample) for sample in self.activation_amax_samples[fqn]]
                for fqn in sorted(self.activation_amax_samples)
            },
            CALIBRATION_NONFINITE_FIELD: {
                fqn: int(self.nonfinite_observations.get(fqn, 0)) for fqn in sorted(self.activation_amax)
            },
        }
        if checkpoint_sha256 is not None:
            payload[CALIBRATION_CHECKPOINT_FIELD] = _validate_checkpoint_sha256(checkpoint_sha256, "checkpoint_sha256")
        return save_calibration(calibration_path, payload, overwrite=overwrite)

    def _make_hook(self, fqn: str):
        """Build a forward pre-hook that records the finite absolute maximum of the module input."""

        def hook(module, args):  # pylint: disable=unused-argument
            if not args:
                return None
            tensor = args[0]
            if not isinstance(tensor, torch.Tensor):
                return None
            values = tensor.detach().abs().float()
            finite = values[torch.isfinite(values)]
            if finite.numel() != values.numel():
                # The maximum keeps ignoring the non-finite elements, but the fact that they existed is recorded
                # before any early return, so it survives even for a module that only ever saw NaN or infinity.
                self.nonfinite_observations[fqn] = self.nonfinite_observations.get(fqn, 0) + 1
            values = finite
            if values.numel() == 0:
                return None
            amax = float(values.max().item())
            if amax <= 0:
                return None
            self.activation_amax[fqn] = max(amax, self.activation_amax.get(fqn, 0.0))
            self.activation_amax_samples.setdefault(fqn, []).append(amax)
            return None

        return hook


def save_calibration(calibration_path: str, payload: Dict[str, Any], overwrite: bool = False) -> str:
    """
    Atomically write a calibration payload to disk.

    Args:
        calibration_path (str): Destination path.
        payload (Dict[str, Any]): JSON-safe calibration payload.
        overwrite (bool): Replace an existing file. Defaults to ``False``.

    Returns:
        calibration_path (str): The path written.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
    """
    path = Path(calibration_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Calibration file {path} already exists. Set quantization_overwrite_calibration=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as tmp:
        temporary_path = Path(tmp.name)
        json.dump(_normalize_json(payload), tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path)


def merge_calibrations(
    inputs: Sequence[Tuple[str, str]], percentile: float, headroom: float, checkpoint_sha256: str
) -> Dict[str, Any]:
    """
    Merge labelled calibration files into one deterministic schema-v1 runtime artifact.

    Every input is assigned to a data source or streaming stratum (its *group*): a stable logical
    domain/microphone/geometry stratum, not an arbitrary label. Several partition files of the *same* stratum
    reuse one group; different microphone or array-geometry strata must be given different groups, because
    pooling them under one name lets the percentile discard the smaller stratum's larger activations.

    Within a group all per-invocation observations of an FQN are pooled and reduced by a conservative
    nearest-rank percentile ``sorted_values[ceil(P / 100 * n) - 1]``. The per-FQN result is the **maximum** over
    groups, so a high-volume source cannot average away a smaller domain's larger activations, multiplied by
    ``headroom``. The headroom is baked into the 124 written values, so the runtime consumes the artifact with
    ``quantization_scale_margin=1.0``.

    Activation scales are only meaningful for the exact weights that produced them, so the merge must be
    finalized against an explicit ``checkpoint_sha256``. Any input that declares its own
    :data:`CALIBRATION_CHECKPOINT_FIELD` must agree with it, which makes a cross-checkpoint merge of
    self-describing shards fail instead of succeeding silently.

    Inputs written before :data:`CALIBRATION_SAMPLES_FIELD` existed (for example the legacy DavidAI artifact)
    carry maxima only; each such value is treated as a single observation, and the fallback and the resulting
    unknown non-finite/checkpoint provenance are recorded in the output.

    Args:
        inputs (Sequence[Tuple[str, str]]): ``(group, path)`` pairs. Order does not affect the output bytes.
        percentile (float): Percentile in ``(0, 100]`` applied per group and FQN.
        headroom (float): Positive finite multiplier baked into every written value.
        checkpoint_sha256 (str): SHA-256 of the checkpoint every input was collected on, as 64 hexadecimal
            characters.

    Returns:
        payload (Dict[str, Any]): Deterministic schema-v1 calibration payload with sorted ``activation_amax``,
            the common target suffixes, and normalized merge provenance. Sample histories are not copied into it.

    Raises:
        ValueError: If ``inputs`` is empty, the percentile, headroom, or checkpoint digest is invalid, an input
            is malformed, has a different schema version, target list, FQN set, or checkpoint, carries an empty
            or invalid sample history, reports a non-finite observation, or produces a non-finite merged value.
        OSError: If an input cannot be read.
    """
    percentile = _validate_percentile(percentile)
    headroom = _positive_finite(headroom, "headroom")
    checkpoint_sha256 = _validate_checkpoint_sha256(checkpoint_sha256, "checkpoint_sha256")
    entries = [_read_merge_input(group, path) for group, path in inputs]
    if not entries:
        raise ValueError("At least one calibration input is required to merge, but none were given.")

    reference = entries[0]
    for entry in entries:
        declared = entry["checkpoint_sha256"]
        if declared is not None and declared != checkpoint_sha256:
            raise ValueError(
                f"Calibration file {entry['path']} was collected on checkpoint {declared}, but the merge is "
                f"finalized against {checkpoint_sha256}; activation scales are only valid for the checkpoint "
                "that produced them."
            )
    for entry in entries[1:]:
        if entry["targets"] != reference["targets"]:
            raise ValueError(
                f"Calibration file {entry['path']} declares targets {list(entry['targets'])}, but "
                f"{reference['path']} declares {list(reference['targets'])}; merged inputs must share targets."
            )
        if entry["fqns"] != reference["fqns"]:
            only_here = sorted(set(entry["fqns"]) - set(reference["fqns"]))
            only_there = sorted(set(reference["fqns"]) - set(entry["fqns"]))
            raise ValueError(
                f"Calibration file {entry['path']} does not cover the same modules as {reference['path']}: "
                f"extra {only_here[:4]}, missing {only_there[:4]}. Re-run calibration against the same checkpoint."
            )

    samples_by_group: Dict[str, Dict[str, List[float]]] = {}
    inputs_by_group: Dict[str, int] = {}
    for entry in entries:
        group_samples = samples_by_group.setdefault(entry["group"], {})
        inputs_by_group[entry["group"]] = inputs_by_group.get(entry["group"], 0) + 1
        for fqn, values in entry["samples"].items():
            group_samples.setdefault(fqn, []).extend(values)

    groups = sorted(samples_by_group)
    activation_amax: Dict[str, float] = {}
    for fqn in reference["fqns"]:
        envelope = max(_nearest_rank_percentile(samples_by_group[group][fqn], percentile) for group in groups)
        value = envelope * headroom
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Merged activation amax for '{fqn}' is not finite and positive ({value!r}); refusing to write a "
                "calibration file that cannot be used as a static scale."
            )
        activation_amax[fqn] = value

    provenance = {
        "method": CALIBRATION_MERGE_METHOD,
        "method_version": CALIBRATION_MERGE_METHOD_VERSION,
        "percentile": percentile,
        "percentile_rule": "sorted_values[ceil(percentile / 100 * n) - 1] per group and module",
        "group_reduction": "max",
        "headroom": headroom,
        "headroom_baked_in": True,
        "runtime_scale_margin": 1.0,
        "target_module_count": len(reference["fqns"]),
        "checkpoint_sha256": checkpoint_sha256,
        # True when at least one input did not declare its own checkpoint, so its identity rests on the caller's
        # assertion alone rather than on the file's own record.
        "checkpoint_identity_asserted_only": any(entry["checkpoint_sha256"] is None for entry in entries),
        "nonfinite_status": (
            CALIBRATION_PROVENANCE_CLEAN
            if all(entry["nonfinite_status"] == CALIBRATION_PROVENANCE_CLEAN for entry in entries)
            else CALIBRATION_PROVENANCE_UNKNOWN_LEGACY
        ),
        "groups": groups,
        "group_semantics": "one group per logical domain/microphone/geometry stratum",
        "group_statistics": {
            group: _group_statistics(samples_by_group[group], reference["fqns"], inputs_by_group[group])
            for group in groups
        },
        "inputs": sorted(
            (_input_provenance(entry) for entry in entries),
            key=lambda summary: (summary["group"], summary["name"], summary["sha256"]),
        ),
        "legacy_max_only_fallback": any(entry["legacy_fallback"] for entry in entries),
    }
    return _normalize_json(
        {
            "version": CALIBRATION_SCHEMA_VERSION,
            "recipe": "disabled",
            "scale_mode": "static",
            "targets": list(reference["targets"]),
            "metadata": provenance,
            "activation_amax": {fqn: activation_amax[fqn] for fqn in sorted(activation_amax)},
        }
    )


def merge_calibration_files(
    inputs: Sequence[Tuple[str, str]],
    percentile: float,
    headroom: float,
    checkpoint_sha256: str,
    output_path: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Merge labelled calibration files and atomically write the result.

    Args:
        inputs (Sequence[Tuple[str, str]]): ``(group, path)`` pairs, as for :func:`merge_calibrations`.
        percentile (float): Percentile in ``(0, 100]``.
        headroom (float): Positive finite multiplier baked into every written value.
        checkpoint_sha256 (str): SHA-256 of the checkpoint every input was collected on.
        output_path (str): Destination path.
        overwrite (bool): Replace an existing destination. Defaults to ``False``.

    Returns:
        payload (Dict[str, Any]): The payload that was written.

    Raises:
        ValueError: If any input or option is invalid.
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
    """
    payload = merge_calibrations(inputs, percentile, headroom, checkpoint_sha256)
    save_calibration(output_path, payload, overwrite=overwrite)
    return payload


def ensure_calibration_output_writable(config: SortformerQuantizationConfig) -> None:
    """
    Fail fast, before inference, when the calibration destination cannot be written.

    Args:
        config (SortformerQuantizationConfig): Quantization options.

    Raises:
        FileExistsError: If the destination exists and ``overwrite_calibration`` is ``False``.
    """
    if not config.calibration_output:
        return
    path = Path(config.calibration_output).expanduser()
    if path.exists() and not config.overwrite_calibration:
        raise FileExistsError(
            f"Calibration file {path} already exists. Set quantization_overwrite_calibration=True to replace it."
        )


def prediction_cache_identity(config: SortformerQuantizationConfig) -> Optional[Dict[str, Any]]:
    """
    Describe the quantization settings that make cached predictions non-interchangeable.

    Predictions produced by a quantized run are numerically different from BF16 ones, so a prediction cache must
    not be shared between them. Adding this to the evaluator's cache metadata turns such a reuse into the
    existing actionable cache-mismatch error instead of a silently mislabelled DER.

    A disabled recipe returns ``None`` rather than a dict, so a BF16 run keeps matching prediction caches written
    before quantization existed, while a quantized run never matches them.

    Static runs also record the calibration file's size and modification time, mirroring how the evaluator
    identifies the checkpoint and the manifest: re-collecting calibration to the same path changes the activation
    scales, and therefore the predictions, without changing any other cache-metadata field.

    Global-scale folding and its activation exponent are recorded for the same reason: a folded run is
    numerically different from the matched unfolded static NVFP4 run, and each exponent of a sweep produces its
    own predictions, so none of them may share a cache entry.

    Producer-packing fusion is recorded for the same reason: the fused LayerNorm/GELU packs round to BF16 exactly
    where the unfused path does, but the two are not bit-identical, so a fused run may not reuse an unfused cache
    entry.

    A BF16 override is recorded by path, file identity *and* content digest: two sensitivity runs differ only in
    the contents of that one file, and each of them assigns a different precision to the same modules, so a cache
    entry may never be shared between them. The digest is read from disk here, which makes a missing or unreadable
    override file fail closed instead of collapsing two different runs onto the same identity.

    The block-MSE weight-scale search is recorded by method *and* by algorithm version, but only when it is on:
    a searched run stores different weights than the matched amax run, so the two may never share a cache entry,
    while the ordinary amax identity stays exactly what it was before the search existed and keeps matching
    prediction caches written by earlier amax runs.

    Four-Over-Six is recorded the same way, with its normalization maximum and candidate magnitudes alongside the
    algorithm version: those two constants define the arithmetic, so a build that changed either would store
    different weights under the same method name and must not reuse this one's predictions.

    AWQ-clip is recorded by method, algorithm version, ratio list, block size and tie rule, by the artifact's path,
    file identity and exact content digest, *and* by the content digest of the activation-calibration file: the
    codes were selected against activations quantized with that file's values, so two runs that differ in either
    file execute different weights or different activations and may never share a cache entry.

    GPTQ is recorded the same way, with its damping, update block and NVFP4 block alongside the algorithm version:
    the artifact carries the packed payload itself, so two artifacts are two different sets of stored weights, and
    the calibration digest separates two payloads selected under Hessians formed from different activations.

    Args:
        config (SortformerQuantizationConfig): Validated quantization options.

    Returns:
        identity (Optional[Dict[str, Any]]): Cache-identity fields, or ``None`` for a disabled recipe.

    Raises:
        ValueError: If a BF16 override file was requested but cannot be read.
    """
    if not config.enabled:
        return None
    override_path = None if not config.has_bf16_override else str(Path(config.bf16_override_path).expanduser())
    weight_scale_identity: Dict[str, Any] = {}
    if config.uses_mse_weight_scales:
        # Added only when the search is on, so the default amax identity stays byte-for-byte what it was before
        # the search existed and keeps matching prediction caches written by earlier amax runs.
        weight_scale_identity = {
            "weight_scale_method": WEIGHT_SCALE_METHOD_MSE,
            "weight_scale_mse_algorithm": WEIGHT_SCALE_MSE_ALGORITHM,
            "weight_scale_mse_algorithm_version": WEIGHT_SCALE_MSE_ALGORITHM_VERSION,
        }
    if config.uses_four_over_six_weight_scales:
        # The normalization maximum and the candidate magnitudes are part of the identity, not of the report only:
        # a future build that changed either would store different weights under the same method name.
        weight_scale_identity = {
            "weight_scale_method": WEIGHT_SCALE_METHOD_FOUR_OVER_SIX,
            "weight_scale_four_over_six_algorithm": WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM,
            "weight_scale_four_over_six_algorithm_version": WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION,
            "weight_scale_four_over_six_fp8_max": float(WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX),
            "weight_scale_four_over_six_magnitudes": [
                int(magnitude) for magnitude in WEIGHT_SCALE_FOUR_OVER_SIX_MAGNITUDES
            ],
        }
    if config.uses_awq_clip_weight_scales:
        # Two AWQ-clip runs differ only in the codes their artifact carries *and* in the calibration file those
        # codes were selected against, and each combination stores different weights or quantizes different
        # activations. Both digests are therefore read from disk here, which also makes a missing or unreadable
        # file fail closed instead of collapsing two different runs onto one identity.
        awq_clip_path = str(Path(config.weight_scale_awq_clip_path).expanduser())
        weight_scale_identity = {
            "weight_scale_method": WEIGHT_SCALE_METHOD_AWQ_CLIP,
            "weight_scale_awq_clip_algorithm": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM,
            "weight_scale_awq_clip_algorithm_version": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION,
            "weight_scale_awq_clip_block_size": int(WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE),
            "weight_scale_awq_clip_ratios": [float(ratio) for ratio in WEIGHT_SCALE_AWQ_CLIP_RATIOS],
            "weight_scale_awq_clip_tie_rule": AWQ_CLIP_TIE_RULE,
            "weight_scale_awq_clip_path": awq_clip_path,
            "weight_scale_awq_clip_file": _file_identity(awq_clip_path),
            "weight_scale_awq_clip_sha256": _file_content_sha256(awq_clip_path, "AWQ-clip artifact"),
            "weight_scale_awq_clip_calibration_sha256": _file_content_sha256(
                config.calibration_path, "Activation calibration file"
            ),
        }
    if config.uses_gptq_weight_scales:
        # A GPTQ run is separated from every other run by the payload bytes its artifact carries *and* by the
        # calibration file whose values quantized the activations its Hessians were formed from. Both digests are
        # read from disk here, which also makes a missing or unreadable file fail closed instead of collapsing two
        # different runs onto one identity. The ordinary-template construction the payload was written under is
        # already separated by ``accelerated_packing`` below, which is what decides that construction.
        gptq_path = str(Path(config.weight_scale_gptq_path).expanduser())
        weight_scale_identity = {
            "weight_scale_method": WEIGHT_SCALE_METHOD_GPTQ,
            "weight_scale_gptq_algorithm": WEIGHT_SCALE_GPTQ_ALGORITHM,
            "weight_scale_gptq_algorithm_version": WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION,
            "weight_scale_gptq_perc_damp": float(WEIGHT_SCALE_GPTQ_PERC_DAMP),
            "weight_scale_gptq_update_block_size": int(WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE),
            "weight_scale_gptq_block_size": int(WEIGHT_SCALE_GPTQ_BLOCK_SIZE),
            "weight_scale_gptq_path": gptq_path,
            "weight_scale_gptq_file": _file_identity(gptq_path),
            "weight_scale_gptq_sha256": _file_content_sha256(gptq_path, "GPTQ artifact"),
            "weight_scale_gptq_calibration_sha256": _file_content_sha256(
                config.calibration_path, "Activation calibration file"
            ),
        }
    if config.uses_local_hessian_weight_scales:
        # The artifact's exact content digest is part of the identity: two runs of this method differ only in the
        # moments that file carries, and each of them stores different weights. Reading it here also makes a
        # missing or unreadable artifact fail closed instead of collapsing two runs onto one identity.
        hessian_path = str(Path(config.weight_scale_hessian_path).expanduser())
        weight_scale_identity = {
            "weight_scale_method": WEIGHT_SCALE_METHOD_LOCAL_HESSIAN,
            "weight_scale_hessian_algorithm": WEIGHT_SCALE_HESSIAN_ALGORITHM,
            "weight_scale_hessian_algorithm_version": WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION,
            "weight_scale_hessian_damping": float(WEIGHT_SCALE_HESSIAN_DAMPING),
            "weight_scale_hessian_path": hessian_path,
            "weight_scale_hessian_file": _file_identity(hessian_path),
            "weight_scale_hessian_sha256": _file_content_sha256(hessian_path, "Diagonal-Hessian artifact"),
        }
    return _normalize_json(
        {
            **weight_scale_identity,
            "recipe": config.recipe,
            "scale_mode": config.scale_mode if config.uses_activation_quantization else "not_applicable",
            "scale_margin": float(config.scale_margin) if config.scale_mode == "static" else None,
            "calibration_path": config.calibration_path,
            "calibration_file": _file_identity(config.calibration_path),
            "accelerated_packing": bool(config.accelerated_packing),
            "fold_global_scales": bool(config.fold_global_scales),
            "fold_activation_exponent": config.fold_activation_exponent if config.fold_global_scales else None,
            "fuse_producer_packing": bool(config.fuse_producer_packing),
            "bf16_override_path": override_path,
            "bf16_override_file": _file_identity(override_path),
            "bf16_override_sha256": _file_content_sha256(override_path, "BF16 override file"),
        }
    )


def validate_calibration_forward_pass(config: SortformerQuantizationConfig, prediction_cache_reused: bool) -> None:
    """
    Reject calibration runs that would reuse cached predictions instead of running a forward pass.

    Args:
        config (SortformerQuantizationConfig): Quantization options.
        prediction_cache_reused (bool): Whether the evaluator is about to load cached prediction tensors.

    Raises:
        ValueError: If calibration output was requested but no forward pass will run.
    """
    if config.calibration_output and prediction_cache_reused:
        raise ValueError(
            "Activation calibration requires a real forward pass, but cached prediction tensors are about to be "
            "reused. Set overwrite_preds_tensors=True, clear out_preds_tensors, or point out_preds_tensors at an "
            "unused path."
        )


def _match_target_modules(model: torch.nn.Module) -> Dict[str, List[str]]:
    """
    Group the model's target-family FQNs by target suffix, rejecting missing or unsupported targets.

    Args:
        model (torch.nn.Module): Model to search.

    Returns:
        fqns_by_suffix (Dict[str, List[str]]): Sorted FQNs for each entry of
            :data:`QUANTIZATION_TARGET_SUFFIXES`.

    Raises:
        ValueError: If a target family has no match, or a matched object is not a ``torch.nn.Linear``.
    """
    fqns_by_suffix: Dict[str, List[str]] = {suffix: [] for suffix in QUANTIZATION_TARGET_SUFFIXES}
    unsupported: List[str] = []
    for name, module in model.named_modules():
        suffix = _matched_suffix(name)
        if suffix is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            unsupported.append(f"{name} ({type(module).__name__})")
            continue
        fqns_by_suffix[suffix].append(name)

    if unsupported:
        raise ValueError(
            "Sortformer quantization targets must be torch.nn.Linear modules, but these matches are not: "
            f"{sorted(unsupported)}"
        )
    missing = [suffix for suffix, fqns in fqns_by_suffix.items() if not fqns]
    if missing:
        raise ValueError(
            f"Model does not contain the expected Sortformer quantization targets {missing}; refusing to "
            "silently quantize a partial set of layers."
        )
    return {suffix: sorted(fqns) for suffix, fqns in fqns_by_suffix.items()}


def _matched_suffix(fqn: str) -> Optional[str]:
    """Return the target suffix that anchors the end of ``fqn``, or ``None`` when it matches no target."""
    for suffix in QUANTIZATION_TARGET_SUFFIXES:
        if fqn == suffix or fqn.endswith("." + suffix):
            return suffix
    return None


def _canonical_fqn(fqn: str) -> str:
    """
    Drop ``torch.compile`` wrapper segments from an FQN.

    ``torch.compile`` re-parents a module under ``_orig_mod``, so a calibration run made with
    ``compile_encoder=True`` would otherwise record names carrying a ``._orig_mod`` segment, which do not exist
    in the uncompiled module tree that quantization is applied to (quantization always runs before
    compilation). Recording the uncompiled name keeps calibration files portable across the two invocations.
    """
    if "_orig_mod" not in fqn:
        return fqn
    return ".".join(part for part in fqn.split(".") if part != "_orig_mod")


def _precision_by_suffix(recipe: str) -> Dict[str, str]:
    """Map each target family to the precision it receives under the given recipe."""
    qkv, rest = "attn.w_qkv", ("attn.out_proj", "ffn.net.0", "ffn.net.3")
    if recipe == "disabled":
        return {suffix: PRECISION_BF16 for suffix in QUANTIZATION_TARGET_SUFFIXES}
    if recipe == "nvfp4_all":
        return {suffix: PRECISION_NVFP4_W4A4 for suffix in QUANTIZATION_TARGET_SUFFIXES}
    if recipe == "nvfp4_weight_only":
        return {suffix: PRECISION_NVFP4_WEIGHT_ONLY for suffix in QUANTIZATION_TARGET_SUFFIXES}
    if recipe == "nvfp4_qkv_only":
        return {qkv: PRECISION_NVFP4_W4A4, **{suffix: PRECISION_BF16 for suffix in rest}}
    if recipe == "nvfp4_qkv_fp8_rest":
        return {qkv: PRECISION_NVFP4_W4A4, **{suffix: PRECISION_FP8_DYNAMIC for suffix in rest}}
    raise ValueError(f"quantization recipe must be one of {list(QUANTIZATION_RECIPES)}, got '{recipe}'")


def _required_apis(config: SortformerQuantizationConfig) -> List[str]:
    """TorchAO entry points that the requested recipe and scale mode need."""
    apis = [TORCHAO_QUANTIZE_API]
    if config.recipe == "nvfp4_weight_only":
        apis.append(TORCHAO_NVFP4_WEIGHT_ONLY_CONFIG_API)
        return apis
    apis.append(TORCHAO_NVFP4_DYNAMIC_CONFIG_API)
    if config.scale_mode == "static":
        apis.append(TORCHAO_NVFP4_OBSERVED_LINEAR_API)
    if config.recipe == "nvfp4_qkv_fp8_rest":
        apis.append(TORCHAO_FP8_CONFIG_API)
    return apis


def _apply_nvfp4_activation_quantization(
    model: torch.nn.Module,
    nvfp4_fqns: Sequence[str],
    backend: str,
    calibration: Optional[Dict[str, Any]],
    quantize_,
    repack_plan: Optional["_WeightRepackPlan"] = None,
) -> List[Dict[str, Any]]:
    """
    Convert the selected linears to NVFP4 W4A4, using either dynamic or calibrated static activation scales.

    Without a repack plan -- ``weight_scale_method='amax'`` -- the conversion is the ordinary batched TorchAO one.
    With one, the selected FQNs are converted one at a time in sorted order and each converted weight is immediately
    repacked by exactly the named repacker, so at most one high-precision weight clone is alive at any moment; the
    calibrated activation amax values are still assigned to every observed linear before the first conversion.

    Args:
        model (torch.nn.Module): Model to convert in place.
        nvfp4_fqns (Sequence[str]): FQNs selected for NVFP4 W4A4.
        backend (str): Resolved backend label; controls the TorchAO Triton-kernel flag.
        calibration (Optional[Dict[str, Any]]): Validated calibration payload for ``scale_mode='static'``.
        quantize_ (Callable): The resolved ``torchao.quantization.quantize_`` entry point.
        repack_plan (Optional[_WeightRepackPlan]): The single named repacker to apply to every converted weight,
            or ``None`` for the ordinary batched amax conversion.

    Returns:
        entries (List[Dict[str, Any]]): One per-FQN report entry per repacked weight; empty for ``'amax'``.

    Raises:
        RuntimeError: If an observed linear does not expose an assignable activation amax, or a converted weight
            cannot be repacked by the requested method.
    """
    nvfp4_config_cls = _require_api(TORCHAO_NVFP4_DYNAMIC_CONFIG_API)
    use_triton_kernel = backend == BACKEND_MSLK_ACCELERATED
    selected = _fqn_filter(nvfp4_fqns)

    if calibration is None:
        dynamic_config = nvfp4_config_cls(
            use_triton_kernel=use_triton_kernel, use_dynamic_per_tensor_scale=True, step=None
        )
        if repack_plan is None:
            quantize_(model, dynamic_config, filter_fn=selected)
            return []
        return _convert_and_repack_each(
            model,
            nvfp4_fqns,
            dynamic_config,
            quantize_,
            lambda fqn: _fqn_filter([fqn]),
            "dynamic NVFP4 W4A4",
            repack_plan,
        )

    prepare_config = nvfp4_config_cls(
        use_triton_kernel=use_triton_kernel, use_dynamic_per_tensor_scale=True, step="prepare"
    )
    quantize_(model, prepare_config, filter_fn=selected)
    observed_linear_cls = _require_api(TORCHAO_NVFP4_OBSERVED_LINEAR_API)
    modules = dict(model.named_modules())
    activation_amax = calibration["activation_amax"]
    for fqn in sorted(activation_amax):
        module = modules.get(fqn)
        if module is None or not isinstance(module, observed_linear_cls):
            raise RuntimeError(
                f"Static NVFP4 preparation did not produce an observed linear at '{fqn}'; refusing to convert "
                "with an unvalidated activation scale."
            )
        _assign_activation_amax(module, fqn, activation_amax[fqn])

    convert_config = nvfp4_config_cls(
        use_triton_kernel=use_triton_kernel, use_dynamic_per_tensor_scale=True, step="convert"
    )
    if repack_plan is not None:
        return _convert_and_repack_each(
            model,
            nvfp4_fqns,
            convert_config,
            quantize_,
            lambda fqn: (lambda module, name: name == fqn and isinstance(module, observed_linear_cls)),
            "static NVFP4 W4A4",
            repack_plan,
        )
    converted = set(nvfp4_fqns)
    quantize_(
        model,
        convert_config,
        filter_fn=lambda module, fqn: fqn in converted and isinstance(module, observed_linear_cls),
    )
    return []


def _apply_nvfp4_weight_only_quantization(
    model: torch.nn.Module,
    weight_only_fqns: Sequence[str],
    weight_only_config: Any,
    quantize_,
    repack_plan: Optional["_WeightRepackPlan"] = None,
) -> List[Dict[str, Any]]:
    """
    Convert the selected linears to NVFP4 weight-only storage, batched or one FQN at a time under a repack plan.

    Args:
        model (torch.nn.Module): Model to convert in place.
        weight_only_fqns (Sequence[str]): FQNs selected for NVFP4 weight-only storage.
        weight_only_config (Any): The already-built TorchAO ``NVFP4WeightOnlyConfig``.
        quantize_ (Callable): The resolved ``torchao.quantization.quantize_`` entry point.
        repack_plan (Optional[_WeightRepackPlan]): The single named repacker to apply to every converted weight,
            or ``None`` for the ordinary batched amax conversion.

    Returns:
        entries (List[Dict[str, Any]]): One per-FQN report entry per repacked weight; empty for ``'amax'``.
    """
    if repack_plan is None:
        quantize_(model, weight_only_config, filter_fn=_fqn_filter(weight_only_fqns))
        return []
    return _convert_and_repack_each(
        model,
        weight_only_fqns,
        weight_only_config,
        quantize_,
        lambda fqn: _fqn_filter([fqn]),
        "NVFP4 weight-only",
        repack_plan,
    )


@dataclass(frozen=True)
class _WeightRepackPlan:
    """The single weight repacker a run applies, named explicitly so no branch can infer it from other data."""

    method: str
    # Only ever set for :data:`WEIGHT_SCALE_METHOD_LOCAL_HESSIAN`, whose repacker needs one moment vector per FQN.
    hessian: Optional[Dict[str, Any]] = None
    # Only ever set for :data:`WEIGHT_SCALE_METHOD_AWQ_CLIP`, whose repacker needs one ratio-code matrix per FQN.
    awq_clip: Optional[Dict[str, Any]] = None
    # Only ever set for :data:`WEIGHT_SCALE_METHOD_GPTQ`, whose repacker needs one packed payload per FQN.
    gptq: Optional[Dict[str, Any]] = None


def _weight_repack_plan(
    config: SortformerQuantizationConfig,
    hessian: Optional[Dict[str, Any]],
    awq_clip: Optional[Dict[str, Any]] = None,
    gptq: Optional[Dict[str, Any]] = None,
) -> Optional[_WeightRepackPlan]:
    """
    Resolve the requested weight-scale method into exactly one repacker, or ``None`` for the ordinary amax path.

    Args:
        config (SortformerQuantizationConfig): Validated quantization options.
        hessian (Optional[Dict[str, Any]]): Validated diagonal-Hessian artifact, required by ``'local_hessian'``.
        awq_clip (Optional[Dict[str, Any]]): Validated AWQ-clip artifact, required by ``'awq_clip'``.
        gptq (Optional[Dict[str, Any]]): Validated GPTQ artifact, required by ``'gptq'``.

    Returns:
        plan (Optional[_WeightRepackPlan]): The named repacker, or ``None`` when TorchAO's batched amax conversion
            is what runs.

    Raises:
        RuntimeError: If ``'local_hessian'``, ``'awq_clip'`` or ``'gptq'`` was requested without its loaded
            artifact, or the method is not one this dispatch implements. None of those is reachable through a
            validated config, and all of them fail closed rather than silently selecting another repacker.
    """
    if config.weight_scale_method == WEIGHT_SCALE_METHOD_AMAX:
        return None
    if config.weight_scale_method == WEIGHT_SCALE_METHOD_MSE:
        return _WeightRepackPlan(method=WEIGHT_SCALE_METHOD_MSE)
    if config.weight_scale_method == WEIGHT_SCALE_METHOD_FOUR_OVER_SIX:
        return _WeightRepackPlan(method=WEIGHT_SCALE_METHOD_FOUR_OVER_SIX)
    if config.weight_scale_method == WEIGHT_SCALE_METHOD_LOCAL_HESSIAN:
        if hessian is None:
            raise RuntimeError(
                f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_LOCAL_HESSIAN}' reached the conversion "
                "without a loaded diagonal-Hessian artifact; refusing to repack with another method's scales."
            )
        return _WeightRepackPlan(method=WEIGHT_SCALE_METHOD_LOCAL_HESSIAN, hessian=hessian)
    if config.weight_scale_method == WEIGHT_SCALE_METHOD_AWQ_CLIP:
        if awq_clip is None:
            raise RuntimeError(
                f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_AWQ_CLIP}' reached the conversion without "
                "a loaded AWQ-clip artifact; refusing to repack with another method's scales."
            )
        return _WeightRepackPlan(method=WEIGHT_SCALE_METHOD_AWQ_CLIP, awq_clip=awq_clip)
    if config.weight_scale_method == WEIGHT_SCALE_METHOD_GPTQ:
        if gptq is None:
            raise RuntimeError(
                f"quantization_weight_scale_method='{WEIGHT_SCALE_METHOD_GPTQ}' reached the conversion without a "
                "loaded GPTQ artifact; refusing to repack with another method's scales."
            )
        return _WeightRepackPlan(method=WEIGHT_SCALE_METHOD_GPTQ, gptq=gptq)
    raise RuntimeError(
        f"quantization_weight_scale_method='{config.weight_scale_method}' has no weight repacker in this build; "
        f"the implemented methods are {list(WEIGHT_SCALE_METHODS)}."
    )


def _convert_and_repack_each(
    model: torch.nn.Module,
    fqns: Sequence[str],
    torchao_config: Any,
    quantize_,
    filter_for,
    description: str,
    repack_plan: "_WeightRepackPlan",
) -> List[Dict[str, Any]]:
    """
    Convert and repack exactly one selected FQN at a time, in stable sorted order.

    The high-precision weight of a module is cloned immediately before that module is converted and released
    immediately after it has been repacked and measured, so the model is never held twice: at most one original
    weight clone is alive at any point, whatever the size of the selected set. Under ``'local_hessian'`` the same
    holds for the device copy of that FQN's second-moment vector, which is the only other per-FQN allocation.

    Args:
        model (torch.nn.Module): Model to convert in place.
        fqns (Sequence[str]): FQNs to convert and repack.
        torchao_config (Any): The TorchAO quantization config to apply to each FQN.
        quantize_ (Callable): The resolved ``torchao.quantization.quantize_`` entry point.
        filter_for (Callable): Builds the single-FQN ``quantize_`` filter for one FQN.
        description (str): Conversion path named in error messages.
        repack_plan (_WeightRepackPlan): The single named repacker applied to every converted weight.

    Returns:
        entries (List[Dict[str, Any]]): One per-FQN report entry, in the same sorted order.
    """
    entries: List[Dict[str, Any]] = []
    for fqn in sorted(fqns):
        original_weight = _clone_original_weight(model, fqn, description)
        quantize_(model, torchao_config, filter_fn=filter_for(fqn))
        entries.append(_repack_one_weight(model, fqn, original_weight, description, repack_plan))
        # Released before the next FQN is cloned, which is what bounds the peak memory of the whole pass.
        del original_weight
    return entries


def _repack_one_weight(
    model: torch.nn.Module,
    fqn: str,
    original_weight: torch.Tensor,
    description: str,
    repack_plan: "_WeightRepackPlan",
) -> Dict[str, Any]:
    """Apply exactly the plan's repacker to one just-converted weight, or fail closed on an unimplemented method."""
    if repack_plan.method == WEIGHT_SCALE_METHOD_MSE:
        return _repack_weight_with_mse(model, fqn, original_weight, description)
    if repack_plan.method == WEIGHT_SCALE_METHOD_FOUR_OVER_SIX:
        return _repack_weight_with_four_over_six(model, fqn, original_weight, description)
    if repack_plan.method == WEIGHT_SCALE_METHOD_LOCAL_HESSIAN:
        return _repack_weight_with_local_hessian(
            model, fqn, original_weight, description, repack_plan.hessian["second_moments"][fqn]
        )
    if repack_plan.method == WEIGHT_SCALE_METHOD_AWQ_CLIP:
        return _repack_weight_with_awq_clip(
            model,
            fqn,
            original_weight,
            description,
            repack_plan.awq_clip["ratio_codes"][fqn],
            repack_plan.awq_clip["code_shapes"][fqn],
            repack_plan.awq_clip["template_arithmetic"],
        )
    if repack_plan.method == WEIGHT_SCALE_METHOD_GPTQ:
        return _repack_weight_with_gptq(model, fqn, original_weight, description, repack_plan.gptq)
    raise RuntimeError(
        f"No NVFP4 weight repacker is implemented for weight_scale_method='{repack_plan.method}'; refusing to "
        "leave the converted weight in place under another method's name."
    )


def _clone_original_weight(model: torch.nn.Module, fqn: str, description: str) -> torch.Tensor:
    """Detached contiguous copy of one module's still-high-precision weight, taken just before its conversion."""
    module = _module_by_fqn(model, fqn, description)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(
            f"{description} module '{fqn}' ({type(module).__name__}) does not expose a 'weight' tensor, so the "
            "NVFP4 block-MSE weight-scale search has no original weight to search against."
        )
    return weight.detach().clone().contiguous()


def _repack_weight_with_mse(
    model: torch.nn.Module, fqn: str, original_weight: torch.Tensor, description: str
) -> Dict[str, Any]:
    """
    Replace one converted NVFP4 weight with its MSE-searched repack and return the per-FQN evidence.

    Everything here fails closed: an unconverted or incompatible weight wrapper, a non-finite reconstruction, a
    searched MSE worse than the converted one beyond the documented tolerance, and an assignment that does not
    preserve the NVFP4 ``Parameter`` all raise instead of leaving the amax-derived weight silently in place.

    Args:
        model (torch.nn.Module): Model holding the just-converted module.
        fqn (str): FQN of the converted module.
        original_weight (torch.Tensor): The high-precision weight this module was converted from.
        description (str): Conversion path named in error messages.

    Returns:
        entry (Dict[str, Any]): FQN, shape, weight count, both reconstruction MSEs, and their ratio/reduction.

    Raises:
        RuntimeError: On any violated contract above.
    """
    module = _module_by_fqn(model, fqn, description)
    template = getattr(module, "weight", None)
    _require_nvfp4_wrapper(
        template,
        f"{description} conversion of '{fqn}' left {{kind}} weight instead of a TorchAO NVFP4 tensor, so the "
        "block-MSE weight-scale search cannot repack it.",
    )

    # Any incompatible wrapper, shape, dtype or metadata is rejected by the packer itself, which never falls back
    # to the template's own scales.
    repacked = repack_nvfp4_weight_mse(original_weight, template)
    _require_nvfp4_wrapper(
        repacked,
        f"The block-MSE repack of '{fqn}' produced {{kind}} weight instead of a TorchAO NVFP4 tensor; it cannot "
        "replace the converted one.",
    )

    reference = original_weight.detach().to(torch.float32)
    template_mse = _reconstruction_mse(template, reference, fqn, "converted")
    searched_mse = _reconstruction_mse(repacked, reference, fqn, "repacked")
    tolerated = template_mse * (1.0 + WEIGHT_SCALE_MSE_RELATIVE_TOLERANCE) + WEIGHT_SCALE_MSE_ABSOLUTE_TOLERANCE
    if searched_mse > tolerated:
        raise RuntimeError(
            f"The block-MSE weight-scale search made '{fqn}' worse: reconstruction MSE {searched_mse!r} against "
            f"the converted {template_mse!r}. The search is exhaustive over the encodings the amax rule also uses, "
            "so this is a defect, not a tolerable regression."
        )

    module.weight = torch.nn.Parameter(repacked, requires_grad=False)
    assigned = module.weight
    if not isinstance(assigned, torch.nn.Parameter) or not isinstance(assigned, type(template)):
        raise RuntimeError(
            f"Assigning the repacked NVFP4 weight of '{fqn}' produced a {type(assigned).__name__} instead of an "
            f"{type(template).__name__} Parameter; the module would no longer execute the searched weight."
        )

    ratio, reduction = _mse_ratio_and_reduction(template_mse, searched_mse)
    return {
        "fqn": fqn,
        "shape": [int(size) for size in original_weight.shape],
        "weight_count": int(original_weight.numel()),
        "template_mse": template_mse,
        "searched_mse": searched_mse,
        "ratio": ratio,
        "relative_reduction": reduction,
    }


def _repack_weight_with_local_hessian(
    model: torch.nn.Module,
    fqn: str,
    original_weight: torch.Tensor,
    description: str,
    second_moments: Sequence[float],
) -> Dict[str, Any]:
    """
    Replace one converted NVFP4 weight with its activation-weighted repack and return the per-FQN evidence.

    The moment vector is materialized on the original weight's device for this one module and released with it, so
    the artifact is only ever resident as Python floats plus one live device vector. Everything fails closed: an
    unconverted or incompatible weight wrapper, a moment vector the packer refuses, a non-finite reconstruction, a
    weighted objective worse than the converted one beyond the documented tolerance, and an assignment that does
    not preserve the NVFP4 ``Parameter`` all raise instead of leaving the amax-derived weight silently in place.

    The unweighted reconstruction MSE is measured before and after as *diagnostic evidence only*: the weighted
    search is free to spend unweighted error on channels the activations never excite, so a larger unweighted MSE
    is an expected outcome here and is reported rather than rejected.

    Args:
        model (torch.nn.Module): Model holding the just-converted module.
        fqn (str): FQN of the converted module.
        original_weight (torch.Tensor): The high-precision weight this module was converted from.
        description (str): Conversion path named in error messages.
        second_moments (Sequence[float]): Validated input-channel second moments of this module.

    Returns:
        entry (Dict[str, Any]): FQN, shape, weight count, both weighted objectives -- absolute, in the units of the
            artifact's second moments -- with their ratio/reduction, and both unweighted reconstruction MSEs.

    Raises:
        RuntimeError: On any violated contract above.
        ValueError: If the moment vector does not describe this weight.
    """
    module = _module_by_fqn(model, fqn, description)
    template = getattr(module, "weight", None)
    _require_nvfp4_wrapper(
        template,
        f"{description} conversion of '{fqn}' left {{kind}} weight instead of a TorchAO NVFP4 tensor, so the "
        "activation-weighted weight-scale search cannot repack it.",
    )

    moments = torch.tensor(list(second_moments), dtype=torch.float32, device=original_weight.device)
    try:
        # Any incompatible wrapper, shape, dtype, device or metadata is rejected by the packer itself, which never
        # falls back to the template's own scales.
        repacked = repack_nvfp4_weight_local_hessian(original_weight, template, moments)
        _require_nvfp4_wrapper(
            repacked,
            f"The activation-weighted repack of '{fqn}' produced {{kind}} weight instead of a TorchAO NVFP4 "
            "tensor; it cannot replace the converted one.",
        )

        reference = original_weight.detach().to(torch.float32)
        # The exact ``h + 0.01 * mean(h)``, not the maximum-rescaled vector the search reduces with: the numbers
        # reported below are the absolute objective in the units of the artifact's own second moments, which is
        # what makes them comparable across layers and meaningful in the aggregate.
        damped = damped_second_moments(moments, int(reference.shape[1]), reference.device)
        template_mse, template_objective = _reconstruction_errors(template, reference, damped, fqn, "converted")
        searched_mse, searched_objective = _reconstruction_errors(repacked, reference, damped, fqn, "repacked")
    finally:
        # Released before the next FQN's vector is built, whatever happened above.
        del moments

    # Both objectives are the same layer's absolute damped error, measured with the same vector and the same FP64
    # reduction, so the exhaustive search's guarantee is compared against float reduction noise only.
    tolerated = template_objective * (1.0 + WEIGHT_SCALE_MSE_RELATIVE_TOLERANCE) + WEIGHT_SCALE_MSE_ABSOLUTE_TOLERANCE
    if searched_objective > tolerated:
        raise RuntimeError(
            f"The activation-weighted weight-scale search made '{fqn}' worse: weighted objective "
            f"{searched_objective!r} against the converted {template_objective!r}. The search is exhaustive over "
            "the encodings the amax rule also uses, so this is a defect, not a tolerable regression."
        )

    module.weight = torch.nn.Parameter(repacked, requires_grad=False)
    assigned = module.weight
    if not isinstance(assigned, torch.nn.Parameter) or not isinstance(assigned, type(template)):
        raise RuntimeError(
            f"Assigning the repacked NVFP4 weight of '{fqn}' produced a {type(assigned).__name__} instead of an "
            f"{type(template).__name__} Parameter; the module would no longer execute the searched weight."
        )

    ratio, reduction = _mse_ratio_and_reduction(template_objective, searched_objective)
    return {
        "fqn": fqn,
        "shape": [int(size) for size in original_weight.shape],
        "weight_count": int(original_weight.numel()),
        "template_objective": template_objective,
        "searched_objective": searched_objective,
        "ratio": ratio,
        "relative_reduction": reduction,
        "template_mse": template_mse,
        "searched_mse": searched_mse,
    }


def _repack_weight_with_four_over_six(
    model: torch.nn.Module, fqn: str, original_weight: torch.Tensor, description: str
) -> Dict[str, Any]:
    """
    Replace one converted NVFP4 weight with its Four-Over-Six repack and return the per-FQN evidence.

    Unlike the two searches, this method is *not* exhaustive over the encodings the amax rule rounds into: it
    compares exactly two representations against a differently normalized global scale, so a larger reconstruction
    MSE than the ordinary conversion is a possible and reportable outcome, not a defect. Both MSEs are therefore
    measured and reported, and neither is used to reject a valid repack.

    Everything else fails closed: an unconverted or incompatible weight wrapper, a repack result that is not the
    packer's own :class:`FourOverSixRepack`, block counts that do not add up to this weight's blocks, a non-finite
    reconstruction, and an assignment that does not preserve the NVFP4 ``Parameter`` all raise instead of leaving
    the amax-derived weight silently in place.

    Args:
        model (torch.nn.Module): Model holding the just-converted module.
        fqn (str): FQN of the converted module.
        original_weight (torch.Tensor): The high-precision weight this module was converted from.
        description (str): Conversion path named in error messages.

    Returns:
        entry (Dict[str, Any]): FQN, shape, weight count, the M=6/M=4 block counts, both reconstruction MSEs, and
            their ratio/reduction.

    Raises:
        RuntimeError: On any violated contract above.
    """
    module = _module_by_fqn(model, fqn, description)
    template = getattr(module, "weight", None)
    _require_nvfp4_wrapper(
        template,
        f"{description} conversion of '{fqn}' left {{kind}} weight instead of a TorchAO NVFP4 tensor, so the "
        "Four-Over-Six weight-scale repack cannot replace it.",
    )

    # Any incompatible wrapper, shape, dtype, metadata or renormalized global scale is rejected by the packer
    # itself, which never falls back to the template's own scales.
    result = repack_nvfp4_weight_four_over_six(original_weight, template)
    if not isinstance(result, FourOverSixRepack):
        raise RuntimeError(
            f"The Four-Over-Six repack of '{fqn}' returned {type(result).__name__} instead of a FourOverSixRepack, "
            "so the M=6/M=4 evidence of this layer cannot be reported and the result cannot be trusted."
        )
    _require_nvfp4_wrapper(
        result.weight,
        f"The Four-Over-Six repack of '{fqn}' produced {{kind}} weight instead of a TorchAO NVFP4 tensor; it "
        "cannot replace the converted one.",
    )
    block_count = int(original_weight.numel()) // NVFP4_BLOCK_SIZE
    counts = (int(result.block_count), int(result.m6_block_count), int(result.m4_block_count))
    if min(counts) < 0 or counts[0] != block_count or counts[1] + counts[2] != block_count:
        raise RuntimeError(
            f"The Four-Over-Six repack of '{fqn}' reports {counts[1]} M=6 and {counts[2]} M=4 blocks of a declared "
            f"{counts[0]}, but the weight has {block_count} blocks of {NVFP4_BLOCK_SIZE}; its evidence does not "
            "describe the weight it repacked."
        )

    reference = original_weight.detach().to(torch.float32)
    template_mse = _reconstruction_mse(template, reference, fqn, "converted")
    repacked_mse = _reconstruction_mse(result.weight, reference, fqn, "repacked")

    module.weight = torch.nn.Parameter(result.weight, requires_grad=False)
    assigned = module.weight
    if not isinstance(assigned, torch.nn.Parameter) or not isinstance(assigned, type(template)):
        raise RuntimeError(
            f"Assigning the repacked NVFP4 weight of '{fqn}' produced a {type(assigned).__name__} instead of an "
            f"{type(template).__name__} Parameter; the module would no longer execute the repacked weight."
        )

    ratio, reduction = _mse_ratio_and_reduction(template_mse, repacked_mse)
    return {
        "fqn": fqn,
        "shape": [int(size) for size in original_weight.shape],
        "weight_count": int(original_weight.numel()),
        "block_count": block_count,
        "m6_block_count": counts[1],
        "m4_block_count": counts[2],
        "template_mse": template_mse,
        "searched_mse": repacked_mse,
        "ratio": ratio,
        "relative_reduction": reduction,
    }


def _repack_weight_with_awq_clip(
    model: torch.nn.Module,
    fqn: str,
    original_weight: torch.Tensor,
    description: str,
    ratio_codes: bytes,
    code_shape: Sequence[int],
    template_arithmetic: str,
) -> Dict[str, Any]:
    """
    Replace one converted NVFP4 weight with its AWQ-clip repack and return the per-FQN evidence.

    The artifact's codes for this one module are materialized as a uint8 tensor on the weight's device and released
    with it, so the whole artifact only ever stays resident as compact ``bytes``. Nothing is searched here: the
    packer reconstructs exactly the candidate each code names.

    Like Four-Over-Six, and unlike the two exhaustive searches, this method optimizes the layer's *output* error
    against activations that are not present at runtime, so a larger plain reconstruction MSE than the ordinary
    conversion is an expected and reportable outcome rather than a defect. Both MSEs are measured and reported, and
    neither is used to reject a valid repack.

    Everything else fails closed: an unconverted or incompatible weight wrapper, a code matrix the packer refuses,
    a non-finite reconstruction, and an assignment that does not preserve the NVFP4 ``Parameter`` all raise instead
    of leaving the amax-derived weight silently in place.

    Args:
        model (torch.nn.Module): Model holding the just-converted module.
        fqn (str): FQN of the converted module.
        original_weight (torch.Tensor): The high-precision weight this module was converted from.
        description (str): Conversion path named in error messages.
        ratio_codes (bytes): This module's decoded row-major uint8 clipping-ratio codes.
        code_shape (Sequence[int]): The codes' validated ``[out_features, in_features / 16]`` shape.
        template_arithmetic (str): The ordinary-template construction the codes were selected against, re-checked
            here against the wrapper TorchAO actually produced for this module.

    Returns:
        entry (Dict[str, Any]): FQN, shape, weight count, block count, the per-ratio block histogram, both
            reconstruction MSEs, and their ratio/reduction.

    Raises:
        RuntimeError: On any violated contract above.
        ValueError: If the codes do not describe this weight.
    """
    module = _module_by_fqn(model, fqn, description)
    template = getattr(module, "weight", None)
    _require_nvfp4_wrapper(
        template,
        f"{description} conversion of '{fqn}' left {{kind}} weight instead of a TorchAO NVFP4 tensor, so the "
        "AWQ-clip weight-scale repack cannot replace it.",
    )
    # The unclipped code keeps this very template's bytes, so the template TorchAO produced has to be the one the
    # codes were scored against. The backend was already checked before conversion; this catches the case where the
    # wrapper's own flag disagrees with the backend the run resolved.
    expected_triton = template_arithmetic == WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED
    triton_names = ("use_triton_kernel", "_use_triton_kernel")
    triton_flags = [bool(getattr(template, name)) for name in triton_names if hasattr(template, name)]
    produced_triton = triton_flags[0] if triton_flags else None
    if produced_triton is not expected_triton:
        raise RuntimeError(
            f"{description} conversion of '{fqn}' produced a template whose Triton-kernel flag is "
            f"{produced_triton!r}, but the AWQ-clip codes were selected against the '{template_arithmetic}' "
            "ordinary-template construction. Its unclipped blocks would keep bytes the offline search never "
            "scored, so this is a failed run rather than a repack."
        )

    rows, blocks = int(code_shape[0]), int(code_shape[1])
    # ``torch.frombuffer`` views the bytes, so the copy is what detaches the codes from the artifact's own buffer.
    flat = torch.frombuffer(bytearray(ratio_codes), dtype=torch.uint8)
    codes = flat.reshape(rows, blocks).to(device=original_weight.device).clone()
    try:
        # Any incompatible wrapper, shape, dtype, device, code range or metadata is rejected by the packer itself,
        # which never falls back to the template's own scales.
        repacked = repack_nvfp4_weight_awq_clip(original_weight, template, codes)
        _require_nvfp4_wrapper(
            repacked,
            f"The AWQ-clip repack of '{fqn}' produced {{kind}} weight instead of a TorchAO NVFP4 tensor; it cannot "
            "replace the converted one.",
        )
        counts = torch.bincount(codes.reshape(-1).to(torch.long), minlength=WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
        histogram = [int(count) for count in counts.tolist()]
    finally:
        # Released before the next FQN's codes are materialized, whatever happened above.
        del codes

    block_count = int(original_weight.numel()) // NVFP4_BLOCK_SIZE
    if rows * blocks != block_count or sum(histogram) != block_count:
        raise RuntimeError(
            f"The AWQ-clip codes of '{fqn}' cover {rows * blocks} block(s), but the weight has {block_count} "
            f"block(s) of {NVFP4_BLOCK_SIZE}; its codes do not describe the weight they repacked."
        )

    reference = original_weight.detach().to(torch.float32)
    template_mse = _reconstruction_mse(template, reference, fqn, "converted")
    repacked_mse = _reconstruction_mse(repacked, reference, fqn, "repacked")

    module.weight = torch.nn.Parameter(repacked, requires_grad=False)
    assigned = module.weight
    if not isinstance(assigned, torch.nn.Parameter) or not isinstance(assigned, type(template)):
        raise RuntimeError(
            f"Assigning the repacked NVFP4 weight of '{fqn}' produced a {type(assigned).__name__} instead of an "
            f"{type(template).__name__} Parameter; the module would no longer execute the repacked weight."
        )

    ratio, reduction = _mse_ratio_and_reduction(template_mse, repacked_mse)
    return {
        "fqn": fqn,
        "shape": [int(size) for size in original_weight.shape],
        "weight_count": int(original_weight.numel()),
        "block_count": block_count,
        "ratio_histogram": histogram,
        "template_mse": template_mse,
        "searched_mse": repacked_mse,
        "ratio": ratio,
        "relative_reduction": reduction,
    }


def _repack_weight_with_gptq(
    model: torch.nn.Module,
    fqn: str,
    original_weight: torch.Tensor,
    description: str,
    gptq: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Replace one converted NVFP4 weight's packed payload with the artifact's, and return the per-FQN evidence.

    Nothing is selected here and no scale is touched: the artifact's bytes for this one module are materialized as
    a uint8 tensor on the weight's device, handed to the packer, and released with it, so the whole artifact only
    ever stays resident as compact ``bytes`` and at most one module's payload is a device tensor at a time.

    This is also where the *template* binding is checked, which the loader could not check yet: the ordinary
    template TorchAO has just produced must carry exactly the scale-buffer shape, dtype, byte length and content
    digest, and exactly the global scale, that the payload was written under. A mismatch means the run's conversion
    is not the conversion the offline selection scored, so it is a failed run rather than a repack. The repacked
    wrapper is then re-read and its scale identity compared against the template's, which is what proves the
    replacement changed the payload and nothing else.

    Like AWQ-clip, and unlike the two exhaustive searches, GPTQ minimizes the layer's output error under a Hessian
    that is absent at runtime, so a larger plain reconstruction MSE than the ordinary conversion is an expected and
    reportable outcome rather than a defect. Both MSEs are measured here -- these are the run's own numbers, not the
    artifact's -- and neither is used to reject a valid replacement.

    Args:
        model (torch.nn.Module): Model holding the just-converted module.
        fqn (str): FQN of the converted module.
        original_weight (torch.Tensor): The high-precision weight this module was converted from.
        description (str): Conversion path named in error messages.
        gptq (Dict[str, Any]): The validated artifact from :func:`load_gptq_artifact`.

    Returns:
        entry (Dict[str, Any]): FQN, shape, weight count, block count, payload byte length and digest, the bound
            template-scale digest, the artifact's offline objectives, both runtime-measured reconstruction MSEs,
            and their ratio/reduction.

    Raises:
        RuntimeError: On any violated contract above.
        ValueError: If the payload does not describe this weight.
    """
    module = _module_by_fqn(model, fqn, description)
    template = getattr(module, "weight", None)
    _require_nvfp4_wrapper(
        template,
        f"{description} conversion of '{fqn}' left {{kind}} weight instead of a TorchAO NVFP4 tensor, so the GPTQ "
        "payload cannot replace it.",
    )
    # The payload was written under the block scales one construction produces; the backend was already checked
    # before conversion, and this catches a wrapper whose own flag disagrees with the backend the run resolved.
    template_arithmetic = str(gptq["template_arithmetic"])
    expected_triton = template_arithmetic == WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED
    triton_names = ("use_triton_kernel", "_use_triton_kernel")
    triton_flags = [bool(getattr(template, name)) for name in triton_names if hasattr(template, name)]
    produced_triton = triton_flags[0] if triton_flags else None
    if produced_triton is not expected_triton:
        raise RuntimeError(
            f"{description} conversion of '{fqn}' produced a template whose Triton-kernel flag is "
            f"{produced_triton!r}, but the GPTQ payload was written under the '{template_arithmetic}' "
            "ordinary-template construction. Its blocks would decode against scales nobody selected them under, so "
            "this is a failed run rather than a repack."
        )

    identity = nvfp4_template_identity(template)
    rows, columns = int(original_weight.shape[0]), int(original_weight.shape[1])
    if (identity.rows, identity.columns) != (rows, columns):
        raise RuntimeError(
            f"{description} conversion of '{fqn}' produced a ({identity.rows}, {identity.columns}) NVFP4 template, "
            f"but its original weight is ({rows}, {columns}); the GPTQ payload cannot describe it."
        )
    _require_gptq_template_binding(identity, gptq["template_scale"][fqn], fqn)

    raw = gptq["qdata"][fqn]
    shape = [int(size) for size in gptq["qdata_shapes"][fqn]]
    if len(raw) != identity.qdata.numel():
        raise RuntimeError(
            f"The GPTQ payload of '{fqn}' holds {len(raw)} byte(s), but the template TorchAO produced holds "
            f"{identity.qdata.numel()}; the payload does not describe this weight."
        )
    # ``torch.frombuffer`` views the bytes, so the copy is what detaches the payload from the artifact's buffer.
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    qdata = flat.reshape(shape[0], shape[1]).to(device=original_weight.device).clone()
    try:
        # Any incompatible wrapper, shape, dtype, device or metadata is rejected by the packer itself, which never
        # falls back to the template's own payload.
        repacked = repack_nvfp4_weight_gptq(template, qdata)
        _require_nvfp4_wrapper(
            repacked,
            f"The GPTQ payload replacement of '{fqn}' produced {{kind}} weight instead of a TorchAO NVFP4 tensor; "
            "it cannot replace the converted one.",
        )
    finally:
        # Released before the next FQN's payload is materialized, whatever happened above.
        del qdata, flat

    # The replacement must have changed the payload and nothing else, so the new wrapper is re-read and bound to
    # exactly the same scale identity the template was.
    _require_gptq_template_binding(nvfp4_template_identity(repacked), gptq["template_scale"][fqn], fqn)

    reference = original_weight.detach().to(torch.float32)
    template_mse = _reconstruction_mse(template, reference, fqn, "converted")
    repacked_mse = _reconstruction_mse(repacked, reference, fqn, "repacked")

    module.weight = torch.nn.Parameter(repacked, requires_grad=False)
    assigned = module.weight
    if not isinstance(assigned, torch.nn.Parameter) or not isinstance(assigned, type(template)):
        raise RuntimeError(
            f"Assigning the GPTQ NVFP4 weight of '{fqn}' produced a {type(assigned).__name__} instead of an "
            f"{type(template).__name__} Parameter; the module would no longer execute the selected payload."
        )

    evidence = gptq["provenance"]["modules"][fqn]
    ratio, reduction = _mse_ratio_and_reduction(template_mse, repacked_mse)
    return {
        "fqn": fqn,
        "shape": [rows, columns],
        "weight_count": rows * columns,
        "block_count": rows * columns // NVFP4_BLOCK_SIZE,
        "qdata_byte_length": len(raw),
        "qdata_sha256": gptq["qdata_digests"][fqn],
        "template_scale_sha256": str(gptq["template_scale"][fqn]["sha256"]),
        "hessian_sha256": str(gptq["hessian"][fqn]["sha256"]),
        "dead_column_count": int(gptq["hessian"][fqn]["dead_column_count"]),
        "offline_template_objective": float(evidence["template_objective"]),
        "offline_selected_objective": float(evidence["selected_objective"]),
        "template_mse": template_mse,
        "searched_mse": repacked_mse,
        "ratio": ratio,
        "relative_reduction": reduction,
    }


def _require_gptq_template_binding(identity: Any, recorded: Dict[str, Any], fqn: str) -> None:
    """
    Reject a template whose scale buffer or global scale is not the one a GPTQ payload was written under.

    The payload carries no scale of its own: it is a set of FP4 codes that only reconstruct the intended values
    under exactly these bytes. Comparing the buffer's shape, dtype, byte length and content digest, and the global
    scale's own digest, is therefore the whole binding between the offline selection and the deployed wrapper.
    """
    actual = {
        "shape": [int(size) for size in identity.scale.shape],
        "dtype": str(identity.scale.dtype),
        "byte_length": int(identity.scale.numel()),
        "sha256": nvfp4_weight_digest(identity.scale),
        "global_scale_sha256": nvfp4_weight_digest(identity.global_scale),
    }
    mismatched = [field for field in sorted(actual) if recorded[field] != actual[field]]
    if mismatched:
        field = mismatched[0]
        raise RuntimeError(
            f"The GPTQ payload of '{fqn}' was written under another ordinary template: it records "
            f"'{field}' as {recorded[field]!r}, but the template this run produced gives {actual[field]!r} "
            f"({len(mismatched)} mismatched field(s): {mismatched}). The stored FP4 codes only reconstruct the "
            "selected values under the exact scale bytes they were written for."
        )


def _require_nvfp4_wrapper(weight: Any, message: str) -> None:
    """
    Reject anything that is not a TorchAO NVFP4 tensor wrapper, naming what was found instead.

    An ordinary ``Tensor`` or ``Parameter`` is refused explicitly: it carries no block scales to search or to
    execute, and ``torch.Tensor.dequantize`` would otherwise make it look superficially like an NVFP4 wrapper.

    Args:
        weight (Any): Candidate NVFP4 weight.
        message (str): Error message with a ``{kind}`` placeholder for what was found.

    Raises:
        RuntimeError: If the candidate is not a tensor, or is a plain tensor or parameter.
    """
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(message.format(kind="no"))
    if type(weight) in (torch.Tensor, torch.nn.Parameter):
        raise RuntimeError(message.format(kind=f"an ordinary {type(weight).__name__}"))


def _reconstruction_mse(weight: torch.Tensor, reference: torch.Tensor, fqn: str, stage: str) -> float:
    """
    FP32 mean squared error between an NVFP4 weight's dequantization and the original high-precision weight.

    Args:
        weight (torch.Tensor): NVFP4 weight wrapper to dequantize.
        reference (torch.Tensor): FP32 view of the original weight.
        fqn (str): FQN named in error messages.
        stage (str): ``'converted'`` or ``'repacked'``, named in error messages.

    Returns:
        mse (float): Finite mean squared error over every weight element.

    Raises:
        RuntimeError: If the wrapper cannot be dequantized, dequantizes to a different shape, or produces
            non-finite values.
    """
    dequantized = _checked_dequantization(weight, reference, fqn, stage)
    mse = float(((dequantized - reference) ** 2).mean())
    if not math.isfinite(mse):
        raise RuntimeError(f"The {stage} reconstruction MSE of '{fqn}' is not finite ({mse!r}).")
    return mse


def _reconstruction_errors(
    weight: torch.Tensor, reference: torch.Tensor, damped: torch.Tensor, fqn: str, stage: str
) -> Tuple[float, float]:
    """
    Unweighted reconstruction MSE and activation-weighted objective of one NVFP4 weight, from one dequantization.

    The weighted value is the mean over every weight element of ``h_damped[j] * (W - Q(W)) ** 2`` with the *exact*
    damped moments ``h_damped = h + 0.01 * mean(h)``: it is the per-block objective
    ``sum_{j in b} h_damped[j] * (W - Q(W)) ** 2`` the search minimizes, divided by the block size and averaged
    over the layer's blocks, and by nothing else. Nothing per-layer is normalized away, so the value carries the
    units of the artifact's second moments and two layers -- and the weight-count-weighted aggregate over them --
    are in the same units. The products are accumulated in FP64 so that a layer with large second moments is
    summed without avoidable rounding.

    Args:
        weight (torch.Tensor): NVFP4 weight wrapper to dequantize.
        reference (torch.Tensor): FP32 view of the original weight.
        damped (torch.Tensor): Exact damped ``(K,)`` second moments on ``reference``'s device.
        fqn (str): FQN named in error messages.
        stage (str): ``'converted'`` or ``'repacked'``, named in error messages.

    Returns:
        errors (Tuple[float, float]): The unweighted MSE and the weighted objective, both finite.

    Raises:
        RuntimeError: If the wrapper cannot be dequantized, dequantizes to a different shape, produces non-finite
            values, or yields a non-finite error.
    """
    dequantized = _checked_dequantization(weight, reference, fqn, stage)
    squared = (dequantized - reference) ** 2
    mse = float(squared.mean())
    objective = float((squared.to(torch.float64) * damped.to(torch.float64)[None, :]).mean())
    for name, value in (("reconstruction MSE", mse), ("weighted objective", objective)):
        if not math.isfinite(value):
            raise RuntimeError(f"The {stage} {name} of '{fqn}' is not finite ({value!r}).")
    return mse, objective


def _checked_dequantization(weight: torch.Tensor, reference: torch.Tensor, fqn: str, stage: str) -> torch.Tensor:
    """FP32 dequantization of an NVFP4 wrapper, checked for the reference's shape and for finiteness."""
    dequantized = _dequantize_nvfp4(weight, fqn, stage).to(torch.float32)
    if tuple(dequantized.shape) != tuple(reference.shape):
        raise RuntimeError(
            f"The {stage} NVFP4 weight of '{fqn}' dequantizes to shape {tuple(dequantized.shape)} but the original "
            f"weight has shape {tuple(reference.shape)}; its reconstruction error cannot be measured."
        )
    if not bool(torch.isfinite(dequantized).all()):
        raise RuntimeError(
            f"The {stage} NVFP4 weight of '{fqn}' dequantizes to non-finite values; refusing to report a "
            "weight-scale search result for it."
        )
    return dequantized


def _dequantize_nvfp4(weight: torch.Tensor, fqn: str, stage: str) -> torch.Tensor:
    """Dequantize an NVFP4 wrapper to FP32; pinned TorchAO 0.17 spells this ``dequantize(torch.float32)``."""
    for name in ("dequantize", "to_dtype"):
        helper = getattr(weight, name, None)
        if callable(helper):
            return helper(torch.float32)
    raise RuntimeError(
        f"The {stage} weight of '{fqn}' ({type(weight).__name__}) exposes no NVFP4 dequantization helper "
        "('dequantize(torch.float32)'), so the block-MSE weight-scale search cannot measure its reconstruction "
        "error. The installed torchao NVFP4 wrapper is not the pinned one."
    )


def _module_by_fqn(model: torch.nn.Module, fqn: str, description: str) -> torch.nn.Module:
    """Look up a module by exact FQN, failing closed when the conversion removed or renamed it."""
    modules = dict(model.named_modules())
    module = modules.get(fqn)
    if module is None:
        raise RuntimeError(
            f"{description} module '{fqn}' is not present in the model; the block-MSE weight-scale search cannot "
            "locate the module it must repack."
        )
    return module


def _weight_scale_mse_summary(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic summary section for an enabled search: per-FQN evidence in sorted order plus the aggregate."""
    layers = sorted(entries, key=lambda entry: entry["fqn"])
    return {
        "enabled": True,
        "method": WEIGHT_SCALE_METHOD_MSE,
        "algorithm": WEIGHT_SCALE_MSE_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_MSE_ALGORITHM_VERSION,
        "target_count": len(layers),
        "total_weight_count": sum(int(layer["weight_count"]) for layer in layers),
        "layers": layers,
        "aggregate": _weight_scale_mse_aggregate(layers),
        "notes": [
            "Reconstruction MSE is computed in FP32 against the original high-precision weight; the aggregate is "
            "weighted by each layer's weight count.",
            "The global per-tensor scales and the NVFP4 wire format are TorchAO's own; only the per-block E4M3 "
            "scales and the matching FP4 payload were re-selected.",
        ],
    }


def _weight_scale_mse_aggregate(layers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Weight-count-weighted aggregate of the per-layer MSEs, or an all-``None`` section when nothing was searched."""
    total = sum(int(layer["weight_count"]) for layer in layers)
    if not layers or total <= 0:
        return {"template_mse": None, "searched_mse": None, "ratio": None, "relative_reduction": None}
    template_mse = sum(float(layer["template_mse"]) * int(layer["weight_count"]) for layer in layers) / total
    searched_mse = sum(float(layer["searched_mse"]) * int(layer["weight_count"]) for layer in layers) / total
    ratio, reduction = _mse_ratio_and_reduction(template_mse, searched_mse)
    return {
        "template_mse": template_mse,
        "searched_mse": searched_mse,
        "ratio": ratio,
        "relative_reduction": reduction,
    }


def _weight_scale_four_over_six_summary(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic summary section for an enabled Four-Over-Six repack.

    It states the algorithm identity, the two constants that *are* that identity -- the 256 normalization maximum
    and the ``(6, 4)`` candidate magnitudes -- how many blocks of every layer took each representation, and the
    reconstruction MSE of the ordinary TorchAO conversion next to the one this method achieved. The comparison is
    reported, never enforced: this method is not exhaustive over the ordinary encodings and renormalizes the global
    scale, so a worse MSE is a possible outcome of a correct run.
    """
    layers = sorted(entries, key=lambda entry: entry["fqn"])
    return {
        "enabled": True,
        "method": WEIGHT_SCALE_METHOD_FOUR_OVER_SIX,
        "algorithm": WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION,
        "fp8_max_for_normalization": float(WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX),
        "candidate_magnitudes": [int(magnitude) for magnitude in WEIGHT_SCALE_FOUR_OVER_SIX_MAGNITUDES],
        "target_count": len(layers),
        "target_fqns": [layer["fqn"] for layer in layers],
        "total_weight_count": sum(int(layer["weight_count"]) for layer in layers),
        "total_block_count": sum(int(layer["block_count"]) for layer in layers),
        "total_m6_block_count": sum(int(layer["m6_block_count"]) for layer in layers),
        "total_m4_block_count": sum(int(layer["m4_block_count"]) for layer in layers),
        "layers": layers,
        "aggregate": _weight_scale_four_over_six_aggregate(layers),
        "notes": [
            "Every 16-weight block was written with its amax mapped onto FP4's largest magnitude 6 (M=6) or onto 4 "
            "(M=4, the same scale multiplied by 1.5), whichever reconstructed the block with the lower plain "
            "squared error; an exact tie kept M=6.",
            "The weight global scale is the ordinary TorchAO one renormalized against "
            f"{WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX}, i.e. multiplied by 448 / "
            f"{WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX}; the activation global scale, the NVFP4 wire format and the "
            "inference kernels are TorchAO's own and are unchanged.",
            "Reconstruction MSE is computed in FP32 against the original high-precision weight; the aggregate is "
            "weighted by each layer's weight count.",
            "This method is not exhaustive over the encodings the amax rule rounds into, so it carries no "
            "guarantee of a lower MSE than the ordinary conversion; the comparison is reported, not enforced.",
        ],
    }


def _weight_scale_four_over_six_aggregate(layers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Weight-count-weighted aggregate of the per-layer MSEs, plus the explicit M=6/M=4 block totals."""
    return {
        **_weight_scale_mse_aggregate(layers),
        "block_count": sum(int(layer["block_count"]) for layer in layers),
        "m6_block_count": sum(int(layer["m6_block_count"]) for layer in layers),
        "m4_block_count": sum(int(layer["m4_block_count"]) for layer in layers),
    }


def _weight_scale_awq_clip_summary(entries: Sequence[Dict[str, Any]], awq_clip: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic summary section for an enabled AWQ-clip repack.

    It states the algorithm identity and every constant that *is* that identity -- the eleven clipping ratios in
    their fixed order, the block size, the tie rule and the ModelOpt reference the arithmetic was adapted from --
    which artifact selected the codes, by path, whole-file digest and the two verified component digests, which
    activation calibration that artifact is bound to, how many blocks of every layer took each ratio, and the
    reconstruction MSE of the ordinary TorchAO conversion next to the one these codes produced.

    The offline objective is *not* recomputed here and deliberately cannot be: the activation rows it was measured
    on are intentionally absent from the artifact. The two objective numbers are therefore reported under
    ``offline_objectives``, explicitly labelled as evidence the artifact carries rather than as a runtime
    measurement. The MSE comparison is likewise reported and never enforced: this method optimizes output error,
    so a larger plain weight MSE is a possible outcome of a correct run.
    """
    layers = sorted(entries, key=lambda entry: entry["fqn"])
    aggregate_provenance = awq_clip["provenance"]["aggregate"]
    return {
        "enabled": True,
        "method": WEIGHT_SCALE_METHOD_AWQ_CLIP,
        "algorithm": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION,
        "block_size": int(WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE),
        "clip_ratios": [float(ratio) for ratio in WEIGHT_SCALE_AWQ_CLIP_RATIOS],
        "tie_rule": AWQ_CLIP_TIE_RULE,
        "template_arithmetic": awq_clip["template_arithmetic"],
        "unclipped_code": int(WEIGHT_SCALE_AWQ_CLIP_UNCLIPPED_CODE),
        "modelopt_reference_version": MODELOPT_REFERENCE_VERSION,
        "modelopt_reference_wheel_sha256": MODELOPT_REFERENCE_WHEEL_SHA256,
        "artifact_path": awq_clip["path"],
        "artifact_sha256": awq_clip["sha256"],
        "ratio_code_sha256": awq_clip["ratio_code_sha256"],
        "provenance_sha256": awq_clip["provenance_sha256"],
        "checkpoint_sha256": awq_clip["checkpoint_sha256"],
        "calibration_path": awq_clip["calibration_path"],
        "calibration_sha256": awq_clip["calibration"]["sha256"],
        "scale_margin": float(awq_clip["calibration"]["scale_margin"]),
        "target_count": len(layers),
        "target_fqns": [layer["fqn"] for layer in layers],
        "total_weight_count": sum(int(layer["weight_count"]) for layer in layers),
        "total_block_count": sum(int(layer["block_count"]) for layer in layers),
        "ratio_histogram": _summed_ratio_histogram(layers),
        "layers": layers,
        "aggregate": _weight_scale_awq_clip_aggregate(layers),
        "offline_objectives": {
            "source": "artifact",
            "measured_at_runtime": False,
            "selected": float(aggregate_provenance["selected_objective"]),
            "unclipped": float(aggregate_provenance["unclipped_objective"]),
            "block_count": int(aggregate_provenance["block_count"]),
        },
        "notes": [
            "Every clipped 16-weight block was written with its own absolute maximum multiplied by the clipping "
            f"ratio its artifact code names, i.e. scale_fp8 = E4M3(clamp(((block_amax * ratio) / {NVFP4_MAX:g}) / "
            "weight_global_scale, 2 ** -9, 448)); an exact tie in the offline search kept the earliest ratio.",
            f"Blocks whose code is the unclipped {int(WEIGHT_SCALE_AWQ_CLIP_UNCLIPPED_CODE)} were not rewritten at "
            "all: their E4M3 scale byte and their eight packed payload bytes are the ordinary TorchAO conversion's "
            f"own, taken from the '{awq_clip['template_arithmetic']}' template this backend produced, which is the "
            "construction the offline search scored them against.",
            "The weight global per-tensor scale, the activation scale, the NVFP4 wire format and the inference "
            "kernels are TorchAO's own and are unchanged; only the per-block E4M3 scales and the matching FP4 "
            "payload were re-selected.",
            "'offline_objectives' are the values the artifact recorded when it selected these codes against "
            "quantized activations; they are not recomputed here, because the activation rows they were measured "
            "on are deliberately absent from the artifact.",
            "Reconstruction MSE is computed in FP32 against the original high-precision weight; the aggregate is "
            "weighted by each layer's weight count. This method minimizes the layer's output error, not the plain "
            "weight error, so a larger MSE than the ordinary conversion is reported and never rejected.",
            "This section is evidence about the stored weights only and makes no claim about DER.",
        ],
    }


def _weight_scale_awq_clip_aggregate(layers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Weight-count-weighted aggregate of the per-layer MSEs, plus the block total and the ratio histogram."""
    return {
        **_weight_scale_mse_aggregate(layers),
        "block_count": sum(int(layer["block_count"]) for layer in layers),
        "ratio_histogram": _summed_ratio_histogram(layers),
    }


def _summed_ratio_histogram(layers: Sequence[Dict[str, Any]]) -> List[int]:
    """Element-wise sum of the per-layer clip-ratio histograms, in the fixed candidate order."""
    return [
        sum(int(layer["ratio_histogram"][index]) for layer in layers)
        for index in range(WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
    ]


def _weight_scale_gptq_summary(entries: Sequence[Dict[str, Any]], gptq: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic summary section for an enabled GPTQ payload replacement.

    It states the algorithm identity and every constant that *is* that identity -- the damping, the Hessian and
    group rules, the dead-column rule, the inverse, the 128-column update block, the 16-value NVFP4 block, the
    fixed-template-scale rule, the activation QDQ and the ModelOpt reference the arithmetic was adapted from --
    which artifact carried the payload, by path, whole-file digest and the three verified section digests, which
    activation calibration and which ordinary-template construction that artifact is bound to, the aggregate
    evidence of the sources and Hessians it was built from, the per-layer and aggregate payload byte counts, the
    offline Hessian quadratic objectives the artifact recorded, and the reconstruction MSE this run measured for
    the ordinary template next to the one the GPTQ payload produced.

    The offline objectives are *not* recomputed here and deliberately cannot be: the activation rows and the
    Hessians they were measured on are intentionally absent from the artifact. They are reported under
    ``offline_objectives``, explicitly labelled as artifact-carried evidence. The two MSEs next to them are the
    opposite -- measured in this process, against this run's own original weights -- and the comparison is
    reported and never enforced, because this method minimizes output error rather than plain weight error.
    """
    layers = sorted(entries, key=lambda entry: entry["fqn"])
    aggregate_provenance = gptq["provenance"]["aggregate"]
    sources = gptq["provenance"]["sources"]
    return {
        "enabled": True,
        "method": WEIGHT_SCALE_METHOD_GPTQ,
        "algorithm": WEIGHT_SCALE_GPTQ_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION,
        "perc_damp": float(WEIGHT_SCALE_GPTQ_PERC_DAMP),
        "update_block_size": int(WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE),
        "block_size": int(WEIGHT_SCALE_GPTQ_BLOCK_SIZE),
        "hessian_rule": GPTQ_HESSIAN_RULE,
        "group_reduction": GPTQ_GROUP_REDUCTION,
        "dead_column_rule": GPTQ_DEAD_COLUMN_RULE,
        "inverse_rule": GPTQ_INVERSE_RULE,
        "template_scale_rule": GPTQ_TEMPLATE_SCALE_RULE,
        "activation_qdq": GPTQ_ACTIVATION_QDQ,
        "objective": GPTQ_OBJECTIVE,
        "hessian_digest_method": GPTQ_HESSIAN_DIGEST_METHOD,
        "template_arithmetic": gptq["template_arithmetic"],
        "modelopt_reference_version": MODELOPT_REFERENCE_VERSION,
        "modelopt_reference_wheel_sha256": MODELOPT_REFERENCE_WHEEL_SHA256,
        "artifact_path": gptq["path"],
        "artifact_sha256": gptq["sha256"],
        "qdata_sha256": gptq["qdata_sha256"],
        "hessian_sha256": gptq["hessian_sha256"],
        "provenance_sha256": gptq["provenance_sha256"],
        "checkpoint_sha256": gptq["checkpoint_sha256"],
        "calibration_path": gptq["calibration_path"],
        "calibration_sha256": gptq["calibration"]["sha256"],
        "scale_margin": float(gptq["calibration"]["scale_margin"]),
        "target_count": len(layers),
        "target_fqns": [layer["fqn"] for layer in layers],
        "total_weight_count": sum(int(layer["weight_count"]) for layer in layers),
        "total_block_count": sum(int(layer["block_count"]) for layer in layers),
        "total_qdata_byte_length": sum(int(layer["qdata_byte_length"]) for layer in layers),
        "total_dead_column_count": sum(int(layer["dead_column_count"]) for layer in layers),
        "source_count": len(sources),
        "source_labels": [str(source["label"]) for source in sources],
        "total_sampled_row_count": sum(int(source["sampled_row_count"]) for source in sources),
        "layers": layers,
        "aggregate": _weight_scale_gptq_aggregate(layers),
        "offline_objectives": {
            "source": "artifact",
            "measured_at_runtime": False,
            "template": float(aggregate_provenance["template_objective"]),
            "selected": float(aggregate_provenance["selected_objective"]),
            "weight_count": int(aggregate_provenance["weight_count"]),
        },
        "notes": [
            "The block scales, the swizzled E4M3 scale buffer including its padding, the weight global per-tensor "
            "scale, the activation scale, the NVFP4 wire format and the inference kernels are the ordinary "
            "TorchAO conversion's own and are unchanged: only the packed FP4 payload bytes were replaced, and each "
            "layer's scale-buffer identity was re-checked against the artifact before and after the replacement.",
            "Payload bytes were selected offline by visiting input columns in order under the fixed template "
            "scales and propagating each column's residual to the following columns over "
            f"{int(WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE)}-column blocks, under a group-balanced input Hessian "
            f"damped by {float(WEIGHT_SCALE_GPTQ_PERC_DAMP)} of its own diagonal mean.",
            "'offline_objectives' are the damped Hessian quadratic forms the artifact recorded when it selected "
            "this payload; they are not recomputed here, because the activation rows and the Hessians they were "
            "measured on are deliberately absent from the artifact.",
            "Reconstruction MSE is measured in this run, in FP32, against the original high-precision weight; the "
            "aggregate is weighted by each layer's weight count. This method minimizes the layer's output error, "
            "not the plain weight error, so a larger MSE than the ordinary conversion is reported and never "
            "rejected.",
            "This section is evidence about the stored weights only and makes no claim about DER.",
        ],
    }


def _weight_scale_gptq_aggregate(layers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Weight-count-weighted aggregate of the measured MSEs, plus the block and payload byte totals."""
    return {
        **_weight_scale_mse_aggregate(layers),
        "block_count": sum(int(layer["block_count"]) for layer in layers),
        "qdata_byte_length": sum(int(layer["qdata_byte_length"]) for layer in layers),
        "dead_column_count": sum(int(layer["dead_column_count"]) for layer in layers),
    }


def _weight_scale_hessian_summary(entries: Sequence[Dict[str, Any]], hessian: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic summary section for an enabled activation-weighted search.

    It states which artifact selected the scales -- by path, whole-file digest and the two verified component
    digests of its moments and its provenance -- what that artifact was built on, the per-layer and aggregate
    weighted objectives the search achieved, and the unweighted reconstruction MSE before and after as diagnostic
    context. It deliberately makes no accuracy claim: this section is evidence about the weights, and only an
    evaluation can say what the technique does to DER.
    """
    layers = sorted(entries, key=lambda entry: entry["fqn"])
    return {
        "enabled": True,
        "method": WEIGHT_SCALE_METHOD_LOCAL_HESSIAN,
        "algorithm": WEIGHT_SCALE_HESSIAN_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION,
        "damping": float(WEIGHT_SCALE_HESSIAN_DAMPING),
        "artifact_path": hessian["path"],
        "artifact_sha256": hessian["sha256"],
        # The two component digests the loader recomputed and verified, so the run's own evidence records which
        # moments and which provenance selected these scales, not only the digest of the whole file.
        "moment_sha256": hessian["moment_sha256"],
        "provenance_sha256": hessian["provenance_sha256"],
        "checkpoint_sha256": hessian["checkpoint_sha256"],
        "target_count": len(layers),
        "target_fqns": [layer["fqn"] for layer in layers],
        "total_weight_count": sum(int(layer["weight_count"]) for layer in layers),
        "layers": layers,
        "aggregate": _weight_scale_hessian_aggregate(layers),
        "notes": [
            "Block scales were chosen by minimizing sum_j h_damped[j] * (W - Q(W))^2 per 16-weight block, with "
            f"h from the artifact's diagonal second moments damped by {WEIGHT_SCALE_HESSIAN_DAMPING} of their "
            "mean.",
            "A reported objective is that same quantity per weight element -- the per-block sum divided by the "
            "block size 16, averaged over the layer's blocks -- in the units of the artifact's second moments, "
            "with no per-layer normalization; the aggregate is the same quantity weighted by each layer's weight "
            "count, so it is comparable across layers and runs built on the same statistics.",
            "The unweighted reconstruction MSE is reported as diagnostic evidence only: the weighted search may "
            "raise it on channels the measured activations never excite, which is the point of the objective.",
            "The global per-tensor scales, the NVFP4 wire format and the inference kernels are TorchAO's own and "
            "are unchanged; only the per-block E4M3 scales and the matching FP4 payload were re-selected.",
            "This section is evidence about the stored weights only and makes no claim about DER.",
        ],
    }


def _weight_scale_hessian_aggregate(layers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Weight-count-weighted aggregate of the weighted objectives and of the diagnostic unweighted MSEs."""
    total = sum(int(layer["weight_count"]) for layer in layers)
    fields = ("template_objective", "searched_objective", "template_mse", "searched_mse")
    if not layers or total <= 0:
        return {**{field: None for field in fields}, "ratio": None, "relative_reduction": None}
    aggregate = {
        field: sum(float(layer[field]) * int(layer["weight_count"]) for layer in layers) / total for field in fields
    }
    ratio, reduction = _mse_ratio_and_reduction(aggregate["template_objective"], aggregate["searched_objective"])
    return {**aggregate, "ratio": ratio, "relative_reduction": reduction}


def _mse_ratio_and_reduction(template_mse: float, searched_mse: float) -> Tuple[float, float]:
    """
    Ratio ``searched / template`` and relative reduction ``1 - ratio``, with an explicit zero baseline.

    A zero baseline means the amax-derived scales already reconstructed the weight exactly, which the search can
    match but never improve on; that case is reported as ratio ``1.0`` and reduction ``0.0`` rather than as an
    undefined division.
    """
    if template_mse <= 0.0:
        return 1.0, 0.0
    ratio = searched_mse / template_mse
    return float(ratio), float(1.0 - ratio)


def _assign_activation_amax(module: torch.nn.Module, fqn: str, amax: float) -> None:
    """
    Write a validated activation amax onto a TorchAO observed linear.

    TorchAO 0.17's ``NVFP4ObservedLinear`` keeps the running activation amax directly in ``module.amax``
    (a scalar tensor created in ``__init__`` and updated in ``forward``), and the convert step reads that
    tensor to build the static per-tensor scale. Older observer layouts that expose ``act_obs.amax`` or
    ``act_amax`` are still handled so calibration keeps working across TorchAO versions.
    """
    current = getattr(module, "amax", None)
    if isinstance(current, torch.Tensor):
        module.amax = torch.full_like(current, float(amax))
        return

    reference = next(module.parameters(), None)
    device = None if reference is None else reference.device
    value = torch.tensor(float(amax), dtype=torch.float32, device=device)
    observer = getattr(module, "act_obs", None)
    if observer is not None and hasattr(observer, "amax"):
        observer.amax = value
        return
    if hasattr(module, "act_amax"):
        module.act_amax = value
        return
    raise RuntimeError(
        f"Observed linear '{fqn}' ({type(module).__name__}) does not expose an assignable activation amax "
        "('amax', 'act_obs.amax' or 'act_amax'); static NVFP4 cannot be applied with this TorchAO version."
    )


def _fqn_filter(fqns: Sequence[str]):
    """Build a ``quantize_`` filter that selects exactly the given FQNs and only ``torch.nn.Linear`` modules."""
    selected = set(fqns)

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        return fqn in selected and isinstance(module, torch.nn.Linear)

    return filter_fn


def _acceleration_report(model: torch.nn.Module, nvfp4_fqns: Sequence[str], backend: str) -> Dict[str, Any]:
    """
    Qualify the acceleration claim of the resolved backend against the TorchAO packing constraints.

    ``K % 64 == 0`` is a static weight property and is enforced here for the accelerated backend. ``M % 128 == 0``
    is the runtime token count of each matmul: it is unknown at quantization time, a guard placed on the quantized
    modules would be traced away by ``torch.compile`` or bypassed by the module swap, and TorchAO can fall back
    internally when it is violated. The accelerated backend is therefore reported as *conditional and unverified*
    rather than as an accomplished acceleration.

    Args:
        model (torch.nn.Module): Model holding the selected modules; read for their static ``in_features``.
        nvfp4_fqns (Sequence[str]): FQNs selected for NVFP4 W4A4.
        backend (str): Resolved backend label.

    Returns:
        acceleration (Dict[str, Any]): Acceleration status, the two constraints, and the conditionality caveat.

    Raises:
        RuntimeError: If accelerated packing was requested but a selected module violates ``K % 64 == 0``.
    """
    modules = dict(model.named_modules())
    violations = []
    for fqn in nvfp4_fqns:
        if getattr(modules[fqn], "in_features", 0) % ACCELERATED_PACKING_K_MULTIPLE != 0:
            violations.append(fqn)
    violations.sort()
    if backend == BACKEND_MSLK_ACCELERATED and violations:
        raise RuntimeError(
            f"Accelerated NVFP4 packing requires in_features % {ACCELERATED_PACKING_K_MULTIPLE} == 0, but these "
            f"selected modules violate it: {violations}. TorchAO would fall back internally, which would not be "
            "accelerated execution."
        )

    if backend == BACKEND_MSLK_ACCELERATED:
        status = ACCELERATION_CONDITIONAL
    elif backend == BACKEND_REFERENCE_UNACCELERATED:
        status = ACCELERATION_UNACCELERATED
    else:
        status = ACCELERATION_NOT_APPLICABLE
    return {
        "status": status,
        "verified": False,
        "k_constraint": f"in_features % {ACCELERATED_PACKING_K_MULTIPLE} == 0",
        "k_constraint_satisfied": not violations,
        "k_constraint_violations": violations,
        "m_constraint": f"runtime token count M % {ACCELERATED_PACKING_M_MULTIPLE} == 0",
        "m_constraint_checked": False,
        "caveat": (
            "M is a runtime property and is not checked here; when it is violated TorchAO can fall back "
            "internally, and such an execution is NOT accelerated. This summary reports the configured packing "
            "path, never a measured acceleration."
        ),
    }


def _summary_notes(config: SortformerQuantizationConfig, backend: str, facts: CapabilityFacts) -> List[str]:
    """Human-readable caveats attached to the structured summary."""
    notes = ["Quantized models are in-process evaluation artifacts; quantized .nemo export is not supported."]
    if config.recipe == "nvfp4_weight_only":
        notes.append(
            "nvfp4_weight_only stores NVFP4 weights with BF16 activations: it is a storage/accuracy comparator, "
            "not accelerated W4A4 execution."
        )
    if config.has_bf16_override:
        notes.append(
            f"A BF16 override file restored part of the '{config.recipe}' target set to BF16; the 'bf16_override' "
            "section records the file identity and the exact restored FQNs of this run."
        )
    if backend == BACKEND_REFERENCE_UNACCELERATED:
        notes.append("Backend is UNACCELERATED reference NVFP4; throughput numbers from this run are not valid.")
    if backend == BACKEND_MSLK_ACCELERATED:
        notes.append(
            f"Accelerated packing is conditional: TorchAO can fall back internally unless every matmul satisfies "
            f"M % {ACCELERATED_PACKING_M_MULTIPLE} == 0 at runtime, and that is not verified here."
        )
    if config.fold_global_scales:
        notes.append(
            f"Global-scale folding is ON with activation exponent {config.fold_activation_exponent}: the "
            "calibrated scale product is carried by the _scaled_mm block scales and the bias is applied by the "
            "GEMM, so results differ numerically from the unfolded static NVFP4 path."
        )
    if config.fuse_producer_packing:
        notes.append(
            "Producer packing fusion is ON: the LayerNorm and GELU producers in front of attn.w_qkv, ffn.net.0 "
            "and ffn.net.3 are packed to NVFP4 in one kernel each, so results differ numerically from the "
            "unfused static NVFP4 path."
        )
    if config.uses_mse_weight_scales:
        notes.append(
            "NVFP4 weight block scales were re-selected by the per-block MSE search "
            f"({WEIGHT_SCALE_MSE_ALGORITHM} v{WEIGHT_SCALE_MSE_ALGORITHM_VERSION}) after the ordinary TorchAO "
            "conversion, so the stored weights differ from the amax-derived ones; see 'weight_scale_mse'."
        )
    if config.uses_four_over_six_weight_scales:
        notes.append(
            "NVFP4 weight block scales were re-selected by the Four-Over-Six comparison "
            f"({WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM} v{WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION}, "
            f"normalization maximum {WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX}, candidate magnitudes "
            f"{list(WEIGHT_SCALE_FOUR_OVER_SIX_MAGNITUDES)}) after the ordinary TorchAO conversion, and the weight "
            "global scale was renormalized with them, so the stored weights differ from the amax-derived ones; see "
            "'weight_scale_four_over_six'."
        )
    if config.uses_local_hessian_weight_scales:
        notes.append(
            "NVFP4 weight block scales were re-selected by the activation-weighted diagonal-Hessian search "
            f"({WEIGHT_SCALE_HESSIAN_ALGORITHM} v{WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION}, damping "
            f"{WEIGHT_SCALE_HESSIAN_DAMPING}) after the ordinary TorchAO conversion. The packed layout and the "
            "inference kernels are unchanged; see 'weight_scale_hessian' for the artifact identity and the "
            "achieved objectives, which are not an accuracy claim."
        )
    if config.uses_awq_clip_weight_scales:
        notes.append(
            "NVFP4 weight block scales were re-selected from an offline AWQ-clip artifact "
            f"({WEIGHT_SCALE_AWQ_CLIP_ALGORITHM} v{WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION}, clipping ratios "
            f"{[float(ratio) for ratio in WEIGHT_SCALE_AWQ_CLIP_RATIOS]}) after the ordinary TorchAO conversion. "
            "The codes were chosen offline against activations quantized with the configured static calibration, "
            "which the artifact is bound to by digest, and against the ordinary template this backend constructs, "
            "which the artifact names and a mismatched backend is refused for; blocks carrying the unclipped code "
            "keep that template's own bytes. The packed layout and the inference kernels are unchanged. "
            "See 'weight_scale_awq_clip' for the artifact identity and the achieved errors, which are not an "
            "accuracy claim."
        )
    if config.uses_gptq_weight_scales:
        notes.append(
            "NVFP4 weight payloads were replaced from an offline GPTQ artifact "
            f"({WEIGHT_SCALE_GPTQ_ALGORITHM} v{WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION}, damping "
            f"{WEIGHT_SCALE_GPTQ_PERC_DAMP}, {WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE}-column update blocks) after the "
            "ordinary TorchAO conversion. The payload was selected offline under Hessians formed from activations "
            "quantized with the configured static calibration, which the artifact is bound to by digest, and under "
            "the ordinary template this backend constructs, whose exact scale bytes every layer was re-checked "
            "against. The block scales, the global scale, the packed layout and the inference kernels are "
            "unchanged; only the FP4 payload bytes differ. See 'weight_scale_gptq' for the artifact identity and "
            "the measured errors, which are not an accuracy claim."
        )
    if config.enabled and facts.qualification == QUALIFICATION_POLICY_ONLY:
        notes.append(
            f"Compute capability {facts.compute_capability} is accepted by capability policy only; this run is "
            "not evidence that the architecture is qualified for NVFP4."
        )
    return notes


def _model_device(model: torch.nn.Module) -> torch.device:
    """Device of the model's first parameter or buffer, defaulting to CPU."""
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _file_identity(path: Optional[str]) -> Optional[Dict[str, int]]:
    """Size and modification time of a file, or ``None`` when the path is unset or does not name a file."""
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return None
    stat = resolved.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _file_content_sha256(path: Optional[str], description: str) -> Optional[str]:
    """
    SHA-256 of a file's exact bytes, or ``None`` when the path is unset.

    Raises:
        ValueError: If a path was given but its bytes cannot be read. Falling back to ``None`` would make two runs
            with different file contents share one identity, which is exactly what the digest exists to prevent.
    """
    if not path:
        return None
    resolved = Path(path).expanduser()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"{description} {resolved} could not be read: {error}") from error
    return hashlib.sha256(raw).hexdigest()


def _read_merge_input(group: str, calibration_path: str) -> Dict[str, Any]:
    """
    Read and strictly validate one labelled calibration input for :func:`merge_calibrations`.

    Args:
        group (str): Data source or streaming stratum this file belongs to.
        calibration_path (str): Path to a schema-v1 calibration JSON file.

    Returns:
        entry (Dict[str, Any]): Group, path, basename, SHA-256, byte size, declared targets, sorted FQNs, the
            per-FQN sample histories, the declared checkpoint (or ``None``), the non-finite provenance status,
            and whether the legacy max-only fallback was used.

    Raises:
        ValueError: If the group is empty or the file is not a valid, complete schema-v1 calibration artifact,
            or if it reports any non-finite activation observation.
    """
    group = str(group).strip()
    if not group:
        raise ValueError(f"Calibration input '{calibration_path}' was given an empty group name.")

    path = Path(calibration_path).expanduser()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"Calibration file {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file {path} must contain a JSON object")
    version = payload.get("version")
    if version != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Calibration file {path} has version {version!r}, but version {CALIBRATION_SCHEMA_VERSION} is required."
        )

    raw_amax = payload.get("activation_amax")
    if not isinstance(raw_amax, dict) or not raw_amax:
        raise ValueError(f"Calibration file {path} must contain a non-empty 'activation_amax' object")
    for fqn in sorted(raw_amax):
        _positive_finite(raw_amax[fqn], f"Calibration entry '{fqn}' in {path}")

    targets = payload.get("targets")
    if targets is None:
        # The legacy artifact predates the targets field; it is a full-model calibration of the same families.
        targets = list(QUANTIZATION_TARGET_SUFFIXES)
    if not isinstance(targets, (list, tuple)) or not all(isinstance(target, str) for target in targets):
        raise ValueError(f"Calibration file {path} must declare 'targets' as a list of strings")

    raw_samples = payload.get(CALIBRATION_SAMPLES_FIELD)
    legacy_fallback = raw_samples is None
    if legacy_fallback:
        # A max-only artifact contributes exactly one observation per module, which makes every percentile of a
        # group that contains it fall back to that maximum rather than silently under-estimating it.
        samples = {fqn: [float(raw_amax[fqn])] for fqn in sorted(raw_amax)}
    else:
        samples = _validate_sample_history(raw_samples, raw_amax, path)

    raw_counts = payload.get(CALIBRATION_NONFINITE_FIELD)
    if raw_counts is None:
        if not legacy_fallback:
            # A shard collected with per-invocation histories is written by this module, which always records the
            # counts; without them the shard cannot testify that its activations were finite.
            raise ValueError(
                f"Calibration file {path} carries '{CALIBRATION_SAMPLES_FIELD}' but no "
                f"'{CALIBRATION_NONFINITE_FIELD}'; re-collect it so the merge can verify that no non-finite "
                "activation was observed."
            )
        nonfinite_status = CALIBRATION_PROVENANCE_UNKNOWN_LEGACY
    else:
        _validate_nonfinite_counts(raw_counts, raw_amax, path)
        nonfinite_status = CALIBRATION_PROVENANCE_CLEAN

    declared_checkpoint = payload.get(CALIBRATION_CHECKPOINT_FIELD)
    if declared_checkpoint is not None:
        declared_checkpoint = _validate_checkpoint_sha256(
            declared_checkpoint, f"'{CALIBRATION_CHECKPOINT_FIELD}' in {path}"
        )

    return {
        "group": group,
        "path": str(path),
        "name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "targets": tuple(str(target) for target in targets),
        "fqns": tuple(sorted(raw_amax)),
        "samples": samples,
        "legacy_fallback": legacy_fallback,
        "nonfinite_status": nonfinite_status,
        "checkpoint_sha256": declared_checkpoint,
    }


def _validate_sample_history(raw_samples: Any, raw_amax: Dict[str, Any], path: Path) -> Dict[str, List[float]]:
    """
    Validate a :data:`CALIBRATION_SAMPLES_FIELD` object against the file's ``activation_amax`` keys.

    The declared maximum must equal the maximum of the history exactly. Both are written from the same Python
    float by :meth:`ActivationAmaxCollector.save`, and JSON round-trips a float exactly, so any difference means
    the two fields describe different data and the file must not be merged.

    Raises:
        ValueError: If the history is not an object over exactly the same FQNs, any history is empty or contains
            a value that is not a finite positive number, or a history disagrees with ``activation_amax``.
    """
    if not isinstance(raw_samples, dict):
        raise ValueError(f"Calibration file {path} must contain '{CALIBRATION_SAMPLES_FIELD}' as an object")
    if set(raw_samples) != set(raw_amax):
        only_samples = sorted(set(raw_samples) - set(raw_amax))
        only_amax = sorted(set(raw_amax) - set(raw_samples))
        raise ValueError(
            f"Calibration file {path} has '{CALIBRATION_SAMPLES_FIELD}' keys that differ from 'activation_amax': "
            f"extra {only_samples[:4]}, missing {only_amax[:4]}."
        )

    samples: Dict[str, List[float]] = {}
    for fqn in sorted(raw_samples):
        history = raw_samples[fqn]
        if not isinstance(history, list) or not history:
            raise ValueError(
                f"Calibration file {path} must give a non-empty list of observations for '{fqn}' in "
                f"'{CALIBRATION_SAMPLES_FIELD}'"
            )
        samples[fqn] = [
            _positive_finite(value, f"Observation {index} of '{fqn}' in {path}") for index, value in enumerate(history)
        ]
        declared = float(raw_amax[fqn])
        observed = max(samples[fqn])
        if observed != declared:
            raise ValueError(
                f"Calibration file {path} declares activation_amax[{fqn!r}] = {declared!r}, but the maximum of "
                f"its '{CALIBRATION_SAMPLES_FIELD}' history is {observed!r}; the two fields describe different "
                "data, so the file is not merged."
            )
    return samples


def _validate_nonfinite_counts(raw_counts: Any, raw_amax: Dict[str, Any], path: Path) -> None:
    """
    Validate a :data:`CALIBRATION_NONFINITE_FIELD` object and reject any file that observed a non-finite value.

    Raises:
        ValueError: If the counts are not an object over exactly the ``activation_amax`` FQNs, a count is not a
            non-negative integer, or any count is positive.
    """
    if not isinstance(raw_counts, dict):
        raise ValueError(f"Calibration file {path} must contain '{CALIBRATION_NONFINITE_FIELD}' as an object")
    if set(raw_counts) != set(raw_amax):
        only_counts = sorted(set(raw_counts) - set(raw_amax))
        only_amax = sorted(set(raw_amax) - set(raw_counts))
        raise ValueError(
            f"Calibration file {path} has '{CALIBRATION_NONFINITE_FIELD}' keys that differ from "
            f"'activation_amax': extra {only_counts[:4]}, missing {only_amax[:4]}."
        )

    affected = []
    for fqn in sorted(raw_counts):
        count = raw_counts[fqn]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Count for '{fqn}' in '{CALIBRATION_NONFINITE_FIELD}' of {path} must be a non-negative integer, "
                f"got {count!r}"
            )
        if count > 0:
            affected.append(fqn)
    if affected:
        raise ValueError(
            f"Calibration file {path} observed non-finite activations in {len(affected)} module(s), e.g. "
            f"{affected[:4]}; its statistics describe only the finite remainder and are not merged. Re-collect "
            "this shard on inputs that do not produce NaN or infinite activations."
        )


def _require_exact_keys(payload: Dict[str, Any], expected: frozenset, described: str, section: str) -> None:
    """Reject any object whose key set is not exactly ``expected``, naming both the extra and the missing keys."""
    unknown = sorted(set(payload) - expected)
    absent = sorted(expected - set(payload))
    if unknown or absent:
        raise ValueError(
            f"{described} must carry exactly the {section} keys {sorted(expected)}; unknown keys {unknown}, "
            f"missing keys {absent}."
        )


def _validate_hessian_provenance(provenance: Any, required: Sequence[str], path: Path) -> Dict[str, Any]:
    """
    Validate the provenance of a diagonal-Hessian artifact: how it was built and from which labelled samples.

    The construction, its version, the objective and the group reduction are constants of this implementation
    rather than free text, so an artifact that states any other value was produced by a different builder and is
    refused instead of being read as if it were this one's.
    """
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Diagonal-Hessian artifact {path} must give 'provenance' as an object, got {type(provenance).__name__}"
        )
    _require_exact_keys(provenance, HESSIAN_PROVENANCE_KEYS, f"Diagonal-Hessian artifact {path}", "provenance")
    for field, expected in (
        ("method", HESSIAN_CONSTRUCTION_METHOD),
        ("method_version", HESSIAN_CONSTRUCTION_METHOD_VERSION),
        ("objective", HESSIAN_OBJECTIVE),
        ("group_reduction", HESSIAN_GROUP_REDUCTION),
    ):
        if provenance[field] != expected or isinstance(provenance[field], bool):
            raise ValueError(
                f"Diagonal-Hessian artifact {path} records '{field}' as {provenance[field]!r}, but this build only "
                f"consumes statistics built with {expected!r}."
            )
    if list(provenance["targets"] or []) != list(QUANTIZATION_TARGET_SUFFIXES):
        raise ValueError(
            f"Diagonal-Hessian artifact {path} declares targets {provenance['targets']!r}, but this build "
            f"quantizes exactly {list(QUANTIZATION_TARGET_SUFFIXES)}."
        )
    if list(provenance["target_fqns"] or []) != list(required):
        raise ValueError(
            f"Diagonal-Hessian artifact {path} names {len(provenance['target_fqns'] or [])} target FQN(s) in its "
            f"provenance, but this model selects {len(required)}; the artifact describes a different target set."
        )
    count = _validated_non_negative_int(provenance["target_module_count"], f"'target_module_count' in {path}")
    if count != len(required):
        raise ValueError(
            f"Diagonal-Hessian artifact {path} records target_module_count {count}, but this model selects "
            f"{len(required)} NVFP4 W4A4 module(s)."
        )

    _validate_hessian_sources(provenance["sources"], path)
    aggregate = provenance["aggregate"]
    if not isinstance(aggregate, dict):
        raise ValueError(f"Diagonal-Hessian artifact {path} must give 'aggregate' as an object.")
    _require_exact_keys(aggregate, HESSIAN_AGGREGATE_KEYS, f"Diagonal-Hessian artifact {path}", "aggregate")
    for field in ("module_count", "source_count", "moment_count"):
        _validated_non_negative_int(aggregate[field], f"'aggregate.{field}' in {path}")
    for field in ("moment_min", "moment_max"):
        value = aggregate[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"'aggregate.{field}' in {path} must be a finite number, got {value!r}")
    if not isinstance(aggregate["source_labels"], list):
        raise ValueError(f"'aggregate.source_labels' in {path} must be a list.")
    return provenance


_HESSIAN_SOURCE_COUNT_FIELDS: Tuple[str, ...] = (
    "size_bytes",
    "seed",
    "max_rows",
    "sampled_row_count",
    "finite_row_count",
    "nonfinite_row_count",
)


def _validate_hessian_sources(sources: Any, path: Path) -> None:
    """Validate the labelled activation sources the moments were built from, key by key and count by count."""
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} must record a non-empty 'sources' list; an artifact that names no "
            "activation source cannot say which data produced its moments."
        )
    labels: List[str] = []
    for index, source in enumerate(sources):
        described = f"Entry {index} of 'sources' in {path}"
        if not isinstance(source, dict):
            raise ValueError(f"{described} must be an object, got {type(source).__name__}")
        _require_exact_keys(source, HESSIAN_SOURCE_KEYS, described, "source")
        for field in ("label", "name"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"{described} must carry a non-empty '{field}'.")
        _validate_checkpoint_sha256(source["sha256"], f"{described} 'sha256'")
        counts = {
            field: _validated_non_negative_int(source[field], f"{described} '{field}'")
            for field in _HESSIAN_SOURCE_COUNT_FIELDS
        }
        for field in ("size_bytes", "max_rows", "sampled_row_count"):
            if counts[field] <= 0:
                raise ValueError(f"{described} must give a positive '{field}', got {counts[field]}.")
        if counts["sampled_row_count"] > counts["finite_row_count"]:
            raise ValueError(
                f"{described} retained {counts['sampled_row_count']} row(s) out of {counts['finite_row_count']} "
                "finite one(s); a bounded reservoir cannot keep more rows than it saw."
            )
        if not isinstance(source["metadata"], dict):
            raise ValueError(f"{described} must give 'metadata' as an object.")
        labels.append(source["label"])
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} names the source label(s) {duplicates} more than once; every group "
            "is weighted equally, so a group named twice would have been counted twice."
        )


def _validate_hessian_aggregate(
    provenance: Dict[str, Any], second_moments: Dict[str, List[float]], required: Sequence[str], path: Path
) -> None:
    """
    Reject a provenance aggregate that contradicts the sections it claims to summarize.

    The aggregate is the one place where the artifact restates what the rest of it contains: how many modules and
    source groups it covers, which groups those were, and how many second moments it carries between which
    extremes. A recomputed ``provenance_sha256`` proves only that nobody edited the section after it was hashed, so
    every field is recomputed here from the already-validated target, source and moment sections and compared. The
    comparison is exact rather than approximate on purpose: the builder writes these values from the very floats it
    writes into the moments, and JSON round-trips a float exactly, so any difference at all is an artifact whose
    summary was not produced from its own contents.

    Args:
        provenance (Dict[str, Any]): Structurally validated provenance section.
        second_moments (Dict[str, List[float]]): Validated per-FQN moment vectors.
        required (Sequence[str]): The selected NVFP4 W4A4 FQNs the artifact must describe.
        path (Path): Artifact path, named in error messages.

    Raises:
        ValueError: If any aggregate field disagrees with the section it summarizes.
    """
    aggregate = provenance["aggregate"]
    values = [value for fqn in sorted(second_moments) for value in second_moments[fqn]]
    labels = [source["label"] for source in provenance["sources"]]
    for field, expected in (
        ("module_count", len(required)),
        ("source_count", len(provenance["sources"])),
        ("moment_count", len(values)),
    ):
        if int(aggregate[field]) != int(expected):
            raise ValueError(
                f"'aggregate.{field}' in {path} records {aggregate[field]!r}, but the artifact's own sections give "
                f"{expected}; the aggregate does not summarize this artifact."
            )
    if list(aggregate["source_labels"]) != labels:
        raise ValueError(
            f"'aggregate.source_labels' in {path} records {aggregate['source_labels']!r}, but its 'sources' are "
            f"labelled {labels}; the aggregate does not summarize this artifact."
        )
    for field, expected_value in (("moment_min", min(values)), ("moment_max", max(values))):
        if float(aggregate[field]) != float(expected_value):
            raise ValueError(
                f"'aggregate.{field}' in {path} records {aggregate[field]!r}, but the artifact's second moments "
                f"give {expected_value!r}; the aggregate does not summarize this artifact."
            )


def _validate_hessian_moments(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, List[float]]:
    """
    Validate the per-FQN second-moment vectors against the live modules they will weight.

    Coverage must be exact and canonical, every vector must be as wide as its module's ``in_features``, and every
    value must be a finite non-negative number with a positive per-vector mean -- in FP32, the precision the search
    actually weights with, not only as a JSON double. A vector that fails any of these describes a layer other than
    the one it names, or a vector the packer would reject after the conversion had already started.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Diagonal-Hessian artifact {path} must contain a non-empty 'diagonal_hessian' object")
    noncanonical = sorted(fqn for fqn in raw if not isinstance(fqn, str) or fqn != _canonical_fqn(fqn))
    if noncanonical:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} names {len(noncanonical)} module(s) in non-canonical form, e.g. "
            f"{noncanonical[:4]}; entries must be exact uncompiled target FQNs."
        )
    if sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required))
        missing = sorted(set(required) - set(raw))
        raise ValueError(
            f"Diagonal-Hessian artifact {path} must cover exactly the {len(required)} selected NVFP4 W4A4 "
            f"module(s); extra {extra[:4]}, missing {missing[:4]}."
        )

    moments: Dict[str, List[float]] = {}
    for fqn in required:
        values = raw[fqn]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Diagonal-Hessian artifact {path} must give a non-empty list of second moments for '{fqn}', got "
                f"{type(values).__name__}"
            )
        in_features = int(getattr(modules[fqn], "in_features", 0))
        if len(values) != in_features:
            raise ValueError(
                f"Diagonal-Hessian artifact {path} gives {len(values)} second moment(s) for '{fqn}', but the live "
                f"module has {in_features} input channel(s)."
            )
        vector: List[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Second moment {index} of '{fqn}' in {path} must be a number, got {value!r}")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"Second moment {index} of '{fqn}' in {path} must be finite and non-negative, got {value!r}"
                )
            vector.append(number)
        if sum(vector) <= 0.0:
            raise ValueError(
                f"The second moments of '{fqn}' in {path} are identically zero, which describes a layer whose "
                "inputs never varied; no activation sample can support that."
            )
        # The runtime materializes this vector as an FP32 device tensor before the search, so it is validated as
        # exactly that here. A value that is finite as a JSON double but overflows FP32, or a vector whose FP32
        # mean is not finite and positive, must fail now, on a still-unconverted model, rather than halfway
        # through the conversion when the packer first sees it.
        try:
            damped_second_moments(torch.tensor(vector, dtype=torch.float32))
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"The second moments of '{fqn}' in {path} are not usable as the FP32 vector the activation-"
                f"weighted search weights with: {error}"
            ) from error
        moments[fqn] = vector
    return moments


def _validate_hessian_weight_digests(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, str]:
    """
    Validate the recorded weight digests and verify them against the live, still-unconverted weights.

    This is the binding between the offline statistics and the model in memory, and it runs before any TorchAO
    conversion has mutated a weight: a checkpoint whose weights differ from the ones the moments were built on is
    a different model, and its moments would weight the wrong input channels.
    """
    if not isinstance(raw, dict) or sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required)) if isinstance(raw, dict) else []
        missing = sorted(set(required) - set(raw)) if isinstance(raw, dict) else list(required)
        raise ValueError(
            f"Diagonal-Hessian artifact {path} must give 'weight_sha256' over exactly the {len(required)} selected "
            f"module(s); extra {extra[:4]}, missing {missing[:4]}."
        )
    digests: Dict[str, str] = {}
    mismatched: List[str] = []
    for fqn in required:
        expected = _validate_checkpoint_sha256(raw[fqn], f"'weight_sha256[{fqn}]' in {path}")
        weight = getattr(modules[fqn], "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                f"Module '{fqn}' exposes no weight tensor, so the diagonal-Hessian artifact {path} cannot be bound "
                "to this model."
            )
        if nvfp4_weight_digest(weight) != expected:
            mismatched.append(fqn)
        digests[fqn] = expected
    if mismatched:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} was built on different weights: {len(mismatched)} module(s) do not "
            f"match their recorded digest, e.g. {mismatched[:4]}. Rebuild the statistics on this checkpoint."
        )
    return digests


def _verify_hessian_section_digest(payload: Dict[str, Any], digest_key: str, section_key: str, path: Path) -> str:
    """
    Verify one recorded section digest against the section the artifact actually carries.

    The recorded value must be a well-formed digest and must equal the :data:`SECTION_DIGEST_METHOD` digest of the
    parsed section. An artifact whose moments or provenance were edited after the builder recorded them is refused
    rather than consumed as if the builder had produced those values.
    """
    expected = _validate_checkpoint_sha256(payload[digest_key], f"'{digest_key}' in {path}")
    try:
        actual = nvfp4_section_digest(payload[section_key])
    except ValueError as error:
        raise ValueError(
            f"The '{section_key}' section of diagonal-Hessian artifact {path} cannot be canonicalized for its "
            f"'{digest_key}': {error}"
        ) from error
    if actual != expected:
        raise ValueError(
            f"Diagonal-Hessian artifact {path} records {digest_key} {expected}, but its '{section_key}' section "
            f"hashes to {actual}; the artifact was modified after it was built."
        )
    return expected


def _validate_awq_clip_arithmetic(arithmetic: Any, path: Path) -> str:
    """
    Reject an AWQ-clip artifact whose recorded arithmetic is not the one this build implements.

    Every value here defines the selection rather than describing it: a different ratio list, a different order, a
    different block size, a different tie rule, a different candidate scale rule or a different activation
    quantize/dequantize would have chosen different codes, and reconstructing those codes with *this* arithmetic
    would produce weights nobody ever evaluated.

    Returns:
        template_arithmetic (str): The ordinary-template construction the unclipped code was scored against.
    """
    if not isinstance(arithmetic, dict):
        raise ValueError(f"AWQ-clip artifact {path} must give 'arithmetic' as an object, got {type(arithmetic)}")
    _require_exact_keys(arithmetic, AWQ_CLIP_ARITHMETIC_KEYS, f"AWQ-clip artifact {path}", "arithmetic")

    # The one arithmetic field with two legitimate values, because the unclipped code reproduces whichever ordinary
    # template the deployment constructs. Which of the two this artifact was scored against is checked against the
    # run's own backend separately, by :func:`require_awq_clip_template_arithmetic`.
    if arithmetic["template_arithmetic"] not in WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS:
        raise ValueError(
            f"AWQ-clip artifact {path} records 'template_arithmetic' as {arithmetic['template_arithmetic']!r}, but "
            f"this build reproduces exactly {list(WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS)}; the unclipped code "
            "means 'leave this block as that construction wrote it', so an unknown construction is unreproducible."
        )

    ratios = arithmetic["clip_ratios"]
    if not isinstance(ratios, list) or len(ratios) != WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT:
        raise ValueError(
            f"AWQ-clip artifact {path} records {len(ratios) if isinstance(ratios, list) else ratios!r} clip "
            f"ratio(s), but this build implements exactly {WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT}."
        )
    for index, (recorded, expected) in enumerate(zip(ratios, WEIGHT_SCALE_AWQ_CLIP_RATIOS)):
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or float(recorded) != float(expected):
            raise ValueError(
                f"Clip ratio {index} in {path} is {recorded!r}, but this build evaluates {expected!r} at that "
                "position; the ratio list and its order are part of the algorithm's identity."
            )
    for field, expected_value in (
        ("block_size", int(WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE)),
        ("tie_rule", AWQ_CLIP_TIE_RULE),
        ("objective", AWQ_CLIP_OBJECTIVE),
        ("group_reduction", AWQ_CLIP_GROUP_REDUCTION),
        ("candidate_scale_rule", AWQ_CLIP_CANDIDATE_SCALE_RULE),
        ("activation_qdq", AWQ_CLIP_ACTIVATION_QDQ),
        ("fp4_max", float(NVFP4_MAX)),
        ("fp8_e4m3_max", float(FP8_E4M3_MAX)),
        ("scale_min", float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MIN)),
        ("modelopt_reference_version", MODELOPT_REFERENCE_VERSION),
        ("modelopt_reference_wheel_sha256", MODELOPT_REFERENCE_WHEEL_SHA256),
    ):
        recorded = arithmetic[field]
        if isinstance(recorded, bool) or recorded != expected_value:
            raise ValueError(
                f"AWQ-clip artifact {path} records '{field}' as {recorded!r}, but this build implements exactly "
                f"{expected_value!r}; a different value is a different algorithm."
            )
    return str(arithmetic["template_arithmetic"])


def _validate_awq_clip_calibration(
    section: Any, calibration_path: str, scale_margin: float, checkpoint_sha256: str, path: Path
) -> Dict[str, Any]:
    """
    Bind an AWQ-clip artifact to the exact activation-calibration file the run is configured with.

    The ratio codes were selected against activations quantized with these values, so a run that consumes another
    calibration file -- or the same file re-merged, or the same values with another margin -- would execute codes
    chosen for activations it never produces. The recorded identity is therefore recomputed from the configured
    file's actual bytes and compared field by field.
    """
    if not isinstance(section, dict):
        raise ValueError(
            f"AWQ-clip artifact {path} must give 'activation_calibration' as an object, got {type(section).__name__}"
        )
    _require_exact_keys(section, AWQ_CLIP_CALIBRATION_KEYS, f"AWQ-clip artifact {path}", "activation-calibration")
    margin = section["scale_margin"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError(f"'activation_calibration.scale_margin' in {path} must be a number, got {margin!r}")
    if float(margin) != float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN) or float(margin) != float(scale_margin):
        raise ValueError(
            f"AWQ-clip artifact {path} was built with activation scale margin {float(margin)!r}, but this run uses "
            f"{float(scale_margin)!r} and this build requires exactly "
            f"{float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN)!r}; the codes were selected against activations quantized "
            "with the artifact's margin."
        )

    # The artifact's own copy of the baked-headroom contract is checked before the field-by-field comparison, so a
    # file that recorded 'headroom_baked_in: false' is named for what it is rather than reported as a mismatch.
    if section["headroom_baked_in"] is not True:
        raise ValueError(
            f"AWQ-clip artifact {path} records its calibration's 'headroom_baked_in' as "
            f"{section['headroom_baked_in']!r}; this method only consumes a calibration whose headroom is already "
            f"baked into its values, because it runs at scale margin {float(margin)!r}."
        )
    headroom = _validated_finite_number(section["headroom"], f"'activation_calibration.headroom' in {path}")
    if headroom <= 0.0:
        raise ValueError(f"'activation_calibration.headroom' in {path} must be positive, got {headroom!r}")

    actual = nvfp4_awq_clip_calibration_identity(calibration_path)
    if actual["scale_mode"] != WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE:
        raise ValueError(
            f"Calibration file {actual['name']} declares scale_mode {actual['scale_mode']!r}, but an AWQ-clip run "
            f"consumes exactly a '{WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE}' calibration artifact."
        )
    mismatched = [field for field in sorted(actual) if section[field] != actual[field]]
    if mismatched:
        field = mismatched[0]
        raise ValueError(
            f"AWQ-clip artifact {path} was built against another activation calibration: it records "
            f"'{field}' as {section[field]!r}, but the configured calibration file gives {actual[field]!r} "
            f"({len(mismatched)} mismatched field(s): {mismatched}). Rebuild the codes against this calibration."
        )
    if section["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            f"AWQ-clip artifact {path} is bound to a calibration collected on checkpoint "
            f"{section['checkpoint_sha256']!r}, but the artifact itself names checkpoint {checkpoint_sha256}; "
            "activation scales are only valid for the checkpoint that produced them."
        )
    recorded_runtime_margin = _validated_finite_number(
        section["runtime_scale_margin"], f"'activation_calibration.runtime_scale_margin' in {path}"
    )
    if recorded_runtime_margin != float(margin):
        raise ValueError(
            f"The calibration file bound to {path} presumes a runtime scale margin of "
            f"{float(recorded_runtime_margin)!r}, but the artifact was built with {float(margin)!r}; its baked-in "
            "headroom would be applied twice or not at all."
        )
    return {**actual, "scale_margin": float(margin)}


def _validate_awq_clip_codes(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, Any]:
    """
    Decode and validate the per-FQN ratio-code payloads against the live modules they will repack.

    Coverage must be exact and canonical, every payload must decode to exactly the number of bytes its declared
    ``[out_features, in_features / 16]`` shape needs, that shape must be the live module's, and every code must
    index the fixed ratio list. A payload that fails any of these describes a layer other than the one it names.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"AWQ-clip artifact {path} must contain a non-empty 'ratio_codes' object")
    noncanonical = sorted(fqn for fqn in raw if not isinstance(fqn, str) or fqn != _canonical_fqn(fqn))
    if noncanonical:
        raise ValueError(
            f"AWQ-clip artifact {path} names {len(noncanonical)} module(s) in non-canonical form, e.g. "
            f"{noncanonical[:4]}; entries must be exact uncompiled target FQNs."
        )
    if sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required))
        missing = sorted(set(required) - set(raw))
        raise ValueError(
            f"AWQ-clip artifact {path} must cover exactly the {len(required)} selected NVFP4 W4A4 module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}."
        )

    ratio_codes: Dict[str, bytes] = {}
    code_shapes: Dict[str, List[int]] = {}
    histograms: Dict[str, List[int]] = {}
    for fqn in required:
        described = f"'ratio_codes[{fqn}]' in {path}"
        entry = raw[fqn]
        if not isinstance(entry, dict):
            raise ValueError(f"{described} must be an object, got {type(entry).__name__}")
        _require_exact_keys(entry, AWQ_CLIP_CODE_KEYS, described, "ratio-code")
        shape = entry["shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"{described} must declare 'shape' as [out_features, in_features / 16], got {shape!r}")
        rows = _validated_non_negative_int(shape[0], f"{described} 'shape[0]'")
        blocks = _validated_non_negative_int(shape[1], f"{described} 'shape[1]'")
        module = modules[fqn]
        out_features = int(getattr(module, "out_features", 0))
        in_features = int(getattr(module, "in_features", 0))
        if in_features % NVFP4_BLOCK_SIZE != 0 or in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Module '{fqn}' has shape ({out_features}, {in_features}), which no AWQ-clip code matrix of "
                f"{NVFP4_BLOCK_SIZE}-weight blocks can describe."
            )
        if (rows, blocks) != (out_features, in_features // NVFP4_BLOCK_SIZE):
            raise ValueError(
                f"{described} declares shape {[rows, blocks]}, but the live module needs "
                f"{[out_features, in_features // NVFP4_BLOCK_SIZE]}; the codes describe a different weight."
            )
        decoded = _decode_awq_clip_codes(entry["codes"], rows * blocks, described)
        codes = torch.frombuffer(bytearray(decoded), dtype=torch.uint8)
        maximum = int(codes.max())
        if maximum >= WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT:
            raise ValueError(
                f"{described} holds the code {maximum}, but the {WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT} clip ratios "
                f"are indexed by 0..{WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT - 1}."
            )
        counts = torch.bincount(codes.to(torch.long), minlength=WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
        ratio_codes[fqn] = decoded
        code_shapes[fqn] = [rows, blocks]
        histograms[fqn] = [int(count) for count in counts.tolist()]
    return {"ratio_codes": ratio_codes, "code_shapes": code_shapes, "ratio_histogram": histograms}


def _decode_awq_clip_codes(value: Any, expected_bytes: int, described: str) -> bytes:
    """Decode one strict base64 ratio-code payload, refusing anything that is not exactly the declared length."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{described} must carry 'codes' as a non-empty base64 string, got {value!r}")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{described} does not carry valid base64 ratio codes: {error}") from error
    if len(decoded) != expected_bytes:
        raise ValueError(
            f"{described} decodes to {len(decoded)} code byte(s), but its declared shape needs exactly "
            f"{expected_bytes}; the payload is truncated or padded."
        )
    return decoded


def _validate_awq_clip_provenance(provenance: Any, required: Sequence[str], path: Path) -> Dict[str, Any]:
    """
    Validate the provenance of an AWQ-clip artifact: how it was built, from which samples, and what it selected.

    The construction, its version, the objective and the group reduction are constants of this implementation
    rather than free text, so an artifact that states any other value was produced by a different builder.
    """
    if not isinstance(provenance, dict):
        raise ValueError(
            f"AWQ-clip artifact {path} must give 'provenance' as an object, got {type(provenance).__name__}"
        )
    _require_exact_keys(provenance, AWQ_CLIP_PROVENANCE_KEYS, f"AWQ-clip artifact {path}", "provenance")
    for field, expected in (
        ("method", AWQ_CLIP_CONSTRUCTION_METHOD),
        ("method_version", AWQ_CLIP_CONSTRUCTION_METHOD_VERSION),
        ("objective", AWQ_CLIP_OBJECTIVE),
        ("group_reduction", AWQ_CLIP_GROUP_REDUCTION),
    ):
        if provenance[field] != expected or isinstance(provenance[field], bool):
            raise ValueError(
                f"AWQ-clip artifact {path} records '{field}' as {provenance[field]!r}, but this build only consumes "
                f"codes built with {expected!r}."
            )
    if list(provenance["targets"] or []) != list(QUANTIZATION_TARGET_SUFFIXES):
        raise ValueError(
            f"AWQ-clip artifact {path} declares targets {provenance['targets']!r}, but this build quantizes exactly "
            f"{list(QUANTIZATION_TARGET_SUFFIXES)}."
        )
    if list(provenance["target_fqns"] or []) != list(required):
        raise ValueError(
            f"AWQ-clip artifact {path} names {len(provenance['target_fqns'] or [])} target FQN(s) in its "
            f"provenance, but this model selects {len(required)}; the artifact describes a different target set."
        )
    count = _validated_non_negative_int(provenance["target_module_count"], f"'target_module_count' in {path}")
    if count != len(required):
        raise ValueError(
            f"AWQ-clip artifact {path} records target_module_count {count}, but this model selects {len(required)} "
            "NVFP4 W4A4 module(s)."
        )

    _validate_awq_clip_sources(provenance["sources"], path)
    _validate_awq_clip_modules(provenance["modules"], required, path)
    aggregate = provenance["aggregate"]
    if not isinstance(aggregate, dict):
        raise ValueError(f"AWQ-clip artifact {path} must give 'aggregate' as an object.")
    _require_exact_keys(aggregate, AWQ_CLIP_AGGREGATE_KEYS, f"AWQ-clip artifact {path}", "aggregate")
    for field in ("module_count", "source_count", "block_count"):
        _validated_non_negative_int(aggregate[field], f"'aggregate.{field}' in {path}")
    for field in ("selected_objective", "unclipped_objective"):
        _validated_finite_number(aggregate[field], f"'aggregate.{field}' in {path}")
    if not isinstance(aggregate["source_labels"], list):
        raise ValueError(f"'aggregate.source_labels' in {path} must be a list.")
    _validated_ratio_histogram(aggregate["ratio_histogram"], f"'aggregate.ratio_histogram' in {path}")
    return provenance


def _validate_awq_clip_sources(sources: Any, path: Path) -> None:
    """Validate the labelled activation sources the codes were selected on, key by key and count by count."""
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"AWQ-clip artifact {path} must record a non-empty 'sources' list; an artifact that names no activation "
            "source cannot say which data selected its codes."
        )
    labels: List[str] = []
    for index, source in enumerate(sources):
        described = f"Entry {index} of 'sources' in {path}"
        if not isinstance(source, dict):
            raise ValueError(f"{described} must be an object, got {type(source).__name__}")
        _require_exact_keys(source, AWQ_CLIP_SOURCE_KEYS, described, "source")
        for field in ("label", "name"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"{described} must carry a non-empty '{field}'.")
        _validate_checkpoint_sha256(source["sha256"], f"{described} 'sha256'")
        counts = {
            field: _validated_non_negative_int(source[field], f"{described} '{field}'")
            for field in _HESSIAN_SOURCE_COUNT_FIELDS
        }
        for field in ("size_bytes", "max_rows", "sampled_row_count"):
            if counts[field] <= 0:
                raise ValueError(f"{described} must give a positive '{field}', got {counts[field]}.")
        if counts["sampled_row_count"] > counts["finite_row_count"]:
            raise ValueError(
                f"{described} retained {counts['sampled_row_count']} row(s) out of {counts['finite_row_count']} "
                "finite one(s); a bounded reservoir cannot keep more rows than it saw."
            )
        if not isinstance(source["metadata"], dict):
            raise ValueError(f"{described} must give 'metadata' as an object.")
        labels.append(source["label"])
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"AWQ-clip artifact {path} names the source label(s) {duplicates} more than once; every group is "
            "weighted equally, so a group named twice would have been counted twice."
        )


def _validate_awq_clip_modules(modules: Any, required: Sequence[str], path: Path) -> None:
    """Validate the per-module evidence section: exact coverage, a well-formed histogram and finite objectives."""
    if not isinstance(modules, dict) or sorted(modules) != list(required):
        extra = sorted(set(modules) - set(required)) if isinstance(modules, dict) else []
        missing = sorted(set(required) - set(modules)) if isinstance(modules, dict) else list(required)
        raise ValueError(
            f"AWQ-clip artifact {path} must give 'provenance.modules' over exactly the {len(required)} selected "
            f"module(s); extra {extra[:4]}, missing {missing[:4]}."
        )
    for fqn in required:
        described = f"'provenance.modules[{fqn}]' in {path}"
        entry = modules[fqn]
        if not isinstance(entry, dict):
            raise ValueError(f"{described} must be an object, got {type(entry).__name__}")
        _require_exact_keys(entry, AWQ_CLIP_MODULE_KEYS, described, "module")
        blocks = _validated_non_negative_int(entry["block_count"], f"{described} 'block_count'")
        if blocks <= 0:
            raise ValueError(f"{described} must describe at least one 16-weight block, got {blocks}.")
        histogram = _validated_ratio_histogram(entry["ratio_histogram"], f"{described} 'ratio_histogram'")
        if sum(histogram) != blocks:
            raise ValueError(
                f"{described} counts {sum(histogram)} block(s) in its ratio histogram but declares {blocks}; every "
                "block selects exactly one ratio."
            )
        for field in ("selected_objective", "unclipped_objective"):
            value = _validated_finite_number(entry[field], f"{described} '{field}'")
            if value < 0.0:
                raise ValueError(f"{described} '{field}' is negative ({value!r}); it is a mean of squared errors.")


def _validate_awq_clip_aggregate(
    provenance: Dict[str, Any], histograms: Dict[str, List[int]], required: Sequence[str], path: Path
) -> None:
    """
    Reject an artifact whose per-module and aggregate claims contradict the codes it actually carries.

    A recomputed ``provenance_sha256`` proves only that nobody edited the section after it was hashed, so every
    claim is recomputed here from the decoded ratio codes and from the already-validated sections. The comparison
    is exact rather than approximate: the counts are integers, and the two aggregate objectives are recomputed
    from the artifact's own per-module numbers in the same canonical order the builder reduced them in, which
    python float arithmetic and a JSON round trip both reproduce exactly.
    """
    modules = provenance["modules"]
    aggregate = provenance["aggregate"]
    labels = [source["label"] for source in provenance["sources"]]
    for fqn in required:
        if list(modules[fqn]["ratio_histogram"]) != list(histograms[fqn]):
            raise ValueError(
                f"'provenance.modules[{fqn}].ratio_histogram' in {path} records "
                f"{list(modules[fqn]['ratio_histogram'])}, but its own ratio codes give {list(histograms[fqn])}; "
                "the summary does not describe the codes this artifact carries."
            )
    expected_histogram = [
        sum(int(histograms[fqn][index]) for fqn in required) for index in range(WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
    ]
    for field, expected in (
        ("module_count", len(required)),
        ("source_count", len(provenance["sources"])),
        ("block_count", sum(int(modules[fqn]["block_count"]) for fqn in required)),
    ):
        if int(aggregate[field]) != int(expected):
            raise ValueError(
                f"'aggregate.{field}' in {path} records {aggregate[field]!r}, but the artifact's own sections give "
                f"{expected}; the aggregate does not summarize this artifact."
            )
    if list(aggregate["source_labels"]) != labels:
        raise ValueError(
            f"'aggregate.source_labels' in {path} records {aggregate['source_labels']!r}, but its 'sources' are "
            f"labelled {labels}; the aggregate does not summarize this artifact."
        )
    if list(aggregate["ratio_histogram"]) != expected_histogram:
        raise ValueError(
            f"'aggregate.ratio_histogram' in {path} records {list(aggregate['ratio_histogram'])}, but its own ratio "
            f"codes give {expected_histogram}; the aggregate does not summarize this artifact."
        )
    for field in ("selected_objective", "unclipped_objective"):
        expected_value = nvfp4_awq_clip_weighted_objective(modules, required, field)
        if float(aggregate[field]) != float(expected_value):
            raise ValueError(
                f"'aggregate.{field}' in {path} records {aggregate[field]!r}, but its per-module values reduce to "
                f"{expected_value!r}; the aggregate does not summarize this artifact."
            )


def _validate_awq_clip_weight_digests(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, str]:
    """
    Validate the recorded weight digests and verify them against the live, still-unconverted weights.

    This is the binding between the offline ratio search and the model in memory, and it runs before any TorchAO
    conversion has mutated a weight: a checkpoint whose weights differ from the ones the codes were selected on is
    a different model, and its codes would clip the wrong blocks.
    """
    if not isinstance(raw, dict) or sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required)) if isinstance(raw, dict) else []
        missing = sorted(set(required) - set(raw)) if isinstance(raw, dict) else list(required)
        raise ValueError(
            f"AWQ-clip artifact {path} must give 'weight_sha256' over exactly the {len(required)} selected "
            f"module(s); extra {extra[:4]}, missing {missing[:4]}."
        )
    digests: Dict[str, str] = {}
    mismatched: List[str] = []
    for fqn in required:
        expected = _validate_checkpoint_sha256(raw[fqn], f"'weight_sha256[{fqn}]' in {path}")
        weight = getattr(modules[fqn], "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                f"Module '{fqn}' exposes no weight tensor, so the AWQ-clip artifact {path} cannot be bound to this "
                "model."
            )
        if nvfp4_weight_digest(weight) != expected:
            mismatched.append(fqn)
        digests[fqn] = expected
    if mismatched:
        raise ValueError(
            f"AWQ-clip artifact {path} was built on different weights: {len(mismatched)} module(s) do not match "
            f"their recorded digest, e.g. {mismatched[:4]}. Rebuild the codes on this checkpoint."
        )
    return digests


def _verify_awq_clip_section_digest(payload: Dict[str, Any], digest_key: str, section_key: str, path: Path) -> str:
    """Verify one recorded AWQ-clip section digest against the section the artifact actually carries."""
    expected = _validate_checkpoint_sha256(payload[digest_key], f"'{digest_key}' in {path}")
    try:
        actual = nvfp4_section_digest(payload[section_key])
    except ValueError as error:
        raise ValueError(
            f"The '{section_key}' section of AWQ-clip artifact {path} cannot be canonicalized for its "
            f"'{digest_key}': {error}"
        ) from error
    if actual != expected:
        raise ValueError(
            f"AWQ-clip artifact {path} records {digest_key} {expected}, but its '{section_key}' section hashes to "
            f"{actual}; the artifact was modified after it was built."
        )
    return expected


def _validate_gptq_arithmetic(arithmetic: Any, path: Path) -> str:
    """
    Reject a GPTQ artifact whose recorded arithmetic is not the one this build implements.

    Every value here defines the payload rather than describing it: a different damping, Hessian rule, group
    reduction, dead-column rule, inverse, update block, NVFP4 block, template-scale rule or activation
    quantize/dequantize would have written different FP4 codes, and deploying those codes under *this* arithmetic
    would execute a weight nobody ever selected.

    Returns:
        template_arithmetic (str): The ordinary-template construction the payload was written under.
    """
    if not isinstance(arithmetic, dict):
        raise ValueError(f"GPTQ artifact {path} must give 'arithmetic' as an object, got {type(arithmetic).__name__}")
    _require_exact_keys(arithmetic, GPTQ_ARITHMETIC_KEYS, f"GPTQ artifact {path}", "arithmetic")

    # The one arithmetic field with two legitimate values, because the payload is written under whichever ordinary
    # template the deployment constructs. Which of the two this artifact was written for is checked against the
    # run's own backend separately, by :func:`require_gptq_template_arithmetic`.
    if arithmetic["template_arithmetic"] not in WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS:
        raise ValueError(
            f"GPTQ artifact {path} records 'template_arithmetic' as {arithmetic['template_arithmetic']!r}, but this "
            f"build reproduces exactly {list(WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS)}; the payload only decodes "
            "against the scales one of those constructions produces."
        )
    for field, expected_value in (
        ("perc_damp", float(WEIGHT_SCALE_GPTQ_PERC_DAMP)),
        ("update_block_size", int(WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE)),
        ("block_size", int(WEIGHT_SCALE_GPTQ_BLOCK_SIZE)),
        ("hessian_rule", GPTQ_HESSIAN_RULE),
        ("group_reduction", GPTQ_GROUP_REDUCTION),
        ("dead_column_rule", GPTQ_DEAD_COLUMN_RULE),
        ("inverse_rule", GPTQ_INVERSE_RULE),
        ("template_scale_rule", GPTQ_TEMPLATE_SCALE_RULE),
        ("activation_qdq", GPTQ_ACTIVATION_QDQ),
        ("objective", GPTQ_OBJECTIVE),
        ("hessian_digest_method", GPTQ_HESSIAN_DIGEST_METHOD),
        ("fp4_max", float(NVFP4_MAX)),
        ("fp8_e4m3_max", float(FP8_E4M3_MAX)),
        ("modelopt_reference_version", MODELOPT_REFERENCE_VERSION),
        ("modelopt_reference_wheel_sha256", MODELOPT_REFERENCE_WHEEL_SHA256),
    ):
        recorded = arithmetic[field]
        if isinstance(recorded, bool) or recorded != expected_value:
            raise ValueError(
                f"GPTQ artifact {path} records '{field}' as {recorded!r}, but this build implements exactly "
                f"{expected_value!r}; a different value is a different algorithm."
            )
    return str(arithmetic["template_arithmetic"])


def _validate_gptq_calibration(
    section: Any, calibration_path: str, scale_margin: float, checkpoint_sha256: str, path: Path
) -> Dict[str, Any]:
    """
    Bind a GPTQ artifact to the exact activation-calibration file the run is configured with.

    The Hessians were formed from activations quantized with these values, so a run that consumes another
    calibration file -- or the same file re-merged, or the same values with another margin -- would execute a
    payload selected for activations it never produces. The recorded identity is therefore recomputed from the
    configured file's actual bytes and compared field by field, exactly as the AWQ-clip binding does.
    """
    if not isinstance(section, dict):
        raise ValueError(
            f"GPTQ artifact {path} must give 'activation_calibration' as an object, got {type(section).__name__}"
        )
    _require_exact_keys(section, GPTQ_CALIBRATION_KEYS, f"GPTQ artifact {path}", "activation-calibration")
    margin = section["scale_margin"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError(f"'activation_calibration.scale_margin' in {path} must be a number, got {margin!r}")
    if float(margin) != float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN) or float(margin) != float(scale_margin):
        raise ValueError(
            f"GPTQ artifact {path} was built with activation scale margin {float(margin)!r}, but this run uses "
            f"{float(scale_margin)!r} and this build requires exactly "
            f"{float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN)!r}; the Hessians were formed from activations quantized with "
            "the artifact's margin."
        )
    if section["headroom_baked_in"] is not True:
        raise ValueError(
            f"GPTQ artifact {path} records its calibration's 'headroom_baked_in' as "
            f"{section['headroom_baked_in']!r}; this method only consumes a calibration whose headroom is already "
            f"baked into its values, because it runs at scale margin {float(margin)!r}."
        )
    headroom = _validated_finite_number(section["headroom"], f"'activation_calibration.headroom' in {path}")
    if headroom <= 0.0:
        raise ValueError(f"'activation_calibration.headroom' in {path} must be positive, got {headroom!r}")

    actual = nvfp4_awq_clip_calibration_identity(calibration_path)
    if actual["scale_mode"] != WEIGHT_SCALE_GPTQ_SCALE_MODE:
        raise ValueError(
            f"Calibration file {actual['name']} declares scale_mode {actual['scale_mode']!r}, but a GPTQ run "
            f"consumes exactly a '{WEIGHT_SCALE_GPTQ_SCALE_MODE}' calibration artifact."
        )
    mismatched = [field for field in sorted(actual) if section[field] != actual[field]]
    if mismatched:
        field = mismatched[0]
        raise ValueError(
            f"GPTQ artifact {path} was built against another activation calibration: it records '{field}' as "
            f"{section[field]!r}, but the configured calibration file gives {actual[field]!r} "
            f"({len(mismatched)} mismatched field(s): {mismatched}). Rebuild the payload against this calibration."
        )
    if section["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            f"GPTQ artifact {path} is bound to a calibration collected on checkpoint "
            f"{section['checkpoint_sha256']!r}, but the artifact itself names checkpoint {checkpoint_sha256}; "
            "activation scales are only valid for the checkpoint that produced them."
        )
    recorded_runtime_margin = _validated_finite_number(
        section["runtime_scale_margin"], f"'activation_calibration.runtime_scale_margin' in {path}"
    )
    if recorded_runtime_margin != float(margin):
        raise ValueError(
            f"The calibration file bound to {path} presumes a runtime scale margin of "
            f"{float(recorded_runtime_margin)!r}, but the artifact was built with {float(margin)!r}; its baked-in "
            "headroom would be applied twice or not at all."
        )
    return {**actual, "scale_margin": float(margin)}


def _validate_gptq_qdata(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, Any]:
    """
    Decode and validate the per-FQN packed payloads against the live modules they will be written into.

    Coverage must be exact and canonical, every payload must decode to exactly the number of bytes its declared
    ``[out_features, in_features / 2]`` shape needs, that shape must be the live module's, the declared byte length
    must agree with both, and the recorded content digest must be the digest of the bytes the artifact carries. A
    payload that fails any of these describes a layer other than the one it names, or has been altered since it was
    written -- including a single flipped FP4 nibble, which changes the digest.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"GPTQ artifact {path} must contain a non-empty 'qdata' object")
    noncanonical = sorted(fqn for fqn in raw if not isinstance(fqn, str) or fqn != _canonical_fqn(fqn))
    if noncanonical:
        raise ValueError(
            f"GPTQ artifact {path} names {len(noncanonical)} module(s) in non-canonical form, e.g. "
            f"{noncanonical[:4]}; entries must be exact uncompiled target FQNs."
        )
    if sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required))
        missing = sorted(set(required) - set(raw))
        raise ValueError(
            f"GPTQ artifact {path} must cover exactly the {len(required)} selected NVFP4 W4A4 module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}."
        )

    payloads: Dict[str, bytes] = {}
    shapes: Dict[str, List[int]] = {}
    digests: Dict[str, str] = {}
    for fqn in required:
        described = f"'qdata[{fqn}]' in {path}"
        entry = raw[fqn]
        if not isinstance(entry, dict):
            raise ValueError(f"{described} must be an object, got {type(entry).__name__}")
        _require_exact_keys(entry, GPTQ_QDATA_KEYS, described, "payload")
        if entry["dtype"] != "uint8":
            raise ValueError(
                f"{described} declares dtype {entry['dtype']!r}, but a packed NVFP4 payload is uint8 bytes, two "
                "FP4 codes each."
            )
        shape = entry["shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"{described} must declare 'shape' as [out_features, in_features / 2], got {shape!r}")
        rows = _validated_non_negative_int(shape[0], f"{described} 'shape[0]'")
        packed = _validated_non_negative_int(shape[1], f"{described} 'shape[1]'")
        out_features, in_features = _gptq_live_module_shape(modules[fqn], fqn)
        if (rows, packed) != (out_features, in_features // 2):
            raise ValueError(
                f"{described} declares shape {[rows, packed]}, but the live module needs "
                f"{[out_features, in_features // 2]}; the payload describes a different weight."
            )
        expected_bytes = rows * packed
        declared = _validated_non_negative_int(entry["byte_length"], f"{described} 'byte_length'")
        if declared != expected_bytes:
            raise ValueError(
                f"{described} declares {declared} payload byte(s), but its own shape needs exactly "
                f"{expected_bytes}."
            )
        decoded = _decode_gptq_payload(entry["payload"], expected_bytes, described)
        digest = _validate_checkpoint_sha256(entry["sha256"], f"{described} 'sha256'")
        actual = hashlib.sha256(decoded).hexdigest()
        if actual != digest:
            raise ValueError(
                f"{described} records the payload digest {digest}, but its own bytes hash to {actual}; the packed "
                "FP4 payload was altered after it was selected."
            )
        payloads[fqn] = decoded
        shapes[fqn] = [rows, packed]
        digests[fqn] = digest
    return {"qdata": payloads, "qdata_shapes": shapes, "qdata_digests": digests}


def _decode_gptq_payload(value: Any, expected_bytes: int, described: str) -> bytes:
    """Decode one strict base64 packed payload, refusing anything that is not exactly the declared length."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{described} must carry 'payload' as a non-empty base64 string, got {value!r}")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{described} does not carry a valid base64 payload: {error}") from error
    if len(decoded) != expected_bytes:
        raise ValueError(
            f"{described} decodes to {len(decoded)} payload byte(s), but its declared shape needs exactly "
            f"{expected_bytes}; the payload is truncated or padded."
        )
    return decoded


def _validate_gptq_template_scale(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Validate the recorded ordinary-template scale identities, structurally, before anything is converted.

    The *comparison* against the live template happens later, per FQN, in :func:`_repack_weight_with_gptq`: TorchAO
    has not converted a single module at this point, so there is no scale buffer to compare against yet. What can
    be checked here is that every claim is well formed and covers exactly the selected modules, and that the
    declared scale-buffer byte length is at least the one block count the live module implies -- a buffer smaller
    than that could not describe this weight under any swizzle padding.
    """
    if not isinstance(raw, dict) or sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required)) if isinstance(raw, dict) else []
        missing = sorted(set(required) - set(raw)) if isinstance(raw, dict) else list(required)
        raise ValueError(
            f"GPTQ artifact {path} must give 'template_scale' over exactly the {len(required)} selected module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}."
        )
    validated: Dict[str, Dict[str, Any]] = {}
    for fqn in required:
        described = f"'template_scale[{fqn}]' in {path}"
        entry = raw[fqn]
        if not isinstance(entry, dict):
            raise ValueError(f"{described} must be an object, got {type(entry).__name__}")
        _require_exact_keys(entry, GPTQ_TEMPLATE_SCALE_KEYS, described, "template-scale")
        shape = entry["shape"]
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"{described} must declare the scale buffer's 'shape' as a non-empty list, got {shape!r}")
        sizes = []
        for index, size in enumerate(shape):
            sizes.append(_validated_non_negative_int(size, f"{described} 'shape[{index}]'"))
        if not isinstance(entry["dtype"], str) or not entry["dtype"].strip():
            raise ValueError(f"{described} must declare the scale buffer's 'dtype' as a non-empty string.")
        declared = _validated_non_negative_int(entry["byte_length"], f"{described} 'byte_length'")
        expected = 1
        for size in sizes:
            expected *= size
        if declared != expected:
            raise ValueError(
                f"{described} declares {declared} scale byte(s), but its own shape {sizes} holds {expected}."
            )
        out_features, in_features = _gptq_live_module_shape(modules[fqn], fqn)
        blocks = out_features * (in_features // NVFP4_BLOCK_SIZE)
        if declared < blocks:
            raise ValueError(
                f"{described} declares {declared} scale byte(s), which is fewer than the {blocks} block(s) of the "
                f"live ({out_features}, {in_features}) module; no swizzle padding can make it describe this weight."
            )
        for field in ("sha256", "global_scale_sha256"):
            _validate_checkpoint_sha256(entry[field], f"{described} '{field}'")
        validated[fqn] = {
            "shape": sizes,
            "dtype": str(entry["dtype"]),
            "byte_length": declared,
            "sha256": str(entry["sha256"]),
            "global_scale_sha256": str(entry["global_scale_sha256"]),
        }
    return validated


def _validate_gptq_hessian_section(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Validate the per-module Hessian identities and statistics the artifact carries in place of the matrices.

    The matrices themselves are deliberately absent -- a ``(K, K)`` FP32 matrix per module would dwarf the payload
    and would carry activation statistics into a runtime artifact -- so what is checked is that every claim is well
    formed: a digest, the module's own input width, a positive sampled-row count, a dead-column count that cannot
    exceed the input width, a positive finite damping, and a finite diagonal whose minimum, mean and maximum are
    ordered and at least the damping (every diagonal entry has that damping added to a non-negative value).
    """
    if not isinstance(raw, dict) or sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required)) if isinstance(raw, dict) else []
        missing = sorted(set(required) - set(raw)) if isinstance(raw, dict) else list(required)
        raise ValueError(
            f"GPTQ artifact {path} must give 'hessian' over exactly the {len(required)} selected module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}."
        )
    validated: Dict[str, Dict[str, Any]] = {}
    for fqn in required:
        described = f"'hessian[{fqn}]' in {path}"
        entry = raw[fqn]
        if not isinstance(entry, dict):
            raise ValueError(f"{described} must be an object, got {type(entry).__name__}")
        _require_exact_keys(entry, GPTQ_HESSIAN_KEYS, described, "Hessian")
        _validate_checkpoint_sha256(entry["sha256"], f"{described} 'sha256'")
        _, in_features = _gptq_live_module_shape(modules[fqn], fqn)
        width = _validated_non_negative_int(entry["input_features"], f"{described} 'input_features'")
        if width != in_features:
            raise ValueError(
                f"{described} records {width} input channel(s), but the live module has {in_features}; the Hessian "
                "does not describe this weight."
            )
        sampled = _validated_non_negative_int(entry["sampled_row_count"], f"{described} 'sampled_row_count'")
        if sampled <= 0:
            raise ValueError(f"{described} must record a positive 'sampled_row_count', got {sampled}.")
        dead = _validated_non_negative_int(entry["dead_column_count"], f"{described} 'dead_column_count'")
        if dead > in_features:
            raise ValueError(
                f"{described} records {dead} dead column(s) of {in_features}; a weight cannot have more dead input "
                "columns than input channels."
            )
        damping = _validated_finite_number(entry["damping"], f"{described} 'damping'")
        if damping <= 0.0:
            raise ValueError(f"{described} must record a positive 'damping', got {damping!r}.")
        statistics = {
            field: _validated_finite_number(entry[field], f"{described} '{field}'")
            for field in ("diagonal_min", "diagonal_max", "diagonal_mean")
        }
        if not statistics["diagonal_min"] <= statistics["diagonal_mean"] <= statistics["diagonal_max"]:
            raise ValueError(
                f"{described} records a diagonal whose minimum, mean and maximum are not ordered: "
                f"{statistics}; they do not describe one damped Hessian diagonal."
            )
        if statistics["diagonal_min"] < damping:
            raise ValueError(
                f"{described} records a diagonal minimum {statistics['diagonal_min']!r} below its own damping "
                f"{damping!r}; every diagonal entry has that damping added to a non-negative value."
            )
        validated[fqn] = {
            "sha256": str(entry["sha256"]),
            "input_features": width,
            "sampled_row_count": sampled,
            "dead_column_count": dead,
            "damping": damping,
            **statistics,
        }
    return validated


def _validate_gptq_provenance(provenance: Any, required: Sequence[str], path: Path) -> Dict[str, Any]:
    """
    Validate the provenance of a GPTQ artifact: how it was built, from which samples, and what it selected.

    The construction, its version, the objective and the group reduction are constants of this implementation
    rather than free text, so an artifact that states any other value was produced by a different builder.
    """
    if not isinstance(provenance, dict):
        raise ValueError(f"GPTQ artifact {path} must give 'provenance' as an object, got {type(provenance).__name__}")
    _require_exact_keys(provenance, GPTQ_PROVENANCE_KEYS, f"GPTQ artifact {path}", "provenance")
    for field, expected in (
        ("method", GPTQ_CONSTRUCTION_METHOD),
        ("method_version", GPTQ_CONSTRUCTION_METHOD_VERSION),
        ("objective", GPTQ_OBJECTIVE),
        ("group_reduction", GPTQ_GROUP_REDUCTION),
    ):
        if provenance[field] != expected or isinstance(provenance[field], bool):
            raise ValueError(
                f"GPTQ artifact {path} records '{field}' as {provenance[field]!r}, but this build only consumes a "
                f"payload built with {expected!r}."
            )
    if list(provenance["targets"] or []) != list(QUANTIZATION_TARGET_SUFFIXES):
        raise ValueError(
            f"GPTQ artifact {path} declares targets {provenance['targets']!r}, but this build quantizes exactly "
            f"{list(QUANTIZATION_TARGET_SUFFIXES)}."
        )
    if list(provenance["target_fqns"] or []) != list(required):
        raise ValueError(
            f"GPTQ artifact {path} names {len(provenance['target_fqns'] or [])} target FQN(s) in its provenance, "
            f"but this model selects {len(required)}; the artifact describes a different target set."
        )
    count = _validated_non_negative_int(provenance["target_module_count"], f"'target_module_count' in {path}")
    if count != len(required):
        raise ValueError(
            f"GPTQ artifact {path} records target_module_count {count}, but this model selects {len(required)} "
            "NVFP4 W4A4 module(s)."
        )

    _validate_gptq_sources(provenance["sources"], path)
    _validate_gptq_modules(provenance["modules"], required, path)
    aggregate = provenance["aggregate"]
    if not isinstance(aggregate, dict):
        raise ValueError(f"GPTQ artifact {path} must give 'aggregate' as an object.")
    _require_exact_keys(aggregate, GPTQ_AGGREGATE_KEYS, f"GPTQ artifact {path}", "aggregate")
    for field in (
        "module_count",
        "source_count",
        "block_count",
        "weight_count",
        "qdata_byte_length",
        "dead_column_count",
    ):
        _validated_non_negative_int(aggregate[field], f"'aggregate.{field}' in {path}")
    for field in ("template_mse", "selected_mse", "template_objective", "selected_objective"):
        value = _validated_finite_number(aggregate[field], f"'aggregate.{field}' in {path}")
        if value < 0.0:
            raise ValueError(f"'aggregate.{field}' in {path} is negative ({value!r}); it is a mean of squares.")
    if not isinstance(aggregate["source_labels"], list):
        raise ValueError(f"'aggregate.source_labels' in {path} must be a list.")
    return provenance


def _validate_gptq_sources(sources: Any, path: Path) -> None:
    """Validate the labelled activation sources the Hessians were formed on, key by key and count by count."""
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"GPTQ artifact {path} must record a non-empty 'sources' list; an artifact that names no activation "
            "source cannot say which data selected its payload."
        )
    labels: List[str] = []
    for index, source in enumerate(sources):
        described = f"Entry {index} of 'sources' in {path}"
        if not isinstance(source, dict):
            raise ValueError(f"{described} must be an object, got {type(source).__name__}")
        _require_exact_keys(source, GPTQ_SOURCE_KEYS, described, "source")
        for field in ("label", "name"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"{described} must carry a non-empty '{field}'.")
        _validate_checkpoint_sha256(source["sha256"], f"{described} 'sha256'")
        counts = {
            field: _validated_non_negative_int(source[field], f"{described} '{field}'")
            for field in _HESSIAN_SOURCE_COUNT_FIELDS
        }
        for field in ("size_bytes", "max_rows", "sampled_row_count"):
            if counts[field] <= 0:
                raise ValueError(f"{described} must give a positive '{field}', got {counts[field]}.")
        if counts["sampled_row_count"] > counts["finite_row_count"]:
            raise ValueError(
                f"{described} retained {counts['sampled_row_count']} row(s) out of {counts['finite_row_count']} "
                "finite one(s); a bounded reservoir cannot keep more rows than it saw."
            )
        if not isinstance(source["metadata"], dict):
            raise ValueError(f"{described} must give 'metadata' as an object.")
        labels.append(source["label"])
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"GPTQ artifact {path} names the source label(s) {duplicates} more than once; every group is weighted "
            "equally, so a group named twice would have been counted twice."
        )


def _validate_gptq_modules(modules: Any, required: Sequence[str], path: Path) -> None:
    """Validate the per-module evidence section: exact coverage, consistent counts and finite offline errors."""
    if not isinstance(modules, dict) or sorted(modules) != list(required):
        extra = sorted(set(modules) - set(required)) if isinstance(modules, dict) else []
        missing = sorted(set(required) - set(modules)) if isinstance(modules, dict) else list(required)
        raise ValueError(
            f"GPTQ artifact {path} must give 'provenance.modules' over exactly the {len(required)} selected "
            f"module(s); extra {extra[:4]}, missing {missing[:4]}."
        )
    for fqn in required:
        described = f"'provenance.modules[{fqn}]' in {path}"
        entry = modules[fqn]
        if not isinstance(entry, dict):
            raise ValueError(f"{described} must be an object, got {type(entry).__name__}")
        _require_exact_keys(entry, GPTQ_MODULE_KEYS, described, "module")
        shape = entry["shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"{described} must declare 'shape' as [out_features, in_features], got {shape!r}")
        rows = _validated_non_negative_int(shape[0], f"{described} 'shape[0]'")
        columns = _validated_non_negative_int(shape[1], f"{described} 'shape[1]'")
        weights = _validated_non_negative_int(entry["weight_count"], f"{described} 'weight_count'")
        blocks = _validated_non_negative_int(entry["block_count"], f"{described} 'block_count'")
        payload = _validated_non_negative_int(entry["qdata_byte_length"], f"{described} 'qdata_byte_length'")
        if weights <= 0 or weights != rows * columns:
            raise ValueError(
                f"{described} records {weights} weight(s) for a {[rows, columns]} module, which holds "
                f"{rows * columns}."
            )
        if blocks * NVFP4_BLOCK_SIZE != weights or payload * 2 != weights:
            raise ValueError(
                f"{described} records {blocks} block(s) and {payload} payload byte(s) for {weights} weight(s), but "
                f"a NVFP4 weight has one {NVFP4_BLOCK_SIZE}-weight block per {NVFP4_BLOCK_SIZE} weights and two "
                "FP4 codes per byte."
            )
        for field in ("template_mse", "selected_mse", "template_objective", "selected_objective"):
            value = _validated_finite_number(entry[field], f"{described} '{field}'")
            if value < 0.0:
                raise ValueError(f"{described} '{field}' is negative ({value!r}); it is a mean of squares.")


def _validate_gptq_aggregate(
    provenance: Dict[str, Any],
    decoded: Dict[str, Any],
    hessian: Dict[str, Dict[str, Any]],
    required: Sequence[str],
    path: Path,
) -> None:
    """
    Reject an artifact whose per-module and aggregate claims contradict the payload it actually carries.

    A recomputed ``provenance_sha256`` proves only that nobody edited the section after it was hashed, so every
    claim is recomputed here from the decoded payload lengths and the already-validated sections. The comparison is
    exact rather than approximate: the counts are integers, and the four aggregate errors are recomputed from the
    artifact's own per-module numbers in the same canonical order the builder reduced them in, which python float
    arithmetic and a JSON round trip both reproduce exactly.
    """
    modules = provenance["modules"]
    aggregate = provenance["aggregate"]
    labels = [source["label"] for source in provenance["sources"]]
    for fqn in required:
        declared = int(modules[fqn]["qdata_byte_length"])
        carried = len(decoded["qdata"][fqn])
        if declared != carried:
            raise ValueError(
                f"'provenance.modules[{fqn}].qdata_byte_length' in {path} records {declared}, but its own payload "
                f"holds {carried} byte(s); the summary does not describe the payload this artifact carries."
            )
    for field, expected in (
        ("module_count", len(required)),
        ("source_count", len(provenance["sources"])),
        ("block_count", sum(int(modules[fqn]["block_count"]) for fqn in required)),
        ("weight_count", sum(int(modules[fqn]["weight_count"]) for fqn in required)),
        ("qdata_byte_length", sum(len(decoded["qdata"][fqn]) for fqn in required)),
        ("dead_column_count", sum(int(hessian[fqn]["dead_column_count"]) for fqn in required)),
    ):
        if int(aggregate[field]) != int(expected):
            raise ValueError(
                f"'aggregate.{field}' in {path} records {aggregate[field]!r}, but the artifact's own sections give "
                f"{expected}; the aggregate does not summarize this artifact."
            )
    if list(aggregate["source_labels"]) != labels:
        raise ValueError(
            f"'aggregate.source_labels' in {path} records {aggregate['source_labels']!r}, but its 'sources' are "
            f"labelled {labels}; the aggregate does not summarize this artifact."
        )
    for field in ("template_mse", "selected_mse", "template_objective", "selected_objective"):
        expected_value = nvfp4_gptq_weighted_objective(modules, required, field)
        if float(aggregate[field]) != float(expected_value):
            raise ValueError(
                f"'aggregate.{field}' in {path} records {aggregate[field]!r}, but its per-module values reduce to "
                f"{expected_value!r}; the aggregate does not summarize this artifact."
            )


def _validate_gptq_weight_digests(
    raw: Any, required: Sequence[str], modules: Dict[str, torch.nn.Module], path: Path
) -> Dict[str, str]:
    """
    Validate the recorded weight digests and verify them against the live, still-unconverted weights.

    This is the binding between the offline payload selection and the model in memory, and it runs before any
    TorchAO conversion has mutated a weight: a checkpoint whose weights differ from the ones the payload was
    selected on is a different model, and its stored FP4 codes would replace the wrong values.
    """
    if not isinstance(raw, dict) or sorted(raw) != list(required):
        extra = sorted(set(raw) - set(required)) if isinstance(raw, dict) else []
        missing = sorted(set(required) - set(raw)) if isinstance(raw, dict) else list(required)
        raise ValueError(
            f"GPTQ artifact {path} must give 'weight_sha256' over exactly the {len(required)} selected module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}."
        )
    digests: Dict[str, str] = {}
    mismatched: List[str] = []
    for fqn in required:
        expected = _validate_checkpoint_sha256(raw[fqn], f"'weight_sha256[{fqn}]' in {path}")
        weight = getattr(modules[fqn], "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                f"Module '{fqn}' exposes no weight tensor, so the GPTQ artifact {path} cannot be bound to this "
                "model."
            )
        if nvfp4_weight_digest(weight) != expected:
            mismatched.append(fqn)
        digests[fqn] = expected
    if mismatched:
        raise ValueError(
            f"GPTQ artifact {path} was built on different weights: {len(mismatched)} module(s) do not match their "
            f"recorded digest, e.g. {mismatched[:4]}. Rebuild the payload on this checkpoint."
        )
    return digests


def _verify_gptq_section_digest(payload: Dict[str, Any], digest_key: str, section_key: str, path: Path) -> str:
    """Verify one recorded GPTQ section digest against the section the artifact actually carries."""
    expected = _validate_checkpoint_sha256(payload[digest_key], f"'{digest_key}' in {path}")
    try:
        actual = nvfp4_section_digest(payload[section_key])
    except ValueError as error:
        raise ValueError(
            f"The '{section_key}' section of GPTQ artifact {path} cannot be canonicalized for its "
            f"'{digest_key}': {error}"
        ) from error
    if actual != expected:
        raise ValueError(
            f"GPTQ artifact {path} records {digest_key} {expected}, but its '{section_key}' section hashes to "
            f"{actual}; the artifact was modified after it was built."
        )
    return expected


def _gptq_live_module_shape(module: torch.nn.Module, fqn: str) -> Tuple[int, int]:
    """Return one live target's ``(out_features, in_features)``, refusing a shape no NVFP4 payload can describe."""
    out_features = int(getattr(module, "out_features", 0))
    in_features = int(getattr(module, "in_features", 0))
    if in_features % NVFP4_BLOCK_SIZE != 0 or in_features <= 0 or out_features <= 0:
        raise ValueError(
            f"Module '{fqn}' has shape ({out_features}, {in_features}), which no NVFP4 payload of "
            f"{NVFP4_BLOCK_SIZE}-weight blocks can describe."
        )
    return out_features, in_features


def _validated_ratio_histogram(value: Any, description: str) -> List[int]:
    """Validate a per-ratio block-count histogram as exactly one non-negative integer per clip ratio."""
    if not isinstance(value, list) or len(value) != WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT:
        raise ValueError(
            f"{description} must be a list of exactly {WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT} non-negative integers, "
            f"one per clip ratio, got {value!r}"
        )
    return [_validated_non_negative_int(count, f"{description}[{index}]") for index, count in enumerate(value)]


def _validated_finite_number(value: Any, description: str) -> float:
    """Validate a reported number as finite, rejecting bools and every non-number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{description} must be a finite number, got {value!r}")
    return float(value)


def _resolve_calibration_checkpoint(payload: Dict[str, Any], metadata: Dict[str, Any], path: Path) -> str:
    """
    Resolve the one checkpoint digest a calibration file claims, from wherever that file records it.

    :func:`merge_calibrations` records it inside ``metadata``, which is where every production calibration carries
    it; a top-level :data:`CALIBRATION_CHECKPOINT_FIELD` is the older single-collector spelling and is accepted as
    well. Both are validated whenever they are present, two present claims must be identical, and a file that
    claims neither is refused: ratio codes that cannot be bound to a checkpoint could be applied to any weights at
    all, which is exactly what the binding exists to prevent.
    """
    top_level = payload.get(CALIBRATION_CHECKPOINT_FIELD)
    if top_level is not None:
        top_level = _validate_checkpoint_sha256(top_level, f"'{CALIBRATION_CHECKPOINT_FIELD}' in {path}")
    nested = metadata.get(CALIBRATION_CHECKPOINT_FIELD)
    if nested is not None:
        nested = _validate_checkpoint_sha256(nested, f"'metadata.{CALIBRATION_CHECKPOINT_FIELD}' in {path}")
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError(
            f"Calibration file {path} claims checkpoint {top_level} at the top level but {nested} in its metadata; "
            "two conflicting claims cannot both describe the checkpoint these activation maxima were collected on."
        )
    resolved = nested if nested is not None else top_level
    if resolved is None:
        raise ValueError(
            f"Calibration file {path} declares no checkpoint digest, in neither "
            f"'{CALIBRATION_CHECKPOINT_FIELD}' nor 'metadata.{CALIBRATION_CHECKPOINT_FIELD}'; AWQ-clip ratio codes "
            "cannot be bound to a checkpoint through it."
        )
    return resolved


def _is_readable_file(path: Any) -> bool:
    """Whether a path names an existing readable file; used by option validation before anything is read."""
    resolved = Path(str(path)).expanduser()
    return resolved.is_file() and os.access(resolved, os.R_OK)


def _validated_non_negative_int(value: Any, description: str) -> int:
    """Validate a count as a non-negative integer, rejecting bools and every non-integer number."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer, got {value!r}")
    return int(value)


_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_checkpoint_sha256(value: Any, description: str) -> str:
    """Validate a checkpoint identity as 64 lowercase hexadecimal characters, returning the normalized digest."""
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a 64-character hexadecimal SHA-256 string, got {value!r}")
    digest = value.strip().lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValueError(f"{description} must be a 64-character hexadecimal SHA-256 string, got {value!r}")
    return digest


def _group_statistics(group_samples: Dict[str, List[float]], fqns: Sequence[str], input_count: int) -> Dict[str, int]:
    """Deterministic observation counts of one group, reported in the merged artifact's provenance."""
    counts = [len(group_samples[fqn]) for fqn in fqns]
    return {
        "input_count": input_count,
        "observation_count": sum(counts),
        "min_observations_per_module": min(counts),
        "max_observations_per_module": max(counts),
    }


def _input_provenance(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Identity of one merged input: a stable basename plus digest and size, never an absolute path."""
    return {
        "group": entry["group"],
        "name": entry["name"],
        "sha256": entry["sha256"],
        "size_bytes": entry["size_bytes"],
        "observation_count": sum(len(values) for values in entry["samples"].values()),
        "legacy_max_only_fallback": entry["legacy_fallback"],
        "nonfinite_status": entry["nonfinite_status"],
        "checkpoint_sha256_declared": entry["checkpoint_sha256"] is not None,
    }


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Conservative nearest-rank percentile ``sorted_values[ceil(percentile / 100 * n) - 1]``."""
    ordered = sorted(values)
    index = math.ceil(percentile / 100.0 * len(ordered)) - 1
    return float(ordered[min(max(index, 0), len(ordered) - 1)])


def _validate_percentile(value: Any) -> float:
    """Coerce a percentile to ``float``, rejecting non-numbers and anything outside ``(0, 100]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"percentile must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number > 100:
        raise ValueError(f"percentile must satisfy 0 < percentile <= 100, got {value!r}")
    return number


def _resolve_api(api: str):
    """Import and return a ``module:attribute`` entry point, or ``None`` when it is unavailable."""
    module_name, attribute = api.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attribute, None)


def _require_api(api: str):
    """Resolve a ``module:attribute`` entry point, raising an actionable error when it is missing."""
    resolved = _resolve_api(api)
    if resolved is None:
        raise RuntimeError(
            f"Required quantization entry point '{api}' is unavailable. Install a TorchAO build that exports it."
        )
    return resolved


def _module_version(module_name: str) -> Optional[str]:
    """Version string of an optional dependency, or ``None`` when it is not installed."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    return None if version is None else str(version)


def _parse_version(version: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse the leading numeric components of a version string."""
    if not version:
        return None
    parts = re.findall(r"\d+", str(version))
    if not parts:
        return None
    return tuple(int(part) for part in parts)


def _positive_finite(value: Any, description: str) -> float:
    """
    Coerce a scale value to ``float``, rejecting non-numbers, NaN, infinities, and non-positive values.

    Args:
        value (Any): Candidate scale margin or activation amax.
        description (str): Prefix used in the error message.

    Returns:
        value (float): The validated value.

    Raises:
        ValueError: If the value is not a finite positive number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{description} must be finite and positive, got {value!r}")
    return number


def _reject_duplicate_keys(pairs):
    """JSON object hook that rejects duplicate keys instead of silently keeping the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Calibration file contains duplicate key '{key}'")
        result[key] = value
    return result


def _reject_duplicate_bf16_override_keys(pairs):
    """JSON object hook for the BF16 override file that rejects duplicate keys instead of keeping the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"BF16 override file contains duplicate key '{key}'")
        result[key] = value
    return result


def _reject_duplicate_hessian_keys(pairs):
    """JSON object hook for the diagonal-Hessian artifact that rejects duplicate keys at every nesting level."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Diagonal-Hessian artifact contains duplicate key '{key}'")
        result[key] = value
    return result


def _reject_duplicate_awq_clip_keys(pairs):
    """JSON object hook for the AWQ-clip artifact that rejects duplicate keys at every nesting level."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"AWQ-clip artifact contains duplicate key '{key}'")
        result[key] = value
    return result


def _reject_duplicate_gptq_keys(pairs):
    """JSON object hook for the GPTQ artifact that rejects duplicate keys at every nesting level."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"GPTQ artifact contains duplicate key '{key}'")
        result[key] = value
    return result


def _normalize_json(value: Any) -> Any:
    """Convert a payload into deterministic, JSON/OmegaConf-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _normalize_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _normalize_json(value.detach().cpu().tolist())
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return float(value)
    return str(value)


_ALL_TORCHAO_APIS: Tuple[str, ...] = (
    TORCHAO_QUANTIZE_API,
    TORCHAO_NVFP4_DYNAMIC_CONFIG_API,
    TORCHAO_NVFP4_WEIGHT_ONLY_CONFIG_API,
    TORCHAO_NVFP4_OBSERVED_LINEAR_API,
    TORCHAO_FP8_CONFIG_API,
)

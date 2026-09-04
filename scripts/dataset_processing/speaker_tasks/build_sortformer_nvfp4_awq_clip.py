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
Build the AWQ-clip ratio codes that select NVFP4 weight block scales for the Sortformer transformer linears.

The NVFP4 block scale TorchAO picks is the block's own absolute maximum, which is the best choice for the *weight*
and not for the *layer output*. ``quantization_weight_scale_method='awq_clip'`` instead picks, per output row ``r``
and contiguous 16-weight input block ``b``, one of the eleven fixed clipping ratios NVIDIA ModelOpt 0.46.0
evaluates, keeping the ratio that minimizes that block's contribution to the layer's output error

    mean_g(mean_rows((sum_j Xq_g[row, j] * (W[r, j] - Q_ratio(W[r, j]))) ** 2))

where ``Xq_g`` are a labelled source group's activation rows *after the exact NVFP4 quantize/dequantize the
quantized runtime will perform on them*, using the frozen static activation calibration this artifact is bound to.
Candidates are compared in the reference's fixed ratio order and one only replaces the incumbent on a strictly
smaller loss, so an exact tie -- an all-zero block, whose eleven candidates reconstruct it identically -- keeps the
earliest ratio 0.5. The objective keeps the block-local *full covariance* of the activations, which is exactly the
within-block cross terms the diagonal ``local_hessian`` objective drops; it scores no interaction between blocks.

``Q_ratio`` is what the *runtime* deploys for that code, never an approximation of it: the ten clipping ratios
``0.50..0.95`` are the adapted TorchAO reconstruction, and the eleventh -- the unclipped ratio 1.00 -- is the FP32
decode of the ordinary NVFP4 template's own stored bytes, because the runtime repack leaves such a block's bytes
exactly as the ordinary conversion wrote them. Every one of the eleven deltas is therefore FP32. That conversion is
not unique, so ``--template-arithmetic`` says which one the deployment uses, the codes are scored against that one,
and the runtime refuses the other backend.

This script writes those codes as the strict runtime artifact a quantized run binds to. It is one-shot PTQ code
selection: no labels, no gradients, no optimizer, no training, and the model is only ever read. ModelOpt is not
imported and is not a dependency; only the identity of the wheel whose semantics were adapted is recorded.

Every ``--input`` file is one labelled source group -- a stable domain/microphone/geometry stratum -- of bounded
activation samples, read with ``torch.load(..., weights_only=True)`` so that inspecting one can never execute code.
Groups are weighted **equally**, so a high-volume corpus cannot outvote a small stratum however many rows each of
them retained. Every sample file must declare ``--checkpoint-sha256``, which is also verified against the actual
bytes of ``--model-path`` before the checkpoint is restored, and ``--activation-calibration-path`` must name the
very calibration file the quantized run will consume, because the codes are only valid for activations quantized
with its values.

The written JSON carries no activation row, no weight, no quantized payload, no label, no RTTM and no task metric:
one compact base64 uint8 code matrix per module, the exact arithmetic identity of the selection, the identity of the
calibration file, one canonical digest per original weight, one digest each over the codes and over the provenance,
the checkpoint identity, and the provenance of the construction. The runtime recomputes every one of those digests,
and every per-module and aggregate claim, before it converts anything.

Example (the digests are pasted in explicitly so that the values the codes bind to are visible in the command that
produced them):
    python scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_awq_clip.py \
        --model-path diar_sortformer_4spk-v2.nemo \
        --checkpoint-sha256 6f4b2c0d1e8a7395fd0c1b2a3e4d5f60718293a4b5c6d7e8f9012a3b4c5d6e7f \
        --activation-calibration-path p100_h1375.json \
        --template-arithmetic mslk_triton \
        --device cuda \
        --input near_field=samples_ami.pt \
        --input far_field=samples_notsofar.pt \
        --output awq_clip_codes.json
"""

import argparse
import base64
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.sortformer_nvfp4_weight_mse import (
    DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE,
    DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE,
    FP8_E4M3_MAX,
    NVFP4_MAX,
    nvfp4_awq_clip_activation_qdq,
    nvfp4_awq_clip_template_reconstruction,
    nvfp4_weight_global_scale,
    select_nvfp4_ratio_codes_awq_clip,
)
from nemo.collections.asr.parts.utils.sortformer_quantization import (
    AWQ_CLIP_ACTIVATION_QDQ,
    AWQ_CLIP_CANDIDATE_SCALE_RULE,
    AWQ_CLIP_CODE_ENCODING,
    AWQ_CLIP_CONSTRUCTION_METHOD,
    AWQ_CLIP_CONSTRUCTION_METHOD_VERSION,
    AWQ_CLIP_GROUP_REDUCTION,
    AWQ_CLIP_OBJECTIVE,
    AWQ_CLIP_SCHEMA,
    AWQ_CLIP_SCHEMA_VERSION,
    AWQ_CLIP_TIE_RULE,
    MODELOPT_REFERENCE_VERSION,
    MODELOPT_REFERENCE_WHEEL_SHA256,
    PRECISION_NVFP4_W4A4,
    QUANTIZATION_TARGET_SUFFIXES,
    WEIGHT_DIGEST_METHOD,
    WEIGHT_SCALE_AWQ_CLIP_ALGORITHM,
    WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION,
    WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE,
    WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT,
    WEIGHT_SCALE_AWQ_CLIP_RATIOS,
    WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN,
    WEIGHT_SCALE_AWQ_CLIP_SCALE_MIN,
    WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE,
    WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED,
    WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS,
    nvfp4_awq_clip_calibration_identity,
    nvfp4_awq_clip_weighted_objective,
    nvfp4_section_digest,
    nvfp4_weight_digest,
    select_quantization_targets,
    validate_sha256_digest,
)

# The codes describe every 16-weight block of exactly the W4A4 target set of this recipe, which is the only recipe
# ``awq_clip`` accepts.
AWQ_CLIP_RECIPE = "nvfp4_all"

# Schema of the bounded activation-sample artifacts this builder consumes. They are internal calibration evidence
# -- they carry real activation rows -- and are never read by a quantized run; the runtime only reads the JSON
# written here.
ACTIVATION_SAMPLE_SCHEMA = "sortformer_nvfp4_activation_samples"
ACTIVATION_SAMPLE_VERSION = 1
ACTIVATION_SAMPLE_KEYS: frozenset = frozenset(
    {
        "schema",
        "version",
        "checkpoint_sha256",
        "targets",
        "metadata",
        "seed",
        "max_rows",
        "total_finite_rows",
        "nonfinite_rows",
        "samples",
    }
)

# The collector retains BF16 rows on the host; anything else was not written by it.
ACTIVATION_SAMPLE_DTYPE = torch.bfloat16

# The selection is eleven candidate reconstructions per block scored against bounded activation rows, so it runs on
# the GPU by default. ``cuda`` without an index means the first device this process can see, which inside a
# GPU-scoped container is the device the operator meant.
DEFAULT_DEVICE = "cuda"
SUPPORTED_DEVICE_TYPES: Tuple[str, ...] = ("cpu", "cuda")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse arguments, select the AWQ-clip ratio codes, and write the runtime artifact.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        exit_code (int): ``0`` on success. Invalid inputs raise :class:`SystemExit` with a nonzero code.
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = build_awq_clip_codes(
            model_path=args.model_path,
            checkpoint_sha256=args.checkpoint_sha256,
            activation_calibration_path=args.activation_calibration_path,
            inputs=_parse_inputs(args.input),
            output_path=args.output,
            template_arithmetic=args.template_arithmetic,
            overwrite=args.overwrite,
            device=args.device,
            row_chunk_size=args.row_chunk_size,
            block_chunk_size=args.block_chunk_size,
        )
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        # FileExistsError is an OSError, so refusing to overwrite also exits nonzero with an actionable message.
        raise SystemExit(f"error: {error}") from error

    aggregate = payload["provenance"]["aggregate"]
    print(
        f"Wrote {args.output} for checkpoint {payload['checkpoint_sha256']}: "
        f"{aggregate['module_count']} module(s) from {aggregate['source_count']} equally weighted source group(s) "
        f"{aggregate['source_labels']}, {aggregate['block_count']} block(s) with clip-ratio histogram "
        f"{aggregate['ratio_histogram']} over ratios {list(WEIGHT_SCALE_AWQ_CLIP_RATIOS)}."
    )
    print(
        f"Codes only: {payload['algorithm']} v{payload['algorithm_version']}, unclipped blocks kept from the "
        f"'{payload['arithmetic']['template_arithmetic']}' ordinary template, bound to activation calibration "
        f"{payload['activation_calibration']['name']} ({payload['activation_calibration']['sha256']}) at margin "
        f"{payload['activation_calibration']['scale_margin']}. Offline objective "
        f"{aggregate['selected_objective']} against {aggregate['unclipped_objective']} unclipped. The artifact "
        "carries no activation rows, weights or metrics, and says nothing about DER."
    )
    return 0


def build_awq_clip_codes(
    model_path: str,
    checkpoint_sha256: str,
    activation_calibration_path: str,
    inputs: Sequence[Tuple[str, str]],
    output_path: str,
    template_arithmetic: str,
    overwrite: bool = False,
    device: str = DEFAULT_DEVICE,
    row_chunk_size: int = DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE,
) -> Dict[str, Any]:
    """
    Restore the checkpoint, select one clipping-ratio code per block from the labelled samples, and write it out.

    Args:
        model_path (str): Path to the ``.nemo`` checkpoint the samples were collected on.
        checkpoint_sha256 (str): Expected SHA-256 of that file; verified against its actual bytes.
        activation_calibration_path (str): Path of the static activation-calibration JSON the quantized run will
            consume; the codes are selected against activations quantized with exactly its values.
        inputs (Sequence[Tuple[str, str]]): ``(label, path)`` pairs, one per balanced source group.
        output_path (str): Destination of the artifact JSON.
        template_arithmetic (str): The ordinary-template construction the quantized run will convert with, one of
            :data:`WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS`.
        overwrite (bool): Replace an existing destination. Defaults to ``False``.
        device (str): PyTorch device the selection runs on, e.g. ``'cuda'``, ``'cuda:1'`` or ``'cpu'``. Resolved
            and checked before anything is read; a CUDA device is never downgraded to the CPU.
        row_chunk_size (int): Output rows scored at once; bounds peak memory only.
        block_chunk_size (int): Input blocks scored at once; bounds peak memory only.

    Returns:
        payload (Dict[str, Any]): The artifact that was written.

    Raises:
        ValueError: If a label or path is duplicated, the device is unusable, the digest does not match the
            checkpoint, or an input or the calibration file is invalid.
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
    """
    digest = validate_sha256_digest(checkpoint_sha256, "--checkpoint-sha256")
    _require_unique_inputs(inputs)
    # Resolved before the checkpoint is restored or a sample file is read, so an unusable device fails in a second
    # instead of after a multi-gigabyte restore.
    resolved = resolve_device(device)
    entries = [load_activation_sample_file(label, path) for label, path in inputs]
    weights = restore_target_weights(model_path, digest, resolved)
    calibration = load_activation_calibration(activation_calibration_path, digest, sorted(weights))
    payload = build_awq_clip_artifact(
        weights,
        entries,
        calibration,
        checkpoint_sha256=digest,
        template_arithmetic=template_arithmetic,
        row_chunk_size=row_chunk_size,
        block_chunk_size=block_chunk_size,
    )
    write_awq_clip_artifact(payload, output_path, overwrite=overwrite)
    return payload


def build_awq_clip_artifact(
    weights: Dict[str, torch.Tensor],
    entries: Sequence[Dict[str, Any]],
    calibration: Dict[str, Any],
    checkpoint_sha256: str,
    template_arithmetic: str,
    row_chunk_size: int = DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE,
) -> Dict[str, Any]:
    """
    Select the clipping-ratio codes module by module and assemble the deterministic runtime artifact payload.

    Every sample file must declare the same checkpoint as the one the codes are finalized against and must cover
    exactly the selected modules with exactly their input widths: rows collected from another checkpoint, or for
    another module set, describe a different model. Modules are visited in sorted FQN order and groups in sorted
    label order, and only one module's quantized group rows are alive at a time, so the peak memory of the whole
    pass is one module's bounded rows plus the search's own chunked working set. No activation tensor, no weight
    and no quantized payload is copied into the artifact.

    Args:
        weights (Dict[str, torch.Tensor]): Unconverted weights of exactly the selected FQNs, on the search device.
        entries (Sequence[Dict[str, Any]]): Loaded sample entries, one per source group.
        calibration (Dict[str, Any]): Validated activation calibration from :func:`load_activation_calibration`.
        checkpoint_sha256 (str): SHA-256 of the checkpoint the weights and the samples come from.
        template_arithmetic (str): The ordinary-template construction the quantized run will convert with, one of
            :data:`WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS`. The unclipped code keeps that construction's own
            bytes, so it is scored against it and recorded in the artifact; the runtime refuses the other backend.
        row_chunk_size (int): Output rows scored at once; bounds peak memory only.
        block_chunk_size (int): Input blocks scored at once; bounds peak memory only.

    Returns:
        payload (Dict[str, Any]): The artifact, ready for :func:`write_awq_clip_artifact`.

    Raises:
        ValueError: If the digest is malformed, the construction is not one this build reproduces, an entry declares
            another checkpoint, target list or module set, a sample width does not match its weight, or a selection
            is unusable.
    """
    digest = validate_sha256_digest(checkpoint_sha256, "checkpoint_sha256")
    if template_arithmetic not in WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS:
        raise ValueError(
            f"template_arithmetic must be one of {list(WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS)}, got "
            f"{template_arithmetic!r}; the unclipped code keeps the bytes that construction writes, so there is no "
            "default and no fallback."
        )
    if not entries:
        raise ValueError("At least one labelled activation-sample file is required, but none were given.")
    fqns = sorted(weights)
    if not fqns:
        raise ValueError("No NVFP4 W4A4 target module was selected, so there are no ratio codes to select.")
    if sorted(calibration["activation_amax"]) != fqns:
        raise ValueError(
            f"Activation calibration {calibration['path']} covers "
            f"{len(calibration['activation_amax'])} module(s), but the codes are built for {len(fqns)}; the "
            "artifact describes every NVFP4 W4A4 target and cannot be built from a partial calibration."
        )
    _validate_sample_entries(entries, weights, fqns, digest)

    ordered = sorted(entries, key=lambda entry: entry["label"])
    ratio_codes: Dict[str, Dict[str, Any]] = {}
    modules: Dict[str, Dict[str, Any]] = {}
    for fqn in fqns:
        selection = select_module_ratio_codes(
            weights[fqn],
            [entry["samples"][fqn] for entry in ordered],
            calibration["activation_amax"][fqn],
            template_arithmetic,
            row_chunk_size=row_chunk_size,
            block_chunk_size=block_chunk_size,
        )
        codes = selection.ratio_codes
        ratio_codes[fqn] = {
            "shape": [int(codes.shape[0]), int(codes.shape[1])],
            "codes": encode_ratio_codes(codes),
        }
        modules[fqn] = {
            "block_count": int(selection.block_count),
            "ratio_histogram": [int(count) for count in selection.code_counts],
            "selected_objective": float(selection.selected_objective),
            "unclipped_objective": float(selection.unclipped_objective),
        }
        # Released before the next module's rows are quantized, which is what bounds the peak memory of the pass.
        del selection, codes

    provenance = {
        "method": AWQ_CLIP_CONSTRUCTION_METHOD,
        "method_version": AWQ_CLIP_CONSTRUCTION_METHOD_VERSION,
        "objective": AWQ_CLIP_OBJECTIVE,
        "group_reduction": AWQ_CLIP_GROUP_REDUCTION,
        "targets": list(QUANTIZATION_TARGET_SUFFIXES),
        "target_module_count": len(fqns),
        "target_fqns": list(fqns),
        "sources": [
            {
                "label": entry["label"],
                "name": entry["name"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
                "seed": entry["seed"],
                "max_rows": entry["max_rows"],
                "sampled_row_count": sum(int(tensor.shape[0]) for tensor in entry["samples"].values()),
                "finite_row_count": sum(entry["total_finite_rows"].values()),
                "nonfinite_row_count": sum(entry["nonfinite_rows"].values()),
                "metadata": entry["metadata"],
            }
            for entry in ordered
        ],
        "modules": modules,
        "aggregate": {
            "module_count": len(fqns),
            "source_count": len(ordered),
            "source_labels": [entry["label"] for entry in ordered],
            "block_count": sum(modules[fqn]["block_count"] for fqn in fqns),
            "ratio_histogram": [
                sum(modules[fqn]["ratio_histogram"][index] for fqn in fqns)
                for index in range(WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
            ],
            # Reduced with the very helper the runtime loader recomputes them with, in the same canonical order,
            # so the comparison there is exact rather than approximate.
            "selected_objective": nvfp4_awq_clip_weighted_objective(modules, fqns, "selected_objective"),
            "unclipped_objective": nvfp4_awq_clip_weighted_objective(modules, fqns, "unclipped_objective"),
        },
    }
    return {
        "schema": AWQ_CLIP_SCHEMA,
        "version": AWQ_CLIP_SCHEMA_VERSION,
        "checkpoint_sha256": digest,
        "algorithm": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION,
        "arithmetic": {
            "block_size": int(WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE),
            "clip_ratios": [float(ratio) for ratio in WEIGHT_SCALE_AWQ_CLIP_RATIOS],
            "tie_rule": AWQ_CLIP_TIE_RULE,
            "objective": AWQ_CLIP_OBJECTIVE,
            "group_reduction": AWQ_CLIP_GROUP_REDUCTION,
            "candidate_scale_rule": AWQ_CLIP_CANDIDATE_SCALE_RULE,
            # Which ordinary conversion the unclipped code's bytes are; the runtime refuses the other backend.
            "template_arithmetic": template_arithmetic,
            "activation_qdq": AWQ_CLIP_ACTIVATION_QDQ,
            "fp4_max": float(NVFP4_MAX),
            "fp8_e4m3_max": float(FP8_E4M3_MAX),
            "scale_min": float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MIN),
            "modelopt_reference_version": MODELOPT_REFERENCE_VERSION,
            "modelopt_reference_wheel_sha256": MODELOPT_REFERENCE_WHEEL_SHA256,
        },
        "activation_calibration": {
            **calibration["identity"],
            "scale_margin": float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN),
        },
        "weight_digest_method": WEIGHT_DIGEST_METHOD,
        "weight_sha256": {fqn: nvfp4_weight_digest(weights[fqn]) for fqn in fqns},
        "code_encoding": AWQ_CLIP_CODE_ENCODING,
        "ratio_codes": ratio_codes,
        # One digest per section, canonicalized exactly as the runtime loader recomputes them, so the codes and the
        # provenance are each verifiable on their own rather than only through the whole file's bytes.
        "ratio_code_sha256": nvfp4_section_digest(ratio_codes),
        "provenance_sha256": nvfp4_section_digest(provenance),
        "provenance": provenance,
    }


def select_module_ratio_codes(
    weight: torch.Tensor,
    group_rows: Sequence[torch.Tensor],
    activation_amax: float,
    template_arithmetic: str,
    row_chunk_size: int = DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE,
    block_chunk_size: int = DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE,
):
    """
    Select one module's ratio codes from its bounded per-group sample rows.

    The rows of each group are moved to the weight's device and passed through the *runtime-matched* NVFP4
    activation quantize/dequantize with this module's calibrated amax before anything is scored, so the search
    measures the error the quantized layer will actually make and never the error against raw BF16 rows. The
    weight's global per-tensor scale is the ordinary TorchAO one, ``global_amax / (448 * 6)``, which is the scale
    the runtime template will carry unchanged.

    The unclipped candidate is likewise the runtime's own: this converts the weight with the ordinary TorchAO
    conversion of the named construction and scores the FP32 decode of that template's stored bytes, because those
    are exactly the bytes the repack leaves in place for a block whose code is the unclipped one.

    Args:
        weight (torch.Tensor): The module's unconverted ``(M, K)`` weight, on the search device.
        group_rows (Sequence[torch.Tensor]): One bounded ``(N_g, K)`` sample tensor per equally weighted group, in
            canonical group order.
        activation_amax (float): The module's calibrated activation maximum, after the run's scale margin.
        template_arithmetic (str): The ordinary-template construction the deployment will convert with, one of
            :data:`WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS`.
        row_chunk_size (int): Output rows scored at once; bounds peak memory only.
        block_chunk_size (int): Input blocks scored at once; bounds peak memory only.

    Returns:
        selection (AWQClipSelection): The uint8 codes, the two offline objectives and the per-ratio block counts.
    """
    quantized: List[torch.Tensor] = []
    global_scale = nvfp4_weight_global_scale(weight)
    template = nvfp4_awq_clip_template_reconstruction(weight, global_scale, template_arithmetic)
    try:
        for rows in group_rows:
            quantized.append(nvfp4_awq_clip_activation_qdq(rows.to(device=weight.device), activation_amax))
        return select_nvfp4_ratio_codes_awq_clip(
            weight,
            global_scale,
            quantized,
            template,
            row_chunk_size=row_chunk_size,
            block_chunk_size=block_chunk_size,
        )
    finally:
        # Released before the next module's rows are quantized, whatever happened above.
        quantized.clear()
        del template


def encode_ratio_codes(codes: torch.Tensor) -> str:
    """Encode a uint8 code matrix as base64 over its contiguous row-major host bytes, as the schema declares."""
    if codes.dtype != torch.uint8:
        raise ValueError(f"Ratio codes must be {torch.uint8} to be encoded, got {codes.dtype}.")
    raw = codes.detach().to("cpu").contiguous().reshape(-1).numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def load_activation_calibration(calibration_path: str, checkpoint_sha256: str, fqns: Sequence[str]) -> Dict[str, Any]:
    """
    Load and strictly validate the static activation calibration the codes will be selected against.

    Beyond the schema itself this checks everything that decides the activations the search quantizes: the static
    scale mode, the checkpoint the maxima were collected on, the exact target families, the exact selected module
    set -- an AWQ-clip artifact describes every W4A4 target, so a partial or over-broad calibration is refused --
    every value being finite and positive, and the baked-headroom contract that makes the runtime margin exactly
    :data:`WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN`.

    Args:
        calibration_path (str): Path to the schema-v1 calibration JSON.
        checkpoint_sha256 (str): The checkpoint the codes are finalized against.
        fqns (Sequence[str]): The selected NVFP4 W4A4 FQNs the calibration must cover exactly.

    Returns:
        calibration (Dict[str, Any]): ``path``, the canonical ``identity`` the artifact records, and the per-FQN
            ``activation_amax`` after the run's margin.

    Raises:
        ValueError: If the file is not a valid, complete, checkpoint-matching static calibration of exactly these
            modules.
        OSError: If the file cannot be read.
    """
    path = Path(calibration_path).expanduser()
    identity = nvfp4_awq_clip_calibration_identity(str(path))
    if identity["scale_mode"] != WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE:
        raise ValueError(
            f"Calibration file {path} declares scale_mode {identity['scale_mode']!r}, but AWQ-clip codes are "
            f"selected against a '{WEIGHT_SCALE_AWQ_CLIP_SCALE_MODE}' calibration artifact."
        )
    if identity["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            f"Calibration file {path} was collected on checkpoint {identity['checkpoint_sha256']!r}, but the codes "
            f"are finalized against {checkpoint_sha256}; activation scales are only valid for the checkpoint that "
            "produced them, and an AWQ-clip artifact must name the one it is bound to."
        )
    # ``nvfp4_awq_clip_calibration_identity`` has already refused a file that declares no checkpoint, states no
    # finite positive headroom, does not state ``headroom_baked_in`` as boolean true, or presumes another runtime
    # margin. This repeats the margin comparison against the value the codes are actually selected at, so the two
    # cannot drift apart if the shared identity ever gains a second accepted margin.
    margin = identity["runtime_scale_margin"]
    if float(margin) != float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN):
        raise ValueError(
            f"Calibration file {path} presumes a runtime scale margin of {float(margin)!r}, but AWQ-clip codes are "
            f"selected at exactly {float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN)!r}; its baked-in headroom would be "
            "applied twice or not at all."
        )

    payload = json.loads(path.read_bytes().decode("utf-8"))
    targets = payload.get("targets")
    if targets is not None and list(targets) != list(QUANTIZATION_TARGET_SUFFIXES):
        raise ValueError(
            f"Calibration file {path} declares targets {list(targets)}, but AWQ-clip codes are selected for exactly "
            f"{list(QUANTIZATION_TARGET_SUFFIXES)}."
        )
    raw_amax = payload["activation_amax"]
    if sorted(raw_amax) != sorted(fqns):
        extra = sorted(set(raw_amax) - set(fqns))
        missing = sorted(set(fqns) - set(raw_amax))
        raise ValueError(
            f"Calibration file {path} must cover exactly the {len(list(fqns))} selected NVFP4 W4A4 module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}. An AWQ-clip artifact describes all of them."
        )
    activation_amax: Dict[str, float] = {}
    for fqn in sorted(raw_amax):
        value = raw_amax[fqn]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Calibration entry '{fqn}' in {path} must be a number, got {value!r}")
        number = float(value) * float(WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"Calibration entry '{fqn}' in {path} must be finite and positive, got {value!r}")
        activation_amax[fqn] = number
    return {"path": str(path), "identity": identity, "activation_amax": activation_amax}


def load_activation_sample_file(label: str, sample_path: str) -> Dict[str, Any]:
    """
    Load and strictly validate one labelled bounded activation-sample artifact.

    The file is read with ``torch.load(..., weights_only=True)``, so a sample file can never execute code while
    being inspected. Everything else about it is checked: the closed key set, the schema and version, the
    checkpoint digest, the recorded seed and row bound, and every retained tensor's rank, dtype, device,
    contiguity, row count, width and finiteness.

    Args:
        label (str): Non-empty source-group label this file is assigned to.
        sample_path (str): Path to the artifact written by the evaluator's sample collector.

    Returns:
        entry (Dict[str, Any]): ``label``, ``path``, ``name``, ``sha256``, ``size_bytes``, ``checkpoint_sha256``,
            ``seed``, ``max_rows``, ``targets``, ``metadata``, sorted ``fqns``, per-FQN ``samples``, and the
            recorded ``total_finite_rows`` and ``nonfinite_rows``.

    Raises:
        ValueError: If the label is empty or the file is not a valid, non-empty sample artifact.
        OSError: If the file cannot be read.
    """
    label = str(label).strip()
    if not label:
        raise ValueError(f"Activation sample file '{sample_path}' was given an empty source label.")
    path = Path(sample_path).expanduser()
    raw = path.read_bytes()
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Activation sample file {path} must contain a dict, got {type(payload).__name__}")
    unknown = sorted(set(payload) - ACTIVATION_SAMPLE_KEYS)
    absent = sorted(ACTIVATION_SAMPLE_KEYS - set(payload))
    if unknown or absent:
        raise ValueError(
            f"Activation sample file {path} must contain exactly the keys {sorted(ACTIVATION_SAMPLE_KEYS)}; "
            f"unknown keys {unknown}, missing keys {absent}."
        )
    if payload["schema"] != ACTIVATION_SAMPLE_SCHEMA:
        raise ValueError(
            f"Activation sample file {path} declares schema {payload['schema']!r}, but "
            f"{ACTIVATION_SAMPLE_SCHEMA!r} is required."
        )
    if isinstance(payload["version"], bool) or payload["version"] != ACTIVATION_SAMPLE_VERSION:
        raise ValueError(
            f"Activation sample file {path} has version {payload['version']!r}, but version "
            f"{ACTIVATION_SAMPLE_VERSION} is required."
        )
    checkpoint_sha256 = validate_sha256_digest(payload["checkpoint_sha256"], f"'checkpoint_sha256' in {path}")
    seed = _non_negative_int(payload["seed"], f"'seed' in {path}")
    max_rows = _non_negative_int(payload["max_rows"], f"'max_rows' in {path}")
    if max_rows <= 0:
        raise ValueError(f"'max_rows' in {path} must be positive, got {max_rows!r}")
    targets = payload["targets"]
    if not isinstance(targets, (list, tuple)) or not all(isinstance(target, str) for target in targets):
        raise ValueError(f"Activation sample file {path} must declare 'targets' as a list of strings")
    if not isinstance(payload["metadata"], dict):
        raise ValueError(f"Activation sample file {path} must declare 'metadata' as an object")

    samples = _validated_sample_tensors(payload["samples"], max_rows, path)
    counts: Dict[str, Dict[str, int]] = {}
    for field in ("total_finite_rows", "nonfinite_rows"):
        recorded = payload[field]
        if not isinstance(recorded, dict) or set(recorded) != set(samples):
            raise ValueError(
                f"Activation sample file {path} must give '{field}' over exactly the sampled modules, got "
                f"{sorted(recorded)[:4] if isinstance(recorded, dict) else type(recorded).__name__}."
            )
        counts[field] = {
            fqn: _non_negative_int(recorded[fqn], f"'{field}[{fqn}]' in {path}") for fqn in sorted(recorded)
        }
    for fqn in sorted(samples):
        retained, finite = int(samples[fqn].shape[0]), counts["total_finite_rows"][fqn]
        if retained > finite:
            raise ValueError(
                f"Activation sample file {path} retained {retained} row(s) for '{fqn}' out of {finite} finite "
                "one(s); a bounded reservoir cannot keep more rows than it saw."
            )
    return {
        "label": label,
        "path": str(path),
        "name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "max_rows": max_rows,
        "targets": tuple(str(target) for target in targets),
        "metadata": payload["metadata"],
        "fqns": tuple(sorted(samples)),
        "samples": samples,
        "total_finite_rows": counts["total_finite_rows"],
        "nonfinite_rows": counts["nonfinite_rows"],
    }


def restore_target_weights(model_path: str, checkpoint_sha256: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Restore the BF16 checkpoint on ``device`` and take the unconverted weights of exactly the NVFP4 W4A4 targets.

    The digest is verified against the file's actual bytes *before* the checkpoint is restored, so codes can never
    bind to a checkpoint they were not selected on. The model is moved, frozen and put in eval mode, and nothing
    here mutates it: the returned weights are detached references the selection reads.

    Args:
        model_path (str): Path to the ``.nemo`` checkpoint.
        checkpoint_sha256 (str): Expected SHA-256 of that file.
        device (torch.device): Device to restore onto.

    Returns:
        weights (Dict[str, torch.Tensor]): FQN -> detached original weight.

    Raises:
        ValueError: If the file does not hash to ``checkpoint_sha256``.
    """
    path = Path(model_path).expanduser()
    actual = file_sha256(path)
    if actual != checkpoint_sha256:
        raise ValueError(
            f"Checkpoint {path} hashes to {actual}, but --checkpoint-sha256 asserts {checkpoint_sha256}; the codes "
            "would name a checkpoint they were not selected on."
        )
    model = SortformerEncLabelModel.restore_from(restore_path=str(path), map_location=device)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    selection = select_quantization_targets(model, AWQ_CLIP_RECIPE)
    modules = dict(model.named_modules())
    return {fqn: modules[fqn].weight.detach() for fqn in selection.fqns_for_precision(PRECISION_NVFP4_W4A4)}


def write_awq_clip_artifact(payload: Dict[str, Any], output_path: str, overwrite: bool = False) -> str:
    """
    Atomically write an AWQ-clip artifact as deterministic, sorted UTF-8 JSON.

    ``allow_nan=False`` is what makes a non-finite objective a failed write rather than a JSON file the runtime
    loader would have to reject later.

    The destination only ever appears complete: the payload is serialized, flushed and fsynced into a temporary
    file beside it and then renamed over it in one step. Every failure -- an unserializable payload, a full disk on
    the flush or the fsync, a rename that cannot happen -- removes that temporary file before propagating, so a
    failed run leaves neither a half-written artifact beside the destination nor a damaged destination.

    Args:
        payload (Dict[str, Any]): Artifact from :func:`build_awq_clip_artifact`.
        output_path (str): Destination path.
        overwrite (bool): Replace an existing destination. Defaults to ``False``.

    Returns:
        output_path (str): The path written.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
        ValueError: If the payload holds a non-finite number, which ``allow_nan=False`` refuses.
        TypeError: If the payload holds a value JSON cannot represent.
        OSError: If the temporary file cannot be written, synced or renamed.
    """
    path = Path(output_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"AWQ-clip artifact {path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        # Nothing was renamed, so this file is the only trace of the failed write; an interrupt has to clean up
        # after itself here too, which is why this catches BaseException and re-raises unchanged.
        temporary_path.unlink(missing_ok=True)
        raise
    return str(path)


def resolve_device(device: str) -> torch.device:
    """
    Resolve and check the device the selection must run on, without ever falling back to another one.

    Args:
        device (str): PyTorch device string, e.g. ``'cuda'``, ``'cuda:1'`` or ``'cpu'``.

    Returns:
        resolved (torch.device): The device, with a CUDA request resolved to the concrete current index.

    Raises:
        ValueError: If the string does not name a device, names a device type this builder does not run on, or
            names a CUDA device this runtime does not have. A requested GPU is never silently downgraded.
    """
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"--device must be a non-empty PyTorch device string, got {device!r}")
    try:
        resolved = torch.device(device.strip())
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"--device {device!r} is not a valid PyTorch device string: {error}") from error
    if resolved.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"--device {device!r} selects a '{resolved.type}' device, but this builder only runs on "
            f"{list(SUPPORTED_DEVICE_TYPES)}."
        )
    if resolved.type != "cuda":
        return resolved
    if not torch.cuda.is_available():
        raise ValueError(
            f"--device {device!r} requests CUDA, but no CUDA device is available to this process. Run on a GPU "
            "node, or pass --device cpu to accept the slower host selection explicitly."
        )
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if resolved.index >= torch.cuda.device_count():
        raise ValueError(
            f"--device {device!r} requests CUDA device {resolved.index}, but this process can see only "
            f"{torch.cuda.device_count()} CUDA device(s)."
        )
    return resolved


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's exact bytes, read in chunks so a multi-gigabyte checkpoint is not held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the AWQ-clip ratio-code CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the per-block AWQ clipping-ratio codes that select NVFP4 weight block scales for the Sortformer "
            "transformer linears."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to the .nemo checkpoint to select codes for.")
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        metavar="HEX64",
        help=(
            "SHA-256 of --model-path, as 64 hexadecimal characters. It is verified against the file's actual "
            "bytes, against every sample file and against the activation calibration, so codes cannot be built "
            "across checkpoints."
        ),
    )
    parser.add_argument(
        "--activation-calibration-path",
        required=True,
        metavar="PATH",
        help=(
            "Static activation-calibration JSON the quantized run will consume. The codes are selected against "
            "activations quantized with exactly its values, at scale margin "
            f"{WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN}, and the artifact is bound to this file's exact bytes."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Bounded activation-sample artifact labelled with its source group. Every group is weighted equally, "
            "so give different domain/microphone/geometry strata different LABELs. Repeat the flag once per group."
        ),
    )
    parser.add_argument("--output", required=True, help="Destination path of the AWQ-clip artifact JSON.")
    parser.add_argument(
        "--template-arithmetic",
        required=True,
        choices=list(WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS),
        help=(
            "Which ordinary NVFP4 conversion the quantized run will use: "
            f"'{WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED}' for the MSLK-accelerated backend, "
            "the other value for the reference non-Triton one. A block whose selected code is the unclipped ratio "
            "1.00 keeps that conversion's own scale and payload bytes, and the two conversions do not write them "
            "identically, so this is recorded in the artifact and a run on the other backend is refused. There is "
            "no default."
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        metavar="DEVICE",
        help=(
            "PyTorch device the selection runs on, e.g. 'cuda', 'cuda:1' or 'cpu'. The default is the first "
            "visible CUDA device; a requested CUDA device this process cannot see is an error, never a silent CPU "
            "fallback. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=DEFAULT_AWQ_CLIP_ROW_CHUNK_SIZE,
        metavar="N",
        help="Output rows scored at once. Bounds peak memory only; the codes are identical. Default: %(default)s.",
    )
    parser.add_argument(
        "--block-chunk-size",
        type=int,
        default=DEFAULT_AWQ_CLIP_BLOCK_CHUNK_SIZE,
        metavar="N",
        help="Input blocks scored at once. Bounds peak memory only; the codes are identical. Default: %(default)s.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file. Without this flag an existing destination is an error.",
    )
    return parser


def _parse_inputs(raw_inputs: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Split every ``LABEL=PATH`` argument into a ``(label, path)`` pair.

    Raises:
        ValueError: If an argument does not carry a non-empty label and path.
    """
    inputs: List[Tuple[str, str]] = []
    for raw in raw_inputs:
        label, separator, path = raw.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(f"--input must be given as LABEL=PATH, got {raw!r}")
        inputs.append((label.strip(), path.strip()))
    return inputs


def _require_unique_inputs(inputs: Sequence[Tuple[str, str]]) -> None:
    """Reject a repeated label or a repeated file, either of which would count one group twice."""
    if not inputs:
        raise ValueError("At least one --input LABEL=PATH is required.")
    labels = [label for label, _ in inputs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"--input labels must be unique, but {duplicates} were given more than once; every source group is "
            "weighted equally, so a group named twice would be counted twice."
        )
    paths = [str(Path(path).expanduser()) for _, path in inputs]
    repeated = sorted({path for path in paths if paths.count(path) > 1})
    if repeated:
        raise ValueError(
            f"--input files must be unique, but {repeated} were given more than once; the same rows would then "
            "contribute under two group labels."
        )


def _validate_sample_entries(
    entries: Sequence[Dict[str, Any]], weights: Dict[str, torch.Tensor], fqns: Sequence[str], digest: str
) -> None:
    """Reject any labelled sample file that does not describe this checkpoint's exact selected modules."""
    for entry in entries:
        if entry["checkpoint_sha256"] != digest:
            raise ValueError(
                f"Activation sample file {entry['path']} was collected on checkpoint {entry['checkpoint_sha256']}, "
                f"but the codes are finalized against {digest}; clipping ratios are only valid for the checkpoint "
                "whose activations produced them."
            )
        if list(entry["targets"]) != list(QUANTIZATION_TARGET_SUFFIXES):
            raise ValueError(
                f"Activation sample file {entry['path']} declares targets {list(entry['targets'])}, but this "
                f"builder selects codes for exactly {list(QUANTIZATION_TARGET_SUFFIXES)}."
            )
        if list(entry["fqns"]) != list(fqns):
            extra = sorted(set(entry["fqns"]) - set(fqns))
            missing = sorted(set(fqns) - set(entry["fqns"]))
            raise ValueError(
                f"Activation sample file {entry['path']} does not cover the {len(list(fqns))} selected module(s): "
                f"extra {extra[:4]}, missing {missing[:4]}. Re-collect it against this checkpoint."
            )
        for fqn in fqns:
            width, expected = int(entry["samples"][fqn].shape[1]), int(weights[fqn].shape[1])
            if width != expected:
                raise ValueError(
                    f"Activation sample file {entry['path']} holds {width}-wide rows for '{fqn}', but its weight "
                    f"has {expected} input channel(s)."
                )


def _validated_sample_tensors(samples: Any, max_rows: int, path: Path) -> Dict[str, torch.Tensor]:
    """Validate the retained per-FQN sample rows of one activation-sample artifact."""
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"Activation sample file {path} must contain a non-empty 'samples' object")
    validated: Dict[str, torch.Tensor] = {}
    for fqn in sorted(samples, key=str):
        if not isinstance(fqn, str) or not fqn.strip():
            raise ValueError(f"Activation sample file {path} has a sample key that is not a module FQN: {fqn!r}")
        tensor = samples[fqn]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Activation sample file {path} must hold a tensor for '{fqn}', got {type(tensor).__name__}"
            )
        if tensor.dim() != 2 or int(tensor.shape[0]) == 0 or int(tensor.shape[1]) == 0:
            raise ValueError(
                f"Activation sample file {path} must hold non-empty rank-2 rows for '{fqn}', got shape "
                f"{tuple(tensor.shape)}"
            )
        if int(tensor.shape[0]) > max_rows:
            raise ValueError(
                f"Activation sample file {path} retained {int(tensor.shape[0])} rows for '{fqn}', which exceeds "
                f"the declared bound of {max_rows}."
            )
        if tensor.dtype != ACTIVATION_SAMPLE_DTYPE:
            raise ValueError(
                f"Activation sample file {path} holds '{fqn}' as {tensor.dtype}; the collector retains "
                f"{ACTIVATION_SAMPLE_DTYPE} rows."
            )
        if tensor.device.type != "cpu":
            raise ValueError(
                f"Activation sample file {path} holds '{fqn}' on {tensor.device}; sample rows are retained on the "
                "host, so this file was not written by the collector."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"Activation sample file {path} holds non-contiguous rows for '{fqn}'.")
        if not bool(torch.isfinite(tensor.to(torch.float32)).all().item()):
            raise ValueError(
                f"Activation sample file {path} holds non-finite rows for '{fqn}'; the collector filters them "
                "out, so this file was not written by it."
            )
        validated[fqn] = tensor
    return validated


def _non_negative_int(value: Any, description: str) -> int:
    """Validate a count as a non-negative integer, rejecting bools and every non-integer number."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer, got {value!r}")
    return int(value)


if __name__ == "__main__":
    sys.exit(main())

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
Build the NVFP4 GPTQ payload that the Sortformer transformer linears are deployed with.

Every other weight-side method in this overlay chooses a *scale*. ``quantization_weight_scale_method='gptq'`` chooses
a *payload*: it keeps the ordinary TorchAO conversion's block scales, its global per-tensor scale and its exact scale
bytes, and changes only which FP4 code each weight is written as, so that the rounding error an already-written input
column made is compensated by the columns that follow it.

This script performs that selection, adapting NVIDIA ModelOpt 0.46.0's GPTQ:

* every ``--input`` file is one labelled source group of bounded activation samples; its rows are passed through the
  exact static NVFP4 activation quantize/dequantize the quantized runtime will perform, using the frozen calibration
  ``--activation-calibration-path`` names;
* each group contributes ``H_g = scaled @ scaled.T`` with ``scaled = sqrt(2 / N_g) * Xq_g.T`` in FP32, and the
  groups are visited in sorted label order and averaged, so every group counts once however many rows it retained;
* columns whose original weight is exactly zero in every output row have their Hessian row and column zeroed and
  their diagonal set to 1, and the diagonal is then damped by ``perc_damp = 0.01`` of its own mean;
* ``h_inv = cholesky(cholesky_inverse(cholesky(H_damped)), upper=True)``, and the reference's column-wise update
  runs over 128-column blocks, writing each column with the payload the *fixed template scales* quantize the current
  working column into and propagating ``(w_col - q_col) / h_inv[i, i]`` to the columns that follow it.

The reference falls back to the identity when a factorization fails. This builder does not: a non-finite Hessian,
inverse, divisor, working value, residual, decoded value or objective is a precise error and no artifact is written.

It is one-shot PTQ payload selection: no labels, no gradients, no optimizer, no training, and the model is only ever
read. ModelOpt is not imported and is not a dependency; only the identity of the wheel whose semantics were adapted
is recorded.

The written JSON carries the final packed payload -- that is the artifact's whole point -- and nothing else that
could identify the data: no activation row, no high-precision weight, no Hessian matrix, no label, no RTTM and no
task metric. Besides the payload it carries one content digest per payload, one digest and a handful of statistics
per Hessian, one digest per original weight and per ordinary-template scale buffer, the identity of the calibration
file, one digest each over the payload, the Hessian evidence and the provenance, and the provenance of the
construction. The runtime recomputes every one of those digests and every aggregate claim before it converts
anything, and re-checks each template's scale bytes before it replaces that module's payload.

Example (the digests are pasted in explicitly so that the values the payload binds to are visible in the command
that produced it):
    python scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_gptq.py \
        --model-path diar_sortformer_4spk-v2.nemo \
        --checkpoint-sha256 6f4b2c0d1e8a7395fd0c1b2a3e4d5f60718293a4b5c6d7e8f9012a3b4c5d6e7f \
        --activation-calibration-path p100_h1375.json \
        --template-arithmetic mslk_triton \
        --device cuda \
        --input near_field=samples_ami.pt \
        --input far_field=samples_notsofar.pt \
        --output gptq_payload.json
"""

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.sortformer_nvfp4_weight_mse import (
    FP8_E4M3_MAX,
    NVFP4_MAX,
    nvfp4_awq_clip_activation_qdq,
    nvfp4_gptq_damped_hessian,
    nvfp4_gptq_hessian,
    nvfp4_gptq_objective,
    nvfp4_ordinary_template,
    nvfp4_template_block_scales,
    nvfp4_template_identity,
    nvfp4_template_values,
    nvfp4_weight_global_scale,
    select_nvfp4_gptq_payload,
)
from nemo.collections.asr.parts.utils.sortformer_quantization import (
    GPTQ_ACTIVATION_QDQ,
    GPTQ_CONSTRUCTION_METHOD,
    GPTQ_CONSTRUCTION_METHOD_VERSION,
    GPTQ_DEAD_COLUMN_RULE,
    GPTQ_GROUP_REDUCTION,
    GPTQ_HESSIAN_DIGEST_METHOD,
    GPTQ_HESSIAN_RULE,
    GPTQ_INVERSE_RULE,
    GPTQ_OBJECTIVE,
    GPTQ_PAYLOAD_ENCODING,
    GPTQ_SCHEMA,
    GPTQ_SCHEMA_VERSION,
    GPTQ_TEMPLATE_SCALE_RULE,
    MODELOPT_REFERENCE_VERSION,
    MODELOPT_REFERENCE_WHEEL_SHA256,
    NVFP4_BLOCK_SIZE,
    PRECISION_NVFP4_W4A4,
    QUANTIZATION_TARGET_SUFFIXES,
    SECTION_DIGEST_METHOD,
    WEIGHT_DIGEST_METHOD,
    WEIGHT_SCALE_GPTQ_ALGORITHM,
    WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION,
    WEIGHT_SCALE_GPTQ_BLOCK_SIZE,
    WEIGHT_SCALE_GPTQ_PERC_DAMP,
    WEIGHT_SCALE_GPTQ_SCALE_MARGIN,
    WEIGHT_SCALE_GPTQ_SCALE_MODE,
    WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED,
    WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS,
    WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE,
    nvfp4_awq_clip_calibration_identity,
    nvfp4_gptq_hessian_digest,
    nvfp4_gptq_weighted_objective,
    nvfp4_section_digest,
    nvfp4_weight_digest,
    select_quantization_targets,
    validate_sha256_digest,
)

# The payload describes every weight of exactly the W4A4 target set of this recipe, which is the only recipe
# ``gptq`` accepts.
GPTQ_RECIPE = "nvfp4_all"

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

# The selection forms one full ``(K, K)`` Hessian per module and then walks its columns, so it runs on the GPU by
# default. ``cuda`` without an index means the first device this process can see, which inside a GPU-scoped
# container is the device the operator meant.
DEFAULT_DEVICE = "cuda"
SUPPORTED_DEVICE_TYPES: Tuple[str, ...] = ("cpu", "cuda")

# Mode of the written artifact. It is set explicitly rather than left to the process umask, so a build that runs as
# root inside a container still leaves a file the host user can read, and two builds leave the same permissions.
ARTIFACT_FILE_MODE = 0o644


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse arguments, select the GPTQ payload, and write the runtime artifact.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        exit_code (int): ``0`` on success. Invalid inputs raise :class:`SystemExit` with a nonzero code.
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = build_gptq_payload(
            model_path=args.model_path,
            checkpoint_sha256=args.checkpoint_sha256,
            activation_calibration_path=args.activation_calibration_path,
            inputs=_parse_inputs(args.input),
            output_path=args.output,
            template_arithmetic=args.template_arithmetic,
            overwrite=args.overwrite,
            device=args.device,
        )
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        # FileExistsError is an OSError, so refusing to overwrite also exits nonzero with an actionable message.
        raise SystemExit(f"error: {error}") from error

    aggregate = payload["provenance"]["aggregate"]
    print(
        f"Wrote {args.output} for checkpoint {payload['checkpoint_sha256']}: "
        f"{aggregate['module_count']} module(s) from {aggregate['source_count']} equally weighted source group(s) "
        f"{aggregate['source_labels']}, {aggregate['weight_count']} weight(s) in "
        f"{aggregate['qdata_byte_length']} packed payload byte(s), {aggregate['dead_column_count']} dead input "
        "column(s)."
    )
    print(
        f"Payload only: {payload['algorithm']} v{payload['algorithm_version']}, perc_damp "
        f"{payload['arithmetic']['perc_damp']}, {payload['arithmetic']['update_block_size']}-column update blocks, "
        f"written under the '{payload['arithmetic']['template_arithmetic']}' ordinary template's own unchanged "
        f"scales and bound to activation calibration {payload['activation_calibration']['name']} "
        f"({payload['activation_calibration']['sha256']}) at margin "
        f"{payload['activation_calibration']['scale_margin']}. Offline Hessian objective "
        f"{aggregate['selected_objective']} against {aggregate['template_objective']} for the ordinary payload, and "
        f"reconstruction MSE {aggregate['selected_mse']} against {aggregate['template_mse']}. Those are offline "
        "weight-side objectives, not DER."
    )
    return 0


def build_gptq_payload(
    model_path: str,
    checkpoint_sha256: str,
    activation_calibration_path: str,
    inputs: Sequence[Tuple[str, str]],
    output_path: str,
    template_arithmetic: str,
    overwrite: bool = False,
    device: str = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    """
    Restore the checkpoint, select the packed payload of every target, and write it out.

    Args:
        model_path (str): Path to the ``.nemo`` checkpoint the samples were collected on.
        checkpoint_sha256 (str): Expected SHA-256 of that file; verified against its actual bytes.
        activation_calibration_path (str): Path of the static activation-calibration JSON the quantized run will
            consume; the Hessians are formed from activations quantized with exactly its values.
        inputs (Sequence[Tuple[str, str]]): ``(label, path)`` pairs, one per balanced source group.
        output_path (str): Destination of the artifact JSON.
        template_arithmetic (str): The ordinary-template construction the quantized run will convert with, one of
            :data:`WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS`.
        overwrite (bool): Replace an existing destination. Defaults to ``False``.
        device (str): PyTorch device the selection runs on, e.g. ``'cuda'``, ``'cuda:1'`` or ``'cpu'``. Resolved
            and checked before anything is read; a CUDA device is never downgraded to the CPU.

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
    payload = build_gptq_artifact(
        weights,
        entries,
        calibration,
        checkpoint_sha256=digest,
        template_arithmetic=template_arithmetic,
    )
    write_gptq_artifact(payload, output_path, overwrite=overwrite)
    return payload


def build_gptq_artifact(
    weights: Dict[str, torch.Tensor],
    entries: Sequence[Dict[str, Any]],
    calibration: Dict[str, Any],
    checkpoint_sha256: str,
    template_arithmetic: str,
) -> Dict[str, Any]:
    """
    Select the payload module by module and assemble the deterministic runtime artifact.

    Every sample file must declare the same checkpoint as the one the payload is finalized against and must cover
    exactly the selected modules with exactly their input widths: rows collected from another checkpoint, or for
    another module set, describe a different model. Modules are visited in sorted FQN order and groups in sorted
    label order, and only one module's quantized rows, Hessian and payload are alive at a time, so the peak memory
    of the whole pass is one module's bounded rows plus one ``(K, K)`` FP32 matrix. No activation tensor, no
    high-precision weight and no Hessian matrix is copied into the artifact.

    Args:
        weights (Dict[str, torch.Tensor]): Unconverted weights of exactly the selected FQNs, on the build device.
        entries (Sequence[Dict[str, Any]]): Loaded sample entries, one per source group.
        calibration (Dict[str, Any]): Validated activation calibration from :func:`load_activation_calibration`.
        checkpoint_sha256 (str): SHA-256 of the checkpoint the weights and the samples come from.
        template_arithmetic (str): The ordinary-template construction the quantized run will convert with, one of
            :data:`WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS`. The payload is written under exactly that
            construction's block scales, and the runtime refuses the other backend.

    Returns:
        payload (Dict[str, Any]): The artifact, ready for :func:`write_gptq_artifact`.

    Raises:
        ValueError: If the digest is malformed, the construction is not one this build reproduces, an entry declares
            another checkpoint, target list or module set, a sample width does not match its weight, or a selection
            is unusable.
    """
    digest = validate_sha256_digest(checkpoint_sha256, "checkpoint_sha256")
    if template_arithmetic not in WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS:
        raise ValueError(
            f"template_arithmetic must be one of {list(WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS)}, got "
            f"{template_arithmetic!r}; the payload is written under the block scales that construction produces, so "
            "there is no default and no fallback."
        )
    if not entries:
        raise ValueError("At least one labelled activation-sample file is required, but none were given.")
    fqns = sorted(weights)
    if not fqns:
        raise ValueError("No NVFP4 W4A4 target module was selected, so there is no payload to select.")
    if sorted(calibration["activation_amax"]) != fqns:
        raise ValueError(
            f"Activation calibration {calibration['path']} covers "
            f"{len(calibration['activation_amax'])} module(s), but the payload is built for {len(fqns)}; the "
            "artifact describes every NVFP4 W4A4 target and cannot be built from a partial calibration."
        )
    _validate_sample_entries(entries, weights, fqns, digest)

    ordered = sorted(entries, key=lambda entry: entry["label"])
    qdata: Dict[str, Dict[str, Any]] = {}
    template_scale: Dict[str, Dict[str, Any]] = {}
    hessian: Dict[str, Dict[str, Any]] = {}
    modules: Dict[str, Dict[str, Any]] = {}
    for fqn in fqns:
        selected = select_module_payload(
            weights[fqn],
            [entry["samples"][fqn] for entry in ordered],
            calibration["activation_amax"][fqn],
            template_arithmetic,
        )
        qdata[fqn] = selected["qdata"]
        template_scale[fqn] = selected["template_scale"]
        hessian[fqn] = selected["hessian"]
        modules[fqn] = selected["module"]
        # Released before the next module's rows are quantized, which is what bounds the peak memory of the pass.
        del selected

    provenance = {
        "method": GPTQ_CONSTRUCTION_METHOD,
        "method_version": GPTQ_CONSTRUCTION_METHOD_VERSION,
        "objective": GPTQ_OBJECTIVE,
        "group_reduction": GPTQ_GROUP_REDUCTION,
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
            "weight_count": sum(modules[fqn]["weight_count"] for fqn in fqns),
            "qdata_byte_length": sum(modules[fqn]["qdata_byte_length"] for fqn in fqns),
            "dead_column_count": sum(hessian[fqn]["dead_column_count"] for fqn in fqns),
            # Reduced with the very helper the runtime loader recomputes them with, in the same canonical order,
            # so the comparison there is exact rather than approximate.
            **{
                field: nvfp4_gptq_weighted_objective(modules, fqns, field)
                for field in ("template_mse", "selected_mse", "template_objective", "selected_objective")
            },
        },
    }
    return {
        "schema": GPTQ_SCHEMA,
        "version": GPTQ_SCHEMA_VERSION,
        "checkpoint_sha256": digest,
        "algorithm": WEIGHT_SCALE_GPTQ_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION,
        "arithmetic": {
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
            # Which ordinary conversion the payload's fixed scales are; the runtime refuses the other backend.
            "template_arithmetic": template_arithmetic,
            "fp4_max": float(NVFP4_MAX),
            "fp8_e4m3_max": float(FP8_E4M3_MAX),
            "modelopt_reference_version": MODELOPT_REFERENCE_VERSION,
            "modelopt_reference_wheel_sha256": MODELOPT_REFERENCE_WHEEL_SHA256,
        },
        "activation_calibration": {
            **calibration["identity"],
            "scale_margin": float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN),
        },
        "weight_digest_method": WEIGHT_DIGEST_METHOD,
        "section_digest_method": SECTION_DIGEST_METHOD,
        "weight_sha256": {fqn: nvfp4_weight_digest(weights[fqn]) for fqn in fqns},
        "payload_encoding": GPTQ_PAYLOAD_ENCODING,
        "qdata": qdata,
        "template_scale": template_scale,
        "hessian": hessian,
        # One digest per section, canonicalized exactly as the runtime loader recomputes them, so the payload, the
        # Hessian evidence and the provenance are each verifiable on their own rather than only through the whole
        # file's bytes.
        "qdata_sha256": nvfp4_section_digest(qdata),
        "hessian_sha256": nvfp4_section_digest(hessian),
        "provenance_sha256": nvfp4_section_digest(provenance),
        "provenance": provenance,
    }


def select_module_payload(
    weight: torch.Tensor,
    group_rows: Sequence[torch.Tensor],
    activation_amax: float,
    template_arithmetic: str,
) -> Dict[str, Any]:
    """
    Select one module's packed payload from its bounded per-group sample rows.

    The rows of each group are moved to the weight's device and passed through the *runtime-matched* NVFP4
    activation quantize/dequantize with this module's calibrated amax before the Hessian is formed, so the update
    compensates the error the quantized layer will actually make and not the error against raw BF16 rows.

    The scales are the runtime's own and are never recomputed: the weight is converted with the ordinary TorchAO
    conversion of the named construction, that template's blocked E4M3 scales are un-swizzled into FP32, and every
    column is written under those scales and the template's unchanged global scale. The ordinary template's own
    decoded values are scored alongside the selected ones, with the same Hessian and the same reduction, so the
    artifact's two objectives are directly comparable.

    Args:
        weight (torch.Tensor): The module's unconverted ``(M, K)`` weight, on the build device.
        group_rows (Sequence[torch.Tensor]): One bounded ``(N_g, K)`` sample tensor per equally weighted group, in
            canonical group order.
        activation_amax (float): The module's calibrated activation maximum, after the run's scale margin.
        template_arithmetic (str): The ordinary-template construction the deployment will convert with.

    Returns:
        selected (Dict[str, Any]): The artifact's ``qdata``, ``template_scale``, ``hessian`` and ``module`` entries
            for this FQN.
    """
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    global_scale = nvfp4_weight_global_scale(weight)
    template = nvfp4_ordinary_template(weight, global_scale, template_arithmetic)
    identity = nvfp4_template_identity(template)
    block_scale = nvfp4_template_block_scales(template)
    template_values = nvfp4_template_values(template)

    quantized: List[torch.Tensor] = []
    try:
        for group in group_rows:
            quantized.append(nvfp4_awq_clip_activation_qdq(group.to(device=weight.device), activation_amax))
        sampled_rows = sum(int(group.shape[0]) for group in quantized)
        damped = nvfp4_gptq_damped_hessian(nvfp4_gptq_hessian(quantized), weight)
    finally:
        # Released before the payload search allocates, whatever happened above.
        quantized.clear()

    selection = select_nvfp4_gptq_payload(weight, global_scale, block_scale, damped.matrix)
    reference = weight.detach().to(torch.float32)
    module = {
        "shape": [rows, columns],
        "weight_count": rows * columns,
        "block_count": rows * columns // NVFP4_BLOCK_SIZE,
        "qdata_byte_length": rows * columns // 2,
        "template_mse": _finite_error(((template_values - reference) ** 2).mean(), "template reconstruction MSE"),
        "selected_mse": _finite_error(((selection.values - reference) ** 2).mean(), "selected reconstruction MSE"),
        "template_objective": nvfp4_gptq_objective(template_values, weight, damped.matrix),
        "selected_objective": nvfp4_gptq_objective(selection.values, weight, damped.matrix),
    }
    raw = encode_qdata(selection.qdata, rows * columns // 2)
    return {
        "qdata": {
            "shape": [rows, columns // 2],
            "dtype": "uint8",
            "byte_length": len(raw),
            "payload": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "template_scale": {
            "shape": [int(size) for size in identity.scale.shape],
            "dtype": str(identity.scale.dtype),
            "byte_length": int(identity.scale.numel()),
            "sha256": nvfp4_weight_digest(identity.scale),
            "global_scale_sha256": nvfp4_weight_digest(identity.global_scale),
        },
        "hessian": {
            "sha256": nvfp4_gptq_hessian_digest(damped.matrix),
            "input_features": columns,
            "sampled_row_count": sampled_rows,
            "dead_column_count": int(damped.dead_columns),
            "damping": float(damped.damping),
            "diagonal_min": float(damped.diagonal_min),
            "diagonal_max": float(damped.diagonal_max),
            "diagonal_mean": float(damped.diagonal_mean),
        },
        "module": module,
    }


def encode_qdata(qdata: torch.Tensor, expected_bytes: int) -> bytes:
    """Return the contiguous row-major host bytes of a packed NVFP4 payload, as the schema declares them."""
    if not isinstance(qdata, torch.Tensor):
        raise TypeError(f"qdata must be a torch.Tensor, got {type(qdata).__name__}.")
    if qdata.dtype != torch.uint8:
        raise ValueError(f"A packed NVFP4 payload must be {torch.uint8} to be encoded, got {qdata.dtype}.")
    raw = qdata.detach().to("cpu").contiguous().reshape(-1).numpy().tobytes()
    if len(raw) != expected_bytes:
        raise ValueError(
            f"The packed NVFP4 payload holds {len(raw)} byte(s), but this weight needs exactly {expected_bytes}; "
            "the installed torchao does not pack as this builder requires."
        )
    return raw


def load_activation_calibration(calibration_path: str, checkpoint_sha256: str, fqns: Sequence[str]) -> Dict[str, Any]:
    """
    Load and strictly validate the static activation calibration the Hessians will be formed against.

    Beyond the schema itself this checks everything that decides the activations the Hessians see: the static scale
    mode, the checkpoint the maxima were collected on, the exact target families, the exact selected module set --
    a GPTQ artifact describes every W4A4 target, so a partial or over-broad calibration is refused -- every value
    being finite and positive, and the baked-headroom contract that makes the runtime margin exactly
    :data:`WEIGHT_SCALE_GPTQ_SCALE_MARGIN`.

    Args:
        calibration_path (str): Path to the schema-v1 calibration JSON.
        checkpoint_sha256 (str): The checkpoint the payload is finalized against.
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
    if identity["scale_mode"] != WEIGHT_SCALE_GPTQ_SCALE_MODE:
        raise ValueError(
            f"Calibration file {path} declares scale_mode {identity['scale_mode']!r}, but a GPTQ payload is "
            f"selected against a '{WEIGHT_SCALE_GPTQ_SCALE_MODE}' calibration artifact."
        )
    if identity["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            f"Calibration file {path} was collected on checkpoint {identity['checkpoint_sha256']!r}, but the "
            f"payload is finalized against {checkpoint_sha256}; activation scales are only valid for the checkpoint "
            "that produced them, and a GPTQ artifact must name the one it is bound to."
        )
    # ``nvfp4_awq_clip_calibration_identity`` has already refused a file that declares no checkpoint, states no
    # finite positive headroom, does not state ``headroom_baked_in`` as boolean true, or presumes another runtime
    # margin. This repeats the margin comparison against the value the payload is actually selected at, so the two
    # cannot drift apart if the shared identity ever gains a second accepted margin.
    margin = identity["runtime_scale_margin"]
    if float(margin) != float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN):
        raise ValueError(
            f"Calibration file {path} presumes a runtime scale margin of {float(margin)!r}, but a GPTQ payload is "
            f"selected at exactly {float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN)!r}; its baked-in headroom would be applied "
            "twice or not at all."
        )

    payload = json.loads(path.read_bytes().decode("utf-8"))
    targets = payload.get("targets")
    if targets is not None and list(targets) != list(QUANTIZATION_TARGET_SUFFIXES):
        raise ValueError(
            f"Calibration file {path} declares targets {list(targets)}, but a GPTQ payload is selected for exactly "
            f"{list(QUANTIZATION_TARGET_SUFFIXES)}."
        )
    raw_amax = payload["activation_amax"]
    if sorted(raw_amax) != sorted(fqns):
        extra = sorted(set(raw_amax) - set(fqns))
        missing = sorted(set(fqns) - set(raw_amax))
        raise ValueError(
            f"Calibration file {path} must cover exactly the {len(list(fqns))} selected NVFP4 W4A4 module(s); "
            f"extra {extra[:4]}, missing {missing[:4]}. A GPTQ artifact describes all of them."
        )
    activation_amax: Dict[str, float] = {}
    for fqn in sorted(raw_amax):
        value = raw_amax[fqn]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Calibration entry '{fqn}' in {path} must be a number, got {value!r}")
        number = float(value) * float(WEIGHT_SCALE_GPTQ_SCALE_MARGIN)
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

    The digest is verified against the file's actual bytes *before* the checkpoint is restored, so a payload can
    never bind to a checkpoint it was not selected on. The model is moved, frozen and put in eval mode, and nothing
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
            f"Checkpoint {path} hashes to {actual}, but --checkpoint-sha256 asserts {checkpoint_sha256}; the "
            "payload would name a checkpoint it was not selected on."
        )
    model = SortformerEncLabelModel.restore_from(restore_path=str(path), map_location=device)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    selection = select_quantization_targets(model, GPTQ_RECIPE)
    modules = dict(model.named_modules())
    return {fqn: modules[fqn].weight.detach() for fqn in selection.fqns_for_precision(PRECISION_NVFP4_W4A4)}


def write_gptq_artifact(payload: Dict[str, Any], output_path: str, overwrite: bool = False) -> str:
    """
    Atomically write a GPTQ artifact as deterministic, sorted UTF-8 JSON with mode ``0644``.

    ``allow_nan=False`` is what makes a non-finite objective a failed write rather than a JSON file the runtime
    loader would have to reject later.

    The destination only ever appears complete: the payload is serialized, flushed, fsynced and chmod-ed into a
    temporary file beside it and then renamed over it in one step, so the final artifact is host-readable even when
    the build ran as root in a container and never carries a temporary file's private mode. Every failure -- an
    unserializable payload, a full disk on the flush or the fsync, a chmod or rename that cannot happen -- removes
    that temporary file before propagating, so a failed run leaves neither a half-written artifact beside the
    destination nor a damaged destination.

    Args:
        payload (Dict[str, Any]): Artifact from :func:`build_gptq_artifact`.
        output_path (str): Destination path.
        overwrite (bool): Replace an existing destination. Defaults to ``False``.

    Returns:
        output_path (str): The path written.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
        ValueError: If the payload holds a non-finite number, which ``allow_nan=False`` refuses.
        TypeError: If the payload holds a value JSON cannot represent.
        OSError: If the temporary file cannot be written, synced, chmod-ed or renamed.
    """
    path = Path(output_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"GPTQ artifact {path} already exists. Pass --overwrite to replace it.")
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
        # Set on the temporary file, so the destination has the final mode from the instant it appears.
        os.chmod(temporary_path, ARTIFACT_FILE_MODE)
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


def artifact_file_mode(path: Path) -> int:
    """The permission bits of a written artifact, for the builder's own checks and for tests."""
    return stat.S_IMODE(Path(path).stat().st_mode)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the GPTQ payload CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the packed NVFP4 GPTQ payload the Sortformer transformer linears are deployed with, under the "
            "ordinary template's own unchanged block and global scales."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to the .nemo checkpoint to select a payload for.")
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        metavar="HEX64",
        help=(
            "SHA-256 of --model-path, as 64 hexadecimal characters. It is verified against the file's actual "
            "bytes, against every sample file and against the activation calibration, so a payload cannot be built "
            "across checkpoints."
        ),
    )
    parser.add_argument(
        "--activation-calibration-path",
        required=True,
        metavar="PATH",
        help=(
            "Static activation-calibration JSON the quantized run will consume. The Hessians are formed from "
            "activations quantized with exactly its values, at scale margin "
            f"{WEIGHT_SCALE_GPTQ_SCALE_MARGIN}, and the artifact is bound to this file's exact bytes."
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
    parser.add_argument("--output", required=True, help="Destination path of the GPTQ artifact JSON.")
    parser.add_argument(
        "--template-arithmetic",
        required=True,
        choices=list(WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS),
        help=(
            "Which ordinary NVFP4 conversion the quantized run will use: "
            f"'{WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED}' for the MSLK-accelerated backend, the other "
            "value for the reference non-Triton one. The payload is written under exactly that construction's "
            "block scales and global scale, and the two conversions do not produce the same ones, so this is "
            "recorded in the artifact and a run on the other backend is refused. There is no default."
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
                f"but the payload is finalized against {digest}; a GPTQ payload is only valid for the checkpoint "
                "whose activations produced its Hessians."
            )
        if list(entry["targets"]) != list(QUANTIZATION_TARGET_SUFFIXES):
            raise ValueError(
                f"Activation sample file {entry['path']} declares targets {list(entry['targets'])}, but this "
                f"builder selects a payload for exactly {list(QUANTIZATION_TARGET_SUFFIXES)}."
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


def _finite_error(value: torch.Tensor, description: str) -> float:
    """Read one reduced error as a python float, refusing a non-finite one instead of writing it out."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"The {description} is not finite ({number!r}); refusing to write it into an artifact.")
    return number


def _non_negative_int(value: Any, description: str) -> int:
    """Validate a count as a non-negative integer, rejecting bools and every non-integer number."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer, got {value!r}")
    return int(value)


if __name__ == "__main__":
    sys.exit(main())

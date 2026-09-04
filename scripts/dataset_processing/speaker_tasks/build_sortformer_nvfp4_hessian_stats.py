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
Build the diagonal-Hessian statistics that select NVFP4 weight block scales for the Sortformer transformer linears.

The NVFP4 block scale TorchAO picks is the block's own absolute maximum, which is the best choice for the *weight*
and not for the *layer output*. ``quantization_weight_scale_method='local_hessian'`` instead picks, per 16-weight
block, the E4M3 scale minimizing

    sum_j h_damped[j] * (W[r, j] - Q_scale(W[r, j])) ** 2,   h_damped = h + 0.01 * mean(h)

where ``h[j] = E[x_j^2]`` is the second moment of input channel ``j`` on activations this checkpoint actually
produced. This script computes those ``h`` vectors and writes them as the strict runtime artifact that the quantized
run binds to. It is one-shot PTQ statistics collection: no labels, no gradients, no optimizer, no training, and the
model is only ever read.

Every ``--input`` file is one labelled source group -- a stable domain/microphone/geometry stratum -- of bounded
activation samples, read with ``torch.load(..., weights_only=True)`` so that inspecting one can never execute code.
Groups are weighted **equally**: ``h = mean_s(mean_rows(X_s ** 2))``, so a high-volume corpus cannot outvote a small
stratum however many rows each of them retained. Every sample file must declare ``--checkpoint-sha256``, which is
also verified against the actual bytes of ``--model-path`` before the checkpoint is restored.

The written JSON carries no activation row, no weight, no quantized payload, no label, no RTTM and no task metric:
the moment vectors, one canonical digest per original weight, one digest each over the moments and over the
provenance, the checkpoint identity, the algorithm identity with its damping, and the provenance of the
construction. The runtime recomputes every one of those digests before it converts anything.

Example (the digest is the output of ``sha256sum diar_sortformer_4spk-v2.nemo``, pasted in explicitly so that the
value the statistics bind to is visible in the command that produced them):
    python scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_hessian_stats.py \
        --model-path diar_sortformer_4spk-v2.nemo \
        --checkpoint-sha256 6f4b2c0d1e8a7395fd0c1b2a3e4d5f60718293a4b5c6d7e8f9012a3b4c5d6e7f \
        --device cuda \
        --input near_field=samples_ami.pt \
        --input far_field=samples_notsofar.pt \
        --output diagonal_hessian.json
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.sortformer_quantization import (
    HESSIAN_CONSTRUCTION_METHOD,
    HESSIAN_CONSTRUCTION_METHOD_VERSION,
    HESSIAN_GROUP_REDUCTION,
    HESSIAN_OBJECTIVE,
    HESSIAN_SCHEMA,
    HESSIAN_SCHEMA_VERSION,
    PRECISION_NVFP4_W4A4,
    QUANTIZATION_TARGET_SUFFIXES,
    WEIGHT_DIGEST_METHOD,
    WEIGHT_SCALE_HESSIAN_ALGORITHM,
    WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION,
    WEIGHT_SCALE_HESSIAN_DAMPING,
    nvfp4_section_digest,
    nvfp4_weight_digest,
    select_quantization_targets,
    validate_sha256_digest,
)

# The statistics describe the inputs of exactly the W4A4 target set of this recipe, which is the only recipe
# ``local_hessian`` accepts.
HESSIAN_RECIPE = "nvfp4_all"

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

# The per-source reduction is one squared mean over a bounded row set per module, so it runs on the GPU by default.
# ``cuda`` without an index means the first device this process can see, which inside a GPU-scoped container is the
# device the operator meant.
DEFAULT_DEVICE = "cuda"
SUPPORTED_DEVICE_TYPES: Tuple[str, ...] = ("cpu", "cuda")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse arguments, build the diagonal-Hessian statistics, and write the runtime artifact.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        exit_code (int): ``0`` on success. Invalid inputs raise :class:`SystemExit` with a nonzero code.
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = build_hessian_stats(
            model_path=args.model_path,
            checkpoint_sha256=args.checkpoint_sha256,
            inputs=_parse_inputs(args.input),
            output_path=args.output,
            overwrite=args.overwrite,
            device=args.device,
        )
    except (ValueError, RuntimeError, OSError) as error:
        # FileExistsError is an OSError, so refusing to overwrite also exits nonzero with an actionable message.
        raise SystemExit(f"error: {error}") from error

    aggregate = payload["provenance"]["aggregate"]
    print(
        f"Wrote {args.output} for checkpoint {payload['checkpoint_sha256']}: "
        f"{aggregate['module_count']} module(s) from {aggregate['source_count']} equally weighted source group(s) "
        f"{aggregate['source_labels']}, {aggregate['moment_count']} second moment(s) in "
        f"[{aggregate['moment_min']}, {aggregate['moment_max']}]."
    )
    print(
        f"Statistics only: {payload['algorithm']} v{payload['algorithm_version']} with damping "
        f"{payload['damping']}. The artifact carries no activation rows, weights or metrics, and says nothing "
        "about DER."
    )
    return 0


def build_hessian_stats(
    model_path: str,
    checkpoint_sha256: str,
    inputs: Sequence[Tuple[str, str]],
    output_path: str,
    overwrite: bool = False,
    device: str = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    """
    Restore the checkpoint, merge the labelled samples into one moment vector per module, and write the artifact.

    Args:
        model_path (str): Path to the ``.nemo`` checkpoint the samples were collected on.
        checkpoint_sha256 (str): Expected SHA-256 of that file; verified against its actual bytes.
        inputs (Sequence[Tuple[str, str]]): ``(label, path)`` pairs, one per balanced source group.
        output_path (str): Destination of the artifact JSON.
        overwrite (bool): Replace an existing destination. Defaults to ``False``.
        device (str): PyTorch device the reduction runs on, e.g. ``'cuda'``, ``'cuda:1'`` or ``'cpu'``. Resolved
            and checked before anything is read; a CUDA device is never downgraded to the CPU.

    Returns:
        payload (Dict[str, Any]): The artifact that was written.

    Raises:
        ValueError: If a label or path is duplicated, the device is unusable, the digest does not match the
            checkpoint, or an input is invalid.
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
    """
    digest = validate_sha256_digest(checkpoint_sha256, "--checkpoint-sha256")
    _require_unique_inputs(inputs)
    # Resolved before the checkpoint is restored or a sample file is read, so an unusable device fails in a second
    # instead of after a multi-gigabyte restore.
    resolved = resolve_device(device)
    entries = [load_activation_sample_file(label, path) for label, path in inputs]
    weights = restore_target_weights(model_path, digest, resolved)
    payload = build_hessian_artifact(weights, entries, checkpoint_sha256=digest, device=resolved)
    write_hessian_artifact(payload, output_path, overwrite=overwrite)
    return payload


def build_hessian_artifact(
    weights: Dict[str, torch.Tensor],
    entries: Sequence[Dict[str, Any]],
    checkpoint_sha256: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Merge the labelled samples and assemble the deterministic runtime artifact payload.

    Every sample file must declare the same checkpoint as the one the statistics are finalized against and must
    cover exactly the selected modules with exactly their input widths: moments collected from another checkpoint,
    or for another module set, describe a different model. No activation tensor is copied into the payload.

    Args:
        weights (Dict[str, torch.Tensor]): Unconverted weights of exactly the selected FQNs.
        entries (Sequence[Dict[str, Any]]): Loaded sample entries, one per source group.
        checkpoint_sha256 (str): SHA-256 of the checkpoint the weights and the samples come from.
        device (Optional[torch.device]): Device the per-source reduction runs on; the host when ``None``.

    Returns:
        payload (Dict[str, Any]): The artifact, ready for :func:`write_hessian_artifact`.

    Raises:
        ValueError: If the digest is malformed, an entry declares another checkpoint, target list or module set,
            a sample width does not match its weight, or a merged vector is unusable.
    """
    digest = validate_sha256_digest(checkpoint_sha256, "checkpoint_sha256")
    if not entries:
        raise ValueError("At least one labelled activation-sample file is required, but none were given.")
    fqns = sorted(weights)
    if not fqns:
        raise ValueError("No NVFP4 W4A4 target module was selected, so there is nothing to collect statistics for.")

    for entry in entries:
        if entry["checkpoint_sha256"] != digest:
            raise ValueError(
                f"Activation sample file {entry['path']} was collected on checkpoint {entry['checkpoint_sha256']}, "
                f"but the statistics are finalized against {digest}; second moments are only valid for the "
                "checkpoint whose activations produced them."
            )
        if list(entry["targets"]) != list(QUANTIZATION_TARGET_SUFFIXES):
            raise ValueError(
                f"Activation sample file {entry['path']} declares targets {list(entry['targets'])}, but this "
                f"builder collects statistics for exactly {list(QUANTIZATION_TARGET_SUFFIXES)}."
            )
        if list(entry["fqns"]) != fqns:
            extra = sorted(set(entry["fqns"]) - set(fqns))
            missing = sorted(set(fqns) - set(entry["fqns"]))
            raise ValueError(
                f"Activation sample file {entry['path']} does not cover the {len(fqns)} selected module(s): extra "
                f"{extra[:4]}, missing {missing[:4]}. Re-collect it against this checkpoint."
            )
        for fqn in fqns:
            width, expected = int(entry["samples"][fqn].shape[1]), int(weights[fqn].shape[1])
            if width != expected:
                raise ValueError(
                    f"Activation sample file {entry['path']} holds {width}-wide rows for '{fqn}', but its weight "
                    f"has {expected} input channel(s)."
                )

    moments = merge_second_moments(entries, fqns, device=device)
    values = [value for fqn in fqns for value in moments[fqn]]
    ordered = sorted(entries, key=lambda entry: entry["label"])
    diagonal_hessian = {fqn: moments[fqn] for fqn in fqns}
    provenance = {
        "method": HESSIAN_CONSTRUCTION_METHOD,
        "method_version": HESSIAN_CONSTRUCTION_METHOD_VERSION,
        "objective": HESSIAN_OBJECTIVE,
        "group_reduction": HESSIAN_GROUP_REDUCTION,
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
        "aggregate": {
            "module_count": len(fqns),
            "source_count": len(ordered),
            "source_labels": [entry["label"] for entry in ordered],
            "moment_count": len(values),
            "moment_min": min(values),
            "moment_max": max(values),
        },
    }
    return {
        "schema": HESSIAN_SCHEMA,
        "version": HESSIAN_SCHEMA_VERSION,
        "checkpoint_sha256": digest,
        "algorithm": WEIGHT_SCALE_HESSIAN_ALGORITHM,
        "algorithm_version": WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION,
        "damping": float(WEIGHT_SCALE_HESSIAN_DAMPING),
        "weight_digest_method": WEIGHT_DIGEST_METHOD,
        "weight_sha256": {fqn: nvfp4_weight_digest(weights[fqn]) for fqn in fqns},
        # One digest per section, canonicalized exactly as the runtime loader recomputes them, so the moments and
        # the provenance are each verifiable on their own rather than only through the whole file's bytes.
        "moment_sha256": nvfp4_section_digest(diagonal_hessian),
        "provenance_sha256": nvfp4_section_digest(provenance),
        "diagonal_hessian": diagonal_hessian,
        "provenance": provenance,
    }


def merge_second_moments(
    entries: Sequence[Dict[str, Any]], fqns: Sequence[str], device: Optional[torch.device] = None
) -> Dict[str, List[float]]:
    """
    Merge the labelled sample groups into one second-moment vector per module, weighting every group equally.

    For source group ``s`` the per-module contribution is ``h_s[fqn] = mean_rows(X_s ** 2)``, reduced with float64
    accumulation over the retained rows. The merged vector is the plain mean over groups,
    ``h[fqn] = mean_s(h_s[fqn])``, so a group that retained ten times more rows than another still counts once.
    Groups are visited in sorted label order and accumulated in float64, so the result does not depend on the order
    the files were given in.

    Args:
        entries (Sequence[Dict[str, Any]]): Loaded sample entries, one per source group.
        fqns (Sequence[str]): Modules to merge, in the order the caller wants them keyed.
        device (Optional[torch.device]): Device the squared reduction runs on; the host when ``None``.

    Returns:
        moments (Dict[str, List[float]]): One finite non-negative vector per module.

    Raises:
        ValueError: If ``entries`` is empty or a merged vector is non-finite, negative or identically zero.
    """
    ordered = sorted(entries, key=lambda entry: entry["label"])
    if not ordered:
        raise ValueError("At least one labelled activation-sample file is required to merge second moments.")

    merged: Dict[str, List[float]] = {}
    for fqn in fqns:
        total = None
        for entry in ordered:
            rows = entry["samples"][fqn]
            if device is not None:
                rows = rows.to(device=device)
            # Squares are formed in float32 and accumulated in float64, so a long bounded row set does not lose
            # the small channels to the large ones.
            per_source = rows.to(torch.float32).pow(2).mean(dim=0, dtype=torch.float64).to("cpu")
            total = per_source if total is None else total + per_source
        moments = total / float(len(ordered))
        if not bool(torch.isfinite(moments).all()) or bool((moments < 0.0).any()):
            raise ValueError(
                f"The merged second moments of '{fqn}' are not all finite and non-negative; the sample files do "
                "not describe usable activations for it."
            )
        if float(moments.sum()) <= 0.0:
            raise ValueError(
                f"The merged second moments of '{fqn}' are identically zero, which describes a module whose "
                "inputs never varied; refusing to write statistics that cannot weight anything."
            )
        merged[fqn] = [float(value) for value in moments.tolist()]
    return merged


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

    The digest is verified against the file's actual bytes *before* the checkpoint is restored, so statistics can
    never bind to a checkpoint they were not built on. The model is moved, frozen and put in eval mode, and nothing
    here mutates it: the returned weights are detached references used for their digests and their shapes only.

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
            "statistics would name a checkpoint they were not built on."
        )
    model = SortformerEncLabelModel.restore_from(restore_path=str(path), map_location=device)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    selection = select_quantization_targets(model, HESSIAN_RECIPE)
    modules = dict(model.named_modules())
    return {fqn: modules[fqn].weight.detach() for fqn in selection.fqns_for_precision(PRECISION_NVFP4_W4A4)}


def write_hessian_artifact(payload: Dict[str, Any], output_path: str, overwrite: bool = False) -> str:
    """
    Atomically write a diagonal-Hessian artifact as deterministic, sorted UTF-8 JSON.

    ``allow_nan=False`` is what makes a non-finite moment a failed write rather than a JSON file the runtime
    loader would have to reject later.

    The destination only ever appears complete: the payload is serialized, flushed and fsynced into a temporary
    file beside it and then renamed over it in one step. Every failure -- an unserializable payload, a full disk on
    the flush or the fsync, a rename that cannot happen -- removes that temporary file before propagating, so a
    failed run leaves neither a half-written statistics JSON beside the destination nor a damaged destination.

    Args:
        payload (Dict[str, Any]): Artifact from :func:`build_hessian_artifact`.
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
        raise FileExistsError(f"Diagonal-Hessian artifact {path} already exists. Pass --overwrite to replace it.")
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
    Resolve and check the device the reduction must run on, without ever falling back to another one.

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
            "node, or pass --device cpu to accept the slower host reduction explicitly."
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
    """Build the argument parser for the diagonal-Hessian statistics CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the per-module diagonal activation second moments that select NVFP4 weight block scales for "
            "the Sortformer transformer linears."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to the .nemo checkpoint to collect statistics for.")
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        metavar="HEX64",
        help=(
            "SHA-256 of --model-path, as 64 hexadecimal characters. It is verified against the file's actual "
            "bytes and against every sample file, so statistics cannot be built across checkpoints."
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
    parser.add_argument("--output", required=True, help="Destination path of the diagonal-Hessian artifact JSON.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        metavar="DEVICE",
        help=(
            "PyTorch device the reduction runs on, e.g. 'cuda', 'cuda:1' or 'cpu'. The default is the first "
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

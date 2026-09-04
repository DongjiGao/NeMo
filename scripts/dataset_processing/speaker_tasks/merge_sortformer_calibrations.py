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
Merge Sortformer NVFP4 activation calibrations collected on several data sources into one runtime artifact.

Each input is labelled with the GROUP it came from: a stable logical domain/microphone/geometry stratum. Several
partition files of the *same* stratum reuse one GROUP; different microphone or array-geometry strata must be
given different GROUPs, because everything sharing a GROUP is pooled before the percentile is taken and a
smaller stratum's larger activations would then be discarded as tail.

Within a group the per-invocation observations of every module are pooled and reduced by a conservative
nearest-rank percentile; the value written for a module is the maximum of the group percentiles, multiplied by
``--headroom``. Taking the maximum across groups keeps a small domain from being averaged away by a high-volume
one, and the headroom is baked into the written values, so the merged artifact is consumed with
``quantization_scale_margin=1.0``.

Activation scales are only valid for the weights that produced them, so ``--checkpoint-sha256`` is required and
any input declaring a different checkpoint is rejected. Inputs that predate per-invocation histories (max-only
artifacts) are accepted: each maximum counts as a single observation, and the fallback and their unknown
non-finite/checkpoint provenance are recorded in the output.

Example:
    python scripts/dataset_processing/speaker_tasks/merge_sortformer_calibrations.py \
        --input near_field=calib_ami.json \
        --input near_field=calib_ami_batch8.json \
        --input far_field=calib_ch109.json \
        --checkpoint-sha256 $(sha256sum diar_sortformer_4spk-v2.nemo | cut -d' ' -f1) \
        --percentile 99.9 --headroom 1.1 --output calib_merged.json
"""

import argparse
import sys
from typing import List, Optional, Sequence, Tuple

from nemo.collections.asr.parts.utils.sortformer_quantization import (
    CALIBRATION_PROVENANCE_CLEAN,
    merge_calibration_files,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse arguments, merge the labelled calibration files, and write the merged artifact.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        exit_code (int): ``0`` on success. Invalid inputs raise :class:`SystemExit` with a nonzero code.
    """
    args = _build_parser().parse_args(argv)
    try:
        inputs = _parse_inputs(args.input)
        payload = merge_calibration_files(
            inputs,
            percentile=args.percentile,
            headroom=args.headroom,
            checkpoint_sha256=args.checkpoint_sha256,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError) as error:
        # FileExistsError is an OSError, so refusing to overwrite also exits nonzero with an actionable message.
        raise SystemExit(f"error: {error}") from error

    provenance = payload["metadata"]
    print(
        f"Wrote {args.output} from {len(inputs)} input(s) in {len(provenance['groups'])} group(s) "
        f"{provenance['groups']}: {provenance['target_module_count']} modules at percentile "
        f"{provenance['percentile']} with headroom {provenance['headroom']} baked in "
        f"(runtime scale margin {provenance['runtime_scale_margin']}), for checkpoint "
        f"{provenance['checkpoint_sha256']}."
    )
    if (
        provenance["nonfinite_status"] != CALIBRATION_PROVENANCE_CLEAN
        or provenance["checkpoint_identity_asserted_only"]
    ):
        print(
            f"Note: non-finite provenance is '{provenance['nonfinite_status']}'; checkpoint identity rests on "
            f"--checkpoint-sha256 alone for at least one input: {provenance['checkpoint_identity_asserted_only']}. "
            f"See 'metadata' in {args.output}."
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the merge CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge labelled Sortformer NVFP4 activation calibrations into one schema-v1 runtime artifact whose "
            "headroom is already baked into the per-module values."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="GROUP=PATH",
        help=(
            "Calibration JSON labelled with its stratum. GROUP is a stable logical domain/microphone/geometry "
            "stratum: repeat the same GROUP for several partition files of that one stratum (for example "
            "latency or batch partitions of the same corpus), and give different microphone or array-geometry "
            "strata different GROUPs, since everything in a GROUP is pooled before the percentile is taken."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Nearest-rank percentile in (0, 100] applied per group and module. Default: %(default)s.",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.0,
        help=(
            "Positive multiplier baked into every written value, so the run that consumes the artifact uses "
            "quantization_scale_margin=1.0. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        metavar="HEX64",
        help=(
            "SHA-256 of the checkpoint every input was calibrated on, as 64 hexadecimal characters. Any input "
            "that declares a different checkpoint is rejected, so calibrations of different checkpoints cannot "
            "be merged by accident."
        ),
    )
    parser.add_argument("--output", required=True, help="Destination path of the merged calibration JSON.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file. Without this flag an existing destination is an error.",
    )
    return parser


def _parse_inputs(raw_inputs: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Split every ``GROUP=PATH`` argument into a ``(group, path)`` pair.

    Raises:
        ValueError: If an argument does not carry a non-empty group and path.
    """
    inputs: List[Tuple[str, str]] = []
    for raw in raw_inputs:
        group, separator, path = raw.partition("=")
        if not separator or not group.strip() or not path.strip():
            raise ValueError(f"--input must be given as GROUP=PATH, got {raw!r}")
        inputs.append((group.strip(), path.strip()))
    return inputs


if __name__ == "__main__":
    sys.exit(main())

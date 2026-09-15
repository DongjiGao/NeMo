#!/usr/bin/env python3
"""Bake a static per-tensor FP8 encoder recipe into a SpeechLM checkpoint.

Writes an ``encoder_quantization`` block into ``config.json`` so a checkpoint
reproduces its own encoder quantization without the caller setting any
``NEMO_ENC_QUANT*`` environment variable. The serving stack picks it up in
``salm/model.py`` and hands it to the ASR encoder, which swaps its Linears on the
first forward.

This is the scales-only stage. Weights stay BF16 in ``perception.safetensors``
and are quantized at load, so the file does not shrink; what it buys is a
self-contained checkpoint whose numbers are reproducible from the artifact alone.
Moving the FP8 weights themselves into the checkpoint is a separate change, since
the Linears would have to be replaced before ``load_state_dict`` rather than on
first forward.

Only ``config.json`` is written. Every other file is hardlinked, so a variant
costs kilobytes rather than the 33 GB the checkpoint occupies.

Usage:
    python bake_encoder_quant_config.py \
        --source /data/.../speechlm_hr8_step17000_fp8_text512 \
        --amax   /data/.../hr8_enc_calib_8set/encoder_amax.json \
        --margin 1.5 \
        --output /data/.../speechlm_hr8_step17000_fp8_encfp8static
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The four dense GEMMs per encoder block. 32 blocks -> 128 matrices, which is
# also the assertion the encoder enforces after the swap.
PATTERNS = ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
EXPECT_REPLACED = 128


def build_recipe(amax: dict[str, float], margin: float, amax_source: str) -> dict:
    return {
        "format": "fp8_e4m3",
        # Matches enc_nvfp4.FP8Linear: weight scale is amax over the input dim per
        # output channel, activation scale is one scalar per matrix. Both use the
        # divisor convention, scale = amax / 448, so the GEMM multiplies back.
        "weight_scale": "per_output_channel",
        "activation_scale": "static_per_tensor",
        "scale_margin": margin,
        "patterns": list(PATTERNS),
        "expect_replaced": EXPECT_REPLACED,
        # Regex selecting which matched matrices go FP8 rather than NVFP4. "." is
        # every one of them; the encoder's activation range (amax up to ~72 on
        # ffn.net.3) is too wide for FP4's single mantissa bit.
        "fp8_re": ".",
        "activation_amax": amax,
        "provenance": {
            "amax_source": amax_source,
            "note": (
                "Activation amax collected on held-out dev audio, then multiplied by "
                "scale_margin at load. The margin covers the gap between the "
                "calibration set and the serving distribution; a scale that is too "
                "small clips activations."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="checkpoint to derive from")
    ap.add_argument("--amax", type=Path, required=True, help="encoder_amax.json from a calib run")
    ap.add_argument("--margin", type=float, default=1.5, help="multiplier applied to amax at load")
    ap.add_argument("--output", type=Path, required=True, help="new checkpoint directory")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    args = ap.parse_args()

    config_path = args.source / "config.json"
    if not config_path.exists():
        raise SystemExit(f"no config.json in {args.source}")

    amax = json.loads(args.amax.read_text())
    if len(amax) != EXPECT_REPLACED:
        raise SystemExit(
            f"{args.amax} holds {len(amax)} entries but the recipe quantizes {EXPECT_REPLACED} "
            "matrices. A short file usually means the calibration run died early."
        )
    if args.margin <= 0:
        raise SystemExit(f"--margin must be positive, got {args.margin}")

    config = json.loads(config_path.read_text())
    if "encoder_quantization" in config and not args.force:
        raise SystemExit(f"{config_path} already carries encoder_quantization; pass --force to replace")
    config["encoder_quantization"] = build_recipe(amax, args.margin, str(args.amax))

    args.output.mkdir(parents=True, exist_ok=True)
    linked = 0
    for path in sorted(args.source.iterdir()):
        if path.name == "config.json" or not path.is_file():
            continue
        dest = args.output / path.name
        if dest.exists():
            if not args.force:
                raise SystemExit(f"{dest} exists; pass --force to replace")
            dest.unlink()
        # Hardlink where the filesystem allows it, else symlink. Both are free;
        # copying is the fallback of last resort because the weights are 32 GB and
        # this script exists to make cheap config variants. safetensors opens by
        # path, so a symlinked shard loads identically to a real one.
        try:
            os.link(path, dest)
        except OSError:
            try:
                dest.symlink_to(path.resolve())
            except OSError:
                import shutil

                shutil.copy2(path, dest)
        linked += 1

    # Subdirectories such as llm_backbone are referenced by config, so mirror them.
    for path in sorted(args.source.iterdir()):
        if path.is_dir() and not (args.output / path.name).exists():
            (args.output / path.name).symlink_to(path.resolve())

    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    scales = sorted(amax.values())
    print(f"linked {linked} files, wrote config.json -> {args.output}")
    print(
        f"recipe: fp8_e4m3, per-output-channel weights, static per-tensor activations, "
        f"margin {args.margin}x, {len(amax)} scales"
    )
    print(
        f"amax range: min={scales[0]:.4g} median={scales[len(scales) // 2]:.4g} max={scales[-1]:.4g}"
    )
    print("serve with no NEMO_ENC_QUANT* variables set; expect '128 FP8 (128 static / 0 dynamic)'")


if __name__ == "__main__":
    main()

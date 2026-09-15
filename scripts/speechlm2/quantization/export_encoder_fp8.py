#!/usr/bin/env python3
"""Write FP8 encoder weights and per-output-channel scales into a checkpoint.

Option B of the encoder quantization port. Where bake_encoder_quant_config.py
ships only activation scales and leaves the weights BF16 to be quantized on
first forward, this converts the weights themselves, so perception.safetensors
shrinks by ~598 MiB and nothing is quantized at load.

Per target Linear it emits, following ModelOpt's layout:
    <name>.weight        float8_e4m3fn, [out, in]
    <name>.weight_scale  float32,       [out]
with scale = amax(|W|, dim=1) / 448, the divisor convention the CUTLASS GEMM
multiplies back. This is the same arithmetic FP8Linear.__init__ performs at
runtime, so the exported tensors must match it bitwise; that is what
test_encoder_fp8_equivalence.py checks.

SCOPE IS THE DANGEROUS PART. The Sortformer diarizer lives in the same file
under perception.encoder.diarization_model.* and uses identical module names, so
a suffix match on "attn.w_qkv" and friends selects 252 tensors rather than 128.
Quantizing the diarizer is not a mild error: its output is fused additively into
the ASR features, so it corrupts every frame. At runtime the swap is protected by
the _enc_quant_is_asr tag set by ParallelExpertEncoder; offline there is no tag,
so selection is by the asr_encoder prefix and the count is asserted.

Usage:
    python export_encoder_fp8.py --source CKPT --output CKPT_FP8ENC
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ASR_PREFIX = "perception.encoder.asr_encoder."
PATTERNS = ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
EXPECT = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def target_keys(keys: list[str]) -> list[str]:
    """The 128 ASR-encoder weight tensors, explicitly not the diarizer's 124."""
    return sorted(
        k
        for k in keys
        if k.startswith(ASR_PREFIX) and any(k.endswith(p + ".weight") for p in PATTERNS)
    )


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of FP8Linear.__init__, kept arithmetically identical on purpose.

    Returns (fp8 weight, fp32 per-output-channel scale of shape [out]).
    """
    w = weight.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    wq = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    # Stored flat as [out], the ModelOpt layout; the runtime buffer is [out, 1].
    return wq, scale.squeeze(-1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_perc = args.source / "perception.safetensors"
    if not src_perc.exists():
        raise SystemExit(f"no perception.safetensors in {args.source}")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force")
    args.output.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src_perc, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        targets = target_keys(keys)
        if len(targets) != EXPECT:
            raise SystemExit(
                f"selected {len(targets)} weight tensors but expected {EXPECT}. Refusing to "
                "export: a wrong selection here silently quantizes the diarizer, whose output "
                "is fused into the ASR features."
            )
        target_set = set(targets)
        for k in keys:
            t = f.get_tensor(k)
            if k in target_set:
                wq, scale = quantize_weight(t)
                tensors[k] = wq
                tensors[k[: -len(".weight")] + ".weight_scale"] = scale
            else:
                tensors[k] = t

    before = src_perc.stat().st_size
    dest = args.output / "perception.safetensors"
    save_file(tensors, dest)
    after = dest.stat().st_size

    # Everything else is symlinked; only perception.safetensors actually changes.
    for path in sorted(args.source.iterdir()):
        if path.name in ("perception.safetensors", "config.json"):
            continue
        target = args.output / path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path.resolve())

    config = json.loads((args.source / "config.json").read_text())
    quant = config.setdefault("encoder_quantization", {})
    quant["weights_prequantized"] = True
    quant["format"] = "fp8_e4m3"
    quant["weight_scale"] = "per_output_channel"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"quantized {len(targets)} ASR-encoder weight tensors, diarizer untouched")
    print(f"perception.safetensors {before / 2**20:.1f} MiB -> {after / 2**20:.1f} MiB "
          f"(saved {(before - after) / 2**20:.1f} MiB)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

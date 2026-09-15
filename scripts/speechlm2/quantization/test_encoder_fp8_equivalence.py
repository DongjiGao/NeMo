#!/usr/bin/env python3
"""Check an exported FP8 encoder against what FP8Linear builds at runtime.

The primary acceptance test for the Option B port, and deliberately not an
end-to-end WER run: the union-excluded WER band across five runs is ~0.007, so a
real porting bug of moderate size could hide inside measurement noise, while it
shows up here immediately and for free. CPU only, no GPU, no eval.

Compares, for all 128 ASR-encoder matrices:

    exported weight / weight_scale   (after a safetensors round-trip)
    vs
    FP8Linear(nn.Linear(bf16)).weight_fp8 / .weight_scale   (computed at load)

Bitwise equality is the bar, not closeness. Both sides run the same arithmetic,
so any difference means something structural: a dtype not surviving the file
format, a transposed or squeezed scale, the reciprocal-vs-divisor convention
confusion that separates FP8 from NVFP4, or the wrong 128 tensors selected.

It also re-checks scope independently of the exporter, since selecting the
diarizer instead would corrupt every frame and is the failure mode worth two
separate guards.

Usage:
    python test_encoder_fp8_equivalence.py --source BF16_CKPT --exported FP8_CKPT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_encoder_fp8 import ASR_PREFIX, EXPECT, PATTERNS, target_keys  # noqa: E402

# Runs inside the eval container, not on the host: FP8Linear builds its weights
# through vllm._custom_ops, so vLLM has to be importable even though no GPU is
# touched. enc_nvfp4 lives in NeMo now rather than on PYTHONPATH.

FP8_MAX = 448.0


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="BF16 checkpoint")
    ap.add_argument("--exported", type=Path, required=True, help="checkpoint from export_encoder_fp8.py")
    ap.add_argument("--limit", type=int, default=0, help="only check the first N matrices")
    args = ap.parse_args()

    from nemo.collections.asr.parts.submodules.enc_nvfp4 import FP8Linear

    src = args.source / "perception.safetensors"
    exp = args.exported / "perception.safetensors"

    failures = 0
    with safe_open(src, framework="pt", device="cpu") as fs, safe_open(
        exp, framework="pt", device="cpu"
    ) as fe:
        src_keys = list(fs.keys())
        exp_keys = set(fe.keys())
        targets = target_keys(src_keys)

        print(f"scope: {len(targets)} ASR-encoder matrices selected")
        if len(targets) != EXPECT:
            fail(f"expected {EXPECT} targets, got {len(targets)}")
            failures += 1

        # Independent scope check: nothing under the diarizer may have gained a scale.
        diar_scales = [
            k for k in exp_keys if "diarization_model" in k and k.endswith(".weight_scale")
        ]
        if diar_scales:
            fail(f"{len(diar_scales)} diarizer tensors were quantized, e.g. {diar_scales[0]}")
            failures += 1
        else:
            print("scope: diarizer carries no weight_scale, correctly untouched")

        checked = targets[: args.limit] if args.limit else targets
        print(f"comparing {len(checked)} matrices bitwise ...")
        for key in checked:
            scale_key = key[: -len(".weight")] + ".weight_scale"
            if scale_key not in exp_keys:
                fail(f"{scale_key} missing from export")
                failures += 1
                continue

            bf16 = fs.get_tensor(key)
            got_w = fe.get_tensor(key)
            got_s = fe.get_tensor(scale_key)

            lin = nn.Linear(bf16.shape[1], bf16.shape[0], bias=False)
            with torch.no_grad():
                lin.weight.copy_(bf16)
            lin.weight.data = lin.weight.data.to(bf16.dtype)
            ref = FP8Linear(lin)
            ref_w = ref.weight_fp8
            ref_s = ref.weight_scale.squeeze(-1)

            if got_w.dtype != ref_w.dtype:
                fail(f"{key}: dtype {got_w.dtype} != {ref_w.dtype}")
                failures += 1
                continue
            if got_w.shape != ref_w.shape or got_s.shape != ref_s.shape:
                fail(f"{key}: shape {tuple(got_w.shape)}/{tuple(got_s.shape)} != "
                     f"{tuple(ref_w.shape)}/{tuple(ref_s.shape)}")
                failures += 1
                continue
            # Compare fp8 bit patterns as integers; fp8 has no reliable ==.
            if not torch.equal(got_w.view(torch.uint8), ref_w.view(torch.uint8)):
                n = int((got_w.view(torch.uint8) != ref_w.view(torch.uint8)).sum())
                fail(f"{key}: {n}/{got_w.numel()} weight elements differ")
                failures += 1
                continue
            if not torch.equal(got_s.float(), ref_s.float()):
                d = (got_s.float() - ref_s.float()).abs().max().item()
                fail(f"{key}: weight_scale differs, max |delta| {d:.3e}")
                failures += 1
                continue

            # Guard the convention itself: dequantized weight must approximate the
            # original, which catches a reciprocal scale that would still be
            # self-consistent between the two paths.
            deq = got_w.float() * got_s.float().unsqueeze(-1)
            denom = bf16.float().abs().amax().clamp_min(1e-12)
            rel = (deq - bf16.float()).abs().amax() / denom
            if rel > 0.15:
                fail(f"{key}: dequant rel error {rel:.3f} too large; scale convention suspect")
                failures += 1

    print()
    if failures:
        print(f"FAILED: {failures} problem(s)")
        return 1
    print(f"PASSED: {len(checked)} matrices match FP8Linear bitwise, diarizer untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

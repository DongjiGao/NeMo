#!/usr/bin/env python3
"""Strip ModelOpt's unresolved-name sentinel from a checkpoint's exclude lists.

ModelOpt main's HF export can emit module names with a placeholder appended:

    lm_head.\x00backbone.pt_name_sentinel

instead of the resolved ``lm_head``. The sentinel is an internal marker for a
name it failed to resolve during export, and it leaks into both places a
consumer looks for the "do not quantize" list:

    config.json           -> quantization_config.ignore
    hf_quant_config.json  -> quantization.exclude_modules

Left alone, nothing in those lists matches a real module, so vLLM tries to build
quantized layers for modules whose on-disk weights are still BF16 -- the
load_qkv_weight / load_merged_column_weight shape assert we hit before with
mis-prefixed exclude names.

This only rewrites metadata; weights are untouched. Every repaired name is
checked against the weight index: it must name an existing tensor that has no
matching weight_scale, i.e. a module the exporter really did leave alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SENTINEL = ".\x00backbone.pt_name_sentinel"


def strip(name: str) -> str:
    return name[: -len(SENTINEL)] if name.endswith(SENTINEL) else name


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def save(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint
    index = load(ckpt / "model.safetensors.index.json")["weight_map"]

    # (file, path-to-list) for each place the exclusions are duplicated.
    targets = [
        ("config.json", ("quantization_config", "ignore")),
        ("hf_quant_config.json", ("quantization", "exclude_modules")),
    ]

    total_fixed = 0
    for filename, keys in targets:
        path = ckpt / filename
        if not path.is_file():
            print(f"{filename}: absent, skipping")
            continue

        doc = load(path)
        node = doc
        for k in keys[:-1]:
            node = node.get(k) or {}
        names = node.get(keys[-1])
        if not names:
            print(f"{filename}: no {'.'.join(keys)}, skipping")
            continue

        repaired = [strip(n) for n in names]
        n_fixed = sum(1 for a, b in zip(names, repaired) if a != b)

        bad = []
        for name in repaired:
            weight = f"{name}.weight"
            if weight not in index:
                bad.append(f"{name}: no such tensor")
            elif f"{name}.weight_scale" in index:
                bad.append(f"{name}: has weight_scale, so it was quantized")
        if bad:
            raise SystemExit(
                f"{filename}: refusing to write, {len(bad)} name(s) do not check out:\n  "
                + "\n  ".join(bad[:10])
            )

        print(f"{filename}: {n_fixed}/{len(names)} names repaired, all {len(repaired)} verified")
        total_fixed += n_fixed
        if args.dry_run:
            continue

        backup = path.with_suffix(path.suffix + ".sentinel-bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        node[keys[-1]] = repaired
        save(path, doc)

    if args.dry_run:
        print(f"\ndry run: {total_fixed} names would be repaired")
    else:
        print(f"\nrepaired {total_fixed} names; originals kept as *.sentinel-bak")


if __name__ == "__main__":
    main()

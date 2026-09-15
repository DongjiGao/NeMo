#!/usr/bin/env python3
"""Create a ModelOpt static-FP8 SpeechLM checkpoint for vLLM plugin loading.

This exports the NemotronH language backbone with ModelOpt FP8 metadata, then
adds the original SpeechLM perception weights. The LLM weights are intentionally
kept in HuggingFace/vLLM names (for example ``backbone.*``), not re-prefixed
with ``llm.*``; the SpeechLM vLLM plugin's HybridBackend can pass those names
through to the inner NemotronH model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = Path(
    "/data/dongjig/results/speechlm-2026h1/"
    "2404-lr1e-4-omnidata-multiling-4node/eval-step-40000"
)
DEFAULT_OUTPUT = Path(
    "/data/dongjig/results/quantization/"
    "speechlm_nemotronh_modelopt_fp8_eval-step-40000"
)
DEFAULT_MODELOPT_REPO = Path("/home/dongjig/Model-Optimizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm-name",
        default=None,
        help="HF LLM backbone id. Defaults to config.json pretrained_llm.",
    )
    parser.add_argument("--hf-home", type=Path, default=Path("/data/dongjig/.cache/huggingface"))
    parser.add_argument("--modelopt-repo", type=Path, default=DEFAULT_MODELOPT_REPO)
    parser.add_argument("--llm-export-dir", type=Path, default=None)
    parser.add_argument("--skip-llm-export", action="store_true")
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--calib-seq", type=int, default=512)
    parser.add_argument("--kv-cache-qformat", default="fp8")
    parser.add_argument("--gpu-max-mem-percentage", type=float, default=0.8)
    parser.add_argument("--use-seq-device-map", action="store_true")
    parser.add_argument("--calib-manifest", type=Path, default=None)
    parser.add_argument("--calib-limit", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def calibration_texts(manifest: Path | None, limit: int) -> list[str]:
    if manifest is not None:
        texts: list[str] = []
        with manifest.open() as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text") or item.get("norm_text") or item.get("expected_answer")
                if text:
                    texts.append(text)
                if len(texts) >= limit:
                    break
        if texts:
            return texts

    base = [
        "Transcribe the following audio accurately and return only the transcription.",
        "The committee discussed the proposal and agreed to revisit it next week.",
        "Researchers measured a clear improvement in latency without changing accuracy.",
        "The model should preserve punctuation independent word error rate after normalization.",
        "Audio understanding requires combining acoustic features with language model context.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "During the meeting the team reviewed benchmark results and open questions.",
        "Speech recognition quality and inference throughput both matter for deployment.",
    ]
    repeats = max(1, (limit + len(base) - 1) // len(base))
    return (base * repeats)[:limit]


def run_official_hf_ptq(args: argparse.Namespace, llm_name: str, export_dir: Path) -> None:
    """Run NVIDIA Model-Optimizer's official HF PTQ script for LLM export."""
    # Upstream renamed examples/llm_ptq -> examples/hf_ptq in f335459dc (2026-06-27).
    # The old name can survive locally as a stale directory holding only __pycache__,
    # because git removes the tracked files but leaves untracked ones behind, so
    # probing for the directory is not enough -- probe for the script itself.
    candidates = [
        args.modelopt_repo / "examples/hf_ptq/hf_ptq.py",
        args.modelopt_repo / "examples/llm_ptq/hf_ptq.py",
    ]
    hf_ptq = next((p for p in candidates if p.exists()), None)
    if hf_ptq is None:
        raise FileNotFoundError(
            f"Missing hf_ptq.py under {args.modelopt_repo}/examples/{{hf_ptq,llm_ptq}}. "
            "Clone https://github.com/NVIDIA/Model-Optimizer there first."
        )

    cmd = [
        sys.executable,
        str(hf_ptq),
        "--pyt_ckpt_path",
        llm_name,
        "--qformat",
        "fp8",
        "--kv_cache_qformat",
        args.kv_cache_qformat,
        "--export_path",
        str(export_dir),
        "--trust_remote_code",
        "--calib_size",
        str(args.calib_size),
        "--calib_seq",
        str(args.calib_seq),
        "--batch_size",
        str(args.calib_batch_size),
        "--skip_generate",
    ]
    if args.use_seq_device_map:
        cmd.append("--use_seq_device_map")
    if args.gpu_max_mem_percentage is not None:
        cmd.extend(["--gpu_max_mem_percentage", str(args.gpu_max_mem_percentage)])

    print("=== Step 1: Run official ModelOpt HF PTQ export ===")
    print("Command:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home)
    # The checkout's modelopt must shadow any installed one. example_utils.py here
    # imports modelopt.torch.utils.image_processor, which the quantdev container's
    # installed 0.44 does not have; without this the run dies at import. cwd is the
    # example directory, so the repo root has to be named explicitly.
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{args.modelopt_repo}{os.pathsep}{existing}" if existing else str(args.modelopt_repo)
    )
    subprocess.run(cmd, cwd=hf_ptq.parent, env=env, check=True)

    for path in sorted(export_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


def copy_sidecar_files(source: Path, output: Path) -> None:
    skip = {"model.safetensors", "model.safetensors.index.json", "config.json"}
    for path in source.iterdir():
        if path.name in skip:
            continue
        target = output / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def save_perception_weights(source: Path, output: Path) -> dict[str, str]:
    model_path = source / "model.safetensors"
    perception: dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("perception.") or key.startswith("projection."):
                perception[key] = f.get_tensor(key)

    if not perception:
        raise RuntimeError(f"No perception weights found in {model_path}")

    perception_path = output / "perception.safetensors"
    save_file(perception, perception_path)
    print(f"Perception weights: {len(perception)} tensors -> {perception_path}")
    return {key: perception_path.name for key in perception}


def copy_llm_export(llm_export_dir: Path, output: Path) -> dict:
    for path in llm_export_dir.iterdir():
        if path.suffix == ".safetensors":
            shutil.copy2(path, output / path.name)
        elif path.name in {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "hf_quant_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        }:
            shutil.copy2(path, output / path.name)
        elif path.name.startswith(("configuration", "modeling")) and path.suffix == ".py":
            shutil.copy2(path, output / path.name)

    llm_config = load_json(llm_export_dir / "config.json")
    return llm_config


def merge_index(output: Path, perception_map: dict[str, str]) -> None:
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
    else:
        index = {"metadata": {}, "weight_map": {}}
        for path in output.glob("*.safetensors"):
            if path.name == "perception.safetensors":
                continue
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    index["weight_map"][key] = path.name

    index.setdefault("weight_map", {}).update(perception_map)
    index.setdefault("metadata", {})["total_size"] = sum(
        p.stat().st_size for p in output.glob("*.safetensors")
    )
    write_json(index_path, index)


def write_speechlm_config(source: Path, output: Path, llm_config: dict) -> None:
    cfg = load_json(source / "config.json")
    qcfg = llm_config.get("quantization_config")
    if not qcfg:
        raise RuntimeError("ModelOpt export did not include quantization_config in LLM config.json")

    cfg["quantization_config"] = qcfg
    # The SpeechLM plugin wraps the LLM under `language_model` and vLLM's NemotronH
    # renames backbone->model, so MIXED_PRECISION `quantized_layers` must be keyed by
    # the mapped names (`language_model.model.*`) or vLLM can't match a module and
    # mis-builds it (e.g. static-FP8 in_proj loaded as dynamic -> KeyError on
    # `*.input_scale`). config_groups can stay `backbone.*` (matches working ckpts).
    ql = qcfg.get("quantized_layers")
    if isinstance(ql, dict):
        qcfg["quantized_layers"] = {
            (k.replace("backbone.", "language_model.model.", 1) if k.startswith("backbone.") else k): v
            for k, v in ql.items()
        }
    if (output / "hf_quant_config.json").exists():
        hf_quant = load_json(output / "hf_quant_config.json")
        if hf_quant.get("producer", {}).get("name") == "modelopt":
            # Match the published ModelOpt checkpoint style: keep both config.json
            # quantization_config and hf_quant_config.json available to vLLM.
            cfg["quantization_config"].setdefault("producer", hf_quant.get("producer", {}))
    cfg["model_type"] = "nemo_speechlm"
    cfg["architectures"] = ["NeMoSpeechLMForConditionalGeneration"]
    write_json(output / "config.json", cfg)


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = str(args.hf_home)

    source = args.source.resolve()
    output = args.output.resolve()
    llm_export_dir = (
        args.llm_export_dir.resolve()
        if args.llm_export_dir is not None
        else output.with_name(output.name + "_llm_only").resolve()
    )

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    if llm_export_dir.exists() and not (args.overwrite or args.skip_llm_export or args.llm_export_dir is not None):
        raise FileExistsError(f"{llm_export_dir} exists; pass --overwrite to replace it")

    if args.overwrite:
        shutil.rmtree(output, ignore_errors=True)
        if not args.skip_llm_export and args.llm_export_dir is None:
            shutil.rmtree(llm_export_dir, ignore_errors=True)

    source_cfg = load_json(source / "config.json")
    llm_name = args.llm_name or source_cfg["pretrained_llm"]

    print(f"Source SpeechLM checkpoint: {source}")
    print(f"Output SpeechLM checkpoint: {output}")
    print(f"LLM backbone: {llm_name}")
    print(f"LLM export dir: {llm_export_dir}")
    print(f"ModelOpt repo: {args.modelopt_repo}")

    if args.skip_llm_export:
        print("=== Step 1: Skip LLM export; using existing LLM export dir ===")
        if not llm_export_dir.exists():
            raise FileNotFoundError(llm_export_dir)
    else:
        run_official_hf_ptq(args, llm_name, llm_export_dir)

    print("\n=== Step 4: Assemble SpeechLM checkpoint ===")
    output.mkdir(parents=True, exist_ok=True)
    copy_sidecar_files(source, output)
    llm_config = copy_llm_export(llm_export_dir, output)
    perception_map = save_perception_weights(source, output)
    merge_index(output, perception_map)
    write_speechlm_config(source, output, llm_config)

    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    original = (source / "model.safetensors").stat().st_size
    summary = {
        "source": str(source),
        "output": str(output),
        "llm_name": llm_name,
        "llm_export_dir": str(llm_export_dir),
        "calib_size": args.calib_size,
        "calib_batch_size": args.calib_batch_size,
        "calib_seq": args.calib_seq,
        "kv_cache_qformat": args.kv_cache_qformat,
        "modelopt_repo": str(args.modelopt_repo),
        "total_size_gb": total / 1e9,
        "original_model_safetensors_gb": original / 1e9,
        "compression_vs_original_model_file": total / original,
    }
    write_json(output / "modelopt_fp8_export_summary.json", summary)
    print("\n=== Done ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

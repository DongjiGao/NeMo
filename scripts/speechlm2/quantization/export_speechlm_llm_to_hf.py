#!/usr/bin/env python3
"""Export the fine-tuned SpeechLM decoder as a standalone HF NemotronH checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    # No path defaults on purpose. These used to point at one specific checkpoint
    # on one machine, so omitting --source did not fail, it silently exported a
    # different and much older model; the --sidecar-source default outlived the
    # directory it named and could not have worked at all.
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="SpeechLM checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="HF LLM checkpoint to write")
    parser.add_argument(
        "--sidecar-source",
        type=Path,
        required=True,
        help="directory holding the NemotronH config.json (usually "
        "<source>/llm_backbone) plus tokenizer.json from the SpeechLM root",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="defaults to the ambient HF_HOME, then ~/.cache/huggingface",
    )
    parser.add_argument("--max-shard-size-gb", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def copy_hf_sidecars(base_model: str, output: Path, sidecar_source: Path) -> None:
    # Prefer the complete local sidecar export so gated Nemotron checkpoints can
    # be exported reproducibly without Hugging Face authentication. The sidecar
    # came from a quantized checkpoint, so strip its quantization metadata before
    # using the architecture config for this BF16 decoder export.
    if (sidecar_source / "config.json").is_file() and (sidecar_source / "tokenizer.json").is_file():
        config = load_json(sidecar_source / "config.json")
        config.pop("quantization_config", None)
        write_json(output / "config.json", config)

        for path in sidecar_source.iterdir():
            if path.name in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "generation_config.json",
            } or (path.name.startswith(("configuration", "modeling")) and path.suffix == ".py"):
                shutil.copy2(path, output / path.name)
        return

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    cfg.save_pretrained(output)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(output)

    for path in sidecar_source.iterdir():
        if path.name in {
            "config.json",
            "hf_quant_config.json",
            "model.safetensors.index.json",
            "generation_config.json",
        }:
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name.startswith(("configuration", "modeling")) and path.suffix == ".py":
            shutil.copy2(path, output / path.name)

    generation_config = sidecar_source / "generation_config.json"
    if generation_config.exists():
        shutil.copy2(generation_config, output / "generation_config.json")


def mapped_llm_weights(source_model: Path):
    with safe_open(source_model, framework="pt", device="cpu") as f:
        for name in f.keys():
            if not name.startswith("llm."):
                continue
            # Transformer Engine bookkeeping, not model parameters. These are
            # zero-length, and carrying them through makes them unexpected keys
            # for HF loading and for ModelOpt's module traversal.
            if name.endswith("_extra_state"):
                continue
            # Multi-token-prediction heads. Inference runs without speculative
            # decoding, and NemotronH does not declare them, so they would load
            # as unexpected keys.
            if name.startswith("llm.mtp."):
                continue

            tensor = f.get_tensor(name)
            hf_name = name.replace("llm.model.", "backbone.")
            hf_name = hf_name.replace("llm.lm_head", "lm_head")
            # Newer NeMo keeps Mamba's A_log/D/dt_bias in an fp32 side-container.
            # NemotronH expects them directly on the mixer; leaving the indirection
            # in place drops them silently and the SSM initialises at random.
            hf_name = hf_name.replace("._fp32_params.", ".")
            if hf_name == "backbone.embed_tokens.weight":
                hf_name = "backbone.embeddings.weight"
            elif hf_name == "backbone.norm.weight":
                hf_name = "backbone.norm_f.weight"

            if hf_name.endswith(".experts.down_projs"):
                prefix = hf_name.replace(".experts.down_projs", "")
                for expert_idx in range(tensor.shape[0]):
                    yield f"{prefix}.experts.{expert_idx}.down_proj.weight", tensor[expert_idx].t().contiguous()
            elif hf_name.endswith(".experts.gate_and_up_projs"):
                prefix = hf_name.replace(".experts.gate_and_up_projs", "")
                for expert_idx in range(tensor.shape[0]):
                    yield f"{prefix}.experts.{expert_idx}.up_proj.weight", tensor[expert_idx].t().contiguous()
            else:
                yield hf_name, tensor


def tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_sharded(weights, output: Path, max_shard_size: int) -> dict:
    tmp_files: list[tuple[Path, list[str]]] = []
    weight_map: dict[str, str] = {}
    shard: dict = {}
    shard_bytes = 0
    total_size = 0

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        tmp_path = output / f"model-{len(tmp_files) + 1:05d}.safetensors"
        save_file(shard, tmp_path)
        tmp_files.append((tmp_path, list(shard)))
        print(f"wrote {tmp_path.name}: {len(shard)} tensors, {shard_bytes / 1e9:.2f} GB")
        shard = {}
        shard_bytes = 0

    for name, tensor in weights:
        nbytes = tensor_nbytes(tensor)
        if shard and shard_bytes + nbytes > max_shard_size:
            flush()
        shard[name] = tensor
        shard_bytes += nbytes
        total_size += nbytes
    flush()

    num_shards = len(tmp_files)
    for idx, (tmp_path, keys) in enumerate(tmp_files, start=1):
        final_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        final_path = output / final_name
        tmp_path.replace(final_path)
        for key in keys:
            weight_map[key] = final_name

    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def main() -> None:
    args = parse_args()
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)

    source = args.source.resolve()
    output = args.output.resolve()
    source_model = source / "model.safetensors"
    source_cfg = load_json(source / "config.json")
    base_model = source_cfg["pretrained_llm"]

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    print(f"Source SpeechLM checkpoint: {source}")
    print(f"Output HF LLM checkpoint: {output}")
    print(f"Base NemotronH config/tokenizer: {base_model}")
    copy_hf_sidecars(base_model, output, args.sidecar_source.resolve())

    max_shard_size = int(args.max_shard_size_gb * 1e9)
    index = save_sharded(mapped_llm_weights(source_model), output, max_shard_size)
    write_json(output / "model.safetensors.index.json", index)

    summary = {
        "source": str(source),
        "output": str(output),
        "base_model": base_model,
        "num_tensors": len(index["weight_map"]),
        "total_size_gb": index["metadata"]["total_size"] / 1e9,
        "max_shard_size_gb": args.max_shard_size_gb,
    }
    write_json(output / "speechlm_llm_hf_export_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

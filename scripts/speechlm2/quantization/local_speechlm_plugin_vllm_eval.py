#!/usr/bin/env python3
"""Local vLLM plugin benchmark/eval for NeMo SpeechLM checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import soundfile as sf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _register_nemo_plugin() -> None:
    try:
        from nemo.collections.speechlm2.vllm.salm import register

        register()
    except Exception as exc:
        raise RuntimeError(
            "Failed to register NeMo SpeechLM vLLM plugin. It is registered via "
            "the nemo_speechlm entry point "
            "(nemo.collections.speechlm2.vllm.salm:register), so NeMo must be "
            "importable in the same environment as vLLM."
        ) from exc


def get_gpu_memory_mb() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    return int(result.stdout.strip().splitlines()[0])


class GpuMemorySampler:
    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.samples_mb: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples_mb.append(get_gpu_memory_mb())
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    @property
    def peak_mb(self) -> int:
        return max(self.samples_mb) if self.samples_mb else 0


def apply_path_rewrites(path: str, rewrites: list[tuple[str, str]]) -> str:
    for old, new in rewrites:
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    rewrites = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Path rewrite must be OLD=NEW; got {value!r}")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError(f"Path rewrite OLD prefix cannot be empty: {value!r}")
        rewrites.append((old, new))
    return rewrites


def load_manifest(
    manifest: Path,
    limit: int | None,
    path_rewrites: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], float]:
    path_rewrites = path_rewrites or []
    items: list[dict] = []
    total_audio_sec = 0.0
    with manifest.open() as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text") or item.get("norm_text") or item.get("expected_answer")
            if not text:
                raise ValueError(f"Manifest item missing text/norm_text: {item}")

            if "audio_filepath" not in item:
                try:
                    audio = next(message["audio"] for message in item["messages"] if "audio" in message)
                    item["audio_filepath"] = audio["path"]
                    item["duration"] = audio["duration"]
                except (KeyError, StopIteration, TypeError) as e:
                    raise ValueError(f"Manifest item missing audio_filepath or messages[*].audio: {item}") from e

            item["text"] = text
            item["audio_filepath"] = apply_path_rewrites(item["audio_filepath"], path_rewrites)
            item["duration"] = float(item["duration"])
            item["dataset"] = item.get("subset_for_metrics") or item.get("dataset") or "all"
            items.append(item)
            total_audio_sec += item["duration"]
            if limit is not None and len(items) >= limit:
                break
    return items, total_audio_sec


SCORING_CONTRACT = (
    "OpenASREnglishTextNormalizer + kaldialign.batch_error_rate(merge_compounds=True)"
)

_FROZEN_SCORER: tuple | None = None


def load_scorer() -> tuple:
    """Load Piotr's frozen scoring contract, the only one HR8 numbers are quoted in.

    The scoring itself lives in `score_frozen.py` and is imported rather than
    reimplemented, so the inline pass/subset numbers and the offline rescore
    cannot drift. An earlier inline scorer here (whisper_normalizer plus a plain
    token edit distance) read the same predictions about `0.6` micro WER higher
    -- `+1.45` on gigaspeech, `+0.45` on the clean subsets, almost all of it
    compound splits like "to day"/"today" that `merge_compounds` forgives -- so
    a number was meaningless without knowing which scorer produced it.

    Raises rather than falling back: a silent fallback is what made the two
    scales ambiguous, and a scoring failure discovered after a full eval costs
    the whole GPU run.
    """
    global _FROZEN_SCORER
    if _FROZEN_SCORER is None:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from score_frozen import _load_frozen_normalizer, score_rows

        # Imported here purely so a stale kaldialign (< 0.10 has no
        # batch_error_rate) fails now rather than after the last pass.
        from kaldialign import batch_error_rate  # noqa: F401

        normalizer, source = _load_frozen_normalizer()
        _FROZEN_SCORER = (normalizer, score_rows, source)
    return _FROZEN_SCORER


def normalize_text(text: str) -> str:
    normalizer, _, _ = load_scorer()
    return normalizer(text)


def corpus_wer(hypotheses: list[str], references: list[str]) -> float:
    """Micro WER as a fraction: total errors over total reference words.

    Micro, not a mean of per-utterance rates -- averaging per-row WER does not
    reproduce Piotr's saved fields.
    """
    normalizer, score_rows, _ = load_scorer()
    rows = [
        {"reference": ref, "hypothesis": hyp, "dataset": "all"}
        for hyp, ref in zip(hypotheses, references)
    ]
    scored = score_rows(rows, normalizer)
    if not scored:
        return 0.0
    bucket = scored["all"]
    return bucket["errors"] / bucket["ref_words"] if bucket["ref_words"] else 0.0


def per_dataset_wer(
    hypotheses: list[str], items: list[dict]
) -> tuple[dict[str, float], float, dict]:
    """Per-dataset micro WER (%) grouped by item['dataset'], the equal-dataset
    average (the canonical Open ASR Leaderboard metric), and the raw per-subset
    buckets so the caller can report runaway rows.

    Rows are labelled with the manifest id, falling back to position, because a
    missing id would make the offline paired ex-runaway comparison unable to key
    its exclusion set.
    """
    normalizer, score_rows, _ = load_scorer()
    rows = [
        {
            "reference": item["text"],
            "hypothesis": hyp,
            "dataset": item.get("dataset", "all"),
            "id": item.get("id") or f"row{idx}",
        }
        for idx, (hyp, item) in enumerate(zip(hypotheses, items))
    ]
    scored = score_rows(rows, normalizer)
    per = {ds: bucket["wer"] for ds, bucket in scored.items()}
    avg = sum(per.values()) / len(per) if per else 0.0
    return per, avg, scored


def build_prompt(
    tokenizer,
    audio_placeholder: str,
    user_prompt: str,
    system_prompt: str | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # The `audio` key is what makes --prompt-contract apply: render_messages only
    # honours locator placement for messages carrying audio, so without it this
    # path silently stayed on the legacy '<|audio|>\n<text>' layout even under
    # --prompt-contract salm.
    messages.append({"role": "user", "content": user_prompt, "audio": True})
    return render_messages(tokenizer, messages, audio_placeholder)


# "salm" reproduces the locator placement enforced by vLLM serving in
# nemo/collections/speechlm2/vllm/salm/runtime_compat.py: text, one ASCII space,
# then a final <|audio|>. Thinking stays suppressed in both contracts; leaving the
# template's bare <think> open makes the model emit reasoning instead of a transcript.
PROMPT_CONTRACT = {"audio_last": False, "suppress_thinking": True}


def render_messages(tokenizer, messages: list[dict], audio_placeholder: str) -> str:
    rendered_messages = []
    for message in messages:
        rendered = {key: value for key, value in message.items() if key != "audio"}
        if "audio" in message:
            content = rendered.get("content", "")
            if PROMPT_CONTRACT["audio_last"]:
                rendered["content"] = f"{content} {audio_placeholder}".strip()
            else:
                rendered["content"] = f"{audio_placeholder}\n{content}".strip()
        rendered_messages.append(rendered)

    if PROMPT_CONTRACT["suppress_thinking"]:
        try:
            return tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(rendered_messages, tokenize=False, add_generation_prompt=True)


def build_requests(
    items: list[dict],
    prompt: str,
    tokenizer=None,
    audio_placeholder: str = "<|audio|>",
    use_manifest_messages: bool = True,
) -> list[dict]:
    requests = []
    for item in items:
        audio, sample_rate = sf.read(item["audio_filepath"], dtype="float32")
        item_prompt = prompt
        # Manifest messages win by default, which is why --user-prompt has had no
        # effect on any manifest carrying them (ours does, system turn included).
        if use_manifest_messages and tokenizer is not None and "messages" in item:
            item_prompt = render_messages(tokenizer, item["messages"], audio_placeholder)
        requests.append(
            {
                "prompt": item_prompt,
                "multi_modal_data": {"audio": (audio, sample_rate)},
            }
        )
    return requests


def generate_pass(
    llm,
    items,
    prompt,
    tokenizer,
    sampling_params,
    chunk_size,
    prebuilt_requests=None,
    use_manifest_messages: bool = True,
):
    """Run one full pass over ``items``; return ``(elapsed_sec, hypotheses)``.

    Only ``llm.generate`` is timed (audio decode is excluded), matching the
    upfront path. With ``chunk_size`` set, requests are built and submitted in
    chunks so host RAM holds at most ``chunk_size`` decoded waveforms at a time;
    expect a small inter-chunk drain in RTFx since each chunk drains the engine.
    """
    if prebuilt_requests is not None:
        start = time.perf_counter()
        outputs = llm.generate(prebuilt_requests, sampling_params=sampling_params, use_tqdm=True)
        elapsed = time.perf_counter() - start
        return elapsed, [out.outputs[0].text for out in outputs]
    elapsed = 0.0
    hypotheses: list[str] = []
    for start_idx in range(0, len(items), chunk_size):
        chunk = items[start_idx : start_idx + chunk_size]
        chunk_requests = build_requests(
            chunk, prompt, tokenizer=tokenizer, use_manifest_messages=use_manifest_messages
        )
        start = time.perf_counter()
        outputs = llm.generate(chunk_requests, sampling_params=sampling_params, use_tqdm=True)
        elapsed += time.perf_counter() - start
        hypotheses.extend(out.outputs[0].text for out in outputs)
        del chunk_requests, outputs
    return elapsed, hypotheses


def make_llm(args: argparse.Namespace) -> LLM:
    _register_nemo_plugin()
    kwargs = dict(
        model=str(args.model_dir),
        tokenizer=str(args.tokenizer or args.model_dir),
        dtype=args.dtype,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 1},
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=not args.enable_log_stats,
    )
    if args.data_parallel_size > 1:
        kwargs["data_parallel_size"] = args.data_parallel_size
    if args.enable_expert_parallel:
        kwargs["enable_expert_parallel"] = True
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.moe_backend:
        kwargs["moe_backend"] = args.moe_backend
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.async_scheduling is not None:
        kwargs["async_scheduling"] = args.async_scheduling
    if args.enable_chunked_prefill is not None:
        kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.mm_processor_cache_gb is not None:
        kwargs["mm_processor_cache_gb"] = args.mm_processor_cache_gb
    if args.num_speculative_tokens is not None:
        # The draft model is the same checkpoint: the SALM plugin's
        # hf_config_override rewrites it to the NeMoSpeechLMMTPModel
        # architecture and remaps llm.mtp.layers.* onto vLLM's aliases. The
        # checkpoint's num_nextn_predict_layers=2 is the sublayer count for one
        # prediction step (attention + MoE, pattern "*E"), not the draft depth;
        # vLLM reuses that single physical step num_speculative_tokens times.
        kwargs["speculative_config"] = {
            "model": str(args.speculative_model or args.model_dir),
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    compilation_config: dict = {}
    if args.disable_cudagraph:
        compilation_config["cudagraph_mode"] = 0
    if args.max_cudagraph_capture_size is not None:
        compilation_config["max_cudagraph_capture_size"] = args.max_cudagraph_capture_size
    if compilation_config:
        kwargs["compilation_config"] = compilation_config
    if args.enable_flashinfer_autotune and args.disable_flashinfer_autotune:
        raise ValueError("Use only one of --enable-flashinfer-autotune or --disable-flashinfer-autotune")
    if args.enable_flashinfer_autotune or args.disable_flashinfer_autotune:
        kwargs["kernel_config"] = {
            "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
        }
    return LLM(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--warmup-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup-limit", type=int, default=None)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, shuffle eval items with this seed for steadier batch packing.",
    )
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--submit-chunk-size",
        type=int,
        default=None,
        help="If set, decode+submit audio in chunks of this many utterances per "
        "llm.generate call to bound host RAM (default None = all decoded upfront, "
        "one call). Only generate time is summed into RTFx; expect a small "
        "inter-chunk drain since each chunk drains the engine.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--memory-sample-interval", type=float, default=0.2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None, help="Optional vLLM quantization mode, e.g. fp8")
    parser.add_argument("--kv-cache-dtype", default=None, help="Optional vLLM KV cache dtype, e.g. fp8 / fp8_e4m3 (default: auto = model dtype).")
    parser.add_argument("--moe-backend", default=None, help="Optional vLLM MoE backend override, e.g. triton")
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Replicate the engine across this many GPUs. Throughput from a DP>1 run is "
        "NOT comparable to a DP1 run, and WER can move slightly because DP changes which "
        "requests batch together and vLLM is not batch-invariant. Re-measure any baseline "
        "at the same DP before comparing.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Shard MoE experts across the DP ranks instead of replicating them. Only "
        "meaningful with --data-parallel-size > 1.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=None,
        help="Enable MTP speculative decoding with this draft depth K. Under greedy "
        "decoding output must be identical to K=None, so a WER change indicates a bug. "
        "Compare RTFx on a warm pass only: enabling this changes what gets compiled "
        "and CUDA-graph captured, so cold-start cost differs between arms.",
    )
    parser.add_argument(
        "--speculative-model",
        default=None,
        help="Draft model path (default: --model-dir, since MTP heads ship in the checkpoint).",
    )
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=0.0,
        help="Size of vLLM's multimodal preprocessor cache. DEFAULT 0 (disabled) on "
        "purpose: it is keyed by a hash of the raw audio, and a repeated pass over the "
        "same manifest would hit it and skip mel extraction, inflating RTFx by an amount "
        "no deployment ever sees. vLLM's own default is 4 GiB. Raise it only when "
        "deliberately measuring cache behaviour, never for a throughput number.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spaces-between-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cudagraph", action="store_true", help="Keep torch.compile but disable CUDA Graph.")
    parser.add_argument(
        "--max-cudagraph-capture-size",
        type=int,
        default=None,
        help="Override vLLM's max CUDA graph capture size. Default caps at "
        "min(max_num_seqs*2, 512); set e.g. 2048 to also capture wide decode "
        "batches up to max_num_seqs.",
    )
    parser.add_argument(
        "--enable-flashinfer-autotune",
        action="store_true",
        help="Enable FlashInfer GEMM/MoE tactic autotuning during vLLM warmup.",
    )
    parser.add_argument(
        "--disable-flashinfer-autotune",
        action="store_true",
        help="Disable FlashInfer GEMM/MoE tactic autotuning even if the vLLM version enables it by default.",
    )
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-mm-cache-after-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset-mm-cache-each-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reset the mm processor cache before each measured pass, emulating real "
        "unique-audio serving (no cross-pass cache reuse).",
    )
    parser.add_argument("--user-prompt", default="Transcribe the following:")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System turn to prepend when --no-manifest-messages is set. The manifest's "
        "own system turn is 'You are a helpful assistant. /no_think'; dropping the "
        "/no_think directive can make the model emit reasoning instead of a transcript.",
    )
    parser.add_argument(
        "--manifest-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use each manifest item's own `messages` (default, and what every HR8 "
        "result so far used). --no-manifest-messages instead applies --user-prompt "
        "and --system-prompt uniformly, which is how prompt variants are tested.",
    )
    parser.add_argument(
        "--prompt-contract",
        choices=("legacy", "salm"),
        default="legacy",
        help="Audio locator placement. legacy: '<|audio|>\\n<text>'. "
        "salm: '<text> <|audio|>', matching what vLLM serving renders for NeMo SpeechLM.",
    )
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="If set, write per-item {reference,hypothesis,dataset} JSONL in manifest "
        "order (last measured pass) for external scoring, e.g. score_oal.py.",
    )
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        help="Rewrite audio paths while loading manifests. Format: OLD=NEW. Can be repeated.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call CUDA profiler start/stop around generation for nsys --capture-range=cudaProfilerApi.",
    )
    parser.add_argument(
        "--cuda-profiler-stop-sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep after cudaProfilerStop() to let Nsight flush short capture ranges.",
    )
    parser.add_argument(
        "--cuda-profiler-no-stop-sync",
        action="store_true",
        help="Skip torch.cuda.synchronize() after cudaProfilerStop(); useful when memory is tight.",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable-log-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable vLLM engine stat logging (Running/Waiting/KV over time) for efficiency diagnostics.",
    )
    args = parser.parse_args()

    PROMPT_CONTRACT["audio_last"] = args.prompt_contract == "salm"

    # Load the scorer before the engine, so a missing normalizer or a stale
    # kaldialign costs a second instead of a completed eval.
    _, _, scorer_source = load_scorer()
    from score_frozen import TAIL_RULE

    print(f"Scoring contract: {SCORING_CONTRACT}\n  normalizer: {scorer_source}")

    # Leave HF_HOME to the caller. This used to setdefault a path from one
    # machine, which on any other box silently pointed the cache at a directory
    # that does not exist.
    path_rewrites = parse_path_rewrites(args.path_rewrite)
    manifest_start = time.perf_counter()
    items, total_audio_sec = load_manifest(args.manifest, args.limit, path_rewrites)
    if args.shuffle_seed is not None:
        import random

        # Shuffle so the continuous batch sees mixed durations instead of the
        # dataset/duration-clustered manifest order (steadier batch packing).
        # references are rebuilt from items below, so order stays consistent.
        random.Random(args.shuffle_seed).shuffle(items)
    warmup_items: list[dict] = []
    warmup_audio_sec = 0.0
    if args.warmup_manifest is not None:
        warmup_items, warmup_audio_sec = load_manifest(args.warmup_manifest, args.warmup_limit, path_rewrites)
    manifest_load_sec = time.perf_counter() - manifest_start
    references = [item["text"] for item in items]
    print(
        f"Loaded {len(items)} utterances, {total_audio_sec:.1f}s audio "
        f"in {manifest_load_sec:.3f}s"
    )
    if warmup_items:
        print(f"Loaded {len(warmup_items)} warmup utterances, {warmup_audio_sec:.1f}s audio")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer or args.model_dir),
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    prompt = build_prompt(tokenizer, "<|audio|>", args.user_prompt, args.system_prompt)
    if args.manifest_messages:
        rendered_prompt = render_messages(tokenizer, items[0]["messages"], "<|audio|>")
        print("Prompt source: manifest `messages` (per item)")
    else:
        rendered_prompt = prompt
        print("Prompt source: --user-prompt / --system-prompt (uniform)")
    print(f"Rendered prompt: {rendered_prompt!r}")
    # The template always emits a thinking block; only an *unclosed* one is the
    # failure mode, and enable_thinking=False normally closes it even without the
    # manifest's `/no_think` system directive.
    if rendered_prompt.count("<think>") > rendered_prompt.count("</think>"):
        print(
            "WARNING: the rendered prompt leaves a <think> block open. The model will "
            "emit reasoning instead of a transcript."
        )

    memory_before_mb = get_gpu_memory_mb()
    sampler = GpuMemorySampler(args.memory_sample_interval)
    sampler.start()
    per_dataset_map: dict = {}
    per_dataset_runaways: dict = {}
    runaway_ids: list = []
    equal_avg_wer = None
    llm = None
    try:
        llm_start = time.perf_counter()
        llm = make_llm(args)
        llm_init_sec = time.perf_counter() - llm_start
        memory_after_load_mb = get_gpu_memory_mb()
        request_start = time.perf_counter()
        requests = (
            build_requests(
                items, prompt, tokenizer=tokenizer, use_manifest_messages=args.manifest_messages
            )
            if args.submit_chunk_size is None
            else None
        )
        warmup_requests = (
            build_requests(
                warmup_items,
                prompt,
                tokenizer=tokenizer,
                use_manifest_messages=args.manifest_messages,
            )
            if warmup_items
            else []
        )
        request_build_sec = time.perf_counter() - request_start
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=args.top_p,
            seed=args.seed,
            skip_special_tokens=args.skip_special_tokens,
            spaces_between_special_tokens=args.spaces_between_special_tokens,
        )
        print(
            f"Setup timing: llm_init={llm_init_sec:.3f}s "
            f"request_build={request_build_sec:.3f}s"
        )

        warmup_results = []
        if warmup_requests:
            for warmup_idx in range(args.warmup_passes):
                start = time.perf_counter()
                outputs = llm.generate(warmup_requests, sampling_params=sampling_params, use_tqdm=True)
                elapsed = time.perf_counter() - start
                hypotheses = [out.outputs[0].text for out in outputs]
                warmup_references = [item["text"] for item in warmup_items]
                wer = corpus_wer(hypotheses, warmup_references)
                warmup_result = {
                    "pass": warmup_idx,
                    "elapsed_sec": elapsed,
                    "rtfx": warmup_audio_sec / elapsed,
                    "wer": wer * 100,
                    "first_hypothesis": hypotheses[0],
                    "first_reference": warmup_references[0],
                }
                warmup_results.append(warmup_result)
                print(
                    f"warmup_pass={warmup_idx} WER={warmup_result['wer']:.2f} "
                    f"RTFx={warmup_result['rtfx']:.1f} elapsed={elapsed:.1f}s"
                )
            if args.reset_mm_cache_after_warmup:
                llm.reset_mm_cache()

        pass_results = []
        if args.cuda_profiler_range:
            import torch

            torch.cuda.cudart().cudaProfilerStart()
        for pass_idx in range(args.passes):
            if args.reset_mm_cache_each_pass:
                llm.reset_mm_cache()
            elapsed, hypotheses = generate_pass(
                llm,
                items,
                prompt,
                tokenizer,
                sampling_params,
                args.submit_chunk_size,
                prebuilt_requests=requests,
                use_manifest_messages=args.manifest_messages,
            )
            wer = corpus_wer(hypotheses, references)
            pass_result = {
                "pass": pass_idx,
                "is_warmup": False if warmup_requests else pass_idx < args.warmup_passes,
                "elapsed_sec": elapsed,
                "rtfx": total_audio_sec / elapsed,
                "wer": wer * 100,
                "first_hypothesis": hypotheses[0],
                "first_reference": references[0],
            }
            pass_results.append(pass_result)
            # Honour reset_mm_cache_after_warmup on THIS path too. The reset above only
            # fires when a separate --warmup-dataset is supplied, so with warmup taken
            # from the first --warmup-passes passes the flag was recorded as True in the
            # output JSON while never executing -- which made warm passes look faster
            # than they were. Belt-and-braces now that the cache defaults to disabled.
            if args.reset_mm_cache_after_warmup and not warmup_requests and pass_idx + 1 == args.warmup_passes:
                llm.reset_mm_cache()
            print(
                f"pass={pass_idx} warmup={pass_result['is_warmup']} "
                f"WER={pass_result['wer']:.2f} RTFx={pass_result['rtfx']:.1f} "
                f"elapsed={elapsed:.1f}s"
            )
        if args.cuda_profiler_range:
            import torch

            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            if not args.cuda_profiler_no_stop_sync:
                torch.cuda.synchronize()
            if args.cuda_profiler_stop_sleep_sec > 0:
                time.sleep(args.cuda_profiler_stop_sleep_sec)
        if pass_results:
            per_dataset_map, equal_avg_wer, scored_buckets = per_dataset_wer(hypotheses, items)
            per_dataset_runaways = {ds: b["tail_rows"] for ds, b in scored_buckets.items()}
            runaway_ids = sorted(
                row_id for b in scored_buckets.values() for row_id in b["tail_ids"]
            )
            print("Per-dataset WER (%) and runaway rows:")
            for ds in sorted(per_dataset_map):
                print(f"  {ds}: {per_dataset_map[ds]:.2f}  runaways={per_dataset_runaways.get(ds, 0)}")
            print(f"equal_dataset_average_wer={equal_avg_wer:.2f}")
            # Reported as a first-class number because it is the quantity that
            # actually moves: one runaway is worth ~0.013 micro WER on the
            # six-subset set, and a BF16 eager-vs-compiled switch alone shifts
            # the count from 6-8 to 10-12. Compare counts across repeats, never
            # as a single number. No ex-runaway WER is emitted here on purpose:
            # the exclusion set is the union across the arms being compared, so
            # it is only meaningful in `score_frozen.py --baseline`.
            print(f"runaway_rows={len(runaway_ids)}  rule: {TAIL_RULE}")
            if args.dump_predictions is not None:
                args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
                with args.dump_predictions.open("w") as pf:
                    for hyp, item in zip(hypotheses, items):
                        pf.write(
                            json.dumps(
                                {
                                    "reference": item["text"],
                                    "hypothesis": hyp,
                                    "dataset": item.get("dataset", "all"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(f"Wrote predictions: {args.dump_predictions} ({len(hypotheses)} rows)")
    finally:
        sampler.stop()
        if llm is not None:
            del llm

    measured = [r for r in pass_results if not r["is_warmup"]]
    rtfx_values = [r["rtfx"] for r in measured]
    wer_values = [r["wer"] for r in measured]
    result = {
        "mode": "plugin",
        "platform": "vllm_plugin",
        "model": str(args.model_dir),
        "tokenizer": str(args.tokenizer or args.model_dir),
        "precision": args.dtype,
        "prompt_contract": args.prompt_contract,
        # The prompt was never recorded, which made a WER unattributable to the
        # text that produced it once variants existed.
        "manifest_messages": args.manifest_messages,
        "user_prompt": args.user_prompt,
        "system_prompt": args.system_prompt,
        "rendered_prompt": rendered_prompt,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "moe_backend": args.moe_backend,
        "num_speculative_tokens": args.num_speculative_tokens,
        "speculative_model": args.speculative_model,
        "attention_backend": args.attention_backend,
        "async_scheduling": args.async_scheduling,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "mm_processor_cache_gb": args.mm_processor_cache_gb,
        "top_p": args.top_p,
        "seed": args.seed,
        "skip_special_tokens": args.skip_special_tokens,
        "spaces_between_special_tokens": args.spaces_between_special_tokens,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "reset_mm_cache_after_warmup": args.reset_mm_cache_after_warmup,
        "reset_mm_cache_each_pass": args.reset_mm_cache_each_pass,
        "dataset": str(args.manifest),
        "warmup_dataset": str(args.warmup_manifest) if args.warmup_manifest else None,
        "num_files": len(items),
        "warmup_num_files": len(warmup_items),
        "total_audio_sec": total_audio_sec,
        "warmup_total_audio_sec": warmup_audio_sec,
        "passes": args.passes,
        "warmup_passes": args.warmup_passes,
        "shuffle_seed": args.shuffle_seed,
        "manifest_load_sec": manifest_load_sec,
        "llm_init_sec": llm_init_sec,
        "request_build_sec": request_build_sec,
        "warmup_results": warmup_results,
        "pass_results": pass_results,
        "measured_rtfx_mean": statistics.mean(rtfx_values) if rtfx_values else None,
        "measured_rtfx_stdev": statistics.stdev(rtfx_values) if len(rtfx_values) > 1 else 0.0,
        "measured_wer_mean": statistics.mean(wer_values) if wer_values else None,
        # Recorded so a WER in this file is never ambiguous again: results
        # written before 2026-09-11 used whisper_normalizer plus a plain token
        # edit distance, which reads ~0.6 micro WER higher on the same audio.
        "scoring_contract": SCORING_CONTRACT,
        "per_dataset_wer": per_dataset_map,
        "equal_dataset_average_wer": equal_avg_wer,
        # Runaway rows are reported, not excluded: the ex-runaway WER needs the
        # union across the arms being compared, so it belongs to
        # `score_frozen.py --baseline`. The ids are recorded here so that
        # pairing needs no rescore.
        "runaway_rows": len(runaway_ids),
        "per_dataset_runaways": per_dataset_runaways,
        "runaway_ids": runaway_ids,
        "runaway_rule": TAIL_RULE,
        "gpu_memory_before_gb": memory_before_mb / 1024,
        "gpu_memory_after_load_gb": memory_after_load_mb / 1024,
        "gpu_memory_after_run_gb": get_gpu_memory_mb() / 1024,
        "gpu_memory_peak_gb": sampler.peak_mb / 1024,
        "data_parallel_size": args.data_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
        "disable_cudagraph": args.disable_cudagraph,
        "enable_flashinfer_autotune": args.enable_flashinfer_autotune,
        "disable_flashinfer_autotune": args.disable_flashinfer_autotune,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

# SpeechLM FP8 quantization (NemotronH decoder + ASR encoder)

Builds and evaluates an FP8 SpeechLM checkpoint for vLLM. Two independent halves:

| half | quantized by | activates with |
|---|---|---|
| NemotronH decoder | NVIDIA ModelOpt (`hf_ptq.py`) | stock vLLM, `quant_method: modelopt` |
| ASR encoder (128 Linears) | this repo's `FP8Linear` | **only** with this NeMo plugin |

The encoder half is **not** ModelOpt. It matches ModelOpt's numerics — E4M3,
per-output-channel weight scales, static per-tensor activation scales — but is
applied by `nemo/collections/asr/parts/submodules/enc_nvfp4.py` through the
SpeechLM vLLM plugin. ModelOpt's HF path only walks the LLM, and the perception
module lives in a separate `perception.safetensors` it never touches.

**Consequence worth internalising:** stock vLLM loads the decoder's FP8 correctly
and **silently ignores** the `encoder_quantization` block in `config.json`. You
get a BF16 encoder with no warning. If you are measuring encoder speedup, confirm
this line appears at load:

```
[NeMo I audio:221] Prequantized encoder: 128 FP8 Linears prepared for load, 128 static act scales
```

## Versions this was built and validated against

- vLLM `0.23.0`
- Model-Optimizer on `main` (see the trap below about older branches)
- transformers `5.x` — required; `4.57.6` has the `nemotron_h` module but does not
  register it in the auto-config mapping, and `llm_backbone/config.json` has no
  `auto_map`, so there is no remote-code fallback
- plugin entry point: `nemo_speechlm = "nemo.collections.speechlm2.vllm.salm:register"`

## Pipeline

ModelOpt cannot read a SpeechLM checkpoint directly, because the decoder lives
inside it as `llm.*` tensors. It has to be lifted out first.

```bash
# 1. decoder -> standalone HF NemotronH checkpoint (~59 GB, deletable afterwards)
python export_speechlm_llm_to_hf.py \
    --source <speechlm-ckpt> --output <llm-bf16> --sidecar-source <sidecar>

# 2. ModelOpt FP8 PTQ + reassemble with the perception weights
python create_speechlm_modelopt_fp8.py \
    --source <speechlm-ckpt> --output <fp8-ckpt> \
    --llm-name <llm-bf16> --llm-export-dir <llm-fp8> \
    --modelopt-repo <Model-Optimizer> \
    --calib-size 128 --calib-batch-size 1 --calib-seq 512 --kv-cache-qformat fp8

# 3. MANDATORY after any ModelOpt-main export -- see trap below
python repair_modelopt_exclude_names.py <fp8-ckpt>
```

`--sidecar-source` needs a directory holding both `config.json` (the NemotronH
one, i.e. `<speechlm-ckpt>/llm_backbone/config.json`) and `tokenizer.json` (from
the SpeechLM checkpoint root). Stage 1 takes a fast path only when both exist.

Stage 2 streams its calibration set. With no `--dataset` it uses ModelOpt's
default pair, `cnn_dailymail` + `nemotron-post-training-dataset-v2`; the second
is **gated**, so pass an HF token even though only ~128 samples are fetched.

### Optional: FP8 encoder on top

```bash
# 4. collect activation amax on HELD-OUT audio (never the eval set)
#    NEMO_ENC_QUANT=calib NEMO_ENC_QUANT_CALIB_OUT=<dir>/encoder_amax
#    then run local_speechlm_plugin_vllm_eval.py over a small dev manifest

# 5. bake the static per-tensor recipe into config.json
python bake_encoder_quant_config.py \
    --source <fp8-ckpt> --amax <dir>/encoder_amax.json --margin 1.5 --output <baked>

# 6. quantize the 128 encoder weights in-checkpoint (saves ~599 MiB)
python export_encoder_fp8.py --source <baked> --output <final>

# 7. verify bitwise on CPU before spending GPU time
python test_encoder_fp8_equivalence.py --source <baked> --exported <final>
```

Step 4 writes `encoder_amax.<pid>.json`, one per engine worker, so concurrent
workers cannot clobber each other. Steps 5+ want a single canonical
`encoder_amax.json`; merge the shards by elementwise max.

Step 7 needs vLLM importable (`FP8Linear` builds weights through
`vllm._custom_ops`) but touches no GPU. Run it in the eval container.

## Traps, each of which has cost real time

**ModelOpt `main` corrupts every exclude name.** It emits them with a NUL-byte
sentinel suffix — `lm_head.\x00backbone.pt_name_sentinel` — in *both*
`config.json`'s `quantization_config.ignore` and `hf_quant_config.json`'s
`exclude_modules`. Nothing then matches a real module, so vLLM builds quantized
layers over weights that are still BF16 and dies in `load_qkv_weight` with a
shape assert. `repair_modelopt_exclude_names.py` strips the sentinel and refuses
to write unless every repaired name resolves to an existing tensor that has no
`weight_scale`.

**Older ModelOpt branches carrying NemotronH remote-code support are dead ends
on transformers 5.x.** They fail in `modelopt/torch/opt/dynamic.py` with
`issubclass() arg 2 must be a class`, because they register a transformers name
that is no longer a class. Use `main`: transformers 5.x knows `nemotron_h`
natively, so the remote-code support is moot.

**The diarizer shares the encoder's module names.** A suffix match on
`attn.w_qkv` / `attn.out_proj` / `ffn.net.0` / `ffn.net.3` selects **252**
tensors, not 128, because the Sortformer diarizer uses identical names.
Selection is by the `perception.encoder.asr_encoder.` prefix and asserted at 128.
Do not loosen that: FP8 on the diarizer measured as a **regression** (its GEMMs
are small enough that per-token scaling overhead exceeds the saving), and its
output is fused additively into the ASR features, so damage is diffuse and does
not show up cleanly in WER.

**Checkpoints written by containers land `root:root` mode `0600`,** with `0700`
directories. Collaborators get "access denied" and transfers fail before moving
a byte. `chmod -R a+rX` through a container fixes it; verify by reading a byte,
not by checking mode bits.

**Baked checkpoints are symlink farms.** `bake_encoder_quant_config.py` links
rather than copies, so only `perception.safetensors` is real and the chain can be
two deep. Copy with `rsync -aL`, `cp -rL`, or `tar -h`, or the destination gets
dangling absolute paths.

## Known wart

`enc_nvfp4.py` is named for NVFP4 but is used almost entirely for FP8, and still
carries an `NVFP4Linear` alongside `FP8Linear`. Note that NVFP4 needs Blackwell
(sm100+); there are no FP4 tensor cores on H100/sm90, so FP8 is the right target
there.

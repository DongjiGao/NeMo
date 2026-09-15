"""NVFP4 quantization for the NeMo SpeechLM audio encoder's Linear layers.

Why this exists: the encoder is plain PyTorch inside NeMo's perception module and
never passes through vLLM's quantization machinery, so neither ModelOpt nor
torchao can accelerate it here. ModelOpt only simulates (quantize/dequantize in
BF16, which is *slower* than BF16), and torchao's NVFP4 path needs the external
MSLK package. vLLM's CUTLASS FP4 kernels are already installed and already serve
the quantized decoder, so we call them directly.

Measured at encoder shapes (M=4107): 2.0-2.8x over BF16 per GEMM, versus
0.34-1.11x for torchao FP8 and 0.73-0.89x for ModelOpt's simulated path.

Activation scaling is dynamic by default: each forward computes its own global
scale via an amax reduction. Set static_scales=True after calibration to skip
that extra pass over the activations.

FP8 is available per layer as well, via ``fp8_re``. Two of the four matrices per
block are badly conditioned for FP4 -- see FP8Linear for why -- and they are only
5 of the 12 GEMM-FLOP units in a block, so holding them at FP8 costs little
throughput.
"""

from __future__ import annotations

import json
import re

import torch
import torch.nn as nn

# NVFP4: 4-bit E2M1 values (max 6.0) grouped in blocks of 16, each block scaled
# by an FP8 E4M3 factor (max 448.0). The global scale maps a tensor's amax onto
# the product of those two ranges.
FP4_MAX = 6.0
FP8_MAX = 448.0
BLOCK = 16

FP8_DTYPE = torch.float8_e4m3fn

# Which encoder Linear layers to quantize. Deliberately excludes pre_encode
# (1.3M params) and leaves all norms in BF16.
DEFAULT_PATTERNS = ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")

# Subtree that must never be quantized. See the guard in `quantize_encoder`.
DIARIZER_MARKER = "diarization_model"

# HR8 (Nemotron-Speech 2.0) module paths, from the eval-step-17000 safetensors
# header. The quantizable ASR branch is 4 linears x 32 layers = 128 GEMMs:
#   perception.encoder.asr_encoder.layers.{0..31}.attn.w_qkv      [3840, 1280]
#   perception.encoder.asr_encoder.layers.{0..31}.attn.out_proj   [1280, 1280]
#   perception.encoder.asr_encoder.layers.{0..31}.ffn.net.0       [5120, 1280]
#   perception.encoder.asr_encoder.layers.{0..31}.ffn.net.3       [1280, 5120]
# head_dim is 80 (q_norm/k_norm are [80]), so d_model 1280 = 16 heads.
HR8_ASR_ENCODER_PATH = "perception.encoder.asr_encoder"


def _global_scale(amax: torch.Tensor) -> torch.Tensor:
    """Per-tensor scale placing amax at the top of the representable range."""
    amax = amax.float().clamp_min(1e-8)
    return (FP8_MAX * FP4_MAX / amax).reshape(1)


class NVFP4Linear(nn.Module):
    """Drop-in replacement for nn.Linear using vLLM's CUTLASS NVFP4 GEMM.

    Weights are quantized once at construction. Activations are quantized per
    forward, since their dynamic range varies per layer and per batch.
    """

    def __init__(self, linear: nn.Linear, static_input_scale: torch.Tensor | None = None):
        super().__init__()
        import vllm._custom_ops as ops

        self._ops = ops
        weight = linear.weight.data
        self.out_features, self.in_features = weight.shape
        if self.in_features % BLOCK != 0:
            raise ValueError(f"in_features={self.in_features} must be a multiple of {BLOCK} for NVFP4")

        w_gs = _global_scale(weight.abs().max())
        wq, w_sf = ops.scaled_fp4_quant(weight, w_gs)
        self.register_buffer("weight_packed", wq, persistent=False)
        self.register_buffer("weight_scale", w_sf, persistent=False)
        self.register_buffer("weight_global_scale", w_gs, persistent=False)
        if static_input_scale is not None:
            self.register_buffer("input_global_scale", static_input_scale.reshape(1), persistent=False)
        else:
            self.input_global_scale = None

        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = self._ops
        out_shape = (*x.shape[:-1], self.out_features)
        x2d = x.reshape(-1, self.in_features)

        x_gs = self.input_global_scale
        if x_gs is None:
            x_gs = _global_scale(x2d.abs().max())

        xq, x_sf = ops.scaled_fp4_quant(x2d, x_gs)
        # CUTLASS applies a single fused dequant factor, so both global scales
        # are folded into alpha rather than rescaling the output afterwards.
        alpha = 1.0 / (x_gs * self.weight_global_scale)
        out = ops.cutlass_scaled_fp4_mm(xq, self.weight_packed, x_sf, self.weight_scale, alpha, x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        mode = "static" if self.input_global_scale is not None else "dynamic"
        return f"in={self.in_features}, out={self.out_features}, nvfp4, act_scale={mode}"


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using vLLM's CUTLASS FP8 GEMM.

    Exists for the encoder tensors that FP4 handles badly. E2M1 elements carry a
    single mantissa bit, so within a block of 16 channels one large value forces
    the block scale up and its neighbours round to zero; measured on this encoder,
    ``ffn.net.3`` sees activation amax up to 72 against a typical magnitude near 1.
    E4M3's four exponent bits absorb that spread instead, at roughly half of FP4's
    arithmetic throughput.

    Weights are always per output channel. Activations default to per token and
    dynamic, which needs no calibration because the amax is fused into vLLM's
    quantization kernel, unlike the whole-tensor reduction that made NVFP4's
    dynamic mode expensive.

    Passing ``static_input_scale`` switches activations to static per tensor.
    Note these are not independent choices: a per-token scale cannot be
    precomputed for a token that has not been seen, so going static necessarily
    coarsens granularity.

    Static was added because it is what ModelOpt exports, making it the
    configuration a checkpoint-resident quantization would actually ship, and it
    was expected to cost a little throughput for that convenience. Measured on the
    six-subset set it is instead the faster of the two, 878.98 RTFx against
    859.11, with WER 4.4193 against 4.4290. Fusing the amax into the quantization
    kernel evidently does not make it free -- the reduction still reads the whole
    activation tensor, and skipping it is worth ~2.3% end to end. One clean
    replicate per arm, so treat the size as approximate but the sign as real.
    """

    def __init__(self, linear: nn.Linear, static_input_scale: torch.Tensor | None = None):
        super().__init__()
        import vllm._custom_ops as ops

        self._ops = ops
        weight = linear.weight.data
        self.out_features, self.in_features = weight.shape
        # 16-byte alignment for the CUTLASS tile; both encoder widths (1280, 5120)
        # already satisfy it.
        if self.in_features % 16 != 0:
            raise ValueError(f"in_features={self.in_features} must be a multiple of 16 for CUTLASS FP8")

        w = weight.float()
        w_scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
        wq = (w / w_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        self.register_buffer("weight_fp8", wq, persistent=False)
        # CUTLASS requires fp32 scales. numel must be 1 or match the output width.
        self.register_buffer("weight_scale", w_scale, persistent=False)

        # Same divisor convention as the weights: xq = x / scale, and CUTLASS
        # multiplies by scale to recover. This is NOT NVFP4's _global_scale,
        # which is the reciprocal form (FP8_MAX * FP4_MAX / amax).
        if static_input_scale is not None:
            self.register_buffer(
                "input_scale", static_input_scale.float().reshape(1), persistent=False
            )
        else:
            self.input_scale = None

        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

    @classmethod
    def from_prequantized(
        cls,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        static_input_scale: torch.Tensor | None = None,
        bias_dtype: torch.dtype = torch.bfloat16,
        device=None,
    ) -> "FP8Linear":
        """Build an empty module for a checkpoint that already holds FP8 weights.

        The weights are not computed here; ``load_state_dict`` fills them. That is
        the whole point of the prequantized path, and it is why the module has to
        exist *before* the load rather than being swapped in on first forward like
        the runtime path does.

        Buffer names deliberately match the checkpoint (``weight``,
        ``weight_scale``) and are persistent, so the load neither reports them as
        unexpected nor leaves them unfilled. ``weight_scale`` is stored flat as
        ``[out]``, the layout ModelOpt emits; ``forward`` views it as ``[out, 1]``
        for CUTLASS, which is free.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)

        import vllm._custom_ops as ops

        self._ops = ops
        self._prequantized = True
        self.out_features, self.in_features = out_features, in_features
        if in_features % 16 != 0:
            raise ValueError(f"in_features={in_features} must be a multiple of 16 for CUTLASS FP8")

        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, dtype=torch.float32, device=device)
        )
        if static_input_scale is not None:
            self.register_buffer(
                "input_scale", static_input_scale.float().reshape(1), persistent=False
            )
        else:
            self.input_scale = None

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=bias_dtype, device=device), requires_grad=False
            )
        return self

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_prequantized", False):
            return self.weight, self.weight_scale.unsqueeze(-1)
        return self.weight_fp8, self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = self._ops
        out_shape = (*x.shape[:-1], self.out_features)
        x2d = x.reshape(-1, self.in_features)

        if self.input_scale is not None:
            # Providing a scale makes this static per-tensor; the kernel ignores
            # use_per_token_if_dynamic in that case, so do not pass it.
            xq, x_scale = ops.scaled_fp8_quant(x2d, self.input_scale)
        else:
            xq, x_scale = ops.scaled_fp8_quant(x2d, None, use_per_token_if_dynamic=True)
        weight, weight_scale = self._weights()
        # B must be column-major; transposing the row-major (out, in) weight is a
        # free metadata change that yields exactly that.
        out = ops.cutlass_scaled_mm(xq, weight.t(), x_scale, weight_scale, x.dtype, self.bias)
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        mode = "static_per_tensor" if self.input_scale is not None else "per_token_dynamic"
        src = "checkpoint" if getattr(self, "_prequantized", False) else "runtime"
        return (
            f"in={self.in_features}, out={self.out_features}, fp8_e4m3, "
            f"act_scale={mode}, weights={src}"
        )


def quantize_encoder(
    module: nn.Module,
    patterns=DEFAULT_PATTERNS,
    verbose: bool = True,
    scales: dict | None = None,
    scale_margin: float = 1.0,
    exclude_re: str | None = None,
    fp8_re: str | None = None,
    fp8_amax_above: float | None = None,
) -> int:
    """Swap matching nn.Linear children of ``module`` for NVFP4Linear in place.

    ``scales`` maps layer name to a calibrated activation amax; layers found there
    get a static input scale, the rest fall back to computing one per forward.

    ``scale_margin`` inflates each calibrated amax. Calibration sees less data than
    inference, so its amax underestimates the true range and unmargined static
    scales clamp the largest activations. A margin trades a slightly coarser block
    scale for not clipping. Only meaningful together with ``scales``.

    ``exclude_re`` keeps matching layers in BF16 and ``fp8_re`` puts them in FP8,
    for mixed-precision recipes; ``exclude_re`` wins where both match. Activation
    amax grows with depth and is dominated by two of the four matrices per block:
    ``ffn.net.3`` (post-GELU) and ``attn.out_proj`` (post-attention) reach 40-72 in
    layers 22-31, while the two that read LayerNorm output stay under 16 everywhere.
    Those two are also only 5 of 12 GEMM-FLOP units per block, so FP8 there is
    cheaper than the BF16 alternative for the same protection.

    ``fp8_amax_above`` selects by calibrated activation amax instead of by name,
    requiring ``scales``. Measured caveat: the WER cost of FP4 tracks how *many*
    matrices are quantized, not their amax -- protecting the 20 highest-amax
    matrices (40% of summed amax) recovered only 19% of the damage, matching their
    16% share of the count. So treat this as a coverage-count knob.

    Returns the number of layers replaced. Skips layers whose input dim is not a
    multiple of the NVFP4 block size.
    """
    if fp8_amax_above is not None and scales is None:
        raise ValueError("fp8_amax_above needs calibrated scales to compare against")

    # HR8 nests the ASR encoder under a ParallelExpertEncoder alongside a Sortformer
    # diarizer, and the diarizer is itself a TransformerEncoder -- so 217 of its
    # tensors match DEFAULT_PATTERNS by substring and would be silently quantized
    # along with the branch we actually want. The diarizer must stay BF16: its
    # output is fused additively into the ASR features, so degrading it corrupts
    # every frame rather than showing up as an isolated layer error.
    if any(DIARIZER_MARKER in n for n, _ in module.named_modules()):
        raise ValueError(
            f"module contains a {DIARIZER_MARKER!r} subtree, which matches the quantization "
            "patterns but must stay in BF16. Pass the ASR branch directly "
            "(e.g. perception.encoder.asr_encoder), or set "
            f"exclude_re={DIARIZER_MARKER!r} if you really mean to scope by regex."
        )

    excl = re.compile(exclude_re) if exclude_re else None
    as_fp8 = re.compile(fp8_re) if fp8_re else None
    targets = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and any(p in name for p in patterns):
            targets.append(name)

    replaced, skipped, static, excluded, n_fp8, saved_bytes = 0, 0, 0, 0, 0, 0
    static_fp8 = 0
    for name in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        lin = getattr(parent, attr)
        if excl is not None and excl.search(name):
            excluded += 1
            continue
        if lin.in_features % BLOCK != 0:
            skipped += 1
            continue
        before = lin.weight.numel() * lin.weight.element_size()
        by_amax = fp8_amax_above is not None and float(scales.get(name, 0.0)) > fp8_amax_above
        if by_amax or (as_fp8 is not None and as_fp8.search(name)):
            # FP8 wants the divisor form amax/FP8_MAX, not NVFP4's reciprocal
            # global scale, so the two branches cannot share _global_scale.
            fs = None
            if scales is not None and name in scales:
                amax = float(scales[name]) * scale_margin
                fs = torch.tensor(
                    max(amax / FP8_MAX, 1e-12), device=lin.weight.device
                )
                static_fp8 += 1
            q = FP8Linear(lin, static_input_scale=fs)
            after = q.weight_fp8.numel() * q.weight_fp8.element_size()
            n_fp8 += 1
        else:
            gs = None
            if scales is not None and name in scales:
                amax = float(scales[name]) * scale_margin
                gs = _global_scale(torch.tensor(amax, device=lin.weight.device))
                static += 1
            q = NVFP4Linear(lin, static_input_scale=gs)
            after = q.weight_packed.numel() * q.weight_packed.element_size()
        after += q.weight_scale.numel() * q.weight_scale.element_size()
        saved_bytes += before - after
        setattr(parent, attr, q)
        replaced += 1

    if verbose:
        n_fp4 = replaced - n_fp8
        margin = f", margin {scale_margin:g}x" if scales else ""
        fp4_mode = f"{static} static / {n_fp4 - static} dynamic" if scales else "dynamic"
        fp8_mode = f"{static_fp8} static / {n_fp8 - static_fp8} dynamic" if scales else "dynamic"
        print(
            f"[enc_nvfp4] replaced {replaced} Linear layers: "
            f"{n_fp4} NVFP4 ({fp4_mode} act scales), "
            f"{n_fp8} FP8 ({fp8_mode} act scales{margin}), "
            f"skipped {skipped}, kept {excluded} in BF16, "
            f"freed {saved_bytes / 2**20:.1f} MiB of weights",
            flush=True,
        )
    return replaced


class ActivationAmaxCollector:
    """Records the running max |activation| entering each target Linear.

    Used to derive static input scales. Calibrate on audio held out from the
    evaluation set: the project's own history shows that calibrating on
    ``representative_128`` leaks, because those clips are all inside the eval set.
    """

    def __init__(self, module: nn.Module, patterns=DEFAULT_PATTERNS, save_every: int = 32, save_path: str | None = None):
        self.amax: dict[str, torch.Tensor] = {}
        self._handles = []
        self._anchor: str | None = None
        self._anchor_calls = 0
        self._save_every = save_every
        self._save_path = save_path
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear) and any(p in name for p in patterns):
                if self._anchor is None:
                    self._anchor = name
                self._handles.append(child.register_forward_pre_hook(self._make_hook(name)))
        print(f"[enc_nvfp4] calibrating {len(self._handles)} Linear layers", flush=True)

    def _make_hook(self, name: str):
        def hook(_mod, inputs):
            v = inputs[0].detach().abs().max()
            prev = self.amax.get(name)
            self.amax[name] = v if prev is None else torch.maximum(prev, v)
            # Checkpoint periodically: this runs inside vLLM's EngineCore child,
            # which may be killed hard enough that atexit never fires. The anchor
            # layer fires once per encoder call, so this counts encoder calls.
            if name == self._anchor and self._save_path:
                self._anchor_calls += 1
                if self._anchor_calls % self._save_every == 0:
                    self.save(self._save_path)

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # vLLM profiles startup memory by running the encoder on dummy audio, so a
    # calibration that dies before reading any real audio still has a fully
    # populated amax dict. Since save() is also an atexit handler, such a run
    # writes a file that looks completely legitimate. It bit once: a crashed run
    # (missing audio mount) produced 128 values every one of which was BELOW the
    # real calibration, median ratio 0.426 -- scales ~2.3x too small, which would
    # have clipped activations throughout and read as "static FP8 does not work".
    MIN_CALLS = 4

    def save(self, path: str, force: bool = False) -> None:
        if not force and self._anchor_calls < self.MIN_CALLS:
            print(
                f"[enc_nvfp4] REFUSING to write {path}: only {self._anchor_calls} encoder "
                f"call(s) observed (need >= {self.MIN_CALLS}). This is almost certainly "
                "vLLM's dummy-audio startup pass after a failed run, not real calibration. "
                "Fix the failure and rerun rather than using these scales.",
                flush=True,
            )
            return
        payload = {k: float(v) for k, v in self.amax.items()}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(
            f"[enc_nvfp4] wrote {len(payload)} activation amax values from "
            f"{self._anchor_calls} encoder calls to {path}",
            flush=True,
        )


def load_scales(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)

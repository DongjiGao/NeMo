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

"""Unit tests for serving SpeechLM checkpoints whose audio encoder is quantized with ModelOpt FP8."""

from fnmatch import fnmatch

import pytest
import torch
from torch import nn

try:
    from nemo.collections.speechlm2.vllm.salm.fp8_encoder import (
        FP8_DTYPE,
        FP8_MAX,
        FP8Linear,
        build_fp8_encoder,
        check_fp8_encoder_weights,
    )

    _HAS_PLUGIN = True
except (ImportError, RuntimeError):
    _HAS_PLUGIN = False

D_MODEL = 32
N_LAYERS = 2
N_QUANTIZED = N_LAYERS * 4
DECODER_IGNORE = ["backbone.embeddings", "lm_head", "mtp*"]
PERCEPTION_IGNORE = [
    "perception.encoder.diarization_model*",
    "perception.encoder.asr_encoder.pre_encode*",
    "perception.proj",
]


class _QuantConfig:
    """The parts of vLLM's ModelOpt quantization config that the FP8 encoder reads."""

    def __init__(self, ignore, name="modelopt", quant_method="FP8"):
        self.exclude_modules = list(ignore)
        self.quant_method = quant_method
        self._name = name

    def get_name(self):
        return self._name

    def is_layer_excluded(self, prefix):
        return any(prefix == entry or fnmatch(prefix, entry) for entry in self.exclude_modules)


class _Block(nn.Module):
    """Mirrors the Linear names of a SpeechLM transformer encoder block."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.w_qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.attn.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.ffn = nn.Module()
        self.ffn.net = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.ReLU(), nn.Identity(), nn.Linear(4 * D_MODEL, D_MODEL)
        )


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_encode = nn.Module()
        self.pre_encode.proj = nn.Linear(2 * D_MODEL, D_MODEL)
        self.layers = nn.ModuleList([_Block() for _ in range(N_LAYERS)])


def _speaker_aware_perception() -> nn.Module:
    """A freshly constructed (float32) perception module laid out like a speaker-aware SpeechLM's: an ASR
    branch and a diarizer with identical Linear names, plus the connector."""
    perception = nn.Module()
    perception.encoder = nn.Module()
    perception.encoder.asr_encoder = _Encoder()
    perception.encoder.diarization_model = nn.Module()
    perception.encoder.diarization_model.encoder = _Encoder()
    perception.proj = nn.Linear(D_MODEL, D_MODEL)
    return perception


def _asr_linears(perception: nn.Module) -> list[nn.Module]:
    return [
        linear
        for layer in perception.encoder.asr_encoder.layers
        for linear in (layer.attn.w_qkv, layer.attn.out_proj, layer.ffn.net[0], layer.ffn.net[3])
    ]


def _fp8_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _checkpoint(perception: nn.Module) -> dict[str, torch.Tensor]:
    """A random perception state dict in the checkpoint's dtypes: FP8 weights, float32 scalar scales, bfloat16 rest."""
    state = {}
    for name, tensor in perception.state_dict().items():
        if tensor.dtype == FP8_DTYPE:
            state[name] = torch.randn(tensor.shape).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        elif name.endswith(("weight_scale", "input_scale")):
            state[name] = torch.rand(()) * 1e-3 + 1e-4  # not representable in bfloat16
        else:
            state[name] = torch.randn(tensor.shape).to(torch.bfloat16)
    return state


def _quantized_perception() -> nn.Module:
    perception = _speaker_aware_perception()
    build_fp8_encoder(perception, _QuantConfig(DECODER_IGNORE + PERCEPTION_IGNORE))
    return perception


@pytest.mark.skipif(not _HAS_PLUGIN, reason="SpeechLM vLLM plugin not available")
class TestBuildFP8Encoder:
    def test_builds_fp8_linears_for_the_layers_not_ignored(self):
        perception = _speaker_aware_perception()

        replaced = build_fp8_encoder(perception, _QuantConfig(DECODER_IGNORE + PERCEPTION_IGNORE))

        assert replaced == N_QUANTIZED
        assert all(isinstance(linear, FP8Linear) for linear in _asr_linears(perception))
        assert sum(isinstance(m, FP8Linear) for m in perception.modules()) == N_QUANTIZED
        assert type(perception.encoder.asr_encoder.pre_encode.proj) is nn.Linear
        assert type(perception.proj) is nn.Linear

    def test_decoder_only_quant_config_leaves_the_encoder_unquantized(self):
        perception = _speaker_aware_perception()

        assert build_fp8_encoder(perception, _QuantConfig(DECODER_IGNORE)) == 0
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    @pytest.mark.parametrize("quant_config", [None, object(), _QuantConfig(DECODER_IGNORE + ["perception*"])])
    def test_no_encoder_quantization(self, quant_config):
        perception = _speaker_aware_perception()

        assert build_fp8_encoder(perception, quant_config) == 0
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    @pytest.mark.parametrize(
        ("name", "quant_method"), [("modelopt", "FP8_PER_CHANNEL_PER_TOKEN"), ("modelopt_fp4", "NVFP4")]
    )
    def test_rejects_other_schemes_before_changing_the_encoder(self, name, quant_method):
        perception = _speaker_aware_perception()

        with pytest.raises(NotImplementedError, match="ModelOpt FP8"):
            build_fp8_encoder(perception, _QuantConfig(DECODER_IGNORE + PERCEPTION_IGNORE, name, quant_method))
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    def test_rejects_widths_cutlass_cannot_run_before_changing_the_encoder(self):
        perception = _speaker_aware_perception()
        perception.encoder.asr_encoder.layers[1].ffn.net[3] = nn.Linear(24, D_MODEL)

        with pytest.raises(ValueError, match="multiple of 16"):
            build_fp8_encoder(perception, _QuantConfig(DECODER_IGNORE + PERCEPTION_IGNORE))
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    def test_vllm_modelopt_config_selects_the_same_layers(self):
        pytest.importorskip("vllm")
        from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

        quant_config = ModelOptFp8Config.from_config(
            {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": DECODER_IGNORE + PERCEPTION_IGNORE}
        )
        perception = _speaker_aware_perception()

        assert build_fp8_encoder(perception, quant_config) == N_QUANTIZED
        assert all(isinstance(linear, FP8Linear) for linear in _asr_linears(perception))
        assert sum(isinstance(m, FP8Linear) for m in perception.modules()) == N_QUANTIZED

    def test_bfloat16_cast_keeps_fp8_weights_and_float32_scales(self):
        perception = _quantized_perception()
        state = _checkpoint(perception)
        perception.load_state_dict(state, strict=True)

        perception.to(torch.bfloat16)

        layer = perception.encoder.asr_encoder.layers[0].attn.w_qkv
        prefix = "encoder.asr_encoder.layers.0.attn.w_qkv."
        assert layer.weight.dtype == FP8_DTYPE
        assert torch.equal(_fp8_bits(layer.weight), _fp8_bits(state[prefix + "weight"]))
        for scale_name in ("weight_scale", "input_scale"):
            scale = getattr(layer, scale_name)
            assert scale.dtype == torch.float32 and torch.equal(scale, state[prefix + scale_name])
        assert layer.bias.dtype == torch.bfloat16

    def test_checkpoint_state_dict_fills_fp8_modules(self):
        perception = _quantized_perception()
        perception.to(torch.bfloat16)  # the plugin casts perception right before loading
        state = _checkpoint(perception)

        result = perception.load_state_dict(state, strict=True)

        assert not result.missing_keys and not result.unexpected_keys
        w_qkv = perception.encoder.asr_encoder.layers[1].attn.w_qkv
        assert torch.equal(_fp8_bits(w_qkv.weight), _fp8_bits(state["encoder.asr_encoder.layers.1.attn.w_qkv.weight"]))
        assert torch.equal(w_qkv.input_scale, state["encoder.asr_encoder.layers.1.attn.w_qkv.input_scale"])

    def test_weight_check_accepts_a_matching_checkpoint(self):
        perception = _quantized_perception()

        check_fp8_encoder_weights(perception, _checkpoint(perception))

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("encoder.asr_encoder.layers.0.attn.w_qkv.input_scale", None, "input_scale is missing"),
            ("encoder.asr_encoder.layers.0.attn.w_qkv.weight", torch.zeros(3 * D_MODEL, D_MODEL), "not torch.float8"),
            ("encoder.asr_encoder.layers.0.ffn.net.0.weight_scale", torch.ones(4 * D_MODEL), "not one scale"),
            (
                "encoder.diarization_model.encoder.layers.0.attn.w_qkv.weight",
                torch.zeros(3 * D_MODEL, D_MODEL).to(torch.float8_e4m3fn),
                "leaves its layer unquantized.*encoder_quantization block is not read",
            ),
        ],
    )
    def test_weight_check_rejects_a_mismatched_checkpoint(self, key, value, match):
        perception = _quantized_perception()
        state = _checkpoint(perception)
        if value is None:
            del state[key]
        else:
            state[key] = value

        with pytest.raises(ValueError, match=match):
            check_fp8_encoder_weights(perception, state)

    def test_plugin_perception_loader_loads_fp8_checkpoint(self):
        pytest.importorskip("vllm")
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        perception = _quantized_perception()
        state = _checkpoint(perception)
        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        nn.Module.__init__(model)
        model.perception = perception
        model._uses_pe_encoder = True  # turns on the exact-architecture check

        loaded = model._load_perception_weights(state)

        assert loaded == {f"perception.{name}" for name in state}
        w_qkv = model.perception.encoder.asr_encoder.layers[1].attn.w_qkv
        assert w_qkv.weight.dtype == FP8_DTYPE
        assert torch.equal(_fp8_bits(w_qkv.weight), _fp8_bits(state["encoder.asr_encoder.layers.1.attn.w_qkv.weight"]))
        assert model.perception.encoder.diarization_model.encoder.layers[0].attn.w_qkv.weight.dtype == torch.bfloat16

    def test_plugin_perception_loader_rejects_fp8_weights_for_an_unquantized_encoder(self):
        pytest.importorskip("vllm")
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        state = _checkpoint(_quantized_perception())
        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        nn.Module.__init__(model)
        model.perception = _speaker_aware_perception()
        model._uses_pe_encoder = True

        with pytest.raises(ValueError, match="leaves its layer unquantized"):
            model._load_perception_weights(state)


@pytest.mark.skipif(not _HAS_PLUGIN or not torch.cuda.is_available(), reason="needs CUDA")
def test_device_move_keeps_fp8_dtypes():
    perception = _quantized_perception()

    perception.to("cuda", torch.bfloat16)

    layer = perception.encoder.asr_encoder.layers[0].attn.w_qkv
    assert layer.weight.device.type == "cuda" and layer.weight.dtype == FP8_DTYPE
    assert layer.weight_scale.device.type == "cuda" and layer.weight_scale.dtype == torch.float32
    assert layer.input_scale.device.type == "cuda" and layer.input_scale.dtype == torch.float32
    assert layer.bias.device.type == "cuda" and layer.bias.dtype == torch.bfloat16


def _fp8_gemm_available() -> bool:
    if not _HAS_PLUGIN or not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        return False
    try:
        import vllm._custom_ops  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _fp8_gemm_available(), reason="needs an FP8-capable GPU and vLLM's CUTLASS kernels")
def test_fp8_linear_matches_dequantized_reference():
    torch.manual_seed(0)
    weight = torch.randn(64, 48, device="cuda")
    weight_scale = weight.abs().amax() / FP8_MAX
    weight_fp8 = (weight / weight_scale).to(FP8_DTYPE)
    bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(5, 7, 48, device="cuda", dtype=torch.bfloat16)
    input_scale = x.float().abs().amax() / FP8_MAX

    layer = FP8Linear(48, 64, bias=True, device="cuda")
    layer.load_state_dict(
        {"weight": weight_fp8, "weight_scale": weight_scale, "input_scale": input_scale, "bias": bias}
    )
    out = layer(x)

    # Quantize the activations the way vLLM's kernel does (multiplying by the reciprocal scale),
    # so the comparison isolates the GEMM's scaling.
    x_dequant = (x.float() * (1.0 / input_scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float() * input_scale
    reference = x_dequant @ (weight_fp8.float() * weight_scale).t() + bias.float()
    assert out.shape == (5, 7, 64)
    torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.1)

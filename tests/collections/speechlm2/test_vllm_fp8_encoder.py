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

"""Unit tests for serving SpeechLM checkpoints with a prequantized FP8 audio encoder."""

import pytest
import torch
from torch import nn

try:
    from nemo.collections.speechlm2.vllm.salm.fp8_encoder import FP8_DTYPE, FP8_MAX, FP8Linear, build_fp8_encoder

    _HAS_PLUGIN = True
except (ImportError, RuntimeError):
    _HAS_PLUGIN = False

PATTERNS = ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
D_MODEL = 32
N_LAYERS = 2


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
        self.layers = nn.ModuleList([_Block() for _ in range(N_LAYERS)])


def _speaker_aware_perception() -> nn.Module:
    """A freshly constructed (float32) perception module whose encoder holds an ASR branch and a diarizer
    with identical Linear names."""
    perception = nn.Module()
    perception.encoder = nn.Module()
    perception.encoder.asr_encoder = _Encoder()
    perception.encoder.diarization_model = nn.Module()
    perception.encoder.diarization_model.encoder = _Encoder()
    return perception


def _fp8_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _recipe(**overrides) -> dict:
    cfg = {
        "format": "fp8_e4m3",
        "weights_prequantized": True,
        "weight_scale": "per_output_channel",
        "activation_scale": "static_per_tensor",
        "patterns": list(PATTERNS),
        "activation_amax": {f"layers.{i}.{p}": 2.0 + i for i in range(N_LAYERS) for p in PATTERNS},
        "scale_margin": 1.5,
        "expect_replaced": N_LAYERS * len(PATTERNS),
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.skipif(not _HAS_PLUGIN, reason="SpeechLM vLLM plugin not available")
class TestBuildFP8Encoder:
    def test_replaces_only_asr_encoder_linears(self):
        perception = _speaker_aware_perception()
        replaced = build_fp8_encoder(perception, _recipe())

        assert replaced == N_LAYERS * len(PATTERNS)
        asr = perception.encoder.asr_encoder
        for layer in asr.layers:
            for linear in (layer.attn.w_qkv, layer.attn.out_proj, layer.ffn.net[0], layer.ffn.net[3]):
                assert isinstance(linear, FP8Linear)
        diarizer = perception.encoder.diarization_model.encoder
        assert not any(isinstance(m, FP8Linear) for m in diarizer.modules())

    def test_static_input_scale_uses_amax_and_margin(self):
        perception = _speaker_aware_perception()
        build_fp8_encoder(perception, _recipe())

        layer1_qkv = perception.encoder.asr_encoder.layers[1].attn.w_qkv
        assert layer1_qkv.input_scale.item() == pytest.approx(3.0 * 1.5 / FP8_MAX)

    def test_dynamic_activations_need_no_amax(self):
        perception = _speaker_aware_perception()
        build_fp8_encoder(perception, _recipe(activation_scale="dynamic_per_token", activation_amax=None))

        assert perception.encoder.asr_encoder.layers[0].attn.w_qkv.input_scale is None

    @pytest.mark.parametrize("cfg", [None, {}, {"format": "fp8_e4m3", "patterns": list(PATTERNS)}])
    def test_noop_without_prequantized_weights(self, cfg):
        perception = _speaker_aware_perception()

        assert build_fp8_encoder(perception, cfg) == 0
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    def test_block_without_prequantized_weights_warns(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm import fp8_encoder

        warnings = []
        monkeypatch.setattr(fp8_encoder.logging, "warning", lambda msg, *args, **kwargs: warnings.append(msg))
        perception = _speaker_aware_perception()

        assert build_fp8_encoder(perception, _recipe(weights_prequantized=False)) == 0
        assert len(warnings) == 1 and "weights_prequantized" in warnings[0]
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    def test_patterns_match_whole_name_components(self):
        perception = nn.Module()
        perception.encoder = nn.Module()
        perception.encoder.net = nn.Sequential(nn.Linear(D_MODEL, D_MODEL))
        perception.encoder.subnet = nn.Sequential(nn.Linear(D_MODEL, D_MODEL))
        recipe = _recipe(patterns=["net.0"], activation_amax={"net.0": 1.0, "subnet.0": 1.0}, expect_replaced=1)

        assert build_fp8_encoder(perception, recipe) == 1
        assert isinstance(perception.encoder.net[0], FP8Linear)
        assert isinstance(perception.encoder.subnet[0], nn.Linear)

    def test_plain_encoder_without_asr_branch(self):
        perception = nn.Module()
        perception.encoder = _Encoder()

        assert build_fp8_encoder(perception, _recipe()) == N_LAYERS * len(PATTERNS)

    @pytest.mark.parametrize(
        ("overrides", "error", "match"),
        [
            ({"format": "nvfp4"}, ValueError, "format"),
            ({"weight_scale": "per_tensor"}, ValueError, "weight_scale"),
            ({"activation_scale": "per_block"}, ValueError, "activation_scale"),
            ({"patterns": []}, ValueError, "patterns"),
            ({"activation_amax": {"layers.0.attn.w_qkv": 1.0}}, ValueError, "activation_amax"),
            ({"expect_replaced": 7}, RuntimeError, "expect_replaced"),
        ],
    )
    def test_rejects_inconsistent_recipes_before_changing_the_encoder(self, overrides, error, match):
        perception = _speaker_aware_perception()

        with pytest.raises(error, match=match):
            build_fp8_encoder(perception, _recipe(**overrides))
        assert not any(isinstance(m, FP8Linear) for m in perception.modules())

    def test_bfloat16_cast_keeps_fp8_weights_and_float32_scales(self):
        perception = _speaker_aware_perception()
        build_fp8_encoder(perception, _recipe())
        layer = perception.encoder.asr_encoder.layers[0].attn.w_qkv
        weight = torch.randn(layer.weight.shape).to(FP8_DTYPE)
        weight_scale = torch.rand(layer.out_features) * 1e-3 + 1e-4  # not representable in bfloat16
        layer.load_state_dict(
            {"weight": weight, "weight_scale": weight_scale, "bias": torch.zeros(layer.out_features)}
        )
        input_scale = layer.input_scale.clone()

        perception.to(torch.bfloat16)

        assert layer.weight.dtype == FP8_DTYPE
        assert torch.equal(_fp8_bits(layer.weight), _fp8_bits(weight))
        assert layer.weight_scale.dtype == torch.float32
        assert torch.equal(layer.weight_scale, weight_scale)
        assert torch.equal(layer.input_scale, input_scale)
        assert layer.bias.dtype == torch.bfloat16

    def test_checkpoint_state_dict_fills_fp8_modules(self):
        perception = _speaker_aware_perception()
        build_fp8_encoder(perception, _recipe())
        perception.to(torch.bfloat16)  # the plugin casts perception right before loading

        state = {}
        for name, tensor in perception.state_dict().items():
            if tensor.dtype == FP8_DTYPE:
                state[name] = torch.randn(tensor.shape).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            else:
                state[name] = torch.randn(tensor.shape).to(tensor.dtype)
        result = perception.load_state_dict(state, strict=True)

        assert not result.missing_keys and not result.unexpected_keys
        w_qkv = perception.encoder.asr_encoder.layers[0].attn.w_qkv
        assert w_qkv.weight.dtype == FP8_DTYPE
        assert torch.equal(_fp8_bits(w_qkv.weight), _fp8_bits(state["encoder.asr_encoder.layers.0.attn.w_qkv.weight"]))
        assert w_qkv.weight_scale.dtype == torch.float32
        assert torch.equal(w_qkv.weight_scale, state["encoder.asr_encoder.layers.0.attn.w_qkv.weight_scale"])

    def test_plugin_perception_loader_loads_fp8_checkpoint(self):
        pytest.importorskip("vllm")
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        perception = _speaker_aware_perception()
        build_fp8_encoder(perception, _recipe())
        state = {
            name: (
                torch.randn(t.shape).to(FP8_DTYPE)
                if t.dtype == FP8_DTYPE
                else torch.randn(t.shape).to(torch.bfloat16 if t.is_floating_point() else t.dtype)
            )
            for name, t in perception.state_dict().items()
        }
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


@pytest.mark.skipif(not _HAS_PLUGIN or not torch.cuda.is_available(), reason="needs CUDA")
def test_device_move_keeps_fp8_dtypes():
    perception = _speaker_aware_perception()
    build_fp8_encoder(perception, _recipe())

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
@pytest.mark.parametrize("static", [True, False])
def test_fp8_linear_matches_dequantized_reference(static):
    torch.manual_seed(0)
    weight = torch.randn(64, 48, device="cuda")
    weight_scale = weight.abs().amax(dim=1) / FP8_MAX
    weight_fp8 = (weight / weight_scale[:, None]).to(FP8_DTYPE)
    bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(5, 7, 48, device="cuda", dtype=torch.bfloat16)
    input_scale = x.float().abs().amax() / FP8_MAX if static else None

    layer = FP8Linear(48, 64, bias=True, input_scale=input_scale, device="cuda")
    layer.load_state_dict({"weight": weight_fp8, "weight_scale": weight_scale, "bias": bias})
    out = layer(x)

    # Quantize the activations the way vLLM's kernels do (multiplying by the reciprocal scale),
    # so the comparison isolates the GEMM's scaling.
    x_scale = input_scale if static else x.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX
    x_dequant = (x.float() * (1.0 / x_scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float() * x_scale
    reference = x_dequant @ (weight_fp8.float() * weight_scale[:, None]).t() + bias.float()
    assert out.shape == (5, 7, 64)
    if static:
        torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.1)
    else:
        # The kernel's per-token scales round a few activations to a neighbouring FP8 code,
        # so compare in aggregate; a misapplied scale would be off by order 1.
        assert (out.float() - reference).norm() / reference.norm() < 0.01

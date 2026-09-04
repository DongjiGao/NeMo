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

"""Tests for NVFP4 global-scale folding.

The mocked tests run anywhere and cover configuration, exact-FQN wrapping, and every fail-closed path. The CUDA
tests build real TorchAO 0.17 static NVFP4 linears and are skipped only when the required CUDA, TorchAO or MSLK
APIs are genuinely unavailable. No test asserts on wall-clock time.
"""

import copy
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_nvfp4_scale_fold as sf

K_FEATURES = 64
N_FEATURES = 128


def _torch_dtypes_available():
    """Whether this torch build exposes the FP4/FP8 dtypes the folded contract depends on."""
    return all(isinstance(getattr(torch, name, None), torch.dtype) for name in sf.REQUIRED_TORCH_DTYPES)


def _cpu_fp8_roundtrip_available():
    """Whether float32 <-> float8_e4m3fn conversion works on CPU, which the mocked setup path needs."""
    if not _torch_dtypes_available():
        return False
    try:
        torch.zeros(4).to(torch.float8_e4m3fn).to(torch.float32)
        torch.zeros(4, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    except (RuntimeError, TypeError):
        return False
    return True


requires_cpu_fp4_fp8 = pytest.mark.skipif(
    not _cpu_fp8_roundtrip_available(),
    reason="this torch build cannot represent NVFP4/FP8 tensors on CPU",
)


def _transposed_storage(tensor):
    """Transpose an FP4/FP8 tensor through its uint8 storage.

    CPU has no ``copy_`` for ``torch.float4_e2m1fn_x2``, so materializing a transpose in that dtype is impossible
    here. Byte storage transposes fine and carries exactly the same shape and layout information, which is all the
    folded path reads: it accepts packed data and blocked scales as either the narrow dtype or its uint8 storage.
    """
    storage = tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)
    return storage.t().contiguous().view(tensor.dtype)


class _FakeNVFP4Tensor:
    """Stand-in for ``torchao.prototype.mx_formats.NVFP4Tensor`` with the attributes the folded path reads."""

    def __init__(
        self,
        qdata,
        scale,
        per_tensor_scale,
        act_per_tensor_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=SimpleNamespace(use_triton_kernel=True),
    ):
        self.qdata = qdata
        self.scale = scale
        self.per_tensor_scale = per_tensor_scale
        self.act_per_tensor_scale = act_per_tensor_scale
        self.is_swizzled_scales = is_swizzled_scales
        self.act_quant_kwargs = act_quant_kwargs

    def t(self):
        """Transposed view, exposing exactly what TorchAO hands to the native GEMM.

        Only the attributes this stand-in actually carries are exposed, so that a test which deletes one of them
        reaches the folded path's own fail-closed check instead of an AttributeError here.
        """
        transposed = {
            name: _transposed_storage(getattr(self, name)) for name in ("qdata", "scale") if hasattr(self, name)
        }
        return SimpleNamespace(**transposed)

    @classmethod
    def to_nvfp4(cls, data, per_tensor_scale=None, is_swizzled_scales=False, use_triton_kernel=False):
        """Only the keyword contract matters for the mocked tests; the forward path is exercised on CUDA."""
        raise AssertionError("the mocked NVFP4 packing path must not be executed")


def _fake_weight(**overrides):
    """A converted-weight stand-in whose blocked scales are deliberately non-square."""
    values = dict(
        qdata=torch.zeros(N_FEATURES, K_FEATURES // 2, dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
        scale=torch.full((N_FEATURES, K_FEATURES // 16), 2.0).to(torch.float8_e4m3fn),
        per_tensor_scale=torch.tensor([0.5]),
        act_per_tensor_scale=torch.tensor([0.25]),
    )
    values.update(overrides)
    return _FakeNVFP4Tensor(**values)


def _converted_linear(weight=None, bias=True, in_features=K_FEATURES):
    """An ``nn.Linear`` whose weight has been replaced by a converted-weight stand-in, as TorchAO leaves it."""
    linear = torch.nn.Linear(in_features, N_FEATURES, bias=bias).to(torch.bfloat16).eval()
    del linear.weight
    linear.weight = _fake_weight() if weight is None else weight
    return linear


def _model_with(fqn="encoder.layers.0.attn.w_qkv", **kwargs):
    """A tiny module tree holding one converted linear at ``fqn`` and one untouched BF16 linear beside it."""
    layer = torch.nn.Module()
    layer.attn = torch.nn.Module()
    layer.attn.w_qkv = _converted_linear(**kwargs)
    layer.attn.out_proj = torch.nn.Linear(K_FEATURES, N_FEATURES).to(torch.bfloat16)
    layers = torch.nn.ModuleList([layer])
    encoder = torch.nn.Module()
    encoder.layers = layers
    model = torch.nn.Module()
    model.encoder = encoder
    assert dict(model.named_modules())[fqn] is layer.attn.w_qkv
    return model


@pytest.mark.unit
def test_importing_the_module_does_not_import_torchao_or_mslk(monkeypatch):
    real_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torchao", "mslk"), f"scale folding imported '{name}' eagerly"
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    for name in [module for module in sys.modules if module.endswith("sortformer_nvfp4_scale_fold")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("nemo.collections.asr.parts.utils.sortformer_nvfp4_scale_fold")


@pytest.mark.unit
@pytest.mark.parametrize("exponent", [sf.FOLD_EXPONENT_MIN, -12, -10, 0, sf.FOLD_EXPONENT_MAX])
def test_valid_exponents_are_accepted(exponent):
    assert sf.validate_fold_exponent(exponent) == exponent


@pytest.mark.unit
@pytest.mark.parametrize(
    "exponent", [sf.FOLD_EXPONENT_MIN - 1, sf.FOLD_EXPONENT_MAX + 1, True, False, 1.0, "-10", None]
)
def test_invalid_exponents_are_rejected(exponent):
    with pytest.raises(ValueError, match="quantization_fold_activation_exponent"):
        sf.validate_fold_exponent(exponent)


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_only_the_listed_fqns_are_wrapped():
    model = _model_with()

    summary = sf.apply_global_scale_folding(
        model, ["encoder.layers.0.attn.w_qkv"], -10, nvfp4_tensor_cls=_FakeNVFP4Tensor
    )

    wrapped = model.get_submodule("encoder.layers.0.attn.w_qkv")
    assert isinstance(wrapped, sf.ScaleFoldedNVFP4Linear)
    assert wrapped.in_features == K_FEATURES and wrapped.out_features == N_FEATURES
    assert wrapped.bias is not None and wrapped.training is False
    # The sibling linear, which the recipe did not select, is untouched.
    assert isinstance(model.get_submodule("encoder.layers.0.attn.out_proj"), torch.nn.Linear)
    assert summary == {
        "enabled": True,
        "activation_exponent": -10,
        "activation_scale_factor": 2.0**-10,
        "wrapped_count": 1,
        "wrapped_fqns": ["encoder.layers.0.attn.w_qkv"],
        "notes": summary["notes"],
    }
    assert summary["notes"]


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_training_state_and_quantized_weight_are_preserved():
    model = _model_with()
    original_weight = model.get_submodule("encoder.layers.0.attn.w_qkv").weight
    original_bias = model.get_submodule("encoder.layers.0.attn.w_qkv").bias
    model.train()

    sf.apply_global_scale_folding(model, ["encoder.layers.0.attn.w_qkv"], -10, nvfp4_tensor_cls=_FakeNVFP4Tensor)

    wrapped = model.get_submodule("encoder.layers.0.attn.w_qkv")
    assert wrapped.training is True
    assert wrapped.weight is original_weight
    assert wrapped.bias is original_bias
    assert wrapped.weight_qdata.dtype == torch.float4_e2m1fn_x2


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_folded_weight_scale_keeps_the_original_orientation_and_value():
    weight = _fake_weight()
    module = sf.ScaleFoldedNVFP4Linear(
        _converted_linear(weight=weight), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_FakeNVFP4Tensor
    )

    # ``weight.t().scale.t()`` is the original ``weight.scale``: transposing it again would be a double transpose.
    assert tuple(weight.t().scale.t().shape) == tuple(weight.scale.shape)
    assert tuple(module.weight_scale_folded.shape) == tuple(weight.scale.shape)
    assert tuple(module.weight_scale_folded.shape) != tuple(reversed(weight.scale.shape))

    product = 0.5 * 0.25
    expected = (weight.scale.to(torch.float32) * (product / 2.0**-10)).to(torch.float8_e4m3fn)
    assert torch.equal(module.weight_scale_folded.to(torch.float32), expected.to(torch.float32))
    assert module.weight_scale_folded.dtype == torch.float8_e4m3fn
    assert module.activation_scale_factor == 2.0**-10
    assert float(module.global_scale_product) == pytest.approx(product)


@requires_cpu_fp4_fp8
@pytest.mark.unit
@pytest.mark.parametrize(
    "scales, exponent",
    [
        ((1e-30, 1e-8), sf.FOLD_EXPONENT_MAX),  # underflows float8_e4m3fn after folding
        ((1e30, 1e10), sf.FOLD_EXPONENT_MIN),  # overflows float8_e4m3fn after folding
    ],
)
def test_a_fold_that_destroys_weight_scales_fails_closed(scales, exponent):
    activation_scale, weight_scale = scales
    weight = _fake_weight(
        per_tensor_scale=torch.tensor([weight_scale]), act_per_tensor_scale=torch.tensor([activation_scale])
    )

    with pytest.raises(RuntimeError, match="destroys"):
        sf.ScaleFoldedNVFP4Linear(
            _converted_linear(weight=weight), fqn="w_qkv", exponent=exponent, nvfp4_tensor_cls=_FakeNVFP4Tensor
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_a_missing_fqn_fails_closed():
    model = _model_with()

    with pytest.raises(RuntimeError, match="could not find the selected module"):
        sf.apply_global_scale_folding(model, ["encoder.layers.1.attn.w_qkv"], -10, nvfp4_tensor_cls=_FakeNVFP4Tensor)


@pytest.mark.unit
def test_folding_nothing_is_rejected():
    with pytest.raises(ValueError, match="nothing to fold"):
        sf.apply_global_scale_folding(torch.nn.Module(), [], -10, nvfp4_tensor_cls=_FakeNVFP4Tensor)


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_an_unconverted_linear_fails_closed():
    model = _model_with()

    with pytest.raises(RuntimeError, match="must run after the static NVFP4 conversion"):
        sf.apply_global_scale_folding(
            model, ["encoder.layers.0.attn.out_proj"], -10, nvfp4_tensor_cls=_FakeNVFP4Tensor
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_a_dynamic_activation_scale_fails_closed():
    weight = _fake_weight(act_per_tensor_scale=None)

    with pytest.raises(RuntimeError, match="scale_mode='static'"):
        sf.ScaleFoldedNVFP4Linear(
            _converted_linear(weight=weight), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_FakeNVFP4Tensor
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_unswizzled_scales_fail_closed():
    weight = _fake_weight(is_swizzled_scales=False)

    with pytest.raises(RuntimeError, match="swizzled"):
        sf.ScaleFoldedNVFP4Linear(
            _converted_linear(weight=weight), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_FakeNVFP4Tensor
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_a_k_dimension_that_cannot_be_accelerated_fails_closed():
    weight = _fake_weight(
        qdata=torch.zeros(N_FEATURES, 16, dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
        scale=torch.full((N_FEATURES, 2), 2.0).to(torch.float8_e4m3fn),
    )

    with pytest.raises(RuntimeError, match="in_features % 64 == 0"):
        sf.ScaleFoldedNVFP4Linear(
            _converted_linear(weight=weight, in_features=32),
            fqn="w_qkv",
            exponent=-10,
            nvfp4_tensor_cls=_FakeNVFP4Tensor,
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_missing_torchao_metadata_fails_closed():
    weight = _fake_weight()
    del weight.scale

    with pytest.raises(RuntimeError, match="requires 'scale'"):
        sf.ScaleFoldedNVFP4Linear(
            _converted_linear(weight=weight), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_FakeNVFP4Tensor
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_a_torchao_packing_signature_mismatch_fails_closed():
    class _StalePackingTensor(_FakeNVFP4Tensor):
        @classmethod
        def to_nvfp4(cls, data, per_tensor_scale=None):  # missing the swizzle/triton keywords
            raise AssertionError("unreachable")

    weight = _StalePackingTensor(
        qdata=torch.zeros(N_FEATURES, K_FEATURES // 2, dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
        scale=torch.full((N_FEATURES, K_FEATURES // 16), 2.0).to(torch.float8_e4m3fn),
        per_tensor_scale=torch.tensor([0.5]),
        act_per_tensor_scale=torch.tensor([0.25]),
    )

    with pytest.raises(RuntimeError, match="is missing"):
        sf.ScaleFoldedNVFP4Linear(
            _converted_linear(weight=weight), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_StalePackingTensor
        )


@requires_cpu_fp4_fp8
@pytest.mark.unit
@pytest.mark.parametrize("shape", [(K_FEATURES,), (2, 2, 2, K_FEATURES)])
def test_unsupported_input_ranks_are_rejected(shape):
    module = sf.ScaleFoldedNVFP4Linear(
        _converted_linear(), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_FakeNVFP4Tensor
    )

    with pytest.raises(ValueError, match="supports rank"):
        module(torch.zeros(shape, dtype=torch.bfloat16))


@requires_cpu_fp4_fp8
@pytest.mark.unit
def test_a_wrong_feature_count_is_rejected():
    module = sf.ScaleFoldedNVFP4Linear(
        _converted_linear(), fqn="w_qkv", exponent=-10, nvfp4_tensor_cls=_FakeNVFP4Tensor
    )

    with pytest.raises(ValueError, match="expects a last dimension"):
        module(torch.zeros(4, K_FEATURES + 64, dtype=torch.bfloat16))


# --------------------------------------------------------------------------------------------------------------
# CUDA tests against real TorchAO 0.17 static NVFP4 conversions.
# --------------------------------------------------------------------------------------------------------------


def _nvfp4_runtime_available():
    """Whether a real static NVFP4 conversion can be built and executed on this machine."""
    if not torch.cuda.is_available() or not _torch_dtypes_available():
        return False
    if torch.cuda.get_device_capability() < (10, 0):
        return False
    for api in ("torchao.quantization:quantize_", "torchao.prototype.mx_formats:NVFP4ObservedLinear"):
        module_name, _, attribute = api.partition(":")
        try:
            if getattr(importlib.import_module(module_name), attribute, None) is None:
                return False
        except ImportError:
            return False
    try:
        importlib.import_module("mslk.quantize.triton.fp4_quantize")
    except ImportError:
        return False
    return True


requires_nvfp4_runtime = pytest.mark.skipif(
    not _nvfp4_runtime_available(),
    reason="requires a Blackwell CUDA device with TorchAO NVFP4 and MSLK available",
)

CUDA_K = 512
CALIBRATION_TOKENS = 256


def _build_static_nvfp4_linear(out_features, bias=True, seed=0):
    """Convert one BF16 linear with TorchAO's static NVFP4 recipe and return it with its BF16 original."""
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig, NVFP4ObservedLinear
    from torchao.quantization import quantize_

    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(CUDA_K, out_features, bias=bias))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    reference = copy.deepcopy(model)

    def config(step):
        return NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=True, use_dynamic_per_tensor_scale=True, step=step
        )

    quantize_(model, config("prepare"), filter_fn=lambda module, fqn: isinstance(module, torch.nn.Linear))
    with torch.no_grad():
        model(torch.randn(CALIBRATION_TOKENS, CUDA_K, device="cuda", dtype=torch.bfloat16))
    quantize_(model, config("convert"), filter_fn=lambda module, fqn: isinstance(module, NVFP4ObservedLinear))
    return model, model[0], reference


def _relative_error(actual, expected):
    """Mean absolute deviation normalized by the reference magnitude."""
    actual, expected = actual.float(), expected.float()
    return float((actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-6))


@requires_nvfp4_runtime
@pytest.mark.unit
@pytest.mark.parametrize("out_features", [512, 1536])
@pytest.mark.parametrize("leading", [(CALIBRATION_TOKENS,), (2, CALIBRATION_TOKENS)])
def test_folded_output_matches_the_unfolded_static_nvfp4_path(out_features, leading):
    quantized, linear, reference = _build_static_nvfp4_linear(out_features)
    x = torch.randn(*leading, CUDA_K, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        unfolded = quantized(x)
        baseline = reference(x)
        module = sf.ScaleFoldedNVFP4Linear(linear, fqn="w_qkv", exponent=-10)
        folded = module(x)

    assert folded.shape == (*leading, out_features)
    assert folded.dtype == torch.bfloat16
    assert folded.device == x.device
    assert torch.isfinite(folded.float()).all()

    # Folding re-rounds every weight block scale into float8_e4m3fn against a rebased exponent, so the folded and
    # unfolded static-NVFP4 paths are close but never bit-identical. A measured TorchAO 0.17 / MSLK exponent sweep
    # (M=4096, K=512, N=1536) puts the folded-vs-unfolded deviation at 0.0366 for exponent -13 through 0.0401 for
    # the default -10, rising to 0.0594 at -9 and 0.2887 at -8; the corresponding deviation from BF16 stays at
    # 0.1435-0.1438 against 0.1395 for the unfolded static path. The bounds below clear those measured values with
    # room for the smaller shapes exercised here, while still rejecting the 0.29 of a two-step exponent slip and
    # the O(1) of a transposed operand or a dropped global scale. They are a layout/scale regression guard only --
    # product accuracy is gated on final DER, not on this unit bound.
    unfolded_deviation = _relative_error(unfolded, baseline)
    assert _relative_error(folded, unfolded) < 0.08
    assert _relative_error(folded, baseline) < 0.20
    assert _relative_error(folded, baseline) < 1.25 * unfolded_deviation
    with torch.no_grad():
        assert torch.equal(module(x), folded)


@requires_nvfp4_runtime
@pytest.mark.unit
def test_the_gemm_applies_the_bias_natively():
    quantized, linear, _ = _build_static_nvfp4_linear(512, bias=True)
    x = torch.randn(CALIBRATION_TOKENS, CUDA_K, device="cuda", dtype=torch.bfloat16)
    module = sf.ScaleFoldedNVFP4Linear(linear, fqn="w_qkv", exponent=-10)
    bias = module.bias.detach().clone()

    with torch.no_grad():
        with_bias = module(x)
        module.bias = None
        without_bias = module(x)

    assert _relative_error(with_bias, without_bias + bias) < 0.01


@requires_nvfp4_runtime
@pytest.mark.unit
def test_folded_weight_scale_is_not_transposed_twice():
    _, linear, _ = _build_static_nvfp4_linear(1536)
    module = sf.ScaleFoldedNVFP4Linear(linear, fqn="w_qkv", exponent=-10)
    weight_scale = linear.weight.scale

    assert tuple(module.weight_scale_folded.shape) == tuple(weight_scale.shape)
    assert tuple(linear.weight.t().scale.t().shape) == tuple(weight_scale.shape)
    if weight_scale.dim() == 2 and weight_scale.shape[0] != weight_scale.shape[1]:
        assert tuple(module.weight_scale_folded.shape) != tuple(reversed(weight_scale.shape))
    assert module.weight_scale_folded.dtype == torch.float8_e4m3fn
    assert module.weight_scale_folded.is_contiguous()


@requires_nvfp4_runtime
@pytest.mark.unit
def test_an_overflowing_fold_fails_closed_on_real_metadata():
    # Overwriting ``per_tensor_scale`` on the converted weight is not a reliable way to force this: a TorchAO
    # tensor subclass may keep serving the state the folded path reads from elsewhere, so the fold stays healthy
    # and the test passes vacuously. Instead the fold is driven off genuinely calibrated metadata into an exponent
    # that a measured sweep shows is destructive: at exponent -7, 99 of this layer's 49152 real weight block
    # scales overflow float8_e4m3fn once rebased by the calibrated global-scale product.
    _, linear, _ = _build_static_nvfp4_linear(1536)
    healthy = sf.ScaleFoldedNVFP4Linear(linear, fqn="w_qkv", exponent=-10)
    assert torch.isfinite(healthy.weight_scale_folded.to(torch.float32)).all()

    with pytest.raises(RuntimeError, match="destroys"):
        sf.ScaleFoldedNVFP4Linear(linear, fqn="w_qkv", exponent=-7)


@requires_nvfp4_runtime
@pytest.mark.unit
def test_the_folded_module_compiles_for_a_fixed_shape():
    _, linear, _ = _build_static_nvfp4_linear(1536)
    module = sf.ScaleFoldedNVFP4Linear(linear, fqn="w_qkv", exponent=-10)
    x = torch.randn(CALIBRATION_TOKENS, CUDA_K, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        eager = module(x)
        compiled = torch.compile(module, dynamic=False)(x)

    assert compiled.shape == eager.shape
    assert _relative_error(compiled, eager) < 1e-3

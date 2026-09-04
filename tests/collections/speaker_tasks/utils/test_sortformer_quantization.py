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

import base64
import hashlib
import importlib
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_nvfp4_producer_fusion as producer_fusion
from nemo.collections.asr.parts.utils import sortformer_nvfp4_scale_fold as scale_fold
from nemo.collections.asr.parts.utils import sortformer_quantization as sq

D_MODEL = 64
FF_HIDDEN = 256
NUM_LAYERS = 2


class _FakeFeedForward(torch.nn.Module):
    """Feed-forward block mirroring the Sortformer transformer encoder's ``net.0`` / ``net.3`` layout."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D_MODEL, FF_HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(FF_HIDDEN, D_MODEL),
            torch.nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class _FakeAttention(torch.nn.Module):
    """Attention block exposing the ``w_qkv`` / ``out_proj`` target names."""

    def __init__(self):
        super().__init__()
        self.w_qkv = torch.nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out_proj = torch.nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        return self.out_proj(self.w_qkv(x)[..., :D_MODEL])


class _FakeLayer(torch.nn.Module):
    """Single transformer layer with the norms that must never be quantized."""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(D_MODEL)
        self.attn = _FakeAttention()
        self.norm2 = torch.nn.LayerNorm(D_MODEL)
        self.ffn = _FakeFeedForward()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _FakeEncoder(torch.nn.Module):
    """Stack of transformer layers."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(NUM_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _FakeSortformer(torch.nn.Module):
    """Sortformer stand-in: transformer targets plus decoys that must never be selected."""

    def __init__(self):
        super().__init__()
        self.pre_encode = torch.nn.Linear(D_MODEL, D_MODEL)
        self.transformer_encoder = _FakeEncoder()
        # Model-level projection: shares the ``out_proj`` name but not the anchored ``attn.out_proj`` suffix.
        self.out_proj = torch.nn.Linear(D_MODEL, D_MODEL)
        self.head = torch.nn.Linear(D_MODEL, 4)

    def forward(self, x):
        return self.head(self.out_proj(self.transformer_encoder(self.pre_encode(x))))


class _FakeOptimizedModule(torch.nn.Module):
    """Stand-in for ``torch._dynamo.OptimizedModule``: re-parents the wrapped module under ``_orig_mod``."""

    def __init__(self, module):
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


def _expected_target_fqns():
    """Every FQN that must be selected, grouped nowhere: a flat sorted list."""
    return sorted(
        f"transformer_encoder.layers.{index}.{suffix}"
        for index in range(NUM_LAYERS)
        for suffix in sq.QUANTIZATION_TARGET_SUFFIXES
    )


def _fqns_for_suffix(suffix):
    """FQNs of a single target family across all layers."""
    return sorted(f"transformer_encoder.layers.{index}.{suffix}" for index in range(NUM_LAYERS))


def _facts(**overrides):
    """Fully capable injected runtime facts, overridable per test."""
    values = dict(
        device_type="cuda",
        compute_capability=(12, 0),
        torch_version="2.12.0+cu132",
        torchao_version="0.17.0",
        mslk_version="1.2.0",
        available_apis=sq._ALL_TORCHAO_APIS,
        available_dtypes=sq.REQUIRED_TORCH_DTYPES,
    )
    values.update(overrides)
    return sq.CapabilityFacts(**values)


def _eval_cfg(**overrides):
    """Minimal stand-in for the evaluator's ``DiarizationConfig`` quantization fields."""
    values = dict(
        quantization_recipe="disabled",
        quantization_scale_mode="dynamic",
        quantization_calibration_path=None,
        quantization_calibration_output=None,
        quantization_scale_margin=1.0,
        quantization_accelerated_packing=True,
        quantization_allow_reference_kernels=False,
        quantization_overwrite_calibration=False,
        quantization_fold_global_scales=False,
        quantization_fold_activation_exponent=-10,
        quantization_fuse_producer_packing=False,
        quantization_bf16_override_path=None,
        quantization_weight_scale_method="amax",
        quantization_weight_scale_hessian_path=None,
        quantization_weight_scale_awq_clip_path=None,
        quantization_weight_scale_gptq_path=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# Deterministic stand-ins for the two NVFP4 reconstructions: the ordinary amax-derived conversion is modelled as a
# coarse grid and the block-MSE repack as a finer one, so the searched error is always the smaller of the two.
TEMPLATE_GRID_STEP = 0.05
SEARCHED_GRID_STEP = 0.005


class _FakeNVFP4Tensor(torch.Tensor):
    """Tensor subclass standing in for torchao's ``NVFP4Tensor``: it carries its own dequantized values."""

    # ``_facts()`` resolves the MSLK-accelerated backend, so the conversion these fakes stand in for is the one
    # torchao performs with the Triton kernel on. The AWQ-clip repack re-checks this against the construction its
    # artifact was selected against, so a fake that claimed the other one would describe a different template.
    use_triton_kernel = True

    def dequantize(self, dtype=torch.float32):
        """Pinned torchao 0.17 spells the dequantization ``dequantize(output_dtype)``."""
        return self.detach().to(dtype)

    # The three buffers ``nvfp4_template_identity`` reads off a real wrapper, as *properties* rather than instance
    # attributes: ``torch.nn.Parameter`` re-wraps a custom tensor subclass through ``detach()``, which keeps the
    # class but not an instance ``__dict__``, so only a class-level definition survives the assignment a
    # conversion makes. The payload is derived from the wrapper's own values -- replacing it is exactly what GPTQ
    # does -- while the scale buffer and the global scale are derived from the *shape* alone, which is how a real
    # GPTQ repack behaves: it carries the template's scales through unchanged however the payload was rewritten.
    @property
    def qdata(self):
        """Deterministic packed-payload stand-in: one byte per two weights, derived from the stored values."""
        rows, columns = int(self.shape[0]), int(self.shape[1])
        codes = (self.detach().float().abs() * 1000.0).to(torch.int64) % 256
        return codes.reshape(rows, columns)[:, : columns // 2].to(torch.uint8).contiguous()

    @property
    def _scale_e4m3(self):
        """Deterministic swizzled scale-buffer stand-in, padded like the Blackwell layout and shape-derived."""
        rows, columns = int(self.shape[0]), int(self.shape[1])
        padded_rows = ((rows + 127) // 128) * 128
        padded_blocks = ((columns // sq.NVFP4_BLOCK_SIZE + 3) // 4) * 4
        positions = torch.arange(padded_rows * padded_blocks, dtype=torch.int64)
        return ((positions + rows + columns) % 251).to(torch.uint8).contiguous()

    @property
    def _per_tensor_scale(self):
        """Deterministic global-scale stand-in; a real one is derived from the weight and is never re-derived."""
        return torch.full((), 1.0 + int(self.shape[0]) * 0.001 + int(self.shape[1]) * 0.0001)


def _fake_nvfp4(values):
    """Wrap plain values in the wrapper class a completed NVFP4 conversion leaves behind."""
    return _FakeNVFP4Tensor._make_subclass(_FakeNVFP4Tensor, values.detach().float().contiguous(), False)


def _round_to(values, step):
    """Snap weights onto a fixed grid; a coarser step is a worse reconstruction of the same weight."""
    return torch.round(values.detach().float() / step) * step


class _FakeNVFP4DynamicConfig:
    """Stand-in for ``NVFP4DynamicActivationNVFP4WeightConfig`` with the TorchAO 0.17 keyword contract."""

    def __init__(self, use_triton_kernel=False, use_dynamic_per_tensor_scale=False, step=None):
        self.use_triton_kernel = use_triton_kernel
        self.use_dynamic_per_tensor_scale = use_dynamic_per_tensor_scale
        self.step = step


class _FakeNVFP4WeightOnlyConfig:
    """Stand-in for ``NVFP4WeightOnlyConfig``."""

    def __init__(self, use_dynamic_per_tensor_scale=False):
        self.use_dynamic_per_tensor_scale = use_dynamic_per_tensor_scale
        self.step = None


class _FakeFP8Config:
    """Stand-in for ``Float8DynamicActivationFloat8WeightConfig``."""

    def __init__(self):
        self.step = None


class _FakeObservedLinear(torch.nn.Module):
    """Stand-in for ``NVFP4ObservedLinear``: carries the direct ``amax`` tensor that calibration writes into."""

    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        self.in_features = linear.in_features
        self.amax = torch.tensor(0.0)


class _FakeQuantizedLinear(torch.nn.Module):
    """Marker left behind by a completed conversion; carries the NVFP4 weight parameter TorchAO leaves behind."""

    def __init__(self, source):
        super().__init__()
        self.in_features = getattr(source, "in_features", 0)
        weight = getattr(source, "weight", None)
        if isinstance(weight, torch.Tensor):
            self.weight = torch.nn.Parameter(_fake_nvfp4(_round_to(weight, TEMPLATE_GRID_STEP)), requires_grad=False)


def _swap_module(model, fqn, replacement):
    """Replace a submodule in place, the way ``quantize_`` does."""
    parent_name, _, attribute = fqn.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, replacement)


class _FakeQuantize:
    """Records every ``quantize_`` call and performs the prepare/convert module swaps."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, config, filter_fn=None):
        selected = sorted(name for name, module in model.named_modules() if filter_fn(module, name))
        call = SimpleNamespace(config=config, selected=selected, observed_amax={})
        step = getattr(config, "step", None)
        for name in selected:
            module = model.get_submodule(name)
            if step == "prepare":
                _swap_module(model, name, _FakeObservedLinear(module))
            elif step == "convert":
                call.observed_amax[name] = float(module.amax)
                _swap_module(model, name, _FakeQuantizedLinear(module))
            else:
                _swap_module(model, name, _FakeQuantizedLinear(module))
        self.calls.append(call)


@pytest.fixture
def fake_torchao(monkeypatch):
    """Replace every TorchAO entry point with an in-process fake."""
    quantize = _FakeQuantize()
    registry = {
        sq.TORCHAO_QUANTIZE_API: quantize,
        sq.TORCHAO_NVFP4_DYNAMIC_CONFIG_API: _FakeNVFP4DynamicConfig,
        sq.TORCHAO_NVFP4_WEIGHT_ONLY_CONFIG_API: _FakeNVFP4WeightOnlyConfig,
        sq.TORCHAO_NVFP4_OBSERVED_LINEAR_API: _FakeObservedLinear,
        sq.TORCHAO_FP8_CONFIG_API: _FakeFP8Config,
    }
    monkeypatch.setattr(sq, "_resolve_api", lambda api: registry.get(api))
    return quantize


@pytest.fixture
def fake_mse_repack(fake_torchao, monkeypatch):
    """
    Replace the accepted block-MSE repacker with a deterministic stand-in and record every call.

    Besides the arguments, each record keeps the number of ``quantize_`` calls made so far -- which pins the repack
    to its own conversion -- and whether any *earlier* original weight was still alive when this call started,
    which is how the one-clone-at-a-time memory contract is checked.
    """
    calls = []
    originals = []

    def repack(weight, template):
        calls.append(
            SimpleNamespace(
                weight=weight.detach().clone(),
                template=template,
                quantize_calls=len(fake_torchao.calls),
                earlier_originals_alive=[reference() is not None for reference in originals],
            )
        )
        originals.append(weakref.ref(weight))
        return _fake_nvfp4(_round_to(weight, SEARCHED_GRID_STEP))

    monkeypatch.setattr(sq, "repack_nvfp4_weight_mse", repack)
    return calls


@pytest.mark.unit
def test_selection_matches_exact_targets_and_excludes_everything_else():
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")

    assert sorted(selection.precision_by_fqn) == _expected_target_fqns()
    assert selection.fqns_by_suffix["attn.w_qkv"] == _fqns_for_suffix("attn.w_qkv")
    for excluded in ("pre_encode", "out_proj", "head"):
        assert excluded not in selection.precision_by_fqn
    assert not any(".norm" in fqn for fqn in selection.precision_by_fqn)


@pytest.mark.unit
@pytest.mark.parametrize(
    "recipe, expected_counts",
    [
        (
            "disabled",
            {sq.PRECISION_BF16: 4 * NUM_LAYERS},
        ),
        (
            "nvfp4_all",
            {sq.PRECISION_NVFP4_W4A4: 4 * NUM_LAYERS},
        ),
        (
            "nvfp4_qkv_only",
            {sq.PRECISION_BF16: 3 * NUM_LAYERS, sq.PRECISION_NVFP4_W4A4: NUM_LAYERS},
        ),
        (
            "nvfp4_qkv_fp8_rest",
            {sq.PRECISION_FP8_DYNAMIC: 3 * NUM_LAYERS, sq.PRECISION_NVFP4_W4A4: NUM_LAYERS},
        ),
        (
            "nvfp4_weight_only",
            {sq.PRECISION_NVFP4_WEIGHT_ONLY: 4 * NUM_LAYERS},
        ),
    ],
)
def test_recipe_precision_counts(recipe, expected_counts):
    selection = sq.select_quantization_targets(_FakeSortformer(), recipe)

    assert selection.counts_by_precision == dict(sorted(expected_counts.items()))
    if recipe in ("nvfp4_qkv_only", "nvfp4_qkv_fp8_rest"):
        assert selection.fqns_for_precision(sq.PRECISION_NVFP4_W4A4) == _fqns_for_suffix("attn.w_qkv")


@pytest.mark.unit
def test_unknown_recipe_is_rejected():
    with pytest.raises(ValueError, match="quantization recipe must be one of"):
        sq.select_quantization_targets(_FakeSortformer(), "nvfp4_everything")


@pytest.mark.unit
def test_missing_target_family_is_rejected_instead_of_silently_skipped():
    model = _FakeSortformer()
    del model.transformer_encoder.layers[1].attn.w_qkv
    del model.transformer_encoder.layers[0].attn.w_qkv

    with pytest.raises(ValueError, match="attn.w_qkv"):
        sq.select_quantization_targets(model, "nvfp4_all")


@pytest.mark.unit
def test_non_linear_target_is_rejected():
    model = _FakeSortformer()
    model.transformer_encoder.layers[0].attn.out_proj = torch.nn.Identity()

    with pytest.raises(ValueError, match="must be torch.nn.Linear"):
        sq.select_quantization_targets(model, "nvfp4_all")


@pytest.mark.unit
@pytest.mark.parametrize(
    "compute_capability, architecture",
    [((10, 0), "sm100"), ((10, 3), "sm103"), ((11, 0), "sm110"), ((12, 0), "sm120"), ((12, 1), "sm121")],
)
def test_accepted_compute_capabilities(compute_capability, architecture):
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")
    facts = _facts(compute_capability=compute_capability)

    assert sq.check_nvfp4_capability(config, facts) == sq.BACKEND_MSLK_ACCELERATED
    assert facts.architecture == architecture


@pytest.mark.unit
@pytest.mark.parametrize("compute_capability", [(9, 0), (8, 9), (12, 2), (10, 1), None])
def test_rejected_compute_capabilities(compute_capability):
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")

    with pytest.raises(RuntimeError, match="compute capabilit"):
        sq.check_nvfp4_capability(config, _facts(compute_capability=compute_capability))


@pytest.mark.unit
def test_non_cuda_device_is_rejected():
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        sq.check_nvfp4_capability(config, _facts(device_type="cpu", compute_capability=None))


@pytest.mark.unit
def test_capability_acceptance_is_policy_not_qualification():
    facts = _facts(compute_capability=(10, 3))
    assert facts.qualification == sq.QUALIFICATION_POLICY_ONLY

    qualified = _facts(compute_capability=(10, 3), qualified_compute_capabilities=((10, 3),))
    assert qualified.qualification == sq.QUALIFICATION_TESTED


@pytest.mark.unit
def test_disabled_recipe_reports_disabled_backend():
    assert sq.check_nvfp4_capability(sq.SortformerQuantizationConfig(), _facts(device_type="cpu")) == (
        sq.BACKEND_DISABLED
    )


@pytest.mark.unit
def test_missing_torchao_api_is_a_hard_error():
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")
    available = tuple(api for api in sq._ALL_TORCHAO_APIS if api != sq.TORCHAO_NVFP4_DYNAMIC_CONFIG_API)

    with pytest.raises(RuntimeError, match="does not provide the APIs required"):
        sq.check_nvfp4_capability(config, _facts(available_apis=available))


@pytest.mark.unit
def test_missing_torch_dtype_is_a_hard_error():
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")

    with pytest.raises(RuntimeError, match="does not expose the dtypes"):
        sq.check_nvfp4_capability(config, _facts(available_dtypes=("float8_e4m3fn",)))


@pytest.mark.unit
@pytest.mark.parametrize("mslk_version", [None, "1.1.9", "0.9"])
def test_missing_or_old_mslk_is_a_hard_error_without_fallback(mslk_version):
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")

    with pytest.raises(RuntimeError, match="MSLK"):
        sq.check_nvfp4_capability(config, _facts(mslk_version=mslk_version))


@pytest.mark.unit
def test_reference_kernels_require_explicit_permission():
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all", accelerated_packing=False)

    with pytest.raises(ValueError, match="quantization_allow_reference_kernels"):
        config.validate()


@pytest.mark.unit
def test_explicit_reference_backend_is_labelled(caplog):
    config = sq.SortformerQuantizationConfig(
        recipe="nvfp4_all", accelerated_packing=False, allow_reference_kernels=True
    )
    config.validate()

    with caplog.at_level("WARNING"):
        backend = sq.check_nvfp4_capability(config, _facts(mslk_version=None))

    assert backend == sq.BACKEND_REFERENCE_UNACCELERATED
    assert "UNACCELERATED" in caplog.text


@pytest.mark.unit
def test_weight_only_backend_does_not_require_mslk():
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_weight_only")
    config.validate()

    assert sq.check_nvfp4_capability(config, _facts(mslk_version=None)) == sq.BACKEND_WEIGHT_ONLY


@pytest.mark.unit
def test_dynamic_quantization_uses_documented_torchao_contract(fake_torchao):
    model = _FakeSortformer()
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all")

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    assert len(fake_torchao.calls) == 1
    call = fake_torchao.calls[0]
    assert isinstance(call.config, _FakeNVFP4DynamicConfig)
    assert (call.config.step, call.config.use_triton_kernel, call.config.use_dynamic_per_tensor_scale) == (
        None,
        True,
        True,
    )
    assert call.selected == _expected_target_fqns()
    assert summary["backend"] == sq.BACKEND_MSLK_ACCELERATED
    assert summary["scale_mode"] == "dynamic"


@pytest.mark.unit
def test_accelerated_backend_is_reported_as_conditional_and_unverified(fake_torchao):
    summary = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )

    acceleration = summary["acceleration"]
    assert acceleration["status"] == sq.ACCELERATION_CONDITIONAL
    assert acceleration["verified"] is False
    assert acceleration["m_constraint_checked"] is False
    assert acceleration["k_constraint_satisfied"] is True
    assert "128" in acceleration["m_constraint"]
    assert any("M % 128 == 0" in note for note in summary["notes"])


@pytest.mark.unit
def test_reference_backend_reports_unaccelerated_status(fake_torchao):
    config = sq.SortformerQuantizationConfig(
        recipe="nvfp4_all", accelerated_packing=False, allow_reference_kernels=True
    )

    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts(mslk_version=None))

    assert summary["backend"] == sq.BACKEND_REFERENCE_UNACCELERATED
    assert summary["acceleration"]["status"] == sq.ACCELERATION_UNACCELERATED
    assert summary["explicit_reference_kernels"] is True
    assert fake_torchao.calls[0].config.use_triton_kernel is False


@pytest.mark.unit
def test_k_constraint_violation_blocks_accelerated_packing(fake_torchao):
    model = _FakeSortformer()
    model.transformer_encoder.layers[0].attn.w_qkv = torch.nn.Linear(D_MODEL - 8, 3 * D_MODEL)

    with pytest.raises(RuntimeError, match="in_features % 64 == 0"):
        sq.quantize_sortformer_model(model, sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts())


@pytest.mark.unit
def test_mixed_recipe_applies_fp8_only_to_non_qkv_targets(fake_torchao):
    model = _FakeSortformer()

    summary = sq.quantize_sortformer_model(
        model, sq.SortformerQuantizationConfig(recipe="nvfp4_qkv_fp8_rest"), facts=_facts()
    )

    assert len(fake_torchao.calls) == 2
    assert fake_torchao.calls[0].selected == _fqns_for_suffix("attn.w_qkv")
    assert isinstance(fake_torchao.calls[1].config, _FakeFP8Config)
    assert fake_torchao.calls[1].selected == sorted(
        _fqns_for_suffix("attn.out_proj") + _fqns_for_suffix("ffn.net.0") + _fqns_for_suffix("ffn.net.3")
    )
    assert summary["counts_by_precision"] == {
        sq.PRECISION_FP8_DYNAMIC: 3 * NUM_LAYERS,
        sq.PRECISION_NVFP4_W4A4: NUM_LAYERS,
    }


@pytest.mark.unit
def test_weight_only_recipe_is_labelled_as_a_comparator(fake_torchao):
    summary = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_weight_only"), facts=_facts()
    )

    assert isinstance(fake_torchao.calls[0].config, _FakeNVFP4WeightOnlyConfig)
    assert fake_torchao.calls[0].config.use_dynamic_per_tensor_scale is True
    assert summary["backend"] == sq.BACKEND_WEIGHT_ONLY
    assert summary["acceleration"]["status"] == sq.ACCELERATION_NOT_APPLICABLE
    assert any("comparator" in note for note in summary["notes"])


def _write_calibration(tmp_path, amax_by_fqn, version=sq.CALIBRATION_SCHEMA_VERSION, name="calib.json"):
    """Write a calibration JSON file and return its path."""
    path = tmp_path / name
    payload = {
        "version": version,
        "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
        "activation_amax": amax_by_fqn,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.unit
def test_static_quantization_prepares_assigns_then_converts(tmp_path, fake_torchao):
    model = _FakeSortformer()
    path = _write_calibration(tmp_path, {fqn: 1.5 for fqn in _expected_target_fqns()})
    config = sq.SortformerQuantizationConfig(
        recipe="nvfp4_all", scale_mode="static", calibration_path=str(path), scale_margin=2.0
    )

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    assert [call.config.step for call in fake_torchao.calls] == ["prepare", "convert"]
    assert fake_torchao.calls[0].selected == _expected_target_fqns()
    # The margin-scaled amax was assigned to every observed linear before the convert step ran.
    assert fake_torchao.calls[1].observed_amax == {fqn: 3.0 for fqn in _expected_target_fqns()}
    assert all(call.config.use_triton_kernel for call in fake_torchao.calls)
    assert summary["scale_margin"] == 2.0
    assert summary["calibration_path"] == str(path)


@pytest.mark.unit
def test_activation_amax_is_written_to_the_torchao_direct_amax_tensor():
    linear = torch.nn.Linear(4, 4)
    observed = _FakeObservedLinear(linear)

    sq._assign_activation_amax(observed, "fake.fqn", 2.5)

    # TorchAO 0.17 reads ``module.amax`` at convert; the tensor must be updated in place-compatible form.
    assert isinstance(observed.amax, torch.Tensor)
    assert observed.amax.shape == torch.Size([])
    assert float(observed.amax) == pytest.approx(2.5)


@pytest.mark.unit
@pytest.mark.parametrize("attribute", ["act_obs", "act_amax"])
def test_activation_amax_falls_back_to_older_observer_layouts(attribute):
    observed = torch.nn.Linear(4, 4)
    if attribute == "act_obs":
        observed.act_obs = SimpleNamespace(amax=None)
    else:
        observed.act_amax = None

    sq._assign_activation_amax(observed, "fake.fqn", 2.5)

    written = observed.act_obs.amax if attribute == "act_obs" else observed.act_amax
    assert float(written) == pytest.approx(2.5)


@pytest.mark.unit
def test_activation_amax_rejects_an_unsupported_observed_linear():
    with pytest.raises(RuntimeError, match="does not expose an assignable activation amax"):
        sq._assign_activation_amax(torch.nn.Linear(4, 4), "fake.fqn", 2.5)


@pytest.mark.unit
def test_static_quantization_rejects_a_missing_observed_linear(tmp_path, monkeypatch):
    model = _FakeSortformer()
    path = _write_calibration(tmp_path, {fqn: 1.5 for fqn in _expected_target_fqns()})
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all", scale_mode="static", calibration_path=str(path))
    # A prepare step that leaves the plain Linear in place must not be converted with an unvalidated scale.
    registry = {
        sq.TORCHAO_QUANTIZE_API: lambda model, config, filter_fn=None: None,
        sq.TORCHAO_NVFP4_DYNAMIC_CONFIG_API: _FakeNVFP4DynamicConfig,
        sq.TORCHAO_NVFP4_WEIGHT_ONLY_CONFIG_API: _FakeNVFP4WeightOnlyConfig,
        sq.TORCHAO_NVFP4_OBSERVED_LINEAR_API: _FakeObservedLinear,
        sq.TORCHAO_FP8_CONFIG_API: _FakeFP8Config,
    }
    monkeypatch.setattr(sq, "_resolve_api", registry.get)

    with pytest.raises(RuntimeError, match="did not produce an observed linear"):
        sq.quantize_sortformer_model(model, config, facts=_facts())


@pytest.mark.unit
def test_calibration_applies_margin_and_normalizes_deterministically(tmp_path):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_qkv_only")
    unordered = {fqn: 0.5 for fqn in reversed(_expected_target_fqns())}
    path = _write_calibration(tmp_path, unordered)

    calibration = sq.load_calibration(str(path), selection, scale_margin=1.25)

    assert list(calibration["activation_amax"]) == _fqns_for_suffix("attn.w_qkv")
    assert set(calibration["activation_amax"].values()) == {0.625}
    assert list(calibration["raw_activation_amax"]) == sorted(calibration["raw_activation_amax"])
    assert calibration["unused_fqns"] == sorted(set(_expected_target_fqns()) - set(_fqns_for_suffix("attn.w_qkv")))
    assert calibration["scale_margin"] == 1.25


@pytest.mark.unit
def test_calibration_requires_complete_coverage(tmp_path):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")
    incomplete = {fqn: 1.0 for fqn in _expected_target_fqns()[1:]}
    path = _write_calibration(tmp_path, incomplete)

    with pytest.raises(ValueError, match="missing activation amax values"):
        sq.load_calibration(str(path), selection)


@pytest.mark.unit
def test_calibration_rejects_entries_that_are_not_targets(tmp_path):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")
    extra = {fqn: 1.0 for fqn in _expected_target_fqns()}
    extra["head"] = 1.0
    path = _write_calibration(tmp_path, extra)

    with pytest.raises(ValueError, match="not Sortformer quantization targets"):
        sq.load_calibration(str(path), selection)


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", ["NaN", "Infinity", "-1.0", "0.0", '"1.0"'])
def test_calibration_rejects_invalid_amax_values(tmp_path, bad_value):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")
    entries = ", ".join(f'"{fqn}": 1.0' for fqn in _expected_target_fqns()[1:])
    path = tmp_path / "calib.json"
    path.write_text(
        f'{{"version": {sq.CALIBRATION_SCHEMA_VERSION}, "activation_amax": '
        f'{{"{_expected_target_fqns()[0]}": {bad_value}, {entries}}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Calibration entry"):
        sq.load_calibration(str(path), selection)


@pytest.mark.unit
def test_calibration_rejects_duplicate_keys(tmp_path):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")
    duplicated = _expected_target_fqns()[0]
    entries = ", ".join(f'"{fqn}": 1.0' for fqn in _expected_target_fqns())
    path = tmp_path / "calib.json"
    path.write_text(
        f'{{"version": {sq.CALIBRATION_SCHEMA_VERSION}, "activation_amax": ' f'{{{entries}, "{duplicated}": 2.0}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        sq.load_calibration(str(path), selection)


@pytest.mark.unit
def test_calibration_rejects_unknown_schema_version(tmp_path):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()}, version=99)

    with pytest.raises(ValueError, match="version"):
        sq.load_calibration(str(path), selection)


@pytest.mark.unit
@pytest.mark.parametrize("margin", [0.0, -1.0, float("inf"), float("nan")])
def test_calibration_rejects_non_positive_margin(tmp_path, margin):
    selection = sq.select_quantization_targets(_FakeSortformer(), "nvfp4_all")
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})

    with pytest.raises(ValueError, match="scale_margin"):
        sq.load_calibration(str(path), selection, scale_margin=margin)


@pytest.mark.unit
def test_collector_records_target_activation_maxima():
    model = _FakeSortformer().eval()
    inputs = torch.randn(2, 5, D_MODEL)
    qkv_fqn = "transformer_encoder.layers.0.attn.w_qkv"

    with torch.no_grad():
        hidden = model.pre_encode(inputs)
        expected = float(model.transformer_encoder.layers[0].norm1(hidden).abs().max())
        collector = sq.ActivationAmaxCollector(model)
        with collector:
            model(inputs)

    assert sorted(collector.activation_amax) == _expected_target_fqns()
    assert collector.activation_amax[qkv_fqn] == pytest.approx(expected, rel=1e-5)


@pytest.mark.unit
def test_collector_records_uncompiled_fqns_for_a_compiled_submodule(tmp_path):
    """A calibration run with compile_encoder=True must stay loadable by the (uncompiled) quantized run."""
    model = _FakeSortformer().eval()
    uncompiled_selection = sq.select_quantization_targets(model, "nvfp4_all")
    model.transformer_encoder = _FakeOptimizedModule(model.transformer_encoder)

    collector = sq.ActivationAmaxCollector(model)
    assert any("_orig_mod" in fqn for fqn in collector.target_fqns)
    with collector:
        with torch.no_grad():
            model(torch.randn(1, 4, D_MODEL))

    assert sorted(collector.activation_amax) == _expected_target_fqns()
    path = tmp_path / "calib.json"
    collector.save(str(path))
    calibration = sq.load_calibration(str(path), uncompiled_selection)
    assert sorted(calibration["activation_amax"]) == _expected_target_fqns()


@pytest.mark.unit
def test_collector_keeps_the_running_maximum_and_ignores_non_finite_values():
    model = _FakeSortformer().eval()
    qkv = model.transformer_encoder.layers[0].attn.w_qkv
    qkv_fqn = "transformer_encoder.layers.0.attn.w_qkv"
    collector = sq.ActivationAmaxCollector(model)
    collector.install()
    collector.install()  # idempotent
    finite = torch.full((1, D_MODEL), 2.0)
    with_inf = torch.full((1, D_MODEL), 3.0)
    with_inf[0, 0] = float("inf")

    with torch.no_grad():
        qkv(with_inf)
        assert collector.activation_amax[qkv_fqn] == pytest.approx(3.0)
        qkv(finite)
        assert collector.activation_amax[qkv_fqn] == pytest.approx(3.0)

    collector.remove()
    collector.remove()  # idempotent
    assert not collector.installed
    with torch.no_grad():
        qkv(torch.full((1, D_MODEL), 100.0))
    assert collector.activation_amax[qkv_fqn] == pytest.approx(3.0)


@pytest.mark.unit
def test_collector_save_is_atomic_deterministic_and_round_trips(tmp_path):
    model = _FakeSortformer().eval()
    collector = sq.ActivationAmaxCollector(model)
    with collector:
        with torch.no_grad():
            model(torch.randn(2, 5, D_MODEL))
    path = tmp_path / "calib.json"

    metadata = {"precision": "bf16", "batch_size": 1}
    collector.save(str(path), metadata=metadata)

    assert [entry.name for entry in tmp_path.iterdir()] == ["calib.json"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == sq.CALIBRATION_SCHEMA_VERSION
    assert payload["targets"] == list(sq.QUANTIZATION_TARGET_SUFFIXES)
    assert payload["metadata"] == {"batch_size": 1, "precision": "bf16"}
    assert sorted(payload["activation_amax"]) == _expected_target_fqns()

    first = path.read_text(encoding="utf-8")
    collector.save(str(path), metadata=metadata, overwrite=True)
    assert path.read_text(encoding="utf-8") == first

    selection = sq.select_quantization_targets(model, "nvfp4_all")
    calibration = sq.load_calibration(str(path), selection, scale_margin=1.0)
    assert sorted(calibration["activation_amax"]) == _expected_target_fqns()


@pytest.mark.unit
def test_collector_save_refuses_to_overwrite(tmp_path):
    model = _FakeSortformer().eval()
    collector = sq.ActivationAmaxCollector(model)
    with collector:
        with torch.no_grad():
            model(torch.randn(1, 4, D_MODEL))
    path = tmp_path / "calib.json"
    collector.save(str(path))

    with pytest.raises(FileExistsError, match="quantization_overwrite_calibration"):
        collector.save(str(path))


@pytest.mark.unit
def test_collector_save_requires_a_forward_pass(tmp_path):
    collector = sq.ActivationAmaxCollector(_FakeSortformer())

    with pytest.raises(RuntimeError, match="No activations were observed"):
        collector.save(str(tmp_path / "calib.json"))


@pytest.mark.unit
def test_ensure_calibration_output_writable(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text("{}", encoding="utf-8")
    config = sq.SortformerQuantizationConfig(calibration_output=str(path))

    with pytest.raises(FileExistsError):
        sq.ensure_calibration_output_writable(config)

    config.overwrite_calibration = True
    sq.ensure_calibration_output_writable(config)


@pytest.mark.unit
def test_calibration_and_quantized_execution_are_mutually_exclusive():
    cfg = _eval_cfg(quantization_recipe="nvfp4_all", quantization_calibration_output="/tmp/calib.json")

    with pytest.raises(ValueError, match="cannot happen in one invocation"):
        sq.quantization_config_from_eval_cfg(cfg)


@pytest.mark.unit
def test_static_scale_mode_requires_a_calibration_file():
    cfg = _eval_cfg(quantization_recipe="nvfp4_all", quantization_scale_mode="static")

    with pytest.raises(ValueError, match="requires quantization_calibration_path"):
        sq.quantization_config_from_eval_cfg(cfg)


@pytest.mark.unit
def test_calibration_path_requires_static_scale_mode():
    cfg = _eval_cfg(quantization_recipe="nvfp4_all", quantization_calibration_path="/tmp/calib.json")

    with pytest.raises(ValueError, match="only used with scale_mode='static'"):
        sq.quantization_config_from_eval_cfg(cfg)


@pytest.mark.unit
def test_eval_cfg_defaults_are_disabled():
    config = sq.quantization_config_from_eval_cfg(_eval_cfg())

    assert config.enabled is False
    assert config.uses_activation_quantization is False


@pytest.mark.unit
def test_calibration_run_rejects_prediction_cache_reuse():
    config = sq.SortformerQuantizationConfig(calibration_output="/tmp/calib.json")

    with pytest.raises(ValueError, match="requires a real forward pass"):
        sq.validate_calibration_forward_pass(config, prediction_cache_reused=True)

    sq.validate_calibration_forward_pass(config, prediction_cache_reused=False)
    sq.validate_calibration_forward_pass(sq.SortformerQuantizationConfig(), prediction_cache_reused=True)


@pytest.mark.unit
def test_disabled_recipe_does_not_import_optional_quantization_dependencies(monkeypatch, tmp_path):
    real_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torchao", "mslk"), f"disabled path imported '{name}'"
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(sq.importlib, "import_module", guarded_import)
    model = _FakeSortformer().eval()

    summary = sq.quantize_sortformer_model(model, sq.SortformerQuantizationConfig())

    assert summary["backend"] == sq.BACKEND_DISABLED
    assert summary["counts_by_precision"] == {sq.PRECISION_BF16: 4 * NUM_LAYERS}
    assert summary["versions"]["torchao"] is None
    # The BF16 modules are untouched by the disabled path.
    assert isinstance(model.transformer_encoder.layers[0].attn.w_qkv, torch.nn.Linear)

    collector = sq.ActivationAmaxCollector(model)
    with collector:
        with torch.no_grad():
            model(torch.randn(1, 4, D_MODEL))
    collector.save(str(tmp_path / "calib.json"))


@pytest.mark.unit
def test_summary_is_json_serializable(fake_torchao):
    summary = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )

    assert json.loads(json.dumps(summary)) == summary
    assert summary["architecture_qualification"]["status"] == sq.QUALIFICATION_POLICY_ONLY
    assert summary["architecture_qualification"]["qualified_architectures"] == []
    assert summary["architecture_qualification"]["policy_accepted_architectures"] == [
        "sm100",
        "sm103",
        "sm110",
        "sm120",
        "sm121",
    ]


@pytest.mark.unit
def test_global_scale_folding_is_off_by_default(fake_torchao):
    summary = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )

    assert summary["global_scale_folding"] == {
        "activation_exponent": None,
        "activation_scale_factor": None,
        "enabled": False,
        "notes": [],
        "wrapped_count": 0,
        "wrapped_fqns": [],
    }
    assert all("Global-scale folding is ON" not in note for note in summary["notes"])
    assert sq.quantization_config_from_eval_cfg(_eval_cfg()).fold_global_scales is False


@pytest.mark.unit
def test_global_scale_folding_runs_after_conversion_and_is_reported(tmp_path, fake_torchao, monkeypatch):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    config = sq.quantization_config_from_eval_cfg(
        _eval_cfg(
            quantization_recipe="nvfp4_all",
            quantization_scale_mode="static",
            quantization_calibration_path=str(path),
            quantization_fold_global_scales=True,
            quantization_fold_activation_exponent=-12,
        )
    )
    recorded = {}

    def fake_fold(model, fqns, exponent):
        recorded["fqns"] = list(fqns)
        recorded["exponent"] = exponent
        # Folding must only ever see fully converted modules, never the observed linears of the prepare step.
        recorded["types"] = sorted({type(model.get_submodule(fqn)).__name__ for fqn in fqns})
        return {
            "enabled": True,
            "activation_exponent": exponent,
            "activation_scale_factor": 2.0**exponent,
            "wrapped_count": len(fqns),
            "wrapped_fqns": sorted(fqns),
            "notes": [],
        }

    monkeypatch.setattr(sq, "apply_global_scale_folding", fake_fold)
    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    assert recorded["fqns"] == _expected_target_fqns()
    assert recorded["exponent"] == -12
    assert recorded["types"] == ["_FakeQuantizedLinear"]
    assert summary["global_scale_folding"]["enabled"] is True
    assert summary["global_scale_folding"]["activation_exponent"] == -12
    assert summary["global_scale_folding"]["wrapped_count"] == 4 * NUM_LAYERS
    assert summary["global_scale_folding"]["wrapped_fqns"] == _expected_target_fqns()
    assert any("Global-scale folding is ON" in note for note in summary["notes"])
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"quantization_recipe": "nvfp4_weight_only", "quantization_scale_mode": "dynamic"},
            "does not quantize activations",
        ),
        ({"quantization_scale_mode": "dynamic", "quantization_calibration_path": None}, "requires scale_mode"),
        (
            {"quantization_accelerated_packing": False, "quantization_allow_reference_kernels": True},
            "requires quantization_accelerated_packing=True",
        ),
        ({"quantization_allow_reference_kernels": True}, "different"),
    ],
)
def test_global_scale_folding_rejects_incompatible_options(tmp_path, overrides, message):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    values = dict(
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(path),
        quantization_fold_global_scales=True,
    )
    values.update(overrides)
    if values["quantization_scale_mode"] != "static":
        values["quantization_calibration_path"] = None

    with pytest.raises(ValueError, match=message):
        sq.quantization_config_from_eval_cfg(_eval_cfg(**values))


@pytest.mark.unit
@pytest.mark.parametrize("exponent", [-25, 9, True, 1.0, "-10", None])
def test_global_scale_folding_rejects_an_out_of_range_or_non_integer_exponent(tmp_path, exponent):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    cfg = _eval_cfg(
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(path),
        quantization_fold_global_scales=True,
        quantization_fold_activation_exponent=exponent,
    )

    with pytest.raises(ValueError, match="quantization_fold_activation_exponent"):
        sq.quantization_config_from_eval_cfg(cfg)


@pytest.mark.unit
def test_global_scale_folding_accepts_the_documented_exponent_bounds(tmp_path):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})

    for exponent in (scale_fold.FOLD_EXPONENT_MIN, -10, scale_fold.FOLD_EXPONENT_MAX):
        config = sq.quantization_config_from_eval_cfg(
            _eval_cfg(
                quantization_recipe="nvfp4_all",
                quantization_scale_mode="static",
                quantization_calibration_path=str(path),
                quantization_fold_global_scales=True,
                quantization_fold_activation_exponent=exponent,
            )
        )
        assert config.fold_activation_exponent == exponent


@pytest.mark.unit
def test_producer_fusion_is_off_by_default(fake_torchao):
    summary = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )

    assert summary["producer_fusion"] == {
        "enabled": False,
        "fused_block_count": 0,
        "out_proj_restored_block_count": 0,
        "fused_block_fqns": [],
        "fused_boundaries": [],
        "notes": [],
    }
    assert all("Producer packing fusion is ON" not in note for note in summary["notes"])
    assert sq.quantization_config_from_eval_cfg(_eval_cfg()).fuse_producer_packing is False


@pytest.mark.unit
def test_producer_fusion_runs_after_conversion_and_is_reported(tmp_path, fake_torchao, monkeypatch):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    config = sq.quantization_config_from_eval_cfg(
        _eval_cfg(
            quantization_recipe="nvfp4_all",
            quantization_scale_mode="static",
            quantization_calibration_path=str(path),
            quantization_fuse_producer_packing=True,
        )
    )
    recorded = {}

    def fake_fuse(model, fqns):
        recorded["fqns"] = list(fqns)
        # Fusion must only ever see fully converted modules, never the observed linears of the prepare step.
        recorded["types"] = sorted({type(model.get_submodule(fqn)).__name__ for fqn in fqns})
        blocks = producer_fusion.group_producer_fusion_blocks(fqns)
        return {
            "enabled": True,
            "fused_block_count": len(blocks),
            "fused_block_fqns": list(blocks),
            "fused_boundaries": list(producer_fusion.FUSED_PRODUCER_BOUNDARIES),
            "notes": [],
        }

    monkeypatch.setattr(sq, "apply_producer_fusion", fake_fuse)
    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    assert recorded["fqns"] == _expected_target_fqns()
    assert recorded["types"] == ["_FakeQuantizedLinear"]
    assert summary["producer_fusion"]["enabled"] is True
    assert summary["producer_fusion"]["fused_block_count"] == NUM_LAYERS
    assert any("Producer packing fusion is ON" in note for note in summary["notes"])
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"quantization_recipe": "nvfp4_qkv_only"}, "requires recipe='nvfp4_all'"),
        ({"quantization_scale_mode": "dynamic", "quantization_calibration_path": None}, "requires scale_mode"),
        (
            {"quantization_accelerated_packing": False, "quantization_allow_reference_kernels": True},
            "requires quantization_accelerated_packing=True",
        ),
        ({"quantization_allow_reference_kernels": True}, "different"),
        ({"quantization_fold_global_scales": True}, "mutually"),
    ],
)
def test_producer_fusion_rejects_incompatible_options(tmp_path, overrides, message):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    values = dict(
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(path),
        quantization_fuse_producer_packing=True,
    )
    values.update(overrides)
    if values["quantization_scale_mode"] != "static":
        values["quantization_calibration_path"] = None

    with pytest.raises(ValueError, match=message):
        sq.quantization_config_from_eval_cfg(_eval_cfg(**values))


@pytest.mark.unit
def test_prediction_cache_identity_separates_quantized_and_bf16_runs(tmp_path):
    """A quantized run must not silently report DER computed from a BF16 prediction cache, or the reverse."""
    from nemo.collections.asr.parts.utils.sortformer_utils import load_prediction_tensors, save_prediction_tensors

    def metadata_for(config):
        return {
            "version": 1,
            "recording_ids": ["rec-0"],
            "num_speakers": 4,
            "quantization": sq.prediction_cache_identity(config),
        }

    bf16 = sq.SortformerQuantizationConfig()
    quantized = sq.SortformerQuantizationConfig(recipe="nvfp4_all")
    assert sq.prediction_cache_identity(bf16) is None
    assert sq.prediction_cache_identity(quantized)["recipe"] == "nvfp4_all"

    bf16_cache = tmp_path / "bf16.pt"
    save_prediction_tensors(str(bf16_cache), [torch.zeros(1, 3, 4)], metadata_for(bf16))
    # A BF16 cache is reusable by another BF16 run, and by a cache written before quantization existed.
    assert len(load_prediction_tensors(str(bf16_cache), metadata_for(bf16))) == 1
    legacy_metadata = {key: value for key, value in metadata_for(bf16).items() if key != "quantization"}
    legacy_cache = tmp_path / "legacy.pt"
    save_prediction_tensors(str(legacy_cache), [torch.zeros(1, 3, 4)], legacy_metadata)
    assert len(load_prediction_tensors(str(legacy_cache), metadata_for(bf16))) == 1

    with pytest.raises(ValueError, match="quantization"):
        load_prediction_tensors(str(bf16_cache), metadata_for(quantized))
    with pytest.raises(ValueError, match="quantization"):
        load_prediction_tensors(str(legacy_cache), metadata_for(quantized))

    quantized_cache = tmp_path / "nvfp4.pt"
    save_prediction_tensors(str(quantized_cache), [torch.zeros(1, 3, 4)], metadata_for(quantized))
    with pytest.raises(ValueError, match="quantization"):
        load_prediction_tensors(str(quantized_cache), metadata_for(bf16))
    with pytest.raises(ValueError, match="quantization"):
        load_prediction_tensors(
            str(quantized_cache), metadata_for(sq.SortformerQuantizationConfig(recipe="nvfp4_qkv_only"))
        )


@pytest.mark.unit
def test_prediction_cache_identity_tracks_the_calibration_file_itself(tmp_path):
    """Re-collecting calibration to the same path must invalidate the previous static run's prediction cache."""
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_all", scale_mode="static", calibration_path=str(path))

    original = sq.prediction_cache_identity(config)
    assert original["calibration_file"]["size"] == path.stat().st_size

    # Same path, different activation scales: nothing else about the run changes.
    _write_calibration(tmp_path, {fqn: 12.25 for fqn in _expected_target_fqns()})
    assert sq.prediction_cache_identity(config) != original

    # A rewrite that happens to keep the same size is still caught through the modification time.
    _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    later = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(later, later))
    assert sq.prediction_cache_identity(config) != original

    missing = sq.SortformerQuantizationConfig(
        recipe="nvfp4_all", scale_mode="static", calibration_path=str(tmp_path / "absent.json")
    )
    assert sq.prediction_cache_identity(missing)["calibration_file"] is None


@pytest.mark.unit
def test_prediction_cache_identity_separates_folded_runs_and_every_swept_exponent(tmp_path):
    """Folded predictions differ from the unfolded ones and from every other exponent, so caches must not mix."""
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})

    def config_for(**overrides):
        return sq.SortformerQuantizationConfig(
            recipe="nvfp4_all", scale_mode="static", calibration_path=str(path), **overrides
        )

    unfolded = sq.prediction_cache_identity(config_for())
    folded = sq.prediction_cache_identity(config_for(fold_global_scales=True, fold_activation_exponent=-10))
    swept = sq.prediction_cache_identity(config_for(fold_global_scales=True, fold_activation_exponent=-12))

    assert unfolded["fold_global_scales"] is False
    assert unfolded["fold_activation_exponent"] is None
    assert folded["fold_activation_exponent"] == -10
    assert unfolded != folded
    assert folded != swept
    # The inert exponent of a run that is not folding must not invalidate an otherwise identical cache.
    assert sq.prediction_cache_identity(config_for(fold_activation_exponent=-12)) == unfolded


@pytest.mark.unit
def test_evaluator_quantizes_before_compiling():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert source.index("quantize_sortformer_model(diar_model") < source.index("torch.compile(diar_model.encoder")
    assert source.index("ActivationAmaxCollector(diar_model") < source.index("activation_collector.remove()")
    # The quantization identity is folded into the prediction-cache metadata before the cache is read or written.
    assert 'prediction_cache_metadata["quantization"] = prediction_cache_identity(' in source
    assert source.index("prediction_cache_identity(quantization_config)") < source.index("reuse_prediction_cache = ")


# --- Per-invocation sample histories and source-balanced merging -------------------------------------------------

FQN_A = "transformer_encoder.layers.0.attn.w_qkv"
FQN_B = "transformer_encoder.layers.0.ffn.net.0"
CHECKPOINT_SHA = "a1" * 32
OTHER_CHECKPOINT_SHA = "b2" * 32


@pytest.mark.unit
def test_collector_keeps_every_invocation_maximum_in_order():
    model = _FakeSortformer().eval()
    qkv = model.transformer_encoder.layers[0].attn.w_qkv
    collector = sq.ActivationAmaxCollector(model)

    with collector:
        with torch.no_grad():
            for magnitude in (2.0, 5.0, 3.0):
                qkv(torch.full((1, D_MODEL), magnitude))

    # The running maximum is unchanged, and each invocation contributed exactly one float, in invocation order.
    assert collector.activation_amax[FQN_A] == pytest.approx(5.0)
    assert collector.activation_amax_samples[FQN_A] == pytest.approx([2.0, 5.0, 3.0])
    assert all(isinstance(sample, float) for sample in collector.activation_amax_samples[FQN_A])


@pytest.mark.unit
def test_collector_samples_ignore_non_finite_and_non_positive_inputs():
    model = _FakeSortformer().eval()
    qkv = model.transformer_encoder.layers[0].attn.w_qkv
    collector = sq.ActivationAmaxCollector(model)
    mixed = torch.full((1, D_MODEL), 3.0)
    mixed[0, 0] = float("inf")

    with collector:
        with torch.no_grad():
            qkv(mixed)
            qkv(torch.full((1, D_MODEL), float("nan")))
            qkv(torch.zeros(1, D_MODEL))
            qkv(torch.full((1, D_MODEL), 1.0))

    # Only the finite maximum of the mixed input and the ordinary input were recorded.
    assert collector.activation_amax_samples[FQN_A] == pytest.approx([3.0, 1.0])
    assert collector.activation_amax[FQN_A] == pytest.approx(3.0)


@pytest.mark.unit
def test_collector_samples_use_canonical_fqns_for_a_compiled_submodule():
    model = _FakeSortformer().eval()
    model.transformer_encoder = _FakeOptimizedModule(model.transformer_encoder)

    collector = sq.ActivationAmaxCollector(model)
    with collector:
        with torch.no_grad():
            model(torch.randn(1, 4, D_MODEL))
            model(torch.randn(1, 4, D_MODEL))

    assert sorted(collector.activation_amax_samples) == _expected_target_fqns()
    assert all(len(samples) == 2 for samples in collector.activation_amax_samples.values())


@pytest.mark.unit
def test_collector_save_records_sample_histories_and_stays_loadable(tmp_path):
    model = _FakeSortformer().eval()
    collector = sq.ActivationAmaxCollector(model)
    with collector:
        with torch.no_grad():
            model(torch.randn(2, 5, D_MODEL))
            model(torch.randn(2, 5, D_MODEL))
    path = tmp_path / "calib.json"

    collector.save(str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload[sq.CALIBRATION_SAMPLES_FIELD]
    assert sorted(samples) == _expected_target_fqns()
    assert list(samples) == sorted(samples)
    assert all(len(history) == 2 for history in samples.values())
    assert all(max(samples[fqn]) == pytest.approx(payload["activation_amax"][fqn]) for fqn in samples)

    # Deterministic bytes, and the extra field is ignored by the runtime loader.
    first = path.read_text(encoding="utf-8")
    collector.save(str(path), overwrite=True)
    assert path.read_text(encoding="utf-8") == first
    calibration = sq.load_calibration(str(path), sq.select_quantization_targets(model, "nvfp4_all"))
    assert sorted(calibration["activation_amax"]) == _expected_target_fqns()


def _write_merge_input(
    path,
    samples,
    version=sq.CALIBRATION_SCHEMA_VERSION,
    targets=(),
    include_samples=True,
    nonfinite=None,
    checkpoint=None,
):
    """
    Write a calibration input for the merger.

    ``targets=None`` omits the field the way a legacy file does, and ``include_samples=False`` writes the legacy
    max-only artifact, which carries neither a sample history nor non-finite counts.
    """
    payload = {
        "version": version,
        "recipe": "disabled",
        "scale_mode": "static",
        "activation_amax": {fqn: max(values) for fqn, values in samples.items()},
    }
    if targets is not None:
        payload["targets"] = list(targets) if targets else list(sq.QUANTIZATION_TARGET_SUFFIXES)
    if include_samples:
        payload[sq.CALIBRATION_SAMPLES_FIELD] = {fqn: list(values) for fqn, values in samples.items()}
        payload[sq.CALIBRATION_NONFINITE_FIELD] = {fqn: (nonfinite or {}).get(fqn, 0) for fqn in samples}
    if checkpoint is not None:
        payload[sq.CALIBRATION_CHECKPOINT_FIELD] = checkpoint
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "percentile, expected",
    [(50.0, 5.0), (75.0, 8.0), (95.0, 10.0), (100.0, 10.0), (10.0, 1.0)],
)
def test_merge_uses_conservative_nearest_rank_percentile(tmp_path, percentile, expected):
    values = [float(index) for index in range(1, 11)]
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: values, FQN_B: values})

    payload = sq.merge_calibrations(
        [("only", path)], percentile=percentile, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
    )

    assert payload["activation_amax"] == {FQN_A: pytest.approx(expected), FQN_B: pytest.approx(expected)}
    assert payload["version"] == sq.CALIBRATION_SCHEMA_VERSION
    assert payload["targets"] == list(sq.QUANTIZATION_TARGET_SUFFIXES)
    assert sq.CALIBRATION_SAMPLES_FIELD not in payload


@pytest.mark.unit
def test_merge_pools_repeated_strata_and_takes_the_group_maximum_with_baked_headroom(tmp_path):
    near_a = _write_merge_input(tmp_path / "near_a.json", {FQN_A: [1.0, 2.0, 3.0], FQN_B: [1.0, 1.0, 1.0]})
    near_b = _write_merge_input(tmp_path / "near_b.json", {FQN_A: [4.0, 5.0, 6.0], FQN_B: [1.0, 1.0, 1.0]})
    far = _write_merge_input(tmp_path / "far.json", {FQN_A: [10.0], FQN_B: [0.5]})

    payload = sq.merge_calibrations(
        [("near", near_a), ("near", near_b), ("far", far)],
        percentile=50.0,
        headroom=2.0,
        checkpoint_sha256=CHECKPOINT_SHA,
    )

    # near pools to [1..6] whose 50th nearest rank is 3.0; the far group's single 10.0 must not be averaged away.
    assert payload["activation_amax"][FQN_A] == pytest.approx(20.0)
    assert payload["activation_amax"][FQN_B] == pytest.approx(2.0)
    provenance = payload["metadata"]
    assert provenance["method"] == sq.CALIBRATION_MERGE_METHOD
    assert provenance["groups"] == ["far", "near"]
    assert provenance["headroom"] == pytest.approx(2.0)
    assert provenance["headroom_baked_in"] is True
    assert provenance["runtime_scale_margin"] == pytest.approx(1.0)
    assert provenance["legacy_max_only_fallback"] is False
    assert provenance["group_statistics"]["near"] == {
        "input_count": 2,
        "observation_count": 12,
        "min_observations_per_module": 6,
        "max_observations_per_module": 6,
    }
    assert [entry["name"] for entry in provenance["inputs"]] == ["far.json", "near_a.json", "near_b.json"]
    assert all(len(entry["sha256"]) == 64 for entry in provenance["inputs"])
    assert all(entry["size_bytes"] > 0 for entry in provenance["inputs"])


@pytest.mark.unit
def test_merge_output_bytes_do_not_depend_on_input_order(tmp_path):
    first = _write_merge_input(tmp_path / "first.json", {FQN_A: [1.0, 4.0], FQN_B: [2.0]})
    second = _write_merge_input(tmp_path / "second.json", {FQN_A: [3.0], FQN_B: [7.0, 1.0]})
    third = _write_merge_input(tmp_path / "third.json", {FQN_A: [2.0], FQN_B: [5.0]})
    forward = tmp_path / "forward.json"
    reversed_output = tmp_path / "reversed.json"

    inputs = [("near", first), ("near", second), ("far", third)]
    sq.merge_calibration_files(
        inputs, percentile=90.0, headroom=1.25, checkpoint_sha256=CHECKPOINT_SHA, output_path=str(forward)
    )
    sq.merge_calibration_files(
        list(reversed(inputs)),
        percentile=90.0,
        headroom=1.25,
        checkpoint_sha256=CHECKPOINT_SHA,
        output_path=str(reversed_output),
    )

    assert forward.read_text(encoding="utf-8") == reversed_output.read_text(encoding="utf-8")


@pytest.mark.unit
def test_merge_accepts_a_legacy_max_only_input_and_records_the_fallback(tmp_path):
    legacy = _write_merge_input(
        tmp_path / "davidai.json", {FQN_A: [8.0], FQN_B: [2.0]}, targets=None, include_samples=False
    )
    modern = _write_merge_input(tmp_path / "modern.json", {FQN_A: [1.0, 2.0], FQN_B: [1.0, 1.0]})

    payload = sq.merge_calibrations(
        [("legacy", legacy), ("modern", modern)], percentile=50.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
    )

    # The single legacy observation survives every percentile, so the envelope is still the legacy maximum.
    assert payload["activation_amax"][FQN_A] == pytest.approx(8.0)
    provenance = payload["metadata"]
    assert provenance["legacy_max_only_fallback"] is True
    fallbacks = {entry["name"]: entry["legacy_max_only_fallback"] for entry in provenance["inputs"]}
    assert fallbacks == {"davidai.json": True, "modern.json": False}
    assert provenance["group_statistics"]["legacy"]["observation_count"] == 2


@pytest.mark.unit
def test_merged_artifact_is_consumable_by_the_runtime_loader(tmp_path):
    model = _FakeSortformer().eval()
    samples = {fqn: [1.0, 2.0] for fqn in _expected_target_fqns()}
    path = _write_merge_input(tmp_path / "a.json", samples)
    merged = tmp_path / "merged.json"

    sq.merge_calibration_files(
        [("only", path)],
        percentile=100.0,
        headroom=1.5,
        checkpoint_sha256=CHECKPOINT_SHA,
        output_path=str(merged),
    )

    calibration = sq.load_calibration(
        str(merged), sq.select_quantization_targets(model, "nvfp4_all"), scale_margin=1.0
    )
    assert sorted(calibration["activation_amax"]) == _expected_target_fqns()
    assert all(value == pytest.approx(3.0) for value in calibration["activation_amax"].values())


@pytest.mark.unit
@pytest.mark.parametrize("percentile", [0.0, -1.0, 100.5, float("inf"), float("nan"), "50", True])
def test_merge_rejects_an_invalid_percentile(tmp_path, percentile):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})

    with pytest.raises(ValueError, match="percentile"):
        sq.merge_calibrations([("only", path)], percentile=percentile, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
@pytest.mark.parametrize("headroom", [0.0, -1.0, float("inf"), float("nan"), "1.0"])
def test_merge_rejects_an_invalid_headroom(tmp_path, headroom):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})

    with pytest.raises(ValueError, match="headroom"):
        sq.merge_calibrations([("only", path)], percentile=99.0, headroom=headroom, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_requires_at_least_one_input():
    with pytest.raises(ValueError, match="At least one calibration input"):
        sq.merge_calibrations([], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_rejects_an_empty_group_label(tmp_path):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})

    with pytest.raises(ValueError, match="empty group name"):
        sq.merge_calibrations([("  ", path)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update({"version": 2}), "version"),
        (lambda payload: payload.update({"activation_amax": {}}), "non-empty 'activation_amax'"),
        (lambda payload: payload["activation_amax"].update({FQN_A: -1.0}), "Calibration entry"),
        (lambda payload: payload["activation_amax"].update({FQN_A: "1.0"}), "Calibration entry"),
        (lambda payload: payload.update({"targets": "attn.w_qkv"}), "list of strings"),
        (lambda payload: payload[sq.CALIBRATION_SAMPLES_FIELD].update({FQN_A: []}), "non-empty list"),
        (lambda payload: payload[sq.CALIBRATION_SAMPLES_FIELD].update({FQN_A: [0.0]}), "Observation 0"),
        (
            lambda payload: payload[sq.CALIBRATION_SAMPLES_FIELD].update({FQN_A: [1.0, float("inf")]}),
            "Observation 1",
        ),
        (lambda payload: payload[sq.CALIBRATION_SAMPLES_FIELD].pop(FQN_A), "differ from 'activation_amax'"),
        (lambda payload: payload.update({sq.CALIBRATION_SAMPLES_FIELD: [1.0]}), "as an object"),
    ],
)
def test_merge_rejects_malformed_inputs(tmp_path, mutate, message):
    path = tmp_path / "a.json"
    _write_merge_input(path, {FQN_A: [1.0, 2.0], FQN_B: [1.0, 2.0]})
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        sq.merge_calibrations([("only", str(path))], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        f'{{"version": {sq.CALIBRATION_SCHEMA_VERSION}, "activation_amax": ' f'{{"{FQN_A}": 1.0, "{FQN_A}": 2.0}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        sq.merge_calibrations([("only", str(path))], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_rejects_invalid_json(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        sq.merge_calibrations([("only", str(path))], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_rejects_mismatched_module_sets(tmp_path):
    first = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})
    second = _write_merge_input(tmp_path / "b.json", {FQN_A: [1.0]})

    with pytest.raises(ValueError, match="does not cover the same modules"):
        sq.merge_calibrations(
            [("a", first), ("b", second)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
        )


@pytest.mark.unit
def test_merge_rejects_mismatched_targets(tmp_path):
    first = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})
    second = _write_merge_input(tmp_path / "b.json", {FQN_A: [1.0], FQN_B: [1.0]}, targets=("attn.w_qkv",))

    with pytest.raises(ValueError, match="must share targets"):
        sq.merge_calibrations(
            [("a", first), ("b", second)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
        )


@pytest.mark.unit
def test_merge_refuses_to_overwrite_without_permission(tmp_path):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})
    output = tmp_path / "merged.json"

    def merge(overwrite=False):
        return sq.merge_calibration_files(
            [("only", path)],
            percentile=99.0,
            headroom=1.0,
            checkpoint_sha256=CHECKPOINT_SHA,
            output_path=str(output),
            overwrite=overwrite,
        )

    merge()
    first = output.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        merge()

    merge(overwrite=True)
    assert output.read_text(encoding="utf-8") == first


@pytest.mark.unit
def test_merge_rejects_an_input_collected_on_a_different_checkpoint(tmp_path):
    ours = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]}, checkpoint=CHECKPOINT_SHA)
    theirs = _write_merge_input(tmp_path / "b.json", {FQN_A: [9.0], FQN_B: [9.0]}, checkpoint=OTHER_CHECKPOINT_SHA)

    with pytest.raises(ValueError, match="was collected on checkpoint"):
        sq.merge_calibrations(
            [("a", ours), ("b", theirs)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
        )


@pytest.mark.unit
def test_merge_records_the_checkpoint_identity_of_self_describing_inputs(tmp_path):
    declared = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]}, checkpoint=CHECKPOINT_SHA)
    silent = _write_merge_input(tmp_path / "b.json", {FQN_A: [1.0], FQN_B: [1.0]})

    both = sq.merge_calibrations(
        [("a", declared), ("b", silent)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
    )
    only_declared = sq.merge_calibrations(
        [("a", declared)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA.upper()
    )

    assert both["metadata"]["checkpoint_sha256"] == CHECKPOINT_SHA
    assert both["metadata"]["checkpoint_identity_asserted_only"] is True
    assert {entry["name"]: entry["checkpoint_sha256_declared"] for entry in both["metadata"]["inputs"]} == {
        "a.json": True,
        "b.json": False,
    }
    # An upper-case digest is normalized, so it still matches the input's own lower-case record.
    assert only_declared["metadata"]["checkpoint_sha256"] == CHECKPOINT_SHA
    assert only_declared["metadata"]["checkpoint_identity_asserted_only"] is False


@pytest.mark.unit
@pytest.mark.parametrize("digest", ["", "not-a-digest", "a" * 63, "a" * 65, "g" * 64, None, 1234])
def test_merge_rejects_an_invalid_checkpoint_digest(tmp_path, digest):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})

    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        sq.merge_calibrations([("only", path)], percentile=99.0, headroom=1.0, checkpoint_sha256=digest)


@pytest.mark.unit
def test_merge_rejects_an_input_whose_declared_checkpoint_is_malformed(tmp_path):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]}, checkpoint="deadbeef")

    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        sq.merge_calibrations([("only", path)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_rejects_a_shard_that_observed_non_finite_activations(tmp_path):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0, 2.0], FQN_B: [1.0]}, nonfinite={FQN_A: 3})

    with pytest.raises(ValueError, match="observed non-finite activations"):
        sq.merge_calibrations([("only", path)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.pop(sq.CALIBRATION_NONFINITE_FIELD), "no 'activation_nonfinite_counts'"),
        (lambda payload: payload.update({sq.CALIBRATION_NONFINITE_FIELD: [0]}), "as an object"),
        (lambda payload: payload[sq.CALIBRATION_NONFINITE_FIELD].pop(FQN_A), "keys that differ"),
        (lambda payload: payload[sq.CALIBRATION_NONFINITE_FIELD].update({FQN_A: -1}), "non-negative integer"),
        (lambda payload: payload[sq.CALIBRATION_NONFINITE_FIELD].update({FQN_A: 0.5}), "non-negative integer"),
        (lambda payload: payload[sq.CALIBRATION_SAMPLES_FIELD].update({FQN_A: [1.0]}), "describe different data"),
    ],
)
def test_merge_rejects_inconsistent_statistics_fields(tmp_path, mutate, message):
    path = tmp_path / "a.json"
    _write_merge_input(path, {FQN_A: [1.0, 2.0], FQN_B: [1.0, 2.0]})
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        sq.merge_calibrations([("only", str(path))], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_merge_marks_a_legacy_input_as_unknown_non_finite_provenance(tmp_path):
    legacy = _write_merge_input(
        tmp_path / "davidai.json", {FQN_A: [8.0], FQN_B: [2.0]}, targets=None, include_samples=False
    )
    modern = _write_merge_input(tmp_path / "modern.json", {FQN_A: [1.0], FQN_B: [1.0]})

    mixed = sq.merge_calibrations(
        [("legacy", legacy), ("modern", modern)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
    )
    clean = sq.merge_calibrations(
        [("modern", modern)], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA
    )

    assert mixed["metadata"]["nonfinite_status"] == sq.CALIBRATION_PROVENANCE_UNKNOWN_LEGACY
    assert clean["metadata"]["nonfinite_status"] == sq.CALIBRATION_PROVENANCE_CLEAN
    statuses = {entry["name"]: entry["nonfinite_status"] for entry in mixed["metadata"]["inputs"]}
    assert statuses == {
        "davidai.json": sq.CALIBRATION_PROVENANCE_UNKNOWN_LEGACY,
        "modern.json": sq.CALIBRATION_PROVENANCE_CLEAN,
    }


@pytest.mark.unit
def test_collector_counts_non_finite_invocations_and_writes_them(tmp_path):
    model = _FakeSortformer().eval()
    qkv = model.transformer_encoder.layers[0].attn.w_qkv
    collector = sq.ActivationAmaxCollector(model)
    mixed = torch.full((1, D_MODEL), 3.0)
    mixed[0, 0] = float("inf")

    with collector:
        with torch.no_grad():
            model(torch.randn(2, 5, D_MODEL))
            qkv(mixed)
    path = tmp_path / "calib.json"
    collector.save(str(path))

    assert collector.nonfinite_observations == {FQN_A: 1}
    counts = json.loads(path.read_text(encoding="utf-8"))[sq.CALIBRATION_NONFINITE_FIELD]
    assert sorted(counts) == _expected_target_fqns()
    assert counts[FQN_A] == 1
    assert sum(counts.values()) == 1

    # A shard that saw a non-finite activation is not merged as if it were healthy.
    with pytest.raises(ValueError, match="observed non-finite activations"):
        sq.merge_calibrations([("only", str(path))], percentile=99.0, headroom=1.0, checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.unit
def test_collector_save_refuses_a_target_that_only_saw_non_finite_activations(tmp_path):
    model = _FakeSortformer().eval()
    qkv = model.transformer_encoder.layers[0].attn.w_qkv
    collector = sq.ActivationAmaxCollector(model)

    with collector:
        with torch.no_grad():
            model(torch.randn(1, 4, D_MODEL))
            del collector.activation_amax[FQN_A]  # simulate a target whose only observations were non-finite
            qkv(torch.full((1, D_MODEL), float("nan")))

    assert collector.nonfinite_observations == {FQN_A: 1}
    with pytest.raises(RuntimeError, match="only non-finite activations"):
        collector.save(str(tmp_path / "calib.json"))


@pytest.mark.unit
def test_collector_save_records_the_checkpoint_identity(tmp_path):
    model = _FakeSortformer().eval()
    collector = sq.ActivationAmaxCollector(model)
    with collector:
        with torch.no_grad():
            model(torch.randn(1, 4, D_MODEL))
    path = tmp_path / "calib.json"

    collector.save(str(path), checkpoint_sha256=CHECKPOINT_SHA.upper())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[sq.CALIBRATION_CHECKPOINT_FIELD] == CHECKPOINT_SHA
    assert sq.load_calibration(str(path), sq.select_quantization_targets(model, "nvfp4_all"))["activation_amax"]

    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        collector.save(str(tmp_path / "other.json"), checkpoint_sha256="nope")
    assert not (tmp_path / "other.json").exists()


@pytest.fixture(scope="module")
def merge_cli():
    """Load the thin merge CLI, which lives outside any importable package."""
    import importlib.util  # the module-level ``import importlib`` does not load this submodule

    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "dataset_processing"
        / "speaker_tasks"
        / "merge_sortformer_calibrations.py"
    )
    if not script.exists():
        pytest.skip("merge CLI is not available in this checkout")
    spec = importlib.util.spec_from_file_location("merge_sortformer_calibrations", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_merge_cli_writes_the_merged_artifact(tmp_path, merge_cli, capsys):
    near = _write_merge_input(tmp_path / "near.json", {FQN_A: [1.0, 2.0], FQN_B: [1.0, 1.0]})
    far = _write_merge_input(tmp_path / "far.json", {FQN_A: [4.0], FQN_B: [1.0]})
    output = tmp_path / "merged.json"

    exit_code = merge_cli.main(
        [
            "--input",
            f"near={near}",
            "--input",
            f"far={far}",
            "--percentile",
            "100",
            "--headroom",
            "1.5",
            "--checkpoint-sha256",
            CHECKPOINT_SHA,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["activation_amax"][FQN_A] == pytest.approx(6.0)
    assert payload["metadata"]["groups"] == ["far", "near"]
    assert payload["metadata"]["checkpoint_sha256"] == CHECKPOINT_SHA
    assert str(output) in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("bad_input", ["near", "=path.json", "near="])
def test_merge_cli_rejects_a_malformed_input_argument(tmp_path, merge_cli, bad_input):
    with pytest.raises(SystemExit) as excinfo:
        merge_cli.main(
            ["--input", bad_input, "--checkpoint-sha256", CHECKPOINT_SHA, "--output", str(tmp_path / "merged.json")]
        )

    assert excinfo.value.code != 0


@pytest.mark.unit
def test_merge_cli_fails_on_a_malformed_calibration_file(tmp_path, merge_cli):
    path = tmp_path / "a.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        merge_cli.main(
            [
                "--input",
                f"only={path}",
                "--checkpoint-sha256",
                CHECKPOINT_SHA,
                "--output",
                str(tmp_path / "merged.json"),
            ]
        )

    assert excinfo.value.code != 0


@pytest.mark.unit
def test_merge_cli_requires_a_checkpoint_identity(tmp_path, merge_cli):
    path = _write_merge_input(tmp_path / "a.json", {FQN_A: [1.0], FQN_B: [1.0]})

    # Omitting the checkpoint is an argparse error, and a malformed digest fails before anything is written.
    with pytest.raises(SystemExit) as missing:
        merge_cli.main(["--input", f"only={path}", "--output", str(tmp_path / "merged.json")])
    assert missing.value.code != 0

    with pytest.raises(SystemExit) as malformed:
        merge_cli.main(
            [
                "--input",
                f"only={path}",
                "--checkpoint-sha256",
                "not-a-digest",
                "--output",
                str(tmp_path / "merged.json"),
            ]
        )
    assert malformed.value.code != 0
    assert not (tmp_path / "merged.json").exists()


# --- BF16 restoration on top of the weight-only and W4A4 recipes ---------------------------------------------------


def _weight_only_selection():
    """Selection of the fake model under the weight-only recipe a BF16 override may be combined with."""
    return sq.select_quantization_targets(_FakeSortformer(), "nvfp4_weight_only")


def _write_bf16_override(tmp_path, fqns, version=sq.BF16_OVERRIDE_SCHEMA_VERSION, name="bf16.json"):
    """Write a well-formed BF16 override file and return its path."""
    path = tmp_path / name
    payload = {"version": version, sq.BF16_OVERRIDE_FQNS_FIELD: list(fqns)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bf16_config(path, **overrides):
    """Quantization config for the weight-only recipe with a BF16 override file."""
    values = dict(quantization_recipe="nvfp4_weight_only", quantization_bf16_override_path=str(path))
    values.update(overrides)
    return sq.quantization_config_from_eval_cfg(_eval_cfg(**values))


@pytest.mark.unit
def test_bf16_override_is_absent_by_default(fake_torchao):
    config = sq.quantization_config_from_eval_cfg(_eval_cfg(quantization_recipe="nvfp4_weight_only"))
    assert config.bf16_override_path is None
    assert config.has_bf16_override is False

    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    # The disabled section keeps the same schema as the enabled one, and nothing else about the run changes.
    assert summary["bf16_override"] == {"count": 0, "enabled": False, "fqns": [], "path": None, "sha256": None}
    assert summary["counts_by_precision"] == {sq.PRECISION_NVFP4_WEIGHT_ONLY: 4 * NUM_LAYERS}
    assert summary["skipped_fqns"] == []
    assert fake_torchao.calls[0].selected == _expected_target_fqns()
    assert all("BF16 override file" not in note for note in summary["notes"])


@pytest.mark.unit
@pytest.mark.parametrize("recipe", ["disabled", "nvfp4_qkv_only", "nvfp4_qkv_fp8_rest"])
def test_bf16_override_is_rejected_for_every_other_recipe(tmp_path, recipe):
    path = _write_bf16_override(tmp_path, _fqns_for_suffix("ffn.net.0"))

    with pytest.raises(ValueError, match="requires one of recipe="):
        _bf16_config(path, quantization_recipe=recipe)


@pytest.mark.unit
@pytest.mark.parametrize("recipe", ["nvfp4_all", "nvfp4_weight_only"])
def test_bf16_override_is_accepted_by_both_uniform_recipes(tmp_path, recipe):
    path = _write_bf16_override(tmp_path, _fqns_for_suffix("ffn.net.0"))

    config = _bf16_config(path, quantization_recipe=recipe)

    assert sq.BF16_OVERRIDE_RECIPES == ("nvfp4_all", "nvfp4_weight_only")
    assert config.recipe == recipe
    assert config.has_bf16_override is True


@pytest.mark.unit
def test_bf16_override_rejects_an_empty_path():
    with pytest.raises(ValueError, match="quantization_bf16_override_path is empty"):
        _bf16_config("")


@pytest.mark.unit
def test_bf16_override_restores_one_family(tmp_path, fake_torchao):
    model = _FakeSortformer()
    restored = _fqns_for_suffix("ffn.net.3")
    config = _bf16_config(_write_bf16_override(tmp_path, restored))

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    remaining = sorted(set(_expected_target_fqns()) - set(restored))
    assert summary["counts_by_precision"] == {
        sq.PRECISION_BF16: NUM_LAYERS,
        sq.PRECISION_NVFP4_WEIGHT_ONLY: 3 * NUM_LAYERS,
    }
    assert summary["selected_fqns"][sq.PRECISION_BF16] == restored
    assert summary["selected_fqns"][sq.PRECISION_NVFP4_WEIGHT_ONLY] == remaining
    assert summary["skipped_fqns"] == restored
    # Only the unlisted FQNs are ever handed to TorchAO, and the restored modules stay ordinary BF16 linears.
    assert fake_torchao.calls[0].selected == remaining
    assert all(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in restored)
    assert not any(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in remaining)
    assert any("BF16 override file" in note for note in summary["notes"])


@pytest.mark.unit
def test_bf16_override_restores_one_family_under_w4a4(tmp_path, fake_torchao):
    """The unfused W4A4 sensitivity experiment: every attn.out_proj in BF16, every other target in NVFP4 W4A4."""
    model = _FakeSortformer()
    restored = _fqns_for_suffix("attn.out_proj")
    config = _bf16_config(_write_bf16_override(tmp_path, restored), quantization_recipe="nvfp4_all")

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    remaining = sorted(set(_expected_target_fqns()) - set(restored))
    assert summary["counts_by_precision"] == {
        sq.PRECISION_BF16: NUM_LAYERS,
        sq.PRECISION_NVFP4_W4A4: 3 * NUM_LAYERS,
    }
    assert summary["selected_fqns"][sq.PRECISION_BF16] == restored
    assert summary["selected_fqns"][sq.PRECISION_NVFP4_W4A4] == remaining
    assert summary["skipped_fqns"] == restored
    # Only the unlisted QKV/FFN FQNs are handed to TorchAO; the restored ones stay ordinary BF16 linears.
    assert [call.selected for call in fake_torchao.calls] == [remaining]
    assert all(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in restored)
    assert not any(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in remaining)
    assert any("'nvfp4_all' target set to BF16" in note for note in summary["notes"])


@pytest.mark.unit
def test_bf16_override_under_w4a4_calibrates_only_the_remaining_targets(tmp_path, fake_torchao):
    """A common calibration artifact stays usable: restored entries are unused, not rejected."""
    restored = _fqns_for_suffix("attn.out_proj")
    calibration = _write_calibration(tmp_path, {fqn: 1.5 for fqn in _expected_target_fqns()})
    config = _bf16_config(
        _write_bf16_override(tmp_path, restored),
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(calibration),
        quantization_scale_margin=2.0,
    )

    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    remaining = sorted(set(_expected_target_fqns()) - set(restored))
    assert [call.config.step for call in fake_torchao.calls] == ["prepare", "convert"]
    assert [call.selected for call in fake_torchao.calls] == [remaining, remaining]
    # The margin-scaled amax reaches the remaining W4A4 modules only.
    assert fake_torchao.calls[1].observed_amax == {fqn: 3.0 for fqn in remaining}
    assert summary["unused_calibration_fqns"] == restored
    assert summary["calibration_path"] == str(calibration)


@pytest.mark.unit
def test_producer_fusion_accepts_an_out_proj_bf16_override(tmp_path):
    """attn.out_proj is not a fused consumer: fusion never packs into it and reaches it only through
    attn.forward_from_qkv, so restoring it must be accepted rather than rejected at config time."""
    path = _write_bf16_override(tmp_path, _fqns_for_suffix("attn.out_proj"))

    config = _bf16_config(
        path,
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(tmp_path / "calib.json"),
        quantization_fuse_producer_packing=True,
    )

    assert config.fuse_producer_packing
    assert config.has_bf16_override


@pytest.mark.unit
@pytest.mark.parametrize("consumer", sq.PRODUCER_FUSION_REQUIRED_CONSUMERS)
def test_producer_fusion_rejects_a_bf16_override_of_any_fused_consumer(tmp_path, fake_torchao, consumer):
    """Restoring attn.w_qkv, ffn.net.0 or ffn.net.3 leaves a fused kernel with no pack to write into. The
    rejection must fire while the model is still unmodified, not after TorchAO has rewritten the weights."""
    path = _write_bf16_override(tmp_path, _fqns_for_suffix(consumer))
    config = _bf16_config(
        path,
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(tmp_path / "calib.json"),
        quantization_fuse_producer_packing=True,
    )

    with pytest.raises(ValueError, match="cannot be combined with a bf16_override that restores a"):
        sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    # The point of checking early is that nothing was converted, so assert it rather than trusting the message.
    assert fake_torchao.calls == []


@pytest.mark.unit
def test_bf16_override_restores_selected_layers(tmp_path, fake_torchao):
    restored = sorted(f"transformer_encoder.layers.0.{suffix}" for suffix in sq.QUANTIZATION_TARGET_SUFFIXES)
    config = _bf16_config(_write_bf16_override(tmp_path, reversed(restored)))

    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    assert summary["bf16_override"]["count"] == 4
    # The reported list is sorted regardless of the order the file happens to use.
    assert summary["bf16_override"]["fqns"] == restored
    assert summary["selected_fqns"][sq.PRECISION_BF16] == restored
    assert fake_torchao.calls[0].selected == sorted(set(_expected_target_fqns()) - set(restored))


@pytest.mark.unit
def test_bf16_override_summary_records_the_exact_file_identity(tmp_path, fake_torchao):
    path = _write_bf16_override(tmp_path, _fqns_for_suffix("attn.w_qkv"))
    config = _bf16_config(path)

    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    assert summary["bf16_override"] == {
        "count": NUM_LAYERS,
        "enabled": True,
        "fqns": _fqns_for_suffix("attn.w_qkv"),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.unit
def test_bf16_override_is_validated_before_the_model_is_mutated(tmp_path, fake_torchao):
    model = _FakeSortformer()
    path = _write_bf16_override(tmp_path, ["transformer_encoder.layers.0.attn.w_qkv", "head"])

    with pytest.raises(ValueError, match="not Sortformer quantization targets"):
        sq.quantize_sortformer_model(model, _bf16_config(path), facts=_facts())

    assert fake_torchao.calls == []
    assert all(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in _expected_target_fqns())


@pytest.mark.unit
def test_bf16_override_rejects_restoring_every_target(tmp_path):
    path = _write_bf16_override(tmp_path, _expected_target_fqns())

    with pytest.raises(ValueError, match="restores all 8 target modules"):
        sq.load_bf16_override(str(path), _weight_only_selection())


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload, message",
    [
        ("[]", "must contain a JSON object"),
        ('"nope"', "must contain a JSON object"),
        ("{", "not valid JSON"),
        ('{"version": 1, "bf16_fqns": ["transformer_encoder.layers.0.attn.w_qkv"], "version": 1}', "duplicate key"),
        ('{"version": 1, "bf16_fqns": ["transformer_encoder.layers.0.attn.w_qkv"], "extra": 1}', "unknown keys"),
        ('{"version": 1}', "missing keys"),
        ('{"bf16_fqns": ["transformer_encoder.layers.0.attn.w_qkv"]}', "missing keys"),
        ('{"version": 2, "bf16_fqns": ["transformer_encoder.layers.0.attn.w_qkv"]}', "version"),
        ('{"version": true, "bf16_fqns": ["transformer_encoder.layers.0.attn.w_qkv"]}', "version"),
        ('{"version": 1, "bf16_fqns": {"transformer_encoder.layers.0.attn.w_qkv": true}}', "as a list"),
        ('{"version": 1, "bf16_fqns": "transformer_encoder.layers.0.attn.w_qkv"}', "as a list"),
        ('{"version": 1, "bf16_fqns": [7]}', "non-empty module FQN string"),
        ('{"version": 1, "bf16_fqns": [""]}', "non-empty module FQN string"),
        ('{"version": 1, "bf16_fqns": ["   "]}', "non-empty module FQN string"),
        ('{"version": 1, "bf16_fqns": [null]}', "non-empty module FQN string"),
        (
            '{"version": 1, "bf16_fqns": ["transformer_encoder.layers.0.attn.w_qkv", '
            '"transformer_encoder.layers.0.attn.w_qkv"]}',
            "more than once",
        ),
        ('{"version": 1, "bf16_fqns": []}', "restores no module at all"),
        ('{"version": 1, "bf16_fqns": ["head"]}', "not Sortformer quantization targets"),
        ('{"version": 1, "bf16_fqns": ["transformer_encoder.layers.0.norm1"]}', "not Sortformer quantization"),
        ('{"version": 1, "bf16_fqns": ["transformer_encoder.layers.9.attn.w_qkv"]}', "not Sortformer quantization"),
    ],
)
def test_bf16_override_rejects_malformed_files(tmp_path, payload, message):
    path = tmp_path / "bf16.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        sq.load_bf16_override(str(path), _weight_only_selection())


@pytest.mark.unit
def test_bf16_override_rejects_a_non_utf8_file(tmp_path):
    path = tmp_path / "bf16.json"
    path.write_bytes(b'{"version": 1, "bf16_fqns": ["\xff"]}')

    with pytest.raises(ValueError, match="must be UTF-8"):
        sq.load_bf16_override(str(path), _weight_only_selection())


@pytest.mark.unit
def test_bf16_override_selection_rejects_a_non_target_fqn():
    with pytest.raises(ValueError, match="not Sortformer quantization targets"):
        sq.select_quantization_targets(_FakeSortformer(), "nvfp4_weight_only", bf16_fqns=["head"])


@pytest.mark.unit
def test_bf16_override_does_not_import_optional_dependencies_when_absent(monkeypatch):
    """The BF16 path of a disabled recipe must stay free of TorchAO and MSLK, exactly as before."""
    real_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torchao", "mslk"), f"disabled path imported '{name}'"
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(sq.importlib, "import_module", guarded_import)

    summary = sq.quantize_sortformer_model(_FakeSortformer(), sq.SortformerQuantizationConfig())

    assert summary["bf16_override"] == sq.disabled_bf16_override_summary()


@pytest.mark.unit
def test_prediction_cache_identity_tracks_the_bf16_override_contents(tmp_path):
    """Two sensitivity runs differ only in this file's bytes, so its digest must separate their caches."""
    path = _write_bf16_override(tmp_path, ["transformer_encoder.layers.0.attn.w_qkv"])
    config = sq.SortformerQuantizationConfig(recipe="nvfp4_weight_only", bf16_override_path=str(path))

    original = sq.prediction_cache_identity(config)
    assert original["bf16_override_path"] == str(path)
    assert original["bf16_override_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    # Same path, same byte count, same modification time: only the digest can tell the two runs apart.
    stat = path.stat()
    _write_bf16_override(tmp_path, ["transformer_encoder.layers.1.attn.w_qkv"])
    os.utime(path, ns=(stat.st_mtime_ns, stat.st_mtime_ns))
    rewritten = sq.prediction_cache_identity(config)
    assert path.stat().st_size == stat.st_size
    assert rewritten["bf16_override_file"] == original["bf16_override_file"]
    assert rewritten["bf16_override_sha256"] != original["bf16_override_sha256"]
    assert rewritten != original

    # A run without an override is a third, distinct identity.
    without = sq.prediction_cache_identity(sq.SortformerQuantizationConfig(recipe="nvfp4_weight_only"))
    assert without["bf16_override_path"] is None
    assert without["bf16_override_sha256"] is None
    assert without != original


@pytest.mark.unit
def test_prediction_cache_identity_fails_closed_on_an_unreadable_override(tmp_path):
    missing = sq.SortformerQuantizationConfig(
        recipe="nvfp4_weight_only", bf16_override_path=str(tmp_path / "absent.json")
    )

    with pytest.raises(ValueError, match="could not be read"):
        sq.prediction_cache_identity(missing)


@pytest.mark.unit
def test_evaluator_exposes_the_bf16_override_option():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert "quantization_bf16_override_path: Optional[str] = None" in source


def _mse_config(**overrides):
    """Quantization config with the block-MSE weight-scale search on, overridable per test."""
    values = dict(recipe="nvfp4_all", weight_scale_method=sq.WEIGHT_SCALE_METHOD_MSE)
    values.update(overrides)
    return sq.SortformerQuantizationConfig(**values)


def _tolerated_mse(template_mse):
    """The largest searched MSE the integration accepts for a given converted MSE, using its own tolerances."""
    return template_mse * (1.0 + sq.WEIGHT_SCALE_MSE_RELATIVE_TOLERANCE) + sq.WEIGHT_SCALE_MSE_ABSOLUTE_TOLERANCE


def _original_weights(model, fqns):
    """Snapshot of the high-precision weights the search must be measured against."""
    return {fqn: model.get_submodule(fqn).weight.detach().clone() for fqn in fqns}


@pytest.mark.unit
def test_weight_scale_method_defaults_to_amax_everywhere():
    """The searched repack is opt-in: nothing about a default run mentions or performs it."""
    assert sq.SortformerQuantizationConfig().weight_scale_method == sq.WEIGHT_SCALE_METHOD_AMAX
    assert sq.SortformerQuantizationConfig().uses_mse_weight_scales is False

    default = sq.quantization_config_from_eval_cfg(_eval_cfg())
    assert default.weight_scale_method == "amax"
    assert default.uses_mse_weight_scales is False

    # The evaluator field is wired to the core option exactly, with no coercion of an unknown value.
    enabled = sq.quantization_config_from_eval_cfg(
        _eval_cfg(quantization_recipe="nvfp4_all", quantization_weight_scale_method="mse")
    )
    assert enabled.weight_scale_method == "mse"
    assert enabled.uses_mse_weight_scales is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"quantization_weight_scale_method": "argmin"}, "weight_scale_method must be one of"),
        ({"quantization_recipe": "disabled"}, "quantizes nothing"),
        ({"quantization_fold_global_scales": True}, "mutually exclusive"),
    ],
)
def test_weight_scale_method_rejects_incompatible_options(tmp_path, overrides, message):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    values = dict(
        quantization_recipe="nvfp4_all",
        quantization_scale_mode="static",
        quantization_calibration_path=str(path),
        quantization_weight_scale_method="mse",
    )
    values.update(overrides)
    if values["quantization_recipe"] == "disabled":
        values["quantization_scale_mode"] = "dynamic"
        values["quantization_calibration_path"] = None

    with pytest.raises(ValueError, match=message):
        sq.quantization_config_from_eval_cfg(_eval_cfg(**values))


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"weight_scale_method": "argmin"}, "weight_scale_method must be one of"),
        ({"recipe": "disabled"}, "quantizes nothing"),
        ({"fold_global_scales": True, "scale_mode": "static"}, "mutually exclusive"),
    ],
)
def test_weight_scale_method_is_rejected_before_the_model_is_mutated(tmp_path, fake_torchao, overrides, message):
    """An incompatible request must fail on a still-unquantized model, never halfway through the conversion."""
    model = _FakeSortformer()
    values = dict(overrides)
    if values.get("scale_mode") == "static":
        values["calibration_path"] = str(_write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()}))

    with pytest.raises(ValueError, match=message):
        sq.quantize_sortformer_model(model, _mse_config(**values), facts=_facts())

    assert fake_torchao.calls == []
    assert all(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in _expected_target_fqns())


@pytest.mark.unit
def test_amax_default_keeps_the_batched_torchao_call_sequence(tmp_path, fake_torchao, monkeypatch):
    """The ordinary path must still issue one batched conversion call and never touch the repacker."""

    def forbidden(weight, template):
        raise AssertionError("the amax weight-scale path must not call the block-MSE repacker")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_mse", forbidden)
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})

    dynamic = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )
    assert [call.selected for call in fake_torchao.calls] == [_expected_target_fqns()]
    assert dynamic["weight_scale_method"] == "amax"
    assert dynamic["weight_scale_mse"] == sq.disabled_weight_scale_mse_summary()
    assert dynamic["weight_scale_mse"]["enabled"] is False
    assert any("OFF" in note for note in dynamic["weight_scale_mse"]["notes"])

    fake_torchao.calls.clear()
    sq.quantize_sortformer_model(
        _FakeSortformer(),
        sq.SortformerQuantizationConfig(recipe="nvfp4_all", scale_mode="static", calibration_path=str(path)),
        facts=_facts(),
    )
    assert [call.config.step for call in fake_torchao.calls] == ["prepare", "convert"]
    assert [call.selected for call in fake_torchao.calls] == [_expected_target_fqns()] * 2

    fake_torchao.calls.clear()
    sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_weight_only"), facts=_facts()
    )
    assert [call.selected for call in fake_torchao.calls] == [_expected_target_fqns()]


@pytest.mark.unit
def test_dynamic_mse_path_converts_and_repacks_each_selected_fqn_once_in_order(fake_torchao, fake_mse_repack):
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)

    summary = sq.quantize_sortformer_model(model, _mse_config(), facts=_facts())

    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert [call.config.step for call in fake_torchao.calls] == [None] * len(expected)
    assert len(fake_mse_repack) == len(expected)
    for index, fqn in enumerate(expected):
        record = fake_mse_repack[index]
        assert torch.equal(record.weight, originals[fqn])
        # Each repack ran after its own conversion and before the next one was requested.
        assert record.quantize_calls == index + 1
        # No earlier original weight was still alive: at most one high-precision clone exists at a time.
        assert record.earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_mse"]["layers"]] == expected


@pytest.mark.unit
def test_static_mse_path_assigns_every_amax_before_the_per_fqn_converts(tmp_path, fake_torchao, fake_mse_repack):
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)
    path = _write_calibration(tmp_path, {fqn: 1.5 for fqn in expected})
    config = _mse_config(scale_mode="static", calibration_path=str(path), scale_margin=2.0)

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    prepare, *converts = fake_torchao.calls
    assert prepare.config.step == "prepare"
    assert prepare.selected == expected
    assert [call.config.step for call in converts] == ["convert"] * len(expected)
    assert [call.selected for call in converts] == [[fqn] for fqn in expected]
    # Every calibrated amax was written before the first per-FQN convert, so no module converted without it.
    assert [call.observed_amax for call in converts] == [{fqn: 3.0} for fqn in expected]
    for index, fqn in enumerate(expected):
        assert torch.equal(fake_mse_repack[index].weight, originals[fqn])
        assert fake_mse_repack[index].quantize_calls == index + 2
        assert fake_mse_repack[index].earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_mse"]["layers"]] == expected


@pytest.mark.unit
def test_weight_only_mse_path_converts_and_repacks_each_selected_fqn_once_in_order(fake_torchao, fake_mse_repack):
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)

    summary = sq.quantize_sortformer_model(model, _mse_config(recipe="nvfp4_weight_only"), facts=_facts())

    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert all(isinstance(call.config, _FakeNVFP4WeightOnlyConfig) for call in fake_torchao.calls)
    for index, fqn in enumerate(expected):
        assert torch.equal(fake_mse_repack[index].weight, originals[fqn])
        assert fake_mse_repack[index].quantize_calls == index + 1
        assert fake_mse_repack[index].earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_mse"]["layers"]] == expected


@pytest.mark.unit
def test_mse_never_repacks_bf16_restored_targets(tmp_path, fake_torchao, fake_mse_repack):
    restored = [f"transformer_encoder.layers.{index}.attn.w_qkv" for index in range(NUM_LAYERS)]
    path = _write_bf16_override(tmp_path, restored)
    config = _mse_config(recipe="nvfp4_weight_only", bf16_override_path=str(path))

    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    quantized = sorted(set(_expected_target_fqns()) - set(restored))
    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in quantized]
    assert [layer["fqn"] for layer in summary["weight_scale_mse"]["layers"]] == quantized
    assert summary["weight_scale_mse"]["target_count"] == len(quantized)


@pytest.mark.unit
def test_mse_never_repacks_fp8_modules(fake_torchao, fake_mse_repack):
    summary = sq.quantize_sortformer_model(_FakeSortformer(), _mse_config(recipe="nvfp4_qkv_fp8_rest"), facts=_facts())

    qkv = _fqns_for_suffix("attn.w_qkv")
    # One conversion per NVFP4 target, then the single unchanged batched FP8 call, which is never repacked.
    assert [call.selected for call in fake_torchao.calls[:-1]] == [[fqn] for fqn in qkv]
    assert isinstance(fake_torchao.calls[-1].config, _FakeFP8Config)
    assert fake_torchao.calls[-1].selected == sorted(
        _fqns_for_suffix("attn.out_proj") + _fqns_for_suffix("ffn.net.0") + _fqns_for_suffix("ffn.net.3")
    )
    assert len(fake_mse_repack) == len(qkv)
    assert [layer["fqn"] for layer in summary["weight_scale_mse"]["layers"]] == qkv


@pytest.mark.unit
def test_producer_fusion_runs_after_every_mse_repack(tmp_path, fake_torchao, fake_mse_repack, monkeypatch):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    config = _mse_config(scale_mode="static", calibration_path=str(path), fuse_producer_packing=True)
    recorded = {}

    def fake_fuse(model, fqns):
        recorded["repacks"] = len(fake_mse_repack)
        recorded["fqns"] = list(fqns)
        return {
            "enabled": True,
            "fused_block_count": len(producer_fusion.group_producer_fusion_blocks(fqns)),
            "fused_block_fqns": list(producer_fusion.group_producer_fusion_blocks(fqns)),
            "fused_boundaries": list(producer_fusion.FUSED_PRODUCER_BOUNDARIES),
            "notes": [],
        }

    monkeypatch.setattr(sq, "apply_producer_fusion", fake_fuse)
    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    assert recorded["fqns"] == _expected_target_fqns()
    assert recorded["repacks"] == len(_expected_target_fqns())
    assert summary["producer_fusion"]["enabled"] is True
    assert summary["weight_scale_mse"]["target_count"] == len(_expected_target_fqns())


@pytest.mark.unit
def test_mse_replacement_preserves_parameter_semantics(fake_torchao, fake_mse_repack):
    model = _FakeSortformer()

    sq.quantize_sortformer_model(model, _mse_config(), facts=_facts())

    for fqn in _expected_target_fqns():
        module = model.get_submodule(fqn)
        weight = module.weight
        assert isinstance(weight, torch.nn.Parameter)
        assert isinstance(weight, _FakeNVFP4Tensor)
        assert weight.requires_grad is False
        assert "weight" in dict(module.named_parameters())


@pytest.mark.unit
def test_mse_repack_rejects_a_weight_the_conversion_did_not_replace():
    """A module the conversion left in high precision must fail closed, not be repacked as if it were NVFP4."""
    model = _FakeSortformer()
    fqn = "transformer_encoder.layers.0.attn.w_qkv"
    original = model.get_submodule(fqn).weight.detach().clone()

    with pytest.raises(RuntimeError, match="instead of a TorchAO NVFP4 tensor"):
        sq._repack_weight_with_mse(model, fqn, original, "dynamic NVFP4 W4A4")


@pytest.mark.unit
def test_mse_repack_rejects_a_module_without_a_weight_or_without_an_fqn():
    model = _FakeSortformer()

    with pytest.raises(RuntimeError, match="does not expose a 'weight' tensor"):
        sq._clone_original_weight(model, "transformer_encoder.layers.0.ffn.net.1", "NVFP4 weight-only")
    with pytest.raises(RuntimeError, match="is not present in the model"):
        sq._clone_original_weight(model, "transformer_encoder.layers.9.attn.w_qkv", "NVFP4 weight-only")


@pytest.mark.unit
@pytest.mark.parametrize(
    "produce, message",
    [
        (lambda weight: _fake_nvfp4(weight.float() + 1.0), "made"),
        (lambda weight: _fake_nvfp4(torch.full(weight.shape, float("nan"))), "non-finite"),
        (lambda weight: _fake_nvfp4(_round_to(weight, SEARCHED_GRID_STEP)[:, :16]), "dequantizes to shape"),
        (lambda weight: torch.nn.Parameter(weight.detach().clone()), "produced an ordinary Parameter"),
        (lambda weight: None, "produced no weight"),
    ],
)
def test_mse_repack_fails_closed_on_an_unusable_result(fake_torchao, monkeypatch, produce, message):
    """No fallback: an unusable repack raises instead of leaving the amax-derived weight silently in place."""
    monkeypatch.setattr(sq, "repack_nvfp4_weight_mse", lambda weight, template: produce(weight))

    with pytest.raises(RuntimeError, match=message):
        sq.quantize_sortformer_model(_FakeSortformer(), _mse_config(), facts=_facts())


@pytest.mark.unit
def test_mse_report_is_deterministic_json_safe_and_weight_count_weighted(fake_torchao, fake_mse_repack):
    torch.manual_seed(1234)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)

    summary = sq.quantize_sortformer_model(model, _mse_config(), facts=_facts())

    report = summary["weight_scale_mse"]
    assert report["enabled"] is True
    assert report["method"] == sq.WEIGHT_SCALE_METHOD_MSE
    assert report["algorithm"] == sq.WEIGHT_SCALE_MSE_ALGORITHM
    assert report["algorithm_version"] == sq.WEIGHT_SCALE_MSE_ALGORITHM_VERSION
    assert report["target_count"] == len(expected)
    assert report["total_weight_count"] == sum(originals[fqn].numel() for fqn in expected)
    assert [layer["fqn"] for layer in report["layers"]] == expected

    for layer in report["layers"]:
        original = originals[layer["fqn"]]
        assert layer["shape"] == list(original.shape)
        assert layer["weight_count"] == original.numel()
        assert 0.0 < layer["searched_mse"] < layer["template_mse"]
        assert layer["ratio"] == pytest.approx(layer["searched_mse"] / layer["template_mse"])
        assert layer["relative_reduction"] == pytest.approx(1.0 - layer["ratio"])

    total = report["total_weight_count"]
    for field in ("template_mse", "searched_mse"):
        assert report["aggregate"][field] == pytest.approx(
            sum(layer[field] * layer["weight_count"] for layer in report["layers"]) / total
        )
    assert report["aggregate"]["ratio"] == pytest.approx(
        report["aggregate"]["searched_mse"] / report["aggregate"]["template_mse"]
    )
    assert report["aggregate"]["relative_reduction"] == pytest.approx(1.0 - report["aggregate"]["ratio"])

    assert summary["weight_scale_method"] == "mse"
    assert any(sq.WEIGHT_SCALE_MSE_ALGORITHM in note for note in summary["notes"])
    assert json.loads(json.dumps(summary)) == summary

    # The same weights produce the same evidence, run after run.
    torch.manual_seed(1234)
    repeated = sq.quantize_sortformer_model(_FakeSortformer(), _mse_config(), facts=_facts())
    assert repeated["weight_scale_mse"] == report


@pytest.mark.unit
def test_mse_report_uses_a_defined_zero_baseline():
    """An exactly reconstructed weight cannot be improved on, so it is reported as ratio 1.0, reduction 0.0."""
    assert sq._mse_ratio_and_reduction(0.0, 0.0) == (1.0, 0.0)
    assert sq._mse_ratio_and_reduction(4.0, 1.0) == (0.25, 0.75)
    assert sq._weight_scale_mse_aggregate([]) == {
        "template_mse": None,
        "searched_mse": None,
        "ratio": None,
        "relative_reduction": None,
    }


@pytest.mark.unit
def test_prediction_cache_identity_separates_searched_from_amax_runs():
    """MSE predictions must never reuse an amax cache, while the amax identity itself stays unchanged."""
    amax = sq.prediction_cache_identity(sq.SortformerQuantizationConfig(recipe="nvfp4_all"))
    searched = sq.prediction_cache_identity(_mse_config())

    # The default identity carries no new key at all, so caches written before the search existed still match.
    assert sorted(amax) == [
        "accelerated_packing",
        "bf16_override_file",
        "bf16_override_path",
        "bf16_override_sha256",
        "calibration_file",
        "calibration_path",
        "fold_activation_exponent",
        "fold_global_scales",
        "fuse_producer_packing",
        "recipe",
        "scale_margin",
        "scale_mode",
    ]
    assert searched["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_MSE
    assert searched["weight_scale_mse_algorithm"] == sq.WEIGHT_SCALE_MSE_ALGORITHM
    assert searched["weight_scale_mse_algorithm_version"] == sq.WEIGHT_SCALE_MSE_ALGORITHM_VERSION
    assert searched != amax
    assert {key: searched[key] for key in amax} == amax


@pytest.mark.unit
def test_evaluator_exposes_the_weight_scale_method_option():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert 'quantization_weight_scale_method: str = "amax"' in source


def _hessian_moments(width, index=0):
    """Deterministic, strictly positive second moments; ``index`` makes different modules differ."""
    return [round(0.5 + 0.25 * ((position + index) % 7), 6) for position in range(width)]


def _hessian_payload(model, fqns=None, moments=None, **overrides):
    """A complete, valid diagonal-Hessian artifact for ``model``, overridable key by key."""
    fqns = list(fqns if fqns is not None else _expected_target_fqns())
    modules = dict(model.named_modules())
    if moments is None:
        moments = {fqn: _hessian_moments(modules[fqn].in_features, index) for index, fqn in enumerate(fqns)}
    values = [value for fqn in fqns for value in moments[fqn]]
    payload = {
        "schema": sq.HESSIAN_SCHEMA,
        "version": sq.HESSIAN_SCHEMA_VERSION,
        "checkpoint_sha256": "a" * 64,
        "algorithm": sq.WEIGHT_SCALE_HESSIAN_ALGORITHM,
        "algorithm_version": sq.WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION,
        "damping": sq.WEIGHT_SCALE_HESSIAN_DAMPING,
        "weight_digest_method": sq.WEIGHT_DIGEST_METHOD,
        "weight_sha256": {fqn: sq.nvfp4_weight_digest(modules[fqn].weight) for fqn in fqns},
        "diagonal_hessian": {fqn: list(moments[fqn]) for fqn in fqns},
        "provenance": {
            "method": sq.HESSIAN_CONSTRUCTION_METHOD,
            "method_version": sq.HESSIAN_CONSTRUCTION_METHOD_VERSION,
            "objective": sq.HESSIAN_OBJECTIVE,
            "group_reduction": sq.HESSIAN_GROUP_REDUCTION,
            "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
            "target_module_count": len(fqns),
            "target_fqns": list(fqns),
            "sources": [
                {
                    "label": "near_field",
                    "name": "samples_near.pt",
                    "sha256": "b" * 64,
                    "size_bytes": 4096,
                    "seed": 7,
                    "max_rows": 512,
                    "sampled_row_count": 256,
                    "finite_row_count": 4096,
                    "nonfinite_row_count": 0,
                    "metadata": {"manifest": "near.json"},
                }
            ],
            "aggregate": {
                "module_count": len(fqns),
                "source_count": 1,
                "source_labels": ["near_field"],
                "moment_count": len(values),
                "moment_min": min(values),
                "moment_max": max(values),
            },
        },
    }
    # Recorded exactly as the builder records them, so a payload is only ever mutated below with the digests it
    # would have carried before the mutation.
    payload["moment_sha256"] = sq.nvfp4_section_digest(payload["diagonal_hessian"])
    payload["provenance_sha256"] = sq.nvfp4_section_digest(payload["provenance"])
    payload.update(overrides)
    return payload


def _write_hessian(tmp_path, payload, name="hessian.json"):
    """Write an artifact payload and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _hessian_config(path, **overrides):
    """Quantization config with the activation-weighted search on, overridable per test."""
    values = dict(
        recipe="nvfp4_all",
        weight_scale_method=sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN,
        weight_scale_hessian_path=str(path),
    )
    values.update(overrides)
    return sq.SortformerQuantizationConfig(**values)


@pytest.fixture
def fake_hessian_repack(fake_torchao, monkeypatch):
    """Replace the activation-weighted repacker with a deterministic stand-in and record every call."""
    calls = []
    originals = []

    def repack(weight, template, second_moments):
        calls.append(
            SimpleNamespace(
                weight=weight.detach().clone(),
                template=template,
                moments=second_moments.detach().clone(),
                quantize_calls=len(fake_torchao.calls),
                earlier_originals_alive=[reference() is not None for reference in originals],
            )
        )
        originals.append(weakref.ref(weight))
        return _fake_nvfp4(_round_to(weight, SEARCHED_GRID_STEP))

    monkeypatch.setattr(sq, "repack_nvfp4_weight_local_hessian", repack)
    return calls


@pytest.mark.unit
def test_local_hessian_is_off_by_default_and_maps_from_the_evaluator(tmp_path):
    """The technique is opt-in end to end, and both evaluator fields are wired to the core options exactly."""
    default = sq.SortformerQuantizationConfig()
    assert default.weight_scale_method == sq.WEIGHT_SCALE_METHOD_AMAX
    assert default.weight_scale_hessian_path is None
    assert default.uses_local_hessian_weight_scales is False
    assert default.uses_searched_weight_scales is False

    path = _write_hessian(tmp_path, _hessian_payload(_FakeSortformer()))
    config = sq.quantization_config_from_eval_cfg(
        _eval_cfg(
            quantization_recipe="nvfp4_all",
            quantization_weight_scale_method="local_hessian",
            quantization_weight_scale_hessian_path=str(path),
        )
    )
    assert config.weight_scale_method == "local_hessian"
    assert config.weight_scale_hessian_path == str(path)
    assert config.uses_local_hessian_weight_scales is True
    assert config.uses_searched_weight_scales is True
    assert config.uses_mse_weight_scales is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recipe": "nvfp4_weight_only"}, "requires one of recipe"),
        ({"recipe": "nvfp4_qkv_only"}, "requires one of recipe"),
        ({"weight_scale_hessian_path": None}, "requires quantization_weight_scale_hessian_path"),
        ({"weight_scale_hessian_path": "  "}, "requires quantization_weight_scale_hessian_path"),
        ({"weight_scale_hessian_path": "/nonexistent/hessian.json"}, "is not a readable file"),
        ({"fold_global_scales": True, "scale_mode": "static"}, "mutually exclusive"),
        ({"bf16_override_path": "override.json"}, "cannot be combined"),
    ],
)
def test_local_hessian_rejects_incompatible_options(tmp_path, overrides, message):
    path = _write_hessian(tmp_path, _hessian_payload(_FakeSortformer()))
    values = dict(overrides)
    if values.get("scale_mode") == "static":
        values["calibration_path"] = str(_write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()}))
    with pytest.raises(ValueError, match=message):
        _hessian_config(path, **values).validate()


@pytest.mark.unit
@pytest.mark.parametrize("method", [sq.WEIGHT_SCALE_METHOD_AMAX, sq.WEIGHT_SCALE_METHOD_MSE])
def test_other_methods_reject_a_supplied_hessian_path(tmp_path, method):
    """An ignored path would let a run believe it was activation weighted while executing other scales."""
    path = _write_hessian(tmp_path, _hessian_payload(_FakeSortformer()))
    with pytest.raises(ValueError, match="is only used with"):
        sq.SortformerQuantizationConfig(
            recipe="nvfp4_all", weight_scale_method=method, weight_scale_hessian_path=str(path)
        ).validate()


@pytest.mark.unit
def test_local_hessian_converts_and_repacks_each_fqn_once_with_its_own_moments(
    tmp_path, fake_torchao, fake_hessian_repack
):
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)
    payload = _hessian_payload(model)
    path = _write_hessian(tmp_path, payload)

    summary = sq.quantize_sortformer_model(model, _hessian_config(path), facts=_facts())

    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert len(fake_hessian_repack) == len(expected)
    for index, fqn in enumerate(expected):
        record = fake_hessian_repack[index]
        assert torch.equal(record.weight, originals[fqn])
        assert record.moments.tolist() == pytest.approx(payload["diagonal_hessian"][fqn])
        assert record.moments.device == originals[fqn].device
        # Each repack ran after its own conversion, and no earlier original weight was still alive.
        assert record.quantize_calls == index + 1
        assert record.earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_hessian"]["layers"]] == expected
    # The disabled MSE section names the method that actually selected the scales instead of claiming amax.
    assert summary["weight_scale_mse"] == sq.disabled_weight_scale_mse_summary(sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN)
    assert summary["weight_scale_mse"]["method"] == sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN
    assert not any("amax-derived" in note for note in summary["weight_scale_mse"]["notes"])


@pytest.mark.unit
def test_local_hessian_summary_records_the_artifact_and_both_objectives(tmp_path, fake_torchao, fake_hessian_repack):
    torch.manual_seed(99)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)
    payload = _hessian_payload(model)
    path = _write_hessian(tmp_path, payload)

    summary = sq.quantize_sortformer_model(model, _hessian_config(path), facts=_facts())

    report = summary["weight_scale_hessian"]
    assert report["enabled"] is True
    assert report["method"] == sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN
    assert report["algorithm"] == sq.WEIGHT_SCALE_HESSIAN_ALGORITHM
    assert report["algorithm_version"] == sq.WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION
    assert report["damping"] == sq.WEIGHT_SCALE_HESSIAN_DAMPING == 0.01
    assert report["artifact_path"] == str(path)
    assert report["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # The verified component digests are part of the run's evidence, not only of the loader's internals.
    assert report["moment_sha256"] == sq.nvfp4_section_digest(payload["diagonal_hessian"])
    assert report["provenance_sha256"] == sq.nvfp4_section_digest(payload["provenance"])
    assert report["checkpoint_sha256"] == payload["checkpoint_sha256"]
    assert report["target_count"] == len(expected)
    assert report["target_fqns"] == expected
    assert report["total_weight_count"] == sum(originals[fqn].numel() for fqn in expected)

    for layer in report["layers"]:
        assert layer["shape"] == list(originals[layer["fqn"]].shape)
        assert 0.0 < layer["searched_objective"] < layer["template_objective"]
        assert layer["ratio"] == pytest.approx(layer["searched_objective"] / layer["template_objective"])
        assert layer["relative_reduction"] == pytest.approx(1.0 - layer["ratio"])
        # The unweighted MSE is carried as diagnostic evidence, not as the objective.
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] > 0.0
        # The objective is the absolute sum_j h_damped[j] * (w_j - q_j)^2 per weight element, with h_damped the
        # artifact's own moments damped by 1% of their mean and nothing divided back out: a per-layer
        # normalization would report this scaled down by that layer's max(h_damped).
        original = originals[layer["fqn"]].float()
        damped = torch.tensor(payload["diagonal_hessian"][layer["fqn"]], dtype=torch.float32)
        damped = (damped + sq.WEIGHT_SCALE_HESSIAN_DAMPING * damped.mean()).double()
        assert float(damped.max()) > 1.0
        for stage, step in (("template", TEMPLATE_GRID_STEP), ("searched", SEARCHED_GRID_STEP)):
            squared = (_round_to(original, step) - original) ** 2
            expected_objective = float((squared.double() * damped[None, :]).mean())
            assert layer[f"{stage}_objective"] == pytest.approx(expected_objective, rel=1e-9)

    total = report["total_weight_count"]
    for field in ("template_objective", "searched_objective", "template_mse", "searched_mse"):
        assert report["aggregate"][field] == pytest.approx(
            sum(layer[field] * layer["weight_count"] for layer in report["layers"]) / total
        )
    assert summary["weight_scale_method"] == "local_hessian"
    assert any(sq.WEIGHT_SCALE_HESSIAN_ALGORITHM in note for note in summary["notes"])
    assert any("inference kernels" in note for note in report["notes"])
    assert any("makes no claim about DER" in note for note in report["notes"])
    assert json.loads(json.dumps(summary)) == summary

    # The same weights and the same artifact select the same scales, so the reported evidence is reproducible.
    torch.manual_seed(99)
    repeated = sq.quantize_sortformer_model(_FakeSortformer(), _hessian_config(path), facts=_facts())
    assert repeated["weight_scale_hessian"]["layers"] == report["layers"]


@pytest.mark.unit
def test_local_hessian_fails_before_the_model_is_mutated_when_the_weights_moved(
    tmp_path, fake_torchao, fake_hessian_repack
):
    """The artifact is bound to the exact original weights, and the check runs before any conversion."""
    model = _FakeSortformer()
    path = _write_hessian(tmp_path, _hessian_payload(model))
    with torch.no_grad():
        model.get_submodule("transformer_encoder.layers.0.attn.w_qkv").weight.add_(1.0)

    with pytest.raises(ValueError, match="built on different weights"):
        sq.quantize_sortformer_model(model, _hessian_config(path), facts=_facts())

    assert fake_torchao.calls == []
    assert fake_hessian_repack == []
    assert all(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in _expected_target_fqns())


@pytest.mark.unit
def test_local_hessian_summary_is_disabled_for_the_other_methods(fake_torchao, fake_mse_repack):
    summary = sq.quantize_sortformer_model(_FakeSortformer(), _mse_config(), facts=_facts())

    assert summary["weight_scale_hessian"] == sq.disabled_weight_scale_hessian_summary(sq.WEIGHT_SCALE_METHOD_MSE)
    assert summary["weight_scale_hessian"]["enabled"] is False
    # The disabled section names the method that ran, so it never reads as evidence of an amax run either.
    assert summary["weight_scale_hessian"]["method"] == sq.WEIGHT_SCALE_METHOD_MSE
    assert summary["weight_scale_hessian"]["artifact_sha256"] is None
    assert summary["weight_scale_hessian"]["moment_sha256"] is None
    assert summary["weight_scale_hessian"]["provenance_sha256"] is None
    assert any("OFF" in note for note in summary["weight_scale_hessian"]["notes"])


@pytest.mark.unit
def test_hessian_artifact_loads_and_reports_exactly_what_it_carries(tmp_path):
    model = _FakeSortformer()
    payload = _hessian_payload(model)
    path = _write_hessian(tmp_path, payload)
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    loaded = sq.load_diagonal_hessian(str(path), model, selection)

    assert loaded["path"] == str(path)
    assert loaded["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded["checkpoint_sha256"] == payload["checkpoint_sha256"]
    assert loaded["damping"] == sq.WEIGHT_SCALE_HESSIAN_DAMPING
    assert loaded["fqns"] == _expected_target_fqns()
    assert loaded["second_moments"] == payload["diagonal_hessian"]
    assert loaded["weight_sha256"] == payload["weight_sha256"]
    # The component digests were recomputed from the parsed sections, not copied out of the file.
    assert loaded["moment_sha256"] == sq.nvfp4_section_digest(payload["diagonal_hessian"]) == payload["moment_sha256"]
    assert loaded["provenance_sha256"] == sq.nvfp4_section_digest(payload["provenance"])
    assert loaded["provenance_sha256"] == payload["provenance_sha256"]


@pytest.mark.unit
def test_section_digest_is_canonical_and_value_bound():
    """The section digest ignores key order and formatting, and changes with any value it covers."""
    section = {"b": [1.0, 2.5], "a": {"x": 1, "y": "two"}}

    assert sq.nvfp4_section_digest(section) == sq.nvfp4_section_digest({"a": {"y": "two", "x": 1}, "b": [1.0, 2.5]})
    # Round-tripping through the artifact's own pretty-printed JSON must not change the digest, which is what
    # lets the builder record it and the runtime recompute it.
    assert sq.nvfp4_section_digest(section) == sq.nvfp4_section_digest(json.loads(json.dumps(section, indent=2)))
    assert sq.nvfp4_section_digest(section) != sq.nvfp4_section_digest({"b": [1.0, 2.6], "a": {"x": 1, "y": "two"}})
    with pytest.raises(ValueError, match="cannot be canonicalized"):
        sq.nvfp4_section_digest({"a": float("nan")})


@pytest.mark.unit
def test_weight_digest_binds_dtype_shape_and_bytes():
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    assert sq.nvfp4_weight_digest(weight) == sq.nvfp4_weight_digest(weight.clone())
    assert sq.nvfp4_weight_digest(weight) != sq.nvfp4_weight_digest(weight.reshape(4, 2))
    assert sq.nvfp4_weight_digest(weight) != sq.nvfp4_weight_digest(weight.to(torch.bfloat16))
    mutated = weight.clone()
    mutated[0, 0] += 1.0
    assert sq.nvfp4_weight_digest(weight) != sq.nvfp4_weight_digest(mutated)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload, model: payload.pop("damping"), "top-level keys"),
        (lambda payload, model: payload.update(extra=1), "top-level keys"),
        (lambda payload, model: payload.update(schema="something_else"), "declares schema"),
        (lambda payload, model: payload.update(version=2), "version"),
        (lambda payload, model: payload.update(version=True), "version"),
        (lambda payload, model: payload.update(algorithm="other"), "was built for algorithm"),
        (lambda payload, model: payload.update(algorithm_version=2), "was built for algorithm"),
        (lambda payload, model: payload.update(damping=0.02), "records damping"),
        (lambda payload, model: payload.update(damping="0.01"), "records damping"),
        (lambda payload, model: payload.update(weight_digest_method="md5"), "weight-digest method"),
        (lambda payload, model: payload.update(checkpoint_sha256="nope"), "hexadecimal SHA-256"),
        (lambda payload, model: payload["provenance"].pop("sources"), "provenance keys"),
        (lambda payload, model: payload["provenance"].update(extra=1), "provenance keys"),
        (lambda payload, model: payload["provenance"].update(method="other"), "records 'method'"),
        (lambda payload, model: payload["provenance"].update(method_version=2), "records 'method_version'"),
        (lambda payload, model: payload["provenance"].update(objective="argmin"), "records 'objective'"),
        (lambda payload, model: payload["provenance"].update(group_reduction="max"), "records 'group_reduction'"),
        (lambda payload, model: payload["provenance"].update(targets=["attn.w_qkv"]), "declares targets"),
        (lambda payload, model: payload["provenance"].update(target_fqns=["a"]), "target FQN"),
        (lambda payload, model: payload["provenance"].update(target_module_count=3), "target_module_count"),
        (lambda payload, model: payload["provenance"].update(sources=[]), "non-empty 'sources'"),
        (lambda payload, model: payload["provenance"]["sources"][0].pop("seed"), "source keys"),
        (lambda payload, model: payload["provenance"]["sources"][0].update(label=" "), "non-empty 'label'"),
        (lambda payload, model: payload["provenance"]["sources"][0].update(sha256="x"), "hexadecimal SHA-256"),
        (lambda payload, model: payload["provenance"]["sources"][0].update(max_rows=0), "positive 'max_rows'"),
        (
            lambda payload, model: payload["provenance"]["sources"][0].update(sampled_row_count=99999),
            "cannot keep more rows",
        ),
        (
            lambda payload, model: payload["provenance"]["sources"].append(dict(payload["provenance"]["sources"][0])),
            "more than once",
        ),
        (lambda payload, model: payload["provenance"]["aggregate"].pop("moment_min"), "aggregate keys"),
        (
            lambda payload, model: payload["provenance"]["aggregate"].update(module_count=-1),
            "non-negative integer",
        ),
        (lambda payload, model: payload["diagonal_hessian"].pop(_expected_target_fqns()[0]), "cover exactly"),
        (lambda payload, model: payload["diagonal_hessian"].update(other=[1.0]), "cover exactly"),
        (
            lambda payload, model: payload["diagonal_hessian"].update(
                {"transformer_encoder._orig_mod.layers.0.attn.w_qkv": [1.0]}
            ),
            "non-canonical",
        ),
        (
            lambda payload, model: payload["diagonal_hessian"].update({_expected_target_fqns()[0]: [1.0, 2.0]}),
            "input channel",
        ),
        (
            lambda payload, model: payload["diagonal_hessian"].update({_expected_target_fqns()[0]: "1.0"}),
            "non-empty list",
        ),
        (
            lambda payload, model: payload["diagonal_hessian"][_expected_target_fqns()[0]].__setitem__(0, -1.0),
            "finite and non-negative",
        ),
        (
            lambda payload, model: payload["diagonal_hessian"][_expected_target_fqns()[0]].__setitem__(0, True),
            "must be a number",
        ),
        (
            lambda payload, model: payload["diagonal_hessian"].update(
                {_expected_target_fqns()[0]: [0.0] * model.get_submodule(_expected_target_fqns()[0]).in_features}
            ),
            "identically zero",
        ),
        (
            # Finite as a JSON double, but +inf as the FP32 vector the search actually weights with: it must fail
            # here, before anything is converted, and not when the packer first materializes it.
            lambda payload, model: payload["diagonal_hessian"][_expected_target_fqns()[0]].__setitem__(0, 1e308),
            "FP32 vector",
        ),
        (lambda payload, model: payload["weight_sha256"].pop(_expected_target_fqns()[0]), "weight_sha256"),
        (lambda payload, model: payload["weight_sha256"].update({_expected_target_fqns()[0]: "z" * 64}), "z" * 8),
        (
            lambda payload, model: payload["weight_sha256"].update({_expected_target_fqns()[0]: "a" * 64}),
            "built on different weights",
        ),
        # Structurally valid edits that the component digests are there to catch.
        (
            lambda payload, model: payload["diagonal_hessian"][_expected_target_fqns()[0]].__setitem__(0, 9.0),
            "hashes to",
        ),
        (
            lambda payload, model: payload["provenance"]["sources"][0]["metadata"].update(manifest="other.json"),
            "hashes to",
        ),
        (lambda payload, model: payload["provenance"]["aggregate"].update(source_count=7), "hashes to"),
        (lambda payload, model: payload.update(moment_sha256="0" * 64), "hashes to"),
        (lambda payload, model: payload.update(provenance_sha256="0" * 64), "hashes to"),
        (lambda payload, model: payload.update(moment_sha256="not-a-digest"), "hexadecimal SHA-256"),
        (lambda payload, model: payload.pop("provenance_sha256"), "top-level keys"),
    ],
)
def test_hessian_artifact_rejection_matrix(tmp_path, mutate, message):
    model = _FakeSortformer()
    payload = _hessian_payload(model)
    mutate(payload, model)
    path = _write_hessian(tmp_path, payload)
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    with pytest.raises(ValueError, match=message):
        sq.load_diagonal_hessian(str(path), model, selection)


@pytest.mark.unit
@pytest.mark.parametrize(
    "fabricate, message",
    [
        (lambda aggregate: aggregate.update(module_count=aggregate["module_count"] + 1), "aggregate.module_count"),
        (lambda aggregate: aggregate.update(module_count=0), "aggregate.module_count"),
        (lambda aggregate: aggregate.update(source_count=2), "aggregate.source_count"),
        (lambda aggregate: aggregate.update(source_labels=["far_field"]), "aggregate.source_labels"),
        (lambda aggregate: aggregate.update(source_labels=[]), "aggregate.source_labels"),
        (lambda aggregate: aggregate.update(source_labels=["near_field", "near_field"]), "aggregate.source_labels"),
        (lambda aggregate: aggregate.update(moment_count=aggregate["moment_count"] - 1), "aggregate.moment_count"),
        (lambda aggregate: aggregate.update(moment_count=0), "aggregate.moment_count"),
        (lambda aggregate: aggregate.update(moment_min=0.0), "aggregate.moment_min"),
        (lambda aggregate: aggregate.update(moment_min=aggregate["moment_max"]), "aggregate.moment_min"),
        (lambda aggregate: aggregate.update(moment_max=99.0), "aggregate.moment_max"),
        (lambda aggregate: aggregate.update(moment_max=aggregate["moment_min"]), "aggregate.moment_max"),
    ],
)
def test_hessian_aggregate_must_summarize_the_artifact_it_ships_with(tmp_path, fabricate, message):
    """A recomputed provenance digest buys no aggregate that contradicts the sections it claims to summarize.

    Every one of these edits is re-hashed exactly as the builder would have, so the section digest matches and the
    artifact is structurally valid; only recomputing the aggregate from the validated target, source and moment
    sections can catch them.
    """
    model = _FakeSortformer()
    payload = _hessian_payload(model)
    fabricate(payload["provenance"]["aggregate"])
    payload["provenance_sha256"] = sq.nvfp4_section_digest(payload["provenance"])
    path = _write_hessian(tmp_path, payload)
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    with pytest.raises(ValueError, match=message):
        sq.load_diagonal_hessian(str(path), model, selection)


@pytest.mark.unit
def test_hessian_aggregate_accepts_only_the_summary_of_its_own_sections(tmp_path):
    """The consistency check is exact, and an honest aggregate of an honest artifact passes it unchanged."""
    model = _FakeSortformer()
    payload = _hessian_payload(model)
    aggregate = payload["provenance"]["aggregate"]
    moments = payload["diagonal_hessian"]
    values = [value for fqn in sorted(moments) for value in moments[fqn]]
    assert aggregate["module_count"] == len(moments) == len(_expected_target_fqns())
    assert aggregate["source_count"] == len(payload["provenance"]["sources"])
    assert aggregate["source_labels"] == [source["label"] for source in payload["provenance"]["sources"]]
    assert (aggregate["moment_count"], aggregate["moment_min"], aggregate["moment_max"]) == (
        len(values),
        min(values),
        max(values),
    )

    loaded = sq.load_diagonal_hessian(
        str(_write_hessian(tmp_path, payload)), model, sq.select_quantization_targets(model, "nvfp4_all")
    )
    assert loaded["provenance"]["aggregate"] == aggregate


@pytest.mark.unit
def test_hessian_artifact_rejects_duplicate_keys_and_malformed_json(tmp_path):
    model = _FakeSortformer()
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    payload = _hessian_payload(model)

    duplicated = json.dumps(payload)
    duplicated = "{" + f'"version": {sq.HESSIAN_SCHEMA_VERSION}, ' + duplicated[1:]
    (tmp_path / "dup.json").write_text(duplicated, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        sq.load_diagonal_hessian(str(tmp_path / "dup.json"), model, selection)

    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        sq.load_diagonal_hessian(str(tmp_path / "bad.json"), model, selection)

    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        sq.load_diagonal_hessian(str(tmp_path / "list.json"), model, selection)


@pytest.mark.unit
def test_prediction_cache_identity_separates_local_hessian_runs(tmp_path):
    """Every distinct artifact is a distinct run, and the amax and mse identities are untouched by all of this."""
    model = _FakeSortformer()
    first = _write_hessian(tmp_path, _hessian_payload(model), name="first.json")
    modules = dict(model.named_modules())
    other = {
        fqn: _hessian_moments(modules[fqn].in_features, index + 3) for index, fqn in enumerate(_expected_target_fqns())
    }
    second = _write_hessian(tmp_path, _hessian_payload(model, moments=other), name="second.json")

    amax = sq.prediction_cache_identity(sq.SortformerQuantizationConfig(recipe="nvfp4_all"))
    mse = sq.prediction_cache_identity(_mse_config())
    hessian = sq.prediction_cache_identity(_hessian_config(first))
    changed = sq.prediction_cache_identity(_hessian_config(second))

    assert hessian["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN
    assert hessian["weight_scale_hessian_algorithm"] == sq.WEIGHT_SCALE_HESSIAN_ALGORITHM
    assert hessian["weight_scale_hessian_algorithm_version"] == sq.WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION
    assert hessian["weight_scale_hessian_damping"] == sq.WEIGHT_SCALE_HESSIAN_DAMPING
    assert hessian["weight_scale_hessian_path"] == str(first)
    assert hessian["weight_scale_hessian_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert hessian != changed
    assert hessian["weight_scale_hessian_sha256"] != changed["weight_scale_hessian_sha256"]
    # A searched identity never collides with an amax or an mse one, and it carries the amax fields unchanged.
    assert hessian not in (amax, mse)
    assert {key: hessian[key] for key in amax} == amax
    assert "weight_scale_mse_algorithm" not in hessian
    assert "weight_scale_hessian_algorithm" not in mse
    assert "weight_scale_hessian_algorithm" not in amax


@pytest.mark.unit
def test_evaluator_exposes_the_hessian_artifact_option():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert "quantization_weight_scale_hessian_path: Optional[str] = None" in source
    assert "amax, mse, local_hessian" in source


@pytest.mark.unit
def test_integrated_local_hessian_repack_runs_the_real_w4a4_production_path(tmp_path):
    """Pinned runtime: static W4A4 over every target, repacked by the real activation-weighted search."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")
    if not torch.cuda.is_available():
        pytest.skip("the integrated NVFP4 repack runs on CUDA only")
    pytest.importorskip("torchao.prototype.mx_formats", reason="the integrated repack requires torchao 0.17")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in sq.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(f"NVFP4 quantization is not accepted on compute capability {capability}")

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _FakeSortformer().to(device=device, dtype=torch.bfloat16).eval()
    expected = _expected_target_fqns()
    path = _write_hessian(tmp_path, _hessian_payload(model))
    calibration = _write_calibration(tmp_path, {fqn: 2.0 for fqn in expected})
    config = _hessian_config(path, scale_mode="static", calibration_path=str(calibration))
    try:
        sq.check_nvfp4_capability(config, sq.collect_capability_facts(device))
    except RuntimeError as error:
        pytest.skip(f"the real W4A4 path is unavailable here: {error}")

    summary = sq.quantize_sortformer_model(model, config)

    report = summary["weight_scale_hessian"]
    assert report["enabled"] is True
    assert report["target_fqns"] == expected
    for layer in report["layers"]:
        assert layer["searched_objective"] <= _tolerated_mse(layer["template_objective"])
    assert report["aggregate"]["searched_objective"] <= _tolerated_mse(report["aggregate"]["template_objective"])
    assert json.loads(json.dumps(summary)) == summary

    with torch.no_grad():
        output = model(torch.randn(2, 8, D_MODEL, device=device, dtype=torch.bfloat16))
    assert torch.isfinite(output.float()).all()


@pytest.mark.unit
def test_integrated_mse_repack_never_worsens_real_nvfp4_weight_error():
    """Pinned runtime: the integrated repack improves every real NVFP4 weight and leaves the model runnable."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")
    if not torch.cuda.is_available():
        pytest.skip("the integrated NVFP4 repack runs on CUDA only")
    pytest.importorskip("torchao.prototype.mx_formats", reason="the integrated repack requires torchao 0.17")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in sq.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(f"NVFP4 quantization is not accepted on compute capability {capability}")

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _FakeSortformer().to(device=device, dtype=torch.bfloat16).eval()
    expected = _expected_target_fqns()
    shapes = {fqn: tuple(model.get_submodule(fqn).weight.shape) for fqn in expected}

    summary = sq.quantize_sortformer_model(model, _mse_config(recipe="nvfp4_weight_only"))

    report = summary["weight_scale_mse"]
    assert report["enabled"] is True
    assert [layer["fqn"] for layer in report["layers"]] == expected
    for layer in report["layers"]:
        assert layer["shape"] == list(shapes[layer["fqn"]])
        assert layer["searched_mse"] <= _tolerated_mse(layer["template_mse"])
    # The search is not merely tolerated: on real weights it strictly beats the amax rule.
    assert all(layer["searched_mse"] < layer["template_mse"] for layer in report["layers"])
    assert report["aggregate"]["searched_mse"] < report["aggregate"]["template_mse"]
    assert json.loads(json.dumps(summary)) == summary

    with torch.no_grad():
        output = model(torch.randn(2, 8, D_MODEL, device=device, dtype=torch.bfloat16))
    assert torch.isfinite(output.float()).all()


# Deterministic stand-in for the Four-Over-Six reconstruction. It is *coarser* than the grid modelling the
# ordinary amax conversion on purpose: this method is not exhaustive over the encodings the amax rule rounds into
# and it renormalizes the weight global scale, so a worse reconstruction is a legitimate outcome the runtime must
# report rather than reject.
FOUR_OVER_SIX_GRID_STEP = 0.08


def _four_over_six_config(**overrides):
    """Quantization config with the ModelOpt Four-Over-Six weight repack on, overridable per test."""
    values = dict(recipe="nvfp4_all", weight_scale_method=sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX)
    values.update(overrides)
    return sq.SortformerQuantizationConfig(**values)


def _four_over_six_counts(weight):
    """Deterministic ``(blocks, M=6, M=4)`` split of one weight; every block is counted exactly once."""
    block_count = weight.numel() // sq.NVFP4_BLOCK_SIZE
    m4_block_count = block_count // 4
    return block_count, block_count - m4_block_count, m4_block_count


def _four_over_six_result(original, **overrides):
    """A ``FourOverSixRepack`` for ``original``, with any field replaced to model a malformed repacker result."""
    block_count, m6_block_count, m4_block_count = _four_over_six_counts(original)
    values = dict(
        weight=_fake_nvfp4(_round_to(original, FOUR_OVER_SIX_GRID_STEP)),
        block_count=block_count,
        m6_block_count=m6_block_count,
        m4_block_count=m4_block_count,
    )
    values.update(overrides)
    return sq.FourOverSixRepack(**values)


@pytest.fixture
def fake_four_over_six_repack(fake_torchao, monkeypatch):
    """
    Replace the accepted Four-Over-Six repacker with a deterministic stand-in and record every call.

    Each record keeps the number of ``quantize_`` calls made so far -- which pins the repack to its own
    conversion -- and whether any *earlier* original weight was still alive when this call started, which is how
    the one-clone-at-a-time memory contract is checked, exactly as for the two exhaustive searches.
    """
    calls = []
    originals = []

    def repack(weight, template):
        calls.append(
            SimpleNamespace(
                weight=weight.detach().clone(),
                template=template,
                quantize_calls=len(fake_torchao.calls),
                earlier_originals_alive=[reference() is not None for reference in originals],
            )
        )
        originals.append(weakref.ref(weight))
        return _four_over_six_result(weight)

    monkeypatch.setattr(sq, "repack_nvfp4_weight_four_over_six", repack)
    return calls


def _forbid_other_repackers(monkeypatch):
    """Make every repacker except the Four-Over-Six one fatal, so the dispatch cannot silently pick another."""

    def forbidden(*args, **kwargs):
        raise AssertionError("only the Four-Over-Six repacker may run under weight_scale_method='four_over_six'")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_mse", forbidden)
    monkeypatch.setattr(sq, "repack_nvfp4_weight_local_hessian", forbidden)


@pytest.mark.unit
def test_four_over_six_is_off_by_default_and_maps_from_the_evaluator():
    """The method is opt-in end to end, and the evaluator field is wired to the core option exactly."""
    default = sq.SortformerQuantizationConfig()
    assert default.weight_scale_method == sq.WEIGHT_SCALE_METHOD_AMAX
    assert default.uses_four_over_six_weight_scales is False
    assert default.uses_searched_weight_scales is False

    config = sq.quantization_config_from_eval_cfg(
        _eval_cfg(quantization_recipe="nvfp4_all", quantization_weight_scale_method="four_over_six")
    )
    assert config.weight_scale_method == "four_over_six"
    assert config.uses_four_over_six_weight_scales is True
    assert config.uses_searched_weight_scales is True
    assert config.uses_mse_weight_scales is False
    assert config.uses_local_hessian_weight_scales is False

    # The identity constants are the reference's, not knobs: the arithmetic is defined by exactly these.
    assert sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX == "four_over_six"
    assert sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX in sq.WEIGHT_SCALE_METHODS
    assert sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM == "nvfp4_block_e4m3_modelopt_four_over_six"
    assert sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION == 1
    assert sq.WEIGHT_SCALE_FOUR_OVER_SIX_FP8_MAX == 256.0
    assert sq.WEIGHT_SCALE_FOUR_OVER_SIX_MAGNITUDES == (6, 4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recipe": "disabled"}, "quantizes nothing"),
        ({"fold_global_scales": True, "scale_mode": "static"}, "mutually exclusive"),
        ({"weight_scale_hessian_path": "hessian.json"}, "is only used with"),
    ],
)
def test_four_over_six_rejects_incompatible_options(tmp_path, overrides, message):
    values = dict(overrides)
    if values.get("scale_mode") == "static":
        values["calibration_path"] = str(_write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()}))
    with pytest.raises(ValueError, match=message):
        _four_over_six_config(**values).validate()


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recipe": "disabled"}, "quantizes nothing"),
        ({"fold_global_scales": True, "scale_mode": "static"}, "mutually exclusive"),
        ({"weight_scale_hessian_path": "hessian.json"}, "is only used with"),
    ],
)
def test_four_over_six_is_rejected_before_the_model_is_mutated(tmp_path, fake_torchao, overrides, message):
    """An incompatible request must fail on a still-unquantized model, never halfway through the conversion."""
    model = _FakeSortformer()
    values = dict(overrides)
    if values.get("scale_mode") == "static":
        values["calibration_path"] = str(_write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()}))

    with pytest.raises(ValueError, match=message):
        sq.quantize_sortformer_model(model, _four_over_six_config(**values), facts=_facts())

    assert fake_torchao.calls == []
    assert all(isinstance(model.get_submodule(fqn), torch.nn.Linear) for fqn in _expected_target_fqns())


@pytest.mark.unit
def test_four_over_six_dispatch_names_its_repacker_explicitly():
    """A plan names its method: 'hessian is None' must never be read as 'this run is the MSE repack'."""
    plan = sq._weight_repack_plan(_four_over_six_config(), None)
    assert plan == sq._WeightRepackPlan(method=sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX)
    assert plan.hessian is None
    # The other two methods resolve exactly as before, and amax still resolves to the batched path.
    assert sq._weight_repack_plan(_mse_config(), None) == sq._WeightRepackPlan(method=sq.WEIGHT_SCALE_METHOD_MSE)
    assert sq._weight_repack_plan(sq.SortformerQuantizationConfig(recipe="nvfp4_all"), None) is None

    model = _FakeSortformer()
    fqn = "transformer_encoder.layers.0.attn.w_qkv"
    original = model.get_submodule(fqn).weight.detach().clone()
    with pytest.raises(RuntimeError, match="No NVFP4 weight repacker is implemented"):
        sq._repack_one_weight(model, fqn, original, "dynamic NVFP4 W4A4", sq._WeightRepackPlan(method="argmin"))


@pytest.mark.unit
def test_dynamic_four_over_six_converts_and_repacks_each_selected_fqn_once_in_order(
    fake_torchao, fake_four_over_six_repack, monkeypatch
):
    _forbid_other_repackers(monkeypatch)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)

    summary = sq.quantize_sortformer_model(model, _four_over_six_config(), facts=_facts())

    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert [call.config.step for call in fake_torchao.calls] == [None] * len(expected)
    assert len(fake_four_over_six_repack) == len(expected)
    for index, fqn in enumerate(expected):
        record = fake_four_over_six_repack[index]
        assert torch.equal(record.weight, originals[fqn])
        # Each repack ran after its own conversion and before the next one was requested.
        assert record.quantize_calls == index + 1
        # No earlier original weight was still alive: at most one high-precision clone exists at a time.
        assert record.earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_four_over_six"]["layers"]] == expected


@pytest.mark.unit
def test_static_four_over_six_assigns_every_amax_before_the_per_fqn_converts(
    tmp_path, fake_torchao, fake_four_over_six_repack, monkeypatch
):
    _forbid_other_repackers(monkeypatch)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)
    path = _write_calibration(tmp_path, {fqn: 1.5 for fqn in expected})
    config = _four_over_six_config(scale_mode="static", calibration_path=str(path), scale_margin=2.0)

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    prepare, *converts = fake_torchao.calls
    assert prepare.config.step == "prepare"
    assert prepare.selected == expected
    assert [call.config.step for call in converts] == ["convert"] * len(expected)
    assert [call.selected for call in converts] == [[fqn] for fqn in expected]
    # Every calibrated amax was written before the first per-FQN convert, so no module converted without it.
    assert [call.observed_amax for call in converts] == [{fqn: 3.0} for fqn in expected]
    for index, fqn in enumerate(expected):
        assert torch.equal(fake_four_over_six_repack[index].weight, originals[fqn])
        assert fake_four_over_six_repack[index].quantize_calls == index + 2
        assert fake_four_over_six_repack[index].earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_four_over_six"]["layers"]] == expected


@pytest.mark.unit
def test_weight_only_four_over_six_converts_and_repacks_each_selected_fqn_once_in_order(
    fake_torchao, fake_four_over_six_repack, monkeypatch
):
    _forbid_other_repackers(monkeypatch)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)

    summary = sq.quantize_sortformer_model(model, _four_over_six_config(recipe="nvfp4_weight_only"), facts=_facts())

    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert all(isinstance(call.config, _FakeNVFP4WeightOnlyConfig) for call in fake_torchao.calls)
    for index, fqn in enumerate(expected):
        assert torch.equal(fake_four_over_six_repack[index].weight, originals[fqn])
        assert fake_four_over_six_repack[index].quantize_calls == index + 1
        assert fake_four_over_six_repack[index].earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_four_over_six"]["layers"]] == expected


@pytest.mark.unit
def test_four_over_six_never_repacks_fp8_modules(fake_torchao, fake_four_over_six_repack):
    """The hybrid recipe repacks only its NVFP4 weights; the FP8 families keep their single batched call."""
    summary = sq.quantize_sortformer_model(
        _FakeSortformer(), _four_over_six_config(recipe="nvfp4_qkv_fp8_rest"), facts=_facts()
    )

    qkv = _fqns_for_suffix("attn.w_qkv")
    assert [call.selected for call in fake_torchao.calls[:-1]] == [[fqn] for fqn in qkv]
    assert isinstance(fake_torchao.calls[-1].config, _FakeFP8Config)
    assert fake_torchao.calls[-1].selected == sorted(
        _fqns_for_suffix("attn.out_proj") + _fqns_for_suffix("ffn.net.0") + _fqns_for_suffix("ffn.net.3")
    )
    assert len(fake_four_over_six_repack) == len(qkv)
    assert [layer["fqn"] for layer in summary["weight_scale_four_over_six"]["layers"]] == qkv
    assert summary["weight_scale_four_over_six"]["target_fqns"] == qkv


@pytest.mark.unit
def test_four_over_six_never_repacks_bf16_restored_targets(tmp_path, fake_torchao, fake_four_over_six_repack):
    restored = [f"transformer_encoder.layers.{index}.attn.w_qkv" for index in range(NUM_LAYERS)]
    path = _write_bf16_override(tmp_path, restored)
    config = _four_over_six_config(recipe="nvfp4_weight_only", bf16_override_path=str(path))

    model = _FakeSortformer()
    quantized = sorted(set(_expected_target_fqns()) - set(restored))
    originals = _original_weights(model, quantized)

    summary = sq.quantize_sortformer_model(model, config, facts=_facts())

    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in quantized]
    # Exactly the unrestored targets were repacked, and each with its own high-precision weight.
    assert len(fake_four_over_six_repack) == len(quantized)
    for record, fqn in zip(fake_four_over_six_repack, quantized):
        assert torch.equal(record.weight, originals[fqn])
    assert [layer["fqn"] for layer in summary["weight_scale_four_over_six"]["layers"]] == quantized
    assert summary["weight_scale_four_over_six"]["target_count"] == len(quantized)
    assert summary["skipped_fqns"] == restored


@pytest.mark.unit
def test_producer_fusion_runs_after_every_four_over_six_repack(
    tmp_path, fake_torchao, fake_four_over_six_repack, monkeypatch
):
    path = _write_calibration(tmp_path, {fqn: 1.0 for fqn in _expected_target_fqns()})
    config = _four_over_six_config(scale_mode="static", calibration_path=str(path), fuse_producer_packing=True)
    recorded = {}

    def fake_fuse(model, fqns):
        recorded["repacks"] = len(fake_four_over_six_repack)
        recorded["fqns"] = list(fqns)
        return {
            "enabled": True,
            "fused_block_count": len(producer_fusion.group_producer_fusion_blocks(fqns)),
            "fused_block_fqns": list(producer_fusion.group_producer_fusion_blocks(fqns)),
            "fused_boundaries": list(producer_fusion.FUSED_PRODUCER_BOUNDARIES),
            "notes": [],
        }

    monkeypatch.setattr(sq, "apply_producer_fusion", fake_fuse)
    summary = sq.quantize_sortformer_model(_FakeSortformer(), config, facts=_facts())

    assert recorded["fqns"] == _expected_target_fqns()
    assert recorded["repacks"] == len(_expected_target_fqns())
    assert summary["producer_fusion"]["enabled"] is True
    assert summary["weight_scale_four_over_six"]["target_count"] == len(_expected_target_fqns())


@pytest.mark.unit
def test_four_over_six_replacement_preserves_parameter_semantics(fake_torchao, fake_four_over_six_repack):
    model = _FakeSortformer()

    sq.quantize_sortformer_model(model, _four_over_six_config(), facts=_facts())

    for fqn in _expected_target_fqns():
        module = model.get_submodule(fqn)
        weight = module.weight
        assert isinstance(weight, torch.nn.Parameter)
        assert isinstance(weight, _FakeNVFP4Tensor)
        assert weight.requires_grad is False
        assert "weight" in dict(module.named_parameters())


@pytest.mark.unit
@pytest.mark.parametrize(
    "produce, message",
    [
        (lambda weight: _fake_nvfp4(_round_to(weight, FOUR_OVER_SIX_GRID_STEP)), "instead of a FourOverSixRepack"),
        (lambda weight: None, "instead of a FourOverSixRepack"),
        (lambda weight: _four_over_six_result(weight, weight=None), "produced no weight"),
        (
            lambda weight: _four_over_six_result(weight, weight=torch.nn.Parameter(weight.detach().clone())),
            "produced an ordinary Parameter",
        ),
        (lambda weight: _four_over_six_result(weight, m4_block_count=0), "does not describe the weight it repacked"),
        (lambda weight: _four_over_six_result(weight, block_count=0), "does not describe the weight it repacked"),
        (
            lambda weight: _four_over_six_result(
                weight,
                m6_block_count=weight.numel() // sq.NVFP4_BLOCK_SIZE + 1,
                m4_block_count=-1,
            ),
            "does not describe the weight it repacked",
        ),
        (
            lambda weight: _four_over_six_result(weight, weight=_fake_nvfp4(torch.full(weight.shape, float("nan")))),
            "non-finite",
        ),
        (
            lambda weight: _four_over_six_result(
                weight, weight=_fake_nvfp4(_round_to(weight, FOUR_OVER_SIX_GRID_STEP)[:, :16])
            ),
            "dequantizes to shape",
        ),
    ],
)
def test_four_over_six_fails_closed_on_an_unusable_result(fake_torchao, monkeypatch, produce, message):
    """No fallback: an unusable repack raises instead of leaving the amax-derived weight silently in place."""
    monkeypatch.setattr(sq, "repack_nvfp4_weight_four_over_six", lambda weight, template: produce(weight))

    with pytest.raises(RuntimeError, match=message):
        sq.quantize_sortformer_model(_FakeSortformer(), _four_over_six_config(), facts=_facts())


@pytest.mark.unit
def test_four_over_six_repack_rejects_a_weight_the_conversion_did_not_replace():
    """A module the conversion left in high precision must fail closed, not be repacked as if it were NVFP4."""
    model = _FakeSortformer()
    fqn = "transformer_encoder.layers.0.attn.w_qkv"
    original = model.get_submodule(fqn).weight.detach().clone()

    with pytest.raises(RuntimeError, match="instead of a TorchAO NVFP4 tensor"):
        sq._repack_weight_with_four_over_six(model, fqn, original, "dynamic NVFP4 W4A4")


@pytest.mark.unit
def test_four_over_six_report_is_deterministic_json_safe_and_weight_count_weighted(
    fake_torchao, fake_four_over_six_repack
):
    torch.manual_seed(4646)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)

    summary = sq.quantize_sortformer_model(model, _four_over_six_config(), facts=_facts())

    report = summary["weight_scale_four_over_six"]
    assert report["enabled"] is True
    assert report["method"] == sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX
    assert report["algorithm"] == sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM
    assert report["algorithm_version"] == sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION
    assert report["fp8_max_for_normalization"] == 256.0
    assert report["candidate_magnitudes"] == [6, 4]
    assert report["target_count"] == len(expected)
    assert report["target_fqns"] == expected
    assert report["total_weight_count"] == sum(originals[fqn].numel() for fqn in expected)
    assert [layer["fqn"] for layer in report["layers"]] == expected

    for layer in report["layers"]:
        original = originals[layer["fqn"]]
        block_count, m6_block_count, m4_block_count = _four_over_six_counts(original)
        assert layer["shape"] == list(original.shape)
        assert layer["weight_count"] == original.numel()
        # Every 16-weight block is accounted for under exactly one of the two representations.
        assert layer["block_count"] == block_count == original.numel() // sq.NVFP4_BLOCK_SIZE
        assert (layer["m6_block_count"], layer["m4_block_count"]) == (m6_block_count, m4_block_count)
        assert layer["m6_block_count"] + layer["m4_block_count"] == layer["block_count"]
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] > 0.0
        assert layer["ratio"] == pytest.approx(layer["searched_mse"] / layer["template_mse"])
        assert layer["relative_reduction"] == pytest.approx(1.0 - layer["ratio"])

    # The totals and the aggregate's own copies are the exact sums of the per-layer counts, never estimates.
    for field in ("block_count", "m6_block_count", "m4_block_count"):
        assert report[f"total_{field}"] == sum(layer[field] for layer in report["layers"])
        assert report["aggregate"][field] == report[f"total_{field}"]
    assert report["total_m6_block_count"] + report["total_m4_block_count"] == report["total_block_count"]

    total = report["total_weight_count"]
    for field in ("template_mse", "searched_mse"):
        assert report["aggregate"][field] == pytest.approx(
            sum(layer[field] * layer["weight_count"] for layer in report["layers"]) / total
        )
    assert report["aggregate"]["ratio"] == pytest.approx(
        report["aggregate"]["searched_mse"] / report["aggregate"]["template_mse"]
    )
    assert report["aggregate"]["relative_reduction"] == pytest.approx(1.0 - report["aggregate"]["ratio"])

    assert summary["weight_scale_method"] == "four_over_six"
    assert any(sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM in note for note in summary["notes"])
    assert any("448 / 256" in note for note in report["notes"])
    assert any("not exhaustive" in note for note in report["notes"])
    assert json.loads(json.dumps(summary)) == summary

    # The same weights produce the same evidence, run after run.
    torch.manual_seed(4646)
    repeated = sq.quantize_sortformer_model(_FakeSortformer(), _four_over_six_config(), facts=_facts())
    assert repeated["weight_scale_four_over_six"] == report


@pytest.mark.unit
def test_four_over_six_reports_a_worse_reconstruction_instead_of_rejecting_it(fake_torchao, fake_four_over_six_repack):
    """The MSE comparison is diagnostic here: unlike the two searches, this method may legitimately be worse."""
    model = _FakeSortformer()

    summary = sq.quantize_sortformer_model(model, _four_over_six_config(), facts=_facts())

    report = summary["weight_scale_four_over_six"]
    assert len(report["layers"]) == len(_expected_target_fqns())
    for layer in report["layers"]:
        assert layer["searched_mse"] > _tolerated_mse(layer["template_mse"])
        assert layer["ratio"] > 1.0
        assert layer["relative_reduction"] < 0.0
    assert report["aggregate"]["searched_mse"] > report["aggregate"]["template_mse"]
    # The worse repack really was installed; nothing fell back to the amax-derived weight.
    for fqn in _expected_target_fqns():
        assert isinstance(model.get_submodule(fqn).weight, _FakeNVFP4Tensor)


@pytest.mark.unit
def test_four_over_six_summary_is_present_only_for_its_own_runs(fake_torchao, fake_four_over_six_repack):
    """The section is added for this method only, so every other run keeps exactly the keys it had before."""
    summary = sq.quantize_sortformer_model(_FakeSortformer(), _four_over_six_config(), facts=_facts())

    # The two search sections are OFF and name the method that really selected the scales.
    assert summary["weight_scale_mse"] == sq.disabled_weight_scale_mse_summary(sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX)
    assert summary["weight_scale_hessian"] == sq.disabled_weight_scale_hessian_summary(
        sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX
    )
    assert summary["weight_scale_mse"]["method"] == sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX
    assert summary["weight_scale_hessian"]["method"] == sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX
    assert not any("amax-derived" in note for note in summary["weight_scale_mse"]["notes"])

    amax = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )
    assert "weight_scale_four_over_six" not in amax
    assert amax["weight_scale_mse"] == sq.disabled_weight_scale_mse_summary()


@pytest.mark.unit
def test_the_other_methods_never_call_the_four_over_six_repacker(
    tmp_path, fake_torchao, fake_mse_repack, fake_hessian_repack, monkeypatch
):
    """Regression: amax stays batched and the two searches keep their own repackers and their own sections."""

    def forbidden(weight, template):
        raise AssertionError("only weight_scale_method='four_over_six' may call the Four-Over-Six repacker")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_four_over_six", forbidden)
    expected = _expected_target_fqns()

    amax = sq.quantize_sortformer_model(
        _FakeSortformer(), sq.SortformerQuantizationConfig(recipe="nvfp4_all"), facts=_facts()
    )
    assert [call.selected for call in fake_torchao.calls] == [expected]
    assert fake_mse_repack == [] and fake_hessian_repack == []
    assert "weight_scale_four_over_six" not in amax

    fake_torchao.calls.clear()
    mse = sq.quantize_sortformer_model(_FakeSortformer(), _mse_config(), facts=_facts())
    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert len(fake_mse_repack) == len(expected)
    assert mse["weight_scale_mse"]["enabled"] is True
    assert "weight_scale_four_over_six" not in mse

    fake_torchao.calls.clear()
    model = _FakeSortformer()
    path = _write_hessian(tmp_path, _hessian_payload(model))
    hessian = sq.quantize_sortformer_model(model, _hessian_config(path), facts=_facts())
    assert [call.selected for call in fake_torchao.calls] == [[fqn] for fqn in expected]
    assert len(fake_hessian_repack) == len(expected)
    assert hessian["weight_scale_hessian"]["enabled"] is True
    assert "weight_scale_four_over_six" not in hessian


@pytest.mark.unit
def test_prediction_cache_identity_separates_four_over_six_runs(tmp_path):
    """Four-Over-Six stores different weights than every other method, and leaves their identities untouched."""
    model = _FakeSortformer()
    hessian_path = _write_hessian(tmp_path, _hessian_payload(model))

    amax = sq.prediction_cache_identity(sq.SortformerQuantizationConfig(recipe="nvfp4_all"))
    mse = sq.prediction_cache_identity(_mse_config())
    hessian = sq.prediction_cache_identity(_hessian_config(hessian_path))
    four_over_six = sq.prediction_cache_identity(_four_over_six_config())

    assert four_over_six["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX
    assert four_over_six["weight_scale_four_over_six_algorithm"] == sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM
    assert (
        four_over_six["weight_scale_four_over_six_algorithm_version"]
        == sq.WEIGHT_SCALE_FOUR_OVER_SIX_ALGORITHM_VERSION
    )
    # The two constants that *are* the arithmetic are part of the identity, not of the report only.
    assert four_over_six["weight_scale_four_over_six_fp8_max"] == 256.0
    assert four_over_six["weight_scale_four_over_six_magnitudes"] == [6, 4]
    assert four_over_six not in (amax, mse, hessian)

    # The default amax identity is byte-for-byte what it was before this method existed, and the other two
    # methods gained nothing from it either.
    assert {key: four_over_six[key] for key in amax} == amax
    assert "weight_scale_four_over_six_algorithm" not in amax
    assert "weight_scale_four_over_six_algorithm" not in mse
    assert "weight_scale_four_over_six_algorithm" not in hessian
    assert "weight_scale_mse_algorithm" not in four_over_six
    assert "weight_scale_hessian_algorithm" not in four_over_six
    assert sq.prediction_cache_identity(_four_over_six_config(recipe="disabled")) is None


@pytest.mark.unit
def test_evaluator_exposes_the_four_over_six_option():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert "amax, mse, local_hessian, four_over_six" in source
    assert "'weight_scale_four_over_six' summary section" in source


def _skip_without_the_real_nvfp4_runtime():
    """Skip unless this machine really can run the pinned NVFP4 conversion; returns the CUDA device."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")
    if not torch.cuda.is_available():
        pytest.skip("the integrated NVFP4 repack runs on CUDA only")
    pytest.importorskip("torchao.prototype.mx_formats", reason="the integrated repack requires torchao 0.17")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in sq.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(f"NVFP4 quantization is not accepted on compute capability {capability}")
    return torch.device("cuda")


def _assert_four_over_six_evidence(report, expected, shapes):
    """Shared checks over a real Four-Over-Six report: identity, coverage and the exact block-count invariants."""
    assert report["enabled"] is True
    assert report["method"] == sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX
    assert report["fp8_max_for_normalization"] == 256.0
    assert report["candidate_magnitudes"] == [6, 4]
    assert report["target_fqns"] == expected
    assert [layer["fqn"] for layer in report["layers"]] == expected
    for layer in report["layers"]:
        rows, columns = shapes[layer["fqn"]]
        assert layer["shape"] == [rows, columns]
        assert layer["block_count"] == rows * columns // sq.NVFP4_BLOCK_SIZE
        assert layer["m6_block_count"] + layer["m4_block_count"] == layer["block_count"]
        assert layer["m6_block_count"] >= 0 and layer["m4_block_count"] >= 0
        # Both MSEs are real measurements against the original weight; neither is used to reject the repack.
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] > 0.0
    assert report["total_block_count"] == sum(layer["block_count"] for layer in report["layers"])
    assert report["total_m6_block_count"] + report["total_m4_block_count"] == report["total_block_count"]
    # Real weights exercise both representations: a run that took only one of them did not compare them.
    assert report["total_m6_block_count"] > 0
    assert report["total_m4_block_count"] > 0


@pytest.mark.unit
def test_integrated_four_over_six_repack_runs_the_real_weight_only_path():
    """Pinned runtime: the real ModelOpt arithmetic repacks every real NVFP4 weight and stays runnable."""
    device = _skip_without_the_real_nvfp4_runtime()
    torch.manual_seed(0)
    model = _FakeSortformer().to(device=device, dtype=torch.bfloat16).eval()
    expected = _expected_target_fqns()
    shapes = {fqn: tuple(model.get_submodule(fqn).weight.shape) for fqn in expected}

    summary = sq.quantize_sortformer_model(model, _four_over_six_config(recipe="nvfp4_weight_only"))

    _assert_four_over_six_evidence(summary["weight_scale_four_over_six"], expected, shapes)
    assert json.loads(json.dumps(summary)) == summary

    with torch.no_grad():
        output = model(torch.randn(2, 8, D_MODEL, device=device, dtype=torch.bfloat16))
    assert torch.isfinite(output.float()).all()


@pytest.mark.unit
def test_integrated_four_over_six_repack_runs_the_real_static_w4a4_path(tmp_path):
    """Pinned runtime: static W4A4 over every target, repacked by the real Four-Over-Six arithmetic."""
    device = _skip_without_the_real_nvfp4_runtime()
    torch.manual_seed(0)
    model = _FakeSortformer().to(device=device, dtype=torch.bfloat16).eval()
    expected = _expected_target_fqns()
    shapes = {fqn: tuple(model.get_submodule(fqn).weight.shape) for fqn in expected}
    calibration = _write_calibration(tmp_path, {fqn: 2.0 for fqn in expected})
    config = _four_over_six_config(scale_mode="static", calibration_path=str(calibration))
    try:
        sq.check_nvfp4_capability(config, sq.collect_capability_facts(device))
    except RuntimeError as error:
        pytest.skip(f"the real W4A4 path is unavailable here: {error}")

    summary = sq.quantize_sortformer_model(model, config)

    _assert_four_over_six_evidence(summary["weight_scale_four_over_six"], expected, shapes)
    assert summary["backend"] == sq.BACKEND_MSLK_ACCELERATED
    assert json.loads(json.dumps(summary)) == summary

    with torch.no_grad():
        output = model(torch.randn(2, 8, D_MODEL, device=device, dtype=torch.bfloat16))
    assert torch.isfinite(output.float()).all()


AWQ_CHECKPOINT_SHA256 = "a" * 64


@pytest.mark.unit
def test_awq_clip_is_off_by_default_and_maps_from_the_evaluator(tmp_path):
    """The technique is opt-in end to end, and both evaluator fields are wired to the core options exactly."""
    default = sq.SortformerQuantizationConfig()
    assert default.weight_scale_method == sq.WEIGHT_SCALE_METHOD_AMAX
    assert default.weight_scale_awq_clip_path is None
    assert default.uses_awq_clip_weight_scales is False
    assert default.uses_searched_weight_scales is False

    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    config = sq.quantization_config_from_eval_cfg(
        _eval_cfg(
            quantization_recipe="nvfp4_all",
            quantization_scale_mode="static",
            quantization_calibration_path=str(calibration),
            quantization_weight_scale_method="awq_clip",
            quantization_weight_scale_awq_clip_path=str(path),
        )
    )
    assert config.weight_scale_method == "awq_clip"
    assert config.weight_scale_awq_clip_path == str(path)
    assert config.uses_awq_clip_weight_scales is True
    assert config.uses_searched_weight_scales is True
    assert config.uses_mse_weight_scales is False
    assert config.uses_local_hessian_weight_scales is False
    assert config.uses_four_over_six_weight_scales is False

    # The identity constants are the adapted reference's, not knobs: the arithmetic is defined by exactly these.
    assert sq.WEIGHT_SCALE_METHOD_AWQ_CLIP == "awq_clip"
    assert sq.WEIGHT_SCALE_METHOD_AWQ_CLIP in sq.WEIGHT_SCALE_METHODS
    assert sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM == "nvfp4_block_e4m3_modelopt_awq_clip"
    assert sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION == 1
    assert sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS == (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
    assert sq.WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT == 11
    assert sq.WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE == 16
    assert sq.WEIGHT_SCALE_AWQ_CLIP_SCALE_MARGIN == 1.0
    assert sq.WEIGHT_SCALE_AWQ_CLIP_RECIPES == ("nvfp4_all",)
    assert sq.WEIGHT_SCALE_AWQ_CLIP_UNCLIPPED_CODE == 10
    assert sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETICS == ("torchao_non_triton", "mslk_triton")
    assert sq.MODELOPT_REFERENCE_VERSION == "0.46.0"
    assert sq.MODELOPT_REFERENCE_WHEEL_SHA256 == ("1864b4e9921e287b065be3861ab48345144e673273ebb2b94bd9a6119a9eba8e")


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recipe": "nvfp4_weight_only", "scale_mode": "dynamic", "calibration_path": None}, "requires one of recipe"),
        ({"recipe": "nvfp4_qkv_only"}, "requires one of recipe"),
        ({"scale_mode": "dynamic", "calibration_path": None}, "requires scale_mode='static'"),
        ({"scale_margin": 1.375}, "requires quantization_scale_margin"),
        ({"weight_scale_awq_clip_path": None}, "requires quantization_weight_scale_awq_clip_path"),
        ({"weight_scale_awq_clip_path": "  "}, "requires quantization_weight_scale_awq_clip_path"),
        ({"weight_scale_awq_clip_path": "/nonexistent/awq.json"}, "is not a readable file"),
        ({"fold_global_scales": True}, "mutually exclusive"),
        ({"bf16_override_path": "override.json"}, "cannot be combined"),
        ({"weight_scale_hessian_path": "hessian.json"}, "is only used with"),
    ],
)
def test_awq_clip_rejects_incompatible_options(tmp_path, overrides, message):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    with pytest.raises(ValueError, match=message):
        _awq_config(path, calibration, **overrides).validate()


@pytest.mark.unit
def test_awq_clip_requires_a_readable_calibration_file(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    with pytest.raises(ValueError, match="readable static activation-calibration"):
        _awq_config(path, tmp_path / "missing.json").validate()


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    [
        sq.WEIGHT_SCALE_METHOD_AMAX,
        sq.WEIGHT_SCALE_METHOD_MSE,
        sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN,
        sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX,
    ],
)
def test_other_methods_reject_a_supplied_awq_clip_path(tmp_path, method):
    """An ignored path would let a run believe it executed the AWQ-clip codes while executing other scales."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    values = {"weight_scale_hessian_path": None}
    if method == sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN:
        values["weight_scale_hessian_path"] = str(_write_hessian(tmp_path, _hessian_payload(model)))
    with pytest.raises(ValueError, match="is only used with"):
        sq.SortformerQuantizationConfig(
            recipe="nvfp4_all",
            scale_mode="static",
            calibration_path=str(calibration),
            weight_scale_method=method,
            weight_scale_awq_clip_path=str(path),
            **values,
        ).validate()


@pytest.mark.unit
def test_awq_clip_loader_accepts_a_valid_artifact_and_decodes_its_codes(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    payload = _awq_payload(model, calibration)
    path = _write_awq(tmp_path, payload)
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    loaded = sq.load_awq_clip_artifact(str(path), model, selection, str(calibration))

    assert loaded["checkpoint_sha256"] == AWQ_CHECKPOINT_SHA256
    assert loaded["algorithm"] == sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM
    assert loaded["fqns"] == _expected_target_fqns()
    assert loaded["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded["ratio_code_sha256"] == payload["ratio_code_sha256"]
    assert loaded["provenance_sha256"] == payload["provenance_sha256"]
    assert loaded["calibration_path"] == str(calibration)
    assert loaded["calibration"]["sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert loaded["calibration"]["scale_margin"] == 1.0
    assert loaded["clip_ratios"] == list(sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS)
    assert loaded["template_arithmetic"] == sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED
    for fqn in loaded["fqns"]:
        module = model.get_submodule(fqn)
        assert loaded["code_shapes"][fqn] == [module.out_features, module.in_features // 16]
        assert isinstance(loaded["ratio_codes"][fqn], bytes)
        assert len(loaded["ratio_codes"][fqn]) == module.out_features * module.in_features // 16
        assert loaded["ratio_codes"][fqn] == base64.b64decode(payload["ratio_codes"][fqn]["codes"], validate=True)
        assert max(loaded["ratio_codes"][fqn]) <= 10
        assert loaded["ratio_histogram"][fqn] == payload["provenance"]["modules"][fqn]["ratio_histogram"]
    # The runtime artifact carries no activation row: only codes, digests and provenance.
    assert set(payload) == set(sq.AWQ_CLIP_ARTIFACT_KEYS)
    assert "activation_rows" not in json.dumps(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update(schema="other"), "declares schema"),
        (lambda payload: payload.update(version=2), "version 1 is required"),
        (lambda payload: payload.update(algorithm="other"), "was built for algorithm"),
        (lambda payload: payload.update(algorithm_version=2), "was built for algorithm"),
        (lambda payload: payload.update(weight_digest_method="md5"), "weight-digest method"),
        (lambda payload: payload.update(code_encoding="hex"), "code encoding"),
        (lambda payload: payload.update(checkpoint_sha256="nope"), "hexadecimal SHA-256"),
        (lambda payload: payload.pop("provenance"), "exactly the top-level keys"),
        (lambda payload: payload.update(extra=1), "exactly the top-level keys"),
        (lambda payload: payload["arithmetic"].update(block_size=32), "'block_size'"),
        (lambda payload: payload["arithmetic"].update(tie_rule="latest wins"), "'tie_rule'"),
        (lambda payload: payload["arithmetic"].update(objective="something else"), "'objective'"),
        (lambda payload: payload["arithmetic"].update(activation_qdq="bf16"), "'activation_qdq'"),
        (lambda payload: payload["arithmetic"].update(template_arithmetic="guessed"), "'template_arithmetic'"),
        (lambda payload: payload["arithmetic"].update(template_arithmetic=None), "'template_arithmetic'"),
        (lambda payload: payload["arithmetic"].pop("template_arithmetic"), "arithmetic keys"),
        (lambda payload: payload["arithmetic"].update(fp4_max=8.0), "'fp4_max'"),
        (lambda payload: payload["arithmetic"].update(scale_min=0.0), "'scale_min'"),
        (lambda payload: payload["arithmetic"].update(modelopt_reference_version="0.45.0"), "reference_version"),
        (lambda payload: payload["arithmetic"].update(modelopt_reference_wheel_sha256="b" * 64), "wheel_sha256"),
        (lambda payload: payload["arithmetic"].update(clip_ratios=[0.5] * 11), "Clip ratio"),
        (lambda payload: payload["arithmetic"].update(clip_ratios=[0.5]), "clip ratio"),
        (lambda payload: payload["arithmetic"].pop("tie_rule"), "arithmetic keys"),
        (lambda payload: payload["provenance"].update(method="other"), "records 'method'"),
        (lambda payload: payload["provenance"].update(targets=["attn.w_qkv"]), "declares targets"),
        (lambda payload: payload["provenance"].update(target_module_count=3), "records target_module_count"),
        (lambda payload: payload["provenance"].update(sources=[]), "non-empty 'sources' list"),
        (lambda payload: payload["provenance"]["aggregate"].update(module_count=99), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(ratio_histogram=[0] * 11), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(selected_objective=99.0), "does not summarize"),
        (lambda payload: payload["activation_calibration"].update(sha256="b" * 64), "another activation calibration"),
        (lambda payload: payload["activation_calibration"].update(size_bytes=1), "another activation calibration"),
        (lambda payload: payload["activation_calibration"].update(scale_margin=1.375), "activation scale margin"),
    ],
)
def test_awq_clip_loader_rejection_matrix(tmp_path, mutate, message):
    """A foreign, edited, stale or self-inconsistent artifact must fail on an unmodified model."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    payload = _awq_payload(model, calibration)
    mutate(payload)
    _reseal_awq(payload)
    path = _write_awq(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        sq.load_awq_clip_artifact(
            str(path), model, sq.select_quantization_targets(model, "nvfp4_all"), str(calibration)
        )


@pytest.mark.unit
def test_awq_clip_loader_rejects_broken_code_payloads(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    fqn = _expected_target_fqns()[0]

    def load(payload):
        _reseal_awq(payload)
        path = _write_awq(tmp_path, payload, name="broken.json")
        return sq.load_awq_clip_artifact(str(path), model, selection, str(calibration))

    truncated = _awq_payload(model, calibration)
    raw = base64.b64decode(truncated["ratio_codes"][fqn]["codes"], validate=True)
    truncated["ratio_codes"][fqn]["codes"] = base64.b64encode(raw[:-1]).decode("ascii")
    with pytest.raises(ValueError, match="code byte"):
        load(truncated)

    invalid = _awq_payload(model, calibration)
    invalid["ratio_codes"][fqn]["codes"] = "not base64!!"
    with pytest.raises(ValueError, match="valid base64"):
        load(invalid)

    out_of_range = _awq_payload(model, calibration)
    raw = bytearray(base64.b64decode(out_of_range["ratio_codes"][fqn]["codes"], validate=True))
    raw[0] = 11
    out_of_range["ratio_codes"][fqn]["codes"] = base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(ValueError, match="indexed by 0"):
        load(out_of_range)

    reshaped = _awq_payload(model, calibration)
    reshaped["ratio_codes"][fqn]["shape"] = [1, 1]
    with pytest.raises(ValueError, match="the live module needs"):
        load(reshaped)

    extra_key = _awq_payload(model, calibration)
    extra_key["ratio_codes"][fqn]["histogram"] = []
    with pytest.raises(ValueError, match="ratio-code keys"):
        load(extra_key)

    partial = _awq_payload(model, calibration)
    partial["ratio_codes"].pop(fqn)
    with pytest.raises(ValueError, match="must cover exactly"):
        load(partial)

    stale = _awq_payload(model, calibration)
    stale["weight_sha256"][fqn] = "b" * 64
    with pytest.raises(ValueError, match="built on different weights"):
        load(stale)

    inconsistent = _awq_payload(model, calibration)
    inconsistent["provenance"]["modules"][fqn]["ratio_histogram"] = [0] * 11
    with pytest.raises(ValueError, match="counts 0 block"):
        load(inconsistent)


@pytest.mark.unit
def test_awq_clip_loader_rejects_a_corrupted_section_digest_and_duplicate_keys(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    fqn = _expected_target_fqns()[0]

    edited = _awq_payload(model, calibration)
    raw = bytearray(base64.b64decode(edited["ratio_codes"][fqn]["codes"], validate=True))
    raw[0] = (raw[0] + 1) % 11
    edited["ratio_codes"][fqn]["codes"] = base64.b64encode(bytes(raw)).decode("ascii")
    # Deliberately *not* resealed: this is an artifact someone edited after the builder digested it.
    with pytest.raises(ValueError, match="ratio_code_sha256"):
        sq.load_awq_clip_artifact(
            str(_write_awq(tmp_path, edited, name="edited.json")), model, selection, str(calibration)
        )

    reprovenanced = _awq_payload(model, calibration)
    reprovenanced["provenance"]["sources"][0]["label"] = "somewhere_else"
    with pytest.raises(ValueError, match="provenance_sha256"):
        sq.load_awq_clip_artifact(
            str(_write_awq(tmp_path, reprovenanced, name="reprov.json")), model, selection, str(calibration)
        )

    duplicate = tmp_path / "dup.json"
    duplicate.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        sq.load_awq_clip_artifact(str(duplicate), model, selection, str(calibration))

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        sq.load_awq_clip_artifact(str(broken), model, selection, str(calibration))

    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        sq.load_awq_clip_artifact(str(listed), model, selection, str(calibration))


@pytest.mark.unit
def test_awq_clip_loader_binds_the_configured_calibration_file(tmp_path):
    """The artifact is bound to the bytes of the calibration whose values quantized the scored activations."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    other = _write_awq_calibration(tmp_path, name="other.json", amax=3.0)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    with pytest.raises(ValueError, match="another activation calibration"):
        sq.load_awq_clip_artifact(str(path), model, selection, str(other))
    with pytest.raises(ValueError, match="activation scale margin"):
        sq.load_awq_clip_artifact(str(path), model, selection, str(calibration), scale_margin=1.5)

    dynamic = _write_awq_calibration(tmp_path, name="dynamic.json", scale_mode="dynamic")
    dynamic_artifact = _write_awq(tmp_path, _awq_payload(model, dynamic), name="dynamic_awq.json")
    with pytest.raises(ValueError, match="declares scale_mode"):
        sq.load_awq_clip_artifact(str(dynamic_artifact), model, selection, str(dynamic))

    foreign = _write_awq_calibration(tmp_path, name="foreign.json", checkpoint="d" * 64)
    foreign_artifact = _write_awq(tmp_path, _awq_payload(model, foreign), name="foreign_awq.json")
    with pytest.raises(ValueError, match="bound to a calibration collected on checkpoint"):
        sq.load_awq_clip_artifact(str(foreign_artifact), model, selection, str(foreign))


@pytest.mark.unit
def test_awq_clip_resolves_the_checkpoint_from_the_production_calibration_shape(tmp_path):
    """The frozen production calibration has no top-level checkpoint at all; the merge records it in metadata.

    ``merge_calibrations`` writes exactly the shape asserted here -- ``activation_amax``, ``metadata``, ``recipe``,
    ``scale_mode``, ``targets``, ``version`` -- so an identity that only ever read a top-level claim could not bind
    a single production artifact to its checkpoint. The older single-collector spelling still resolves, and two
    agreeing claims resolve to the one digest they agree on.
    """
    production = _write_awq_calibration(tmp_path, name="production.json")
    payload = json.loads(production.read_text(encoding="utf-8"))
    assert sorted(payload) == ["activation_amax", "metadata", "recipe", "scale_mode", "targets", "version"]
    assert payload["metadata"]["checkpoint_sha256"] == AWQ_CHECKPOINT_SHA256

    identity = sq.nvfp4_awq_clip_calibration_identity(str(production))
    assert identity["checkpoint_sha256"] == AWQ_CHECKPOINT_SHA256
    assert identity["scale_mode"] == "static"
    assert identity["headroom"] == 1.375
    assert identity["headroom_baked_in"] is True
    assert identity["runtime_scale_margin"] == 1.0
    assert identity["merge_method"] == sq.CALIBRATION_MERGE_METHOD
    assert identity["merge_method_version"] == sq.CALIBRATION_MERGE_METHOD_VERSION
    assert identity["target_module_count"] == len(_expected_target_fqns())
    assert identity["sha256"] == hashlib.sha256(production.read_bytes()).hexdigest()
    assert identity["size_bytes"] == len(production.read_bytes())

    legacy = _write_awq_calibration(tmp_path, name="legacy.json", checkpoint_location="top_level")
    assert "checkpoint_sha256" not in json.loads(legacy.read_text(encoding="utf-8"))["metadata"]
    assert sq.nvfp4_awq_clip_calibration_identity(str(legacy))["checkpoint_sha256"] == AWQ_CHECKPOINT_SHA256

    both = _write_awq_calibration(tmp_path, name="both.json", checkpoint_location="both")
    assert sq.nvfp4_awq_clip_calibration_identity(str(both))["checkpoint_sha256"] == AWQ_CHECKPOINT_SHA256


@pytest.mark.unit
def test_awq_clip_refuses_a_conflicting_or_absent_calibration_checkpoint(tmp_path):
    """Two conflicting claims cannot both be right, and no claim at all cannot bind the codes to a checkpoint."""
    conflicting = _write_awq_calibration(
        tmp_path, name="conflict.json", checkpoint_location="both", top_level_checkpoint="d" * 64
    )
    with pytest.raises(ValueError, match="conflicting claims"):
        sq.nvfp4_awq_clip_calibration_identity(str(conflicting))

    unbound = _write_awq_calibration(tmp_path, name="unbound.json", checkpoint_location="none")
    with pytest.raises(ValueError, match="declares no checkpoint digest"):
        sq.nvfp4_awq_clip_calibration_identity(str(unbound))

    malformed = _write_awq_calibration(tmp_path, name="malformed.json", checkpoint="not-a-digest")
    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        sq.nvfp4_awq_clip_calibration_identity(str(malformed))


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"headroom_baked_in": False}, "headroom_baked_in"),
        ({"headroom_baked_in": None}, "headroom_baked_in"),
        ({"headroom_baked_in": 1}, "headroom_baked_in"),
        ({"headroom_baked_in": "true"}, "headroom_baked_in"),
        ({"headroom": None}, "'metadata.headroom'"),
        ({"headroom": "1.375"}, "'metadata.headroom'"),
        ({"headroom": float("inf")}, "'metadata.headroom'"),
        ({"headroom": 0.0}, "must be positive"),
        ({"headroom": -1.0}, "must be positive"),
        ({"runtime_scale_margin": None}, "runtime_scale_margin"),
        ({"runtime_scale_margin": 1.375}, "presumes a runtime scale margin"),
        ({"method_version": "1"}, "method_version"),
        ({"method": 1}, "'metadata.method'"),
    ],
)
def test_awq_clip_enforces_the_baked_headroom_contract(tmp_path, metadata, message):
    """A calibration whose headroom is not already baked in would be applied twice or not at all at margin 1.0."""
    path = _write_awq_calibration(tmp_path, name="headroom.json", metadata=metadata)
    with pytest.raises(ValueError, match=message):
        sq.nvfp4_awq_clip_calibration_identity(str(path))


@pytest.mark.unit
def test_awq_clip_runtime_rejects_a_calibration_without_baked_headroom(tmp_path):
    """The runtime loader fails closed on the same contract, before it converts anything."""
    model = _FakeSortformer()
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    calibration = _write_awq_calibration(tmp_path)
    payload = _awq_payload(model, calibration)

    # An artifact that recorded 'headroom_baked_in: false' is named for what it is rather than reported as a
    # mismatch against the file, and it is refused even though its own digests are intact.
    unbaked = json.loads(json.dumps(payload))
    unbaked["activation_calibration"]["headroom_baked_in"] = False
    with pytest.raises(ValueError, match="headroom_baked_in"):
        sq.load_awq_clip_artifact(
            str(_write_awq(tmp_path, _reseal_awq(unbaked), name="unbaked.json")),
            model,
            selection,
            str(calibration),
        )

    absent = json.loads(json.dumps(payload))
    absent["activation_calibration"]["headroom"] = None
    with pytest.raises(ValueError, match="activation_calibration.headroom"):
        sq.load_awq_clip_artifact(
            str(_write_awq(tmp_path, _reseal_awq(absent), name="absent.json")), model, selection, str(calibration)
        )

    # And a calibration file that stops claiming baked headroom is refused even when the artifact still claims it.
    rewritten = _write_awq_calibration(tmp_path, name="rewritten.json", metadata={"headroom_baked_in": False})
    with pytest.raises(ValueError, match="headroom_baked_in"):
        sq.load_awq_clip_artifact(
            str(_write_awq(tmp_path, payload, name="ok_awq.json")), model, selection, str(rewritten)
        )


@pytest.mark.unit
def test_awq_clip_validates_the_artifact_before_converting_anything(tmp_path, fake_torchao, monkeypatch):
    """A failed artifact check leaves every target module exactly as it was."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    payload = _awq_payload(model, calibration)
    payload["weight_sha256"][_expected_target_fqns()[0]] = "b" * 64
    _reseal_awq(payload)
    path = _write_awq(tmp_path, payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("no weight may be repacked when the artifact does not describe this model")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_awq_clip", forbidden)

    with pytest.raises(ValueError, match="built on different weights"):
        sq.quantize_sortformer_model(model, _awq_config(path, calibration), facts=_facts())

    assert fake_torchao.calls == []
    for fqn in _expected_target_fqns():
        assert isinstance(model.get_submodule(fqn), torch.nn.Linear)


@pytest.mark.unit
def test_awq_clip_refuses_a_backend_that_builds_the_other_ordinary_template(tmp_path, fake_torchao, monkeypatch):
    """The unclipped code keeps the ordinary conversion's own bytes, and the two backends do not write them alike.

    The artifact says which conversion it was scored against, so a run whose backend would construct the other one
    is refused before a single module is converted rather than silently deploying blocks nobody scored.
    """
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))

    def forbidden(*args, **kwargs):
        raise AssertionError("no weight may be repacked when the artifact describes the other backend's template")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_awq_clip", forbidden)

    assert sq.awq_clip_template_arithmetic_for_backend(sq.BACKEND_MSLK_ACCELERATED) == (
        sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED
    )
    assert sq.awq_clip_template_arithmetic_for_backend(sq.BACKEND_REFERENCE_UNACCELERATED) == (
        sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE
    )
    # The reference-kernel acknowledgement is the existing requirement of an unaccelerated run and is given here
    # explicitly, so this reaches the artifact/backend check instead of failing on that unrelated validation.
    unaccelerated = _awq_config(path, calibration, accelerated_packing=False, allow_reference_kernels=True)
    assert sq.check_nvfp4_capability(unaccelerated, _facts()) == sq.BACKEND_REFERENCE_UNACCELERATED
    with pytest.raises(ValueError, match="ordinary-template construction"):
        sq.quantize_sortformer_model(model, unaccelerated, facts=_facts())

    assert fake_torchao.calls == []
    for fqn in _expected_target_fqns():
        assert isinstance(model.get_submodule(fqn), torch.nn.Linear)

    # And the artifact built for the reference conversion is the one that run accepts.
    reference = _awq_payload(model, calibration)
    reference["arithmetic"]["template_arithmetic"] = sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE
    loaded = sq.load_awq_clip_artifact(
        str(_write_awq(tmp_path, reference, name="reference_awq.json")),
        model,
        sq.select_quantization_targets(model, "nvfp4_all"),
        str(calibration),
    )
    assert loaded["template_arithmetic"] == sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE
    with pytest.raises(ValueError, match="ordinary-template construction"):
        sq.require_awq_clip_template_arithmetic(loaded, sq.BACKEND_MSLK_ACCELERATED)


@pytest.mark.unit
def test_awq_clip_converts_and_repacks_each_fqn_once_with_its_own_codes(
    tmp_path, fake_torchao, fake_awq_clip_repack, monkeypatch
):
    _forbid_non_awq_repackers(monkeypatch)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    originals = _original_weights(model, expected)
    calibration = _write_awq_calibration(tmp_path)
    payload = _awq_payload(model, calibration)
    path = _write_awq(tmp_path, payload)

    summary = sq.quantize_sortformer_model(model, _awq_config(path, calibration), facts=_facts())

    # One prepare call for the whole set, then exactly one convert call per FQN in sorted order.
    assert [call.selected for call in fake_torchao.calls[1:]] == [[fqn] for fqn in expected]
    assert len(fake_awq_clip_repack) == len(expected)
    for index, fqn in enumerate(expected):
        record = fake_awq_clip_repack[index]
        assert torch.equal(record.weight, originals[fqn])
        assert record.codes.dtype == torch.uint8
        assert record.codes.device == originals[fqn].device
        assert bytes(record.codes.reshape(-1).tolist()) == base64.b64decode(
            payload["ratio_codes"][fqn]["codes"], validate=True
        )
        assert list(record.codes.shape) == payload["ratio_codes"][fqn]["shape"]
        # Each repack ran after its own conversion, and no earlier original weight was still alive.
        assert record.quantize_calls == index + 2
        assert record.earlier_originals_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_awq_clip"]["layers"]] == expected
    # The other methods' summary sections name the method that actually selected the scales.
    assert summary["weight_scale_mse"] == sq.disabled_weight_scale_mse_summary(sq.WEIGHT_SCALE_METHOD_AWQ_CLIP)
    assert summary["weight_scale_hessian"] == sq.disabled_weight_scale_hessian_summary(sq.WEIGHT_SCALE_METHOD_AWQ_CLIP)
    assert "weight_scale_four_over_six" not in summary


@pytest.mark.unit
def test_awq_clip_summary_records_the_artifact_identity_and_the_ratio_evidence(
    tmp_path, fake_torchao, fake_awq_clip_repack
):
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    # Read off the live model *before* conversion: the reported shape is the shape of the weight the codes were
    # selected on, and conversion may leave a module that no longer advertises the original Linear's dimensions.
    shapes = {
        fqn: [int(model.get_submodule(fqn).out_features), int(model.get_submodule(fqn).in_features)]
        for fqn in expected
    }
    calibration = _write_awq_calibration(tmp_path)
    payload = _awq_payload(model, calibration)
    path = _write_awq(tmp_path, payload)

    summary = sq.quantize_sortformer_model(model, _awq_config(path, calibration), facts=_facts())
    report = summary["weight_scale_awq_clip"]

    assert summary["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_AWQ_CLIP
    assert report["enabled"] is True
    assert report["method"] == sq.WEIGHT_SCALE_METHOD_AWQ_CLIP
    assert report["algorithm"] == sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM
    assert report["algorithm_version"] == sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION
    assert report["clip_ratios"] == list(sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS)
    assert report["block_size"] == 16
    assert report["tie_rule"] == sq.AWQ_CLIP_TIE_RULE
    assert report["template_arithmetic"] == sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED
    assert report["unclipped_code"] == 10
    assert report["artifact_path"] == str(path)
    assert report["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["ratio_code_sha256"] == payload["ratio_code_sha256"]
    assert report["provenance_sha256"] == payload["provenance_sha256"]
    assert report["checkpoint_sha256"] == AWQ_CHECKPOINT_SHA256
    assert report["calibration_path"] == str(calibration)
    assert report["calibration_sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert report["scale_margin"] == 1.0
    assert report["target_fqns"] == expected
    assert report["target_count"] == len(expected)

    total_blocks = 0
    aggregate_histogram = [0] * 11
    for layer in report["layers"]:
        blocks = payload["provenance"]["modules"][layer["fqn"]]["block_count"]
        assert layer["block_count"] == blocks
        assert layer["ratio_histogram"] == payload["provenance"]["modules"][layer["fqn"]]["ratio_histogram"]
        assert sum(layer["ratio_histogram"]) == blocks
        assert layer["weight_count"] == blocks * 16
        assert layer["shape"] == shapes[layer["fqn"]]
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] >= 0.0
        total_blocks += blocks
        aggregate_histogram = [a + b for a, b in zip(aggregate_histogram, layer["ratio_histogram"])]
    assert report["total_block_count"] == total_blocks
    assert report["ratio_histogram"] == aggregate_histogram
    assert report["aggregate"]["block_count"] == total_blocks
    assert report["aggregate"]["ratio_histogram"] == aggregate_histogram
    assert report["total_weight_count"] == total_blocks * 16

    # The offline objective is artifact-carried evidence and is explicitly not a runtime measurement.
    offline = report["offline_objectives"]
    assert offline["source"] == "artifact"
    assert offline["measured_at_runtime"] is False
    assert offline["selected"] == payload["provenance"]["aggregate"]["selected_objective"]
    assert offline["unclipped"] == payload["provenance"]["aggregate"]["unclipped_objective"]
    assert any("deliberately absent" in note for note in report["notes"])
    assert any("makes no claim about DER" in note for note in report["notes"])
    assert any("AWQ-clip artifact" in note for note in summary["notes"])
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.unit
def test_awq_clip_reports_a_worse_plain_mse_without_rejecting_it(tmp_path, fake_torchao, monkeypatch):
    """AWQ optimizes output error, so a larger stored-weight MSE is reported rather than treated as a defect."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))

    def coarse(weight, template, ratio_codes):
        # A deliberately coarser reconstruction than the ordinary conversion's.
        return _fake_nvfp4(_round_to(weight, TEMPLATE_GRID_STEP * 4))

    monkeypatch.setattr(sq, "repack_nvfp4_weight_awq_clip", coarse)

    report = sq.quantize_sortformer_model(model, _awq_config(path, calibration), facts=_facts())[
        "weight_scale_awq_clip"
    ]

    assert report["aggregate"]["searched_mse"] > report["aggregate"]["template_mse"]
    assert report["aggregate"]["relative_reduction"] < 0.0


@pytest.mark.unit
def test_awq_clip_dispatch_is_explicit_and_never_falls_back(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    config = _awq_config(path, calibration)
    artifact = {"ratio_codes": {}, "code_shapes": {}}

    plan = sq._weight_repack_plan(config, None, artifact)
    assert plan == sq._WeightRepackPlan(method=sq.WEIGHT_SCALE_METHOD_AWQ_CLIP, awq_clip=artifact)
    # The absence of another method's data can never imply this one, and vice versa.
    assert sq._weight_repack_plan(_mse_config(), None, artifact) == sq._WeightRepackPlan(
        method=sq.WEIGHT_SCALE_METHOD_MSE
    )
    assert sq._weight_repack_plan(sq.SortformerQuantizationConfig(recipe="nvfp4_all"), None, artifact) is None
    with pytest.raises(RuntimeError, match="without a loaded AWQ-clip artifact"):
        sq._weight_repack_plan(config, None, None)


@pytest.mark.unit
def test_awq_clip_runs_before_producer_fusion(tmp_path, fake_torchao, fake_awq_clip_repack, monkeypatch):
    """Fusion rewrites the blocks the repacked weights live in, so every repack must already have happened."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    seen = {}

    def fuse(module, fqns):
        seen["repacks"] = len(fake_awq_clip_repack)
        return {"enabled": True, "fqns": list(fqns)}

    monkeypatch.setattr(sq, "apply_producer_fusion", fuse)

    sq.quantize_sortformer_model(model, _awq_config(path, calibration, fuse_producer_packing=True), facts=_facts())

    assert seen["repacks"] == len(_expected_target_fqns())


@pytest.mark.unit
def test_awq_clip_cache_identity_separates_artifacts_and_calibrations(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    identity = sq.prediction_cache_identity(_awq_config(path, calibration))

    assert identity["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_AWQ_CLIP
    assert identity["weight_scale_awq_clip_algorithm"] == sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM
    assert identity["weight_scale_awq_clip_algorithm_version"] == sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION
    assert identity["weight_scale_awq_clip_ratios"] == list(sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS)
    assert identity["weight_scale_awq_clip_block_size"] == 16
    assert identity["weight_scale_awq_clip_tie_rule"] == sq.AWQ_CLIP_TIE_RULE
    assert identity["weight_scale_awq_clip_path"] == str(path)
    assert identity["weight_scale_awq_clip_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["weight_scale_awq_clip_calibration_sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()

    # A different artifact, and a different calibration, each separate the cache on their own.
    other_codes = _write_awq(tmp_path, _awq_payload(model, calibration, fill=3), name="other.json")
    assert sq.prediction_cache_identity(_awq_config(other_codes, calibration)) != identity
    other_calibration = _write_awq_calibration(tmp_path, name="other_calib.json", amax=4.0)
    rebuilt = _write_awq(tmp_path, _awq_payload(model, other_calibration), name="rebuilt.json")
    assert sq.prediction_cache_identity(_awq_config(rebuilt, other_calibration)) != identity

    # An unreadable artifact fails closed instead of collapsing two runs onto one identity.
    missing = _awq_config(path, calibration)
    missing.weight_scale_awq_clip_path = str(tmp_path / "gone.json")
    with pytest.raises(ValueError, match="AWQ-clip artifact"):
        sq.prediction_cache_identity(missing)


@pytest.mark.unit
def test_existing_cache_identities_are_unchanged_by_awq_clip(tmp_path):
    """Every identity that existed before this method must stay byte-for-byte what it was."""
    assert sq.prediction_cache_identity(sq.SortformerQuantizationConfig()) is None
    amax = sq.prediction_cache_identity(sq.SortformerQuantizationConfig(recipe="nvfp4_all"))
    assert not any(key.startswith("weight_scale") for key in amax)

    for config in (
        _mse_config(),
        _four_over_six_config(),
        _hessian_config(_write_hessian(tmp_path, _hessian_payload(_FakeSortformer()))),
    ):
        identity = sq.prediction_cache_identity(config)
        assert not any("awq_clip" in key for key in identity)


@pytest.mark.unit
def test_amax_stays_batched_while_awq_clip_converts_one_fqn_at_a_time(tmp_path, fake_torchao, fake_awq_clip_repack):
    """The default method's conversion path is untouched by the per-FQN AWQ-clip path."""
    calibration = _write_awq_calibration(tmp_path)
    batched = _FakeSortformer()
    sq.quantize_sortformer_model(
        batched,
        sq.SortformerQuantizationConfig(recipe="nvfp4_all", scale_mode="static", calibration_path=str(calibration)),
        facts=_facts(),
    )
    prepare, convert = fake_torchao.calls
    assert prepare.selected == _expected_target_fqns()
    assert convert.selected == _expected_target_fqns()
    assert fake_awq_clip_repack == []

    searched = _FakeSortformer()
    path = _write_awq(tmp_path, _awq_payload(searched, calibration), name="searched.json")
    sq.quantize_sortformer_model(searched, _awq_config(path, calibration), facts=_facts())
    assert [call.selected for call in fake_torchao.calls[3:]] == [[fqn] for fqn in _expected_target_fqns()]
    assert len(fake_awq_clip_repack) == len(_expected_target_fqns())


@pytest.mark.unit
def test_evaluator_declares_the_awq_clip_fields():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert "amax, mse, local_hessian, four_over_six, awq_clip" in source
    assert "quantization_weight_scale_awq_clip_path: Optional[str] = None" in source
    assert "'weight_scale_awq_clip' section" in source


@pytest.mark.unit
def test_integrated_awq_clip_repack_runs_the_real_static_w4a4_path(tmp_path):
    """Pinned runtime: static W4A4 over every target, repacked from real AWQ-clip codes."""
    device = _skip_without_the_real_nvfp4_runtime()
    torch.manual_seed(0)
    model = _FakeSortformer().to(device=device, dtype=torch.bfloat16).eval()
    expected = _expected_target_fqns()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_awq(tmp_path, _awq_payload(model, calibration))
    config = _awq_config(path, calibration)
    try:
        sq.check_nvfp4_capability(config, sq.collect_capability_facts(device))
    except RuntimeError as error:
        pytest.skip(f"the real W4A4 path is unavailable here: {error}")

    summary = sq.quantize_sortformer_model(model, config)
    report = summary["weight_scale_awq_clip"]

    assert summary["backend"] == sq.BACKEND_MSLK_ACCELERATED
    assert report["target_fqns"] == expected
    assert sum(report["ratio_histogram"]) == report["total_block_count"]
    for layer in report["layers"]:
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] > 0.0
    assert json.loads(json.dumps(summary)) == summary

    with torch.no_grad():
        output = model(torch.randn(2, 8, D_MODEL, device=device, dtype=torch.bfloat16))
    assert torch.isfinite(output.float()).all()


def _write_awq_calibration(tmp_path, name="awq_calib.json", amax=2.0, **overrides):
    """Write a merged-style static calibration artifact of exactly the target set, overridable field by field.

    The default is the *production* shape that :func:`merge_calibrations` writes and that the frozen cross-domain
    calibration has: the checkpoint digest lives in ``metadata`` and there is no top-level ``checkpoint_sha256``
    at all. ``checkpoint_location`` moves or duplicates that claim, and a ``metadata`` override dict replaces
    individual metadata claims or -- with a ``None`` value -- removes one entirely.
    """
    fqns = _expected_target_fqns()
    checkpoint = overrides.get("checkpoint", AWQ_CHECKPOINT_SHA256)
    location = overrides.get("checkpoint_location", "metadata")
    metadata = {
        "method": sq.CALIBRATION_MERGE_METHOD,
        "method_version": sq.CALIBRATION_MERGE_METHOD_VERSION,
        "percentile": 100.0,
        "headroom": 1.375,
        "headroom_baked_in": True,
        "runtime_scale_margin": 1.0,
    }
    if location in ("metadata", "both"):
        metadata["checkpoint_sha256"] = checkpoint
    for key, value in overrides.get("metadata", {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    payload = {
        "version": overrides.get("version", sq.CALIBRATION_SCHEMA_VERSION),
        "recipe": "disabled",
        "scale_mode": overrides.get("scale_mode", "static"),
        "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
        "metadata": metadata,
        "activation_amax": {fqn: float(amax) for fqn in fqns},
    }
    if location in ("top_level", "both"):
        payload["checkpoint_sha256"] = overrides.get("top_level_checkpoint", checkpoint)
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _awq_payload(model, calibration_path, fqns=None, fill=None, **overrides):
    """A complete, valid AWQ-clip artifact for ``model``, overridable key by key."""
    fqns = list(fqns if fqns is not None else _expected_target_fqns())
    modules = dict(model.named_modules())
    ratio_codes = {}
    module_evidence = {}
    for index, fqn in enumerate(fqns):
        module = modules[fqn]
        rows, blocks = int(module.out_features), int(module.in_features) // sq.NVFP4_BLOCK_SIZE
        raw = bytes(
            (fill if fill is not None else (position + index) % sq.WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
            for position in range(rows * blocks)
        )
        ratio_codes[fqn] = {"shape": [rows, blocks], "codes": base64.b64encode(raw).decode("ascii")}
        histogram = [0] * sq.WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT
        for value in raw:
            histogram[value] += 1
        module_evidence[fqn] = {
            "block_count": rows * blocks,
            "ratio_histogram": histogram,
            "selected_objective": 0.125 * (index + 1),
            "unclipped_objective": 0.25 * (index + 1),
        }
    provenance = {
        "method": sq.AWQ_CLIP_CONSTRUCTION_METHOD,
        "method_version": sq.AWQ_CLIP_CONSTRUCTION_METHOD_VERSION,
        "objective": sq.AWQ_CLIP_OBJECTIVE,
        "group_reduction": sq.AWQ_CLIP_GROUP_REDUCTION,
        "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
        "target_module_count": len(fqns),
        "target_fqns": list(fqns),
        "sources": [
            {
                "label": "near_field",
                "name": "samples_near.pt",
                "sha256": "b" * 64,
                "size_bytes": 4096,
                "seed": 7,
                "max_rows": 512,
                "sampled_row_count": 256,
                "finite_row_count": 4096,
                "nonfinite_row_count": 0,
                "metadata": {"manifest": "near.json"},
            }
        ],
        "modules": module_evidence,
        "aggregate": {
            "module_count": len(fqns),
            "source_count": 1,
            "source_labels": ["near_field"],
            "block_count": sum(module_evidence[fqn]["block_count"] for fqn in fqns),
            "ratio_histogram": [
                sum(module_evidence[fqn]["ratio_histogram"][index] for fqn in fqns)
                for index in range(sq.WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT)
            ],
            "selected_objective": sq.nvfp4_awq_clip_weighted_objective(module_evidence, fqns, "selected_objective"),
            "unclipped_objective": sq.nvfp4_awq_clip_weighted_objective(module_evidence, fqns, "unclipped_objective"),
        },
    }
    payload = {
        "schema": sq.AWQ_CLIP_SCHEMA,
        "version": sq.AWQ_CLIP_SCHEMA_VERSION,
        "checkpoint_sha256": AWQ_CHECKPOINT_SHA256,
        "algorithm": sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM,
        "algorithm_version": sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM_VERSION,
        "arithmetic": {
            "block_size": int(sq.WEIGHT_SCALE_AWQ_CLIP_BLOCK_SIZE),
            "clip_ratios": [float(ratio) for ratio in sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS],
            "tie_rule": sq.AWQ_CLIP_TIE_RULE,
            "objective": sq.AWQ_CLIP_OBJECTIVE,
            "group_reduction": sq.AWQ_CLIP_GROUP_REDUCTION,
            "candidate_scale_rule": sq.AWQ_CLIP_CANDIDATE_SCALE_RULE,
            "template_arithmetic": sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED,
            "activation_qdq": sq.AWQ_CLIP_ACTIVATION_QDQ,
            "fp4_max": 6.0,
            "fp8_e4m3_max": 448.0,
            "scale_min": float(sq.WEIGHT_SCALE_AWQ_CLIP_SCALE_MIN),
            "modelopt_reference_version": sq.MODELOPT_REFERENCE_VERSION,
            "modelopt_reference_wheel_sha256": sq.MODELOPT_REFERENCE_WHEEL_SHA256,
        },
        "activation_calibration": {
            **sq.nvfp4_awq_clip_calibration_identity(str(calibration_path)),
            "scale_margin": 1.0,
        },
        "weight_digest_method": sq.WEIGHT_DIGEST_METHOD,
        "weight_sha256": {fqn: sq.nvfp4_weight_digest(modules[fqn].weight) for fqn in fqns},
        "code_encoding": sq.AWQ_CLIP_CODE_ENCODING,
        "ratio_codes": ratio_codes,
        "provenance": provenance,
    }
    # Recorded exactly as the builder records them, so a payload is only ever mutated below with the digests it
    # would have carried before the mutation.
    payload["ratio_code_sha256"] = sq.nvfp4_section_digest(payload["ratio_codes"])
    payload["provenance_sha256"] = sq.nvfp4_section_digest(payload["provenance"])
    payload.update(overrides)
    return payload


def _reseal_awq(payload):
    """Recompute both section digests, so a structural mutation is judged on its own terms, not as tampering."""
    if isinstance(payload.get("ratio_codes"), dict):
        payload["ratio_code_sha256"] = sq.nvfp4_section_digest(payload["ratio_codes"])
    if isinstance(payload.get("provenance"), dict):
        payload["provenance_sha256"] = sq.nvfp4_section_digest(payload["provenance"])
    return payload


def _write_awq(tmp_path, payload, name="awq_clip.json"):
    """Write an artifact payload and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _awq_config(path, calibration, **overrides):
    """Quantization config with the AWQ-clip repack on, overridable per test.

    The calibration is taken positionally and lands in the ``calibration_path`` option below, so ``overrides`` may
    replace that option like any other -- which is what lets a caller build the invalid combinations this method
    has to reject.
    """
    values = dict(
        recipe="nvfp4_all",
        scale_mode="static",
        calibration_path=str(calibration),
        scale_margin=1.0,
        weight_scale_method=sq.WEIGHT_SCALE_METHOD_AWQ_CLIP,
        weight_scale_awq_clip_path=str(path),
    )
    values.update(overrides)
    return sq.SortformerQuantizationConfig(**values)


def _forbid_non_awq_repackers(monkeypatch):
    """Make every repacker except the AWQ-clip one fatal, so the dispatch cannot silently pick another."""

    def forbidden(*args, **kwargs):
        raise AssertionError("only the AWQ-clip repacker may run under weight_scale_method='awq_clip'")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_mse", forbidden)
    monkeypatch.setattr(sq, "repack_nvfp4_weight_local_hessian", forbidden)
    monkeypatch.setattr(sq, "repack_nvfp4_weight_four_over_six", forbidden)


@pytest.fixture
def fake_awq_clip_repack(fake_torchao, monkeypatch):
    """Replace the accepted AWQ-clip repacker with a deterministic stand-in and record every call."""
    calls = []
    originals = []

    def repack(weight, template, ratio_codes):
        calls.append(
            SimpleNamespace(
                weight=weight.detach().clone(),
                template=template,
                codes=ratio_codes.detach().clone(),
                quantize_calls=len(fake_torchao.calls),
                earlier_originals_alive=[reference() is not None for reference in originals],
            )
        )
        originals.append(weakref.ref(weight))
        return _fake_nvfp4(_round_to(weight, SEARCHED_GRID_STEP))

    monkeypatch.setattr(sq, "repack_nvfp4_weight_awq_clip", repack)
    return calls


# The GPTQ payload binds to the same static calibration shape the AWQ-clip tests write, so both artifacts name the
# same checkpoint here and the shared calibration writer serves both.
GPTQ_CHECKPOINT_SHA256 = AWQ_CHECKPOINT_SHA256


@pytest.mark.unit
def test_gptq_is_off_by_default_and_maps_from_the_evaluator(tmp_path):
    """The technique is opt-in end to end, and both evaluator fields are wired to the core options exactly."""
    default = sq.SortformerQuantizationConfig()
    assert default.weight_scale_method == sq.WEIGHT_SCALE_METHOD_AMAX
    assert default.weight_scale_gptq_path is None
    assert default.uses_gptq_weight_scales is False
    assert default.uses_searched_weight_scales is False

    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    config = sq.quantization_config_from_eval_cfg(
        _eval_cfg(
            quantization_recipe="nvfp4_all",
            quantization_scale_mode="static",
            quantization_calibration_path=str(calibration),
            quantization_weight_scale_method="gptq",
            quantization_weight_scale_gptq_path=str(path),
        )
    )
    assert config.weight_scale_method == "gptq"
    assert config.weight_scale_gptq_path == str(path)
    assert config.uses_gptq_weight_scales is True
    assert config.uses_searched_weight_scales is True
    assert config.uses_mse_weight_scales is False
    assert config.uses_local_hessian_weight_scales is False
    assert config.uses_four_over_six_weight_scales is False
    assert config.uses_awq_clip_weight_scales is False

    # The identity constants are the adapted reference's, not knobs: the arithmetic is defined by exactly these.
    assert sq.WEIGHT_SCALE_METHOD_GPTQ == "gptq"
    assert sq.WEIGHT_SCALE_METHOD_GPTQ in sq.WEIGHT_SCALE_METHODS
    assert sq.WEIGHT_SCALE_GPTQ_ALGORITHM == "nvfp4_qdata_modelopt_gptq"
    assert sq.WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION == 1
    assert sq.WEIGHT_SCALE_GPTQ_PERC_DAMP == 0.01
    assert sq.WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE == 128
    assert sq.WEIGHT_SCALE_GPTQ_BLOCK_SIZE == 16
    assert sq.WEIGHT_SCALE_GPTQ_SCALE_MARGIN == 1.0
    assert sq.WEIGHT_SCALE_GPTQ_SCALE_MODE == "static"
    assert sq.WEIGHT_SCALE_GPTQ_RECIPES == ("nvfp4_all",)
    assert sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETICS == ("torchao_non_triton", "mslk_triton")
    assert sq.GPTQ_SCHEMA == "sortformer_nvfp4_gptq"
    assert sq.GPTQ_SCHEMA_VERSION == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recipe": "nvfp4_weight_only", "scale_mode": "dynamic", "calibration_path": None}, "requires one of recipe"),
        ({"recipe": "nvfp4_qkv_only"}, "requires one of recipe"),
        ({"scale_mode": "dynamic", "calibration_path": None}, "requires scale_mode='static'"),
        ({"scale_margin": 1.375}, "requires quantization_scale_margin"),
        ({"weight_scale_gptq_path": None}, "requires quantization_weight_scale_gptq_path"),
        ({"weight_scale_gptq_path": "  "}, "requires quantization_weight_scale_gptq_path"),
        ({"weight_scale_gptq_path": "/nonexistent/gptq.json"}, "is not a readable file"),
        ({"fold_global_scales": True}, "mutually exclusive"),
        ({"bf16_override_path": "override.json"}, "cannot be combined"),
        ({"weight_scale_hessian_path": "hessian.json"}, "is only used with"),
        ({"weight_scale_awq_clip_path": "awq.json"}, "is only used with"),
    ],
)
def test_gptq_rejects_incompatible_options(tmp_path, overrides, message):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    with pytest.raises(ValueError, match=message):
        _gptq_config(path, calibration, **overrides).validate()


@pytest.mark.unit
def test_gptq_requires_a_readable_calibration_file(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    with pytest.raises(ValueError, match="readable static activation-calibration"):
        _gptq_config(path, tmp_path / "missing.json").validate()


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    [
        sq.WEIGHT_SCALE_METHOD_AMAX,
        sq.WEIGHT_SCALE_METHOD_MSE,
        sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN,
        sq.WEIGHT_SCALE_METHOD_FOUR_OVER_SIX,
        sq.WEIGHT_SCALE_METHOD_AWQ_CLIP,
    ],
)
def test_other_methods_reject_a_supplied_gptq_path(tmp_path, method):
    """An ignored path would let a run believe it executed the GPTQ payload while executing other weights."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    values = {"weight_scale_hessian_path": None, "weight_scale_awq_clip_path": None}
    if method == sq.WEIGHT_SCALE_METHOD_LOCAL_HESSIAN:
        values["weight_scale_hessian_path"] = str(_write_hessian(tmp_path, _hessian_payload(model)))
    if method == sq.WEIGHT_SCALE_METHOD_AWQ_CLIP:
        values["weight_scale_awq_clip_path"] = str(_write_awq(tmp_path, _awq_payload(model, calibration)))
    with pytest.raises(ValueError, match="is only used with"):
        sq.SortformerQuantizationConfig(
            recipe="nvfp4_all",
            scale_mode="static",
            calibration_path=str(calibration),
            weight_scale_method=method,
            weight_scale_gptq_path=str(path),
            **values,
        ).validate()


@pytest.mark.unit
def test_gptq_loader_accepts_a_valid_artifact_and_decodes_its_payload(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    payload = _gptq_payload(model, calibration)
    path = _write_gptq(tmp_path, payload)
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    loaded = sq.load_gptq_artifact(str(path), model, selection, str(calibration))

    assert loaded["checkpoint_sha256"] == GPTQ_CHECKPOINT_SHA256
    assert loaded["algorithm"] == sq.WEIGHT_SCALE_GPTQ_ALGORITHM
    assert loaded["perc_damp"] == 0.01
    assert loaded["update_block_size"] == 128
    assert loaded["block_size"] == 16
    assert loaded["fqns"] == _expected_target_fqns()
    assert loaded["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded["qdata_sha256"] == payload["qdata_sha256"]
    assert loaded["hessian_sha256"] == payload["hessian_sha256"]
    assert loaded["provenance_sha256"] == payload["provenance_sha256"]
    assert loaded["calibration_path"] == str(calibration)
    assert loaded["calibration"]["sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert loaded["calibration"]["scale_margin"] == 1.0
    assert loaded["template_arithmetic"] == sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED
    for fqn in loaded["fqns"]:
        module = model.get_submodule(fqn)
        assert loaded["qdata_shapes"][fqn] == [module.out_features, module.in_features // 2]
        assert isinstance(loaded["qdata"][fqn], bytes)
        assert len(loaded["qdata"][fqn]) == module.out_features * module.in_features // 2
        assert loaded["qdata"][fqn] == base64.b64decode(payload["qdata"][fqn]["payload"], validate=True)
        assert loaded["qdata_digests"][fqn] == hashlib.sha256(loaded["qdata"][fqn]).hexdigest()
        assert loaded["template_scale"][fqn] == payload["template_scale"][fqn]
        assert loaded["hessian"][fqn]["input_features"] == module.in_features
    # The runtime artifact carries the payload and nothing that could identify the data it was selected from.
    assert set(payload) == set(sq.GPTQ_ARTIFACT_KEYS)
    text = json.dumps(payload)
    assert "activation_rows" not in text and "second_moments" not in text
    # The Hessians themselves are never stored: only a digest and a handful of scalar statistics per module.
    for entry in payload["hessian"].values():
        assert set(entry) == set(sq.GPTQ_HESSIAN_KEYS)
        assert all(not isinstance(value, list) for value in entry.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update(schema="other"), "declares schema"),
        (lambda payload: payload.update(version=2), "version 1 is required"),
        (lambda payload: payload.update(version=True), "version 1 is required"),
        (lambda payload: payload.update(algorithm="other"), "was built for algorithm"),
        (lambda payload: payload.update(algorithm_version=2), "was built for algorithm"),
        (lambda payload: payload.update(weight_digest_method="md5"), "'weight_digest_method'"),
        (lambda payload: payload.update(section_digest_method="md5"), "'section_digest_method'"),
        (lambda payload: payload.update(payload_encoding="hex"), "'payload_encoding'"),
        (lambda payload: payload.update(checkpoint_sha256="nope"), "hexadecimal SHA-256"),
        (lambda payload: payload.pop("provenance"), "exactly the top-level keys"),
        (lambda payload: payload.update(extra=1), "exactly the top-level keys"),
        (lambda payload: payload["arithmetic"].update(perc_damp=0.02), "'perc_damp'"),
        (lambda payload: payload["arithmetic"].update(update_block_size=64), "'update_block_size'"),
        (lambda payload: payload["arithmetic"].update(block_size=32), "'block_size'"),
        (lambda payload: payload["arithmetic"].update(hessian_rule="something else"), "'hessian_rule'"),
        (lambda payload: payload["arithmetic"].update(group_reduction="sum"), "'group_reduction'"),
        (lambda payload: payload["arithmetic"].update(dead_column_rule="drop"), "'dead_column_rule'"),
        (lambda payload: payload["arithmetic"].update(inverse_rule="identity fallback"), "'inverse_rule'"),
        (lambda payload: payload["arithmetic"].update(template_scale_rule="recomputed"), "'template_scale_rule'"),
        (lambda payload: payload["arithmetic"].update(activation_qdq="bf16"), "'activation_qdq'"),
        (lambda payload: payload["arithmetic"].update(objective="something else"), "'objective'"),
        (lambda payload: payload["arithmetic"].update(hessian_digest_method="md5"), "'hessian_digest_method'"),
        (lambda payload: payload["arithmetic"].update(fp4_max=8.0), "'fp4_max'"),
        (lambda payload: payload["arithmetic"].update(fp8_e4m3_max=256.0), "'fp8_e4m3_max'"),
        (lambda payload: payload["arithmetic"].update(template_arithmetic="guessed"), "'template_arithmetic'"),
        (lambda payload: payload["arithmetic"].pop("template_arithmetic"), "arithmetic keys"),
        (lambda payload: payload["arithmetic"].update(modelopt_reference_version="0.45.0"), "reference_version"),
        (lambda payload: payload["arithmetic"].update(modelopt_reference_wheel_sha256="b" * 64), "wheel_sha256"),
        (lambda payload: payload["provenance"].update(method="other"), "records 'method'"),
        (lambda payload: payload["provenance"].update(objective="other"), "records 'objective'"),
        (lambda payload: payload["provenance"].update(targets=["attn.w_qkv"]), "declares targets"),
        (lambda payload: payload["provenance"].update(target_module_count=3), "records target_module_count"),
        (lambda payload: payload["provenance"].update(sources=[]), "non-empty 'sources' list"),
        (lambda payload: payload["provenance"]["aggregate"].update(module_count=99), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(weight_count=1), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(qdata_byte_length=1), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(dead_column_count=7), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(selected_mse=99.0), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(selected_objective=99.0), "does not summarize"),
        (lambda payload: payload["provenance"]["aggregate"].update(source_labels=["other"]), "does not summarize"),
        (lambda payload: payload["activation_calibration"].update(sha256="b" * 64), "another activation calibration"),
        (lambda payload: payload["activation_calibration"].update(scale_margin=1.375), "activation scale margin"),
    ],
)
def test_gptq_loader_rejection_matrix(tmp_path, mutate, message):
    """A foreign, edited, stale or self-inconsistent artifact must fail on an unmodified model."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    payload = _gptq_payload(model, calibration)
    mutate(payload)
    _reseal_gptq(payload)
    path = _write_gptq(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        sq.load_gptq_artifact(str(path), model, sq.select_quantization_targets(model, "nvfp4_all"), str(calibration))


@pytest.mark.unit
def test_gptq_loader_rejects_broken_payloads(tmp_path):
    """Truncation, padding, an unusable base64 string, a wrong shape and an altered nibble are each refused."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    fqn = _expected_target_fqns()[0]

    def load(payload):
        _reseal_gptq(payload)
        path = _write_gptq(tmp_path, payload, name="broken.json")
        return sq.load_gptq_artifact(str(path), model, selection, str(calibration))

    truncated = _gptq_payload(model, calibration)
    raw = base64.b64decode(truncated["qdata"][fqn]["payload"], validate=True)
    truncated["qdata"][fqn]["payload"] = base64.b64encode(raw[:-1]).decode("ascii")
    with pytest.raises(ValueError, match="payload byte"):
        load(truncated)

    padded = _gptq_payload(model, calibration)
    padded["qdata"][fqn]["payload"] = base64.b64encode(raw + b"\x00").decode("ascii")
    with pytest.raises(ValueError, match="payload byte"):
        load(padded)

    malformed = _gptq_payload(model, calibration)
    malformed["qdata"][fqn]["payload"] = "not base64!!"
    with pytest.raises(ValueError, match="valid base64"):
        load(malformed)

    empty = _gptq_payload(model, calibration)
    empty["qdata"][fqn]["payload"] = ""
    with pytest.raises(ValueError, match="non-empty base64"):
        load(empty)

    reshaped = _gptq_payload(model, calibration)
    reshaped["qdata"][fqn]["shape"] = [1, len(raw)]
    with pytest.raises(ValueError, match="the live module needs"):
        load(reshaped)

    mistyped = _gptq_payload(model, calibration)
    mistyped["qdata"][fqn]["dtype"] = "float8_e4m3fn"
    with pytest.raises(ValueError, match="packed NVFP4 payload is uint8"):
        load(mistyped)

    mislengthed = _gptq_payload(model, calibration)
    mislengthed["qdata"][fqn]["byte_length"] = len(raw) - 1
    with pytest.raises(ValueError, match="payload byte"):
        load(mislengthed)

    # One flipped FP4 nibble is a different stored weight, and the recorded content digest is what catches it.
    flipped = _gptq_payload(model, calibration)
    altered = bytes([raw[0] ^ 0x01]) + raw[1:]
    flipped["qdata"][fqn]["payload"] = base64.b64encode(altered).decode("ascii")
    with pytest.raises(ValueError, match="altered after it was selected"):
        load(flipped)

    partial = _gptq_payload(model, calibration)
    partial["qdata"].pop(fqn)
    with pytest.raises(ValueError, match="must cover exactly"):
        load(partial)

    extra = _gptq_payload(model, calibration)
    extra["qdata"]["transformer_encoder.layers.0.attn.other"] = extra["qdata"][fqn]
    with pytest.raises(ValueError, match="must cover exactly"):
        load(extra)

    noncanonical = _gptq_payload(model, calibration)
    noncanonical["qdata"]["_orig_mod." + fqn] = noncanonical["qdata"].pop(fqn)
    with pytest.raises(ValueError, match="non-canonical form"):
        load(noncanonical)


@pytest.mark.unit
def test_gptq_loader_rejects_broken_template_scale_and_hessian_claims(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    fqn = _expected_target_fqns()[0]

    def load(payload):
        _reseal_gptq(payload)
        path = _write_gptq(tmp_path, payload, name="claims.json")
        return sq.load_gptq_artifact(str(path), model, selection, str(calibration))

    for mutate, message in (
        (lambda p: p["template_scale"][fqn].update(sha256="zz"), "hexadecimal SHA-256"),
        (lambda p: p["template_scale"][fqn].update(global_scale_sha256="zz"), "hexadecimal SHA-256"),
        (lambda p: p["template_scale"][fqn].update(byte_length=3), "its own shape"),
        (lambda p: p["template_scale"][fqn].update(shape=[1], byte_length=1), "fewer than"),
        (lambda p: p["template_scale"][fqn].pop("dtype"), "template-scale keys"),
        (lambda p: p["template_scale"].pop(fqn), "must give 'template_scale'"),
        (lambda p: p["hessian"][fqn].update(sha256="zz"), "hexadecimal SHA-256"),
        (lambda p: p["hessian"][fqn].update(input_features=8), "does not describe this weight"),
        (lambda p: p["hessian"][fqn].update(sampled_row_count=0), "positive 'sampled_row_count'"),
        (lambda p: p["hessian"][fqn].update(dead_column_count=10**6), "dead input"),
        (lambda p: p["hessian"][fqn].update(damping=0.0), "positive 'damping'"),
        (lambda p: p["hessian"][fqn].update(damping="0.25"), "finite number"),
        (lambda p: p["hessian"][fqn].update(diagonal_min=9.0), "not ordered"),
        (lambda p: p["hessian"][fqn].update(diagonal_min=0.1, diagonal_mean=0.2), "below its own damping"),
        (lambda p: p["hessian"][fqn].pop("damping"), "Hessian keys"),
        (lambda p: p["hessian"].pop(fqn), "must give 'hessian'"),
        (lambda p: p["provenance"]["modules"][fqn].update(weight_count=3), "module, which holds"),
        (lambda p: p["provenance"]["modules"][fqn].update(block_count=3), "FP4 codes per byte"),
        (lambda p: p["provenance"]["modules"][fqn].update(selected_mse=-1.0), "is negative"),
        (lambda p: p["provenance"]["modules"][fqn].update(selected_objective="0.5"), "finite number"),
        (lambda p: p["provenance"]["modules"][fqn].pop("shape"), "module keys"),
        (lambda p: p["provenance"]["modules"].pop(fqn), "exactly the"),
        (lambda p: p["provenance"]["modules"][fqn].update(qdata_byte_length=2), "FP4 codes per byte"),
    ):
        payload = _gptq_payload(model, calibration)
        mutate(payload)
        with pytest.raises(ValueError, match=message):
            load(payload)


@pytest.mark.unit
def test_gptq_loader_rejects_a_corrupted_section_digest_and_duplicate_keys(tmp_path):
    """Each of the three section digests is recomputed, and duplicate keys are refused at every nesting level."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    fqn = _expected_target_fqns()[0]

    for key, mutate in (
        ("qdata_sha256", lambda p: p["qdata"][fqn].update(byte_length=p["qdata"][fqn]["byte_length"])),
        ("hessian_sha256", lambda p: p["hessian"][fqn].update(damping=0.5)),
        ("provenance_sha256", lambda p: p["provenance"].update(target_module_count=len(_expected_target_fqns()))),
    ):
        payload = _gptq_payload(model, calibration)
        mutate(payload)
        payload[key] = "0" * 64
        path = _write_gptq(tmp_path, payload, name=f"{key}.json")
        with pytest.raises(ValueError, match="modified after it was built"):
            sq.load_gptq_artifact(str(path), model, selection, str(calibration))

    duplicated = tmp_path / "duplicate.json"
    duplicated.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        sq.load_gptq_artifact(str(duplicated), model, selection, str(calibration))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        sq.load_gptq_artifact(str(malformed), model, selection, str(calibration))


@pytest.mark.unit
def test_gptq_loader_rejects_a_foreign_weight_and_a_foreign_calibration(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    selection = sq.select_quantization_targets(model, "nvfp4_all")

    stale = _gptq_payload(model, calibration)
    stale["weight_sha256"][_expected_target_fqns()[0]] = "b" * 64
    with pytest.raises(ValueError, match="built on different weights"):
        sq.load_gptq_artifact(str(_write_gptq(tmp_path, stale, name="stale.json")), model, selection, str(calibration))

    other = _write_awq_calibration(tmp_path, name="other_calib.json", amax=4.0)
    with pytest.raises(ValueError, match="another activation calibration"):
        sq.load_gptq_artifact(
            str(_write_gptq(tmp_path, _gptq_payload(model, calibration), name="ok.json")),
            model,
            selection,
            str(other),
        )

    dynamic = _write_awq_calibration(tmp_path, name="dynamic.json", scale_mode="dynamic")
    with pytest.raises(ValueError, match="declares scale_mode"):
        sq.load_gptq_artifact(
            str(_write_gptq(tmp_path, _gptq_payload(model, dynamic), name="dyn.json")),
            model,
            selection,
            str(dynamic),
        )


@pytest.mark.unit
def test_gptq_validates_the_artifact_before_converting_anything(tmp_path, fake_torchao, monkeypatch):
    """A failed artifact check leaves every target module exactly as it was."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    payload = _gptq_payload(model, calibration)
    payload["weight_sha256"][_expected_target_fqns()[0]] = "b" * 64
    _reseal_gptq(payload)
    path = _write_gptq(tmp_path, payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("no payload may be replaced when the artifact does not describe this model")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_gptq", forbidden)

    with pytest.raises(ValueError, match="built on different weights"):
        sq.quantize_sortformer_model(model, _gptq_config(path, calibration), facts=_facts())

    assert fake_torchao.calls == []
    for fqn in _expected_target_fqns():
        assert isinstance(model.get_submodule(fqn), torch.nn.Linear)


@pytest.mark.unit
def test_gptq_refuses_a_backend_that_builds_the_other_ordinary_template(tmp_path, fake_torchao, monkeypatch):
    """The payload only decodes under the block scales one construction produces, so the other backend is refused."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))

    def forbidden(*args, **kwargs):
        raise AssertionError("no payload may be replaced when the artifact describes the other backend's template")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_gptq", forbidden)

    unaccelerated = _gptq_config(path, calibration, accelerated_packing=False, allow_reference_kernels=True)
    assert sq.check_nvfp4_capability(unaccelerated, _facts()) == sq.BACKEND_REFERENCE_UNACCELERATED
    with pytest.raises(ValueError, match="ordinary-template construction"):
        sq.quantize_sortformer_model(model, unaccelerated, facts=_facts())

    assert fake_torchao.calls == []
    for fqn in _expected_target_fqns():
        assert isinstance(model.get_submodule(fqn), torch.nn.Linear)

    reference = _gptq_payload(model, calibration)
    reference["arithmetic"]["template_arithmetic"] = sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_REFERENCE
    loaded = sq.load_gptq_artifact(
        str(_write_gptq(tmp_path, reference, name="reference_gptq.json")),
        model,
        sq.select_quantization_targets(model, "nvfp4_all"),
        str(calibration),
    )
    assert loaded["template_arithmetic"] == sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_REFERENCE
    with pytest.raises(ValueError, match="ordinary-template construction"):
        sq.require_gptq_template_arithmetic(loaded, sq.BACKEND_MSLK_ACCELERATED)
    assert sq.require_gptq_template_arithmetic(loaded, sq.BACKEND_REFERENCE_UNACCELERATED) == (
        sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_REFERENCE
    )


@pytest.mark.unit
def test_gptq_converts_and_replaces_each_fqn_once_with_its_own_payload(
    tmp_path, fake_torchao, fake_gptq_repack, monkeypatch
):
    _forbid_non_gptq_repackers(monkeypatch)
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    calibration = _write_awq_calibration(tmp_path)
    payload = _gptq_payload(model, calibration)
    path = _write_gptq(tmp_path, payload)

    summary = sq.quantize_sortformer_model(model, _gptq_config(path, calibration), facts=_facts())

    # One prepare call for the whole set, then exactly one convert call per FQN in sorted order.
    assert [call.selected for call in fake_torchao.calls[1:]] == [[fqn] for fqn in expected]
    assert len(fake_gptq_repack) == len(expected)
    for index, fqn in enumerate(expected):
        record = fake_gptq_repack[index]
        assert record.qdata.dtype == torch.uint8
        assert bytes(record.qdata.reshape(-1).tolist()) == base64.b64decode(
            payload["qdata"][fqn]["payload"], validate=True
        )
        assert list(record.qdata.shape) == payload["qdata"][fqn]["shape"]
        # Each replacement ran after its own conversion, and no earlier FQN's payload was still resident.
        assert record.quantize_calls == index + 2
        assert record.earlier_payloads_alive == [False] * index
    assert [layer["fqn"] for layer in summary["weight_scale_gptq"]["layers"]] == expected
    # The other methods' summary sections name the method that actually selected the weights.
    assert summary["weight_scale_mse"] == sq.disabled_weight_scale_mse_summary(sq.WEIGHT_SCALE_METHOD_GPTQ)
    assert summary["weight_scale_hessian"] == sq.disabled_weight_scale_hessian_summary(sq.WEIGHT_SCALE_METHOD_GPTQ)
    assert "weight_scale_four_over_six" not in summary
    assert "weight_scale_awq_clip" not in summary


@pytest.mark.unit
def test_gptq_summary_records_the_artifact_identity_and_the_payload_evidence(tmp_path, fake_torchao, fake_gptq_repack):
    model = _FakeSortformer()
    expected = _expected_target_fqns()
    shapes = {
        fqn: [int(model.get_submodule(fqn).out_features), int(model.get_submodule(fqn).in_features)]
        for fqn in expected
    }
    calibration = _write_awq_calibration(tmp_path)
    payload = _gptq_payload(model, calibration)
    path = _write_gptq(tmp_path, payload)

    summary = sq.quantize_sortformer_model(model, _gptq_config(path, calibration), facts=_facts())
    report = summary["weight_scale_gptq"]

    assert summary["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_GPTQ
    assert report["enabled"] is True
    assert report["method"] == sq.WEIGHT_SCALE_METHOD_GPTQ
    assert report["algorithm"] == sq.WEIGHT_SCALE_GPTQ_ALGORITHM
    assert report["algorithm_version"] == sq.WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION
    assert report["perc_damp"] == 0.01
    assert report["update_block_size"] == 128
    assert report["block_size"] == 16
    assert report["hessian_rule"] == sq.GPTQ_HESSIAN_RULE
    assert report["dead_column_rule"] == sq.GPTQ_DEAD_COLUMN_RULE
    assert report["inverse_rule"] == sq.GPTQ_INVERSE_RULE
    assert report["template_scale_rule"] == sq.GPTQ_TEMPLATE_SCALE_RULE
    assert report["activation_qdq"] == sq.GPTQ_ACTIVATION_QDQ
    assert report["template_arithmetic"] == sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED
    assert report["modelopt_reference_version"] == sq.MODELOPT_REFERENCE_VERSION
    assert report["modelopt_reference_wheel_sha256"] == sq.MODELOPT_REFERENCE_WHEEL_SHA256
    assert report["artifact_path"] == str(path)
    assert report["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["qdata_sha256"] == payload["qdata_sha256"]
    assert report["hessian_sha256"] == payload["hessian_sha256"]
    assert report["provenance_sha256"] == payload["provenance_sha256"]
    assert report["checkpoint_sha256"] == GPTQ_CHECKPOINT_SHA256
    assert report["calibration_path"] == str(calibration)
    assert report["calibration_sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert report["scale_margin"] == 1.0
    assert report["target_fqns"] == expected
    assert report["target_count"] == len(expected)
    assert report["source_count"] == 1
    assert report["source_labels"] == ["near_field"]
    assert report["total_sampled_row_count"] == 256

    total_weights = 0
    total_bytes = 0
    for layer in report["layers"]:
        rows, columns = shapes[layer["fqn"]]
        assert layer["shape"] == [rows, columns]
        assert layer["weight_count"] == rows * columns
        assert layer["block_count"] == rows * columns // 16
        assert layer["qdata_byte_length"] == rows * columns // 2
        assert layer["qdata_sha256"] == payload["qdata"][layer["fqn"]]["sha256"]
        assert layer["template_scale_sha256"] == payload["template_scale"][layer["fqn"]]["sha256"]
        assert layer["hessian_sha256"] == payload["hessian"][layer["fqn"]]["sha256"]
        assert layer["dead_column_count"] == payload["hessian"][layer["fqn"]]["dead_column_count"]
        # Both MSEs are this run's own measurement against its own original weights.
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] >= 0.0
        total_weights += layer["weight_count"]
        total_bytes += layer["qdata_byte_length"]
    assert report["total_weight_count"] == total_weights
    assert report["total_qdata_byte_length"] == total_bytes
    assert report["aggregate"]["qdata_byte_length"] == total_bytes
    assert report["aggregate"]["block_count"] == total_weights // 16

    # The offline objective is artifact-carried evidence and is explicitly not a runtime measurement.
    offline = report["offline_objectives"]
    assert offline["source"] == "artifact"
    assert offline["measured_at_runtime"] is False
    assert offline["selected"] == payload["provenance"]["aggregate"]["selected_objective"]
    assert offline["template"] == payload["provenance"]["aggregate"]["template_objective"]
    assert offline["weight_count"] == payload["provenance"]["aggregate"]["weight_count"]
    assert any("only the packed FP4 payload bytes were replaced" in note for note in report["notes"])
    assert any("deliberately absent" in note for note in report["notes"])
    assert any("makes no claim about DER" in note for note in report["notes"])
    assert any("GPTQ artifact" in note for note in summary["notes"])
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.unit
def test_gptq_refuses_a_template_whose_scale_bytes_are_not_the_ones_it_was_written_under(
    tmp_path, fake_torchao, fake_gptq_repack
):
    """The payload's FP4 codes only reconstruct the selected values under the exact scale bytes they name."""
    calibration = _write_awq_calibration(tmp_path)
    fqn = _expected_target_fqns()[0]

    for field, value in (
        ("sha256", "b" * 64),
        ("global_scale_sha256", "b" * 64),
        ("dtype", "torch.float8_e4m3fn"),
    ):
        # A fresh model per case, because ``quantize_sortformer_model`` converts the one it is given and the
        # artifact's weight digests bind it to exactly that model's own weights.
        model = _FakeSortformer()
        payload = _gptq_payload(model, calibration)
        payload["template_scale"][fqn][field] = value
        path = _write_gptq(tmp_path, payload, name=f"scale_{field}.json")
        with pytest.raises(RuntimeError, match="written under another ordinary template"):
            sq.quantize_sortformer_model(model, _gptq_config(path, calibration), facts=_facts())


@pytest.mark.unit
def test_gptq_reports_a_worse_plain_mse_without_rejecting_it(tmp_path, fake_torchao, monkeypatch):
    """GPTQ minimizes output error, so a larger stored-weight MSE is reported rather than treated as a defect."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))

    def coarse(template, qdata):
        return _fake_nvfp4(_round_to(template.detach().float(), TEMPLATE_GRID_STEP * 4))

    monkeypatch.setattr(sq, "repack_nvfp4_weight_gptq", coarse)

    report = sq.quantize_sortformer_model(model, _gptq_config(path, calibration), facts=_facts())["weight_scale_gptq"]

    assert report["aggregate"]["searched_mse"] > report["aggregate"]["template_mse"]
    assert report["aggregate"]["relative_reduction"] < 0.0


@pytest.mark.unit
def test_gptq_dispatch_is_explicit_and_never_falls_back(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    config = _gptq_config(path, calibration)
    artifact = {"qdata": {}, "qdata_shapes": {}}

    plan = sq._weight_repack_plan(config, None, None, artifact)
    assert plan == sq._WeightRepackPlan(method=sq.WEIGHT_SCALE_METHOD_GPTQ, gptq=artifact)
    # The absence of another method's data can never imply this one, and vice versa.
    assert sq._weight_repack_plan(_mse_config(), None, None, artifact) == sq._WeightRepackPlan(
        method=sq.WEIGHT_SCALE_METHOD_MSE
    )
    assert sq._weight_repack_plan(sq.SortformerQuantizationConfig(recipe="nvfp4_all"), None, None, artifact) is None
    with pytest.raises(RuntimeError, match="without a loaded GPTQ artifact"):
        sq._weight_repack_plan(config, None, None, None)


@pytest.mark.unit
def test_gptq_cache_identity_separates_artifacts_and_calibrations(tmp_path):
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    identity = sq.prediction_cache_identity(_gptq_config(path, calibration))

    assert identity["weight_scale_method"] == sq.WEIGHT_SCALE_METHOD_GPTQ
    assert identity["weight_scale_gptq_algorithm"] == sq.WEIGHT_SCALE_GPTQ_ALGORITHM
    assert identity["weight_scale_gptq_algorithm_version"] == sq.WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION
    assert identity["weight_scale_gptq_perc_damp"] == 0.01
    assert identity["weight_scale_gptq_update_block_size"] == 128
    assert identity["weight_scale_gptq_block_size"] == 16
    assert identity["weight_scale_gptq_path"] == str(path)
    assert identity["weight_scale_gptq_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["weight_scale_gptq_calibration_sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()

    # A different payload, and a different calibration, each separate the cache on their own.
    other_payload = _write_gptq(tmp_path, _gptq_payload(model, calibration, fill=3), name="other.json")
    assert sq.prediction_cache_identity(_gptq_config(other_payload, calibration)) != identity
    other_calibration = _write_awq_calibration(tmp_path, name="other_calib.json", amax=4.0)
    rebuilt = _write_gptq(tmp_path, _gptq_payload(model, other_calibration), name="rebuilt.json")
    assert sq.prediction_cache_identity(_gptq_config(rebuilt, other_calibration)) != identity

    # An unreadable artifact fails closed instead of collapsing two runs onto one identity.
    missing = _gptq_config(path, calibration)
    missing.weight_scale_gptq_path = str(tmp_path / "gone.json")
    with pytest.raises(ValueError, match="GPTQ artifact"):
        sq.prediction_cache_identity(missing)


@pytest.mark.unit
def test_existing_cache_identities_are_unchanged_by_gptq(tmp_path):
    """Every identity that existed before this method must stay byte-for-byte what it was."""
    assert sq.prediction_cache_identity(sq.SortformerQuantizationConfig()) is None
    amax = sq.prediction_cache_identity(sq.SortformerQuantizationConfig(recipe="nvfp4_all"))
    assert not any(key.startswith("weight_scale") for key in amax)

    calibration = _write_awq_calibration(tmp_path)
    for config in (
        _mse_config(),
        _four_over_six_config(),
        _hessian_config(_write_hessian(tmp_path, _hessian_payload(_FakeSortformer()))),
        _awq_config(_write_awq(tmp_path, _awq_payload(_FakeSortformer(), calibration)), calibration),
    ):
        identity = sq.prediction_cache_identity(config)
        assert not any("gptq" in key for key in identity)


@pytest.mark.unit
def test_amax_stays_batched_while_gptq_converts_one_fqn_at_a_time(tmp_path, fake_torchao, fake_gptq_repack):
    """The default method's conversion path is untouched by the per-FQN GPTQ path."""
    calibration = _write_awq_calibration(tmp_path)
    batched = _FakeSortformer()
    sq.quantize_sortformer_model(
        batched,
        sq.SortformerQuantizationConfig(recipe="nvfp4_all", scale_mode="static", calibration_path=str(calibration)),
        facts=_facts(),
    )
    prepare, convert = fake_torchao.calls
    assert prepare.selected == _expected_target_fqns()
    assert convert.selected == _expected_target_fqns()
    assert fake_gptq_repack == []

    replaced = _FakeSortformer()
    path = _write_gptq(tmp_path, _gptq_payload(replaced, calibration), name="replaced.json")
    sq.quantize_sortformer_model(replaced, _gptq_config(path, calibration), facts=_facts())
    assert [call.selected for call in fake_torchao.calls[3:]] == [[fqn] for fqn in _expected_target_fqns()]
    assert len(fake_gptq_repack) == len(_expected_target_fqns())


@pytest.mark.unit
def test_gptq_runs_before_producer_fusion(tmp_path, fake_torchao, fake_gptq_repack, monkeypatch):
    """Fusion rewrites the blocks the replaced weights live in, so every replacement must already have happened."""
    model = _FakeSortformer()
    calibration = _write_awq_calibration(tmp_path)
    path = _write_gptq(tmp_path, _gptq_payload(model, calibration))
    seen = {}

    def fuse(module, fqns):
        seen["replacements"] = len(fake_gptq_repack)
        return {"enabled": True, "fqns": list(fqns)}

    monkeypatch.setattr(sq, "apply_producer_fusion", fuse)

    sq.quantize_sortformer_model(model, _gptq_config(path, calibration, fuse_producer_packing=True), facts=_facts())

    assert seen["replacements"] == len(_expected_target_fqns())


@pytest.mark.unit
def test_evaluator_declares_the_gptq_fields():
    script = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "speaker_tasks"
        / "diarization"
        / "neural_diarizer"
        / "e2e_diarize_speech.py"
    )
    if not script.exists():
        pytest.skip("evaluator script is not available in this checkout")
    source = script.read_text(encoding="utf-8")

    assert "amax, mse, local_hessian, four_over_six, awq_clip, gptq" in source
    assert "quantization_weight_scale_gptq_path: Optional[str] = None" in source
    assert "'weight_scale_gptq' section" in source


@pytest.mark.unit
def test_integrated_gptq_replacement_runs_the_real_static_w4a4_path(tmp_path):
    """Pinned runtime: static W4A4 over every target, with a real GPTQ payload replacing every packed weight."""
    device = _skip_without_the_real_nvfp4_runtime()
    weight_mse = importlib.import_module("nemo.collections.asr.parts.utils.sortformer_nvfp4_weight_mse")
    torch.manual_seed(0)
    model = _FakeSortformer().to(device=device, dtype=torch.bfloat16).eval()
    expected = _expected_target_fqns()
    calibration = _write_awq_calibration(tmp_path)
    config = _gptq_config(tmp_path / "unused.json", calibration)
    try:
        sq.check_nvfp4_capability(config, sq.collect_capability_facts(device))
    except RuntimeError as error:
        pytest.skip(f"the real W4A4 path is unavailable here: {error}")

    arithmetic = sq.awq_clip_template_arithmetic_for_backend(sq.BACKEND_MSLK_ACCELERATED)
    payload = _real_gptq_payload(weight_mse, model, calibration, arithmetic)
    path = _write_gptq(tmp_path, payload, name="real_gptq.json")
    config = _gptq_config(path, calibration)

    summary = sq.quantize_sortformer_model(model, config)
    report = summary["weight_scale_gptq"]

    assert summary["backend"] == sq.BACKEND_MSLK_ACCELERATED
    assert report["target_fqns"] == expected
    for layer in report["layers"]:
        assert layer["template_mse"] > 0.0 and layer["searched_mse"] > 0.0
    assert json.loads(json.dumps(summary)) == summary

    with torch.no_grad():
        output = model(torch.randn(2, 8, D_MODEL, device=device, dtype=torch.bfloat16))
    assert torch.isfinite(output.float()).all()


def _gptq_payload(model, calibration_path, fqns=None, fill=None, **overrides):
    """A complete, valid GPTQ artifact for ``model``, overridable key by key.

    The payload bytes are deterministic filler rather than a real selection: this file's fakes stand in for the
    TorchAO conversion, so what these tests bind is the schema, the identities and the dispatch, not the update
    itself, which ``test_sortformer_nvfp4_weight_mse.py`` pins against the reference transcription.
    """
    fqns = list(fqns if fqns is not None else _expected_target_fqns())
    modules = dict(model.named_modules())
    qdata = {}
    template_scale = {}
    hessian = {}
    module_evidence = {}
    for index, fqn in enumerate(fqns):
        module = modules[fqn]
        rows, columns = int(module.out_features), int(module.in_features)
        raw = bytes(
            (fill if fill is not None else (position + index)) % 256 for position in range(rows * columns // 2)
        )
        qdata[fqn] = {
            "shape": [rows, columns // 2],
            "dtype": "uint8",
            "byte_length": len(raw),
            "payload": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        # The scale identity of the template this file's fake conversion produces for that weight; the runtime
        # re-checks it against the wrapper it actually built, so it has to be that wrapper's own.
        identity = sq.nvfp4_template_identity(_fake_nvfp4(_round_to(module.weight, TEMPLATE_GRID_STEP)))
        template_scale[fqn] = {
            "shape": [int(size) for size in identity.scale.shape],
            "dtype": str(identity.scale.dtype),
            "byte_length": int(identity.scale.numel()),
            "sha256": sq.nvfp4_weight_digest(identity.scale),
            "global_scale_sha256": sq.nvfp4_weight_digest(identity.global_scale),
        }
        hessian[fqn] = {
            "sha256": hashlib.sha256(f"hessian:{fqn}".encode("utf-8")).hexdigest(),
            "input_features": columns,
            "sampled_row_count": 256,
            "dead_column_count": 0,
            "damping": 0.25,
            "diagonal_min": 1.5,
            "diagonal_mean": 2.0,
            "diagonal_max": 3.0,
        }
        module_evidence[fqn] = {
            "shape": [rows, columns],
            "weight_count": rows * columns,
            "block_count": rows * columns // sq.NVFP4_BLOCK_SIZE,
            "qdata_byte_length": rows * columns // 2,
            "template_mse": 0.5 * (index + 1),
            "selected_mse": 0.25 * (index + 1),
            "template_objective": 0.125 * (index + 1),
            "selected_objective": 0.0625 * (index + 1),
        }
    provenance = {
        "method": sq.GPTQ_CONSTRUCTION_METHOD,
        "method_version": sq.GPTQ_CONSTRUCTION_METHOD_VERSION,
        "objective": sq.GPTQ_OBJECTIVE,
        "group_reduction": sq.GPTQ_GROUP_REDUCTION,
        "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
        "target_module_count": len(fqns),
        "target_fqns": list(fqns),
        "sources": [
            {
                "label": "near_field",
                "name": "samples_near.pt",
                "sha256": "b" * 64,
                "size_bytes": 4096,
                "seed": 7,
                "max_rows": 512,
                "sampled_row_count": 256,
                "finite_row_count": 4096,
                "nonfinite_row_count": 0,
                "metadata": {"manifest": "near.json"},
            }
        ],
        "modules": module_evidence,
        "aggregate": {
            "module_count": len(fqns),
            "source_count": 1,
            "source_labels": ["near_field"],
            "block_count": sum(module_evidence[fqn]["block_count"] for fqn in fqns),
            "weight_count": sum(module_evidence[fqn]["weight_count"] for fqn in fqns),
            "qdata_byte_length": sum(module_evidence[fqn]["qdata_byte_length"] for fqn in fqns),
            "dead_column_count": sum(hessian[fqn]["dead_column_count"] for fqn in fqns),
            **{
                field: sq.nvfp4_gptq_weighted_objective(module_evidence, fqns, field)
                for field in ("template_mse", "selected_mse", "template_objective", "selected_objective")
            },
        },
    }
    payload = {
        "schema": sq.GPTQ_SCHEMA,
        "version": sq.GPTQ_SCHEMA_VERSION,
        "checkpoint_sha256": GPTQ_CHECKPOINT_SHA256,
        "algorithm": sq.WEIGHT_SCALE_GPTQ_ALGORITHM,
        "algorithm_version": sq.WEIGHT_SCALE_GPTQ_ALGORITHM_VERSION,
        "arithmetic": {
            "perc_damp": float(sq.WEIGHT_SCALE_GPTQ_PERC_DAMP),
            "update_block_size": int(sq.WEIGHT_SCALE_GPTQ_UPDATE_BLOCK_SIZE),
            "block_size": int(sq.WEIGHT_SCALE_GPTQ_BLOCK_SIZE),
            "hessian_rule": sq.GPTQ_HESSIAN_RULE,
            "group_reduction": sq.GPTQ_GROUP_REDUCTION,
            "dead_column_rule": sq.GPTQ_DEAD_COLUMN_RULE,
            "inverse_rule": sq.GPTQ_INVERSE_RULE,
            "template_scale_rule": sq.GPTQ_TEMPLATE_SCALE_RULE,
            "activation_qdq": sq.GPTQ_ACTIVATION_QDQ,
            "objective": sq.GPTQ_OBJECTIVE,
            "hessian_digest_method": sq.GPTQ_HESSIAN_DIGEST_METHOD,
            "template_arithmetic": sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_ACCELERATED,
            "fp4_max": 6.0,
            "fp8_e4m3_max": 448.0,
            "modelopt_reference_version": sq.MODELOPT_REFERENCE_VERSION,
            "modelopt_reference_wheel_sha256": sq.MODELOPT_REFERENCE_WHEEL_SHA256,
        },
        "activation_calibration": {
            **sq.nvfp4_awq_clip_calibration_identity(str(calibration_path)),
            "scale_margin": 1.0,
        },
        "weight_digest_method": sq.WEIGHT_DIGEST_METHOD,
        "section_digest_method": sq.SECTION_DIGEST_METHOD,
        "weight_sha256": {fqn: sq.nvfp4_weight_digest(modules[fqn].weight) for fqn in fqns},
        "payload_encoding": sq.GPTQ_PAYLOAD_ENCODING,
        "qdata": qdata,
        "template_scale": template_scale,
        "hessian": hessian,
        "provenance": provenance,
    }
    # Recorded exactly as the builder records them, so a payload is only ever mutated below with the digests it
    # would have carried before the mutation.
    payload["qdata_sha256"] = sq.nvfp4_section_digest(payload["qdata"])
    payload["hessian_sha256"] = sq.nvfp4_section_digest(payload["hessian"])
    payload["provenance_sha256"] = sq.nvfp4_section_digest(payload["provenance"])
    payload.update(overrides)
    return payload


def _real_gptq_payload(weight_mse, model, calibration_path, arithmetic):
    """A GPTQ artifact built from real torchao templates and a real column-wise selection, for the GPU test."""
    payload = _gptq_payload(model, calibration_path)
    modules = dict(model.named_modules())
    for fqn in _expected_target_fqns():
        weight = modules[fqn].weight.detach()
        rows, columns = int(weight.shape[0]), int(weight.shape[1])
        scale = weight_mse.nvfp4_weight_global_scale(weight)
        template = weight_mse.nvfp4_ordinary_template(weight, scale, arithmetic)
        identity = weight_mse.nvfp4_template_identity(template)
        rows_of_activations = torch.randn(64, columns, device=weight.device, dtype=torch.float32)
        quantized = weight_mse.nvfp4_awq_clip_activation_qdq(rows_of_activations, 2.0)
        damped = weight_mse.nvfp4_gptq_damped_hessian(weight_mse.nvfp4_gptq_hessian([quantized]), weight)
        selection = weight_mse.select_nvfp4_gptq_payload(
            weight, scale, weight_mse.nvfp4_template_block_scales(template), damped.matrix
        )
        raw = selection.qdata.detach().to("cpu").contiguous().reshape(-1).numpy().tobytes()
        payload["qdata"][fqn] = {
            "shape": [rows, columns // 2],
            "dtype": "uint8",
            "byte_length": len(raw),
            "payload": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        payload["template_scale"][fqn] = {
            "shape": [int(size) for size in identity.scale.shape],
            "dtype": str(identity.scale.dtype),
            "byte_length": int(identity.scale.numel()),
            "sha256": sq.nvfp4_weight_digest(identity.scale),
            "global_scale_sha256": sq.nvfp4_weight_digest(identity.global_scale),
        }
        payload["hessian"][fqn]["dead_column_count"] = int(damped.dead_columns)
    payload["provenance"]["aggregate"]["dead_column_count"] = sum(
        int(payload["hessian"][fqn]["dead_column_count"]) for fqn in _expected_target_fqns()
    )
    return _reseal_gptq(payload)


def _reseal_gptq(payload):
    """Recompute the three section digests, so a structural mutation is judged on its own terms."""
    for section, digest in (
        ("qdata", "qdata_sha256"),
        ("hessian", "hessian_sha256"),
        ("provenance", "provenance_sha256"),
    ):
        if isinstance(payload.get(section), dict):
            payload[digest] = sq.nvfp4_section_digest(payload[section])
    return payload


def _write_gptq(tmp_path, payload, name="gptq.json"):
    """Write an artifact payload and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _gptq_config(path, calibration, **overrides):
    """Quantization config with the GPTQ payload replacement on, overridable per test."""
    values = dict(
        recipe="nvfp4_all",
        scale_mode="static",
        calibration_path=str(calibration),
        scale_margin=1.0,
        weight_scale_method=sq.WEIGHT_SCALE_METHOD_GPTQ,
        weight_scale_gptq_path=str(path),
    )
    values.update(overrides)
    return sq.SortformerQuantizationConfig(**values)


def _forbid_non_gptq_repackers(monkeypatch):
    """Make every repacker except the GPTQ one fatal, so the dispatch cannot silently pick another."""

    def forbidden(*args, **kwargs):
        raise AssertionError("only the GPTQ payload replacement may run under weight_scale_method='gptq'")

    monkeypatch.setattr(sq, "repack_nvfp4_weight_mse", forbidden)
    monkeypatch.setattr(sq, "repack_nvfp4_weight_local_hessian", forbidden)
    monkeypatch.setattr(sq, "repack_nvfp4_weight_four_over_six", forbidden)
    monkeypatch.setattr(sq, "repack_nvfp4_weight_awq_clip", forbidden)


@pytest.fixture
def fake_gptq_repack(fake_torchao, monkeypatch):
    """Replace the accepted GPTQ payload replacement with a deterministic stand-in and record every call.

    Besides the arguments, each record keeps the number of ``quantize_`` calls made so far -- which pins the
    replacement to its own conversion -- and whether any *earlier* FQN's device payload was still alive when this
    call started, which is how the one-payload-at-a-time memory contract is checked.
    """
    calls = []
    payloads = []

    def repack(template, qdata):
        calls.append(
            SimpleNamespace(
                template=template,
                qdata=qdata.detach().clone(),
                quantize_calls=len(fake_torchao.calls),
                earlier_payloads_alive=[reference() is not None for reference in payloads],
            )
        )
        payloads.append(weakref.ref(qdata))
        return _fake_nvfp4(_round_to(template.detach().float(), SEARCHED_GRID_STEP))

    monkeypatch.setattr(sq, "repack_nvfp4_weight_gptq", repack)
    return calls

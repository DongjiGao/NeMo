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

"""Tests for the opt-in NVFP4 producer-packing fusion of the Sortformer transformer blocks."""

import copy
import importlib.util
import subprocess
import sys
import textwrap

import pytest
import torch

from nemo.collections.asr.modules.transformer_encoder import TransformerBlock, TransformerEncoderConfig
from nemo.collections.asr.parts.submodules.multi_head_attention import RotaryPositionalEncoding
from nemo.collections.asr.parts.utils import sortformer_nvfp4_producer_fusion as producer_fusion
from nemo.collections.asr.parts.utils.sortformer_quantization import (
    QUANTIZATION_TARGET_SUFFIXES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    ActivationAmaxCollector,
    SortformerQuantizationConfig,
    prediction_cache_identity,
    quantize_sortformer_model,
)

_HAS_FP4_DTYPES = all(
    isinstance(getattr(torch, name, None), torch.dtype) for name in producer_fusion.REQUIRED_TORCH_DTYPES
)
_requires_fp4_dtypes = pytest.mark.skipif(
    not _HAS_FP4_DTYPES, reason="torch build does not expose float4_e2m1fn_x2 / float8_e4m3fn"
)


def _real_nvfp4_available() -> bool:
    """Whether this machine can run the real converted path: Blackwell CUDA plus the pinned optional backends."""
    if not _HAS_FP4_DTYPES or not torch.cuda.is_available():
        return False
    if tuple(torch.cuda.get_device_capability()) not in SUPPORTED_COMPUTE_CAPABILITIES:
        return False
    return all(importlib.util.find_spec(name) is not None for name in ("torchao", "mslk", "triton"))


_requires_real_nvfp4 = pytest.mark.skipif(
    not _real_nvfp4_available(),
    reason="needs an accepted Blackwell CUDA device with TorchAO, MSLK and Triton installed",
)

_D_MODEL = 64
_N_HEADS = 4
_FF_HIDDEN = 256

# Real-conversion shapes: K is a multiple of 64 for both the residual (256) and the FFN hidden width (1024), and
# B * T = 256 keeps the runtime token count a multiple of 128 so the accelerated pack applies.
_REAL_D_MODEL = 256
_REAL_N_HEADS = 4
_REAL_BATCH = 2
_REAL_TIME = 128


class _StubQuantizedWeight:
    """Minimal stand-in for a converted ``NVFP4Tensor``, exposing exactly the attributes the adapter reads."""

    def __init__(self, out_features, in_features, device="cpu"):
        # Packed FP4 stores two values per byte along the reduction dim; ``t()`` is the native GEMM operand.
        self.qdata = torch.randint(0, 255, (out_features, in_features // 2), dtype=torch.uint8, device=device)
        self.scale = torch.randint(
            1, 255, (out_features, in_features // producer_fusion.NVFP4_BLOCK_SIZE), dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn)
        self.per_tensor_scale = torch.tensor([0.25], dtype=torch.float32, device=device)
        self.act_per_tensor_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
        self.is_swizzled_scales = True
        self.act_quant_kwargs = object()
        self._transposed = torch.randint(0, 255, (in_features, out_features // 2), dtype=torch.uint8, device=device)

    def t(self):
        """Transposed view whose ``qdata`` is the native ``_scaled_mm`` weight operand."""
        transposed = object.__new__(_StubQuantizedWeight)
        transposed.qdata = self._transposed
        return transposed


def _quantized_block(qkv_bias=False):
    """Build a BF16 transformer block whose four target linears carry stub converted weights."""
    cfg = TransformerEncoderConfig(
        d_model=_D_MODEL,
        n_heads=_N_HEADS,
        ff_expansion=4.0,
        drop_rate=0.0,
        qkv_bias=qkv_bias,
        self_attention_model="rope",
    )
    rope = RotaryPositionalEncoding(d_k=_D_MODEL // _N_HEADS)
    block = TransformerBlock(cfg, pos_enc=rope).to(torch.bfloat16).eval()
    for linear in (block.attn.w_qkv, block.attn.out_proj, block.ffn.net[0], block.ffn.net[3]):
        out_features, in_features = linear.out_features, linear.in_features
        del linear.weight
        linear.weight = _StubQuantizedWeight(out_features, in_features)
    return block


def _model_with_block(block):
    """Wrap a block in a ``layers`` container so the adapter sees realistic nested FQNs."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([block])
    return model


def _real_bf16_block(qkv_bias=False):
    """Build an unquantized BF16 CUDA block wide enough for the real NVFP4 conversion."""
    cfg = TransformerEncoderConfig(
        d_model=_REAL_D_MODEL,
        n_heads=_REAL_N_HEADS,
        ff_expansion=4.0,
        drop_rate=0.0,
        qkv_bias=qkv_bias,
        self_attention_model="no_pos",
    )
    return TransformerBlock(cfg).to(device="cuda", dtype=torch.bfloat16).eval()


def _real_quantized_pair(tmp_path, qkv_bias=False):
    """
    Convert one BF16 block twice -- once unfused, once producer-fused -- from a shared calibration file.

    Returns:
        (reference_model, fused_model, fused_summary, x): the two converted models, the fused run's quantization
        summary, and the BF16 activation both were calibrated on.
    """
    torch.manual_seed(0)
    block = _real_bf16_block(qkv_bias=qkv_bias)
    x = torch.randn(_REAL_BATCH, _REAL_TIME, _REAL_D_MODEL, dtype=torch.bfloat16, device="cuda")

    reference_model = _model_with_block(block)
    fused_model = _model_with_block(copy.deepcopy(block))

    with ActivationAmaxCollector(reference_model) as collector, torch.no_grad():
        reference_model.layers[0](x)
    calibration_path = collector.save(str(tmp_path / "calibration.json"))

    def config(fuse):
        return SortformerQuantizationConfig(
            recipe="nvfp4_all",
            scale_mode="static",
            calibration_path=calibration_path,
            fuse_producer_packing=fuse,
        )

    quantize_sortformer_model(reference_model, config(False))
    fused_summary = quantize_sortformer_model(fused_model, config(True))
    return reference_model, fused_model, fused_summary, x


def _relative_error(actual, expected):
    """Relative Frobenius error, the shape-independent way these tests compare two NVFP4 results."""
    return float((actual - expected).float().norm() / expected.float().norm())


def _fqns(prefix="layers.0"):
    return [f"{prefix}.{suffix}" for suffix in producer_fusion.PRODUCER_FUSION_BLOCK_MEMBERS]


@pytest.fixture
def cpu_device(monkeypatch):
    """Let the CPU tests exercise the adapter itself; the device policy has its own dedicated tests."""
    monkeypatch.setattr(producer_fusion, "validate_producer_fusion_device", lambda *args, **kwargs: (12, 0))


class TestTargetCoverage:
    """The fused set must be exactly the quantized set, and never a partial or unrecognized one."""

    @pytest.mark.unit
    def test_block_members_match_the_quantization_targets(self):
        assert producer_fusion.PRODUCER_FUSION_BLOCK_MEMBERS == QUANTIZATION_TARGET_SUFFIXES

    @pytest.mark.unit
    def test_pack_layout_constants_match_the_pack_primitives(self):
        """The adapter repeats the pack constants so that its import stays free of the ASR collection."""
        from nemo.collections.asr.parts.utils import sortformer_nvfp4_fused_pack as fused_pack

        for name in ("NVFP4_BLOCK_SIZE", "SCALE_TILE_ROWS", "SCALE_TILE_COLS", "REQUIRED_K_MULTIPLE"):
            assert getattr(producer_fusion, name) == getattr(fused_pack, name), name

    @pytest.mark.unit
    def test_grouping_covers_every_complete_block(self):
        blocks = producer_fusion.group_producer_fusion_blocks(_fqns("layers.0") + _fqns("layers.1"))
        assert list(blocks) == ["layers.0", "layers.1"]
        assert blocks["layers.0"] == sorted(_fqns("layers.0"))

    @pytest.mark.unit
    def test_partial_block_is_rejected(self):
        with pytest.raises(ValueError, match="incomplete"):
            producer_fusion.group_producer_fusion_blocks(_fqns()[:-1])

    @pytest.mark.unit
    def test_unrecognized_fqn_is_rejected(self):
        with pytest.raises(ValueError, match="matches none"):
            producer_fusion.group_producer_fusion_blocks(_fqns() + ["layers.0.attn.q_norm"])

    @pytest.mark.unit
    def test_duplicate_member_is_rejected(self):
        with pytest.raises(ValueError, match="twice"):
            producer_fusion.group_producer_fusion_blocks(_fqns() + ["layers.0.attn.w_qkv"])

    @pytest.mark.unit
    def test_top_level_member_is_rejected(self):
        with pytest.raises(ValueError, match="top-level"):
            producer_fusion.group_producer_fusion_blocks(list(producer_fusion.PRODUCER_FUSION_BLOCK_MEMBERS))

    @pytest.mark.unit
    def test_empty_selection_is_rejected(self):
        with pytest.raises(ValueError, match="nothing to fuse"):
            producer_fusion.apply_producer_fusion(torch.nn.Module(), [])


class TestDevicePolicy:
    """One common Blackwell path: no device-name branch, no fallback, no CPU execution."""

    @pytest.mark.unit
    def test_cpu_is_rejected(self):
        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            producer_fusion.validate_producer_fusion_device(torch.device("cpu"), ((12, 0),))

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_capability_outside_the_policy_is_rejected(self):
        with pytest.raises(RuntimeError, match="only accepted on compute capabilities"):
            producer_fusion.validate_producer_fusion_device(torch.device("cuda"), ((9, 0),))

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_default_policy_is_the_quantization_policy(self):
        from nemo.collections.asr.parts.utils.sortformer_quantization import SUPPORTED_COMPUTE_CAPABILITIES

        capability = tuple(torch.cuda.get_device_capability(torch.device("cuda")))
        if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
            pytest.skip("device is outside the accepted Blackwell policy")
        assert producer_fusion.validate_producer_fusion_device(torch.device("cuda")) == capability


class TestAdapterInstallation:
    """Installing the adapter must consume the converted weights and preserve the public model exactly."""

    @pytest.mark.unit
    @_requires_fp4_dtypes
    @pytest.mark.parametrize("qkv_bias", [False, True])
    def test_operands_and_scales_are_derived_from_the_converted_weight(self, cpu_device, qkv_bias):
        block = _quantized_block(qkv_bias=qkv_bias)
        model = _model_with_block(block)
        weight = block.attn.w_qkv.weight
        summary = producer_fusion.apply_producer_fusion(model, _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight)

        fused = model.layers[0]
        assert summary["enabled"] and summary["fused_block_fqns"] == ["layers.0"]
        assert summary["fused_boundaries"] == list(producer_fusion.FUSED_PRODUCER_BOUNDARIES)
        # The GEMM operands are the converted tensors themselves, never a second weight format.
        assert fused.qkv_weight_qdata.data_ptr() == weight.t().qdata.data_ptr()
        assert fused.qkv_weight_scale.data_ptr() == weight.scale.data_ptr()
        # MSLK packs with the reciprocal of TorchAO's per-tensor dequantization scale...
        assert torch.equal(fused.qkv_activation_scale, torch.tensor([2.0]))
        # ... and the epilogue multiplies by the BF16-rounded product of the two global scales.
        assert fused.qkv_output_scale.dtype == torch.bfloat16
        assert float(fused.qkv_output_scale) == pytest.approx(0.125)
        assert float(fused.qkv_output_scale_f32) == pytest.approx(0.125)
        # The bias stays owned by the linear; whether the checkpoint has one is read from the module.
        assert (block.attn.w_qkv.bias is not None) is qkv_bias

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_state_dict_and_public_fqns_are_unchanged(self, cpu_device):
        block = _quantized_block()
        model = _model_with_block(block)
        before = sorted(model.state_dict())
        modules_before = sorted(name for name, _ in model.named_modules())

        producer_fusion.apply_producer_fusion(model, _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight)

        assert sorted(model.state_dict()) == before
        assert sorted(name for name, _ in model.named_modules()) == modules_before
        parameter_ids = [id(parameter) for parameter in model.parameters()]
        assert len(parameter_ids) == len(set(parameter_ids)), "a parameter was registered twice"

        fused = model.layers[0]
        # The GEMM operands stay owned by the converted linears: they are held as plain attributes, so the
        # weight storage is never re-registered under a second owner that ``nn.Module`` would transform twice.
        buffer_names = {name for name, _ in model.named_buffers()}
        assert not any(name.endswith(("_weight_qdata", "_weight_scale")) for name in buffer_names)
        assert isinstance(fused.qkv_weight_qdata, torch.Tensor)
        # Only the scalars the adapter computes itself are registered, and never persistently.
        assert {name for name in buffer_names if "activation_scale" in name or "output_scale" in name}
        buffer_storages = {buffer.untyped_storage().data_ptr() for _, buffer in model.named_buffers()}
        parameter_storages = {parameter.untyped_storage().data_ptr() for _, parameter in model.named_parameters()}
        assert not buffer_storages & parameter_storages, "a registered buffer aliases parameter storage"

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_gemm_operands_follow_the_weights_through_module_transforms(self, cpu_device):
        """``.to()`` and friends replace the converted weights; the cached operands must not go stale."""
        block = _quantized_block()
        model = _model_with_block(block)
        producer_fusion.apply_producer_fusion(model, _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight)
        fused = model.layers[0]

        replacement = _StubQuantizedWeight(fused.attn.w_qkv.out_features, fused.attn.w_qkv.in_features)
        fused.attn.w_qkv.weight = replacement
        model._apply(lambda tensor: tensor)

        assert fused.qkv_weight_qdata.data_ptr() == replacement.t().qdata.data_ptr()
        assert fused.qkv_weight_scale.data_ptr() == replacement.scale.data_ptr()

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_unconverted_linear_is_rejected(self, cpu_device):
        block = _quantized_block()
        del block.ffn.net[3].weight
        block.ffn.net[3].weight = torch.nn.Parameter(torch.zeros(_D_MODEL, _FF_HIDDEN, dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="TorchAO-converted"):
            producer_fusion.apply_producer_fusion(
                _model_with_block(block), _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight
            )

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_weight_only_conversion_is_rejected(self, cpu_device):
        block = _quantized_block()
        block.ffn.net[0].weight.act_quant_kwargs = None
        with pytest.raises(RuntimeError, match="weight-only"):
            producer_fusion.apply_producer_fusion(
                _model_with_block(block), _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight
            )

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_unswizzled_scales_are_rejected(self, cpu_device):
        block = _quantized_block()
        block.attn.out_proj.weight.is_swizzled_scales = False
        with pytest.raises(RuntimeError, match="is_swizzled_scales=False"):
            producer_fusion.apply_producer_fusion(
                _model_with_block(block), _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight
            )

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_non_bf16_norm_is_rejected(self, cpu_device):
        block = _quantized_block()
        block.norm2.to(torch.float32)
        with pytest.raises(ValueError, match="must be torch.bfloat16"):
            producer_fusion.apply_producer_fusion(
                _model_with_block(block), _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight
            )

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_training_mode_is_rejected_at_install_and_in_forward(self, cpu_device):
        block = _quantized_block().train()
        with pytest.raises(RuntimeError, match="inference-only"):
            producer_fusion.apply_producer_fusion(
                _model_with_block(block), _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight
            )

        model = _model_with_block(_quantized_block())
        producer_fusion.apply_producer_fusion(model, _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight)
        model.train()
        with pytest.raises(RuntimeError, match="training mode"):
            model.layers[0](torch.zeros(2, 8, _D_MODEL, dtype=torch.bfloat16))

    @pytest.mark.unit
    @_requires_fp4_dtypes
    def test_grad_bearing_input_is_rejected(self, cpu_device):
        model = _model_with_block(_quantized_block())
        producer_fusion.apply_producer_fusion(model, _fqns(), nvfp4_tensor_cls=_StubQuantizedWeight)
        x = torch.zeros(2, 8, _D_MODEL, dtype=torch.bfloat16, requires_grad=True)
        with pytest.raises(RuntimeError, match="requires grad"):
            model.layers[0](x)


class TestCustomOps:
    """The fused packs are reached through opaque custom operators with faithful fake implementations."""

    @pytest.mark.unit
    def test_registration_is_idempotent(self):
        producer_fusion.register_producer_pack_ops()
        producer_fusion.register_producer_pack_ops()
        assert producer_fusion._op_is_registered(producer_fusion.LAYER_NORM_PACK_OP_NAME)
        assert producer_fusion._op_is_registered(producer_fusion.SCALED_GELU_PACK_OP_NAME)

    @pytest.mark.unit
    def test_reimport_does_not_raise(self):
        """A fresh interpreter loading the module twice and registering twice must not double-register."""
        script = textwrap.dedent(
            f"""
            import importlib
            import importlib.util

            spec = importlib.util.spec_from_file_location("_probe", {producer_fusion.__file__!r})
            for _ in range(2):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # A reload resets the module's own registration flag, so this is the real double-registration
                # risk: the second module object must still recognize the operators already on torch.ops.
                module.register_producer_pack_ops()
                assert module._op_is_registered(module.LAYER_NORM_PACK_OP_NAME)
                assert module._op_is_registered(module.SCALED_GELU_PACK_OP_NAME)
            """
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim,k_dim", [(256, 512), (100, 64)])
    def test_fake_shapes_dtypes_and_device(self, m_dim, k_dim):
        producer = torch.empty(m_dim, k_dim, dtype=torch.bfloat16, device="meta")
        scale = torch.ones(1, dtype=torch.float32, device="meta")
        for packed, scales in (
            producer_fusion._layer_norm_pack_fake(producer, producer[0], None, scale, 1e-5),
            producer_fusion._scaled_gelu_pack_fake(producer, scale, scale, None),
        ):
            assert packed.shape == (m_dim, k_dim // 2)
            assert scales.shape == (-(-m_dim // 128) * 128, -(-(k_dim // 16) // 4) * 4)
            assert packed.dtype == scales.dtype == torch.uint8
            assert packed.device == scales.device == producer.device

    @pytest.mark.unit
    def test_layer_norm_fake_accepts_the_real_fp32_residual_shape(self):
        """Inductor calls the op with the evaluator's real fixed shape and an FP32 residual: M=32*11304, K=512."""
        m_dim, k_dim = 32 * 11304, 512
        producer = torch.empty(m_dim, k_dim, dtype=torch.float32, device="meta")
        weight = torch.empty(k_dim, dtype=torch.bfloat16, device="meta")
        scale = torch.ones(1, dtype=torch.float32, device="meta")

        packed, scales = producer_fusion._layer_norm_pack_fake(producer, weight, weight, scale, 1e-5)

        assert packed.shape == (m_dim, k_dim // 2)
        assert scales.shape == (m_dim, k_dim // 16)
        assert packed.dtype == scales.dtype == torch.uint8
        assert packed.device == scales.device == producer.device

    @pytest.mark.unit
    def test_no_optional_dependency_is_imported_by_the_module(self):
        """Importing the module must not pull in Triton, MSLK or TorchAO."""
        script = textwrap.dedent(
            f"""
            import importlib.util
            import sys

            import torch  # noqa: F401 -- baseline, so torch itself is not blamed for a triton import

            def optional(modules):
                roots = ("triton", "mslk", "torchao")
                return sorted(n for n in modules if n.split(".")[0] in roots)

            baseline = optional(sys.modules)
            spec = importlib.util.spec_from_file_location("_probe", {producer_fusion.__file__!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            leaked = [n for n in optional(sys.modules) if n not in baseline]
            assert not leaked, leaked
            """
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


class TestConfigurationOption:
    """The option is off by default and legal only for the complete static accelerated NVFP4 recipe."""

    @staticmethod
    def _config(**overrides):
        options = {
            "recipe": "nvfp4_all",
            "scale_mode": "static",
            "calibration_path": "calibration.json",
            "fuse_producer_packing": True,
        }
        options.update(overrides)
        return SortformerQuantizationConfig(**options)

    @pytest.mark.unit
    def test_default_is_disabled(self):
        assert SortformerQuantizationConfig().fuse_producer_packing is False

    @pytest.mark.unit
    def test_valid_combination_is_accepted(self):
        self._config().validate()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"recipe": "nvfp4_qkv_only"}, "recipe='nvfp4_all'"),
            ({"recipe": "nvfp4_weight_only", "scale_mode": "dynamic", "calibration_path": None}, "nvfp4_all"),
            ({"scale_mode": "dynamic", "calibration_path": None}, "scale_mode='static'"),
            ({"accelerated_packing": False, "allow_reference_kernels": True}, "accelerated_packing=True"),
            ({"allow_reference_kernels": True}, "different"),
            ({"fold_global_scales": True}, "mutually"),
        ],
    )
    def test_incompatible_combinations_are_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            self._config(**overrides).validate()

    @pytest.mark.unit
    def test_cache_identity_records_the_option(self):
        enabled = prediction_cache_identity(self._config())
        disabled = prediction_cache_identity(self._config(fuse_producer_packing=False))
        assert enabled["fuse_producer_packing"] is True
        assert disabled["fuse_producer_packing"] is False
        assert enabled != disabled

    @pytest.mark.unit
    def test_disabled_summary_is_inert(self):
        summary = producer_fusion.disabled_producer_fusion_summary()
        assert summary["enabled"] is False
        assert summary["fused_block_count"] == 0
        assert summary["fused_block_fqns"] == [] and summary["fused_boundaries"] == []


class TestRealConvertedPath:
    """
    End-to-end coverage of the real TorchAO-converted, MSLK-packed path.

    Every test here needs an accepted Blackwell device with the pinned optional backends and skips cleanly
    everywhere else. The assertions are stated as a relative Frobenius error rather than an elementwise tolerance,
    and the bounds below are *regression* bounds, not an accuracy gate: the product gate for this path is the
    final DER delta, and these tests only have to catch an operand orientation, an operand order or a reciprocated
    global scale, each of which lands at an error of order 1.

    The two paths are deliberately not bit-identical:

    * TorchAO's unfused block rounds intermediates to BF16 -- the FFN-up rescale, then the bias, then the GELU --
      while the fused pack (like the accepted reference primitive it reproduces) evaluates rescale, bias and the
      exact GELU in FP32 and rounds once, at the same BF16 boundary the pack consumes;
    * NVFP4's 16-value grid amplifies such a sub-BF16 difference whenever an element crosses a block-quantization
      boundary, so a fraction of a percent on the producer becomes a few percent on the packed activation and
      survives the following GEMM.

    The bounds are set from measured values on B6000/SM120 with the pinned container, with roughly 3-7x headroom:
    fused-vs-unfused was 0.0322 at worst (the FFN, which carries two fused boundaries), and eager-vs-compiled was
    0.0028 for the producers and 0.0070 for the whole block, where Inductor fuses the eager BF16 epilogue into one
    FP32 kernel and shifts the same boundary.
    """

    # Fused static NVFP4 against the unfused static NVFP4 reference (measured worst case 0.0322).
    PARITY_BOUND = 0.10
    # Compiled against eager, same fused implementation (measured worst case 0.0070).
    COMPILE_BOUND = 0.05

    @pytest.mark.unit
    @_requires_real_nvfp4
    @pytest.mark.parametrize("qkv_bias", [False, True])
    def test_fused_qkv_matches_the_torchao_linear(self, tmp_path, qkv_bias):
        """The packed activation, the GEMM operand order and the rescale-then-bias epilogue reproduce TorchAO."""
        reference_model, fused_model, _, x = _real_quantized_pair(tmp_path, qkv_bias=qkv_bias)
        reference_block, fused_block = reference_model.layers[0], fused_model.layers[0]
        assert (reference_block.attn.w_qkv.bias is not None) is qkv_bias

        with torch.no_grad():
            expected = reference_block.attn.w_qkv(reference_block.norm1(x)).reshape(-1, 3 * _REAL_D_MODEL)
            actual = fused_block._fused_qkv(x)

        assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
        assert _relative_error(actual, expected) < self.PARITY_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_fused_qkv_accepts_the_fp32_autocast_residual(self, tmp_path):
        """The real compiled encoder hands the pre-norm boundary an FP32 residual, not a BF16 one.

        Under CUDA BF16 autocast ``layer_norm`` keeps FP32 input and FP32 arithmetic, and only the following
        quantized linear consumes the normalized activation at BF16. The fused path must reproduce exactly that
        boundary from the same FP32 residual, without casting or mutating it.
        """
        reference_model, fused_model, _, x = _real_quantized_pair(tmp_path)
        reference_block, fused_block = reference_model.layers[0], fused_model.layers[0]
        residual = x.float()
        untouched = residual.clone()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized = reference_block.norm1(residual)
            expected = reference_block.attn.w_qkv(normalized).reshape(-1, 3 * _REAL_D_MODEL)
        with torch.no_grad():
            actual = fused_block._fused_qkv(residual)

        assert normalized.dtype == torch.float32, "autocast must keep the pre-norm LayerNorm in FP32"
        assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
        assert residual.dtype == torch.float32 and torch.equal(residual, untouched)
        assert _relative_error(actual, expected) < self.PARITY_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_fused_block_matches_the_unfused_block_on_the_fp32_autocast_residual(self, tmp_path):
        """Whole-block parity in the real mixed-precision configuration, with the residual precision preserved."""
        reference_model, fused_model, _, x = _real_quantized_pair(tmp_path)
        reference_block, fused_block = reference_model.layers[0], fused_model.layers[0]
        residual = x.float()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            expected = reference_block(residual)
            actual = fused_block(residual)

        assert actual.shape == residual.shape
        assert actual.dtype == expected.dtype == torch.float32
        assert _relative_error(actual, expected) < self.PARITY_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_unsupported_residual_dtype_fails_closed(self, tmp_path):
        """A dtype outside the declared pre-norm contract raises instead of being cast into the fused pack."""
        _, fused_model, _, x = _real_quantized_pair(tmp_path)
        with torch.no_grad(), pytest.raises(ValueError, match="must be torch.bfloat16 or torch.float32"):
            fused_model.layers[0](x.to(torch.float16))

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_fused_ffn_matches_the_torchao_feed_forward(self, tmp_path):
        """The FFN-up raw output, the fused scaled-GELU pack and the FFN-down epilogue reproduce TorchAO."""
        reference_model, fused_model, _, x = _real_quantized_pair(tmp_path)
        reference_block, fused_block = reference_model.layers[0], fused_model.layers[0]

        with torch.no_grad():
            expected = reference_block.ffn(reference_block.norm2(x)).reshape(-1, _REAL_D_MODEL)
            actual = fused_block._fused_ffn(x)

        assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
        assert _relative_error(actual, expected) < self.PARITY_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    @pytest.mark.parametrize("backend", ["flex", "fa4_cute"])
    def test_fused_block_matches_the_unfused_static_nvfp4_block(self, tmp_path, backend):
        """Whole-block parity on both post-QKV attention routes; the attention body itself is unchanged."""
        if backend == "fa4_cute" and importlib.util.find_spec("flash_attn") is None:
            pytest.skip("the fa4_cute backend needs flash_attn")
        reference_model, fused_model, summary, x = _real_quantized_pair(tmp_path)
        reference_block, fused_block = reference_model.layers[0], fused_model.layers[0]
        assert summary["producer_fusion"]["enabled"] is True
        assert summary["producer_fusion"]["fused_block_fqns"] == ["layers.0"]

        seqused_k = None
        if backend == "fa4_cute":
            reference_block.attn.set_attention_backend(backend)
            fused_block.attn.set_attention_backend(backend)
            seqused_k = torch.full((_REAL_BATCH,), _REAL_TIME, dtype=torch.int32, device="cuda")

        with torch.no_grad():
            expected = reference_block(x, seqused_k=seqused_k)
            actual = fused_block(x, seqused_k=seqused_k)

        assert actual.shape == x.shape
        assert _relative_error(actual, expected) < self.PARITY_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_producer_custom_ops_compile_fullgraph(self, tmp_path):
        """Both fused producer boundaries trace into a single fixed-shape graph with no break."""
        _, fused_model, _, x = _real_quantized_pair(tmp_path)
        fused_block = fused_model.layers[0]

        def producers(activation):
            return fused_block._fused_qkv(activation), fused_block._fused_ffn(activation)

        compiled = torch.compile(producers, fullgraph=True, dynamic=False)
        with torch.no_grad():
            expected_qkv, expected_ffn = producers(x)
            actual_qkv, actual_ffn = compiled(x)

        assert _relative_error(actual_qkv, expected_qkv) < self.COMPILE_BOUND
        assert _relative_error(actual_ffn, expected_ffn) < self.COMPILE_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_producer_custom_ops_compile_fullgraph_on_the_fp32_residual(self, tmp_path):
        """The compiled path must accept the FP32 residual the real evaluator supplies, not only a BF16 one."""
        _, fused_model, _, x = _real_quantized_pair(tmp_path)
        fused_block = fused_model.layers[0]
        residual = x.float()

        def producers(activation):
            return fused_block._fused_qkv(activation), fused_block._fused_ffn(activation)

        compiled = torch.compile(producers, fullgraph=True, dynamic=False)
        with torch.no_grad():
            expected_qkv, expected_ffn = producers(residual)
            actual_qkv, actual_ffn = compiled(residual)

        assert actual_qkv.dtype == actual_ffn.dtype == torch.bfloat16
        assert _relative_error(actual_qkv, expected_qkv) < self.COMPILE_BOUND
        assert _relative_error(actual_ffn, expected_ffn) < self.COMPILE_BOUND

    @pytest.mark.unit
    @_requires_real_nvfp4
    def test_converted_block_compiles_at_fixed_shape(self, tmp_path):
        """The whole fused block survives the evaluator's ``dynamic=False`` compilation."""
        _, fused_model, _, x = _real_quantized_pair(tmp_path)
        fused_block = fused_model.layers[0]
        compiled = torch.compile(fused_block, dynamic=False)

        with torch.no_grad():
            expected = fused_block(x)
            actual = compiled(x)
            # A second call at the same shape must reuse the compiled graph rather than recompile.
            repeated = compiled(x)

        assert _relative_error(actual, expected) < self.COMPILE_BOUND
        assert torch.equal(actual, repeated)


@pytest.mark.unit
def test_required_consumers_match_the_linears_the_block_actually_drives():
    """M-1: the safety argument is that fusion packs into exactly these three. Assert it instead of
    documenting it, so adding attn.out_proj to _gemm_linears without updating the consumer tuple fails here
    rather than silently allowing an override to restore a linear fusion packs into."""
    assert producer_fusion.PRODUCER_FUSION_CONSUMER_PREFIXES == ("qkv", "ffn_up", "ffn_down")
    assert len(producer_fusion.PRODUCER_FUSION_REQUIRED_CONSUMERS) == len(
        producer_fusion.PRODUCER_FUSION_CONSUMER_PREFIXES
    )
    assert set(producer_fusion.PRODUCER_FUSION_REQUIRED_CONSUMERS) < set(producer_fusion.PRODUCER_FUSION_BLOCK_MEMBERS)
    assert "attn.out_proj" not in producer_fusion.PRODUCER_FUSION_REQUIRED_CONSUMERS


@pytest.mark.unit
def test_grouping_accepts_a_block_without_out_proj():
    """M-7: the case the change exists for -- three consumers present, attn.out_proj restored."""
    prefix = "transformer_encoder.layers.0"
    selected = [f"{prefix}.{suffix}" for suffix in producer_fusion.PRODUCER_FUSION_REQUIRED_CONSUMERS]

    blocks = producer_fusion.group_producer_fusion_blocks(selected)

    assert list(blocks) == [prefix]
    assert not any(fqn.endswith(".attn.out_proj") for fqn in blocks[prefix])


@pytest.mark.unit
@pytest.mark.parametrize("consumer", producer_fusion.PRODUCER_FUSION_REQUIRED_CONSUMERS)
def test_grouping_still_rejects_a_block_missing_any_fused_consumer(consumer):
    """M-7: out_proj became optional; the three consumers must not have."""
    prefix = "transformer_encoder.layers.0"
    selected = [f"{prefix}.{suffix}" for suffix in producer_fusion.PRODUCER_FUSION_BLOCK_MEMBERS if suffix != consumer]

    with pytest.raises(ValueError, match="incomplete"):
        producer_fusion.group_producer_fusion_blocks(selected)


@pytest.mark.unit
def test_disabled_summary_keeps_the_enabled_schema():
    """M-3: a consumer reading out_proj_restored_block_count must not KeyError on a fusion-off run."""
    disabled = producer_fusion.disabled_producer_fusion_summary()

    assert disabled["out_proj_restored_block_count"] == 0
    assert disabled["enabled"] is False

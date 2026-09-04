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

"""Tests for the standalone NVFP4 MSE block-scale weight packer."""

import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_nvfp4_weight_mse as weight_mse

# Attribute-name candidates the module itself accepts, reused here so the tests read the template exactly as the
# packer does instead of pinning one private torchao spelling.
_QDATA_ATTRS = ("qdata", "_data")
_SCALE_ATTRS = ("_scale_e4m3", "scale", "_scale")
_PER_TENSOR_ATTRS = ("_per_tensor_scale", "per_tensor_scale")
_ACT_PER_TENSOR_ATTRS = ("_act_per_tensor_scale", "act_per_tensor_scale")
_SWIZZLE_ATTRS = ("_is_swizzled_scales", "is_swizzled_scales")
_TRITON_ATTRS = ("use_triton_kernel", "_use_triton_kernel")
_ACT_KWARGS_ATTRS = ("act_quant_kwargs", "_act_quant_kwargs")
_BLOCK_SIZE_ATTRS = ("_block_size", "block_size")
_ORIG_DTYPE_ATTRS = ("_orig_dtype", "orig_dtype")

# Smallest positive E4M3 encoding (bit pattern 0x01, the subnormal 2**-9). Ties resolve to it.
_SMALLEST_E4M3 = 2.0**-9


class TestLazyOptionalDependencies:
    """Importing the module must not pull in torchao, and validation must fail before it would be needed."""

    @pytest.mark.unit
    def test_import_does_not_resolve_torchao(self):
        """A bare import of the module must not add torchao to sys.modules."""
        probe = f"""
            import importlib.util
            import sys

            import torch  # noqa: F401  -- baseline: torch itself must not be blamed for a torchao import

            def optional(modules):
                return sorted(n for n in modules if n == "torchao" or n.startswith("torchao."))

            baseline = optional(sys.modules)
            spec = importlib.util.spec_from_file_location("_nvfp4_weight_mse_probe", {weight_mse.__file__!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            leaked = [n for n in optional(sys.modules) if n not in baseline]
            assert not leaked, leaked
            assert module._torchao_backend is None
            """
        script = textwrap.dedent(probe)
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

    @pytest.mark.unit
    def test_missing_torchao_fails_closed_at_call_time(self, monkeypatch):
        """An absent torchao raises an actionable error only when the packer is actually called."""
        _skip_without_e4m3()

        def _missing(module_path):
            raise weight_mse.TorchAOUnavailableError(f"'{module_path}' could not be imported.")

        monkeypatch.setattr(weight_mse, "_torchao_backend", None)
        monkeypatch.setattr(weight_mse, "_import_module", _missing)
        with pytest.raises(ValueError, match="could not be imported"):
            weight_mse.repack_nvfp4_weight_mse(torch.zeros(1, 16, dtype=torch.bfloat16), object())
        # The error is catchable both ways, so callers may treat it as a missing backend or as a bad call.
        assert issubclass(weight_mse.TorchAOUnavailableError, (ImportError, ValueError))


class TestScaleCandidates:
    """The candidate set is the exact E4M3 encoding set, not an approximate float grid."""

    @pytest.mark.unit
    def test_candidates_are_the_126_positive_finite_encodings(self):
        _skip_without_e4m3()
        candidates = weight_mse.nvfp4_scale_candidates()

        assert candidates.numel() == weight_mse.NVFP4_SCALE_CANDIDATE_COUNT == 126
        assert candidates.dtype == torch.float32
        assert torch.isfinite(candidates).all()
        assert (candidates > 0).all()
        assert candidates.unique().numel() == candidates.numel()

    @pytest.mark.unit
    def test_candidates_are_bit_patterns_1_to_126_in_order(self):
        """Bit patterns 0x00 (+0.0) and 0x7F (NaN) are the only ones dropped, and the order is the bit order."""
        _skip_without_e4m3()
        expected = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
        candidates = weight_mse.nvfp4_scale_candidates()

        assert torch.equal(candidates, expected)
        assert bool((candidates[1:] > candidates[:-1]).all()), "bit order must also be ascending in value"
        assert candidates[0].item() == _SMALLEST_E4M3
        assert candidates[-1].item() == weight_mse.FP8_E4M3_MAX
        # Every candidate must survive an E4M3 round trip, or the search could select an unrepresentable scale.
        assert torch.equal(candidates.to(torch.float8_e4m3fn).to(torch.float32), candidates)


@pytest.mark.usefixtures("nvfp4_backend")
class TestScaleSearch:
    """The search itself, which needs torchao's kernels but no converted template."""

    @pytest.mark.unit
    def test_ties_select_the_first_candidate(self, nvfp4_backend):
        """Every candidate reconstructs an all-zero block exactly, so the first one in bit order must win."""
        _, device = nvfp4_backend
        weight = torch.zeros(2, 32, dtype=torch.float32, device=device)
        scales = weight_mse.select_nvfp4_block_scales(weight, torch.ones((), device=device))

        assert scales.shape == (2, 2)
        assert scales.dtype == torch.float8_e4m3fn
        assert torch.equal(scales.float(), torch.full_like(scales.float(), _SMALLEST_E4M3))

    @pytest.mark.unit
    def test_never_worse_than_the_amax_rule(self, nvfp4_backend):
        """Blockwise MSE under the selected scales never exceeds the ordinary max-based scales' MSE."""
        _, device = nvfp4_backend
        weight = _random_weight(24, 96, device)
        global_scale = _global_scale(weight)

        selected = weight_mse.select_nvfp4_block_scales(weight, global_scale)
        amax_based = _amax_based_scales(weight, global_scale)
        selected_mse = weight_mse.nvfp4_blockwise_mse(weight, selected, global_scale)
        amax_mse = weight_mse.nvfp4_blockwise_mse(weight, amax_based, global_scale)

        assert torch.isfinite(selected_mse).all()
        assert bool((selected_mse <= amax_mse + 1e-12).all()), "MSE search must never increase a block's error"

    @pytest.mark.unit
    def test_strictly_beats_the_amax_rule_on_a_constructed_block(self, nvfp4_backend):
        """A block whose amax/6 rounds *up* in E4M3 is fit better by clipping the maximum slightly.

        With a unit global scale the amax rule picks ``3.3 / 6 = 0.55``, which rounds to the E4M3 value 0.5625 and
        leaves every 1.5 between two representable points. The scale 0.5 represents 1.5 exactly for the fifteen
        small weights and costs only a small clip on the single large one, which is the lower-MSE choice.
        """
        _, device = nvfp4_backend
        weight = torch.tensor([[1.5] * 15 + [3.3]], dtype=torch.float32, device=device)
        global_scale = torch.ones((), device=device)

        selected = weight_mse.select_nvfp4_block_scales(weight, global_scale)
        amax_based = _amax_based_scales(weight, global_scale)
        selected_mse = weight_mse.nvfp4_blockwise_mse(weight, selected, global_scale)
        amax_mse = weight_mse.nvfp4_blockwise_mse(weight, amax_based, global_scale)

        assert not torch.equal(selected.float(), amax_based.float())
        assert float(selected_mse) < float(amax_mse)

    @pytest.mark.unit
    @pytest.mark.parametrize("candidate_chunk_size,block_chunk_size", [(1, 1), (5, 7), (126, 10**6)])
    def test_chunking_does_not_change_the_selection(self, nvfp4_backend, candidate_chunk_size, block_chunk_size):
        """The chunk sizes bound memory only; the selected scales must be byte-identical."""
        _, device = nvfp4_backend
        weight = _random_weight(6, 64, device, seed=3)
        global_scale = _global_scale(weight)

        reference = weight_mse.select_nvfp4_block_scales(weight, global_scale)
        chunked = weight_mse.select_nvfp4_block_scales(weight, global_scale, candidate_chunk_size, block_chunk_size)
        assert torch.equal(chunked.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.usefixtures("nvfp4_backend")
class TestRepack:
    """End-to-end repacking against a real converted template."""

    @pytest.mark.unit
    def test_preserves_wrapper_class_metadata_and_layout(self, nvfp4_backend):
        """Everything except the block scales and the packed payload comes through untouched."""
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device)
        template = _make_template(cls, weight, _global_scale(weight))

        result = weight_mse.repack_nvfp4_weight_mse(weight, template)

        assert type(result) is type(template)
        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert result.device == template.device
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            got, expected = _attr(result, attrs), _attr(template, attrs)
            assert got.shape == expected.shape
            assert got.dtype == expected.dtype
            assert got.device == expected.device
            assert got.is_contiguous()
        # The global scale is not re-optimized: the very same tensor is handed to the new wrapper.
        assert _attr(result, _PER_TENSOR_ATTRS) is _attr(template, _PER_TENSOR_ATTRS)
        assert _attr(result, _ACT_PER_TENSOR_ATTRS) is _attr(template, _ACT_PER_TENSOR_ATTRS)
        assert bool(_attr(result, _SWIZZLE_ATTRS)) is bool(_attr(template, _SWIZZLE_ATTRS)) is True
        assert _attr(result, _TRITON_ATTRS) == _attr(template, _TRITON_ATTRS)
        assert _attr(result, _ACT_KWARGS_ATTRS) == _attr(template, _ACT_KWARGS_ATTRS)

    @pytest.mark.unit
    def test_preserves_a_triton_kernel_template(self, nvfp4_backend):
        """The deployment template carries ``use_triton_kernel=True``, and the repack must hand that flag on.

        The wrapper is built directly from a standard converted template's own packed buffers rather than through
        another conversion, so this exercises the flag on a real ``NVFP4Tensor`` without requiring the Triton/MSLK
        quantization kernel to be installed. The packer itself never dispatches, so it must simply carry the flag.
        """
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=19)
        standard = _make_template(cls, weight, _global_scale(weight))
        template = cls(
            _attr(standard, _QDATA_ATTRS),
            _attr(standard, _SCALE_ATTRS),
            _attr(standard, _BLOCK_SIZE_ATTRS),
            _attr(standard, _ORIG_DTYPE_ATTRS),
            per_tensor_scale=_attr(standard, _PER_TENSOR_ATTRS),
            act_per_tensor_scale=_attr(standard, _ACT_PER_TENSOR_ATTRS),
            is_swizzled_scales=True,
            use_triton_kernel=True,
            act_quant_kwargs=_attr(standard, _ACT_KWARGS_ATTRS),
        )
        assert bool(_attr(template, _TRITON_ATTRS)) is True

        result = weight_mse.repack_nvfp4_weight_mse(weight, template)

        assert bool(_attr(result, _TRITON_ATTRS)) is True
        assert bool(_attr(result, _SWIZZLE_ATTRS)) is True
        assert _attr(result, _BLOCK_SIZE_ATTRS) == _attr(template, _BLOCK_SIZE_ATTRS)
        assert _attr(result, _ORIG_DTYPE_ATTRS) == _attr(template, _ORIG_DTYPE_ATTRS)
        assert _attr(result, _PER_TENSOR_ATTRS) is _attr(template, _PER_TENSOR_ATTRS)
        assert _attr(result, _ACT_KWARGS_ATTRS) == _attr(template, _ACT_KWARGS_ATTRS)
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            got, expected = _attr(result, attrs), _attr(template, attrs)
            assert (got.shape, got.dtype, got.device) == (expected.shape, expected.dtype, expected.device)
            assert got.is_contiguous()

    @pytest.mark.unit
    def test_qdata_matches_the_selected_scales(self, nvfp4_backend):
        """The packed payload is exactly torchao's quantization of the weight under the selected scales."""
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=5)
        global_scale = _global_scale(weight)
        template = _make_template(cls, weight, global_scale)

        result = weight_mse.repack_nvfp4_weight_mse(weight, template)

        backend = weight_mse._resolve_torchao()
        scales = weight_mse.select_nvfp4_block_scales(weight, global_scale)
        expected = weight_mse._pack_qdata(weight.float(), scales, global_scale.float(), backend)
        assert torch.equal(_attr(result, _QDATA_ATTRS).view(torch.uint8).reshape(-1), expected.reshape(-1))

        # And the swizzled scale buffer is torchao's own blocked view of those same linear scales.
        blocked = backend.to_blocked(scales).flatten()
        assert torch.equal(_attr(result, _SCALE_ATTRS).view(torch.uint8).reshape(-1), blocked.view(torch.uint8))

    @pytest.mark.unit
    def test_dequantizes_no_worse_than_the_template(self, nvfp4_backend):
        """The repacked tensor dequantizes finitely and closer to the original weight than the template does."""
        cls, device = nvfp4_backend
        weight = _random_weight(16, 64, device, seed=7)
        global_scale = _global_scale(weight)
        template = _make_template(cls, weight, global_scale)
        result = weight_mse.repack_nvfp4_weight_mse(weight, template)

        repacked = _dequantize(result)
        original = _dequantize(template)
        reference = weight.float()

        assert torch.isfinite(repacked).all()
        assert repacked.shape == reference.shape
        repacked_mse = float(((repacked - reference) ** 2).mean())
        assert repacked_mse <= float(((original - reference) ** 2).mean()) + 1e-12
        # The dequantized values must track the objective the search minimized. The tolerance only covers whatever
        # rounding the wrapper's own dequantization applies on the way out; it is not a numerical free pass.
        selected = weight_mse.select_nvfp4_block_scales(weight, global_scale)
        expected_mse = float(weight_mse.nvfp4_blockwise_mse(weight, selected, global_scale).mean())
        assert repacked_mse <= expected_mse * 1.05 + 1e-12

    @pytest.mark.unit
    def test_does_not_mutate_inputs_or_template_storage(self, nvfp4_backend):
        """The weight and every buffer owned by the template are read only."""
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=11)
        global_scale = _global_scale(weight)
        template = _make_template(cls, weight, global_scale)

        before = {
            "weight": weight.clone(),
            "qdata": _attr(template, _QDATA_ATTRS).view(torch.uint8).clone(),
            "scale": _attr(template, _SCALE_ATTRS).view(torch.uint8).clone(),
            "global": global_scale.clone(),
        }
        result = weight_mse.repack_nvfp4_weight_mse(weight, template)

        assert torch.equal(weight, before["weight"])
        assert torch.equal(_attr(template, _QDATA_ATTRS).view(torch.uint8), before["qdata"])
        assert torch.equal(_attr(template, _SCALE_ATTRS).view(torch.uint8), before["scale"])
        assert torch.equal(global_scale, before["global"])
        assert _attr(result, _QDATA_ATTRS).data_ptr() != _attr(template, _QDATA_ATTRS).data_ptr()
        assert _attr(result, _SCALE_ATTRS).data_ptr() != _attr(template, _SCALE_ATTRS).data_ptr()

    @pytest.mark.unit
    @pytest.mark.parametrize("candidate_chunk_size,block_chunk_size", [(1, 1), (7, 3)])
    def test_chunked_repack_is_byte_identical(self, nvfp4_backend, candidate_chunk_size, block_chunk_size):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=13)
        template = _make_template(cls, weight, _global_scale(weight))

        reference = weight_mse.repack_nvfp4_weight_mse(weight, template)
        chunked = weight_mse.repack_nvfp4_weight_mse(weight, template, candidate_chunk_size, block_chunk_size)
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            assert torch.equal(_attr(chunked, attrs).view(torch.uint8), _attr(reference, attrs).view(torch.uint8))

    @pytest.mark.unit
    def test_rejects_a_weight_that_does_not_match_the_template(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=17)
        template = _make_template(cls, weight, _global_scale(weight))

        with pytest.raises(ValueError, match="but the template covers"):
            weight_mse.repack_nvfp4_weight_mse(_random_weight(8, 32, device), template)


class TestValidationWithoutTorchAO:
    """Fail-closed validation, proved against a stand-in backend so it also runs where torchao is absent."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "overrides,error,message",
        [
            ({"block_size": 32}, ValueError, "block size 16"),
            ({"is_swizzled_scales": False}, ValueError, "swizzled"),
            ({"orig_dtype": torch.float16}, ValueError, "orig_dtype"),
            ({"per_tensor_scale": None}, ValueError, "per_tensor_scale is missing"),
            ({"per_tensor_scale": torch.tensor(0.0)}, ValueError, "positive and finite"),
            ({"per_tensor_scale": torch.tensor(float("inf"))}, ValueError, "positive and finite"),
            ({"per_tensor_scale": torch.tensor([1.0, 2.0])}, ValueError, "scalar tensor"),
            ({"per_tensor_scale": 1.0}, TypeError, "must be a torch.Tensor"),
            ({"qdata": torch.zeros(8, 8, dtype=torch.uint8)}, ValueError, "cannot be preserved safely"),
            ({"drop": "use_triton_kernel"}, ValueError, "does not expose"),
            ({"drop": "act_quant_kwargs"}, ValueError, "does not expose"),
        ],
    )
    def test_incompatible_templates_fail_closed(self, monkeypatch, overrides, error, message):
        _skip_without_e4m3()
        template = _fake_template(**overrides)
        _install_fake_backend(monkeypatch)
        with pytest.raises(error, match=message):
            weight_mse.repack_nvfp4_weight_mse(_fake_weight(), template)

    @pytest.mark.unit
    def test_non_nvfp4_template_is_rejected(self, monkeypatch):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(TypeError, match="must be a torchao"):
            weight_mse.repack_nvfp4_weight_mse(_fake_weight(), torch.zeros(8, 64))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "case,error,message",
        [
            ("not_a_tensor", TypeError, "must be a torch.Tensor"),
            ("rank", ValueError, "rank 2"),
            ("dtype", ValueError, "bfloat16"),
            ("contiguous", ValueError, "contiguous"),
            ("shape", ValueError, "but the template covers"),
            ("non_finite", ValueError, "non-finite"),
        ],
    )
    def test_invalid_weights_fail_closed(self, monkeypatch, case, error, message):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        weight = _fake_weight()
        if case == "not_a_tensor":
            weight = [[0.0] * 64] * 8
        elif case == "rank":
            weight = torch.zeros(8, 8, 8, dtype=torch.bfloat16)
        elif case == "dtype":
            weight = torch.zeros(8, 64, dtype=torch.float16)
        elif case == "contiguous":
            weight = torch.zeros(64, 8, dtype=torch.bfloat16).t()
        elif case == "shape":
            weight = torch.zeros(8, 32, dtype=torch.bfloat16)
        elif case == "non_finite":
            weight = _fake_weight()
            weight[0, 0] = float("nan")

        with pytest.raises(error, match=message):
            weight_mse.repack_nvfp4_weight_mse(weight, _fake_template())

    @pytest.mark.unit
    def test_k_not_divisible_by_the_block_size_is_rejected(self, monkeypatch):
        """A ``K`` that is not a multiple of 16 is caught on the template, before any weight is touched."""
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(ValueError, match="multiple of 16"):
            weight_mse.repack_nvfp4_weight_mse(torch.zeros(8, 24, dtype=torch.bfloat16), _fake_template(columns=24))

    @pytest.mark.unit
    @pytest.mark.parametrize("candidate_chunk_size,block_chunk_size", [(0, 8), (8, 0), (-1, 8), (8, -1)])
    def test_non_positive_chunk_sizes_are_rejected(self, monkeypatch, candidate_chunk_size, block_chunk_size):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(ValueError, match="must be a positive int"):
            weight_mse.repack_nvfp4_weight_mse(
                _fake_weight(), _fake_template(), candidate_chunk_size, block_chunk_size
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [True, 1.5, "8", None])
    def test_non_integer_chunk_sizes_are_rejected(self, monkeypatch, value):
        """``True`` is an int in python but never a meaningful chunk size, so bools are rejected too."""
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(ValueError, match="must be a positive int"):
            weight_mse.repack_nvfp4_weight_mse(_fake_weight(), _fake_template(), value, 8)


@pytest.mark.usefixtures("nvfp4_backend")
class TestLocalHessianSearch:
    """The activation-weighted search: same candidates and arithmetic, different reduction."""

    @pytest.mark.unit
    def test_constant_moments_reproduce_the_mse_selection_exactly(self, nvfp4_backend):
        """Unweighted MSE is the constant-``h`` special case, so the two selections must be byte-identical."""
        _, device = nvfp4_backend
        weight = _random_weight(12, 64, device, seed=23)
        global_scale = _global_scale(weight)

        for value in (1.0, 7.5, 1e-3):
            moments = torch.full((64,), value, dtype=torch.float32, device=device)
            # The damped vector is the exact h + 0.01 * mean(h), in h's own units; the search then rescales it to
            # a maximum of one, and a constant vector rescales to exactly one, which is what makes the reduction
            # bit-for-bit the unweighted one.
            damped = weight_mse.damped_second_moments(moments)
            assert torch.equal(damped, moments + weight_mse.NVFP4_HESSIAN_DAMPING * moments.mean())
            assert torch.equal(weight_mse._search_weights(damped), torch.ones(64, dtype=torch.float32, device=device))
            weighted = weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, moments)
            unweighted = weight_mse.select_nvfp4_block_scales(weight, global_scale)
            assert torch.equal(weighted.view(torch.uint8), unweighted.view(torch.uint8))

    @pytest.mark.unit
    def test_constant_moments_reproduce_the_mse_repack_byte_for_byte(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=29)
        template = _make_template(cls, weight, _global_scale(weight))
        moments = torch.ones(64, dtype=torch.float32, device=device)

        weighted = weight_mse.repack_nvfp4_weight_local_hessian(weight, template, moments)
        unweighted = weight_mse.repack_nvfp4_weight_mse(weight, template)

        assert type(weighted) is type(unweighted)
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            assert torch.equal(_attr(weighted, attrs).view(torch.uint8), _attr(unweighted, attrs).view(torch.uint8))
        # Every carried wrapper attribute is the template's own, exactly as for the unweighted repack.
        assert _attr(weighted, _PER_TENSOR_ATTRS) is _attr(template, _PER_TENSOR_ATTRS)
        assert _attr(weighted, _ACT_PER_TENSOR_ATTRS) is _attr(template, _ACT_PER_TENSOR_ATTRS)
        assert bool(_attr(weighted, _SWIZZLE_ATTRS)) is True
        assert _attr(weighted, _TRITON_ATTRS) == _attr(template, _TRITON_ATTRS)
        assert _attr(weighted, _ACT_KWARGS_ATTRS) == _attr(template, _ACT_KWARGS_ATTRS)
        assert _attr(weighted, _BLOCK_SIZE_ATTRS) == _attr(template, _BLOCK_SIZE_ATTRS)
        assert _attr(weighted, _ORIG_DTYPE_ATTRS) == _attr(template, _ORIG_DTYPE_ATTRS)

    @pytest.mark.unit
    def test_nonuniform_moments_pick_another_scale_and_lower_the_weighted_error(self, nvfp4_backend):
        """Weighting the block's largest column heavily makes clipping it expensive, so the scale must change.

        With constant weights the MSE search clips the single 3.3 to fit the fifteen 1.5s exactly. Here the 3.3's
        own input channel carries essentially all of the activation energy, so the block that reconstructs *it*
        well is the better one, and the search must find it.
        """
        _, device = nvfp4_backend
        weight = torch.tensor([[1.5] * 15 + [3.3]], dtype=torch.float32, device=device)
        global_scale = torch.ones((), device=device)
        moments = torch.tensor([1e-4] * 15 + [1.0], dtype=torch.float32, device=device)

        unweighted = weight_mse.select_nvfp4_block_scales(weight, global_scale)
        weighted = weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, moments)
        assert not torch.equal(weighted.float(), unweighted.float())

        objective = weight_mse.nvfp4_blockwise_local_hessian_objective
        assert float(objective(weight, weighted, global_scale, moments)) < float(
            objective(weight, unweighted, global_scale, moments)
        )
        # ... and it is still no better than the weighted optimum on the unweighted objective's own terms:
        # the unweighted search remains the minimizer of the plain MSE.
        assert float(weight_mse.nvfp4_blockwise_mse(weight, unweighted, global_scale)) <= float(
            weight_mse.nvfp4_blockwise_mse(weight, weighted, global_scale)
        )

    @pytest.mark.unit
    def test_weighted_objective_matches_a_scalar_reference(self, nvfp4_backend):
        """The vectorized objective must equal an element-by-element Python computation of the same formula."""
        _, device = nvfp4_backend
        weight = _random_weight(3, 32, device, seed=31)
        global_scale = _global_scale(weight)
        generator = torch.Generator().manual_seed(5)
        moments = torch.rand(32, generator=generator).to(device=device, dtype=torch.float32)
        scales = weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, moments)

        produced = weight_mse.nvfp4_blockwise_local_hessian_objective(weight, scales, global_scale, moments)
        expected = _scalar_weighted_objective(weight, scales, global_scale, moments)
        assert produced.shape == expected.shape
        assert torch.allclose(produced.cpu(), expected, rtol=1e-5, atol=1e-9)

    @pytest.mark.unit
    def test_reported_objectives_are_absolute_and_scale_with_the_moments(self, nvfp4_backend):
        """A non-unit moment scale must show up in the reported value, exactly as the formula says.

        The reference weights are damped in plain Python, so this pins the absolute magnitude rather than merely
        agreeing with the packer's own helper: any per-layer normalization would divide the moment scale back out
        and report the same number for ``h`` and for ``1024 h``.
        """
        _, device = nvfp4_backend
        weight = _random_weight(2, 32, device, seed=53)
        global_scale = _global_scale(weight)
        base = torch.tensor(
            [0.5 + 0.25 * (position % 5) for position in range(32)], dtype=torch.float32, device=device
        )
        scales = weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, base)

        values = base.tolist()
        mean = sum(values) / len(values)
        hand_damped = torch.tensor([value + 0.01 * mean for value in values], dtype=torch.float32)
        produced = weight_mse.nvfp4_blockwise_local_hessian_objective(weight, scales, global_scale, base)
        assert torch.allclose(
            produced.cpu(),
            _scalar_weighted_objective(weight, scales, global_scale, base, damped=hand_damped),
            rtol=1e-5,
            atol=1e-12,
        )
        assert float(produced.max()) > 0.0

        # 1024 is a power of two, so h -> 1024 h scales every FP32 step of the damping and of the reduction
        # exactly: the objective is multiplied by 1024 and nothing else moves.
        scaled = weight_mse.nvfp4_blockwise_local_hessian_objective(weight, scales, global_scale, base * 1024.0)
        assert torch.equal(scaled, produced * 1024.0)
        # ... and the selection itself is invariant under that positive factor, because only reporting is absolute.
        assert torch.equal(
            weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, base * 1024.0).view(torch.uint8),
            scales.view(torch.uint8),
        )

    @pytest.mark.unit
    def test_the_weighted_search_minimizes_its_own_objective(self, nvfp4_backend):
        """No other candidate scale can beat the selected one on the weighted objective, block by block."""
        _, device = nvfp4_backend
        weight = _random_weight(4, 32, device, seed=37)
        global_scale = _global_scale(weight)
        generator = torch.Generator().manual_seed(11)
        moments = torch.rand(32, generator=generator).to(device=device, dtype=torch.float32)
        selected = weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, moments)
        best = weight_mse.nvfp4_blockwise_local_hessian_objective(weight, selected, global_scale, moments)

        for candidate in weight_mse.nvfp4_scale_candidates(device):
            scales = torch.full_like(selected.float(), float(candidate)).to(torch.float8_e4m3fn)
            other = weight_mse.nvfp4_blockwise_local_hessian_objective(weight, scales, global_scale, moments)
            assert bool((best <= other + 1e-12).all())

    @pytest.mark.unit
    @pytest.mark.parametrize("candidate_chunk_size,block_chunk_size", [(1, 1), (5, 7), (126, 10**6)])
    def test_chunking_does_not_change_the_weighted_selection(
        self, nvfp4_backend, candidate_chunk_size, block_chunk_size
    ):
        _, device = nvfp4_backend
        weight = _random_weight(6, 64, device, seed=41)
        global_scale = _global_scale(weight)
        generator = torch.Generator().manual_seed(13)
        moments = torch.rand(64, generator=generator).to(device=device, dtype=torch.float32)

        reference = weight_mse.select_nvfp4_block_scales_local_hessian(weight, global_scale, moments)
        chunked = weight_mse.select_nvfp4_block_scales_local_hessian(
            weight, global_scale, moments, candidate_chunk_size, block_chunk_size
        )
        assert torch.equal(chunked.view(torch.uint8), reference.view(torch.uint8))

    @pytest.mark.unit
    def test_the_weighted_repack_does_not_mutate_its_inputs(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=43)
        global_scale = _global_scale(weight)
        template = _make_template(cls, weight, global_scale)
        generator = torch.Generator().manual_seed(17)
        moments = torch.rand(64, generator=generator).to(device=device, dtype=torch.float32)
        before = {
            "weight": weight.clone(),
            "moments": moments.clone(),
            "qdata": _attr(template, _QDATA_ATTRS).view(torch.uint8).clone(),
            "scale": _attr(template, _SCALE_ATTRS).view(torch.uint8).clone(),
        }

        weight_mse.repack_nvfp4_weight_local_hessian(weight, template, moments)

        assert torch.equal(weight, before["weight"])
        assert torch.equal(moments, before["moments"])
        assert torch.equal(_attr(template, _QDATA_ATTRS).view(torch.uint8), before["qdata"])
        assert torch.equal(_attr(template, _SCALE_ATTRS).view(torch.uint8), before["scale"])


class TestLocalHessianValidation:
    """Fail-closed validation of the moment vector, proved without torchao where possible."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "moments,error,message",
        [
            ([0.0] * 64, TypeError, "must be a torch.Tensor"),
            (torch.ones(8, 8), ValueError, "rank 1"),
            (torch.ones(64, dtype=torch.int64), ValueError, "must be a float tensor"),
            (torch.ones(32), ValueError, "input channels"),
            (torch.full((64,), float("nan")), ValueError, "non-finite"),
            (torch.full((64,), -1.0), ValueError, "negative"),
            (torch.zeros(64), ValueError, "finite positive mean"),
        ],
    )
    def test_invalid_moments_fail_closed(self, monkeypatch, moments, error, message):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(error, match=message):
            weight_mse.repack_nvfp4_weight_local_hessian(_fake_weight(), _fake_template(), moments)

    @pytest.mark.unit
    def test_a_moment_vector_on_another_device_is_rejected(self):
        """The vector is never moved for the caller: a host vector for a device weight is a caller mistake."""
        _skip_without_e4m3()
        with pytest.raises(ValueError, match="but the weight is on"):
            weight_mse.damped_second_moments(torch.ones(8), 8, torch.device("meta"))

    @pytest.mark.unit
    @pytest.mark.parametrize("candidate_chunk_size,block_chunk_size", [(0, 8), (8, 0), (True, 8), (8, 1.5)])
    def test_invalid_chunk_sizes_are_rejected_before_the_moments_are_read(
        self, monkeypatch, candidate_chunk_size, block_chunk_size
    ):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(ValueError, match="must be a positive int"):
            weight_mse.repack_nvfp4_weight_local_hessian(
                _fake_weight(), _fake_template(), torch.ones(64), candidate_chunk_size, block_chunk_size
            )

    @pytest.mark.unit
    def test_damping_is_the_documented_constant_and_the_damped_vector_is_not_rescaled(self):
        _skip_without_e4m3()
        assert weight_mse.NVFP4_HESSIAN_DAMPING == 0.01
        moments = torch.tensor([0.0, 1.0, 3.0, 4.0])
        damped = weight_mse.damped_second_moments(moments)

        # Exactly h + 0.01 * mean(h) and nothing else: mean([0, 1, 3, 4]) is 2.0, so the floor is 0.02.
        assert damped.tolist() == pytest.approx([0.02, 1.02, 3.02, 4.02])
        assert torch.equal(damped, moments + weight_mse.NVFP4_HESSIAN_DAMPING * moments.mean())
        # A never-observed channel keeps a positive floor rather than dropping out of the objective entirely.
        assert float(damped[0]) > 0.0
        # Only the search rescales, and only by the positive constant that cannot move an argmin.
        rescaled = weight_mse._search_weights(damped)
        assert float(rescaled.max()) == 1.0
        assert rescaled.tolist() == pytest.approx([value / 4.02 for value in (0.02, 1.02, 3.02, 4.02)])


@pytest.mark.usefixtures("nvfp4_backend")
class TestFourOverSixSelection:
    """The two-candidate ModelOpt selection, checked against arithmetic written out independently here."""

    @pytest.mark.unit
    def test_m6_wins_when_mapping_the_amax_onto_six_is_exact(self, nvfp4_backend):
        """With a unit global scale, a block of exact multiples of 1.5 up to 6 is reconstructed exactly by M=6.

        The M=4 candidate stores the same block with scale 1.5, which cannot represent the 0.5 element (0.5 / 1.5
        is not on the E2M1 grid), so M=6 is strictly better and must be selected.
        """
        _, device = nvfp4_backend
        weight = torch.tensor([[6.0, 3.0, 1.5, 0.5] * 4], dtype=torch.float32, device=device)
        global_scale = torch.ones((), device=device)

        selection = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)

        assert selection.magnitude_index.tolist() == [[0]]
        assert (selection.m6_block_count, selection.m4_block_count, selection.block_count) == (1, 0, 1)
        assert selection.scale_fp8.float().tolist() == [[1.0]]
        assert _reference_four_over_six(weight, global_scale) == ([[0]], [[1.0]])

    @pytest.mark.unit
    def test_m4_wins_when_mapping_the_amax_onto_four_is_exact(self, nvfp4_backend):
        """An amax of 4 makes the M=4 scale exactly 1.0, while M=6 rounds 4/6 to 0.6875 and clips the maximum."""
        _, device = nvfp4_backend
        weight = torch.tensor([[4.0, 2.0, 1.5, 0.5] * 4], dtype=torch.float32, device=device)
        global_scale = torch.ones((), device=device)

        selection = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)

        assert selection.magnitude_index.tolist() == [[1]]
        assert (selection.m6_block_count, selection.m4_block_count, selection.block_count) == (0, 1, 1)
        assert selection.scale_fp8.float().tolist() == [[1.0]]
        assert _reference_four_over_six(weight, global_scale) == ([[1]], [[1.0]])
        # M=4 really is the exact representation here, and M=6 really is not.
        assert float(weight_mse.nvfp4_blockwise_mse(weight, selection.scale_fp8, global_scale)) == 0.0

    @pytest.mark.unit
    def test_an_exact_tie_keeps_m6(self, nvfp4_backend):
        """A single 6.0 is represented exactly by both candidates, so the first one in order must win."""
        _, device = nvfp4_backend
        weight = torch.tensor([[6.0] + [0.0] * 15], dtype=torch.float32, device=device)
        global_scale = torch.ones((), device=device)

        selection = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)

        # Both candidates reconstruct the block exactly: 6 * 1.0 and 4 * 1.5.
        assert float(weight_mse.nvfp4_blockwise_mse(weight, selection.scale_fp8, global_scale)) == 0.0
        other = torch.full_like(selection.scale_fp8.float(), 1.5).to(torch.float8_e4m3fn)
        assert float(weight_mse.nvfp4_blockwise_mse(weight, other, global_scale)) == 0.0
        assert selection.magnitude_index.tolist() == [[0]]
        assert selection.scale_fp8.float().tolist() == [[1.0]]

    @pytest.mark.unit
    def test_an_all_zero_block_takes_the_reference_zero_scale_substitution(self, nvfp4_backend):
        """A zero unnormalized scale becomes 1.0 *before* normalization and clamping, exactly as the export does."""
        _, device = nvfp4_backend
        weight = torch.zeros(1, 32, dtype=torch.float32, device=device)

        unit = weight_mse.select_nvfp4_block_scales_four_over_six(weight, torch.ones((), device=device))
        assert unit.scale_fp8.float().tolist() == [[1.0, 1.0]]
        assert unit.magnitude_index.tolist() == [[0, 0]]

        # 1.0 / a tiny global scale overflows the clamp, so the stored scale is the E4M3 maximum, not zero.
        tiny = weight_mse.select_nvfp4_block_scales_four_over_six(weight, torch.full((), 1e-6, device=device))
        assert tiny.scale_fp8.float().tolist() == [[weight_mse.FP8_E4M3_MAX] * 2]

    @pytest.mark.unit
    def test_the_stored_scale_is_clamped_into_the_reference_range(self, nvfp4_backend):
        """Both ends of ``[2 ** -9, 448]`` are enforced before the E4M3 rounding, and the ends are representable."""
        _, device = nvfp4_backend
        assert weight_mse.NVFP4_FOUR_OVER_SIX_SCALE_MIN == 2.0**-9
        assert weight_mse.NVFP4_FOUR_OVER_SIX_SCALE_MAX == weight_mse.FP8_E4M3_MAX == 448.0

        tiny_weight = torch.full((1, 16), 1e-8, dtype=torch.float32, device=device)
        underflow = weight_mse.select_nvfp4_block_scales_four_over_six(
            tiny_weight, torch.full((), 448.0, device=device)
        )
        assert underflow.scale_fp8.float().tolist() == [[weight_mse.NVFP4_FOUR_OVER_SIX_SCALE_MIN]]

        large_weight = torch.full((1, 16), 1e4, dtype=torch.float32, device=device)
        overflow = weight_mse.select_nvfp4_block_scales_four_over_six(
            large_weight, torch.full((), 1e-4, device=device)
        )
        assert overflow.scale_fp8.float().tolist() == [[weight_mse.FP8_E4M3_MAX]]

    @pytest.mark.unit
    def test_every_selected_scale_is_the_better_of_its_own_two_candidates(self, nvfp4_backend):
        """On real weights every block still stores one of its two reference candidates, and the better one.

        The candidate *values* are pinned against the reference arithmetic; the *decision* is checked against
        torchao's own per-block reconstruction error, so this cannot be satisfied by a selection that merely agrees
        with the reference's rounding of the loss.
        """
        _, device = nvfp4_backend
        weight = _random_weight(6, 64, device, dtype=torch.float32, seed=61)
        global_scale = _four_over_six_scale(_global_scale(weight))

        selection = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)
        candidates = _reference_four_over_six_candidates(weight, global_scale)
        index = selection.magnitude_index.cpu().tolist()
        selected = selection.scale_fp8.float().cpu().tolist()

        rows, blocks = len(index), len(index[0])
        assert [len(row) for row in index] == [weight.shape[1] // 16] * rows
        for row in range(rows):
            for block in range(blocks):
                assert selected[row][block] == candidates[row][block][index[row][block]]

        errors = [
            weight_mse.nvfp4_blockwise_mse(weight, _scales_like(selection.scale_fp8, candidates, side), global_scale)
            for side in (0, 1)
        ]
        for row in range(rows):
            for block in range(blocks):
                m6, m4 = float(errors[0][row][block]), float(errors[1][row][block])
                # M=4 is taken only when it is strictly better, so an exact tie keeps M=6.
                assert index[row][block] == (1 if m4 < m6 else 0)

        assert selection.m6_block_count + selection.m4_block_count == selection.block_count == weight.numel() // 16
        assert selection.m4_block_count == sum(sum(row) for row in index)
        # Nothing outside E4M3's positive finite range is ever stored.
        stored = selection.scale_fp8.float()
        assert bool((stored > 0).all()) and bool((stored <= weight_mse.FP8_E4M3_MAX).all())

    @pytest.mark.unit
    @pytest.mark.parametrize("block_chunk_size", [1, 3, 10**6])
    def test_block_chunking_does_not_change_the_selection(self, nvfp4_backend, block_chunk_size):
        _, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=67)
        global_scale = _four_over_six_scale(_global_scale(weight))

        reference = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)
        chunked = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale, block_chunk_size)

        assert torch.equal(chunked.scale_fp8.view(torch.uint8), reference.scale_fp8.view(torch.uint8))
        assert torch.equal(chunked.magnitude_index, reference.magnitude_index)
        assert (chunked.m6_block_count, chunked.m4_block_count) == (
            reference.m6_block_count,
            reference.m4_block_count,
        )


@pytest.mark.usefixtures("nvfp4_backend")
class TestFourOverSixRepack:
    """End-to-end Four-Over-Six repacking against a real converted template."""

    @pytest.mark.unit
    def test_the_weight_global_scale_is_the_template_scale_times_448_over_256(self, nvfp4_backend):
        """The one attribute this method replaces, and it replaces it with exactly the reference normalization."""
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=71)
        template_scale = _global_scale(weight)
        template = _make_template(cls, weight, template_scale)

        result = weight_mse.repack_nvfp4_weight_four_over_six(weight, template)

        produced = _attr(result.weight, _PER_TENSOR_ATTRS)
        expected = _attr(template, _PER_TENSOR_ATTRS) * (448.0 / 256.0)
        assert torch.equal(produced, expected)
        assert produced.shape == expected.shape
        assert produced.dtype == _attr(template, _PER_TENSOR_ATTRS).dtype
        assert produced.device == _attr(template, _PER_TENSOR_ATTRS).device
        # A new tensor: the template's own scale is neither handed on nor written into.
        assert produced is not _attr(template, _PER_TENSOR_ATTRS)
        assert produced.data_ptr() != _attr(template, _PER_TENSOR_ATTRS).data_ptr()
        assert torch.equal(_attr(template, _PER_TENSOR_ATTRS), template_scale)
        # ... while the exhaustive searches keep handing the template's very same scale object on.
        assert _attr(weight_mse.repack_nvfp4_weight_mse(weight, template), _PER_TENSOR_ATTRS) is _attr(
            template, _PER_TENSOR_ATTRS
        )
        moments = torch.ones(int(weight.shape[1]), device=device)
        assert _attr(
            weight_mse.repack_nvfp4_weight_local_hessian(weight, template, moments), _PER_TENSOR_ATTRS
        ) is _attr(template, _PER_TENSOR_ATTRS)

    @pytest.mark.unit
    def test_every_other_wrapper_attribute_and_layout_is_preserved(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=73)
        template = _make_template(cls, weight, _global_scale(weight))

        result = weight_mse.repack_nvfp4_weight_four_over_six(weight, template).weight

        assert type(result) is type(template)
        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert result.device == template.device
        # The activation global scale is untouched: Four-Over-Six is a weight-side technique only.
        assert _attr(result, _ACT_PER_TENSOR_ATTRS) is _attr(template, _ACT_PER_TENSOR_ATTRS)
        assert _attr(result, _BLOCK_SIZE_ATTRS) == _attr(template, _BLOCK_SIZE_ATTRS)
        assert _attr(result, _ORIG_DTYPE_ATTRS) == _attr(template, _ORIG_DTYPE_ATTRS)
        assert bool(_attr(result, _SWIZZLE_ATTRS)) is bool(_attr(template, _SWIZZLE_ATTRS)) is True
        assert _attr(result, _TRITON_ATTRS) == _attr(template, _TRITON_ATTRS)
        assert _attr(result, _ACT_KWARGS_ATTRS) == _attr(template, _ACT_KWARGS_ATTRS)
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            got, expected = _attr(result, attrs), _attr(template, attrs)
            assert (got.shape, got.dtype, got.device) == (expected.shape, expected.dtype, expected.device)
            assert got.is_contiguous()

    @pytest.mark.unit
    def test_payload_and_swizzled_scales_match_the_selected_rounded_scales(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=79)
        template = _make_template(cls, weight, _global_scale(weight))

        result = weight_mse.repack_nvfp4_weight_four_over_six(weight, template)

        backend = weight_mse._resolve_torchao()
        global_scale = weight_mse.four_over_six_global_scale(_attr(template, _PER_TENSOR_ATTRS))
        selection = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)
        expected = weight_mse._pack_qdata(
            weight.float(), selection.scale_fp8, global_scale.float().reshape(()), backend
        )
        assert torch.equal(_attr(result.weight, _QDATA_ATTRS).view(torch.uint8).reshape(-1), expected.reshape(-1))
        blocked = backend.to_blocked(selection.scale_fp8).flatten()
        assert torch.equal(_attr(result.weight, _SCALE_ATTRS).view(torch.uint8).reshape(-1), blocked.view(torch.uint8))

        # The reported counts are the selection's own, so a caller never has to re-quantize to report them.
        assert result.block_count == weight.numel() // 16 == selection.block_count
        assert result.m6_block_count == selection.m6_block_count
        assert result.m4_block_count == selection.m4_block_count
        assert result.m6_block_count + result.m4_block_count == result.block_count

    @pytest.mark.unit
    def test_dequantization_follows_the_selected_scales_and_stays_finite(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(16, 64, device, seed=83)
        template = _make_template(cls, weight, _global_scale(weight))

        result = weight_mse.repack_nvfp4_weight_four_over_six(weight, template)

        dequantized = _dequantize(result.weight)
        reference = weight.float()
        assert torch.isfinite(dequantized).all()
        assert dequantized.shape == reference.shape

        global_scale = weight_mse.four_over_six_global_scale(_attr(template, _PER_TENSOR_ATTRS))
        selection = weight_mse.select_nvfp4_block_scales_four_over_six(weight, global_scale)
        expected_mse = float(weight_mse.nvfp4_blockwise_mse(weight, selection.scale_fp8, global_scale).mean())
        assert float(((dequantized - reference) ** 2).mean()) <= expected_mse * 1.05 + 1e-12

    @pytest.mark.unit
    def test_does_not_mutate_the_weight_or_any_template_buffer(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=89)
        global_scale = _global_scale(weight)
        template = _make_template(cls, weight, global_scale)
        before = {
            "weight": weight.clone(),
            "qdata": _attr(template, _QDATA_ATTRS).view(torch.uint8).clone(),
            "scale": _attr(template, _SCALE_ATTRS).view(torch.uint8).clone(),
            "global": global_scale.clone(),
        }

        result = weight_mse.repack_nvfp4_weight_four_over_six(weight, template)

        assert torch.equal(weight, before["weight"])
        assert torch.equal(_attr(template, _QDATA_ATTRS).view(torch.uint8), before["qdata"])
        assert torch.equal(_attr(template, _SCALE_ATTRS).view(torch.uint8), before["scale"])
        assert torch.equal(global_scale, before["global"])
        assert _attr(result.weight, _QDATA_ATTRS).data_ptr() != _attr(template, _QDATA_ATTRS).data_ptr()
        assert _attr(result.weight, _SCALE_ATTRS).data_ptr() != _attr(template, _SCALE_ATTRS).data_ptr()

    @pytest.mark.unit
    @pytest.mark.parametrize("block_chunk_size", [1, 5])
    def test_chunked_repack_is_byte_identical(self, nvfp4_backend, block_chunk_size):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=97)
        template = _make_template(cls, weight, _global_scale(weight))

        reference = weight_mse.repack_nvfp4_weight_four_over_six(weight, template)
        chunked = weight_mse.repack_nvfp4_weight_four_over_six(weight, template, block_chunk_size)

        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            assert torch.equal(
                _attr(chunked.weight, attrs).view(torch.uint8), _attr(reference.weight, attrs).view(torch.uint8)
            )
        assert chunked[1:] == reference[1:]


class TestFourOverSixValidation:
    """Fail-closed validation, proved against the same stand-in backend the searches use."""

    @pytest.mark.unit
    def test_the_global_rescaling_is_exact_and_validated(self):
        _skip_without_e4m3()
        assert weight_mse.NVFP4_FOUR_OVER_SIX_FP8_MAX == 256.0
        assert weight_mse.NVFP4_FOUR_OVER_SIX_MAGNITUDES == (6, 4)

        scale = torch.tensor(0.25)
        rescaled = weight_mse.four_over_six_global_scale(scale)
        assert float(rescaled) == 0.25 * (448.0 / 256.0)
        assert rescaled.shape == scale.shape and rescaled.dtype == scale.dtype
        assert torch.equal(scale, torch.tensor(0.25))

        with pytest.raises(ValueError, match="positive and finite"):
            weight_mse.four_over_six_global_scale(torch.tensor(0.0))
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.four_over_six_global_scale(0.25)
        # A rescaling that overflows the scale's own dtype is refused rather than stored as an infinity.
        with pytest.raises(ValueError, match="positive and finite"):
            weight_mse.four_over_six_global_scale(torch.tensor(60000.0, dtype=torch.float16))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "overrides,error,message",
        [
            ({"block_size": 32}, ValueError, "block size 16"),
            ({"is_swizzled_scales": False}, ValueError, "swizzled"),
            ({"orig_dtype": torch.float16}, ValueError, "orig_dtype"),
            ({"per_tensor_scale": None}, ValueError, "per_tensor_scale is missing"),
            ({"per_tensor_scale": torch.tensor(0.0)}, ValueError, "positive and finite"),
            ({"qdata": torch.zeros(8, 8, dtype=torch.uint8)}, ValueError, "cannot be preserved safely"),
            ({"drop": "act_quant_kwargs"}, ValueError, "does not expose"),
        ],
    )
    def test_incompatible_templates_fail_closed(self, monkeypatch, overrides, error, message):
        _skip_without_e4m3()
        template = _fake_template(**overrides)
        _install_fake_backend(monkeypatch)
        with pytest.raises(error, match=message):
            weight_mse.repack_nvfp4_weight_four_over_six(_fake_weight(), template)

    @pytest.mark.unit
    def test_a_non_nvfp4_template_and_an_invalid_weight_fail_closed(self, monkeypatch):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(TypeError, match="must be a torchao"):
            weight_mse.repack_nvfp4_weight_four_over_six(_fake_weight(), torch.zeros(8, 64))
        with pytest.raises(ValueError, match="but the template covers"):
            weight_mse.repack_nvfp4_weight_four_over_six(torch.zeros(8, 32, dtype=torch.bfloat16), _fake_template())
        non_finite = _fake_weight()
        non_finite[0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.repack_nvfp4_weight_four_over_six(non_finite, _fake_template())

    @pytest.mark.unit
    @pytest.mark.parametrize("block_chunk_size", [0, -1, True, 1.5, None])
    def test_invalid_block_chunk_sizes_are_rejected(self, monkeypatch, block_chunk_size):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        with pytest.raises(ValueError, match="block_chunk_size must be a positive int"):
            weight_mse.repack_nvfp4_weight_four_over_six(_fake_weight(), _fake_template(), block_chunk_size)


@pytest.mark.unit
def test_awq_clip_ratio_list_is_the_reference_insertion_order():
    """The eleven ratios, their order and the unclipped code are the algorithm's identity, not tunables."""
    assert weight_mse.NVFP4_AWQ_CLIP_RATIOS == (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
    assert weight_mse.NVFP4_AWQ_CLIP_RATIO_COUNT == 11
    assert weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE == 10
    assert weight_mse.NVFP4_AWQ_CLIP_RATIOS[0] == 0.5
    assert weight_mse.NVFP4_AWQ_CLIP_RATIOS[weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE] == 1.0
    ratios = weight_mse.NVFP4_AWQ_CLIP_RATIOS
    assert all(later > earlier for earlier, later in zip(ratios, ratios[1:]))
    assert weight_mse.NVFP4_AWQ_CLIP_SCALE_MIN == 2.0**-9
    assert weight_mse.NVFP4_AWQ_CLIP_SCALE_MAX == weight_mse.FP8_E4M3_MAX


@pytest.mark.unit
def test_awq_clip_repack_fails_closed_without_torchao(monkeypatch):
    """An absent torchao raises an actionable error only when the AWQ-clip packer is actually called."""
    _skip_without_e4m3()

    def _missing(module_path):
        raise weight_mse.TorchAOUnavailableError(f"'{module_path}' could not be imported.")

    monkeypatch.setattr(weight_mse, "_torchao_backend", None)
    monkeypatch.setattr(weight_mse, "_import_module", _missing)
    with pytest.raises(ValueError, match="could not be imported"):
        weight_mse.repack_nvfp4_weight_awq_clip(
            torch.zeros(1, 16, dtype=torch.bfloat16), object(), torch.zeros(1, 1, dtype=torch.uint8)
        )


@pytest.mark.usefixtures("nvfp4_backend")
class TestAWQClipCandidateScales:
    """The eleven candidate scales, checked against a scalar reference that borrows nothing from the packer."""

    @pytest.mark.unit
    def test_candidates_match_a_scalar_reference(self, nvfp4_backend):
        _, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=31)
        global_scale = _global_scale(weight)

        candidates = weight_mse.nvfp4_awq_clip_candidate_scales(weight, global_scale)
        expected = _reference_awq_clip_candidates(weight, global_scale)

        assert tuple(candidates.shape) == (4, 4, 11)
        assert candidates.dtype == torch.float32
        assert candidates.detach().cpu().tolist() == expected
        # Every candidate is an exact E4M3 encoding, or the runtime could not store the one that was selected.
        assert torch.equal(candidates.to(torch.float8_e4m3fn).to(torch.float32), candidates)

    @pytest.mark.unit
    def test_the_unclipped_candidate_is_the_ordinary_amax_scale(self, nvfp4_backend):
        """Ratio 1.00 is the ordinary rule, which is what makes code 10 reproduce the plain conversion."""
        _, device = nvfp4_backend
        weight = _random_weight(6, 96, device, seed=32)
        global_scale = _global_scale(weight)

        candidates = weight_mse.nvfp4_awq_clip_candidate_scales(weight, global_scale)
        amax_based = _amax_based_scales(weight, global_scale).float()

        assert torch.equal(candidates[..., weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE], amax_based)
        # Clipping never raises a block's stored scale, and clamping keeps every candidate representable.
        assert bool((candidates <= amax_based[..., None] + 1e-12).all())
        assert bool((candidates >= weight_mse.NVFP4_AWQ_CLIP_SCALE_MIN).all())

    @pytest.mark.unit
    def test_an_all_zero_block_takes_the_scale_floor_under_every_ratio(self, nvfp4_backend):
        _, device = nvfp4_backend
        weight = torch.zeros(2, 32, dtype=torch.float32, device=device)

        candidates = weight_mse.nvfp4_awq_clip_candidate_scales(weight, torch.ones((), device=device))

        assert bool((candidates == weight_mse.NVFP4_AWQ_CLIP_SCALE_MIN).all())


@pytest.mark.usefixtures("nvfp4_backend")
class TestAWQClipSelection:
    """The offline ratio search: its objective, its tie rule and its independence from the chunk sizes."""

    @pytest.mark.unit
    def test_ties_keep_the_earliest_ratio(self, nvfp4_backend):
        """Every candidate reconstructs an all-zero block identically, so the earliest ratio must win."""
        _, device = nvfp4_backend
        weight = torch.zeros(2, 32, dtype=torch.float32, device=device)
        scale = torch.ones((), device=device)
        rows = torch.randn(8, 32, generator=torch.Generator().manual_seed(33)).to(device=device)

        selection = weight_mse.select_nvfp4_ratio_codes_awq_clip(
            weight, scale, [rows], _template_values(weight, scale)
        )

        assert tuple(selection.ratio_codes.shape) == (2, 2)
        assert selection.ratio_codes.dtype == torch.uint8
        assert bool((selection.ratio_codes == 0).all())
        assert selection.code_counts == (4,) + (0,) * 10
        assert selection.block_count == 4
        assert selection.selected_objective == 0.0
        assert selection.unclipped_objective == 0.0

    @pytest.mark.unit
    def test_selection_matches_a_scalar_reference(self, nvfp4_backend):
        """Codes and both objectives are checked against a plain-Python reference of the specified arithmetic.

        The reference scores all eleven candidates of every block independently, emulating the FP32 dot product
        and the FP64 reduction the objective is defined with, so a candidate indexed off by one -- or a ratio
        applied to the wrong block -- moves at least one code or one reported objective.
        """
        _, device = nvfp4_backend
        weight = _random_weight(4, 32, device, seed=34)
        global_scale = _global_scale(weight)
        groups = [
            torch.randn(5, 32, generator=torch.Generator().manual_seed(35)).to(device=device),
            torch.randn(3, 32, generator=torch.Generator().manual_seed(36)).to(device=device),
        ]

        template = _template_values(weight, global_scale)
        selection = weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, groups, template)
        losses = _reference_awq_clip_losses(weight, global_scale, groups, template)

        assert [len(block) for row in losses for block in row] == [weight_mse.NVFP4_AWQ_CLIP_RATIO_COUNT] * 8
        expected_codes = [[_earliest_argmin(block) for block in row] for row in losses]
        assert selection.ratio_codes.detach().cpu().tolist() == expected_codes
        # Every block's winner is the minimum over all eleven reference losses, not merely better than its neighbours.
        for row, codes in enumerate(expected_codes):
            for block, code in enumerate(codes):
                assert losses[row][block][code] == min(losses[row][block])
        histogram = [0] * weight_mse.NVFP4_AWQ_CLIP_RATIO_COUNT
        for codes in expected_codes:
            for code in codes:
                histogram[code] += 1
        assert list(selection.code_counts) == histogram

        selected = [
            losses[row][block][code] for row, codes in enumerate(expected_codes) for block, code in enumerate(codes)
        ]
        unclipped = [block[weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE] for row in losses for block in row]
        assert selection.selected_objective == pytest.approx(sum(selected) / len(selected), rel=1e-6)
        assert selection.unclipped_objective == pytest.approx(sum(unclipped) / len(unclipped), rel=1e-6)
        assert selection.selected_objective <= selection.unclipped_objective

    @pytest.mark.unit
    def test_clipping_strictly_improves_the_objective_on_a_constructed_block(self, nvfp4_backend):
        """A block whose amax rounds its scale up is fit better by clipping, and AWQ must take that ratio.

        With a unit global scale the unclipped rule stores 3.3 / 6 = 0.55, which rounds to the E4M3 value 0.5625
        and leaves every one of the fifteen 1.5s between two representable points. Clipping to 0.9 of the amax
        stores 0.5 instead, which represents 1.5 exactly and pays only for the single clipped maximum -- and the
        activations weight those fifteen errors fifteen times as heavily as the one.
        """
        _, device = nvfp4_backend
        weight = torch.tensor([[1.5] * 15 + [3.3]], dtype=torch.float32, device=device)
        scale = torch.ones((), device=device)
        rows = torch.randn(32, 16, generator=torch.Generator().manual_seed(37)).to(device=device)

        selection = weight_mse.select_nvfp4_ratio_codes_awq_clip(
            weight, scale, [rows], _template_values(weight, scale)
        )

        assert int(selection.ratio_codes[0, 0]) != weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE
        assert selection.selected_objective < selection.unclipped_objective

    @pytest.mark.unit
    def test_the_full_covariance_choice_differs_from_the_diagonal_one(self, nvfp4_backend):
        """Correlated activations make the within-block cross terms decide, which the diagonal objective drops."""
        _, device = nvfp4_backend
        weight = _random_weight(16, 64, device, seed=38)
        global_scale = _global_scale(weight)
        groups = [_correlated_rows(24, 64, device, seed=39), _correlated_rows(16, 64, device, seed=40)]

        template = _template_values(weight, global_scale)
        selection = weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, groups, template)
        diagonal = _diagonal_ratio_codes(weight, global_scale, groups, template)

        assert tuple(selection.ratio_codes.shape) == tuple(diagonal.shape)
        assert not torch.equal(selection.ratio_codes.cpu(), diagonal.cpu())

    @pytest.mark.unit
    @pytest.mark.parametrize("row_chunk_size,block_chunk_size", [(1, 1), (3, 2), (10**6, 10**6)])
    def test_chunking_and_group_order_do_not_change_the_codes(self, nvfp4_backend, row_chunk_size, block_chunk_size):
        _, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=41)
        global_scale = _global_scale(weight)
        first = torch.randn(6, 64, generator=torch.Generator().manual_seed(42)).to(device=device)
        second = torch.randn(4, 64, generator=torch.Generator().manual_seed(43)).to(device=device)

        template = _template_values(weight, global_scale)
        reference = weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [first, second], template)
        chunked = weight_mse.select_nvfp4_ratio_codes_awq_clip(
            weight, global_scale, [first, second], template, row_chunk_size, block_chunk_size
        )
        swapped = weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [second, first], template)

        assert torch.equal(chunked.ratio_codes, reference.ratio_codes)
        assert chunked.code_counts == reference.code_counts
        assert chunked.selected_objective == reference.selected_objective
        assert torch.equal(swapped.ratio_codes, reference.ratio_codes)

    @pytest.mark.unit
    def test_group_validation_is_fail_closed(self, nvfp4_backend):
        _, device = nvfp4_backend
        weight = _random_weight(2, 32, device, seed=44)
        global_scale = _global_scale(weight)
        rows = torch.zeros(4, 32, dtype=torch.float32, device=device)
        template = _template_values(weight, global_scale)

        with pytest.raises(ValueError, match="non-empty sequence"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [], template)
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [object()], template)
        with pytest.raises(ValueError, match="input channel"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows[:, :16]], template)
        with pytest.raises(ValueError, match="non-empty rank-2"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows[:0]], template)
        with pytest.raises(ValueError, match="never raw low-precision"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows.to(torch.bfloat16)], template)
        nonfinite = rows.clone()
        nonfinite[0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [nonfinite], template)
        with pytest.raises(ValueError, match="row_chunk_size must be a positive int"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], template, 0)
        with pytest.raises(ValueError, match="block_chunk_size must be a positive int"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], template, 1, -1)

    @pytest.mark.unit
    def test_the_unclipped_candidate_must_describe_this_very_weight(self, nvfp4_backend):
        """Scoring anything but this weight's own ordinary template would rank a block nobody deploys."""
        _, device = nvfp4_backend
        weight = _random_weight(2, 32, device, seed=54)
        global_scale = _global_scale(weight)
        rows = torch.zeros(4, 32, dtype=torch.float32, device=device)
        template = _template_values(weight, global_scale)

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], object())
        with pytest.raises(ValueError, match="must be the ordinary NVFP4 template's readback"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], template[:, :16])
        with pytest.raises(ValueError, match="must be a float tensor"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], template.to(torch.int32))
        nonfinite = template.clone()
        nonfinite[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], nonfinite)
        with pytest.raises(ValueError, match="template_arithmetic must be one of"):
            weight_mse.nvfp4_awq_clip_template_reconstruction(weight, global_scale, "guess")

    @pytest.mark.unit
    def test_the_unclipped_candidate_is_the_templates_own_decoded_bytes(self, nvfp4_backend):
        """The eleventh candidate is the template, not the ratio-1.00 formula, so its loss is the template's."""
        cls, device = nvfp4_backend
        weight = _random_weight(4, 32, device, seed=55).clone()
        # A block the E4M3 floor decides, where the formula and the ordinary conversion disagree by construction.
        weight[1, 16:32] = 0.0
        global_scale = _global_scale(weight)
        rows = torch.randn(6, 32, generator=torch.Generator().manual_seed(56)).to(device=device)
        template = _template_values(weight, global_scale)

        reference_template = _template_in_mode(
            cls, weight, global_scale, weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE
        )
        assert torch.equal(template, _decoded_template_values(reference_template, global_scale))
        losses = _reference_awq_clip_losses(weight, global_scale, [rows], template)
        selection = weight_mse.select_nvfp4_ratio_codes_awq_clip(weight, global_scale, [rows], template)

        unclipped = [block[weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE] for row in losses for block in row]
        assert selection.unclipped_objective == pytest.approx(sum(unclipped) / len(unclipped), rel=1e-6)


@pytest.mark.usefixtures("nvfp4_backend")
@pytest.mark.parametrize("arithmetic", list(weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETICS))
class TestAWQClipTemplateReconstruction:
    """The unclipped candidate: the FP32 decode of the exact bytes an ordinary template stores.

    Both constructions are covered because the unclipped code means "leave this block as *that* conversion wrote
    it", and one weight carries a normal, an all-zero and a far-below-amax block, which are the three cases where
    the stored E4M3 scale is decided by the conversion's own floor rather than by the block's values.
    """

    @pytest.mark.unit
    def test_decodes_the_stored_bytes_with_fp32_scale_arithmetic(self, nvfp4_backend, arithmetic):
        """Every candidate delta is FP32, including the eleventh; nothing here is cast through ``orig_dtype``."""
        cls, device = nvfp4_backend
        weight = _awq_clip_mixed_magnitude_weight(device)
        global_scale = _global_scale(weight)
        template = _template_in_mode(cls, weight, global_scale, arithmetic)

        values = weight_mse.nvfp4_awq_clip_template_reconstruction(weight, global_scale, arithmetic)

        assert values.dtype == torch.float32
        assert values.device == weight.device
        assert values.is_contiguous()
        assert tuple(values.shape) == tuple(weight.shape)
        assert torch.equal(values, _decoded_template_values(template, global_scale))

        # Independently of that decode: every value is an E2M1 magnitude times its block's *FP32* multiplier
        # ``global_scale * block_scale``. A multiplier rounded through BF16 is up to 2 ** -9 relative away from
        # that grid, which is four orders of magnitude looser than this tolerance.
        payload, multiplier = _template_payload_and_multiplier(template, global_scale)
        usable = multiplier.expand_as(payload) > 0.0
        assert bool(usable.any())
        magnitudes = (values.reshape(payload.shape).abs() / multiplier)[usable]
        grid = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=values.device)
        nearest = grid[(magnitudes[..., None] - grid).abs().argmin(dim=-1)]
        assert torch.allclose(magnitudes, nearest, rtol=1e-6, atol=0.0)

    @pytest.mark.unit
    def test_a_scale_rounded_through_bf16_would_not_pass(self, nvfp4_backend, arithmetic):
        """The discriminator: the same decode with a BF16-rounded block multiplier must differ from the result.

        Pinned torchao's own ``get_hp_scales`` casts the stored E4M3 scale through the wrapper's ``orig_dtype``,
        which for these weights is BF16, so a wrapper readback is not the FP32 candidate this method is specified
        with. This test fails if that rounding ever reaches the unclipped candidate.
        """
        cls, device = nvfp4_backend
        weight = _awq_clip_mixed_magnitude_weight(device)
        global_scale = _global_scale(weight)
        template = _template_in_mode(cls, weight, global_scale, arithmetic)

        values = weight_mse.nvfp4_awq_clip_template_reconstruction(weight, global_scale, arithmetic)
        rounded = _decoded_template_values(template, global_scale, scale_dtype=torch.bfloat16)

        assert tuple(rounded.shape) == tuple(values.shape)
        assert not torch.equal(values, rounded)
        # The two differ by exactly one BF16 rounding of the block multiplier and by nothing structural, which is
        # what makes this a precision discriminator rather than a comparison of two different reconstructions.
        assert torch.allclose(values, rounded, rtol=2.0**-8, atol=0.0)


@pytest.mark.usefixtures("nvfp4_backend")
class TestAWQClipActivationQuantization:
    """The activation quantize/dequantize the offline objective scores must be the runtime's own."""

    @pytest.mark.unit
    def test_matches_a_scalar_reference_of_the_torchao_arithmetic(self, nvfp4_backend):
        _, device = nvfp4_backend
        rows = torch.randn(5, 32, generator=torch.Generator().manual_seed(45)).to(device=device)
        amax = float(rows.abs().max()) * 1.5

        quantized = weight_mse.nvfp4_awq_clip_activation_qdq(rows, amax)
        expected = _reference_activation_qdq(rows, amax)

        assert quantized.dtype == torch.float32
        assert tuple(quantized.shape) == (5, 32)
        flattened = [value for row in expected for value in row]
        assert quantized.detach().cpu().reshape(-1).tolist() == pytest.approx(flattened, rel=1e-6, abs=1e-9)
        # The reconstruction lives on the FP4 grid scaled by the two-level scales, not on the original rows.
        assert not torch.equal(quantized.cpu(), rows.float().cpu())

    @pytest.mark.unit
    def test_the_global_scale_is_the_calibrated_amax_over_448_times_6(self, nvfp4_backend):
        """A single block whose values sit exactly on the grid must come back unchanged."""
        _, device = nvfp4_backend
        amax = 6.0 * 448.0
        rows = torch.full((1, 16), 448.0, dtype=torch.float32, device=device)

        quantized = weight_mse.nvfp4_awq_clip_activation_qdq(rows, amax)

        # global scale = amax / (448 * 6) = 1.0, block scale = 448 / 6 rounded to E4M3, payload = 6.
        block_scale = float(torch.tensor(448.0 / 6.0).to(torch.float8_e4m3fn).to(torch.float32))
        assert quantized.detach().cpu().tolist() == [[pytest.approx(6.0 * block_scale)] * 16]

    @pytest.mark.unit
    def test_rejects_unusable_rows_and_maxima(self, nvfp4_backend):
        _, device = nvfp4_backend
        rows = torch.randn(4, 32, generator=torch.Generator().manual_seed(46)).to(device=device)

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.nvfp4_awq_clip_activation_qdq([1.0], 1.0)
        with pytest.raises(ValueError, match="non-empty rank-2"):
            weight_mse.nvfp4_awq_clip_activation_qdq(rows[:0], 1.0)
        with pytest.raises(ValueError, match="multiple of 16"):
            weight_mse.nvfp4_awq_clip_activation_qdq(rows[:, :20], 1.0)
        with pytest.raises(ValueError, match="must be a float tensor"):
            weight_mse.nvfp4_awq_clip_activation_qdq(rows.to(torch.int32), 1.0)
        for amax in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite and positive"):
                weight_mse.nvfp4_awq_clip_activation_qdq(rows, amax)
        with pytest.raises(ValueError, match="must be a number"):
            weight_mse.nvfp4_awq_clip_activation_qdq(rows, "1.0")
        nonfinite = rows.clone()
        nonfinite[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.nvfp4_awq_clip_activation_qdq(nonfinite, 1.0)

    @pytest.mark.unit
    def test_weight_global_scale_is_the_ordinary_torchao_scale(self, nvfp4_backend):
        _, device = nvfp4_backend
        weight = _random_weight(4, 32, device, seed=47)

        assert float(weight_mse.nvfp4_weight_global_scale(weight)) == float(_global_scale(weight))
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.nvfp4_weight_global_scale([1.0])
        with pytest.raises(ValueError, match="positive and finite"):
            weight_mse.nvfp4_weight_global_scale(torch.zeros(2, 16, device=device))


@pytest.mark.usefixtures("nvfp4_backend")
@pytest.mark.parametrize("arithmetic", list(weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETICS))
class TestAWQClipRepack:
    """Reconstructing stored codes against a real converted template, in both ordinary-template constructions.

    Every case runs once per construction because the unclipped code means "leave this block as *that* conversion
    wrote it", and the two conversions disagree: the non-Triton path floors a near-zero block scale at ``2 ** -6``
    where the accelerated kernel stores an E4M3 zero, and their payloads differ on ordinary blocks too. A repack
    that reproduced either by formula would pass in one mode and fail in the other, which is exactly what these
    parametrized cases are here to catch. The accelerated parameter skips only where that kernel does not run.
    """

    @pytest.mark.unit
    def test_code_ten_reproduces_the_ordinary_template_byte_for_byte(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=48)
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)
        codes = torch.full((8, 4), weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE, dtype=torch.uint8, device=device)

        result = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)

        assert result is not template
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            produced, expected = _attr(result, attrs), _attr(template, attrs)
            assert produced is not expected
            assert torch.equal(produced.view(torch.uint8), expected.view(torch.uint8))

    @pytest.mark.unit
    def test_code_ten_is_byte_identical_across_an_all_zero_block(self, nvfp4_backend, arithmetic):
        """The one code that must be a no-op stays a no-op where the E4M3 scale floor decides the scale.

        An all-zero 16-value block has amax ``0``, so its stored scale is whatever the conversion's own floor or
        epsilon produces rather than whatever the block's own values imply -- and the two supported conversions do
        not produce the same byte there. Code 10 *is* "leave it as the ordinary conversion wrote it", so the repack
        has to reproduce torchao's own conversion of that block in whichever mode built the template: a formula, a
        different floor or a clamp applied in another order shows up here as one differing scale byte while every
        other block still matches, which the random-weight case above cannot see.
        """
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=52).clone()
        weight[3, 16:32] = 0.0
        assert float(weight[3, 16:32].abs().max()) == 0.0
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)
        codes = torch.full((8, 4), weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE, dtype=torch.uint8, device=device)

        result = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)

        assert result is not template
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            produced, expected = _attr(result, attrs), _attr(template, attrs)
            assert torch.equal(produced.view(torch.uint8), expected.view(torch.uint8))
        # The template's own global scales are handed on as the very same objects, never re-derived.
        assert _attr(result, _PER_TENSOR_ATTRS) is _attr(template, _PER_TENSOR_ATTRS)
        assert _attr(result, _ACT_PER_TENSOR_ATTRS) is _attr(template, _ACT_PER_TENSOR_ATTRS)

    @pytest.mark.unit
    def test_code_ten_is_byte_identical_across_a_block_the_scale_floor_decides(self, nvfp4_backend, arithmetic):
        """A block whose amax is far below the tensor's puts the stored scale under the conversion's own floor.

        With the ordinary global scale, a block's unclamped scale is ``448 * block_amax / global_amax``, so a block
        four orders of magnitude below the tensor maximum lands below both candidate floors -- pinned torchao's
        ``finfo(float8_e4m3fn).tiny`` and this module's ``2 ** -9`` -- without being all-zero. That is the case
        that separates "reproduce the template" from "reproduce a plausible formula".
        """
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=57).clone()
        weight[2, 32:48] = 1e-7
        weight[5, 0:16] = 3e-6
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)
        codes = torch.full((8, 4), weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE, dtype=torch.uint8, device=device)

        result = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)

        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            produced, expected = _attr(result, attrs), _attr(template, attrs)
            assert torch.equal(produced.view(torch.uint8), expected.view(torch.uint8))

    @pytest.mark.unit
    def test_mixed_codes_keep_the_template_only_where_the_code_is_unclipped(self, nvfp4_backend, arithmetic):
        """Per block: an unclipped code keeps the template's exact bytes, every other code stores its candidate.

        The payload is compared byte by byte, which is what proves the packed-nibble placement is right: the eight
        bytes of block ``b`` of row ``r`` come from one source or the other and never from a mixture. The stored
        *scales* are compared the same way, after un-swizzling both buffers with torchao's own ``from_blocked``:
        that reads the layout back rather than restating the placement the packer computes, so a wrong swizzle
        mask, a rewritten unclipped byte or a wrong clipped byte all move a compared byte. The whole buffer is then
        checked to differ from the template's in no more bytes than there are clipped blocks, which is what catches
        a rewritten padding byte.
        """
        cls, device = nvfp4_backend
        rows, blocks = 6, 4
        weight = _random_weight(rows, blocks * 16, device, seed=58).clone()
        weight[1, 16:32] = 0.0
        global_scale = _global_scale(weight)
        template = _template_in_mode(cls, weight, global_scale, arithmetic)
        codes = _mixed_codes(rows, blocks, device)
        unclipped = (codes == weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE).cpu()
        assert bool(unclipped.any()) and not bool(unclipped.all())

        result = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)

        backend = weight_mse._resolve_torchao()
        clipped_scales = weight_mse.select_nvfp4_block_scales_awq_clip(weight, global_scale, codes)
        clipped_qdata = weight_mse._pack_qdata(weight.float(), clipped_scales, global_scale.float(), backend)
        produced = _attr(result, _QDATA_ATTRS).view(torch.uint8).reshape(rows, blocks, 8).cpu()
        from_template = _attr(template, _QDATA_ATTRS).view(torch.uint8).reshape(rows, blocks, 8).cpu()
        from_formula = clipped_qdata.reshape(rows, blocks, 8).cpu()
        assert torch.equal(produced[unclipped], from_template[unclipped])
        assert torch.equal(produced[~unclipped], from_formula[~unclipped])

        # And the stored scales, byte for byte in the template's own swizzled layout: an unclipped block keeps the
        # template's E4M3 scale byte and a clipped one stores exactly its candidate's.
        produced_scales = _linear_scale_bytes(result, rows, blocks)
        template_scales = _linear_scale_bytes(template, rows, blocks)
        expected_scales = clipped_scales.detach().cpu().contiguous().view(torch.uint8).reshape(rows, blocks)
        assert torch.equal(produced_scales[unclipped], template_scales[unclipped])
        assert torch.equal(produced_scales[~unclipped], expected_scales[~unclipped])
        # Nothing outside the clipped blocks was rewritten, the swizzle's padding bytes included.
        produced_buffer = _attr(result, _SCALE_ATTRS).reshape(-1).contiguous().view(torch.uint8).cpu()
        template_buffer = _attr(template, _SCALE_ATTRS).reshape(-1).contiguous().view(torch.uint8).cpu()
        assert int((produced_buffer != template_buffer).sum()) <= int((~unclipped).sum())

        # Like for like, through identical wrapper semantics: an unclipped block reads back exactly what the
        # template reads back, and a clipped one does not.
        produced_values = _dequantize(result).float().reshape(rows, blocks, 16).cpu()
        template_values = _dequantize(template).float().reshape(rows, blocks, 16).cpu()
        assert torch.equal(produced_values[unclipped], template_values[unclipped])
        assert not torch.equal(produced_values[~unclipped], template_values[~unclipped])

    @pytest.mark.unit
    def test_preserves_wrapper_class_metadata_layout_and_the_global_scale_object(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=49)
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)
        codes = _mixed_codes(8, 4, device)
        before = {
            "qdata": _attr(template, _QDATA_ATTRS).clone(),
            "scale": _attr(template, _SCALE_ATTRS).clone(),
            "weight": weight.clone(),
            "codes": codes.clone(),
        }

        result = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)

        assert result is not template
        assert type(result) is type(template)
        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert result.device == template.device
        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            produced, expected = _attr(result, attrs), _attr(template, attrs)
            assert produced.shape == expected.shape
            assert produced.dtype == expected.dtype
            assert produced.device == expected.device
            assert produced.is_contiguous()
        # The global scale is not re-derived: the very same tensor is handed to the new wrapper.
        assert _attr(result, _PER_TENSOR_ATTRS) is _attr(template, _PER_TENSOR_ATTRS)
        assert _attr(result, _ACT_PER_TENSOR_ATTRS) is _attr(template, _ACT_PER_TENSOR_ATTRS)
        assert bool(_attr(result, _SWIZZLE_ATTRS)) is bool(_attr(template, _SWIZZLE_ATTRS)) is True
        assert _attr(result, _TRITON_ATTRS) == _attr(template, _TRITON_ATTRS)
        assert _attr(result, _ACT_KWARGS_ATTRS) == _attr(template, _ACT_KWARGS_ATTRS)
        assert _attr(result, _BLOCK_SIZE_ATTRS) == _attr(template, _BLOCK_SIZE_ATTRS)
        assert _attr(result, _ORIG_DTYPE_ATTRS) == _attr(template, _ORIG_DTYPE_ATTRS)
        # Nothing the caller owns was touched, including the template's own storage.
        assert torch.equal(_attr(template, _QDATA_ATTRS).view(torch.uint8), before["qdata"].view(torch.uint8))
        assert torch.equal(_attr(template, _SCALE_ATTRS).view(torch.uint8), before["scale"].view(torch.uint8))
        assert torch.equal(weight, before["weight"])
        assert torch.equal(codes, before["codes"])

    @pytest.mark.unit
    def test_clipped_codes_store_exactly_the_selected_candidate_scales(self, nvfp4_backend, arithmetic):
        """Every clipped code stores its own independently gathered candidate, in the template's swizzled layout.

        Only the clipped codes are asserted here: the unclipped one stores the template's byte rather than the
        formula's, which the per-block case above checks. Making every code clipped is what lets the whole buffer
        be compared at once against an independent swizzle of the gathered candidates.
        """
        cls, device = nvfp4_backend
        weight = _random_weight(6, 64, device, seed=50)
        global_scale = _global_scale(weight)
        template = _template_in_mode(cls, weight, global_scale, arithmetic)
        codes = (_mixed_codes(6, 4, device).long() % weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE).to(torch.uint8)
        assert int(codes.max()) < weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE

        result = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)
        scales = weight_mse.select_nvfp4_block_scales_awq_clip(weight, global_scale, codes)
        candidates = weight_mse.nvfp4_awq_clip_candidate_scales(weight, global_scale)
        expected_scales = candidates.gather(2, codes.long()[..., None]).squeeze(-1)

        assert scales.dtype == torch.float8_e4m3fn
        assert torch.equal(scales.float(), expected_scales)
        # The stored scale buffer is exactly those independently gathered scales in the template's swizzled layout.
        backend = weight_mse._resolve_torchao()
        selected = expected_scales.to(torch.float8_e4m3fn)
        blocked = backend.to_blocked(selected).flatten()
        assert torch.equal(_attr(result, _SCALE_ATTRS).view(torch.uint8).reshape(-1), blocked.view(torch.uint8))
        # And the packed payload is exactly torchao's quantization of the weight under those same scales, compared
        # byte for byte rather than through the wrapper's dequantization, which would round the comparison itself.
        expected_qdata = weight_mse._pack_qdata(weight.float(), selected, global_scale.float(), backend)
        assert torch.equal(_attr(result, _QDATA_ATTRS).view(torch.uint8).reshape(-1), expected_qdata.reshape(-1))
        # A repack that stored the template's own scales would be indistinguishable from an amax conversion.
        assert not torch.equal(
            _attr(result, _QDATA_ATTRS).view(torch.uint8), _attr(template, _QDATA_ATTRS).view(torch.uint8)
        )

    @pytest.mark.unit
    def test_the_repack_selects_scales_independently_of_the_chunk_size(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=51)
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)
        codes = _mixed_codes(4, 4, device)

        reference = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes)
        chunked = weight_mse.repack_nvfp4_weight_awq_clip(weight, template, codes, block_chunk_size=1)

        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            assert torch.equal(_attr(chunked, attrs).view(torch.uint8), _attr(reference, attrs).view(torch.uint8))


class TestAWQClipRepackValidation:
    """Everything the AWQ-clip packer refuses, checked without any torchao kernel ever running."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "codes, message",
        [
            (None, "must be a torch.Tensor"),
            (torch.zeros(8, 4, dtype=torch.int64), "one byte per block"),
            (torch.zeros(8, 5, dtype=torch.uint8), "must have shape"),
            (torch.zeros(8, dtype=torch.uint8), "must have shape"),
            (torch.full((8, 4), 11, dtype=torch.uint8), "indexed by 0"),
        ],
    )
    def test_invalid_ratio_codes_are_rejected(self, monkeypatch, codes, message):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        error = TypeError if codes is None else ValueError
        with pytest.raises(error, match=message):
            weight_mse.repack_nvfp4_weight_awq_clip(_fake_weight(), _fake_template(), codes)

    @pytest.mark.unit
    @pytest.mark.parametrize("block_chunk_size", [0, -1, True, 1.5, None])
    def test_invalid_block_chunk_sizes_are_rejected(self, monkeypatch, block_chunk_size):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        codes = torch.zeros(8, 4, dtype=torch.uint8)
        with pytest.raises(ValueError, match="block_chunk_size must be a positive int"):
            weight_mse.repack_nvfp4_weight_awq_clip(_fake_weight(), _fake_template(), codes, block_chunk_size)

    @pytest.mark.unit
    def test_an_incompatible_template_or_weight_is_rejected(self, monkeypatch):
        _skip_without_e4m3()
        _install_fake_backend(monkeypatch)
        codes = torch.zeros(8, 4, dtype=torch.uint8)
        with pytest.raises(TypeError, match="must be a torchao"):
            weight_mse.repack_nvfp4_weight_awq_clip(_fake_weight(), object(), codes)
        with pytest.raises(ValueError, match="swizzled"):
            weight_mse.repack_nvfp4_weight_awq_clip(_fake_weight(), _fake_template(is_swizzled_scales=False), codes)
        non_finite = _fake_weight()
        non_finite[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.repack_nvfp4_weight_awq_clip(non_finite, _fake_template(), codes)


@pytest.mark.unit
def test_gptq_constants_are_the_reference_ones():
    """The three constants that define the update, plus the constructions it inherits from AWQ-clip."""
    assert weight_mse.NVFP4_GPTQ_UPDATE_BLOCK_SIZE == 128
    assert weight_mse.NVFP4_GPTQ_PERC_DAMP == 0.01
    assert weight_mse.NVFP4_GPTQ_HESSIAN_FACTOR == 2.0
    assert weight_mse.NVFP4_GPTQ_TEMPLATE_ARITHMETICS == ("torchao_non_triton", "mslk_triton")


class TestGPTQHessian:
    """The group-balanced input Hessian. It needs no torchao: it is a plain FP32 reduction over quantized rows."""

    @pytest.mark.unit
    def test_matches_the_reference_formula_and_balances_groups_equally(self):
        first = _integer_rows([[1.0, 2.0, 0.0, -1.0], [2.0, 0.0, 1.0, 1.0]])
        second = _integer_rows([[0.0, 1.0, -2.0, 1.0]])

        hessian = weight_mse.nvfp4_gptq_hessian([first, second])

        expected = (_reference_group_hessian(first) + _reference_group_hessian(second)) / 2.0
        assert torch.equal(hessian, expected)
        assert hessian.dtype is torch.float32
        assert hessian.is_contiguous()

    @pytest.mark.unit
    def test_duplicating_rows_inside_one_group_does_not_reweight_it(self):
        """``sqrt(2 / N_g)`` divides by the group's own row count, so a group counts once however large it is.

        The comparison is to a tight tolerance rather than bit-exact, and unavoidably so: the two normalizations
        are ``sqrt(2 / N)`` and ``sqrt(1 / N)``, and no pair of row counts makes both of them exactly representable
        in FP32, so the doubled group's entries differ from the original's in their last unit in the last place.
        A relative tolerance alone is not enough either, because an entry where two groups' contributions cancel is
        *exactly* zero in one reduction and a rounding residue of the cancelled magnitude in the other; the
        tolerance is therefore an absolute one, scaled to a millionth of the matrix's own largest entry. A group
        that was genuinely counted twice would move entries by a factor of two, six orders of magnitude outside it.
        """
        rows = _integer_rows([[1.0, 2.0, 0.0, -1.0], [2.0, 0.0, 1.0, 1.0]])
        doubled = torch.cat((rows, rows), dim=0).contiguous()
        other = _integer_rows([[0.0, 1.0, -2.0, 1.0]])

        single = weight_mse.nvfp4_gptq_hessian([rows])
        assert torch.allclose(
            single, weight_mse.nvfp4_gptq_hessian([doubled]), rtol=1e-6, atol=1e-6 * float(single.abs().max())
        )
        combined = weight_mse.nvfp4_gptq_hessian([rows, other])
        assert torch.allclose(
            combined,
            weight_mse.nvfp4_gptq_hessian([doubled, other]),
            rtol=1e-6,
            atol=1e-6 * float(combined.abs().max()),
        )
        # The same group listed twice is two equally weighted groups whose mean is that group's own Hessian, and
        # *that* reduction is exact: ``(H + H) / 2`` loses nothing.
        assert torch.equal(single, weight_mse.nvfp4_gptq_hessian([rows, rows]))
        # A second, *different* group does change the balance, which is what equal weighting means.
        assert not torch.equal(
            weight_mse.nvfp4_gptq_hessian([rows, other]), weight_mse.nvfp4_gptq_hessian([rows, rows, other])
        )

    @pytest.mark.unit
    def test_group_order_is_the_callers_and_is_summed_in_it(self):
        """The builder passes sorted labels; the reduction adds in exactly that order and divides once."""
        first = _integer_rows([[1.0, 2.0, 0.0, -1.0]])
        second = _integer_rows([[0.0, 1.0, -2.0, 1.0]])
        expected = (_reference_group_hessian(first) + _reference_group_hessian(second)) / 2.0

        assert torch.equal(weight_mse.nvfp4_gptq_hessian([first, second]), expected)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "groups, error, message",
        [
            ([], ValueError, "non-empty sequence"),
            ("rows", ValueError, "non-empty sequence"),
            ([object()], TypeError, "must be a torch.Tensor"),
            ([torch.zeros(4, dtype=torch.float32)], ValueError, "non-empty rank-2"),
            ([torch.zeros(0, 4, dtype=torch.float32)], ValueError, "non-empty rank-2"),
        ],
    )
    def test_rejects_unusable_groups(self, groups, error, message):
        with pytest.raises(error, match=message):
            weight_mse.nvfp4_gptq_hessian(groups)

    @pytest.mark.unit
    def test_rejects_mismatched_width_dtype_and_non_finite_rows(self):
        rows = _integer_rows([[1.0, 2.0, 0.0, -1.0]])
        with pytest.raises(ValueError, match="input channel"):
            weight_mse.nvfp4_gptq_hessian([rows, torch.zeros(1, 8, dtype=torch.float32)])
        with pytest.raises(ValueError, match="must be torch.float32"):
            weight_mse.nvfp4_gptq_hessian([rows, torch.zeros(1, 4, dtype=torch.bfloat16)])
        broken = rows.clone()
        broken[0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.nvfp4_gptq_hessian([broken])


class TestGPTQDampedHessian:
    """Dead columns and damping, in the reference's order: zero, unit diagonal, then ``0.01 * mean(diag)``."""

    @pytest.mark.unit
    def test_dead_columns_are_zeroed_and_given_a_unit_diagonal_before_damping(self):
        weight = torch.ones(2, 32, dtype=torch.float32)
        weight[:, 3] = 0.0
        hessian = torch.full((32, 32), 2.0)
        hessian.fill_diagonal_(4.0)

        damped = weight_mse.nvfp4_gptq_damped_hessian(hessian, weight)

        # After zeroing the dead row/column and setting its diagonal to 1, the diagonal is 31 fours and one one.
        expected_damping = 0.01 * ((31 * 4.0 + 1.0) / 32.0)
        assert damped.dead_columns == 1
        assert damped.damping == pytest.approx(expected_damping, rel=1e-6)
        assert float(damped.matrix[3, 3]) == pytest.approx(1.0 + expected_damping, rel=1e-6)
        assert torch.equal(damped.matrix[3, :3], torch.zeros(3))
        assert torch.equal(damped.matrix[:3, 3], torch.zeros(3))
        assert float(damped.matrix[0, 0]) == pytest.approx(4.0 + expected_damping, rel=1e-6)
        assert float(damped.matrix[0, 1]) == pytest.approx(2.0, rel=1e-6)
        assert damped.diagonal_min == pytest.approx(1.0 + expected_damping, rel=1e-6)
        assert damped.diagonal_max == pytest.approx(4.0 + expected_damping, rel=1e-6)
        assert damped.diagonal_min <= damped.diagonal_mean <= damped.diagonal_max
        # The caller's matrix is never touched.
        assert float(hessian[3, 3]) == 4.0

    @pytest.mark.unit
    def test_damping_is_exactly_one_percent_of_the_diagonal_mean(self):
        weight = torch.ones(2, 32, dtype=torch.float32)
        hessian = torch.eye(32) * 5.0

        damped = weight_mse.nvfp4_gptq_damped_hessian(hessian, weight)

        assert damped.dead_columns == 0
        assert damped.damping == pytest.approx(0.05, rel=1e-6)
        assert weight_mse.NVFP4_GPTQ_PERC_DAMP == 0.01

    @pytest.mark.unit
    def test_a_zero_second_moment_reports_a_diagonal_minimum_of_exactly_the_damping(self):
        """A channel the activation quantizer always underflows has a second moment of exactly zero.

        Its weight column is not zero, so it is not a dead column: its damped diagonal entry *is* the damping. The
        recorded damping therefore has to be exactly the FP32 scalar the diagonal received, because the runtime
        loader rejects an artifact whose 'diagonal_min' sits below its own 'damping' as self-inconsistent, and the
        FP64 product of ``0.01`` and the mean is not an FP32 value.
        """
        weight = torch.ones(2, 32, dtype=torch.float32)
        hessian = torch.eye(32) * 3.0
        hessian[7, 7] = 0.0

        damped = weight_mse.nvfp4_gptq_damped_hessian(hessian, weight)

        assert damped.dead_columns == 0
        assert float(damped.matrix[7, 7]) == damped.damping
        assert damped.diagonal_min == damped.damping
        assert damped.diagonal_min <= damped.diagonal_mean <= damped.diagonal_max

    @pytest.mark.unit
    def test_rejects_an_all_zero_diagonal_and_a_non_finite_or_misshaped_matrix(self):
        weight = torch.ones(2, 32, dtype=torch.float32)
        with pytest.raises(ValueError, match="non-positive or non-finite mean"):
            weight_mse.nvfp4_gptq_damped_hessian(torch.zeros(32, 32), weight)
        with pytest.raises(ValueError, match="must have shape"):
            weight_mse.nvfp4_gptq_damped_hessian(torch.eye(16), weight)
        broken = torch.eye(32)
        broken[0, 1] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.nvfp4_gptq_damped_hessian(broken, weight)
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.nvfp4_gptq_damped_hessian(object(), weight)


class TestNVFP4TemplateIdentity:
    """The runtime binding check reads a wrapper's buffers without importing torchao at all.

    A converted NVFP4 weight *is* a ``torch.Tensor`` -- torchao's wrapper is a tensor subclass -- and the binding
    check requires one, so these exercise a tensor-subclass stand-in rather than a bare object. The one object that
    is not a tensor is still passed in, because refusing it is part of the contract.
    """

    @pytest.mark.unit
    def test_returns_the_templates_own_buffers(self):
        _skip_without_e4m3()
        template = _tensor_template(rows=8, columns=64)

        identity = weight_mse.nvfp4_template_identity(template)

        assert isinstance(template, torch.Tensor)
        assert (identity.rows, identity.columns) == (8, 64)
        assert identity.qdata is template.qdata
        assert identity.scale is template._scale_e4m3
        assert identity.global_scale is template._per_tensor_scale

    @pytest.mark.unit
    def test_rejects_a_wrapper_it_cannot_address_byte_by_byte(self):
        _skip_without_e4m3()
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.nvfp4_template_identity(object())
        with pytest.raises(ValueError, match="rank 2"):
            weight_mse.nvfp4_template_identity(torch.zeros(8, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="does not expose"):
            weight_mse.nvfp4_template_identity(_tensor_template(drop="per_tensor_scale"))
        with pytest.raises(ValueError, match="packed byte"):
            weight_mse.nvfp4_template_identity(_tensor_template(qdata=torch.zeros(8, 8, dtype=torch.uint8)))
        with pytest.raises(ValueError, match="2-byte elements"):
            weight_mse.nvfp4_template_identity(_tensor_template(qdata=torch.zeros(8, 32, dtype=torch.bfloat16)))
        with pytest.raises(ValueError, match="fewer than"):
            weight_mse.nvfp4_template_identity(_tensor_template(scale_e4m3=torch.zeros(4, dtype=torch.float8_e4m3fn)))
        with pytest.raises(ValueError, match="positive and finite"):
            weight_mse.nvfp4_template_identity(_tensor_template(per_tensor_scale=torch.zeros(())))


@pytest.mark.usefixtures("nvfp4_backend")
class TestGPTQPayloadSelection:
    """The column-wise update itself, against a literal transcription of the ModelOpt reference."""

    @pytest.mark.unit
    @pytest.mark.parametrize("columns", [32, 128, 144, 272])
    def test_matches_the_reference_update_including_the_block_boundary_and_tail(self, nvfp4_backend, columns):
        """128 columns is exactly one update block; 144 and 272 exercise the cross-block ``addmm_`` and the tail."""
        cls, device = nvfp4_backend
        weight = _random_weight(4, columns, device, seed=71)
        scale = _global_scale(weight)
        template = _make_template(cls, weight, scale)
        block_scale = weight_mse.nvfp4_template_block_scales(template)
        hessian = _correlated_hessian(columns, device, seed=72)

        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, hessian)

        expected = _reference_gptq_codes(weight, scale, block_scale, hessian)
        backend = weight_mse._resolve_torchao()
        assert torch.equal(selection.qdata, backend.pack_uint4(expected))
        assert selection.qdata.dtype is torch.uint8
        assert selection.qdata.numel() == 4 * columns // 2
        assert torch.isfinite(selection.values).all()

    @pytest.mark.unit
    def test_a_diagonal_hessian_reproduces_the_plain_fixed_scale_payload(self, nvfp4_backend):
        """With no cross-column coupling no residual can propagate, so GPTQ writes the ordinary payload."""
        cls, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=73)
        scale = _global_scale(weight)
        template = _make_template(cls, weight, scale)
        block_scale = weight_mse.nvfp4_template_block_scales(template)
        hessian = torch.eye(64, dtype=torch.float32, device=device) * 3.0

        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, hessian)

        backend = weight_mse._resolve_torchao()
        plain = weight_mse._pack_qdata(weight.float(), block_scale, scale.to(device), backend)
        assert torch.equal(selection.qdata, plain)

    @pytest.mark.unit
    def test_a_correlated_hessian_changes_later_codes_and_lowers_the_quadratic_objective(self, nvfp4_backend):
        """The point of the method: later columns absorb earlier rounding error under a correlated Hessian."""
        cls, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=74)
        scale = _global_scale(weight)
        template = _make_template(cls, weight, scale)
        block_scale = weight_mse.nvfp4_template_block_scales(template)
        rows = _correlated_rows(64, 64, device, seed=75)
        damped = weight_mse.nvfp4_gptq_damped_hessian(weight_mse.nvfp4_gptq_hessian([rows]), weight)

        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, damped.matrix)

        backend = weight_mse._resolve_torchao()
        ordinary = weight_mse._pack_qdata(weight.float(), block_scale, scale.to(device), backend)
        assert not torch.equal(selection.qdata, ordinary)

        ordinary_values = weight_mse._reconstruct(
            weight.float().reshape(4, 4, 16), block_scale.reshape(4, 4, 1), scale.to(device), backend
        ).reshape(4, 64)
        # The first column cannot change: nothing has been written before it whose error it could absorb.
        assert torch.equal(selection.values[:, 0], ordinary_values[:, 0])
        # Later columns do, which is the whole point of the update.
        assert not torch.equal(selection.values, ordinary_values)
        selected_objective = weight_mse.nvfp4_gptq_objective(selection.values, weight, damped.matrix)
        ordinary_objective = weight_mse.nvfp4_gptq_objective(ordinary_values, weight, damped.matrix)
        assert selected_objective < ordinary_objective

    @pytest.mark.unit
    def test_the_selected_values_decode_the_selected_payload(self, nvfp4_backend):
        """The reported values are the payload's own decode under the fixed scales, not the working weight."""
        cls, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=76)
        scale = _global_scale(weight)
        template = _make_template(cls, weight, scale)
        block_scale = weight_mse.nvfp4_template_block_scales(template)
        hessian = _correlated_hessian(64, device, seed=77)

        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, hessian)

        backend = weight_mse._resolve_torchao()
        readback = weight_mse._resolve_awq_clip_readback()
        payload = backend.f4_unpacked_to_f32(readback.unpack_uint4(selection.qdata)).to(torch.float32)
        multiplier = (scale.to(device).float() * block_scale).reshape(4, 4, 1)
        decoded = (payload.reshape(4, 4, 16) * multiplier).reshape(4, 64)
        assert torch.equal(decoded, selection.values)

    @pytest.mark.unit
    def test_a_hessian_this_build_cannot_factorize_is_a_hard_error(self, nvfp4_backend):
        """The reference substitutes the identity here; this build refuses instead, with no fallback payload."""
        cls, device = nvfp4_backend
        weight = _random_weight(4, 32, device, seed=78)
        scale = _global_scale(weight)
        block_scale = weight_mse.nvfp4_template_block_scales(_make_template(cls, weight, scale))

        with pytest.raises(ValueError, match="could not be factorized"):
            weight_mse.select_nvfp4_gptq_payload(
                weight, scale, block_scale, -torch.eye(32, dtype=torch.float32, device=device)
            )
        with pytest.raises(ValueError, match="could not be factorized"):
            weight_mse.select_nvfp4_gptq_payload(
                weight, scale, block_scale, torch.zeros(32, 32, dtype=torch.float32, device=device)
            )

    @pytest.mark.unit
    def test_rejects_unusable_scales_hessians_and_weights(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(4, 32, device, seed=79)
        scale = _global_scale(weight)
        block_scale = weight_mse.nvfp4_template_block_scales(_make_template(cls, weight, scale))
        hessian = _correlated_hessian(32, device, seed=80)

        with pytest.raises(ValueError, match="must have shape"):
            weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale[:, :1], hessian)
        # A zero block scale is a legitimate template encoding and is covered below; a negative or non-finite one
        # is not something any construction produces, so it is refused rather than written under.
        with pytest.raises(ValueError, match="non-negative"):
            weight_mse.select_nvfp4_gptq_payload(weight, scale, -block_scale, hessian)
        with pytest.raises(ValueError, match="non-negative"):
            weight_mse.select_nvfp4_gptq_payload(weight, scale, torch.full_like(block_scale, float("inf")), hessian)
        with pytest.raises(ValueError, match="must have shape"):
            weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, hessian[:16, :16])
        non_finite = weight.clone()
        non_finite[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            weight_mse.select_nvfp4_gptq_payload(non_finite, scale, block_scale, hessian)
        with pytest.raises(ValueError, match="positive and finite"):
            weight_mse.select_nvfp4_gptq_payload(weight, torch.zeros(()), block_scale, hessian)

    @pytest.mark.unit
    def test_an_all_zero_template_scale_is_written_as_code_zero_without_dividing(self, nvfp4_backend):
        """The ordinary template encodes an all-zero or underflowed block with a blocked scale of exactly zero.

        That is the fixed template this method writes under, so every row of such a block is stored as FP4 code
        zero and decodes to ``0.0``. Nothing is floored and no substitute scale is invented, and no reciprocal of
        zero is ever taken -- a payload of infinities or NaNs would have failed the finiteness checks instead.
        """
        cls, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=91)
        scale = _global_scale(weight)
        block_scale = weight_mse.nvfp4_template_block_scales(_make_template(cls, weight, scale))
        # A diagonal Hessian keeps the residual of an unrepresentable block from propagating anywhere, so this
        # case pins the payload itself rather than the arithmetic of the update.
        hessian = torch.eye(64, dtype=torch.float32, device=device) * 3.0

        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, torch.zeros_like(block_scale), hessian)

        assert torch.equal(selection.values, torch.zeros_like(selection.values))
        assert torch.equal(_unpacked_codes(selection.qdata, 4, 64), torch.zeros(4, 64, dtype=torch.uint8))
        assert selection.qdata.dtype is torch.uint8
        assert selection.qdata.numel() == 4 * 64 // 2

    @pytest.mark.unit
    def test_zero_and_nonzero_scale_rows_of_one_column_are_decided_independently(self, nvfp4_backend):
        """One column decision serves both kinds of row: the update is per output row, so neither disturbs the other.

        Row 0 has no usable scale in any block and row 1 only loses its first block, while rows 2 and 3 keep the
        template's own scales. The zero-scale rows must be code zero, and the ordinary rows must be bit-identical
        to the payload they get when nothing is zeroed at all.
        """
        cls, device = nvfp4_backend
        weight = _random_weight(4, 64, device, seed=92)
        scale = _global_scale(weight)
        block_scale = weight_mse.nvfp4_template_block_scales(_make_template(cls, weight, scale))
        mixed = block_scale.clone()
        mixed[0, :] = 0.0
        mixed[1, 0] = 0.0
        hessian = _correlated_hessian(64, device, seed=93)

        baseline = weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, hessian)
        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, mixed, hessian)

        codes = _unpacked_codes(selection.qdata, 4, 64)
        assert torch.equal(selection.values[0], torch.zeros_like(selection.values[0]))
        assert torch.equal(codes[0], torch.zeros_like(codes[0]))
        block = weight_mse.NVFP4_BLOCK_SIZE
        assert torch.equal(selection.values[1, :block], torch.zeros_like(selection.values[1, :block]))
        assert torch.equal(codes[1, :block], torch.zeros_like(codes[1, :block]))
        # The rows with a usable scale went through TorchAO's own arithmetic, unchanged by the zeroed rows.
        assert torch.equal(selection.values[2:], baseline.values[2:])
        assert torch.equal(codes[2:], _unpacked_codes(baseline.qdata, 4, 64)[2:])
        # And the zeroed rows really did change something: they are not the payload the template's scales give.
        assert not torch.equal(selection.values[0], baseline.values[0])


@pytest.mark.usefixtures("nvfp4_backend")
@pytest.mark.parametrize("arithmetic", list(weight_mse.NVFP4_GPTQ_TEMPLATE_ARITHMETICS))
class TestGPTQRepack:
    """Replacing a template's payload must leave every other byte of the wrapper exactly where it was."""

    @pytest.mark.unit
    def test_replaces_only_the_payload(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=81)
        scale = _global_scale(weight)
        template = _template_in_mode(cls, weight, scale, arithmetic)
        block_scale = weight_mse.nvfp4_template_block_scales(template)
        hessian = _correlated_hessian(64, device, seed=82)
        selection = weight_mse.select_nvfp4_gptq_payload(weight, scale, block_scale, hessian)

        result = weight_mse.repack_nvfp4_weight_gptq(template, selection.qdata)

        assert result is not template
        assert type(result) is type(template)
        assert tuple(result.shape) == tuple(template.shape)
        assert result.device == template.device
        # The scale buffer is byte-identical, padding included, and the global scale is the very same object.
        produced, expected = _attr(result, _SCALE_ATTRS), _attr(template, _SCALE_ATTRS)
        assert produced.dtype == expected.dtype and produced.shape == expected.shape
        assert torch.equal(produced.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        assert _attr(result, _PER_TENSOR_ATTRS) is _attr(template, _PER_TENSOR_ATTRS)
        for attrs in (
            _BLOCK_SIZE_ATTRS,
            _ORIG_DTYPE_ATTRS,
            _ACT_PER_TENSOR_ATTRS,
            _SWIZZLE_ATTRS,
            _TRITON_ATTRS,
            _ACT_KWARGS_ATTRS,
        ):
            assert _attr(result, attrs) == _attr(template, attrs)
        # And the payload is the selected one, in the template's own storage layout.
        stored = _attr(result, _QDATA_ATTRS)
        assert stored.dtype == _attr(template, _QDATA_ATTRS).dtype
        assert stored.shape == _attr(template, _QDATA_ATTRS).shape
        assert torch.equal(stored.contiguous().view(torch.uint8).reshape(-1), selection.qdata.reshape(-1))
        assert torch.isfinite(_dequantize(result)).all()

    @pytest.mark.unit
    def test_replacing_with_the_templates_own_payload_reproduces_it_byte_for_byte(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=83)
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)
        payload = _attr(template, _QDATA_ATTRS).contiguous().view(torch.uint8).clone()

        result = weight_mse.repack_nvfp4_weight_gptq(template, payload)

        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            produced, expected = _attr(result, attrs), _attr(template, attrs)
            assert torch.equal(produced.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        assert torch.equal(_dequantize(result), _dequantize(template))

    @pytest.mark.unit
    def test_rejects_a_payload_that_does_not_describe_the_template(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=84)
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            weight_mse.repack_nvfp4_weight_gptq(template, object())
        with pytest.raises(ValueError, match="packed bytes"):
            weight_mse.repack_nvfp4_weight_gptq(template, torch.zeros(8, 32, dtype=torch.bfloat16, device=device))
        with pytest.raises(ValueError, match="packed byte"):
            weight_mse.repack_nvfp4_weight_gptq(template, torch.zeros(8, 8, dtype=torch.uint8, device=device))
        with pytest.raises(ValueError, match="contiguous"):
            weight_mse.repack_nvfp4_weight_gptq(template, torch.zeros(32, 8, dtype=torch.uint8, device=device).t())


@pytest.mark.usefixtures("nvfp4_backend")
class TestGPTQTemplateReadback:
    """The builder's two template readbacks: the linear block scales and the ordinary payload's own values."""

    @pytest.mark.unit
    @pytest.mark.parametrize("arithmetic", list(weight_mse.NVFP4_GPTQ_TEMPLATE_ARITHMETICS))
    def test_block_scales_are_the_templates_own_unswizzled_bytes(self, nvfp4_backend, arithmetic):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=85)
        template = _template_in_mode(cls, weight, _global_scale(weight), arithmetic)

        scales = weight_mse.nvfp4_template_block_scales(template)

        assert scales.shape == (8, 4)
        assert scales.dtype is torch.float32
        assert torch.equal(
            scales.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).cpu(), _linear_scale_bytes(template, 8, 4)
        )

    @pytest.mark.unit
    def test_template_values_decode_the_templates_own_payload(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=86)
        scale = _global_scale(weight)
        template = _make_template(cls, weight, scale)

        values = weight_mse.nvfp4_template_values(template)

        assert torch.equal(values, _decoded_template_values(template, scale))

    @pytest.mark.unit
    def test_ordinary_template_refuses_an_unknown_construction(self, nvfp4_backend):
        _, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=87)
        with pytest.raises(ValueError, match="template_arithmetic must be one of"):
            weight_mse.nvfp4_ordinary_template(weight, _global_scale(weight), "guessed")

    @pytest.mark.unit
    def test_ordinary_template_is_the_conversion_the_runtime_performs(self, nvfp4_backend):
        cls, device = nvfp4_backend
        weight = _random_weight(8, 64, device, seed=88)
        scale = _global_scale(weight)

        built = weight_mse.nvfp4_ordinary_template(weight, scale, weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE)
        expected = _template_in_mode(cls, weight, scale, weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE)

        for attrs in (_QDATA_ATTRS, _SCALE_ATTRS):
            assert torch.equal(
                _attr(built, attrs).contiguous().view(torch.uint8),
                _attr(expected, attrs).contiguous().view(torch.uint8),
            )


@pytest.mark.unit
def test_gptq_entry_points_fail_closed_without_torchao(monkeypatch):
    """Every GPTQ entry point that needs the pinned kernels says so instead of guessing a replacement."""
    _skip_without_e4m3()

    def _missing(module_path):
        raise weight_mse.TorchAOUnavailableError(f"'{module_path}' could not be imported.")

    monkeypatch.setattr(weight_mse, "_torchao_backend", None)
    monkeypatch.setattr(weight_mse, "_import_module", _missing)
    with pytest.raises(ValueError, match="could not be imported"):
        weight_mse.repack_nvfp4_weight_gptq(object(), torch.zeros(8, dtype=torch.uint8))
    with pytest.raises(ValueError, match="could not be imported"):
        weight_mse.select_nvfp4_gptq_payload(
            torch.zeros(1, 16, dtype=torch.bfloat16), torch.ones(()), torch.ones(1, 1), torch.eye(16)
        )


def _integer_rows(values):
    """Activation rows of small integers, so every FP32 product and sum in a Hessian is exact."""
    return torch.tensor(values, dtype=torch.float32).contiguous()


def _reference_group_hessian(rows):
    """``H_g = scaled @ scaled.T`` with ``scaled = sqrt(2 / N_g) * Xq_g.T``, spelled out."""
    scaled = (math.sqrt(2.0 / float(int(rows.shape[0]))) * rows.t()).contiguous()
    return scaled @ scaled.t()


def _correlated_hessian(columns, device, seed):
    """A damped, strictly positive definite Hessian with real cross-column coupling."""
    rows = _correlated_rows(max(4 * columns, 64), columns, device, seed)
    hessian = weight_mse.nvfp4_gptq_hessian([rows])
    diagonal = torch.diagonal(hessian)
    diagonal += weight_mse.NVFP4_GPTQ_PERC_DAMP * float(diagonal.mean())
    return hessian


def _reference_gptq_codes(weight, per_tensor_scale, block_scale, hessian):
    """A literal transcription of the ModelOpt 0.46.0 column-wise GPTQ update, one column at a time.

    Only the fixed-scale quantizer is shared with the packer -- it is torchao's own arithmetic, verified elsewhere
    -- so what this pins is the update itself: the per-block clone, the divisor, the within-block ``addr_``, the
    cross-block ``addmm_``, and the order they happen in.
    """
    backend = weight_mse._resolve_torchao()
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    factor = torch.linalg.cholesky(hessian)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    working = weight.detach().float().clone()
    codes = torch.zeros(rows, columns, dtype=torch.uint8, device=weight.device)
    scale = per_tensor_scale.detach().reshape(()).to(device=weight.device, dtype=torch.float32)
    for start in range(0, columns, 128):
        stop = min(start + 128, columns)
        block_inverse = inverse[start:stop, start:stop]
        block_working = working.clone()
        errors = torch.zeros(rows, stop - start, dtype=torch.float32, device=weight.device)
        for index in range(stop - start):
            column = start + index
            current = block_working[:, column]
            divisor = block_inverse[index, index]
            block = block_scale[:, column // weight_mse.NVFP4_BLOCK_SIZE]
            reciprocal = (1.0 / scale) / block
            code = backend.f32_to_f4_unpacked(
                torch.clamp(current * reciprocal, -weight_mse.NVFP4_MAX, weight_mse.NVFP4_MAX)
            )
            quantized = backend.f4_unpacked_to_f32(code) * (scale * block)
            codes[:, column] = code
            working[:, column] = quantized
            error = (current - quantized) / divisor
            block_working[:, column:stop].addr_(error, block_inverse[index, index:], alpha=-1)
            errors[:, index] = error
        if stop < columns:
            working[:, stop:].addmm_(errors, inverse[start:stop, stop:], alpha=-1)
    return codes


def _unpacked_codes(qdata, rows, columns):
    """The FP4 codes a packed payload carries, as a host ``(rows, columns)`` uint8 matrix in torchao's own order."""
    readback = weight_mse._resolve_awq_clip_readback()
    return readback.unpack_uint4(qdata).reshape(rows, columns).to(torch.uint8).cpu()


def _mixed_codes(rows, blocks, device):
    """A deterministic code matrix that exercises several ratios, including the unclipped one."""
    positions = torch.arange(rows * blocks, dtype=torch.long, device=device)
    return (positions % weight_mse.NVFP4_AWQ_CLIP_RATIO_COUNT).reshape(rows, blocks).to(torch.uint8)


# The eight E2M1 magnitudes an NVFP4 payload nibble can decode to; every stored value is one of them times its
# block's ``global_scale * block_scale`` multiplier.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# The all-zero 16-weight block of :func:`_awq_clip_mixed_magnitude_weight`.
_AWQ_CLIP_ZERO_BLOCK = (2, slice(16, 32))


def _awq_clip_mixed_magnitude_weight(device, seed=59):
    """A weight carrying an ordinary, an all-zero and a far-below-amax block, all in one conversion.

    The last two are where a conversion's own E4M3 scale floor -- and not the block's values -- decides the stored
    scale byte, and where the two supported ordinary conversions disagree with each other and with any formula.
    """
    weight = _random_weight(8, 64, device, seed=seed).clone()
    weight[_AWQ_CLIP_ZERO_BLOCK] = 0.0
    weight[5, 32:48] = 1e-7
    return weight.contiguous()


def _template_payload_and_multiplier(template, per_tensor_scale):
    """A converted template's stored FP4 payload values and its per-block FP32 ``global * block`` multiplier.

    Both come off the wrapper's own buffers through torchao's ``unpack_uint4`` and ``from_blocked``, so this reads
    the bytes the runtime deploys rather than recomputing what they ought to be.
    """
    backend = weight_mse._resolve_torchao()
    readback = weight_mse._resolve_awq_clip_readback()
    rows, columns = int(template.shape[0]), int(template.shape[1])
    num_blocks = columns // weight_mse.NVFP4_BLOCK_SIZE
    packed = _attr(template, _QDATA_ATTRS).contiguous().view(torch.uint8)
    payload = backend.f4_unpacked_to_f32(readback.unpack_uint4(packed))
    payload = payload.to(torch.float32).reshape(rows, num_blocks, weight_mse.NVFP4_BLOCK_SIZE)
    scales = readback.from_blocked(_attr(template, _SCALE_ATTRS), rows, num_blocks)
    scales = scales.to(torch.float32).reshape(rows, num_blocks, 1)
    global_scale = per_tensor_scale.detach().reshape(()).to(device=payload.device, dtype=torch.float32)
    return payload, global_scale * scales


def _decoded_template_values(template, per_tensor_scale, scale_dtype=torch.float32):
    """Decode a template's exact stored bytes; ``scale_dtype`` rounds the block multiplier before it is applied."""
    payload, multiplier = _template_payload_and_multiplier(template, per_tensor_scale)
    rows, columns = int(template.shape[0]), int(template.shape[1])
    return (payload * multiplier.to(scale_dtype).to(torch.float32)).reshape(rows, columns)


def _linear_scale_bytes(tensor, rows, blocks):
    """The un-swizzled ``(rows, blocks)`` E4M3 scale bytes a converted wrapper stores, on the host."""
    readback = weight_mse._resolve_awq_clip_readback()
    scales = readback.from_blocked(_attr(tensor, _SCALE_ATTRS), rows, blocks)
    return scales.reshape(rows, blocks).contiguous().view(torch.uint8).cpu()


def _correlated_rows(count, columns, device, seed):
    """Rows whose neighbouring columns are strongly anti-correlated, so within-block cross terms dominate."""
    generator = torch.Generator().manual_seed(seed)
    common = torch.randn(count, 1, generator=generator)
    rows = 0.15 * torch.randn(count, columns, generator=generator) + common
    rows[:, 1::2] = rows[:, 1::2] * -1.0
    return rows.to(device=device, dtype=torch.float32).contiguous()


def _reference_awq_clip_candidates(weight, per_tensor_scale):
    """The eleven reference candidate scales of every block, as ``[rows][blocks][11]``.

    The reference order of operations is reproduced exactly -- ``amax * ratio``, the division by 6, the division by
    the global scale, the ``[2 ** -9, 448]`` clamp, and only then the E4M3 rounding -- one scalar at a time in
    FP32, without calling anything the packer uses.
    """
    global_scale = per_tensor_scale.detach().reshape(()).float().cpu()
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    values = weight.detach().float().cpu().tolist()
    candidates = []
    for row in range(rows):
        row_candidates = []
        for start in range(0, columns, weight_mse.NVFP4_BLOCK_SIZE):
            block = values[row][start : start + weight_mse.NVFP4_BLOCK_SIZE]
            amax = torch.tensor(max(abs(value) for value in block), dtype=torch.float32)
            block_candidates = []
            for ratio in weight_mse.NVFP4_AWQ_CLIP_RATIOS:
                clipped = amax * torch.tensor(float(ratio), dtype=torch.float32)
                unnormalized = clipped / torch.tensor(6.0, dtype=torch.float32)
                clamped = torch.clamp(unnormalized / global_scale, 2.0**-9, 448.0)
                block_candidates.append(float(clamped.to(torch.float8_e4m3fn).to(torch.float32)))
            row_candidates.append(block_candidates)
        candidates.append(row_candidates)
    return candidates


def _reference_awq_clip_losses(weight, per_tensor_scale, groups, template_reconstruction):
    """Plain-Python reference for the AWQ-clip loss of every block and candidate, as ``[rows][blocks][11]``.

    The E2M1 grid, the FP32 dot products and the group-balanced mean of per-row squared block outputs are all
    written out here, so nothing about the objective is borrowed from the packer. The specified arithmetic is
    emulated step by step: the reciprocal ordering, the ``[-6, 6]`` clamp, the E2M1 round trip and the
    reconstruction error are FP32, and each fixed 16-element dot product accumulates in FP32 -- explicitly, one
    ``numpy.float32`` addition at a time -- so its rounding is the arithmetic the deployed kernel would perform.
    Only the *finished* dot is widened to a Python double, and only then is it squared, averaged over the group's
    rows and averaged over the equally weighted groups.

    The eleventh candidate is deliberately *not* re-derived from the clipping formula: the unclipped code deploys
    the ordinary template's own bytes, so its reference delta is ``W - template_reconstruction``.
    """
    candidates = _reference_awq_clip_candidates(weight, per_tensor_scale)
    global_scale = np.float32(float(per_tensor_scale.detach().reshape(()).float()))
    values = weight.detach().float().cpu().tolist()
    template_values = template_reconstruction.detach().float().cpu().tolist()
    group_rows = [group.detach().float().cpu().tolist() for group in groups]
    losses = []
    for row, row_candidates in enumerate(candidates):
        row_losses = []
        for block, block_candidates in enumerate(row_candidates):
            start = block * weight_mse.NVFP4_BLOCK_SIZE
            elements = [np.float32(value) for value in values[row][start : start + weight_mse.NVFP4_BLOCK_SIZE]]
            block_losses = []
            for candidate, scale in enumerate(block_candidates):
                block_scale = np.float32(scale)
                # torchao's own ordering: the reciprocal is formed before it multiplies, and the dequantization
                # multiplier is the product of the two scales. Both are FP32, and a different order would round
                # a different value onto the FP4 grid.
                reciprocal = np.float32(np.float32(1.0) / global_scale) / block_scale
                multiplier = global_scale * block_scale
                delta = []
                for position, value in enumerate(elements):
                    if candidate == weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE:
                        delta.append(np.float32(value - np.float32(template_values[row][start + position])))
                        continue
                    scaled = min(max(float(value * reciprocal), -6.0), 6.0)
                    delta.append(np.float32(value - np.float32(_reference_e2m1(scaled)) * multiplier))
                total = 0.0
                for rows_of_group in group_rows:
                    squares = 0.0
                    for activation in rows_of_group:
                        dot = np.float32(0.0)
                        for position in range(weight_mse.NVFP4_BLOCK_SIZE):
                            dot = np.float32(dot + np.float32(activation[start + position]) * delta[position])
                        # The dot is complete, so widening it here is exactly where the objective widens it.
                        squares += float(dot) * float(dot)
                    total += squares / len(rows_of_group)
                block_losses.append(total / len(group_rows))
            row_losses.append(block_losses)
        losses.append(row_losses)
    return losses


def _reference_activation_qdq(rows, amax):
    """Plain-Python reference for the runtime-matched NVFP4 activation quantize/dequantize."""
    global_scale = float(torch.tensor(float(amax), dtype=torch.float32) / (6.0 * 448.0))
    values = rows.detach().float().cpu().tolist()
    reconstructed = []
    for row in values:
        out = []
        for start in range(0, len(row), weight_mse.NVFP4_BLOCK_SIZE):
            block = row[start : start + weight_mse.NVFP4_BLOCK_SIZE]
            block_amax = torch.tensor(max(abs(value) for value in block), dtype=torch.float32)
            scaled = torch.clamp((block_amax / 6.0) / global_scale, 2.0**-9, 448.0)
            scale = float(scaled.to(torch.float8_e4m3fn).to(torch.float32))
            out += [
                _reference_e2m1(min(max(value / (global_scale * scale), -6.0), 6.0)) * global_scale * scale
                for value in block
            ]
        reconstructed.append(out)
    return reconstructed


def _diagonal_ratio_codes(weight, per_tensor_scale, groups, template_reconstruction):
    """Earliest-wins argmin of the *diagonal* activation-weighted error over the same eleven candidates.

    Only the objective differs from the packer's: the per-column squared errors are weighted by that column's
    second moment instead of being combined through the block's full activation covariance. The candidate set is
    the packer's own, unclipped code included, so the two choices differ in the objective and in nothing else.
    """
    backend = weight_mse._resolve_torchao()
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    num_blocks = columns // weight_mse.NVFP4_BLOCK_SIZE
    scale = per_tensor_scale.detach().reshape(()).float().to(weight.device)
    blocks = weight.detach().float().reshape(rows, num_blocks, weight_mse.NVFP4_BLOCK_SIZE)
    candidates = weight_mse.nvfp4_awq_clip_candidate_scales(weight, per_tensor_scale)
    clipped = weight_mse._reconstruct(
        blocks[:, :, None, :],
        candidates[..., : weight_mse.NVFP4_AWQ_CLIP_UNCLIPPED_CODE, None],
        scale,
        backend,
    )
    template_blocks = (
        template_reconstruction.detach().float().reshape(rows, num_blocks, 1, weight_mse.NVFP4_BLOCK_SIZE)
    )
    delta = blocks[:, :, None, :] - torch.cat((clipped, template_blocks), dim=2)
    moments = torch.stack([group.double().pow(2).mean(dim=0) for group in groups]).mean(dim=0)
    weights = moments.reshape(1, num_blocks, 1, weight_mse.NVFP4_BLOCK_SIZE)
    loss = ((delta.double() ** 2) * weights).sum(dim=-1)

    best = loss[..., 0].clone()
    index = torch.zeros(loss.shape[:-1], dtype=torch.long, device=loss.device)
    for candidate in range(1, weight_mse.NVFP4_AWQ_CLIP_RATIO_COUNT):
        improved = loss[..., candidate] < best
        best = torch.where(improved, loss[..., candidate], best)
        index = torch.where(improved, torch.full_like(index, candidate), index)
    return index.to(torch.uint8)


def _earliest_argmin(values):
    """Index of the smallest value, keeping the earliest on an exact tie, exactly as the search's ``<`` does."""
    best = 0
    for index in range(1, len(values)):
        if values[index] < values[best]:
            best = index
    return best


def _four_over_six_scale(per_tensor_scale):
    """The Four-Over-Six weight global scale: the ordinary one renormalized against 256 instead of 448."""
    return per_tensor_scale * (448.0 / 256.0)


# The eight E2M1 magnitudes, in encoding order; the reference below rounds onto them with ties to even.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _reference_four_over_six_candidates(weight, per_tensor_scale):
    """The two reference candidate scales of every block, in M=6 then M=4 order, as ``[rows][blocks][2]``.

    The reference order of operations is reproduced exactly -- ``amax / magnitude``, the zero substitution, the
    division by the global scale, the ``[2 ** -9, 448]`` clamp, and only then the E4M3 rounding -- one scalar at a
    time in FP32, without calling anything the packer uses.
    """
    global_scale = per_tensor_scale.detach().reshape(()).float().cpu()
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    values = weight.detach().float().cpu().tolist()
    candidates = []
    for row in range(rows):
        row_candidates = []
        for start in range(0, columns, weight_mse.NVFP4_BLOCK_SIZE):
            block = values[row][start : start + weight_mse.NVFP4_BLOCK_SIZE]
            amax = torch.tensor(max(abs(value) for value in block), dtype=torch.float32)
            block_candidates = []
            for magnitude in weight_mse.NVFP4_FOUR_OVER_SIX_MAGNITUDES:
                unnormalized = amax / torch.tensor(float(magnitude), dtype=torch.float32)
                if float(unnormalized) == 0.0:
                    unnormalized = torch.tensor(1.0, dtype=torch.float32)
                clamped = torch.clamp(unnormalized / global_scale, 2.0**-9, 448.0)
                block_candidates.append(float(clamped.to(torch.float8_e4m3fn).to(torch.float32)))
            row_candidates.append(block_candidates)
        candidates.append(row_candidates)
    return candidates


def _reference_four_over_six(weight, per_tensor_scale):
    """Full plain-Python reference: the winning magnitude index and stored scale of every block.

    The candidates come from :func:`_reference_four_over_six_candidates` and the score is the squared
    reconstruction error computed here, with the E2M1 grid written out below, so nothing about the decision is
    borrowed from the packer.
    """
    values = weight.detach().float().cpu().tolist()
    global_scale = float(per_tensor_scale.detach().reshape(()).float())
    candidates = _reference_four_over_six_candidates(weight, per_tensor_scale)
    indices, scales = [], []
    for row, row_candidates in enumerate(candidates):
        row_indices, row_scales = [], []
        for block, block_candidates in enumerate(row_candidates):
            start = block * weight_mse.NVFP4_BLOCK_SIZE
            elements = values[row][start : start + weight_mse.NVFP4_BLOCK_SIZE]
            errors = [_reference_block_error(elements, candidate, global_scale) for candidate in block_candidates]
            index = 1 if errors[1] < errors[0] else 0
            row_indices.append(index)
            row_scales.append(block_candidates[index])
        indices.append(row_indices)
        scales.append(row_scales)
    return indices, scales


def _scales_like(reference, candidates, side):
    """An E4M3 scale tensor of ``reference``'s shape holding every block's ``side``-th reference candidate."""
    values = [[block[side] for block in row] for row in candidates]
    return torch.tensor(values, dtype=torch.float32, device=reference.device).to(torch.float8_e4m3fn)


def _reference_block_error(block, scale, global_scale):
    """Squared reconstruction error of one block under one E4M3 scale, with the E2M1 grid written out here."""
    total = 0.0
    for value in block:
        scaled = min(max(value / (global_scale * scale), -6.0), 6.0)
        total += (_reference_e2m1(scaled) * global_scale * scale - value) ** 2
    return total


def _reference_e2m1(value):
    """Round one value onto the signed E2M1 grid, ties going to the even encoding, as torchao's kernel does."""
    sign = -1.0 if value < 0.0 else 1.0
    magnitude = min(abs(value), 6.0)
    best, distance = 0, abs(magnitude - _E2M1_MAGNITUDES[0])
    for index in range(1, len(_E2M1_MAGNITUDES)):
        candidate = abs(magnitude - _E2M1_MAGNITUDES[index])
        if candidate < distance or (candidate == distance and index % 2 == 0):
            best, distance = index, candidate
    return sign * _E2M1_MAGNITUDES[best]


def _scalar_weighted_objective(weight, scale_fp8, per_tensor_scale, moments, damped=None):
    """Element-by-element reference for the weighted per-block objective, computed in plain Python.

    ``damped`` overrides the packer's own damped vector, which is what lets a caller pin the absolute magnitude
    against weights it damped itself.
    """
    backend = weight_mse._resolve_torchao()
    damped = (weight_mse.damped_second_moments(moments) if damped is None else damped).cpu()
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    num_blocks = columns // weight_mse.NVFP4_BLOCK_SIZE
    blocks = weight.float().reshape(rows * num_blocks, weight_mse.NVFP4_BLOCK_SIZE)
    scales = scale_fp8.reshape(rows * num_blocks, 1).float()
    reconstruction = weight_mse._reconstruct(
        blocks, scales, per_tensor_scale.float().reshape(()).to(weight.device), backend
    ).cpu()
    blocks = blocks.cpu()

    values = []
    for row in range(rows):
        row_values = []
        for block in range(num_blocks):
            index = row * num_blocks + block
            total = 0.0
            for position in range(weight_mse.NVFP4_BLOCK_SIZE):
                error = float(reconstruction[index, position]) - float(blocks[index, position])
                total += float(damped[block * weight_mse.NVFP4_BLOCK_SIZE + position]) * error * error
            row_values.append(total / weight_mse.NVFP4_BLOCK_SIZE)
        values.append(row_values)
    return torch.tensor(values, dtype=torch.float32)


@pytest.fixture(scope="module")
def nvfp4_backend():
    """Skip cleanly unless the pinned torchao NVFP4 API and the E4M3 dtype are usable on this machine."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")
    module = pytest.importorskip(
        "torchao.prototype.mx_formats.nvfp4_tensor", reason="the NVFP4 MSE packer requires torchao 0.17"
    )
    pytest.importorskip("torchao.prototype.mx_formats.kernels", reason="the NVFP4 MSE packer requires torchao 0.17")
    pytest.importorskip("torchao.prototype.mx_formats.utils", reason="the NVFP4 MSE packer requires torchao 0.17")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = torch.zeros(1, 16, dtype=torch.bfloat16, device=device)
    try:
        _make_template(module.NVFP4Tensor, probe, torch.ones((), device=device))
    except (NotImplementedError, RuntimeError) as err:  # pragma: no cover - depends on the install
        pytest.skip(f"torchao NVFP4 conversion is unavailable on {device}: {err}")
    return module.NVFP4Tensor, device


def _make_template(cls, weight, per_tensor_scale):
    """Convert ``weight`` with torchao's ordinary amax-based packing, in the swizzled layout the runtime uses."""
    return cls.to_nvfp4(
        weight,
        block_size=weight_mse.NVFP4_BLOCK_SIZE,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
    )


def _template_in_mode(cls, weight, per_tensor_scale, arithmetic):
    """Convert with the ordinary conversion of one construction mode, skipping when that mode is unavailable.

    The accelerated mode needs the MSLK Triton kernel and therefore a GPU that has it; the reference non-Triton
    mode is the one the module fixture already probed, so a failure there is a real failure and is not skipped.
    """
    accelerated = arithmetic == weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETIC_ACCELERATED
    try:
        return cls.to_nvfp4(
            weight,
            block_size=weight_mse.NVFP4_BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=accelerated,
        )
    except Exception as err:  # pragma: no cover - depends on the install and the GPU
        if accelerated:
            pytest.skip(f"the accelerated ordinary NVFP4 template is unavailable here: {err}")
        raise


def _template_values(weight, per_tensor_scale, arithmetic=weight_mse.AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE):
    """The unclipped candidate: what the ordinary template of ``weight`` reads back in one construction mode."""
    return weight_mse.nvfp4_awq_clip_template_reconstruction(weight, per_tensor_scale, arithmetic)


def _random_weight(rows, columns, device, dtype=torch.bfloat16, seed=0):
    """Deterministic high-precision weight; generated on CPU so the values do not depend on the device."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator, dtype=torch.float32)
    return values.to(device=device, dtype=dtype).contiguous()


def _global_scale(weight):
    """The per-tensor scale torchao's own two-level recipe derives from the weight, kept fixed by the packer."""
    return (weight.float().abs().max() / (weight_mse.NVFP4_MAX * weight_mse.FP8_E4M3_MAX)).reshape(())


def _amax_based_scales(weight, per_tensor_scale):
    """The ordinary max-based block scales: the block amax mapped onto FP4's largest magnitude, rounded to E4M3."""
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    blocks = weight.float().reshape(rows, columns // weight_mse.NVFP4_BLOCK_SIZE, weight_mse.NVFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    scales = amax / (weight_mse.NVFP4_MAX * per_tensor_scale.float())
    return scales.clamp(0.0, weight_mse.FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def _attr(template, names):
    """Read a wrapper attribute the same way the packer does."""
    return weight_mse._template_attr(template, names, "test attribute")


def _dequantize(tensor, dtype=torch.float32):
    """Dequantize through the wrapper's own helper; pinned torchao 0.17 spells it ``dequantize(output_dtype)``."""
    for name in ("dequantize", "to_dtype"):
        helper = getattr(tensor, name, None)
        if helper is not None:
            return helper(dtype)
    pytest.skip("the installed torchao NVFP4Tensor exposes no dequantization helper")


def _skip_without_e4m3():
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")


class _FakeNVFP4Tensor:
    """Minimal stand-in exposing exactly the surface the packer's validation reads off a real ``NVFP4Tensor``."""

    def __init__(self, rows=8, columns=64, **overrides):
        num_blocks = columns // weight_mse.NVFP4_BLOCK_SIZE
        padded_blocks = ((num_blocks + 3) // 4) * 4
        self.shape = torch.Size((rows, columns))
        self.device = torch.device("cpu")
        self.qdata = torch.zeros(rows, columns // 2, dtype=torch.uint8)
        self._scale_e4m3 = torch.zeros(128 * padded_blocks, dtype=torch.float8_e4m3fn)
        self._block_size = weight_mse.NVFP4_BLOCK_SIZE
        self._orig_dtype = torch.bfloat16
        self._per_tensor_scale = torch.ones(())
        self._act_per_tensor_scale = None
        self._is_swizzled_scales = True
        self.use_triton_kernel = False
        self.act_quant_kwargs = None
        for name, value in overrides.items():
            setattr(self, f"_{name}" if hasattr(self, f"_{name}") else name, value)

    def dim(self):
        return len(self.shape)


def _fake_template(rows=8, columns=64, drop=None, **overrides):
    """Build a stand-in template, optionally with one attribute perturbed or removed entirely."""
    template = _FakeNVFP4Tensor(rows=rows, columns=columns, **overrides)
    if drop is not None:
        for name in (drop, f"_{drop}"):
            if name in template.__dict__:
                delattr(template, name)
    return template


class _TensorNVFP4Template(torch.Tensor):
    """A ``torch.Tensor`` subclass shaped like a converted NVFP4 weight, for the runtime binding check.

    ``nvfp4_template_identity`` deliberately requires a real tensor, because a converted weight always is one:
    torchao's ``NVFP4Tensor`` is itself a tensor subclass carrying its buffers as attributes. This stand-in has the
    same shape -- the logical ``(M, K)`` shape is the tensor's own and the three buffers are instance attributes --
    without needing torchao installed.
    """


def _tensor_template(rows=8, columns=64, drop=None, **overrides):
    """Build a tensor-subclass stand-in template, optionally with one buffer perturbed or removed entirely."""
    template = torch.zeros(rows, columns, dtype=torch.bfloat16).as_subclass(_TensorNVFP4Template)
    padded_blocks = ((columns // weight_mse.NVFP4_BLOCK_SIZE + 3) // 4) * 4
    buffers = {
        "qdata": torch.zeros(rows, columns // 2, dtype=torch.uint8),
        "_scale_e4m3": torch.zeros(128 * padded_blocks, dtype=torch.float8_e4m3fn),
        "_per_tensor_scale": torch.ones(()),
    }
    for name, value in overrides.items():
        buffers["qdata" if name == "qdata" else f"_{name}"] = value
    for name, value in buffers.items():
        # ``drop`` is only ever a name no bare ``torch.Tensor`` carries, so a dropped buffer is genuinely absent
        # rather than shadowed by an attribute the base class happens to define.
        if drop is None or name.lstrip("_") != drop:
            setattr(template, name, value)
    return template


def _fake_weight(rows=8, columns=64):
    return torch.zeros(rows, columns, dtype=torch.bfloat16)


def _install_fake_backend(monkeypatch):
    """Point the packer at a backend whose kernels raise, so only validation can run."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("validation must fail before any torchao kernel is called")

    backend = weight_mse._TorchAOBackend(
        NVFP4Tensor=_FakeNVFP4Tensor,
        f32_to_f4_unpacked=_forbidden,
        f4_unpacked_to_f32=_forbidden,
        pack_uint4=_forbidden,
        to_blocked=_forbidden,
    )
    monkeypatch.setattr(weight_mse, "_resolve_torchao", lambda: backend)

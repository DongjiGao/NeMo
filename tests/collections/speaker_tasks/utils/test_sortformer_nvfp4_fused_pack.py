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

"""Tests for the fused Sortformer producer-to-NVFP4-pack prototype primitives."""

import inspect
import subprocess
import sys
import textwrap

import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_nvfp4_fused_pack as fused_pack

# Fraction of packed/scale bytes allowed to differ between the fused and reference paths. The producers agree to
# within an FP32 ULP -- the fused kernel and torch reduce the LayerNorm moments in a different order, and Triton's
# libdevice erf and torch's erf are not bit-identical -- so an element sitting exactly on a BF16 rounding boundary
# can round the other way and shift a single FP4 nibble or E4M3 scale byte. That is a ~2**-15 per-element event.
_MAX_BYTE_MISMATCH_FRACTION = 1e-3

# The exact signature of ``dequantize_nvfp4`` in the pinned MSLK 1.2 runtime. The tests bind positionally against it
# rather than guessing parameter names, so a runtime bump that changes the contract fails loudly instead of silently
# dequantizing something else.
_DEQUANTIZE_PARAMETERS = ("input_quantized", "scale", "global_scale", "group_size")
_DEQUANTIZE_GROUP_SIZE = 16


class TestLazyOptionalDependencies:
    """Importing the module and rejecting bad input must not pull in Triton or MSLK."""

    @pytest.mark.unit
    def test_import_does_not_resolve_triton_or_mslk(self):
        """A bare import of the module must not add triton or mslk to sys.modules."""
        script = textwrap.dedent(
            f"""
            import importlib.util
            import sys

            import torch  # noqa: F401  -- baseline: torch itself must not be blamed for a triton import

            def optional(modules):
                return sorted(n for n in modules if n == "triton" or n.startswith("triton.")
                              or n == "mslk" or n.startswith("mslk."))

            baseline = optional(sys.modules)
            spec = importlib.util.spec_from_file_location("_nvfp4_fused_pack_probe", {fused_pack.__file__!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            leaked = [n for n in optional(sys.modules) if n not in baseline]
            assert not leaked, leaked
            assert module.triton is None and module.tl is None
            assert module._mslk_quantize_nvfp4 is None
            """
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

    @pytest.mark.unit
    @pytest.mark.parametrize("entry_point", ["layer_norm", "scaled_gelu"])
    @pytest.mark.parametrize(
        "case,message",
        [
            ("rank", "rank 2"),
            ("dtype", "bfloat16"),
            ("contiguous", "contiguous"),
            ("device", "CUDA tensor"),
            ("k_multiple", "multiple of 64"),
            ("scale_numel", "exactly one element"),
            ("scale_dtype", "float32"),
        ],
    )
    def test_validation_is_actionable_and_resolves_nothing(self, monkeypatch, entry_point, case, message):
        """Every representative contract violation raises a named ValueError before any optional import."""

        def _forbidden(*args, **kwargs):
            raise AssertionError("validation must not resolve Triton or MSLK")

        monkeypatch.setattr(fused_pack, "_resolve_fused_backend", _forbidden)
        monkeypatch.setattr(fused_pack, "_resolve_mslk_quantize", _forbidden)

        producer = torch.zeros(8, 64, dtype=torch.bfloat16)
        vector = torch.zeros(64, dtype=torch.bfloat16)
        scale = torch.ones(1, dtype=torch.float32)
        if case == "rank":
            producer = torch.zeros(8, dtype=torch.bfloat16)
        elif case == "dtype":
            # FP16 is unsupported at both boundaries; FP32 is only accepted by the LayerNorm one, and that
            # asymmetry has its own tests below.
            producer = torch.zeros(8, 64, dtype=torch.float16)
        elif case == "contiguous":
            producer = torch.zeros(64, 8, dtype=torch.bfloat16).t()
        elif case == "k_multiple":
            producer = torch.zeros(8, 96, dtype=torch.bfloat16)
            vector = torch.zeros(96, dtype=torch.bfloat16)
        elif case == "scale_numel":
            scale = torch.ones(2, dtype=torch.float32)
        elif case == "scale_dtype":
            scale = torch.ones(1, dtype=torch.float64)

        # The "device" case needs no mutation: every tensor above is on CPU, and the CUDA requirement is the only
        # check left once the structural ones pass.
        for callable_ in _entry_points(entry_point):
            with pytest.raises(ValueError, match=message):
                if entry_point == "layer_norm":
                    callable_(producer, vector, scale, bias=None, eps=1e-5)
                else:
                    callable_(producer, scale, scale, bias=None)

    @pytest.mark.unit
    @pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
    def test_layer_norm_rejects_invalid_eps(self, monkeypatch, eps):
        """A non-positive or non-finite eps is rejected before any optional import."""

        def _forbidden(*args, **kwargs):
            raise AssertionError("validation must not resolve Triton or MSLK")

        monkeypatch.setattr(fused_pack, "_resolve_fused_backend", _forbidden)
        monkeypatch.setattr(fused_pack, "_resolve_mslk_quantize", _forbidden)

        producer = torch.zeros(8, 64, dtype=torch.bfloat16)
        vector = torch.zeros(64, dtype=torch.bfloat16)
        scale = torch.ones(1, dtype=torch.float32)
        for callable_ in _entry_points("layer_norm"):
            with pytest.raises(ValueError, match="eps must be"):
                callable_(producer, vector, scale, bias=None, eps=eps)


class TestProducerInputDtypes:
    """The two boundaries accept different producer dtypes, and neither of them accepts anything else."""

    @pytest.mark.unit
    def test_layer_norm_accepts_the_fp32_autocast_residual(self, monkeypatch):
        """An FP32 residual passes validation; only the CUDA requirement is left to reject this CPU call."""

        def _forbidden(*args, **kwargs):
            raise AssertionError("validation must not resolve Triton or MSLK")

        monkeypatch.setattr(fused_pack, "_resolve_fused_backend", _forbidden)
        monkeypatch.setattr(fused_pack, "_resolve_mslk_quantize", _forbidden)

        x = torch.zeros(8, 64, dtype=torch.float32)
        weight = torch.zeros(64, dtype=torch.bfloat16)
        scale = torch.ones(1, dtype=torch.float32)
        for callable_ in _entry_points("layer_norm"):
            with pytest.raises(ValueError, match="CUDA tensor"):
                callable_(x, weight, scale, bias=None, eps=1e-5)

    @pytest.mark.unit
    def test_scaled_gelu_stays_bf16_only(self, monkeypatch):
        """The scaled-GELU input is a ``_scaled_mm`` result, which is always BF16; FP32 is still rejected."""

        def _forbidden(*args, **kwargs):
            raise AssertionError("validation must not resolve Triton or MSLK")

        monkeypatch.setattr(fused_pack, "_resolve_fused_backend", _forbidden)
        monkeypatch.setattr(fused_pack, "_resolve_mslk_quantize", _forbidden)

        raw = torch.zeros(8, 64, dtype=torch.float32)
        scale = torch.ones(1, dtype=torch.float32)
        for callable_ in _entry_points("scaled_gelu"):
            with pytest.raises(ValueError, match="must be torch.bfloat16"):
                callable_(raw, scale, scale, bias=None)

    @pytest.mark.unit
    def test_accepted_dtypes_are_declared_per_boundary(self):
        assert fused_pack.LAYER_NORM_INPUT_DTYPES == (torch.bfloat16, torch.float32)
        assert fused_pack.SCALED_GELU_INPUT_DTYPES == (torch.bfloat16,)


class TestPackOutputShapes:
    """The declared MSLK output shapes, checkable without CUDA."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "m_dim,k_dim,expected_scales",
        [
            (128, 64, (128, 4)),
            (200, 128, (256, 8)),
            (1, 512, (128, 32)),
            (361728, 2048, (361728, 128)),
        ],
    )
    def test_shapes(self, m_dim, k_dim, expected_scales):
        data_shape, scale_shape = fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)
        assert data_shape == (m_dim, k_dim // 2)
        assert scale_shape == expected_scales

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim,k_dim", [(0, 64), (8, 0), (8, 96), (8, -64)])
    def test_rejects_unsupported_shapes(self, m_dim, k_dim):
        with pytest.raises(ValueError):
            fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)


class TestPackOutputAllocation:
    """The output buffers are allocated uninitialized, and no fill of any kind is issued before the launch."""

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim,k_dim", [(128, 64), (256, 128), (1024, 512)])
    def test_aligned_m_allocates_empty_and_launches_over_m_rows(self, monkeypatch, m_dim, k_dim):
        """For ``M % 128 == 0`` there are no padded rows, so the grid is ``M`` and nothing is initialized."""
        packed, scales, grid_rows = _allocate_without_fills(monkeypatch, m_dim, k_dim)
        data_shape, scale_shape = fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)
        assert (packed.shape, packed.dtype) == (data_shape, torch.uint8)
        assert (scales.shape, scales.dtype) == (scale_shape, torch.uint8)
        assert grid_rows == m_dim

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim,k_dim", [(1, 64), (200, 128), (300, 192), (11304, 512)])
    def test_unaligned_m_allocates_empty_and_launches_over_padded_rows(self, monkeypatch, m_dim, k_dim):
        """A padded tail is written by the same launch, so allocation still issues no fill and the grid grows."""
        packed, scales, grid_rows = _allocate_without_fills(monkeypatch, m_dim, k_dim)
        data_shape, scale_shape = fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)
        assert (packed.shape, packed.dtype) == (data_shape, torch.uint8)
        assert (scales.shape, scales.dtype) == (scale_shape, torch.uint8)
        assert grid_rows == scale_shape[0] > m_dim

    @pytest.mark.unit
    def test_the_separate_tail_fill_helper_is_gone(self):
        """The padded rows are a kernel responsibility now; no pre-launch tail fill may come back."""
        assert not hasattr(fused_pack, "_zero_padded_scale_tail")

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim,k_dim", [(1, 64), (200, 128), (300, 192), (11304, 512)])
    def test_padded_rows_complete_the_scale_buffer_exactly(self, m_dim, k_dim):
        """Valid and padded rows of the grid partition every blocked-scale byte, and nothing else.

        The scale store is 128x4-swizzled, so a padded row is not a contiguous slice; this replays the kernel's own
        address arithmetic to show that the launch over ``ceil(M / 128) * 128`` rows writes each byte of the buffer
        exactly once, which is what lets the buffer stay ``torch.empty``.
        """
        _, scale_shape = fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)
        padded_rows, padded_cols = scale_shape
        n_col_blocks = padded_cols // fused_pack.SCALE_TILE_COLS
        num_blocks = k_dim // fused_pack.NVFP4_BLOCK_SIZE

        valid = _swizzled_scale_indices(range(m_dim), k_dim, n_col_blocks)
        padded = _swizzled_scale_indices(range(m_dim, padded_rows), k_dim, n_col_blocks)

        assert len(valid) == m_dim * num_blocks
        assert len(padded) == (padded_rows - m_dim) * num_blocks
        assert not (valid & padded)
        assert valid | padded == set(range(padded_rows * padded_cols))


class TestFusedLaunchContract:
    """Both fused entry points must issue exactly one launch, over the padded scale rows, with ``M`` passed in."""

    @pytest.mark.unit
    @pytest.mark.parametrize("entry_point", ["layer_norm", "scaled_gelu"])
    @pytest.mark.parametrize("m_dim,k_dim", [(1, 64), (128, 128), (200, 192), (300, 128)])
    def test_single_launch_over_padded_rows(self, monkeypatch, entry_point, m_dim, k_dim):
        """The grid is the padded scale-row count and ``(M, K)`` are handed to the kernel positionally."""
        for name in ("float4_e2m1fn_x2", "float8_e4m3fn"):
            if not isinstance(getattr(torch, name, None), torch.dtype):
                pytest.skip(f"torch {torch.__version__} does not expose torch.{name}.")

        kernel = _RecordingKernel()
        monkeypatch.setattr(fused_pack, "triton", _TritonStub)
        monkeypatch.setattr(fused_pack, "_resolve_fused_backend", lambda *args, **kwargs: kernel)
        # The tensors below are CPU tensors, so only the device check has to stand aside; every structural check,
        # the real allocation and the real grid computation still run.
        monkeypatch.setattr(fused_pack, "_validate_devices", lambda *args, **kwargs: None)

        producer = torch.zeros(m_dim, k_dim, dtype=torch.bfloat16)
        vector = torch.zeros(k_dim, dtype=torch.bfloat16)
        scale = torch.ones(1, dtype=torch.float32)
        if entry_point == "layer_norm":
            fused_pack.layer_norm_nvfp4_pack_triton(producer, vector, scale, bias=vector, eps=1e-5)
        else:
            fused_pack.scaled_gelu_nvfp4_pack_triton(producer, scale, scale, bias=vector)

        _, scale_shape = fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)
        assert len(kernel.launches) == 1
        grid, args, kwargs = kernel.launches[0]
        assert grid == (scale_shape[0],)
        # Positions 6 and 7 are ``M`` and ``K`` in both kernels; the padded programs need ``M`` to know to skip.
        assert (args[6], args[7]) == (m_dim, k_dim)
        assert kwargs["BLOCK_K"] == _TritonStub.next_power_of_2(k_dim)


@pytest.mark.usefixtures("nvfp4_backend")
class TestFusedPackOnCuda:
    """Numerical and layout equivalence against the pinned MSLK pack. Requires CUDA, Triton and MSLK."""

    @pytest.mark.unit
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_layer_norm_reference_matches_torch_layer_norm(self, with_bias):
        """The reference path's producer is exactly ``F.layer_norm(x, (K,), weight, bias, eps)``."""
        x, weight, bias, scale = _layer_norm_case(m_dim=192, k_dim=128, with_bias=with_bias)
        quantize = fused_pack._resolve_mslk_quantize()

        got = fused_pack.layer_norm_nvfp4_pack_reference(x, weight, scale, bias=bias, eps=1e-5)
        producer = torch.nn.functional.layer_norm(x, (x.shape[1],), weight, bias, 1e-5)
        expected = quantize(producer.contiguous(), scale)

        assert producer.dtype == torch.bfloat16
        _assert_bytes_equal(got, expected)

    @pytest.mark.unit
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_layer_norm_fp32_reference_keeps_fp32_arithmetic_and_one_bf16_boundary(self, with_bias):
        """For the real FP32 residual the reference normalizes in FP32 and rounds to BF16 exactly once."""
        x, weight, bias, scale = _layer_norm_case(m_dim=192, k_dim=128, with_bias=with_bias, dtype=torch.float32)
        quantize = fused_pack._resolve_mslk_quantize()

        got = fused_pack.layer_norm_nvfp4_pack_reference(x, weight, scale, bias=bias, eps=1e-5)
        # Autocast widens the BF16 affine parameters and keeps the FP32 input, so this is the arithmetic the
        # ordinary pre-norm boundary performs; the quantized linear then consumes it at BF16.
        producer = torch.nn.functional.layer_norm(
            x, (x.shape[1],), weight.float(), None if bias is None else bias.float(), 1e-5
        )
        assert producer.dtype == torch.float32
        expected = quantize(producer.to(torch.bfloat16).contiguous(), scale)

        _assert_bytes_equal(got, expected)

    @pytest.mark.unit
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("m_dim,k_dim", [(128, 128), (200, 192)])
    def test_fused_layer_norm_matches_reference_pack_for_an_fp32_residual(self, with_bias, m_dim, k_dim):
        """The fused kernel reproduces the FP32-residual reference, in the same MSLK layout."""
        x, weight, bias, scale = _layer_norm_case(m_dim=m_dim, k_dim=k_dim, with_bias=with_bias, dtype=torch.float32)
        fused = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        reference = fused_pack.layer_norm_nvfp4_pack_reference(x, weight, scale, bias=bias, eps=1e-5)

        _assert_mslk_layout(fused[0], fused[1], m_dim, k_dim)
        _assert_pack_agrees(fused, reference)

    @pytest.mark.unit
    def test_fused_layer_norm_does_not_mutate_the_fp32_residual(self):
        """The residual is read, never cast in place or written back."""
        x, weight, bias, scale = _layer_norm_case(m_dim=200, k_dim=128, with_bias=True, dtype=torch.float32)
        before = x.clone()
        fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)

        assert x.dtype == torch.float32
        assert torch.equal(x, before)

    @pytest.mark.unit
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_scaled_gelu_reference_matches_torch_gelu(self, with_bias):
        """The reference path's producer is exactly ``F.gelu(raw * out_scale + bias, approximate='none')``."""
        raw, bias, out_scale, act_scale = _scaled_gelu_case(m_dim=192, k_dim=128, with_bias=with_bias)
        quantize = fused_pack._resolve_mslk_quantize()

        got = fused_pack.scaled_gelu_nvfp4_pack_reference(raw, out_scale, act_scale, bias=bias)
        producer = raw * out_scale
        if bias is not None:
            producer = producer + bias
        producer = torch.nn.functional.gelu(producer, approximate='none').to(torch.bfloat16)
        expected = quantize(producer.contiguous(), act_scale)

        _assert_bytes_equal(got, expected)

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim", [1, 128, 200, 300])
    @pytest.mark.parametrize("k_dim", [64, 128, 192])
    def test_fused_layer_norm_layout(self, m_dim, k_dim):
        """Fused LayerNorm output has the exact MSLK shapes, dtypes and padded-scale zeros."""
        x, weight, bias, scale = _layer_norm_case(m_dim=m_dim, k_dim=k_dim, with_bias=True)
        data, scales = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        _assert_mslk_layout(data, scales, m_dim, k_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize("m_dim", [1, 128, 200, 300])
    @pytest.mark.parametrize("k_dim", [64, 128, 192])
    def test_fused_scaled_gelu_layout(self, m_dim, k_dim):
        """Fused scaled-GELU output has the exact MSLK shapes, dtypes and padded-scale zeros."""
        raw, bias, out_scale, act_scale = _scaled_gelu_case(m_dim=m_dim, k_dim=k_dim, with_bias=True)
        data, scales = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=bias)
        _assert_mslk_layout(data, scales, m_dim, k_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("m_dim", [1, 128, 200, 300])
    @pytest.mark.parametrize("k_dim", [128, 192])
    def test_fused_layer_norm_matches_reference_pack(self, with_bias, m_dim, k_dim):
        """Fused LayerNorm packing agrees with the MSLK reference pack byte for byte, up to BF16 tie flips."""
        x, weight, bias, scale = _layer_norm_case(m_dim=m_dim, k_dim=k_dim, with_bias=with_bias)
        fused = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        reference = fused_pack.layer_norm_nvfp4_pack_reference(x, weight, scale, bias=bias, eps=1e-5)
        _assert_pack_agrees(fused, reference)

    @pytest.mark.unit
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("m_dim", [1, 128, 200, 300])
    @pytest.mark.parametrize("k_dim", [128, 192])
    def test_fused_scaled_gelu_matches_reference_pack(self, with_bias, m_dim, k_dim):
        """Fused scaled-GELU packing agrees with the MSLK reference pack byte for byte, up to BF16 tie flips."""
        raw, bias, out_scale, act_scale = _scaled_gelu_case(m_dim=m_dim, k_dim=k_dim, with_bias=with_bias)
        fused = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=bias)
        reference = fused_pack.scaled_gelu_nvfp4_pack_reference(raw, out_scale, act_scale, bias=bias)
        _assert_pack_agrees(fused, reference)

    @pytest.mark.unit
    def test_fused_layer_norm_dequantizes_like_the_reference(self):
        """Dequantized fused values stay finite and track the pinned MSLK oracle, no worse than its own error."""
        m_dim, k_dim = 200, 128
        x, weight, bias, _ = _layer_norm_case(m_dim=m_dim, k_dim=k_dim, with_bias=True)
        # A unit activation scale keeps the dequantized values on the producer's own scale, independently of
        # whether MSLK's dequantize helper folds the global scale back in.
        scale = torch.ones(1, dtype=torch.float32, device=x.device)

        fused = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        reference = fused_pack.layer_norm_nvfp4_pack_reference(x, weight, scale, bias=bias, eps=1e-5)
        producer = torch.nn.functional.layer_norm(x, (k_dim,), weight, bias, 1e-5).float()
        _assert_dequant_bound(fused, reference, producer, scale, m_dim, k_dim)

    @pytest.mark.unit
    def test_fused_scaled_gelu_dequantizes_like_the_reference(self):
        """Dequantized fused values stay finite and track the pinned MSLK oracle, no worse than its own error."""
        m_dim, k_dim = 200, 128
        raw, bias, out_scale, _ = _scaled_gelu_case(m_dim=m_dim, k_dim=k_dim, with_bias=True)
        act_scale = torch.ones(1, dtype=torch.float32, device=raw.device)

        fused = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=bias)
        reference = fused_pack.scaled_gelu_nvfp4_pack_reference(raw, out_scale, act_scale, bias=bias)
        producer = torch.nn.functional.gelu(raw * out_scale + bias, approximate='none').to(torch.bfloat16).float()
        _assert_dequant_bound(fused, reference, producer, act_scale, m_dim, k_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize("producer", ["layer_norm", "scaled_gelu"])
    def test_fused_all_zero_producer_packs_and_dequantizes_like_the_reference(self, producer):
        """An all-zero producer, whose 16-wide blocks have no finite amax, still packs and dequantizes as MSLK does."""
        m_dim, k_dim = 200, 128
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        if producer == "layer_norm":
            # Arbitrary input, but a zero weight and zero bias drive the normalized output to exactly zero.
            x, _, _, _ = _layer_norm_case(m_dim=m_dim, k_dim=k_dim, with_bias=False)
            zeros = torch.zeros(k_dim, dtype=torch.bfloat16, device="cuda")
            fused = fused_pack.layer_norm_nvfp4_pack_triton(x, zeros, scale, bias=zeros, eps=1e-5)
            reference = fused_pack.layer_norm_nvfp4_pack_reference(x, zeros, scale, bias=zeros, eps=1e-5)
        else:
            raw = torch.zeros(m_dim, k_dim, dtype=torch.bfloat16, device="cuda")
            out_scale = torch.full((1,), 0.37, device="cuda", dtype=torch.float32)
            fused = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, scale, bias=None)
            reference = fused_pack.scaled_gelu_nvfp4_pack_reference(raw, out_scale, scale, bias=None)

        _assert_mslk_layout(fused[0], fused[1], m_dim, k_dim)
        _assert_bytes_equal(fused, reference)

        # MSLK emits zero scale bytes and a nonzero packed sentinel for an all-zero block, but the pair still
        # dequantizes to exact zeros; assert the values, not the bytes.
        for values in (_dequantize(fused, scale, m_dim, k_dim), _dequantize(reference, scale, m_dim, k_dim)):
            assert torch.isfinite(values).all()
            assert torch.equal(values, torch.zeros_like(values))

    @pytest.mark.unit
    @pytest.mark.parametrize("producer", ["layer_norm", "scaled_gelu"])
    @pytest.mark.parametrize("m_dim,k_dim", [(1, 64), (200, 128), (300, 192)])
    def test_padded_scale_rows_are_written_by_the_kernel_not_by_the_allocation(
        self, monkeypatch, producer, m_dim, k_dim
    ):
        """Padded rows read back as zero even when the uninitialized buffer starts as garbage.

        ``torch.empty`` normally hands back recycled allocator memory, so a test on a fresh buffer could pass on
        accidental zeros. Poisoning the uint8 allocations makes the padded rows zero only if a program wrote them.
        """
        x, weight, bias, scale = _layer_norm_case(m_dim=m_dim, k_dim=k_dim, with_bias=True)
        raw, gelu_bias, out_scale, act_scale = _scaled_gelu_case(m_dim=m_dim, k_dim=k_dim, with_bias=True)
        # Warm the JIT before poisoning, so nothing Triton allocates while compiling is affected.
        fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=gelu_bias)

        real_empty = torch.empty

        def _poisoned_empty(*args, **kwargs):
            tensor = real_empty(*args, **kwargs)
            if tensor.dtype == torch.uint8:
                tensor.fill_(0xEF)
            return tensor

        monkeypatch.setattr(torch, "empty", _poisoned_empty)
        if producer == "layer_norm":
            data, scales = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        else:
            data, scales = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=gelu_bias)

        _assert_mslk_layout(data, scales, m_dim, k_dim)

    @pytest.mark.unit
    def test_fused_paths_are_deterministic(self):
        """Repeated fused calls on the same inputs produce byte-identical packs."""
        x, weight, bias, scale = _layer_norm_case(m_dim=200, k_dim=128, with_bias=True)
        raw, gelu_bias, out_scale, act_scale = _scaled_gelu_case(m_dim=200, k_dim=128, with_bias=True)

        first_ln = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        second_ln = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        _assert_bytes_equal(first_ln, second_ln)

        first_gelu = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=gelu_bias)
        second_gelu = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=gelu_bias)
        _assert_bytes_equal(first_gelu, second_gelu)

    @pytest.mark.unit
    def test_fused_paths_never_touch_the_reference_pack(self, monkeypatch):
        """With every reference entry point poisoned, the already-warm fused paths still succeed."""
        x, weight, bias, scale = _layer_norm_case(m_dim=200, k_dim=128, with_bias=True)
        raw, gelu_bias, out_scale, act_scale = _scaled_gelu_case(m_dim=200, k_dim=128, with_bias=True)
        # Warm up so the poisoning below cannot be satisfied by a lazy resolution that already happened.
        fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=gelu_bias)

        def _poisoned(*args, **kwargs):
            raise AssertionError("the fused path must not use the reference MSLK pack")

        monkeypatch.setattr(fused_pack, "_resolve_mslk_quantize", _poisoned)
        monkeypatch.setattr(fused_pack, "_mslk_quantize_nvfp4", _poisoned)
        monkeypatch.setattr(fused_pack, "layer_norm_nvfp4_pack_reference", _poisoned)
        monkeypatch.setattr(fused_pack, "scaled_gelu_nvfp4_pack_reference", _poisoned)

        data, scales = fused_pack.layer_norm_nvfp4_pack_triton(x, weight, scale, bias=bias, eps=1e-5)
        _assert_mslk_layout(data, scales, 200, 128)
        data, scales = fused_pack.scaled_gelu_nvfp4_pack_triton(raw, out_scale, act_scale, bias=gelu_bias)
        _assert_mslk_layout(data, scales, 200, 128)


@pytest.fixture
def nvfp4_backend():
    """Skip cleanly unless CUDA, Triton, MSLK and the FP4/FP8 torch dtypes are all available."""
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 packing requires CUDA.")
    for name in ("float4_e2m1fn_x2", "float8_e4m3fn"):
        if not isinstance(getattr(torch, name, None), torch.dtype):
            pytest.skip(f"torch {torch.__version__} does not expose torch.{name}.")
    pytest.importorskip("triton", reason="NVFP4 packing requires Triton.")
    pytest.importorskip("mslk.quantize.triton.fp4_quantize", reason="NVFP4 packing requires MSLK >= 1.2.")
    pytest.importorskip("mslk.quantize.triton.fp4_utils", reason="NVFP4 packing requires MSLK >= 1.2.")


def _entry_points(name):
    """Return the (reference, fused) pair for a producer, so validation is asserted on both."""
    if name == "layer_norm":
        return (fused_pack.layer_norm_nvfp4_pack_reference, fused_pack.layer_norm_nvfp4_pack_triton)
    return (fused_pack.scaled_gelu_nvfp4_pack_reference, fused_pack.scaled_gelu_nvfp4_pack_triton)


class _TritonStub:
    """Stands in for the lazily-imported ``triton`` module; output allocation only needs ``next_power_of_2``."""

    @staticmethod
    def next_power_of_2(value):
        return 1 << (value - 1).bit_length()


class _RecordingKernel:
    """Stands in for a JIT-wrapped Triton kernel and records the grid and arguments of every launch."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return _launch


def _allocate_without_fills(monkeypatch, m_dim, k_dim):
    """Allocate the pack outputs with every zero-filling entry point poisoned.

    ``torch.zeros``/``torch.zeros_like`` and the in-place ``zero_``/``fill_`` methods are made to raise for the
    duration of the call, so an allocation that reintroduced any pre-launch fill -- of the whole output or only of
    the padded scale tail -- fails here rather than silently costing an extra kernel launch per pack.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError("pack outputs must be allocated with torch.empty and never pre-filled")

    monkeypatch.setattr(fused_pack, "triton", _TritonStub)
    monkeypatch.setattr(torch, "zeros", _forbidden)
    monkeypatch.setattr(torch, "zeros_like", _forbidden)
    monkeypatch.setattr(torch.Tensor, "zero_", _forbidden)
    monkeypatch.setattr(torch.Tensor, "fill_", _forbidden)

    reference = torch.empty(0, dtype=torch.bfloat16)
    packed, scales, block_k, n_col_blocks, grid_rows = fused_pack._allocate_pack_outputs(reference, m_dim, k_dim)

    assert block_k == _TritonStub.next_power_of_2(k_dim)
    assert n_col_blocks == scales.shape[1] // fused_pack.SCALE_TILE_COLS
    return packed, scales, grid_rows


def _swizzled_scale_indices(rows, k_dim, n_col_blocks):
    """Replay the kernel's swizzled scale-store addressing and return the byte offsets written for ``rows``."""
    indices = set()
    for row in rows:
        row_in_tile = row % 128
        for block in range(k_dim // fused_pack.NVFP4_BLOCK_SIZE):
            tile_offset = (row // 128) * n_col_blocks + block // 4
            inner = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4 + block % 4
            indices.add(tile_offset * 512 + inner)
    return indices


def _layer_norm_case(m_dim, k_dim, with_bias, seed=0, dtype=torch.bfloat16):
    """Build a deterministic LayerNorm case on CUDA; the affine parameters stay BF16 for either input dtype."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m_dim, k_dim, generator=generator, device="cuda", dtype=torch.float32).to(dtype)
    weight = torch.randn(k_dim, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    bias = None
    if with_bias:
        bias = torch.randn(k_dim, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.full((1,), 1.7, device="cuda", dtype=torch.float32)
    return x, weight, bias, scale


def _scaled_gelu_case(m_dim, k_dim, with_bias, seed=1):
    """Build a deterministic BF16 scaled-GELU case on CUDA."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(m_dim, k_dim, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    bias = None
    if with_bias:
        bias = torch.randn(k_dim, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    out_scale = torch.full((1,), 0.37, device="cuda", dtype=torch.float32)
    act_scale = torch.full((1,), 2.3, device="cuda", dtype=torch.float32)
    return raw, bias, out_scale, act_scale


def _assert_mslk_layout(data, scales, m_dim, k_dim):
    """Assert the MSLK packed-data and blocked-scale shapes, dtypes and zero padding."""
    data_shape, scale_shape = fused_pack.nvfp4_pack_output_shapes(m_dim, k_dim)
    assert data.shape == data_shape
    assert data.dtype == torch.float4_e2m1fn_x2
    assert data.is_contiguous()
    assert scales.shape == scale_shape
    assert scales.dtype == torch.float8_e4m3fn
    assert scales.is_contiguous()

    if scale_shape[0] > m_dim:
        # Rows padding M up to a multiple of 128 are written as zero by the padded programs of the same launch.
        # Their bytes are swizzled, not a contiguous slice, so replay the kernel addressing and check exactly them.
        n_col_blocks = scale_shape[1] // fused_pack.SCALE_TILE_COLS
        offsets = sorted(_swizzled_scale_indices(range(m_dim, scale_shape[0]), k_dim, n_col_blocks))
        flat = scales.view(torch.uint8).reshape(-1)
        padded_bytes = flat[torch.tensor(offsets, dtype=torch.long, device=flat.device)]
        assert torch.equal(padded_bytes, torch.zeros_like(padded_bytes)), "padded blocked-scale rows must be zero"


def _byte_mismatch_fraction(left, right):
    """Fraction of differing bytes between two same-shaped packed/scale tensors."""
    left_bytes = left.view(torch.uint8)
    right_bytes = right.view(torch.uint8)
    return float((left_bytes != right_bytes).sum().item()) / max(left_bytes.numel(), 1)


def _assert_bytes_equal(got, expected):
    """Assert two ``(data, scales)`` pairs are byte-identical."""
    for name, left, right in (("data", got[0], expected[0]), ("scales", got[1], expected[1])):
        assert left.shape == right.shape, name
        assert left.dtype == right.dtype, name
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8)), name


def _assert_pack_agrees(fused, reference):
    """Assert the fused pack matches the MSLK reference pack in format exactly and in content near-exactly."""
    assert fused[0].shape == reference[0].shape
    assert fused[0].dtype == reference[0].dtype == torch.float4_e2m1fn_x2
    assert fused[1].shape == reference[1].shape
    assert fused[1].dtype == reference[1].dtype == torch.float8_e4m3fn

    data_mismatch = _byte_mismatch_fraction(fused[0], reference[0])
    scale_mismatch = _byte_mismatch_fraction(fused[1], reference[1])
    assert data_mismatch <= _MAX_BYTE_MISMATCH_FRACTION, f"packed byte mismatch fraction {data_mismatch}"
    assert scale_mismatch <= _MAX_BYTE_MISMATCH_FRACTION, f"scale byte mismatch fraction {scale_mismatch}"


def _assert_dequant_bound(fused, reference, producer, global_scale, m_dim, k_dim):
    """Dequantize both packs with MSLK and hold the fused path to the pinned reference, not to an absolute bound.

    The pinned MSLK pack is the numerical oracle here: whatever mean relative error NVFP4 costs on a given producer
    is the format's cost, not a fused-kernel defect, so the only meaningful tensor-level contracts are that the
    fused values stay finite, are no worse than the reference, and stay on top of the reference. End-to-end
    diarization accuracy is gated separately by the matched final DER, not by a tensor error percentage.
    """
    fused_values = _dequantize(fused, global_scale, m_dim, k_dim)
    reference_values = _dequantize(reference, global_scale, m_dim, k_dim)

    assert torch.isfinite(fused_values).all(), "fused dequantized values contain non-finite entries"
    assert torch.isfinite(reference_values).all(), "reference dequantized values contain non-finite entries"

    magnitude = producer.abs().mean().clamp_min(1e-6)
    fused_error = (fused_values - producer).abs().mean() / magnitude
    reference_error = (reference_values - producer).abs().mean() / magnitude

    assert (
        fused_error <= reference_error * 1.05 + 1e-6
    ), f"fused error {fused_error.item()} exceeds MSLK reference error {reference_error.item()}"
    # The two packs are already required to agree near-byte-exactly, so their dequantized values must coincide.
    assert (fused_values - reference_values).abs().mean() <= 1e-6 * magnitude


def _dequantize(pack, global_scale, m_dim, k_dim):
    """Call the pinned MSLK ``dequantize_nvfp4`` on a ``(data, scales)`` pack and reshape to ``(M, K)``."""
    from mslk.quantize.triton.fp4_utils import dequantize_nvfp4

    signature = inspect.signature(dequantize_nvfp4)
    parameters = list(signature.parameters.values())
    names = tuple(parameter.name for parameter in parameters)
    if names != _DEQUANTIZE_PARAMETERS:
        pytest.fail(f"pinned MSLK dequantize_nvfp4{signature} changed: expected parameters {_DEQUANTIZE_PARAMETERS}")
    if any(parameter.default is not parameter.empty for parameter in parameters[:3]):
        pytest.fail(f"pinned MSLK dequantize_nvfp4{signature} changed: first three parameters must be required")
    if parameters[3].default != _DEQUANTIZE_GROUP_SIZE:
        pytest.fail(f"pinned MSLK dequantize_nvfp4{signature} changed: group_size default must be 16")

    data, scales = pack
    # The pinned helper masks nibbles with ``torch.bitwise_and``, which rejects ``torch.float4_e2m1fn_x2``; the
    # packed bytes have to be handed over as raw uint8.
    return dequantize_nvfp4(data.view(torch.uint8), scales, global_scale).float().reshape(m_dim, k_dim)

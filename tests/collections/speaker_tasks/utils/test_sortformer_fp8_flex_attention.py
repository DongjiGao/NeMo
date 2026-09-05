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

"""Tests for the opt-in FP8 (``float8_e4m3fn``) Triton FlexAttention Sortformer attention backend."""

import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from nemo.collections.asr.parts.utils import sortformer_fp8_flex_attention
from nemo.collections.asr.parts.utils.sortformer_attention_backends import (
    FLEX_BACKEND,
    SUPPORTED_ATTENTION_BACKENDS,
    attention_backend_cache_identity,
    attention_backend_info,
    configure_attention_backend,
    validate_attention_backend,
)
from nemo.collections.asr.parts.utils.sortformer_fp8_flex_attention import (
    BLOCK_MASK_TENSOR_FIELDS,
    FP8_DTYPE,
    FP8_FLEX_BACKEND,
    FP8_FLEX_CUSTOM_OP_NAME,
    SUPPORTED_CAPABILITY_MAJORS,
    fp8_flex_attention,
    fp8_flex_backend_info,
    prepare_fp8_flex_valid_lengths,
    validate_fp8_flex_attention_config,
    validate_fp8_flex_inference_mode,
)

BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = 3, 128, 4, 64
# A direct E4M3 cast of Q/K/V is a lossy but bounded approximation of the BF16 kernel, so these bounds are
# diagnostic drift detectors, not accuracy gates: final DER, not either number, is the accuracy decision.
# Measured max abs deviation of this helper against BF16 FlexAttention on the pinned B6000 stack at the
# shapes below: 1.09e-1 (a single element; the bulk of the tensor is two orders of magnitude closer). The
# threshold keeps roughly 1.4x margin over that measured maximum -- enough that BF16 output rounding and
# reference-path differences cannot flip it, while a regression that meaningfully widens the cast error
# still fails. It does not claim FP8 is bit exact.
FP8_BF16_PARITY_ATOL = 1.5e-1
# Compiled and eager FP8 differ by more than the BF16 comparison does: Inductor tiles the FP8 accumulation
# differently, and the measured max abs compiled-vs-eager deviation on the same pinned stack was 1.09e-1.
FP8_COMPILE_PARITY_ATOL = 2e-1

_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
FP8_FLEX_SUPPORTED = _CAPABILITY is not None and _CAPABILITY[0] in SUPPORTED_CAPABILITY_MAJORS

CUDA_FP8_FLEX = pytest.mark.skipif(not FP8_FLEX_SUPPORTED, reason="requires CUDA and a Blackwell GPU")

# Sentinel for "delete this attribute" in :class:`_MaskStandIn`.
_MISSING = object()


class _DummyEncoder:
    """Stand-in for a TransformerEncoder that only exposes the backend setter."""

    def __init__(self):
        self.backend = FLEX_BACKEND

    def set_attention_backend(self, backend):
        self.backend = validate_attention_backend(backend)
        return self.backend


class _DummyModel:
    def __init__(self, encoder=None, transformer_encoder=None):
        self.encoder = encoder
        self.transformer_encoder = transformer_encoder


class _MaskStandIn:
    """A duck-typed ``BlockMask`` whose individual fields can be corrupted or removed."""

    def __init__(self, block_mask, **overrides):
        for name in BLOCK_MASK_TENSOR_FIELDS + ("seq_lengths", "BLOCK_SIZE", "mask_mod"):
            setattr(self, name, getattr(block_mask, name))
        for name, value in overrides.items():
            if value is _MISSING:
                delattr(self, name)
            else:
                setattr(self, name, value)


def _padding_block_mask(lengths, device, seq_len=SEQ_LEN):
    """Exactly the per-sample key-padding block mask ``TransformerEncoder.forward_internal`` builds."""
    length_tensor = torch.tensor(lengths, device=device, dtype=torch.int64)

    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < length_tensor[b]

    return create_block_mask(pad_mask, B=len(lengths), H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def _valid_lengths(lengths, device, seq_len=SEQ_LEN):
    """The ``(B,)`` int32 tensor the encoder prepares once per forward and shares across layers."""
    return prepare_fp8_flex_valid_lengths(torch.tensor(lengths, device=device, dtype=torch.int64), seq_len)


def _qkv(device, dtype=torch.bfloat16, seq_len=SEQ_LEN, head_dim=HEAD_DIM, seed=0):
    """``(B, H, T, D)`` Q/K/V exactly as ``forward_from_qkv`` hands them to the backend."""
    torch.manual_seed(seed)
    shape = (BATCH, NUM_HEADS, seq_len, head_dim)
    return tuple(torch.randn(shape, device=device, dtype=dtype) for _ in range(3))


def _op_args(query, key, value, block_mask, valid_lengths):
    """The positional argument list :func:`fp8_flex_attention` builds for the custom operator."""
    return (
        query,
        key,
        value,
        valid_lengths,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        block_mask.q_num_blocks,
        block_mask.q_indices,
        block_mask.full_q_num_blocks,
        block_mask.full_q_indices,
        block_mask.seq_lengths[0],
        block_mask.seq_lengths[1],
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
    )


def _count_opaque_op_nodes(targets):
    """Count FX ``call_function`` targets that are the FP8 custom operator.

    Dynamo records the ``OpOverloadPacket`` for a ``torch.library.custom_op`` call; older/newer tracers may
    record the resolved ``.default`` overload instead. Both denote exactly one opaque node.
    """
    packet = torch.ops.nemo_sortformer.fp8_flex_attention
    return sum(target is packet or target is packet.default for target in targets)


def _capture_boundary(monkeypatch):
    """Replace the compiled private root with a recorder; also stand down the device admission check."""
    captured = {}

    def fake_boundary(*args):
        captured.setdefault("calls", []).append(args)
        captured["args"] = args
        return torch.zeros(args[0].shape, dtype=torch.bfloat16, device=args[0].device)

    monkeypatch.setattr(sortformer_fp8_flex_attention, "_fp8_flex_attention_compiled", fake_boundary)
    # The tensorized boundary contract is device independent; only the device admission check stands down.
    monkeypatch.setattr(sortformer_fp8_flex_attention, "_validate_fp8_flex_device", lambda query: None)
    return captured


class TestBackendSelection:
    @pytest.mark.unit
    def test_backend_is_registered(self):
        assert FP8_FLEX_BACKEND == "fp8_flex"
        assert FP8_FLEX_BACKEND in SUPPORTED_ATTENTION_BACKENDS
        assert validate_attention_backend(FP8_FLEX_BACKEND) == FP8_FLEX_BACKEND

    @pytest.mark.unit
    def test_default_selection_is_unchanged(self):
        assert validate_attention_backend(None) == FLEX_BACKEND
        assert validate_attention_backend(FLEX_BACKEND) == FLEX_BACKEND

    @pytest.mark.unit
    @pytest.mark.parametrize("backend", ["fp8", "FP8_FLEX", "fp8-flex", "fp8_flex ", "flex_fp8", "fp8flex"])
    def test_misspelled_backend_raises(self, backend):
        with pytest.raises(ValueError, match="is not supported"):
            validate_attention_backend(backend)

    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ["rope", "abs_pos", "no_pos"])
    def test_dense_no_score_mod_configs_are_accepted(self, self_attention_model):
        validate_fp8_flex_attention_config("full", self_attention_model)

    @pytest.mark.unit
    def test_causal_mode_is_rejected(self):
        with pytest.raises(ValueError, match="attn_mode='full'"):
            validate_fp8_flex_attention_config("causal", "rope")

    @pytest.mark.unit
    def test_rel_pos_score_mod_is_rejected(self):
        with pytest.raises(ValueError, match="relative-position"):
            validate_fp8_flex_attention_config("full", "rel_pos")

    @pytest.mark.unit
    def test_training_mode_is_rejected(self):
        """A module left in training mode must fail even under no_grad, where the grad check cannot fire."""
        validate_fp8_flex_inference_mode(False)
        with pytest.raises(RuntimeError, match="inference-only"):
            validate_fp8_flex_inference_mode(True)


class TestModelWiring:
    @pytest.mark.unit
    def test_backend_applied_to_every_transformer_encoder(self):
        model = _DummyModel(encoder=_DummyEncoder(), transformer_encoder=_DummyEncoder())
        applied = configure_attention_backend(model, FP8_FLEX_BACKEND)
        assert applied == {"encoder": FP8_FLEX_BACKEND, "transformer_encoder": FP8_FLEX_BACKEND}
        assert model.encoder.backend == model.transformer_encoder.backend == FP8_FLEX_BACKEND

    @pytest.mark.unit
    def test_misspelled_backend_raises_before_touching_the_model(self):
        model = _DummyModel(encoder=_DummyEncoder())
        with pytest.raises(ValueError, match="is not supported"):
            configure_attention_backend(model, "fp8")
        assert model.encoder.backend == FLEX_BACKEND

    @pytest.mark.unit
    def test_non_default_backend_without_any_target_raises(self):
        model = _DummyModel(encoder=object(), transformer_encoder=None)
        with pytest.raises(ValueError, match="no attention-backend selectable"):
            configure_attention_backend(model, FP8_FLEX_BACKEND)

    @pytest.mark.unit
    def test_cache_identity_is_distinct_and_non_null(self):
        assert attention_backend_cache_identity(FP8_FLEX_BACKEND) == FP8_FLEX_BACKEND
        # Unchanged defaults: caches written before any backend option existed keep matching a default run.
        assert attention_backend_cache_identity(FLEX_BACKEND) is None
        assert attention_backend_cache_identity(None) is None

    @pytest.mark.unit
    def test_backend_info_identifies_fp8_flex(self):
        info = fp8_flex_backend_info()
        assert info["backend"] == FP8_FLEX_BACKEND
        assert info["fp8_dtype"] == str(FP8_DTYPE)
        assert info["flex_kernel_options"] == {"BACKEND": "TRITON"}
        assert info["inference_only"] is True
        assert info["supported_capability_majors"] == list(SUPPORTED_CAPABILITY_MAJORS)
        assert set(info) >= {"torch_version", "cuda_version", "cuda_available", "device_name", "compute_capability"}
        # Must never be mislabeled as FlashAttention.
        assert "flash_attn_version" not in info

    @pytest.mark.unit
    def test_backend_info_names_the_custom_op_without_claiming_a_custom_kernel(self):
        """The operator is a compilation boundary; the attention kernel is still PyTorch FlexAttention."""
        info = fp8_flex_backend_info()
        assert info["custom_op"] == FP8_FLEX_CUSTOM_OP_NAME == "nemo_sortformer::fp8_flex_attention"
        assert info["custom_attention_kernel"] is False
        assert "flex_attention" in info["attention_kernel"] and "Triton" in info["attention_kernel"]
        assert "boundary" in info["custom_op_role"]
        for claim in ("cutlass", "cuda kernel", "flash"):
            assert claim not in repr(info).lower()

    @pytest.mark.unit
    def test_dispatched_backend_info_matches_the_selected_backend(self):
        assert attention_backend_info(FP8_FLEX_BACKEND)["backend"] == FP8_FLEX_BACKEND
        assert attention_backend_info(FLEX_BACKEND)["backend"] == FLEX_BACKEND

    @pytest.mark.unit
    def test_backend_info_does_not_import_flash_attn(self):
        """The FP8 path must be loggable without resolving the FlashAttention optional dependency."""
        imported_before = "flash_attn" in sys.modules
        fp8_flex_backend_info()
        assert ("flash_attn" in sys.modules) is imported_before

    @pytest.mark.unit
    def test_custom_op_is_registered(self):
        assert hasattr(torch.ops.nemo_sortformer, "fp8_flex_attention")


class TestValidLengthPreparation:
    @pytest.mark.unit
    def test_lengths_are_converted_once_to_contiguous_int32(self):
        lengths = torch.tensor([SEQ_LEN, 41, 0], dtype=torch.int64)
        valid_lengths = prepare_fp8_flex_valid_lengths(lengths, SEQ_LEN)
        assert valid_lengths.dtype == torch.int32
        assert valid_lengths.is_contiguous()
        assert valid_lengths.tolist() == [SEQ_LEN, 41, 0]

    @pytest.mark.unit
    def test_noncontiguous_input_is_made_contiguous(self):
        lengths = torch.tensor([[SEQ_LEN, 0], [41, 0], [7, 0]], dtype=torch.int64)[:, 0]
        assert not lengths.is_contiguous()
        assert prepare_fp8_flex_valid_lengths(lengths, SEQ_LEN).is_contiguous()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "length, seq_len",
        [
            (torch.tensor([SEQ_LEN + 1], dtype=torch.int64), SEQ_LEN),
            (torch.tensor([-1], dtype=torch.int64), SEQ_LEN),
            (torch.tensor([[4, 4]], dtype=torch.int64), SEQ_LEN),
            (torch.tensor([4.0]), SEQ_LEN),
            (torch.tensor([4], dtype=torch.int64), 0),
            (torch.tensor([4], dtype=torch.int64), 4.0),
        ],
    )
    def test_malformed_lengths_are_rejected(self, length, seq_len):
        with pytest.raises(ValueError):
            prepare_fp8_flex_valid_lengths(length, seq_len)

    @pytest.mark.unit
    def test_non_tensor_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            prepare_fp8_flex_valid_lengths([4, 4], SEQ_LEN)

    @pytest.mark.unit
    def test_value_range_check_is_skipped_while_tracing(self):
        """The range check is a host read, so it must not appear in a traced graph."""

        @torch.compile(fullgraph=True, dynamic=True)
        def to_valid_lengths(x, lengths):
            return prepare_fp8_flex_valid_lengths(lengths, x.shape[1])

        out = to_valid_lengths(torch.zeros(2, SEQ_LEN, 4), torch.tensor([SEQ_LEN, 5], dtype=torch.int64))
        assert out.dtype == torch.int32


class TestInputContract:
    """Contract failures reportable without a GPU: everything but the device/capability check."""

    def _cpu_inputs(self, dtype=torch.bfloat16, head_dim=HEAD_DIM):
        shape = (BATCH, NUM_HEADS, SEQ_LEN, head_dim)
        q, k, v = (torch.zeros(shape, dtype=dtype) for _ in range(3))
        return q, k, v, _padding_block_mask([SEQ_LEN, 41, 7], "cpu"), _valid_lengths([SEQ_LEN, 41, 7], "cpu")

    @pytest.mark.unit
    def test_cpu_tensors_are_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        with pytest.raises(RuntimeError, match="requires CUDA"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_no_optional_dependency_is_imported_by_the_contract_checks(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        imported_before = "flash_attn" in sys.modules
        with pytest.raises(RuntimeError):
            fp8_flex_attention(q, k, v, block_mask, lengths)
        assert ("flash_attn" in sys.modules) is imported_before

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_non_bf16_is_rejected(self, dtype):
        q, k, v, block_mask, lengths = self._cpu_inputs(dtype=dtype)
        with pytest.raises(ValueError, match="bfloat16"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_non_tensor_is_rejected(self):
        _, k, v, block_mask, lengths = self._cpu_inputs()
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            fp8_flex_attention(None, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_rank_mismatch_is_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        with pytest.raises(ValueError, match="rank 4"):
            fp8_flex_attention(q[0], k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_shape_mismatch_is_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        with pytest.raises(ValueError, match="must share one shape"):
            fp8_flex_attention(q, k[:, :, :-1], v, block_mask, lengths)

    @pytest.mark.unit
    def test_non_head_dim_contiguous_input_is_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        transposed = torch.zeros(BATCH, NUM_HEADS, HEAD_DIM, SEQ_LEN, dtype=torch.bfloat16).transpose(-2, -1)
        assert transposed.shape == q.shape and transposed.stride(-1) != 1
        with pytest.raises(ValueError, match="head dimension"):
            fp8_flex_attention(q, transposed, v, block_mask, lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize("head_dim", [8, 24, 144])
    def test_unsupported_head_dim_is_rejected(self, head_dim):
        q, k, v, block_mask, lengths = self._cpu_inputs(head_dim=head_dim)
        with pytest.raises(ValueError, match="head dim"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_missing_block_mask_is_rejected(self):
        q, k, v, _, lengths = self._cpu_inputs()
        with pytest.raises(ValueError, match="block mask"):
            fp8_flex_attention(q, k, v, None, lengths)

    @pytest.mark.unit
    def test_grad_requiring_inputs_are_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        q = q.detach().requires_grad_(True)
        with pytest.raises(RuntimeError, match="inference-only"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_no_grad_bypasses_the_training_guard(self):
        """Under no_grad the guard must not fire; the device check is what stops a CPU run."""
        q, k, v, block_mask, lengths = self._cpu_inputs()
        q = q.detach().requires_grad_(True)
        with torch.no_grad(), pytest.raises(RuntimeError, match="requires CUDA"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_mixed_devices_are_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        with pytest.raises(ValueError, match="is on device"):
            fp8_flex_attention(q.cuda(), k, v, block_mask, lengths)

    @pytest.mark.unit
    @pytest.mark.skipif(
        not torch.cuda.is_available() or (_CAPABILITY is not None and _CAPABILITY[0] in SUPPORTED_CAPABILITY_MAJORS),
        reason="requires a CUDA device that is not Blackwell",
    )
    def test_unsupported_capability_is_rejected(self):
        q, k, v, _, _ = self._cpu_inputs()
        block_mask = _padding_block_mask([SEQ_LEN, 41, 7], "cuda")
        lengths = _valid_lengths([SEQ_LEN, 41, 7], "cuda")
        with pytest.raises(RuntimeError, match="compute capability"):
            fp8_flex_attention(q.cuda(), k.cuda(), v.cuda(), block_mask, lengths)


class TestValidLengthContract:
    """The per-sample lengths are validated before dispatch, exactly like Q/K/V and the mask."""

    def _cpu_inputs(self):
        q, k, v = _qkv("cpu")
        return q, k, v, _padding_block_mask([SEQ_LEN, 41, 7], "cpu")

    @pytest.mark.unit
    def test_missing_lengths_are_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match="valid key lengths"):
            fp8_flex_attention(q, k, v, block_mask, None)

    @pytest.mark.unit
    def test_non_tensor_lengths_are_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match="valid_lengths must be a torch.Tensor"):
            fp8_flex_attention(q, k, v, block_mask, [SEQ_LEN, 41, 7])

    @pytest.mark.unit
    def test_non_int32_lengths_are_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match="must be int32"):
            fp8_flex_attention(q, k, v, block_mask, torch.tensor([SEQ_LEN, 41, 7], dtype=torch.int64))

    @pytest.mark.unit
    def test_batch_mismatch_is_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match="matching the query batch"):
            fp8_flex_attention(q, k, v, block_mask, _valid_lengths([SEQ_LEN, 41], "cpu"))

    @pytest.mark.unit
    def test_rank_mismatch_is_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match="matching the query batch"):
            fp8_flex_attention(q, k, v, block_mask, torch.zeros(BATCH, 2, dtype=torch.int32))

    @pytest.mark.unit
    def test_noncontiguous_lengths_are_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        lengths = torch.zeros(BATCH, 2, dtype=torch.int32)[:, 0]
        with pytest.raises(ValueError, match="must be contiguous"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_out_of_range_lengths_are_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match=r"must lie in \[0, 128\]"):
            fp8_flex_attention(q, k, v, block_mask, torch.tensor([SEQ_LEN + 1, 41, 7], dtype=torch.int32))

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_lengths_on_another_device_are_rejected(self):
        q, k, v, block_mask = self._cpu_inputs()
        with pytest.raises(ValueError, match="valid_lengths is on device"):
            fp8_flex_attention(q, k, v, block_mask, _valid_lengths([SEQ_LEN, 41, 7], "cuda"))


class TestBlockMaskContract:
    """Every mask tensor field is checked before dispatch; nothing malformed reaches the operator."""

    def _cpu_inputs(self):
        q, k, v = _qkv("cpu")
        return q, k, v, _padding_block_mask([SEQ_LEN, 41, 7], "cpu"), _valid_lengths([SEQ_LEN, 41, 7], "cpu")

    @pytest.mark.unit
    def test_the_real_mask_passes_every_structural_check(self, monkeypatch):
        captured = _capture_boundary(monkeypatch)
        q, k, v, block_mask, lengths = self._cpu_inputs()
        fp8_flex_attention(q, k, v, block_mask, lengths)
        assert "args" in captured

    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["kv_num_blocks", "kv_indices"])
    def test_missing_required_mask_field_is_rejected(self, field):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        stand_in = _MaskStandIn(block_mask, **{field: _MISSING})
        with pytest.raises(ValueError, match=f"block_mask.{field}"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["kv_num_blocks", "kv_indices"])
    def test_non_tensor_required_mask_field_is_rejected(self, field):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        stand_in = _MaskStandIn(block_mask, **{field: [0, 1]})
        with pytest.raises(ValueError, match=f"block_mask.{field}"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field", ["full_kv_num_blocks", "full_kv_indices", "q_num_blocks", "q_indices", "full_q_num_blocks"]
    )
    def test_half_present_optional_mask_pairs_are_rejected(self, field):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        if getattr(block_mask, field) is None:
            pytest.skip(f"this block mask has no {field} to drop")
        stand_in = _MaskStandIn(block_mask, **{field: None})
        with pytest.raises(ValueError, match="both present or both absent"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    def test_float_mask_field_is_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        stand_in = _MaskStandIn(block_mask, kv_indices=block_mask.kv_indices.to(torch.float32))
        with pytest.raises(ValueError, match="must be an integer tensor"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    def test_low_rank_kv_indices_is_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        stand_in = _MaskStandIn(block_mask, kv_indices=block_mask.kv_indices.reshape(-1))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    def test_mask_batch_mismatch_is_rejected(self):
        q, k, v, _, lengths = self._cpu_inputs()
        block_mask = _padding_block_mask([SEQ_LEN, 41], "cpu")
        with pytest.raises(ValueError, match="broadcast over the query batch"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    def test_sequence_length_mismatch_is_rejected(self):
        q, k, v, _, lengths = self._cpu_inputs()
        block_mask = _padding_block_mask([SEQ_LEN, 41, 7], "cpu", seq_len=2 * SEQ_LEN)
        with pytest.raises(ValueError, match="seq_lengths"):
            fp8_flex_attention(q, k, v, block_mask, lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["seq_lengths", "BLOCK_SIZE"])
    def test_malformed_context_pair_is_rejected(self, field):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        stand_in = _MaskStandIn(block_mask, **{field: (SEQ_LEN,)})
        with pytest.raises(ValueError, match="pair of ints"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    def test_non_positive_block_size_is_rejected(self):
        q, k, v, block_mask, lengths = self._cpu_inputs()
        stand_in = _MaskStandIn(block_mask, BLOCK_SIZE=(0, 128))
        with pytest.raises(ValueError, match="two positive ints"):
            fp8_flex_attention(q, k, v, stand_in, lengths)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_mask_on_another_device_is_rejected(self):
        q, k, v, _, lengths = self._cpu_inputs()
        block_mask = _padding_block_mask([SEQ_LEN, 41, 7], "cuda")
        with pytest.raises(ValueError, match="block_mask.kv_num_blocks is on device"):
            fp8_flex_attention(q, k, v, block_mask, lengths)


class TestOperatorContract:
    """Only tensors and ints cross the custom-operator boundary, and they carry the whole mask."""

    LENGTHS = [SEQ_LEN, 41, 7]

    @pytest.mark.unit
    def test_operator_receives_only_tensors_and_ints(self, monkeypatch):
        captured = _capture_boundary(monkeypatch)
        q, k, v = _qkv("cpu")
        block_mask = _padding_block_mask(self.LENGTHS, "cpu")
        lengths = _valid_lengths(self.LENGTHS, "cpu")
        fp8_flex_attention(q, k, v, block_mask, lengths)

        for arg in captured["args"]:
            assert arg is None or isinstance(arg, (torch.Tensor, int)), f"non-tensor state crossed: {arg!r}"
        assert not any(isinstance(arg, BlockMask) for arg in captured["args"])

    @pytest.mark.unit
    def test_operator_arguments_are_the_callers_mask_and_lengths(self, monkeypatch):
        captured = _capture_boundary(monkeypatch)
        q, k, v = _qkv("cpu")
        block_mask = _padding_block_mask(self.LENGTHS, "cpu")
        lengths = _valid_lengths(self.LENGTHS, "cpu")
        fp8_flex_attention(q, k, v, block_mask, lengths)

        expected = _op_args(q, k, v, block_mask, lengths)
        assert len(captured["args"]) == len(expected)
        for actual, want in zip(captured["args"], expected):
            if isinstance(want, torch.Tensor):
                assert torch.equal(actual, want)
            else:
                assert actual == want
        # The eight BlockMask tensor fields are threaded through positions 4..11 in BlockMask's own order.
        assert len(BLOCK_MASK_TENSOR_FIELDS) == 8
        for offset, name in enumerate(BLOCK_MASK_TENSOR_FIELDS):
            field = getattr(block_mask, name)
            actual = captured["args"][4 + offset]
            assert (actual is None) == (field is None)
            if field is not None:
                assert torch.equal(actual, field)

    @pytest.mark.unit
    def test_structurally_identical_calls_share_one_tensorized_contract(self, monkeypatch):
        """Two distinct BlockMask objects, two distinct length tensors: the operator sees the same state."""
        captured = _capture_boundary(monkeypatch)
        q, k, v = _qkv("cpu")
        for _ in range(3):
            fp8_flex_attention(q, k, v, _padding_block_mask(self.LENGTHS, "cpu"), _valid_lengths(self.LENGTHS, "cpu"))

        calls = captured["calls"]
        assert len(calls) == 3
        first = calls[0]
        for other in calls[1:]:
            assert len(other) == len(first)
            for actual, want in zip(other, first):
                if isinstance(want, torch.Tensor):
                    assert torch.equal(actual, want)
                else:
                    assert actual == want
        # The mask/length state really was rebuilt per call: equal values, different objects.
        mask_and_length_positions = range(3, 4 + len(BLOCK_MASK_TENSOR_FIELDS))
        assert any(calls[0][i] is not calls[1][i] for i in mask_and_length_positions)


class TestPrivateBoundary:
    """What the separately compiled private root hands to FlexAttention, checked through a call seam."""

    LENGTHS = [SEQ_LEN, 41, 7]

    def _capture_flex(self, monkeypatch):
        captured = {}

        def fake_flex(query, key, value, block_mask=None, kernel_options=None, **kwargs):
            captured.update(
                query=query, key=key, value=value, block_mask=block_mask, kernel_options=kernel_options, extra=kwargs
            )
            # Real FP8 FlexAttention returns the FP8 accumulation layout; a constant makes the cast visible.
            return torch.full(query.shape, 0.5, dtype=torch.bfloat16).to(FP8_DTYPE)

        monkeypatch.setattr(sortformer_fp8_flex_attention, "flex_attention", fake_flex)
        return captured

    def _run(self, monkeypatch, lengths=None, seq_len=SEQ_LEN):
        lengths = self.LENGTHS if lengths is None else lengths
        captured = self._capture_flex(monkeypatch)
        q, k, v = _qkv("cpu", seq_len=seq_len)
        block_mask = _padding_block_mask(lengths, "cpu", seq_len=seq_len)
        valid_lengths = _valid_lengths(lengths, "cpu", seq_len=seq_len)
        out = sortformer_fp8_flex_attention._fp8_flex_attention_boundary(*_op_args(q, k, v, block_mask, valid_lengths))
        return captured, out, block_mask, valid_lengths

    @pytest.mark.unit
    def test_query_and_key_are_row_major_fp8(self, monkeypatch):
        captured, _, _, _ = self._run(monkeypatch)
        for name in ("query", "key"):
            tensor = captured[name]
            assert tensor.dtype == FP8_DTYPE
            assert tensor.shape == (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
            assert tensor.is_contiguous()

    @pytest.mark.unit
    def test_value_is_fp8_in_the_column_major_attention_layout(self, monkeypatch):
        captured, _, _, _ = self._run(monkeypatch)
        value = captured["value"]
        assert value.dtype == FP8_DTYPE
        assert value.shape == (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        # Column major over (T, D): stride 1 along time, stride T along the head dim.
        assert value.stride() == (NUM_HEADS * SEQ_LEN * HEAD_DIM, SEQ_LEN * HEAD_DIM, 1, SEQ_LEN)

    @pytest.mark.unit
    def test_only_the_triton_kernel_option_is_forwarded(self, monkeypatch):
        captured, _, _, _ = self._run(monkeypatch)
        assert captured["kernel_options"] == {"BACKEND": "TRITON"}
        assert captured["extra"] == {}

    @pytest.mark.unit
    def test_reconstructed_mask_matches_the_callers_mask(self, monkeypatch):
        captured, _, block_mask, _ = self._run(monkeypatch)
        rebuilt = captured["block_mask"]
        assert isinstance(rebuilt, BlockMask)
        assert rebuilt is not block_mask
        for name in BLOCK_MASK_TENSOR_FIELDS:
            field, want = getattr(rebuilt, name), getattr(block_mask, name)
            assert (field is None) == (want is None)
            if want is not None:
                assert torch.equal(field, want)
        assert rebuilt.seq_lengths == tuple(block_mask.seq_lengths)
        assert rebuilt.BLOCK_SIZE == tuple(block_mask.BLOCK_SIZE)

    @pytest.mark.unit
    @pytest.mark.parametrize("lengths", [[SEQ_LEN, 41, 7], [SEQ_LEN, SEQ_LEN, 1]])
    def test_reconstructed_mask_mod_is_the_key_padding_predicate(self, monkeypatch, lengths):
        """``kv_idx < valid_lengths[b]`` -- exact even where the last valid key is not block aligned."""
        captured, _, _, valid_lengths = self._run(monkeypatch, lengths=lengths)
        mask_mod = captured["block_mask"].mask_mod
        kv_idx = torch.arange(SEQ_LEN)
        zero = torch.tensor(0)
        for sample, valid in enumerate(lengths):
            actual = mask_mod(torch.tensor(sample), zero, zero, kv_idx)
            torch.testing.assert_close(actual, kv_idx < valid, atol=0, rtol=0)
        assert valid_lengths.tolist() == lengths

    @pytest.mark.unit
    def test_result_is_converted_back_to_bf16(self, monkeypatch):
        _, out, _, _ = self._run(monkeypatch)
        assert out.dtype == torch.bfloat16
        assert out.shape == (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        assert out.is_contiguous()
        assert torch.equal(out, torch.full(out.shape, 0.5, dtype=torch.bfloat16))


class TestFakeImplementation:
    """The registered fake must describe the real output, so a traced graph stays honest."""

    LENGTHS = [SEQ_LEN, 41, 7]

    def _fake_output(self, q, k, v, block_mask, lengths):
        with FakeTensorMode() as mode:
            args = [
                mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in _op_args(q, k, v, block_mask, lengths)
            ]
            return torch.ops.nemo_sortformer.fp8_flex_attention(*args)

    @pytest.mark.unit
    def test_fake_output_is_a_contiguous_bf16_query_shaped_tensor(self):
        q, k, v = _qkv("cpu")
        fake = self._fake_output(
            q, k, v, _padding_block_mask(self.LENGTHS, "cpu"), _valid_lengths(self.LENGTHS, "cpu")
        )
        assert fake.shape == q.shape
        assert fake.dtype == torch.bfloat16
        assert fake.device == q.device
        assert fake.stride() == q.contiguous().stride()

    @CUDA_FP8_FLEX
    def test_fake_output_matches_the_real_output_contract(self):
        q, k, v = _qkv("cuda")
        block_mask = _padding_block_mask(self.LENGTHS, "cuda")
        lengths = _valid_lengths(self.LENGTHS, "cuda")
        with torch.no_grad():
            real = fp8_flex_attention(q, k, v, block_mask, lengths)
        fake = self._fake_output(q, k, v, block_mask, lengths)

        assert fake.shape == real.shape
        assert fake.dtype == real.dtype
        assert fake.device.type == real.device.type
        assert fake.stride() == real.stride()


class TestOuterCompileBoundary:
    """The outer encoder compiler must see one opaque node per attention call, and no graph break."""

    LENGTHS = [SEQ_LEN, 41, 7]

    @pytest.mark.unit
    def test_outer_fullgraph_capture_yields_one_custom_op_node(self, monkeypatch):
        captured = _capture_boundary(monkeypatch)
        graphs = []

        def record_backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        # fullgraph=True turns any graph break into an error, so a single recorded graph is the assertion.
        @torch.compile(backend=record_backend, fullgraph=True, dynamic=False)
        def outer(query, key, value, mask, lengths):
            return fp8_flex_attention(query, key, value, mask, lengths)

        q, k, v = _qkv("cpu")
        out = outer(q, k, v, _padding_block_mask(self.LENGTHS, "cpu"), _valid_lengths(self.LENGTHS, "cpu"))

        assert out.shape == q.shape
        assert len(graphs) == 1
        targets = [node.target for node in graphs[0].graph.nodes if node.op == "call_function"]
        assert _count_opaque_op_nodes(targets) == 1
        # The FP8 attention region itself must not have been inlined into the outer graph.
        assert not any("flex_attention" in str(target) and "nemo_sortformer" not in str(target) for target in targets)
        # ... and the opaque node really does run the separately compiled private root.
        assert len(captured["calls"]) == 1

    @pytest.mark.unit
    def test_repeated_layers_reuse_one_opaque_node_each(self, monkeypatch):
        """A stack of structurally identical layers must not depend on layer or mask object identity."""
        captured = _capture_boundary(monkeypatch)
        graphs = []

        def record_backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        @torch.compile(backend=record_backend, fullgraph=True, dynamic=False)
        def outer(query, key, value, mask, lengths):
            for _ in range(3):
                query = fp8_flex_attention(query, key, value, mask, lengths)
            return query

        q, k, v = _qkv("cpu")
        outer(q, k, v, _padding_block_mask(self.LENGTHS, "cpu"), _valid_lengths(self.LENGTHS, "cpu"))

        assert len(graphs) == 1
        targets = [node.target for node in graphs[0].graph.nodes if node.op == "call_function"]
        assert _count_opaque_op_nodes(targets) == 3
        assert len(captured["calls"]) == 3


class TestCudaParity:
    """Real FP8 Triton FlexAttention evidence; only runs on a supported Blackwell CUDA runtime."""

    # Unequal nonzero lengths: a fully padded sample has no defined FP8 row, so it is deliberately absent.
    LENGTHS = [SEQ_LEN, 41, 7]

    def _reference(self, q, k, v, block_mask):
        """The BF16 FlexAttention semantics this backend replaces, called directly and uncompiled.

        Deliberately not routed through any ``torch.compile``-wrapped FlexAttention: every such wrapper --
        the encoder's ``flex_attention_compiled`` included -- shares one Dynamo cache and one recompile
        budget for the ``flex_attention`` code object, so in a long-lived test process the reference would
        be order dependent and could silently degrade to eager. A direct call evaluates the same masked
        softmax over the same block mask, at a shape small enough (B=3, H=4, T=128, D=64) for the unfused
        path, and depends on no compiled-code cache state; the encoder itself takes this uncompiled call on
        CPU.
        """
        return flex_attention(q, k, v, block_mask=block_mask)

    @CUDA_FP8_FLEX
    def test_matches_bf16_flex_attention_on_valid_positions(self):
        q, k, v = _qkv("cuda")
        block_mask = _padding_block_mask(self.LENGTHS, "cuda")
        lengths = _valid_lengths(self.LENGTHS, "cuda")

        with torch.no_grad():
            out = fp8_flex_attention(q, k, v, block_mask, lengths)
            reference = self._reference(q, k, v, block_mask)

        assert out.shape == q.shape
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        for sample, valid in enumerate(self.LENGTHS):
            torch.testing.assert_close(
                out[sample, :, :valid].float(),
                reference[sample, :, :valid].float(),
                atol=FP8_BF16_PARITY_ATOL,
                rtol=0,
            )

    @CUDA_FP8_FLEX
    def test_padding_keys_do_not_affect_valid_outputs(self):
        lengths = [41, 41, 41]
        q, k, v = _qkv("cuda")
        block_mask = _padding_block_mask(lengths, "cuda")
        valid_lengths = _valid_lengths(lengths, "cuda")
        k_perturbed, v_perturbed = k.clone(), v.clone()
        k_perturbed[:, :, 41:] = torch.randn_like(k_perturbed[:, :, 41:])
        v_perturbed[:, :, 41:] = torch.randn_like(v_perturbed[:, :, 41:])

        with torch.no_grad():
            out = fp8_flex_attention(q, k, v, block_mask, valid_lengths)
            out_perturbed = fp8_flex_attention(q, k_perturbed, v_perturbed, block_mask, valid_lengths)

        torch.testing.assert_close(out[:, :, :41], out_perturbed[:, :, :41], atol=0, rtol=0)

    @CUDA_FP8_FLEX
    def test_padding_invariance_with_a_non_block_aligned_valid_length(self):
        """The last valid key falls inside a block, so the rebuilt mask_mod -- not block granularity -- decides."""
        seq_len, valid = 384, 200
        lengths = [valid, valid, valid]
        q, k, v = _qkv("cuda", seq_len=seq_len)
        block_mask = _padding_block_mask(lengths, "cuda", seq_len=seq_len)
        valid_lengths = _valid_lengths(lengths, "cuda", seq_len=seq_len)
        k_perturbed, v_perturbed = k.clone(), v.clone()
        k_perturbed[:, :, valid:] = torch.randn_like(k_perturbed[:, :, valid:])
        v_perturbed[:, :, valid:] = torch.randn_like(v_perturbed[:, :, valid:])

        with torch.no_grad():
            out = fp8_flex_attention(q, k, v, block_mask, valid_lengths)
            out_perturbed = fp8_flex_attention(q, k_perturbed, v_perturbed, block_mask, valid_lengths)

        torch.testing.assert_close(out[:, :, :valid], out_perturbed[:, :, :valid], atol=0, rtol=0)

    @CUDA_FP8_FLEX
    def test_repeated_calls_with_fresh_mask_objects_agree(self):
        """31 encoder layers share one mask object; nothing may depend on that identity holding."""
        q, k, v = _qkv("cuda")
        lengths = _valid_lengths(self.LENGTHS, "cuda")

        with torch.no_grad():
            shared_mask = _padding_block_mask(self.LENGTHS, "cuda")
            first = fp8_flex_attention(q, k, v, shared_mask, lengths)
            second = fp8_flex_attention(q, k, v, shared_mask, lengths)
            third = fp8_flex_attention(q, k, v, _padding_block_mask(self.LENGTHS, "cuda"), lengths)

        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first, third, atol=0, rtol=0)

    @CUDA_FP8_FLEX
    def test_fullgraph_static_compile(self):
        """The helper must survive ``torch.compile(dynamic=False, fullgraph=True)`` at a fixed shape."""
        q, k, v = _qkv("cuda")
        block_mask = _padding_block_mask(self.LENGTHS, "cuda")
        lengths = _valid_lengths(self.LENGTHS, "cuda")

        @torch.compile(dynamic=False, fullgraph=True)
        def compiled_attention(query, key, value, mask, valid_lengths):
            return fp8_flex_attention(query, key, value, mask, valid_lengths)

        with torch.no_grad():
            compiled_out = compiled_attention(q, k, v, block_mask, lengths)
            eager_out = fp8_flex_attention(q, k, v, block_mask, lengths)

        assert compiled_out.shape == q.shape
        assert compiled_out.dtype == torch.bfloat16
        assert torch.isfinite(compiled_out).all()
        torch.testing.assert_close(compiled_out.float(), eager_out.float(), atol=FP8_COMPILE_PARITY_ATOL, rtol=0)

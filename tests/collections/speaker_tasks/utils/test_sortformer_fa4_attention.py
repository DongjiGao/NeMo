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

"""Tests for the opt-in FlashAttention-4 (``flash_attn.cute``) Sortformer attention backend."""

import importlib.util

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo.collections.asr.parts.utils import sortformer_fa4_attention
from nemo.collections.asr.parts.utils.sortformer_fa4_attention import (
    FA4_CUTE_BACKEND,
    FLEX_BACKEND,
    SUPPORTED_ATTENTION_BACKENDS,
    SUPPORTED_CAPABILITY_MAJORS,
    attention_backend_cache_identity,
    configure_attention_backend,
    fa4_cute_attention,
    fa4_cute_backend_info,
    prepare_fa4_seqused_k,
    validate_attention_backend,
    validate_fa4_cute_attention_config,
    validate_fa4_cute_inference_mode,
)

BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = 3, 48, 4, 64
# Sampled valid-output max |FA4 - Triton FlexAttention| on the pinned stack was 4.88e-4 in BF16;
# this tolerance keeps a comfortable margin over that measured bound.
BF16_ATOL = 2e-3

_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
FA4_SUPPORTED = FLASH_ATTN_AVAILABLE and _CAPABILITY is not None and _CAPABILITY[0] in SUPPORTED_CAPABILITY_MAJORS

CUDA_FA4 = pytest.mark.skipif(not FA4_SUPPORTED, reason="requires CUDA, flash_attn and a Blackwell GPU")


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


def _dense_inputs(lengths, device="cuda", dtype=torch.bfloat16, seq_len=SEQ_LEN):
    """Padded ``(B, T, H, D)`` Q/K/V plus the matching int32 ``seqused_k``."""
    torch.manual_seed(0)
    shape = (len(lengths), seq_len, NUM_HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device=device, dtype=dtype) for _ in range(3))
    seqused_k = prepare_fa4_seqused_k(torch.tensor(lengths, device=device, dtype=torch.int64), seq_len)
    return q, k, v, seqused_k


def _flex_reference(q, k, v, lengths):
    """Current Triton FlexAttention path: (B, H, T, D) tensors with a ``kv_idx < lengths[b]`` block mask."""
    length_tensor = torch.tensor(lengths, device=q.device, dtype=torch.int64)

    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < length_tensor[b]

    batch, seq_len = q.shape[0], q.shape[1]
    block_mask = create_block_mask(pad_mask, B=batch, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=q.device)
    # Compiled exactly the way TransformerEncoder runs it, so this is the Triton kernel FA4 replaces.
    attn_fn = torch.compile(flex_attention, dynamic=True)
    out = attn_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
    return out.transpose(1, 2)


class TestBackendSelection:
    @pytest.mark.unit
    @pytest.mark.parametrize("backend", SUPPORTED_ATTENTION_BACKENDS)
    def test_valid_backends(self, backend):
        assert validate_attention_backend(backend) == backend

    @pytest.mark.unit
    def test_none_defaults_to_flex(self):
        assert validate_attention_backend(None) == FLEX_BACKEND

    @pytest.mark.unit
    @pytest.mark.parametrize("backend", ["FA4", "fa4", "flash", "", "fa4_cute "])
    def test_invalid_backend_raises(self, backend):
        with pytest.raises(ValueError, match="is not supported"):
            validate_attention_backend(backend)

    @pytest.mark.unit
    def test_non_string_backend_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_attention_backend(2)

    @pytest.mark.unit
    def test_dense_rope_config_is_accepted(self):
        validate_fa4_cute_attention_config("full", "rope")

    @pytest.mark.unit
    def test_causal_mode_is_rejected(self):
        with pytest.raises(ValueError, match="attn_mode='full'"):
            validate_fa4_cute_attention_config("causal", "rope")

    @pytest.mark.unit
    def test_rel_pos_score_mod_is_rejected(self):
        with pytest.raises(ValueError, match="relative-position"):
            validate_fa4_cute_attention_config("full", "rel_pos")

    @pytest.mark.unit
    def test_training_mode_is_rejected(self):
        """A module left in training mode must fail even under no_grad, where the grad check cannot fire."""
        validate_fa4_cute_inference_mode(False)
        with pytest.raises(RuntimeError, match="inference-only"):
            validate_fa4_cute_inference_mode(True)


class TestSequsedK:
    @pytest.mark.unit
    def test_conversion_is_int32_and_contiguous(self):
        lengths = torch.tensor([48, 17, 0], dtype=torch.int64)
        seqused_k = prepare_fa4_seqused_k(lengths, SEQ_LEN)
        assert seqused_k.dtype == torch.int32
        assert seqused_k.is_contiguous()
        assert seqused_k.tolist() == [48, 17, 0]

    @pytest.mark.unit
    def test_noncontiguous_input_is_made_contiguous(self):
        lengths = torch.tensor([[10, 20], [30, 40]], dtype=torch.int64)[:, 0]
        assert not lengths.is_contiguous()
        assert prepare_fa4_seqused_k(lengths, SEQ_LEN).is_contiguous()

    @pytest.mark.unit
    @pytest.mark.parametrize("lengths", [[SEQ_LEN + 1, 4, 4], [-1, 4, 4]])
    def test_out_of_range_lengths_raise(self, lengths):
        with pytest.raises(ValueError, match="must lie in"):
            prepare_fa4_seqused_k(torch.tensor(lengths, dtype=torch.int64), SEQ_LEN)

    @pytest.mark.unit
    def test_rank_and_dtype_are_validated(self):
        with pytest.raises(ValueError, match="rank 1"):
            prepare_fa4_seqused_k(torch.tensor([[4, 4]], dtype=torch.int64), SEQ_LEN)
        with pytest.raises(ValueError, match="integer tensor"):
            prepare_fa4_seqused_k(torch.tensor([4.0, 4.0]), SEQ_LEN)
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            prepare_fa4_seqused_k([4, 4], SEQ_LEN)
        with pytest.raises(ValueError, match="positive int"):
            prepare_fa4_seqused_k(torch.tensor([4], dtype=torch.int64), 0)
        with pytest.raises(ValueError, match="positive int"):
            prepare_fa4_seqused_k(torch.tensor([4], dtype=torch.int64), 4.0)

    @pytest.mark.unit
    def test_symbolic_seq_len_is_accepted_under_dynamic_tracing(self):
        """With a dynamic time axis ``T`` arrives as a ``SymInt``; the host-side range check is skipped."""

        @torch.compile(dynamic=True, fullgraph=True, backend="eager")
        def to_seqused_k(x, lengths):
            return prepare_fa4_seqused_k(lengths, x.shape[1])

        # The second call has a different T, so the time axis is recompiled as a symbolic dimension.
        for seq_len in (12, 20):
            out = to_seqused_k(torch.zeros(2, seq_len, 4), torch.tensor([seq_len, 5], dtype=torch.int64))
            assert out.dtype == torch.int32
            assert out.tolist() == [seq_len, 5]


class TestInputContract:
    """Contract failures reportable without a GPU: everything but the device/capability check."""

    def _cpu_inputs(self, dtype=torch.bfloat16):
        shape = (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM)
        q, k, v = (torch.zeros(shape, dtype=dtype) for _ in range(3))
        seqused_k = torch.tensor([SEQ_LEN, 17, 0], dtype=torch.int32)
        return q, k, v, seqused_k

    @pytest.mark.unit
    def test_cpu_tensors_are_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(RuntimeError, match="requires CUDA"):
            fa4_cute_attention(q, k, v, seqused_k)

    @pytest.mark.unit
    def test_flash_attn_is_not_imported_by_the_contract_checks(self):
        """The optional dependency must stay unresolved until a supported call actually executes."""
        q, k, v, seqused_k = self._cpu_inputs()
        before = sortformer_fa4_attention._flash_attn_varlen_func
        with pytest.raises(RuntimeError):
            fa4_cute_attention(q, k, v, seqused_k)
        assert sortformer_fa4_attention._flash_attn_varlen_func is before

    @pytest.mark.unit
    def test_non_bf16_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs(dtype=torch.float32)
        with pytest.raises(ValueError, match="bfloat16"):
            fa4_cute_attention(q, k, v, seqused_k)

    @pytest.mark.unit
    def test_rank_mismatch_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(ValueError, match="rank 4"):
            fa4_cute_attention(q[0], k, v, seqused_k)

    @pytest.mark.unit
    def test_shape_mismatch_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(ValueError, match="must share one shape"):
            fa4_cute_attention(q, k[:, :-1], v, seqused_k)

    @pytest.mark.unit
    def test_non_last_dim_contiguous_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(ValueError, match="last dimension"):
            fa4_cute_attention(q.transpose(1, 3), k, v, seqused_k)

    @pytest.mark.unit
    def test_non_int32_seqused_k_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(ValueError, match="must be int32"):
            fa4_cute_attention(q, k, v, seqused_k.to(torch.int64))

    @pytest.mark.unit
    def test_noncontiguous_seqused_k_is_rejected(self):
        q, k, v, _ = self._cpu_inputs()
        seqused_k = torch.zeros(BATCH, 2, dtype=torch.int32)[:, 0]
        with pytest.raises(ValueError, match="must be contiguous"):
            fa4_cute_attention(q, k, v, seqused_k)

    @pytest.mark.unit
    def test_seqused_k_batch_mismatch_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(ValueError, match="matching the query batch"):
            fa4_cute_attention(q, k, v, seqused_k[:-1])

    @pytest.mark.unit
    def test_unsupported_head_dim_is_rejected(self):
        q = torch.zeros(BATCH, SEQ_LEN, NUM_HEADS, 12, dtype=torch.bfloat16)
        seqused_k = torch.zeros(BATCH, dtype=torch.int32)
        with pytest.raises(ValueError, match="head dim"):
            fa4_cute_attention(q, q, q, seqused_k)

    @pytest.mark.unit
    def test_grad_requiring_inputs_are_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        q = q.detach().requires_grad_(True)
        with pytest.raises(RuntimeError, match="inference-only"):
            fa4_cute_attention(q, k, v, seqused_k)

    @pytest.mark.unit
    def test_no_grad_bypasses_the_training_guard(self):
        """Under no_grad the guard must not fire; the device check is what stops a CPU run."""
        q, k, v, seqused_k = self._cpu_inputs()
        q = q.detach().requires_grad_(True)
        with torch.no_grad(), pytest.raises(RuntimeError, match="requires CUDA"):
            fa4_cute_attention(q, k, v, seqused_k)

    @pytest.mark.unit
    @pytest.mark.skipif(
        not torch.cuda.is_available() or (_CAPABILITY is not None and _CAPABILITY[0] in SUPPORTED_CAPABILITY_MAJORS),
        reason="requires a CUDA device that is not Blackwell",
    )
    def test_unsupported_capability_is_rejected(self):
        q, k, v, seqused_k = self._cpu_inputs()
        with pytest.raises(RuntimeError, match="compute capability"):
            fa4_cute_attention(q.cuda(), k.cuda(), v.cuda(), seqused_k.cuda())


class TestCustomOp:
    @pytest.mark.unit
    def test_operator_is_registered(self):
        assert hasattr(torch.ops.nemo_sortformer, "fa4_cute_attention")

    @pytest.mark.unit
    def test_fake_implementation_matches_the_real_output_layout(self):
        """Dynamo needs a fake output with the real (B, T, H, D) contiguous shape/dtype."""
        with FakeTensorMode():
            q = torch.empty(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
            seqused_k = torch.empty(BATCH, dtype=torch.int32)
            out = torch.ops.nemo_sortformer.fa4_cute_attention(q, q, q, seqused_k, 0.125)
            assert out.shape == q.shape
            assert out.dtype == torch.bfloat16
            assert out.is_contiguous()


class TestModelWiring:
    @pytest.mark.unit
    def test_backend_applied_to_every_transformer_encoder(self):
        model = _DummyModel(encoder=_DummyEncoder(), transformer_encoder=_DummyEncoder())
        applied = configure_attention_backend(model, FA4_CUTE_BACKEND)
        assert applied == {"encoder": FA4_CUTE_BACKEND, "transformer_encoder": FA4_CUTE_BACKEND}
        assert model.encoder.backend == model.transformer_encoder.backend == FA4_CUTE_BACKEND

    @pytest.mark.unit
    def test_missing_optional_transformer_encoder_is_skipped(self):
        model = _DummyModel(encoder=_DummyEncoder(), transformer_encoder=None)
        assert configure_attention_backend(model, FA4_CUTE_BACKEND) == {"encoder": FA4_CUTE_BACKEND}

    @pytest.mark.unit
    def test_non_selectable_modules_keep_the_default_backend(self):
        model = _DummyModel(encoder=object(), transformer_encoder=None)
        assert configure_attention_backend(model, FLEX_BACKEND) == {}

    @pytest.mark.unit
    def test_non_default_backend_without_any_target_raises(self):
        model = _DummyModel(encoder=object(), transformer_encoder=None)
        with pytest.raises(ValueError, match="no attention-backend selectable"):
            configure_attention_backend(model, FA4_CUTE_BACKEND)

    @pytest.mark.unit
    def test_invalid_backend_raises_before_touching_the_model(self):
        model = _DummyModel(encoder=_DummyEncoder())
        with pytest.raises(ValueError, match="is not supported"):
            configure_attention_backend(model, "fa4")
        assert model.encoder.backend == FLEX_BACKEND

    @pytest.mark.unit
    def test_cache_identity_is_none_for_the_default_backend(self):
        """Caches written before this option existed must keep matching a default run."""
        assert attention_backend_cache_identity(FLEX_BACKEND) is None
        assert attention_backend_cache_identity(None) is None
        assert attention_backend_cache_identity(FA4_CUTE_BACKEND) == FA4_CUTE_BACKEND

    @pytest.mark.unit
    def test_backend_info_is_reportable_without_flash_attn(self):
        info = fa4_cute_backend_info()
        assert info["backend"] == FA4_CUTE_BACKEND
        assert info["custom_op"] == sortformer_fa4_attention.FA4_CUSTOM_OP_NAME
        assert set(info) >= {"torch_version", "cuda_available", "device_name", "compute_capability"}


class TestCudaParity:
    @CUDA_FA4
    def test_matches_flex_attention_on_valid_positions(self):
        """Unequal lengths plus a fully padded sample, against the current Triton FlexAttention path."""
        lengths = [SEQ_LEN, 17, 0]
        q, k, v, seqused_k = _dense_inputs(lengths)

        with torch.no_grad():
            out = fa4_cute_attention(q, k, v, seqused_k)
            reference = _flex_reference(q, k, v, lengths)

        assert out.shape == q.shape
        assert torch.isfinite(out).all()
        for sample, valid in enumerate(lengths):
            if valid == 0:
                # Every key is masked: FlexAttention has no defined value here, FA4 returns finite zeros.
                assert torch.count_nonzero(out[sample]) == 0
                continue
            torch.testing.assert_close(
                out[sample, :valid].float(), reference[sample, :valid].float(), atol=BF16_ATOL, rtol=0
            )

    @CUDA_FA4
    def test_padding_keys_do_not_affect_valid_outputs(self):
        lengths = [17, 17, 17]
        q, k, v, seqused_k = _dense_inputs(lengths)
        k_perturbed, v_perturbed = k.clone(), v.clone()
        k_perturbed[:, 17:] = torch.randn_like(k_perturbed[:, 17:])
        v_perturbed[:, 17:] = torch.randn_like(v_perturbed[:, 17:])

        with torch.no_grad():
            out = fa4_cute_attention(q, k, v, seqused_k)
            out_perturbed = fa4_cute_attention(q, k_perturbed, v_perturbed, seqused_k)

        torch.testing.assert_close(out[:, :17], out_perturbed[:, :17], atol=0, rtol=0)

    @CUDA_FA4
    def test_fullgraph_static_compile(self):
        """The custom op must survive ``torch.compile(dynamic=False, fullgraph=True)`` at a fixed shape."""
        lengths = [SEQ_LEN, 17, 0]
        q, k, v, seqused_k = _dense_inputs(lengths)

        @torch.compile(dynamic=False, fullgraph=True)
        def compiled_attention(query, key, value, lengths_int32):
            return fa4_cute_attention(query, key, value, lengths_int32)

        with torch.no_grad():
            compiled_out = compiled_attention(q, k, v, seqused_k)
            eager_out = fa4_cute_attention(q, k, v, seqused_k)

        assert compiled_out.shape == q.shape
        assert torch.isfinite(compiled_out).all()
        torch.testing.assert_close(compiled_out, eager_out, atol=0, rtol=0)

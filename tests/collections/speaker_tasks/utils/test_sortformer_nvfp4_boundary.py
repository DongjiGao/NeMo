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

"""Tests for the fused static-NVFP4 Sortformer QKV output boundary primitive."""

import importlib.util

import pytest
import torch

from nemo.collections.asr.parts.submodules.multi_head_attention import RotaryPositionalEncoding
from nemo.collections.asr.parts.utils.sortformer_nvfp4_boundary import (
    fused_qkv_rope_boundary_reference,
    fused_qkv_rope_boundary_triton,
)

BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = 3, 7, 4, 8

TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
CUDA_TRITON = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="requires CUDA and Triton",
)


def _rope_tables(rotary_fraction, dtype=torch.float32, device="cpu", head_dim=HEAD_DIM, seq_len=SEQ_LEN):
    """cos/sin tables produced exactly the way RotaryPositionalEncoding produces them."""
    rope = RotaryPositionalEncoding(d_k=head_dim, rotary_fraction=rotary_fraction)
    rope.extend_pe(seq_len, device=device, dtype=dtype)
    return rope, rope.cos[:seq_len].contiguous(), rope.sin[:seq_len].contiguous()


def _model_path(raw_qkv, global_scale, rope, num_heads, head_dim, bias=None):
    """The layout + RoPE path currently implemented inside MultiHeadAttention.forward."""
    batch, seq_len, _ = raw_qkv.shape
    qkv = raw_qkv * global_scale
    if bias is not None:
        qkv = qkv + bias
    qkv = qkv.view(batch, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = rope(q, k)
    return q, k, v


@pytest.mark.unit
def test_reference_matches_model_path_full_rotary():
    torch.manual_seed(0)
    raw_qkv = torch.randn(BATCH, SEQ_LEN, 3 * NUM_HEADS * HEAD_DIM)
    scale = 0.3721
    rope, cos, sin = _rope_tables(rotary_fraction=1.0)

    expected = _model_path(raw_qkv, scale, rope, NUM_HEADS, HEAD_DIM)
    got = fused_qkv_rope_boundary_reference(raw_qkv, scale, cos, sin, NUM_HEADS, HEAD_DIM)

    for name, ref, out in zip("qkv", expected, got):
        assert out.shape == (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), name
        assert out.is_contiguous(), name
        torch.testing.assert_close(out, ref, msg=f"{name} mismatch")


@pytest.mark.unit
def test_reference_matches_model_path_partial_rotary():
    torch.manual_seed(1)
    raw_qkv = torch.randn(BATCH, SEQ_LEN, 3 * NUM_HEADS * HEAD_DIM)
    scale = torch.tensor(1.75)
    rope, cos, sin = _rope_tables(rotary_fraction=0.5)
    assert cos.shape == (SEQ_LEN, HEAD_DIM // 2)

    expected = _model_path(raw_qkv, scale, rope, NUM_HEADS, HEAD_DIM)
    got = fused_qkv_rope_boundary_reference(raw_qkv, scale, cos, sin, NUM_HEADS, HEAD_DIM)

    for name, ref, out in zip("qkv", expected, got):
        torch.testing.assert_close(out, ref, msg=f"{name} mismatch")

    # Features beyond the rotary dim are rescaled but never rotated.
    scaled = (raw_qkv * scale).view(BATCH, SEQ_LEN, 3, NUM_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
    torch.testing.assert_close(got[0][..., HEAD_DIM // 2 :], scaled[0][..., HEAD_DIM // 2 :])


@pytest.mark.unit
def test_reference_applies_bias_before_rope():
    torch.manual_seed(2)
    raw_qkv = torch.randn(BATCH, SEQ_LEN, 3 * NUM_HEADS * HEAD_DIM)
    bias = torch.randn(3 * NUM_HEADS * HEAD_DIM)
    scale = 0.5
    rope, cos, sin = _rope_tables(rotary_fraction=1.0)

    expected = _model_path(raw_qkv, scale, rope, NUM_HEADS, HEAD_DIM, bias=bias)
    got = fused_qkv_rope_boundary_reference(raw_qkv, scale, cos, sin, NUM_HEADS, HEAD_DIM, bias=bias)

    for name, ref, out in zip("qkv", expected, got):
        torch.testing.assert_close(out, ref, msg=f"{name} mismatch")

    # V carries the last third of the bias without rotation.
    v_bias = bias.view(3, NUM_HEADS, HEAD_DIM)[2]
    scaled_v = (raw_qkv * scale).view(BATCH, SEQ_LEN, 3, NUM_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)[2]
    torch.testing.assert_close(got[2], (scaled_v + v_bias[None, :, None, :]).contiguous())

    # Bias actually changes the result.
    without_bias = fused_qkv_rope_boundary_reference(raw_qkv, scale, cos, sin, NUM_HEADS, HEAD_DIM)
    assert not torch.allclose(got[0], without_bias[0])


@pytest.mark.unit
def test_reference_preserves_input_dtype_with_fp32_scale_tensor():
    """A fp32 scale tensor must not promote Q/K/V away from the input dtype."""
    torch.manual_seed(5)
    raw_qkv = torch.randn(BATCH, SEQ_LEN, 3 * NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    _, cos, sin = _rope_tables(rotary_fraction=1.0, dtype=torch.bfloat16)

    got = fused_qkv_rope_boundary_reference(raw_qkv, torch.tensor(1.25), cos, sin, NUM_HEADS, HEAD_DIM)
    expected = fused_qkv_rope_boundary_reference(raw_qkv, 1.25, cos, sin, NUM_HEADS, HEAD_DIM)

    for name, ref, out in zip("qkv", expected, got):
        assert out.dtype == torch.bfloat16, name
        torch.testing.assert_close(out, ref, msg=f"{name} mismatch")


@pytest.mark.unit
def test_module_import_does_not_import_triton_eagerly():
    """A CPU-only install must be able to import the module without Triton being resolved."""
    spec = importlib.util.find_spec("nemo.collections.asr.parts.utils.sortformer_nvfp4_boundary")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.triton is None
    assert module.tl is None
    assert module._compiled_kernel is None


@pytest.mark.unit
def test_triton_entry_point_rejects_cpu_inputs():
    raw_qkv = torch.randn(BATCH, SEQ_LEN, 3 * NUM_HEADS * HEAD_DIM)
    _, cos, sin = _rope_tables(rotary_fraction=1.0)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fused_qkv_rope_boundary_triton(raw_qkv, 1.0, cos, sin, NUM_HEADS, HEAD_DIM)


@pytest.mark.unit
def test_validation_errors():
    hidden = 3 * NUM_HEADS * HEAD_DIM
    raw_qkv = torch.randn(BATCH, SEQ_LEN, hidden)
    _, cos, sin = _rope_tables(rotary_fraction=1.0)
    ref = fused_qkv_rope_boundary_reference

    with pytest.raises(ValueError, match="rank 3"):
        ref(raw_qkv.reshape(BATCH * SEQ_LEN, hidden), 1.0, cos, sin, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="floating point"):
        ref(torch.zeros(BATCH, SEQ_LEN, hidden, dtype=torch.int32), 1.0, cos, sin, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="contiguous"):
        ref(raw_qkv.transpose(0, 1), 1.0, cos, sin, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="last dim"):
        ref(raw_qkv, 1.0, cos, sin, NUM_HEADS + 1, HEAD_DIM)
    with pytest.raises(ValueError, match="positive ints"):
        ref(raw_qkv, 1.0, cos, sin, 0, HEAD_DIM)
    with pytest.raises(ValueError, match="scalar"):
        ref(raw_qkv, torch.ones(2), cos, sin, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="float or a 1-element tensor"):
        ref(raw_qkv, "1.0", cos, sin, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="cos must be rank 2"):
        ref(raw_qkv, 1.0, cos[0], sin, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="same shape"):
        ref(raw_qkv, 1.0, cos, sin[..., :2].contiguous(), NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="positions"):
        ref(raw_qkv, 1.0, cos[:2], sin[:2], NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="positive even"):
        ref(raw_qkv, 1.0, cos[:, :3].contiguous(), sin[:, :3].contiguous(), NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="must not exceed head_dim"):
        ref(
            torch.randn(BATCH, SEQ_LEN, 3 * NUM_HEADS * 2),
            1.0,
            cos,
            sin,
            NUM_HEADS,
            2,
        )
    with pytest.raises(ValueError, match="cos dtype"):
        ref(raw_qkv, 1.0, cos.to(torch.bfloat16), sin.to(torch.bfloat16), NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match=r"bias must have shape"):
        ref(raw_qkv, 1.0, cos, sin, NUM_HEADS, HEAD_DIM, bias=torch.zeros(hidden + 1))
    with pytest.raises(ValueError, match="bias dtype"):
        ref(raw_qkv, 1.0, cos, sin, NUM_HEADS, HEAD_DIM, bias=torch.zeros(hidden, dtype=torch.bfloat16))


@pytest.mark.unit
@CUDA_TRITON
def test_triton_matches_reference_bf16_full_rotary():
    torch.manual_seed(3)
    device = "cuda"
    batch, seq_len, num_heads, head_dim = 2, 37, 4, 64
    raw_qkv = torch.randn(batch, seq_len, 3 * num_heads * head_dim, device=device, dtype=torch.bfloat16)
    scale = 0.4271
    _, cos, sin = _rope_tables(
        rotary_fraction=1.0, dtype=torch.bfloat16, device=device, head_dim=head_dim, seq_len=seq_len
    )

    expected = fused_qkv_rope_boundary_reference(raw_qkv, scale, cos, sin, num_heads, head_dim)
    got = fused_qkv_rope_boundary_triton(raw_qkv, scale, cos, sin, num_heads, head_dim)

    for name, ref, out in zip("qkv", expected, got):
        assert out.shape == (batch, num_heads, seq_len, head_dim), name
        assert out.is_contiguous(), name
        assert out.dtype == torch.bfloat16, name
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2, msg=f"{name} mismatch")


@pytest.mark.unit
@CUDA_TRITON
def test_triton_matches_reference_bf16_partial_rotary_with_bias():
    torch.manual_seed(4)
    device = "cuda"
    batch, seq_len, num_heads, head_dim = 3, 11, 3, 16
    hidden = 3 * num_heads * head_dim
    raw_qkv = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)
    bias = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    # Exactly representable in bf16 so the reference and Triton paths see the same multiplier.
    scale = torch.tensor(1.25, device=device, dtype=torch.float32)
    _, cos, sin = _rope_tables(
        rotary_fraction=0.5, dtype=torch.bfloat16, device=device, head_dim=head_dim, seq_len=seq_len
    )

    expected = fused_qkv_rope_boundary_reference(
        raw_qkv, scale.to(torch.bfloat16), cos, sin, num_heads, head_dim, bias
    )
    got = fused_qkv_rope_boundary_triton(raw_qkv, scale, cos, sin, num_heads, head_dim, bias)

    for name, ref, out in zip("qkv", expected, got):
        torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2, msg=f"{name} mismatch")

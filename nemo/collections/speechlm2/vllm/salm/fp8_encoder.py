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

"""FP8 audio encoder for SpeechLM checkpoints whose encoder weights are stored in FP8.

The audio encoder runs as plain PyTorch inside the plugin, outside vLLM's
quantization machinery, so an FP8 encoder is described by its own block in the
checkpoint's ``config.json``::

    "encoder_quantization": {
        "format": "fp8_e4m3",
        "weights_prequantized": true,
        "weight_scale": "per_output_channel",
        "activation_scale": "static_per_tensor",   # or "dynamic_per_token"
        "patterns": ["attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3"],
        "activation_amax": {"layers.0.attn.w_qkv": 11.5, ...},
        "scale_margin": 1.5,
        "expect_replaced": 128
    }

Every ``nn.Linear`` of the ASR encoder whose name ends with one of ``patterns``
(matched on whole name components) holds an FP8 E4M3 ``weight`` and a float32
``weight_scale`` with one entry per output channel. With static activation
scales, each such layer also needs an ``activation_amax`` entry, keyed by its
name inside the ASR encoder; the input scale is ``amax * scale_margin / 448``.
The diarizer of a speaker-aware encoder is never quantized: its output is fused
additively into the ASR features.

``build_fp8_encoder`` creates these layers when the model is constructed, as vLLM
does for a quantized decoder, so the checkpoint loads into them directly.
"""

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from nemo.utils import logging

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
SUPPORTED_FORMAT = "fp8_e4m3"
_ACTIVATION_SCALES = ("static_per_tensor", "dynamic_per_token")


class FP8Linear(nn.Module):
    """Linear layer with FP8 E4M3 weights from the checkpoint, run with vLLM's CUTLASS FP8 GEMM.

    The module is built empty and filled by ``load_state_dict``: ``weight`` and
    ``weight_scale`` are persistent buffers named like the checkpoint tensors.
    Weights are scaled per output channel. Activations are quantized with a
    static per-tensor scale when ``input_scale`` is given, otherwise dynamically
    per token. Dtype casts of an enclosing module convert only the bias; the
    FP8 weight and the float32 scales keep their dtypes and follow device moves.

    Args:
        in_features: Input width. CUTLASS needs a multiple of 16.
        out_features: Output width.
        bias: Whether the layer has a bias.
        input_scale: Static activation scale (divisor convention, ``xq = x / scale``),
            or None for dynamic per-token scaling.
        bias_dtype: Dtype of the bias parameter.
        device: Device for the empty buffers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        input_scale: Optional[torch.Tensor] = None,
        bias_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if in_features % 16 != 0:
            raise ValueError(f"in_features={in_features} must be a multiple of 16 for the CUTLASS FP8 GEMM")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.float().reshape(1).to(device), persistent=False)
        else:
            self.input_scale = None
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=bias_dtype, device=device), requires_grad=False)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "FP8Linear":
        """Apply ``fn`` to the bias as usual but only its device change to the buffers.

        The perception module is cast to bfloat16 as a whole; applied to the buffers,
        that cast would turn the FP8 weight into bfloat16 values and round the scales.
        """
        buffers = dict(self._buffers)
        self._buffers.clear()
        try:
            super()._apply(fn, recurse)
        finally:
            for name, buf in buffers.items():
                if buf is not None:
                    device = fn(torch.empty(0, dtype=buf.dtype, device=buf.device)).device
                    buf = buf.to(device=device)
                self._buffers[name] = buf
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize ``x`` to FP8 and multiply by the FP8 weight, returning ``x.dtype``."""
        import vllm._custom_ops as ops

        out_shape = (*x.shape[:-1], self.out_features)
        x2d = x.reshape(-1, self.in_features)
        if self.input_scale is not None:
            xq, x_scale = ops.scaled_fp8_quant(x2d, self.input_scale)
        else:
            xq, x_scale = ops.scaled_fp8_quant(x2d, None, use_per_token_if_dynamic=True)
        # CUTLASS wants a column-major B and an [out, 1] weight scale; both are views.
        out = ops.cutlass_scaled_mm(xq, self.weight.t(), x_scale, self.weight_scale.unsqueeze(-1), x.dtype, self.bias)
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        """Summarize the layer shape and activation scaling mode."""
        mode = "static_per_tensor" if self.input_scale is not None else "dynamic_per_token"
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8_e4m3, activations={mode}"


def _asr_encoder(perception: nn.Module) -> nn.Module:
    """Return the encoder that holds the quantized Linears.

    A speaker-aware encoder keeps its ASR branch in ``asr_encoder`` next to the
    diarizer, so selecting that subtree leaves the diarizer out by construction.
    """
    encoder = getattr(perception, "encoder", None)
    if encoder is None:
        raise ValueError("perception module has no 'encoder' to quantize")
    return getattr(encoder, "asr_encoder", encoder)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether ``name`` ends with one of ``patterns`` on whole name components, so ``net.0`` misses ``subnet.0``."""
    return any(name == pattern or name.endswith("." + pattern) for pattern in patterns)


def build_fp8_encoder(perception: nn.Module, quant_cfg: Optional[dict]) -> int:
    """Build empty ``FP8Linear`` layers in place of the ASR encoder's quantized Linears.

    Call this when the model is constructed, before any weights load: the
    checkpoint's FP8 ``weight`` and ``weight_scale`` then load into these layers
    directly, and later dtype casts of the perception module leave them intact.

    Args:
        perception: The SpeechLM perception module.
        quant_cfg: The checkpoint's ``encoder_quantization`` block, or None.

    Returns:
        The number of Linears replaced; 0 when the checkpoint has no prequantized encoder. A block
        without ``weights_prequantized`` is ignored with a warning: the encoder then runs unquantized.

    Raises:
        ValueError: If the block requests an unsupported format or scaling mode, has no
            patterns, or lacks a static activation amax for a quantized layer.
        RuntimeError: If the number of replaced Linears differs from ``expect_replaced``.
    """
    if not quant_cfg:
        return 0
    if not quant_cfg.get("weights_prequantized"):
        logging.warning(
            "config.json has an encoder_quantization block without weights_prequantized; only prequantized "
            "FP8 encoder weights are supported, so the audio encoder runs unquantized"
        )
        return 0
    if quant_cfg.get("format") != SUPPORTED_FORMAT:
        raise ValueError(
            f"encoder_quantization.format={quant_cfg.get('format')!r} is not supported; expected {SUPPORTED_FORMAT!r}"
        )
    weight_scale = quant_cfg.get("weight_scale", "per_output_channel")
    if weight_scale != "per_output_channel":
        raise ValueError(f"encoder_quantization.weight_scale={weight_scale!r} is not supported")
    activation_scale = quant_cfg.get("activation_scale", "static_per_tensor")
    if activation_scale not in _ACTIVATION_SCALES:
        raise ValueError(
            f"encoder_quantization.activation_scale={activation_scale!r} is not one of {_ACTIVATION_SCALES}"
        )
    patterns = tuple(quant_cfg.get("patterns") or ())
    if not patterns:
        raise ValueError("encoder_quantization.patterns is required to locate the FP8 Linears")

    encoder = _asr_encoder(perception)
    targets = [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, patterns)
    ]
    static = activation_scale == "static_per_tensor"
    amax = quant_cfg.get("activation_amax") or {}
    if static:
        missing = [name for name, _ in targets if name not in amax]
        if missing:
            raise ValueError(
                f"encoder_quantization uses static activation scales but has no activation_amax for {len(missing)} "
                f"quantized Linear(s), e.g. {missing[:3]}"
            )
    expected = quant_cfg.get("expect_replaced")
    if expected is not None and int(expected) != len(targets):
        raise RuntimeError(
            f"encoder_quantization.expect_replaced={expected} but {len(targets)} Linears match the patterns; "
            "refusing to load a checkpoint whose quantized layers do not line up with the encoder"
        )
    margin = float(quant_cfg.get("scale_margin", 1.0))

    for name, module in targets:
        input_scale = None
        if static:
            input_scale = torch.tensor(max(float(amax[name]) * margin / FP8_MAX, 1e-12))
        parent_name, _, child_name = name.rpartition(".")
        parent = encoder.get_submodule(parent_name) if parent_name else encoder
        setattr(
            parent,
            child_name,
            FP8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                input_scale=input_scale,
                bias_dtype=module.bias.dtype if module.bias is not None else torch.bfloat16,
                device=module.weight.device,
            ),
        )

    logging.info(
        f"Prequantized encoder: {len(targets)} FP8 Linears prepared for load, "
        f"{len(targets) if static else 0} static act scales"
    )
    return len(targets)

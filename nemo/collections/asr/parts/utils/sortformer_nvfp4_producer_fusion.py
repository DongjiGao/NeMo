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

"""
Opt-in producer-to-NVFP4-pack fusion for static-NVFP4 Sortformer transformer blocks.

A statically quantized transformer block currently materializes a full BF16 ``(M, K)`` producer tensor in front of
every NVFP4 activation pack: the LayerNorm output before ``attn.w_qkv``, the LayerNorm output before ``ffn.net.0``,
and the rescaled/biased GELU output before ``ffn.net.3``. This module removes exactly those three boundaries by
replacing each complete converted block with a :class:`ProducerFusedTransformerBlock` that drives the fused
producer+pack kernels of :mod:`~nemo.collections.asr.parts.utils.sortformer_nvfp4_fused_pack` and then calls the
native FP4 ``torch._scaled_mm`` against the weights TorchAO has *already* converted.

What this path does **not** do:

* it never creates a second weight format -- the ``_scaled_mm`` operands are views of the converted
  ``NVFP4Tensor`` (``weight.t().qdata`` and ``weight.scale``), held as non-persistent buffers;
* it never changes the residual stream, the residual adds, the attention body, or the public module tree --
  ``norm1``, ``attn``, ``drop``, ``norm2`` and ``ffn`` keep their identity, their names and their state-dict keys.
  The residual is read, never cast or written: under CUDA BF16 autocast the compiled encoder carries an FP32
  residual into the pre-norm boundary (``layer_norm`` is on autocast's FP32 list), the fused LayerNorm keeps that
  FP32 arithmetic, and only the packed activation crosses the BF16 producer boundary, exactly as the ordinary
  TorchAO path does when autocast casts the linear's input;
* it never touches ``attn.out_proj``, which stays on the ordinary TorchAO dispatch in this first integration;
* it never falls back. An unsupported device, dtype, rank, shape, layout, scale, dependency or compute capability
  raises, and a partially converted or unrecognized block is rejected rather than silently left unfused.

Scale contract, matching TorchAO's ``_addmm_nvfp4_dispatch`` exactly:

* activations are packed with the MSLK global scale ``weight.act_per_tensor_scale.reciprocal()``;
* the GEMM is ``_scaled_mm(act_qdata, weight.t().qdata, act_blocked_scale, weight.scale, out_dtype=bfloat16)``;
* the result is multiplied by ``(act_per_tensor_scale * per_tensor_scale)`` *cast to BF16* and the BF16 bias is
  added afterwards, because TorchAO always adds the bias separately once a per-tensor scale exists.

``torch.compile`` compatibility: the two fused Triton launchers are reached through narrowly scoped
``torch.library.custom_op`` operators with registered fake implementations, following the pattern already used by
:mod:`~nemo.collections.asr.parts.utils.sortformer_fa4_attention`. Registration happens when the adapter is
installed, not at import, and is idempotent, so importing or reloading this module twice is safe. Like TorchAO's
own packing op, the operators return the ``uint8`` storage of the packed data and blocked scales; the FP4/FP8
``view`` is taken at the ``_scaled_mm`` call site, which is exactly the pattern TorchAO traces today.

Every optional dependency is imported lazily, so importing this module on a CPU-only or minimal install neither
imports nor requires Triton, MSLK or TorchAO. That includes the pack primitives of
:mod:`~nemo.collections.asr.parts.utils.sortformer_nvfp4_fused_pack`: importing them eagerly would import the
NeMo ASR collection, which does import Triton when it is installed.
"""

import importlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Linear families that must all be present, converted, and selected before a block may be fused. This is the exact
# set of ``sortformer_quantization.QUANTIZATION_TARGET_SUFFIXES``, used to recognise which block member an FQN
# names. Completeness is required over PRODUCER_FUSION_REQUIRED_CONSUMERS, not over this tuple; it is repeated
# here rather than imported so that
# this module stays importable from the quantization module without a cycle, and a test asserts the two agree.
PRODUCER_FUSION_BLOCK_MEMBERS: Tuple[str, ...] = (
    "attn.w_qkv",
    "attn.out_proj",
    "ffn.net.0",
    "ffn.net.3",
)

# The block members fusion actually packs into and dispatches itself, i.e. the ones _gemm_linears owns.
# ``attn.out_proj`` is deliberately absent: fusion never writes a pack for it and never dispatches it, so it may
# be restored to BF16 by a bf16_override while the block is still fully fusable. Completeness is required over
# these three, not over PRODUCER_FUSION_BLOCK_MEMBERS.
PRODUCER_FUSION_REQUIRED_CONSUMERS: Tuple[str, ...] = (
    "attn.w_qkv",
    "ffn.net.0",
    "ffn.net.3",
)
# Operand prefixes ``_gemm_linears`` returns, in the same order, so a test can assert the two agree. If a future
# integration adds attn.out_proj to _gemm_linears it must be added here too, otherwise a bf16_override could
# restore a linear that fusion packs into.
PRODUCER_FUSION_CONSUMER_PREFIXES: Tuple[str, ...] = ("qkv", "ffn_up", "ffn_down")

# The three BF16 producer materializations this path removes, reported in the quantization summary.
FUSED_PRODUCER_BOUNDARIES: Tuple[str, ...] = (
    "norm1 -> attn.w_qkv",
    "norm2 -> ffn.net.0",
    "ffn.net.0 -> ffn.net.3",
)

# Packed-layout constants of ``sortformer_nvfp4_fused_pack``. Like the block members above they are repeated
# rather than imported, here so that importing this module stays free of the NeMo ASR collection (and therefore of
# Triton); the pack primitives themselves are resolved lazily by :func:`_pack_primitives`. A test asserts that
# these values agree with the ones the pack module defines.
NVFP4_BLOCK_SIZE = 16
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4
REQUIRED_K_MULTIPLE = 64

LAYER_NORM_PACK_OP_NAME = "nemo_sortformer::nvfp4_layer_norm_pack"
SCALED_GELU_PACK_OP_NAME = "nemo_sortformer::nvfp4_scaled_gelu_pack"

# TorchAO entry point, addressed as ``module:attribute`` so that availability can be probed and mocked.
NVFP4_TENSOR_API = "torchao.prototype.mx_formats.nvfp4_tensor:NVFP4Tensor"

REQUIRED_TORCH_DTYPES: Tuple[str, ...] = ("float4_e2m1fn_x2", "float8_e4m3fn")

# Rank of the activation the fused block consumes: the encoder residual stream is always ``(B, T, d_model)``.
RESIDUAL_RANK = 3

_producer_pack_ops_registered = False
_pack_primitives_cache: Optional[Tuple[Any, Any]] = None


class ProducerFusedTransformerBlock(torch.nn.Module):
    """
    Static-NVFP4 Sortformer transformer block whose three producer-to-pack boundaries are fused.

    The block adopts the original ``norm1``/``attn``/``drop``/``norm2``/``ffn`` submodules under their original
    names, so every parameter, buffer and state-dict key of the wrapped block is preserved exactly and no
    parameter is registered twice. Only the execution of the two LayerNorm producers, the GELU producer and the
    three quantized GEMMs in front of them changes; the residual adds and the residual's own precision (FP32 under
    autocast, BF16 without it) are untouched.
    """

    def __init__(
        self,
        block: torch.nn.Module,
        fqn: str,
        nvfp4_tensor_cls: Optional[type] = None,
        out_proj_restored: bool = False,
    ):
        """
        Args:
            block (torch.nn.Module): ``TransformerBlock`` whose four target linears TorchAO has already converted
                to static NVFP4.
            fqn (str): Fully qualified name of ``block``, used in provenance and in every error message.
            out_proj_restored (bool): Whether ``attn.out_proj`` was deliberately restored to BF16 by a
                ``bf16_override``. When False the linear must be converted, so a half-converted block is
                rejected rather than half-fused.
            nvfp4_tensor_cls (Optional[type]): Injected ``NVFP4Tensor`` class. Resolved lazily from TorchAO when
                ``None``.

        Raises:
            RuntimeError: If the block is not a recognizable Sortformer transformer block, if any of its four
                target linears is not a converted static-NVFP4 linear, if a required dtype or TorchAO API is
                missing, or if the block is in training mode.
            ValueError: If a shape, dtype or device of the block's own tensors violates the fused contract.
        """
        super().__init__()
        self.fqn = str(fqn)
        _require_torch_dtypes(f"NVFP4 producer fusion of '{self.fqn}'")
        # Registered here rather than at import, so that a disabled recipe never pays for it, and idempotently, so
        # that a block built outside :func:`apply_producer_fusion` still finds its operators in ``torch.ops``.
        register_producer_pack_ops()
        tensor_cls = nvfp4_tensor_cls if nvfp4_tensor_cls is not None else _require_api(NVFP4_TENSOR_API)
        _require_transformer_block(block, self.fqn)
        if block.training:
            raise RuntimeError(
                f"NVFP4 producer fusion is inference-only, but '{self.fqn}' is in training mode. Call .eval() on "
                "the model before quantizing with quantization_fuse_producer_packing=True."
            )

        # Adopted, not copied: the same module objects keep the same names, so public FQNs and state-dict keys of
        # the wrapped block are byte-for-byte what they were before fusion.
        self.norm1 = block.norm1
        self.attn = block.attn
        self.drop = block.drop
        self.norm2 = block.norm2
        self.ffn = block.ffn

        self.d_model = int(self.attn.w_qkv.in_features)
        self.norm1_eps = _validated_layer_norm(self.norm1, "norm1", self.d_model, self.fqn)
        self.norm2_eps = _validated_layer_norm(self.norm2, "norm2", self.d_model, self.fqn)

        # ``attn.out_proj`` keeps the ordinary TorchAO dispatch. Fusion neither packs into it nor dispatches
        # it -- forward() reaches it only through ``attn.forward_from_qkv`` -- so it may legitimately be a BF16
        # ``torch.nn.Linear`` when a bf16_override restored it. When it was NOT restored it must still be
        # converted, so that a half-converted block is rejected instead of half-fused.
        if out_proj_restored:
            _require_restored_linear(self.attn.out_proj, f"{self.fqn}.attn.out_proj", tensor_cls)
        else:
            _require_converted_linear(self.attn.out_proj, f"{self.fqn}.attn.out_proj", tensor_cls)

        for prefix, linear, linear_fqn in self._gemm_linears():
            self._register_gemm_operands(prefix, linear, linear_fqn, tensor_cls)

        if int(self.ffn.net[0].in_features) != self.d_model:
            raise ValueError(
                f"NVFP4 producer fusion of '{self.fqn}' requires ffn.net.0 to consume the same residual width as "
                f"attn.w_qkv ({self.d_model}), got {self.ffn.net[0].in_features}."
            )
        self.train(block.training)

    def forward(self, x, block_mask=None, pos_emb=None, seqused_k=None):
        """
        Run the block with fused producer packing; the signature matches ``TransformerBlock.forward``.

        Args:
            x (torch.Tensor): ``(B, T, d_model)`` residual stream, FP32 under CUDA BF16 autocast or BF16 without
                it. It is read as-is: an unsupported dtype is rejected by the fused pack rather than cast.
            block_mask: FlexAttention block mask, forwarded unchanged to the attention body.
            pos_emb: Relative positional embedding, forwarded unchanged to the attention body.
            seqused_k: Per-sample key lengths, forwarded unchanged to the attention body.

        Returns:
            x (torch.Tensor): ``(B, T, d_model)`` residual stream, in the dtype of the incoming residual.
        """
        self._require_inference(x)
        if x.dim() != RESIDUAL_RANK:
            raise ValueError(
                f"NVFP4 producer fusion at '{self.fqn}' expects a rank-{RESIDUAL_RANK} (B, T, d_model) residual, "
                f"got shape {tuple(x.shape)}."
            )
        batch, time = x.shape[0], x.shape[1]

        attn_out = self.attn.forward_from_qkv(
            self._fused_qkv(x).view(batch, time, -1),
            block_mask=block_mask,
            pos_emb=pos_emb,
            seqused_k=seqused_k,
        )
        x = x + self.drop(attn_out)
        return x + self.drop(self._fused_ffn(x).view(batch, time, -1))

    def _apply(self, *args, **kwargs):
        """Re-derive the GEMM operands after any ``nn.Module`` tensor transform (``.to()``, ``.cuda()``, ...).

        The operands are views of the converted weights, which stay owned by their own linears. Whenever
        ``nn.Module`` replaces those weights with transformed ones, the cached views would otherwise silently keep
        pointing at the previous storage, so they are read off the live weights again here.
        """
        module = super()._apply(*args, **kwargs)
        module._rebind_gemm_operands()
        return module

    def extra_repr(self) -> str:
        """Residual width and the fused boundaries, so ``print(model)`` states what this block replaced."""
        return f"d_model={self.d_model}, fused_boundaries={list(FUSED_PRODUCER_BOUNDARIES)}"

    def _fused_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Fused ``norm1`` + NVFP4 pack, native FP4 QKV GEMM, and TorchAO's rescale-then-bias epilogue."""
        qdata, scales = torch.ops.nemo_sortformer.nvfp4_layer_norm_pack(
            x.reshape(-1, self.d_model).contiguous(),
            self.norm1.weight,
            self.norm1.bias,
            self.qkv_activation_scale,
            self.norm1_eps,
        )
        raw = self._scaled_mm(qdata, scales, self.qkv_weight_qdata, self.qkv_weight_scale)
        return _apply_output_scale_and_bias(raw, self.qkv_output_scale, self.attn.w_qkv.bias)

    def _fused_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Fused ``norm2`` + pack, FFN-up GEMM, fused rescale+bias+GELU+pack, FFN-down GEMM and its epilogue."""
        up_qdata, up_scales = torch.ops.nemo_sortformer.nvfp4_layer_norm_pack(
            x.reshape(-1, self.d_model).contiguous(),
            self.norm2.weight,
            self.norm2.bias,
            self.ffn_up_activation_scale,
            self.norm2_eps,
        )
        # The FFN-up GEMM stops at the raw BF16 product: its global-scale rescale, its bias and the exact GELU are
        # applied inside the fused pack that feeds the FFN-down GEMM.
        up_raw = self._scaled_mm(up_qdata, up_scales, self.ffn_up_weight_qdata, self.ffn_up_weight_scale)
        down_qdata, down_scales = torch.ops.nemo_sortformer.nvfp4_scaled_gelu_pack(
            up_raw,
            self.ffn_up_output_scale_f32,
            self.ffn_down_activation_scale,
            self.ffn.net[0].bias,
        )
        down_raw = self._scaled_mm(down_qdata, down_scales, self.ffn_down_weight_qdata, self.ffn_down_weight_scale)
        return _apply_output_scale_and_bias(down_raw, self.ffn_down_output_scale, self.ffn.net[3].bias)

    @staticmethod
    def _scaled_mm(
        act_qdata: torch.Tensor, act_scale: torch.Tensor, weight_qdata: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        """One native NVFP4 GEMM, with the FP4/FP8 views taken exactly where TorchAO takes them."""
        return torch._scaled_mm(
            act_qdata.view(torch.float4_e2m1fn_x2),
            weight_qdata.view(torch.float4_e2m1fn_x2),
            act_scale.view(torch.float8_e4m3fn),
            weight_scale.view(torch.float8_e4m3fn),
            bias=None,
            out_dtype=torch.bfloat16,
        )

    def _require_inference(self, x: torch.Tensor) -> None:
        """Reject training mode and gradient-bearing execution; no autograd formula backs the fused packs."""
        if self.training:
            raise RuntimeError(
                f"NVFP4 producer fusion at '{self.fqn}' is inference-only and the fused pack operators have no "
                "autograd formula, but the module is in training mode. Call .eval() before running it."
            )
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError(
                f"NVFP4 producer fusion at '{self.fqn}' is inference-only, but it received an input that requires "
                "grad. Run inside torch.no_grad()/torch.inference_mode()."
            )

    def _gemm_linears(self) -> Tuple[Tuple[str, torch.nn.Module, str], ...]:
        """The three converted linears this block drives itself, as ``(operand prefix, module, FQN)``."""
        return (
            ("qkv", self.attn.w_qkv, f"{self.fqn}.attn.w_qkv"),
            ("ffn_up", self.ffn.net[0], f"{self.fqn}.ffn.net.0"),
            ("ffn_down", self.ffn.net[3], f"{self.fqn}.ffn.net.3"),
        )

    def _rebind_gemm_operands(self) -> None:
        """Point the cached ``_scaled_mm`` weight operands at the live storage of the converted weights."""
        for prefix, linear, fqn in self._gemm_linears():
            weight = linear.weight
            qdata = _as_uint8(_require_attribute(weight.t(), "qdata", f"{fqn} (transposed weight)"), fqn)
            scale = _as_uint8(_require_attribute(weight, "scale", fqn), fqn)
            # Plain attributes, deliberately not buffers: the storage belongs to the linear's own weight, and
            # registering a second owner for it would duplicate it on every subsequent ``nn.Module`` transform and
            # would report another module's parameter storage under this module's ``named_buffers()``.
            setattr(self, f"{prefix}_weight_qdata", qdata)
            setattr(self, f"{prefix}_weight_scale", scale)

    def _register_gemm_operands(self, prefix: str, linear: torch.nn.Module, fqn: str, tensor_cls: type) -> None:
        """
        Derive the ``_scaled_mm`` operands and epilogue scalars of one converted linear.

        The qdata and block scales stay owned by the converted ``NVFP4Tensor``; only the scalars this module
        computes itself (the reciprocal activation scale and the BF16-rounded output scale) are registered, as
        non-persistent buffers, so no weight is duplicated and no state-dict key is added.
        """
        weight = _require_converted_linear(linear, fqn, tensor_cls)
        in_features, out_features = int(linear.in_features), int(linear.out_features)
        if in_features % REQUIRED_K_MULTIPLE != 0:
            raise ValueError(
                f"NVFP4 producer fusion of '{fqn}' requires in_features % {REQUIRED_K_MULTIPLE} == 0 so that the "
                f"packed block count is a whole number of scale tiles, got in_features={in_features}."
            )

        activation_scale = _require_scalar_scale(getattr(weight, "act_per_tensor_scale", None), "act", fqn)
        weight_global_scale = _require_scalar_scale(getattr(weight, "per_tensor_scale", None), "weight", fqn)
        weight_qdata = _as_uint8(_require_attribute(weight.t(), "qdata", f"{fqn} (transposed weight)"), fqn)
        # TorchAO reaches the native weight operand through ``weight.t().scale.t()``, which is the *original*
        # ``weight.scale`` layout; transposing ``weight.scale`` here would transpose it a second time.
        weight_scale = _as_uint8(_require_attribute(weight, "scale", fqn), fqn)
        bias = _validated_bias(linear, out_features, fqn)
        _require_same_device(fqn, weight_qdata, weight_scale, activation_scale, weight_global_scale, bias)

        # MSLK packs with the reciprocal of TorchAO's per-tensor *dequantization* scale.
        self.register_buffer(
            f"{prefix}_activation_scale", activation_scale.reshape(1).reciprocal().contiguous(), persistent=False
        )
        setattr(self, f"{prefix}_weight_qdata", weight_qdata)
        setattr(self, f"{prefix}_weight_scale", weight_scale)
        # ``_addmm_nvfp4_dispatch`` multiplies the BF16 GEMM result by ``scale_result.to(orig_dtype)``, so the
        # product is rounded to BF16 once, here, instead of once per forward.
        output_scale = (activation_scale.reshape(()) * weight_global_scale.reshape(())).to(torch.bfloat16)
        self.register_buffer(f"{prefix}_output_scale", output_scale, persistent=False)
        # The fused GELU pack takes its rescale as a one-element float32 tensor; widening the same BF16-rounded
        # value keeps the fused epilogue on the identical scale as the unfused one.
        self.register_buffer(
            f"{prefix}_output_scale_f32", output_scale.to(torch.float32).reshape(1).contiguous(), persistent=False
        )


def apply_producer_fusion(
    model: torch.nn.Module,
    fqns: Sequence[str],
    nvfp4_tensor_cls: Optional[type] = None,
    supported_compute_capabilities: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Replace every complete converted block covering ``fqns`` with a producer-fused block.

    Only the blocks reached from the given FQNs are considered, so this can never widen the set of modules chosen
    by the quantization recipe. A block whose three fused consumers (``attn.w_qkv``, ``ffn.net.0``, ``ffn.net.3``) are not all present in
    ``fqns`` is rejected rather than left unfused. ``attn.out_proj`` is optional: fusion never packs into or
    dispatches it, so a ``bf16_override`` may restore it while the block stays fully fusable.

    Args:
        model (torch.nn.Module): Model whose selected linears TorchAO has already converted in place.
        fqns (Sequence[str]): Exact FQNs selected for NVFP4 W4A4.
        nvfp4_tensor_cls (Optional[type]): Injected ``NVFP4Tensor`` class; resolved from TorchAO when ``None``.
        supported_compute_capabilities (Optional[Sequence[Tuple[int, int]]]): Accepted Blackwell capability
            policy; the quantization module's policy is used when ``None``.

    Returns:
        summary (Dict[str, Any]): Whether fusion is enabled, the fused boundaries, and the number and names of
            the fused blocks.

    Raises:
        RuntimeError: If a block is absent, is not a Sortformer transformer block, is not fully converted, or the
            device is not an accepted Blackwell device.
        ValueError: If ``fqns`` is empty, contains an unrecognized name, or covers a block only partially.
    """
    selected = sorted(str(fqn) for fqn in fqns)
    if not selected:
        raise ValueError(
            "NVFP4 producer fusion was requested but no NVFP4 W4A4 module was selected; the recipe would have "
            "nothing to fuse."
        )
    _require_torch_dtypes("NVFP4 producer fusion")
    blocks = group_producer_fusion_blocks(selected)
    tensor_cls = nvfp4_tensor_cls if nvfp4_tensor_cls is not None else _require_api(NVFP4_TENSOR_API)
    validate_producer_fusion_device(_model_device(model), supported_compute_capabilities)
    register_producer_pack_ops()

    modules = dict(model.named_modules())
    for block_fqn in blocks:
        block = modules.get(block_fqn)
        if block is None:
            raise RuntimeError(
                f"NVFP4 producer fusion could not find the transformer block '{block_fqn}' that owns the selected "
                "linears; refusing to fuse a partial set of blocks."
            )
        # Authoritative from the recipe's own selection: out_proj is restored exactly when it is absent from
        # the block's selected members. Never inferred from the weight's type, which would silently accept a
        # genuinely half-converted block.
        out_proj_restored = not any(member.endswith(".attn.out_proj") for member in blocks[block_fqn])
        _swap_module(
            model,
            block_fqn,
            ProducerFusedTransformerBlock(
                block, block_fqn, nvfp4_tensor_cls=tensor_cls, out_proj_restored=out_proj_restored
            ),
        )

    restored_out_proj_blocks = sorted(
        block_fqn
        for block_fqn, members in blocks.items()
        if not any(member.endswith(".attn.out_proj") for member in members)
    )
    return {
        "enabled": True,
        "fused_block_count": len(blocks),
        "out_proj_restored_block_count": len(restored_out_proj_blocks),
        "fused_block_fqns": list(blocks),
        "fused_boundaries": list(FUSED_PRODUCER_BOUNDARIES),
        "notes": [
            "The LayerNorm and GELU producers in front of attn.w_qkv, ffn.net.0 and ffn.net.3 are packed to NVFP4 "
            "in one kernel each; attn.out_proj is never packed into or dispatched by the fused block, which "
            "reaches it only through attn.forward_from_qkv, so it keeps whichever precision the recipe and any "
            "bf16_override assign it. The residual stream, the attention body and the public module tree are "
            "unchanged.",
        ],
    }


def disabled_producer_fusion_summary() -> Dict[str, Any]:
    """Summary block reported when producer fusion was not requested."""
    return {
        "enabled": False,
        "fused_block_count": 0,
        "out_proj_restored_block_count": 0,
        "fused_block_fqns": [],
        "fused_boundaries": [],
        "notes": [],
    }


def group_producer_fusion_blocks(fqns: Sequence[str]) -> Dict[str, List[str]]:
    """
    Group selected linear FQNs by the transformer block that owns them, requiring every block to be complete.

    Args:
        fqns (Sequence[str]): Exact FQNs selected for NVFP4 W4A4.

    Returns:
        blocks (Dict[str, List[str]]): Block FQN -> its sorted member FQNs, in sorted block order.

    Raises:
        ValueError: If a name does not end in one of :data:`PRODUCER_FUSION_BLOCK_MEMBERS`, if it names a bare
            member with no owning block, if a member appears twice, or if a block is only partially selected.
    """
    members_by_block: Dict[str, Dict[str, str]] = {}
    for fqn in sorted(str(name) for name in fqns):
        member = _matched_member(fqn)
        if member is None:
            raise ValueError(
                f"NVFP4 producer fusion only understands the transformer-block linears "
                f"{list(PRODUCER_FUSION_BLOCK_MEMBERS)}, but '{fqn}' matches none of them."
            )
        block_fqn = fqn[: -(len(member) + 1)]
        if not block_fqn:
            raise ValueError(
                f"NVFP4 producer fusion needs an owning transformer block, but '{fqn}' is a top-level module."
            )
        owned = members_by_block.setdefault(block_fqn, {})
        if member in owned:
            raise ValueError(f"NVFP4 producer fusion received '{fqn}' twice for block '{block_fqn}'.")
        owned[member] = fqn

    required = set(PRODUCER_FUSION_REQUIRED_CONSUMERS)
    incomplete = sorted(
        f"{block_fqn} (missing {sorted(required - set(owned))})"
        for block_fqn, owned in members_by_block.items()
        if not required.issubset(owned)
    )
    if incomplete:
        raise ValueError(
            "NVFP4 producer fusion requires every fused block to have all of "
            f"{list(PRODUCER_FUSION_REQUIRED_CONSUMERS)} selected, but these blocks are incomplete: "
            f"{incomplete}."
        )
    return {block_fqn: sorted(owned.values()) for block_fqn, owned in sorted(members_by_block.items())}


def validate_producer_fusion_device(
    device: torch.device, supported_compute_capabilities: Optional[Sequence[Tuple[int, int]]] = None
) -> Tuple[int, int]:
    """
    Require an accepted Blackwell CUDA device, using the quantization module's capability policy.

    There is one implementation for every accepted capability: no device-name branch, no per-architecture
    specialization, and no fallback for a device outside the policy.

    Args:
        device (torch.device): Device the fused blocks will run on.
        supported_compute_capabilities (Optional[Sequence[Tuple[int, int]]]): Accepted capability policy; the
            quantization module's policy is used when ``None``.

    Returns:
        capability (Tuple[int, int]): The device's compute capability.

    Raises:
        RuntimeError: If the device is not CUDA, or its compute capability is outside the accepted policy.
    """
    if supported_compute_capabilities is None:
        # Imported lazily: the quantization module imports this one, and the policy must have a single owner.
        from nemo.collections.asr.parts.utils.sortformer_quantization import SUPPORTED_COMPUTE_CAPABILITIES

        supported_compute_capabilities = SUPPORTED_COMPUTE_CAPABILITIES

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"NVFP4 producer fusion requires a CUDA device, but the model is on '{device}'. The fused packs and "
            "the native FP4 GEMM have no CPU implementation."
        )
    capability = tuple(torch.cuda.get_device_capability(device))
    accepted = {tuple(item) for item in supported_compute_capabilities}
    if capability not in accepted:
        raise RuntimeError(
            f"NVFP4 producer fusion is only accepted on compute capabilities "
            f"{[list(item) for item in sorted(accepted)]}, but this device reports {list(capability)}."
        )
    return capability


def register_producer_pack_ops() -> None:
    """
    Register the two fused producer-pack custom operators, at most once per process.

    Called from the install path, never at import: registration reaches into ``torch.library`` machinery that a
    disabled or CPU-only run has no reason to touch.

    Registration is idempotent across repeated calls and across a module reload: each operator is checked
    individually on ``torch.ops`` and an already registered one is left alone, so re-importing or reloading this
    module never raises and never double-registers.
    """
    global _producer_pack_ops_registered
    if _producer_pack_ops_registered:
        return
    _register_pack_op(LAYER_NORM_PACK_OP_NAME, _layer_norm_pack_impl, _layer_norm_pack_fake)
    _register_pack_op(SCALED_GELU_PACK_OP_NAME, _scaled_gelu_pack_impl, _scaled_gelu_pack_fake)
    _producer_pack_ops_registered = True


def _register_pack_op(name: str, implementation, fake) -> None:
    """Register one opaque pack operator with its fake implementation, unless ``torch.ops`` already has it."""
    if _op_is_registered(name):
        return
    torch.library.custom_op(name, implementation, mutates_args=()).register_fake(fake)


def _layer_norm_pack_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation_global_scale: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runtime implementation: one fused LayerNorm+NVFP4-pack Triton launch, returned as its uint8 storage."""
    layer_norm_pack, _ = _pack_primitives()
    packed, scales = layer_norm_pack(x, weight, activation_global_scale, bias=bias, eps=eps)
    return packed.view(torch.uint8), scales.view(torch.uint8)


def _layer_norm_pack_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation_global_scale: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shapes/dtypes/device of the packed data and blocked scales, keeping ``M`` symbolic."""
    return _fake_pack_outputs(x)


def _scaled_gelu_pack_impl(
    raw: torch.Tensor,
    output_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runtime implementation: one fused rescale+bias+GELU+NVFP4-pack Triton launch, as its uint8 storage."""
    _, scaled_gelu_pack = _pack_primitives()
    packed, scales = scaled_gelu_pack(raw, output_global_scale, activation_global_scale, bias=bias)
    return packed.view(torch.uint8), scales.view(torch.uint8)


def _scaled_gelu_pack_fake(
    raw: torch.Tensor,
    output_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shapes/dtypes/device of the packed data and blocked scales, keeping ``M`` symbolic."""
    return _fake_pack_outputs(raw)


def _fake_pack_outputs(producer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate empty pack outputs with the MSLK layout, without reading any leading dimension as a python int.

    The padding arithmetic is written with ``//`` and ``+`` only -- never a comparison -- so a symbolic ``M``
    survives tracing instead of being specialized by a guard. ``K`` is a static weight dimension in every caller.
    """
    m_dim, k_dim = producer.shape[0], producer.shape[1]
    packed = producer.new_empty((m_dim, k_dim // 2), dtype=torch.uint8)
    padded_rows = ((m_dim + SCALE_TILE_ROWS - 1) // SCALE_TILE_ROWS) * SCALE_TILE_ROWS
    num_blocks = k_dim // NVFP4_BLOCK_SIZE
    padded_cols = ((num_blocks + SCALE_TILE_COLS - 1) // SCALE_TILE_COLS) * SCALE_TILE_COLS
    return packed, producer.new_empty((padded_rows, padded_cols), dtype=torch.uint8)


def _apply_output_scale_and_bias(
    raw: torch.Tensor, output_scale: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Apply TorchAO's unfolded epilogue: rescale the BF16 GEMM result, then add the BF16 bias separately."""
    out = raw * output_scale
    if bias is not None:
        out = out + bias
    return out


def _require_transformer_block(block: Any, fqn: str) -> None:
    """Reject anything that is not a Sortformer ``TransformerBlock`` with the expected sublayer structure."""
    # Imported lazily so that importing this utility does not pull in the ASR modules package.
    from nemo.collections.asr.modules.transformer_encoder import FeedForward, MultiHeadAttention, TransformerBlock

    if not isinstance(block, TransformerBlock):
        raise RuntimeError(
            f"NVFP4 producer fusion expects '{fqn}' to be a TransformerBlock, got {type(block).__name__}."
        )
    if not isinstance(block.attn, MultiHeadAttention) or not hasattr(block.attn, "forward_from_qkv"):
        raise RuntimeError(
            f"NVFP4 producer fusion drives the attention body of '{fqn}' through "
            f"MultiHeadAttention.forward_from_qkv, which {type(block.attn).__name__} does not provide."
        )
    # The exact sequence matters: the fused path evaluates net.0 -> net.1 -> net.3 itself and skips net.2,
    # which is an identity only because it is a Dropout module in eval mode.
    expected_ffn = (torch.nn.Linear, torch.nn.GELU, torch.nn.Dropout, torch.nn.Linear)
    if (
        not isinstance(block.ffn, FeedForward)
        or len(block.ffn.net) != len(expected_ffn)
        or not all(isinstance(layer, cls) for layer, cls in zip(block.ffn.net, expected_ffn))
    ):
        raise RuntimeError(
            f"NVFP4 producer fusion expects '{fqn}.ffn' to be the Sortformer FeedForward with an "
            f"(up, activation, dropout, down) sequence, got {type(block.ffn).__name__} with "
            f"{[type(layer).__name__ for layer in getattr(block.ffn, 'net', [])]}."
        )
    if not isinstance(block.ffn.net[1], torch.nn.GELU) or block.ffn.net[1].approximate != 'none':
        raise RuntimeError(
            f"NVFP4 producer fusion fuses the exact erf GELU of '{fqn}.ffn.net.1', but that module is "
            f"{type(block.ffn.net[1]).__name__} with approximate="
            f"{getattr(block.ffn.net[1], 'approximate', None)!r}."
        )


def _require_restored_linear(linear: Any, fqn: str, tensor_cls: type) -> None:
    """Require a ``torch.nn.Linear`` whose weight is NOT an NVFP4 tensor, i.e. genuinely restored to BF16.

    The mirror of :func:`_require_converted_linear`. Asserting this matters because the summary reports the
    restoration: without it, a selection that merely omits ``attn.out_proj`` would fuse a block whose out_proj is
    still converted, and the run would claim a precision it does not execute.
    """
    weight = getattr(linear, "weight", None)
    if not isinstance(linear, torch.nn.Linear):
        raise RuntimeError(
            f"NVFP4 producer fusion expects the restored '{fqn}' to be a torch.nn.Linear, but found "
            f"{type(linear).__name__}."
        )
    if isinstance(weight, tensor_cls):
        raise RuntimeError(
            f"NVFP4 producer fusion was told '{fqn}' is restored to BF16, but its weight is an "
            f"{tensor_cls.__name__}. The selection and the converted model disagree; refusing to report a "
            "restoration the block would not execute."
        )


def _require_converted_linear(linear: Any, fqn: str, tensor_cls: type) -> Any:
    """Require a ``torch.nn.Linear`` whose weight is a converted, swizzled, activation-quantized NVFP4 tensor."""
    weight = getattr(linear, "weight", None)
    if not isinstance(linear, torch.nn.Linear) or not isinstance(weight, tensor_cls):
        raise RuntimeError(
            f"NVFP4 producer fusion expects '{fqn}' to be a TorchAO-converted torch.nn.Linear whose weight is an "
            f"{tensor_cls.__name__}, but found {type(linear).__name__} with weight {type(weight).__name__}. "
            "Fusion must run after the static NVFP4 conversion step."
        )
    if not bool(getattr(weight, "is_swizzled_scales", False)):
        raise RuntimeError(
            f"NVFP4 producer fusion requires swizzled blocked scales, but the weight of '{fqn}' reports "
            "is_swizzled_scales=False. Convert with the accelerated MSLK packing path."
        )
    if getattr(weight, "act_quant_kwargs", None) is None:
        raise RuntimeError(
            f"The weight of '{fqn}' has no 'act_quant_kwargs', so it is a weight-only NVFP4 tensor. Producer "
            "fusion packs activations to NVFP4 and requires the W4A4 conversion."
        )
    return weight


def _validated_layer_norm(norm: Any, name: str, d_model: int, fqn: str) -> float:
    """Require a BF16 affine LayerNorm over the residual width and return its epsilon."""
    if not isinstance(norm, torch.nn.LayerNorm):
        raise RuntimeError(
            f"NVFP4 producer fusion fuses '{fqn}.{name}' into the following pack, but it is "
            f"{type(norm).__name__} rather than a torch.nn.LayerNorm."
        )
    if tuple(norm.normalized_shape) != (d_model,):
        raise ValueError(f"'{fqn}.{name}' must normalize over ({d_model},), got {tuple(norm.normalized_shape)}.")
    _require_bf16_vector(norm.weight, f"{fqn}.{name}.weight", d_model, required=True)
    _require_bf16_vector(norm.bias, f"{fqn}.{name}.bias", d_model, required=False)
    return float(norm.eps)


def _validated_bias(linear: torch.nn.Module, out_features: int, fqn: str) -> Optional[torch.Tensor]:
    """Return the linear's bias, requiring the BF16 ``(out_features,)`` layout the fused epilogue adds."""
    bias = getattr(linear, "bias", None)
    _require_bf16_vector(bias, f"{fqn}.bias", out_features, required=False)
    return bias


def _require_bf16_vector(vector: Any, name: str, size: int, required: bool) -> None:
    """Require a contiguous BF16 ``(size,)`` tensor, or ``None`` when the vector is optional."""
    if vector is None:
        if required:
            raise ValueError(f"NVFP4 producer fusion requires '{name}' to exist with shape ({size},).")
        return
    if not isinstance(vector, torch.Tensor):
        raise ValueError(f"'{name}' must be a torch.Tensor or None, got {type(vector).__name__}.")
    if vector.dim() != 1 or int(vector.shape[0]) != size:
        raise ValueError(f"'{name}' must have shape ({size},), got {tuple(vector.shape)}.")
    if vector.dtype != torch.bfloat16:
        raise ValueError(
            f"'{name}' must be torch.bfloat16 for the fused NVFP4 path, got {vector.dtype}. Cast the model to "
            "bfloat16 before quantizing."
        )
    if not vector.is_contiguous():
        raise ValueError(f"'{name}' must be contiguous.")


def _require_scalar_scale(scale: Any, kind: str, fqn: str) -> torch.Tensor:
    """Validate a calibrated per-tensor scale and return it as a detached float32 tensor."""
    attribute = "act_per_tensor_scale" if kind == "act" else "per_tensor_scale"
    if not isinstance(scale, torch.Tensor):
        raise RuntimeError(
            f"NVFP4 producer fusion requires a calibrated '{attribute}' on the weight of '{fqn}', but found "
            f"{type(scale).__name__}. This path is for scale_mode='static' conversions only."
        )
    if scale.numel() != 1:
        raise RuntimeError(f"'{attribute}' of '{fqn}' must be a one-element tensor, got {tuple(scale.shape)}.")
    if not scale.is_floating_point():
        raise RuntimeError(f"'{attribute}' of '{fqn}' must be a floating point tensor, got dtype {scale.dtype}.")
    return scale.detach().to(torch.float32)


def _require_attribute(owner: Any, name: str, fqn: str) -> torch.Tensor:
    """Read a required tensor attribute off a TorchAO tensor, naming the module when it is missing."""
    value = getattr(owner, name, None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"NVFP4 producer fusion requires '{name}' on the quantized weight of '{fqn}', but "
            f"{type(owner).__name__} exposes {type(value).__name__}. The installed TorchAO does not match the "
            "NVFP4 layout this path is written against."
        )
    return value


def _as_uint8(tensor: torch.Tensor, fqn: str) -> torch.Tensor:
    """View packed FP4 data or FP8 blocked scales as their uint8 storage, rejecting any other element type."""
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return tensor.view(torch.uint8)
    raise RuntimeError(
        f"NVFP4 producer fusion of '{fqn}' expects packed data as torch.float4_e2m1fn_x2 and blocked scales as "
        f"torch.float8_e4m3fn (or their uint8 storage), got {tensor.dtype}."
    )


def _require_same_device(fqn: str, *tensors: Optional[torch.Tensor]) -> None:
    """Require every execution operand of one fused linear to live on a single device."""
    devices = {tensor.device for tensor in tensors if tensor is not None}
    if len(devices) != 1:
        raise RuntimeError(
            f"NVFP4 producer fusion of '{fqn}' requires its qdata, block scales, calibrated scales and bias on one "
            f"device, but found {sorted(str(device) for device in devices)}."
        )


def _matched_member(fqn: str) -> Optional[str]:
    """Return the block-member suffix that anchors the end of ``fqn``, or ``None`` when it matches none."""
    for member in PRODUCER_FUSION_BLOCK_MEMBERS:
        if fqn == member or fqn.endswith("." + member):
            return member
    return None


def _swap_module(model: torch.nn.Module, fqn: str, replacement: torch.nn.Module) -> None:
    """Replace exactly one submodule in place, addressed by its full name."""
    parent_name, _, attribute = fqn.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, replacement)


def _model_device(model: torch.nn.Module) -> torch.device:
    """Device of the model's first parameter or buffer, defaulting to CPU."""
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _op_is_registered(name: str) -> bool:
    """Whether ``namespace::op`` already exists on ``torch.ops``."""
    namespace, _, op_name = name.partition("::")
    try:
        return hasattr(getattr(torch.ops, namespace), op_name)
    except (AttributeError, RuntimeError):  # pragma: no cover - depends on the torch build
        return False


def _require_torch_dtypes(entry_point: str) -> None:
    """Raise unless this torch build exposes the FP4/FP8 dtypes the fused contract needs."""
    missing = [name for name in REQUIRED_TORCH_DTYPES if not isinstance(getattr(torch, name, None), torch.dtype)]
    if missing:
        raise RuntimeError(
            f"{entry_point} requires torch dtypes {missing}, which torch {torch.__version__} does not expose."
        )


def _pack_primitives() -> Tuple[Any, Any]:
    """
    Import the accepted fused pack primitives on first use and cache them.

    They live in the NeMo ASR collection, so importing them eagerly would import Triton on any install that has
    it, which is exactly what the disabled/CPU path must not do.
    """
    global _pack_primitives_cache
    if _pack_primitives_cache is None:
        from nemo.collections.asr.parts.utils.sortformer_nvfp4_fused_pack import (
            layer_norm_nvfp4_pack_triton,
            scaled_gelu_nvfp4_pack_triton,
        )

        _pack_primitives_cache = (layer_norm_nvfp4_pack_triton, scaled_gelu_nvfp4_pack_triton)
    return _pack_primitives_cache


def _require_api(api: str):
    """Import a ``module:attribute`` entry point lazily, raising an actionable error when it is unavailable."""
    module_name, _, attribute = api.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:  # pragma: no cover - depends on the install
        raise RuntimeError(
            f"NVFP4 producer fusion requires '{api}', but '{module_name}' could not be imported. Install a TorchAO "
            "build that exports it."
        ) from err
    resolved = getattr(module, attribute, None)
    if resolved is None:
        raise RuntimeError(f"NVFP4 producer fusion requires '{api}', which the installed TorchAO does not provide.")
    return resolved

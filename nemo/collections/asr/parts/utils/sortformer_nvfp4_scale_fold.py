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
Global-scale folding for static-NVFP4 Sortformer linears.

TorchAO's static NVFP4 linear quantizes the activation with a calibrated per-tensor scale ``a_g``, quantizes the
weight with ``w_g``, calls ``torch._scaled_mm`` with the two *normalized* FP8 block-scale tensors, and only then
materializes the result: ``out = mm_result * (a_g * w_g) + bias``. That trailing rescale-and-bias is a separate
full-size elementwise pass over the GEMM output.

The path in this module removes it. The calibrated product ``P = a_g * w_g`` is redistributed into the two FP8
block-scale operands that ``_scaled_mm`` already consumes, so the native kernel produces the final BF16 values and
applies the bias itself:

* activation block scales are rebased by ``F_a = 2 ** exponent`` at runtime -- a power of two, so an E4M3 block
  scale is rescaled exactly unless it leaves the E4M3 range;
* weight block scales are rebased once, at conversion time, by ``P / F_a``.

Neither the activation nor the weight FP4 ``qdata`` changes: packing still uses the original calibrated ``a_g``,
so the quantized values remain exactly the matched static NVFP4 ones and only the dequantization scales move.

The rebased weight block scales are re-rounded into E4M3, so a folded linear is numerically close to, but not
bit-identical with, the unfolded static NVFP4 path; a measured TorchAO 0.17 sweep at ``K=512, N=1536`` puts the
deviation near 3.7% at the rounding floor and 4% at the default exponent, against a static-NVFP4-to-BF16
deviation of 14% that folding leaves essentially unchanged. Accuracy is gated on final DER, not on that number.

``exponent`` is configurable because the split between the two operands is the only free parameter and it is what
a final-DER sweep varies; it is bounded by :data:`FOLD_EXPONENT_MIN` / :data:`FOLD_EXPONENT_MAX` so that a sweep
cannot silently ask for a fold that annihilates the weight scales.

The path is fail-closed throughout: a TorchAO layout this module does not recognize, a non-static activation
scale, unswizzled scales, a disabled MSLK packing path, or a weight scale that underflows or overflows E4M3 after
folding all raise. There is no fallback to BF16, to reference NVFP4, or to the unfolded global-scale path. TorchAO
is imported lazily, so importing this module on a CPU-only install neither imports nor requires it.

Like the rest of the Sortformer quantization utilities, a folded model is an in-process evaluation artifact:
quantized ``.nemo`` export is not supported.
"""

import importlib
import inspect
from typing import Any, Dict, Optional, Sequence

import torch

# Bounds on the configurable activation exponent. Wide enough for a controlled sweep around the working point,
# narrow enough that a typo cannot request a fold that no calibrated scale could survive.
FOLD_EXPONENT_MIN = -24
FOLD_EXPONENT_MAX = 8

# TorchAO entry point, addressed as ``module:attribute`` so that availability can be probed and mocked.
NVFP4_TENSOR_API = "torchao.prototype.mx_formats.nvfp4_tensor:NVFP4Tensor"

# Keyword contract of ``NVFP4Tensor.to_nvfp4`` that the accelerated static packing path depends on.
REQUIRED_PACK_KWARGS = ("per_tensor_scale", "is_swizzled_scales", "use_triton_kernel")

REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn")

# Accelerated NVFP4 packing requires ``K`` to be a multiple of 64, so that the block count ``K // 16`` is already a
# multiple of the 4-wide blocked-scale tile.
REQUIRED_K_MULTIPLE = 64

# Leading-shape behaviour supported by TorchAO's NVFP4 linear that this module reproduces.
SUPPORTED_INPUT_DIMS = (2, 3)


class ScaleFoldedNVFP4Linear(torch.nn.Module):
    """
    Static-NVFP4 linear whose calibrated global scales live inside the ``_scaled_mm`` block-scale operands.

    The module keeps the converted TorchAO weight (and therefore its FP4 ``qdata`` and calibrated scales) so that
    provenance is preserved, but execution never touches the post-GEMM global rescale: it packs the activation
    with the original ``a_g``, rebases the returned block scales by ``2 ** exponent``, and calls ``_scaled_mm``
    against the pre-folded weight block scales with a native ``bias=``.
    """

    def __init__(
        self,
        linear: torch.nn.Module,
        fqn: str,
        exponent: int,
        nvfp4_tensor_cls: Optional[type] = None,
    ):
        """
        Args:
            linear (torch.nn.Module): TorchAO static-NVFP4 converted ``torch.nn.Linear``.
            fqn (str): Fully qualified name of ``linear``, used in provenance and in every error message.
            exponent (int): Activation fold exponent; ``F_a = 2 ** exponent``.
            nvfp4_tensor_cls (Optional[type]): Injected ``NVFP4Tensor`` class. Resolved lazily from TorchAO when
                ``None``.

        Raises:
            RuntimeError: If the module is not a recognizable TorchAO static NVFP4 converted linear, if the
                required torch dtypes or TorchAO APIs are unavailable, or if folding would destroy a weight scale.
            ValueError: If ``exponent`` is not an integer inside the documented bounds.
        """
        super().__init__()
        self.fqn = str(fqn)
        self.fold_exponent = validate_fold_exponent(exponent)
        self.activation_scale_factor = float(2.0**self.fold_exponent)

        _require_torch_dtypes(f"NVFP4 global-scale folding of '{self.fqn}'")
        tensor_cls = nvfp4_tensor_cls if nvfp4_tensor_cls is not None else _require_api(NVFP4_TENSOR_API)
        self._to_nvfp4 = _require_pack_entry_point(tensor_cls, self.fqn)

        weight = getattr(linear, "weight", None)
        if not isinstance(linear, torch.nn.Linear) or not isinstance(weight, tensor_cls):
            raise RuntimeError(
                f"NVFP4 global-scale folding expects '{self.fqn}' to be a TorchAO-converted torch.nn.Linear whose "
                f"weight is an {tensor_cls.__name__}, but found {type(linear).__name__} with weight "
                f"{type(weight).__name__}. Folding must run after the static NVFP4 conversion step."
            )

        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        if self.in_features % REQUIRED_K_MULTIPLE != 0:
            raise RuntimeError(
                f"NVFP4 global-scale folding requires in_features % {REQUIRED_K_MULTIPLE} == 0, but '{self.fqn}' "
                f"has in_features={self.in_features}; accelerated packing would not apply to it."
            )
        if not bool(getattr(weight, "is_swizzled_scales", False)):
            raise RuntimeError(
                f"NVFP4 global-scale folding requires swizzled blocked scales, but the weight of '{self.fqn}' "
                "reports is_swizzled_scales=False. Convert with the accelerated MSLK packing path."
            )
        if not hasattr(weight, "act_quant_kwargs"):
            raise RuntimeError(
                f"The weight of '{self.fqn}' ({type(weight).__name__}) does not expose 'act_quant_kwargs'; this "
                "TorchAO version does not match the NVFP4 layout this path is written against."
            )

        activation_global_scale = _require_scalar_scale(
            getattr(weight, "act_per_tensor_scale", None), "act_per_tensor_scale", self.fqn
        )
        weight_global_scale = _require_scalar_scale(
            getattr(weight, "per_tensor_scale", None), "per_tensor_scale", self.fqn
        )
        weight_qdata = _as_fp4(_require_attribute(weight.t(), "qdata", f"{self.fqn} (transposed weight)"))
        # TorchAO reaches the native weight operand through ``weight.t().scale.t()``, which is the *original*
        # ``weight.scale`` layout. Transposing ``weight.scale`` here would transpose it a second time.
        weight_scale = _as_float32(_require_attribute(weight, "scale", self.fqn))
        _require_same_device(self.fqn, weight_qdata, weight_scale, activation_global_scale, weight_global_scale)

        global_scale_product = activation_global_scale.reshape(()) * weight_global_scale.reshape(())
        folded_weight_scale = _fold_weight_scale(
            weight_scale, global_scale_product / self.activation_scale_factor, self.fqn, self.fold_exponent
        )

        # Kept for provenance and so that the converted weight stays reachable under its original name. A
        # ``Parameter`` is registered as one; a bare TorchAO tensor subclass is held as a plain attribute, exactly
        # as ``torch.nn.Linear`` held it after conversion.
        self.weight = weight
        self.bias = _validated_bias(linear, self.out_features, self.fqn)
        self.register_buffer("weight_qdata", weight_qdata, persistent=False)
        self.register_buffer("weight_scale_folded", folded_weight_scale, persistent=False)
        self.register_buffer("activation_global_scale", activation_global_scale, persistent=False)
        self.register_buffer("weight_global_scale", weight_global_scale, persistent=False)
        self.register_buffer("global_scale_product", global_scale_product, persistent=False)
        self.train(linear.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the folded static-NVFP4 matmul.

        Args:
            x (torch.Tensor): BF16 activation of rank 2 or 3 whose last dimension is ``in_features``.

        Returns:
            output (torch.Tensor): BF16 tensor with ``x``'s leading dimensions and ``out_features`` features.

        Raises:
            ValueError: If the input rank, feature count, or device does not match this module.
        """
        if x.dim() not in SUPPORTED_INPUT_DIMS:
            raise ValueError(
                f"NVFP4 global-scale folding at '{self.fqn}' supports rank {list(SUPPORTED_INPUT_DIMS)} inputs, "
                f"got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"NVFP4 global-scale folding at '{self.fqn}' expects a last dimension of {self.in_features}, got "
                f"shape {tuple(x.shape)}."
            )
        if x.device != self.weight_qdata.device:
            raise ValueError(
                f"NVFP4 global-scale folding at '{self.fqn}' expects the activation on {self.weight_qdata.device}, "
                f"got a tensor on {x.device}."
            )

        # Packing keeps the original calibrated ``a_g``, so the FP4 qdata is exactly the matched static NVFP4
        # qdata; only the returned normalized block scales are rebased, by an exact power of two.
        activation = self._to_nvfp4(
            x.reshape(-1, self.in_features),
            per_tensor_scale=self.activation_global_scale,
            is_swizzled_scales=True,
            use_triton_kernel=True,
        )
        activation_scale = _as_float32(activation.scale) * self.activation_scale_factor
        output = torch._scaled_mm(
            _as_fp4(activation.qdata),
            self.weight_qdata,
            activation_scale.to(torch.float8_e4m3fn),
            self.weight_scale_folded,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        )
        return output.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        """Shape, bias and fold provenance, in the style of ``torch.nn.Linear``."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"fold_exponent={self.fold_exponent}"
        )


def validate_fold_exponent(exponent: Any) -> int:
    """
    Coerce and bound-check the activation fold exponent.

    Args:
        exponent (Any): Candidate exponent; ``F_a = 2 ** exponent``.

    Returns:
        exponent (int): The validated exponent.

    Raises:
        ValueError: If the value is not an integer (``bool`` included) or lies outside the documented bounds.
    """
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise ValueError(
            f"quantization_fold_activation_exponent must be an integer, got {exponent!r} "
            f"({type(exponent).__name__})."
        )
    if not FOLD_EXPONENT_MIN <= exponent <= FOLD_EXPONENT_MAX:
        raise ValueError(
            f"quantization_fold_activation_exponent must be in [{FOLD_EXPONENT_MIN}, {FOLD_EXPONENT_MAX}], "
            f"got {exponent}."
        )
    return int(exponent)


def apply_global_scale_folding(
    model: torch.nn.Module,
    fqns: Sequence[str],
    exponent: int,
    nvfp4_tensor_cls: Optional[type] = None,
) -> Dict[str, Any]:
    """
    Replace exactly the given converted static-NVFP4 linears with folded ones.

    Only the listed FQNs are considered: there is no suffix matching and no traversal beyond a direct lookup, so
    this can never widen the set of modules chosen by the quantization recipe.

    Args:
        model (torch.nn.Module): Model whose selected linears TorchAO has already converted in place.
        fqns (Sequence[str]): Exact FQNs selected for NVFP4 W4A4.
        exponent (int): Activation fold exponent.
        nvfp4_tensor_cls (Optional[type]): Injected ``NVFP4Tensor`` class; resolved from TorchAO when ``None``.

    Returns:
        summary (Dict[str, Any]): Whether folding is enabled, the exponent and its factor, and the number and
            names of the wrapped modules.

    Raises:
        RuntimeError: If an FQN is absent, is not a converted static NVFP4 linear, or cannot be folded.
        ValueError: If the exponent is invalid or no FQN was given.
    """
    resolved_exponent = validate_fold_exponent(exponent)
    selected = sorted(str(fqn) for fqn in fqns)
    if not selected:
        raise ValueError(
            "NVFP4 global-scale folding was requested but no NVFP4 W4A4 module was selected; the recipe would "
            "have nothing to fold."
        )
    _require_torch_dtypes("NVFP4 global-scale folding")
    tensor_cls = nvfp4_tensor_cls if nvfp4_tensor_cls is not None else _require_api(NVFP4_TENSOR_API)

    modules = dict(model.named_modules())
    for fqn in selected:
        module = modules.get(fqn)
        if module is None:
            raise RuntimeError(
                f"NVFP4 global-scale folding could not find the selected module '{fqn}'; refusing to fold a "
                "partial set of layers."
            )
        folded = ScaleFoldedNVFP4Linear(module, fqn=fqn, exponent=resolved_exponent, nvfp4_tensor_cls=tensor_cls)
        _swap_module(model, fqn, folded)

    return {
        "enabled": True,
        "activation_exponent": resolved_exponent,
        "activation_scale_factor": float(2.0**resolved_exponent),
        "wrapped_count": len(selected),
        "wrapped_fqns": selected,
        "notes": [
            "The calibrated activation/weight global-scale product is carried by the _scaled_mm block-scale "
            "operands; the GEMM applies the linear bias natively and no post-GEMM rescale is performed.",
        ],
    }


def disabled_folding_summary() -> Dict[str, Any]:
    """Summary block reported when global-scale folding was not requested."""
    return {
        "enabled": False,
        "activation_exponent": None,
        "activation_scale_factor": None,
        "wrapped_count": 0,
        "wrapped_fqns": [],
        "notes": [],
    }


def _fold_weight_scale(weight_scale: torch.Tensor, fold_factor: torch.Tensor, fqn: str, exponent: int) -> torch.Tensor:
    """
    Rebase the normalized weight block scales by ``P / F_a`` and reject a fold that destroys any of them.

    Runs once, at conversion time, so the host synchronization implied by the check is paid exactly once per
    module and never in the forward path.

    Raises:
        RuntimeError: If a finite nonzero original scale becomes zero, NaN, or infinite after folding.
    """
    folded = (weight_scale * fold_factor).to(torch.float8_e4m3fn)
    original_usable = torch.isfinite(weight_scale) & (weight_scale != 0)
    folded_f32 = folded.to(torch.float32)
    destroyed = original_usable & (~torch.isfinite(folded_f32) | (folded_f32 == 0))
    if bool(destroyed.any().item()):
        raise RuntimeError(
            f"NVFP4 global-scale folding with quantization_fold_activation_exponent={exponent} destroys "
            f"{int(destroyed.sum().item())} of {int(original_usable.sum().item())} usable weight block scales of "
            f"'{fqn}': they underflow or overflow float8_e4m3fn after being rebased by the global-scale product. "
            "Choose an exponent that keeps the folded weight scales representable."
        )
    return folded.contiguous()


def _validated_bias(linear: torch.nn.Module, out_features: int, fqn: str) -> Optional[torch.nn.Parameter]:
    """Return the linear's bias, requiring the shape and dtype that native ``_scaled_mm`` accepts."""
    bias = getattr(linear, "bias", None)
    if bias is None:
        return None
    if bias.dtype != torch.bfloat16:
        raise RuntimeError(
            f"NVFP4 global-scale folding applies the bias of '{fqn}' inside _scaled_mm, which requires a "
            f"torch.bfloat16 bias, but this one is {bias.dtype}."
        )
    if bias.dim() != 1 or int(bias.shape[0]) != out_features:
        raise RuntimeError(f"The bias of '{fqn}' must have shape ({out_features},), got {tuple(bias.shape)}.")
    return bias


def _require_scalar_scale(scale: Any, name: str, fqn: str) -> torch.Tensor:
    """Validate a calibrated per-tensor scale and return it as a float32 scalar tensor."""
    if not isinstance(scale, torch.Tensor):
        raise RuntimeError(
            f"NVFP4 global-scale folding requires a calibrated '{name}' on the weight of '{fqn}', but found "
            f"{type(scale).__name__}. This path is for scale_mode='static' conversions only."
        )
    if scale.numel() != 1:
        raise RuntimeError(f"'{name}' of '{fqn}' must be a one-element tensor, got {tuple(scale.shape)}.")
    if not scale.is_floating_point():
        raise RuntimeError(f"'{name}' of '{fqn}' must be a floating point tensor, got dtype {scale.dtype}.")
    return scale.detach().to(torch.float32)


def _require_attribute(owner: Any, name: str, fqn: str) -> torch.Tensor:
    """Read a required tensor attribute off a TorchAO tensor, naming the module when it is missing."""
    value = getattr(owner, name, None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"NVFP4 global-scale folding requires '{name}' on the quantized weight of '{fqn}', but "
            f"{type(owner).__name__} exposes {type(value).__name__}. The installed TorchAO does not match the "
            "NVFP4 layout this path is written against."
        )
    return value


def _require_same_device(fqn: str, *tensors: torch.Tensor) -> None:
    """Require every execution operand of one folded linear to live on a single device."""
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RuntimeError(
            f"NVFP4 global-scale folding of '{fqn}' requires its qdata, block scales and calibrated scales on one "
            f"device, but found {sorted(str(device) for device in devices)}."
        )


def _as_float32(scale: torch.Tensor) -> torch.Tensor:
    """Read an FP8 blocked-scale tensor (or its uint8 storage) as float32."""
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e4m3fn)
    if scale.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"NVFP4 blocked scales must be torch.float8_e4m3fn (or its uint8 storage), got {scale.dtype}."
        )
    return scale.to(torch.float32)


def _as_fp4(qdata: torch.Tensor) -> torch.Tensor:
    """Read packed NVFP4 data (or its uint8 storage) as ``torch.float4_e2m1fn_x2``."""
    if qdata.dtype == torch.uint8:
        return qdata.view(torch.float4_e2m1fn_x2)
    if qdata.dtype != torch.float4_e2m1fn_x2:
        raise RuntimeError(
            f"NVFP4 packed data must be torch.float4_e2m1fn_x2 (or its uint8 storage), got {qdata.dtype}."
        )
    return qdata


def _require_pack_entry_point(tensor_cls: type, fqn: str):
    """Resolve ``NVFP4Tensor.to_nvfp4`` and require the keyword contract the accelerated static path depends on."""
    to_nvfp4 = getattr(tensor_cls, "to_nvfp4", None)
    if to_nvfp4 is None:
        raise RuntimeError(
            f"NVFP4 global-scale folding of '{fqn}' requires {tensor_cls.__name__}.to_nvfp4, which the installed "
            "TorchAO does not provide."
        )
    try:
        parameters = inspect.signature(to_nvfp4).parameters
    except (TypeError, ValueError) as err:  # pragma: no cover - depends on the install
        raise RuntimeError(
            f"NVFP4 global-scale folding of '{fqn}' could not inspect {tensor_cls.__name__}.to_nvfp4."
        ) from err
    missing = [name for name in REQUIRED_PACK_KWARGS if name not in parameters]
    if missing:
        raise RuntimeError(
            f"NVFP4 global-scale folding of '{fqn}' packs activations through {tensor_cls.__name__}.to_nvfp4 with "
            f"{list(REQUIRED_PACK_KWARGS)}, but the installed TorchAO signature is missing {missing}."
        )
    return to_nvfp4


def _swap_module(model: torch.nn.Module, fqn: str, replacement: torch.nn.Module) -> None:
    """Replace exactly one submodule in place, addressed by its full name."""
    parent_name, _, attribute = fqn.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, replacement)


def _require_torch_dtypes(entry_point: str) -> None:
    """Raise unless this torch build exposes the FP4/FP8 dtypes the folded contract needs."""
    missing = [name for name in REQUIRED_TORCH_DTYPES if not isinstance(getattr(torch, name, None), torch.dtype)]
    if missing:
        raise RuntimeError(
            f"{entry_point} requires torch dtypes {missing}, which torch {torch.__version__} does not expose."
        )


def _require_api(api: str):
    """Import a ``module:attribute`` entry point lazily, raising an actionable error when it is unavailable."""
    module_name, _, attribute = api.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:  # pragma: no cover - depends on the install
        raise RuntimeError(
            f"NVFP4 global-scale folding requires '{api}', but '{module_name}' could not be imported. Install a "
            "TorchAO build that exports it."
        ) from err
    resolved = getattr(module, attribute, None)
    if resolved is None:
        raise RuntimeError(
            f"NVFP4 global-scale folding requires '{api}', which the installed TorchAO does not provide."
        )
    return resolved

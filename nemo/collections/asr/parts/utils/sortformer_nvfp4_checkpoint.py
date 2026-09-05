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
``SaveRestoreConnector`` that serializes an NVFP4-quantized Sortformer model to a ``.nemo`` archive.

A model quantized by the NVFP4 recipe holds TorchAO ``NVFP4Tensor`` weights, which the default
connector cannot write. This connector stores them so that ``restore_from`` reconstructs the
quantized model directly, without the calibration artifacts or recipe options that produced it.

Archive layout
--------------
Alongside NeMo's usual ``model_config.yaml``, the archive holds two members:

``model_weights.safetensors``
    Unquantized weights unchanged. Each quantized weight is decomposed with
    ``NVFP4Tensor.__tensor_flatten__`` into four tensors -- the packed FP4 ``qdata``, the
    per-16-element-block E4M3 ``scale``, the ``per_tensor_scale`` weight global scale and the
    ``act_per_tensor_scale`` static activation scale -- stored under ``<parameter>::<attribute>``
    keys. The separator identifies which entries are payload, so no list of quantized weights is
    stored anywhere.

``quantization_config.json``
    The flatten context needed to rebuild an ``NVFP4Tensor``, plus provenance. Keeping it a separate
    member rather than safetensors metadata makes it readable without a binary parse, and marks the
    archive as this format.

The context is stored once, not per weight: this recipe quantizes every target identically, so all
93 contexts of a produced checkpoint were byte-identical. Export refuses a model whose weights
disagree rather than silently keeping one of them; per-weight contexts can be added as a format
version when a recipe needs them.

The flattened primitives are stored rather than a pickled ``NVFP4Tensor`` because ``torch.save`` of
the subclass binds the archive to one TorchAO release: 0.17 and 0.18 disagree on the
``is_swizzled_scales`` default, which changes the scale layout. Recording that flag in the context
and reconstructing through the installed TorchAO keeps the archive readable across both.

Restore
-------
Reconstructed weights are assigned onto their modules directly, because a model built from the
archive's config has BF16 ``torch.nn.Linear`` weights whose shape and dtype do not accept a packed
NVFP4 payload. The remaining weights load normally, and the keys missing from that load are required
to equal exactly the reconstructed ones, so bypassing ``load_state_dict`` does not weaken it.

:func:`resolve_nvfp4_save_restore_connector` identifies these archives by their config member, so a
caller can restore one from ``model_path`` alone.
"""

import json
import os
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import torch

from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "PAYLOAD_SEPARATOR",
    "QUANTIZATION_CONFIG_MEMBER",
    "SortformerNVFP4SaveRestoreConnector",
    "is_nvfp4_checkpoint",
    "resolve_nvfp4_save_restore_connector",
]

# Identity of this container, so a reader can refuse a file it does not understand instead of
# misinterpreting one.
CHECKPOINT_FORMAT = "sortformer_nvfp4"
CHECKPOINT_FORMAT_VERSION = 1

# Archive member holding the quantization config. Its presence is what marks an archive as this
# format.
QUANTIZATION_CONFIG_MEMBER = "quantization_config.json"

# Separates a parameter's state-dict key from the flattened attribute it carries. Chosen because it
# cannot occur in a module FQN, so splitting is unambiguous and a plain BF16 key is never mistaken
# for part of a quantized payload.
PAYLOAD_SEPARATOR = "::"

TORCHAO_VERSION_MISMATCH_MESSAGE = (
    "The checkpoint records TorchAO {recorded} but TorchAO {installed} is installed. The stored context is "
    "layout-explicit and is expected to reconstruct correctly, but the packing arithmetic was validated against "
    "the recorded version."
)

_TORCH_DTYPE_PREFIX = "torch."


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Serialize a torch dtype as its ``torch.``-qualified name."""
    return str(dtype)


def _name_to_dtype(name: str) -> torch.dtype:
    """
    Resolve a ``torch.``-qualified dtype name back to the dtype.

    Args:
        name (str): Name as produced by :func:`_dtype_to_name`, e.g. ``"torch.float32"``.

    Returns:
        dtype (torch.dtype): The resolved dtype.

    Raises:
        ValueError: If the name is not a dtype this build of PyTorch provides.
    """
    if not name.startswith(_TORCH_DTYPE_PREFIX):
        raise ValueError(f"Not a torch dtype name: {name!r}")
    resolved = getattr(torch, name[len(_TORCH_DTYPE_PREFIX) :], None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"torch does not provide the dtype {name!r} this checkpoint requires.")
    return resolved


def _installed_torchao_version() -> Optional[str]:
    """TorchAO version string, or ``None`` when TorchAO is not importable."""
    try:
        import torchao
    except ImportError:
        return None
    return getattr(torchao, "__version__", None)


def _context_to_json(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a flatten context into a JSON-representable mapping.

    Args:
        context (Dict[str, Any]): The context returned by ``NVFP4Tensor.__tensor_flatten__``.

    Returns:
        encoded (Dict[str, Any]): JSON-safe mapping carrying the same values.

    Raises:
        TypeError: If the context holds a tensor or any other value this format cannot represent.
            Silently dropping such a value would produce a checkpoint that executes different
            arithmetic, so it is refused.
    """
    encoded: Dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"Flatten context entry '{key}' is a Tensor. This format stores the context as JSON, so a tensor "
                "there would be dropped and the restored weight would not execute the exported arithmetic."
            )
        if isinstance(value, torch.dtype):
            encoded[key] = {"__dtype__": _dtype_to_name(value)}
        elif value is None or isinstance(value, (bool, int, float, str)):
            encoded[key] = value
        elif hasattr(value, "__dict__"):
            # Dataclass-like, e.g. QuantizeTensorToNVFP4Kwargs. Its type name is recorded so the
            # loader rebuilds the same class rather than guessing from the field names.
            fields = dict(vars(value))
            for field_key, field_value in fields.items():
                if isinstance(field_value, torch.Tensor):
                    raise TypeError(
                        f"Flatten context entry '{key}.{field_key}' is a Tensor, which this format cannot store "
                        "as JSON without changing the restored arithmetic."
                    )
            encoded[key] = {"__class__": type(value).__name__, "__fields__": fields}
        else:
            raise TypeError(f"Flatten context entry '{key}' has unsupported type {type(value).__name__}.")
    return encoded


def _context_from_json(encoded: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild a flatten context from its JSON form.

    Args:
        encoded (Dict[str, Any]): Mapping produced by :func:`_context_to_json`.

    Returns:
        context (Dict[str, Any]): Context suitable for ``__tensor_unflatten__``.

    Raises:
        ValueError: If a recorded class name is not available, since constructing the wrong
            activation quantization config would change what the weight executes.
    """
    context: Dict[str, Any] = {}
    for key, value in encoded.items():
        if isinstance(value, dict) and "__dtype__" in value:
            context[key] = _name_to_dtype(value["__dtype__"])
        elif isinstance(value, dict) and "__class__" in value:
            class_name = value["__class__"]
            factory = _kwargs_factory(class_name)
            if factory is None:
                raise ValueError(
                    f"The checkpoint's context entry '{key}' names {class_name!r}, which this TorchAO build does "
                    "not provide. The exported activation quantization config cannot be reconstructed."
                )
            context[key] = factory(**value["__fields__"])
        else:
            context[key] = value
    return context


def _kwargs_factory(class_name: str) -> Optional[Any]:
    """
    Constructor for a context entry's dataclass-like value, or ``None`` if TorchAO lacks it.

    Resolved from the module that defines ``NVFP4Tensor`` rather than a hardcoded path, because that
    module is already known to be importable and TorchAO has moved these helpers between releases.
    """
    try:
        from torchao.prototype.mx_formats import nvfp4_tensor
    except ImportError:
        return None
    return getattr(nvfp4_tensor, class_name, None)


def _is_quantized(tensor: Any) -> bool:
    """Whether a state-dict value is a TorchAO tensor subclass carrying a flatten contract."""
    return isinstance(tensor, torch.Tensor) and hasattr(tensor, "__tensor_flatten__")


def is_nvfp4_checkpoint(path: str) -> bool:
    """
    Whether a ``.nemo`` archive carries an NVFP4 quantization config.

    Decided from the archive's table of contents, which does not extract any member. A path that is
    not a readable tar is reported as ``False`` rather than raising, so the ordinary restore path
    still produces its own error message for a genuinely broken file.

    Args:
        path (str): Path to a ``.nemo`` archive.

    Returns:
        is_nvfp4 (bool): ``True`` when the archive contains :data:`QUANTIZATION_CONFIG_MEMBER`.
    """
    try:
        with tarfile.open(path, "r:*") as archive:
            return any(name.lstrip("./") == QUANTIZATION_CONFIG_MEMBER for name in archive.getnames())
    except (OSError, tarfile.TarError):
        return False


def resolve_nvfp4_save_restore_connector(path: str) -> Optional["SortformerNVFP4SaveRestoreConnector"]:
    """
    Connector for an NVFP4 archive, or ``None`` when the ordinary connector should be used.

    Args:
        path (str): Path to a ``.nemo`` archive.

    Returns:
        connector (Optional[SortformerNVFP4SaveRestoreConnector]): Connector to pass to
            ``restore_from``, or ``None`` for a conventional checkpoint.
    """
    if not is_nvfp4_checkpoint(path):
        return None
    logging.info(f"{path} carries an NVFP4 quantization config; restoring with the NVFP4 connector.")
    return SortformerNVFP4SaveRestoreConnector()


class SortformerNVFP4SaveRestoreConnector(SaveRestoreConnector):
    """
    ``SaveRestoreConnector`` that writes and reads NVFP4 Sortformer weights as safetensors.

    Save and restore go through the ordinary ``ModelPT.save_to`` / ``restore_from`` paths, so the
    result is a normal ``.nemo`` archive and ``push_to_hf_hub`` works unchanged.

    Args:
        export_precision (Optional[str]): Dtype the weights were cast to before conversion. Block
            amax is read from the weight as it stands, so an FP32 weight and its BF16 rounding give
            different FP4 codes; recorded so that condition is recoverable from the archive.
        source_checkpoint_sha256 (Optional[str]): Digest of the checkpoint this was quantized from,
            recorded so an artifact can be traced to its origin.
    """

    def __init__(
        self,
        export_precision: Optional[str] = None,
        source_checkpoint_sha256: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Not "model_weights.ckpt": the name marks the payload as safetensors rather than a pickle.
        self._model_weights_ckpt = "model_weights.safetensors"
        self._export_precision = export_precision
        self._source_checkpoint_sha256 = source_checkpoint_sha256
        self._restored_context: Dict[str, Any] = {}

    def _save_state_dict_to_disk(self, state_dict: Dict[str, Any], filepath: str) -> None:
        """
        Write the state dict as safetensors and the quantization config beside it.

        The config is written into the same directory because ``save_to`` archives that directory
        wholesale, so no override of the archiving step is needed.

        Args:
            state_dict (Dict[str, Any]): Model state dict, possibly holding TorchAO tensor subclasses.
            filepath (str): Destination path for the safetensors file.

        Raises:
            TypeError: If a flatten context cannot be represented as JSON.
            ValueError: If the quantized weights do not all share one flatten context.
        """
        from safetensors.torch import save_file

        tensors: Dict[str, torch.Tensor] = {}
        contexts: Dict[str, Dict[str, Any]] = {}

        for key, value in state_dict.items():
            if not _is_quantized(value):
                tensors[key] = value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
                continue

            names, context = value.__tensor_flatten__()
            for name in names:
                tensors[f"{key}{PAYLOAD_SEPARATOR}{name}"] = getattr(value, name).detach().cpu().contiguous()
            contexts[key] = _context_to_json(context)

        if not contexts:
            raise ValueError("No quantized weights found; use the ordinary connector for an unquantized model.")

        distinct = {json.dumps(c, sort_keys=True) for c in contexts.values()}
        if len(distinct) != 1:
            raise ValueError(
                f"The {len(contexts)} quantized weights carry {len(distinct)} different flatten contexts. Format "
                f"version {CHECKPOINT_FORMAT_VERSION} stores one shared context and cannot represent this model."
            )

        config = {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "producer": {
                "torchao_version": _installed_torchao_version(),
                "export_precision": self._export_precision,
                "source_checkpoint_sha256": self._source_checkpoint_sha256,
            },
            "context": json.loads(distinct.pop()),
        }
        config_path = os.path.join(os.path.dirname(filepath), QUANTIZATION_CONFIG_MEMBER)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

        save_file(tensors, filepath)
        logging.info(f"Wrote {len(contexts)} quantized and {len(tensors)} total entries to {filepath}")

    def _load_state_dict_from_disk(self, model_weights: str, map_location: Optional[Any] = "cpu") -> Dict[str, Any]:
        """
        Read the safetensors payload and the quantization config beside it.

        The context is stashed on the connector because ``restore_from`` reads the weights before it
        installs them, and reconstruction needs it.

        Args:
            model_weights (str): Path to the safetensors file inside the unpacked archive.
            map_location (Optional[Any]): Device for the loaded tensors.

        Returns:
            state_dict (Dict[str, Any]): Plain keys plus separator-suffixed quantized attributes.

        Raises:
            ValueError: If the config is absent, is not this format, or records an unreadable version.
        """
        from safetensors.torch import load_file

        config_path = os.path.join(os.path.dirname(model_weights), QUANTIZATION_CONFIG_MEMBER)
        if not os.path.isfile(config_path):
            raise ValueError(f"{model_weights} has no {QUANTIZATION_CONFIG_MEMBER} beside it.")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)

        if config.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{config_path} records format {config.get('format')!r}, not {CHECKPOINT_FORMAT!r}. This connector "
                "will not guess at the layout of an unknown container."
            )
        if config.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"{config_path} records format version {config.get('format_version')!r}, but this build reads "
                f"version {CHECKPOINT_FORMAT_VERSION}."
            )

        self._restored_context = config["context"]

        recorded = (config.get("producer") or {}).get("torchao_version")
        installed = _installed_torchao_version()
        if recorded and recorded != installed:
            logging.warning(TORCHAO_VERSION_MISMATCH_MESSAGE.format(recorded=recorded, installed=installed))

        device = "cpu" if map_location is None else str(map_location)
        return load_file(model_weights, device=device)

    def load_instance_with_state_dict(self, instance: Any, state_dict: Dict[str, Any], strict: bool) -> None:
        """
        Reconstruct quantized weights onto the instance, then load the remaining plain tensors.

        Quantized weights are assigned directly rather than routed through ``load_state_dict``,
        because the freshly built instance holds ordinary BF16 ``nn.Linear`` weights whose shape and
        dtype do not match a packed NVFP4 payload. The plain remainder is then loaded and the missing
        keys are required to be exactly the quantized ones, which preserves the strictness that
        skipping ``load_state_dict`` for those keys would otherwise lose.

        Args:
            instance (Any): Model built from the archive's config.
            state_dict (Dict[str, Any]): Output of :meth:`_load_state_dict_from_disk`.
            strict (bool): Whether unexpected or unaccounted keys are an error.

        Raises:
            ValueError: If a target module is absent, or if the plain load leaves keys unaccounted
                for while ``strict`` is set.
        """
        grouped, plain = self._partition(state_dict)
        context = _context_from_json(self._restored_context)
        modules = dict(instance.named_modules())
        installed: List[str] = []

        for key, attributes in sorted(grouped.items()):
            module_fqn, _, attribute_name = key.rpartition(".")
            module = modules.get(module_fqn)
            if module is None:
                raise ValueError(f"The checkpoint quantizes '{module_fqn}', which this model does not contain.")
            tensor = self._rebuild(attributes, context)
            setattr(module, attribute_name, torch.nn.Parameter(tensor, requires_grad=False))
            installed.append(key)

        missing, unexpected = instance.load_state_dict(plain, strict=False)
        unaccounted_missing = sorted(set(missing) - set(installed))
        if strict and (unaccounted_missing or unexpected):
            raise ValueError(
                "Restoring the NVFP4 checkpoint left keys unaccounted for. "
                f"missing={unaccounted_missing} unexpected={sorted(unexpected)}"
            )

        logging.info(f"Restored {len(installed)} quantized and {len(plain)} plain weights.")
        instance._set_model_restore_state(is_being_restored=False)

    @staticmethod
    def _partition(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Split a loaded state dict into quantized attribute groups and plain tensors."""
        grouped: Dict[str, Dict[str, torch.Tensor]] = {}
        plain: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if PAYLOAD_SEPARATOR in key:
                parameter_key, _, attribute = key.partition(PAYLOAD_SEPARATOR)
                grouped.setdefault(parameter_key, {})[attribute] = value
            else:
                plain[key] = value
        return grouped, plain

    @staticmethod
    def _rebuild(attributes: Dict[str, torch.Tensor], context: Dict[str, Any]) -> torch.Tensor:
        """
        Rebuild an ``NVFP4Tensor`` from stored primitives using the installed TorchAO.

        Args:
            attributes (Dict[str, torch.Tensor]): Flattened attribute name to tensor.
            context (Dict[str, Any]): Rebuilt flatten context.

        Returns:
            tensor (torch.Tensor): The reconstructed tensor subclass.

        Raises:
            RuntimeError: If TorchAO does not provide ``NVFP4Tensor``.
        """
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        except ImportError as error:
            raise RuntimeError(
                "Restoring an NVFP4 Sortformer checkpoint requires TorchAO's NVFP4Tensor. Install a TorchAO build "
                "that provides torchao.prototype.mx_formats.nvfp4_tensor."
            ) from error
        return NVFP4Tensor.__tensor_unflatten__(attributes, context, None, None)

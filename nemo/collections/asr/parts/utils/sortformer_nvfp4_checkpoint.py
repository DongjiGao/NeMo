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
Self-contained NVFP4 ``.nemo`` serialization for Sortformer diarization.

:mod:`~nemo.collections.asr.parts.utils.sortformer_quantization` quantizes a restored BF16 model in
place and is explicit that the result is an in-process artifact: shipping it previously meant
shipping the BF16 checkpoint plus a calibration artifact plus roughly a dozen Hydra flags, and a
consumer who mismatched one flag would silently evaluate a different model. This module makes the
quantized model itself the shippable object.

**What is stored.** A quantized weight is a TorchAO ``NVFP4Tensor`` held in an ``nn.Parameter``.
Its ``__tensor_flatten__`` decomposes it into four plain tensors -- the packed FP4 ``qdata``, the
per-16-element-block E4M3 ``scale``, the ``per_tensor_scale`` weight global scale and the
``act_per_tensor_scale`` static activation scale -- plus a context of plain scalars. Every quantized
weight of the production recipe was audited and none carries a tensor inside its context, so those
four tensors are the entire numeric payload and the context is fully JSON-representable. The payload
goes into safetensors and the context into the safetensors metadata header.

**Why not pickle the subclass.** ``torch.save`` of an ``NVFP4Tensor`` binds the artifact to one
TorchAO release. That is not hypothetical churn: TorchAO 0.17 and 0.18 disagree on the
``is_swizzled_scales`` default, which changes the on-disk scale layout, and
:mod:`~nemo.collections.asr.parts.utils.sortformer_nvfp4_weight_mse` still documents its arithmetic
against 0.17. Storing the flattened primitives together with an explicitly recorded
``is_swizzled_scales`` means the layout travels with the data and is never inherited from whatever
default the installed TorchAO happens to have. Both layouts were verified to round-trip bit-exactly.

**What is verified on load.** The recorded per-weight payload digest is recomputed and compared
before the reconstructed weight is installed, so an edited or truncated payload fails loudly rather
than quietly executing different numbers. The TorchAO version present at export is recorded and
compared, and a mismatch is reported through :data:`TORCHAO_VERSION_MISMATCH_MESSAGE` rather than
being ignored.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "PAYLOAD_SEPARATOR",
    "SortformerNVFP4SaveRestoreConnector",
    "is_nvfp4_checkpoint",
    "payload_digest",
    "resolve_nvfp4_save_restore_connector",
]

# Identity of this container, written into the safetensors metadata so a reader can refuse a file it
# does not understand instead of misinterpreting one.
CHECKPOINT_FORMAT = "sortformer_nvfp4"
CHECKPOINT_FORMAT_VERSION = 1

# Separates a parameter's state-dict key from the flattened attribute it carries. Chosen because it
# cannot occur in a module FQN, so splitting is unambiguous and a plain BF16 key is never mistaken
# for part of a quantized payload.
PAYLOAD_SEPARATOR = "::"

# Metadata keys. safetensors metadata values must be strings, so every structured value is JSON.
META_FORMAT = "format"
META_FORMAT_VERSION = "format_version"
META_TORCHAO_VERSION = "torchao_version"
META_CONTEXTS = "nvfp4_contexts"
META_DIGESTS = "payload_digests"
META_QUANTIZATION_SUMMARY = "quantization_summary"
META_SOURCE_CHECKPOINT_SHA256 = "source_checkpoint_sha256"
# The dtype the weights were cast to before conversion. Recorded because quantization is
# precision-dependent: block amax is taken from the weight as it stands, so converting an FP32
# weight and converting its BF16 rounding produce different E4M3 block scales and different FP4
# codes. An artifact exported for one serving precision is therefore not the right artifact for
# another, and without this field that mismatch is silent.
META_EXPORT_PRECISION = "export_precision"

# Canonical per-weight payload digest. Hashing each attribute's dtype and shape alongside its raw
# bytes makes a reshaped or re-dtyped payload a different payload, matching the intent of
# ``WEIGHT_DIGEST_METHOD`` in the quantization module.
PAYLOAD_DIGEST_METHOD = (
    "sha256 over sorted attributes of (name + '|' + str(dtype) + '|' + str(tuple(shape)) + '|' + "
    "contiguous_cpu_raw_bytes)"
)

TORCHAO_VERSION_MISMATCH_MESSAGE = (
    "The checkpoint records TorchAO {recorded} but TorchAO {installed} is installed. The stored payload is "
    "layout-explicit and is expected to reconstruct correctly, but the packing arithmetic was validated against "
    "the recorded version. Re-export with the installed version to remove this warning."
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


def payload_digest(attributes: Dict[str, torch.Tensor]) -> str:
    """
    Canonical digest of one quantized weight's payload.

    The digest is :data:`PAYLOAD_DIGEST_METHOD`. It is computed over the attributes in sorted order so
    that it depends on the values carried and not on dictionary ordering, and it survives the
    safetensors write/read round trip.

    Args:
        attributes (Dict[str, torch.Tensor]): Flattened attribute name to tensor.

    Returns:
        digest (str): 64-character lowercase hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    for name in sorted(attributes):
        tensor = attributes[name].detach().cpu().contiguous()
        header = f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
        digest.update(header)
        # view(uint8) needs at least one dimension, and the two global scales are 0-dim scalars.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _installed_torchao_version() -> Optional[str]:
    """TorchAO version string, or ``None`` when TorchAO is not importable."""
    try:
        import torchao
    except ImportError:
        return None
    return getattr(torchao, "__version__", None)


def _context_to_json(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one flatten context into a JSON-representable mapping.

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
                f"Flatten context entry '{key}' is a Tensor. This format stores context as JSON, so a tensor "
                "there would be dropped and the restored weight would not execute the exported arithmetic."
            )
        if isinstance(value, torch.dtype):
            encoded[key] = {"__dtype__": _dtype_to_name(value)}
        elif value is None or isinstance(value, (bool, int, float, str)):
            encoded[key] = value
        elif hasattr(value, "__dict__") and type(value).__name__:
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


def _context_from_json(encoded: Dict[str, Any], kwargs_factory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild a flatten context from its JSON form.

    Args:
        encoded (Dict[str, Any]): Mapping produced by :func:`_context_to_json`.
        kwargs_factory (Dict[str, Any]): Class name to callable for the dataclass-like entries.

    Returns:
        context (Dict[str, Any]): Context suitable for ``__tensor_unflatten__``.

    Raises:
        ValueError: If a recorded class name has no factory, since constructing the wrong activation
            quantization config would change what the weight executes.
    """
    context: Dict[str, Any] = {}
    for key, value in encoded.items():
        if isinstance(value, dict) and "__dtype__" in value:
            context[key] = _name_to_dtype(value["__dtype__"])
        elif isinstance(value, dict) and "__class__" in value:
            class_name = value["__class__"]
            factory = kwargs_factory.get(class_name)
            if factory is None:
                raise ValueError(
                    f"The checkpoint's context entry '{key}' names {class_name!r}, which this TorchAO build does "
                    "not provide. The exported activation quantization config cannot be reconstructed."
                )
            context[key] = factory(**value["__fields__"])
        else:
            context[key] = value
    return context


def _kwargs_factories() -> Dict[str, Any]:
    """Class-name to constructor map for the context's dataclass-like entries."""
    factories: Dict[str, Any] = {}
    try:
        from torchao.quantization.quantize_.common import QuantizeTensorToNVFP4Kwargs

        factories["QuantizeTensorToNVFP4Kwargs"] = QuantizeTensorToNVFP4Kwargs
    except ImportError:
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

            factories["QuantizeTensorToNVFP4Kwargs"] = QuantizeTensorToNVFP4Kwargs
        except ImportError:
            logging.debug("QuantizeTensorToNVFP4Kwargs is not importable from either known location.")
    return factories


def _is_quantized(tensor: Any) -> bool:
    """Whether a state-dict value is a TorchAO tensor subclass carrying a flatten contract."""
    return isinstance(tensor, torch.Tensor) and hasattr(tensor, "__tensor_flatten__")


# Name of the weights member that identifies an archive as this format, checked without extracting.
NVFP4_WEIGHTS_MEMBER = "model_weights.safetensors"


def is_nvfp4_checkpoint(path: str) -> bool:
    """
    Whether a ``.nemo`` archive carries an NVFP4 safetensors payload.

    Decided from the archive's table of contents, which is cheap and does not extract a 90 MB
    member. A path that is not a readable tar is reported as ``False`` rather than raising, so the
    ordinary restore path still produces its own error message for a genuinely broken file.

    Args:
        path (str): Path to a ``.nemo`` archive.

    Returns:
        is_nvfp4 (bool): ``True`` when the archive contains :data:`NVFP4_WEIGHTS_MEMBER`.
    """
    import tarfile

    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getnames():
                if member.lstrip("./") == NVFP4_WEIGHTS_MEMBER:
                    return True
    except (OSError, tarfile.TarError):
        return False
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
    logging.info(f"{path} carries an NVFP4 payload; restoring with the NVFP4 connector.")
    return SortformerNVFP4SaveRestoreConnector()


class SortformerNVFP4SaveRestoreConnector(SaveRestoreConnector):
    """
    ``SaveRestoreConnector`` that writes and reads NVFP4 Sortformer weights as safetensors.

    Save and restore go through the ordinary ``ModelPT.save_to`` / ``restore_from`` paths, so the
    result is a normal ``.nemo`` archive and ``push_to_hf_hub`` works unchanged.

    Args:
        quantization_summary (Optional[Dict[str, Any]]): Summary returned by
            ``quantize_sortformer_model``, recorded for provenance.
        source_checkpoint_sha256 (Optional[str]): Digest of the BF16 checkpoint this was quantized
            from, recorded so an artifact can be traced to its origin.
        strict_torchao_version (bool): Fail rather than warn when the installed TorchAO differs from
            the one recorded at export.
    """

    def __init__(
        self,
        quantization_summary: Optional[Dict[str, Any]] = None,
        source_checkpoint_sha256: Optional[str] = None,
        strict_torchao_version: bool = False,
        export_precision: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Not "model_weights.ckpt": the extension advertises that this is not a torch.save pickle,
        # so a consumer that loads it with torch.load fails immediately instead of confusingly.
        self._model_weights_ckpt = "model_weights.safetensors"
        self._quantization_summary = quantization_summary
        self._source_checkpoint_sha256 = source_checkpoint_sha256
        self._strict_torchao_version = strict_torchao_version
        self._export_precision = export_precision
        self._restored_contexts: Dict[str, Dict[str, Any]] = {}
        self._restored_digests: Dict[str, str] = {}
        self.restored_export_precision: Optional[str] = None

    def _save_state_dict_to_disk(self, state_dict: Dict[str, Any], filepath: str) -> None:
        """
        Write the state dict as safetensors, decomposing every quantized weight into plain tensors.

        Args:
            state_dict (Dict[str, Any]): Model state dict, possibly holding TorchAO tensor subclasses.
            filepath (str): Destination path for the safetensors file.

        Raises:
            TypeError: If a quantized weight's flatten context cannot be represented as JSON.
        """
        from safetensors.torch import save_file

        tensors: Dict[str, torch.Tensor] = {}
        contexts: Dict[str, Dict[str, Any]] = {}
        digests: Dict[str, str] = {}

        for key, value in state_dict.items():
            if not _is_quantized(value):
                tensors[key] = value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
                continue

            names, context = value.__tensor_flatten__()
            attributes = {name: getattr(value, name) for name in names}
            for name, attribute in attributes.items():
                tensors[f"{key}{PAYLOAD_SEPARATOR}{name}"] = attribute.detach().cpu().contiguous()
            contexts[key] = _context_to_json(context)
            digests[key] = payload_digest(attributes)

        metadata = {
            META_FORMAT: CHECKPOINT_FORMAT,
            META_FORMAT_VERSION: str(CHECKPOINT_FORMAT_VERSION),
            META_TORCHAO_VERSION: str(_installed_torchao_version()),
            META_CONTEXTS: json.dumps(contexts, sort_keys=True, separators=(",", ":")),
            META_DIGESTS: json.dumps(digests, sort_keys=True, separators=(",", ":")),
        }
        if self._quantization_summary is not None:
            metadata[META_QUANTIZATION_SUMMARY] = json.dumps(
                self._quantization_summary, sort_keys=True, separators=(",", ":"), default=str
            )
        if self._source_checkpoint_sha256 is not None:
            metadata[META_SOURCE_CHECKPOINT_SHA256] = self._source_checkpoint_sha256
        if self._export_precision is not None:
            metadata[META_EXPORT_PRECISION] = str(self._export_precision)

        save_file(tensors, filepath, metadata=metadata)
        logging.info(
            f"Wrote {len(contexts)} quantized and {len(tensors) - sum(len(c) for c in contexts.values())} "
            f"plain entries to {filepath}"
        )

    def _load_state_dict_from_disk(self, model_weights: str, map_location: Optional[Any] = "cpu") -> Dict[str, Any]:
        """
        Read the safetensors payload, keeping quantized attributes separate for later reconstruction.

        The contexts and digests are stashed on the connector because ``restore_from`` reads the file
        before it installs weights, and reconstruction needs both.

        Args:
            model_weights (str): Path to the safetensors file inside the unpacked archive.
            map_location (Optional[Any]): Device for the loaded tensors.

        Returns:
            state_dict (Dict[str, Any]): Plain keys plus separator-suffixed quantized attributes.

        Raises:
            ValueError: If the file is not this format or records an unreadable version.
        """
        from safetensors import safe_open

        device = "cpu"
        if map_location is not None and str(map_location) != "cpu":
            device = str(map_location)

        state_dict: Dict[str, Any] = {}
        with safe_open(model_weights, framework="pt", device=device) as handle:
            metadata = handle.metadata() or {}
            recorded_format = metadata.get(META_FORMAT)
            if recorded_format != CHECKPOINT_FORMAT:
                raise ValueError(
                    f"{model_weights} records format {recorded_format!r}, not {CHECKPOINT_FORMAT!r}. This "
                    "connector will not guess at the layout of an unknown container."
                )
            recorded_version = int(metadata.get(META_FORMAT_VERSION, -1))
            if recorded_version != CHECKPOINT_FORMAT_VERSION:
                raise ValueError(
                    f"{model_weights} records format version {recorded_version}, but this build reads version "
                    f"{CHECKPOINT_FORMAT_VERSION}."
                )

            self._restored_contexts = json.loads(metadata.get(META_CONTEXTS, "{}"))
            self._restored_digests = json.loads(metadata.get(META_DIGESTS, "{}"))
            self.restored_export_precision = metadata.get(META_EXPORT_PRECISION)
            if self.restored_export_precision is not None:
                logging.info(
                    f"{model_weights} was quantized at precision={self.restored_export_precision}; serving at a "
                    "different precision executes different FP4 codes than were validated."
                )

            recorded_torchao = metadata.get(META_TORCHAO_VERSION)
            installed_torchao = str(_installed_torchao_version())
            if recorded_torchao and recorded_torchao != installed_torchao:
                message = TORCHAO_VERSION_MISMATCH_MESSAGE.format(
                    recorded=recorded_torchao, installed=installed_torchao
                )
                if self._strict_torchao_version:
                    raise ValueError(message)
                logging.warning(message)

            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
        return state_dict

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
            ValueError: If a payload digest does not match, if a target module is absent, or if the
                plain load leaves keys unaccounted for while ``strict`` is set.
        """
        grouped, plain = self._partition(state_dict)
        factories = _kwargs_factories()
        modules = dict(instance.named_modules())
        installed: List[str] = []

        for key, attributes in sorted(grouped.items()):
            encoded_context = self._restored_contexts.get(key)
            if encoded_context is None:
                raise ValueError(f"The checkpoint carries a payload for '{key}' but records no flatten context.")

            expected = self._restored_digests.get(key)
            if expected is not None:
                actual = payload_digest(attributes)
                if actual != expected:
                    raise ValueError(
                        f"Payload digest mismatch for '{key}': the file records {expected} but its bytes hash to "
                        f"{actual}. The payload was edited or truncated; refusing to execute it."
                    )

            module_fqn, _, attribute_name = key.rpartition(".")
            module = modules.get(module_fqn)
            if module is None:
                raise ValueError(f"The checkpoint quantizes '{module_fqn}', which this model does not contain.")

            context = _context_from_json(encoded_context, factories)
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

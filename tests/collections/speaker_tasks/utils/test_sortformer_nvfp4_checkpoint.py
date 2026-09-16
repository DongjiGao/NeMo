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
Tests for the self-contained NVFP4 checkpoint format.

Everything here runs on CPU: ``NVFP4Tensor.to_nvfp4`` only packs and scales, and both
``__tensor_flatten__`` and ``__tensor_unflatten__`` are metadata operations, so no FP4 kernel and no
Blackwell device is involved. The archive's whole purpose is to survive a round trip bit-exactly and to
refuse anything it cannot reconstruct faithfully, and both of those are checkable without a GPU.
"""

import json
import os
import tarfile

import pytest
import torch

from nemo.collections.asr.parts.utils.sortformer_nvfp4_checkpoint import (
    CHECKPOINT_FORMAT,
    CONFIG_SECTION,
    QUANTIZATION_CONFIG_MEMBER,
    SortformerNVFP4SaveRestoreConnector,
    _context_from_json,
    is_nvfp4_checkpoint,
    resolve_nvfp4_save_restore_connector,
)

torchao = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor", reason="requires TorchAO's NVFP4Tensor")
NVFP4Tensor = torchao.NVFP4Tensor

WEIGHTS_MEMBER = "model_weights.safetensors"


def _quantized(out_features=64, in_features=32, with_optional_scales=False):
    """An ``NVFP4Tensor`` like the export path produces, optionally carrying both global scales."""
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    if not with_optional_scales:
        return NVFP4Tensor.to_nvfp4(weight)
    return NVFP4Tensor.to_nvfp4(
        weight,
        per_tensor_scale=torch.tensor(2.0),
        act_per_tensor_scale=torch.tensor(3.0),
    )


def _state_dict(**quantized):
    """A state dict mixing quantized weights with the plain tensors that must pass through untouched."""
    state_dict = {
        "encoder.norm.weight": torch.randn(64, dtype=torch.bfloat16),
        "encoder.norm.bias": torch.randn(64, dtype=torch.bfloat16),
    }
    state_dict.update(quantized)
    return state_dict


def _write(tmp_path, state_dict, **connector_kwargs):
    """Run the save half and return the connector plus the weights path."""
    connector = SortformerNVFP4SaveRestoreConnector(**connector_kwargs)
    weights = str(tmp_path / WEIGHTS_MEMBER)
    connector._save_state_dict_to_disk(state_dict, weights)
    return connector, weights


def _read_config(tmp_path):
    with open(tmp_path / QUANTIZATION_CONFIG_MEMBER, encoding="utf-8") as handle:
        return json.load(handle)


def _write_config(tmp_path, config):
    with open(tmp_path / QUANTIZATION_CONFIG_MEMBER, "w", encoding="utf-8") as handle:
        json.dump(config, handle)


def _archive(tmp_path, name, members):
    """Build a ``.nemo``-shaped tar so detection can be exercised on realistic inputs."""
    path = str(tmp_path / name)
    with tarfile.open(path, "w") as tar:
        for member_name, payload in members.items():
            member_path = tmp_path / member_name
            member_path.write_text(payload, encoding="utf-8")
            tar.add(str(member_path), arcname=f"./{member_name}")
    return path


class TestRoundTrip:
    """A payload written by the save half must come back bit-identical through the load half."""

    @pytest.mark.unit
    @pytest.mark.parametrize("with_optional_scales", [False, True], ids=["mandatory-only", "with-global-scales"])
    def test_quantized_weights_survive_a_round_trip_bit_exactly(self, tmp_path, with_optional_scales):
        quantized = _quantized(with_optional_scales=with_optional_scales)
        connector, weights = _write(tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": quantized}))

        loaded = connector._load_state_dict_from_disk(weights)
        grouped, plain = connector._partition(loaded)
        rebuilt = connector._rebuild(
            grouped["encoder.attn.w_qkv.weight"], _context_from_json(connector._restored_context)
        )

        names, _ = quantized.__tensor_flatten__()
        for name in names:
            assert torch.equal(getattr(rebuilt, name), getattr(quantized, name)), name
        # The plain entries are the remainder, keyed exactly as they went in.
        assert set(plain) == {"encoder.norm.weight", "encoder.norm.bias"}

    @pytest.mark.unit
    def test_the_config_records_the_layers_and_the_attribute_set(self, tmp_path):
        quantized = _quantized(with_optional_scales=True)
        _write(tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": quantized}))

        section = _read_config(tmp_path)[CONFIG_SECTION]
        names, _ = quantized.__tensor_flatten__()
        assert section["format"] == CHECKPOINT_FORMAT
        # Recorded so the load half can tell a dropped optional scale from one that never existed.
        assert set(section["attributes"]) == set(names)
        assert list(_read_config(tmp_path)["quantization"]["quantized_layers"]) == ["encoder.attn.w_qkv"]

    @pytest.mark.unit
    def test_provenance_is_carried_through(self, tmp_path):
        _write(
            tmp_path,
            _state_dict(**{"encoder.attn.w_qkv.weight": _quantized()}),
            export_precision="bf16",
            source_checkpoint_sha256="abc123",
        )

        section = _read_config(tmp_path)[CONFIG_SECTION]
        assert section["export_precision"] == "bf16"
        assert section["source_checkpoint_sha256"] == "abc123"


class TestSaveRefusesWhatItCannotAddress:
    """The archive keys payloads by module, so anything that breaks that mapping must fail at write time."""

    @pytest.mark.unit
    def test_a_quantized_tensor_not_named_weight_is_rejected(self, tmp_path):
        # The load half rebuilds f"{module}.weight" and assigns to `weight`, so any other leaf name would be
        # written over the module's real weight and land on the wrong parameter.
        with pytest.raises(ValueError, match="can only represent a parameter named 'weight'"):
            _write(tmp_path, _state_dict(**{"encoder.attn.in_proj_weight": _quantized()}))

    @pytest.mark.unit
    def test_the_weight_requirement_covers_every_other_leaf_name(self, tmp_path):
        # Requiring 'weight' is also what makes the module key unique, since the stored key is then exactly
        # f"{module}.weight" and state-dict keys cannot repeat. A second quantized tensor in the same module
        # must therefore have a different leaf name, and is caught by the same check.
        with pytest.raises(ValueError, match="quantized tensor named 'bias'"):
            _write(
                tmp_path,
                {
                    "encoder.attn.weight": _quantized(),
                    "encoder.attn.bias": _quantized(out_features=32, in_features=32),
                },
            )

    @pytest.mark.unit
    def test_an_unquantized_model_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="No quantized weights found"):
            _write(tmp_path, _state_dict())


class TestLoadRefusesAnIncompletePayload:
    """A payload missing part of itself must raise, not restore a silently different weight."""

    @pytest.mark.unit
    def test_a_dropped_optional_scale_is_rejected(self, tmp_path):
        # TorchAO restores an absent optional name as None, and get_hp_scales() then omits the global scale
        # entirely, so the dequantized weight is wrong by the amax ratio with nothing raised.
        from safetensors.torch import load_file, save_file

        connector, weights = _write(
            tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": _quantized(with_optional_scales=True)})
        )
        tensors = load_file(weights)
        del tensors["encoder.attn.w_qkv.weight_scale_2"]
        save_file(tensors, weights)

        loaded = connector._load_state_dict_from_disk(weights)
        with pytest.raises(ValueError, match="payload is missing"):
            connector._partition(loaded)

    @pytest.mark.unit
    def test_a_context_from_another_torchao_is_rejected(self, tmp_path):
        # __tensor_unflatten__ indexes the context by the installed class's names and ignores the rest, so an
        # unrecognized key would be dropped and its class default used instead -- silent layout drift.
        connector, weights = _write(tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": _quantized()}))
        config = _read_config(tmp_path)
        config[CONFIG_SECTION]["context"]["a_future_attribute"] = 1
        _write_config(tmp_path, config)

        loaded = connector._load_state_dict_from_disk(weights)
        grouped, _ = connector._partition(loaded)
        with pytest.raises(ValueError, match="does not match the installed TorchAO"):
            connector._rebuild(grouped["encoder.attn.w_qkv.weight"], _context_from_json(connector._restored_context))

    @pytest.mark.unit
    def test_a_config_from_another_format_is_rejected(self, tmp_path):
        connector, weights = _write(tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": _quantized()}))
        config = _read_config(tmp_path)
        del config[CONFIG_SECTION]
        _write_config(tmp_path, config)

        with pytest.raises(ValueError, match=f"has no '{CONFIG_SECTION}' section"):
            connector._load_state_dict_from_disk(weights)

    @pytest.mark.unit
    def test_a_missing_config_is_rejected(self, tmp_path):
        connector, weights = _write(tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": _quantized()}))
        os.remove(tmp_path / QUANTIZATION_CONFIG_MEMBER)

        with pytest.raises(ValueError, match=f"has no {QUANTIZATION_CONFIG_MEMBER}"):
            connector._load_state_dict_from_disk(weights)


class TestDetection:
    """Detection runs on every .nemo path, so it must answer False rather than raise on anything odd."""

    @pytest.mark.unit
    def test_an_nvfp4_archive_is_detected(self, tmp_path):
        _write(tmp_path, _state_dict(**{"encoder.attn.w_qkv.weight": _quantized()}))
        config = json.dumps(_read_config(tmp_path))
        path = _archive(tmp_path, "nvfp4.nemo", {QUANTIZATION_CONFIG_MEMBER: config})

        assert is_nvfp4_checkpoint(path) is True
        assert isinstance(resolve_nvfp4_save_restore_connector(path), SortformerNVFP4SaveRestoreConnector)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "members, reason",
        [
            ({"model_config.yaml": "name: sortformer\n"}, "plain BF16 archive, no config member"),
            ({QUANTIZATION_CONFIG_MEMBER: '{"producer": {"name": "modelopt"}}'}, "genuine ModelOpt archive"),
            ({QUANTIZATION_CONFIG_MEMBER: '["not", "an", "object"]'}, "valid JSON that is not an object"),
            ({QUANTIZATION_CONFIG_MEMBER: "not json at all"}, "unparseable config"),
        ],
        ids=["bf16", "modelopt", "json-array", "malformed-json"],
    )
    def test_other_archives_are_not_detected(self, tmp_path, members, reason):
        assert is_nvfp4_checkpoint(_archive(tmp_path, "other.nemo", members)) is False, reason
        assert resolve_nvfp4_save_restore_connector(_archive(tmp_path, "other.nemo", members)) is None, reason

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ["missing.nemo", "not_a_tar.nemo", ""], ids=["absent", "not-tar", "empty-path"])
    def test_unreadable_paths_are_not_detected(self, tmp_path, name):
        path = str(tmp_path / name) if name else ""
        if name == "not_a_tar.nemo":
            (tmp_path / name).write_bytes(b"\x00\x01\x02 definitely not a tar")

        assert is_nvfp4_checkpoint(path) is False

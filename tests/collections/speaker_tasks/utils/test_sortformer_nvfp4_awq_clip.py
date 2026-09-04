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

"""Tests for the offline builder of the Sortformer NVFP4 AWQ-clip ratio codes."""

import base64
import hashlib
import json
from pathlib import Path

import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_nvfp4_weight_mse as weight_mse
from nemo.collections.asr.parts.utils import sortformer_quantization as sq

D_MODEL = 16
FF_HIDDEN = 32
NUM_LAYERS = 2

CHECKPOINT_SHA256 = "c" * 64

# The builder tests run on the host, so they build for the ordinary conversion that runs there. The unclipped code
# keeps that conversion's own bytes, so the construction is part of what every artifact here is bound to.
TEMPLATE_ARITHMETIC = sq.WEIGHT_SCALE_AWQ_CLIP_TEMPLATE_ARITHMETIC_REFERENCE


@pytest.fixture(scope="module")
def builder():
    """Load the AWQ-clip builder, which lives outside any importable package."""
    import importlib.util  # a plain ``import importlib`` does not load this submodule

    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "dataset_processing"
        / "speaker_tasks"
        / "build_sortformer_nvfp4_awq_clip.py"
    )
    if not script.exists():
        pytest.skip("the AWQ-clip ratio-code builder is not available in this checkout")
    spec = importlib.util.spec_from_file_location("build_sortformer_nvfp4_awq_clip", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def awq_backend():
    """Skip cleanly unless the pinned torchao NVFP4 kernels and the E4M3 dtype are usable on this machine."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")
    pytest.importorskip("torchao.prototype.mx_formats.kernels", reason="the AWQ-clip builder requires torchao 0.17")
    pytest.importorskip("torchao.prototype.mx_formats.utils", reason="the AWQ-clip builder requires torchao 0.17")
    pytest.importorskip(
        "torchao.prototype.mx_formats.nvfp4_tensor", reason="the AWQ-clip builder requires torchao 0.17"
    )
    return True


class _FakeFeedForward(torch.nn.Module):
    """Feed-forward block mirroring the Sortformer transformer encoder's ``net.0`` / ``net.3`` layout."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D_MODEL, FF_HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(FF_HIDDEN, D_MODEL),
            torch.nn.Dropout(0.0),
        )


class _FakeAttention(torch.nn.Module):
    """Attention block exposing the ``w_qkv`` / ``out_proj`` target names."""

    def __init__(self):
        super().__init__()
        self.w_qkv = torch.nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out_proj = torch.nn.Linear(D_MODEL, D_MODEL)


class _FakeLayer(torch.nn.Module):
    """Single transformer layer with the norms that must never be selected."""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(D_MODEL)
        self.attn = _FakeAttention()
        self.norm2 = torch.nn.LayerNorm(D_MODEL)
        self.ffn = _FakeFeedForward()


class _FakeEncoder(torch.nn.Module):
    """Stack of transformer layers."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(NUM_LAYERS)])


class _FakeSortformer(torch.nn.Module):
    """Sortformer stand-in carrying exactly the quantization targets the builder selects codes for."""

    def __init__(self):
        super().__init__()
        self.pre_encode = torch.nn.Linear(D_MODEL, D_MODEL)
        self.transformer_encoder = _FakeEncoder()
        self.head = torch.nn.Linear(D_MODEL, 4)


@pytest.mark.unit
def test_sample_loader_reports_the_file_identity_and_the_retained_rows(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    path = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, seed=1, rows=3))

    entry = builder.load_activation_sample_file("  near_field  ", str(path))

    assert entry["label"] == "near_field"
    assert entry["name"] == "near.pt"
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["size_bytes"] == len(path.read_bytes())
    assert entry["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert entry["fqns"] == tuple(sorted(weights))
    assert all(int(tensor.shape[0]) == 3 for tensor in entry["samples"].values())
    assert all(tensor.dtype is torch.bfloat16 for tensor in entry["samples"].values())


@pytest.mark.unit
def test_sample_loader_reads_without_executing_code(builder, tmp_path, monkeypatch):
    """A sample file is inspected with ``weights_only=True``, so it can never execute code while being read."""
    weights = _target_weights(_FakeSortformer())
    path = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, seed=2))
    seen = {}
    original = torch.load

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording)
    builder.load_activation_sample_file("near_field", str(path))

    assert seen["weights_only"] is True
    assert seen["map_location"] == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update(schema="other"), "declares schema"),
        (lambda payload: payload.update(version=2), "version 1 is required"),
        (lambda payload: payload.update(version=True), "version 1 is required"),
        (lambda payload: payload.update(checkpoint_sha256="nope"), "hexadecimal SHA-256"),
        (lambda payload: payload.update(seed=-1), "non-negative integer"),
        (lambda payload: payload.update(max_rows=0), "must be positive"),
        (lambda payload: payload.update(targets="attn.w_qkv"), "list of strings"),
        (lambda payload: payload.update(metadata=[]), "'metadata' as an object"),
        (lambda payload: payload.pop("samples"), "exactly the keys"),
        (lambda payload: payload.update(extra=1), "exactly the keys"),
        (lambda payload: payload.update(total_finite_rows={}), "exactly the sampled modules"),
        (lambda payload: payload.update(nonfinite_rows={}), "exactly the sampled modules"),
    ],
)
def test_sample_loader_rejection_matrix(builder, tmp_path, mutate, message):
    weights = _target_weights(_FakeSortformer())
    payload = _sample_payload(builder, _sample_rows(weights, seed=3))
    mutate(payload)
    path = tmp_path / "bad.pt"
    torch.save(payload, str(path))

    with pytest.raises(ValueError, match=message):
        builder.load_activation_sample_file("near_field", str(path))


@pytest.mark.unit
def test_sample_loader_rejects_unusable_rows(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    fqn = sorted(weights)[0]

    for rows, message in (
        ({fqn: torch.zeros(2, D_MODEL, dtype=torch.float32)}, "retains"),
        ({fqn: torch.zeros(0, D_MODEL, dtype=torch.bfloat16)}, "non-empty rank-2"),
        ({fqn: torch.full((2, D_MODEL), float("nan"), dtype=torch.bfloat16)}, "non-finite"),
    ):
        samples = _sample_rows(weights, seed=4)
        samples[fqn] = rows[fqn]
        path = tmp_path / f"{message[:6]}.pt"
        torch.save(_sample_payload(builder, samples), str(path))
        with pytest.raises(ValueError, match=message):
            builder.load_activation_sample_file("near_field", str(path))

    with pytest.raises(ValueError, match="empty source label"):
        builder.load_activation_sample_file("   ", str(tmp_path / "missing.pt"))


@pytest.mark.unit
def test_calibration_must_describe_this_checkpoint_and_exactly_these_modules(builder, tmp_path):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)

    good = _write_calibration(tmp_path, fqns)
    calibration = builder.load_activation_calibration(str(good), CHECKPOINT_SHA256, fqns)
    assert calibration["identity"]["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest()
    assert calibration["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert calibration["identity"]["scale_mode"] == "static"
    assert calibration["identity"]["runtime_scale_margin"] == 1.0
    # The margin this build requires is exactly 1.0, so the consumed values are the recorded ones.
    assert calibration["activation_amax"] == {fqn: pytest.approx(_amax_for(fqn)) for fqn in fqns}

    for overrides, message in (
        ({"checkpoint": "d" * 64}, "was collected on checkpoint"),
        ({"scale_mode": "dynamic"}, "declares scale_mode"),
        ({"runtime_scale_margin": 1.375}, "presumes a runtime scale margin"),
        ({"targets": ["attn.w_qkv"]}, "declares targets"),
        ({"drop": fqns[0]}, "must cover exactly"),
        ({"extra": "transformer_encoder.layers.0.attn.other"}, "must cover exactly"),
        ({"value": -1.0}, "finite and positive"),
        ({"version": 2}, "version 1 is required"),
    ):
        path = _write_calibration(tmp_path, fqns, name=f"calib_{len(message)}_{message[:4]}.json", **overrides)
        with pytest.raises(ValueError, match=message):
            builder.load_activation_calibration(str(path), CHECKPOINT_SHA256, fqns)


@pytest.mark.unit
def test_calibration_checkpoint_is_read_from_the_production_metadata_shape(builder, tmp_path):
    """The frozen production calibration carries its checkpoint in ``metadata`` and no top-level claim at all.

    ``merge_calibrations`` writes exactly the six top-level keys asserted here, so a builder that only ever read a
    top-level claim could not bind a single production artifact to its checkpoint. The older single-collector
    spelling still resolves, two agreeing claims resolve to the digest they agree on, and neither a conflict nor an
    absent claim is read optimistically.
    """
    fqns = sorted(_target_weights(_FakeSortformer()))

    production = _write_calibration(tmp_path, fqns, name="production.json")
    payload = json.loads(production.read_text(encoding="utf-8"))
    assert sorted(payload) == ["activation_amax", "metadata", "recipe", "scale_mode", "targets", "version"]
    assert payload["metadata"]["checkpoint_sha256"] == CHECKPOINT_SHA256

    for name, overrides in (
        ("production.json", {}),
        ("legacy.json", {"checkpoint_location": "top_level"}),
        ("agreeing.json", {"checkpoint_location": "both"}),
    ):
        path = _write_calibration(tmp_path, fqns, name=name, **overrides)
        loaded = builder.load_activation_calibration(str(path), CHECKPOINT_SHA256, fqns)
        assert loaded["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256

    conflicting = _write_calibration(
        tmp_path, fqns, name="conflict.json", checkpoint_location="both", top_level_checkpoint="d" * 64
    )
    with pytest.raises(ValueError, match="conflicting claims"):
        builder.load_activation_calibration(str(conflicting), CHECKPOINT_SHA256, fqns)

    unbound = _write_calibration(tmp_path, fqns, name="unbound.json", checkpoint_location="none")
    with pytest.raises(ValueError, match="declares no checkpoint digest"):
        builder.load_activation_calibration(str(unbound), CHECKPOINT_SHA256, fqns)


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"headroom_baked_in": False}, "headroom_baked_in"),
        ({"headroom_baked_in": None}, "headroom_baked_in"),
        ({"headroom_baked_in": 1}, "headroom_baked_in"),
        ({"headroom": None}, "'metadata.headroom'"),
        ({"headroom": 0.0}, "must be positive"),
        ({"headroom": -1.375}, "must be positive"),
        ({"runtime_scale_margin": None}, "runtime_scale_margin"),
        ({"runtime_scale_margin": 1.375}, "presumes a runtime scale margin"),
    ],
)
def test_builder_requires_a_calibration_whose_headroom_is_already_baked_in(builder, tmp_path, metadata, message):
    """Codes are selected at margin 1.0, so a calibration that does not bake its headroom in is refused."""
    fqns = sorted(_target_weights(_FakeSortformer()))
    path = _write_calibration(tmp_path, fqns, name="headroom.json", metadata=metadata)

    with pytest.raises(ValueError, match=message):
        builder.load_activation_calibration(str(path), CHECKPOINT_SHA256, fqns)


@pytest.mark.unit
def test_artifact_binds_the_checkpoint_the_weights_the_calibration_and_the_labelled_sources(
    builder, awq_backend, tmp_path
):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    entries = [
        _entry(builder, tmp_path, "near_field", weights, seed=5, rows=6),
        _entry(builder, tmp_path, "far_field", weights, seed=6, rows=2),
    ]
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)

    payload = builder.build_awq_clip_artifact(
        weights, entries, calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )

    assert payload["schema"] == sq.AWQ_CLIP_SCHEMA
    assert payload["version"] == sq.AWQ_CLIP_SCHEMA_VERSION
    assert payload["algorithm"] == sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM
    assert payload["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert set(payload) == set(sq.AWQ_CLIP_ARTIFACT_KEYS)
    assert set(payload["arithmetic"]) == set(sq.AWQ_CLIP_ARITHMETIC_KEYS)
    assert payload["arithmetic"]["clip_ratios"] == list(sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS)
    assert payload["arithmetic"]["block_size"] == 16
    assert payload["arithmetic"]["modelopt_reference_version"] == "0.46.0"
    assert payload["arithmetic"]["modelopt_reference_wheel_sha256"] == sq.MODELOPT_REFERENCE_WHEEL_SHA256
    assert payload["arithmetic"]["template_arithmetic"] == TEMPLATE_ARITHMETIC
    assert payload["activation_calibration"]["sha256"] == calibration["identity"]["sha256"]
    assert payload["activation_calibration"]["scale_margin"] == 1.0
    assert payload["weight_sha256"] == {fqn: sq.nvfp4_weight_digest(weights[fqn]) for fqn in fqns}
    assert sorted(payload["ratio_codes"]) == fqns
    assert payload["ratio_code_sha256"] == sq.nvfp4_section_digest(payload["ratio_codes"])
    assert payload["provenance_sha256"] == sq.nvfp4_section_digest(payload["provenance"])

    for fqn in fqns:
        entry = payload["ratio_codes"][fqn]
        assert set(entry) == set(sq.AWQ_CLIP_CODE_KEYS)
        rows, blocks = int(weights[fqn].shape[0]), int(weights[fqn].shape[1]) // 16
        assert entry["shape"] == [rows, blocks]
        decoded = base64.b64decode(entry["codes"], validate=True)
        assert len(decoded) == rows * blocks
        assert max(decoded) <= 10
        module = payload["provenance"]["modules"][fqn]
        assert module["block_count"] == rows * blocks
        assert sum(module["ratio_histogram"]) == rows * blocks

    provenance = payload["provenance"]
    assert provenance["method"] == sq.AWQ_CLIP_CONSTRUCTION_METHOD
    assert provenance["objective"] == sq.AWQ_CLIP_OBJECTIVE
    assert provenance["target_fqns"] == fqns
    assert [source["label"] for source in provenance["sources"]] == ["far_field", "near_field"]
    assert provenance["aggregate"]["source_labels"] == ["far_field", "near_field"]
    assert provenance["aggregate"]["module_count"] == len(fqns)
    assert provenance["aggregate"]["block_count"] == sum(
        payload["provenance"]["modules"][fqn]["block_count"] for fqn in fqns
    )
    assert provenance["aggregate"]["selected_objective"] == sq.nvfp4_awq_clip_weighted_objective(
        provenance["modules"], fqns, "selected_objective"
    )


@pytest.mark.unit
def test_artifact_carries_no_activation_row_and_no_weight(builder, awq_backend, tmp_path):
    """The runtime artifact is codes and provenance: no rows, no weights, no labels beyond the group names.

    This is a *structural* contract rather than a search for particular values, because a scalar's repr collides
    with legitimate evidence -- ``0.5`` is a clip ratio as well as a plausible weight. Instead every key set is
    pinned to its closed constant, every value must be a JSON scalar, no list may carry a float or more numbers
    than there are clip ratios (an activation row and a weight row are both exactly that), and the only bulk
    payload in the file must decode to exactly the ratio codes its own shape declares.
    """
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    entry = _entry(builder, tmp_path, "near_field", weights, seed=7)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)

    payload = builder.build_awq_clip_artifact(
        weights, [entry], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )

    # Closed key sets: an artifact carrying an unknown section could carry anything at all in it.
    assert set(payload) == set(sq.AWQ_CLIP_ARTIFACT_KEYS)
    assert set(payload["arithmetic"]) == set(sq.AWQ_CLIP_ARITHMETIC_KEYS)
    assert set(payload["activation_calibration"]) == set(sq.AWQ_CLIP_CALIBRATION_KEYS)
    assert set(payload["provenance"]) == set(sq.AWQ_CLIP_PROVENANCE_KEYS)
    assert set(payload["provenance"]["aggregate"]) == set(sq.AWQ_CLIP_AGGREGATE_KEYS)
    for source in payload["provenance"]["sources"]:
        assert set(source) == set(sq.AWQ_CLIP_SOURCE_KEYS)
    for fqn in fqns:
        assert set(payload["ratio_codes"][fqn]) == set(sq.AWQ_CLIP_CODE_KEYS)
        assert set(payload["provenance"]["modules"][fqn]) == set(sq.AWQ_CLIP_MODULE_KEYS)
    assert sorted(payload["weight_sha256"]) == fqns
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in payload["weight_sha256"].values())

    # The one float list the artifact may carry is the fixed clip-ratio list, which is identity, not data.
    assert payload["arithmetic"]["clip_ratios"] == list(sq.WEIGHT_SCALE_AWQ_CLIP_RATIOS)
    assert _artifact_structure_violations(payload) == []

    # No forbidden semantic key, at any depth, under any spelling of a training or task artifact.
    assert _artifact_keys_matching(payload, _FORBIDDEN_ARTIFACT_KEYS) == []

    # The only bulk payload is base64 ratio codes, and it decodes to exactly the codes its shape declares.
    for fqn in fqns:
        rows, blocks = payload["ratio_codes"][fqn]["shape"]
        decoded = base64.b64decode(payload["ratio_codes"][fqn]["codes"], validate=True)
        assert len(decoded) == rows * blocks
        assert max(decoded) < sq.WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT

    # And it round-trips as strict JSON, which is what the runtime will read it back as.
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.unit
def test_groups_are_weighted_equally_regardless_of_how_many_rows_they_kept(builder, awq_backend, tmp_path):
    """Duplicating a group's rows must not change its influence: the reduction is a mean over groups."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    small = _sample_rows(weights, seed=8, rows=3)
    doubled = {fqn: torch.cat([rows, rows], dim=0) for fqn, rows in small.items()}
    other = _sample_rows(weights, seed=9, rows=4)

    reference = builder.build_awq_clip_artifact(
        weights,
        [_load(builder, tmp_path, "a", small, "a.pt"), _load(builder, tmp_path, "b", other, "b.pt")],
        calibration,
        checkpoint_sha256=CHECKPOINT_SHA256,
        template_arithmetic=TEMPLATE_ARITHMETIC,
    )
    balanced = builder.build_awq_clip_artifact(
        weights,
        [_load(builder, tmp_path, "a", doubled, "a2.pt"), _load(builder, tmp_path, "b", other, "b2.pt")],
        calibration,
        checkpoint_sha256=CHECKPOINT_SHA256,
        template_arithmetic=TEMPLATE_ARITHMETIC,
    )

    assert balanced["ratio_codes"] == reference["ratio_codes"]
    assert balanced["ratio_code_sha256"] == reference["ratio_code_sha256"]


@pytest.mark.unit
def test_codes_do_not_depend_on_input_order_or_on_the_chunk_sizes(builder, awq_backend, tmp_path):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    near = _entry(builder, tmp_path, "near_field", weights, seed=10, rows=5)
    far = _entry(builder, tmp_path, "far_field", weights, seed=11, rows=3)

    forward = builder.build_awq_clip_artifact(
        weights, [near, far], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )
    reverse = builder.build_awq_clip_artifact(
        weights, [far, near], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )
    chunked = builder.build_awq_clip_artifact(
        weights,
        [near, far],
        calibration,
        checkpoint_sha256=CHECKPOINT_SHA256,
        template_arithmetic=TEMPLATE_ARITHMETIC,
        row_chunk_size=1,
        block_chunk_size=1,
    )

    assert reverse == forward
    assert chunked["ratio_codes"] == forward["ratio_codes"]
    assert chunked["provenance"]["modules"] == forward["provenance"]["modules"]


@pytest.mark.unit
def test_artifact_rejects_samples_or_calibration_describing_another_run(builder, awq_backend, tmp_path):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)

    other_checkpoint = _entry(
        builder, tmp_path, "near_field", weights, seed=12, name="other.pt", checkpoint_sha256="d" * 64
    )
    with pytest.raises(ValueError, match="was collected on checkpoint"):
        builder.build_awq_clip_artifact(
            weights,
            [other_checkpoint],
            calibration,
            checkpoint_sha256=CHECKPOINT_SHA256,
            template_arithmetic=TEMPLATE_ARITHMETIC,
        )

    other_targets = _entry(
        builder, tmp_path, "near_field", weights, seed=13, name="targets.pt", targets=["attn.w_qkv"]
    )
    with pytest.raises(ValueError, match="declares targets"):
        builder.build_awq_clip_artifact(
            weights,
            [other_targets],
            calibration,
            checkpoint_sha256=CHECKPOINT_SHA256,
            template_arithmetic=TEMPLATE_ARITHMETIC,
        )

    narrow = _sample_rows(weights, seed=14)
    narrow.pop(fqns[0])
    partial = _load(builder, tmp_path, "near_field", narrow, "partial.pt")
    with pytest.raises(ValueError, match="does not cover"):
        builder.build_awq_clip_artifact(
            weights,
            [partial],
            calibration,
            checkpoint_sha256=CHECKPOINT_SHA256,
            template_arithmetic=TEMPLATE_ARITHMETIC,
        )

    wide = _sample_rows(weights, seed=15)
    wide[fqns[0]] = torch.zeros(2, int(weights[fqns[0]].shape[1]) + 16, dtype=torch.bfloat16)
    mismatched = _load(builder, tmp_path, "near_field", wide, "wide.pt")
    with pytest.raises(ValueError, match="input channel"):
        builder.build_awq_clip_artifact(
            weights,
            [mismatched],
            calibration,
            checkpoint_sha256=CHECKPOINT_SHA256,
            template_arithmetic=TEMPLATE_ARITHMETIC,
        )

    entry = _entry(builder, tmp_path, "near_field", weights, seed=16, name="ok.pt")
    with pytest.raises(ValueError, match="At least one labelled"):
        builder.build_awq_clip_artifact(
            weights, [], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
        )
    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        builder.build_awq_clip_artifact(
            weights, [entry], calibration, checkpoint_sha256="nope", template_arithmetic=TEMPLATE_ARITHMETIC
        )
    with pytest.raises(ValueError, match="template_arithmetic must be one of"):
        builder.build_awq_clip_artifact(
            weights, [entry], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic="guessed"
        )


@pytest.mark.unit
def test_write_is_deterministic_atomic_and_refuses_to_overwrite(builder, awq_backend, tmp_path):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    entry = _entry(builder, tmp_path, "near_field", weights, seed=17)
    payload = builder.build_awq_clip_artifact(
        weights, [entry], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )
    destination = tmp_path / "out" / "awq.json"

    written = builder.write_awq_clip_artifact(payload, str(destination))
    first = Path(written).read_bytes()
    assert sorted(item.name for item in destination.parent.iterdir()) == ["awq.json"]

    with pytest.raises(FileExistsError, match="already exists"):
        builder.write_awq_clip_artifact(payload, str(destination))
    assert Path(written).read_bytes() == first

    builder.write_awq_clip_artifact(payload, str(destination), overwrite=True)
    assert Path(written).read_bytes() == first


@pytest.mark.unit
def test_a_failed_write_leaves_no_temporary_file_beside_the_destination(builder, awq_backend, tmp_path):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    entry = _entry(builder, tmp_path, "near_field", weights, seed=18)
    payload = builder.build_awq_clip_artifact(
        weights, [entry], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )
    destination = tmp_path / "out" / "awq.json"

    broken = json.loads(json.dumps(payload))
    broken["provenance"]["aggregate"]["selected_objective"] = float("nan")
    with pytest.raises(ValueError):
        builder.write_awq_clip_artifact(broken, str(destination))
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []

    with pytest.raises(TypeError):
        builder.write_awq_clip_artifact({"schema": {"a", "b"}}, str(destination))
    assert list(destination.parent.iterdir()) == []

    assert builder.write_awq_clip_artifact(payload, str(destination)) == str(destination)
    assert sorted(item.name for item in destination.parent.iterdir()) == ["awq.json"]


@pytest.mark.unit
def test_written_artifact_is_consumable_by_the_runtime_loader(builder, awq_backend, tmp_path):
    """The builder's output and the runtime's strict loader are two halves of one contract."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    calibration_path = _write_calibration(tmp_path, fqns)
    calibration = builder.load_activation_calibration(str(calibration_path), CHECKPOINT_SHA256, fqns)
    entries = [
        _entry(builder, tmp_path, "near_field", weights, seed=19, rows=5),
        _entry(builder, tmp_path, "far_field", weights, seed=20, rows=2),
    ]
    payload = builder.build_awq_clip_artifact(
        weights, entries, calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )
    output = builder.write_awq_clip_artifact(payload, str(tmp_path / "awq.json"))

    loaded = sq.load_awq_clip_artifact(
        output, model, sq.select_quantization_targets(model, "nvfp4_all"), str(calibration_path)
    )

    assert loaded["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert loaded["fqns"] == fqns
    assert loaded["sha256"] == hashlib.sha256(Path(output).read_bytes()).hexdigest()
    assert loaded["ratio_code_sha256"] == payload["ratio_code_sha256"]
    assert loaded["provenance_sha256"] == payload["provenance_sha256"]
    assert loaded["calibration"]["sha256"] == calibration["identity"]["sha256"]
    for fqn in fqns:
        expected = base64.b64decode(payload["ratio_codes"][fqn]["codes"], validate=True)
        assert loaded["ratio_codes"][fqn] == expected
        assert loaded["code_shapes"][fqn] == payload["ratio_codes"][fqn]["shape"]


@pytest.mark.unit
def test_selection_matches_the_packers_own_entry_point(builder, awq_backend, tmp_path):
    """The builder is orchestration only: the codes it writes are exactly the packer's own selection."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    fqn = fqns[0]
    rows = _sample_rows(weights, seed=21, rows=4)[fqn]
    amax = _amax_for(fqn)

    selection = builder.select_module_ratio_codes(weights[fqn], [rows], amax, TEMPLATE_ARITHMETIC)
    quantized = weight_mse.nvfp4_awq_clip_activation_qdq(rows, amax)
    global_scale = weight_mse.nvfp4_weight_global_scale(weights[fqn])
    # The unclipped candidate is the ordinary template's own readback, not the ratio-1.00 formula, because that is
    # what the runtime repack leaves in place for a block whose code is the unclipped one.
    template = weight_mse.nvfp4_awq_clip_template_reconstruction(weights[fqn], global_scale, TEMPLATE_ARITHMETIC)
    expected = weight_mse.select_nvfp4_ratio_codes_awq_clip(weights[fqn], global_scale, [quantized], template)

    assert torch.equal(selection.ratio_codes, expected.ratio_codes)
    assert selection.selected_objective == expected.selected_objective
    assert selection.unclipped_objective == expected.unclipped_objective
    # Base64 is a lossless view of exactly those bytes.
    encoded = builder.encode_ratio_codes(selection.ratio_codes)
    decoded = torch.frombuffer(bytearray(base64.b64decode(encoded, validate=True)), dtype=torch.uint8)
    assert torch.equal(decoded.reshape(selection.ratio_codes.shape), selection.ratio_codes)


@pytest.mark.unit
def test_device_resolution_never_downgrades_a_requested_gpu(builder):
    assert builder.resolve_device("cpu") == torch.device("cpu")
    assert builder.DEFAULT_DEVICE == "cuda"

    with pytest.raises(ValueError, match="non-empty PyTorch device string"):
        builder.resolve_device("  ")
    with pytest.raises(ValueError, match="only runs on"):
        builder.resolve_device("meta")
    with pytest.raises(ValueError, match="not a valid PyTorch device string"):
        builder.resolve_device("not-a-device")

    if torch.cuda.is_available():
        assert builder.resolve_device("cuda") == torch.device("cuda", torch.cuda.current_device())
        with pytest.raises(ValueError, match="can see only"):
            builder.resolve_device(f"cuda:{torch.cuda.device_count()}")
    else:
        with pytest.raises(ValueError, match="no CUDA device is available"):
            builder.resolve_device("cuda")


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["samples.pt", "=samples.pt", "near_field=", "  =  "])
def test_input_parsing_requires_a_label_and_a_path(builder, raw):
    with pytest.raises(ValueError, match="LABEL=PATH"):
        builder._parse_inputs([raw])


@pytest.mark.unit
def test_inputs_must_be_unique_by_label_and_by_file(builder, tmp_path):
    with pytest.raises(ValueError, match="At least one --input"):
        builder._require_unique_inputs([])
    with pytest.raises(ValueError, match="labels must be unique"):
        builder._require_unique_inputs([("near", "a.pt"), ("near", "b.pt")])
    with pytest.raises(ValueError, match="files must be unique"):
        builder._require_unique_inputs([("near", "a.pt"), ("far", "a.pt")])


@pytest.mark.unit
def test_cli_writes_the_artifact_and_prints_codes_only(builder, awq_backend, tmp_path, monkeypatch, capsys):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    near = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, seed=22, rows=6))
    far = _write_sample(builder, tmp_path / "far.pt", _sample_rows(weights, seed=23, rows=2))
    calibration = _write_calibration(tmp_path, fqns)
    output = tmp_path / "awq.json"
    # The restore itself needs a real .nemo checkpoint; everything around it is exercised here.
    monkeypatch.setattr(builder, "restore_target_weights", lambda path, digest, device: weights)

    exit_code = builder.main(_argv(tmp_path, calibration, output, [f"near_field={near}", f"far_field={far}"]))

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert payload["provenance"]["aggregate"]["source_labels"] == ["far_field", "near_field"]

    printed = capsys.readouterr().out
    assert str(output) in printed
    assert "2 equally weighted source group(s)" in printed
    assert sq.WEIGHT_SCALE_AWQ_CLIP_ALGORITHM in printed
    assert "says nothing about DER" in printed


@pytest.mark.unit
def test_cli_fails_before_restoring_on_a_repeated_group(builder, tmp_path, monkeypatch):
    weights = _target_weights(_FakeSortformer())
    near = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, seed=24))
    calibration = _write_calibration(tmp_path, sorted(weights))

    def refuse(*args, **kwargs):
        raise AssertionError("the checkpoint must not be restored for an invalid input set")

    monkeypatch.setattr(builder, "restore_target_weights", refuse)

    output = tmp_path / "awq.json"
    with pytest.raises(SystemExit, match="labels must be unique"):
        builder.main(_argv(tmp_path, calibration, output, [f"near_field={near}", f"near_field={near}"]))
    assert not output.exists()


@pytest.mark.unit
def test_cli_refuses_an_existing_output_without_overwrite(builder, awq_backend, tmp_path, monkeypatch):
    model = _FakeSortformer()
    weights = _target_weights(model)
    near = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, seed=25))
    calibration = _write_calibration(tmp_path, sorted(weights))
    output = tmp_path / "awq.json"
    output.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(builder, "restore_target_weights", lambda path, digest, device: weights)

    argv = _argv(tmp_path, calibration, output, [f"near_field={near}"])
    with pytest.raises(SystemExit, match="already exists"):
        builder.main(argv)
    assert output.read_text(encoding="utf-8") == "keep me"

    assert builder.main(argv + ["--overwrite"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == sq.AWQ_CLIP_SCHEMA


@pytest.mark.unit
def test_restore_verifies_the_checkpoint_digest_before_reading_it(builder, tmp_path, monkeypatch):
    """A checkpoint that does not hash to the asserted digest is never restored at all."""
    checkpoint = tmp_path / "model.nemo"
    checkpoint.write_bytes(b"not the checkpoint the codes claim")

    def refuse(*args, **kwargs):
        raise AssertionError("restore_from must not run for a mismatched checkpoint")

    monkeypatch.setattr(builder.SortformerEncLabelModel, "restore_from", staticmethod(refuse))

    with pytest.raises(ValueError, match="hashes to"):
        builder.restore_target_weights(str(checkpoint), CHECKPOINT_SHA256, torch.device("cpu"))

    assert builder.file_sha256(checkpoint) == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


# Keys a runtime artifact may never carry, at any depth, whatever it holds under them. ``weight_sha256`` and
# ``weight_digest_method`` are legitimate, so these are matched as whole keys and never as substrings.
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "activation_rows",
        "activations",
        "der",
        "diarization_error_rate",
        "gradient",
        "gradients",
        "labels",
        "loss",
        "optimizer",
        "packed",
        "qdata",
        "rows",
        "rttm",
        "samples",
        "scales",
        "weight",
        "weights",
    }
)

# The single float list the artifact may carry: the fixed clip-ratio list, which is part of the algorithm's
# identity rather than measured data, and which the caller pins against the constant before scanning.
_ARTIFACT_FLOAT_LIST_PATHS = frozenset({"artifact.arithmetic.clip_ratios"})


def _artifact_structure_violations(value, path="artifact"):
    """Every violation of the artifact's closed structural contract, as readable paths.

    A tensor, a float array or a numeric list longer than the clip-ratio list is exactly the shape an activation
    row, a weight row or a quantized payload would take, so each of them is a violation wherever it appears.
    """
    if path in _ARTIFACT_FLOAT_LIST_PATHS:
        return []
    problems = []
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(f"{path}: non-string key {key!r}")
            problems += _artifact_structure_violations(item, f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) > sq.WEIGHT_SCALE_AWQ_CLIP_RATIO_COUNT and any(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            problems.append(f"{path}: a numeric list of {len(value)} entries is a row, not evidence")
        for index, item in enumerate(value):
            if isinstance(item, float):
                problems.append(f"{path}[{index}]: a float array is a sample or a weight, never evidence")
            problems += _artifact_structure_violations(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        problems.append(f"{path}: {type(value).__name__} is not a JSON value")
    return problems


def _artifact_keys_matching(value, forbidden, path="artifact"):
    """Every path in the artifact whose key is one of ``forbidden``."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in forbidden:
                found.append(f"{path}.{key}")
            found += _artifact_keys_matching(item, forbidden, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += _artifact_keys_matching(item, forbidden, f"{path}[{index}]")
    return found


def _target_weights(model):
    """The unconverted weights of exactly the NVFP4 W4A4 targets, keyed by FQN."""
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    modules = dict(model.named_modules())
    return {fqn: modules[fqn].weight.detach() for fqn in selection.fqns_for_precision(sq.PRECISION_NVFP4_W4A4)}


def _sample_rows(weights, seed, rows=4):
    """Deterministic BF16 sample rows of the right width for every module."""
    generator = torch.Generator().manual_seed(seed)
    return {
        fqn: torch.randn(rows, int(weight.shape[1]), generator=generator).to(torch.bfloat16).contiguous()
        for fqn, weight in sorted(weights.items())
    }


def _sample_payload(builder, samples, **overrides):
    """A complete, valid bounded activation-sample artifact, overridable key by key."""
    payload = {
        "schema": builder.ACTIVATION_SAMPLE_SCHEMA,
        "version": builder.ACTIVATION_SAMPLE_VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
        "metadata": {"manifest": "near.json"},
        "seed": 11,
        "max_rows": 512,
        "total_finite_rows": {fqn: 1000 for fqn in samples},
        "nonfinite_rows": {fqn: 0 for fqn in samples},
        "samples": samples,
    }
    payload.update(overrides)
    return payload


def _write_sample(builder, path, samples, **overrides):
    """Write a sample artifact the way the collector does and return its path."""
    torch.save(_sample_payload(builder, samples, **overrides), str(path))
    return path


def _load(builder, tmp_path, label, samples, name, **overrides):
    """Write one labelled source group and load it back through the strict loader."""
    path = _write_sample(builder, tmp_path / name, samples, **overrides)
    return builder.load_activation_sample_file(label, str(path))


def _entry(builder, tmp_path, label, weights, seed, rows=4, name=None, **overrides):
    """Write and load one labelled source group of deterministic rows."""
    return _load(builder, tmp_path, label, _sample_rows(weights, seed, rows), name or f"{label}.pt", **overrides)


def _amax_for(fqn):
    """Deterministic, strictly positive calibrated activation maximum of one module."""
    return 1.0 + (len(fqn) % 5) * 0.25


def _write_calibration(tmp_path, fqns, name="calib.json", **overrides):
    """Write a merged-style static calibration artifact, overridable field by field.

    The default is the *production* shape ``merge_calibrations`` writes: the checkpoint digest lives in
    ``metadata`` and there is no top-level ``checkpoint_sha256`` at all. ``checkpoint_location`` moves or
    duplicates that claim, and a ``metadata`` override dict replaces individual metadata claims or -- with a
    ``None`` value -- removes one entirely.
    """
    amax = {fqn: _amax_for(fqn) for fqn in fqns}
    if "drop" in overrides:
        amax.pop(overrides["drop"])
    if "extra" in overrides:
        amax[overrides["extra"]] = 1.0
    if "value" in overrides:
        amax[sorted(amax)[0]] = overrides["value"]
    checkpoint = overrides.get("checkpoint", CHECKPOINT_SHA256)
    location = overrides.get("checkpoint_location", "metadata")
    metadata = {
        "method": sq.CALIBRATION_MERGE_METHOD,
        "method_version": sq.CALIBRATION_MERGE_METHOD_VERSION,
        "percentile": 100.0,
        "headroom": 1.375,
        "headroom_baked_in": True,
        "runtime_scale_margin": overrides.get("runtime_scale_margin", 1.0),
    }
    if location in ("metadata", "both"):
        metadata["checkpoint_sha256"] = checkpoint
    for key, value in overrides.get("metadata", {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    payload = {
        "version": overrides.get("version", 1),
        "recipe": "disabled",
        "scale_mode": overrides.get("scale_mode", "static"),
        "targets": overrides.get("targets", list(sq.QUANTIZATION_TARGET_SUFFIXES)),
        "metadata": metadata,
        "activation_amax": amax,
    }
    if location in ("top_level", "both"):
        payload["checkpoint_sha256"] = overrides.get("top_level_checkpoint", checkpoint)
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _argv(tmp_path, calibration, output, inputs):
    """Build a complete CLI argument vector for the builder."""
    argv = [
        "--model-path",
        str(tmp_path / "model.nemo"),
        "--checkpoint-sha256",
        CHECKPOINT_SHA256,
        "--activation-calibration-path",
        str(calibration),
        "--template-arithmetic",
        TEMPLATE_ARITHMETIC,
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    for value in inputs:
        argv += ["--input", value]
    return argv

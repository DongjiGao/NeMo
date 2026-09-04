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

"""Tests for the offline builder of the Sortformer NVFP4 GPTQ payload."""

import base64
import hashlib
import json
import stat
from pathlib import Path

import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_nvfp4_weight_mse as weight_mse
from nemo.collections.asr.parts.utils import sortformer_quantization as sq

D_MODEL = 16
FF_HIDDEN = 32
NUM_LAYERS = 2

CHECKPOINT_SHA256 = "c" * 64

# The builder tests run on the host, so they build for the ordinary conversion that runs there. The payload is
# written under exactly that construction's block scales, so the construction is part of what every artifact here
# is bound to.
TEMPLATE_ARITHMETIC = sq.WEIGHT_SCALE_GPTQ_TEMPLATE_ARITHMETIC_REFERENCE


@pytest.fixture(scope="module")
def builder():
    """Load the GPTQ builder, which lives outside any importable package."""
    import importlib.util  # a plain ``import importlib`` does not load this submodule

    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "dataset_processing"
        / "speaker_tasks"
        / "build_sortformer_nvfp4_gptq.py"
    )
    if not script.exists():
        pytest.skip("the GPTQ payload builder is not available in this checkout")
    spec = importlib.util.spec_from_file_location("build_sortformer_nvfp4_gptq", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gptq_backend():
    """Skip cleanly unless the pinned torchao NVFP4 kernels and the E4M3 dtype are usable on this machine."""
    if not isinstance(getattr(torch, "float8_e4m3fn", None), torch.dtype):
        pytest.skip(f"torch {torch.__version__} does not expose torch.float8_e4m3fn.")
    pytest.importorskip("torchao.prototype.mx_formats.kernels", reason="the GPTQ builder requires torchao 0.17")
    pytest.importorskip("torchao.prototype.mx_formats.utils", reason="the GPTQ builder requires torchao 0.17")
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor", reason="the GPTQ builder requires torchao 0.17")
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
    """Sortformer stand-in carrying exactly the quantization targets the builder selects a payload for."""

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
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert entry["fqns"] == tuple(sorted(weights))
    assert all(int(tensor.shape[0]) == 3 for tensor in entry["samples"].values())


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
        (lambda payload: payload.update(checkpoint_sha256="nope"), "hexadecimal SHA-256"),
        (lambda payload: payload.update(seed=-1), "non-negative integer"),
        (lambda payload: payload.update(max_rows=0), "must be positive"),
        (lambda payload: payload.update(targets="attn.w_qkv"), "list of strings"),
        (lambda payload: payload.update(metadata=[]), "'metadata' as an object"),
        (lambda payload: payload.pop("samples"), "exactly the keys"),
        (lambda payload: payload.update(extra=1), "exactly the keys"),
        (lambda payload: payload.update(total_finite_rows={}), "exactly the sampled modules"),
    ],
)
def test_sample_loader_rejection_matrix(builder, tmp_path, mutate, message):
    weights = _target_weights(_FakeSortformer())
    payload = _sample_payload(builder, _sample_rows(weights, seed=3))
    mutate(payload)
    path = tmp_path / "broken.pt"
    torch.save(payload, str(path))

    with pytest.raises(ValueError, match=message):
        builder.load_activation_sample_file("near_field", str(path))


@pytest.mark.unit
def test_calibration_loader_binds_the_checkpoint_the_targets_and_the_margin(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)

    assert sorted(calibration["activation_amax"]) == fqns
    assert calibration["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert calibration["identity"]["runtime_scale_margin"] == 1.0
    assert all(value > 0.0 for value in calibration["activation_amax"].values())

    with pytest.raises(ValueError, match="was collected on checkpoint"):
        builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), "d" * 64, fqns)
    dynamic = _write_calibration(tmp_path, fqns, name="dyn.json", scale_mode="dynamic")
    with pytest.raises(ValueError, match="declares scale_mode"):
        builder.load_activation_calibration(str(dynamic), CHECKPOINT_SHA256, fqns)
    partial = _write_calibration(tmp_path, fqns, name="partial.json", drop=fqns[0])
    with pytest.raises(ValueError, match="must cover exactly"):
        builder.load_activation_calibration(str(partial), CHECKPOINT_SHA256, fqns)
    unbaked = _write_calibration(tmp_path, fqns, name="unbaked.json", metadata={"headroom_baked_in": False})
    with pytest.raises(ValueError, match="headroom_baked_in"):
        builder.load_activation_calibration(str(unbaked), CHECKPOINT_SHA256, fqns)


@pytest.mark.unit
def test_device_is_resolved_before_anything_is_read_and_never_downgraded(builder):
    assert builder.resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="non-empty PyTorch device string"):
        builder.resolve_device("  ")
    with pytest.raises(ValueError, match="only runs on"):
        builder.resolve_device("meta")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="no CUDA device is available"):
            builder.resolve_device("cuda")


@pytest.mark.unit
def test_encode_qdata_refuses_a_payload_that_is_not_packed_bytes(builder):
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        builder.encode_qdata(object(), 8)
    with pytest.raises(ValueError, match="must be torch.uint8"):
        builder.encode_qdata(torch.zeros(4, 4, dtype=torch.int8), 16)
    with pytest.raises(ValueError, match="needs exactly"):
        builder.encode_qdata(torch.zeros(4, 4, dtype=torch.uint8), 8)
    assert builder.encode_qdata(torch.arange(4, dtype=torch.uint8), 4) == b"\x00\x01\x02\x03"


@pytest.mark.unit
def test_duplicate_labels_and_files_are_refused(builder):
    with pytest.raises(ValueError, match="labels must be unique"):
        builder._require_unique_inputs([("a", "one.pt"), ("a", "two.pt")])
    with pytest.raises(ValueError, match="files must be unique"):
        builder._require_unique_inputs([("a", "one.pt"), ("b", "one.pt")])
    with pytest.raises(ValueError, match="At least one"):
        builder._require_unique_inputs([])
    assert builder._parse_inputs(["near = one.pt"]) == [("near", "one.pt")]
    with pytest.raises(ValueError, match="LABEL=PATH"):
        builder._parse_inputs(["near"])


@pytest.mark.unit
def test_artifact_is_a_complete_runtime_payload_the_loader_accepts(builder, gptq_backend, tmp_path):
    """The end-to-end product: a real selection whose artifact the runtime loader validates against the model."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    entry = _entry(builder, tmp_path, "near_field", weights, seed=11)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)

    payload = builder.build_gptq_artifact(
        weights, [entry], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )

    assert set(payload) == set(sq.GPTQ_ARTIFACT_KEYS)
    assert payload["schema"] == sq.GPTQ_SCHEMA
    assert payload["algorithm"] == sq.WEIGHT_SCALE_GPTQ_ALGORITHM
    assert payload["arithmetic"]["template_arithmetic"] == TEMPLATE_ARITHMETIC
    assert payload["arithmetic"]["perc_damp"] == 0.01
    assert payload["arithmetic"]["update_block_size"] == 128
    assert sorted(payload["qdata"]) == fqns
    assert payload["provenance"]["aggregate"]["source_labels"] == ["near_field"]
    for fqn in fqns:
        rows, columns = int(weights[fqn].shape[0]), int(weights[fqn].shape[1])
        raw = base64.b64decode(payload["qdata"][fqn]["payload"], validate=True)
        assert payload["qdata"][fqn]["shape"] == [rows, columns // 2]
        assert payload["qdata"][fqn]["byte_length"] == rows * columns // 2
        assert len(raw) == rows * columns // 2
        assert payload["qdata"][fqn]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert payload["hessian"][fqn]["input_features"] == columns
        assert payload["hessian"][fqn]["sampled_row_count"] == int(entry["samples"][fqn].shape[0])
        assert payload["hessian"][fqn]["damping"] > 0.0
        assert payload["provenance"]["modules"][fqn]["weight_count"] == rows * columns

    # And the runtime loader accepts it against the very model it was built for.
    path = tmp_path / "gptq.json"
    builder.write_gptq_artifact(payload, str(path))
    loaded = sq.load_gptq_artifact(
        str(path),
        model,
        sq.select_quantization_targets(model, "nvfp4_all"),
        str(tmp_path / "calib.json"),
    )
    assert loaded["fqns"] == fqns
    assert loaded["template_arithmetic"] == TEMPLATE_ARITHMETIC
    for fqn in fqns:
        assert loaded["qdata"][fqn] == base64.b64decode(payload["qdata"][fqn]["payload"], validate=True)


@pytest.mark.unit
def test_artifact_carries_no_activation_row_weight_or_hessian(builder, gptq_backend, tmp_path):
    """The payload is the only bulk data the artifact may carry; the evidence it was selected from is not in it."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    payload = _build(builder, tmp_path, weights, [("near_field", 12)])

    text = json.dumps(payload)
    for forbidden in ("activation_rows", "second_moments", "diagonal_hessian", "rttm"):
        assert forbidden not in text
    # The Hessian and template-scale sections carry scalars, digests and short shape lists only: no matrix, no
    # weight and no activation row survives into the artifact.
    for section in ("hessian", "template_scale"):
        for entry in payload[section].values():
            for value in entry.values():
                assert isinstance(value, (str, int, float, list))
                if isinstance(value, list):
                    assert len(value) <= 8 and all(isinstance(item, int) for item in value)
    assert all(set(entry) == set(sq.GPTQ_HESSIAN_KEYS) for entry in payload["hessian"].values())
    # The only bulk data is the packed payload itself, which is the artifact's whole point.
    assert all(isinstance(entry["payload"], str) for entry in payload["qdata"].values())


@pytest.mark.unit
def test_groups_are_balanced_and_row_counts_do_not_reweight_a_group(builder, gptq_backend, tmp_path):
    """Duplicating a group's own rows must not change the payload; adding a second group must."""
    weights = _target_weights(_FakeSortformer())
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    rows = _sample_rows(weights, seed=13, rows=4)
    doubled = {fqn: torch.cat((tensor, tensor), dim=0).contiguous() for fqn, tensor in rows.items()}
    other = _sample_rows(weights, seed=14, rows=4)

    single = _artifact(builder, tmp_path, weights, calibration, [("near_field", rows, "near.pt")])
    twice = _artifact(builder, tmp_path, weights, calibration, [("near_a", rows, "a1.pt"), ("near_b", rows, "a2.pt")])
    repeated = _artifact(builder, tmp_path, weights, calibration, [("near_field", doubled, "near_x2.pt")])
    balanced = _artifact(
        builder, tmp_path, weights, calibration, [("near_field", rows, "a.pt"), ("far_field", other, "b.pt")]
    )

    # The same rows listed as two equally weighted groups reduce to exactly the same Hessian and therefore to
    # exactly the same payload: ``(H + H) / 2`` loses nothing.
    assert _payloads(twice) == _payloads(single)
    # Duplicating the rows *inside* one group divides by that group's own row count, so the Hessian is the same up
    # to the last unit in the last place of two FP32 normalizations, and the per-module objectives agree.
    for fqn in fqns:
        for field in ("selected_objective", "template_objective", "selected_mse", "template_mse"):
            assert repeated["provenance"]["modules"][fqn][field] == pytest.approx(
                single["provenance"]["modules"][fqn][field], rel=1e-4
            )
    # A second, genuinely different group does change the balance, which is what equal weighting means.
    assert _payloads(balanced) != _payloads(single)
    assert balanced["provenance"]["aggregate"]["source_labels"] == ["far_field", "near_field"]
    assert balanced["provenance"]["aggregate"]["source_count"] == 2


@pytest.mark.unit
def test_group_label_order_does_not_depend_on_the_command_line_order(builder, gptq_backend, tmp_path):
    """Groups are reduced in sorted label order, so two orderings of the same inputs build the same artifact."""
    weights = _target_weights(_FakeSortformer())
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    near = ("near_field", _sample_rows(weights, seed=15, rows=4), "near.pt")
    far = ("far_field", _sample_rows(weights, seed=16, rows=4), "far.pt")

    forward = _artifact(builder, tmp_path, weights, calibration, [near, far])
    backward = _artifact(builder, tmp_path, weights, calibration, [far, near])

    assert forward == backward


@pytest.mark.unit
def test_build_is_deterministic(builder, gptq_backend, tmp_path):
    weights = _target_weights(_FakeSortformer())
    first = _build(builder, tmp_path, weights, [("near_field", 17)])
    second = _build(builder, tmp_path, weights, [("near_field", 17)])

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.unit
def test_write_refuses_to_overwrite_cleans_up_and_sets_mode_0644(builder, gptq_backend, tmp_path):
    weights = _target_weights(_FakeSortformer())
    payload = _build(builder, tmp_path, weights, [("near_field", 18)])
    path = tmp_path / "out" / "gptq.json"

    builder.write_gptq_artifact(payload, str(path))

    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert builder.artifact_file_mode(path) == 0o644
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        builder.write_gptq_artifact(payload, str(path))
    assert path.read_bytes() == original
    assert not _temporaries(path)

    builder.write_gptq_artifact(payload, str(path), overwrite=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert json.loads(path.read_text(encoding="utf-8")) == payload

    # A failed write leaves neither a damaged destination nor a temporary file beside it.
    broken = dict(payload)
    broken["provenance"] = {"unserializable": object()}
    with pytest.raises(TypeError):
        builder.write_gptq_artifact(broken, str(path), overwrite=True)
    assert path.read_bytes() == original
    assert not _temporaries(path)


@pytest.mark.unit
def test_build_refuses_a_foreign_checkpoint_target_set_or_construction(builder, gptq_backend, tmp_path):
    weights = _target_weights(_FakeSortformer())
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    entry = _entry(builder, tmp_path, "near_field", weights, seed=19)

    with pytest.raises(ValueError, match="template_arithmetic must be one of"):
        builder.build_gptq_artifact(
            weights, [entry], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic="guessed"
        )
    with pytest.raises(ValueError, match="At least one labelled"):
        builder.build_gptq_artifact(
            weights, [], calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
        )
    foreign = _entry(builder, tmp_path, "far_field", weights, seed=20, name="far.pt", checkpoint_sha256="d" * 64)
    with pytest.raises(ValueError, match="was collected on checkpoint"):
        builder.build_gptq_artifact(
            weights,
            [foreign],
            calibration,
            checkpoint_sha256=CHECKPOINT_SHA256,
            template_arithmetic=TEMPLATE_ARITHMETIC,
        )
    narrowed = {fqn: weights[fqn] for fqn in fqns[:2]}
    with pytest.raises(ValueError, match="cannot be built from a partial calibration"):
        builder.build_gptq_artifact(
            narrowed,
            [entry],
            calibration,
            checkpoint_sha256=CHECKPOINT_SHA256,
            template_arithmetic=TEMPLATE_ARITHMETIC,
        )


@pytest.mark.unit
def test_module_payload_is_written_under_the_templates_own_fixed_scales(builder, gptq_backend, tmp_path):
    """The builder never derives a scale of its own: it reads the ordinary template's and records its identity."""
    weights = _target_weights(_FakeSortformer())
    fqn = sorted(weights)[0]
    weight = weights[fqn]
    rows = _sample_rows(weights, seed=21, rows=4)[fqn].to(torch.float32)

    selected = builder.select_module_payload(weight, [rows], 2.0, TEMPLATE_ARITHMETIC)

    scale = weight_mse.nvfp4_weight_global_scale(weight)
    template = weight_mse.nvfp4_ordinary_template(weight, scale, TEMPLATE_ARITHMETIC)
    identity = weight_mse.nvfp4_template_identity(template)
    assert selected["template_scale"]["sha256"] == sq.nvfp4_weight_digest(identity.scale)
    assert selected["template_scale"]["global_scale_sha256"] == sq.nvfp4_weight_digest(identity.global_scale)
    assert selected["template_scale"]["byte_length"] == int(identity.scale.numel())

    # And the recorded payload is exactly the one the packer selects under those scales and that Hessian.
    quantized = weight_mse.nvfp4_awq_clip_activation_qdq(rows, 2.0)
    damped = weight_mse.nvfp4_gptq_damped_hessian(weight_mse.nvfp4_gptq_hessian([quantized]), weight)
    expected = weight_mse.select_nvfp4_gptq_payload(
        weight, scale, weight_mse.nvfp4_template_block_scales(template), damped.matrix
    )
    raw = expected.qdata.detach().to("cpu").contiguous().reshape(-1).numpy().tobytes()
    assert base64.b64decode(selected["qdata"]["payload"], validate=True) == raw
    assert selected["hessian"]["sha256"] == sq.nvfp4_gptq_hessian_digest(damped.matrix)
    assert selected["hessian"]["dead_column_count"] == int(damped.dead_columns)
    assert selected["hessian"]["damping"] == pytest.approx(float(damped.damping))
    assert selected["module"]["selected_objective"] >= 0.0
    assert selected["module"]["template_objective"] >= 0.0


@pytest.mark.unit
def test_cli_writes_the_artifact_and_prints_a_summary_without_claiming_der(
    builder, gptq_backend, tmp_path, monkeypatch, capsys
):
    model = _FakeSortformer()
    weights = _target_weights(model)
    fqns = sorted(weights)
    sample = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, seed=22, rows=4))
    calibration = _write_calibration(tmp_path, fqns)
    output = tmp_path / "cli.json"
    monkeypatch.setattr(builder, "restore_target_weights", lambda *args, **kwargs: weights)

    assert builder.main(_argv(tmp_path, calibration, output, [f"near_field={sample}"])) == 0

    printed = capsys.readouterr().out
    assert str(output) in printed
    assert "not DER" in printed
    written = json.loads(output.read_text(encoding="utf-8"))
    assert set(written) == set(sq.GPTQ_ARTIFACT_KEYS)
    assert stat.S_IMODE(output.stat().st_mode) == 0o644

    # And a second run without --overwrite refuses rather than replacing the artifact.
    with pytest.raises(SystemExit, match="already exists"):
        builder.main(_argv(tmp_path, calibration, output, [f"near_field={sample}"]))


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


def _entry(builder, tmp_path, label, weights, seed, rows=4, name=None, **overrides):
    """Write and load one labelled source group of deterministic rows."""
    path = _write_sample(builder, tmp_path / (name or f"{label}.pt"), _sample_rows(weights, seed, rows), **overrides)
    return builder.load_activation_sample_file(label, str(path))


def _artifact(builder, tmp_path, weights, calibration, groups):
    """Build one artifact from explicit ``(label, rows, filename)`` groups."""
    entries = []
    for label, rows, name in groups:
        path = _write_sample(builder, tmp_path / name, rows)
        entries.append(builder.load_activation_sample_file(label, str(path)))
    return builder.build_gptq_artifact(
        weights, entries, calibration, checkpoint_sha256=CHECKPOINT_SHA256, template_arithmetic=TEMPLATE_ARITHMETIC
    )


def _build(builder, tmp_path, weights, groups):
    """Build one artifact from ``(label, seed)`` groups, writing each sample file once."""
    fqns = sorted(weights)
    calibration = builder.load_activation_calibration(str(_write_calibration(tmp_path, fqns)), CHECKPOINT_SHA256, fqns)
    return _artifact(
        builder,
        tmp_path,
        weights,
        calibration,
        [(label, _sample_rows(weights, seed=seed), f"{label}.pt") for label, seed in groups],
    )


def _payloads(artifact):
    """The packed payload strings of one artifact, which is what two builds must agree on byte for byte."""
    return {fqn: entry["payload"] for fqn, entry in artifact["qdata"].items()}


def _temporaries(path):
    """Any leftover temporary file the atomic writer might have left beside the destination."""
    return sorted(str(candidate) for candidate in path.parent.glob(f".{path.name}.*"))


def _amax_for(fqn):
    """Deterministic, strictly positive calibrated activation maximum of one module."""
    return 1.0 + (len(fqn) % 5) * 0.25


def _write_calibration(tmp_path, fqns, name="calib.json", **overrides):
    """Write a merged-style static calibration artifact of exactly the target set, overridable field by field."""
    amax = {fqn: _amax_for(fqn) for fqn in fqns}
    if "drop" in overrides:
        amax.pop(overrides["drop"])
    metadata = {
        "method": sq.CALIBRATION_MERGE_METHOD,
        "method_version": sq.CALIBRATION_MERGE_METHOD_VERSION,
        "percentile": 100.0,
        "headroom": 1.375,
        "headroom_baked_in": True,
        "runtime_scale_margin": 1.0,
        "checkpoint_sha256": overrides.get("checkpoint", CHECKPOINT_SHA256),
    }
    for key, value in overrides.get("metadata", {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    payload = {
        "version": sq.CALIBRATION_SCHEMA_VERSION,
        "recipe": "disabled",
        "scale_mode": overrides.get("scale_mode", "static"),
        "targets": list(sq.QUANTIZATION_TARGET_SUFFIXES),
        "metadata": metadata,
        "activation_amax": amax,
    }
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

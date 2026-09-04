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

"""Tests for the offline builder of the Sortformer NVFP4 diagonal-Hessian statistics."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from nemo.collections.asr.parts.utils import sortformer_quantization as sq

D_MODEL = 8
FF_HIDDEN = 16
NUM_LAYERS = 2


@pytest.fixture(scope="module")
def builder():
    """Load the statistics builder, which lives outside any importable package."""
    import importlib.util  # a plain ``import importlib`` does not load this submodule

    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "dataset_processing"
        / "speaker_tasks"
        / "build_sortformer_nvfp4_hessian_stats.py"
    )
    if not script.exists():
        pytest.skip("the diagonal-Hessian statistics builder is not available in this checkout")
    spec = importlib.util.spec_from_file_location("build_sortformer_nvfp4_hessian_stats", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    """Sortformer stand-in carrying exactly the quantization targets the builder collects statistics for."""

    def __init__(self):
        super().__init__()
        self.pre_encode = torch.nn.Linear(D_MODEL, D_MODEL)
        self.transformer_encoder = _FakeEncoder()
        self.head = torch.nn.Linear(D_MODEL, 4)


CHECKPOINT_SHA256 = "c" * 64


def _target_weights(model):
    """The unconverted weights of exactly the NVFP4 W4A4 targets, keyed by FQN."""
    selection = sq.select_quantization_targets(model, "nvfp4_all")
    modules = dict(model.named_modules())
    return {fqn: modules[fqn].weight.detach() for fqn in selection.fqns_for_precision(sq.PRECISION_NVFP4_W4A4)}


def _sample_rows(weights, value, rows=4):
    """Constant sample rows of the right width for every module, so the expected moments are exactly computable."""
    return {
        fqn: torch.full((rows, int(weight.shape[1])), float(value), dtype=torch.bfloat16)
        for fqn, weight in weights.items()
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


def _entry(builder, tmp_path, label, weights, value, rows=4, **overrides):
    """Write one labelled source group and load it back through the strict loader."""
    path = _write_sample(builder, tmp_path / f"{label}.pt", _sample_rows(weights, value, rows), **overrides)
    return builder.load_activation_sample_file(label, str(path))


@pytest.mark.unit
def test_sample_loader_reports_the_file_identity_and_the_retained_rows(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    path = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, 2.0, rows=3))

    entry = builder.load_activation_sample_file("  near_field  ", str(path))

    assert entry["label"] == "near_field"
    assert entry["path"] == str(path)
    assert entry["name"] == "near.pt"
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["size_bytes"] == path.stat().st_size
    assert entry["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert entry["seed"] == 11
    assert entry["max_rows"] == 512
    assert entry["targets"] == tuple(sq.QUANTIZATION_TARGET_SUFFIXES)
    assert entry["fqns"] == tuple(sorted(weights))
    assert all(tensor.shape[0] == 3 for tensor in entry["samples"].values())


@pytest.mark.unit
def test_sample_loader_reads_without_executing_code(builder, tmp_path, monkeypatch):
    """Sample files are untrusted evidence, so they are always loaded with ``weights_only=True``."""
    weights = _target_weights(_FakeSortformer())
    path = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, 1.0))
    seen = {}
    original = torch.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    builder.load_activation_sample_file("near_field", str(path))

    assert seen["weights_only"] is True
    assert seen["map_location"] == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload, samples: payload.pop("seed"), "exactly the keys"),
        (lambda payload, samples: payload.update(extra=1), "exactly the keys"),
        (lambda payload, samples: payload.update(schema="other"), "declares schema"),
        (lambda payload, samples: payload.update(version=2), "has version"),
        (lambda payload, samples: payload.update(version=True), "has version"),
        (lambda payload, samples: payload.update(checkpoint_sha256="nope"), "hexadecimal SHA-256"),
        (lambda payload, samples: payload.update(seed=-1), "non-negative integer"),
        (lambda payload, samples: payload.update(seed=True), "non-negative integer"),
        (lambda payload, samples: payload.update(max_rows=0), "must be positive"),
        (lambda payload, samples: payload.update(targets="attn.w_qkv"), "list of strings"),
        (lambda payload, samples: payload.update(metadata=[]), "'metadata' as an object"),
        (lambda payload, samples: payload.update(samples={}), "non-empty 'samples'"),
        (lambda payload, samples: payload.update(max_rows=2), "exceeds the declared bound"),
        (lambda payload, samples: payload["samples"].update({next(iter(samples)): torch.ones(4)}), "rank-2 rows"),
        (
            lambda payload, samples: payload["samples"].update(
                {next(iter(samples)): torch.ones(4, 8, dtype=torch.float32)}
            ),
            "the collector retains",
        ),
        (
            lambda payload, samples: payload["samples"].update(
                {next(iter(samples)): torch.ones(8, 4, dtype=torch.bfloat16).t()}
            ),
            "non-contiguous rows",
        ),
        (
            lambda payload, samples: payload["samples"][next(iter(samples))].__setitem__((0, 0), float("inf")),
            "non-finite rows",
        ),
        (lambda payload, samples: payload["total_finite_rows"].popitem(), "over exactly the sampled modules"),
        (lambda payload, samples: payload.update(nonfinite_rows=[]), "over exactly the sampled modules"),
        (
            lambda payload, samples: payload["total_finite_rows"].update({next(iter(samples)): 1}),
            "cannot keep more rows than it saw",
        ),
    ],
)
def test_sample_loader_rejection_matrix(builder, tmp_path, mutate, message):
    weights = _target_weights(_FakeSortformer())
    samples = _sample_rows(weights, 1.0)
    payload = _sample_payload(builder, samples)
    mutate(payload, samples)
    path = tmp_path / "bad.pt"
    torch.save(payload, str(path))

    with pytest.raises(ValueError, match=message):
        builder.load_activation_sample_file("near_field", str(path))


@pytest.mark.unit
def test_sample_loader_rejects_an_empty_label_and_a_non_dict_file(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    path = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, 1.0))
    with pytest.raises(ValueError, match="empty source label"):
        builder.load_activation_sample_file("   ", str(path))

    torch.save([1, 2, 3], str(tmp_path / "list.pt"))
    with pytest.raises(ValueError, match="must contain a dict"):
        builder.load_activation_sample_file("near_field", str(tmp_path / "list.pt"))


@pytest.mark.unit
def test_groups_are_weighted_equally_regardless_of_how_many_rows_they_kept(builder, tmp_path):
    """A high-volume corpus must not outvote a small stratum: the merge is a mean over groups, not over rows."""
    weights = _target_weights(_FakeSortformer())
    big = _entry(builder, tmp_path, "near_field", weights, value=1.0, rows=64)
    small = _entry(builder, tmp_path, "far_field", weights, value=3.0, rows=1)

    merged = builder.merge_second_moments([big, small], sorted(weights))

    # (1^2 + 3^2) / 2 == 5.0, not the row-weighted (64 * 1 + 1 * 9) / 65.
    for fqn, weight in weights.items():
        assert merged[fqn] == [pytest.approx(5.0)] * int(weight.shape[1])


@pytest.mark.unit
def test_merged_moments_do_not_depend_on_the_order_the_groups_were_given(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    first = _entry(builder, tmp_path, "near_field", weights, value=1.5, rows=5)
    second = _entry(builder, tmp_path, "far_field", weights, value=2.5, rows=9)

    assert builder.merge_second_moments([first, second], sorted(weights)) == builder.merge_second_moments(
        [second, first], sorted(weights)
    )


@pytest.mark.unit
def test_merge_rejects_no_group_and_moments_that_cannot_weight_anything(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    with pytest.raises(ValueError, match="At least one labelled activation-sample file"):
        builder.merge_second_moments([], sorted(weights))

    zero = _entry(builder, tmp_path, "silent", weights, value=0.0, rows=4)
    with pytest.raises(ValueError, match="identically zero"):
        builder.merge_second_moments([zero], sorted(weights))


@pytest.mark.unit
def test_artifact_binds_the_checkpoint_the_weights_and_the_labelled_sources(builder, tmp_path):
    model = _FakeSortformer()
    weights = _target_weights(model)
    far = _entry(builder, tmp_path, "far_field", weights, value=3.0, rows=2)
    near = _entry(builder, tmp_path, "near_field", weights, value=1.0, rows=6)

    payload = builder.build_hessian_artifact(weights, [far, near], checkpoint_sha256=CHECKPOINT_SHA256)

    assert payload["schema"] == sq.HESSIAN_SCHEMA
    assert payload["version"] == sq.HESSIAN_SCHEMA_VERSION
    assert payload["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert payload["algorithm"] == sq.WEIGHT_SCALE_HESSIAN_ALGORITHM
    assert payload["algorithm_version"] == sq.WEIGHT_SCALE_HESSIAN_ALGORITHM_VERSION
    assert payload["damping"] == sq.WEIGHT_SCALE_HESSIAN_DAMPING
    assert payload["weight_digest_method"] == sq.WEIGHT_DIGEST_METHOD
    assert payload["weight_sha256"] == {fqn: sq.nvfp4_weight_digest(weight) for fqn, weight in weights.items()}
    # One digest over the moments and one over the provenance, recorded exactly as the runtime recomputes them.
    assert payload["moment_sha256"] == sq.nvfp4_section_digest(payload["diagonal_hessian"])
    assert payload["provenance_sha256"] == sq.nvfp4_section_digest(payload["provenance"])
    assert sorted(payload["diagonal_hessian"]) == sorted(weights)
    for fqn, weight in weights.items():
        assert len(payload["diagonal_hessian"][fqn]) == int(weight.shape[1])

    provenance = payload["provenance"]
    assert provenance["method"] == sq.HESSIAN_CONSTRUCTION_METHOD
    assert provenance["method_version"] == sq.HESSIAN_CONSTRUCTION_METHOD_VERSION
    assert provenance["objective"] == sq.HESSIAN_OBJECTIVE
    assert provenance["group_reduction"] == sq.HESSIAN_GROUP_REDUCTION
    assert provenance["targets"] == list(sq.QUANTIZATION_TARGET_SUFFIXES)
    assert provenance["target_fqns"] == sorted(weights)
    assert provenance["target_module_count"] == len(weights)
    # Sources are recorded in sorted label order, which is also the order they were merged in.
    assert [source["label"] for source in provenance["sources"]] == ["far_field", "near_field"]
    assert provenance["aggregate"]["source_labels"] == ["far_field", "near_field"]
    assert provenance["aggregate"]["module_count"] == len(weights)
    assert provenance["aggregate"]["source_count"] == 2
    assert provenance["aggregate"]["moment_count"] == sum(int(w.shape[1]) for w in weights.values())
    assert provenance["aggregate"]["moment_min"] == pytest.approx(5.0)
    assert provenance["aggregate"]["moment_max"] == pytest.approx(5.0)
    near_source = provenance["sources"][1]
    assert near_source["sha256"] == near["sha256"]
    assert near_source["size_bytes"] == near["size_bytes"]
    assert near_source["seed"] == 11 and near_source["max_rows"] == 512
    assert near_source["sampled_row_count"] == 6 * len(weights)
    assert near_source["metadata"] == {"manifest": "near.json"}


@pytest.mark.unit
def test_artifact_carries_no_activation_row_and_no_weight(builder, tmp_path):
    """The artifact is statistics only: it must not leak the rows or the weights it was built from."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    entry = _entry(builder, tmp_path, "near_field", weights, value=1.5, rows=3)

    payload = builder.build_hessian_artifact(weights, [entry], checkpoint_sha256=CHECKPOINT_SHA256)
    output = builder.write_hessian_artifact(payload, str(tmp_path / "hessian.json"))
    text = Path(output).read_text(encoding="utf-8")

    assert "samples" not in text
    assert "rttm" not in text.lower()
    assert set(json.loads(text)) == set(sq.HESSIAN_ARTIFACT_KEYS)
    # Nothing in the payload is a tensor, so nothing about a row or a weight survives into the JSON.
    assert not any(isinstance(value, torch.Tensor) for value in payload.values())
    # The only numbers written for a module are its second moments -- here the single constant 1.5 ** 2 -- and
    # never the weights they will be used to quantize.
    written = {round(float(value), 4) for vector in payload["diagonal_hessian"].values() for value in vector}
    assert written == {round(1.5**2, 4)}
    weight_values = {round(float(value), 4) for weight in weights.values() for value in weight.flatten().tolist()}
    assert written.isdisjoint(weight_values)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"checkpoint_sha256": "d" * 64}, "was collected on checkpoint"),
        ({"targets": ["attn.w_qkv"]}, "declares targets"),
    ],
)
def test_artifact_rejects_a_sample_file_describing_another_run(builder, tmp_path, overrides, message):
    weights = _target_weights(_FakeSortformer())
    entry = _entry(builder, tmp_path, "near_field", weights, value=1.0, **overrides)

    with pytest.raises(ValueError, match=message):
        builder.build_hessian_artifact(weights, [entry], checkpoint_sha256=CHECKPOINT_SHA256)


@pytest.mark.unit
def test_artifact_rejects_a_sample_file_covering_other_modules_or_other_widths(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    partial = dict(list(_sample_rows(weights, 1.0).items())[:-1])
    path = _write_sample(builder, tmp_path / "partial.pt", partial)
    with pytest.raises(ValueError, match="does not cover the"):
        builder.build_hessian_artifact(
            weights,
            [builder.load_activation_sample_file("near_field", str(path))],
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    narrow = _sample_rows(weights, 1.0)
    fqn = sorted(narrow)[0]
    narrow[fqn] = torch.ones(4, int(weights[fqn].shape[1]) - 1, dtype=torch.bfloat16)
    path = _write_sample(builder, tmp_path / "narrow.pt", narrow)
    with pytest.raises(ValueError, match="input channel"):
        builder.build_hessian_artifact(
            weights,
            [builder.load_activation_sample_file("near_field", str(path))],
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


@pytest.mark.unit
def test_artifact_requires_a_valid_digest_a_source_and_a_target(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    entry = _entry(builder, tmp_path, "near_field", weights, value=1.0)

    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        builder.build_hessian_artifact(weights, [entry], checkpoint_sha256="nope")
    with pytest.raises(ValueError, match="At least one labelled activation-sample file"):
        builder.build_hessian_artifact(weights, [], checkpoint_sha256=CHECKPOINT_SHA256)
    with pytest.raises(ValueError, match="nothing to collect statistics for"):
        builder.build_hessian_artifact({}, [entry], checkpoint_sha256=CHECKPOINT_SHA256)


@pytest.mark.unit
def test_write_is_deterministic_atomic_and_refuses_to_overwrite(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    entry = _entry(builder, tmp_path, "near_field", weights, value=1.25)
    payload = builder.build_hessian_artifact(weights, [entry], checkpoint_sha256=CHECKPOINT_SHA256)
    output = tmp_path / "out" / "hessian.json"

    written = builder.write_hessian_artifact(payload, str(output))
    first = Path(written).read_bytes()
    builder.write_hessian_artifact(payload, str(tmp_path / "out" / "again.json"))

    assert first == (tmp_path / "out" / "again.json").read_bytes()
    # Deterministic bytes: sorted keys, two-space indent, UTF-8 and a trailing newline.
    expected = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    assert first.decode("utf-8") == expected
    # The write is atomic: no temporary file is left behind next to the destination.
    assert sorted(item.name for item in output.parent.iterdir()) == ["again.json", "hessian.json"]

    with pytest.raises(FileExistsError, match="already exists"):
        builder.write_hessian_artifact(payload, str(output))
    assert Path(written).read_bytes() == first

    builder.write_hessian_artifact(payload, str(output), overwrite=True)
    assert Path(written).read_bytes() == first


@pytest.mark.unit
def test_write_refuses_a_non_finite_moment(builder, tmp_path):
    weights = _target_weights(_FakeSortformer())
    entry = _entry(builder, tmp_path, "near_field", weights, value=1.0)
    payload = builder.build_hessian_artifact(weights, [entry], checkpoint_sha256=CHECKPOINT_SHA256)
    payload["diagonal_hessian"][sorted(weights)[0]][0] = float("nan")

    with pytest.raises(ValueError):
        builder.write_hessian_artifact(payload, str(tmp_path / "nan.json"))
    # The atomic write means the destination is never created from a payload that could not be serialized.
    assert not (tmp_path / "nan.json").exists()


@pytest.mark.unit
def test_a_failed_write_leaves_no_temporary_file_beside_the_destination(builder, tmp_path):
    """Every failure after the temporary file exists must remove it: no partial statistics JSON is left behind."""
    weights = _target_weights(_FakeSortformer())
    entry = _entry(builder, tmp_path, "near_field", weights, value=1.0)
    payload = builder.build_hessian_artifact(weights, [entry], checkpoint_sha256=CHECKPOINT_SHA256)
    destination = tmp_path / "out" / "hessian.json"

    # A serialization failure happens *after* the temporary file was created, halfway through the dump.
    broken = json.loads(json.dumps(payload))
    broken["diagonal_hessian"][sorted(weights)[0]][0] = float("nan")
    with pytest.raises(ValueError):
        builder.write_hessian_artifact(broken, str(destination))
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []

    # So does a payload that cannot be serialized at all.
    with pytest.raises(TypeError):
        builder.write_hessian_artifact({"schema": {"a", "b"}}, str(destination))
    assert list(destination.parent.iterdir()) == []

    # And so does a rename that cannot happen -- here the destination is an existing directory, which the
    # replace refuses -- with the destination left exactly as it was.
    blocked = tmp_path / "blocked" / "hessian.json"
    blocked.mkdir(parents=True)
    with pytest.raises(OSError):
        builder.write_hessian_artifact(payload, str(blocked), overwrite=True)
    assert blocked.is_dir()
    assert sorted(item.name for item in blocked.parent.iterdir()) == ["hessian.json"]

    # The cleanup did not disturb the ordinary path: the same payload still writes exactly one file.
    assert builder.write_hessian_artifact(payload, str(destination)) == str(destination)
    assert sorted(item.name for item in destination.parent.iterdir()) == ["hessian.json"]


@pytest.mark.unit
def test_written_artifact_is_consumable_by_the_runtime_loader(builder, tmp_path):
    """The builder's output and the runtime's strict loader are two halves of one contract."""
    model = _FakeSortformer()
    weights = _target_weights(model)
    entries = [
        _entry(builder, tmp_path, "near_field", weights, value=1.0, rows=5),
        _entry(builder, tmp_path, "far_field", weights, value=2.0, rows=2),
    ]
    payload = builder.build_hessian_artifact(weights, entries, checkpoint_sha256=CHECKPOINT_SHA256)
    output = builder.write_hessian_artifact(payload, str(tmp_path / "hessian.json"))

    loaded = sq.load_diagonal_hessian(output, model, sq.select_quantization_targets(model, "nvfp4_all"))

    assert loaded["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert loaded["fqns"] == sorted(weights)
    assert loaded["second_moments"] == payload["diagonal_hessian"]
    assert loaded["sha256"] == hashlib.sha256(Path(output).read_bytes()).hexdigest()
    # The recorded component digests survive the pretty-printed write and the loader's parse unchanged.
    assert loaded["moment_sha256"] == payload["moment_sha256"]
    assert loaded["provenance_sha256"] == payload["provenance_sha256"]


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
def test_cli_writes_the_artifact_and_prints_statistics_only(builder, tmp_path, monkeypatch, capsys):
    model = _FakeSortformer()
    weights = _target_weights(model)
    near = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, 1.0, rows=6))
    far = _write_sample(builder, tmp_path / "far.pt", _sample_rows(weights, 3.0, rows=2))
    output = tmp_path / "hessian.json"
    # The restore itself needs a real .nemo checkpoint; everything around it is exercised here.
    monkeypatch.setattr(builder, "restore_target_weights", lambda path, digest, device: weights)

    exit_code = builder.main(
        [
            "--model-path",
            str(tmp_path / "model.nemo"),
            "--checkpoint-sha256",
            CHECKPOINT_SHA256,
            "--device",
            "cpu",
            "--input",
            f"near_field={near}",
            "--input",
            f"far_field={far}",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert payload["provenance"]["aggregate"]["source_labels"] == ["far_field", "near_field"]

    printed = capsys.readouterr().out
    assert str(output) in printed
    assert "2 equally weighted source group(s)" in printed
    assert sq.WEIGHT_SCALE_HESSIAN_ALGORITHM in printed
    assert "says nothing about DER" in printed


@pytest.mark.unit
def test_cli_fails_before_restoring_on_a_repeated_group(builder, tmp_path, monkeypatch):
    weights = _target_weights(_FakeSortformer())
    near = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, 1.0))

    def refuse(*args, **kwargs):
        raise AssertionError("the checkpoint must not be restored for an invalid input set")

    monkeypatch.setattr(builder, "restore_target_weights", refuse)

    with pytest.raises(SystemExit, match="labels must be unique"):
        builder.main(
            [
                "--model-path",
                str(tmp_path / "model.nemo"),
                "--checkpoint-sha256",
                CHECKPOINT_SHA256,
                "--device",
                "cpu",
                "--input",
                f"near_field={near}",
                "--input",
                f"near_field={near}",
                "--output",
                str(tmp_path / "hessian.json"),
            ]
        )
    assert not (tmp_path / "hessian.json").exists()


@pytest.mark.unit
def test_cli_refuses_an_existing_output_without_overwrite(builder, tmp_path, monkeypatch):
    model = _FakeSortformer()
    weights = _target_weights(model)
    near = _write_sample(builder, tmp_path / "near.pt", _sample_rows(weights, 1.0))
    output = tmp_path / "hessian.json"
    output.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(builder, "restore_target_weights", lambda path, digest, device: weights)

    argv = [
        "--model-path",
        str(tmp_path / "model.nemo"),
        "--checkpoint-sha256",
        CHECKPOINT_SHA256,
        "--device",
        "cpu",
        "--input",
        f"near_field={near}",
        "--output",
        str(output),
    ]
    with pytest.raises(SystemExit, match="already exists"):
        builder.main(argv)
    assert output.read_text(encoding="utf-8") == "keep me"

    assert builder.main(argv + ["--overwrite"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == sq.HESSIAN_SCHEMA


@pytest.mark.unit
def test_restore_verifies_the_checkpoint_digest_before_reading_it(builder, tmp_path, monkeypatch):
    """A checkpoint that does not hash to the asserted digest is never restored at all."""
    checkpoint = tmp_path / "model.nemo"
    checkpoint.write_bytes(b"not the checkpoint the statistics claim")

    def refuse(*args, **kwargs):
        raise AssertionError("restore_from must not run for a mismatched checkpoint")

    monkeypatch.setattr(builder.SortformerEncLabelModel, "restore_from", staticmethod(refuse))

    with pytest.raises(ValueError, match="hashes to"):
        builder.restore_target_weights(str(checkpoint), CHECKPOINT_SHA256, torch.device("cpu"))

    assert builder.file_sha256(checkpoint) == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

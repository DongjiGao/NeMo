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

"""The vLLM plugin's per-request store for deferred CTC timestamp inputs, on CPU."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo.collections.asr.modules.parallel_expert_encoder import CTCTimestampInputs
from nemo.collections.speechlm2.vllm.salm import ctc_timestamps as ct


class _FakeEncoder(nn.Module):
    """Records collated inputs and returns one word per token of each transcript."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = []

    def generate_ctc_timestamps(self, timestamp_inputs, sot_transcripts, audio_durations):
        self.calls.append((timestamp_inputs, list(sot_transcripts), list(audio_durations)))
        if any("unalignable" in text for text in sot_transcripts):
            raise ValueError("tokenizer disagreement")
        if any("crash" in text for text in sot_transcripts):
            raise RuntimeError("kernel failure")
        return [
            {
                "speaker_word_timestamps": {
                    0: [
                        {"word": word, "speaker": 0, "start": 0.1 * (index + 1), "end": 0.1 * (index + 1) + 0.08}
                        for index, word in enumerate(text.split())
                    ]
                },
                "diarization_timestamps": [{"speaker": 1, "start": 0.05, "end": 0.93}],
                "speaker_tag_to_sortformer_column": {0: 1},
            }
            for text in sot_transcripts
        ]


def _words(result):
    return [w["word"] for w in result["words"]]


def _inputs(batch, frames):
    return CTCTimestampInputs(
        asr_encoded=torch.randn(batch, 4, frames),
        asr_encoded_lengths=torch.full((batch,), frames),
        sortformer_sigmoids=torch.rand(batch, frames, 2),
        sortformer_lengths=torch.full((batch,), frames),
        diarization_labels=torch.zeros(batch, 4, frames * 8, dtype=torch.bool),
        diarization_lengths=torch.full((batch,), frames * 8),
    )


def _capture(batch, frames, hashes, durations=None):
    """Store one forward's inputs and name them as the runner hook would."""
    row_ids = ct.pending_row_ids(batch)
    ct.store_timestamp_inputs(row_ids, _inputs(batch, frames), durations or [1.0] * batch)
    ct._rename_pending(hashes)


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(ct, "_store", {})
    monkeypatch.setattr(ct, "_request_hashes", {})
    monkeypatch.setattr(ct, "_external_ids", {})
    monkeypatch.setattr(ct, "_registry", {})
    monkeypatch.setattr(ct, "_uncompacted", deque())
    monkeypatch.setattr(ct, "_state", SimpleNamespace())
    fake = _FakeEncoder()
    ct.register_encoder(lambda: fake)
    return fake


def test_external_request_id_resolves_and_rows_from_two_forwards_are_padded(encoder):
    _capture(1, 5, ["hash-a"], durations=[0.4])
    _capture(1, 3, ["hash-b"], durations=[0.24])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-89abcdef", "hash-b")])

    results = ct.align_requests([("req-a", "one two"), ("req-b", "three")])

    assert [_words(result) for result in results] == [["one", "two"], ["three"]]
    inputs, texts, durations = encoder.calls[0]
    assert texts == ["one two", "three"] and durations == [0.4, 0.24]
    assert inputs.asr_encoded.shape == (2, 4, 5)
    assert inputs.asr_encoded_lengths.tolist() == [5, 3]
    assert inputs.sortformer_sigmoids.shape == (2, 5, 2)
    assert inputs.diarization_labels.shape == (2, 4, 40)
    assert torch.count_nonzero(inputs.asr_encoded[1, :, 3:]) == 0


def test_cache_hit_request_and_repeated_alignment_find_the_same_inputs(encoder):
    _capture(1, 4, ["hash-a"])
    first = SimpleNamespace(req_id="req-1", mm_features=[SimpleNamespace(identifier="hash-a")])
    repeat = SimpleNamespace(req_id="req-2", mm_features=[SimpleNamespace(identifier="hash-a")])
    ct._record_request_hashes(ct._new_request_hashes(SimpleNamespace(scheduled_new_reqs=[first, repeat])))

    assert ct.align_request("req-1", "a b") == ct.align_request("req-1", "a b")
    assert _words(ct.align_request("req-2", "c")) == ["c"]


def test_placeholders_queued_outside_the_hook_are_discarded(encoder):
    ct.store_timestamp_inputs(ct.pending_row_ids(1), _inputs(1, 9), [3310.0])
    assert ct._discard_pending() == 1
    _capture(1, 4, ["hash-a"], durations=[0.32])
    ct._record_request_hashes([("req", "hash-a")])

    ct.align_request("req", "word")

    assert encoder.calls[0][0].asr_encoded.shape[-1] == 4
    assert encoder.calls[0][2] == [0.32]


def test_retention_evicts_least_recently_aligned_first(encoder, monkeypatch):
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN", "2")
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-c", "hash-c")])
    ct.align_request("req-a", "keep")
    _capture(1, 4, ["hash-c"])

    ct._trim_store()

    assert set(ct._store) == {"hash-a", "hash-c"}
    assert ct.align_request("req-b", "gone") == ct._empty_result()


def test_one_unalignable_transcript_does_not_cost_the_batch(encoder):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-c", "hash-c")])

    results = ct.align_requests([("req-a", "x"), ("req-b", "unalignable"), ("req-c", "y z")])

    assert [_words(result) for result in results] == [["x"], [], ["y", "z"]]


def test_diarization_segments_and_speaker_mapping_pass_through(encoder):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    result = ct.align_request("req", "<spk:0> hi")

    assert result["diarization"] == [{"speaker": 1, "start": 0.05, "end": 0.93}]
    assert result["speaker_tag_to_diarization_speaker"] == {"0": 1}
    assert result["words"][0]["speaker"] == "0"


def test_default_retention_follows_engine_concurrency_unless_overridden(encoder, monkeypatch):
    monkeypatch.delenv("NEMO_CTC_TIMESTAMP_RETAIN", raising=False)
    assert ct._retention_limit() == 64

    ct.set_default_retention(512)
    assert ct._retention_limit() == 1024

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN", "10")
    assert ct._retention_limit() == 10


def test_request_with_several_audio_items_gets_no_timestamps(encoder):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("two-clips", "hash-a"), ("two-clips", "hash-b"), ("one-clip", "hash-b")])

    results = ct.align_requests([("two-clips", "one two"), ("one-clip", "three")])

    assert results[0] == ct._empty_result()
    assert _words(results[1]) == ["three"]
    assert [texts for _, texts, _ in encoder.calls] == [["three"]]


def test_placeholder_and_hash_count_mismatch_discards_the_forward(encoder):
    ct.store_timestamp_inputs(ct.pending_row_ids(2), _inputs(2, 4), [1.0, 1.0])

    assert ct._rename_pending(["hash-a"]) == 0
    assert ct._store == {}

    _capture(1, 3, ["hash-b"])
    ct._record_request_hashes([("req", "hash-b")])
    assert _words(ct.align_request("req", "ok")) == ["ok"]


def test_compaction_trims_each_row_to_its_valid_frames_in_its_own_storage(encoder):
    inputs = CTCTimestampInputs(
        asr_encoded=torch.randn(2, 4, 5),
        asr_encoded_lengths=torch.tensor([5, 3]),
        sortformer_sigmoids=torch.rand(2, 5, 2),
        sortformer_lengths=torch.tensor([5, 3]),
        diarization_labels=torch.ones(2, 4, 40, dtype=torch.bool),
        diarization_lengths=torch.tensor([40, 24]),
    )
    ct.store_timestamp_inputs(ct.pending_row_ids(2), inputs, [0.4, 0.24])
    ct._rename_pending(["hash-a", "hash-b"])

    assert ct._compact_ready() == 2
    short = ct._store["hash-b"]
    assert short["asr_encoded"].shape == (1, 4, 3)
    assert short["sortformer_sigmoids"].shape == (1, 3, 2)
    assert short["diarization_labels"].shape == (1, 4, 24)
    asr = short["asr_encoded"]
    assert asr.untyped_storage().nbytes() == asr.numel() * asr.element_size()
    assert torch.equal(asr[0], inputs.asr_encoded[1, :, :3])

    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])
    ct.align_requests([("req-a", "one"), ("req-b", "two")])
    collated = encoder.calls[0][0]
    assert collated.asr_encoded.shape == (2, 4, 5)
    assert collated.diarization_labels[0].all()
    assert torch.count_nonzero(collated.diarization_labels[1, :, 24:]) == 0


def test_evicted_rows_are_not_compacted(encoder, monkeypatch):
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN", "1")
    _capture(2, 4, ["hash-a", "hash-b"])

    ct._trim_store()

    assert ct._compact_ready() == 1
    assert list(ct._store) == ["hash-b"]


def test_speaker_prior_weight_defaults_and_rejects_negative_values():
    assert ct.speaker_logprob_weight({}) == 0.25
    assert ct.speaker_logprob_weight(SimpleNamespace(speaker_logprob_weight=0)) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        ct.speaker_logprob_weight({"speaker_logprob_weight": -0.1})


def test_byte_budget_evicts_least_recently_used_but_keeps_the_newest(encoder, monkeypatch):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._compact_ready()
    row_bytes = ct._store["hash-a"]["nbytes"]

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", str(2.5 * row_bytes / 1e9))
    ct._trim_store()
    assert list(ct._store) == ["hash-b", "hash-c"]

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", "0")
    ct._trim_store()
    assert list(ct._store) == ["hash-c"]


def test_reencoded_audio_replaces_its_entry_as_the_most_recent(encoder):
    _capture(1, 4, ["hash-a"])
    first = ct._store["hash-a"]
    _capture(1, 4, ["hash-b"])
    _capture(1, 3, ["hash-a"])

    assert list(ct._store) == ["hash-b", "hash-a"]
    assert first["dropped"] and ct._store["hash-a"]["asr_encoded"].shape[-1] == 3


def test_unexpected_alignment_errors_are_raised_not_hidden(encoder):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])

    with pytest.raises(RuntimeError, match="kernel failure"):
        ct.align_requests([("req-a", "fine"), ("req-b", "crash")])


def test_durations_given_as_a_tensor_reach_the_aligner(encoder):
    ct.store_timestamp_inputs(ct.pending_row_ids(2), _inputs(2, 4), torch.tensor([0.5, 0.25], dtype=torch.float64))
    ct._rename_pending(["hash-a", "hash-b"])
    ct._compact_ready()
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])

    ct.align_requests([("req-a", "x"), ("req-b", "y")])

    assert encoder.calls[0][2] == [0.5, 0.25]
    assert ct._store["hash-b"]["duration"] == 0.25


def test_external_ids_resolve_through_an_index_that_follows_eviction(encoder, monkeypatch):
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN", "1")

    ct._record_request_hashes([(f"req{i}-0123abcd", f"hash-{i}") for i in range(5)])

    assert ct.mm_hashes_for_request("req0") == []
    assert ct.mm_hashes_for_request("req4") == ct.mm_hashes_for_request("req4-0123abcd") == ["hash-4"]
    assert ct._external_ids == {f"req{i}": f"req{i}-0123abcd" for i in range(1, 5)}


def test_only_the_first_tensor_parallel_rank_aligns(encoder, monkeypatch):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    monkeypatch.setattr(ct, "_aligns_here", lambda: False)
    assert ct._worker_align_requests(None, [("req", "x")]) == [ct._empty_result()]
    assert encoder.calls == []

    monkeypatch.setattr(ct, "_aligns_here", lambda: True)
    assert _words(ct._worker_align_requests(None, [("req", "x")])[0]) == ["x"]


def test_output_times_are_rounded_to_milliseconds(encoder):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    words = ct.align_request("req", "a b c")["words"]

    assert [(w["start"], w["end"]) for w in words] == [(0.1, 0.18), (0.2, 0.28), (0.3, 0.38)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_collate_assembles_padded_rows_on_the_gpu(encoder):
    _capture(1, 5, ["hash-a"])
    _capture(1, 3, ["hash-b"])

    batch = ct._collate([ct._store["hash-a"], ct._store["hash-b"]], torch.device("cuda"))

    assert batch.asr_encoded.is_cuda and batch.asr_encoded.shape == (2, 4, 5)
    assert torch.equal(batch.asr_encoded[1, :, :3].cpu(), ct._store["hash-b"]["asr_encoded"][0])
    assert torch.count_nonzero(batch.asr_encoded[1, :, 3:]) == 0

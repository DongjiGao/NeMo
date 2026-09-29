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
        return [
            {
                "speaker_word_timestamps": {
                    0: [
                        {"word": word, "speaker": 0, "start": 0.1 * (index + 1), "end": 0.1 * (index + 1) + 0.08}
                        for index, word in enumerate(text.split())
                    ]
                }
            }
            for text in sot_transcripts
        ]


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
    monkeypatch.setattr(ct, "_registry", {})
    monkeypatch.setattr(ct, "_state", SimpleNamespace())
    fake = _FakeEncoder()
    ct.register_encoder(lambda: fake)
    return fake


def test_external_request_id_resolves_and_rows_from_two_forwards_are_padded(encoder):
    _capture(1, 5, ["hash-a"], durations=[0.4])
    _capture(1, 3, ["hash-b"], durations=[0.24])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-89abcdef", "hash-b")])

    words = ct.align_requests([("req-a", "one two"), ("req-b", "three")])

    assert [[w["word"] for w in item] for item in words] == [["one", "two"], ["three"]]
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
    assert [w["word"] for w in ct.align_request("req-2", "c")] == ["c"]


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
    assert ct.align_request("req-b", "gone") == []


def test_one_unalignable_transcript_does_not_cost_the_batch(encoder):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-c", "hash-c")])

    words = ct.align_requests([("req-a", "x"), ("req-b", "unalignable"), ("req-c", "y z")])

    assert [[w["word"] for w in item] for item in words] == [["x"], [], ["y", "z"]]

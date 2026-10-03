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

import asyncio
import sys
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


def _capture(batch, frames, hashes, durations=None, inputs=None):
    """Store one forward's states under its items' hashes, as during a runner-hook step."""
    ct._begin_step(hashes)
    ct.store_alignment_states(
        ct.take_step_hashes(batch),
        _inputs(batch, frames) if inputs is None else inputs,
        [1.0] * batch if durations is None else durations,
    )
    ct._end_step()


def _align_one(request_id, text, *, release):
    return ct.align_finished_requests([(request_id, text)], release=release)[0]


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(ct, "_store", {})
    monkeypatch.setattr(ct, "_request_hashes", {})
    monkeypatch.setattr(ct, "_hash_owners", {})
    monkeypatch.setattr(ct, "_engine_cached", set())
    monkeypatch.setattr(ct, "_external_ids", {})
    monkeypatch.setattr(ct, "_registry", {})
    monkeypatch.setattr(ct, "_uncompacted", deque())
    monkeypatch.setattr(ct, "_state", SimpleNamespace())
    fake = _FakeEncoder()
    ct.register_encoder(fake)
    return fake


def test_external_request_id_resolves_and_rows_from_two_forwards_are_padded(encoder):
    _capture(1, 5, ["hash-a"], durations=[0.4])
    _capture(1, 3, ["hash-b"], durations=[0.24])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-89abcdef", "hash-b")])

    results = ct.align_finished_requests([("req-a", "one two"), ("req-b", "three")], release=True)

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

    kept = _align_one("req-1", "a b", release=False)
    assert kept == _align_one("req-1", "a b", release=True)
    assert _words(_align_one("req-2", "c", release=True)) == ["c"]


def test_forwards_outside_an_encoder_step_get_no_hashes(encoder):
    # vLLM's startup profiling pass runs the encoder outside the runner hook.
    assert ct.take_step_hashes(1) is None
    _capture(1, 4, ["hash-a"], durations=[0.32])
    assert ct.take_step_hashes(1) is None
    ct._record_request_hashes([("req", "hash-a")])

    _align_one("req", "word", release=True)

    assert encoder.calls[0][0].asr_encoded.shape[-1] == 4
    assert encoder.calls[0][2] == [0.32]


def test_a_shared_capture_is_deleted_once_its_last_owner_is_aligned(encoder, caplog):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req-1", "hash-a"), ("req-2", "hash-a")])

    assert _words(_align_one("req-1", "a", release=True)) == ["a"]
    assert "hash-a" in ct._store and _align_one("req-1", "a", release=True) == ct._empty_result()
    assert "already aligned or released" in caplog.text

    assert _words(_align_one("req-2", "b", release=True)) == ["b"]
    assert ct._store == {} and ct._hash_owners == {} and ct._request_hashes == {}


def test_an_unowned_capture_lives_while_vllm_caches_its_audio(encoder):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._follow_engine_cache(encoded=["hash-a", "hash-b"])
    ct._record_request_hashes([("req-1", "hash-a"), ("req-2", "hash-b")])

    _align_one("req-1", "a", release=True)
    # A later request with the same audio is an encoder-cache hit and captures nothing.
    ct._record_request_hashes([("req-3", "hash-a")])
    assert _words(_align_one("req-3", "c", release=True)) == ["c"]
    assert "hash-a" in ct._store

    ct._follow_engine_cache(freed=["hash-a", "hash-b"])
    assert list(ct._store) == ["hash-b"]
    _align_one("req-2", "b", release=True)
    assert ct._store == {}


def test_one_unalignable_transcript_does_not_cost_the_batch(encoder):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-c", "hash-c")])

    results = ct.align_finished_requests([("req-a", "x"), ("req-b", "unalignable"), ("req-c", "y z")], release=True)

    assert [_words(result) for result in results] == [["x"], [], ["y", "z"]]


def test_diarization_segments_and_speaker_mapping_pass_through(encoder):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    result = _align_one("req", "<spk:0> hi", release=True)

    assert result["diarization"] == [{"speaker": 1, "start": 0.05, "end": 0.93}]
    assert result["speaker_tag_to_diarization_speaker"] == {"0": 1}
    assert result["words"][0]["speaker"] == "0"


def test_release_false_keeps_captures_until_the_worker_releases_them(encoder):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-0123abcd", "hash-b")])

    ct.align_finished_requests([("req-a", "x"), ("req-b", "y")], release=False)
    assert set(ct._store) == {"hash-a", "hash-b"}

    ct._worker_release_requests(None, ["req-a", "req-b", "never-seen"])
    assert ct._store == {} and ct._request_hashes == {} and ct._external_ids == {}


def test_ranks_that_hold_no_captures_still_encode_but_store_nothing(encoder, monkeypatch):
    class StubRunner:
        def _batch_mm_inputs_from_scheduler(self, step):
            return step.hashes, None, [(req_id, None) for req_id in step.req_ids]

        def _execute_mm_encoder(self, step):
            # What _encode_with_ctc_capture does: the forward always runs, and states
            # are stored only under the step's hashes.
            count = len(step.hashes)
            hashes = ct.take_step_hashes(count)
            if hashes is not None:
                ct.store_alignment_states(hashes, _inputs(count, 4), [1.0] * count)
            return "encoded"

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", SimpleNamespace(GPUModelRunner=StubRunner))
    ct.install_encoder_cache_binding()
    monkeypatch.setattr(ct, "_holds_captures", lambda: False)
    step = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req-a-0123abcd", mm_features=[SimpleNamespace(identifier="hash-a")])
        ],
        hashes=["hash-a"],
        req_ids=["req-a-0123abcd"],
        free_encoder_mm_hashes=[],
    )

    assert StubRunner()._execute_mm_encoder(step) == "encoded"

    assert ct._store == {} and ct._request_hashes == {}
    assert ct._worker_align_requests(None, [("req-a", "x")]) == [ct._empty_result()]
    assert encoder.calls == []


def test_offline_api_aligns_repeatedly_then_releases(encoder):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([(f"req-{name}-0123abcd", f"hash-{name}") for name in "abc"])
    methods = {
        ct.WORKER_ALIGN_METHOD: ct._worker_align_requests,
        ct.WORKER_RELEASE_METHOD: ct._worker_release_requests,
    }
    llm = SimpleNamespace(collective_rpc=lambda method, args: [methods[method](None, *args)])
    outputs = [SimpleNamespace(request_id=f"req-{name}", outputs=[SimpleNamespace(text=name)]) for name in "ab"]

    generated = ct.ctc_word_timestamps(llm, outputs, release=False)
    reference = ct.ctc_timestamps(llm, outputs, texts=["ref a", "ref b"])

    assert [[word["word"] for word in words] for words in generated] == [["a"], ["b"]]
    assert [_words(result) for result in reference] == [["ref", "a"], ["ref", "b"]]
    assert list(ct._store) == ["hash-c"]

    ct.ctc_release(llm, [SimpleNamespace(request_id="req-c")])
    assert ct._store == {}


def test_sync_and_async_clients_send_the_same_rpcs_and_get_the_same_results(encoder):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([(f"req-{name}-0123abcd", f"hash-{name}") for name in "abc"])
    methods = {
        ct.WORKER_ALIGN_METHOD: ct._worker_align_requests,
        ct.WORKER_RELEASE_METHOD: ct._worker_release_requests,
    }
    sent = []

    def rpc(method, args):
        sent.append((method, args))
        reply = methods[method](None, *args)
        # Every rank replies; only the first one aligns.
        other_rank = None if reply is None else [ct._empty_result() for _ in reply]
        return [reply, other_rank]

    async def rpc_async(method, args):
        return rpc(method, args)

    items = [("req-a", "a"), ("req-b", "b c"), ("req-c", "d")]
    sync = ct.align(rpc, items, release=False, chunk_size=2)
    sync_sent = sent[:]
    sent.clear()
    via_async = asyncio.run(ct.align_async(rpc_async, items, release=False, chunk_size=2))

    assert via_async == sync
    assert [_words(result) for result in sync] == [["a"], ["b", "c"], ["d"]]
    assert sent == sync_sent and [len(args[0]) for _, args in sent] == [2, 1]

    asyncio.run(ct.release_captures_async(rpc_async, ["req-a", "req-b"]))
    assert list(ct._store) == ["hash-c"]
    ct.release_captures(rpc, ["req-c"])
    assert ct._store == {}

    with pytest.raises(ValueError, match="chunk_size"):
        ct.align(rpc, items, chunk_size=0)


def test_request_with_several_audio_items_gets_no_timestamps(encoder):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("two-clips", "hash-a"), ("two-clips", "hash-b"), ("one-clip", "hash-b")])

    results = ct.align_finished_requests([("two-clips", "one two"), ("one-clip", "three")], release=True)

    assert results[0] == ct._empty_result()
    assert _words(results[1]) == ["three"]
    assert [texts for _, texts, _ in encoder.calls] == [["three"]]


def test_a_step_whose_rows_do_not_match_its_hashes_keeps_no_captures(encoder, caplog):
    ct._begin_step(["hash-a", "hash-b"])
    ct.store_alignment_states(ct.take_step_hashes(1), _inputs(1, 4), [1.0])
    ct._end_step()
    assert ct._store == {} and "dropping this step's CTC timestamp captures" in caplog.text

    ct._begin_step(["hash-c"])
    assert ct.take_step_hashes(2) is None
    ct._end_step()

    _capture(1, 3, ["hash-d"])
    ct._record_request_hashes([("req", "hash-d")])
    assert _words(_align_one("req", "ok", release=True)) == ["ok"]


def test_compaction_trims_each_row_to_its_valid_frames_in_its_own_storage(encoder):
    inputs = CTCTimestampInputs(
        asr_encoded=torch.randn(2, 4, 5),
        asr_encoded_lengths=torch.tensor([5, 3]),
        sortformer_sigmoids=torch.rand(2, 5, 2),
        sortformer_lengths=torch.tensor([5, 3]),
        diarization_labels=torch.ones(2, 4, 40, dtype=torch.bool),
        diarization_lengths=torch.tensor([40, 24]),
    )
    _capture(2, 5, ["hash-a", "hash-b"], [0.4, 0.24], inputs=inputs)

    assert ct._compact_ready() == 2
    short = ct._store["hash-b"]
    assert short["asr_encoded"].shape == (1, 4, 3)
    assert short["sortformer_sigmoids"].shape == (1, 3, 2)
    assert short["diarization_labels"].shape == (1, 4, 24)
    asr = short["asr_encoded"]
    assert asr.untyped_storage().nbytes() == asr.numel() * asr.element_size()
    assert torch.equal(asr[0], inputs.asr_encoded[1, :, :3])

    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])
    ct.align_finished_requests([("req-a", "one"), ("req-b", "two")], release=True)
    collated = encoder.calls[0][0]
    assert collated.asr_encoded.shape == (2, 4, 5)
    assert collated.diarization_labels[0].all()
    assert torch.count_nonzero(collated.diarization_labels[1, :, 24:]) == 0


def test_evicted_rows_are_not_compacted(encoder, monkeypatch):
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", "0")
    _capture(2, 4, ["hash-a", "hash-b"])

    ct._trim_store()

    assert ct._compact_ready() == 1
    assert list(ct._store) == ["hash-b"]


def test_speaker_prior_weight_defaults_and_rejects_negative_values():
    assert ct.read_speaker_prior_weight({}) == 0.25
    assert ct.read_speaker_prior_weight(SimpleNamespace(speaker_logprob_weight=0)) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        ct.read_speaker_prior_weight({"speaker_logprob_weight": -0.1})


def test_model_runner_v2_is_refused():
    ct.require_v1_model_runner(SimpleNamespace())
    ct.require_v1_model_runner(SimpleNamespace(use_v2_model_runner=False))
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        ct.require_v1_model_runner(SimpleNamespace(use_v2_model_runner=True))


def test_byte_budget_evicts_least_recently_used_but_keeps_the_newest(encoder, monkeypatch, caplog):
    monkeypatch.setattr(ct.logging, "once_logged", set())
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._compact_ready()
    row_bytes = ct._store["hash-a"]["nbytes"]

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", str(2.5 * row_bytes / 1e9))
    ct._trim_store()
    assert list(ct._store) == ["hash-b", "hash-c"]

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", "0")
    ct._trim_store()
    assert list(ct._store) == ["hash-c"]
    # A server evicts routinely, so this is said once rather than per step.
    assert caplog.text.count("Evicting CTC timestamp captures") == 1


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
        ct.align_finished_requests([("req-a", "fine"), ("req-b", "crash")], release=True)


def test_tensor_durations_survive_compaction(encoder):
    _capture(2, 4, ["hash-a", "hash-b"], torch.tensor([0.5, 0.25], dtype=torch.float64))
    ct._compact_ready()
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])

    ct.align_finished_requests([("req-a", "x"), ("req-b", "y")], release=False)

    assert encoder.calls[0][2] == [0.5, 0.25]
    assert ct._store["hash-b"]["duration"] == 0.25


def test_external_ids_resolve_through_an_index_that_follows_forgotten_requests(encoder, monkeypatch):
    monkeypatch.setattr(ct, "_MAX_TRACKED_REQUESTS", 4)

    ct._record_request_hashes([(f"req{i}-0123abcd", f"hash-{i}") for i in range(5)])

    assert ct.mm_hashes_for_request("req0") == []
    assert ct.mm_hashes_for_request("req4") == ct.mm_hashes_for_request("req4-0123abcd") == ["hash-4"]
    assert ct._external_ids == {f"req{i}": f"req{i}-0123abcd" for i in range(1, 5)}


def test_only_the_first_tensor_parallel_rank_aligns(encoder, monkeypatch):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    monkeypatch.setattr(ct, "_holds_captures", lambda: False)
    assert ct._worker_align_requests(None, [("req", "x")], False) == [ct._empty_result()]
    assert encoder.calls == []

    monkeypatch.setattr(ct, "_holds_captures", lambda: True)
    assert _words(ct._worker_align_requests(None, [("req", "x")])[0]) == ["x"]


def test_output_times_are_rounded_to_milliseconds(encoder):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    words = _align_one("req", "a b c", release=True)["words"]

    assert [(w["start"], w["end"]) for w in words] == [(0.1, 0.18), (0.2, 0.28), (0.3, 0.38)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_collate_assembles_padded_rows_on_the_gpu(encoder):
    _capture(1, 5, ["hash-a"])
    _capture(1, 3, ["hash-b"])

    batch = ct._collate([ct._store["hash-a"], ct._store["hash-b"]], torch.device("cuda"))

    assert batch.asr_encoded.is_cuda and batch.asr_encoded.shape == (2, 4, 5)
    assert torch.equal(batch.asr_encoded[1, :, :3].cpu(), ct._store["hash-b"]["asr_encoded"][0])
    assert torch.count_nonzero(batch.asr_encoded[1, :, 3:]) == 0


def test_runner_hook_keys_captures_by_hash_maps_cache_hits_and_ignores_outside_forwards(encoder, monkeypatch):
    class StubRunner:
        def _batch_mm_inputs_from_scheduler(self, step):
            return step.hashes, None, [(req_id, None) for req_id in step.req_ids]

        def _execute_mm_encoder(self, step):
            # What _process_audio does for the items vLLM encodes this step.
            if step.hashes:
                count = len(step.hashes)
                ct.store_alignment_states(ct.take_step_hashes(count), _inputs(count, 4), [1.0] * count)
            return "encoded"

    def step(new_requests, encoded, freed=()):
        return SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=req_id, mm_features=[SimpleNamespace(identifier=mm_hash)])
                for req_id, mm_hash in new_requests
            ],
            hashes=[mm_hash for _, mm_hash in encoded],
            req_ids=[req_id for req_id, _ in encoded],
            free_encoder_mm_hashes=list(freed),
        )

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", SimpleNamespace(GPUModelRunner=StubRunner))
    ct.install_encoder_cache_binding()
    runner = StubRunner()

    first = step([("req-a-0123abcd", "hash-a")], [("req-a-0123abcd", "hash-a")])
    assert runner._execute_mm_encoder(first) == "encoded"
    # The same audio again is an encoder-cache hit: scheduled, but never encoded.
    runner._execute_mm_encoder(step([("req-b-0123abcd", "hash-a")], []))
    # A forward outside the hook, like vLLM's startup profiling pass on dummy audio, gets no hashes.
    assert ct.take_step_hashes(1) is None
    runner._execute_mm_encoder(step([("req-c-0123abcd", "hash-c")], [("req-c-0123abcd", "hash-c")]))

    assert set(ct._store) == {"hash-a", "hash-c"} and not ct._uncompacted
    results = ct.align_finished_requests([("req-a", "a"), ("req-b", "b"), ("req-c", "c")], release=True)
    assert [_words(result) for result in results] == [["a"], ["b"], ["c"]]
    assert encoder.calls[0][2] == [1.0, 1.0, 1.0]

    # Released, but kept until vLLM evicts the audio from its encoder cache.
    assert set(ct._store) == {"hash-a", "hash-c"}
    runner._execute_mm_encoder(step([], [], freed=["hash-a"]))
    assert list(ct._store) == ["hash-c"]


def test_offline_alignment_refuses_a_model_without_timestamps(monkeypatch):
    monkeypatch.setattr(ct, "_registry", {})

    with pytest.raises(RuntimeError, match="not enabled"):
        ct._worker_align_requests(None, [("req", "<spk:0> hi")])
    assert ct._worker_align_requests(None, [("req", "<spk:0> hi")], True, False) == [ct._empty_result()]


def test_adapter_on_an_encoder_without_timestamp_support_is_refused():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = SimpleNamespace(encoder=SimpleNamespace())
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = None

    model._maybe_enable_ctc_timestamps(None, SimpleNamespace())
    with pytest.raises(ValueError, match="cannot produce CTC timestamp inputs"):
        model._maybe_enable_ctc_timestamps({"adapter_path": "/adapter.pt"}, SimpleNamespace())


def test_timestamps_refuse_encoder_chunking_outside_the_parallel_expert_encoder():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = SimpleNamespace(encoder=SimpleNamespace(supports_ctc_timestamp_inputs=True))
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = 30.0

    with pytest.raises(ValueError, match="unchunked"):
        model._maybe_enable_ctc_timestamps({"adapter_path": "/adapter.pt"}, SimpleNamespace())


def test_perception_built_for_another_sample_rate_is_refused():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import _require_resampling_rate

    def perception(sample_rate):
        return SimpleNamespace(preprocessor=SimpleNamespace(featurizer=SimpleNamespace(sample_rate=sample_rate)))

    _require_resampling_rate(SimpleNamespace())
    _require_resampling_rate(perception(16000))
    with pytest.raises(ValueError, match="8000 Hz"):
        _require_resampling_rate(perception(8000))


class _FakePerception(nn.Module):
    """A perception module whose encoder is not a ParallelExpertEncoder; records each forward."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.forwards = []

    def forward(self, input_signal, input_signal_length, return_ctc_timestamp_inputs=False):
        self.forwards.append((tuple(input_signal.shape), return_ctc_timestamp_inputs))
        batch = input_signal.shape[0]
        outputs = (torch.ones(batch, 3, 4), torch.full((batch,), 3))
        return (*outputs, _inputs(batch, 4)) if return_ctc_timestamp_inputs else outputs


def _model_with_fake_perception():
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = _FakePerception()
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = None
    return model


def _audio(capture=None):
    capture = None if capture is None else torch.tensor(capture)
    return SimpleNamespace(
        audio_signal=torch.ones(2, 16), audio_signal_length=torch.tensor([16, 8]), capture_ctc_timestamps=capture
    )


def test_any_encoder_with_the_flag_captures_through_one_unchunked_forward(encoder):
    pytest.importorskip("vllm")
    model = _model_with_fake_perception()

    ct._begin_step(["hash-a", "hash-b"])
    embeddings = model._process_audio(_audio())
    ct._end_step()

    assert [tuple(e.shape) for e in embeddings] == [(3, 4), (3, 4)]
    assert model.perception.forwards == [((2, 16), True)]
    assert list(ct._store) == ["hash-a", "hash-b"]


def test_a_forward_outside_the_hook_produces_states_but_stores_none(encoder):
    pytest.importorskip("vllm")
    model = _model_with_fake_perception()

    model._process_audio(_audio())

    # Memory is profiled as served, but there is no hash to store anything under.
    assert model.perception.forwards == [((2, 16), True)]
    assert ct._store == {}


def test_items_that_opt_out_take_their_hash_but_store_nothing(encoder):
    pytest.importorskip("vllm")
    model = _model_with_fake_perception()

    ct._begin_step(["hash-a", "hash-b", "hash-c", "hash-d"])
    model._process_audio(_audio([False, True]))
    model._process_audio(_audio([False, False]))
    ct._end_step()

    assert [capture for _, capture in model.perception.forwards] == [True, False]
    # Rows take the step's hashes by position, so an opted-out row still takes one.
    assert list(ct._store) == ["hash-b"] and float(ct._store["hash-b"]["duration"]) == 8 / 16000


def test_processor_marks_the_audio_of_requests_that_opt_into_capture():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

    class _Tokenizer:
        def get_vocab(self):
            return {"<|audio|>": 0}

        def encode(self, prompt, add_special_tokens=True):
            return [0] * len(prompt.split())

    processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_tokenizer=_Tokenizer,
        _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: 2,
        _get_encoder_chunk_size_seconds=lambda: None,
        _get_audio_token_estimator_config=lambda: None,
    )

    def capture_flags(mm_kwargs):
        result = processor._call_hf_processor(
            prompt="<|audio|> <|audio|>",
            mm_data={"audios": [[0.0] * 8, [0.0] * 4]},
            mm_kwargs=mm_kwargs,
            tok_kwargs={},
        )
        return result["capture_ctc_timestamps"].tolist()

    assert capture_flags({}) == [False, False]
    assert capture_flags({"capture_ctc_timestamps": True}) == [True, True]
    # Read on the host while encoding, where a device copy would need a sync.
    assert processor._get_mm_fields_config(None, {})["capture_ctc_timestamps"].field.keep_on_cpu


def test_startup_profiling_inputs_opt_into_capture(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder

    from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMDummyInputsBuilder

    monkeypatch.setattr(
        BaseDummyInputsBuilder,
        "get_dummy_processor_inputs",
        lambda self, seq_len, mm_counts, mm_options: SimpleNamespace(hf_processor_mm_kwargs={}),
    )
    builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)

    inputs = builder.get_dummy_processor_inputs(64, {"audio": 1}, {})

    assert inputs.hf_processor_mm_kwargs == {"capture_ctc_timestamps": True}


def test_transcripts_without_speaker_tags_are_reported(encoder, caplog):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    _align_one("req", "hello world", release=True)

    assert "no <spk:N> speaker tags" in caplog.text


def test_requests_without_recorded_audio_are_not_reported_as_evicted(encoder, caplog):
    assert ct.align_finished_requests([("text-only", "<spk:0> hello")], release=True) == [ct._empty_result()]
    assert "No capture is recorded" in caplog.text and "evicted" not in caplog.text

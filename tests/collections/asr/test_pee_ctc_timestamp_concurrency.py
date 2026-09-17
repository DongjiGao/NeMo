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

"""Per-request CTC timestamp capture under continuous batching.

``capture_ctc_timestamps`` scopes one generation call and holds a single
``outputs`` slot, which ``_store_ctc_timestamp_outputs`` overwrites on every
forward. A vLLM server calls the encoder repeatedly with interleaved requests, so
that slot loses every batch but the last, and the nesting guard rejects a
concurrent scope outright. These tests pin the per-request replacement.

The storage path only touches ``self.__dict__``, so the tests drive the real
methods on an uninitialized instance rather than building a 630M-parameter
encoder.
"""

import pytest
import torch

from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

# @experimental wraps the class in a wrapt proxy, whose __new__ is the proxy's, not
# the class's. Unwrap to reach the real type.
_ParallelExpertEncoder = getattr(ParallelExpertEncoder, "__wrapped__", ParallelExpertEncoder)


def _encoder():
    """Real class, no __init__, so the dict-backed capture paths are exercised."""
    return _ParallelExpertEncoder.__new__(_ParallelExpertEncoder)


def _batch(batch_size: int, frames: int = 4, vocab: int = 6, speakers: int = 2):
    """Shapes mirroring _store_ctc_timestamp_outputs: (B,T,V), (B,), (B,T,S), (B,)."""
    log_probs = torch.arange(batch_size * frames * vocab, dtype=torch.float32)
    log_probs = log_probs.reshape(batch_size, frames, vocab)
    sigmoids = torch.arange(batch_size * frames * speakers, dtype=torch.float32)
    sigmoids = sigmoids.reshape(batch_size, frames, speakers)
    lengths = torch.full((batch_size,), frames, dtype=torch.int64)
    return log_probs, lengths, sigmoids, lengths.clone()


class TestPerRequestCapture:
    def test_interleaved_requests_do_not_clobber(self):
        """The defect: a later forward used to overwrite an earlier request's rows."""
        enc = _encoder()
        first_lp, first_len, first_sig, first_slen = _batch(1)
        with enc.ctc_timestamp_request_rows(["req-a"]):
            enc._store_ctc_timestamp_outputs(first_lp, first_len, first_sig, first_slen)

        second_lp, second_len, second_sig, second_slen = _batch(1)
        second_lp = second_lp + 1000.0
        with enc.ctc_timestamp_request_rows(["req-b"]):
            enc._store_ctc_timestamp_outputs(second_lp, second_len, second_sig, second_slen)

        assert enc.has_ctc_timestamps("req-a")
        assert enc.has_ctc_timestamps("req-b")
        store = enc.__dict__["_ctc_timestamp_request_store"]
        torch.testing.assert_close(store["req-a"]["ctc_log_probs"], first_lp)
        torch.testing.assert_close(store["req-b"]["ctc_log_probs"], second_lp)

    def test_batch_is_split_per_row(self):
        enc = _encoder()
        log_probs, lengths, sigmoids, spk_lengths = _batch(3)
        with enc.ctc_timestamp_request_rows(["r0", "r1", "r2"]):
            enc._store_ctc_timestamp_outputs(log_probs, lengths, sigmoids, spk_lengths)

        store = enc.__dict__["_ctc_timestamp_request_store"]
        assert sorted(store) == ["r0", "r1", "r2"]
        for row, request_id in enumerate(["r0", "r1", "r2"]):
            got = store[request_id]["ctc_log_probs"]
            assert got.shape == (1, 4, 6)
            torch.testing.assert_close(got, log_probs[row : row + 1])

    def test_none_rows_are_not_stored(self):
        """A batch may mix timestamp and plain requests; plain rows cost nothing."""
        enc = _encoder()
        log_probs, lengths, sigmoids, spk_lengths = _batch(3)
        with enc.ctc_timestamp_request_rows(["keep", None, "also-keep"]):
            enc._store_ctc_timestamp_outputs(log_probs, lengths, sigmoids, spk_lengths)

        store = enc.__dict__["_ctc_timestamp_request_store"]
        assert sorted(store) == ["also-keep", "keep"]
        torch.testing.assert_close(store["also-keep"]["ctc_log_probs"], log_probs[2:3])

    def test_rows_accumulate_across_forwards(self):
        """Audio spanning several forwards must append, not replace."""
        enc = _encoder()
        chunk_one = _batch(1)
        chunk_two = _batch(1)
        for chunk in (chunk_one, chunk_two):
            with enc.ctc_timestamp_request_rows(["long"]):
                enc._store_ctc_timestamp_outputs(*chunk)

        stored = enc.__dict__["_ctc_timestamp_request_store"]["long"]
        assert stored["ctc_log_probs"].shape == (2, 4, 6)
        assert stored["ctc_lengths"].shape == (2,)

    def test_row_count_mismatch_is_rejected(self):
        enc = _encoder()
        log_probs, lengths, sigmoids, spk_lengths = _batch(2)
        with enc.ctc_timestamp_request_rows(["only-one"]):
            with pytest.raises(ValueError, match="declared 1 rows but the batch has 2"):
                enc._store_ctc_timestamp_outputs(log_probs, lengths, sigmoids, spk_lengths)

    def test_no_mapping_means_no_per_request_storage(self):
        """Without a declared mapping the legacy capture path stays in charge."""
        enc = _encoder()
        enc._store_ctc_timestamp_outputs(*_batch(1))
        assert "_ctc_timestamp_request_store" not in enc.__dict__

    def test_mapping_is_restored_after_nesting(self):
        enc = _encoder()
        with enc.ctc_timestamp_request_rows(["outer"]):
            with enc.ctc_timestamp_request_rows(["inner"]):
                assert enc.__dict__["_ctc_timestamp_row_ids"] == ["inner"]
            assert enc.__dict__["_ctc_timestamp_row_ids"] == ["outer"]
        assert "_ctc_timestamp_row_ids" not in enc.__dict__

    def test_discard_releases_rows(self):
        """Abort or disconnect must free rows, which are held until taken."""
        enc = _encoder()
        with enc.ctc_timestamp_request_rows(["doomed"]):
            enc._store_ctc_timestamp_outputs(*_batch(1))
        assert enc.has_ctc_timestamps("doomed")
        enc.discard_ctc_timestamps("doomed")
        assert not enc.has_ctc_timestamps("doomed")
        enc.discard_ctc_timestamps("never-existed")

    def test_take_without_capture_raises(self):
        enc = _encoder()
        with pytest.raises(RuntimeError, match="No CTC outputs were captured for request"):
            enc.take_ctc_timestamps("ghost", sot_transcripts=["hi"], audio_durations=[1.0])

    def test_take_aligns_and_releases(self):
        enc = _encoder()
        with enc.ctc_timestamp_request_rows(["req"]):
            enc._store_ctc_timestamp_outputs(*_batch(1))

        seen = {}

        class _Extractor:
            def extract_from_outputs_batch(self, **kwargs):
                seen.update(kwargs)
                return [{"word": "ok"}]

        enc.__dict__["_ctc_timestamp_extractor_cache"] = ("/fake.nemo", _Extractor())
        result = enc.take_ctc_timestamps("req", sot_transcripts=["ok"], audio_durations=[1.0])

        assert result == [{"word": "ok"}]
        assert seen["sot_transcripts"] == ["ok"]
        assert seen["ctc_log_probs"].shape == (1, 4, 6)
        assert not enc.has_ctc_timestamps("req"), "take must release the rows"


class TestServeGate:
    def test_serve_is_reentrant_unlike_capture(self):
        """capture_ctc_timestamps raises on nesting; serving must not."""
        enc = _encoder()
        enc.__dict__["_ctc_timestamp_extractor_cache"] = ("/fake.nemo", object())
        enc.__dict__["_ctc_timestamp_serve_depth"] = 1
        with enc.serve_ctc_timestamps(torch.device("cpu")):
            assert enc.__dict__["_ctc_timestamp_serve_depth"] == 2
        assert enc.__dict__["_ctc_timestamp_serve_depth"] == 1

    def test_decoder_gate_off_when_nothing_active(self):
        enc = _encoder()
        assert enc._ctc_timestamp_decoder() is None

    def test_decoder_gate_on_while_serving(self):
        enc = _encoder()
        sentinel = object()

        class _Extractor:
            ctc_decoder = sentinel

        enc.__dict__["_ctc_timestamp_extractor_cache"] = ("/fake.nemo", _Extractor())
        enc.__dict__["_ctc_timestamp_serve_depth"] = 1
        assert enc._ctc_timestamp_decoder() is sentinel

    def test_serve_exit_drops_pending_rows(self, monkeypatch):
        """Entering at depth 0 warms the adapter, so stub the loader out."""
        import nemo.collections.asr.modules.parallel_expert_encoder as pee

        monkeypatch.setattr(pee, "_get_ctc_timestamp_extractor", lambda *a, **k: object())
        enc = _encoder()
        enc.__dict__["ctc_timestamp_model_path"] = "/fake.nemo"
        with enc.serve_ctc_timestamps(torch.device("cpu")):
            with enc.ctc_timestamp_request_rows(["leaked"]):
                enc._store_ctc_timestamp_outputs(*_batch(1))
            assert enc.has_ctc_timestamps("leaked")
        assert "_ctc_timestamp_request_store" not in enc.__dict__

    def test_serve_warms_the_adapter_once(self, monkeypatch):
        """A bad path must fail at startup, not on the first request mid-serve."""
        import nemo.collections.asr.modules.parallel_expert_encoder as pee

        calls = []
        monkeypatch.setattr(
            pee, "_get_ctc_timestamp_extractor", lambda *a, **k: calls.append(1) or object()
        )
        enc = _encoder()
        enc.__dict__["ctc_timestamp_model_path"] = "/fake.nemo"
        with enc.serve_ctc_timestamps(torch.device("cpu")):
            with enc.serve_ctc_timestamps(torch.device("cpu")):
                pass
        assert len(calls) == 1, "reentry must not reload the adapter"

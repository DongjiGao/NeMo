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
import os
import random

import pytest
import torch

try:
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate
    from vllm.v1.sample.logits_processor.interface import MoveDirectionality

    from nemo.collections.speechlm2.vllm.salm.loop_guard import (
        LoopGuardConfig,
        LoopGuardLogitsProcessor,
        request_overrides,
    )

    _HAS_LOOP_GUARD = True
except ImportError:
    _HAS_LOOP_GUARD = False

pytestmark = pytest.mark.skipif(not _HAS_LOOP_GUARD, reason="vLLM not available")

EOS = 11
VOCAB = 64
PREFIX = [20, 21, 22, 23, 24, 25, 26, 27]
CYCLE = [30, 31, 32, 33]


@pytest.fixture(autouse=True)
def _no_env_defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEMO_LOOP_GUARD"):
            monkeypatch.delenv(name)


def _params(**loop_guard) -> "SamplingParams":
    extra = {f"loop_guard_{key}": value for key, value in loop_guard.items()}
    return SamplingParams(stop_token_ids=[EOS], extra_args=extra or None)


class _Batch:
    """Drives the processor the way vLLM's persistent batch does: batch changes, then one step."""

    def __init__(self):
        self.proc = LoopGuardLogitsProcessor(None, torch.device("cpu"), False)
        self.outs: dict[int, list[int]] = {}
        self.added, self.removed, self.moved = [], [], []

    def add(self, index: int, params: "SamplingParams") -> None:
        self.outs[index] = []
        self.added.append((index, params, None, self.outs[index]))

    def remove(self, index: int) -> None:
        del self.outs[index]
        self.removed.append(index)

    def swap(self, a: int, b: int) -> None:
        self.outs[a], self.outs[b] = self.outs[b], self.outs[a]
        self.moved.append((a, b, MoveDirectionality.SWAP))

    def move(self, a: int, b: int) -> None:
        self.outs[b] = self.outs.pop(a)
        self.moved.append((a, b, MoveDirectionality.UNIDIRECTIONAL))

    def step(self, tokens: dict[int, int]) -> torch.Tensor:
        for index, token in tokens.items():
            self.outs[index].append(token)
        update = None
        if self.added or self.removed or self.moved:
            update = BatchUpdate(len(self.outs), self.removed, self.added, self.moved)
            self.added, self.removed, self.moved = [], [], []
        self.proc.update_state(update)
        return self.proc.apply(torch.zeros(max(self.outs) + 1, VOCAB))


def _banned(logits: torch.Tensor, index: int) -> list[int]:
    return torch.isinf(logits[index]).nonzero().flatten().tolist()


def _feed(batch: _Batch, index: int, tokens: list[int]) -> list[tuple[int, list[int]]]:
    """Feed tokens one per step; return (output length, banned tokens) for every step with bans."""
    acted = []
    for token in tokens:
        banned = _banned(batch.step({index: token}), index)
        if banned:
            acted.append((len(batch.outs[index]), banned))
    return acted


def test_off_by_default():
    batch = _Batch()
    batch.add(0, _params())
    assert _feed(batch, 0, PREFIX + CYCLE * 40) == []


def test_short_repeats_are_left_alone():
    batch = _Batch()
    batch.add(0, _params(mode="break"))
    assert _feed(batch, 0, PREFIX + [40] * 10 + PREFIX + [41, 42] * 12 + PREFIX) == []


def test_break_bans_the_next_cycle_token_once_the_loop_spans_min_span():
    batch = _Batch()
    batch.add(0, _params(mode="break"))
    acted = _feed(batch, 0, PREFIX + CYCLE * 20)
    length, banned = acted[0]
    assert length == len(PREFIX) + 64
    assert banned == [CYCLE[(length - len(PREFIX)) % len(CYCLE)]]


def test_stop_allows_only_stop_tokens():
    batch = _Batch()
    batch.add(0, _params(mode="stop"))
    acted = _feed(batch, 0, PREFIX + CYCLE * 20)
    assert acted[0][0] == len(PREFIX) + 64
    assert acted[0][1] == [t for t in range(VOCAB) if t != EOS]


def test_breaks_are_limited_then_the_request_is_stopped():
    batch = _Batch()
    batch.add(0, _params(mode="break", max_interventions=2))
    acted = _feed(batch, 0, PREFIX + CYCLE * 20)
    assert [len(banned) for _, banned in acted[:2]] == [1, 1]
    assert acted[2][1] == [t for t in range(VOCAB) if t != EOS]


def test_break_window_bans_recent_ngram_continuations():
    batch = _Batch()
    batch.add(0, _params(mode="break", break_window=8, break_ngram=3, max_interventions=50))
    acted = _feed(batch, 0, PREFIX + CYCLE * 20)
    length, banned = acted[0]
    assert banned == [CYCLE[(length - len(PREFIX)) % len(CYCLE)]]
    window = [bans for step_length, bans in acted if length < step_length <= length + 8]
    assert window and all(token in CYCLE for bans in window for token in bans)


def test_state_follows_requests_through_swaps_moves_and_slot_reuse():
    rng = random.Random(0)
    batch = _Batch()
    batch.add(0, _params(mode="break"))
    batch.add(1, _params(mode="break"))
    loop = iter(PREFIX + CYCLE * 30)
    for _ in range(30):
        batch.step({0: next(loop), 1: rng.randrange(34, VOCAB)})
    batch.swap(0, 1)
    bans = {0: 0, 1: 0}
    for _ in range(60):
        logits = batch.step({1: next(loop), 0: rng.randrange(34, VOCAB)})
        bans[0] += bool(_banned(logits, 0))
        bans[1] += bool(_banned(logits, 1))
    assert bans[0] == 0 and bans[1] > 0

    batch.remove(0)
    batch.move(1, 0)
    logits = batch.step({0: next(loop)})
    assert _banned(logits, 0)

    batch.remove(0)
    batch.add(0, _params(mode="break"))
    logits = batch.step({0: rng.randrange(34, VOCAB)})
    assert not _banned(logits, 0)


def test_request_overrides_and_validation():
    assert request_overrides({"loop_guard_mode": "break", "loop_guard_min_span": 32, "other": 1}) == {
        "mode": "break",
        "min_span": 32,
    }
    assert LoopGuardConfig().with_overrides({"mode": "stop"}).mode == "stop"
    with pytest.raises(ValueError):
        LoopGuardLogitsProcessor.validate_params(_params(min_spam=32))
    with pytest.raises(ValueError):
        LoopGuardLogitsProcessor.validate_params(_params(mode="loop"))
    with pytest.raises(ValueError):
        LoopGuardConfig(break_ngram=8, break_history=4).validated()

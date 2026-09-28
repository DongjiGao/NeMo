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
"""Loop guard for greedy ASR decoding, as a vLLM logits processor.

Greedy ASR decoding occasionally falls into a loop: an exact repetition of a block of ``p`` tokens,
back to back, that does not end until the token limit. On long outputs (long-form chunks,
multi-speaker sessions) such a loop can start early and replace most of the transcript, while
normal transcripts only repeat short spans (for example "no no no"). The guard tracks, per request
and incrementally per token, how long the most recent tokens have been periodic for every ``p`` up
to ``max_period``. Once a stretch of at least ``min_span`` tokens holds at least ``min_repeats``
copies of one block, it acts on the next step:

* ``break``: ban the token that would continue the cycle, so decoding takes its next best token
  and can resume the transcript. Each break also opens a window of ``break_window`` steps (0
  disables it) in which any token that would recreate a ``break_ngram``-gram from the last
  ``break_history`` tokens is banned, so the decoder cannot slide back into the copies still in
  its context. After the first break a loop is re-detected at ``rearm_span`` tokens (``min_span``
  if 0), and after ``max_interventions`` breaks the request is ended.
* ``stop``: end the request at once (every token except the request's stop tokens is banned).

A request that never loops is never touched, so clean outputs are unchanged. vLLM's own
``repetition_detection`` can only end a looping request, which drops the rest of its transcript.

Enable with ``--logits-processors
nemo.collections.speechlm2.vllm.salm.loop_guard:LoopGuardLogitsProcessor`` (``logits_processors=``
offline). vLLM refuses to load custom logits processors when speculative decoding (MTP, DFlash) is
enabled, so the guard cannot be combined with it.

The engine-wide defaults come from ``NEMO_LOOP_GUARD`` (mode, default ``off``) and
``NEMO_LOOP_GUARD_<SETTING>`` for every other field of :class:`LoopGuardConfig` (for example
``NEMO_LOOP_GUARD_BREAK_WINDOW``). A request overrides them with flat ``loop_guard_<setting>`` keys
in ``SamplingParams.extra_args``, which the OpenAI server fills from ``vllm_xargs``:
``extra_body={"vllm_xargs": {"loop_guard_mode": "break", "loop_guard_min_span": 32}}``. The keys are
flat because ``vllm_xargs`` only carries scalars and lists.
"""

import dataclasses
import os
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.interface import MoveDirectionality

from nemo.utils import logging

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig

MODES = ("off", "break", "stop")
REQUEST_KEY_PREFIX = "loop_guard_"


def request_overrides(extra_args: dict | None) -> dict:
    """``loop_guard_<setting>`` keys of a request's extra_args, keyed by setting name."""
    return {
        key[len(REQUEST_KEY_PREFIX) :]: value
        for key, value in (extra_args or {}).items()
        if key.startswith(REQUEST_KEY_PREFIX)
    }


@dataclasses.dataclass(frozen=True)
class LoopGuardConfig:
    """Loop guard settings for one request.

    Attributes:
        mode: ``off``, ``break`` or ``stop``.
        min_span: Periodic stretch length, in tokens, that counts as a loop.
        min_repeats: Minimum number of copies of the block within that stretch.
        max_period: Longest block, in tokens, that is tracked.
        max_interventions: Breaks allowed per request before it is ended.
        break_window: Steps after each break during which recent n-gram continuations are banned
            (0 bans only the next token of the cycle).
        break_ngram: n of the n-grams banned inside the break window.
        break_history: How many recent output tokens the break window searches.
        rearm_span: Loop length that triggers the next break after the first (0: ``min_span``).
    """

    mode: str = "off"
    min_span: int = 64
    min_repeats: int = 3
    max_period: int = 200
    max_interventions: int = 8
    break_window: int = 32
    break_ngram: int = 4
    break_history: int = 512
    rearm_span: int = 32

    @classmethod
    def from_env(cls) -> "LoopGuardConfig":
        """Engine-wide defaults from ``NEMO_LOOP_GUARD`` and ``NEMO_LOOP_GUARD_<SETTING>`` variables."""

        def read(name: str, default: Any, cast: Callable[[str], Any]) -> Any:
            value = os.environ.get(name)
            return default if value in (None, "") else cast(value)

        return cls(
            mode=read("NEMO_LOOP_GUARD", cls.mode, str),
            min_span=read("NEMO_LOOP_GUARD_MIN_SPAN", cls.min_span, int),
            min_repeats=read("NEMO_LOOP_GUARD_MIN_REPEATS", cls.min_repeats, int),
            max_period=read("NEMO_LOOP_GUARD_MAX_PERIOD", cls.max_period, int),
            max_interventions=read("NEMO_LOOP_GUARD_MAX_INTERVENTIONS", cls.max_interventions, int),
            break_window=read("NEMO_LOOP_GUARD_BREAK_WINDOW", cls.break_window, int),
            break_ngram=read("NEMO_LOOP_GUARD_BREAK_NGRAM", cls.break_ngram, int),
            break_history=read("NEMO_LOOP_GUARD_BREAK_HISTORY", cls.break_history, int),
            rearm_span=read("NEMO_LOOP_GUARD_REARM_SPAN", cls.rearm_span, int),
        ).validated()

    def validated(self) -> "LoopGuardConfig":
        """Return self, raising ``ValueError`` if any setting is out of range."""
        if self.mode not in MODES:
            raise ValueError(f"loop_guard mode must be one of {MODES}, got {self.mode!r}")
        if (
            self.min_repeats < 2
            or self.max_period < 1
            or self.min_span < 2
            or self.max_interventions < 0
            or self.break_window < 0
            or self.break_ngram < 2
            or self.break_history < self.break_ngram
            or self.rearm_span < 0
        ):
            raise ValueError(f"invalid loop_guard settings: {self}")
        return self

    def with_overrides(self, overrides: dict | None) -> "LoopGuardConfig":
        """Copy with the given settings replaced; unknown or out-of-range settings raise ``ValueError``."""
        if not overrides:
            return self
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"unknown loop_guard settings: {sorted(unknown)}")
        return dataclasses.replace(self, **overrides).validated()


class _RequestLoopState:
    """Periodicity of one request's generated tokens, updated as tokens arrive.

    ``ring[k % max_period]`` holds output token ``k``; ``runs[p - 1]`` counts how many consecutive
    recent tokens equal the token ``p`` positions earlier, so a loop of period ``p`` spans
    ``runs[p - 1] + p`` tokens.
    """

    __slots__ = (
        "cfg",
        "out",
        "stop_ids",
        "periods",
        "ring",
        "runs",
        "seen",
        "breaks",
        "stopped",
        "bans",
        "window_left",
    )

    def __init__(self, cfg: LoopGuardConfig, output_tok_ids: list[int], stop_ids: set[int]) -> None:
        self.cfg = cfg
        self.out = output_tok_ids
        self.stop_ids = sorted(stop_ids)
        self.periods = np.arange(1, cfg.max_period + 1)
        self.ring = np.full(cfg.max_period, -1, dtype=np.int64)
        self.runs = np.zeros(cfg.max_period, dtype=np.int64)
        self.seen = 0
        self.breaks = 0
        self.stopped = False
        self.bans: list[int] = []
        self.window_left = 0

    def advance(self) -> None:
        """Consume new output tokens and decide what to do on the next step."""
        width = self.cfg.max_period
        self.bans = []
        new_tokens = len(self.out) - self.seen
        while self.seen < len(self.out):
            token = self.out[self.seen]
            earlier = self.ring[(self.seen - self.periods) % width]
            self.runs = np.where(earlier == token, self.runs + 1, 0)
            self.ring[self.seen % width] = token
            self.seen += 1
        self.window_left = max(0, self.window_left - new_tokens)
        threshold = self.cfg.rearm_span if self.breaks and self.cfg.rearm_span else self.cfg.min_span
        span = self.runs + self.periods
        looping = (span >= threshold) & (span >= self.cfg.min_repeats * self.periods)
        if looping.any():
            if self.cfg.mode == "stop" or self.breaks >= self.cfg.max_interventions:
                self.stopped = True
                return
            period = int(self.periods[looping][0])
            self.breaks += 1
            self.bans.append(int(self.ring[(self.seen - period) % width]))
            self.window_left = self.cfg.break_window
        if self.window_left > 0:
            self.bans.extend(self._repeat_continuations())

    def _repeat_continuations(self) -> list[int]:
        """Tokens that would extend the current (n-1)-token suffix into an n-gram seen recently."""
        k = self.cfg.break_ngram - 1
        recent = np.asarray(self.out[-self.cfg.break_history :], dtype=np.int64)
        if len(recent) <= k:
            return []
        suffix = recent[-k:]
        match = np.ones(len(recent) - k, dtype=bool)
        for j in range(k):
            match &= recent[j : len(recent) - k + j] == suffix[j]
        return np.unique(recent[np.flatnonzero(match) + k]).tolist()


class LoopGuardLogitsProcessor(LogitsProcessor):
    """Break or stop exact-repetition loops; leaves requests that never loop untouched."""

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Reject a request whose ``loop_guard_<setting>`` extra args are unknown or out of range."""
        overrides = request_overrides(sampling_params.extra_args)
        if overrides:
            LoopGuardConfig.from_env().with_overrides(overrides)

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool) -> None:
        """Read the engine-wide defaults; per-request state is created as requests join the batch."""
        self.default = LoopGuardConfig.from_env()
        self.device = device
        self.states: dict[int, _RequestLoopState] = {}
        self.guarded_requests = 0
        logging.info(f"Loop guard loaded: {self.default}")

    def is_argmax_invariant(self) -> bool:
        """Banning tokens can change the greedy choice, so vLLM must apply it under greedy decoding too."""
        return False

    def _new_state(self, params: "SamplingParams", output_tok_ids: list[int]) -> _RequestLoopState | None:
        cfg = self.default.with_overrides(request_overrides(params.extra_args))
        if cfg.mode == "off":
            return None
        return _RequestLoopState(cfg, output_tok_ids, set(params.all_stop_token_ids))

    def _retire(self, state: _RequestLoopState | None) -> None:
        if state is not None and (state.breaks or state.stopped):
            self.guarded_requests += 1
            logging.info(
                f"[loop-guard] request finished after {state.breaks} break(s)"
                f"{', then stopped' if state.stopped else ''} at {state.seen} output tokens "
                f"({self.guarded_requests} guarded so far)"
            )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """Track batch membership, then consume each guarded request's new tokens.

        Changes are applied in the order the vLLM interface documents (removed, added, moved), so a
        slot that is freed and reused in one update ends up holding the new request's state.
        """
        if batch_update is not None:
            for index in batch_update.removed:
                self._retire(self.states.pop(index, None))
            for index, params, _prompt_tok_ids, output_tok_ids in batch_update.added:
                self._retire(self.states.pop(index, None))
                state = self._new_state(params, output_tok_ids)
                if state is not None:
                    self.states[index] = state
            for a_index, b_index, direction in batch_update.moved:
                a_state = self.states.pop(a_index, None)
                b_state = self.states.pop(b_index, None)
                if a_state is not None:
                    self.states[b_index] = a_state
                if b_state is not None:
                    if direction == MoveDirectionality.SWAP:
                        self.states[a_index] = b_state
                    else:
                        self._retire(b_state)
        for state in self.states.values():
            state.advance()

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Ban each looping request's cycle and window tokens, or all but its stop tokens once stopped."""
        for index, state in self.states.items():
            if state.stopped and state.stop_ids:
                allowed = logits[index, state.stop_ids].clone()
                logits[index] = float("-inf")
                logits[index, state.stop_ids] = allowed
            elif state.bans:
                logits[index, state.bans] = float("-inf")
        return logits

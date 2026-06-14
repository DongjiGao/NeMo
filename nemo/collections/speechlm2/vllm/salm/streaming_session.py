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

"""Client-side driver for NeMo StreamingSTT decoding over a vLLM session.

Wraps vLLM's resumable streaming-session API (one ``StreamingInput`` per audio
chunk turn) into a reusable :class:`StreamingSTTSession`. For each chunk the
session sends ``<user header> <audio> <assistant header>`` with one audio item
carrying that chunk's precomputed embeddings (see :func:`stream_encode`), lets
the LLM decode until a stop blank / EOS, and -- together with the
``register()``-installed scheduler patch -- carries the chunk's text + blank +
footer into the next chunk's KV. Generated token streams are decoded to text by
splitting on the blank (segment) token, mirroring NeMo's ``decode_with_blank``.

The chunk-boundary marker token IDs and turn-template token IDs are carried in a
:class:`StreamingMarkers`, which can be built from a markers dict / JSON file or
from the model's HF config.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any

# The opt-in extra_args contract keys are owned by the scheduler patch; import
# them so the producer (this file) and the consumer (streaming_scheduler) share
# one source of truth and cannot silently drift.
from nemo.collections.speechlm2.vllm.salm.streaming_scheduler import (
    BLANK_ID_KEY,
    EOS_ID_KEY,
    FOOTER_IDS_KEY,
    RETAIN_FLAG,
)


@dataclass
class StreamingMarkers:
    """Token IDs and turn-template pieces that define a StreamingSTT turn.

    Attributes:
        chunk_size: Encoder output frames per chunk (audio placeholders expand to
            this count).
        blank_token_id: The ``<blank>`` segment-boundary token.
        eos_id: End-of-turn token (e.g. ``<|im_end|>``).
        audio_id: The audio locator token a chunk's embeddings replace.
        asst_footer_ids: Assistant turn footer (e.g. ``[<|im_end|>, \\n]``).
        user_header_ids: User turn header (e.g. ``[<|im_start|>, user, \\n]``).
        uf_ah_ids: User-footer + assistant-header tokens that follow the audio.
        system_prompt: System prompt rendered (via chat template) on chunk 0.
        has_blank: Whether the checkpoint uses a dedicated blank token.
    """

    chunk_size: int
    blank_token_id: int
    eos_id: int
    audio_id: int
    asst_footer_ids: list[int]
    user_header_ids: list[int]
    uf_ah_ids: list[int]
    system_prompt: str = "Transcribe the audio into text."
    has_blank: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "StreamingMarkers":
        """Build markers from a markers dict (e.g. loaded from ``markers.json``)."""
        return cls(
            chunk_size=int(data["chunk_size"]),
            blank_token_id=int(data["blank_token_id"]),
            eos_id=int(data["eos_id"]),
            audio_id=int(data.get("audio_id", 151670)),
            asst_footer_ids=list(data["asst_footer_ids"]),
            user_header_ids=list(data["user_header_ids"]),
            uf_ah_ids=list(data["uf_ah_ids"]),
            system_prompt=data.get("system_prompt", "Transcribe the audio into text."),
            has_blank=bool(data.get("has_blank", True)),
        )

    @classmethod
    def from_config(cls, hf_config: Any) -> "StreamingMarkers":
        """Build markers from a model's HF config ``streaming_markers`` field.

        Lets a checkpoint be self-describing (no external markers file). Raises if
        the config does not carry ``streaming_markers``.
        """
        data = getattr(hf_config, "streaming_markers", None)
        if not data:
            raise ValueError(
                "Model config has no 'streaming_markers'; export the checkpoint with "
                "streaming markers or pass an explicit markers dict via from_dict()."
            )
        return cls.from_dict(data)


class StreamingSTTSession:
    """Decode StreamingSTT utterances over vLLM resumable streaming sessions.

    One :class:`StreamingSTTSession` is reused across utterances; each call to
    :meth:`transcribe` opens an independent vLLM session (its own ``request_id``
    and KV cache) and feeds the utterance's per-chunk embeddings one chunk turn
    at a time.

    Args:
        engine: A vLLM ``AsyncLLM`` built with ``enable_mm_embeds=True`` and (for
            the resumable session API) ``async_scheduling=False``.
        tokenizer: HF tokenizer for the model (used for the chunk-0 system-prompt
            chat-template render and for detokenizing output segments).
        markers: Token IDs / turn-template pieces (:class:`StreamingMarkers`).
        max_tokens: Per-chunk ``max_tokens`` (NeMo uses 64 per chunk turn).
    """

    def __init__(self, engine: Any, tokenizer: Any, markers: StreamingMarkers, *, max_tokens: int = 64):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        self.engine = engine
        self.tokenizer = tokenizer
        self.markers = markers

        self.sys_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": markers.system_prompt}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        # One audio placeholder per chunk: the SALM multimodal processor expands
        # it to the embedding frame count. Emitting chunk_size literal locators
        # instead double-expands and corrupts the audio context.
        self.audio_ph = [markers.audio_id]

        extra_args = {
            RETAIN_FLAG: True,
            BLANK_ID_KEY: markers.blank_token_id,
            EOS_ID_KEY: markers.eos_id,
            FOOTER_IDS_KEY: list(markers.asst_footer_ids),
        }
        self._chunk_sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop_token_ids=[markers.blank_token_id, markers.eos_id],
            output_kind=RequestOutputKind.DELTA,
            extra_args=extra_args,
        )

    def _chunk_wrapper(self, chunk_index: int) -> list[int]:
        """Per-chunk turn token IDs: (system on chunk 0) + user header + audio + asst header."""
        prefix = list(self.sys_ids) if chunk_index == 0 else []
        return prefix + self.markers.user_header_ids + self.audio_ph + self.markers.uf_ah_ids

    async def _run_session(self, chunks: list[Any], request_id: str, throttle: bool) -> list[int]:
        """Run one utterance's streaming session; return the flat generated token IDs."""
        from vllm.engine.protocol import StreamingInput
        from vllm.inputs import TokensPrompt

        blank_id, eos_id = self.markers.blank_token_id, self.markers.eos_id

        def _stream_input(chunk_index: int) -> Any:
            return StreamingInput(
                prompt=TokensPrompt(
                    prompt_token_ids=self._chunk_wrapper(chunk_index),
                    multi_modal_data={"audio": [chunks[chunk_index]]},
                ),
                sampling_params=self._chunk_sp,
            )

        if throttle:
            # One chunk in flight, gated on its output -- useful for debugging /
            # strict per-chunk inspection.
            done: asyncio.Queue = asyncio.Queue()

            async def gen_throttled():
                for i in range(len(chunks)):
                    yield _stream_input(i)
                    await done.get()

            per_chunk: list[list[int]] = []
            cur: list[int] = []
            async for out in self.engine.generate(gen_throttled(), self._chunk_sp, request_id=request_id):
                d = out.outputs[0]
                cur.extend(list(d.token_ids))
                if d.finish_reason is not None or (cur and cur[-1] in (blank_id, eos_id)):
                    per_chunk.append(cur)
                    cur = []
                    await done.put(1)
                if len(per_chunk) >= len(chunks):
                    break
            return [t for group in per_chunk for t in group]

        # Throughput mode: eager-yield all chunk turns; the engine queues and
        # advances the session internally. Drain to the end.
        async def gen_throughput():
            for i in range(len(chunks)):
                yield _stream_input(i)

        flat: list[int] = []
        async for out in self.engine.generate(gen_throughput(), self._chunk_sp, request_id=request_id):
            flat.extend(list(out.outputs[0].token_ids))
            if out.finished:
                break
        return flat

    def decode(self, token_ids: list[int]) -> str:
        """Decode generated token IDs to text by splitting on the blank token.

        Mirrors NeMo ``decode_with_blank``: the blank token separates segments
        (decoded independently to preserve BPE within a turn), EOS is dropped, and
        segments are joined with single spaces.
        """
        blank_id, eos_id = self.markers.blank_token_id, self.markers.eos_id
        segments: list[str] = []
        current: list[int] = []
        for t in token_ids:
            if t == blank_id:
                if current:
                    segments.append(self.tokenizer.decode(current, skip_special_tokens=True))
                    current = []
            elif t == eos_id:
                continue
            else:
                current.append(t)
        if current:
            segments.append(self.tokenizer.decode(current, skip_special_tokens=True))
        return re.sub(r"\s+", " ", " ".join(s for s in segments if s)).strip()

    async def transcribe(self, chunks: list[Any], request_id: str, *, throttle: bool = False) -> str:
        """Transcribe one utterance from its list of per-chunk embedding tensors.

        Args:
            chunks: Per-chunk embedding tensors ``(chunk_size, hidden)`` from
                :func:`stream_encode`.
            request_id: Unique vLLM request/session id for this utterance.
            throttle: If True, gate one chunk in flight at a time (debug mode).

        Returns:
            The decoded transcript string.
        """
        flat = await self._run_session(chunks, request_id, throttle)
        return self.decode(flat)

    async def transcribe_many(
        self,
        chunk_lists: list[list[Any]],
        *,
        tag: str = "streaming_stt",
        concurrency: int = 32,
        throttle: bool = False,
    ) -> list[str]:
        """Transcribe many utterances concurrently (one vLLM session each).

        Args:
            chunk_lists: Per-utterance lists of per-chunk embedding tensors.
            tag: Request-id prefix; each utterance gets ``f"{tag}_{i}"``.
            concurrency: Max concurrent sessions (continuous batching across them).
            throttle: Per-chunk gating (debug mode).

        Returns:
            Transcripts aligned to ``chunk_lists`` order.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(i: int) -> str:
            async with sem:
                return await self.transcribe(chunk_lists[i], f"{tag}_{i}", throttle=throttle)

        return await asyncio.gather(*[_bounded(i) for i in range(len(chunk_lists))])

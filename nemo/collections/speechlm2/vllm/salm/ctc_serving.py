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

"""Serve CTC word timestamps and diarized transcripts from vLLM's transcription endpoint.

Stock vLLM's ``/v1/audio/transcriptions`` reports timestamps only for models that emit
timestamp tokens, and its model hooks receive the transcript text alone, so they cannot
reach the alignment inputs the engine captured. :func:`install_transcription_alignment`,
called from the plugin's ``register()``, wraps the endpoint for models whose config has
a ``ctc_timestamps.adapter_path``. A ``diarized_json`` request, or a ``verbose_json``
request for word timestamps, then runs vLLM's own ``json`` path, with the speaker-tag
prompt and the special tokens kept for ``diarized_json``. The finished transcript is
aligned in the engine through :func:`align_async`, the same worker method offline
callers reach, and the model's ``get_word_timestamps`` and ``parse_diarized_transcript``
hooks build the response. Every other request, and every other model, takes vLLM's
path unchanged.

A request taken over here always releases its capture: aligning does, and a request
that fails or is cancelled first releases it instead of leaving it to the byte cap.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from functools import wraps
from typing import Any

from nemo.collections.speechlm2.vllm.salm.ctc_timestamps import align_async, ctc_adapter_path, release_captures_async
from nemo.utils import logging


@dataclass
class _AlignedRequest:
    """A transcription request the wrapper takes over, filled in as vLLM's path runs."""

    response_format: str
    keep_special_tokens: bool
    request_id: str | None = None
    duration: float | None = None


# The request this asyncio task is serving while vLLM's path runs for it. Context
# variables are per task, so concurrent requests never see each other's.
_current: contextvars.ContextVar[_AlignedRequest | None] = contextvars.ContextVar(
    "nemo_ctc_aligned_request", default=None
)


def aligned_response_format() -> str | None:
    """Return the response format of the request being prepared on this task, when the endpoint aligns it.

    vLLM's prompt hook is not told the response format, so the model reads it here to
    pick the speaker-tag prompt and to keep the audio's alignment inputs.

    Returns:
        str | None: ``"diarized_json"`` or ``"verbose_json"``, or ``None`` for requests
        that are not aligned, including every request outside the endpoint.
    """
    request = _current.get()
    return None if request is None else request.response_format


def install_transcription_alignment() -> None:
    """Wrap vLLM's transcription endpoint to align the requests that need timestamps.

    Leaves the endpoint as it is, with a warning, when this vLLM lacks one of the
    methods wrapped here.
    """
    try:
        from vllm.entrypoints.speech_to_text.base import serving
        from vllm.entrypoints.speech_to_text.transcription import protocol
    except ImportError:  # pragma: no cover - vLLM without the transcription endpoint
        return

    base = serving.SpeechToTextBaseServing
    seams = (
        (base, "_create_speech_to_text"),
        (base, "_preprocess_speech_to_text"),
        (protocol.TranscriptionRequest, "to_sampling_params"),
    )
    missing = [f"{owner.__name__}.{name}" for owner, name in seams if not callable(getattr(owner, name, None))]
    if missing:
        logging.warning(
            "[NeMoSpeechLM] This vLLM's transcription endpoint lacks %s; it will not serve CTC word timestamps "
            "or diarized transcripts.",
            ", ".join(missing),
        )
        return
    if getattr(base._create_speech_to_text, "_nemo_ctc_alignment", False):
        return

    base._create_speech_to_text = _wrap_create(base._create_speech_to_text, serving, protocol)
    base._preprocess_speech_to_text = _wrap_preprocess(base._preprocess_speech_to_text)
    protocol.TranscriptionRequest.to_sampling_params = _wrap_sampling_params(
        protocol.TranscriptionRequest.to_sampling_params
    )


def _aligned_format(endpoint: Any, request: Any) -> str | None:
    """The response format to align a request for, or ``None`` to leave it to vLLM."""
    if endpoint.task_type != "transcribe":
        return None
    if not ctc_adapter_path(getattr(endpoint.model_config.hf_config, "ctc_timestamps", None)):
        return None
    if request.response_format == "diarized_json":
        return request.response_format
    if request.response_format == "verbose_json" and "word" in (request.timestamp_granularities or ()):
        return request.response_format
    return None


def _wrap_create(original: Any, serving: Any, protocol: Any) -> Any:
    @wraps(original)
    async def _create_speech_to_text(self, audio_data, request, raw_request, response_class, stream_generator_method):
        response_format = _aligned_format(self, request)
        if response_format is None:
            return await original(self, audio_data, request, raw_request, response_class, stream_generator_method)
        if request.stream or request.use_beam_search:
            return self.create_error_response(
                f"CTC timestamps for {response_format} support neither streaming nor beam search."
            )

        aligned = _AlignedRequest(
            response_format=response_format,
            keep_special_tokens=response_format == "diarized_json"
            and getattr(self.model_cls, "keep_special_tokens_for_diarization", False),
        )
        token = _current.set(aligned)
        # vLLM's own json path, which neither rejects these formats for a model without
        # timestamp tokens nor parses timestamp tokens out of the transcript.
        request.response_format = "json"
        result = None
        try:
            response = await original(self, audio_data, request, raw_request, response_class, stream_generator_method)
            if aligned.request_id is not None and isinstance(response, protocol.TranscriptionResponse):
                (result,) = await align_async(
                    self.engine_client.collective_rpc,
                    [(aligned.request_id, response.text)],
                    require_enabled=False,
                )
        finally:
            request.response_format = response_format
            _current.reset(token)
            if aligned.request_id is not None and result is None:
                await asyncio.gather(
                    release_captures_async(self.engine_client.collective_rpc, [aligned.request_id]),
                    return_exceptions=True,
                )
        if result is None:
            if isinstance(response, protocol.TranscriptionResponse):
                return self.create_error_response("CTC timestamps need the audio as one engine request.")
            return response
        return _aligned_response(self, request, response, result, aligned, serving, protocol)

    _create_speech_to_text._nemo_ctc_alignment = True
    return _create_speech_to_text


def _wrap_preprocess(original: Any) -> Any:
    @wraps(original)
    async def _preprocess_speech_to_text(self, request, audio_data, request_id):
        engine_inputs, duration, chunk_start_offsets = await original(self, request, audio_data, request_id)
        aligned = _current.get()
        if aligned is not None:
            aligned.duration = duration
            # vLLM submits a lone input under request_id itself, and the pieces of
            # chunked audio as f"{request_id}-{index}".
            aligned.request_id = request_id if len(engine_inputs) == 1 else None
        return engine_inputs, duration, chunk_start_offsets

    return _preprocess_speech_to_text


def _wrap_sampling_params(original: Any) -> Any:
    @wraps(original)
    def to_sampling_params(self, *args, **kwargs):
        params = original(self, *args, **kwargs)
        aligned = _current.get()
        if aligned is not None and aligned.keep_special_tokens:
            # The <spk:N> speaker tags are special tokens, and the aligner reads them.
            params.skip_special_tokens = False
        return params

    return to_sampling_params


def _aligned_response(
    endpoint: Any, request: Any, response: Any, result: dict, aligned: _AlignedRequest, serving: Any, protocol: Any
) -> Any:
    """Build the ``verbose_json`` or ``diarized_json`` response from an aligned ``json`` transcription."""
    model_cls = endpoint.model_cls
    if aligned.response_format == "verbose_json":
        return protocol.TranscriptionResponseVerbose(
            text=response.text,
            language=request.language or next(iter(model_cls.supported_languages), "en"),
            duration=aligned.duration,
            words=model_cls.get_word_timestamps(response.text, worker_output=result),
        )
    segments = model_cls.parse_diarized_transcript(response.text, worker_output=result)
    if not segments:
        return endpoint.create_error_response("Model output did not contain a valid diarized transcript")
    separator = serving.asr_inter_chunk_separator(request.language, model_cls.no_space_languages)
    return protocol.TranscriptionResponseDiarized(
        duration=aligned.duration,
        text=separator.join(segment.text for segment in segments),
        segments=[
            {
                "id": f"seg_{index}",
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": segment.speaker,
            }
            for index, segment in enumerate(segments)
        ],
        usage=response.usage,
    )

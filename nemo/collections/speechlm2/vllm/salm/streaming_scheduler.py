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

"""Engine-side scheduler patch for NeMo StreamingSTT vLLM sessions.

vLLM's resumable streaming-session scheduler (``_update_request_as_session``)
keeps only the *computed* output tokens of a chunk and drops the final sampled
stop token, and it never re-feeds the assistant turn footer. For NeMo
StreamingSTT that diverges from training: each per-chunk assistant turn ends as
``<text> <blank> <|im_end|>\\n`` (silent chunks emit a lone ``<blank>``), and all
of ``text``, the stop ``<blank>``, and the footer must remain in the KV history
the next chunk attends to. Without that retention, later chunks drift (e.g. a
silent chunk emits EOS instead of blank) and the transcript degrades.

This module installs a narrow monkey-patch of
``Scheduler._update_request_as_session`` that restores the NeMo-faithful
retention, but ONLY for requests that opt in via
``SamplingParams.extra_args["streaming_stt_retain_until_blank"]``. Every other
request falls through to the unmodified vLLM implementation, so non-StreamingSTT
serving is unaffected.

The patch is installed from the plugin ``register()`` entry point
(``vllm.general_plugins``), so it runs in every vLLM process -- including the
EngineCore process that owns the scheduler. It is idempotent and
version-guarded: if the target method is absent (vLLM API drift / a build
without the streaming-session API), it logs a warning and no-ops, leaving the
engine fully functional for non-streaming use.

This is an interim mechanism; the upstream-friendly form is an opt-in
``extra_args`` contract contributed to vLLM directly.
"""

from typing import Any

from nemo.collections.speechlm2.vllm.salm.streaming_constants import (
    BLANK_ID_KEY,
    EOS_ID_KEY,
    FOOTER_IDS_KEY,
    RETAIN_FLAG,
)
from nemo.utils import logging

# Sentinel attribute set on the patched class so repeated ``register()`` calls
# (front-end + EngineCore + workers) install the patch at most once per process.
_PATCH_FLAG = "_nemo_streaming_stt_patched"


def install_streaming_session_patch() -> None:
    """Install the StreamingSTT session-retention scheduler patch.

    Idempotent and safe to call from every process. Wraps vLLM's
    ``Scheduler._update_request_as_session`` so that only requests carrying the
    ``streaming_stt_retain_until_blank`` opt-in flag get NeMo-faithful
    text+blank+footer retention; all other requests use the original behavior.

    Returns:
        None. On any incompatibility the function logs a warning and returns
        without modifying the scheduler.
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover - defensive against vLLM layout changes
        logging.warning(
            "StreamingSTT: could not import vLLM Scheduler (%s); session-retention " "patch not installed.",
            e,
        )
        return

    if getattr(Scheduler, _PATCH_FLAG, False):
        return

    original = getattr(Scheduler, "_update_request_as_session", None)
    if original is None:
        logging.warning(
            "StreamingSTT: vLLM Scheduler has no '_update_request_as_session' "
            "(requires the resumable streaming-session API). Session-retention "
            "patch not installed; non-streaming use is unaffected."
        )
        return

    from dataclasses import replace

    from vllm.v1.engine import EngineCoreEventType
    from vllm.v1.request import RequestStatus

    def _update_request_as_session(self: Any, session: Any, update: Any) -> None:
        """vLLM streaming-session update with opt-in StreamingSTT retention."""
        extra_args = update.sampling_params.extra_args if update.sampling_params is not None else None
        if not (extra_args and extra_args.get(RETAIN_FLAG)):
            # Not a StreamingSTT request: defer entirely to vLLM's behavior.
            return original(self, session, update)

        blank_id = extra_args.get(BLANK_ID_KEY)
        eos_id = extra_args.get(EOS_ID_KEY)
        footer_ids = list(extra_args.get(FOOTER_IDS_KEY) or [])

        # All tokens generated this chunk, including the final (uncomputed) stop
        # token that vLLM's default would drop.
        full_generated = list(session._all_token_ids[session.num_prompt_tokens :])
        if full_generated and eos_id is not None and full_generated[-1] == eos_id:
            # Speech chunk stopped on EOS (<|im_end|>): keep the generated text
            # (incl. trailing space) + footer. The blank NeMo appends on EOS is
            # OUTPUT-only and is NOT fed to KV -- injecting it here pollutes the
            # carried context and corrupts later chunks for multi-word turns.
            kept = full_generated[:-1]
        elif full_generated and blank_id is not None and full_generated[-1] == blank_id:
            # Silent chunk stopped on blank: NeMo feeds the blank to KV.
            kept = full_generated  # already ends on blank
        else:
            # max_tokens / other stop: append a blank like NeMo's filler.
            kept = full_generated
            if blank_id is not None:
                kept.append(blank_id)
        kept_output_tokens = kept + footer_ids

        # Everything below mirrors vLLM's own _update_request_as_session tail (only
        # the kept-token policy above is NeMo-specific); keep in sync if that
        # upstream method changes.
        # kept_output_tokens may include tokens without KV (blank/footer); fold
        # them into the next chunk's prefill by resetting to the prompt prefix.
        session._all_token_ids[:] = session._all_token_ids[: session.num_prompt_tokens]
        session._output_token_ids.clear()
        if session.prompt_token_ids is None:
            raise RuntimeError("StreamingSTT: streaming session has no prompt_token_ids to extend.")
        session.prompt_token_ids.extend(kept_output_tokens)
        session._all_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(mm_feature.mm_position, offset=mm_feature.mm_position.offset + base)
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    Scheduler._update_request_as_session = _update_request_as_session
    setattr(Scheduler, _PATCH_FLAG, True)
    logging.info("StreamingSTT: installed vLLM session-retention scheduler patch.")

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

"""Shared StreamingSTT vLLM session-contract constants.

These ``SamplingParams.extra_args`` keys form the opt-in retention contract
between the client-side session driver (``streaming_session``) and the
engine-side scheduler patch (``streaming_scheduler``). They live in this
dependency-free module so the producer and the consumer share a single source of
truth and cannot silently drift -- a key mismatch would quietly disable
retention (the scheduler would just defer to stock vLLM) with no error.
"""

# Presence (truthy) in extra_args opts a request into NeMo-faithful per-chunk KV
# retention; absent, the scheduler patch defers to stock vLLM.
RETAIN_FLAG = "streaming_stt_retain_until_blank"

# Marker token ids the scheduler patch reads to decide what to retain per chunk.
BLANK_ID_KEY = "streaming_stt_blank_id"
EOS_ID_KEY = "streaming_stt_eos_id"
FOOTER_IDS_KEY = "streaming_stt_footer_ids"

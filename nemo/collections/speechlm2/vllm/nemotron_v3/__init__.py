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

"""vLLM plugin registration for NeMo Speech LM models.

Registers NeMoSpeechLMConfig and both model classes into vLLM's
model and config registries via the ``vllm.general_plugins``
entry point.

Model classes:
  - NeMoSpeechLMHybridForConditionalGeneration — hybrid (Mamba+MoE, e.g. NemotronH)
  - NeMoSpeechLMForConditionalGeneration       — alias for hybrid (legacy compat)
  - NeMoSpeechLMStdForConditionalGeneration    — standard transformer (e.g. Qwen3)
"""

_PKG = "nemo.collections.speechlm2.vllm.nemotron_v3"


def register():
    """Register the NeMo Speech LM models and config with vLLM."""
    from transformers import AutoConfig

    from nemo.collections.speechlm2.vllm.nemotron_v3.config import NeMoSpeechLMConfig

    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["nemo_speechlm"] = NeMoSpeechLMConfig

    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMHybridForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "NeMoSpeechLMForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "NeMoSpeechLMStdForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMStdForConditionalGeneration",
    )

    _apply_backend_patches()


def _apply_backend_patches():
    """Apply runtime patches needed for vLLM compatibility.

    Called at plugin registration time. All patches here must be
    pickle-safe (no closures) since vLLM spawns EngineCore as a
    subprocess.
    """
    _patch_tokenizer_thread_safety()

    try:
        from transformers import AutoConfig as _AC

        _nhc = _AC.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
        )
        NHConfigCls = type(_nhc)
        _orig_getattr = getattr(NHConfigCls, "__getattr__", None)

        def _patched_getattr(self, name):
            if name == "rms_norm_eps":
                return getattr(self, "layer_norm_epsilon", 1e-5)
            if _orig_getattr:
                return _orig_getattr(self, name)
            raise AttributeError(name)

        NHConfigCls.__getattr__ = _patched_getattr
    except Exception:
        pass


def _thread_safe_batch_encode_plus(self, *args, **kwargs):
    """Wrapper that serializes _batch_encode_plus calls per instance."""
    import threading

    if not hasattr(self, "_tokenizer_lock"):
        self._tokenizer_lock = threading.Lock()
    with self._tokenizer_lock:
        return type(self)._orig_batch_encode_plus(self, *args, **kwargs)


def _patch_tokenizer_thread_safety():
    """Make HuggingFace fast tokenizer thread-safe for vLLM.

    ``PreTrainedTokenizerFast._batch_encode_plus`` first mutates the
    underlying Rust tokenizer (``set_truncation_and_padding`` calls
    ``enable_truncation`` / ``no_truncation``), then encodes text via
    ``self._tokenizer.encode_batch``.  When vLLM dispatches tokenizer
    calls to a thread pool, concurrent threads race on the Rust borrow
    checker and panic with ``RuntimeError: Already borrowed``.

    This patch wraps the entire ``_batch_encode_plus`` method in a
    per-instance ``threading.Lock`` so the truncation-setup + encode +
    cleanup cycle is atomic.

    Uses a module-level function (not a closure) so the patch survives
    multiprocessing spawn/pickle.
    """
    from transformers import PreTrainedTokenizerFast

    if hasattr(PreTrainedTokenizerFast, "_orig_batch_encode_plus"):
        return  # already patched

    PreTrainedTokenizerFast._orig_batch_encode_plus = (
        PreTrainedTokenizerFast._batch_encode_plus
    )
    PreTrainedTokenizerFast._batch_encode_plus = (
        _thread_safe_batch_encode_plus
    )

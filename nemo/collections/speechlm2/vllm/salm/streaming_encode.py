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

"""Cache-aware streaming audio encode for the NeMo StreamingSTT vLLM path.

The streaming StreamingSTT serving path encodes the audio NeMo-faithfully on the
client side and feeds the already-projected per-chunk embeddings into vLLM via
the precomputed-``audio_embeds`` multimodal input (one item per chunk turn). This
module provides that encode step as a reusable helper.

``stream_encode`` drives an ``AudioPerceptionModule`` chunk-by-chunk, threading
the FastConformer cache (``cache_last_channel`` / ``cache_last_time``) and a
``BatchedCacheFeatureBufferer`` across chunks -- exactly NeMo's
``_chunked_streaming_generate`` / ``_chunked_streaming_step`` encode path, minus
the LLM. This is the trained inference path (``use_offline_embs=False``); a
single full-context perception forward (offline embeddings) is non-causal and
truncates long utterances, so it must not be used for streaming.

The output is one ``list[Tensor(chunk_size, hidden)]`` per utterance -- the exact
shape vLLM expects under ``multi_modal_data={"audio_embeds": ...}`` with one
audio item per chunk.
"""

import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from nemo.utils import logging

# StreamingSTT encoder frames are 80 ms (FastConformer 8x subsampling of 10 ms
# mel frames). Override only for encoders with a different frame rate.
DEFAULT_FRAME_LENGTH_S = 0.08
DEFAULT_SAMPLE_RATE = 16000

# FastConformer keeps its relative-position attention biases as plain tensor
# attributes (not nn.Parameter / registered buffers), so ``Module.to(device)``
# skips them. They are listed here so the encoder can be relocated faithfully;
# extend this if a different encoder holds other plain device-resident tensors.
_PLAIN_DEVICE_TENSOR_ATTRS = ("pos_bias_u", "pos_bias_v")


def _move_perception_plain_tensors(perception: nn.Module, device: torch.device) -> None:
    """Relocate the encoder's plain (non-Parameter) tensors onto ``device``.

    ``nn.Module.to(device)`` only moves registered Parameters and buffers, but
    FastConformer keeps its relative-position attention biases
    (:data:`_PLAIN_DEVICE_TENSOR_ATTRS`) as plain tensor attributes. Those are
    silently left on their original device and later trigger a device mismatch
    inside attention. This walks every submodule and moves any such attribute to
    ``device``, matching the module's floating dtype when the tensor is floating
    point (integer tensors keep their dtype).

    Args:
        perception: The ``AudioPerceptionModule`` to fix up, already moved to
            ``device`` via ``.to(device)`` (which handles its Parameters/buffers).
        device: Target device for the plain tensors.
    """
    dtype = next(perception.parameters()).dtype
    for module in perception.modules():
        for name in _PLAIN_DEVICE_TENSOR_ATTRS:
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
                kwargs: dict = {"device": device}
                if value.is_floating_point():
                    kwargs["dtype"] = dtype
                setattr(module, name, value.to(**kwargs))


def load_streaming_perception(
    model_dir: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, dict]:
    """Load an ``AudioPerceptionModule`` (with weights) for client-side streaming encode.

    Reads the ``perception`` block from ``config.json``, builds the module, loads
    the ``perception.*`` weights from ``model.safetensors``, and moves it to
    ``device`` in ``dtype`` (eval mode, with plain bias tensors relocated).

    Args:
        model_dir: Path to the exported vLLM SALM model directory.
        device: Device to place the encoder on.
        dtype: Floating dtype for the encoder.

    Returns:
        ``(perception, preprocessor_cfg)`` -- the eval-mode module and its
        preprocessor config dict (pass the latter to :func:`stream_encode`).
    """
    import json
    import os

    from safetensors import safe_open

    from nemo.collections.speechlm2.modules import AudioPerceptionModule

    with open(os.path.join(model_dir, "config.json")) as f:
        perception_cfg = json.load(f)["perception"]
    perception = AudioPerceptionModule(DictConfig(perception_cfg)).to(dtype)

    weights: dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("perception."):
                tensor = f.get_tensor(key)
                weights[key[len("perception.") :]] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    perception.load_state_dict(weights, strict=False)
    perception = perception.to(device).eval()
    _move_perception_plain_tensors(perception, device)
    return perception, perception_cfg


def _resolve_pre_encode_cache_size(perception: nn.Module) -> int:
    """Return the encoder's pre-encode cache size (scalar), handling list configs."""
    pre_cache = perception.encoder.streaming_cfg.pre_encode_cache_size
    if isinstance(pre_cache, (list, tuple)):
        # NeMo may store two pre-encode cache sizes (first-chunk padding
        # variants of the same left cache, NOT left/right attention context).
        # Index [1] is the steady-state size used for every chunk after the
        # first; NeMo's own cache-aware CTC/RNNT pipelines also size the feature
        # buffer with [1]. ([0] is only the first chunk when
        # pad_and_drop_preencoded is False.)
        pre_cache = pre_cache[1]
    return int(pre_cache)


def stream_encode(
    perception: nn.Module,
    wavs: list[torch.Tensor],
    chunk_size: int,
    preprocessor_cfg: dict,
    *,
    device: torch.device,
    encode_batch_size: int = 64,
    frame_length_s: float = DEFAULT_FRAME_LENGTH_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[list[torch.Tensor]]:
    """Cache-aware streaming encode of waveforms into per-chunk embeddings.

    Replicates NeMo's chunked streaming encode (``use_offline_embs=False``):
    drives ``perception`` chunk-by-chunk while threading the FastConformer cache
    and a feature buffer, then projects each chunk to the LLM hidden size. The
    encoder is assumed to already live on ``device`` with its plain (non-Parameter)
    tensors (e.g. ``pos_bias_u/v``) moved to ``device`` as well.

    Args:
        perception: A NeMo ``AudioPerceptionModule`` (encoder + modality adapter +
            projection) in eval mode on ``device``.
        wavs: List of 1-D mono float32 waveforms at ``sample_rate``.
        chunk_size: Number of encoder output frames per chunk (the model's
            ``chunk_size``); each returned chunk tensor is padded/trimmed to this.
        preprocessor_cfg: The perception preprocessor config (e.g. the checkpoint
            ``config.json`` ``perception.preprocessor`` block); used to build the
            streaming feature buffer.
        device: CUDA device to run the encoder on.
        encode_batch_size: Number of utterances encoded together per streaming
            pass. Utterances are sorted by length to minimize padding waste.
        frame_length_s: Seconds of audio per encoder output frame.
        sample_rate: Audio sample rate in Hz.

    Returns:
        One ``list[Tensor(chunk_size, hidden)]`` per input utterance, aligned to
        the ``wavs`` order. Each inner list has one tensor per audio chunk.
    """
    from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import (
        BatchedCacheFeatureBufferer,
    )
    from nemo.collections.asr.inference.streaming.framing.request import Frame

    if not wavs:
        return []

    chunk_samples = math.ceil(chunk_size * frame_length_s * sample_rate)
    preprocessor_cfg = DictConfig(preprocessor_cfg)
    pre_cache = _resolve_pre_encode_cache_size(perception)
    buf_secs = pre_cache * float(preprocessor_cfg.window_stride) + chunk_size * frame_length_s
    emb_dtype = next(perception.parameters()).dtype

    # Build the feature bufferer once and reuse across batches via reset_slots:
    # its __init__ instantiates a mel preprocessor, which is expensive to rebuild
    # per batch (far costlier than the encoder forward itself).
    num_slots = max(1, min(encode_batch_size, len(wavs)))
    feature_bufferer = BatchedCacheFeatureBufferer(
        num_slots=num_slots,
        sample_rate=sample_rate,
        buffer_size_in_secs=buf_secs,
        chunk_size_in_secs=buf_secs,
        preprocessor_cfg=preprocessor_cfg,
        device=device,
    )

    def _encode_batch(batch: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        batch_size = len(batch)
        sample_lens = [int(w.shape[-1]) for w in batch]
        num_chunks = [max(1, math.ceil(n / chunk_samples)) for n in sample_lens]
        max_num_chunks = max(num_chunks)
        # Pad each waveform up to a whole number of chunks, so every per-chunk slice
        # is exactly chunk_samples wide -- the trailing zero-pad is what the last
        # partial chunk (and any chunk past an utterance's end) needs anyway. Pad +
        # stack on GPU; CPU pad+stack is the encode bottleneck, not the algorithm.
        audios = torch.stack(
            [F.pad(w.to(device, non_blocking=True), (0, max_num_chunks * chunk_samples - w.shape[-1])) for w in batch]
        )
        sample_lens_t = torch.tensor(sample_lens, device=device)

        feature_bufferer.reset_slots(list(range(batch_size)))  # fresh per-slot buffer state
        cache_last_channel, cache_last_time, cache_last_channel_len = perception.encoder.get_initial_cache_state(
            batch_size=batch_size, dtype=emb_dtype, device=device, max_dim=0
        )

        chunk_stack = []  # per chunk index: (batch, chunk_size, hidden) on GPU
        for chunk_idx in range(max_num_chunks):
            start = chunk_idx * chunk_samples
            # Whole-chunk slice for the entire batch at once; valid_lens marks how many
            # of those samples are real (the rest is the trailing zero-pad).
            stacked = audios[:, start : start + chunk_samples]
            valid_lens = (sample_lens_t - start).clamp(0, chunk_samples)
            frames = [Frame(samples=stacked[b], length=int(valid_lens[b]), stream_id=b) for b in range(batch_size)]

            feats, right_paddings = feature_bufferer.update(frames)
            processed_signal = torch.stack(feats).to(emb_dtype)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(rp) for rp in right_paddings], device=device
            ).long()

            with torch.no_grad():
                enc_emb, enc_len, cache_last_channel, cache_last_time, cache_last_channel_len = (
                    perception.encoder.cache_aware_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=False,
                    )
                )
                enc, _ = perception.modality_adapter(audio_signal=enc_emb, length=enc_len)
                enc = perception.proj(enc.transpose(1, 2))  # (batch, frames, hidden)

            num_frames = enc.shape[1]
            if num_frames < chunk_size:
                enc = F.pad(enc, (0, 0, 0, chunk_size - num_frames))
            elif num_frames > chunk_size:
                enc = enc[:, :chunk_size, :]
            chunk_stack.append(enc.to(emb_dtype))

        all_chunks = torch.stack(chunk_stack).cpu()  # (max_chunks, batch, chunk_size, hidden)
        return [[all_chunks[chunk_idx, b] for chunk_idx in range(num_chunks[b])] for b in range(batch_size)]

    # Sort by duration (descending) like NeMo's sampler to minimize padding waste,
    # then scatter results back to the input order.
    order = sorted(range(len(wavs)), key=lambda i: wavs[i].shape[-1], reverse=True)
    per_utt: list[list[torch.Tensor] | None] = [None] * len(wavs)
    for start in range(0, len(order), encode_batch_size):
        idxs = order[start : start + encode_batch_size]
        encoded = _encode_batch([wavs[i] for i in idxs])
        for k, i in enumerate(idxs):
            per_utt[i] = encoded[k]

    logging.debug("StreamingSTT: stream_encode produced per-chunk embeddings for %d utterances.", len(wavs))
    return per_utt  # type: ignore[return-value]

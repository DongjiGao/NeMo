# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


"""
This script provides an inference and evaluation script for end-to-end speaker diarization models.
The performance of the diarization model is measured using the Diarization Error Rate (DER).
If you want to evaluate its performance, the manifest JSON file should contain the corresponding RTTM
(Rich Transcription Time Marked) file.
Please refer to the NeMo Library Documentation for more details on data preparation for diarization inference:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit
/asr/speaker_diarization/datasets.html#data-preparation-for-inference

Usage for diarization inference:

The end-to-end speaker diarization model can be specified by "model_path".
Data for diarization is fed through the "dataset_manifest".
By default, post-processing is bypassed, and only binarization is performed.
If you want to reproduce DER scores reported on NeMo model cards, you need to apply post-processing steps.
Use batch_size = 1 to have the longest inference window and the highest possible accuracy.

python $BASEPATH/neural_diarizer/e2e_diarize_speech.py \
    model_path=/path/to/diar_sortformer_4spk_v1.nemo \
    batch_size=1 \
    dataset_manifest=/path/to/diarization_manifest.json

"""
import functools
import json
import logging
import os
import tempfile
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch._dynamo
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.models.sortformer_diar_models import COMPILED_FLEX_EXACT_128_BOUNDARY_WIDTH
from nemo.collections.asr.parts.utils.diarization_utils import convert_pred_mat_to_segments
from nemo.collections.asr.parts.utils.sortformer_fa4_attention import (
    FLEX_BACKEND,
    attention_backend_cache_identity,
    attention_backend_info,
    configure_attention_backend,
)
from nemo.collections.asr.parts.utils.sortformer_nvfp4_checkpoint import resolve_nvfp4_save_restore_connector
from nemo.collections.asr.parts.utils.sortformer_quantization import (
    ActivationAmaxCollector,
    ensure_calibration_output_writable,
    prediction_cache_identity,
    quantization_config_from_eval_cfg,
    quantize_sortformer_model,
    validate_calibration_forward_pass,
)
from nemo.collections.asr.parts.utils.sortformer_utils import (
    InferenceProfiler,
    configure_output_subsampling_factor,
    configure_streaming_mode,
    get_prediction_cache_metadata,
    load_prediction_tensors,
    save_prediction_tensors,
)
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.asr.parts.utils.transcribe_utils import read_and_maybe_sort_manifest
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams, load_postprocessing_from_yaml
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.config import hydra_runner
from nemo.utils.dependency import import_optional_dependency

seed_everything(42)
torch.backends.cudnn.deterministic = True


@dataclass
class DiarizationConfig:
    """Diarization configuration parameters for inference."""

    model_path: Optional[str] = None  # Path to a .nemo file
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    presort_manifest: Optional[bool] = True

    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    no_der: bool = False
    out_rttm_dir: Optional[str] = None
    out_preds_tensors: Optional[str] = None  # Explicit cache path; enables prediction loading and saving
    overwrite_preds_tensors: bool = False  # Ignore and replace an existing prediction cache
    precision: str = "bf16"  # 32, bf16, bf16-mixed

    # General configs
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    batch_size: int = 1
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True  # If True, postprocessing will be bypassed
    log: bool = False  # If True, log will be printed
    output_subsampling_factor: Optional[int] = None  # Override prediction step in 10 ms feature frames
    # Override the checkpoint's streaming mode: False runs the offline forward path, True forces streaming.
    streaming_mode: Optional[bool] = None
    profile_inference: bool = True  # Report wall time and detailed streaming-step timings
    # Number of leading model-forward calls excluded from the profiler summary, e.g. compilation warmup.
    profiler_warmup_calls: int = 0
    # Audio duration, in seconds, covered by the measured (non-warmup) forward calls; manifest total otherwise.
    measured_audio_duration: Optional[float] = None

    # Attention backend for the transformer encoders: flex (default, PyTorch FlexAttention), fa4_cute
    # (opt-in inference-only FlashAttention-4 on Blackwell GPUs) or fp8_flex (opt-in inference-only
    # float8_e4m3fn FlexAttention on the Triton backend, Blackwell GPUs, same key-padding block mask).
    # Orthogonal to the quantization recipe.
    attention_backend: str = "flex"

    # Quantization configs. Quantization is disabled by default and never changes the requested recipe silently.
    # Only the transformer-encoder attention/feed-forward linears are affected; see sortformer_quantization.py.
    quantization_recipe: str = "disabled"  # disabled, nvfp4_all, nvfp4_qkv_only, nvfp4_qkv_fp8_rest,
    # nvfp4_weight_only
    quantization_scale_mode: str = "dynamic"  # NVFP4 activation scales: dynamic or static
    # Calibration JSON consumed by scale_mode=static; produced by a separate recipe=disabled run.
    quantization_calibration_path: Optional[str] = None
    # Destination for an activation-amax calibration run; mutually exclusive with a non-disabled recipe.
    quantization_calibration_output: Optional[str] = None
    quantization_scale_margin: float = 1.0  # Positive multiplier applied to every calibrated activation amax
    # Accelerated NVFP4 packing requires MSLK >= 1.2; disabling it runs unaccelerated reference kernels.
    quantization_accelerated_packing: bool = True
    quantization_allow_reference_kernels: bool = False  # Explicit acknowledgement of unaccelerated kernels
    quantization_overwrite_calibration: bool = False  # Replace an existing calibration file
    # Fold the calibrated activation/weight global-scale product into the NVFP4 block scales so that the native
    # GEMM produces final values and applies the bias itself. Requires static W4A4 with accelerated packing.
    quantization_fold_global_scales: bool = False
    # Split of the folded product between the activation and weight block scales: F_a = 2 ** exponent.
    quantization_fold_activation_exponent: int = -10
    # Fuse the LayerNorm/GELU producers into the NVFP4 activation packs of the complete transformer blocks, so the
    # BF16 producer tensors in front of attn.w_qkv, ffn.net.0 and ffn.net.3 are never materialized. Requires
    # recipe='nvfp4_all' with static scales and accelerated packing, and is mutually exclusive with folding.
    quantization_fuse_producer_packing: bool = False
    # UTF-8 JSON file {"version": 1, "bf16_fqns": [...]} restoring an explicit proper subset of the quantization
    # targets to BF16, for family/layer sensitivity experiments. Entries are exact target FQNs, matched exactly and
    # never by suffix; every unlisted target keeps the recipe's precision -- NVFP4 weight-only under
    # recipe='nvfp4_weight_only', NVFP4 W4A4 under recipe='nvfp4_all'. Those are the only accepted recipes. With
    # static scales the calibration file must cover the FQNs that stay W4A4; entries for restored FQNs are
    # reported as unused. May be combined with quantization_fuse_producer_packing only when every restored FQN is
    # attn.out_proj, which fusion never packs into or dispatches; restoring a fused consumer is rejected.
    quantization_bf16_override_path: Optional[str] = None
    # How the NVFP4 weight block scales are chosen: 'amax' (default) is TorchAO's ordinary rule; 'mse' converts
    # with that same rule and then repacks every selected NVFP4 weight with an exhaustive per-16-weight-block E4M3
    # search, keeping TorchAO's global per-tensor scale and wire format. Requires an NVFP4 recipe and is mutually
    # exclusive with quantization_fold_global_scales. The per-layer and aggregate reconstruction MSE it achieved is
    # reported in the 'weight_scale_mse' section of the quantization summary.
    # 'local_hessian' runs the same exhaustive per-block search against the activation-weighted objective
    # sum_j h_damped[j] * (W - Q(W))^2, with h read from quantization_weight_scale_hessian_path. It requires
    # recipe='nvfp4_all' and is rejected together with folding and quantization_bf16_override_path.
    # 'four_over_six' reproduces NVIDIA ModelOpt 0.46.0's Four-Over-Six weight arithmetic: the weight global scale
    # is renormalized against 256 instead of 448 (the template scale times 448/256) and every 16-weight block is
    # written with its amax mapped onto FP4's largest magnitude 6 or onto 4, whichever reconstructs the block with
    # the lower squared error, ties keeping 6. It has no tunable search grid, requires an NVFP4 recipe, is rejected
    # together with folding and with a Hessian artifact path, and reports its per-layer M=6/M=4 block counts and
    # both reconstruction MSEs in the 'weight_scale_four_over_six' summary section.
    # 'awq_clip' adapts NVIDIA ModelOpt 0.46.0's AWQ clipping: for every output row and 16-weight input block an
    # offline builder picked one of eleven fixed clipping ratios of the block's own amax, minimizing that block's
    # contribution to the layer's output error on runtime-matched quantized activations, ties keeping the earliest
    # ratio. The run only reconstructs the ratio codes of quantization_weight_scale_awq_clip_path. It requires
    # recipe='nvfp4_all' with scale_mode='static', quantization_scale_margin=1.0 and a readable calibration file,
    # and is rejected together with folding, quantization_bf16_override_path and a Hessian artifact path. Its
    # per-layer ratio histograms and both reconstruction MSEs are reported in the 'weight_scale_awq_clip' section.
    # 'gptq' adapts NVIDIA ModelOpt 0.46.0's GPTQ and changes no scale at all: an offline builder wrote every input
    # column's FP4 payload under the ordinary template's own fixed block and global scales, propagating each
    # column's rounding residual to the following columns over 128-column blocks under a group-balanced input
    # Hessian damped by 0.01 of its diagonal mean. The run converts each target with the ordinary recipe, verifies
    # that the template's scale bytes and global scale are exactly the ones the payload of
    # quantization_weight_scale_gptq_path was written under, and replaces only the packed payload. It requires
    # recipe='nvfp4_all' with scale_mode='static', quantization_scale_margin=1.0 and a readable calibration file,
    # and is rejected together with folding, quantization_bf16_override_path and any other method's artifact path.
    # Its per-layer payload identities and both reconstruction MSEs are reported in the 'weight_scale_gptq' section.
    quantization_weight_scale_method: str = "amax"  # amax, mse, local_hessian, four_over_six, awq_clip, gptq
    # Diagonal-Hessian artifact JSON built by
    # scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_hessian_stats.py: one non-negative
    # input-channel second-moment vector per target FQN, bound to the checkpoint by per-weight digests. Required by
    # quantization_weight_scale_method='local_hessian' and rejected for every other method.
    quantization_weight_scale_hessian_path: Optional[str] = None
    # AWQ-clip artifact JSON built by
    # scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_awq_clip.py: one uint8 clipping-ratio code per
    # output row and 16-weight input block of every target FQN, bound to the checkpoint by per-weight digests and to
    # quantization_calibration_path by that file's exact content digest. Required by
    # quantization_weight_scale_method='awq_clip' and rejected for every other method.
    quantization_weight_scale_awq_clip_path: Optional[str] = None
    # GPTQ artifact JSON built by
    # scripts/dataset_processing/speaker_tasks/build_sortformer_nvfp4_gptq.py: the final packed FP4 payload of every
    # target FQN, bound to the checkpoint by per-weight digests, to each FQN's exact ordinary-template scale buffer
    # by that buffer's content digest, and to quantization_calibration_path by that file's exact content digest.
    # Required by quantization_weight_scale_method='gptq' and rejected for every other method.
    quantization_weight_scale_gptq_path: Optional[str] = None

    use_lhotse: bool = True
    batch_duration: int = 100000
    # Total padded-audio duration threshold, in seconds, for feature extraction.
    # Above it, batches are split along the batch dimension; <= 0 disables splitting.
    # Lower this value for feature-extraction OOMs.
    max_batch_dur: float = 100000

    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25  # Collar in seconds for DER calculation
    ignore_overlap: bool = False  # If True, DER will be calculated only for non-overlapping segments

    # Streaming diarization configs
    async_streaming: bool = False
    # Use fixed-size encoder inputs, trading extra padded computation for stable shapes. Applies to both
    # asynchronous and synchronous streaming, despite the historical `async_` prefix kept for compatibility.
    async_pad_to_max: bool = False
    # Compile the frontend encoder and optional transformer encoder; dynamic shapes support variable input lengths.
    compile_encoder: bool = False
    # Dynamic shapes handle variable input lengths; set False for static shapes when dynamic compilation fails.
    compile_dynamic: bool = True
    # Capture the compiled outer encoders as CUDA Graphs (torch.compile mode='reduce-overhead'). Opt-in and
    # offline only; requires compile_encoder=True, compile_dynamic=False, streaming_mode=False and a CUDA device.
    # The private FP8 FlexAttention compile boundary is deliberately left outside the captured region.
    compile_cuda_graphs: bool = False
    # Frontend-encoder maximum audio length, in encoder input frames, materialized before compilation so that the
    # retained positional state is created outside the captured region and the integer guard never changes between
    # batches. Required when compile_cuda_graphs=True and ignored otherwise.
    compile_cuda_graph_max_audio_length: Optional[int] = None
    # Capture the compiled streaming encoders as CUDA Graphs (torch.compile mode='reduce-overhead' with
    # dynamic=False). Opt-in and streaming only; requires compile_encoder=True, an explicit streaming_mode=True,
    # async_pad_to_max=True and a CUDA device, and is mutually exclusive with the offline compile_cuda_graphs mode.
    # The captured boundaries are the primary encoder forward that frontend_encoder() calls once per streaming step
    # and, when the checkpoint has one, the optional transformer encoder that forward_infer() calls in the same
    # step. The separately invoked encoder.pre_encode() and the private FP8 FlexAttention compile boundary are
    # deliberately left outside the captured region.
    compile_streaming_encoder_cuda_graphs: bool = False
    # torch.compile backend used for the encoders: 'inductor' (default) keeps the existing behaviour, 'cudagraphs'
    # captures the eager CUDA kernels as CUDA Graphs without Inductor kernel rewriting and is accepted only for
    # compile_streaming_encoder_cuda_graphs=True.
    compile_backend: str = "inductor"
    # Emulate production streams arriving independently; offline batches otherwise update in lockstep.
    async_desync_updates: bool = False
    spkcache_len: Optional[int] = None
    spkcache_update_period: int = 144
    fifo_len: int = 188
    chunk_len: int = 6
    chunk_left_context: Optional[int] = None
    chunk_right_context: int = 7

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # Optuna Config
    launch_pp_optim: bool = False  # If True, launch optimization process for postprocessing parameters
    optuna_study_name: str = "optim_postprocessing"
    optuna_temp_dir: str = "/tmp/optuna"
    optuna_storage: str = f"sqlite:///{optuna_study_name}.db"
    optuna_log_file: str = f"{optuna_study_name}.log"
    optuna_n_trials: int = 100000


CUDA_GRAPH_COMPILE_MODE = "reduce-overhead"  # torch.compile mode whose only option is triton.cudagraphs=True
INDUCTOR_COMPILE_BACKEND = "inductor"  # Default torch.compile backend, which rewrites the kernels it captures
CUDAGRAPHS_COMPILE_BACKEND = "cudagraphs"  # torch.compile backend that graph-captures the eager kernels unchanged
SUPPORTED_COMPILE_BACKENDS = (INDUCTOR_COMPILE_BACKEND, CUDAGRAPHS_COMPILE_BACKEND)
CUDA_GRAPH_MARKER_ATTRIBUTE = "_sortformer_cuda_graph_step_marker"
CUDA_GRAPH_LENGTH_STABILIZER_ATTRIBUTE = "_sortformer_cuda_graph_length_stabilizer"
# Retained per-shape length buffers of the installed stabilizer, exposed on the wrapper so that the buffers stay
# alive exactly as long as the wrapper that reuses them.
CUDA_GRAPH_LENGTH_BUFFERS_ATTRIBUTE = "_sortformer_cuda_graph_length_buffers"
# Keyword name of the length argument that the stabilized per-step boundary keeps in stable storage.
STREAMING_CUDA_GRAPH_LENGTH_ARGUMENT = "processed_signal_length"
# Single per-streaming-step boundary that every captured encoder call passes through: the model method that calls
# the primary encoder forward, before forward_infer() reaches the optional transformer encoder of the same step.
STREAMING_CUDA_GRAPH_STEP_BOUNDARY = "frontend_encoder"
# Development fixed-shape compile adapter. When it is active its frame count must agree with the pre-stabilized
# maximum audio length, otherwise the captured graph and the retained positional state disagree.
FIXED_COMPILE_ENV_VAR = "SORTFORMER_FIXED_COMPILE"
FIXED_COMPILE_TIME_FRAMES_ENV_VAR = "SORTFORMER_COMPILE_TIME_FRAMES"


def validate_cuda_graph_config(cfg: DiarizationConfig, device_type: str) -> None:
    """
    Fail closed on configurations that cannot capture the outer encoders as CUDA Graphs.

    Args:
        cfg (DiarizationConfig): The configuration object containing the compilation options.
        device_type (str): Type of the device selected for inference, for example ``cuda`` or ``cpu``.

    Raises:
        ValueError: If CUDA Graphs are requested without compiled static-shape offline inference on a CUDA device,
            without a positive integer ``compile_cuda_graph_max_audio_length``, or with an active fixed-shape compile
            adapter whose frame count disagrees with that length.
        RuntimeError: If the installed PyTorch build has no ``torch.compiler.cudagraph_mark_step_begin``.
    """
    if not cfg.compile_cuda_graphs:
        return
    if not cfg.compile_encoder:
        raise ValueError(
            "compile_cuda_graphs=True requires compile_encoder=True: the graphs are captured by the outer encoder "
            "torch.compile boundary."
        )
    if cfg.compile_dynamic:
        raise ValueError(
            "compile_cuda_graphs=True requires compile_dynamic=False: CUDA Graph capture needs static shapes."
        )
    if cfg.streaming_mode is not False:
        raise ValueError(
            "compile_cuda_graphs=True requires the offline forward path, so streaming_mode must be set explicitly "
            f"to False, got streaming_mode={cfg.streaming_mode}."
        )
    if device_type != "cuda":
        raise ValueError(f"compile_cuda_graphs=True requires a CUDA device, got device type '{device_type}'.")
    max_audio_length = cfg.compile_cuda_graph_max_audio_length
    if not isinstance(max_audio_length, int) or isinstance(max_audio_length, bool) or max_audio_length <= 0:
        raise ValueError(
            "compile_cuda_graphs=True requires compile_cuda_graph_max_audio_length to be a positive integer number "
            "of encoder input frames, so that the retained positional state is materialized before capture, got "
            f"compile_cuda_graph_max_audio_length={cfg.compile_cuda_graph_max_audio_length!r}."
        )
    if not callable(getattr(torch.compiler, "cudagraph_mark_step_begin", None)):
        raise RuntimeError(
            "compile_cuda_graphs=True requires a callable torch.compiler.cudagraph_mark_step_begin, which this "
            "PyTorch build does not provide."
        )
    validate_fixed_shape_adapter_env(max_audio_length)


def validate_streaming_encoder_cuda_graph_config(cfg: DiarizationConfig, device_type: str) -> None:
    """
    Fail closed on configurations that cannot capture the streaming encoder forwards as CUDA Graphs.

    This is the streaming counterpart of :func:`validate_cuda_graph_config`. It captures the encoder forwards that
    one streaming step calls with a fixed input shape, so it pins those boundaries to static shapes itself instead
    of requiring a globally static ``compile_dynamic``. The fixed input shape comes from ``async_pad_to_max=True``,
    which pads the packed speaker-cache, FIFO and chunk frames to their configured capacity on every streaming step.

    Args:
        cfg (DiarizationConfig): The configuration object containing the compilation options.
        device_type (str): Type of the device selected for inference, for example ``cuda`` or ``cpu``.

    Raises:
        ValueError: If the streaming mode is combined with the offline ``compile_cuda_graphs`` mode, or requested
            without compiled encoders, without explicit streaming inference, without fixed-shape streaming inputs,
            or on a non-CUDA device.
        RuntimeError: If the installed PyTorch build has no ``torch.compiler.cudagraph_mark_step_begin``.
    """
    if not cfg.compile_streaming_encoder_cuda_graphs:
        return
    if cfg.compile_cuda_graphs:
        raise ValueError(
            "compile_streaming_encoder_cuda_graphs=True is mutually exclusive with compile_cuda_graphs=True: the "
            "offline mode captures the outer encoders of one model forward and marks that forward, the streaming "
            "mode captures the fixed-shape encoder forwards of every streaming step and marks every step. Enable "
            "exactly one of them."
        )
    if not cfg.compile_encoder:
        raise ValueError(
            "compile_streaming_encoder_cuda_graphs=True requires compile_encoder=True: the graphs are captured by "
            "the encoder torch.compile boundaries."
        )
    if cfg.streaming_mode is not True:
        raise ValueError(
            "compile_streaming_encoder_cuda_graphs=True requires the streaming forward path, so streaming_mode "
            f"must be set explicitly to True, got streaming_mode={cfg.streaming_mode}."
        )
    if not cfg.async_pad_to_max:
        raise ValueError(
            "compile_streaming_encoder_cuda_graphs=True requires async_pad_to_max=True so that the captured encoder "
            "inputs keep a fixed physical shape across streaming steps."
        )
    if device_type != "cuda":
        raise ValueError(
            f"compile_streaming_encoder_cuda_graphs=True requires a CUDA device, got device type '{device_type}'."
        )
    if not callable(getattr(torch.compiler, "cudagraph_mark_step_begin", None)):
        raise RuntimeError(
            "compile_streaming_encoder_cuda_graphs=True requires a callable "
            "torch.compiler.cudagraph_mark_step_begin, which this PyTorch build does not provide."
        )


def validate_compile_backend(cfg: DiarizationConfig) -> None:
    """
    Fail closed on a torch.compile backend this launcher does not support or cannot use as configured.

    The default ``inductor`` backend rewrites the kernels it captures, which measurably shifts the streaming DER of
    the quantized recipes. The ``cudagraphs`` backend captures the eager CUDA kernels unchanged, so it isolates the
    graph-capture speedup from Inductor codegen, but it needs the fixed input shapes of the streaming graph mode and
    has no defined meaning for ordinary dynamic compilation or for the offline graph mode, which are rejected rather
    than reinterpreted. The backend must also be registered in the installed PyTorch build, never silently replaced
    by Inductor.

    Args:
        cfg (DiarizationConfig): The configuration object containing the compilation options.

    Raises:
        ValueError: If the backend is unknown, or if ``cudagraphs`` is requested outside the fixed-shape streaming
            encoder CUDA Graph mode.
        RuntimeError: If the requested backend is not registered by ``torch._dynamo.list_backends()``.
    """
    if cfg.compile_backend not in SUPPORTED_COMPILE_BACKENDS:
        raise ValueError(
            f"compile_backend={cfg.compile_backend!r} is not supported. Choose one of "
            f"{list(SUPPORTED_COMPILE_BACKENDS)}: '{INDUCTOR_COMPILE_BACKEND}' keeps the default kernel-rewriting "
            f"backend, '{CUDAGRAPHS_COMPILE_BACKEND}' captures the eager CUDA kernels and requires "
            "compile_streaming_encoder_cuda_graphs=True."
        )
    if cfg.compile_backend == INDUCTOR_COMPILE_BACKEND:
        return
    if not cfg.compile_streaming_encoder_cuda_graphs:
        raise ValueError(
            f"compile_backend='{CUDAGRAPHS_COMPILE_BACKEND}' requires compile_streaming_encoder_cuda_graphs=True: it "
            "captures the fixed-shape streaming encoder forwards without Inductor kernel rewriting, and has no "
            "defined meaning for ordinary dynamic compilation or for the offline compile_cuda_graphs mode. Set "
            f"compile_backend='{INDUCTOR_COMPILE_BACKEND}' for those modes."
        )
    available_backends = torch._dynamo.list_backends()
    if CUDAGRAPHS_COMPILE_BACKEND not in available_backends:
        raise RuntimeError(
            f"compile_backend='{CUDAGRAPHS_COMPILE_BACKEND}' is not registered by this PyTorch build, which reports "
            f"{sorted(available_backends)}. Falling back to another backend would silently change the compiled "
            "kernels, so the run is stopped instead."
        )


def resolve_encoder_compile_kwargs(cfg: DiarizationConfig) -> Dict[str, Any]:
    """
    Build the ``torch.compile`` keyword arguments for the encoders this launcher compiles.

    Both CUDA Graph modes capture outer encoder boundaries and therefore need static shapes and the reduce-overhead
    mode; they differ in which forward calls are captured, not in how the encoders are compiled, so the primary and
    the optional transformer encoder share these arguments. The offline mode already requires
    ``compile_dynamic=False``, while the streaming mode overrides it, because only the fixed-shape streaming
    forwards of both encoders are captured. Ordinary compilation keeps the configured ``compile_dynamic`` policy.
    The private FlexAttention compile boundaries keep their own mode, so no graph is captured around them.

    The streaming graph mode additionally honours ``compile_backend``: the explicit ``cudagraphs`` backend captures
    the eager CUDA kernels itself, so it replaces the Inductor-specific reduce-overhead mode instead of joining it.

    Args:
        cfg (DiarizationConfig): The configuration object containing the compilation options.

    Returns:
        compile_kwargs (Dict[str, Any]): Keyword arguments passed to ``torch.compile`` for the encoders.
    """
    if cfg.compile_streaming_encoder_cuda_graphs and cfg.compile_backend == CUDAGRAPHS_COMPILE_BACKEND:
        return {"dynamic": False, "backend": CUDAGRAPHS_COMPILE_BACKEND}
    if cfg.compile_cuda_graphs or cfg.compile_streaming_encoder_cuda_graphs:
        return {"dynamic": False, "mode": CUDA_GRAPH_COMPILE_MODE}
    return {"dynamic": cfg.compile_dynamic}


def should_enable_exact_128_boundary_shim(cfg: DiarizationConfig) -> bool:
    """
    Decide whether the exact-128 compiled FlexAttention boundary shim applies to this configuration.

    The pinned torch 2.12 build fails inside Inductor kernel fusion when a synchronous, unpadded, dynamically
    compiled BF16 FlexAttention encoder is called with a physical time width of exactly
    ``COMPILED_FLEX_EXACT_128_BOUNDARY_WIDTH``. That is the only combination the shim is meant for, so every other
    run - eager, asynchronous, fixed-shape, statically compiled, non-Flex or non-BF16 - keeps its current behaviour.

    Args:
        cfg (DiarizationConfig): The configuration object containing the compilation and streaming options.

    Returns:
        enabled (bool): True when all conditions of the failing combination hold.
    """
    return (
        cfg.compile_encoder
        and cfg.compile_dynamic
        and cfg.streaming_mode is True
        and not cfg.async_streaming
        and not cfg.async_pad_to_max
        and cfg.attention_backend == FLEX_BACKEND
        and str(cfg.precision).startswith("bf16")
    )


def validate_fixed_shape_adapter_env(max_audio_length: int) -> None:
    """
    Reject an active development fixed-shape compile adapter that disagrees with the pre-stabilized maximum length.

    The adapter is considered active when ``SORTFORMER_FIXED_COMPILE`` is ``1`` or ``SORTFORMER_COMPILE_TIME_FRAMES``
    is set. Generic runs, where neither variable is present, are unaffected.

    Args:
        max_audio_length (int): Maximum audio length, in encoder input frames, requested by the configuration.

    Raises:
        ValueError: If the adapter is active and its frame count is missing, malformed, or different.
    """
    fixed_compile = os.environ.get(FIXED_COMPILE_ENV_VAR)
    time_frames = os.environ.get(FIXED_COMPILE_TIME_FRAMES_ENV_VAR)
    if fixed_compile != "1" and time_frames is None:
        return
    try:
        adapter_frames = int(str(time_frames).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"compile_cuda_graphs=True with the fixed-shape compile adapter active ({FIXED_COMPILE_ENV_VAR}="
            f"{fixed_compile!r}) requires {FIXED_COMPILE_TIME_FRAMES_ENV_VAR} to be a positive integer equal to "
            f"compile_cuda_graph_max_audio_length={max_audio_length}, got {time_frames!r}."
        ) from None
    if adapter_frames <= 0 or adapter_frames != max_audio_length:
        raise ValueError(
            f"compile_cuda_graphs=True requires {FIXED_COMPILE_TIME_FRAMES_ENV_VAR}={time_frames!r} to be a positive "
            f"integer equal to compile_cuda_graph_max_audio_length={max_audio_length}, otherwise the captured graph "
            "and the pre-stabilized positional state disagree."
        )


def stabilize_encoder_max_audio_length(encoder: torch.nn.Module, max_audio_length: int) -> None:
    """
    Materialize the encoder positional state for a fixed maximum audio length before compilation.

    Calling this outside any compiled region creates the retained RoPE cos/sin buffers up front and pins the integer
    ``max_audio_length`` guard, so a later batch cannot extend the positional state from inside a captured CUDA Graph.

    Args:
        encoder (torch.nn.Module): Encoder whose maximum audio length is stabilized, before ``torch.compile``.
        max_audio_length (int): Maximum audio length in encoder input frames.

    Raises:
        RuntimeError: If the encoder has no callable ``set_max_audio_length`` or does not adopt the requested length.
    """
    set_max_audio_length = getattr(encoder, "set_max_audio_length", None)
    if not callable(set_max_audio_length):
        raise RuntimeError(
            "compile_cuda_graphs=True requires an encoder with a callable set_max_audio_length() so that the "
            f"positional state can be materialized before capture, got {type(encoder).__name__}."
        )
    set_max_audio_length(max_audio_length)
    observed = getattr(encoder, "max_audio_length", None)
    if observed != max_audio_length:
        raise RuntimeError(
            "compile_cuda_graphs=True could not stabilize the encoder maximum audio length: requested "
            f"{max_audio_length}, but {type(encoder).__name__}.max_audio_length is {observed!r}."
        )


def resolve_streaming_cuda_graph_targets(model: SortformerEncLabelModel) -> Dict[str, torch.nn.Module]:
    """
    Collect the modules that the streaming encoder CUDA Graph mode captures on the restored model.

    Every streaming step calls the primary encoder forward once through ``frontend_encoder()``, and, when the
    checkpoint has a non-empty optional transformer encoder, calls that encoder once through ``forward_infer()`` of
    the same step. Both therefore see a fixed input shape under ``async_pad_to_max=True`` and are captured. A
    checkpoint whose quantized attention/feed-forward blocks all live in the primary encoder has no optional
    transformer encoder, and that absence is accepted rather than rejected. The pre-encode submodule stays outside
    the captured forward, because ``SortformerEncLabelModel._call_pre_encode()`` invokes ``encoder.pre_encode``
    directly instead of going through the compiled encoder forward.

    Args:
        model (SortformerEncLabelModel): Restored model whose streaming capture targets are collected.

    Returns:
        targets (Dict[str, torch.nn.Module]): Attribute name to captured module, primary ``encoder`` first.

    Raises:
        ValueError: If the required primary encoder module or the per-step call boundary that marks it is missing,
            in which case the requested capture has nothing usable to capture or to mark.
    """
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, torch.nn.Module):
        raise ValueError(
            "compile_streaming_encoder_cuda_graphs=True requires the restored model to expose a primary encoder "
            "module, whose forward every streaming step calls once with a fixed input shape, got "
            f"encoder={type(encoder).__name__}."
        )
    if not callable(getattr(model, STREAMING_CUDA_GRAPH_STEP_BOUNDARY, None)):
        raise ValueError(
            "compile_streaming_encoder_cuda_graphs=True requires a callable "
            f"{STREAMING_CUDA_GRAPH_STEP_BOUNDARY}() boundary on the restored model, which is where every "
            f"streaming step is marked exactly once, got {type(model).__name__}."
        )
    targets: Dict[str, torch.nn.Module] = {"encoder": encoder}
    transformer_encoder = getattr(model, "transformer_encoder", None)
    layers = getattr(transformer_encoder, "layers", None)
    if isinstance(transformer_encoder, torch.nn.Module) and layers is not None and len(layers) > 0:
        targets["transformer_encoder"] = transformer_encoder
    return targets


def install_cuda_graph_step_marker(model: torch.nn.Module, method_name: str = "forward") -> Callable:
    """
    Mark every call of one bound method as the beginning of a new CUDA Graph iteration.

    The wrapper replaces the bound method on the instance, so it is also in effect for callers that bypass forward
    hooks, such as ``test_batch()`` calling ``self.forward`` directly. The offline CUDA Graph mode marks the model
    ``forward``, and installing it before the inference profiler keeps exactly one marker per model forward,
    because the profiler wraps whatever method it finds. The streaming mode marks ``frontend_encoder`` instead: it
    is the single boundary every streaming step passes through, before the captured primary encoder call and
    before the optional transformer encoder consumes that output in the same step.

    Args:
        model (torch.nn.Module): Module or model that owns the marked method. Repeated calls are no-ops.
        method_name (str): Name of the bound method to mark. Defaults to ``forward``.

    Returns:
        marked_method (Callable): The installed method, or the existing one when already installed.

    Raises:
        ValueError: If the object has no callable method of that name to mark.
    """
    marker_attribute = f"{CUDA_GRAPH_MARKER_ATTRIBUTE}_{method_name}"
    original_method = getattr(model, method_name, None)
    if not callable(original_method):
        raise ValueError(
            f"CUDA Graph step marking requires a callable {method_name}() on {type(model).__name__}, got "
            f"{type(original_method).__name__}."
        )
    already_installed = getattr(model, marker_attribute, False) or getattr(
        original_method, CUDA_GRAPH_MARKER_ATTRIBUTE, False
    )
    if already_installed:
        return original_method

    @functools.wraps(original_method)
    def marked_method(*args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        return original_method(*args, **kwargs)

    setattr(marked_method, CUDA_GRAPH_MARKER_ATTRIBUTE, True)
    setattr(model, method_name, marked_method)
    setattr(model, marker_attribute, True)
    return marked_method


def install_streaming_cuda_graph_length_stabilizer(
    model: torch.nn.Module, method_name: str = STREAMING_CUDA_GRAPH_STEP_BOUNDARY
) -> Callable:
    """
    Give the captured streaming boundary a length tensor whose storage is stable across steps.

    ``async_pad_to_max=True`` fixes the shape of every captured input, but the streaming loop still builds a freshly
    allocated ``processed_signal_length`` tensor per step. A CUDA Graph is keyed on the data pointers of its static
    inputs, so a new allocation per step makes PyTorch re-record the graph on every step ("static input data pointer
    changed") instead of replaying it. This wrapper retains one buffer per length tensor metadata key (shape, dtype,
    device), copies the current values into the matching buffer outside the compiled encoder, and calls the original
    boundary with that buffer, so the captured call sees one pointer per supported shape while still reading the
    newest values. Alternating supported shapes, such as a final partial batch, each reuse their own retained buffer.

    ``processed_signal`` is deliberately passed through untouched: its address is already stable across steps, and
    copying a feature tensor of that size every step would cost more than the capture saves. A returned length that
    is the retained buffer itself, which the encoder produces when it hands a pre-encoded input length back
    unchanged, is returned as a copy, so that a caller keeping it across steps still reads its own step's values.

    Args:
        model (torch.nn.Module): Model that owns the per-step boundary. Repeated calls are no-ops.
        method_name (str): Name of the bound boundary method to stabilize. Defaults to the streaming boundary.

    Returns:
        stabilized_method (Callable): The installed method, or the existing one when already installed.

    Raises:
        ValueError: If the object has no callable method of that name to stabilize.
    """
    stabilizer_attribute = f"{CUDA_GRAPH_LENGTH_STABILIZER_ATTRIBUTE}_{method_name}"
    original_method = getattr(model, method_name, None)
    if not callable(original_method):
        raise ValueError(
            f"CUDA Graph length stabilization requires a callable {method_name}() on {type(model).__name__}, got "
            f"{type(original_method).__name__}."
        )
    already_installed = getattr(model, stabilizer_attribute, False) or getattr(
        original_method, CUDA_GRAPH_LENGTH_STABILIZER_ATTRIBUTE, False
    )
    if already_installed:
        return original_method

    # Keyed by tensor metadata rather than by call order, so that a recurring shape reuses its own retained buffer.
    length_buffers: Dict[Tuple, torch.Tensor] = {}

    def stabilized_length(length: torch.Tensor) -> torch.Tensor:
        key = (tuple(length.shape), length.dtype, length.device)
        buffer = length_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(length.shape, dtype=length.dtype, device=length.device)
            length_buffers[key] = buffer
        if buffer is not length:
            buffer.copy_(length)
        return buffer

    def detached_result(result: Any, buffer: Optional[torch.Tensor]) -> Any:
        # The encoder forward may hand the untouched length straight back, and a caller that keeps that tensor
        # across steps must not observe the next step overwriting the retained buffer in place.
        if buffer is None or not isinstance(result, tuple):
            return result
        return tuple(buffer.clone() if item is buffer else item for item in result)

    @functools.wraps(original_method)
    def stabilized_method(*args, **kwargs):
        buffer = None
        if STREAMING_CUDA_GRAPH_LENGTH_ARGUMENT in kwargs:
            length = kwargs[STREAMING_CUDA_GRAPH_LENGTH_ARGUMENT]
            if isinstance(length, torch.Tensor):
                buffer = stabilized_length(length)
                kwargs = dict(kwargs)
                kwargs[STREAMING_CUDA_GRAPH_LENGTH_ARGUMENT] = buffer
        elif len(args) > 1 and isinstance(args[1], torch.Tensor):
            buffer = stabilized_length(args[1])
            args = (args[0], buffer) + args[2:]
        return detached_result(original_method(*args, **kwargs), buffer)

    setattr(stabilized_method, CUDA_GRAPH_LENGTH_STABILIZER_ATTRIBUTE, True)
    setattr(stabilized_method, CUDA_GRAPH_LENGTH_BUFFERS_ATTRIBUTE, length_buffers)
    setattr(model, method_name, stabilized_method)
    setattr(model, stabilizer_attribute, True)
    return stabilized_method


def install_streaming_cuda_graph_boundary(cfg: DiarizationConfig, model: torch.nn.Module) -> bool:
    """
    Prepare the per-streaming-step boundary of the streaming encoder CUDA Graph mode.

    Both wrappers are installed after the encoders are compiled, and only when the streaming graph mode is enabled:
    the offline graph mode marks the model forward instead, and the non-graph modes leave the boundary untouched.
    The length stabilizer is installed first, so that the step marker wraps it and every streaming step still begins
    the new CUDA Graph iteration before anything else of that step runs, including the length copy.

    Args:
        cfg (DiarizationConfig): Resolved configuration whose ``compile_streaming_encoder_cuda_graphs`` decides
            whether the boundary is prepared at all.
        model (torch.nn.Module): Restored model that owns the boundary. Repeated calls are no-ops.

    Returns:
        installed (bool): Whether the boundary wrappers were requested by this configuration.
    """
    if not cfg.compile_streaming_encoder_cuda_graphs:
        return False
    install_streaming_cuda_graph_length_stabilizer(model, method_name=STREAMING_CUDA_GRAPH_STEP_BOUNDARY)
    install_cuda_graph_step_marker(model, method_name=STREAMING_CUDA_GRAPH_STEP_BOUNDARY)
    return True


def optuna_suggest_params(postprocessing_cfg: PostProcessingParams, trial) -> PostProcessingParams:
    """
    Suggests hyperparameters for postprocessing using Optuna.
    See the following link for `trial` instance in Optuna framework.
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial

    Args:
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameters.

    Returns:
        PostProcessingParams: The updated postprocessing configuration with suggested hyperparameters.
    """
    postprocessing_cfg.onset = trial.suggest_float("onset", 0.4, 0.8, step=0.01)
    postprocessing_cfg.offset = trial.suggest_float("offset", 0.4, 0.9, step=0.01)
    postprocessing_cfg.pad_onset = trial.suggest_float("pad_onset", 0.1, 0.5, step=0.01)
    postprocessing_cfg.pad_offset = trial.suggest_float("pad_offset", 0.0, 0.2, step=0.01)
    postprocessing_cfg.min_duration_on = trial.suggest_float("min_duration_on", 0.0, 0.75, step=0.01)
    postprocessing_cfg.min_duration_off = trial.suggest_float("min_duration_off", 0.0, 0.75, step=0.01)
    return postprocessing_cfg


def get_tensor_path(cfg: DiarizationConfig) -> tuple[Optional[str], str, str]:
    """
    Resolve the explicit prediction-cache path and derive model/manifest identifiers.

    Args:
        cfg (DiarizationConfig): The configuration object containing model and dataset details.

    Returns:
        tensor_path (Optional[str]): Absolute prediction-cache path, or ``None`` when caching is disabled.
        model_id (str): Model identifier including the configured output subsampling factor.
        tensor_filename (str): Manifest-derived identifier used in the prediction tensor filename.
    """
    tensor_filename = os.path.basename(cfg.dataset_manifest).replace("manifest.", "").replace(".json", "")
    model_path = Path(cfg.model_path).expanduser().absolute()
    model_id = model_path.name.replace(".ckpt", "").replace(".nemo", "")
    model_id = f"{model_id}_sf{cfg.output_subsampling_factor}"
    if cfg.out_preds_tensors:
        tensor_path = Path(cfg.out_preds_tensors).expanduser().absolute()
    else:
        tensor_path = None
    return None if tensor_path is None else str(tensor_path), model_id, tensor_filename


def diarization_objective(
    trial,
    postprocessing_cfg: PostProcessingParams,
    temp_out_dir: str,
    infer_audio_rttm_dict: Dict[str, Dict[str, str]],
    diar_model_preds_total_list: List[torch.Tensor],
    unit_10ms_frame_count: int,
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization in speaker diarization.

    This function evaluates the diarization performance using a set of postprocessing parameters
    suggested by Optuna. It converts prediction matrices to time-stamp segments, scores the
    diarization results, and returns the Diarization Error Rate (DER) as the optimization metric.

    Args:
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameters.
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        temp_out_dir (str): Temporary directory for storing intermediate outputs.
        infer_audio_rttm_dict (Dict[str, Dict[str, str]]): Dictionary containing audio file paths,
            offsets, durations, and RTTM file paths.
        diar_model_preds_total_list (List[torch.Tensor]): List of prediction matrices containing
            sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        unit_10ms_frame_count (int): Number of 10 ms feature frames represented by each prediction frame.
        collar (float, optional): Collar in seconds for DER calculation. Defaults to 0.25.
        ignore_overlap (bool, optional): If True, DER will be calculated only for non-overlapping segments.
            Defaults to False.

    Returns:
        der (float): Diarization Error Rate for the given set of postprocessing parameters.
    """
    with tempfile.TemporaryDirectory(dir=temp_out_dir, prefix="Diar_PostProcessing_") as _:
        if trial is not None:
            postprocessing_cfg = optuna_suggest_params(postprocessing_cfg, trial)
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
            audio_rttm_map_dict=infer_audio_rttm_dict,
            postprocessing_cfg=postprocessing_cfg,
            batch_preds_list=diar_model_preds_total_list,
            unit_10ms_frame_count=unit_10ms_frame_count,
            bypass_postprocessing=False,
        )
        metric, _, _ = score_labels(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=collar,
            ignore_overlap=ignore_overlap,
        )
        der = abs(metric)
    return der


def run_optuna_hyperparam_search(
    cfg: DiarizationConfig,  # type: DiarizationConfig
    postprocessing_cfg: PostProcessingParams,
    infer_audio_rttm_dict: Dict[str, Dict[str, str]],
    preds_list: List[torch.Tensor],
    temp_out_dir: str,
    unit_10ms_frame_count: int,
):
    """
    Run Optuna hyperparameter optimization for speaker diarization.

    Args:
        cfg (DiarizationConfig): The configuration object containing model and dataset details.
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        infer_audio_rttm_dict (dict): dictionary of audio file path, offset, duration and RTTM filepath.
        preds_list (List[torch.Tensor]): list of prediction matrices containing sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        temp_out_dir (str): temporary directory for storing intermediate outputs.
        unit_10ms_frame_count (int): Number of 10 ms feature frames represented by each prediction frame.
    """
    optuna = import_optional_dependency("optuna")

    worker_function = lambda trial: diarization_objective(
        trial=trial,
        postprocessing_cfg=postprocessing_cfg,
        temp_out_dir=temp_out_dir,
        infer_audio_rttm_dict=infer_audio_rttm_dict,
        diar_model_preds_total_list=preds_list,
        unit_10ms_frame_count=unit_10ms_frame_count,
        collar=cfg.collar,
    )
    study = optuna.create_study(
        direction="minimize", study_name=cfg.optuna_study_name, storage=cfg.optuna_storage, load_if_exists=True
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    if cfg.optuna_log_file is not None:
        logger.addHandler(logging.FileHandler(cfg.optuna_log_file, mode="a"))
    logger.addHandler(logging.StreamHandler())
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    study.optimize(worker_function, n_trials=cfg.optuna_n_trials)


@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:
    """Main function for end-to-end speaker diarization inference."""
    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None:
        raise ValueError("cfg.model_path cannot be None. Please specify the path to the model.")

    # Validate the quantization options and the calibration destination before the checkpoint is restored, so an
    # unusable combination fails immediately instead of after model loading.
    quantization_config = quantization_config_from_eval_cfg(cfg)
    ensure_calibration_output_writable(quantization_config)

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    # Reject unsupported CUDA Graph combinations before the checkpoint is restored instead of falling back silently.
    # The streaming mode is checked first so that requesting both modes reports the mutual exclusion instead of the
    # offline requirements the streaming configuration cannot satisfy.
    validate_streaming_encoder_cuda_graph_config(cfg, map_location.type)
    validate_cuda_graph_config(cfg, map_location.type)
    # Likewise for the compile backend, so an unusable or unregistered backend never reaches compilation.
    validate_compile_backend(cfg)

    if cfg.model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=cfg.model_path, map_location=map_location, strict=False
        )
    elif cfg.model_path.endswith(".nemo"):
        # A self-contained NVFP4 archive stores safetensors rather than a torch.save pickle, so it needs its own
        # connector. Detected from the archive's own contents instead of asking for a flag: the whole point of
        # exporting is that a consumer does not have to know the recipe to run it correctly.
        nvfp4_connector = resolve_nvfp4_save_restore_connector(cfg.model_path)
        if nvfp4_connector is not None:
            diar_model = SortformerEncLabelModel.restore_from(
                restore_path=cfg.model_path, map_location=map_location, save_restore_connector=nvfp4_connector
            )
        else:
            diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.model_path, map_location=map_location)
    else:
        raise ValueError("cfg.model_path must end with.ckpt or.nemo!")

    diar_model.max_batch_dur = cfg.max_batch_dur
    configure_streaming_mode(diar_model, cfg.streaming_mode)

    cfg.output_subsampling_factor = configure_output_subsampling_factor(diar_model, cfg.output_subsampling_factor)
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator, precision=cfg.precision)
    diar_model.set_trainer(trainer)

    if torch.cuda.is_bf16_supported() and cfg.precision.startswith("bf16"):
        diar_model = diar_model.to(dtype=torch.bfloat16).eval()
    else:
        diar_model = diar_model.eval()

    sorted_manifest_path = None
    if cfg.presort_manifest:
        audio_key = cfg.get('audio_key', 'audio_filepath')
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            for item in read_and_maybe_sort_manifest(cfg.dataset_manifest, try_sort=cfg.presort_manifest):
                audio_file = get_full_path(audio_file=item[audio_key], manifest_file=cfg.dataset_manifest)
                item[audio_key] = audio_file
                f.write(json.dumps(item) + "\n")
            sorted_manifest_path = f.name
        diar_model._cfg.test_ds.manifest_filepath = sorted_manifest_path
        infer_audio_rttm_dict = audio_rttm_map(sorted_manifest_path)
    else:
        diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
        infer_audio_rttm_dict = audio_rttm_map(cfg.dataset_manifest)
    remove_path_after_done = sorted_manifest_path if sorted_manifest_path is not None else None

    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    diar_model._cfg.test_ds.pin_memory = False
    diar_model._cfg.test_ds.num_spks = -1

    OmegaConf.set_struct(diar_model._cfg, False)
    diar_model._cfg.test_ds.use_lhotse = cfg.use_lhotse
    diar_model._cfg.test_ds.use_bucketing = False
    diar_model._cfg.test_ds.drop_last = False
    diar_model._cfg.test_ds.batch_duration = cfg.batch_duration
    OmegaConf.set_struct(diar_model._cfg, True)

    # Model setup for inference
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)

    # Streaming mode setup (only if enabled)
    if diar_model.streaming_mode:
        diar_model.async_streaming = cfg.async_streaming
        diar_model.async_pad_to_max = cfg.async_pad_to_max
        diar_model.sortformer_modules.async_desync_updates = cfg.async_desync_updates
        diar_model.sortformer_modules.chunk_len = cfg.chunk_len
        if cfg.spkcache_len is not None:
            diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
        if cfg.chunk_left_context is not None:
            diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
        diar_model.sortformer_modules.fifo_len = cfg.fifo_len
        diar_model.sortformer_modules.log = cfg.log
        diar_model.sortformer_modules.spkcache_update_period = cfg.spkcache_update_period
        diar_model._check_streaming_parameters()

    # Select the attention kernel before quantization and compilation so both observe the final path.
    applied_backends = configure_attention_backend(diar_model, cfg.attention_backend)
    logging.info(f"Attention backend '{cfg.attention_backend}' applied to: {sorted(applied_backends)}")
    if cfg.attention_backend != FLEX_BACKEND:
        backend_info = attention_backend_info(cfg.attention_backend, diar_model.device)
        logging.info(f"Attention backend details: {json.dumps(backend_info)}")

    if should_enable_exact_128_boundary_shim(cfg):
        diar_model.compiled_flex_exact_128_boundary_shim = True
        boundary_width = COMPILED_FLEX_EXACT_128_BOUNDARY_WIDTH
        logging.info(
            "Exact-128 compiled FlexAttention boundary shim enabled: this run is synchronous, unpadded, dynamically "
            "compiled BF16 FlexAttention, where the pinned PyTorch build fails Inductor kernel fusion at a physical "
            f"encoder time width of exactly {boundary_width}. One masked frame is appended only at that width, so "
            f"the compiled encoder sees physical T={boundary_width + 1} while the valid length stays "
            f"{boundary_width}. All other widths, the streaming state and the valid lengths are unchanged."
        )

    # Quantize before compiling so that torch.compile traces the quantized modules.
    if quantization_config.enabled:
        quantization_summary = quantize_sortformer_model(diar_model, quantization_config)
        logging.info(f"Sortformer quantization summary:\n{json.dumps(quantization_summary, indent=2)}")

    if cfg.compile_cuda_graphs:
        # Materialize the retained positional state before compilation so that no batch extends it from inside a
        # captured graph, which previously produced overwritten CUDA Graph outputs on the second batch.
        stabilize_encoder_max_audio_length(diar_model.encoder, cfg.compile_cuda_graph_max_audio_length)
        logging.info(
            f"CUDA Graphs requested: applying torch.compile mode='{CUDA_GRAPH_COMPILE_MODE}' to the outer frontend "
            "and optional transformer encoders only, leaving the private FlexAttention compile boundary unchanged. "
            "Static shapes are required (compile_dynamic=False) and every model forward is preceded by an explicit "
            "torch.compiler.cudagraph_mark_step_begin() marker. The frontend encoder positional state was "
            f"materialized before compilation at max_audio_length={cfg.compile_cuda_graph_max_audio_length} frames, "
            "so the integer length guard is already fixed. Whether PyTorch actually captures a graph is reported by "
            "PyTorch at runtime and is not implied by this configuration."
        )

    if cfg.compile_streaming_encoder_cuda_graphs:
        # Fail before inference when the required encoder or its per-step call boundary is missing, instead of
        # running the requested mode as a no-op. A checkpoint without the optional transformer encoder is accepted.
        streaming_graph_targets = resolve_streaming_cuda_graph_targets(diar_model)
        captured_modules = ", ".join(
            f"{name} ({type(module).__name__})" for name, module in streaming_graph_targets.items()
        )
        logging.info(
            "Streaming encoder CUDA Graphs requested: applying torch.compile with "
            f"{resolve_encoder_compile_kwargs(cfg)} to {captured_modules}, using the "
            f"'{cfg.compile_backend}' backend. Every streaming step calls each of them once with the fixed "
            "input shape that async_pad_to_max=True produces, and the per-step "
            f"{STREAMING_CUDA_GRAPH_LENGTH_ARGUMENT} values are copied into one retained buffer per shape before "
            "the captured call, so both the fixed shape and stable length storage that capture requires are "
            "established; the separately invoked encoder.pre_encode() runs outside the captured encoder forward, "
            "and the private FlexAttention compile boundary is left unchanged. Each streaming step is preceded by "
            "exactly one explicit torch.compiler.cudagraph_mark_step_begin() marker at the "
            f"{type(diar_model).__name__}.{STREAMING_CUDA_GRAPH_STEP_BOUNDARY}() boundary, and no marker is "
            "installed on the outer model forward. Whether PyTorch actually captures a graph is reported by "
            "PyTorch at runtime, measured rather than implied by this configuration."
        )

    if cfg.compile_encoder:
        compile_kwargs = resolve_encoder_compile_kwargs(cfg)
        logging.info(f"Encoder compile backend '{cfg.compile_backend}' resolved to torch.compile({compile_kwargs})")
        logging.info(f"Compiling the frontend encoder with {compile_kwargs}")
        diar_model.encoder = torch.compile(diar_model.encoder, **compile_kwargs)
        if diar_model.transformer_encoder is not None and len(diar_model.transformer_encoder.layers) > 0:
            logging.info(f"Compiling the optional transformer encoder with {compile_kwargs}")
            diar_model.transformer_encoder = torch.compile(diar_model.transformer_encoder, **compile_kwargs)

    # Installed at the single per-step boundary, after compilation, and only for the streaming graph mode.
    install_streaming_cuda_graph_boundary(cfg, diar_model)

    if cfg.compile_cuda_graphs:
        # Installed before the profiler so that the profiler wraps the marked forward and each model forward still
        # begins exactly one CUDA Graph iteration, including the direct self.forward calls made by test_batch().
        install_cuda_graph_step_marker(diar_model)

    postprocessing_cfg = load_postprocessing_from_yaml(cfg.postprocessing_yaml)
    tensor_path, model_id, tensor_filename = get_tensor_path(cfg)
    cfg.optuna_study_name = f"__{model_id}_{tensor_filename}"
    cfg.optuna_storage: str = f"sqlite:///{cfg.optuna_temp_dir}/{cfg.optuna_study_name}.db"
    cfg.optuna_log_file: str = f"{cfg.optuna_temp_dir}/{cfg.optuna_study_name}.log"
    inference_profiler = (
        InferenceProfiler(diar_model, warmup_calls=cfg.profiler_warmup_calls) if cfg.profile_inference else None
    )
    if inference_profiler is not None:
        inference_profiler.install()

    prediction_cache_metadata = (
        get_prediction_cache_metadata(cfg, diar_model, infer_audio_rttm_dict) if tensor_path is not None else None
    )
    if prediction_cache_metadata is not None:
        # Quantized predictions are not interchangeable with BF16 ones, so they must not share a cache entry.
        prediction_cache_metadata["quantization"] = prediction_cache_identity(quantization_config)
        # Likewise for the attention backend: every backend is close to, but not identical with, the default.
        prediction_cache_metadata["attention_backend"] = attention_backend_cache_identity(cfg.attention_backend)

    reuse_prediction_cache = (
        tensor_path is not None and os.path.exists(tensor_path) and not cfg.overwrite_preds_tensors
    )
    validate_calibration_forward_pass(quantization_config, reuse_prediction_cache)

    if reuse_prediction_cache:
        logging.info(
            f"A saved prediction tensor has been found. Loading the saved prediction tensors from {tensor_path}..."
        )
        diar_model_preds_total_list = load_prediction_tensors(tensor_path, prediction_cache_metadata)
    else:
        logging.info("No saved prediction tensors found. Running inference on the dataset...")
        activation_collector = ActivationAmaxCollector(diar_model) if quantization_config.calibration_output else None
        try:
            if activation_collector is not None:
                activation_collector.install()
            with torch.inference_mode(), torch.autocast(device_type=diar_model.device.type, dtype=diar_model.dtype):
                diar_model.test_batch()
        finally:
            if activation_collector is not None:
                activation_collector.remove()
        if activation_collector is not None:
            calibration_path = activation_collector.save(
                quantization_config.calibration_output,
                overwrite=quantization_config.overwrite_calibration,
                metadata={
                    "model_path": str(cfg.model_path),
                    "dataset_manifest": str(cfg.dataset_manifest),
                    "precision": str(cfg.precision),
                    "batch_size": int(cfg.batch_size),
                    "streaming_mode": bool(diar_model.streaming_mode),
                    "output_subsampling_factor": int(cfg.output_subsampling_factor),
                },
            )
            logging.info(f"Activation calibration saved to {calibration_path}")

        diar_model_preds_total_list = diar_model.preds_total_list
        if inference_profiler is not None:
            audio_duration = sum(float(item['duration']) for item in infer_audio_rttm_dict.values())
            inference_profiler.log_summary(audio_duration, measured_audio_duration=cfg.measured_audio_duration)
        if tensor_path is not None:
            save_prediction_tensors(tensor_path, diar_model.preds_total_list, prediction_cache_metadata)
            logging.info(f"Prediction tensors saved to {tensor_path}")

    if cfg.launch_pp_optim:
        # Launch a hyperparameter optimization process if launch_pp_optim is True
        run_optuna_hyperparam_search(
            cfg=cfg,
            postprocessing_cfg=postprocessing_cfg,
            infer_audio_rttm_dict=infer_audio_rttm_dict,
            preds_list=diar_model_preds_total_list,
            temp_out_dir=cfg.optuna_temp_dir,
            unit_10ms_frame_count=cfg.output_subsampling_factor,
        )

    # Evaluation
    if not cfg.no_der:
        if cfg.out_rttm_dir is not None and not os.path.exists(cfg.out_rttm_dir):
            os.mkdir(cfg.out_rttm_dir)

        logging.info("Running offline diarization evaluation...")
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
            infer_audio_rttm_dict,
            postprocessing_cfg=postprocessing_cfg,
            batch_preds_list=diar_model_preds_total_list,
            unit_10ms_frame_count=cfg.output_subsampling_factor,
            bypass_postprocessing=cfg.bypass_postprocessing,
            out_rttm_dir=cfg.out_rttm_dir,
        )
        logging.info(f"Evaluating the model on the {len(diar_model_preds_total_list)} audio segments...")
        score_labels(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=cfg.collar,
            ignore_overlap=cfg.ignore_overlap,
        )
        logging.info(f"PostProcessingParams: {postprocessing_cfg}")

    # clean-up
    if cfg.presort_manifest is not None:
        if remove_path_after_done is not None:
            os.unlink(remove_path_after_done)


if __name__ == '__main__':
    main()

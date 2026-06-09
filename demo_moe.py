"""
Wan2.2-TI2V-5B 版本的 Point Prompting 演示脚本。

这个脚本仅使用 WanImageToVideoPipeline 的官方 TI2V/I2V
expand_timesteps 路径执行首帧条件、加噪、去噪和 scheduler。

核心反事实引导公式：
    v_guided = (lam + 1) * v_marked - lam * v_original

其中 marked 条件来自带红点首帧，original 条件来自无红点首帧。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from color_rebalance import rebalance_video
from marker import insert_marker, track_marker_sequence


DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
OFFICIAL_SCHEDULER_CLASS = "UniPCMultistepScheduler"
OFFICIAL_FLOW_SHIFT = 5.0
OFFICIAL_MAX_SEQUENCE_LENGTH = 512
OFFICIAL_LANDSCAPE_SIZE = (1280, 704)
OFFICIAL_PORTRAIT_SIZE = (704, 1280)
T4_LANDSCAPE_SIZE = (832, 480)
T4_PORTRAIT_SIZE = (480, 832)


@dataclass
class FirstFrameCondition:
    latents: torch.Tensor
    condition: torch.Tensor
    first_frame_mask: torch.Tensor


@dataclass
class CounterfactualConditions:
    latents: torch.Tensor
    marked: FirstFrameCondition
    original: FirstFrameCondition
    diff_mean: float
    diff_max: float


@dataclass
class PointResult:
    tracks: np.ndarray
    visible: np.ndarray
    generated_frames: list
    stage_summaries: list


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_vae_internal_cache(pipe) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    for name in ("clear_cache", "_clear_cache", "clear_context_parallel_cache"):
        fn = getattr(vae, name, None)
        if callable(fn):
            try:
                fn()
            except TypeError:
                pass
    for attr in ("_feat_map", "_features", "_cache"):
        if hasattr(vae, attr):
            try:
                setattr(vae, attr, None)
            except Exception:
                pass


def _config_value(config, name, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _resolve_cuda_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda:0")
    return dev


def resolve_vae_device(device: str, vae_device: str) -> torch.device:
    main_dev = _resolve_cuda_device(device)
    if vae_device == "auto":
        if main_dev.type == "cuda" and torch.cuda.device_count() > 1:
            return torch.device("cuda:1" if main_dev.index == 0 else "cuda:0")
        return main_dev
    return _resolve_cuda_device(vae_device)


def parse_torch_dtype(dtype_name: str, device: str = "cuda") -> torch.dtype:
    name = dtype_name.lower()
    if name == "auto":
        dev = _resolve_cuda_device(device)
        return torch.float16 if dev.type == "cuda" else torch.float32
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError("--vae-dtype must be one of: auto, float32, float16, bfloat16")


def _pipe_device(pipe) -> torch.device:
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        try:
            return next(transformer.parameters()).device
        except StopIteration:
            pass
    d = getattr(pipe, "_execution_device", None)
    if d is not None:
        return torch.device(d)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _transformer_dtype(pipe) -> torch.dtype:
    return getattr(pipe.transformer, "dtype", torch.float32)


def _vae_device(pipe) -> torch.device:
    return next(pipe.vae.parameters()).device


def _vae_dtype(pipe) -> torch.dtype:
    return next(pipe.vae.parameters()).dtype


def _vae_norm(pipe, device) -> tuple:
    mean = torch.tensor(
        pipe.vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(
        pipe.vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    return mean, std


def _component_to(component, *args, **kwargs) -> None:
    if component is not None and hasattr(component, "to"):
        component.to(*args, **kwargs)


def scheduler_info(pipe) -> dict:
    config = getattr(getattr(pipe, "scheduler", None), "config", None)
    return {
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config_class": _config_value(config, "_class_name"),
        "scheduler_flow_shift": _config_value(config, "flow_shift"),
        "scheduler_prediction_type": _config_value(config, "prediction_type"),
        "scheduler_use_flow_sigmas": _config_value(config, "use_flow_sigmas"),
        "scheduler_timestep_spacing": _config_value(config, "timestep_spacing"),
        "scheduler_solver_order": _config_value(config, "solver_order"),
        "scheduler_solver_type": _config_value(config, "solver_type"),
        "scheduler_num_train_timesteps": _config_value(config, "num_train_timesteps"),
    }


def tensor_head_tail(tensor: torch.Tensor, count: int = 5) -> tuple:
    values = [float(x) for x in tensor.detach().float().cpu().tolist()]
    return values[:count], values[-count:]


def print_scheduler_info(pipe) -> dict:
    info = scheduler_info(pipe)
    print(
        "[scheduler] "
        f"class={info['scheduler_class']} source=checkpoint_config "
        f"flow_shift={info['scheduler_flow_shift']} "
        f"prediction_type={info['scheduler_prediction_type']} "
        f"use_flow_sigmas={info['scheduler_use_flow_sigmas']} "
        f"timestep_spacing={info['scheduler_timestep_spacing']} "
        f"solver_order={info['scheduler_solver_order']} "
        f"solver_type={info['scheduler_solver_type']} "
        f"num_train_timesteps={info['scheduler_num_train_timesteps']}"
    )
    return {**info, "scheduler_source": "checkpoint_config"}


def validate_wan22_ti2v5b_pipeline(pipe) -> None:
    if pipe.__class__.__name__ != "WanImageToVideoPipeline":
        raise TypeError(
            "demo_moe.py requires WanImageToVideoPipeline, "
            f"got {pipe.__class__.__name__}"
        )
    required = ("transformer", "vae", "scheduler", "video_processor", "prepare_latents")
    missing = [name for name in required if not hasattr(pipe, name)]
    if missing:
        raise TypeError(f"Wan2.2-TI2V-5B pipeline is missing components: {missing}")
    if not bool(_config_value(getattr(pipe, "config", None), "expand_timesteps", False)):
        raise ValueError("Wan2.2-TI2V-5B requires pipe.config.expand_timesteps=True")

    info = scheduler_info(pipe)
    expected = {
        "scheduler_class": OFFICIAL_SCHEDULER_CLASS,
        "scheduler_flow_shift": OFFICIAL_FLOW_SHIFT,
        "scheduler_prediction_type": "flow_prediction",
        "scheduler_use_flow_sigmas": True,
        "scheduler_timestep_spacing": "linspace",
    }
    mismatches = {
        name: (info.get(name), value)
        for name, value in expected.items()
        if info.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Pipeline scheduler does not match official Wan2.2-TI2V-5B config: "
            f"{mismatches}"
        )

    patch_size = tuple(getattr(pipe.transformer.config, "patch_size", ()))
    if len(patch_size) < 3 or tuple(patch_size[-2:]) != (2, 2):
        raise ValueError(
            "Wan2.2-TI2V-5B expanded timestep path requires spatial patch_size=(2, 2), "
            f"got {patch_size}"
        )


def resolve_max_sequence_length(pipe, requested) -> int:
    if requested is not None:
        return int(requested)
    for config in (getattr(pipe, "config", None), getattr(pipe.transformer, "config", None)):
        value = _config_value(config, "max_sequence_length")
        if value is None:
            value = _config_value(config, "max_text_seq_len")
        if value is not None:
            return int(value)
    return OFFICIAL_MAX_SEQUENCE_LENGTH


def load_wan_ti2v_pipe(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    vae_device: str = "auto",
    vae_dtype: str = "float32",
    low_cpu_memory: bool = True,
):
    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    except ImportError as exc:
        raise ImportError(
            "demo_moe.py requires a Diffusers build with Wan2.2 TI2V/I2V support. "
            "Install the current Diffusers main branch."
        ) from exc

    vae_torch_dtype = parse_torch_dtype(vae_dtype, device=device)
    loading_kwargs = {"low_cpu_mem_usage": low_cpu_memory}
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_torch_dtype,
            **loading_kwargs,
        )
    except ValueError as exc:
        if "low_cpu_mem_usage" not in str(exc):
            raise
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_torch_dtype,
            low_cpu_mem_usage=True,
        )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        **loading_kwargs,
    )

    dev = _resolve_cuda_device(device)
    vae_dev = resolve_vae_device(device, vae_device)
    if dev.type == "cuda":
        _component_to(getattr(pipe, "transformer", None), dev)
        _component_to(getattr(pipe, "vae", None), vae_dev, dtype=vae_torch_dtype)
        _component_to(getattr(pipe, "text_encoder", None), torch.device("cpu"))
    else:
        if hasattr(pipe, "to"):
            pipe.to(dev)
        _component_to(getattr(pipe, "vae", None), vae_dev, dtype=vae_torch_dtype)

    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    pipe._low_memory_text_encoder = low_cpu_memory
    validate_wan22_ti2v5b_pipeline(pipe)
    return pipe


def frames_to_tensor(frames_bgr: list, device, dtype) -> torch.Tensor:
    rgb = np.stack([frame[..., ::-1] for frame in frames_bgr], axis=0)
    tensor = torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype) / 127.5 - 1.0
    return tensor


def tensor_to_frames(tensor: torch.Tensor) -> list:
    tensor = tensor.detach().float().cpu().clamp(-1, 1)
    tensor = ((tensor + 1.0) * 127.5).round().to(torch.uint8)
    array = tensor[0].permute(1, 2, 3, 0).numpy()
    return [frame[..., ::-1].copy() for frame in array]


def _retrieve_latents_argmax(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        dist = encoder_output.latent_dist
        if hasattr(dist, "mode"):
            return dist.mode()
        return dist.mean
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder output")


def encode_video_official(pipe, frames_bgr: list) -> torch.Tensor:
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    tensor = frames_to_tensor(frames_bgr, vae_dev, vae_dtype)
    with torch.no_grad():
        latents = _retrieve_latents_argmax(pipe.vae.encode(tensor))
    mean, std = _vae_norm(pipe, latents.device)
    latents = ((latents.float() - mean) * std).to(_transformer_dtype(pipe))
    latents = latents.to(device=_pipe_device(pipe))
    clear_vae_internal_cache(pipe)
    release_memory()
    return latents


def decode_latents_official(pipe, latents: torch.Tensor) -> list:
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    mean, std = _vae_norm(pipe, latents.device)
    latents = (latents.float() / std + mean).to(device=vae_dev, dtype=vae_dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents).sample
    frames = tensor_to_frames(decoded)
    del decoded, latents
    clear_vae_internal_cache(pipe)
    release_memory()
    return frames


def encode_prompt_official(pipe, prompt: str, max_sequence_length: int):
    device = _pipe_device(pipe)
    dtype = _transformer_dtype(pipe)
    if hasattr(pipe, "encode_prompt") and not getattr(pipe, "_low_memory_text_encoder", False):
        prompt_embeds, _negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        return prompt_embeds.to(device=device, dtype=dtype)

    if not hasattr(pipe, "_get_t5_prompt_embeds"):
        raise RuntimeError("Pipeline does not expose encode_prompt or _get_t5_prompt_embeds")

    return pipe._get_t5_prompt_embeds(
        prompt=prompt,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=torch.device("cpu"),
        dtype=getattr(getattr(pipe, "text_encoder", None), "dtype", torch.float32),
    ).to(device=device, dtype=dtype)


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[..., ::-1].copy())


def prepare_wan22_first_frame_condition(
    pipe,
    first_frame_bgr: np.ndarray,
    noisy_latents: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    generator,
) -> FirstFrameCondition:
    transformer_device = _pipe_device(pipe)
    vae_device = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    dtype = _transformer_dtype(pipe)
    image = pipe.video_processor.preprocess(
        _bgr_to_pil(first_frame_bgr), height=height, width=width
    )
    image = image.to(device=vae_device, dtype=vae_dtype)

    outputs = pipe.prepare_latents(
        image,
        batch_size=1,
        num_channels_latents=pipe.vae.config.z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=vae_dtype,
        device=vae_device,
        generator=generator,
        latents=noisy_latents.to(device=vae_device, dtype=vae_dtype),
    )
    if len(outputs) != 3:
        raise ValueError(
            "Wan2.2-TI2V-5B official expand_timesteps path must return "
            f"(latents, condition, first_frame_mask), got {len(outputs)} values"
        )
    prepared_latents, condition, first_frame_mask = outputs
    prepared_latents = prepared_latents.to(device=transformer_device, dtype=dtype)
    condition = condition.to(device=transformer_device, dtype=dtype)
    first_frame_mask = first_frame_mask.to(device=transformer_device, dtype=dtype)

    if prepared_latents.shape != noisy_latents.shape:
        raise ValueError(
            "prepare_latents changed noisy latent shape: "
            f"input={tuple(noisy_latents.shape)} output={tuple(prepared_latents.shape)}"
        )
    try:
        torch.broadcast_shapes(first_frame_mask.shape, prepared_latents.shape)
    except RuntimeError as exc:
        raise ValueError(
            "Wan2.2-TI2V first_frame_mask must broadcast to latent shape: "
            f"mask={tuple(first_frame_mask.shape)} latents={tuple(prepared_latents.shape)}"
        ) from exc

    clear_vae_internal_cache(pipe)
    release_memory()
    return FirstFrameCondition(prepared_latents, condition, first_frame_mask)


def prepare_counterfactual_conditions(
    pipe,
    marked_frame_bgr: np.ndarray,
    original_frame_bgr: np.ndarray,
    noisy_latents: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
) -> CounterfactualConditions:
    gen_marked = torch.Generator(device="cpu").manual_seed(seed)
    gen_original = torch.Generator(device="cpu").manual_seed(seed)
    marked = prepare_wan22_first_frame_condition(
        pipe, marked_frame_bgr, noisy_latents, height, width, num_frames, gen_marked
    )
    original = prepare_wan22_first_frame_condition(
        pipe, original_frame_bgr, noisy_latents, height, width, num_frames, gen_original
    )
    if marked.latents.shape != original.latents.shape:
        raise ValueError("Marked and original prepared latents have different shapes")
    if marked.condition.shape != original.condition.shape:
        raise ValueError("Marked and original conditions have different shapes")
    try:
        torch.broadcast_shapes(marked.first_frame_mask.shape, marked.latents.shape)
        torch.broadcast_shapes(original.first_frame_mask.shape, original.latents.shape)
    except RuntimeError as exc:
        raise ValueError("First-frame masks must broadcast to latent shape") from exc
    if not torch.allclose(marked.first_frame_mask, original.first_frame_mask):
        raise ValueError("Marked and original first-frame masks differ")

    diff = (marked.condition.float() - original.condition.float()).abs()
    diff_mean = float(diff.mean().item())
    diff_max = float(diff.max().item())
    print(
        "[condition] Wan2.2 TI2V counterfactual "
        f"mean_abs_diff={diff_mean:.6f} max_abs_diff={diff_max:.6f}"
    )
    return CounterfactualConditions(
        latents=marked.latents,
        marked=marked,
        original=original,
        diff_mean=diff_mean,
        diff_max=diff_max,
    )


def scheduler_step_official(pipe, noise_pred, timestep, latents):
    return pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]


def model_cache_context(model, name: str):
    if hasattr(model, "cache_context"):
        return model.cache_context(name)
    return nullcontext()


def _expanded_timestep(model, first_frame_mask, timestep, batch_size):
    patch_size = getattr(getattr(model, "config", None), "patch_size", (1, 2, 2))
    patch_h = int(patch_size[1]) if len(patch_size) > 1 else 2
    patch_w = int(patch_size[2]) if len(patch_size) > 2 else 2
    token_mask = first_frame_mask[0][0][:, ::patch_h, ::patch_w]
    temp_ts = (token_mask * timestep).flatten()
    return temp_ts.unsqueeze(0).expand(batch_size, -1).to(first_frame_mask.device)


def _transformer_forward(
    model,
    latent_model_input,
    timestep_batch,
    prompt_embeds,
    attention_kwargs,
    cache_name,
):
    with torch.no_grad(), model_cache_context(model, cache_name):
        return model(
            hidden_states=latent_model_input,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]


def denoise_counterfactual_ti2v(
    pipe,
    latents: torch.Tensor,
    timesteps_run: torch.Tensor,
    conditions: CounterfactualConditions,
    prompt_embeds: torch.Tensor,
    lam: float,
    attention_kwargs=None,
) -> torch.Tensor:
    model = pipe.transformer
    transformer_dtype = _transformer_dtype(pipe)
    mask = conditions.marked.first_frame_mask
    cond_marked = conditions.marked.condition
    cond_original = conditions.original.condition

    for i, timestep in enumerate(timesteps_run):
        timestep_batch = _expanded_timestep(model, mask, timestep, latents.shape[0])
        input_marked = ((1 - mask) * cond_marked + mask * latents).to(
            dtype=transformer_dtype
        )
        input_original = ((1 - mask) * cond_original + mask * latents).to(
            dtype=transformer_dtype
        )
        v_marked = _transformer_forward(
            model, input_marked, timestep_batch, prompt_embeds, attention_kwargs, "marked"
        )
        v_original = _transformer_forward(
            model, input_original, timestep_batch, prompt_embeds, attention_kwargs, "original"
        )
        v_guided = (float(lam) + 1.0) * v_marked - float(lam) * v_original
        latents = scheduler_step_official(pipe, v_guided, timestep, latents)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            print(
                f"[ti2v point denoise {i + 1:3d}/{len(timesteps_run)}] "
                f"t={timestep.item():.1f} latent_norm={latents.norm().item():.1f}"
            )

    return (1 - mask) * cond_marked + mask * latents


def noise_strength_to_start_idx(noise_strength: float, n_steps: int) -> int:
    if not 0.0 <= noise_strength <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {noise_strength}")
    if n_steps <= 0:
        raise ValueError("scheduler_steps must be positive")
    return min(n_steps - 1, max(0, int(round((1.0 - noise_strength) * (n_steps - 1)))))


def denoise_step_count(noise_strength: float, n_steps: int) -> int:
    return n_steps - noise_strength_to_start_idx(noise_strength, n_steps)


def set_denoise_start(pipe, n_steps: int, start_idx: int) -> torch.Tensor:
    pipe.scheduler.set_timesteps(n_steps, device=_pipe_device(pipe))
    timesteps = pipe.scheduler.timesteps
    start_idx = max(0, min(int(start_idx), len(timesteps) - 1))
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(start_idx)
    return timesteps[start_idx:]


def add_noise_at_timestep(pipe, latents, noise, timestep):
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    return pipe.scheduler.add_noise(latents, noise, timestep.to(device=latents.device))


def run_counterfactual_sdedit(
    pipe,
    frames_bgr_clean: list,
    marked_frame_bgr: np.ndarray,
    original_frame_bgr: np.ndarray,
    gamma: float,
    lam: float,
    scheduler_steps: int,
    prompt_embeds: torch.Tensor,
    seed: int,
    decode_noisy: bool = False,
    debug_dir: Optional[str] = None,
    fps: float = 8.0,
) -> tuple:
    height, width = frames_bgr_clean[0].shape[:2]
    latents_clean = encode_video_official(pipe, frames_bgr_clean)
    start_idx = noise_strength_to_start_idx(gamma, scheduler_steps)
    timesteps_run = set_denoise_start(pipe, scheduler_steps, start_idx)
    t_start = pipe.scheduler.timesteps[start_idx]
    if gamma == 0.0:
        timesteps_run = timesteps_run[:0]
        noisy_latents = latents_clean.clone()
    else:
        torch.manual_seed(seed)
        noise = torch.randn_like(latents_clean)
        noisy_latents = add_noise_at_timestep(pipe, latents_clean, noise, t_start)

    all_head, all_tail = tensor_head_tail(pipe.scheduler.timesteps)
    run_head, run_tail = tensor_head_tail(timesteps_run)
    print(
        f"[scheduler gamma={gamma:.3f}] steps={scheduler_steps} "
        f"start_idx={start_idx} t_start={float(t_start.item()):.1f} "
        f"denoise_steps={len(timesteps_run)} "
        f"timesteps_head={all_head} timesteps_tail={all_tail} "
        f"run_head={run_head} run_tail={run_tail}"
    )

    conditions = prepare_counterfactual_conditions(
        pipe,
        marked_frame_bgr,
        original_frame_bgr,
        noisy_latents,
        height,
        width,
        len(frames_bgr_clean),
        seed,
    )
    noisy_latents = conditions.latents
    if decode_noisy and debug_dir is not None:
        save_video(decode_latents_official(pipe, noisy_latents), os.path.join(debug_dir, "noisy.mp4"), fps)
    elif not decode_noisy:
        print("[decode] skipping noisy.mp4 to keep VAE memory for denoising")

    denoised_latents = denoise_counterfactual_ti2v(
        pipe,
        noisy_latents.clone(),
        timesteps_run,
        conditions,
        prompt_embeds,
        lam,
    )
    frames_out = decode_latents_official(pipe, denoised_latents)
    summary = {
        "gamma": gamma,
        "lam": lam,
        "start_idx": start_idx,
        "t_start": float(t_start.item()),
        "denoise_steps": len(timesteps_run),
        "condition_diff_mean": conditions.diff_mean,
        "condition_diff_max": conditions.diff_max,
        "video_latent_shape": list(latents_clean.shape),
        "transformer_latent_shape": list(noisy_latents.shape),
        "condition_shape": list(conditions.marked.condition.shape),
        "first_frame_mask_shape": list(conditions.marked.first_frame_mask.shape),
        "reference_latent_slots": 0,
        "timesteps_head": all_head,
        "timesteps_tail": all_tail,
        "timesteps_run_head": run_head,
        "timesteps_run_tail": run_tail,
        "decode_noisy": decode_noisy,
    }
    del latents_clean, noisy_latents, denoised_latents, conditions
    release_memory()
    return frames_out, summary


def load_video(path: str, max_frames: Optional[int] = None) -> tuple:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames, float(fps)


def save_video(frames: list, path: str, fps: float) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio

        imageio.mimsave(
            path,
            [frame[..., ::-1] for frame in frames],
            fps=fps,
            codec="libx264",
            output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
        )
        print(f"[save] {path} ({len(frames)} frames, imageio/libx264)")
        return
    except Exception as exc:
        print(f"[save] imageio failed ({exc}), fallback to cv2")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[save] {path} ({len(frames)} frames, cv2/mp4v)")


def resize_video(frames: list, width: int, height: int) -> list:
    if width <= 0 or height <= 0:
        return frames
    if not frames or (frames[0].shape[1] == width and frames[0].shape[0] == height):
        return frames
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]


def scale_points(points: list, src_size: tuple, dst_size: tuple) -> list:
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return [(x * sx, y * sy) for x, y in points]


def _aligned_size(width: int, height: int, max_width: int, max_height: int, stride: int) -> Tuple[int, int]:
    if max_width <= 0 or max_height <= 0:
        return width, height
    scale = min(max_width / width, max_height / height, 1.0)
    out_w = int(width * scale)
    out_h = int(height * scale)
    if stride > 1:
        out_w = max(stride, (out_w // stride) * stride)
        out_h = max(stride, (out_h // stride) * stride)
    return out_w, out_h


def official_ti2v_size_for_aspect(width: int, height: int) -> Tuple[int, int]:
    if height > width:
        return OFFICIAL_PORTRAIT_SIZE
    return OFFICIAL_LANDSCAPE_SIZE


def t4_ti2v_size_for_aspect(width: int, height: int) -> Tuple[int, int]:
    if height > width:
        return T4_PORTRAIT_SIZE
    return T4_LANDSCAPE_SIZE


def default_ti2v_size_for_preset(preset: str, width: int, height: int) -> Tuple[int, int]:
    if preset == "official":
        return official_ti2v_size_for_aspect(width, height)
    if preset == "t4":
        return t4_ti2v_size_for_aspect(width, height)
    if preset == "custom":
        return t4_ti2v_size_for_aspect(width, height)
    raise ValueError("--resolution-preset must be one of: t4, official, custom")


def draw_tracks(frames: list, tracks_list: list, visible_list: list) -> list:
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    out = []
    for t, frame in enumerate(frames):
        vis_frame = frame.copy()
        for i, (track, visible) in enumerate(zip(tracks_list, visible_list)):
            if t >= len(track):
                continue
            color = colors[i % len(colors)] if visible[t] else (128, 128, 128)
            x, y = int(round(track[t, 0])), int(round(track[t, 1]))
            cv2.circle(vis_frame, (x, y), 5, color, -1)
            cv2.putText(
                vis_frame,
                str(i),
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        out.append(vis_frame)
    return out


def parse_point(token: str) -> Tuple[float, float]:
    try:
        x_str, y_str = token.split(",", 1)
        return float(x_str), float(y_str)
    except Exception as exc:
        raise ValueError(f"Invalid point '{token}', expected 'x,y'") from exc


def parse_points(values: list) -> list:
    tokens = []
    for value in values:
        tokens.extend(str(value).split())
    return [parse_point(token) for token in tokens]


def run_point(
    pipe,
    frames_bgr: list,
    query_point: Tuple[float, float],
    args,
    prompt_embeds: torch.Tensor,
    fps: float,
    point_index: int,
) -> PointResult:
    orig_h, orig_w = frames_bgr[0].shape[:2]
    model_w, model_h = _aligned_size(
        orig_w, orig_h, args.model_width, args.model_height, args.model_stride
    )
    frames_model = resize_video(frames_bgr, model_w, model_h)
    sx = model_w / orig_w
    sy = model_h / orig_h
    query_model = (query_point[0] * sx, query_point[1] * sy)
    marker_radius = max(2, int(round(args.marker_radius * min(sx, sy))))

    protect_r = marker_radius * 4
    frames_rb = rebalance_video(
        frames_model,
        protect_point=query_model,
        protect_radius=protect_r,
    )
    frame0_original = frames_rb[0]
    frame0_marked = insert_marker(frame0_original, query_model, marker_radius)
    frames_edited = [frame0_marked] + frames_rb[1:]
    total_stages = 1 if args.no_refine else 2

    point_dir = os.path.join(args.output_dir, f"point_{point_index:02d}")
    os.makedirs(point_dir, exist_ok=True)
    print(
        f"\n[point {point_index}] query={query_model} model_size={model_w}x{model_h} "
        f"marker_radius={marker_radius}"
    )
    denoise_steps = denoise_step_count(args.gamma, args.scheduler_steps)
    print(
        f"  [阶段 1/{total_stages}] TI2V 反事实 SDEdit "
        f"gamma={args.gamma} lam={args.lam} denoise_steps={denoise_steps}"
    )
    generated, stage1 = run_counterfactual_sdedit(
        pipe,
        frames_edited,
        frame0_marked,
        frame0_original,
        gamma=args.gamma,
        lam=args.lam,
        scheduler_steps=args.scheduler_steps,
        prompt_embeds=prompt_embeds,
        seed=args.seed + point_index,
        decode_noisy=args.decode_noisy,
        debug_dir=os.path.join(point_dir, "stage1"),
        fps=fps,
    )
    tracks, visible = track_marker_sequence(generated, query_model)
    stage1["stage"] = "sdedit"
    stage1["visible_count"] = int(visible.sum())
    stage1["visible_ratio"] = float(visible.mean()) if len(visible) else 0.0
    print(f"  [阶段 1/{total_stages}] 完成，可见帧 {visible.sum()}/{len(visible)}")

    summaries = [stage1]
    if not args.no_refine:
        refine_steps = denoise_step_count(args.refine_gamma, args.scheduler_steps)
        print(
            f"  [阶段 2/2] TI2V 二次保守 SDEdit "
            f"gamma={args.refine_gamma} lam={args.lam} denoise_steps={refine_steps}"
        )
        refined, stage2 = run_counterfactual_sdedit(
            pipe,
            generated,
            generated[0],
            frame0_original,
            gamma=args.refine_gamma,
            lam=args.lam,
            scheduler_steps=args.scheduler_steps,
            prompt_embeds=prompt_embeds,
            seed=args.seed + point_index + 10000,
            decode_noisy=args.decode_noisy,
            debug_dir=os.path.join(point_dir, "stage2"),
            fps=fps,
        )
        tracks, visible = track_marker_sequence(refined, query_model)
        generated = refined
        stage2["stage"] = "refine"
        stage2["visible_count"] = int(visible.sum())
        stage2["visible_ratio"] = float(visible.mean()) if len(visible) else 0.0
        summaries.append(stage2)
        print(f"  [阶段 2/2] 完成，可见帧 {visible.sum()}/{len(visible)}")

    if sx != 1.0 or sy != 1.0:
        tracks = tracks.copy()
        tracks[:, 0] /= sx
        tracks[:, 1] /= sy

    if args.save_generated:
        gen_path = os.path.join(point_dir, "generated.mp4")
        save_video(generated, gen_path, fps)
        overlay = draw_tracks(generated, [track_marker_sequence(generated, query_model)[0]], [visible])
        save_video(overlay, os.path.join(point_dir, "generated_overlay.mp4"), fps)

    return PointResult(tracks, visible, generated, summaries)


def cuda_preflight_error(device: str) -> Optional[str]:
    if not str(device).startswith("cuda"):
        return None
    if torch.cuda.is_available():
        return None
    return (
        "错误：当前 PyTorch 环境不可用 CUDA，但 --device 指向 cuda。"
        "请在 GPU 环境运行，或显式传入 --device cpu 做功能测试。"
    )


def cuda_oom_hint(args) -> str:
    return (
        "CUDA out of memory。Wan2.2-TI2V-5B 官方 720P 为 "
        "1280x704/704x1280，单卡官方建议约 24GB VRAM；T4 15GB 很容易在 "
        "VAE prepare_latents 阶段 OOM。\n"
        "建议先使用默认 --resolution-preset t4，或显式传入：\n"
        "  --resolution-preset t4 --vae-dtype auto\n"
        "如果仍 OOM，再降到：\n"
        "  --preprocess-width 720 --preprocess-height 416 "
        "--model-width 720 --model-height 416 --vae-dtype float16\n"
        f"当前设置：resolution_preset={args.resolution_preset}, "
        f"preprocess={args.preprocess_width}x{args.preprocess_height}, "
        f"model={args.model_width}x{args.model_height}, vae_dtype={args.vae_dtype}"
    )


def run_demo(args) -> dict:
    if args.guidance_scale != 1.0:
        print("[warn] demo_moe.py 首版不叠加文本 CFG，--guidance-scale 会被记录但不参与去噪")
    if args.negative_prompt:
        print("[warn] 首版仅反事实引导，不使用 --negative-prompt")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    noise_strength_to_start_idx(args.gamma, args.scheduler_steps)
    noise_strength_to_start_idx(args.refine_gamma, args.scheduler_steps)

    query_points = parse_points(args.points)
    if not query_points:
        raise ValueError("--points 至少需要一个有效坐标，例如 --points \"320,240\"")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载视频：{args.video}")
    frames, fps = load_video(args.video, args.max_frames)
    if not frames:
        raise RuntimeError("无法从视频中读取任何帧")
    orig_w, orig_h = frames[0].shape[1], frames[0].shape[0]
    print(f"  {len(frames)} 帧  分辨率 {orig_w}x{orig_h}  fps={fps:.2f}")
    if (len(frames) - 1) % 4 != 0:
        good = ((len(frames) - 1) // 4) * 4 + 1
        good = max(5, good)
        print(f"  帧数 {len(frames)} 不满足 T=4k+1，自动裁剪到 {good} 帧")
        frames = frames[:good]

    frames_orig = [frame.copy() for frame in frames]
    official_w, official_h = official_ti2v_size_for_aspect(orig_w, orig_h)
    preset_w, preset_h = default_ti2v_size_for_preset(
        args.resolution_preset, orig_w, orig_h
    )
    if args.preprocess_width is None:
        args.preprocess_width = preset_w
    if args.preprocess_height is None:
        args.preprocess_height = preset_h
    if args.model_width is None:
        args.model_width = preset_w
    if args.model_height is None:
        args.model_height = preset_h
    print(
        f"  Wan2.2-TI2V-5B 官方 720P 尺寸："
        f"{official_w}x{official_h}（横屏 1280x704 / 竖屏 704x1280）"
    )
    print(
        f"  当前分辨率预设：{args.resolution_preset}，默认处理尺寸："
        f"{args.preprocess_width}x{args.preprocess_height}"
    )
    pre_w, pre_h = orig_w, orig_h
    if args.preprocess_width > 0 and args.preprocess_height > 0:
        pre_w, pre_h = args.preprocess_width, args.preprocess_height
        frames = resize_video(frames, pre_w, pre_h)
        query_points = scale_points(query_points, (orig_w, orig_h), (pre_w, pre_h))
        print(f"  已预处理到 {pre_w}x{pre_h}")

    device_error = cuda_preflight_error(args.device)
    if device_error is not None:
        raise RuntimeError(device_error)

    print(f"加载模型：{args.model_id}")
    pipe = load_wan_ti2v_pipe(
        args.model_id,
        device=args.device,
        vae_device=args.vae_device,
        vae_dtype=args.vae_dtype,
        low_cpu_memory=args.low_memory,
    )
    validate_wan22_ti2v5b_pipeline(pipe)
    sched_info = print_scheduler_info(pipe)
    print(
        f"[model] Wan2.2-TI2V-5B transformer_device={_pipe_device(pipe)} "
        f"transformer_dtype={_transformer_dtype(pipe)} "
        f"vae_device={_vae_device(pipe)} vae_dtype={_vae_dtype(pipe)}"
    )

    max_sequence_length = resolve_max_sequence_length(pipe, args.max_sequence_length)
    prompt_embeds = encode_prompt_official(pipe, args.prompt, max_sequence_length)
    if args.low_memory and getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    release_memory()

    results = []
    for i, point in enumerate(query_points):
        result = run_point(pipe, frames, point, args, prompt_embeds, fps, i)
        results.append(result)
        release_memory()

    sx_inv = orig_w / pre_w
    sy_inv = orig_h / pre_h
    tracks_orig = []
    for result in results:
        tracks = result.tracks.copy()
        tracks[:, 0] *= sx_inv
        tracks[:, 1] *= sy_inv
        tracks_orig.append(tracks)

    for i, (result, track_orig) in enumerate(zip(results, tracks_orig)):
        print(f"\n点 {i} 坐标追踪（预处理{pre_w}x{pre_h} -> 原始{orig_w}x{orig_h}）：")
        for t in range(len(result.tracks)):
            px, py = result.tracks[t, 0], result.tracks[t, 1]
            ox, oy = track_orig[t, 0], track_orig[t, 1]
            vis = "可见" if result.visible[t] else "丢失"
            print(f"  t={t:02d} [{vis}]  预处理({px:.1f},{py:.1f}) -> 原始({ox:.1f},{oy:.1f})")

    n_gen = min(len(result.tracks) for result in results) if results else len(frames_orig)
    annotated = draw_tracks(
        frames_orig[:n_gen],
        [track[:n_gen] for track in tracks_orig],
        [result.visible[:n_gen] for result in results],
    )
    save_video(annotated, args.output, fps)

    summary = {
        "video": args.video,
        "output": args.output,
        "model_id": args.model_id,
        "model_family": "Wan2.2-TI2V-5B",
        "frames": len(frames),
        "preprocess_width": pre_w,
        "preprocess_height": pre_h,
        "model_width": args.model_width,
        "model_height": args.model_height,
        "resolution_preset": args.resolution_preset,
        "official_ti2v_resolution": f"{official_w}x{official_h}",
        "fps": fps,
        "seed": args.seed,
        "gamma": args.gamma,
        "refine_gamma": args.refine_gamma,
        "lam": args.lam,
        "scheduler_steps": args.scheduler_steps,
        "guidance_scale_recorded_only": args.guidance_scale,
        "negative_prompt_recorded_only": args.negative_prompt,
        "max_sequence_length": max_sequence_length,
        "reference_latent_slots": 0,
        "transformer_device": str(_pipe_device(pipe)),
        "transformer_dtype": str(_transformer_dtype(pipe)),
        "vae_device": str(_vae_device(pipe)),
        "vae_dtype": str(_vae_dtype(pipe)),
        **sched_info,
        "points": [
            {
                "index": i,
                "query_preprocessed": list(query_points[i]),
                "visible_count": int(result.visible.sum()),
                "visible_ratio": float(result.visible.mean()) if len(result.visible) else 0.0,
                "stages": result.stage_summaries,
            }
            for i, result in enumerate(results)
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\n已保存至 {args.output}")
    print(f"[result] summary written to {os.path.join(args.output_dir, 'summary.json')}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B Point Prompting demo")
    parser.add_argument("--video", required=True)
    parser.add_argument("--points", nargs="+", required=True)
    parser.add_argument("--output", default="tracked_moe.mp4")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-device", default="auto")
    parser.add_argument(
        "--vae-dtype",
        default="auto",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="VAE dtype. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=8.0)
    parser.add_argument("--scheduler-steps", type=int, default=100)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=9)
    parser.add_argument(
        "--resolution-preset",
        default="t4",
        choices=["t4", "official", "custom"],
        help=(
            "Default resolution policy. t4 uses 832x480/480x832 for 15GB GPUs; "
            "official uses Wan2.2 TI2V 720P 1280x704/704x1280; custom uses "
            "explicit --preprocess/--model sizes."
        ),
    )
    parser.add_argument(
        "--preprocess-width",
        type=int,
        default=None,
        help="Preprocess width. Default auto-selects official TI2V 720P width: 1280 or 704.",
    )
    parser.add_argument(
        "--preprocess-height",
        type=int,
        default=None,
        help="Preprocess height. Default auto-selects official TI2V 720P height: 704 or 1280.",
    )
    parser.add_argument(
        "--model-width",
        type=int,
        default=None,
        help="Model width cap. Default follows official TI2V 720P width.",
    )
    parser.add_argument(
        "--model-height",
        type=int,
        default=None,
        help="Model height cap. Default follows official TI2V 720P height.",
    )
    parser.add_argument("--model-stride", type=int, default=16)
    parser.add_argument("--marker-radius", type=int, default=6)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--refine-gamma", type=float, default=0.3)
    parser.add_argument("--save-generated", action="store_true")
    parser.add_argument("--output-dir", default="outputs/demo_moe")
    parser.add_argument("--decode-noisy", action="store_true")
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Disable reduced-CPU-RAM model loading and T5 release.",
    )
    parser.set_defaults(low_memory=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_demo(args)
    except torch.OutOfMemoryError as exc:
        sys.exit(f"错误：{cuda_oom_hint(args)}\n原始错误：{exc}")
    except Exception as exc:
        sys.exit(f"错误：{exc}")


if __name__ == "__main__":
    main()

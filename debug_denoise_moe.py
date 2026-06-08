"""
Wan2.2 TI2V/I2V SDEdit reconstruction experiment.

This mirrors debug_denoise.py outputs, but uses the official Wan
Image-to-Video latent condition path instead of VACE reference slots:
prepare_latents(image, latents=...), optional expand_timesteps mask fusion,
CFG, transformer / transformer_2 switching, and scheduler steps.

The default model is Wan-AI/Wan2.2-TI2V-5B-Diffusers. Despite this file name,
that checkpoint is a dense 5B TI2V model; the explicitly MoE Wan2.2 checkpoints
are the A14B series.

Example:
  python debug_denoise_moe.py --video input.mp4 --max-frames 9 --gammas 0.5
"""

import argparse
import csv
import gc
import inspect
import json
import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

from debug_denoise import (
    gamma_dir_name,
    json_metric,
    load_video,
    mean_psnr,
    save_comparison,
    save_frames,
    save_video,
)
from model_adapter import _frames_to_tensor, _tensor_to_frames, noise_strength_to_start_idx


DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


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


def _pipe_device(pipe) -> torch.device:
    for attr in ("transformer", "transformer_2"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            try:
                return next(mod.parameters()).device
            except StopIteration:
                pass
    d = getattr(pipe, "_execution_device", None)
    if d is not None:
        return torch.device(d)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _transformer_dtype(pipe) -> torch.dtype:
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_2", None)
    return getattr(transformer, "dtype", torch.float32)


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


def _retrieve_latents_argmax(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        dist = encoder_output.latent_dist
        if hasattr(dist, "mode"):
            return dist.mode()
        return dist.mean
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder output")


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[..., ::-1].copy())


def _component_to(component, *args, **kwargs) -> None:
    if component is not None and hasattr(component, "to"):
        component.to(*args, **kwargs)


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


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError("--vae-dtype must be one of: float32, float16, bfloat16")


def load_wan_ti2v_pipe(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    vae_device: str = "auto",
    vae_dtype: str = "float32",
    flow_shift: float = 3.0,
    low_cpu_memory: bool = True,
):
    try:
        from diffusers import AutoencoderKLWan, DiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "debug_denoise_moe.py requires a Diffusers build with Wan2.2 TI2V/I2V "
            "support. Install the current Diffusers main branch, for example: "
            "pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc

    try:
        from diffusers import WanImageToVideoPipeline
    except ImportError:
        WanImageToVideoPipeline = None

    try:
        from diffusers import UniPCMultistepScheduler
    except ImportError:
        UniPCMultistepScheduler = None

    vae_torch_dtype = parse_torch_dtype(vae_dtype)
    loading_kwargs = {"low_cpu_mem_usage": low_cpu_memory}
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_torch_dtype,
            **loading_kwargs,
        )
    except ValueError as exc:
        if "low_cpu_mem_usage" in str(exc):
            loading_kwargs = {"low_cpu_mem_usage": True}
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=vae_torch_dtype,
                **loading_kwargs,
            )
        else:
            raise

    first_exc = None
    if WanImageToVideoPipeline is not None:
        try:
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.bfloat16,
                **loading_kwargs,
            )
        except Exception as exc:
            first_exc = exc
            pipe = None
    else:
        pipe = None

    if pipe is None:
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.bfloat16,
                **loading_kwargs,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "Could not load Wan2.2 TI2V/I2V pipeline. This usually means the "
                "installed Diffusers version does not expose WanImageToVideoPipeline "
                "or the checkpoint components. Install Diffusers from main branch."
            ) from (first_exc or fallback_exc)

    if not hasattr(pipe, "prepare_latents"):
        raise RuntimeError(
            "Loaded pipeline does not expose prepare_latents; install a Diffusers "
            "version with official Wan TI2V/I2V support."
        )

    if UniPCMultistepScheduler is not None:
        try:
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=flow_shift
            )
        except TypeError:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    dev = _resolve_cuda_device(device)
    vae_dev = resolve_vae_device(device, vae_device)
    if dev.type == "cuda":
        _component_to(getattr(pipe, "transformer", None), dev)
        _component_to(getattr(pipe, "transformer_2", None), dev)
        _component_to(getattr(pipe, "vae", None), vae_dev, dtype=vae_torch_dtype)
        _component_to(getattr(pipe, "text_encoder", None), torch.device("cpu"))
        _component_to(getattr(pipe, "image_encoder", None), torch.device("cpu"))
    else:
        if hasattr(pipe, "to"):
            pipe.to(dev)
        _component_to(getattr(pipe, "vae", None), vae_dev, dtype=vae_torch_dtype)

    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    pipe._low_memory_text_encoder = low_cpu_memory
    return pipe


def encode_video_official(pipe, frames_bgr: list) -> torch.Tensor:
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    tensor = _frames_to_tensor(frames_bgr, vae_dev, vae_dtype)
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
    frames = _tensor_to_frames(decoded)
    del decoded, latents
    clear_vae_internal_cache(pipe)
    release_memory()
    return frames


def encode_prompt_official(
    pipe,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    max_sequence_length: int,
):
    device = _pipe_device(pipe)
    dtype = _transformer_dtype(pipe)
    if hasattr(pipe, "encode_prompt") and not getattr(pipe, "_low_memory_text_encoder", False):
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        return prompt_embeds.to(dtype=dtype, device=device), (
            negative_embeds.to(dtype=dtype, device=device)
            if negative_embeds is not None
            else None
        )

    if not hasattr(pipe, "_get_t5_prompt_embeds"):
        raise RuntimeError("Pipeline does not expose encode_prompt or _get_t5_prompt_embeds")

    prompt_embeds = pipe._get_t5_prompt_embeds(
        prompt=prompt,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=torch.device("cpu"),
        dtype=getattr(getattr(pipe, "text_encoder", None), "dtype", torch.float32),
    ).to(device=device, dtype=dtype)
    negative_embeds = None
    if guidance_scale > 1.0:
        negative_embeds = pipe._get_t5_prompt_embeds(
            prompt=negative_prompt or "",
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
    return prompt_embeds, negative_embeds


def encode_image_embeds_official(pipe, frame_bgr: np.ndarray, height: int, width: int):
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_2", None)
    image_dim = getattr(getattr(transformer, "config", None), "image_dim", None)
    if image_dim is None or not hasattr(pipe, "encode_image"):
        return None

    image = _bgr_to_pil(frame_bgr)
    image_embeds = pipe.encode_image(
        image=image,
        device=torch.device("cpu"),
        num_videos_per_prompt=1,
        output_hidden_states=False,
    )
    if isinstance(image_embeds, tuple):
        image_embeds = image_embeds[0]
    return image_embeds.to(device=_pipe_device(pipe), dtype=_transformer_dtype(pipe))


def prepare_ti2v_condition_official(
    pipe,
    first_frame_bgr: np.ndarray,
    noisy_latents: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    generator,
):
    transformer_device = _pipe_device(pipe)
    vae_device = _vae_device(pipe)
    dtype = _transformer_dtype(pipe)
    image = _bgr_to_pil(first_frame_bgr)
    if not hasattr(pipe, "video_processor"):
        raise RuntimeError("Pipeline does not expose video_processor for image preprocessing")
    image = pipe.video_processor.preprocess(image, height=height, width=width)
    image = image.to(device=vae_device, dtype=torch.float32)

    outputs = pipe.prepare_latents(
        image,
        batch_size=1,
        num_channels_latents=pipe.vae.config.z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=vae_device,
        generator=generator,
        latents=noisy_latents.to(device=vae_device, dtype=torch.float32),
    )
    expand_timesteps = bool(getattr(getattr(pipe, "config", None), "expand_timesteps", False))
    if len(outputs) == 3:
        prepared_latents, condition, first_frame_mask = outputs
        expand_timesteps = True
        first_frame_mask = first_frame_mask.to(device=transformer_device, dtype=dtype)
    elif len(outputs) == 2:
        prepared_latents, condition = outputs
        first_frame_mask = None
    else:
        raise ValueError(f"Unexpected prepare_latents return length: {len(outputs)}")

    prepared_latents = prepared_latents.to(device=transformer_device, dtype=dtype)
    condition = condition.to(device=transformer_device, dtype=dtype)
    if prepared_latents.shape != noisy_latents.shape:
        raise ValueError(
            "prepare_latents changed noisy latent shape: "
            f"input={tuple(noisy_latents.shape)} output={tuple(prepared_latents.shape)}"
        )
    if expand_timesteps and first_frame_mask is None:
        raise ValueError("expand_timesteps=True requires first_frame_mask")
    clear_vae_internal_cache(pipe)
    release_memory()
    return {
        "latents": prepared_latents,
        "condition": condition,
        "first_frame_mask": first_frame_mask,
        "expand_timesteps": expand_timesteps,
    }


def scheduler_step_official(pipe, noise_pred, timestep, latents, timestep_next):
    params = inspect.signature(pipe.scheduler.step).parameters
    if "timestep_back" in params:
        return pipe.scheduler.step(
            noise_pred, timestep, latents, timestep_back=timestep_next
        ).prev_sample
    try:
        out = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)
        return out[0]
    except TypeError:
        return pipe.scheduler.step(noise_pred, timestep, latents).prev_sample


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
    image_embeds,
    attention_kwargs,
    cache_name,
):
    with torch.no_grad(), model_cache_context(model, cache_name):
        return model(
            hidden_states=latent_model_input,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=image_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]


def denoise_with_ti2v_pipeline_internals(
    pipe,
    latents: torch.Tensor,
    timesteps_run: torch.Tensor,
    condition: torch.Tensor,
    first_frame_mask: torch.Tensor,
    expand_timesteps: bool,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    guidance_scale: float,
    guidance_scale_2: float,
    attention_kwargs=None,
):
    transformer_dtype = _transformer_dtype(pipe)
    boundary_ratio = getattr(getattr(pipe, "config", None), "boundary_ratio", None)
    if boundary_ratio is not None and getattr(pipe, "transformer_2", None) is not None:
        boundary_timestep = boundary_ratio * pipe.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    for i, timestep in enumerate(timesteps_run):
        if boundary_timestep is None or timestep >= boundary_timestep:
            model = pipe.transformer
            step_guidance = guidance_scale
        else:
            model = pipe.transformer_2
            step_guidance = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale

        if expand_timesteps:
            latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
            timestep_batch = _expanded_timestep(
                model, first_frame_mask, timestep, latents.shape[0]
            )
        else:
            latent_model_input = torch.cat([latents, condition], dim=1)
            timestep_batch = timestep.expand(latents.shape[0]).to(latents.device)

        latent_model_input = latent_model_input.to(dtype=transformer_dtype)
        noise_pred = _transformer_forward(
            model,
            latent_model_input,
            timestep_batch,
            prompt_embeds,
            image_embeds,
            attention_kwargs,
            "cond",
        )
        if negative_prompt_embeds is not None:
            noise_uncond = _transformer_forward(
                model,
                latent_model_input,
                timestep_batch,
                negative_prompt_embeds,
                image_embeds,
                attention_kwargs,
                "uncond",
            )
            noise_pred = noise_uncond + step_guidance * (noise_pred - noise_uncond)

        timestep_next = (
            timesteps_run[i + 1]
            if i + 1 < len(timesteps_run)
            else torch.zeros_like(timestep)
        )
        latents = scheduler_step_official(pipe, noise_pred, timestep, latents, timestep_next)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            print(
                f"[ti2v denoise {i + 1:3d}/{len(timesteps_run)}] "
                f"t={timestep.item():.1f} latent_norm={latents.norm().item():.1f}"
            )

    if expand_timesteps:
        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
    return latents


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


def run_debug(args):
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if len(set(args.gammas)) != len(args.gammas):
        raise ValueError("--gammas must not contain duplicates")
    for gamma in args.gammas:
        noise_strength_to_start_idx(gamma, args.scheduler_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    frames, fps = load_video(args.video, args.max_frames, args.width, args.height)
    print(f"[input] {len(frames)} frames, {args.width}x{args.height}, {fps:.2f} fps")
    save_video(frames, os.path.join(args.output_dir, "original.mp4"), fps)
    save_frames(frames, os.path.join(args.output_dir, "original_frames"))

    pipe = load_wan_ti2v_pipe(
        args.model_id,
        device=args.device,
        vae_device=args.vae_device,
        vae_dtype=args.vae_dtype,
        flow_shift=args.flow_shift,
        low_cpu_memory=args.low_memory,
    )
    actual_model = "Wan2.2-TI2V-5B"
    print(
        f"[model] {actual_model} model_id={args.model_id} "
        f"transformer_device={_pipe_device(pipe)} transformer_dtype={_transformer_dtype(pipe)} "
        f"vae_device={_vae_device(pipe)} vae_dtype={_vae_dtype(pipe)}"
    )

    latents_clean = encode_video_official(pipe, frames)
    vae_frames = decode_latents_official(pipe, latents_clean)
    vae_psnr, vae_per_frame = mean_psnr(frames, vae_frames)
    save_video(vae_frames, os.path.join(args.output_dir, "vae_roundtrip.mp4"), fps)
    del vae_frames
    release_memory()
    print(f"[baseline] VAE round-trip PSNR={vae_psnr:.2f} dB")

    prompt_embeds, negative_prompt_embeds = encode_prompt_official(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.guidance_scale,
        args.max_sequence_length,
    )
    image_embeds = encode_image_embeds_official(pipe, frames[0], args.height, args.width)
    if args.low_memory and getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    if args.low_memory and getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder = None
    release_memory()

    torch.manual_seed(args.seed)
    base_noise = torch.randn_like(latents_clean)
    torch.save(base_noise.detach().cpu(), os.path.join(args.output_dir, "noise.pt"))

    video_latent_shape = list(latents_clean.shape)
    print(
        f"[latent] video={video_latent_shape} no_reference_slots=True "
        f"guidance_scale={args.guidance_scale} guidance_scale_2={args.guidance_scale_2}"
    )

    summaries = []
    for gamma in args.gammas:
        print(f"\n[gamma={gamma:.3f}] starting")
        gamma_dir = os.path.join(args.output_dir, gamma_dir_name(gamma))
        os.makedirs(gamma_dir, exist_ok=True)

        start_idx = noise_strength_to_start_idx(gamma, args.scheduler_steps)
        timesteps_run = set_denoise_start(pipe, args.scheduler_steps, start_idx)
        t_start = pipe.scheduler.timesteps[start_idx]
        if gamma == 0.0:
            noisy_latents = latents_clean.clone()
            timesteps_run = timesteps_run[:0]
        else:
            noisy_latents = add_noise_at_timestep(pipe, latents_clean, base_noise, t_start)

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        ti2v_condition = prepare_ti2v_condition_official(
            pipe,
            frames[0],
            noisy_latents,
            args.height,
            args.width,
            len(frames),
            generator,
        )
        noisy_latents = ti2v_condition["latents"]
        expand_timesteps = ti2v_condition["expand_timesteps"]
        condition = ti2v_condition["condition"]
        first_frame_mask = ti2v_condition["first_frame_mask"]

        noisy_frames = None
        noisy_psnr = None
        noisy_per_frame = []
        if args.decode_noisy:
            noisy_frames = decode_latents_official(pipe, noisy_latents)
            noisy_psnr, noisy_per_frame = mean_psnr(frames, noisy_frames)
            save_video(noisy_frames, os.path.join(gamma_dir, "noisy.mp4"), fps)
        else:
            print("[decode] skipping noisy.mp4 to keep VAE memory for denoising")

        denoised_latents = denoise_with_ti2v_pipeline_internals(
            pipe,
            noisy_latents.clone(),
            timesteps_run,
            condition,
            first_frame_mask,
            expand_timesteps,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            args.guidance_scale,
            args.guidance_scale_2,
        )
        denoised_frames = decode_latents_official(pipe, denoised_latents)
        denoised_psnr, denoised_per_frame = mean_psnr(frames, denoised_frames)
        save_video(denoised_frames, os.path.join(gamma_dir, "denoised.mp4"), fps)
        save_frames(denoised_frames, os.path.join(gamma_dir, "denoised_frames"))
        if noisy_frames is not None:
            save_comparison(
                frames,
                noisy_frames,
                denoised_frames,
                os.path.join(gamma_dir, "compare.mp4"),
                fps,
            )

        summary = {
            "gamma": gamma,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "expand_timesteps": expand_timesteps,
            "reference_latent_slots": 0,
            "video_latent_shape": str(video_latent_shape),
            "transformer_latent_shape": str(list(noisy_latents.shape)),
            "condition_shape": str(list(condition.shape)),
            "start_idx": start_idx,
            "t_start": float(t_start.item()),
            "denoise_steps": len(timesteps_run),
            "vae_psnr": vae_psnr,
            "noisy_psnr": noisy_psnr,
            "denoised_psnr": denoised_psnr,
            "recovery_gain": (
                denoised_psnr - noisy_psnr if noisy_psnr is not None else None
            ),
            "gap_to_vae": vae_psnr - denoised_psnr,
            "noisy_latent_mse": float(
                torch.mean((noisy_latents.float() - latents_clean.float()) ** 2).item()
            ),
            "denoised_latent_mse": float(
                torch.mean((denoised_latents.float() - latents_clean.float()) ** 2).item()
            ),
            "recovered": denoised_psnr > noisy_psnr if noisy_psnr is not None else None,
            "noisy_psnr_per_frame": noisy_per_frame,
            "denoised_psnr_per_frame": denoised_per_frame,
            "decode_noisy": args.decode_noisy,
        }
        summaries.append(summary)
        if noisy_psnr is None:
            print(
                f"[gamma={gamma:.3f}] expand_timesteps={expand_timesteps} "
                f"denoised={denoised_psnr:.2f} dB noisy_psnr=skipped"
            )
        else:
            print(
                f"[gamma={gamma:.3f}] expand_timesteps={expand_timesteps} "
                f"noisy={noisy_psnr:.2f} dB denoised={denoised_psnr:.2f} dB "
                f"gain={summary['recovery_gain']:+.2f} dB"
            )
        del noisy_latents, denoised_latents, noisy_frames, denoised_frames
        del condition, first_frame_mask, ti2v_condition
        release_memory()

    csv_fields = [
        "gamma",
        "guidance_scale",
        "guidance_scale_2",
        "expand_timesteps",
        "reference_latent_slots",
        "video_latent_shape",
        "transformer_latent_shape",
        "condition_shape",
        "start_idx",
        "t_start",
        "denoise_steps",
        "vae_psnr",
        "noisy_psnr",
        "denoised_psnr",
        "recovery_gain",
        "gap_to_vae",
        "noisy_latent_mse",
        "denoised_latent_mse",
        "recovered",
        "decode_noisy",
    ]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in csv_fields} for row in summaries])

    json_summary = {
        "video": args.video,
        "model_id": args.model_id,
        "model_family": actual_model,
        "frames": len(frames),
        "width": args.width,
        "height": args.height,
        "fps": fps,
        "seed": args.seed,
        "scheduler_steps": args.scheduler_steps,
        "flow_shift": args.flow_shift,
        "transformer_device": str(_pipe_device(pipe)),
        "transformer_dtype": str(_transformer_dtype(pipe)),
        "vae_device": str(_vae_device(pipe)),
        "vae_dtype": str(_vae_dtype(pipe)),
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "decode_noisy": args.decode_noisy,
        "video_latent_shape": video_latent_shape,
        "reference_latent_slots": 0,
        "vae_psnr": json_metric(vae_psnr),
        "vae_psnr_per_frame": [json_metric(value) for value in vae_per_frame],
        "runs": [
            {
                key: (
                    [json_metric(value) for value in val]
                    if isinstance(val, list)
                    else json_metric(val)
                    if isinstance(val, float)
                    else val
                )
                for key, val in row.items()
            }
            for row in summaries
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(json_summary, handle, indent=2, ensure_ascii=False)
    print(f"\n[result] outputs written to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vae-device",
        default="auto",
        help="VAE device. auto uses cuda:1 when multiple GPUs are visible, otherwise --device.",
    )
    parser.add_argument(
        "--vae-dtype",
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="VAE dtype. float32 is safest; float16 can reduce memory if needed.",
    )
    parser.add_argument("--max-frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Intuitive noise strengths: 0=no noise, 1=maximum noise.",
    )
    parser.add_argument("--scheduler-steps", type=int, default=100)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale-2", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=226)
    parser.add_argument(
        "--decode-noisy",
        action="store_true",
        help="Decode/save noisy.mp4 and noisy PSNR. Disabled by default to avoid VAE OOM.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/debug_denoise_moe")
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Disable reduced-CPU-RAM model loading and T5/image encoder release.",
    )
    parser.set_defaults(low_memory=True)
    return parser


if __name__ == "__main__":
    run_debug(build_parser().parse_args())

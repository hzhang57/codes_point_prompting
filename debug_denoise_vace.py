"""
Official Wan VACE pipeline SDEdit reconstruction experiment.

This mirrors debug_denoise.py outputs, but constructs VACE conditions and runs
the denoising loop with WanVACEPipeline internals: preprocess_conditions,
prepare_video_latents, prepare_masks, prepare_latents, CFG, transformer calls,
and scheduler steps.

Example:
  python debug_denoise_vace.py --video input.mp4 --max-frames 9
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
from model_adapter import (
    _frames_to_tensor,
    _tensor_to_frames,
    load_wan_vace_pipe,
    noise_strength_to_start_idx,
    prepend_reference_slots,
    remove_reference_slots,
)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    return transformer.dtype


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


def encode_video_official(pipe, frames_bgr: list) -> torch.Tensor:
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    tensor = _frames_to_tensor(frames_bgr, vae_dev, vae_dtype)
    with torch.no_grad():
        latents = _retrieve_latents_argmax(pipe.vae.encode(tensor))
    mean, std = _vae_norm(pipe, latents.device)
    latents = ((latents.float() - mean) * std).to(_transformer_dtype(pipe))
    return latents.to(device=_pipe_device(pipe))


def decode_latents_official(pipe, latents: torch.Tensor) -> list:
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    mean, std = _vae_norm(pipe, latents.device)
    latents = (latents.float() / std + mean).to(device=vae_dev, dtype=vae_dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents).sample
    return _tensor_to_frames(decoded)


def prepare_reference_condition_official(
    pipe,
    frame_bgr: np.ndarray,
    n_frames_px: int,
    height: int,
    width: int,
    generator,
) -> dict:
    device = _pipe_device(pipe)
    dtype = _transformer_dtype(pipe)
    video, mask, reference_images = pipe.preprocess_conditions(
        video=None,
        mask=None,
        reference_images=_bgr_to_pil(frame_bgr),
        batch_size=1,
        height=height,
        width=width,
        num_frames=n_frames_px,
        dtype=torch.float32,
        device=device,
    )
    conditioning_latents = pipe.prepare_video_latents(
        video, mask, reference_images, generator, device
    )
    mask_latents = pipe.prepare_masks(mask, reference_images, generator)
    control_hidden_states = torch.cat([conditioning_latents, mask_latents], dim=1)
    return {
        "control_hidden_states": control_hidden_states.to(device=device, dtype=dtype),
        "reference_latent_slots": len(reference_images[0]),
    }


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

    # Low-memory loader keeps T5 on CPU; use the official T5 helper there.
    prompt_embeds = pipe._get_t5_prompt_embeds(
        prompt=prompt,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=torch.device("cpu"),
        dtype=getattr(pipe.text_encoder, "dtype", torch.float32)
        if getattr(pipe, "text_encoder", None) is not None
        else torch.float32,
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


def conditioning_scale_tensor(pipe, conditioning_scale: float) -> torch.Tensor:
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_2", None)
    vace_layers = transformer.config.vace_layers
    return torch.full(
        (len(vace_layers),),
        float(conditioning_scale),
        device=_pipe_device(pipe),
        dtype=_transformer_dtype(pipe),
    )


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


def denoise_with_vace_pipeline_internals(
    pipe,
    latents: torch.Tensor,
    timesteps_run: torch.Tensor,
    control_hidden_states: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    conditioning_scale: torch.Tensor,
):
    transformer_dtype = _transformer_dtype(pipe)
    boundary_ratio = getattr(getattr(pipe, "config", None), "boundary_ratio", None)
    if boundary_ratio is not None:
        boundary_timestep = boundary_ratio * pipe.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    for i, timestep in enumerate(timesteps_run):
        if boundary_timestep is None or timestep >= boundary_timestep:
            model = pipe.transformer
            step_guidance = guidance_scale
        else:
            model = pipe.transformer_2
            step_guidance = guidance_scale

        latent_model_input = latents.to(transformer_dtype)
        timestep_batch = timestep.expand(latents.shape[0]).to(latents.device)
        with torch.no_grad(), model_cache_context(model, "cond"):
            noise_pred = model(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=conditioning_scale,
                return_dict=False,
            )[0]
        if negative_prompt_embeds is not None:
            with torch.no_grad(), model_cache_context(model, "uncond"):
                noise_uncond = model(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=negative_prompt_embeds,
                    control_hidden_states=control_hidden_states,
                    control_hidden_states_scale=conditioning_scale,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + step_guidance * (noise_pred - noise_uncond)

        timestep_next = (
            timesteps_run[i + 1]
            if i + 1 < len(timesteps_run)
            else torch.zeros_like(timestep)
        )
        latents = scheduler_step_official(pipe, noise_pred, timestep, latents, timestep_next)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            print(
                f"[vace denoise {i + 1:3d}/{len(timesteps_run)}] "
                f"t={timestep.item():.1f} latent_norm={latents.norm().item():.1f}"
            )
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

    pipe = load_wan_vace_pipe(
        args.model_id,
        device=args.device,
        flow_shift=args.flow_shift,
        low_cpu_memory=args.low_memory,
    )
    print(f"[model] device={_pipe_device(pipe)} dtype={_transformer_dtype(pipe)}")

    latents_clean = encode_video_official(pipe, frames)
    vae_frames = decode_latents_official(pipe, latents_clean)
    vae_psnr, vae_per_frame = mean_psnr(frames, vae_frames)
    save_video(vae_frames, os.path.join(args.output_dir, "vae_roundtrip.mp4"), fps)
    del vae_frames
    release_memory()
    print(f"[baseline] VAE round-trip PSNR={vae_psnr:.2f} dB")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    reference_condition = prepare_reference_condition_official(
        pipe, frames[0], len(frames), args.height, args.width, generator
    )
    reference_slots = reference_condition["reference_latent_slots"]
    model_latents_clean = prepend_reference_slots(latents_clean, reference_slots)
    num_channels = model_latents_clean.shape[1]
    model_latents_clean = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels,
        height=args.height,
        width=args.width,
        num_frames=len(frames) + reference_slots * getattr(pipe, "vae_scale_factor_temporal", 4),
        dtype=torch.float32,
        device=_pipe_device(pipe),
        generator=generator,
        latents=model_latents_clean,
    ).to(dtype=_transformer_dtype(pipe))
    control_hidden_states = reference_condition["control_hidden_states"]
    if control_hidden_states.shape[2] != model_latents_clean.shape[2]:
        raise ValueError(
            "Official VACE control does not match model latent length: "
            f"control={control_hidden_states.shape[2]} latent={model_latents_clean.shape[2]}"
        )

    prompt_embeds, negative_prompt_embeds = encode_prompt_official(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.guidance_scale,
        args.max_sequence_length,
    )
    if args.low_memory and getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    release_memory()

    conditioning_scale = conditioning_scale_tensor(pipe, args.conditioning_scale)
    torch.manual_seed(args.seed)
    base_noise = torch.randn_like(model_latents_clean)
    torch.save(base_noise.detach().cpu(), os.path.join(args.output_dir, "noise.pt"))

    video_latent_shape = list(latents_clean.shape)
    model_latent_shape = list(model_latents_clean.shape)
    print(
        f"[latent] video={video_latent_shape} transformer={model_latent_shape} "
        f"reference_slots={reference_slots} guidance_scale={args.guidance_scale} "
        f"conditioning_scale={args.conditioning_scale}"
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
            noisy_model_latents = model_latents_clean.clone()
            timesteps_run = timesteps_run[:0]
        else:
            noisy_model_latents = add_noise_at_timestep(
                pipe, model_latents_clean, base_noise, t_start
            )

        noisy_latents = remove_reference_slots(noisy_model_latents, reference_slots)
        noisy_frames = decode_latents_official(pipe, noisy_latents)
        noisy_psnr, noisy_per_frame = mean_psnr(frames, noisy_frames)
        save_video(noisy_frames, os.path.join(gamma_dir, "noisy.mp4"), fps)

        denoised_model_latents = denoise_with_vace_pipeline_internals(
            pipe,
            noisy_model_latents.clone(),
            timesteps_run,
            control_hidden_states,
            prompt_embeds,
            negative_prompt_embeds,
            args.guidance_scale,
            conditioning_scale,
        )
        denoised_latents = remove_reference_slots(denoised_model_latents, reference_slots)
        denoised_frames = decode_latents_official(pipe, denoised_latents)
        denoised_psnr, denoised_per_frame = mean_psnr(frames, denoised_frames)
        save_video(denoised_frames, os.path.join(gamma_dir, "denoised.mp4"), fps)
        save_frames(denoised_frames, os.path.join(gamma_dir, "denoised_frames"))
        save_comparison(
            frames,
            noisy_frames,
            denoised_frames,
            os.path.join(gamma_dir, "compare.mp4"),
            fps,
        )

        summary = {
            "gamma": gamma,
            "conditioning_scale": args.conditioning_scale,
            "guidance_scale": args.guidance_scale,
            "reference_latent_slots": reference_slots,
            "video_latent_shape": str(video_latent_shape),
            "transformer_latent_shape": str(model_latent_shape),
            "start_idx": start_idx,
            "t_start": float(t_start.item()),
            "denoise_steps": len(timesteps_run),
            "vae_psnr": vae_psnr,
            "noisy_psnr": noisy_psnr,
            "denoised_psnr": denoised_psnr,
            "recovery_gain": denoised_psnr - noisy_psnr,
            "gap_to_vae": vae_psnr - denoised_psnr,
            "noisy_latent_mse": float(
                torch.mean((noisy_latents.float() - latents_clean.float()) ** 2).item()
            ),
            "denoised_latent_mse": float(
                torch.mean((denoised_latents.float() - latents_clean.float()) ** 2).item()
            ),
            "recovered": denoised_psnr > noisy_psnr,
            "noisy_psnr_per_frame": noisy_per_frame,
            "denoised_psnr_per_frame": denoised_per_frame,
        }
        summaries.append(summary)
        print(
            f"[gamma={gamma:.3f}] noisy={noisy_psnr:.2f} dB "
            f"denoised={denoised_psnr:.2f} dB "
            f"gain={summary['recovery_gain']:+.2f} dB"
        )
        del noisy_model_latents, noisy_latents, denoised_model_latents
        del denoised_latents, noisy_frames, denoised_frames
        release_memory()

    csv_fields = [
        "gamma",
        "conditioning_scale",
        "guidance_scale",
        "reference_latent_slots",
        "video_latent_shape",
        "transformer_latent_shape",
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
    ]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in csv_fields} for row in summaries])

    json_summary = {
        "video": args.video,
        "frames": len(frames),
        "width": args.width,
        "height": args.height,
        "fps": fps,
        "seed": args.seed,
        "scheduler_steps": args.scheduler_steps,
        "conditioning_scale": args.conditioning_scale,
        "guidance_scale": args.guidance_scale,
        "reference_latent_slots": reference_slots,
        "video_latent_shape": video_latent_shape,
        "transformer_latent_shape": model_latent_shape,
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
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=226)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/debug_denoise_vace")
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Disable reduced-CPU-RAM model loading and T5 release.",
    )
    parser.set_defaults(low_memory=True)
    return parser


if __name__ == "__main__":
    run_debug(build_parser().parse_args())

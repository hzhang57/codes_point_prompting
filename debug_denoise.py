"""
SDEdit video reconstruction experiment.

Loads a clean video, applies several intuitive noise strengths in latent space,
denoises with the clean first frame as VACE control, and writes noisy/denoised
videos plus quantitative summaries.

Example:
  python debug_denoise.py --video input.mp4 --max-frames 9
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch

from model_adapter import (
    create_adapter,
    load_wan_vace_pipe,
    noise_strength_to_start_idx,
)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def mean_psnr(orig: list, reco: list) -> tuple:
    if len(orig) != len(reco):
        raise ValueError(f"Frame count mismatch: original={len(orig)}, result={len(reco)}")
    for i, (a, b) in enumerate(zip(orig, reco)):
        if a.shape != b.shape:
            raise ValueError(f"Frame {i} shape mismatch: original={a.shape}, result={b.shape}")
    values = [psnr(a, b) for a, b in zip(orig, reco)]
    return float(np.mean(values)), values


def save_frames(frames: list, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f"{i:03d}.png"), frame)


def save_video(frames: list, path: str, fps: float) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio

        rgb = [frame[..., ::-1] for frame in frames]
        imageio.mimsave(
            path,
            rgb,
            fps=fps,
            codec="libx264",
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )
    except Exception as exc:
        print(f"[save] imageio failed ({exc}), fallback to cv2")
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
    print(f"[save] {path} ({len(frames)} frames, {fps:.2f} fps)")


def save_comparison(orig: list, noisy: list, denoised: list, path: str, fps: float) -> None:
    rows = [
        np.concatenate([clean, noisy_frame, denoised_frame], axis=1)
        for clean, noisy_frame, denoised_frame in zip(orig, noisy, denoised)
    ]
    save_video(rows, path, fps)


def gamma_dir_name(gamma: float) -> str:
    return f"gamma_{gamma:.3f}".rstrip("0").rstrip(".")


def json_metric(value: float):
    return value if np.isfinite(value) else None


def load_video(path: str, max_frames: int, width: int, height: int) -> tuple:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height)))
    cap.release()
    if not frames:
        raise RuntimeError(f"无法读取视频：{path}")

    valid_count = ((len(frames) - 1) // 4) * 4 + 1
    if valid_count != len(frames):
        print(f"[input] 帧数 {len(frames)} 不满足 T=4k+1，裁剪到 {valid_count}")
        frames = frames[:valid_count]
    return frames, float(fps)


def denoise_latents(adapter, latents, timesteps_run, image_cond, text_cond, n_frames_px):
    for i, timestep in enumerate(timesteps_run):
        timestep_batch = timestep.unsqueeze(0).to(adapter.device)
        timestep_next = (
            timesteps_run[i + 1]
            if i + 1 < len(timesteps_run)
            else torch.zeros_like(timestep)
        )
        with torch.no_grad():
            velocity = adapter.forward_transformer(
                noisy_latents=latents,
                timestep=timestep_batch,
                text_cond=text_cond,
                image_cond=image_cond,
                n_frames_px=n_frames_px,
            )
        latents = adapter.scheduler_step(velocity, timestep, latents, timestep_next)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            print(
                f"[denoise {i + 1:3d}/{len(timesteps_run)}] "
                f"t={timestep.item():.1f} latent_norm={latents.norm().item():.1f}"
            )
    return latents


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
        args.model_id, device=args.device, flow_shift=args.flow_shift
    )
    adapter = create_adapter(pipe)
    print(f"[model] device={adapter.device} dtype={adapter.dtype}")

    latents_clean = adapter.encode_video(frames)
    vae_frames = adapter.decode_latents(latents_clean)
    vae_psnr, vae_per_frame = mean_psnr(frames, vae_frames)
    save_video(vae_frames, os.path.join(args.output_dir, "vae_roundtrip.mp4"), fps)
    print(f"[baseline] VAE round-trip PSNR={vae_psnr:.2f} dB")

    image_cond = adapter.encode_image_cond(frames[0], latents_clean)
    text_cond = adapter.encode_text(args.prompt)
    torch.manual_seed(args.seed)
    base_noise = torch.randn_like(latents_clean)
    torch.save(base_noise.detach().cpu(), os.path.join(args.output_dir, "noise.pt"))

    summaries = []
    for gamma in args.gammas:
        print(f"\n[gamma={gamma:.3f}] starting")
        gamma_dir = os.path.join(args.output_dir, gamma_dir_name(gamma))
        os.makedirs(gamma_dir, exist_ok=True)

        start_idx = noise_strength_to_start_idx(gamma, args.scheduler_steps)
        timesteps_run = adapter.prepare_denoise_start(args.scheduler_steps, start_idx)
        t_start = adapter.timesteps[start_idx]
        if gamma == 0.0:
            noisy_latents = latents_clean.clone()
            timesteps_run = timesteps_run[:0]
        else:
            noisy_latents = adapter.add_noise_at_timestep(
                latents_clean, base_noise, t_start
            )

        noisy_frames = adapter.decode_latents(noisy_latents)
        noisy_psnr, noisy_per_frame = mean_psnr(frames, noisy_frames)
        save_video(noisy_frames, os.path.join(gamma_dir, "noisy.mp4"), fps)

        denoised_latents = denoise_latents(
            adapter,
            noisy_latents.clone(),
            timesteps_run,
            image_cond,
            text_cond,
            len(frames),
        )
        denoised_frames = adapter.decode_latents(denoised_latents)
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

    csv_fields = [
        "gamma",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/debug_denoise")
    return parser


if __name__ == "__main__":
    run_debug(build_parser().parse_args())

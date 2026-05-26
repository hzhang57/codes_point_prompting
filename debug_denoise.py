"""
无引导去噪 debug 脚本。

流程：
  1. 读取输入视频前 N 帧
  2. VAE 编码 → latents_clean
  3. 在 gamma=0.5 处加噪 → latents_noisy
  4. 无任何引导，用 VACE transformer 去噪 50 步
  5. VAE 解码 → 输出帧，与原始帧做 PSNR 对比

预期：输出帧应接近原始帧（PSNR > 20dB），否则说明
scheduler/transformer/control 构建有问题。

用法：
  python debug_denoise.py --video input.mp4 --max-frames 9
"""

import argparse
import os
import cv2
import torch
import numpy as np

from model_adapter import load_wan_vace_pipe, create_adapter


# --------------------------------------------------------------------------- #

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def save_side_by_side(orig: list, reco: list, path: str, fps: float = 8.0):
    """将原始帧和重建帧左右拼接保存为 mp4。"""
    if not orig:
        return
    rows = []
    for o, r in zip(orig, reco):
        rows.append(np.concatenate([o, r], axis=1))
    try:
        import imageio
        rgb = [f[..., ::-1] for f in rows]
        imageio.mimsave(path, rgb, fps=fps, codec="libx264",
                        output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
        print(f"[save] {path}  ({len(rows)} frames, left=原始 right=重建)")
    except Exception as e:
        print(f"[save] imageio failed ({e}), fallback cv2")
        H, W = rows[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for f in rows:
            writer.write(f)
        writer.release()


def save_frames(frames: list, prefix: str):
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(f"{prefix}_{i:03d}.png", f)
    print(f"[save] {len(frames)} PNGs → {prefix}_000.png … {prefix}_{len(frames)-1:03d}.png")


# --------------------------------------------------------------------------- #

def run_debug(args):
    # ------------------------------------------------------------------ #
    # 1. 读取视频帧                                                        #
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (args.width, args.height))
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"无法读取视频：{args.video}")
    print(f"[input] 读取 {len(frames)} 帧，分辨率 {args.width}×{args.height}")
    save_frames(frames, "dbg_original")

    # ------------------------------------------------------------------ #
    # 2. 加载模型                                                          #
    # ------------------------------------------------------------------ #
    pipe = load_wan_vace_pipe(args.model_id, device=args.device)
    adapter = create_adapter(pipe)
    print(f"[model] device={adapter.device}  dtype={adapter.dtype}")

    # ------------------------------------------------------------------ #
    # 3. VAE 编码                                                          #
    # ------------------------------------------------------------------ #
    latents_clean = adapter.encode_video(frames)
    print(f"[enc] latents_clean: shape={latents_clean.shape} "
          f"min={latents_clean.min():.3f} max={latents_clean.max():.3f} "
          f"norm={latents_clean.norm():.1f}")

    # 解码确认 round-trip 无损
    reco_vae = adapter.decode_latents(latents_clean)
    save_frames(reco_vae, "dbg_vae_roundtrip")
    psnr_vals = [psnr(o, r) for o, r in zip(frames, reco_vae)]
    print(f"[enc] VAE round-trip PSNR: {np.mean(psnr_vals):.1f} dB  "
          f"(per-frame: {[f'{v:.1f}' for v in psnr_vals]})")

    # ------------------------------------------------------------------ #
    # 4. 加噪到 gamma 处                                                   #
    # ------------------------------------------------------------------ #
    N = args.scheduler_steps
    gamma = args.gamma
    adapter.set_timesteps(N)
    timesteps = adapter.timesteps

    start_idx = min(int(N * gamma), N - 1)
    t_start = timesteps[start_idx]
    sigma = adapter.scheduler.sigmas[start_idx].item()
    print(f"[noise] N={N} gamma={gamma} start_idx={start_idx} "
          f"t_start={t_start.item():.1f} sigma={sigma:.3f}")
    print(f"[noise] 混合比例: content={(1-sigma):.3f}  noise={sigma:.3f}")

    torch.manual_seed(args.seed)
    noise = torch.randn_like(latents_clean)
    latents = adapter.scheduler.scale_noise(latents_clean, t_start.unsqueeze(0), noise)
    print(f"[noise] latents_noisy: min={latents.min():.3f} max={latents.max():.3f} "
          f"norm={latents.norm():.1f}")

    noisy_frames = adapter.decode_latents(latents)
    save_frames(noisy_frames, "dbg_noisy")
    psnr_noisy = [psnr(o, r) for o, r in zip(frames, noisy_frames)]
    print(f"[noise] 加噪后 PSNR: {np.mean(psnr_noisy):.1f} dB  "
          f"(应明显低于 VAE round-trip)")

    # ------------------------------------------------------------------ #
    # 5. 无引导去噪循环                                                    #
    # ------------------------------------------------------------------ #
    image_cond = adapter.encode_image_cond(frames[0], latents_clean)
    text_cond = adapter.encode_text("")   # None（T5 未加载），forward_transformer 内部补全零向量

    # [诊断] 打印第一步的 control_hidden_states 统计量，确认 _build_control 是否正常
    _ctrl = adapter._build_control(latents, image_cond)
    print(f"[diag] control shape={_ctrl.shape} "
          f"norm={_ctrl.norm():.1f} mean={_ctrl.mean():.4f} "
          f"video_ctrl norm={_ctrl[:, :32].norm():.1f}  "
          f"mask_patches norm={_ctrl[:, 32:].norm():.1f}")

    timesteps_run = timesteps[start_idx:]
    print(f"[denoise] 去噪步数={len(timesteps_run)}  "
          f"t: {timesteps_run[0].item():.0f} → {timesteps_run[-1].item():.0f}")

    for i, t in enumerate(timesteps_run):
        t_batch = t.unsqueeze(0).to(adapter.device)
        t_next  = timesteps_run[i + 1] if i + 1 < len(timesteps_run) else torch.zeros_like(t)

        with torch.no_grad():
            velocity = adapter.forward_transformer(
                noisy_latents=latents,
                timestep=t_batch,
                text_cond=text_cond,
                image_cond=image_cond,
            )

        latents = adapter.scheduler_step(velocity, t, latents, t_next)

        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            v_norm = velocity.norm().item()
            print(f"[step {i+1:3d}/{len(timesteps_run)}] t={t.item():.0f}  "
                  f"v_norm={v_norm:.3f}  "
                  f"latents: min={latents.min():.3f} max={latents.max():.3f} "
                  f"norm={latents.norm():.1f}")

    # ------------------------------------------------------------------ #
    # 6. 解码 & 评估                                                       #
    # ------------------------------------------------------------------ #
    reco_frames = adapter.decode_latents(latents)
    save_frames(reco_frames, "dbg_denoised")
    psnr_reco = [psnr(o, r) for o, r in zip(frames, reco_frames)]
    print(f"\n[result] 去噪后 PSNR: {np.mean(psnr_reco):.1f} dB  "
          f"(per-frame: {[f'{v:.1f}' for v in psnr_reco]})")
    print(f"[result] 判断标准：")
    print(f"  > 25 dB → 去噪基本正常，问题在引导信号")
    print(f"  15~25 dB → 去噪部分有效，可能 control 构建或 scheduler 有偏差")
    print(f"  < 15 dB → 去噪完全无效，transformer 调用或 latent 格式有根本错误")

    save_side_by_side(frames, reco_frames, "dbg_compare.mp4")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",          required=True)
    parser.add_argument("--model-id",       default="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--max-frames",     type=int,   default=9)
    parser.add_argument("--height",         type=int,   default=480)
    parser.add_argument("--width",          type=int,   default=832)
    parser.add_argument("--gamma",          type=float, default=0.5)
    parser.add_argument("--scheduler-steps",type=int,   default=100)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()
    run_debug(args)

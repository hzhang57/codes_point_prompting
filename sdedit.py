"""
Point Prompting 核心：带反事实增强引导的 SDEdit 去噪。

基于 CogVideoX-I2V 的 SDEdit 去噪。

反事实引导公式（论文公式 3）：
    v̂_θ(x_t, c_edited) = (λ+1) · v_θ(x_t, c_edited) - λ · v_θ(x_t, c_original)

其中：
    c_edited   = 含红色标记的第 0 帧作为图像条件（正向）
    c_original = 不含标记的原始第 0 帧作为图像条件（负向）
    λ          = 引导权重（默认 8）
    γ          = SDEdit 加噪比例（默认 0.5）

直觉：用原始帧作为"负提示"，迫使扩散模型在生成时保留标记，
而不是把标记当成噪声去除掉。
"""

from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import Optional

from model_adapter import ModelAdapter


def run_sdedit(
    adapter: ModelAdapter,
    frames_bgr_edited: list,
    frame_bgr_original: np.ndarray,
    gamma: float = 0.5,
    lam: float = 8.0,
    scheduler_steps: int = 100,
    prompt: str = "",
    generator: Optional[torch.Generator] = None,
) -> list:
    """执行一次带反事实增强引导的 SDEdit 完整流程。

    Args:
        adapter:          CogVideoXAdapter
        frames_bgr_edited: frame[0] 含红色标记的视频帧列表
        frame_bgr_original: 不含标记的原始 frame[0]（用于负向条件）
        gamma:            SDEdit 加噪比例 γ ∈ (0, 1]，越大生成越自由
        lam:              反事实引导权重 λ，越大标记越显著
        scheduler_steps:  调度器总步数（论文：100）；从 gamma*N 处开始去噪到末尾
        prompt:           文本提示（论文零样本设置为空字符串）
        generator:        可复现性用的随机数生成器

    Returns:
        生成后的帧列表，每帧 (H, W, 3) BGR uint8
    """
    device = adapter.device

    # ------------------------------------------------------------------ #
    # 步骤 1：将编辑后的视频帧编码到潜变量空间                            #
    # ------------------------------------------------------------------ #
    latents_clean = adapter.encode_video(frames_bgr_edited)  # (1, C, T, lH, lW)
    print(f"[DEBUG] vae_scale_factor={adapter._video_scale():.4f}")
    print(f"[DEBUG] latents_clean: shape={latents_clean.shape} "
          f"min={latents_clean.min():.3f} max={latents_clean.max():.3f} mean={latents_clean.mean():.3f}")

    # [DEBUG] 验证 VAE encode→decode 重建质量（无任何去噪，直接解码干净 latent）
    _clean_frames = adapter.decode_latents(latents_clean)
    print(f"[DEBUG] VAE reconstruct: {len(_clean_frames)} frames decoded")
    _save_debug_video(_clean_frames, "debug_vae_reconstruct.mp4")
    print(f"[DEBUG] debug_vae_reconstruct.mp4 saved — 应与输入帧接近")

    # ------------------------------------------------------------------ #
    # 步骤 2：分别编码"含标记"和"原始"帧的图像条件                       #
    # ------------------------------------------------------------------ #
    cond_edited   = adapter.encode_image_cond(frames_bgr_edited[0], latents_clean)
    cond_original = adapter.encode_image_cond(frame_bgr_original)
    print(f"[DEBUG] cond_edited:   shape={cond_edited.shape} "
          f"min={cond_edited.min():.3f} max={cond_edited.max():.3f}")
    print(f"[DEBUG] cond_original: shape={cond_original.shape} "
          f"min={cond_original.min():.3f} max={cond_original.max():.3f}")
    print(f"[DEBUG] cond diff (edited-original): "
          f"mean abs={( cond_edited - cond_original).abs().mean():.4f}")

    # [DEBUG] 将 cond_edited 解码回像素，确认红色标记是否正确保留
    _cond_frame = adapter.decode_latents(cond_edited)[0]
    cv2.imwrite("debug_cond_edited.png", _cond_frame)
    _cond_orig_frame = adapter.decode_latents(cond_original)[0]
    cv2.imwrite("debug_cond_original.png", _cond_orig_frame)
    print(f"[DEBUG] debug_cond_edited.png / debug_cond_original.png saved")

    # ------------------------------------------------------------------ #
    # 步骤 3：编码文本提示                                                  #
    # ------------------------------------------------------------------ #
    text_cond = adapter.encode_text(prompt)
    print(f"[DEBUG] text_cond: shape={text_cond.shape}")

    # ------------------------------------------------------------------ #
    # 步骤 4：在 t ≈ γ·T_max 处加噪（SDEdit 前向过程）                   #
    # 用 scheduler.add_noise() 确保与 DDIM 加噪公式一致。                  #
    # ------------------------------------------------------------------ #
    noise = torch.randn_like(latents_clean, generator=generator)
    adapter.set_timesteps(scheduler_steps)
    timesteps = adapter.timesteps  # 从大到小，共 scheduler_steps 个时间步

    start_idx = min(int(scheduler_steps * gamma), scheduler_steps - 1)
    t_start   = timesteps[start_idx]
    print(f"[DEBUG] scheduler_steps={scheduler_steps} gamma={gamma} "
          f"start_idx={start_idx} t_start={t_start.item():.1f} "
          f"denoise_steps={len(timesteps) - start_idx}")

    latents = adapter.scheduler.add_noise(
        latents_clean,
        noise,
        t_start[None] if t_start.ndim == 0 else t_start,
    )
    # [DEBUG] 打印 add_noise 内部实际使用的 sigma/alpha（flow matching: x_t = (1-sigma)*x0 + sigma*noise）
    _t_norm = t_start.item() / 1000.0  # 归一化到 [0,1]
    print(f"[DEBUG] t_start={t_start.item():.1f} t_norm={_t_norm:.3f}")
    print(f"[DEBUG] 期望 flow-matching 混合: content={(1-_t_norm):.3f} noise={_t_norm:.3f}")
    print(f"[DEBUG] latents_clean norm: {latents_clean.norm():.3f}  noise norm: {noise.norm():.3f}")
    print(f"[DEBUG] latents after add_noise: "
          f"min={latents.min():.3f} max={latents.max():.3f} mean={latents.mean():.3f} norm={latents.norm():.3f}")

    # [DEBUG] 将加噪后的 latent 解码，直观看噪声程度
    _save_debug_video(adapter.decode_latents(latents), "debug_noisy_input.mp4")
    print(f"[DEBUG] debug_noisy_input.mp4 saved")

    # 从 start_idx 去噪到序列末尾（共 scheduler_steps - start_idx 步）
    timesteps_run = timesteps[start_idx:]

    # ------------------------------------------------------------------ #
    # 步骤 5：普通去噪循环（DEBUG：跳过反事实引导，验证去噪本身是否正常）   #
    # ------------------------------------------------------------------ #
    print(f"[DEBUG] timesteps_run: {timesteps_run.cpu().numpy()}")  # 打印实际去噪的时间步列表
    for i, t in enumerate(timesteps_run):
        t_batch = t.unsqueeze(0).to(device)
        t_next  = timesteps_run[i + 1] if i + 1 < len(timesteps_run) else torch.zeros_like(t)
        with torch.no_grad():
            v = adapter.forward_transformer(latents, t_batch, text_cond, cond_edited)
        latents = adapter.scheduler_step(v, t, latents, t_next)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            print(f"[DEBUG] step {i+1}/{len(timesteps_run)} t={t.item():.1f} "
                  f"latents: min={latents.min():.3f} max={latents.max():.3f} mean={latents.mean():.3f}")
            # 取第 0 帧 latent 解码并保存，观察去噪过程中第一帧的恢复情况
            _f0 = adapter.decode_latents(latents[:, :, :1, :, :])[0]
            cv2.imwrite(f"debug_step_{i+1:03d}_frame0.png", _f0)

    # [DEBUG] ---- 反事实引导循环（暂时注释） ----
    # for i, t in enumerate(timesteps_run):
    #     t_batch  = t.unsqueeze(0).to(device)
    #     t_next   = timesteps_run[i + 1] if i + 1 < len(timesteps_run) else torch.zeros_like(t)
    #     v_guided = adapter.predict_with_guidance(
    #         noisy_latents=latents,
    #         timestep=t_batch,
    #         text_cond=text_cond,
    #         image_cond_edited=cond_edited,
    #         image_cond_original=cond_original,
    #         lam=lam,
    #     )
    #     latents = adapter.scheduler_step(v_guided, t, latents, t_next)

    # ------------------------------------------------------------------ #
    # 步骤 6：解码潜变量 → 像素帧                                          #
    # ------------------------------------------------------------------ #
    print(f"[DEBUG] final latents: min={latents.min():.3f} max={latents.max():.3f} mean={latents.mean():.3f}")
    return adapter.decode_latents(latents)


def _save_debug_video(frames: list, path: str, fps: float = 8.0, min_frames: int = 4) -> None:
    """将帧列表保存为 mp4。单帧时复制到 min_frames 帧，避免播放器显示绿色。"""
    if not frames:
        return
    out = frames if len(frames) >= min_frames else frames * (min_frames // len(frames) + 1)
    out = out[:max(len(frames), min_frames)]
    H, W = out[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in out:
        writer.write(f)
    writer.release()

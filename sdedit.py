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

    # ------------------------------------------------------------------ #
    # 步骤 2：分别编码"含标记"和"原始"帧的图像条件                       #
    # ------------------------------------------------------------------ #
    cond_edited   = adapter.encode_image_cond(frames_bgr_edited[0], latents_clean)
    cond_original = adapter.encode_image_cond(frame_bgr_original)

    # ------------------------------------------------------------------ #
    # 步骤 3：编码文本提示                                                  #
    # ------------------------------------------------------------------ #
    text_cond = adapter.encode_text(prompt)

    # ------------------------------------------------------------------ #
    # 步骤 4：在 t ≈ γ·T_max 处加噪（SDEdit 前向过程）                   #
    # 用 scheduler.add_noise() 确保与 DPM 加噪公式一致。                  #
    # ------------------------------------------------------------------ #
    noise = torch.randn_like(latents_clean, generator=generator)
    adapter.set_timesteps(scheduler_steps)
    timesteps = adapter.timesteps  # 从大到小，共 scheduler_steps 个时间步

    start_idx = min(int(scheduler_steps * gamma), scheduler_steps - 1)
    t_start   = timesteps[start_idx]

    latents = adapter.scheduler.add_noise(
        latents_clean,
        noise,
        t_start[None] if t_start.ndim == 0 else t_start,
    )

    # 从 start_idx 去噪到序列末尾（共 scheduler_steps - start_idx 步）
    timesteps_run = timesteps[start_idx:]

    # ------------------------------------------------------------------ #
    # 步骤 5：带反事实引导的去噪循环                                       #
    # ------------------------------------------------------------------ #
    for i, t in enumerate(timesteps_run):
        t_batch  = t.unsqueeze(0).to(device)
        t_next   = timesteps_run[i + 1] if i + 1 < len(timesteps_run) else torch.zeros_like(t)
        v_guided = adapter.predict_with_guidance(
            noisy_latents=latents,
            timestep=t_batch,
            text_cond=text_cond,
            image_cond_edited=cond_edited,
            image_cond_original=cond_original,
            lam=lam,
        )
        latents = adapter.scheduler_step(v_guided, t, latents, t_next)

    # ------------------------------------------------------------------ #
    # 步骤 6：解码潜变量 → 像素帧                                          #
    # ------------------------------------------------------------------ #
    return adapter.decode_latents(latents)

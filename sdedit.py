"""
Point Prompting 核心：带反事实增强引导的 SDEdit 去噪。

与模型无关，支持任意 ModelAdapter（CogVideoX-I2V 或 Wan2.1-I2V）。

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
    num_inference_steps: int = 50,
    prompt: str = "",
    generator: Optional[torch.Generator] = None,
) -> list:
    """执行一次带反事实增强引导的 SDEdit 完整流程。

    Args:
        adapter:             ModelAdapter（CogVideoXAdapter 或 WanAdapter）
        frames_bgr_edited:   frame[0] 含红色标记的视频帧列表
        frame_bgr_original:  不含标记的原始 frame[0]（用于负向条件）
        gamma:               SDEdit 加噪比例 γ ∈ (0, 1]，越大生成越自由
        lam:                 反事实引导权重 λ，越大标记越显著
        num_inference_steps: 总去噪步数
        prompt:              文本提示（论文零样本设置为空字符串）
        generator:           可复现性用的随机数生成器

    Returns:
        生成后的帧列表，每帧 (H, W, 3) BGR uint8
    """
    device = adapter.device
    dtype  = adapter.dtype

    # ------------------------------------------------------------------ #
    # 步骤 1：将编辑后的视频帧编码到潜变量空间                            #
    # ------------------------------------------------------------------ #
    latents_clean = adapter.encode_video(frames_bgr_edited)  # (1, C, T, lH, lW)

    # ------------------------------------------------------------------ #
    # 步骤 2：分别编码"含标记"和"原始"帧的图像条件                       #
    # 直接从已编码的 video latent 切第 0 帧作为正向条件，保证空间尺寸严格一致。
    # 负向条件单独编码原始帧（无标记），使用与 encode_video 相同路径。
    # ------------------------------------------------------------------ #
    cond_edited   = adapter.encode_image_cond(frames_bgr_edited[0], latents_clean)
    cond_original = adapter.encode_image_cond(frame_bgr_original)

    # ------------------------------------------------------------------ #
    # 步骤 3：编码文本提示                                                  #
    # ------------------------------------------------------------------ #
    text_cond = adapter.encode_text(prompt)

    # ------------------------------------------------------------------ #
    # 步骤 4：在 t ≈ γ·T_max 处加噪（Flow Matching SDEdit 前向过程）      #
    # x_t = (1-γ)·x_0 + γ·ε                                             #
    # ------------------------------------------------------------------ #
    noise = torch.randn_like(latents_clean, generator=generator)
    adapter.set_timesteps(num_inference_steps)
    timesteps = adapter.timesteps  # 从大到小的时间步序列

    # 根据 γ 找到去噪的起始时间步索引（跳过已经足够干净的步骤）
    t_start_idx    = max(0, int((1.0 - gamma) * num_inference_steps))
    timesteps_run  = timesteps[t_start_idx:]

    # 加噪比例 = 起始时间步 / 最大时间步 ≈ gamma（对非均匀调度器更鲁棒）
    t_val  = timesteps[t_start_idx].float()
    t_max  = timesteps[0].float()
    t_frac = (t_val / t_max).item()           # ≈ gamma，直接作为插值系数
    latents = adapter.add_noise(latents_clean, noise, t_frac)

    # ------------------------------------------------------------------ #
    # 步骤 5：带反事实引导的去噪循环                                       #
    # ------------------------------------------------------------------ #
    for t in timesteps_run:
        t_batch = t.unsqueeze(0).to(device)

        # 公式：v̂ = (λ+1)·v(c_edited) - λ·v(c_original)
        # predict_with_guidance 内部调用两次 forward_transformer
        v_guided = adapter.predict_with_guidance(
            noisy_latents=latents,
            timestep=t_batch,
            text_cond=text_cond,
            image_cond_edited=cond_edited,
            image_cond_original=cond_original,
            lam=lam,
        )
        latents = adapter.scheduler_step(v_guided, t, latents)

    # ------------------------------------------------------------------ #
    # 步骤 6：解码潜变量 → 像素帧                                          #
    # ------------------------------------------------------------------ #
    return adapter.decode_latents(latents)


# --------------------------------------------------------------------------- #
#  向后兼容的 CogVideoX 专用入口（内部直接委托给 run_sdedit）               #
# --------------------------------------------------------------------------- #

def run_sdedit_cogvideox(
    pipe,
    frames_bgr_edited: list,
    frame_bgr_original: np.ndarray,
    gamma: float = 0.5,
    lam: float = 8.0,
    num_inference_steps: int = 50,
    prompt: str = "",
    generator: Optional[torch.Generator] = None,
) -> list:
    """便捷封装：自动构建 CogVideoXAdapter 并调用 run_sdedit。"""
    from model_adapter import CogVideoXAdapter
    return run_sdedit(
        adapter=CogVideoXAdapter(pipe),
        frames_bgr_edited=frames_bgr_edited,
        frame_bgr_original=frame_bgr_original,
        gamma=gamma, lam=lam,
        num_inference_steps=num_inference_steps,
        prompt=prompt, generator=generator,
    )

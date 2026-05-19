"""
由粗到精的 Inpainting 精细化模块（论文 Section 3.3）。

在粗跟踪完成后，围绕每个检测到的标记位置构建时空二值掩码，
只对掩码区域内的潜变量重新去噪（inpainting），
掩码外的区域每步都从原始干净帧的潜变量中粘贴回来，
从而在保留标记可见性的同时消除初次 SDEdit 引入的伪影。

支持任意 ModelAdapter（CogVideoX-I2V 或 Wan2.1-I2V）。
"""

from __future__ import annotations

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional

from model_adapter import ModelAdapter


# 每个跟踪点周围重新去噪的圆形区域半径（像素）
INPAINT_RADIUS = 24


# --------------------------------------------------------------------------- #
#  空间掩码辅助函数                                                             #
# --------------------------------------------------------------------------- #

def _make_pixel_mask(H: int, W: int, points: np.ndarray, radius: int = INPAINT_RADIUS) -> np.ndarray:
    """构建像素级时空掩码：(T, H, W)，标记位置圆形区域内为 1，其余为 0。

    每帧以对应的 tracks[t] 为圆心，radius 为半径构建圆形区域。
    """
    T = len(points)
    mask = np.zeros((T, H, W), dtype=np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)  # 预计算像素坐标网格
    for t in range(T):
        cx, cy = points[t]
        # 圆形掩码：距中心 <= radius 的像素为 1
        mask[t] = ((xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2).astype(np.float32)
    return mask


def _downsample_mask(mask_px: np.ndarray, lH: int, lW: int, device, dtype) -> torch.Tensor:
    """将像素级掩码 (T, H, W) 下采样到潜变量空间分辨率 (1, 1, T, lH, lW)。

    形状中的 "1, 1" 分别对应批次维和通道维，便于与潜变量张量直接广播相乘。
    使用双线性插值保证掩码边界平滑过渡。
    """
    T = mask_px.shape[0]
    # 添加通道维 → (T, 1, H, W) 以满足 F.interpolate 的要求
    t = torch.from_numpy(mask_px).view(T, 1, *mask_px.shape[1:])
    t = F.interpolate(t, size=(lH, lW), mode="bilinear", align_corners=False)  # (T,1,lH,lW)
    # 调整为 (1, 1, T, lH, lW)：批次=1，通道=1（广播到 C 维）
    return t.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
#  精细化主函数                                                                 #
# --------------------------------------------------------------------------- #

def refine_tracks(
    adapter: ModelAdapter,
    frames_bgr_generated: list,
    frames_bgr_original: list,
    tracks: np.ndarray,              # (T, 2) 粗跟踪阶段检测到的标记坐标
    gamma: float = 0.3,              # 精细化加噪比例，小于主 SDEdit 的 γ
    num_inference_steps: int = 50,
    prompt: str = "",
    generator: Optional[torch.Generator] = None,
) -> list:
    """基于 Inpainting 的精细化去噪。

    仅对标记周围的小区域重新去噪，掩码外的区域在每步去噪后
    都从原始视频的潜变量粘贴回来，实现"局部修复"效果。

    Args:
        adapter:              ModelAdapter 实例
        frames_bgr_generated: 初次 SDEdit 生成的帧（含标记）
        frames_bgr_original:  颜色重平衡后的原始帧（不含标记）
        tracks:               (T, 2) 粗跟踪坐标，用于定位掩码中心
        gamma:                精细化的 SDEdit 加噪比例（应 < 主流程 γ）
        num_inference_steps:  总去噪步数
        prompt:               文本提示
        generator:            可复现性随机数生成器

    Returns:
        精细化后的帧列表，每帧 (H, W, 3) BGR uint8
    """
    device = adapter.device
    dtype  = adapter.dtype

    H, W = frames_bgr_generated[0].shape[:2]

    # ------------------------------------------------------------------ #
    # 步骤 1：将两组视频分别编码到潜变量空间                               #
    # ------------------------------------------------------------------ #
    lat_gen  = adapter.encode_video(frames_bgr_generated)   # (1, C, T, lH, lW) 含标记
    lat_orig = adapter.encode_video(frames_bgr_original)    # (1, C, T, lH, lW) 无标记

    _, C, T, lH, lW = lat_gen.shape

    # ------------------------------------------------------------------ #
    # 步骤 2：构建像素级和潜变量级时空掩码                                 #
    # ------------------------------------------------------------------ #
    mask_px  = _make_pixel_mask(H, W, tracks, INPAINT_RADIUS)    # (T, H, W)
    mask_lat = _downsample_mask(mask_px, lH, lW, device, dtype)  # (1,1,T,lH,lW)

    # ------------------------------------------------------------------ #
    # 步骤 3：掩码内区域加噪，掩码外保留原始干净潜变量                     #
    # ------------------------------------------------------------------ #
    noise     = torch.randn_like(lat_gen, generator=generator)
    lat_noisy = adapter.add_noise(lat_gen, noise, gamma)          # 含标记区域加噪
    # 掩码内用噪声潜变量，掩码外直接用原始潜变量（无需去噪）
    lat_start = mask_lat * lat_noisy + (1.0 - mask_lat) * lat_orig

    # ------------------------------------------------------------------ #
    # 步骤 4：编码条件信息                                                  #
    # ------------------------------------------------------------------ #
    text_cond  = adapter.encode_text(prompt)
    image_cond = adapter.encode_image_cond(frames_bgr_generated[0])  # 以生成帧第 0 帧为条件

    # ------------------------------------------------------------------ #
    # 步骤 5：仅运行后 γ 比例的去噪步骤（跳过前面已完成的步骤）           #
    # ------------------------------------------------------------------ #
    adapter.set_timesteps(num_inference_steps)
    timesteps     = adapter.timesteps
    t_start_idx   = max(0, int((1.0 - gamma) * num_inference_steps))
    timesteps_run = timesteps[t_start_idx:]

    latents = lat_start.clone()

    for t in timesteps_run:
        t_batch = t.unsqueeze(0).to(device)
        with torch.no_grad():
            v = adapter.forward_transformer(latents, t_batch, text_cond, image_cond)
        latents = adapter.scheduler_step(v, t, latents)

        # 关键：每步去噪后将掩码外的区域替换回原始潜变量
        # 确保精细化只影响标记附近，避免其他区域被模型随机改变
        latents = mask_lat * latents + (1.0 - mask_lat) * lat_orig

    # ------------------------------------------------------------------ #
    # 步骤 6：解码最终潜变量为像素帧                                        #
    # ------------------------------------------------------------------ #
    return adapter.decode_latents(latents)

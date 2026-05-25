"""
Point Prompting 主跟踪器。

完整流水线（论文 Section 3）：
  1. 颜色重平衡：抑制视频中的自然红色，防止干扰标记检测。
  2. 插入标记：在第 0 帧的查询点处绘制红色圆形标记。
  3. 反事实 SDEdit：以含标记帧为正向条件、原始帧为负向条件重新生成视频。
  4. 标记检测：在生成帧中逐帧检测标记的质心坐标。
  5. 精细化（可选）：对标记周围区域执行 inpainting 精细化。
  6. 返回轨迹坐标和可见性标志。

基于 Wan2.1-VACE-I2V。
"""

from __future__ import annotations

import numpy as np
import torch
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple

from color_rebalance import rebalance_video
from marker import insert_marker, track_marker_sequence
from sdedit import run_sdedit
from refinement import refine_tracks
from model_adapter import ModelAdapter, create_adapter


@dataclass
class TrackResult:
    """单次跟踪的结果。"""
    tracks: np.ndarray      # (T, 2) float32，每帧的 (x, y) 像素坐标
    visible: np.ndarray     # (T,)   bool，True 表示该帧标记可见
    generated_frames: list  # 生成帧列表 (H, W, 3) BGR uint8，用于调试可视化


@dataclass
class PointPrompterConfig:
    """跟踪器超参数配置。"""
    gamma: float = 0.5             # SDEdit 加噪比例（论文默认 0.5）
    lam: float = 8.0               # 反事实引导权重 λ（论文默认 8）
    scheduler_steps: int = 100     # 调度器总步数，决定时间步粒度（论文默认 100）
    marker_radius: int = 2         # 插入标记的圆形半径（像素）；论文消融最优值为 2px
    do_refine: bool = True         # 是否执行 inpainting 精细化
    refine_gamma: float = 0.7      # 精细化阶段的加噪比例（> gamma，噪声更小，编辑更保守）
    prompt: str = ""               # 文本提示（论文零样本设置为空字符串）
    seed: Optional[int] = None     # 随机种子，None 表示不固定
    model_width: int = 832         # 扩散模型输入的最大宽度，防止高分辨率视频 OOM
    model_height: int = 480        # 扩散模型输入的最大高度，防止高分辨率视频 OOM
    model_stride: int = 16         # 模型输入尺寸对齐倍数


def _aligned_size(width: int, height: int, max_width: int, max_height: int, stride: int) -> Tuple[int, int]:
    """按比例缩放到最大尺寸内，并将宽高向下对齐到 stride 倍数。"""
    if max_width <= 0 or max_height <= 0:
        return width, height
    scale = min(max_width / width, max_height / height, 1.0)
    out_w = int(width * scale)
    out_h = int(height * scale)
    if stride > 1:
        out_w = max(stride, (out_w // stride) * stride)
        out_h = max(stride, (out_h // stride) * stride)
    return out_w, out_h


def _resize_frames(frames_bgr: list, size: Tuple[int, int]) -> list:
    """Resize BGR frames to (width, height), preserving list layout."""
    width, height = size
    if not frames_bgr or (frames_bgr[0].shape[1] == width and frames_bgr[0].shape[0] == height):
        return frames_bgr
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames_bgr]


class PointPrompter:
    """基于预训练图像条件视频扩散模型的零样本点跟踪器。

    接受原始 diffusers pipeline（自动检测类型）或 ModelAdapter 实例。

    使用示例：
        pipe    = load_wan_vace_pipe("Wan-AI/Wan2.1-VACE-1.3B-diffusers")
        tracker = PointPrompter(pipe)
        result  = tracker.track(frames, query_point=(x, y))
    """

    def __init__(self, pipe_or_adapter, config: Optional[PointPrompterConfig] = None):
        # 兼容直接传入 pipeline 或 ModelAdapter 两种方式
        if isinstance(pipe_or_adapter, ModelAdapter):
            self.adapter = pipe_or_adapter
        else:
            self.adapter = create_adapter(pipe_or_adapter)  # 自动识别 pipeline 类型
        self.config = config or PointPrompterConfig()

    def track(
        self,
        frames_bgr: list,
        query_point: Tuple[float, float],
    ) -> TrackResult:
        """跟踪视频中的单个查询点。

        Args:
            frames_bgr:   视频帧列表，每帧 (H, W, 3) BGR uint8，共 T 帧
            query_point:  第 0 帧中需要跟踪的点坐标 (x, y)

        Returns:
            TrackResult：包含轨迹、可见性和生成帧
        """
        cfg = self.config
        # 固定随机种子以便复现
        gen = None
        if cfg.seed is not None:
            gen_device = self.adapter.device if str(self.adapter.device).startswith("cuda") else "cpu"
            gen = torch.Generator(device=gen_device).manual_seed(cfg.seed)
        orig_h, orig_w = frames_bgr[0].shape[:2]
        model_w, model_h = _aligned_size(orig_w, orig_h, cfg.model_width, cfg.model_height, cfg.model_stride)
        frames_model = _resize_frames(frames_bgr, (model_w, model_h))
        sx = model_w / orig_w
        sy = model_h / orig_h
        query_model = (query_point[0] * sx, query_point[1] * sy)
        marker_radius = max(2, int(round(cfg.marker_radius * min(sx, sy))))

        # ---- 步骤 1：颜色重平衡，抑制自然红色 ----
        # 保护查询点周围区域不被截断：若该点原本是红色，截断会使正负向条件差异消失
        protect_r = marker_radius * 4  # 留出足够余量覆盖标记及其邻域
        frames_rb = rebalance_video(frames_model, protect_point=query_model, protect_radius=protect_r)

        # ---- 步骤 2：在第 0 帧插入红色标记 ----
        frame0_original = frames_rb[0]
        frame0_marked   = insert_marker(frame0_original, query_model, marker_radius)
        frames_edited   = [frame0_marked] + frames_rb[1:]  # 仅第 0 帧含标记

        total_stages = 2 if cfg.do_refine else 1

        # ---- 步骤 3：反事实 SDEdit 生成含标记轨迹的视频 ----
        denoise_steps = cfg.scheduler_steps - int(cfg.scheduler_steps * cfg.gamma)
        print(f"  [阶段 1/{total_stages}] SDEdit 生成（调度器 {cfg.scheduler_steps} 步 / 去噪 {denoise_steps} 步）…")
        generated = run_sdedit(
            adapter=self.adapter,
            frames_bgr_edited=frames_edited,
            frame_bgr_original=frame0_original,  # 负向条件：第 0 帧无标记
            gamma=cfg.gamma,
            lam=cfg.lam,
            scheduler_steps=cfg.scheduler_steps,
            prompt=cfg.prompt,
            generator=gen,
            query_point=query_model,
        )

        # ---- 步骤 4：在生成帧中逐帧检测标记质心 ----
        tracks, visible = track_marker_sequence(generated, query_model)
        print(f"  [阶段 1/{total_stages}] 完成，可见帧 {visible.sum()}/{len(visible)}")

        # ---- 步骤 5：可选 inpainting 精细化 ----
        if cfg.do_refine:
            refine_denoise_steps = cfg.scheduler_steps - int(cfg.scheduler_steps * cfg.refine_gamma)
            print(f"  [阶段 2/{total_stages}] Inpainting 精细化（调度器 {cfg.scheduler_steps} 步 / 去噪 {refine_denoise_steps} 步）…")
            refined = refine_tracks(
                adapter=self.adapter,
                frames_bgr_generated=generated,
                frames_bgr_original=frames_rb,
                tracks=tracks,
                gamma=cfg.refine_gamma,
                scheduler_steps=cfg.scheduler_steps,
                prompt=cfg.prompt,
                generator=gen,
            )
            # 在精细化后的帧上重新检测，得到更准确的坐标
            tracks, visible = track_marker_sequence(refined, query_model)
            generated = refined
            print(f"  [阶段 2/{total_stages}] 完成，可见帧 {visible.sum()}/{len(visible)}")

        # 将 tracks 坐标从模型分辨率反算回输入帧分辨率（frames_bgr 的坐标系）
        if sx != 1.0 or sy != 1.0:
            tracks = tracks.copy()
            tracks[:, 0] /= sx
            tracks[:, 1] /= sy

        return TrackResult(tracks=tracks, visible=visible, generated_frames=generated)

    def track_multiple(
        self,
        frames_bgr: list,
        query_points: List[Tuple[float, float]],
    ) -> List[TrackResult]:
        """独立跟踪多个查询点（每个点单独运行完整流水线）。"""
        results = []
        for qp in query_points:
            results.append(self.track(frames_bgr, qp))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

"""
Point Prompting 主跟踪器。

完整流水线（论文 Section 3）：
  1. 颜色重平衡：抑制视频中的自然红色，防止干扰标记检测。
  2. 插入标记：在第 0 帧的查询点处绘制红色圆形标记。
  3. 反事实 SDEdit：以含标记帧为正向条件、原始帧为负向条件重新生成视频。
  4. 标记检测：在生成帧中逐帧检测标记的质心坐标。
  5. 精细化（可选）：对标记周围区域执行 inpainting 精细化。
  6. 返回轨迹坐标和可见性标志。

支持任意 ModelAdapter（CogVideoX 或 Wan）。
"""

from __future__ import annotations

import numpy as np
import torch
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
    gamma: float = 0.5           # SDEdit 加噪比例（论文默认 0.5）
    lam: float = 8.0             # 反事实引导权重 λ（论文默认 8）
    num_inference_steps: int = 50  # 扩散模型去噪总步数
    marker_radius: int = 8         # 插入标记的圆形半径（像素）
    do_refine: bool = True         # 是否执行 inpainting 精细化
    refine_gamma: float = 0.3      # 精细化阶段的加噪比例（< gamma）
    prompt: str = ""               # 文本提示（论文零样本设置为空字符串）
    seed: Optional[int] = None     # 随机种子，None 表示不固定


class PointPrompter:
    """基于预训练图像条件视频扩散模型的零样本点跟踪器。

    接受原始 diffusers pipeline（自动检测类型）或 ModelAdapter 实例。

    使用示例：
        # CogVideoX
        pipe    = load_cogvideox_pipe("THUDM/CogVideoX-5b-I2V")
        tracker = PointPrompter(pipe)

        # Wan 2.1
        pipe    = load_wan_pipe("Wan-AI/Wan2.1-I2V-14B-480P")
        tracker = PointPrompter(pipe)

        result = tracker.track(frames, query_point=(x, y))
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
        gen = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None

        # ---- 步骤 1：颜色重平衡，抑制自然红色 ----
        frames_rb = rebalance_video(frames_bgr)

        # ---- 步骤 2：在第 0 帧插入红色标记 ----
        frame0_original = frames_rb[0]
        frame0_marked   = insert_marker(frame0_original, query_point, cfg.marker_radius)
        frames_edited   = [frame0_marked] + frames_rb[1:]  # 仅第 0 帧含标记

        # ---- 步骤 3：反事实 SDEdit 生成含标记轨迹的视频 ----
        generated = run_sdedit(
            adapter=self.adapter,
            frames_bgr_edited=frames_edited,
            frame_bgr_original=frame0_original,  # 负向条件：无标记的原始帧
            gamma=cfg.gamma,
            lam=cfg.lam,
            num_inference_steps=cfg.num_inference_steps,
            prompt=cfg.prompt,
            generator=gen,
        )

        # ---- 步骤 4：在生成帧中逐帧检测标记质心 ----
        tracks, visible = track_marker_sequence(generated, query_point)

        # ---- 步骤 5：可选 inpainting 精细化 ----
        if cfg.do_refine:
            refined = refine_tracks(
                adapter=self.adapter,
                frames_bgr_generated=generated,
                frames_bgr_original=frames_rb,
                tracks=tracks,
                gamma=cfg.refine_gamma,
                num_inference_steps=cfg.num_inference_steps,
                prompt=cfg.prompt,
                generator=gen,
            )
            # 在精细化后的帧上重新检测，得到更准确的坐标
            tracks, visible = track_marker_sequence(refined, query_point)
            generated = refined

        return TrackResult(tracks=tracks, visible=visible, generated_frames=generated)

    def track_multiple(
        self,
        frames_bgr: list,
        query_points: List[Tuple[float, float]],
    ) -> List[TrackResult]:
        """独立跟踪多个查询点（每个点单独运行完整流水线）。"""
        return [self.track(frames_bgr, qp) for qp in query_points]

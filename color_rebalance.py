"""
颜色重平衡模块：对输入帧中的自然红色像素进行去饱和处理，
防止其被误识别为合成标记，从而干扰标记检测器。

原理：越接近红色色相（0°/180°）的像素，饱和度降低越多；
远离红色的像素（绿、蓝等）饱和度保持不变。
"""

import cv2
import numpy as np


# OpenCV HSV 色相范围为 [0, 180]；红色分布在约 [0,10] 和 [170,180]
_RED_HUE_MARGIN = 15   # 抑制边界宽度：距离红色 15 个色相单位以内开始衰减


def _red_proximity_weight(hue: np.ndarray) -> np.ndarray:
    """计算每个像素的"远红色"权重，取值 ∈ [0, 1]。

    权重 = 0：色相恰好是红色（hue=0 或 hue=180）
    权重 = 1：色相远离红色（如绿色 hue=60、蓝色 hue=120）

    `hue` 为 OpenCV 约定的整数数组，范围 [0, 180]。
    """
    # 分别计算到 0° 和 180° 的距离，取较小值（处理色相环绕）
    dist_to_zero = hue.astype(np.float32)
    dist_to_180  = (180 - hue).astype(np.float32)
    dist         = np.minimum(dist_to_zero, dist_to_180)

    # 线性映射：[0, _RED_HUE_MARGIN] → [0, 1]，超出范围截断到 1
    weight = np.clip(dist / _RED_HUE_MARGIN, 0.0, 1.0)
    return weight


def rebalance_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """对单帧 BGR 图像中的红色像素进行去饱和处理。

    处理步骤：
      1. BGR → HSV
      2. 对接近红色色相的像素按权重降低饱和度
      3. HSV → BGR 输出

    返回与输入相同 shape/dtype 的 BGR 帧（不修改原图）。
    """
    # 转换到 HSV 空间，用 float 防止运算溢出
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 计算每像素的去饱和权重（红色附近权重趋向 0）
    weight = _red_proximity_weight(h.astype(np.int32))
    s_new  = s * weight   # 红色区域饱和度被压低，其余区域不变

    hsv[..., 1] = s_new
    hsv_u8 = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)


def rebalance_video(frames_bgr: list) -> list:
    """对视频的每一帧独立应用 rebalance_frame。"""
    return [rebalance_frame(f) for f in frames_bgr]

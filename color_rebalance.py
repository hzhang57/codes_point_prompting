"""
颜色重平衡模块：对输入帧中靠近红色色相的像素进行硬性饱和度截断，
防止自然红色被误识别为合成标记，同时尽量保留原始色彩信息。

论文做法（Appendix B.1）：HSV 色相在 [-30°, 10°] 范围内的像素，
将其饱和度截断至最大 80（OpenCV 0-255 范围），其余像素不变。

扩展：若查询点处原本就是红色，可传入 protect_point / protect_radius
跳过该圆形区域的截断，避免查询点颜色被平坦化后与合成标记的对比度消失。
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# 饱和度截断上限（论文指定 80，对应 OpenCV 0-255 范围）
_MAX_SAT_RED = 80

# 红色色相邻域范围（OpenCV 色相单位，0-180）
# 标准 HSV [-30°, 10°] 对应 OpenCV [150, 180] ∪ [0, 5]，
# 此处取更保守的 [165, 180] ∪ [0, 10]（约 ±30° 覆盖）
_RED_HUE_MARGIN = 15   # 距离 0 或 180 在此范围内视为红色


def rebalance_frame(
    frame_bgr: np.ndarray,
    protect_point: Optional[Tuple[float, float]] = None,
    protect_radius: int = 0,
) -> np.ndarray:
    """对单帧 BGR 图像中红色色相区域进行饱和度硬截断。

    处理步骤：
      1. BGR → HSV
      2. 标记红色区域（hue ≤ _RED_HUE_MARGIN 或 hue ≥ 180-_RED_HUE_MARGIN）
      3. 若指定了 protect_point，从掩码中排除该圆形区域（保留原色）
      4. 对剩余红色区域的饱和度截断至最大 _MAX_SAT_RED
      5. HSV → BGR 输出

    Args:
        frame_bgr:      (H, W, 3) BGR uint8 图像
        protect_point:  (x, y) 需要保护的点坐标；若为 None 则不保护任何区域
        protect_radius: 保护圆的半径（像素）；为 0 则不保护

    返回与输入相同 shape/dtype 的 BGR 帧（不修改原图）。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).copy()
    h = hsv[:, :, 0].astype(np.int32)

    # 红色区域掩码（距色相 0° 或 180° 不超过 _RED_HUE_MARGIN）
    is_red = (h <= _RED_HUE_MARGIN) | (h >= 180 - _RED_HUE_MARGIN)

    # 从截断掩码中排除查询点周围的保护区域
    if protect_point is not None and protect_radius > 0:
        H, W = frame_bgr.shape[:2]
        cx, cy = protect_point
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        protected = (xs - cx) ** 2 + (ys - cy) ** 2 <= protect_radius ** 2
        is_red = is_red & ~protected

    # 硬截断：饱和度超过 _MAX_SAT_RED 的红色像素降至上限
    s = hsv[:, :, 1].astype(np.int32)
    s[is_red] = np.minimum(s[is_red], _MAX_SAT_RED)
    hsv[:, :, 1] = s.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rebalance_video(
    frames_bgr: list,
    protect_point: Optional[Tuple[float, float]] = None,
    protect_radius: int = 0,
) -> list:
    """对视频的每一帧独立应用 rebalance_frame。

    protect_point / protect_radius 透传到每一帧，保护查询点区域不被截断。
    """
    return [rebalance_frame(f, protect_point, protect_radius) for f in frames_bgr]

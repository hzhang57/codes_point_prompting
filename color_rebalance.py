"""
颜色重平衡模块：对输入帧中靠近红色色相的像素进行硬性饱和度截断，
防止自然红色被误识别为合成标记，同时尽量保留原始色彩信息。

论文做法（Appendix B.1）：HSV 色相在 [-30°, 10°] 范围内的像素，
将其饱和度截断至最大 80（OpenCV 0-255 范围），其余像素不变。
"""

import cv2
import numpy as np


# 饱和度截断上限（论文指定 80，对应 OpenCV 0-255 范围）
_MAX_SAT_RED = 80

# 红色色相邻域范围（OpenCV 色相单位，0-180）
# 标准 HSV [-30°, 10°] 对应 OpenCV [150, 180] ∪ [0, 5]，
# 此处取更保守的 [165, 180] ∪ [0, 10]（约 ±30° 覆盖）
_RED_HUE_MARGIN = 15   # 距离 0 或 180 在此范围内视为红色


def rebalance_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """对单帧 BGR 图像中红色色相区域进行饱和度硬截断。

    处理步骤：
      1. BGR → HSV
      2. 标记红色区域（hue ≤ _RED_HUE_MARGIN 或 hue ≥ 180-_RED_HUE_MARGIN）
      3. 对该区域的饱和度截断至最大 _MAX_SAT_RED
      4. HSV → BGR 输出

    返回与输入相同 shape/dtype 的 BGR 帧（不修改原图）。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).copy()
    h = hsv[:, :, 0].astype(np.int32)

    # 红色区域掩码（距色相 0° 或 180° 不超过 _RED_HUE_MARGIN）
    is_red = (h <= _RED_HUE_MARGIN) | (h >= 180 - _RED_HUE_MARGIN)

    # 硬截断：饱和度超过 _MAX_SAT_RED 的红色像素降至上限
    s = hsv[:, :, 1].astype(np.int32)
    s[is_red] = np.minimum(s[is_red], _MAX_SAT_RED)
    hsv[:, :, 1] = s.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rebalance_video(frames_bgr: list) -> list:
    """对视频的每一帧独立应用 rebalance_frame。"""
    return [rebalance_frame(f) for f in frames_bgr]

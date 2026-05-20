"""
Point Prompting 的标记插入与检测模块。

在视频帧的查询点处插入一个醒目的红色圆形标记，
然后利用 HSV 颜色阈值法在生成帧中逐帧检测该标记的位置。
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ---- 标记外观参数 ----
MARKER_RADIUS = 2             # 标记圆的半径（像素）；论文消融最优值为 2px
MARKER_COLOR_BGR = (0, 0, 255)  # 纯红（BGR 格式：B=0, G=0, R=255）
MARKER_THICKNESS = -1         # -1 表示填充实心圆

# ---- HSV 红色检测阈值 ----
# OpenCV 中色相范围为 [0, 180]，红色横跨 0°/180° 边界，因此需要两段范围
_HUE_LO1, _HUE_HI1 = 0, 10      # 红色区间1：色相 0–10°
_HUE_LO2, _HUE_HI2 = 170, 180   # 红色区间2：色相 170–180°（环绕）
_SAT_LO, _SAT_HI = 100, 255     # 饱和度范围：论文 150，放宽至 100 以兼容生成帧中褪色的标记
_VAL_LO, _VAL_HI = 80, 255      # 亮度范围：排除过暗像素

# ---- 搜索窗口参数 ----
DEFAULT_SEARCH_RADIUS = 90   # 默认以上一帧位置为中心的搜索半径（像素）
MAX_SEARCH_RADIUS = 150      # 连续丢失帧时搜索半径的最大扩展值
_REFINE_RADIUS = 20          # 质心精炼：仅聚合距最近红色像素 20px 以内的像素


def insert_marker(frame: np.ndarray, point: Tuple[int, int], radius: int = MARKER_RADIUS) -> np.ndarray:
    """在 `frame`（H, W, 3 BGR）的 `point`（x, y）处绘制红色实心圆标记。

    返回修改后的副本，不改变原始帧。
    """
    out = frame.copy()  # 不修改原帧
    cv2.circle(out, (int(point[0]), int(point[1])), radius, MARKER_COLOR_BGR, MARKER_THICKNESS)
    return out


def _red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """对 BGR 帧进行 HSV 阈值分割，返回红色像素的二值掩码。

    因红色在 OpenCV HSV 空间跨越 0°/180° 边界，需合并两段阈值区间。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 低色相区间（0–10°）
    lo1 = np.array([_HUE_LO1, _SAT_LO, _VAL_LO])
    hi1 = np.array([_HUE_HI1, _SAT_HI, _VAL_HI])
    # 高色相区间（170–180°，对应负角度红色）
    lo2 = np.array([_HUE_LO2, _SAT_LO, _VAL_LO])
    hi2 = np.array([_HUE_HI2, _SAT_HI, _VAL_HI])

    mask1 = cv2.inRange(hsv, lo1, hi1)
    mask2 = cv2.inRange(hsv, lo2, hi2)
    return cv2.bitwise_or(mask1, mask2)  # 合并两段掩码


def detect_marker(
    frame_bgr: np.ndarray,
    prev_point: Tuple[float, float],
    search_radius: int = DEFAULT_SEARCH_RADIUS,
) -> Optional[Tuple[float, float]]:
    """在 `prev_point` 附近的局部搜索窗口内检测红色标记质心。

    只在以上一帧位置为中心、边长为 2*search_radius 的矩形窗口内搜索，
    避免被画面其他区域的红色像素干扰。

    返回 (x, y) 质心坐标，若未检测到则返回 None。
    """
    h, w = frame_bgr.shape[:2]
    px, py = int(prev_point[0]), int(prev_point[1])

    # 裁剪局部搜索窗口（边界夹紧到图像范围内）
    x1 = max(0, px - search_radius)
    y1 = max(0, py - search_radius)
    x2 = min(w, px + search_radius)
    y2 = min(h, py + search_radius)
    crop = frame_bgr[y1:y2, x1:x2]

    mask = _red_mask(crop)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None  # 窗口内无红色像素，标记丢失

    # 找距上一帧位置最近的红色像素作为锚点（裁剪坐标系）
    crop_px, crop_py = px - x1, py - y1
    dists = np.sqrt((xs - crop_px) ** 2 + (ys - crop_py) ** 2)
    anchor_idx = int(np.argmin(dists))
    anchor_x, anchor_y = xs[anchor_idx], ys[anchor_idx]

    # 仅聚合锚点 _REFINE_RADIUS 范围内的红色像素，抑制离群点
    near = np.sqrt((xs - anchor_x) ** 2 + (ys - anchor_y) ** 2) <= _REFINE_RADIUS
    if not near.any():
        near = np.ones(len(xs), dtype=bool)
    cx = float(xs[near].mean()) + x1
    cy = float(ys[near].mean()) + y1
    return (cx, cy)


def track_marker_sequence(
    frames_bgr: list,
    query_point: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """逐帧检测标记位置，连续丢失时自动扩大搜索半径。

    丢失帧的位置沿用上一帧的最后已知位置（向前传播），
    搜索半径每次丢失后扩大 10%，直到上限 MAX_SEARCH_RADIUS。
    重新检测到后恢复默认搜索半径。

    返回：
        tracks:  (T, 2) float32，每帧的 (x, y) 坐标
        visible: (T,) bool，True 表示该帧成功检测到标记
    """
    T = len(frames_bgr)
    tracks  = np.zeros((T, 2), dtype=np.float32)
    visible = np.zeros(T, dtype=bool)

    # 第 0 帧直接使用查询点，不做检测
    tracks[0]  = query_point
    visible[0] = True

    search_radius = DEFAULT_SEARCH_RADIUS
    prev = query_point

    for t in range(1, T):
        result = detect_marker(frames_bgr[t], prev, search_radius)
        if result is not None:
            # 检测成功：更新位置，重置搜索半径
            tracks[t]  = result
            visible[t] = True
            prev          = result
            search_radius = DEFAULT_SEARCH_RADIUS
        else:
            # 检测失败：传播上一位置，扩大搜索半径
            tracks[t]  = prev
            visible[t] = False
            search_radius = min(int(search_radius * 1.1), MAX_SEARCH_RADIUS)

    return tracks, visible

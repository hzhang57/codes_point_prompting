"""
方案 A：Confetti 网格稠密点跟踪 demo（基于 Wan2.2-TI2V-5B 反事实 SDEdit）。

核心思想：从"一个红点"升级为"一张周期编码的彩点网"，一次生成跟踪全图。

  1. 全帧低饱和化（S *= 0.25），把色相空间腾给标记；
  2. 首帧铺 confetti 点阵：色相沿 x 周期排列（6 色 × 30°），明度沿 y 两档交替，
     单点码字（6×2=12）只需局部唯一；
  3. 复用 demo_moe.py 的官方 TI2V 反事实 SDEdit 管线生成（marked=铺点首帧，
     original=无点首帧）；
  4. 解码：饱和度阈值检测彩点 → 码字分类（首帧自标定）→ 运动连续性消歧
     （"钟表原理"：颜色=分针给精度，上一帧位置=时针消循环歧义）→
     邻域位移中值滤波剔除误匹配。

实验 0（go/no-go）：
  python demo_moe_confetti.py --video input.mp4 --spacing 48
  人眼检查 outputs/demo_moe_confetti/generated.mp4：点阵是跟着物体平流，
  还是贴在屏幕上不动/被抹掉/晕染——这直接裁决方案可行性。
  密度消融：--spacing 64 / 48 / 32。

输出目录结构：
  inputs/frame0_confetti.png   铺点后的首帧（marked 条件）
  inputs/frame0_original.png   无点首帧（counterfactual 条件）
  inputs/frame0_pair.png       两个条件帧的左右拼接对比
  inputs/input_desat.mp4       低饱和化后的输入视频
  inputs/input_marked.mp4      实际送入扩散的视频（confetti 首帧 + 后续帧）
  generated.mp4                反事实 SDEdit 生成结果
  generated_overlay.mp4        生成结果 + 解码出的追踪点
  tracks_trails.mp4            追踪结果带轨迹尾迹版
  compare.mp4                  输入(上) vs 生成+追踪(下) 拼接对比
  tracks.npz / summary.json    轨迹数组与全部统计
  （--save-frames 额外导出逐帧 PNG；--decode-noisy 额外导出 stage1/noisy.mp4）

无 GPU 自检（不加载模型，验证插点与解码器闭环）：
  python demo_moe_confetti.py --video input.mp4 --preview-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---- 码字调色板 ----
HUE_CLASSES = (0, 30, 60, 90, 120, 150)  # OpenCV H ∈ [0,180)，间隔 30° 抗漂移
VALUE_LEVELS = (255, 160)                # 明度两档：亮 / 暗
MARKER_SATURATION = 255

# ---- 默认分辨率（与 demo_moe 的 t4 preset 一致，避免 import torch 链）----
T4_LANDSCAPE_SIZE = (832, 480)
T4_PORTRAIT_SIZE = (480, 832)
SIZE_ALIGN = 32  # VAE /16 × patch /2


@dataclass
class GridPoint:
    index: int
    gx: int
    gy: int
    x: float
    y: float
    hue_idx: int
    val_idx: int
    color_bgr: Tuple[int, int, int]

    @property
    def codeword(self) -> Tuple[int, int]:
        return (self.hue_idx, self.val_idx)


@dataclass
class Detection:
    x: float
    y: float
    hue: float
    val: float
    area: int


@dataclass
class CodewordDecoder:
    """首帧自标定的码字分类器：用生成视频里标记的实测颜色分布做最近邻。"""

    hue_centers: List[float]
    val_threshold: float

    def classify(self, det: Detection) -> Tuple[int, int]:
        dists = [_circular_hue_dist(det.hue, c) for c in self.hue_centers]
        hue_idx = int(np.argmin(dists))
        val_idx = 0 if det.val >= self.val_threshold else 1
        return (hue_idx, val_idx)


def _circular_hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _circular_hue_mean(hues: np.ndarray) -> float:
    ang = np.deg2rad(hues.astype(np.float64) * 2.0)
    mean = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
    return (math.degrees(mean) % 360.0) / 2.0


def de_bruijn_sequence(k: int, n: int) -> List[int]:
    """k 元、窗口 n 的 De Bruijn 序列：任意连续 n 个元素的组合全局唯一。

    v1 解码暂未使用窗口重定位，预留给丢失点的 De Bruijn 找回。
    """
    a = [0] * (k * n)
    seq: List[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


def palette_color_bgr(hue_idx: int, val_idx: int) -> Tuple[int, int, int]:
    hsv = np.uint8([[[HUE_CLASSES[hue_idx], MARKER_SATURATION, VALUE_LEVELS[val_idx]]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(b), int(g), int(r))


def build_confetti_grid(
    width: int,
    height: int,
    spacing: int,
    jitter: int,
    hue_coding: str = "diagonal",
    seed: int = 42,
) -> List[GridPoint]:
    """生成周期编码的 confetti 网格。

    diagonal（推荐）：hue_idx = (gx + 3*gy) % 6，上下相邻行色相相差 90°，
    即使明度档在生成中漂移失效，仅靠色相仍有 2 格的垂直安全距离。
    periodic：hue_idx = gx % 6，同列各行色相相同，依赖明度档区分相邻行。
    debruijn：相邻 3 点色相组合全局唯一，预留窗口重定位能力。
    三种编码同码字最小间隔都是 2 行（2*spacing）。
    """
    rng = np.random.default_rng(seed)
    margin = max(spacing // 2, 8)
    xs = list(range(margin, width - margin + 1, spacing))
    ys = list(range(margin, height - margin + 1, spacing))
    seq = de_bruijn_sequence(len(HUE_CLASSES), 3) if hue_coding == "debruijn" else None

    points: List[GridPoint] = []
    for gy, y0 in enumerate(ys):
        for gx, x0 in enumerate(xs):
            if seq is not None:
                hue_idx = seq[gx % len(seq)]
            elif hue_coding == "diagonal":
                hue_idx = (gx + 3 * gy) % len(HUE_CLASSES)
            else:
                hue_idx = gx % len(HUE_CLASSES)
            val_idx = gy % len(VALUE_LEVELS)
            # 轻微随机抖动：破坏完美规则性，增强"颜料"而非"滤镜图案"的解释
            jx = float(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0.0
            jy = float(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0.0
            x = float(np.clip(x0 + jx, 2, width - 3))
            y = float(np.clip(y0 + jy, 2, height - 3))
            points.append(
                GridPoint(
                    index=len(points),
                    gx=gx,
                    gy=gy,
                    x=x,
                    y=y,
                    hue_idx=hue_idx,
                    val_idx=val_idx,
                    color_bgr=palette_color_bgr(hue_idx, val_idx),
                )
            )
    return points


def insert_confetti(frame_bgr: np.ndarray, grid: List[GridPoint], radius: int) -> np.ndarray:
    out = frame_bgr.copy()
    for p in grid:
        cv2.circle(out, (int(round(p.x)), int(round(p.y))), radius, p.color_bgr, -1)
    return out


def desaturate_frame(frame_bgr: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= factor
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def desaturate_video(frames: List[np.ndarray], factor: float) -> List[np.ndarray]:
    if factor >= 1.0:
        return frames
    return [desaturate_frame(f, factor) for f in frames]


# --------------------------------------------------------------------------- #
#  解码器                                                                       #
# --------------------------------------------------------------------------- #


def detect_confetti(
    frame_bgr: np.ndarray,
    sat_thresh: int = 140,
    val_lo: int = 70,
    min_area: int = 4,
    max_area: int = 900,
) -> List[Detection]:
    """饱和度阈值 + 连通域：背景已压到 S<=0.25*255≈64，标记 S≈255，间隔巨大。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((s >= sat_thresh) & (v >= val_lo)).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets: List[Detection] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        hgt = stats[i, cv2.CC_STAT_HEIGHT]
        sub = labels[y0 : y0 + hgt, x0 : x0 + w] == i
        hue = _circular_hue_mean(h[y0 : y0 + hgt, x0 : x0 + w][sub])
        val = float(v[y0 : y0 + hgt, x0 : x0 + w][sub].mean())
        cx, cy = centroids[i]
        dets.append(Detection(float(cx), float(cy), hue, val, area))
    return dets


def match_frame0(
    dets: List[Detection], grid: List[GridPoint], max_dist: float
) -> Tuple[dict, CodewordDecoder]:
    """首帧：按位置最近邻把检测分配给已知网格点，并自标定码字分类器。"""
    assigned: dict = {}
    if dets:
        det_xy = np.array([[d.x, d.y] for d in dets])
        used = set()
        for p in grid:
            dist = np.hypot(det_xy[:, 0] - p.x, det_xy[:, 1] - p.y)
            order = np.argsort(dist)
            for j in order:
                if dist[j] > max_dist:
                    break
                if int(j) not in used:
                    used.add(int(j))
                    assigned[p.index] = dets[int(j)]
                    break

    # 自标定：每个色相类 / 明度档的实测分布
    hue_centers = []
    for c in range(len(HUE_CLASSES)):
        hues = np.array(
            [d.hue for idx, d in assigned.items() if grid[idx].hue_idx == c]
        )
        hue_centers.append(_circular_hue_mean(hues) if len(hues) else float(HUE_CLASSES[c]))
    val_means = []
    for lv in range(len(VALUE_LEVELS)):
        vals = [d.val for idx, d in assigned.items() if grid[idx].val_idx == lv]
        val_means.append(float(np.mean(vals)) if vals else float(VALUE_LEVELS[lv]))
    val_threshold = (val_means[0] + val_means[1]) / 2.0
    return assigned, CodewordDecoder(hue_centers, val_threshold)


def _grid_neighbors(grid: List[GridPoint]) -> List[List[int]]:
    by_cell = {(p.gx, p.gy): p.index for p in grid}
    neighbors: List[List[int]] = []
    for p in grid:
        ns = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = by_cell.get((p.gx + dx, p.gy + dy))
                if j is not None:
                    ns.append(j)
        neighbors.append(ns)
    return neighbors


def _two_means(values: List[float], init: List[float], iters: int = 8) -> List[float]:
    """一维 2-means（Lloyd 迭代），用上一帧的中心做初始化以保持档位连贯。"""
    c = [float(init[0]), float(init[1])]
    arr = np.asarray(values, dtype=np.float64)
    for _ in range(iters):
        assign = np.abs(arr - c[0]) > np.abs(arr - c[1])
        for k, sel in enumerate((~assign, assign)):
            if sel.any():
                c[k] = float(arr[sel].mean())
    if c[0] < c[1]:  # 约定 c[0]=亮档
        c = [c[1], c[0]]
    return c


def _circ_ema(old_hue: float, new_hue: float, alpha: float) -> float:
    a_old = math.radians(old_hue * 2.0)
    a_new = math.radians(new_hue * 2.0)
    x = (1 - alpha) * math.cos(a_old) + alpha * math.cos(a_new)
    y = (1 - alpha) * math.sin(a_old) + alpha * math.sin(a_new)
    return (math.degrees(math.atan2(y, x)) % 360.0) / 2.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return 99.0  # 上限值，保持 JSON 可序列化
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(s.mean())


def compute_sync_metrics(
    generated: List[np.ndarray], reference: List[np.ndarray]
) -> List[dict]:
    """逐帧同步性：生成帧 vs 输入帧的灰度 PSNR/SSIM。

    量化"生成视频是否还在复刻输入运动"——同步性是反事实跟踪方法的地基。
    预期不同步从某帧开始时，曲线会出现清晰下跌拐点（常与 latent chunk
    边界对齐）。用灰度通道规避标记/去饱和带来的色度差异。
    """
    rows = []
    for t in range(min(len(generated), len(reference))):
        a = cv2.cvtColor(generated[t], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(reference[t], cv2.COLOR_BGR2GRAY)
        rows.append(
            {"frame": t, "psnr": round(_psnr(a, b), 2), "ssim": round(_ssim_gray(a, b), 4)}
        )
    return rows


def print_sync_metrics(rows: List[dict]) -> None:
    print("\n同步性（生成 vs 输入，灰度）：")
    for r in rows:
        print(f"  t={r['frame']:02d} PSNR {r['psnr']:6.2f}dB  SSIM {r['ssim']:.4f}")
    if len(rows) > 1:
        drops = [rows[t - 1]["psnr"] - rows[t]["psnr"] for t in range(1, len(rows))]
        worst = int(np.argmax(drops)) + 1
        print(
            f"  最大单帧 PSNR 跌幅：t={worst}（-{max(drops):.2f}dB）"
            + ("  <- 疑似漂移拐点" if max(drops) > 1.5 else "")
        )


def track_confetti_sequence(
    frames: List[np.ndarray],
    grid: List[GridPoint],
    search_radius: float,
    sat_thresh: int = 140,
    val_lo: int = 70,
    reject_deviation: Optional[float] = None,
    hue_gate: float = 25.0,
    hue_weight: float = 0.8,
    val_penalty: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """解码器 v2：钟表原理（码字=分针，上一帧位置=时针）+ 三项鲁棒化。

    相对 v1 的改动：
      1. 明度档逐帧再标定（2-means 自适应），消除生成亮度衰减导致的码字集体翻转；
      2. 软匹配：cost = 位置距离 + hue_weight*色相差 + 明度档不符惩罚，
         色相差超过 hue_gate 才硬性排除；
      3. 丢失轨迹随已匹配邻居的位移中值平流（兜底用全局中值），
         标记重现时可在正确位置附近重捕获——拆掉"丢失即永久"的死亡螺旋。
    """
    n_pts, n_frames = len(grid), len(frames)
    height, width = frames[0].shape[:2]
    tracks = np.zeros((n_pts, n_frames, 2), dtype=np.float64)
    visible = np.zeros((n_pts, n_frames), dtype=bool)
    stats: List[dict] = []
    if reject_deviation is None:
        reject_deviation = 0.75 * search_radius
    neighbors = _grid_neighbors(grid)

    dets0 = detect_confetti(frames[0], sat_thresh, val_lo)
    assigned0, decoder = match_frame0(dets0, grid, max_dist=search_radius * 0.5)
    hue_centers = list(decoder.hue_centers)
    level_means = [float(VALUE_LEVELS[0]), float(VALUE_LEVELS[1])]
    v0 = [
        [d.val for idx, d in assigned0.items() if grid[idx].val_idx == lv]
        for lv in range(len(VALUE_LEVELS))
    ]
    for lv in range(len(VALUE_LEVELS)):
        if v0[lv]:
            level_means[lv] = float(np.mean(v0[lv]))

    last_pos = np.array([[p.x, p.y] for p in grid], dtype=np.float64)
    for idx, det in assigned0.items():
        last_pos[idx] = (det.x, det.y)
        visible[idx, 0] = True
    tracks[:, 0] = last_pos
    stats.append(
        {
            "frame": 0,
            "detections": len(dets0),
            "matched": len(assigned0),
            "val_level_means": [round(c, 1) for c in level_means],
        }
    )

    for t in range(1, n_frames):
        dets = detect_confetti(frames[t], sat_thresh, val_lo)

        # 1) 明度档逐帧再标定：用上一帧中心初始化的 2-means
        det_vals = [d.val for d in dets]
        if len(det_vals) >= 4:
            level_means = _two_means(det_vals, level_means)
        det_levels = [
            0 if abs(d.val - level_means[0]) <= abs(d.val - level_means[1]) else 1
            for d in dets
        ]

        # 2) 软匹配：色相门限内按综合代价贪心一对一分配
        pairs = []
        for i in range(n_pts):
            hx, hy = last_pos[i]
            for j, det in enumerate(dets):
                dist = math.hypot(det.x - hx, det.y - hy)
                if dist > search_radius:
                    continue
                hue_d = _circular_hue_dist(det.hue, hue_centers[grid[i].hue_idx])
                if hue_d > hue_gate:
                    continue
                cost = dist + hue_weight * hue_d
                if det_levels[j] != grid[i].val_idx:
                    cost += val_penalty
                pairs.append((cost, i, j))
        pairs.sort(key=lambda p: p[0])
        used_tracks, used_dets = set(), set()
        matched: dict = {}
        for _cost, i, j in pairs:
            if i in used_tracks or j in used_dets:
                continue
            used_tracks.add(i)
            used_dets.add(j)
            matched[i] = j

        # 邻域位移中值滤波：和周围邻居都拧着的匹配基本是误检
        disps = {
            i: np.array([dets[j].x - last_pos[i, 0], dets[j].y - last_pos[i, 1]])
            for i, j in matched.items()
        }
        rejected = set()
        for i in list(matched):
            nb = [disps[k] for k in neighbors[i] if k in disps]
            if len(nb) < 3:
                continue
            med = np.median(np.stack(nb), axis=0)
            if np.linalg.norm(disps[i] - med) > reject_deviation:
                rejected.add(i)
        for i in rejected:
            del matched[i]
            del disps[i]

        # 色相中心 EMA 跟随生成漂移
        for c in range(len(HUE_CLASSES)):
            hues = [dets[j].hue for i, j in matched.items() if grid[i].hue_idx == c]
            if hues:
                hue_centers[c] = _circ_ema(hue_centers[c], _circular_hue_mean(np.array(hues)), 0.3)

        # 3) 更新：匹配点取检测位置；丢失点随邻居（兜底全局）位移中值平流
        global_med = (
            np.median(np.stack(list(disps.values())), axis=0)
            if disps
            else np.zeros(2)
        )
        advected = 0
        for i in range(n_pts):
            if i in matched:
                det = dets[matched[i]]
                last_pos[i] = (det.x, det.y)
                visible[i, t] = True
            else:
                nb = [disps[k] for k in neighbors[i] if k in disps]
                med = np.median(np.stack(nb), axis=0) if len(nb) >= 2 else global_med
                last_pos[i, 0] = float(np.clip(last_pos[i, 0] + med[0], 0, width - 1))
                last_pos[i, 1] = float(np.clip(last_pos[i, 1] + med[1], 0, height - 1))
                advected += 1
            tracks[i, t] = last_pos[i]
        stats.append(
            {
                "frame": t,
                "detections": len(dets),
                "matched": len(matched),
                "rejected_by_smoothness": len(rejected),
                "advected": advected,
                "val_level_means": [round(c, 1) for c in level_means],
            }
        )
    return tracks, visible, stats


def draw_confetti_tracks(
    frames: List[np.ndarray],
    grid: List[GridPoint],
    tracks: np.ndarray,
    visible: np.ndarray,
) -> List[np.ndarray]:
    out = []
    for t, frame in enumerate(frames):
        vis = frame.copy()
        for p in grid:
            x, y = int(round(tracks[p.index, t, 0])), int(round(tracks[p.index, t, 1]))
            if visible[p.index, t]:
                cv2.circle(vis, (x, y), 4, p.color_bgr, -1)
                cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)
            else:
                cv2.circle(vis, (x, y), 2, (128, 128, 128), -1)
        out.append(vis)
    return out


def draw_confetti_trails(
    frames: List[np.ndarray],
    grid: List[GridPoint],
    tracks: np.ndarray,
    visible: np.ndarray,
    trail_len: int = 8,
) -> List[np.ndarray]:
    """带轨迹尾迹的可视化：每个点拖出最近 trail_len 帧的运动线。"""
    out = []
    for t, frame in enumerate(frames):
        vis = frame.copy()
        t0 = max(0, t - trail_len)
        for p in grid:
            for k in range(t0, t):
                if not (visible[p.index, k] and visible[p.index, k + 1]):
                    continue
                a = tuple(int(round(c)) for c in tracks[p.index, k])
                b = tuple(int(round(c)) for c in tracks[p.index, k + 1])
                cv2.line(vis, a, b, p.color_bgr, 1, cv2.LINE_AA)
            if visible[p.index, t]:
                x, y = int(round(tracks[p.index, t, 0])), int(round(tracks[p.index, t, 1]))
                cv2.circle(vis, (x, y), 3, p.color_bgr, -1)
        out.append(vis)
    return out


# --------------------------------------------------------------------------- #
#  视频 IO（本地实现，preview 模式不依赖 torch）                                 #
# --------------------------------------------------------------------------- #


def load_video(
    path: str, max_frames: Optional[int] = None, stride: int = 1
) -> Tuple[List[np.ndarray], float]:
    """读取视频，每 stride 帧取一帧；返回的 fps 已按 stride 折算。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    stride = max(1, int(stride))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, float(fps) / stride


def save_video(frames: List[np.ndarray], path: str, fps: float) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio

        imageio.mimsave(
            path,
            [frame[..., ::-1] for frame in frames],
            fps=fps,
            codec="libx264",
            output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
        )
        print(f"[save] {path} ({len(frames)} frames, imageio/libx264)")
        return
    except Exception as exc:
        print(f"[save] imageio failed ({exc}), fallback to cv2")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[save] {path} ({len(frames)} frames, cv2/mp4v)")


def resize_video(frames: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
    if not frames or (frames[0].shape[1] == width and frames[0].shape[0] == height):
        return frames
    return [cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA) for f in frames]


def save_frames(frames: List[np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f"{i:03d}.png"), frame)
    print(f"[save] {output_dir}/ ({len(frames)} frames, png)")


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_pair_image(marked: np.ndarray, original: np.ndarray) -> np.ndarray:
    """左右拼接：marked（confetti 首帧） | original（counterfactual 首帧）。"""
    return cv2.hconcat(
        [_label(marked, "marked (confetti)"), _label(original, "original (counterfactual)")]
    )


def save_comparison(top: List[np.ndarray], bottom: List[np.ndarray], path: str, fps: float) -> None:
    """上下拼接对比视频：输入 marked 视频 | 生成+追踪 overlay。"""
    n = min(len(top), len(bottom))
    rows = [
        cv2.vconcat([_label(top[t], "input (marked)"), _label(bottom[t], "generated + tracks")])
        for t in range(n)
    ]
    save_video(rows, path, fps)


def pick_model_size(orig_w: int, orig_h: int, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if width and height:
        if width % SIZE_ALIGN or height % SIZE_ALIGN:
            raise ValueError(f"--width/--height 必须是 {SIZE_ALIGN} 的倍数（VAE /16 × patch /2）")
        return width, height
    return T4_PORTRAIT_SIZE if orig_h > orig_w else T4_LANDSCAPE_SIZE


def trim_to_4k1(frames: List[np.ndarray]) -> List[np.ndarray]:
    if (len(frames) - 1) % 4 != 0:
        good = max(5, ((len(frames) - 1) // 4) * 4 + 1)
        print(f"  帧数 {len(frames)} 不满足 T=4k+1，自动裁剪到 {good} 帧")
        frames = frames[:good]
    return frames


# --------------------------------------------------------------------------- #
#  主流程                                                                       #
# --------------------------------------------------------------------------- #


def self_test_decoder(frame_marked: np.ndarray, grid: List[GridPoint], args) -> dict:
    """对插好点的首帧跑一遍检测+匹配，闭环验证解码器（不依赖模型）。"""
    dets = detect_confetti(frame_marked, args.sat_thresh, args.val_lo)
    assigned, decoder = match_frame0(dets, grid, max_dist=args.spacing * 0.5)
    errs = [
        math.hypot(d.x - grid[i].x, d.y - grid[i].y) for i, d in assigned.items()
    ]
    code_ok = sum(
        1 for i, d in assigned.items() if decoder.classify(d) == grid[i].codeword
    )
    report = {
        "grid_points": len(grid),
        "detections": len(dets),
        "matched": len(assigned),
        "match_ratio": len(assigned) / len(grid) if grid else 0.0,
        "mean_position_error_px": float(np.mean(errs)) if errs else None,
        "codeword_accuracy": code_ok / len(assigned) if assigned else 0.0,
        "calibrated_hue_centers": [round(c, 1) for c in decoder.hue_centers],
        "calibrated_val_threshold": round(decoder.val_threshold, 1),
    }
    mean_err = report["mean_position_error_px"]
    print(
        f"[self-test] 首帧解码闭环：匹配 {report['matched']}/{report['grid_points']} "
        f"(比例 {report['match_ratio']:.2%}) "
        f"位置误差 {mean_err if mean_err is not None else float('nan'):.2f}px "
        f"码字准确率 {report['codeword_accuracy']:.2%}"
    )
    return report


def run_diffusion(args, frames_edited, frame0_marked, frame0_original, fps, debug_dir):
    """懒加载 demo_moe 管线：marked=铺点首帧，original=无点首帧，引擎不变。"""
    import torch  # noqa: F401
    from demo_moe import (
        cuda_preflight_error,
        encode_prompt_official,
        load_wan_ti2v_pipe,
        print_scheduler_info,
        release_memory,
        resolve_max_sequence_length,
        run_counterfactual_sdedit,
    )

    device_error = cuda_preflight_error(args.device)
    if device_error is not None:
        raise RuntimeError(device_error)

    print(f"加载模型：{args.model_id}")
    pipe = load_wan_ti2v_pipe(
        args.model_id,
        device=args.device,
        vae_device=args.vae_device,
        vae_dtype=args.vae_dtype,
        low_cpu_memory=args.low_memory,
        vae_tiling=args.vae_tiling,
    )
    sched_info = print_scheduler_info(pipe)
    max_sequence_length = resolve_max_sequence_length(pipe, None)
    prompt_embeds = encode_prompt_official(pipe, args.prompt, max_sequence_length)
    if args.low_memory and getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    release_memory()

    generated, stage = run_counterfactual_sdedit(
        pipe,
        frames_edited,
        frame0_marked,
        frame0_original,
        gamma=args.gamma,
        lam=args.lam,
        scheduler_steps=args.scheduler_steps,
        prompt_embeds=prompt_embeds,
        seed=args.seed,
        decode_noisy=args.decode_noisy,
        debug_dir=debug_dir,
        fps=fps,
    )
    release_memory()
    return generated, stage, sched_info


def run_demo(args) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载视频：{args.video}")
    frames, fps = load_video(args.video, args.max_frames, args.frame_stride)
    if not frames:
        raise RuntimeError("无法从视频中读取任何帧")
    orig_w, orig_h = frames[0].shape[1], frames[0].shape[0]
    print(
        f"  {len(frames)} 帧  分辨率 {orig_w}x{orig_h}  "
        f"等效fps={fps:.2f}（stride={args.frame_stride}）"
    )
    frames = trim_to_4k1(frames)

    model_w, model_h = pick_model_size(orig_w, orig_h, args.width, args.height)
    frames = resize_video(frames, model_w, model_h)
    print(f"  模型工作尺寸 {model_w}x{model_h}")

    print(f"低饱和化：S *= {args.desaturation}")
    frames_desat = desaturate_video(frames, args.desaturation)

    grid = build_confetti_grid(
        model_w, model_h, args.spacing, args.jitter, args.hue_coding, args.seed
    )
    n_cols = len({p.gx for p in grid})
    n_rows = len({p.gy for p in grid})
    print(
        f"Confetti 网格：{n_cols}x{n_rows}={len(grid)} 点  间距 {args.spacing}px  "
        f"码字 {len(HUE_CLASSES)}色相x{len(VALUE_LEVELS)}明度  编码 {args.hue_coding}"
    )
    same_code_gap = len(VALUE_LEVELS) * args.spacing
    print(
        f"  同码字最小间隔 ≈ {same_code_gap}px（垂直 2 行）→ "
        f"单帧位移 < {same_code_gap // 2}px 时身份无歧义"
    )

    frame0_original = frames_desat[0]
    frame0_marked = insert_confetti(frame0_original, grid, args.marker_radius)
    frames_edited = [frame0_marked] + frames_desat[1:]

    # ---- 中间结果：输入侧 ----
    inputs_dir = os.path.join(args.output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    cv2.imwrite(os.path.join(inputs_dir, "frame0_confetti.png"), frame0_marked)
    cv2.imwrite(os.path.join(inputs_dir, "frame0_original.png"), frame0_original)
    cv2.imwrite(
        os.path.join(inputs_dir, "frame0_pair.png"),
        make_pair_image(frame0_marked, frame0_original),
    )
    print(f"[save] {inputs_dir}/frame0_confetti.png / frame0_original.png / frame0_pair.png")
    save_video(frames_desat, os.path.join(inputs_dir, "input_desat.mp4"), fps)
    save_video(frames_edited, os.path.join(inputs_dir, "input_marked.mp4"), fps)
    if args.save_frames:
        save_frames(frames_edited, os.path.join(inputs_dir, "input_marked_frames"))

    self_test = self_test_decoder(frame0_marked, grid, args)
    if self_test["match_ratio"] < 0.95:
        print(
            "[warn] 首帧自检匹配率 < 95%，先检查 --sat-thresh/--desaturation/--spacing "
            "再上 GPU"
        )

    summary = {
        "video": args.video,
        "model_size": f"{model_w}x{model_h}",
        "frames": len(frames_edited),
        "fps": fps,
        "frame_stride": args.frame_stride,
        "spacing": args.spacing,
        "jitter": args.jitter,
        "marker_radius": args.marker_radius,
        "desaturation": args.desaturation,
        "hue_coding": args.hue_coding,
        "grid_points": len(grid),
        "grid_cols": n_cols,
        "grid_rows": n_rows,
        "gamma": args.gamma,
        "lam": args.lam,
        "scheduler_steps": args.scheduler_steps,
        "seed": args.seed,
        "frame0_self_test": self_test,
    }

    if args.preview_only:
        summary["mode"] = "preview_only"
        with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print("[preview-only] 跳过扩散生成。检查 inputs/frame0_pair.png 与自检指标。")
        return summary

    generated, stage, sched_info = run_diffusion(
        args,
        frames_edited,
        frame0_marked,
        frame0_original,
        fps,
        debug_dir=os.path.join(args.output_dir, "stage1"),
    )
    save_video(generated, os.path.join(args.output_dir, "generated.mp4"), fps)
    if args.save_frames:
        save_frames(generated, os.path.join(args.output_dir, "generated_frames"))

    sync_rows = compute_sync_metrics(generated, frames_edited)
    print_sync_metrics(sync_rows)

    search_radius = args.search_radius or float(args.spacing)
    tracks, visible, frame_stats = track_confetti_sequence(
        generated,
        grid,
        search_radius=search_radius,
        sat_thresh=args.sat_thresh,
        val_lo=args.val_lo,
    )

    # ---- 中间结果：追踪可视化 ----
    overlay = draw_confetti_tracks(generated, grid, tracks, visible)
    save_video(overlay, os.path.join(args.output_dir, "generated_overlay.mp4"), fps)
    trails = draw_confetti_trails(generated, grid, tracks, visible)
    save_video(trails, os.path.join(args.output_dir, "tracks_trails.mp4"), fps)
    save_comparison(
        frames_edited, trails, os.path.join(args.output_dir, "compare.mp4"), fps
    )
    if args.save_frames:
        save_frames(overlay, os.path.join(args.output_dir, "overlay_frames"))

    per_frame_ratio = visible.mean(axis=0)
    print("\n逐帧解码率（匹配点 / 网格点）：")
    for st in frame_stats:
        t = st["frame"]
        extra = ""
        if "rejected_by_smoothness" in st:
            extra += f"  平滑剔除 {st['rejected_by_smoothness']}"
        if "val_level_means" in st:
            extra += f"  明度档 {st['val_level_means']}"
        print(
            f"  t={t:02d} 检出 {st['detections']:4d}  匹配 {st['matched']:4d}"
            f"  比例 {per_frame_ratio[t]:.2%}{extra}"
        )
    survival = visible[:, -1].mean() if visible.shape[1] else 0.0
    print(f"\n末帧存活率：{survival:.2%}  全程平均可见率：{visible.mean():.2%}")

    npz_path = os.path.join(args.output_dir, "tracks.npz")
    np.savez(
        npz_path,
        tracks=tracks,
        visible=visible,
        grid_xy=np.array([[p.x, p.y] for p in grid]),
        codewords=np.array([p.codeword for p in grid]),
        model_size=np.array([model_w, model_h]),
        orig_size=np.array([orig_w, orig_h]),
    )
    print(f"[save] {npz_path}")

    summary.update(
        {
            "mode": "full",
            "artifacts": {
                "frame0_confetti": "inputs/frame0_confetti.png",
                "frame0_original": "inputs/frame0_original.png",
                "frame0_pair": "inputs/frame0_pair.png",
                "input_desat": "inputs/input_desat.mp4",
                "input_marked": "inputs/input_marked.mp4",
                "generated": "generated.mp4",
                "generated_overlay": "generated_overlay.mp4",
                "tracks_trails": "tracks_trails.mp4",
                "compare": "compare.mp4",
                "tracks": "tracks.npz",
            },
            "stage1": stage,
            "sync_metrics": sync_rows,
            "frame_stats": frame_stats,
            "per_frame_visible_ratio": [float(x) for x in per_frame_ratio],
            "last_frame_survival": float(survival),
            "mean_visible_ratio": float(visible.mean()),
            **sched_info,
        }
    )
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[result] summary written to {os.path.join(args.output_dir, 'summary.json')}")
    print(
        "\n[实验0判读] 打开 generated_overlay.mp4：彩点跟随物体平流=方案成立；"
        "静止贴屏/被抹掉/晕染成块=约束失效，降低 --spacing 密度或调 --lam 重试。"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confetti 网格稠密点跟踪 demo（Wan2.2-TI2V-5B 反事实 SDEdit）",
        epilog="密度消融建议：--spacing 64 / 48 / 32 各跑一次，对比点阵平流质量。",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="outputs/demo_moe_confetti")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-device", default="auto")
    parser.add_argument(
        "--vae-dtype",
        default="auto",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="VAE 空间分块，显著降低 fp32 decode 峰值显存（T4 + float32 时建议开启）",
    )
    # confetti 设计参数
    parser.add_argument("--spacing", type=int, default=48, help="点间距（px），密度消融的主旋钮")
    parser.add_argument("--jitter", type=int, default=3, help="网格随机抖动幅度（px）")
    parser.add_argument("--marker-radius", type=int, default=4)
    parser.add_argument(
        "--desaturation", type=float, default=0.25, help="背景饱和度保留比例（0=纯灰度）"
    )
    parser.add_argument(
        "--hue-coding",
        default="diagonal",
        choices=["diagonal", "periodic", "debruijn"],
        help=(
            "diagonal（默认）上下相邻行色相相差 90°，不依赖明度档稳定性；"
            "periodic 同列同色相；debruijn 预留窗口重定位能力"
        ),
    )
    # 解码参数
    parser.add_argument("--sat-thresh", type=int, default=140, help="检测饱和度阈值（0-255）")
    parser.add_argument("--val-lo", type=int, default=70, help="检测最低明度（0-255）")
    parser.add_argument(
        "--search-radius", type=float, default=None, help="逐帧搜索半径（px），默认=spacing"
    )
    # 扩散参数（与 demo_moe 对齐）
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--lam", type=float, default=8.0, help="反事实引导强度；多点联合引导建议消融 2-8"
    )
    parser.add_argument(
        "--scheduler-steps", type=int, default=50, help="UniPC 步数；30-50 通常够跟踪用"
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=9)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="每 N 帧取一帧。60fps 输入建议 2-3，让运动幅度匹配模型先验",
    )
    parser.add_argument("--width", type=int, default=None, help="自定义宽（须 32 对齐）")
    parser.add_argument("--height", type=int, default=None, help="自定义高（须 32 对齐）")
    parser.add_argument("--decode-noisy", action="store_true")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="额外按帧导出 PNG（inputs/generated/overlay 各一个目录）",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="只做插点 + 解码器闭环自检，不加载模型（无 GPU 可跑）",
    )
    parser.add_argument(
        "--no-low-memory", dest="low_memory", action="store_false",
        help="Disable reduced-CPU-RAM model loading and T5 release.",
    )
    parser.set_defaults(low_memory=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_demo(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

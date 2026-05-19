"""
TAP-Vid 基准评测模块：在 DAVIS 或 Kinetics 数据集上计算 AJ、δ_avg^x 和 OA 三项指标。

评测协议参考 TAP-Vid 论文（Doersch et al., 2022）：
  - AJ  (Average Jaccard)：综合考虑位置精度和遮挡预测的调和平均。
  - δ_avg^x：五个像素阈值（1/2/4/8/16 px）下位置精度的平均值。
  - OA  (Occlusion Accuracy)：遮挡状态预测的帧级准确率。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
#  指标计算                                                                      #
# --------------------------------------------------------------------------- #

# 五个评测阈值（像素），需除以图像对角线归一化
THRESHOLDS = np.array([1, 2, 4, 8, 16], dtype=np.float32)


def _compute_tapvid_metrics(
    tracks_pred: np.ndarray,   # (N, T, 2) 预测轨迹坐标，单位：像素
    visible_pred: np.ndarray,  # (N, T) bool，True 表示预测点可见
    tracks_gt: np.ndarray,     # (N, T, 2) 真实轨迹坐标，单位：像素
    visible_gt: np.ndarray,    # (N, T) bool，True 表示真实点可见
    H: int,
    W: int,
) -> Dict[str, float]:
    """对单个视频计算 AJ、δ_avg^x 和 OA。

    坐标单位为像素；阈值按 max(H, W) 归一化，与 TAP-Vid 论文一致。
    """
    N, T, _ = tracks_gt.shape
    diag = max(H, W)  # 用最长边作为归一化基准

    # ---- 遮挡准确率 (OA) ---------------------------------------------------- #
    # 统计预测可见性与真实可见性完全匹配的帧占比
    oa = (visible_pred == visible_gt).mean()

    # ---- 位置精度 (δ_avg^x) ------------------------------------------------- #
    # 仅在真实标注为可见的帧上计算位置误差
    vis_mask = visible_gt.astype(bool)     # (N, T)
    if vis_mask.sum() == 0:
        pos_acc = 0.0
    else:
        # 计算每帧的欧氏距离误差并归一化
        err = np.linalg.norm(tracks_pred - tracks_gt, axis=-1)  # (N, T)
        err_norm = err / diag                                     # 归一化到 [0, 1]

        accs = []
        for thr in THRESHOLDS / 256.0:   # 阈值转为图像尺寸的分数
            correct = (err_norm < thr) & vis_mask
            accs.append(correct.sum() / vis_mask.sum())
        pos_acc = float(np.mean(accs))

    # ---- 平均 Jaccard (AJ) -------------------------------------------------- #
    # Jaccard = TP / (TP + FP + FN)，在每个阈值下计算后取平均
    if vis_mask.sum() == 0:
        aj = 0.0
    else:
        jaccards = []
        for thr in THRESHOLDS / 256.0:
            err = np.linalg.norm(tracks_pred - tracks_gt, axis=-1) / diag
            # 真正例：位置正确且预测/真实均可见
            true_pos  = ((err < thr) & vis_mask & visible_pred).sum()
            # 假正例：预测可见但位置错误或真实遮挡
            false_pos = ((err >= thr) | ~vis_mask) & visible_pred
            # 假负例：真实可见但预测遮挡
            false_neg = vis_mask & ~visible_pred
            denom = true_pos + false_pos.sum() + false_neg.sum()
            jaccards.append(true_pos / denom if denom > 0 else 0.0)
        aj = float(np.mean(jaccards))

    return {"AJ": aj, "delta_avg": pos_acc, "OA": oa}


# --------------------------------------------------------------------------- #
#  数据集加载                                                                    #
# --------------------------------------------------------------------------- #

def load_tapvid_davis(data_path: str) -> List[Dict]:
    """加载 TAP-Vid DAVIS pickle 格式数据集。

    返回列表，每个元素为包含以下字段的字典：
        'video'   : (T, H, W, 3) uint8 RGB 视频帧
        'points'  : (N, T, 2)    float，坐标归一化到 [0, 1]
        'occluded': (N, T)       bool，True 表示该帧被遮挡
    """
    import pickle
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    samples = []
    for item in data:
        samples.append({
            "video":    item["video"],
            "points":   item["points"],
            "occluded": item["occluded"],
        })
    return samples


# --------------------------------------------------------------------------- #
#  主评测循环                                                                    #
# --------------------------------------------------------------------------- #

def evaluate(
    tracker,
    samples: List[Dict],
    max_videos: int = -1,
) -> Dict[str, float]:
    """在 TAP-Vid 样本上评测 PointPrompter 跟踪器。

    Args:
        tracker:    PointPrompter 实例（或任何实现 track_multiple 的跟踪器）
        samples:    load_tapvid_davis 返回的样本列表
        max_videos: 限制评测视频数量，-1 表示评测全部

    Returns:
        所有视频上 AJ、delta_avg、OA 的均值字典
    """
    import cv2
    all_metrics: List[Dict[str, float]] = []

    for i, sample in enumerate(samples):
        if max_videos > 0 and i >= max_videos:
            break

        video_rgb = sample["video"]           # (T, H, W, 3) uint8 RGB
        pts_norm  = sample["points"]          # (N, T, 2) 归一化坐标 ∈ [0, 1]
        occluded  = sample["occluded"]        # (N, T) bool — True 表示遮挡

        T, H, W, _ = video_rgb.shape
        N = pts_norm.shape[0]

        # RGB → BGR 列表，与跟踪器输入格式一致
        frames_bgr = [video_rgb[t][..., ::-1].copy() for t in range(T)]

        # 将归一化坐标还原为像素坐标
        pts_px = pts_norm.copy()
        pts_px[..., 0] *= (W - 1)
        pts_px[..., 1] *= (H - 1)
        visible_gt = ~occluded  # 可见性 = 非遮挡

        # 取每个点首次可见帧的坐标作为查询点（模拟论文评测设置）
        query_list = []
        for n in range(N):
            first_vis = np.argmax(visible_gt[n])
            query_list.append((float(pts_px[n, first_vis, 0]), float(pts_px[n, first_vis, 1])))

        # 批量跟踪所有查询点
        results = tracker.track_multiple(frames_bgr, query_list)

        tracks_pred  = np.stack([r.tracks  for r in results], axis=0)   # (N, T, 2)
        visible_pred = np.stack([r.visible for r in results], axis=0)   # (N, T)

        m = _compute_tapvid_metrics(tracks_pred, visible_pred, pts_px, visible_gt, H, W)
        all_metrics.append(m)
        print(f"[{i+1}/{len(samples)}] AJ={m['AJ']:.3f}  δ={m['delta_avg']:.3f}  OA={m['OA']:.3f}")

    # 汇总所有视频的均值
    mean_aj    = float(np.mean([m["AJ"]        for m in all_metrics]))
    mean_delta = float(np.mean([m["delta_avg"] for m in all_metrics]))
    mean_oa    = float(np.mean([m["OA"]        for m in all_metrics]))

    print(f"\n=== 评测结果 ===")
    print(f"AJ:      {mean_aj:.3f}")
    print(f"δ_avg^x: {mean_delta:.3f}")
    print(f"OA:      {mean_oa:.3f}")
    return {"AJ": mean_aj, "delta_avg": mean_delta, "OA": mean_oa}

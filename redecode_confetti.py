"""
对已有 confetti run 的 generated.mp4 做离线重解码（解码器 v2）+ 悬崖诊断。

不需要 GPU、不重跑扩散：从 run 目录的 summary.json 读取网格参数，
精确重建当初的 confetti 网格，然后：

  1. 诊断：逐帧统计检出彩点的明度分布——用"生成首帧标定的固定阈值"和
     "逐帧 2-means 自适应"分别给出亮档比例。如果固定阈值下亮档比例在某帧
     突然塌缩而自适应保持 ~50%，即证实"明度档集体翻转"是匹配率悬崖的元凶。
  2. 重解码：用解码器 v2（逐帧明度再标定 + 软匹配 + 丢失点邻居平流）
     重新生成轨迹，输出 v2 的逐帧匹配表和可视化。

用法：
  python redecode_confetti.py --run-dir outputs/demo_moe_confetti
产物（写入 run 目录）：
  redecode_report.json    诊断 + v2 统计
  tracks_v2.npz           v2 轨迹
  generated_overlay_v2.mp4 / tracks_trails_v2.mp4 / compare_v2.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

from demo_moe_confetti import (
    build_confetti_grid,
    compute_sync_metrics,
    detect_confetti,
    draw_confetti_tracks,
    draw_confetti_trails,
    load_video,
    match_frame0,
    print_sync_metrics,
    save_comparison,
    save_video,
    track_confetti_sequence,
    _two_means,
)


def diagnose_value_levels(frames, grid, args) -> list:
    """逐帧明度分布诊断：固定阈值 vs 自适应 2-means 的亮档比例对比。"""
    dets0 = detect_confetti(frames[0], args.sat_thresh, args.val_lo)
    _assigned0, decoder = match_frame0(dets0, grid, max_dist=args.spacing * 0.5)
    fixed_thr = decoder.val_threshold
    level_means = [255.0, 160.0]
    print(f"[diag] 生成首帧标定的固定明度阈值 = {fixed_thr:.1f}")
    print("[diag] 逐帧明度分布（亮档比例：固定阈值 vs 自适应；自适应档位中心）：")
    rows = []
    for t, frame in enumerate(frames):
        dets = detect_confetti(frame, args.sat_thresh, args.val_lo)
        vals = np.array([d.val for d in dets]) if dets else np.array([])
        if len(vals) >= 4:
            level_means = _two_means(list(vals), level_means)
        bright_fixed = float((vals >= fixed_thr).mean()) if len(vals) else 0.0
        mid = (level_means[0] + level_means[1]) / 2.0
        bright_adapt = float((vals >= mid).mean()) if len(vals) else 0.0
        row = {
            "frame": t,
            "detections": int(len(vals)),
            "bright_ratio_fixed": round(bright_fixed, 3),
            "bright_ratio_adaptive": round(bright_adapt, 3),
            "val_mean": round(float(vals.mean()), 1) if len(vals) else None,
            "level_means_adaptive": [round(c, 1) for c in level_means],
        }
        rows.append(row)
        print(
            f"  t={t:02d} 检出 {row['detections']:4d}  "
            f"亮档(固定) {bright_fixed:6.1%}  亮档(自适应) {bright_adapt:6.1%}  "
            f"V均值 {row['val_mean']}  档位 {row['level_means_adaptive']}"
        )
    # 悬崖判定：固定阈值亮档比例的最大单帧跌幅
    fixed_series = [r["bright_ratio_fixed"] for r in rows]
    drops = [fixed_series[t - 1] - fixed_series[t] for t in range(1, len(fixed_series))]
    if drops:
        worst = int(np.argmax(drops)) + 1
        print(
            f"[diag] 固定阈值亮档比例最大单帧跌幅：t={worst} 跌 {max(drops):.1%}"
            f"（自适应同帧 {rows[worst]['bright_ratio_adaptive']:.1%}）"
        )
        if max(drops) > 0.15 and rows[worst]["bright_ratio_adaptive"] > 0.3:
            print("[diag] => 证实：明度整体衰减导致固定阈值档位翻转；自适应标定可恢复")
        else:
            print("[diag] => 明度翻转不明显，悬崖另有原因（看 chunk 边界 / 位移超窗）")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Confetti run 离线重解码（解码器 v2）+ 诊断")
    parser.add_argument("--run-dir", required=True, help="包含 generated.mp4 与 summary.json 的目录")
    parser.add_argument("--sat-thresh", type=int, default=140)
    parser.add_argument("--val-lo", type=int, default=70)
    parser.add_argument("--search-radius", type=float, default=None, help="默认=spacing")
    parser.add_argument("--diag-only", action="store_true", help="只做明度诊断，不重解码")
    args = parser.parse_args()

    summary_path = os.path.join(args.run_dir, "summary.json")
    video_path = os.path.join(args.run_dir, "generated.mp4")
    for path in (summary_path, video_path):
        if not os.path.exists(path):
            sys.exit(f"错误：缺少 {path}")
    with open(summary_path) as handle:
        run = json.load(handle)

    model_w, model_h = (int(v) for v in run["model_size"].split("x"))
    args.spacing = int(run["spacing"])
    grid = build_confetti_grid(
        model_w,
        model_h,
        args.spacing,
        int(run["jitter"]),
        run.get("hue_coding", "periodic"),
        int(run["seed"]),
    )
    print(
        f"[run] {args.run_dir}: {run['model_size']} spacing={args.spacing} "
        f"hue_coding={run.get('hue_coding')} grid={len(grid)}点 "
        f"(summary 记录 {run['grid_points']}点)"
    )
    if len(grid) != int(run["grid_points"]):
        sys.exit("错误：重建网格点数与 summary.json 不符，检查参数/版本")

    frames, _fps = load_video(video_path)
    fps = float(run.get("fps", 8.0))
    print(f"[run] generated.mp4: {len(frames)} 帧")

    diag_rows = diagnose_value_levels(frames, grid, args)
    report = {"run_dir": args.run_dir, "value_level_diagnosis": diag_rows}

    # 同步性诊断：生成 vs 实际送入扩散的输入（量化内容漂移，定位拐点）
    marked_path = os.path.join(args.run_dir, "inputs", "input_marked.mp4")
    marked_frames = None
    if os.path.exists(marked_path):
        marked_frames, _ = load_video(marked_path)
        sync_rows = compute_sync_metrics(frames, marked_frames)
        print_sync_metrics(sync_rows)
        report["sync_metrics"] = sync_rows
    else:
        print(f"[diag] 未找到 {marked_path}，跳过同步性诊断")

    if not args.diag_only:
        search_radius = args.search_radius or float(args.spacing)
        tracks, visible, frame_stats = track_confetti_sequence(
            frames,
            grid,
            search_radius=search_radius,
            sat_thresh=args.sat_thresh,
            val_lo=args.val_lo,
        )
        per_frame_ratio = visible.mean(axis=0)
        print("\n[v2] 逐帧解码率（匹配点 / 网格点）：")
        for st in frame_stats:
            t = st["frame"]
            extra = ""
            if "rejected_by_smoothness" in st:
                extra += f"  平滑剔除 {st['rejected_by_smoothness']}"
            print(
                f"  t={t:02d} 检出 {st['detections']:4d}  匹配 {st['matched']:4d}"
                f"  比例 {per_frame_ratio[t]:.2%}{extra}"
            )
        survival = visible[:, -1].mean() if visible.shape[1] else 0.0
        print(f"\n[v2] 末帧存活率：{survival:.2%}  全程平均可见率：{visible.mean():.2%}")

        overlay = draw_confetti_tracks(frames, grid, tracks, visible)
        save_video(overlay, os.path.join(args.run_dir, "generated_overlay_v2.mp4"), fps)
        trails = draw_confetti_trails(frames, grid, tracks, visible)
        save_video(trails, os.path.join(args.run_dir, "tracks_trails_v2.mp4"), fps)
        if marked_frames is not None:
            save_comparison(
                marked_frames, trails, os.path.join(args.run_dir, "compare_v2.mp4"), fps
            )
        np.savez(
            os.path.join(args.run_dir, "tracks_v2.npz"),
            tracks=tracks,
            visible=visible,
            grid_xy=np.array([[p.x, p.y] for p in grid]),
            codewords=np.array([p.codeword for p in grid]),
            model_size=np.array([model_w, model_h]),
        )
        print(f"[save] {os.path.join(args.run_dir, 'tracks_v2.npz')}")
        report["v2_frame_stats"] = frame_stats
        report["v2_last_frame_survival"] = float(survival)
        report["v2_mean_visible_ratio"] = float(visible.mean())

    report_path = os.path.join(args.run_dir, "redecode_report.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[result] {report_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)

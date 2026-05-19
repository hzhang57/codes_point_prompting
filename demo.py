"""
演示脚本：使用 Point Prompting 跟踪视频中的用户指定点，并将轨迹可视化写入输出视频。

支持 CogVideoX-I2V 和 Wan2.1-I2V 两种模型。

使用示例：
    # Wan 2.1 14B（论文主要模型，质量最佳，需约 40 GB 显存）
    python demo.py --video input.mp4 --points "320,240" "640,360" \\
                   --model-type wan --model-id Wan-AI/Wan2.1-I2V-14B-480P

    # Wan 2.1 1.3B（速度更快，约 10 GB 显存）
    python demo.py --video input.mp4 --points "320,240" \\
                   --model-type wan --model-id Wan-AI/Wan2.1-I2V-1.3B-480P

    # CogVideoX 5B I2V
    python demo.py --video input.mp4 --points "320,240" \\
                   --model-type cogvideox --model-id THUDM/CogVideoX-5b-I2V

依赖安装：
    pip install torch diffusers transformers accelerate opencv-python pillow
"""

import argparse
import sys
import re
import cv2
import numpy as np
import torch

from model_adapter import load_cogvideox_pipe, load_wan_pipe, create_adapter
from tracker import PointPrompter, PointPrompterConfig


# --------------------------------------------------------------------------- #
#  各模型类型的默认配置                                                          #
# --------------------------------------------------------------------------- #

MODEL_DEFAULTS = {
    "wan": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P",   # 默认使用 14B 版本
        "dtype": "bfloat16",
    },
    "cogvideox": {
        "model_id": "THUDM/CogVideoX-5b-I2V",
        "dtype": "float16",
    },
}


# --------------------------------------------------------------------------- #
#  轨迹可视化                                                                    #
# --------------------------------------------------------------------------- #

# 用于区分不同查询点的颜色调色板（BGR 格式）
_PALETTE = [
    (0, 255, 0),   (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255),
]


def draw_tracks(frames: list, tracks_list: list, visible_list: list) -> list:
    """在每帧上绘制所有点的轨迹：可见帧画圆点，相邻可见帧之间画连线。

    Args:
        frames:       原始视频帧列表，每帧 (H, W, 3) BGR
        tracks_list:  每个点的轨迹 (T, 2) 列表
        visible_list: 每个点的可见性 (T,) bool 列表

    Returns:
        带轨迹标注的帧列表（深拷贝，不修改原帧）
    """
    T   = len(frames)
    out = [f.copy() for f in frames]
    for i, (track, vis) in enumerate(zip(tracks_list, visible_list)):
        color = _PALETTE[i % len(_PALETTE)]  # 每个点使用不同颜色
        for t in range(T):
            if vis[t]:
                cx, cy = int(round(track[t, 0])), int(round(track[t, 1]))
                # 在当前帧画实心圆点标记位置
                cv2.circle(out[t], (cx, cy), 5, color, -1)
                # 向前找最近的可见帧，画连线表示运动轨迹
                prev = t - 1
                while prev >= 0 and not vis[prev]:
                    prev -= 1
                if prev >= 0:
                    px, py = int(round(track[prev, 0])), int(round(track[prev, 1]))
                    cv2.line(out[t], (px, py), (cx, cy), color, 2)
    return out


# --------------------------------------------------------------------------- #
#  视频 I/O 工具                                                                 #
# --------------------------------------------------------------------------- #

def load_video(path: str, max_frames: int = 50) -> list:
    """从文件读取视频帧（BGR），最多读取 max_frames 帧。"""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def save_video(frames: list, path: str, fps: float = 15.0):
    """将帧列表写入 MP4 视频文件。"""
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()


def parse_point(s: str):
    """解析 'x,y' 格式的命令行查询点字符串，返回 (float, float)。"""
    parts = s.strip().split(",")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(f"Expected 'x,y', got '{s}'")
    return float(parts[0].strip()), float(parts[1].strip())


def parse_points(values: list) -> list:
    """解析命令行查询点，兼容空白参数以及 'x,y x,y' / 'x,y;x,y' 写法。"""
    tokens = []
    for value in values:
        tokens.extend(t for t in re.split(r"[;\s]+", value.strip()) if t)
    return [parse_point(token) for token in tokens]


# --------------------------------------------------------------------------- #
#  主函数                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Point Prompting 点跟踪演示")
    parser.add_argument("--video",   required=True, help="输入视频文件路径")
    parser.add_argument("--points",  nargs="+", required=True,
                        help="查询点坐标，格式为 'x,y'（第 0 帧像素坐标），可指定多个点")
    parser.add_argument("--model-type", choices=["wan", "cogvideox"], default="wan",
                        help="使用的模型骨干（默认：wan）")
    parser.add_argument("--model-id",   default=None,
                        help="HuggingFace 模型 ID（覆盖各类型的默认值）")
    parser.add_argument("--output",  default="tracked.mp4",
                        help="输出视频路径（默认：tracked.mp4）")
    parser.add_argument("--gamma",   type=float, default=0.5,
                        help="SDEdit 加噪比例 γ（默认：0.5）")
    parser.add_argument("--lam",     type=float, default=8.0,
                        help="反事实引导权重 λ（默认：8.0）")
    parser.add_argument("--steps",   type=int,   default=50,
                        help="扩散模型去噪总步数（默认：50）")
    parser.add_argument("--no-refine", action="store_true",
                        help="跳过 inpainting 精细化步骤（速度更快但精度略低）")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="随机种子（默认：42）")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="最多处理的帧数（默认：50）")
    parser.add_argument("--device",  default="cuda",
                        help="计算设备（默认：cuda）")
    args = parser.parse_args()

    # 确定模型 ID（命令行参数优先，否则使用该类型的默认值）
    model_type = args.model_type
    model_id   = args.model_id or MODEL_DEFAULTS[model_type]["model_id"]

    try:
        query_points = parse_points(args.points)
    except ValueError as exc:
        parser.error(str(exc))
    if not query_points:
        parser.error("--points 至少需要一个有效坐标，例如 --points \"320,240\"")
    print(f"查询点：{query_points}")

    # 读取输入视频
    print(f"加载视频：{args.video}")
    frames = load_video(args.video, args.max_frames)
    if not frames:
        sys.exit("错误：无法从视频中读取任何帧。")
    print(f"  {len(frames)} 帧  分辨率 {frames[0].shape[1]}×{frames[0].shape[0]}")

    # 加载视频扩散模型
    print(f"加载 {model_type} 模型：{model_id}")
    if model_type == "wan":
        pipe = load_wan_pipe(model_id, args.device)
    else:
        pipe = load_cogvideox_pipe(model_id, args.device)

    adapter = create_adapter(pipe)
    print(f"  已使用适配器：{type(adapter).__name__}")

    # 构建跟踪器（传入超参数配置）
    cfg = PointPrompterConfig(
        gamma=args.gamma,
        lam=args.lam,
        num_inference_steps=args.steps,
        do_refine=not args.no_refine,
        seed=args.seed,
    )
    tracker = PointPrompter(adapter, cfg)

    # 执行跟踪（每个查询点独立运行完整流水线）
    print("正在跟踪…")
    results = tracker.track_multiple(frames, query_points)

    # 可视化轨迹并保存输出视频
    annotated = draw_tracks(frames, [r.tracks for r in results], [r.visible for r in results])
    save_video(annotated, args.output)
    print(f"已保存至 {args.output}")

    # 打印每个点的可见帧比例统计
    for i, r in enumerate(results):
        print(f"  点 {i} {query_points[i]}：可见帧占比 {r.visible.mean()*100:.0f}%")


if __name__ == "__main__":
    main()

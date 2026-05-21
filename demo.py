"""
演示脚本：使用 Point Prompting 跟踪视频中的用户指定点，并将轨迹可视化写入输出视频。

支持 CogVideoX-I2V 和 Wan2.1-I2V / VACE 两种模型。

使用示例：
    # Wan 2.1 VACE 1.3B（约 10 GB 显存）
    python demo.py --video input.mp4 --points "1157,635" \\
                   --model-type wan --model-id Wan-AI/Wan2.1-VACE-1.3B-diffusers

    # Wan 2.1 14B（论文主要模型，质量最佳，需约 40 GB 显存）
    python demo.py --video input.mp4 --points "320,240" "640,360" \\
                   --model-type wan --model-id Wan-AI/Wan2.1-I2V-14B-480P

    # CogVideoX 5B I2V
    python demo.py --video input.mp4 --points "320,240" \\
                   --model-type cogvideox --model-id THUDM/CogVideoX-5b-I2V

依赖安装：
    pip install torch diffusers transformers accelerate opencv-python pillow
"""

import argparse
import sys
import re
from typing import Optional
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
    (0, 0, 255),   (255, 128, 0), (0, 128, 255), (255, 0, 255),
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

def load_video(path: str, max_frames: int = 50):
    """从文件读取视频帧（BGR），最多读取 max_frames 帧。

    返回 (frames, fps)：帧列表和原始视频帧率。
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def resize_video(frames: list, width: int, height: int) -> list:
    """将视频帧统一缩放到固定分辨率。"""
    if width <= 0 or height <= 0:
        return frames
    if not frames or (frames[0].shape[1] == width and frames[0].shape[0] == height):
        return frames
    return [cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA) for f in frames]


def scale_points(points: list, src_size: tuple, dst_size: tuple) -> list:
    """将点坐标从源分辨率映射到目标分辨率。"""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / src_w
    sy = dst_h / src_h
    return [(x * sx, y * sy) for x, y in points]


def cuda_preflight_error(device: str) -> Optional[str]:
    """检查 CUDA 是否真的可执行 kernel；不可用时返回面向用户的错误信息。"""
    if not str(device).startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        return "错误：指定了 --device cuda，但当前环境中 torch.cuda.is_available() 为 False。"

    try:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        x = torch.ones(1, device=device)
        _ = x + 1
        torch.cuda.synchronize(index)
    except Exception as exc:
        return (
            "错误：当前 PyTorch/CUDA 版本不能在这张 GPU 上执行 CUDA kernel。\n"
            f"  设备：{device}\n"
            f"  异常：{type(exc).__name__}: {exc}\n"
            "这通常是 torch wheel 不包含该 GPU 的计算架构导致的，例如 Kaggle 分配到较老 GPU，"
            "但环境安装了较新的 CUDA/PyTorch wheel。\n"
            "处理方式：在 Kaggle 切换到 T4/L4/A100 等较新的 GPU，或安装与当前 GPU 架构匹配的 PyTorch CUDA 版本；"
            "否则只能用 --device cpu 运行。"
        )

    print(f"CUDA 预检通过：{name} sm_{capability[0]}{capability[1]}")
    return None


def save_video(frames: list, path: str, fps: float):
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

def _diag_generated(results, query_points, radius: int = 30):
    """打印生成帧中查询点附近区域的红色像素诊断信息。"""
    for i, (r, qp) in enumerate(zip(results, query_points)):
        print(f"\n[诊断] 点 {i} {qp}  生成帧数={len(r.generated_frames)}")
        for t, frame in enumerate(r.generated_frames[:5]):  # 只看前 5 帧
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cx, cy = int(round(r.tracks[t, 0])), int(round(r.tracks[t, 1]))
            h_img, w_img = frame.shape[:2]
            x1 = max(0, cx - radius); x2 = min(w_img, cx + radius)
            y1 = max(0, cy - radius); y2 = min(h_img, cy + radius)
            crop_hsv = hsv[y1:y2, x1:x2]
            hue = crop_hsv[:, :, 0].astype(int)
            sat = crop_hsv[:, :, 1].astype(int)
            is_red = ((hue <= 10) | (hue >= 170)) & (sat >= 80)
            is_red_strict = ((hue <= 10) | (hue >= 170)) & (sat >= 100)
            bgr_at = frame[cy, cx] if 0 <= cy < h_img and 0 <= cx < w_img else [0, 0, 0]
            print(f"  t={t:02d}  BGR@center={bgr_at}  "
                  f"red_px(sat≥80)={is_red.sum():4d}  "
                  f"red_px(sat≥100)={is_red_strict.sum():4d}  "
                  f"sat_max={sat.max():3d}  hue_at_center={hue[radius, radius] if radius < hue.shape[0] and radius < hue.shape[1] else -1}")


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
    parser.add_argument("--preprocess-width", type=int, default=832,
                        help="跟踪前先将视频缩放到此宽度，0 表示不预处理缩放（默认：832）")
    parser.add_argument("--preprocess-height", type=int, default=480,
                        help="跟踪前先将视频缩放到此高度，0 表示不预处理缩放（默认：480）")
    parser.add_argument("--model-width", type=int, default=832,
                        help="送入扩散模型的最大宽度，0 表示不缩放（默认：832）")
    parser.add_argument("--model-height", type=int, default=480,
                        help="送入扩散模型的最大高度，0 表示不缩放（默认：480）")
    parser.add_argument("--model-stride", type=int, default=16,
                        help="模型输入宽高对齐倍数（默认：16）")
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
    frames, src_fps = load_video(args.video, args.max_frames)
    if not frames:
        sys.exit("错误：无法从视频中读取任何帧。")
    orig_w, orig_h = frames[0].shape[1], frames[0].shape[0]
    print(f"  {len(frames)} 帧  分辨率 {orig_w}×{orig_h}  fps={src_fps:.2f}")

    # 保留原始帧用于最终可视化（轨迹坐标会被 tracker 反算回原始分辨率）
    frames_orig = frames

    # 预处理视频到固定分辨率，降低 VAE 编码显存占用；查询点同步缩放到预处理坐标系。
    if args.preprocess_width > 0 and args.preprocess_height > 0:
        frames = resize_video(frames, args.preprocess_width, args.preprocess_height)
        query_points = scale_points(
            query_points,
            src_size=(orig_w, orig_h),
            dst_size=(args.preprocess_width, args.preprocess_height),
        )
        print(f"  已预处理到 {args.preprocess_width}×{args.preprocess_height}")
        print(f"  预处理后查询点：{query_points}")

    device_error = cuda_preflight_error(args.device)
    if device_error is not None:
        sys.exit(device_error)

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
        model_width=args.model_width,
        model_height=args.model_height,
        model_stride=args.model_stride,
    )
    tracker = PointPrompter(adapter, cfg)

    # 执行跟踪（每个查询点独立运行完整流水线）
    print("正在跟踪…")
    results = tracker.track_multiple(frames, query_points)

    # 诊断：打印每个点生成帧中查询点附近的 HSV 红色像素信息
    _diag_generated(results, query_points, radius=30)

    # 可视化轨迹并保存输出视频
    # 用原始分辨率帧绘制，tracks 已被 tracker 反算回原始坐标系
    annotated = draw_tracks(frames_orig, [r.tracks for r in results], [r.visible for r in results])
    save_video(annotated, args.output, src_fps)
    print(f"已保存至 {args.output}")

    # 打印每个点的可见帧比例统计
    for i, r in enumerate(results):
        print(f"  点 {i} {query_points[i]}：可见帧占比 {r.visible.mean()*100:.0f}%")


if __name__ == "__main__":
    main()

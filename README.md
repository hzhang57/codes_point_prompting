# 基于 Wan2.2-TI2V-5B 的 Point Prompting

[English README](README_EN.md)

**v3.0** — 2026-06-09

本项目是反事实 Point Prompting 点跟踪算法的非官方实现，目前仅支持
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`。

从 v3.0 开始，仓库删除了其他模型后端和兼容层，只保留一条模型链路、一套
scheduler 约束，以及一种官方首帧条件注入流程。

## 算法流程

对于每个查询点：

1. 抑制输入视频中原本存在的红色区域，减少红点检测干扰。
2. 在第 0 帧查询点位置绘制红色标记。
3. 编码编辑后的视频，并按照 `gamma` 在 latent 空间执行 SDEdit 加噪。
4. 通过官方 `prepare_latents(image, latents=...)` 分别构造两个 TI2V 首帧条件：
   - 正条件：带红点的第 0 帧
   - 负条件：不带红点的原始第 0 帧
5. 在每个去噪步骤应用反事实引导：

   ```python
   v_guided = (lam + 1) * v_marked - lam * v_original
   ```

6. 解码生成视频，并逐帧检测传播后的红点。
7. 可选执行第二次更保守的 TI2V SDEdit refinement。

## 模型约束

唯一支持的 checkpoint：

```text
Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

Pipeline 必须是启用 `expand_timesteps=True` 的
`WanImageToVideoPipeline`。程序启动时会强校验 checkpoint scheduler：

```text
class=UniPCMultistepScheduler
flow_shift=5.0
prediction_type=flow_prediction
use_flow_sigmas=True
timestep_spacing=linspace
```

Wan2.2-TI2V-5B 实际是 dense 5B 模型。脚本名称中的 `moe` 后缀仅为兼容已有
运行命令而保留。

## 安装

```bash
pip install -r requirements.txt
```

需要使用包含 Wan2.2 TI2V/I2V 支持的较新版本 Diffusers。

## Point Prompting 演示

```bash
python demo_moe.py \
  --video input.mp4 \
  --points "900,535" \
  --gamma 0.5 \
  --lam 8.0 \
  --max-frames 9 \
  --output tracked_moe.mp4 \
  --save-generated
```

`--output` 路径相对于当前命令运行目录解析。调试文件和 `summary.json` 默认写入
`outputs/demo_moe`。

支持同时输入多个点：

```bash
python demo_moe.py --video input.mp4 --points "900,535" "1157,635"
```

### 分辨率与双 T4

Wan2.2-TI2V-5B 官方 720P 分辨率：

- 横屏：`1280x704`
- 竖屏：`704x1280`

官方 720P 通常超过单张 15 GiB T4 的显存能力。默认
`--resolution-preset t4` 使用 `832x480` 或 `480x832`。双卡环境下，
transformer 放在 `cuda:0`，VAE 放在 `cuda:1`；`--vae-dtype auto` 在 CUDA
环境自动使用 float16。

显式使用官方分辨率：

```bash
python demo_moe.py --video input.mp4 --points "900,535" --resolution-preset official
```

如果 T4 默认预设仍然 OOM，可以进一步降低分辨率：

```bash
python demo_moe.py \
  --video input.mp4 \
  --points "900,535" \
  --preprocess-width 720 \
  --preprocess-height 416 \
  --model-width 720 \
  --model-height 416 \
  --vae-dtype float16
```

## 去噪重建调试

使用重建脚本扫描不同 SDEdit 噪声强度，同时保持官方 TI2V 首帧条件和
scheduler：

```bash
python debug_denoise_moe.py \
  --video input.mp4 \
  --max-frames 9 \
  --width 832 \
  --height 480 \
  --gammas 0.5
```

输出包含 VAE round-trip 重建视频、去噪视频、timestep 日志、PSNR 指标，以及
JSON/CSV 汇总文件，默认保存到 `outputs/debug_denoise_moe`。

## 主要文件

```text
demo_moe.py             Wan2.2-TI2V-5B Point Prompting 命令行入口
debug_denoise_moe.py    Wan2.2-TI2V-5B 重建调试工具
marker.py               红点插入与检测
color_rebalance.py      自然红色抑制
distillation.py         模型无关的学生跟踪器训练工具
eval_tapvid.py          模型无关的 TAP-Vid 指标与评估工具
```

## 更新记录

### v3.0 — 2026-06-09

- 将 Wan2.2-TI2V-5B 设为唯一支持的模型。
- 删除此前的模型实现、兼容适配层和对应测试。
- 对 checkpoint scheduler 和官方 TI2V 首帧条件执行 fail-fast 强校验。
- 新增双 T4 分辨率预设、float16 VAE 自动模式和更明确的 OOM 提示。

### 历史版本

早期模型实验仍可从 Git 历史和旧 tag 获取，但不属于 v3.0 工作树的支持范围。

## 引用

```bibtex
@inproceedings{shrivastava2026pointprompting,
  title     = {Point Prompting: Counterfactual Tracking with Video Diffusion Models},
  author    = {Shrivastava, Ayush and Mehta, Sanyam and Geng, Daniel and Owens, Andrew},
  booktitle = {ICLR},
  year      = {2026},
  url       = {https://openreview.net/forum?id=6FFQ007qLX}
}
```

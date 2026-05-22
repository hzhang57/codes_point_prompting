# Point Prompting: Counterfactual Tracking with Video Diffusion Models

**v1.0** — 2026-05-23

A zero-shot point tracker based on pre-trained image-conditioned video diffusion models. This is an implementation of the method from the paper [*Point Prompting: Counterfactual Tracking with Video Diffusion Models*](https://arxiv.org/abs/2510.11715).

## Changelog

### v1.0 (2026-05-23)
- Full pipeline working end-to-end: color rebalance → marker insert → counterfactual SDEdit → marker detection → inpainting refinement
- Supports **CogVideoX-5B-I2V** and **Wan2.1-I2V / VACE** backbones via unified `ModelAdapter`
- Dual-GPU support (2× T4 15 GiB): transformer loaded with `device_map="balanced"`; CogVideoX VAE pinned to cuda:0 via asymmetric `max_memory` to avoid OOM
- Per-stage progress prints: `[阶段 1/2]` SDEdit, `[阶段 2/2]` refinement
- Input video preprocessing: auto-resize to model resolution (720×480 for CogVideoX, 832×480 for Wan) with stride alignment; tracks returned in original-frame coordinates
- `--no-refine` flag to skip refinement and halve runtime
- `--max-frames` flag to limit frame count and VRAM usage

## Overview

The key idea: insert a small red circular marker at a query point in frame 0, then use **counterfactual SDEdit** to regenerate the video so the marker propagates naturally through subsequent frames. The marker's centroid in each generated frame gives the track.

**Pipeline (per query point):**

1. **Color rebalancing** — suppress natural reds in the video (HSV saturation cap at 80 for red-hue pixels) so they don't interfere with marker detection
2. **Marker insertion** — draw a 2 px red circle at the query point in frame 0
3. **Counterfactual SDEdit** — regenerate the video using the marked frame as the positive image condition and the original frame as the negative, with guidance weight λ:
   ```
   v̂ = (λ+1) · v(c_edited) − λ · v(c_original)
   ```
4. **Marker detection** — detect the red marker centroid in each generated frame using HSV thresholding within a local search window
5. **Inpainting refinement** (optional) — re-denoise a small patch around each detected position at lower noise level γ=0.3 for sub-pixel accuracy

Supports **CogVideoX-5B-I2V** and **Wan2.1-I2V / VACE** backbones.

## Requirements

```
torch>=2.1.0
diffusers>=0.30.0
transformers>=4.40.0
accelerate>=0.30.0
opencv-python>=4.9.0
pillow>=10.0.0
numpy>=1.24.0
```

Install:
```bash
pip install -r requirements.txt
```

For Wan VACE support, make sure your diffusers version exposes `WanVACEPipeline` (≥0.32.0 recommended).

## Quick Start

```bash
# Wan2.1 I2V 1.3B (fits on ~12 GB VRAM, two T4s recommended)
python demo.py --video input.mp4 --points "320,240" \
               --model-type wan --model-id Wan-AI/Wan2.1-I2V-1.3B-480P

# Wan2.1 I2V 14B (best quality, needs ~40 GB VRAM)
python demo.py --video input.mp4 --points "320,240" "640,360" \
               --model-type wan --model-id Wan-AI/Wan2.1-I2V-14B-480P

# Wan2.1 VACE 1.3B
python demo.py --video input.mp4 --points "320,240" \
               --model-type wan --model-id Wan-AI/Wan2.1-VACE-1.3B

# CogVideoX 5B I2V
python demo.py --video input.mp4 --points "320,240" \
               --model-type cogvideox --model-id THUDM/CogVideoX-5b-I2V

# CogVideoX 5B I2V — fast test (10 frames, no refinement)
python demo.py --video input.mp4 --points "1157,635" \
               --model-type cogvideox --model-id THUDM/CogVideoX-5b-I2V \
               --max-frames 10 --steps 50 --no-refine

# CogVideoX 5B I2V — full run (10 frames, with refinement)
python demo.py --video input.mp4 --points "1157,635" \
               --model-type cogvideox --model-id THUDM/CogVideoX-5b-I2V \
               --max-frames 10 --steps 50
```

The output is a video (`tracked.mp4` by default) with the trajectory drawn on the original frames.

### Multiple points

```bash
python demo.py --video input.mp4 --points "100,200" "400,300" "600,150" \
               --model-type wan --model-id Wan-AI/Wan2.1-I2V-1.3B-480P
```

Each point runs through the full pipeline independently.

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--points` | required | Query point(s) as `x,y` in frame-0 pixel coords |
| `--model-type` | `wan` | Backbone: `wan` or `cogvideox` |
| `--model-id` | model-type default | HuggingFace model ID |
| `--output` | `tracked.mp4` | Output video path |
| `--gamma` | `0.5` | SDEdit noise ratio γ |
| `--lam` | `8.0` | Counterfactual guidance weight λ |
| `--steps` | `50` | Diffusion denoising steps |
| `--no-refine` | off | Skip inpainting refinement (faster) |
| `--seed` | `42` | Random seed |
| `--max-frames` | `50` | Max frames to process |
| `--preprocess-width` | `832` | Resize video width before tracking |
| `--preprocess-height` | `480` | Resize video height before tracking |
| `--device` | `cuda` | Compute device |

## Python API

```python
from model_adapter import load_wan_pipe, create_adapter
from tracker import PointPrompter, PointPrompterConfig
import cv2

# Load frames
cap = cv2.VideoCapture("input.mp4")
frames = []
while len(frames) < 50:
    ret, f = cap.read()
    if not ret: break
    frames.append(f)
cap.release()

# Load model
pipe    = load_wan_pipe("Wan-AI/Wan2.1-I2V-1.3B-480P")
tracker = PointPrompter(create_adapter(pipe), PointPrompterConfig(seed=42))

# Track a single point
result = tracker.track(frames, query_point=(320.0, 240.0))
print(result.tracks)    # (T, 2) float32 — (x, y) per frame
print(result.visible)   # (T,)   bool    — whether marker was detected

# Track multiple points
results = tracker.track_multiple(frames, [(320, 240), (640, 360)])
```

### PointPrompterConfig

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `0.5` | SDEdit noise ratio (paper default) |
| `lam` | `8.0` | Counterfactual guidance weight (paper default) |
| `num_inference_steps` | `50` | Denoising steps |
| `marker_radius` | `2` | Red marker radius in pixels (paper ablation optimum) |
| `do_refine` | `True` | Enable inpainting refinement pass |
| `refine_gamma` | `0.3` | Noise ratio for refinement (< gamma) |
| `prompt` | `""` | Text prompt (paper uses empty string) |
| `seed` | `None` | Random seed for reproducibility |
| `model_width` | `832` | Max width fed to diffusion model |
| `model_height` | `480` | Max height fed to diffusion model |
| `model_stride` | `16` | Spatial alignment stride |

## Evaluation (TAP-Vid)

```python
from model_adapter import load_wan_pipe, create_adapter
from tracker import PointPrompter, PointPrompterConfig
from eval_tapvid import load_tapvid_davis, evaluate

pipe    = load_wan_pipe("Wan-AI/Wan2.1-I2V-14B-480P")
tracker = PointPrompter(create_adapter(pipe))

samples = load_tapvid_davis("tapvid_davis.pkl")
metrics = evaluate(tracker, samples)
# prints per-video and mean AJ / δ_avg^x / OA
```

Metrics follow the TAP-Vid protocol (Doersch et al., 2022):
- **AJ** (Average Jaccard) — joint position accuracy and occlusion prediction
- **δ_avg^x** — position accuracy averaged over thresholds 1/2/4/8/16 px
- **OA** (Occlusion Accuracy) — frame-level occlusion prediction accuracy

## Multi-GPU Setup

`load_wan_pipe` and `load_cogvideox_pipe` automatically detect the number of available GPUs.

**Wan:**
- **2+ GPUs**: `device_map="balanced"` spreads transformer layers across all GPUs
- **1 GPU**: `enable_model_cpu_offload()`

**CogVideoX (dual T4 15 GiB):**
- Asymmetric `max_memory`: cuda:0 gets `total-5 GiB`, cuda:1 gets `total-1 GiB`
- This forces most transformer layers onto cuda:1, leaving ~5 GiB free on cuda:0
- VAE accelerate hooks are removed after load; VAE is pinned entirely to cuda:0 at fp16
- Result: VAE encode/decode runs on GPU (10–20× faster than CPU fallback); transformer pipeline-parallel across both GPUs

No extra configuration needed — just make sure `accelerate` is installed.

## File Structure

```
├── demo.py            # CLI entry point and visualization
├── tracker.py         # PointPrompter: full tracking pipeline
├── sdedit.py          # Counterfactual SDEdit core loop
├── marker.py          # Red marker insertion and detection
├── color_rebalance.py # HSV saturation clamp for natural reds
├── refinement.py      # Inpainting refinement pass
├── model_adapter.py   # Unified adapter for CogVideoX / Wan I2V / Wan VACE
├── eval_tapvid.py     # TAP-Vid benchmark evaluation
└── requirements.txt
```

## Reference

```bibtex
@article{pointprompting2024,
  title   = {Point Prompting: Counterfactual Tracking with Video Diffusion Models},
  year    = {2024},
  url     = {https://arxiv.org/abs/2510.11715}
}
```

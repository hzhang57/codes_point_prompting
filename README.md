# Point Prompting: Counterfactual Tracking with Video Diffusion Models

**v2.1** — 2026-05-27

An unofficial third-party implementation of [*Point Prompting: Counterfactual Tracking with Video Diffusion Models*](https://openreview.net/forum?id=6FFQ007qLX) (ICLR 2026 Poster).

## Changelog

### Current
- Define `gamma` as intuitive noise strength (`0` = none, `1` = maximum)
- Add multi-strength SDEdit reconstruction experiment with denoised MP4 output

### v2.1 (2026-05-27)
- Switch Wan2.1-VACE-1.3B sampling to the official Diffusers `UniPCMultistepScheduler`
- Use `flow_prediction` / flow sigmas from model config with default `flow_shift=3.0` for 480P
- Add `--flow-shift` for 720P-style runs (`5.0`) and scheduler experiments

### v2.0 (2026-05-24)
- Enable counterfactual guidance loop (paper Eq. 3): `v̂ = (λ+1)·v(c_edited) − λ·v(c_original)`
- Fix VAE frame count handling: valid T = 4k+1; auto-clip input frames
- Fix VAE tiling: `enable_slicing()` only — `enable_tiling()` causes checkerboard artifacts
- Replace all MP4 debug saves with PNG sequences for reliable viewing on Kaggle
- Add configurable refinement noise level
- Debug per-step full-frame decode to verify denoising progress

### v1.0 (2026-05-23)
- Full pipeline: color rebalance → marker insert → SDEdit → marker detection → inpainting refinement
- Video diffusion backbone via unified `ModelAdapter`
- Dual-GPU support (2× T4 15 GiB)

## Overview

Insert a small red circular marker at a query point in frame 0, then use **counterfactual SDEdit** to regenerate the video so the marker propagates naturally through subsequent frames.

**Pipeline (per query point):**

1. **Color rebalancing** — suppress natural reds (HSV saturation cap) so they don't interfere with marker detection
2. **Marker insertion** — draw a 2 px red circle at the query point in frame 0
3. **Counterfactual SDEdit** — regenerate the video with the marked frame as positive condition and the original frame as negative:
   ```
   v̂ = (λ+1) · v(c_edited) − λ · v(c_original)
   ```
4. **Marker detection** — detect red marker centroid per frame via HSV thresholding
5. **Inpainting refinement** (optional) — re-denoise a small patch around each detected position at lower noise level for sub-pixel accuracy

## Requirements

```
torch>=2.1.0
diffusers>=0.30.0
transformers>=4.40.0
accelerate>=0.30.0
opencv-python>=4.9.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
```

## Quick Start

```bash
python demo.py \
  --video input.mp4 \
  --points "320,240" \
  --model-id Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --max-frames 81
```

Wan VACE uses a temporal VAE stride where valid input counts follow T = 4k+1. The default `--max-frames 81` matches the Wan2.1-VACE 480P workflow used by this repo.

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--points` | required | Query point(s) as `x,y` in frame-0 pixel coords |
| `--model-id` | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | HuggingFace model ID |
| `--output` | `tracked.mp4` | Output video path |
| `--gamma` | `0.5` | SDEdit noise strength: `0` = none, `1` = maximum |
| `--lam` | `8.0` | Counterfactual guidance weight λ (paper default) |
| `--scheduler-steps` | `100` | Total scheduler timesteps (paper default) |
| `--flow-shift` | `3.0` | UniPC flow shift; use `5.0` for 720P-style Wan runs |
| `--no-refine` | off | Skip inpainting refinement (faster) |
| `--seed` | `42` | Random seed |
| `--max-frames` | `81` | Max frames; must satisfy T = 4k+1 |
| `--device` | `cuda` | Compute device |

## PointPrompterConfig

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `0.5` | SDEdit noise strength: `0` = none, `1` = maximum |
| `lam` | `8.0` | Counterfactual guidance weight (paper default) |
| `scheduler_steps` | `100` | Total scheduler timesteps |
| `marker_radius` | `2` | Red marker radius in pixels (paper ablation optimum) |
| `do_refine` | `True` | Enable inpainting refinement pass |
| `refine_gamma` | `0.3` | Lower noise strength for conservative refinement |
| `prompt` | `""` | Text prompt (paper uses empty string) |
| `seed` | `None` | Random seed for reproducibility |
| `model_width` | `832` | Max width fed to diffusion model |
| `model_height` | `480` | Max height fed to diffusion model |
| `model_stride` | `16` | Spatial alignment stride |

## Wan VACE Scheduler

This repo follows the official Wan2.1-VACE-1.3B Diffusers setup:

```python
UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
```

The model config provides `prediction_type="flow_prediction"` and `use_flow_sigmas=True`. The default `flow_shift=3.0` targets 480P; pass `--flow-shift 5.0` for 720P-style settings.

## SDEdit Reconstruction Check

Use `debug_denoise.py` to scan noise strengths and verify that denoising recovers
a clean input video:

```bash
python debug_denoise.py --video input.mp4 --max-frames 9
```

By default, the clean first frame is passed through VACE's official single
`reference_images` path. The complete clean video is encoded independently,
noised as the SDEdit state, and never supplied to the VACE control branch.
Use `--condition-mode legacy-first-frame` to compare against the previous
first-frame-plus-zero-video control.

Results are written under `outputs/debug_denoise/`. Each
`gamma_<value>/denoised.mp4` is the fully denoised video; the same directory
also contains `noisy.mp4`, denoised PNG frames, and a three-column
`compare.mp4`. Aggregate PSNR and latent MSE results are saved in
`summary.csv` and `summary.json`.

The reconstruction script enables low-CPU-RAM mode by default: model loading
uses `low_cpu_mem_usage`, T5 keeps its loaded dtype instead of expanding to
float32, and T5 is released after prompt encoding. For a constrained notebook,
start with one strength and fewer pixels:

```bash
python debug_denoise.py --video input.mp4 --max-frames 5 --width 512 --height 288 --gammas 0.5 --conditioning-scale 1.0
```

## File Structure

```
├── demo.py            # CLI entry point and visualization
├── tracker.py         # PointPrompter: full tracking pipeline
├── sdedit.py          # Counterfactual SDEdit core loop
├── marker.py          # Red marker insertion and detection
├── color_rebalance.py # HSV saturation clamp for natural reds
├── refinement.py      # Inpainting refinement pass
├── model_adapter.py   # Wan VACE adapter + pipeline loader
└── requirements.txt
```

## Reference

```bibtex
@inproceedings{shrivastava2026pointprompting,
  title     = {Point Prompting: Counterfactual Tracking with Video Diffusion Models},
  author    = {Shrivastava, Ayush and Mehta, Sanyam and Geng, Daniel and Owens, Andrew},
  booktitle = {ICLR},
  year      = {2026},
  url       = {https://openreview.net/forum?id=6FFQ007qLX}
}
```

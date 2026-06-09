# Point Prompting with Wan2.2-TI2V-5B

[中文说明](README.md)

**v3.0** — 2026-06-09

An unofficial implementation of counterfactual point prompting using only
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`.

Version 3.0 removes the previous model backends and compatibility layers. The
repository now has one model path, one scheduler contract, and one official
first-frame conditioning flow.

## Pipeline

For each query point:

1. Suppress naturally red regions in the input video.
2. Draw a red marker on frame 0.
3. Encode the edited video and add SDEdit noise at strength `gamma`.
4. Build two official TI2V first-frame conditions with
   `prepare_latents(image, latents=...)`:
   - positive: marked frame 0
   - negative: original frame 0
5. Apply counterfactual guidance at every denoising step:

   ```python
   v_guided = (lam + 1) * v_marked - lam * v_original
   ```

6. Decode the generated video and detect the propagated marker.
7. Optionally run a second conservative TI2V SDEdit refinement pass.

## Model Contract

The only supported checkpoint is:

```text
Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

The pipeline must be `WanImageToVideoPipeline` with
`expand_timesteps=True`. The checkpoint scheduler is validated at startup:

```text
class=UniPCMultistepScheduler
flow_shift=5.0
prediction_type=flow_prediction
use_flow_sigmas=True
timestep_spacing=linspace
```

Wan2.2-TI2V-5B is a dense 5B model. The `moe` suffix in script names is kept
for compatibility with the established commands.

## Installation

```bash
pip install -r requirements.txt
```

Use a current Diffusers build that includes Wan2.2 TI2V/I2V support.

## Point Prompting Demo

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

`--output` is resolved relative to the current working directory. Debug files
and `summary.json` are written under `outputs/demo_moe` by default.

Multiple points are supported:

```bash
python demo_moe.py --video input.mp4 --points "900,535" "1157,635"
```

### Kaggle T4 x 2 Commands

Select the dual-T4 accelerator in a Kaggle Notebook, then run:

```python
!cd /kaggle/working
!pip install ftfy
!rm -rf /kaggle/working/*
!git clone https://github.com/hzhang57/codes_point_prompting.git

!python /kaggle/working/codes_point_prompting/demo_moe.py \
  --video /kaggle/working/codes_point_prompting/input.mp4 \
  --points "900,535" \
  --gamma 0.5 \
  --lam 8.0 \
  --max-frames 30 \
  --output tracked_moe.mp4 \
  --save-generated
```

In notebook cells, `!cd` only affects that shell command. The example uses
absolute paths for the script and video, so later commands are unaffected. The
Wan temporal VAE requires `T=4k+1`; `--max-frames 30` is automatically clipped
to 29 frames. `tracked_moe.mp4` is written to the notebook's current working
directory and can be located with:

```python
!find /kaggle/working -name "tracked_moe.mp4" -print
```

### Resolution And Dual T4

The official 720P sizes are:

- landscape: `1280x704`
- portrait: `704x1280`

Official 720P generally requires more memory than a 15 GiB T4 provides. The
default `--resolution-preset t4` uses `832x480` or `480x832`. On two GPUs, the
transformer is placed on `cuda:0` and the VAE on `cuda:1`; `--vae-dtype auto`
uses float16 on CUDA.

Use official resolution explicitly:

```bash
python demo_moe.py --video input.mp4 --points "900,535" --resolution-preset official
```

If the T4 preset still runs out of memory:

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

## Reconstruction Debug

Use the reconstruction script to scan SDEdit noise strengths while preserving
the official TI2V first-frame condition and scheduler:

```bash
python debug_denoise_moe.py \
  --video input.mp4 \
  --max-frames 9 \
  --width 832 \
  --height 480 \
  --gammas 0.5
```

Outputs include VAE round-trip reconstruction, denoised videos, timestep logs,
PSNR metrics, and JSON/CSV summaries under `outputs/debug_denoise_moe`.

## Main Files

```text
demo_moe.py             Wan2.2-TI2V-5B point prompting CLI
debug_denoise_moe.py    Wan2.2-TI2V-5B reconstruction debugger
marker.py               Red marker insertion and detection
color_rebalance.py      Natural-red suppression
distillation.py         Model-independent student tracker training tools
eval_tapvid.py          Model-independent TAP-Vid metrics and evaluation
```

## Changelog

### v3.0 — 2026-06-09

- Make Wan2.2-TI2V-5B the only supported model.
- Remove previous model implementations, compatibility adapters, and tests.
- Keep the checkpoint scheduler and official TI2V first-frame condition as
  fail-fast contracts.
- Add a dual-T4 resolution preset, float16 VAE auto mode, and clearer OOM
  guidance.

### Historical Releases

Earlier model experiments remain available in Git history and previous tags,
but are not supported by the v3.0 working tree.

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

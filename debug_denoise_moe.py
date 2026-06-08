"""
Wan2.2-TI2V-5B 的 SDEdit 视频重建实验。

脚本先把干净视频编码成 latent，再按 gamma 加噪，最后使用首帧作为 I2V 条件
进行去噪。条件注入、mask 融合和 scheduler 都严格遵循 Diffusers 官方流程。

注意：虽然文件名包含 moe，但默认的 TI2V-5B 是 dense 5B 模型；显式 MoE
模型是 Wan2.2 A14B 系列。

示例：
  python debug_denoise_moe.py --video input.mp4 --max-frames 9 --gammas 0.5
"""

import argparse
import csv
import gc
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

from debug_denoise import (
    gamma_dir_name,
    json_metric,
    load_video,
    mean_psnr,
    save_comparison,
    save_frames,
    save_video,
)
from model_adapter import _frames_to_tensor, _tensor_to_frames, noise_strength_to_start_idx


# 这些值来自 Wan-AI/Wan2.2-TI2V-5B-Diffusers 的官方 checkpoint 配置。
# 启动时会逐项校验，避免脚本在错误 scheduler 上“看似正常”地运行。
DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
OFFICIAL_SCHEDULER_CLASS = "UniPCMultistepScheduler"
OFFICIAL_FLOW_SHIFT = 5.0
OFFICIAL_MAX_SEQUENCE_LENGTH = 512


def release_memory() -> None:
    """释放 Python 对象和 PyTorch CUDA 缓存，降低双 T4 上的显存峰值。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_vae_internal_cache(pipe) -> None:
    """清理 Wan VAE 可能保留的时序 feature cache。

    Wan VAE 会逐帧编码/解码，并可能在模块内部缓存中间特征。若不主动释放，
    下一次 VAE decode 可能因为这些残留特征而 OOM。
    """
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    for name in ("clear_cache", "_clear_cache", "clear_context_parallel_cache"):
        fn = getattr(vae, name, None)
        if callable(fn):
            try:
                fn()
            except TypeError:
                pass
    for attr in ("_feat_map", "_features", "_cache"):
        if hasattr(vae, attr):
            try:
                setattr(vae, attr, None)
            except Exception:
                pass


def _pipe_device(pipe) -> torch.device:
    """返回 transformer 所在设备，也就是去噪主计算设备。"""
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        try:
            return next(transformer.parameters()).device
        except StopIteration:
            pass
    d = getattr(pipe, "_execution_device", None)
    if d is not None:
        return torch.device(d)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _transformer_dtype(pipe) -> torch.dtype:
    """返回 transformer 的计算精度，通常为 bfloat16。"""
    return getattr(pipe.transformer, "dtype", torch.float32)


def _vae_device(pipe) -> torch.device:
    """返回 VAE 所在设备；双卡模式下通常是 cuda:1。"""
    return next(pipe.vae.parameters()).device


def _vae_dtype(pipe) -> torch.dtype:
    """返回 VAE 的计算精度，默认 float32 以保证重建质量。"""
    return next(pipe.vae.parameters()).dtype


def _vae_norm(pipe, device) -> tuple:
    """构造官方 Wan VAE latent 的归一化均值和缩放系数。"""
    mean = torch.tensor(
        pipe.vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(
        pipe.vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    return mean, std


def _config_value(config, name, default=None):
    """兼容 dict 和 Diffusers FrozenDict 两种配置读取方式。"""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def scheduler_info(pipe) -> dict:
    """收集会影响加噪与去噪轨迹的关键 scheduler 配置。"""
    config = getattr(getattr(pipe, "scheduler", None), "config", None)
    return {
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config_class": _config_value(config, "_class_name"),
        "scheduler_flow_shift": _config_value(config, "flow_shift"),
        "scheduler_prediction_type": _config_value(config, "prediction_type"),
        "scheduler_use_flow_sigmas": _config_value(config, "use_flow_sigmas"),
        "scheduler_timestep_spacing": _config_value(config, "timestep_spacing"),
        "scheduler_solver_order": _config_value(config, "solver_order"),
        "scheduler_solver_type": _config_value(config, "solver_type"),
        "scheduler_num_train_timesteps": _config_value(config, "num_train_timesteps"),
    }


def tensor_head_tail(tensor: torch.Tensor, count: int = 5) -> tuple:
    """只保留 timestep 序列的开头和结尾，方便写日志而不刷屏。"""
    values = [float(x) for x in tensor.detach().float().cpu().tolist()]
    return values[:count], values[-count:]


def print_scheduler_info(pipe) -> dict:
    """打印实际加载的 scheduler，便于确认当前运行与官方配置一致。"""
    info = scheduler_info(pipe)
    print(
        "[scheduler] "
        f"class={info['scheduler_class']} source=checkpoint_config "
        f"flow_shift={info['scheduler_flow_shift']} "
        f"prediction_type={info['scheduler_prediction_type']} "
        f"use_flow_sigmas={info['scheduler_use_flow_sigmas']} "
        f"timestep_spacing={info['scheduler_timestep_spacing']} "
        f"solver_order={info['scheduler_solver_order']} "
        f"solver_type={info['scheduler_solver_type']} "
        f"num_train_timesteps={info['scheduler_num_train_timesteps']}"
    )
    return {**info, "scheduler_source": "checkpoint_config"}


def validate_wan22_ti2v5b_pipeline(pipe) -> None:
    """对官方 Wan2.2-TI2V-5B pipeline 做启动前的强校验。

    本脚本只支持该模型的 expand_timesteps 条件路径。发现 pipeline、scheduler
    或 patch size 不一致时立即报错，避免得到无法解释的实验结果。
    """
    if pipe.__class__.__name__ != "WanImageToVideoPipeline":
        raise TypeError(
            "debug_denoise_moe.py requires WanImageToVideoPipeline, "
            f"got {pipe.__class__.__name__}"
        )
    required = ("transformer", "vae", "scheduler", "video_processor", "prepare_latents")
    missing = [name for name in required if not hasattr(pipe, name)]
    if missing:
        raise TypeError(f"Wan2.2-TI2V-5B pipeline is missing components: {missing}")
    if not bool(_config_value(getattr(pipe, "config", None), "expand_timesteps", False)):
        raise ValueError("Wan2.2-TI2V-5B requires pipe.config.expand_timesteps=True")

    info = scheduler_info(pipe)
    # TI2V-5B checkpoint 的官方 scheduler 配置。这里不允许静默替换。
    expected = {
        "scheduler_class": OFFICIAL_SCHEDULER_CLASS,
        "scheduler_flow_shift": OFFICIAL_FLOW_SHIFT,
        "scheduler_prediction_type": "flow_prediction",
        "scheduler_use_flow_sigmas": True,
        "scheduler_timestep_spacing": "linspace",
    }
    mismatches = {
        name: (info.get(name), value)
        for name, value in expected.items()
        if info.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Pipeline scheduler does not match official Wan2.2-TI2V-5B config: "
            f"{mismatches}"
        )

    patch_size = tuple(getattr(pipe.transformer.config, "patch_size", ()))
    if len(patch_size) < 3 or tuple(patch_size[-2:]) != (2, 2):
        raise ValueError(
            "Wan2.2-TI2V-5B expanded timestep path requires spatial patch_size=(2, 2), "
            f"got {patch_size}"
        )


def resolve_max_sequence_length(pipe, requested) -> int:
    """优先使用命令行值，其次读取模型配置，最后回退到官方默认 512。"""
    if requested is not None:
        return int(requested)
    for config in (getattr(pipe, "config", None), getattr(pipe.transformer, "config", None)):
        value = _config_value(config, "max_sequence_length")
        if value is None:
            value = _config_value(config, "max_text_seq_len")
        if value is not None:
            return int(value)
    return OFFICIAL_MAX_SEQUENCE_LENGTH


def _retrieve_latents_argmax(encoder_output):
    """从不同版本 Diffusers 的 VAE encode 返回值中取确定性 latent。"""
    if hasattr(encoder_output, "latent_dist"):
        dist = encoder_output.latent_dist
        if hasattr(dist, "mode"):
            return dist.mode()
        return dist.mean
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder output")


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR 图像转为 Diffusers 常用的 RGB PIL 图像。"""
    return Image.fromarray(frame_bgr[..., ::-1].copy())


def _component_to(component, *args, **kwargs) -> None:
    """组件存在且支持 `.to()` 时才移动设备或切换 dtype。"""
    if component is not None and hasattr(component, "to"):
        component.to(*args, **kwargs)


def _resolve_cuda_device(device: str) -> torch.device:
    """把模糊的 `cuda` 显式解析为 `cuda:0`。"""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda:0")
    return dev


def resolve_vae_device(device: str, vae_device: str) -> torch.device:
    """自动把 VAE 放到另一张 GPU，避免和 5B transformer 抢显存。"""
    main_dev = _resolve_cuda_device(device)
    if vae_device == "auto":
        if main_dev.type == "cuda" and torch.cuda.device_count() > 1:
            return torch.device("cuda:1" if main_dev.index == 0 else "cuda:0")
        return main_dev
    return _resolve_cuda_device(vae_device)


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    """把便于命令行输入的 dtype 名称转换成 torch.dtype。"""
    name = dtype_name.lower()
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError("--vae-dtype must be one of: float32, float16, bfloat16")


def load_wan_ti2v_pipe(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    vae_device: str = "auto",
    vae_dtype: str = "float32",
    low_cpu_memory: bool = True,
):
    """加载官方 WanImageToVideoPipeline，并按双卡调试策略放置组件。

    transformer 是主要去噪网络，默认放在 `--device`；VAE 负责视频和首帧条件
    的编码/解码，默认 `--vae-device auto` 会在双 GPU 环境放到另一张卡。
    """
    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    except ImportError as exc:
        raise ImportError(
            "debug_denoise_moe.py requires a Diffusers build with Wan2.2 TI2V/I2V "
            "support. Install the current Diffusers main branch, for example: "
            "pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc

    # VAE 单独加载为用户指定精度；默认 float32，重建 PSNR 通常更稳。
    vae_torch_dtype = parse_torch_dtype(vae_dtype)
    loading_kwargs = {"low_cpu_mem_usage": low_cpu_memory}
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_torch_dtype,
            **loading_kwargs,
        )
    except ValueError as exc:
        if "low_cpu_mem_usage" in str(exc):
            loading_kwargs = {"low_cpu_mem_usage": True}
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=vae_torch_dtype,
                **loading_kwargs,
            )
        else:
            raise

    # 这里故意不使用 DiffusionPipeline fallback：该 checkpoint 的 model_index
    # 可能指向 T2V pipeline，而本脚本必须使用 I2V 的 prepare_latents(image, ...).
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        **loading_kwargs,
    )
    # 双 T4 15GB 时，transformer 和 VAE 放在同一张卡很容易 OOM。
    # 因此默认 transformer -> cuda:0，VAE -> cuda:1。
    dev = _resolve_cuda_device(device)
    vae_dev = resolve_vae_device(device, vae_device)
    if dev.type == "cuda":
        _component_to(getattr(pipe, "transformer", None), dev)
        _component_to(getattr(pipe, "vae", None), vae_dev, dtype=vae_torch_dtype)
        _component_to(getattr(pipe, "text_encoder", None), torch.device("cpu"))
    else:
        if hasattr(pipe, "to"):
            pipe.to(dev)
        _component_to(getattr(pipe, "vae", None), vae_dev, dtype=vae_torch_dtype)

    # VAE slicing 会降低单次 decode/encode 的峰值显存，代价是稍慢。
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    pipe._low_memory_text_encoder = low_cpu_memory
    validate_wan22_ti2v5b_pipeline(pipe)
    return pipe


def encode_video_official(pipe, frames_bgr: list) -> torch.Tensor:
    """把完整视频编码为 Wan latent，作为 SDEdit 的干净起点。"""
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    tensor = _frames_to_tensor(frames_bgr, vae_dev, vae_dtype)
    # VAE encode 产生未标准化 latent；Wan pipeline 使用 (latent - mean) * std。
    with torch.no_grad():
        latents = _retrieve_latents_argmax(pipe.vae.encode(tensor))
    mean, std = _vae_norm(pipe, latents.device)
    latents = ((latents.float() - mean) * std).to(_transformer_dtype(pipe))
    latents = latents.to(device=_pipe_device(pipe))
    clear_vae_internal_cache(pipe)
    release_memory()
    return latents


def decode_latents_official(pipe, latents: torch.Tensor) -> list:
    """把 Wan latent 解码回 BGR 帧，用于保存 MP4/PNG 和计算 PSNR。"""
    vae_dev = _vae_device(pipe)
    vae_dtype = _vae_dtype(pipe)
    mean, std = _vae_norm(pipe, latents.device)
    # 解码前要撤销 encode 阶段的 Wan latent 标准化。
    latents = (latents.float() / std + mean).to(device=vae_dev, dtype=vae_dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents).sample
    frames = _tensor_to_frames(decoded)
    del decoded, latents
    clear_vae_internal_cache(pipe)
    release_memory()
    return frames


def encode_prompt_official(
    pipe,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    max_sequence_length: int,
):
    """编码文本 prompt；低内存模式下优先让 T5 留在 CPU。"""
    device = _pipe_device(pipe)
    dtype = _transformer_dtype(pipe)
    # 如果没有启用低内存路径，直接用 pipeline 官方 encode_prompt。
    if hasattr(pipe, "encode_prompt") and not getattr(pipe, "_low_memory_text_encoder", False):
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        return prompt_embeds.to(dtype=dtype, device=device), (
            negative_embeds.to(dtype=dtype, device=device)
            if negative_embeds is not None
            else None
        )

    if not hasattr(pipe, "_get_t5_prompt_embeds"):
        raise RuntimeError("Pipeline does not expose encode_prompt or _get_t5_prompt_embeds")

    # 低内存路径：只在 CPU 上跑 T5，然后把 prompt embeds 搬到 transformer 设备。
    prompt_embeds = pipe._get_t5_prompt_embeds(
        prompt=prompt,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=torch.device("cpu"),
        dtype=getattr(getattr(pipe, "text_encoder", None), "dtype", torch.float32),
    ).to(device=device, dtype=dtype)
    negative_embeds = None
    if guidance_scale > 1.0:
        negative_embeds = pipe._get_t5_prompt_embeds(
            prompt=negative_prompt or "",
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
    return prompt_embeds, negative_embeds


def prepare_wan22_first_frame_condition(
    pipe,
    first_frame_bgr: np.ndarray,
    noisy_latents: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    generator,
):
    """用官方 I2V `prepare_latents` 构造首帧条件和 mask。

    Wan2.2-TI2V-5B 的 expand_timesteps 路径会返回三件东西：
    - prepared_latents: 与输入 noisy_latents 同形状，作为真正去噪状态
    - condition: 首帧条件 latent
    - first_frame_mask: 哪些位置使用 condition，哪些位置使用当前 noisy latent
    """
    transformer_device = _pipe_device(pipe)
    vae_device = _vae_device(pipe)
    dtype = _transformer_dtype(pipe)
    # 官方 pipeline 先把首帧按目标分辨率预处理成 image tensor，再交给 VAE。
    image = _bgr_to_pil(first_frame_bgr)
    if not hasattr(pipe, "video_processor"):
        raise RuntimeError("Pipeline does not expose video_processor for image preprocessing")
    image = pipe.video_processor.preprocess(image, height=height, width=width)
    image = image.to(device=vae_device, dtype=torch.float32)

    # 关键点：这里显式传入 latents=noisy_latents。
    # 这表示“从我们加噪后的 SDEdit 状态开始”，同时使用首帧作为 I2V 条件。
    outputs = pipe.prepare_latents(
        image,
        batch_size=1,
        num_channels_latents=pipe.vae.config.z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=vae_device,
        generator=generator,
        latents=noisy_latents.to(device=vae_device, dtype=torch.float32),
    )
    if len(outputs) != 3:
        raise ValueError(
            "Wan2.2-TI2V-5B official expand_timesteps path must return "
            f"(latents, condition, first_frame_mask), got {len(outputs)} values"
        )
    prepared_latents, condition, first_frame_mask = outputs
    first_frame_mask = first_frame_mask.to(device=transformer_device, dtype=dtype)

    prepared_latents = prepared_latents.to(device=transformer_device, dtype=dtype)
    condition = condition.to(device=transformer_device, dtype=dtype)
    if prepared_latents.shape != noisy_latents.shape:
        raise ValueError(
            "prepare_latents changed noisy latent shape: "
            f"input={tuple(noisy_latents.shape)} output={tuple(prepared_latents.shape)}"
        )
    try:
        torch.broadcast_shapes(first_frame_mask.shape, prepared_latents.shape)
    except RuntimeError as exc:
        raise ValueError(
            "Wan2.2-TI2V first_frame_mask must broadcast to latent shape: "
            f"mask={tuple(first_frame_mask.shape)} latents={tuple(prepared_latents.shape)}"
        ) from exc
    clear_vae_internal_cache(pipe)
    release_memory()
    return prepared_latents, condition, first_frame_mask


def scheduler_step_official(pipe, noise_pred, timestep, latents):
    """执行官方 Diffusers scheduler step。"""
    return pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]


def model_cache_context(model, name: str):
    """兼容支持 cache_context 的 transformer；不支持时就是普通 no-op。"""
    if hasattr(model, "cache_context"):
        return model.cache_context(name)
    return nullcontext()


def _expanded_timestep(model, first_frame_mask, timestep, batch_size):
    """把单个 timestep 展开到 token/patch 级别。

    Wan2.2-TI2V-5B 的官方 I2V 路径会让首帧条件 token 和待生成 token 使用
    不同的 timestep mask。这里按 transformer patch_size 对 mask 下采样。
    """
    patch_size = getattr(getattr(model, "config", None), "patch_size", (1, 2, 2))
    patch_h = int(patch_size[1]) if len(patch_size) > 1 else 2
    patch_w = int(patch_size[2]) if len(patch_size) > 2 else 2
    token_mask = first_frame_mask[0][0][:, ::patch_h, ::patch_w]
    temp_ts = (token_mask * timestep).flatten()
    return temp_ts.unsqueeze(0).expand(batch_size, -1).to(first_frame_mask.device)


def _transformer_forward(
    model,
    latent_model_input,
    timestep_batch,
    prompt_embeds,
    attention_kwargs,
    cache_name,
):
    """调用 Wan transformer，返回预测的 flow/noise。"""
    with torch.no_grad(), model_cache_context(model, cache_name):
        return model(
            hidden_states=latent_model_input,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]


def denoise_with_ti2v_pipeline_internals(
    pipe,
    latents: torch.Tensor,
    timesteps_run: torch.Tensor,
    condition: torch.Tensor,
    first_frame_mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    attention_kwargs=None,
):
    """执行 Wan2.2-TI2V-5B 的官方 expand_timesteps 去噪循环。"""
    transformer_dtype = _transformer_dtype(pipe)
    model = pipe.transformer

    for i, timestep in enumerate(timesteps_run):
        # 官方 mask fusion：首帧区域固定使用 condition，其余区域使用当前 latent。
        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
        timestep_batch = _expanded_timestep(
            model, first_frame_mask, timestep, latents.shape[0]
        )
        latent_model_input = latent_model_input.to(dtype=transformer_dtype)
        noise_pred = _transformer_forward(
            model,
            latent_model_input,
            timestep_batch,
            prompt_embeds,
            attention_kwargs,
            "cond",
        )
        # CFG：当 guidance_scale > 1 时，额外跑一次 negative prompt 分支。
        if negative_prompt_embeds is not None:
            noise_uncond = _transformer_forward(
                model,
                latent_model_input,
                timestep_batch,
                negative_prompt_embeds,
                attention_kwargs,
                "uncond",
            )
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # scheduler 根据 transformer 预测值，把 latent 从当前 timestep 推到下一步。
        latents = scheduler_step_official(pipe, noise_pred, timestep, latents)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(timesteps_run):
            print(
                f"[ti2v denoise {i + 1:3d}/{len(timesteps_run)}] "
                f"t={timestep.item():.1f} latent_norm={latents.norm().item():.1f}"
            )

    # 最终再融合一次，确保解码前首帧条件区域保持官方 I2V 约束。
    return (1 - first_frame_mask) * condition + first_frame_mask * latents


def set_denoise_start(pipe, n_steps: int, start_idx: int) -> torch.Tensor:
    """设置 scheduler timesteps，并返回本次 gamma 实际要跑的后半段。"""
    pipe.scheduler.set_timesteps(n_steps, device=_pipe_device(pipe))
    timesteps = pipe.scheduler.timesteps
    start_idx = max(0, min(int(start_idx), len(timesteps) - 1))
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(start_idx)
    return timesteps[start_idx:]


def add_noise_at_timestep(pipe, latents, noise, timestep):
    """在指定 timestep 上把干净 latent 加噪，得到 SDEdit 起点。"""
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    return pipe.scheduler.add_noise(latents, noise, timestep.to(device=latents.device))


def run_debug(args):
    """主实验入口：读取视频、编码 latent、逐个 gamma 加噪并重建。"""
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if len(set(args.gammas)) != len(args.gammas):
        raise ValueError("--gammas must not contain duplicates")
    for gamma in args.gammas:
        noise_strength_to_start_idx(gamma, args.scheduler_steps)

    # 1. 读入视频，并裁剪到 Wan temporal VAE 支持的 T=4k+1 帧数。
    os.makedirs(args.output_dir, exist_ok=True)
    frames, fps = load_video(args.video, args.max_frames, args.width, args.height)
    print(f"[input] {len(frames)} frames, {args.width}x{args.height}, {fps:.2f} fps")
    save_video(frames, os.path.join(args.output_dir, "original.mp4"), fps)
    save_frames(frames, os.path.join(args.output_dir, "original_frames"))

    # 2. 加载并校验官方 Wan2.2-TI2V-5B I2V pipeline。
    pipe = load_wan_ti2v_pipe(
        args.model_id,
        device=args.device,
        vae_device=args.vae_device,
        vae_dtype=args.vae_dtype,
        low_cpu_memory=args.low_memory,
    )
    validate_wan22_ti2v5b_pipeline(pipe)
    actual_model = "Wan2.2-TI2V-5B"
    print(
        f"[model] {actual_model} model_id={args.model_id} "
        f"transformer_device={_pipe_device(pipe)} transformer_dtype={_transformer_dtype(pipe)} "
        f"vae_device={_vae_device(pipe)} vae_dtype={_vae_dtype(pipe)}"
    )
    sched_info = print_scheduler_info(pipe)

    # 3. VAE round-trip：检查仅 VAE 编解码造成的基础损失。
    latents_clean = encode_video_official(pipe, frames)
    vae_frames = decode_latents_official(pipe, latents_clean)
    vae_psnr, vae_per_frame = mean_psnr(frames, vae_frames)
    save_video(vae_frames, os.path.join(args.output_dir, "vae_roundtrip.mp4"), fps)
    del vae_frames
    release_memory()
    print(f"[baseline] VAE round-trip PSNR={vae_psnr:.2f} dB")

    # 4. 文本条件只编码一次，之后每个 gamma 共用同一份 prompt embeds。
    max_sequence_length = resolve_max_sequence_length(pipe, args.max_sequence_length)
    print(f"[text] max_sequence_length={max_sequence_length}")
    prompt_embeds, negative_prompt_embeds = encode_prompt_official(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.guidance_scale,
        max_sequence_length,
    )
    if args.low_memory and getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    release_memory()

    # 5. 固定同一份基础噪声，保证不同 gamma 间可比较。
    torch.manual_seed(args.seed)
    base_noise = torch.randn_like(latents_clean)
    torch.save(base_noise.detach().cpu(), os.path.join(args.output_dir, "noise.pt"))

    video_latent_shape = list(latents_clean.shape)
    print(
        f"[latent] video={video_latent_shape} no_reference_slots=True "
        f"guidance_scale={args.guidance_scale}"
    )

    summaries = []
    for gamma in args.gammas:
        # 6. gamma 决定从 scheduler 的哪个 timestep 开始：
        #    gamma 越大，start_idx 越靠前，噪声越强，去噪步数越多。
        print(f"\n[gamma={gamma:.3f}] starting")
        gamma_dir = os.path.join(args.output_dir, gamma_dir_name(gamma))
        os.makedirs(gamma_dir, exist_ok=True)

        start_idx = noise_strength_to_start_idx(gamma, args.scheduler_steps)
        timesteps_run = set_denoise_start(pipe, args.scheduler_steps, start_idx)
        t_start = pipe.scheduler.timesteps[start_idx]
        if gamma == 0.0:
            timesteps_run = timesteps_run[:0]
        all_head, all_tail = tensor_head_tail(pipe.scheduler.timesteps)
        run_head, run_tail = tensor_head_tail(timesteps_run)
        print(
            f"[scheduler gamma={gamma:.3f}] steps={args.scheduler_steps} "
            f"start_idx={start_idx} t_start={float(t_start.item()):.1f} "
            f"denoise_steps={len(timesteps_run)} "
            f"timesteps_head={all_head} timesteps_tail={all_tail} "
            f"run_head={run_head} run_tail={run_tail}"
        )
        if gamma == 0.0:
            noisy_latents = latents_clean.clone()
        else:
            noisy_latents = add_noise_at_timestep(pipe, latents_clean, base_noise, t_start)

        # 7. 使用首帧作为官方 I2V 条件；注意这不是 VACE reference slot。
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        noisy_latents, condition, first_frame_mask = prepare_wan22_first_frame_condition(
            pipe,
            frames[0],
            noisy_latents,
            args.height,
            args.width,
            len(frames),
            generator,
        )

        # noisy.mp4 只是调试中间产物，默认跳过以降低 VAE decode 显存峰值。
        noisy_frames = None
        noisy_psnr = None
        noisy_per_frame = []
        if args.decode_noisy:
            noisy_frames = decode_latents_official(pipe, noisy_latents)
            noisy_psnr, noisy_per_frame = mean_psnr(frames, noisy_frames)
            save_video(noisy_frames, os.path.join(gamma_dir, "noisy.mp4"), fps)
        else:
            print("[decode] skipping noisy.mp4 to keep VAE memory for denoising")

        # 8. 真正的去噪循环：每步都用首帧 condition/mask 约束 latent。
        denoised_latents = denoise_with_ti2v_pipeline_internals(
            pipe,
            noisy_latents.clone(),
            timesteps_run,
            condition,
            first_frame_mask,
            prompt_embeds,
            negative_prompt_embeds,
            args.guidance_scale,
        )
        denoised_frames = decode_latents_official(pipe, denoised_latents)
        denoised_psnr, denoised_per_frame = mean_psnr(frames, denoised_frames)
        save_video(denoised_frames, os.path.join(gamma_dir, "denoised.mp4"), fps)
        save_frames(denoised_frames, os.path.join(gamma_dir, "denoised_frames"))
        if noisy_frames is not None:
            save_comparison(
                frames,
                noisy_frames,
                denoised_frames,
                os.path.join(gamma_dir, "compare.mp4"),
                fps,
            )

        # 9. 每个 gamma 的指标都落盘，便于之后横向比较。
        summary = {
            "gamma": gamma,
            "guidance_scale": args.guidance_scale,
            "reference_latent_slots": 0,
            "video_latent_shape": str(video_latent_shape),
            "transformer_latent_shape": str(list(noisy_latents.shape)),
            "condition_shape": str(list(condition.shape)),
            "start_idx": start_idx,
            "t_start": float(t_start.item()),
            "denoise_steps": len(timesteps_run),
            "scheduler_class": sched_info["scheduler_class"],
            "scheduler_source": sched_info["scheduler_source"],
            "scheduler_flow_shift": sched_info["scheduler_flow_shift"],
            "scheduler_prediction_type": sched_info["scheduler_prediction_type"],
            "scheduler_use_flow_sigmas": sched_info["scheduler_use_flow_sigmas"],
            "scheduler_timestep_spacing": sched_info["scheduler_timestep_spacing"],
            "timesteps_head": all_head,
            "timesteps_tail": all_tail,
            "timesteps_run_head": run_head,
            "timesteps_run_tail": run_tail,
            "vae_psnr": vae_psnr,
            "noisy_psnr": noisy_psnr,
            "denoised_psnr": denoised_psnr,
            "recovery_gain": (
                denoised_psnr - noisy_psnr if noisy_psnr is not None else None
            ),
            "gap_to_vae": vae_psnr - denoised_psnr,
            "noisy_latent_mse": float(
                torch.mean((noisy_latents.float() - latents_clean.float()) ** 2).item()
            ),
            "denoised_latent_mse": float(
                torch.mean((denoised_latents.float() - latents_clean.float()) ** 2).item()
            ),
            "recovered": denoised_psnr > noisy_psnr if noisy_psnr is not None else None,
            "noisy_psnr_per_frame": noisy_per_frame,
            "denoised_psnr_per_frame": denoised_per_frame,
            "decode_noisy": args.decode_noisy,
        }
        summaries.append(summary)
        if noisy_psnr is None:
            print(
                f"[gamma={gamma:.3f}] denoised={denoised_psnr:.2f} dB "
                "noisy_psnr=skipped"
            )
        else:
            print(
                f"[gamma={gamma:.3f}] noisy={noisy_psnr:.2f} dB "
                f"denoised={denoised_psnr:.2f} dB "
                f"gain={summary['recovery_gain']:+.2f} dB"
            )
        del noisy_latents, denoised_latents, noisy_frames, denoised_frames
        del condition, first_frame_mask
        release_memory()

    # 10. 写聚合结果：CSV 适合快速扫表，JSON 保留完整 per-frame/per-run 信息。
    csv_fields = [
        "gamma",
        "guidance_scale",
        "reference_latent_slots",
        "video_latent_shape",
        "transformer_latent_shape",
        "condition_shape",
        "start_idx",
        "t_start",
        "denoise_steps",
        "scheduler_class",
        "scheduler_source",
        "scheduler_flow_shift",
        "scheduler_prediction_type",
        "scheduler_use_flow_sigmas",
        "scheduler_timestep_spacing",
        "timesteps_head",
        "timesteps_tail",
        "timesteps_run_head",
        "timesteps_run_tail",
        "vae_psnr",
        "noisy_psnr",
        "denoised_psnr",
        "recovery_gain",
        "gap_to_vae",
        "noisy_latent_mse",
        "denoised_latent_mse",
        "recovered",
        "decode_noisy",
    ]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in csv_fields} for row in summaries])

    json_summary = {
        "video": args.video,
        "model_id": args.model_id,
        "model_family": actual_model,
        "frames": len(frames),
        "width": args.width,
        "height": args.height,
        "fps": fps,
        "seed": args.seed,
        "scheduler_steps": args.scheduler_steps,
        **sched_info,
        "transformer_device": str(_pipe_device(pipe)),
        "transformer_dtype": str(_transformer_dtype(pipe)),
        "vae_device": str(_vae_device(pipe)),
        "vae_dtype": str(_vae_dtype(pipe)),
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": max_sequence_length,
        "decode_noisy": args.decode_noisy,
        "video_latent_shape": video_latent_shape,
        "reference_latent_slots": 0,
        "vae_psnr": json_metric(vae_psnr),
        "vae_psnr_per_frame": [json_metric(value) for value in vae_per_frame],
        "runs": [
            {
                key: (
                    [json_metric(value) for value in val]
                    if isinstance(val, list)
                    else json_metric(val)
                    if isinstance(val, float)
                    else val
                )
                for key, val in row.items()
            }
            for row in summaries
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(json_summary, handle, indent=2, ensure_ascii=False)
    print(f"\n[result] outputs written to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数定义。默认值面向双 T4 低显存环境。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vae-device",
        default="auto",
        help="VAE device. auto uses cuda:1 when multiple GPUs are visible, otherwise --device.",
    )
    parser.add_argument(
        "--vae-dtype",
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="VAE dtype. float32 is safest; float16 can reduce memory if needed.",
    )
    parser.add_argument("--max-frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Intuitive noise strengths: 0=no noise, 1=maximum noise.",
    )
    parser.add_argument("--scheduler-steps", type=int, default=100)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Override text sequence length. Default reads pipeline config or uses 512.",
    )
    parser.add_argument(
        "--decode-noisy",
        action="store_true",
        help="Decode/save noisy.mp4 and noisy PSNR. Disabled by default to avoid VAE OOM.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/debug_denoise_moe")
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Disable reduced-CPU-RAM model loading and T5/image encoder release.",
    )
    parser.set_defaults(low_memory=True)
    return parser


if __name__ == "__main__":
    run_debug(build_parser().parse_args())

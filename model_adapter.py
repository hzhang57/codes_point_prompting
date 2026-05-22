"""
Point Prompting 的模型适配层。

支持两种图像条件视频扩散模型：
  - CogVideoX-5B-I2V：图像条件 = VAE 潜变量沿通道轴拼接
  - Wan2.1-I2V 1.3B / 14B：图像条件 = SigLIP/CLIP 嵌入向量（交叉注意力）

两者均基于 Flow Matching，去噪器预测的是速度场（velocity），而非 DDPM 中的噪声。

核心区别
--------
CogVideoX-I2V  ：VAE 编码第 0 帧 → 潜变量 → cat([video, img], dim=1)
                 Transformer 输入通道数翻倍（2C），已在 I2V 版本预训练好。
Wan-I2V        ：SigLIP/CLIP 编码第 0 帧 → image_embeds（交叉注意力）
                 同时将 VAE 编码的图像帧沿时序轴前置于视频潜变量之前。
"""

from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from PIL import Image


# --------------------------------------------------------------------------- #
#  共用张量转换工具                                                             #
# --------------------------------------------------------------------------- #

def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """BGR numpy 数组 → RGB PIL 图像（cv2 与 PIL 的通道顺序相反）。"""
    return Image.fromarray(arr[..., ::-1].copy())


def _frames_to_tensor(frames_bgr: list, device, dtype) -> torch.Tensor:
    """BGR uint8 帧列表 → (1, C, T, H, W) float 张量，值域 [-1, 1]。

    同时完成 BGR→RGB 通道转换和归一化：pixel/127.5 - 1.0
    """
    t = torch.stack([
        torch.from_numpy(f[..., ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        for f in frames_bgr
    ])  # (T, C, H, W)
    # permute(1,0,2,3) → (C, T, H, W)，unsqueeze(0) → (1, C, T, H, W)
    return t.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)


def _tensor_to_frames(tensor: torch.Tensor) -> list:
    """(1, C, T, H, W) float [-1,1] → BGR uint8 帧列表。

    执行逆操作：(val+1)*127.5 clamp→uint8，RGB→BGR。
    """
    t = tensor.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
    out = []
    for i in range(t.shape[0]):
        arr = ((t[i].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5)
        out.append(arr.clip(0, 255).astype(np.uint8)[..., ::-1].copy())  # RGB→BGR
    return out


def _retrieve_latents(encoder_output, generator=None, sample_mode: str = "sample") -> torch.Tensor:
    """兼容 diffusers VAE encode 输出，提取 latent tensor。"""
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        if sample_mode == "mode" and hasattr(encoder_output.latent_dist, "mode"):
            return encoder_output.latent_dist.mode()
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    return encoder_output[0]


def _pipe_device(pipe) -> torch.device:
    """Return the primary compute device of a pipeline.

    Works whether the pipeline was loaded with .to(device),
    enable_model_cpu_offload(), or device_map="auto".
    """
    d = getattr(pipe, "_execution_device", None)
    if d is not None:
        return d
    for attr in ("vae", "transformer", "unet"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            try:
                return next(mod.parameters()).device
            except StopIteration:
                pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _max_memory_per_gpu(reserve_gib: int = 3) -> dict:
    """Build a max_memory dict leaving reserve_gib free on each GPU for VAE decode."""
    n = torch.cuda.device_count()
    return {
        i: f"{max(1, torch.cuda.get_device_properties(i).total_memory // 1024**3 - reserve_gib)}GiB"
        for i in range(n)
    }


# --------------------------------------------------------------------------- #
#  抽象基类                                                                     #
# --------------------------------------------------------------------------- #

class ModelAdapter(ABC):
    """I2V 扩散模型的统一接口，供 Point Prompting 各模块调用。"""

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def scheduler(self): ...

    # -- 编码 / 解码 --------------------------------------------------------- #

    @abstractmethod
    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        """将 BGR 帧列表编码为缩放后的潜变量，返回 (1, C, T, H, W)。"""
        ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> list:
        """(1, C, T, H, W) 潜变量 → BGR uint8 帧列表。"""
        ...

    @abstractmethod
    def encode_image_cond(self, frame_bgr: np.ndarray, video_latent=None) -> Any:
        """将单帧 BGR 图像编码为模型特定的图像条件表示。"""
        ...

    @abstractmethod
    def encode_text(self, prompt: str) -> Any:
        """编码文本提示，返回模型特定的格式。"""
        ...

    # -- 去噪器 -------------------------------------------------------------- #

    @abstractmethod
    def forward_transformer(
        self,
        noisy_latents: torch.Tensor,   # (1, C, T, H, W)
        timestep: torch.Tensor,        # scalar 或 (1,)
        text_cond: Any,
        image_cond: Any,
    ) -> torch.Tensor:
        """单步去噪器前向传播，返回速度场 (1, C, T, H, W)。"""
        ...

    # -- 反事实引导（所有子类共享） ------------------------------------------ #

    @torch.no_grad()
    def predict_with_guidance(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_cond: Any,
        image_cond_edited: Any,
        image_cond_original: Any,
        lam: float = 8.0,
    ) -> torch.Tensor:
        """反事实增强引导（论文公式 3）。

        v̂ = (λ+1) · v(c_edited) - λ · v(c_original)

        用原始帧作为"负向图像提示"，放大标记条件的影响，
        使生成视频中标记更加清晰可见。
        """
        v_e = self.forward_transformer(noisy_latents, timestep, text_cond, image_cond_edited)
        v_o = self.forward_transformer(noisy_latents, timestep, text_cond, image_cond_original)
        return (lam + 1.0) * v_e - lam * v_o

    # -- 调度器封装（所有子类共享） ------------------------------------------ #

    def set_timesteps(self, n_steps: int) -> None:
        """设置去噪调度器的时间步序列。"""
        self.scheduler.set_timesteps(n_steps, device=self.device)

    @property
    def timesteps(self) -> torch.Tensor:
        """返回调度器的时间步张量（从大到小）。"""
        return self.scheduler.timesteps

    def scheduler_step(
        self, velocity: torch.Tensor, t: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        """执行一步调度器去噪，返回 x_{t-1}。"""
        return self.scheduler.step(velocity, t, latents).prev_sample

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Flow Matching SDEdit 前向过程：x_t = (1-γ)·x_0 + γ·ε。"""
        return (1.0 - gamma) * latents + gamma * noise


# --------------------------------------------------------------------------- #
#  CogVideoX-I2V 适配器                                                        #
# --------------------------------------------------------------------------- #

class CogVideoXAdapter(ModelAdapter):
    """CogVideoX-I2V 适配器。

    图像条件方式：VAE 编码第 0 帧 → 潜变量 →
    沿通道轴（dim=1）与视频潜变量拼接 → 输入通道数翻倍。
    CogVideoX-I2V 的 Transformer 已在 doubled in_channels 上预训练。
    """

    def __init__(self, pipe):
        self.pipe = pipe

    @property
    def device(self):
        return _pipe_device(self.pipe)

    @property
    def dtype(self):
        return self.pipe.transformer.dtype

    @property
    def scheduler(self):
        return self.pipe.scheduler

    # -- 编码 / 解码 --------------------------------------------------------- #

    def _video_scale(self) -> float:
        """CogVideoX 视频潜变量缩放因子（与图像条件缩放因子相同）。"""
        return float(getattr(self.pipe, "vae_scale_factor",
                     getattr(self.pipe.vae.config, "scaling_factor", 1.0)))

    def _enable_vae_slicing(self):
        """启用 VAE 时序切片（仅 slicing，不开 tiling）。

        tiling 会改变 VAE 的空间输出尺寸，导致 latent 与 transformer 期望尺寸不符。
        slicing 仅在时序方向分块，不影响空间尺寸，是安全的。
        """
        vae = self.pipe.vae
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        # 不调用 enable_tiling()，保持空间尺寸与模型预期一致

    @staticmethod
    def _gc():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        self._enable_vae_slicing()
        t = _frames_to_tensor(frames_bgr, self.device, self.dtype)
        with torch.no_grad():
            lat = self.pipe.vae.encode(t).latent_dist.sample()
        del t
        self._gc()
        return lat * self._video_scale()

    def decode_latents(self, latents: torch.Tensor) -> list:
        self._enable_vae_slicing()
        self._gc()
        lat = latents / self._video_scale()
        with torch.no_grad():
            decoded = self.pipe.vae.decode(lat).sample
        return _tensor_to_frames(decoded)

    def encode_image_cond(self, frame_bgr: np.ndarray,
                          video_latent: torch.Tensor = None) -> torch.Tensor:
        """返回 (1, C_lat, 1, lH, lW) BCTHW 图像条件潜变量。

        若传入 video_latent，直接切第 0 帧——保证与视频 latent 空间尺寸完全一致，
        避免因 tiling 差异导致单帧编码和视频编码的输出尺寸不同。
        否则单独编码 frame_bgr（用于无对应 video_latent 的场景）。
        """
        if video_latent is not None:
            # 直接复用已有的 video latent 第 0 帧，空间尺寸严格一致
            return video_latent[:, :, :1, :, :].clone()  # (1,C,1,lH,lW)

        self._enable_vae_slicing()
        vae = self.pipe.vae
        img_t = _frames_to_tensor([frame_bgr], self.device, self.dtype)  # (1,C,1,H,W)
        with torch.no_grad():
            lat = vae.encode(img_t).latent_dist.sample()  # (1,C_lat,1,lH,lW) BCTHW
        del img_t
        self._gc()
        scale = getattr(self.pipe, "vae_scaling_factor_image",
                        getattr(vae.config, "scaling_factor", 1.0))
        return lat * scale

    def encode_text(self, prompt: str) -> torch.Tensor:
        """T5 文本编码，返回 (1, seq_len, D) 嵌入张量。"""
        tok = self.pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length, truncation=True,
        ).to(self.device)
        with torch.no_grad():
            emb = self.pipe.text_encoder(**tok).last_hidden_state
        return emb.to(self.dtype)

    # -- 去噪器 -------------------------------------------------------------- #

    def _rotary_emb(self, T: int, H: int, W: int) -> Optional[Any]:
        """计算旋转位置编码（RoPE），若模型不使用则返回 None。"""
        if not getattr(self.pipe.transformer.config, "use_rotary_positional_embeddings", False):
            return None
        if hasattr(self.pipe, "_prepare_rotary_positional_embeddings"):
            p = self.pipe.vae_scale_factor_spatial if hasattr(self.pipe, "vae_scale_factor_spatial") else 8
            return self.pipe._prepare_rotary_positional_embeddings(H * p, W * p, T, self.device)
        return None

    def forward_transformer(self, noisy_latents, timestep, text_cond, image_cond):
        # noisy_latents: (1, C, T, lH, lW) BCTHW（内部统一格式）
        # image_cond:    (1, C, 1, lH, lW) BCTHW
        # CogVideoX transformer 期望 BTCHW 输入，通道轴拼接后再转换格式
        _, C, T, lH, lW = noisy_latents.shape
        img_pad = torch.zeros_like(noisy_latents)                      # (1, C, T, lH, lW)
        img_pad[:, :, :image_cond.shape[2], :, :] = image_cond        # 写入第 0 帧
        # 沿通道轴拼接：(1, 2C, T, lH, lW)，然后转为 BTCHW：(1, T, 2C, lH, lW)
        model_input = torch.cat([noisy_latents, img_pad], dim=1)       # (1, 2C, T, lH, lW)
        model_input = model_input.permute(0, 2, 1, 3, 4)              # (1, T, 2C, lH, lW)

        ipe = self._rotary_emb(T, lH, lW)
        t_b = timestep if timestep.ndim >= 1 else timestep.unsqueeze(0)

        out = self.pipe.transformer(
            hidden_states=model_input,
            encoder_hidden_states=text_cond,
            timestep=t_b,
            image_rotary_emb=ipe,
            return_dict=False,
        )
        # 输出为 BTCHW: (1, T, C, lH, lW)，转回 BCTHW: (1, C, T, lH, lW)
        return out[0].permute(0, 2, 1, 3, 4)


# --------------------------------------------------------------------------- #
#  Wan2.1-I2V 适配器                                                           #
# --------------------------------------------------------------------------- #

class WanAdapter(ModelAdapter):
    """Wan2.1-I2V 适配器。

    图像条件方式（双路）：
      1. SigLIP/CLIP 编码第 0 帧 → image_embeds → Transformer 交叉注意力
      2. VAE 编码第 0 帧 → 图像潜变量 → 前置于视频潜变量时序维之前
    """

    def __init__(self, pipe):
        self.pipe = pipe
        # 前置图像帧的数量（通常为 1）
        self._img_lat_frames: int = 1

    @property
    def device(self):
        return _pipe_device(self.pipe)

    @property
    def dtype(self):
        return self.pipe.transformer.dtype

    @property
    def scheduler(self):
        return self.pipe.scheduler

    # -- 编码 / 解码 --------------------------------------------------------- #

    def _vae_scale(self) -> float:
        """读取 Wan VAE 的缩放因子，不存在时默认为 1.0。"""
        return float(getattr(self.pipe.vae.config, "scaling_factor", 1.0))

    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        t = _frames_to_tensor(frames_bgr, self.device, self.dtype)  # (1,C,T,H,W)
        with torch.no_grad():
            lat = self.pipe.vae.encode(t).latent_dist.sample()
        return lat * self._vae_scale()

    def decode_latents(self, latents: torch.Tensor) -> list:
        # 去掉前置的图像帧潜变量，只解码视频部分
        video_lat = latents[:, :, self._img_lat_frames:, :, :]
        lat = video_lat / self._vae_scale()
        with torch.no_grad():
            decoded = self.pipe.vae.decode(lat).sample
        return _tensor_to_frames(decoded)

    def encode_image_cond(self, frame_bgr: np.ndarray, video_latent=None):
        """双路编码：返回包含 clip_emb 和 vae_latent 的字典。

        clip_emb   : (1, N_tokens, D) — 用于 Transformer 交叉注意力
        vae_latent : (1, C, 1, lH, lW) — 用于前置到视频潜变量时序维
        """
        img_pil = _bgr_to_pil(frame_bgr)

        # ---- 路径1：SigLIP/CLIP 语义嵌入 ----
        feat_inputs = self.pipe.feature_extractor(images=img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_out = self.pipe.image_encoder(**feat_inputs)
        # 兼容不同编码器：部分输出 last_hidden_state，部分输出 image_embeds
        if hasattr(clip_out, "last_hidden_state"):
            clip_emb = clip_out.last_hidden_state.to(self.dtype)
        else:
            clip_emb = clip_out.image_embeds.unsqueeze(1).to(self.dtype)

        # ---- 路径2：VAE 空间潜变量（T=1 单帧）----
        if hasattr(self.pipe, "video_processor"):
            img_t = self.pipe.video_processor.preprocess(img_pil)
        else:
            arr   = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
            img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.unsqueeze(2).to(device=self.device, dtype=self.dtype)  # (1,C,1,H,W)
        with torch.no_grad():
            vae_lat = self.pipe.vae.encode(img_t).latent_dist.sample()
        vae_lat = vae_lat * self._vae_scale()

        return {"clip_emb": clip_emb, "vae_latent": vae_lat}

    def encode_text(self, prompt: str):
        """UMT5 文本编码，返回包含 embeds 和 attention_mask 的字典。"""
        tok = self.pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length, truncation=True,
        ).to(self.device)
        with torch.no_grad():
            enc_out = self.pipe.text_encoder(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
            )
        return {
            "embeds": enc_out.last_hidden_state.to(self.dtype),
            "mask":   tok.attention_mask.to(self.device),
        }

    # -- 去噪器 -------------------------------------------------------------- #

    def _prepend_image_latent(
        self, noisy_latents: torch.Tensor, image_cond: dict
    ) -> torch.Tensor:
        """将图像帧潜变量前置到视频潜变量的时序维，返回 (1, C, 1+T, lH, lW)。

        若空间分辨率不匹配（帧被缩放过），先做双线性插值对齐。
        """
        img_lat = image_cond["vae_latent"]  # (1, C, 1, lH_img, lW_img)
        if img_lat.shape[3:] != noisy_latents.shape[3:]:
            # 空间尺寸不一致时，将图像潜变量插值到视频潜变量的分辨率
            img_lat = torch.nn.functional.interpolate(
                img_lat.squeeze(2), size=noisy_latents.shape[3:],
                mode="bilinear", align_corners=False
            ).unsqueeze(2)
        return torch.cat([img_lat, noisy_latents], dim=2)  # 沿时序轴前置

    def forward_transformer(self, noisy_latents, timestep, text_cond, image_cond):
        # 将图像帧潜变量作为第 0 帧前置，使 Transformer 在时序上感知图像条件
        lat_input = self._prepend_image_latent(noisy_latents, image_cond)

        t_b = timestep if timestep.ndim >= 1 else timestep.unsqueeze(0)

        out = self.pipe.transformer(
            hidden_states=lat_input,               # (1, C, 1+T, lH, lW)
            timestep=t_b,
            encoder_hidden_states=text_cond["embeds"],
            encoder_attention_mask=text_cond["mask"],
            image_embeds=image_cond["clip_emb"],   # CLIP 嵌入通过交叉注意力注入
            return_dict=False,
        )
        vel_full = out[0]  # (1, C, 1+T, lH, lW)

        # 只返回视频帧部分的速度场，去掉前置图像帧对应的输出
        return vel_full[:, :, self._img_lat_frames:, :, :]

    def encode_video_with_image(self, frames_bgr: list, image_cond: dict) -> torch.Tensor:
        """编码视频并前置图像帧潜变量，得到 Wan 完整输入序列。"""
        video_lat = self.encode_video(frames_bgr)
        return self._prepend_image_latent(video_lat, image_cond)


# --------------------------------------------------------------------------- #
#  Wan2.1-VACE 适配器                                                         #
# --------------------------------------------------------------------------- #

class WanVACEAdapter(ModelAdapter):
    """Wan2.1-VACE 适配器。

    VACE 不接收 I2V 的 CLIP image_embeds；它通过
    control_hidden_states = cat(mask_latents, condition_latents) 接收条件。
    这里将第 0 帧构造成条件帧，mask 中第 0 帧为黑色（保留/条件），
    后续帧为白色（生成），从而近似 I2V 的首帧条件用法。
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self._last_num_frames: Optional[int] = None
        self._last_height: Optional[int] = None
        self._last_width: Optional[int] = None
        self._last_latent_frames: Optional[int] = None

    @property
    def device(self):
        return _pipe_device(self.pipe)

    @property
    def dtype(self):
        return self.pipe.transformer.dtype

    @property
    def scheduler(self):
        return self.pipe.scheduler

    def _vae_scale(self, dtype, device) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        config = getattr(self.pipe.vae, "config", None)
        latents_mean = getattr(config, "latents_mean", None)
        latents_std = getattr(config, "latents_std", None)
        if latents_mean is not None and latents_std is not None:
            mean = torch.tensor(latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
            std = 1.0 / torch.tensor(latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
            return mean, std
        scale = float(getattr(config, "scaling_factor", 1.0))
        return None, torch.tensor(scale, device=device, dtype=dtype)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, scale = self._vae_scale(latents.dtype, latents.device)
        if mean is None:
            return latents * scale
        return (latents - mean) * scale

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, scale = self._vae_scale(latents.dtype, latents.device)
        if mean is None:
            return latents / scale
        return latents / scale + mean

    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        self._last_num_frames = len(frames_bgr)
        self._last_height, self._last_width = frames_bgr[0].shape[:2]
        t = _frames_to_tensor(frames_bgr, self.device, self.dtype)
        with torch.no_grad():
            lat = _retrieve_latents(self.pipe.vae.encode(t), sample_mode="mode")
        lat = self._normalize_latents(lat.to(device=self.device, dtype=self.dtype))
        self._last_latent_frames = lat.shape[2]
        return lat

    def decode_latents(self, latents: torch.Tensor) -> list:
        lat = self._denormalize_latents(latents)
        with torch.no_grad():
            decoded = self.pipe.vae.decode(lat).sample
        if decoded.ndim == 5 and decoded.shape[1] != 3 and decoded.shape[2] == 3:
            decoded = decoded.permute(0, 2, 1, 3, 4)
        return _tensor_to_frames(decoded)

    def _vace_in_channels(self) -> int:
        cfg = getattr(self.pipe.transformer, "config", None)
        return int(getattr(cfg, "vace_in_channels", 0) or 0)

    def _pad_control(self, ctrl: torch.Tensor) -> torch.Tensor:
        """Zero-pad the channel dimension to vace_in_channels if needed."""
        target = self._vace_in_channels()
        if target and ctrl.shape[1] < target:
            pad = torch.zeros(
                ctrl.shape[0], target - ctrl.shape[1], *ctrl.shape[2:],
                device=ctrl.device, dtype=ctrl.dtype,
            )
            ctrl = torch.cat([ctrl, pad], dim=1)
        return ctrl

    def _fallback_control_cond(self, frame_bgr: np.ndarray) -> dict:
        num_frames = self._last_latent_frames or self._last_num_frames or 1
        t = _frames_to_tensor([frame_bgr], self.device, self.dtype)
        with torch.no_grad():
            first_lat = _retrieve_latents(self.pipe.vae.encode(t), sample_mode="mode")
        first_lat = self._normalize_latents(first_lat.to(device=self.device, dtype=self.dtype))
        cond_lat = torch.zeros(
            first_lat.shape[0], first_lat.shape[1], num_frames, first_lat.shape[3], first_lat.shape[4],
            device=first_lat.device, dtype=first_lat.dtype,
        )
        cond_lat[:, :, 0:1] = first_lat
        mask = torch.ones(
            cond_lat.shape[0], 1, cond_lat.shape[2], cond_lat.shape[3], cond_lat.shape[4],
            device=cond_lat.device, dtype=cond_lat.dtype,
        )
        mask[:, :, 0] = 0.0
        ctrl = self._pad_control(torch.cat([cond_lat, mask], dim=1))
        return {"control": ctrl, "scale": 1.0}

    def encode_image_cond(self, frame_bgr: np.ndarray, video_latent=None):
        """构造 VACE control_hidden_states 条件。"""
        # 官方 prepare_video_latents 会对整段 control video 再跑一次 Wan VAE。
        # 在 14-16GB GPU 上这一步容易 OOM；这里只编码第 0 帧并构造轻量 VACE control。
        if getattr(self, "use_pipeline_conditioning", False):
            return self._pipeline_control_cond(frame_bgr)
        return self._fallback_control_cond(frame_bgr)

    def _pipeline_control_cond(self, frame_bgr: np.ndarray):
        num_frames = self._last_num_frames or 1
        height = self._last_height or frame_bgr.shape[0]
        width = self._last_width or frame_bgr.shape[1]
        first = _bgr_to_pil(frame_bgr)
        gray = Image.new("RGB", (width, height), (127, 127, 127))
        video = [first] + [gray] * max(0, num_frames - 1)

        mask_first = Image.new("L", (width, height), 0)
        mask_generate = Image.new("L", (width, height), 255)
        mask = [mask_first] + [mask_generate] * max(0, num_frames - 1)

        condition, mask, reference_images = self.pipe.preprocess_conditions(
            video=video,
            mask=mask,
            reference_images=None,
            batch_size=1,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
        )
        condition_latents = self.pipe.prepare_video_latents(
            condition,
            mask,
            reference_images,
            None,
            self.device,
        )
        vace_ch = self._vace_in_channels()
        if vace_ch and condition_latents.shape[1] >= vace_ch:
            ctrl = condition_latents[:, :vace_ch]
        else:
            mask_latents = self.pipe.prepare_masks(mask, reference_images, None)
            ctrl = torch.cat([condition_latents, mask_latents], dim=1)
            ctrl = self._pad_control(ctrl)
        return {"control": ctrl.to(self.dtype), "scale": 1.0}

    def encode_text(self, prompt: str):
        if getattr(self.pipe, "text_encoder", None) is None or getattr(self.pipe, "tokenizer", None) is None:
            if prompt:
                raise ValueError(
                    "Wan VACE was loaded without a text encoder to avoid the incomplete UMT5 checkpoint. "
                    "Use an empty prompt, or load a complete text encoder before using non-empty prompts."
                )
            text_dim = int(getattr(getattr(self.pipe.transformer, "config", None), "text_dim", 4096))
            seq_len = 512
            return {
                "embeds": torch.zeros(1, seq_len, text_dim, device=self.device, dtype=self.dtype),
                "mask": torch.zeros(1, seq_len, device=self.device, dtype=torch.long),
            }

        tok = self.pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            max_length=min(getattr(self.pipe.tokenizer, "model_max_length", 512), 512), truncation=True,
        ).to(self.device)
        with torch.no_grad():
            enc_out = self.pipe.text_encoder(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
            )
        return {"embeds": enc_out.last_hidden_state.to(self.dtype), "mask": tok.attention_mask.to(self.device)}

    def forward_transformer(self, noisy_latents, timestep, text_cond, image_cond):
        t_b = timestep if timestep.ndim >= 1 else timestep.unsqueeze(0)
        control_scale = image_cond.get("scale", 1.0)
        if not torch.is_tensor(control_scale):
            control_scale = torch.tensor([control_scale], device=noisy_latents.device, dtype=noisy_latents.dtype)
        else:
            control_scale = control_scale.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
            if control_scale.ndim == 0:
                control_scale = control_scale.unsqueeze(0)
        transformer = self.pipe.transformer
        num_vace = (
            len(getattr(transformer, "vace_blocks", None) or [])
            or len(getattr(transformer, "vace_layers", None) or [])
            or getattr(getattr(transformer, "config", None), "vace_num_layers", 0)
            or (getattr(getattr(transformer, "config", None), "num_layers", 0) // 2)
            or 1
        )
        if control_scale.numel() == 1:
            control_scale = control_scale.expand(num_vace).contiguous()
        kwargs = {
            "hidden_states": noisy_latents,
            "timestep": t_b,
            "encoder_hidden_states": text_cond["embeds"],
            "control_hidden_states": image_cond["control"],
            "control_hidden_states_scale": control_scale,
            "return_dict": False,
        }
        import inspect
        sig = inspect.signature(self.pipe.transformer.forward)
        if "encoder_attention_mask" in sig.parameters:
            kwargs["encoder_attention_mask"] = text_cond["mask"]
        out = self.pipe.transformer(**kwargs)
        return out[0]


# --------------------------------------------------------------------------- #
#  工厂函数                                                                     #
# --------------------------------------------------------------------------- #

_COGVIDEOX_CLASSES = {"CogVideoXImageToVideoPipeline"}
_WAN_CLASSES       = {"WanImageToVideoPipeline"}
_WAN_VACE_CLASSES  = {"WanVACEPipeline"}
_WAN_I2V_MODEL_HINTS = ("I2V", "Image-to-Video", "ImageToVideo")
_WAN_VACE_MODEL_HINTS = ("VACE",)
_WAN_UNSUPPORTED_MODEL_HINTS = ("T2V", "Text-to-Video")


def create_adapter(pipe) -> ModelAdapter:
    """根据 pipeline 类型自动创建对应的 ModelAdapter。

    检测顺序：
      1. 类名精确匹配（CogVideoXImageToVideoPipeline / WanImageToVideoPipeline）
      2. 启发式推断：Wan Transformer 的 forward 签名包含 image_embeds 和
         encoder_attention_mask；CogVideoX Transformer 则不含这两个参数。
    """
    cls_name = type(pipe).__name__
    if cls_name in _COGVIDEOX_CLASSES:
        return CogVideoXAdapter(pipe)
    if cls_name in _WAN_VACE_CLASSES or "VACE" in cls_name:
        return WanVACEAdapter(pipe)
    if cls_name in _WAN_CLASSES:
        return WanAdapter(pipe)

    # 启发式回退：检查 Transformer forward 签名
    import inspect
    sig = inspect.signature(pipe.transformer.forward)
    if "control_hidden_states" in sig.parameters:
        return WanVACEAdapter(pipe)
    if "image_embeds" in sig.parameters and "encoder_attention_mask" in sig.parameters:
        return WanAdapter(pipe)    # Wan 特有参数
    return CogVideoXAdapter(pipe)  # 默认为 CogVideoX


def load_cogvideox_pipe(model_id: str = "THUDM/CogVideoX-5b-I2V", device: str = "cuda"):
    """加载 CogVideoX-5B I2V pipeline（float16）。"""
    from diffusers import CogVideoXImageToVideoPipeline
    n_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    if n_gpus >= 2:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="balanced", max_memory=_max_memory_per_gpu(),
        )
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if str(device).startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    # slicing 仅做时序分块，不改变空间尺寸；tiling 会改变输出尺寸，不启用
    pipe.vae.enable_slicing()
    return pipe


def load_wan_pipe(model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P", device: str = "cuda"):
    """加载 Wan2.1 I2V 或 VACE pipeline。

    14B 模型使用 bfloat16（精度更高），1.3B 模型使用 float16（速度更快）。
    """
    upper_model_id = model_id.upper()
    if any(hint in upper_model_id for hint in _WAN_UNSUPPORTED_MODEL_HINTS):
        raise ValueError(
            f"Unsupported Wan model for this adapter: {model_id}. "
            "This code expects a Wan2.1 I2V/VACE checkpoint, for example "
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers or "
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers. "
            "T2V checkpoints do not provide the required image/video conditioning."
        )
    is_vace = any(hint in upper_model_id for hint in _WAN_VACE_MODEL_HINTS)
    is_i2v = any(hint.upper() in upper_model_id for hint in _WAN_I2V_MODEL_HINTS)
    if not (is_vace or is_i2v):
        raise ValueError(
            f"Unsupported Wan model for this adapter: {model_id}. "
            "Please pass a Wan2.1 I2V or VACE model id."
        )

    dtype = torch.bfloat16 if "14B" in model_id else torch.float16
    if is_vace:
        try:
            from diffusers import WanVACEPipeline
        except ImportError as exc:
            raise ImportError(
                "Your installed diffusers package does not expose WanVACEPipeline. "
                "Upgrade diffusers/transformers/accelerate, then retry the VACE checkpoint."
            ) from exc
        pipeline_cls = WanVACEPipeline
        pipe_kwargs = dict(torch_dtype=dtype, text_encoder=None, tokenizer=None)
    else:
        from diffusers import WanImageToVideoPipeline
        pipeline_cls = WanImageToVideoPipeline
        pipe_kwargs = dict(torch_dtype=dtype)

    n_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    if n_gpus >= 2:
        pipe = pipeline_cls.from_pretrained(
            model_id, **pipe_kwargs,
            device_map="balanced", max_memory=_max_memory_per_gpu(),
        )
    else:
        pipe = pipeline_cls.from_pretrained(model_id, **pipe_kwargs)
        if str(device).startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    return pipe

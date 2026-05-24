"""
Point Prompting 的模型适配层。

支持 CogVideoX-5B-I2V：图像条件 = VAE 潜变量沿通道轴拼接，
Transformer 输入通道数翻倍（2C），已在 I2V 版本预训练好。
"""

from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
from PIL import Image


# --------------------------------------------------------------------------- #
#  共用张量转换工具                                                             #
# --------------------------------------------------------------------------- #

def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """BGR numpy 数组 → RGB PIL 图像（cv2 与 PIL 的通道顺序相反）。"""
    return Image.fromarray(arr[..., ::-1].copy())


def _frames_to_tensor(frames_bgr: list, device, dtype) -> torch.Tensor:
    """BGR uint8 帧列表 → (1, C, T, H, W) float 张量，值域 [-1, 1]。"""
    t = torch.stack([
        torch.from_numpy(f[..., ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        for f in frames_bgr
    ])  # (T, C, H, W)
    return t.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)


def _tensor_to_frames(tensor: torch.Tensor) -> list:
    """(1, C, T, H, W) float [-1,1] → BGR uint8 帧列表。"""
    t = tensor.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
    out = []
    for i in range(t.shape[0]):
        arr = ((t[i].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5)
        out.append(arr.clip(0, 255).astype(np.uint8)[..., ::-1].copy())  # RGB→BGR
    return out


def _pipe_device(pipe) -> torch.device:
    """Return the primary compute device of a pipeline.

    Skips VAE — it may be intentionally pinned to a different device.
    """
    d = getattr(pipe, "_execution_device", None)
    if d is not None:
        return d
    for attr in ("transformer", "unet"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            try:
                return next(mod.parameters()).device
            except StopIteration:
                pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    @abstractmethod
    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        """将 BGR 帧列表编码为缩放后的潜变量，返回 (1, C, T, H, W)。"""
        ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> list:
        """(1, C, T, H, W) 潜变量 → BGR uint8 帧列表。"""
        ...

    @abstractmethod
    def encode_image_cond(self, frame_bgr: np.ndarray,
                          video_latent: Optional[torch.Tensor] = None) -> Any:
        """将单帧 BGR 图像编码为模型特定的图像条件表示。

        若传入 video_latent，实现可直接切第 0 帧以保证尺寸严格一致。
        """
        ...

    @abstractmethod
    def encode_text(self, prompt: str) -> Any:
        """编码文本提示，返回模型特定的格式。"""
        ...

    @abstractmethod
    def forward_transformer(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_cond: Any,
        image_cond: Any,
    ) -> torch.Tensor:
        """单步去噪器前向传播，返回速度场 (1, C, T, H, W)。"""
        ...

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
        """
        v_e = self.forward_transformer(noisy_latents, timestep, text_cond, image_cond_edited)
        v_o = self.forward_transformer(noisy_latents, timestep, text_cond, image_cond_original)
        return (lam + 1.0) * v_e - lam * v_o

    def set_timesteps(self, n_steps: int) -> None:
        self.scheduler.set_timesteps(n_steps, device=self.device)

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps

    def scheduler_step(
        self,
        velocity: torch.Tensor,
        t: torch.Tensor,
        latents: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._dpm_step is None:
            import inspect
            params = inspect.signature(self.scheduler.step).parameters
            self._dpm_step = "timestep_back" in params
        if self._dpm_step:
            t_back = t_next if t_next is not None else torch.zeros_like(t)
            return self.scheduler.step(velocity, t, latents, timestep_back=t_back).prev_sample
        return self.scheduler.step(velocity, t, latents).prev_sample


# --------------------------------------------------------------------------- #
#  CogVideoX-I2V 适配器                                                        #
# --------------------------------------------------------------------------- #

class CogVideoXAdapter(ModelAdapter):
    """CogVideoX-I2V 适配器。

    图像条件方式：VAE 编码第 0 帧 → 潜变量 →
    沿通道轴（dim=1）与视频潜变量拼接 → 输入通道数翻倍。
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self._dpm_step: Optional[bool] = None  # cached after first scheduler_step call

    @property
    def device(self):
        return _pipe_device(self.pipe)

    @property
    def dtype(self):
        return self.pipe.transformer.dtype

    @property
    def scheduler(self):
        return self.pipe.scheduler

    def _video_scale(self) -> float:
        # [DEBUG] 测试 scale=1.0，验证 VAE encode/decode 不需要手动 scale
        return 1.0

    @property
    def _vae_device(self) -> torch.device:
        return next(self.pipe.vae.parameters()).device

    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        vae_dev = self._vae_device
        # _frames_to_tensor → (1, C, T, H, W); VAE encode 期望 (1, T, C, H, W)
        t = _frames_to_tensor(frames_bgr, vae_dev, self.dtype)  # (1, C, T, H, W) — VAE 期望 BCTHW
        T_in = t.shape[2]
        lT_expected = (T_in - 1) // 4
        print(f"[DEBUG] encode_video input: shape={t.shape} min={t.min():.3f} max={t.max():.3f} "
              f"(T={T_in} → expect lT={(T_in-1)//4}, need T=4k+1 e.g. {lT_expected*4+1})")
        with torch.no_grad():
            dist = self.pipe.vae.encode(t).latent_dist
            lat_mean = dist.mean  # (1, C_lat, lT, lH, lW)
        del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[DEBUG] encode_video: lat shape={lat_mean.shape} min={lat_mean.min():.3f} max={lat_mean.max():.3f}")
        return (lat_mean * self._video_scale()).to(device=self.device, dtype=self.dtype)

    def decode_latents(self, latents: torch.Tensor) -> list:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vae_dev = self._vae_device
        lat = (latents / self._video_scale()).to(vae_dev, dtype=self.dtype)  # (1, C, T, H, W) BCTHW
        with torch.no_grad():
            decoded = self.pipe.vae.decode(lat).sample  # (1, C, T, H, W)
        print(f"[DEBUG] decode_latents: decoded shape={decoded.shape} min={decoded.min():.3f} max={decoded.max():.3f}")
        return _tensor_to_frames(decoded)

    def encode_image_cond(self, frame_bgr: np.ndarray,
                          video_latent: torch.Tensor = None) -> torch.Tensor:
        """用 VAE 编码单帧，返回 (1, C_lat, 1, lH, lW) BCTHW 图像潜变量。

        若传入 video_latent，直接切第 0 帧，保证空间尺寸严格一致。
        """
        if video_latent is not None:
            return video_latent[:, :, :1, :, :].clone()

        vae_dev = self._vae_device
        img_t = _frames_to_tensor([frame_bgr], vae_dev, self.dtype)  # (1, C, 1, H, W) BCTHW
        with torch.no_grad():
            lat = self.pipe.vae.encode(img_t).latent_dist.mean  # (1, C_lat, lT, lH, lW)
        return (lat * self._video_scale()).to(device=self.device, dtype=self.dtype)

    def encode_text(self, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length, truncation=True,
        ).to(self.device)
        with torch.no_grad():
            emb = self.pipe.text_encoder(**tok).last_hidden_state
        return emb.to(self.dtype)

    def _rotary_emb(self, T: int, H: int, W: int) -> Optional[Any]:
        if not getattr(self.pipe.transformer.config, "use_rotary_positional_embeddings", False):
            return None
        if hasattr(self.pipe, "_prepare_rotary_positional_embeddings"):
            p = self.pipe.vae_scale_factor_spatial if hasattr(self.pipe, "vae_scale_factor_spatial") else 8
            return self.pipe._prepare_rotary_positional_embeddings(H * p, W * p, T, self.device)
        return None

    def _ofs_tensor(self, T: int) -> Optional[torch.Tensor]:
        """CogVideoX 1.5 新增的 ofs（output frame scale）嵌入。"""
        if not hasattr(self.pipe.transformer, "ofs_proj"):
            return None
        return torch.tensor([T - 1], device=self.device, dtype=self.dtype)

    def forward_transformer(self, noisy_latents, timestep, text_cond, image_cond):
        _, C, T, lH, lW = noisy_latents.shape
        img_pad = torch.zeros_like(noisy_latents)
        img_pad[:, :, :image_cond.shape[2], :, :] = image_cond
        model_input = torch.cat([noisy_latents, img_pad], dim=1).permute(0, 2, 1, 3, 4)  # (1,T,2C,lH,lW)

        ipe = self._rotary_emb(T, lH, lW)
        ofs = self._ofs_tensor(T)
        t_b = timestep if timestep.ndim >= 1 else timestep.unsqueeze(0)

        kwargs = dict(
            hidden_states=model_input,
            encoder_hidden_states=text_cond,
            timestep=t_b,
            image_rotary_emb=ipe,
            return_dict=False,
        )
        if ofs is not None:
            kwargs["ofs"] = ofs

        out = self.pipe.transformer(**kwargs)
        return out[0].permute(0, 2, 1, 3, 4)  # BTCHW → BCTHW


# --------------------------------------------------------------------------- #
#  工厂函数                                                                     #
# --------------------------------------------------------------------------- #

def create_adapter(pipe) -> ModelAdapter:
    """根据 pipeline 类型自动创建 CogVideoXAdapter。"""
    return CogVideoXAdapter(pipe)


def load_cogvideox_pipe(model_id: str = "THUDM/CogVideoX-5b-I2V", device: str = "cuda"):
    """加载 CogVideoX I2V pipeline（float16）。

    双卡策略（方案 A）：
      - cuda:0 限制 total-5 GiB：为 VAE 权重(0.4G) + encode 激活(~1G) 留出空间
      - cuda:1 限制 total-1 GiB：放 transformer 后半 + T5 text encoder
      - 加载完成后摘除 VAE 的 accelerate hooks，整体固定到 cuda:0 fp16
    """
    import os
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDDIMScheduler
    os.environ["TQDM_DISABLE"] = "1"
    n_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    if n_gpus >= 2:
        total_0 = torch.cuda.get_device_properties(0).total_memory // 1024**3
        total_1 = torch.cuda.get_device_properties(1).total_memory // 1024**3
        mem = {
            0: f"{max(1, total_0 - 5)}GiB",
            1: f"{max(1, total_1 - 1)}GiB",
        }
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="balanced", max_memory=mem,
        )
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(pipe.vae, recurse=True)
        pipe.vae.to("cuda:0", dtype=torch.float16)
        vae_dev = next(pipe.vae.parameters()).device
        f0 = torch.cuda.mem_get_info(0)[0] / 1024**3
        f1 = torch.cuda.mem_get_info(1)[0] / 1024**3
        print(f"[load] VAE device: {vae_dev}")
        print(f"[load] GPU 0 free: {f0:.1f} GiB / {total_0} GiB")
        print(f"[load] GPU 1 free: {f1:.1f} GiB / {total_1} GiB")
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if str(device).startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    os.environ.pop("TQDM_DISABLE", None)
    return pipe

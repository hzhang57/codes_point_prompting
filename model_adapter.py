"""
Point Prompting 的模型适配层。

支持后端：
- Wan2.1-VACE-1.3B：图像条件 = control_hidden_states（VAE latent + mask 注入 VACE 层）
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
        n_frames_px: int = 9,
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
        n_frames_px: int = 9,
    ) -> torch.Tensor:
        """反事实增强引导（论文公式 3）。

        v̂ = (λ+1) · v(c_edited) - λ · v(c_original)
        """
        v_e = self.forward_transformer(noisy_latents, timestep, text_cond, image_cond_edited, n_frames_px)
        v_o = self.forward_transformer(noisy_latents, timestep, text_cond, image_cond_original, n_frames_px)
        return (lam + 1.0) * v_e - lam * v_o

    def set_timesteps(self, n_steps: int) -> None:
        self.scheduler.set_timesteps(n_steps, device=self.device)

    def prepare_denoise_start(self, n_steps: int, start_idx: int) -> torch.Tensor:
        self.set_timesteps(n_steps)
        timesteps = self.timesteps
        start_idx = max(0, min(int(start_idx), len(timesteps) - 1))
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(start_idx)
        return timesteps[start_idx:]

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps

    def add_noise_at_timestep(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.scheduler, "add_noise"):
            raise AttributeError(f"{type(self.scheduler).__name__} does not implement add_noise()")
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        return self.scheduler.add_noise(latents, noise, timestep.to(device=latents.device))

    def scheduler_step(
        self,
        velocity: torch.Tensor,
        t: torch.Tensor,
        latents: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(self, "_dpm_step", None) is None:
            import inspect
            params = inspect.signature(self.scheduler.step).parameters
            self._dpm_step = "timestep_back" in params
        if self._dpm_step:
            t_back = t_next if t_next is not None else torch.zeros_like(t)
            return self.scheduler.step(velocity, t, latents, timestep_back=t_back).prev_sample
        return self.scheduler.step(velocity, t, latents).prev_sample


# --------------------------------------------------------------------------- #
#  Wan2.1-VACE 适配器                                                          #
# --------------------------------------------------------------------------- #

class WanVACEAdapter(ModelAdapter):
    """Wan2.1-VACE-1.3B 适配器。对齐官方 pipeline_wan_vace.py 的实现。

    VAE 标准化：(latent - latents_mean) * latents_std（官方做法，非 scaling_factor）
    control_hidden_states：inactive(C) + reactive(C) + mask_patches(64) = 2C+64 ch
    mask 构建：像素空间 mask 经 spatial patch flatten（8×8 → 64ch）得到
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self._dpm_step: Optional[bool] = None
        # 缓存 VAE 标准化参数（与官方 pipeline 对齐）
        self._latents_mean: Optional[torch.Tensor] = None
        self._latents_std: Optional[torch.Tensor] = None

    def _get_vae_norm(self, device):
        """获取 VAE 标准化参数（惰性初始化，缓存复用）。"""
        if self._latents_mean is None:
            mean = torch.tensor(
                self.pipe.vae.config.latents_mean, dtype=torch.float32
            ).view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            std = 1.0 / torch.tensor(
                self.pipe.vae.config.latents_std, dtype=torch.float32
            ).view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            self._latents_mean = mean
            self._latents_std  = std
        return (
            self._latents_mean.to(device=device, dtype=torch.float32),
            self._latents_std.to(device=device, dtype=torch.float32),
        )

    def _normalize(self, lat: torch.Tensor) -> torch.Tensor:
        """官方标准化：(lat - mean) * std，在 float32 计算后转回原 dtype。"""
        mean, std = self._get_vae_norm(lat.device)
        return ((lat.float() - mean) * std).to(lat.dtype)

    def _denormalize(self, lat: torch.Tensor) -> torch.Tensor:
        """逆标准化：lat / std + mean。"""
        mean, std = self._get_vae_norm(lat.device)
        return (lat.float() / std + mean).to(lat.dtype)

    @property
    def device(self) -> torch.device:
        return _pipe_device(self.pipe)

    @property
    def dtype(self) -> torch.dtype:
        return self.pipe.transformer.dtype

    @property
    def _vae_dtype(self) -> torch.dtype:
        return next(self.pipe.vae.parameters()).dtype

    @property
    def scheduler(self):
        return self.pipe.scheduler

    @property
    def _vae_device(self) -> torch.device:
        return next(self.pipe.vae.parameters()).device

    def _video_scale(self) -> float:
        return 1.0  # 标准化由 _normalize/_denormalize 处理，此处不再用 scaling_factor

    def encode_video(self, frames_bgr: list) -> torch.Tensor:
        vae_dev = self._vae_device
        t = _frames_to_tensor(frames_bgr, vae_dev, self._vae_dtype)
        T_in = t.shape[2]
        print(f"[DEBUG] encode_video input: shape={t.shape} (T={T_in} → expect lT={(T_in-1)//4 + 1})")
        with torch.no_grad():
            lat = self.pipe.vae.encode(t).latent_dist.mean
        del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lat = self._normalize(lat)
        print(f"[DEBUG] encode_video: lat shape={lat.shape}")
        return lat.to(device=self.device, dtype=self.dtype)

    def decode_latents(self, latents: torch.Tensor) -> list:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vae_dev = self._vae_device
        lat = self._denormalize(latents).to(vae_dev, dtype=self._vae_dtype)
        with torch.no_grad():
            decoded = self.pipe.vae.decode(lat).sample
        print(f"[DEBUG] decode_latents: decoded shape={decoded.shape}")
        return _tensor_to_frames(decoded)

    def encode_image_cond(self, frame_bgr: np.ndarray,
                          video_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回 (1, C_lat, 1, lH, lW) 第 0 帧的单帧 latent（已标准化）。

        若传入 video_latent，直接切第 0 帧，保证空间尺寸严格一致。
        """
        if video_latent is not None:
            return video_latent[:, :, :1, :, :].clone()
        vae_dev = self._vae_device
        img_t = _frames_to_tensor([frame_bgr], vae_dev, self._vae_dtype)
        with torch.no_grad():
            lat = self.pipe.vae.encode(img_t).latent_dist.mean
        return self._normalize(lat).to(device=self.device, dtype=self.dtype)

    def encode_text(self, prompt: str) -> torch.Tensor:
        """用 T5 编码文本，返回 (1, seq_len, 4096)。空字符串返回 EOS embedding。"""
        # 临时把 T5 移到 GPU 编码，完成后移回 CPU 释放显存
        t5_device = self.device
        self.pipe.text_encoder.to(t5_device)
        with torch.no_grad():
            embeds = self.pipe._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=1,
                max_sequence_length=226,
                device=t5_device,
            )
        self.pipe.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeds

    def _build_control(
        self,
        noisy_latents: torch.Tensor,
        image_cond: torch.Tensor,
        n_frames_px: int = 9,
    ) -> torch.Tensor:
        """构建 VACE control_hidden_states，严格对齐官方 prepare_masks。

        官方 prepare_masks 输入是像素空间原始帧数 T_px（如9），
        不是 lT*4 的近似值。patch_size[1]=2，所以：
          new_height = H_px // (vae_s * patch_size) * patch_size = lH // patch_size * patch_size
          view(T_px, new_height, vae_s, new_width, vae_s) → permute(2,4,0,1,3) → flatten(0,1)
          → interpolate(nearest-exact) to (new_T, new_H, new_W)
        """
        _, C, lT, lH, lW = noisy_latents.shape
        dev, dt = self.device, self.dtype

        # --- video control (2C ch) ---
        inactive = torch.zeros(1, C, lT, lH, lW, device=dev, dtype=dt)
        inactive[:, :, :1, :, :] = image_cond
        reactive = torch.zeros_like(inactive)
        video_ctrl = torch.cat([inactive, reactive], dim=1)  # (1, 2C, lT, lH, lW)

        # --- mask patches（严格对齐官方 prepare_masks）---
        vae_s = 8          # vae_scale_factor_spatial
        p = 2              # transformer patch_size[1] = patch_size[2] = 2
        # 官方用 new_height = H_px // (vae_s * p) * p，等价于 lH // p * p
        new_H = lH // p * p
        new_W = lW // p * p
        # 像素空间 mask：(T_px, H_px, W_px)，第 0 帧=0，其余=1
        H_px, W_px = new_H * vae_s, new_W * vae_s
        mask_px = torch.ones(n_frames_px, H_px, W_px, device=dev, dtype=dt)
        mask_px[0] = 0.0  # 第 0 帧已知
        # spatial patch flatten：与官方完全相同
        mask_px = mask_px.view(n_frames_px, new_H, vae_s, new_W, vae_s)
        mask_px = mask_px.permute(2, 4, 0, 1, 3).flatten(0, 1)  # (64, T_px, new_H, new_W)
        # nearest-exact 在 9→3 时把第0帧映射到输入帧1（坐标系偏移），
        # nearest 才能正确保留第0帧的零值（floor(0 * 9/3) = 0）
        mask_patches = torch.nn.functional.interpolate(
            mask_px.unsqueeze(0), size=(lT, new_H, new_W), mode="nearest"
        )  # (1, 64, lT, new_H, new_W)

        return torch.cat([video_ctrl, mask_patches], dim=1)  # (1, 2C+64, lT, lH, lW)

    def forward_transformer(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_cond: torch.Tensor,
        image_cond: torch.Tensor,
        n_frames_px: int = 9,
    ) -> torch.Tensor:
        control_hidden_states = self._build_control(noisy_latents, image_cond, n_frames_px)

        t_b = timestep if timestep.ndim >= 1 else timestep.unsqueeze(0)

        # WanTransformer3DModel forward：BCTHW 输入，不需要 permute
        # encoder_hidden_states 是必填项；text_cond=None 时用全零占位
        # Wan-1.3B text_dim=512，seq_len 取 transformer config 或默认 512
        if text_cond is None:
            text_dim = getattr(self.pipe.transformer.config, "text_dim", 4096)
            seq_len  = getattr(self.pipe.transformer.config, "max_text_seq_len", 226)
            text_cond = torch.zeros(1, seq_len, text_dim,
                                    device=self.device, dtype=self.dtype)
        out = self.pipe.transformer(
            hidden_states=noisy_latents,
            timestep=t_b,
            encoder_hidden_states=text_cond,
            control_hidden_states=control_hidden_states,
            return_dict=False,
        )
        return out[0]  # (1, C, lT, lH, lW) BCTHW


# --------------------------------------------------------------------------- #
#  工厂函数                                                                     #
# --------------------------------------------------------------------------- #

def create_adapter(pipe) -> ModelAdapter:
    """根据 pipeline 类名自动选择适配器。"""
    return WanVACEAdapter(pipe)


def load_wan_vace_pipe(model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                       device: str = "cuda",
                       flow_shift: float = 3.0) -> Any:
    """加载 Wan2.1-VACE-1.3B pipeline（bfloat16），包含 T5 文本编码器。

    T5 是必须的：全零 text_cond 会让 transformer 输出巨大的固定偏置
    velocity（v_norm≈604），导致去噪完全失效。
    T5 编码空字符串后放到 CPU，推理时按需移到 GPU，常驻显存约 1.4GB。
    """
    import os
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
    os.environ["TQDM_DISABLE"] = "1"

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanVACEPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    if str(device).startswith("cuda"):
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            # transformer 分布到两卡，VAE 固定在 cuda:0
            try:
                from accelerate import dispatch_model, infer_auto_device_map
                total_0 = torch.cuda.get_device_properties(0).total_memory // 1024**3
                total_1 = torch.cuda.get_device_properties(1).total_memory // 1024**3
                mem = {0: f"{max(1, total_0 - 3)}GiB", 1: f"{max(1, total_1 - 3)}GiB"}
                device_map = infer_auto_device_map(pipe.transformer, max_memory=mem)
                pipe.transformer = dispatch_model(pipe.transformer, device_map=device_map)
            except Exception as e:
                print(f"[load] transformer dispatch skipped ({e}), using cuda:0")
                pipe.transformer.to("cuda:0")
        else:
            pipe.transformer.to("cuda:0")
        pipe.vae.to("cuda:0", dtype=torch.float32)
        # T5 放到 CPU，推理时按需移到 GPU（节省常驻显存）
        pipe.text_encoder.to("cpu")
        for i in range(n_gpus if n_gpus >= 2 else 1):
            free = torch.cuda.mem_get_info(i)[0] / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory // 1024**3
            print(f"[load] GPU {i} free: {free:.1f} GiB / {total} GiB")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    print(f"[load] scheduler: UniPCMultistepScheduler (flow_shift={flow_shift})")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    os.environ.pop("TQDM_DISABLE", None)
    return pipe

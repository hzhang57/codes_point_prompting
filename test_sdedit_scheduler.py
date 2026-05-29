import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sdedit import run_sdedit


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.linspace(999, 1, 4).long()
        self.begin_index = None
        self.add_noise_called = False

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.linspace(999, 1, n, device=device).long()

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def add_noise(self, latents, noise, timesteps):
        self.add_noise_called = True
        return latents + noise * 0.1

    def step(self, velocity, timestep, sample):
        return SimpleNamespace(prev_sample=sample)


class _FakeAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.scheduler = _FakeScheduler()
        self.add_noise_timestep = None

    def _video_scale(self):
        return 1.0

    def set_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.device)

    @property
    def timesteps(self):
        return self.scheduler.timesteps

    def prepare_denoise_start(self, n_steps, start_idx):
        self.set_timesteps(n_steps)
        self.scheduler.set_begin_index(start_idx)
        return self.timesteps[start_idx:]

    def add_noise_at_timestep(self, latents, noise, timestep):
        self.add_noise_timestep = timestep
        return self.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))

    def scheduler_step(self, velocity, t, latents, t_next=None):
        return self.scheduler.step(velocity, t, latents).prev_sample

    def encode_video(self, frames_bgr):
        return torch.zeros(1, 4, len(frames_bgr), 2, 2)

    def decode_latents(self, latents):
        return [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(latents.shape[2])]

    def encode_image_cond(self, frame_bgr, video_latent=None):
        return torch.zeros(1, 4, 1, 2, 2)

    def encode_text(self, prompt):
        return torch.zeros(1, 1, 4)

    def predict_with_guidance(
        self,
        noisy_latents,
        timestep,
        text_cond,
        image_cond_edited,
        image_cond_original,
        lam=8.0,
        n_frames_px=9,
    ):
        return torch.zeros_like(noisy_latents)


class TestSDEditSchedulerUsage(unittest.TestCase):
    def test_run_sdedit_uses_add_noise_and_begin_index(self):
        adapter = _FakeAdapter()
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]

        with patch("sdedit._save_frames"), patch("sdedit._save_mp4"), patch("sdedit.cv2.imwrite"):
            out = run_sdedit(
                adapter,
                frames_bgr_edited=frames,
                frame_bgr_original=frames[0],
                gamma=0.5,
                scheduler_steps=4,
            )

        self.assertEqual(len(out), 5)
        self.assertTrue(adapter.scheduler.add_noise_called)
        self.assertEqual(adapter.scheduler.begin_index, 2)
        self.assertEqual(adapter.add_noise_timestep.item(), adapter.timesteps[2].item())


if __name__ == "__main__":
    unittest.main()

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
        self.reference_frames = []
        self.guidance_calls = []
        self.decoded_shapes = []

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
        self.decoded_shapes.append(tuple(latents.shape))
        return [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(latents.shape[2])]

    def encode_image_cond(self, frame_bgr, video_latent=None):
        raise AssertionError("legacy encode_image_cond must not be used")

    def prepare_reference_condition(self, frame_bgr, n_frames_px, height, width):
        self.reference_frames.append(frame_bgr.copy())
        value = float(frame_bgr.max()) / 255.0
        return {
            "mode": "reference",
            "reference_latent_slots": 1,
            "control_hidden_states": torch.full(
                (1, 72, n_frames_px + 1, 2, 2), value
            ),
        }

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
        self.guidance_calls.append(
            (noisy_latents, image_cond_edited, image_cond_original)
        )
        return torch.zeros_like(noisy_latents)


class TestSDEditSchedulerUsage(unittest.TestCase):
    def test_run_sdedit_uses_add_noise_and_begin_index(self):
        adapter = _FakeAdapter()
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
        frames[0][8, 8] = (0, 0, 255)
        original = np.zeros_like(frames[0])

        with patch("sdedit._save_frames"), patch("sdedit._save_mp4"), patch("sdedit.cv2.imwrite"):
            out = run_sdedit(
                adapter,
                frames_bgr_edited=frames,
                frame_bgr_original=original,
                gamma=0.5,
                scheduler_steps=4,
            )

        self.assertEqual(len(out), 5)
        self.assertTrue(adapter.scheduler.add_noise_called)
        self.assertEqual(adapter.scheduler.begin_index, 2)
        self.assertEqual(adapter.add_noise_timestep.item(), adapter.timesteps[2].item())
        self.assertEqual(len(adapter.reference_frames), 2)
        self.assertGreater(adapter.reference_frames[0].max(), adapter.reference_frames[1].max())
        self.assertTrue(adapter.guidance_calls)
        noisy, edited, original_condition = adapter.guidance_calls[0]
        self.assertEqual(noisy.shape[2], len(frames) + 1)
        self.assertGreater(
            edited["control_hidden_states"].max(),
            original_condition["control_hidden_states"].max(),
        )
        self.assertTrue(all(shape[2] == len(frames) for shape in adapter.decoded_shapes))

    def test_zero_noise_strength_skips_noise_and_denoising(self):
        adapter = _FakeAdapter()
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]

        with patch("sdedit._save_frames"), patch("sdedit._save_mp4"), patch("sdedit.cv2.imwrite"):
            out = run_sdedit(
                adapter,
                frames_bgr_edited=frames,
                frame_bgr_original=frames[0],
                gamma=0.0,
                scheduler_steps=4,
            )

        self.assertEqual(len(out), 5)
        self.assertFalse(adapter.scheduler.add_noise_called)
        self.assertEqual(adapter.scheduler.begin_index, 3)
        self.assertEqual(len(adapter.reference_frames), 2)
        self.assertTrue(all(shape[2] == len(frames) for shape in adapter.decoded_shapes))


if __name__ == "__main__":
    unittest.main()

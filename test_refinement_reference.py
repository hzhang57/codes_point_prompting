import unittest
from types import SimpleNamespace

import numpy as np
import torch

from refinement import refine_tracks


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([999, 666, 333, 1])

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.linspace(999, 1, n, device=device).long()

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def add_noise(self, latents, noise, timestep):
        return latents + noise * 0.1

    def step(self, velocity, timestep, sample):
        return SimpleNamespace(prev_sample=sample)


class _FakeAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.scheduler = _FakeScheduler()
        self.reference_calls = []
        self.transformer_shapes = []
        self.decoded_shapes = []

    @property
    def timesteps(self):
        return self.scheduler.timesteps

    def encode_video(self, frames):
        return torch.zeros(1, 4, len(frames), 2, 2)

    def encode_image_cond(self, frame):
        raise AssertionError("legacy encode_image_cond must not be used")

    def prepare_reference_condition(self, frame, n_frames_px, height, width):
        self.reference_calls.append((frame.copy(), n_frames_px, height, width))
        return {
            "mode": "reference",
            "reference_latent_slots": 1,
            "control_hidden_states": torch.zeros(1, 72, n_frames_px + 1, 2, 2),
        }

    def encode_text(self, prompt):
        return torch.zeros(1, 1, 4)

    def prepare_denoise_start(self, n_steps, start_idx):
        self.scheduler.set_timesteps(n_steps, device=self.device)
        self.scheduler.set_begin_index(start_idx)
        return self.timesteps[start_idx:]

    def add_noise_at_timestep(self, latents, noise, timestep):
        return self.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))

    def forward_transformer(
        self, noisy_latents, timestep, text_cond, image_cond, n_frames_px=9
    ):
        self.transformer_shapes.append(tuple(noisy_latents.shape))
        self.assert_reference = image_cond["mode"]
        return torch.zeros_like(noisy_latents)

    def scheduler_step(self, velocity, timestep, latents, timestep_next=None):
        return self.scheduler.step(velocity, timestep, latents).prev_sample

    def decode_latents(self, latents):
        self.decoded_shapes.append(tuple(latents.shape))
        return [
            np.zeros((16, 16, 3), dtype=np.uint8)
            for _ in range(latents.shape[2])
        ]


class TestRefinementReferenceCondition(unittest.TestCase):
    def test_refinement_uses_official_reference_and_removes_reference_slot(self):
        adapter = _FakeAdapter()
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
        tracks = np.full((5, 2), 8.0, dtype=np.float32)

        out = refine_tracks(
            adapter,
            frames_bgr_generated=frames,
            frames_bgr_original=frames,
            tracks=tracks,
            gamma=0.5,
            scheduler_steps=4,
        )

        self.assertEqual(len(adapter.reference_calls), 1)
        self.assertTrue(adapter.transformer_shapes)
        self.assertTrue(all(shape[2] == len(frames) + 1 for shape in adapter.transformer_shapes))
        self.assertEqual(adapter.assert_reference, "reference")
        self.assertEqual(adapter.decoded_shapes[-1][2], len(frames))
        self.assertEqual(len(out), len(frames))


if __name__ == "__main__":
    unittest.main()

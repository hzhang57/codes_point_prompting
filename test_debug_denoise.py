import tempfile
import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import debug_denoise


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([999, 666, 333, 1])


class _FakeAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.scheduler = _FakeScheduler()
        self.text_encoder_released = False
        self.full_video_latent = None
        self.image_cond_video_latent = None
        self.add_noise_latents = []
        self.transformer_latent_shapes = []
        self.decoded_latent_shapes = []

    @property
    def timesteps(self):
        return self.scheduler.timesteps

    def encode_video(self, frames):
        self.full_video_latent = torch.zeros(1, 1, len(frames), 2, 2)
        return self.full_video_latent

    def decode_latents(self, latents):
        self.decoded_latent_shapes.append(tuple(latents.shape))
        value = int(min(255, max(0, latents.mean().item() * 10)))
        return [
            np.full((4, 4, 3), value, dtype=np.uint8)
            for _ in range(latents.shape[2])
        ]

    def encode_image_cond(self, frame, video_latent=None):
        self.image_cond_video_latent = video_latent
        return torch.zeros(1, 1, 1, 2, 2)

    def prepare_reference_condition(self, frame, n_frames_px, height, width):
        return {
            "mode": "reference",
            "reference_latent_slots": 1,
            "reference_latents": torch.full((1, 1, 1, 2, 2), 7.0),
            "control_hidden_states": torch.zeros(1, 66, n_frames_px + 1, 2, 2),
        }

    def encode_text(self, prompt):
        return torch.zeros(1, 1, 1)

    def release_text_encoder(self):
        self.text_encoder_released = True

    def prepare_denoise_start(self, n_steps, start_idx):
        self.scheduler.timesteps = torch.linspace(999, 1, n_steps).long()
        return self.scheduler.timesteps[start_idx:]

    def add_noise_at_timestep(self, latents, noise, timestep):
        self.add_noise_latents.append(latents)
        return latents + noise * 0.1

    def forward_transformer(
        self, noisy_latents, timestep, text_cond, image_cond, n_frames_px,
        conditioning_scale=1.0,
    ):
        self.transformer_latent_shapes.append(tuple(noisy_latents.shape))
        return torch.zeros_like(noisy_latents)

    def scheduler_step(self, velocity, timestep, latents, timestep_next):
        return latents


class TestDebugDenoise(unittest.TestCase):
    def test_save_video_writes_playable_mp4(self):
        frames = [np.full((16, 24, 3), i * 20, dtype=np.uint8) for i in range(3)]
        with tempfile.TemporaryDirectory() as output_dir:
            path = os.path.join(output_dir, "denoised.mp4")
            debug_denoise.save_video(frames, path, 12.0)

            cap = debug_denoise.cv2.VideoCapture(path)
            decoded = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(frame)
            cap.release()

        self.assertEqual(len(decoded), len(frames))
        self.assertEqual(decoded[0].shape, frames[0].shape)

    def test_scan_requests_denoised_video_for_each_gamma(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        args = SimpleNamespace(
            video="input.mp4",
            model_id="fake",
            device="cpu",
            max_frames=5,
            height=4,
            width=4,
            gammas=[0.0, 0.5, 1.0],
            scheduler_steps=4,
            flow_shift=3.0,
            prompt="",
            condition_mode="reference",
            conditioning_scale=1.0,
            seed=42,
            output_dir=None,
            low_memory=True,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            fake_adapter = _FakeAdapter()
            with patch("debug_denoise.load_video", return_value=(frames, 12.0)), \
                    patch("debug_denoise.load_wan_vace_pipe", return_value=object()) as loader_mock, \
                    patch("debug_denoise.create_adapter", return_value=fake_adapter), \
                    patch("debug_denoise.save_video") as save_video_mock, \
                    patch("debug_denoise.save_frames"):
                debug_denoise.run_debug(args)

            loader_mock.assert_called_once_with(
                "fake", device="cpu", flow_shift=3.0, low_cpu_memory=True
            )
            self.assertTrue(fake_adapter.text_encoder_released)
            self.assertIsNone(fake_adapter.image_cond_video_latent)
            self.assertTrue(fake_adapter.add_noise_latents)
            self.assertTrue(all(
                latent.shape[2] == fake_adapter.full_video_latent.shape[2] + 1
                for latent in fake_adapter.add_noise_latents
            ))
            self.assertTrue(fake_adapter.transformer_latent_shapes)
            self.assertTrue(all(shape[2] == len(frames) + 1 for shape in fake_adapter.transformer_latent_shapes))
            self.assertTrue(all(shape[2] == len(frames) for shape in fake_adapter.decoded_latent_shapes))
            saved_paths = [call.args[1] for call in save_video_mock.call_args_list]
            for gamma in args.gammas:
                expected = debug_denoise.os.path.join(
                    output_dir, debug_denoise.gamma_dir_name(gamma), "denoised.mp4"
                )
                self.assertIn(expected, saved_paths)

            self.assertTrue(debug_denoise.os.path.exists(
                debug_denoise.os.path.join(output_dir, "summary.csv")
            ))
            self.assertTrue(debug_denoise.os.path.exists(
                debug_denoise.os.path.join(output_dir, "summary.json")
            ))

    def test_legacy_mode_uses_full_video_first_frame_condition(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        args = SimpleNamespace(
            video="input.mp4",
            model_id="fake",
            device="cpu",
            max_frames=5,
            height=4,
            width=4,
            gammas=[0.5],
            scheduler_steps=4,
            flow_shift=3.0,
            prompt="",
            condition_mode="legacy-first-frame",
            conditioning_scale=1.0,
            seed=42,
            output_dir=None,
            low_memory=True,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            fake_adapter = _FakeAdapter()
            with patch("debug_denoise.load_video", return_value=(frames, 12.0)), \
                    patch("debug_denoise.load_wan_vace_pipe", return_value=object()), \
                    patch("debug_denoise.create_adapter", return_value=fake_adapter), \
                    patch("debug_denoise.save_video"), \
                    patch("debug_denoise.save_frames"):
                debug_denoise.run_debug(args)

        self.assertIs(fake_adapter.image_cond_video_latent, fake_adapter.full_video_latent)
        self.assertTrue(all(
            latent is fake_adapter.full_video_latent for latent in fake_adapter.add_noise_latents
        ))


if __name__ == "__main__":
    unittest.main()

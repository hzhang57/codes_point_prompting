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

    @property
    def timesteps(self):
        return self.scheduler.timesteps

    def encode_video(self, frames):
        return torch.zeros(1, 1, len(frames), 2, 2)

    def decode_latents(self, latents):
        value = int(min(255, max(0, latents.mean().item() * 10)))
        return [
            np.full((4, 4, 3), value, dtype=np.uint8)
            for _ in range(latents.shape[2])
        ]

    def encode_image_cond(self, frame, video_latent=None):
        return torch.zeros(1, 1, 1, 2, 2)

    def encode_text(self, prompt):
        return torch.zeros(1, 1, 1)

    def release_text_encoder(self):
        self.text_encoder_released = True

    def prepare_denoise_start(self, n_steps, start_idx):
        self.scheduler.timesteps = torch.linspace(999, 1, n_steps).long()
        return self.scheduler.timesteps[start_idx:]

    def add_noise_at_timestep(self, latents, noise, timestep):
        return latents + noise * 0.1

    def forward_transformer(self, noisy_latents, timestep, text_cond, image_cond, n_frames_px):
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
            seed=42,
            output_dir=None,
            low_memory=True,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            with patch("debug_denoise.load_video", return_value=(frames, 12.0)), \
                    patch("debug_denoise.load_wan_vace_pipe", return_value=object()) as loader_mock, \
                    patch("debug_denoise.create_adapter", return_value=_FakeAdapter()) as adapter_mock, \
                    patch("debug_denoise.save_video") as save_video_mock, \
                    patch("debug_denoise.save_frames"):
                debug_denoise.run_debug(args)

            loader_mock.assert_called_once_with(
                "fake", device="cpu", flow_shift=3.0, low_cpu_memory=True
            )
            self.assertTrue(adapter_mock.text_encoder_released)
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


if __name__ == "__main__":
    unittest.main()

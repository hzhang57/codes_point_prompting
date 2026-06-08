import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import debug_denoise_vace


class _LatentDist:
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return self.mean


class _EncodeResult:
    def __init__(self, latents):
        self.latent_dist = _LatentDist(latents)


class _DecodeResult:
    def __init__(self, sample):
        self.sample = sample


class _FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(
            z_dim=1,
            latents_mean=[0.0],
            latents_std=[1.0],
        )
        self._param = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))

    def parameters(self):
        return iter([self._param])

    @property
    def dtype(self):
        return self._param.dtype

    def encode(self, tensor):
        b, _, t, _, _ = tensor.shape
        latents = torch.zeros(b, 1, t, 2, 2, device=tensor.device, dtype=tensor.dtype)
        return _EncodeResult(latents)

    def decode(self, latents):
        b, _, t, _, _ = latents.shape
        sample = torch.zeros(b, 3, t, 4, 4, device=latents.device, dtype=latents.dtype)
        return _DecodeResult(sample)


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([999, 666, 333, 1])
        self.add_noise_calls = []
        self.step_calls = []
        self.begin_index = None
        self.order = 1
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, n_steps, device=None):
        self.timesteps = torch.linspace(999, 1, n_steps, device=device).long()

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def add_noise(self, latents, noise, timesteps):
        self.add_noise_calls.append((latents, noise, timesteps))
        return latents + noise * 0.1

    def step(self, noise_pred, timestep, latents, return_dict=False):
        self.step_calls.append((noise_pred, timestep, latents))
        return (latents,)


class _FakeTransformer:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=1, vace_layers=[0, 1], patch_size=(1, 2, 2))
        self._param = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))
        self.calls = []

    def parameters(self):
        return iter([self._param])

    def cache_context(self, _name):
        return nullcontext()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros_like(kwargs["hidden_states"]),)


class _FakeTextEncoder:
    dtype = torch.float32


class _FakePipe:
    def __init__(self):
        self.vae = _FakeVAE()
        self.transformer = _FakeTransformer()
        self.transformer_2 = None
        self.scheduler = _FakeScheduler()
        self.text_encoder = _FakeTextEncoder()
        self.config = SimpleNamespace(boundary_ratio=None)
        self.vae_scale_factor_temporal = 4
        self.prepare_latents_calls = []
        self.preprocess_calls = []

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def _get_t5_prompt_embeds(
        self, prompt, num_videos_per_prompt, max_sequence_length, device, dtype
    ):
        return torch.zeros(
            num_videos_per_prompt, max_sequence_length, 4, device=device, dtype=dtype
        )

    def preprocess_conditions(
        self, video, mask, reference_images, batch_size, height, width, num_frames, dtype, device
    ):
        self.preprocess_calls.append(reference_images)
        video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
        mask = torch.ones_like(video)
        reference = torch.ones(3, height, width, dtype=dtype, device=device)
        return video, mask, [[reference]]

    def prepare_video_latents(self, video, mask, reference_images, generator, device):
        return torch.zeros(1, 2, video.shape[2] + 1, 2, 2, device=device)

    def prepare_masks(self, mask, reference_images, generator):
        return torch.zeros(1, 64, mask.shape[2] + 1, 2, 2, device=mask.device)

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, num_frames,
        dtype, device, generator, latents=None,
    ):
        self.prepare_latents_calls.append(latents)
        return latents.to(device=device, dtype=dtype)


class TestDebugDenoiseVACE(unittest.TestCase):
    def _args(self, output_dir):
        return SimpleNamespace(
            video="input.mp4",
            model_id="fake",
            device="cpu",
            max_frames=5,
            height=4,
            width=4,
            gammas=[0.0, 0.5],
            scheduler_steps=4,
            flow_shift=3.0,
            prompt="",
            negative_prompt="",
            conditioning_scale=1.0,
            guidance_scale=2.0,
            max_sequence_length=8,
            seed=42,
            output_dir=output_dir,
            low_memory=True,
        )

    def test_run_debug_uses_official_vace_helpers_and_pipeline_loop(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        pipe = _FakePipe()

        with tempfile.TemporaryDirectory() as output_dir:
            args = self._args(output_dir)
            with patch("debug_denoise_vace.load_video", return_value=(frames, 12.0)), \
                    patch("debug_denoise_vace.load_wan_vace_pipe", return_value=pipe) as loader, \
                    patch("debug_denoise_vace.save_video") as save_video, \
                    patch("debug_denoise_vace.save_frames"):
                debug_denoise_vace.run_debug(args)

            loader.assert_called_once_with(
                "fake", device="cpu", flow_shift=3.0, low_cpu_memory=True
            )
            self.assertEqual(len(pipe.preprocess_calls), 1)
            self.assertTrue(pipe.prepare_latents_calls)
            self.assertTrue(pipe.scheduler.add_noise_calls)
            self.assertEqual(pipe.scheduler.begin_index, 2)
            self.assertTrue(pipe.scheduler.step_calls)
            self.assertTrue(pipe.transformer.calls)
            self.assertTrue(all(
                call["hidden_states"].shape[2] == len(frames) + 1
                for call in pipe.transformer.calls
            ))

            saved_paths = [call.args[1] for call in save_video.call_args_list]
            self.assertIn(os.path.join(output_dir, "gamma_0.5", "denoised.mp4"), saved_paths)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "summary.json")))

            with open(os.path.join(output_dir, "summary.json")) as handle:
                summary = handle.read()
            self.assertIn('"guidance_scale": 2.0', summary)
            self.assertIn('"conditioning_scale": 1.0', summary)


if __name__ == "__main__":
    unittest.main()

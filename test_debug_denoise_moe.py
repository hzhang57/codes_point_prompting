import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import debug_denoise_moe


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
        self.decode_inputs = []

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
        self.decode_inputs.append(latents.detach().clone())
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
        self.add_noise_calls.append((latents.detach().clone(), noise.detach().clone(), timesteps))
        return latents + noise * 0.1

    def step(self, noise_pred, timestep, latents, return_dict=False):
        self.step_calls.append((noise_pred.detach().clone(), timestep, latents.detach().clone()))
        return (latents - noise_pred * 0.01,)


class _FakeTransformer:
    def __init__(self, in_channels=1):
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=in_channels, patch_size=(1, 1, 1), image_dim=None)
        self._param = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))
        self.calls = []

    def parameters(self):
        return iter([self._param])

    def cache_context(self, _name):
        return nullcontext()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        batch, _, t, h, w = hidden.shape
        embeds = kwargs["encoder_hidden_states"]
        prompt_offset = embeds.mean().to(device=hidden.device, dtype=hidden.dtype)
        return (torch.ones(batch, 1, t, h, w, device=hidden.device, dtype=hidden.dtype) * prompt_offset,)


class _FakeTextEncoder:
    dtype = torch.float32


class _FakeVideoProcessor:
    def __init__(self):
        self.calls = []

    def preprocess(self, image, height, width):
        self.calls.append((image, height, width))
        return torch.zeros(1, 3, height, width)


class _FakePipe:
    def __init__(self, expand_timesteps=True):
        self.vae = _FakeVAE()
        self.transformer = _FakeTransformer(in_channels=1 if expand_timesteps else 3)
        self.transformer_2 = None
        self.scheduler = _FakeScheduler()
        self.text_encoder = _FakeTextEncoder()
        self.image_encoder = None
        self.video_processor = _FakeVideoProcessor()
        self.config = SimpleNamespace(expand_timesteps=expand_timesteps, boundary_ratio=None)
        self.prepare_latents_calls = []
        self.expand_timesteps = expand_timesteps

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def _get_t5_prompt_embeds(
        self, prompt, num_videos_per_prompt, max_sequence_length, device, dtype
    ):
        value = 1.0 if prompt else 0.0
        return torch.full(
            (num_videos_per_prompt, max_sequence_length, 4),
            value,
            device=device,
            dtype=dtype,
        )

    def prepare_latents(
        self, image, batch_size, num_channels_latents, height, width, num_frames,
        dtype, device, generator, latents=None, last_image=None,
    ):
        self.prepare_latents_calls.append(
            {
                "image": image,
                "latents": latents.detach().clone(),
                "height": height,
                "width": width,
                "num_frames": num_frames,
            }
        )
        latents = latents.to(device=device, dtype=dtype)
        condition = torch.full_like(latents, 7.0)
        if self.expand_timesteps:
            mask = torch.ones_like(latents)
            mask[:, :, :1] = 0.0
            return latents, condition, mask
        condition = torch.full(
            (latents.shape[0], 2, latents.shape[2], latents.shape[3], latents.shape[4]),
            3.0,
            device=device,
            dtype=dtype,
        )
        return latents, condition


class TestDebugDenoiseMOE(unittest.TestCase):
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
            negative_prompt="bad",
            guidance_scale=2.0,
            guidance_scale_2=3.0,
            max_sequence_length=8,
            seed=42,
            output_dir=output_dir,
            low_memory=True,
        )

    def _run(self, pipe, output_dir):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        args = self._args(output_dir)
        with patch("debug_denoise_moe.load_video", return_value=(frames, 12.0)), \
                patch("debug_denoise_moe.load_wan_ti2v_pipe", return_value=pipe) as loader, \
                patch("debug_denoise_moe.save_video"), \
                patch("debug_denoise_moe.save_frames"):
            debug_denoise_moe.run_debug(args)
        loader.assert_called_once_with(
            "fake", device="cpu", flow_shift=3.0, low_cpu_memory=True
        )
        return args

    def test_expand_timesteps_uses_prepare_latents_mask_fusion_and_cfg(self):
        with tempfile.TemporaryDirectory() as output_dir:
            pipe = _FakePipe(expand_timesteps=True)
            self._run(pipe, output_dir)

            self.assertEqual(len(pipe.prepare_latents_calls), 2)
            self.assertEqual(len(pipe.scheduler.add_noise_calls), 1)
            self.assertEqual(pipe.scheduler.begin_index, 2)
            self.assertTrue(pipe.scheduler.step_calls)
            self.assertGreaterEqual(len(pipe.transformer.calls), 2)

            first_call = pipe.transformer.calls[0]
            self.assertEqual(first_call["hidden_states"].shape[1], 1)
            self.assertTrue(torch.all(first_call["hidden_states"][:, :, :1] == 7.0))
            self.assertEqual(first_call["timestep"].ndim, 2)
            self.assertIsNone(first_call["encoder_hidden_states_image"])

            cond_calls = [
                call for call in pipe.transformer.calls
                if torch.isclose(call["encoder_hidden_states"].mean(), torch.tensor(0.0))
            ]
            uncond_calls = [
                call for call in pipe.transformer.calls
                if torch.isclose(call["encoder_hidden_states"].mean(), torch.tensor(1.0))
            ]
            self.assertTrue(cond_calls)
            self.assertTrue(uncond_calls)

            decoded_channels = [tensor.shape[1] for tensor in pipe.vae.decode_inputs]
            self.assertTrue(all(channels == 1 for channels in decoded_channels))

            with open(os.path.join(output_dir, "summary.json")) as handle:
                summary = json.load(handle)
            self.assertEqual(summary["guidance_scale"], 2.0)
            self.assertEqual(summary["guidance_scale_2"], 3.0)
            self.assertEqual(summary["reference_latent_slots"], 0)
            run = summary["runs"][1]
            self.assertTrue(run["expand_timesteps"])
            self.assertEqual(run["start_idx"], 2)
            self.assertEqual(run["denoise_steps"], 2)

    def test_non_expand_timesteps_concats_latents_and_condition(self):
        with tempfile.TemporaryDirectory() as output_dir:
            pipe = _FakePipe(expand_timesteps=False)
            self._run(pipe, output_dir)

            self.assertEqual(len(pipe.prepare_latents_calls), 2)
            self.assertTrue(pipe.transformer.calls)
            first_call = pipe.transformer.calls[0]
            self.assertEqual(first_call["hidden_states"].shape[1], 3)
            self.assertEqual(first_call["timestep"].ndim, 1)

            with open(os.path.join(output_dir, "summary.json")) as handle:
                summary = json.load(handle)
            self.assertFalse(summary["runs"][1]["expand_timesteps"])


if __name__ == "__main__":
    unittest.main()

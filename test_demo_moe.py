import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import demo_moe


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

    def encode(self, tensor):
        b, _, t, _, _ = tensor.shape
        return _EncodeResult(torch.zeros(b, 1, t, 2, 2, dtype=tensor.dtype, device=tensor.device))

    def decode(self, latents):
        self.decode_inputs.append(latents.detach().clone())
        b, _, t, _, _ = latents.shape
        sample = torch.zeros(b, 3, t, 4, 4, dtype=latents.dtype, device=latents.device)
        return _DecodeResult(sample)


class UniPCMultistepScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([999, 666, 333, 1])
        self.add_noise_calls = []
        self.step_calls = []
        self.begin_index = None
        self.config = SimpleNamespace(
            flow_shift=5.0,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            timestep_spacing="linspace",
            solver_order=2,
            solver_type="bh2",
            num_train_timesteps=1000,
        )

    def set_timesteps(self, n_steps, device=None):
        self.timesteps = torch.linspace(999, 1, n_steps, device=device).long()

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def add_noise(self, latents, noise, timestep):
        self.add_noise_calls.append((latents.detach().clone(), noise.detach().clone(), timestep))
        return latents + noise * 0.1

    def step(self, noise_pred, timestep, latents, return_dict=False):
        self.step_calls.append((noise_pred.detach().clone(), timestep, latents.detach().clone()))
        return (latents - noise_pred * 0.01,)


class _FakeTransformer:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(patch_size=(1, 2, 2), max_text_seq_len=512)
        self._param = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))
        self.calls = []

    def parameters(self):
        return iter([self._param])

    def cache_context(self, _name):
        return nullcontext()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        value = hidden[:, :, :1].mean().to(dtype=hidden.dtype, device=hidden.device)
        return (torch.ones_like(hidden) * value,)


class _FakeTextEncoder:
    dtype = torch.float32


class _FakeVideoProcessor:
    def __init__(self):
        self.calls = []

    def preprocess(self, image, height, width):
        self.calls.append((image, height, width))
        return torch.zeros(1, 3, height, width)


class WanImageToVideoPipeline:
    def __init__(self, expand_timesteps=True):
        self.vae = _FakeVAE()
        self.transformer = _FakeTransformer()
        self.scheduler = UniPCMultistepScheduler()
        self.text_encoder = _FakeTextEncoder()
        self.video_processor = _FakeVideoProcessor()
        self.config = SimpleNamespace(expand_timesteps=expand_timesteps)
        self.prepare_latents_calls = []

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def _get_t5_prompt_embeds(self, prompt, num_videos_per_prompt, max_sequence_length, device, dtype):
        return torch.zeros(num_videos_per_prompt, max_sequence_length, 4, device=device, dtype=dtype)

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
        prepared = latents.to(device=device, dtype=dtype)
        call_index = len(self.prepare_latents_calls)
        condition_value = 8.0 if call_index % 2 == 1 else 3.0
        condition = torch.full_like(prepared, condition_value)
        if not self.config.expand_timesteps:
            return prepared, condition
        mask = torch.ones_like(prepared)
        mask[:, :, :1] = 0.0
        return prepared, condition, mask


class _WrongPipeline(WanImageToVideoPipeline):
    pass


class TestDemoMOE(unittest.TestCase):
    def test_official_ti2v_size_follows_aspect(self):
        self.assertEqual(demo_moe.official_ti2v_size_for_aspect(1920, 1080), (1280, 704))
        self.assertEqual(demo_moe.official_ti2v_size_for_aspect(1080, 1920), (704, 1280))
        self.assertEqual(demo_moe.default_ti2v_size_for_preset("t4", 1920, 1080), (832, 480))
        self.assertEqual(demo_moe.default_ti2v_size_for_preset("official", 1920, 1080), (1280, 704))

    def test_validate_requires_wan_i2v_and_official_scheduler(self):
        with self.assertRaisesRegex(TypeError, "requires WanImageToVideoPipeline"):
            demo_moe.validate_wan22_ti2v5b_pipeline(_WrongPipeline())
        pipe = WanImageToVideoPipeline()
        pipe.scheduler.config.flow_shift = 3.0
        with self.assertRaisesRegex(ValueError, "official Wan2.2-TI2V-5B config"):
            demo_moe.validate_wan22_ti2v5b_pipeline(pipe)
        pipe = WanImageToVideoPipeline(expand_timesteps=False)
        with self.assertRaisesRegex(ValueError, "expand_timesteps=True"):
            demo_moe.validate_wan22_ti2v5b_pipeline(pipe)

    def test_prepare_counterfactual_conditions_uses_official_prepare_latents(self):
        pipe = WanImageToVideoPipeline()
        latents = torch.zeros(1, 1, 5, 2, 2)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        conditions = demo_moe.prepare_counterfactual_conditions(
            pipe, frame, frame, latents, 4, 4, 5, seed=1
        )
        self.assertEqual(len(pipe.prepare_latents_calls), 2)
        self.assertEqual(conditions.marked.condition.shape, conditions.original.condition.shape)
        self.assertTrue(torch.allclose(conditions.marked.first_frame_mask, conditions.original.first_frame_mask))
        self.assertEqual(conditions.diff_mean, 5.0)
        self.assertEqual(conditions.diff_max, 5.0)

    def test_counterfactual_denoise_uses_mask_fusion_and_formula(self):
        pipe = WanImageToVideoPipeline()
        latents = torch.zeros(1, 1, 5, 2, 2)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        conditions = demo_moe.prepare_counterfactual_conditions(
            pipe, frame, frame, latents, 4, 4, 5, seed=1
        )
        prompt = torch.zeros(1, 512, 4)
        timesteps = demo_moe.set_denoise_start(pipe, 4, 2)
        out = demo_moe.denoise_counterfactual_ti2v(
            pipe, latents.clone(), timesteps, conditions, prompt, lam=2.0
        )
        self.assertEqual(len(pipe.transformer.calls), 4)
        marked_hidden = pipe.transformer.calls[0]["hidden_states"]
        original_hidden = pipe.transformer.calls[1]["hidden_states"]
        self.assertTrue(torch.all(marked_hidden[:, :, :1] == 8.0))
        self.assertTrue(torch.all(original_hidden[:, :, :1] == 3.0))
        first_noise_pred = pipe.scheduler.step_calls[0][0]
        self.assertTrue(torch.all(first_noise_pred == 18.0))
        self.assertEqual(out.shape, latents.shape)

    def test_run_counterfactual_gamma_zero_and_nonzero_noise(self):
        pipe = WanImageToVideoPipeline()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        prompt = torch.zeros(1, 512, 4)
        demo_moe.run_counterfactual_sdedit(
            pipe, frames, frames[0], frames[0], 0.0, 1.0, 4, prompt, 1
        )
        self.assertEqual(len(pipe.scheduler.add_noise_calls), 0)
        demo_moe.run_counterfactual_sdedit(
            pipe, frames, frames[0], frames[0], 0.5, 1.0, 4, prompt, 1
        )
        self.assertEqual(len(pipe.scheduler.add_noise_calls), 1)
        self.assertTrue(all(tensor.shape[1] == 1 for tensor in pipe.vae.decode_inputs))

    def _args(self, output_dir):
        return SimpleNamespace(
            video="input.mp4",
            points=["1,1"],
            output=os.path.join(output_dir, "tracked.mp4"),
            model_id="fake",
            device="cpu",
            vae_device="auto",
            vae_dtype="float32",
            gamma=0.5,
            lam=2.0,
            scheduler_steps=4,
            prompt="",
            negative_prompt="",
            guidance_scale=1.0,
            max_sequence_length=None,
            seed=42,
            max_frames=5,
            resolution_preset="custom",
            preprocess_width=4,
            preprocess_height=4,
            model_width=4,
            model_height=4,
            model_stride=1,
            marker_radius=2,
            no_refine=False,
            refine_gamma=0.25,
            save_generated=False,
            output_dir=output_dir,
            decode_noisy=False,
            low_memory=True,
        )

    def test_run_demo_refinement_and_summary(self):
        pipe = WanImageToVideoPipeline()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        with tempfile.TemporaryDirectory() as output_dir:
            args = self._args(output_dir)
            with patch("demo_moe.load_video", return_value=(frames, 12.0)), \
                    patch("demo_moe.load_wan_ti2v_pipe", return_value=pipe), \
                    patch("demo_moe.save_video"), \
                    patch("demo_moe.track_marker_sequence", return_value=(np.ones((5, 2)), np.ones(5, dtype=bool))):
                summary = demo_moe.run_demo(args)
            self.assertEqual(len(summary["points"][0]["stages"]), 2)
            self.assertEqual(summary["points"][0]["stages"][0]["stage"], "sdedit")
            self.assertEqual(summary["points"][0]["stages"][1]["stage"], "refine")
            self.assertEqual(summary["scheduler_class"], "UniPCMultistepScheduler")
            self.assertEqual(summary["reference_latent_slots"], 0)
            with open(os.path.join(output_dir, "summary.json")) as handle:
                saved = json.load(handle)
            self.assertEqual(saved["lam"], 2.0)
            self.assertEqual(saved["points"][0]["visible_ratio"], 1.0)

    def test_run_demo_no_refine(self):
        pipe = WanImageToVideoPipeline()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        with tempfile.TemporaryDirectory() as output_dir:
            args = self._args(output_dir)
            args.no_refine = True
            with patch("demo_moe.load_video", return_value=(frames, 12.0)), \
                    patch("demo_moe.load_wan_ti2v_pipe", return_value=pipe), \
                    patch("demo_moe.save_video"), \
                    patch("demo_moe.track_marker_sequence", return_value=(np.ones((5, 2)), np.ones(5, dtype=bool))):
                summary = demo_moe.run_demo(args)
            self.assertEqual(len(summary["points"][0]["stages"]), 1)


if __name__ == "__main__":
    unittest.main()

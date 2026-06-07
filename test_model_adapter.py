import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, ".")

from model_adapter import (
    ModelAdapter,
    WanVACEAdapter,
    create_adapter,
    denoise_step_count,
    load_wan_vace_pipe,
    noise_strength_to_start_idx,
    prepend_reference_slots,
    remove_reference_slots,
    _frames_to_tensor,
    _tensor_to_frames,
)


B, C_IMG, T, H, W = 1, 3, 5, 32, 48
LAT_C = 4
TEXT_LEN = 226
TEXT_D = 64


class _LatentDist:
    def __init__(self, data):
        self.mean = data


class _VAEEncodeResult:
    def __init__(self, data):
        self.latent_dist = _LatentDist(data)


class _VAEDecodeResult:
    def __init__(self, data):
        self.sample = data


class _MockVAE:
    def __init__(self, dtype=torch.float32):
        self.config = SimpleNamespace(
            z_dim=LAT_C,
            latents_mean=[0.0] * LAT_C,
            latents_std=[1.0] * LAT_C,
        )
        self._param = torch.nn.Parameter(torch.empty(0, dtype=dtype))
        self.slicing_enabled = False
        self.tiling_enabled = False

    def parameters(self):
        return iter([self._param])

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self._param.data = self._param.data.to(dtype=dtype)
        if device is not None:
            self._param.data = self._param.data.to(device=device)
        return self

    def encode(self, x):
        b, _, t, h, w = x.shape
        lat = torch.zeros(b, LAT_C, t, h, w, device=x.device, dtype=x.dtype)
        return _VAEEncodeResult(lat)

    def decode(self, x):
        b, _, t, h, w = x.shape
        sample = torch.zeros(b, C_IMG, t, h, w, device=x.device, dtype=x.dtype)
        return _VAEDecodeResult(sample)

    def enable_slicing(self):
        self.slicing_enabled = True

    def enable_tiling(self):
        self.tiling_enabled = True


class _MockTransformer:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.config = SimpleNamespace(
            in_channels=LAT_C,
            text_dim=TEXT_D,
            max_text_seq_len=TEXT_LEN,
            vace_layers=[0, 1, 2],
        )
        self._param = torch.nn.Parameter(torch.empty(0, dtype=dtype))
        self.last_kwargs = None

    def parameters(self):
        return iter([self._param])

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self._param.data = self._param.data.to(dtype=dtype)
        if device is not None:
            self._param.data = self._param.data.to(device=device)
        return self

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return (torch.zeros_like(kwargs["hidden_states"]),)


class _MockTextEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.weight])

    def to(self, device=None, dtype=None):
        self.device = torch.device(device)
        self.weight.data = self.weight.data.to(device=device, dtype=dtype)
        return self


class _MockScheduler:
    def __init__(self, n=10):
        self.timesteps = torch.linspace(999, 1, n).long()
        self.begin_index = None
        self.add_noise_calls = []

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.linspace(999, 1, n, device=device).long()

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def add_noise(self, latents, noise, timesteps):
        self.add_noise_calls.append((latents, noise, timesteps))
        return latents + noise * 0.5

    def step(self, velocity, timestep, sample):
        return SimpleNamespace(prev_sample=sample - 0.01 * velocity)


class _MockWanVACEPipeline:
    def __init__(self):
        self.vae = _MockVAE()
        self.transformer = _MockTransformer()
        self.text_encoder = _MockTextEncoder()
        self.scheduler = _MockScheduler()

    def _get_t5_prompt_embeds(self, prompt, num_videos_per_prompt, max_sequence_length, device):
        return torch.zeros(num_videos_per_prompt, max_sequence_length, TEXT_D, device=device)

    def preprocess_conditions(
        self, video, mask, reference_images, batch_size, height, width, num_frames, dtype, device
    ):
        video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
        mask = torch.ones_like(video)
        reference = torch.ones(3, height, width, dtype=dtype, device=device)
        return video, mask, [[reference]]

    def prepare_video_latents(self, video, mask, reference_images, generator, device):
        t = video.shape[2]
        latents = torch.zeros(1, 2 * LAT_C, t + 1, video.shape[3], video.shape[4], device=device)
        latents[:, :LAT_C, :1] = 7
        return latents

    def prepare_masks(self, mask, reference_images, generator):
        return torch.zeros(1, 64, mask.shape[2] + 1, mask.shape[3], mask.shape[4])


class _MinimalAdapter(ModelAdapter):
    def __init__(self, sched=None):
        self._sched = sched or _MockScheduler()
        self._dpm_step = None

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    @property
    def scheduler(self):
        return self._sched

    def encode_video(self, frames_bgr):
        raise NotImplementedError

    def decode_latents(self, latents):
        raise NotImplementedError

    def encode_image_cond(self, frame_bgr, video_latent=None):
        raise NotImplementedError

    def encode_text(self, prompt):
        raise NotImplementedError

    def forward_transformer(
        self, noisy_latents, timestep, text_cond, image_cond, n_frames_px=9,
        conditioning_scale=1.0,
    ):
        return torch.ones_like(noisy_latents)


class TestTensorHelpers(unittest.TestCase):
    def test_frames_tensor_roundtrip_shape_and_channels(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        frame[..., 0] = 10
        frame[..., 1] = 20
        frame[..., 2] = 30

        tensor = _frames_to_tensor([frame], torch.device("cpu"), torch.float32)
        out = _tensor_to_frames(tensor)

        self.assertEqual(tensor.shape, (1, 3, 1, 4, 5))
        np.testing.assert_allclose(out[0], frame, atol=1)

    def test_reference_slot_helpers_roundtrip(self):
        video = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 2, 2)

        model_latents = prepend_reference_slots(video, 2)

        self.assertEqual(model_latents.shape, (1, 1, 5, 2, 2))
        self.assertEqual(model_latents[:, :, :2].count_nonzero().item(), 0)
        torch.testing.assert_close(remove_reference_slots(model_latents, 2), video)


class TestSchedulerHelpers(unittest.TestCase):
    def test_noise_strength_maps_to_descending_scheduler_index(self):
        self.assertEqual(noise_strength_to_start_idx(0.0, 100), 99)
        self.assertEqual(noise_strength_to_start_idx(0.5, 100), 50)
        self.assertEqual(noise_strength_to_start_idx(1.0, 100), 0)
        self.assertEqual(denoise_step_count(0.0, 100), 0)
        self.assertEqual(denoise_step_count(0.5, 100), 50)
        self.assertEqual(denoise_step_count(1.0, 100), 100)

    def test_noise_strength_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            noise_strength_to_start_idx(-0.1, 100)
        with self.assertRaises(ValueError):
            noise_strength_to_start_idx(1.1, 100)
        with self.assertRaises(ValueError):
            noise_strength_to_start_idx(0.5, 0)

    def test_prepare_denoise_start_sets_begin_index_and_returns_slice(self):
        sched = _MockScheduler(n=12)
        adapter = _MinimalAdapter(sched)

        timesteps_run = adapter.prepare_denoise_start(20, 7)

        self.assertEqual(len(adapter.timesteps), 20)
        self.assertEqual(sched.begin_index, 7)
        torch.testing.assert_close(timesteps_run, adapter.timesteps[7:])

    def test_add_noise_at_timestep_uses_scheduler_add_noise(self):
        sched = _MockScheduler()
        adapter = _MinimalAdapter(sched)
        latents = torch.zeros(1, LAT_C, 2, 4, 4)
        noise = torch.ones_like(latents)

        out = adapter.add_noise_at_timestep(latents, noise, torch.tensor(500))

        torch.testing.assert_close(out, torch.full_like(latents, 0.5))
        self.assertEqual(len(sched.add_noise_calls), 1)
        self.assertEqual(sched.add_noise_calls[0][2].shape, (1,))

    def test_scheduler_step_uses_generic_step_for_unipc_style_scheduler(self):
        adapter = _MinimalAdapter(_MockScheduler())
        latents = torch.ones(1, LAT_C, 2, 4, 4)
        velocity = torch.ones_like(latents)

        out = adapter.scheduler_step(velocity, torch.tensor(1), latents)

        torch.testing.assert_close(out, latents - 0.01)

    def test_counterfactual_guidance_reuses_noisy_latents_and_combines_conditions(self):
        class GuidanceAdapter(_MinimalAdapter):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward_transformer(
                self, noisy_latents, timestep, text_cond, image_cond,
                n_frames_px=9, conditioning_scale=1.0,
            ):
                self.calls.append((noisy_latents, image_cond))
                return torch.full_like(noisy_latents, image_cond["value"])

        adapter = GuidanceAdapter()
        noisy = torch.zeros(1, LAT_C, 2, 4, 4)

        guided = adapter.predict_with_guidance(
            noisy,
            torch.tensor([1]),
            text_cond=None,
            image_cond_edited={"value": 2.0},
            image_cond_original={"value": 0.5},
            lam=3.0,
        )

        self.assertIs(adapter.calls[0][0], noisy)
        self.assertIs(adapter.calls[1][0], noisy)
        torch.testing.assert_close(guided, torch.full_like(noisy, 6.5))


class TestWanVACEAdapter(unittest.TestCase):
    def test_create_adapter_returns_wan_vace_adapter(self):
        adapter = create_adapter(_MockWanVACEPipeline())
        self.assertIsInstance(adapter, WanVACEAdapter)

    def test_encode_video_uses_vae_dtype_and_returns_transformer_dtype(self):
        pipe = _MockWanVACEPipeline()
        pipe.vae = _MockVAE(dtype=torch.float64)
        pipe.transformer = _MockTransformer(dtype=torch.float32)
        adapter = WanVACEAdapter(pipe)
        frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(T)]

        latents = adapter.encode_video(frames)

        self.assertEqual(latents.shape, (B, LAT_C, T, H, W))
        self.assertEqual(latents.dtype, torch.float32)

    def test_forward_transformer_passes_vace_control(self):
        pipe = _MockWanVACEPipeline()
        adapter = WanVACEAdapter(pipe)
        noisy = torch.zeros(1, LAT_C, 2, 4, 6)
        image_cond = torch.zeros(1, LAT_C, 1, 4, 6)
        text_cond = torch.zeros(1, TEXT_LEN, TEXT_D)

        out = adapter.forward_transformer(noisy, torch.tensor([1]), text_cond, image_cond, n_frames_px=5)

        self.assertEqual(out.shape, noisy.shape)
        control = pipe.transformer.last_kwargs["control_hidden_states"]
        self.assertEqual(control.shape, (1, 2 * LAT_C + 64, 2, 4, 6))
        torch.testing.assert_close(
            pipe.transformer.last_kwargs["control_hidden_states_scale"],
            torch.ones(3),
        )

    def test_prepare_reference_condition_uses_reference_only_official_helpers(self):
        pipe = _MockWanVACEPipeline()
        adapter = WanVACEAdapter(pipe)
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        condition = adapter.prepare_reference_condition(frame, T, H, W)

        self.assertEqual(condition["mode"], "reference")
        self.assertEqual(condition["reference_latent_slots"], 1)
        self.assertEqual(
            condition["control_hidden_states"].shape,
            (1, 2 * LAT_C + 64, T + 1, H, W),
        )
        # The video part is the reference-only path's zero-video condition.
        self.assertEqual(
            condition["control_hidden_states"][:, : 2 * LAT_C, 1:].count_nonzero().item(),
            0,
        )

    def test_forward_transformer_accepts_prebuilt_reference_control(self):
        pipe = _MockWanVACEPipeline()
        adapter = WanVACEAdapter(pipe)
        noisy = torch.zeros(1, LAT_C, 3, 4, 6)
        control = torch.zeros(1, 2 * LAT_C + 64, 3, 4, 6)
        condition = {"control_hidden_states": control}

        adapter.forward_transformer(
            noisy,
            torch.tensor([1]),
            torch.zeros(1, TEXT_LEN, TEXT_D),
            condition,
            conditioning_scale=0.75,
        )

        self.assertIs(pipe.transformer.last_kwargs["control_hidden_states"], control)
        torch.testing.assert_close(
            pipe.transformer.last_kwargs["control_hidden_states_scale"],
            torch.full((3,), 0.75),
        )

    def test_encode_text_uses_t5_prompt_embeds(self):
        adapter = WanVACEAdapter(_MockWanVACEPipeline())
        embeds = adapter.encode_text("")

        self.assertEqual(embeds.shape, (1, TEXT_LEN, TEXT_D))


class TestWanVACELoader(unittest.TestCase):
    def test_loader_defaults_to_low_cpu_memory(self):
        import inspect

        signature = inspect.signature(load_wan_vace_pipe)

        self.assertTrue(signature.parameters["low_cpu_memory"].default)

    def test_loader_uses_official_unipc_scheduler_with_flow_shift(self):
        class FakeAutoencoderKLWan:
            @classmethod
            def from_pretrained(cls, model_id, subfolder=None, torch_dtype=None, **kwargs):
                vae = _MockVAE(dtype=torch_dtype)
                vae.model_id = model_id
                vae.subfolder = subfolder
                vae.low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage")
                return vae

        class FakeUniPCMultistepScheduler:
            @classmethod
            def from_config(cls, config, flow_shift=None):
                sched = _MockScheduler()
                sched.config = config
                sched.flow_shift = flow_shift
                return sched

        class FakeWanVACEPipeline(_MockWanVACEPipeline):
            @classmethod
            def from_pretrained(cls, model_id, vae=None, torch_dtype=None, **kwargs):
                pipe = cls()
                pipe.model_id = model_id
                pipe.vae = vae
                pipe.transformer = _MockTransformer(dtype=torch_dtype)
                pipe.scheduler.config = {"prediction_type": "flow_prediction"}
                return pipe

        fake_diffusers = SimpleNamespace(
            AutoencoderKLWan=FakeAutoencoderKLWan,
            UniPCMultistepScheduler=FakeUniPCMultistepScheduler,
            WanVACEPipeline=FakeWanVACEPipeline,
        )

        with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            pipe = load_wan_vace_pipe(
                "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                device="cpu",
                flow_shift=3.0,
                low_cpu_memory=True,
            )

        self.assertIsInstance(pipe, FakeWanVACEPipeline)
        self.assertEqual(pipe.vae.subfolder, "vae")
        self.assertTrue(pipe.vae.low_cpu_mem_usage)
        self.assertEqual(next(pipe.vae.parameters()).dtype, torch.float32)
        self.assertEqual(pipe.transformer.dtype, torch.bfloat16)
        self.assertEqual(pipe.scheduler.flow_shift, 3.0)
        self.assertTrue(pipe.vae.slicing_enabled)
        self.assertTrue(pipe.vae.tiling_enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)

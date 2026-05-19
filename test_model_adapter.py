"""
Unit tests for model_adapter.py.

All tests run on CPU without any real model weights.
Each adapter is tested against a minimal mock pipeline whose components
return tensors of known, predictable shapes.

Run:
    python -m pytest test_model_adapter.py -v
  or
    python test_model_adapter.py
"""

import sys
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, ".")

from model_adapter import (
    ModelAdapter,
    CogVideoXAdapter,
    WanAdapter,
    WanVACEAdapter,
    create_adapter,
    load_wan_pipe,
    _bgr_to_pil,
    _frames_to_tensor,
    _tensor_to_frames,
)


# =========================================================================== #
#  Shared test dimensions                                                       #
# =========================================================================== #

B, C_IMG, T, H, W = 1, 3, 4, 32, 32   # pixel-space
LAT_C    = 4                            # latent channels
LAT_H    = H // 8                       # 4
LAT_W    = W // 8                       # 4
SCALE_F  = 0.18215                      # VAE scaling factor
CLIP_N   = 16                           # CLIP token count
CLIP_D   = 64                           # CLIP embedding dim
TEXT_LEN = 32                           # tokenizer max length
TEXT_D   = 64                           # text embedding dim


# =========================================================================== #
#  Mock building blocks                                                         #
# =========================================================================== #

class _LatentDist:
    """Mimics VAE latent distribution (.latent_dist.sample())."""
    def __init__(self, data: torch.Tensor):
        self._data = data
    def sample(self, generator=None):
        return self._data.clone()


class _VAEEncodeResult:
    def __init__(self, data):
        self.latent_dist = _LatentDist(data)


class _VAEDecodeResult:
    def __init__(self, data):
        self.sample = data


class _MockVAE:
    """
    Identity-like VAE: encode keeps spatial dims, decode restores channel count.
    encode: (1, C_in, T, H, W)  →  latent_dist.sample() → (1, LAT_C, T, H, W)
    decode: (1, LAT_C, T, H, W) →  sample              → (1, C_IMG, T, H, W)
    """
    def __init__(self, lat_c=LAT_C, scale=SCALE_F):
        self.config = SimpleNamespace(scaling_factor=scale)
        self._lat_c  = lat_c

    def encode(self, x: torch.Tensor):
        B_, _, T_, H_, W_ = x.shape
        lat = torch.zeros(B_, self._lat_c, T_, H_, W_, dtype=x.dtype)
        return _VAEEncodeResult(lat)

    def decode(self, x: torch.Tensor):
        B_, _, T_, H_, W_ = x.shape
        out = torch.zeros(B_, C_IMG, T_, H_, W_, dtype=x.dtype)
        return _VAEDecodeResult(out)


class _FeatExtractorOutput(dict):
    """Dict-like output from feature extractor that supports .to()."""
    def to(self, device):
        return self


class _MockFeatureExtractor:
    def __call__(self, images, return_tensors="pt"):
        return _FeatExtractorOutput(pixel_values=torch.zeros(1, 3, 224, 224))


class _ClipOutput:
    def __init__(self, use_last_hidden_state=True):
        self._lhs = use_last_hidden_state
        if use_last_hidden_state:
            self.last_hidden_state = torch.zeros(1, CLIP_N, CLIP_D)
        else:
            self.image_embeds = torch.zeros(1, CLIP_D)


class _MockImageEncoder:
    def __init__(self, use_last_hidden_state=True):
        self._use_lhs = use_last_hidden_state

    def __call__(self, **kwargs):
        return _ClipOutput(self._use_lhs)


class _TokOutput:
    def __init__(self):
        self.input_ids      = torch.zeros(1, TEXT_LEN, dtype=torch.long)
        self.attention_mask = torch.ones(1, TEXT_LEN, dtype=torch.long)

    def to(self, device):
        return self

    # Support **unpacking
    def keys(self):
        return ["input_ids", "attention_mask"]
    def __getitem__(self, k):
        return getattr(self, k)


class _MockTokenizer:
    def __init__(self):
        self.model_max_length = TEXT_LEN

    def __call__(self, text, **kwargs):
        return _TokOutput()


class _TextEncOut:
    def __init__(self):
        self.last_hidden_state = torch.zeros(1, TEXT_LEN, TEXT_D)


class _MockTextEncoder:
    def __call__(self, **kwargs):
        return _TextEncOut()


class _MockScheduler:
    def __init__(self, n=50):
        self.timesteps = torch.linspace(999, 1, n).long()

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.linspace(999, 1, n).long()

    def step(self, velocity, t, latents):
        return SimpleNamespace(prev_sample=latents - 0.01 * velocity)


# --------------------------------------------------------------------------- #
#  CogVideoX mock transformer                                                   #
# Receives (1, 2*LAT_C, T, lH, lW), returns (1, LAT_C, T, lH, lW)            #
# --------------------------------------------------------------------------- #

class _MockCogVideoXTransformer:
    def __init__(self, use_rotary=False):
        self.config = SimpleNamespace(use_rotary_positional_embeddings=use_rotary)
        self.dtype  = torch.float32

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                image_rotary_emb=None, return_dict=False, **kwargs):
        B_, combined_C, T_, lH_, lW_ = hidden_states.shape
        lat_c = combined_C // 2
        vel   = torch.zeros(B_, lat_c, T_, lH_, lW_, dtype=hidden_states.dtype)
        return (vel,)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


# --------------------------------------------------------------------------- #
#  Wan mock transformer                                                         #
# Receives (1, LAT_C, 1+T, lH, lW), returns same shape                        #
# --------------------------------------------------------------------------- #

class _MockWanTransformer:
    def __init__(self):
        self.dtype = torch.float32

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, image_embeds=None,
                return_dict=False, **kwargs):
        vel = torch.zeros_like(hidden_states)
        return (vel,)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _MockWanVACETransformer:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=LAT_C, text_dim=TEXT_D)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                control_hidden_states=None, control_hidden_states_scale=1.0,
                return_dict=False, **kwargs):
        vel = torch.zeros_like(hidden_states)
        return (vel,)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


# --------------------------------------------------------------------------- #
#  Complete mock pipelines                                                      #
# --------------------------------------------------------------------------- #

# Named exactly so that type(pipe).__name__ matches what create_adapter looks for.

class CogVideoXImageToVideoPipeline:
    """Mock CogVideoX I2V pipeline."""
    def __init__(self, use_rotary=False):
        self.device          = torch.device("cpu")
        self.vae             = _MockVAE()
        self.transformer     = _MockCogVideoXTransformer(use_rotary)
        self.tokenizer       = _MockTokenizer()
        self.text_encoder    = _MockTextEncoder()
        self.scheduler       = _MockScheduler()


class WanImageToVideoPipeline:
    """Mock Wan I2V pipeline."""
    def __init__(self, use_lhs=True):
        self.device            = torch.device("cpu")
        self.vae               = _MockVAE()
        self.transformer       = _MockWanTransformer()
        self.tokenizer         = _MockTokenizer()
        self.text_encoder      = _MockTextEncoder()
        self.image_encoder     = _MockImageEncoder(use_lhs)
        self.feature_extractor = _MockFeatureExtractor()
        self.scheduler         = _MockScheduler()


class WanVACEPipeline:
    """Mock Wan VACE pipeline."""
    def __init__(self):
        self.device         = torch.device("cpu")
        self.vae            = _MockVAE()
        self.transformer    = _MockWanVACETransformer()
        self.tokenizer      = _MockTokenizer()
        self.text_encoder   = _MockTextEncoder()
        self.scheduler      = _MockScheduler()


def _cogvideox_pipe(use_rotary=False) -> CogVideoXImageToVideoPipeline:
    return CogVideoXImageToVideoPipeline(use_rotary)


def _wan_pipe(use_lhs=True) -> WanImageToVideoPipeline:
    return WanImageToVideoPipeline(use_lhs)


def _wan_vace_pipe() -> WanVACEPipeline:
    return WanVACEPipeline()


# --------------------------------------------------------------------------- #
#  Helpers for making synthetic test data                                       #
# --------------------------------------------------------------------------- #

def _random_frames(n=T, h=H, w=W) -> list:
    """n random (h, w, 3) BGR uint8 frames."""
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _random_latents(t=T) -> torch.Tensor:
    return torch.randn(B, LAT_C, t, LAT_H, LAT_W)


# =========================================================================== #
#  Tests: shared tensor helpers                                                 #
# =========================================================================== #

class TestBgrToPil(unittest.TestCase):

    def test_channel_swap(self):
        """BGR input → PIL should have R and B channels swapped."""
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:, :, 0] = 10   # B
        arr[:, :, 2] = 200  # R
        pil = _bgr_to_pil(arr)
        pil_arr = np.array(pil)
        self.assertEqual(pil_arr[0, 0, 0], 200)  # PIL R == original B channel 2
        self.assertEqual(pil_arr[0, 0, 2], 10)   # PIL B == original B channel 0

    def test_output_is_rgb(self):
        from PIL import Image
        arr = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        pil = _bgr_to_pil(arr)
        self.assertIsInstance(pil, Image.Image)
        self.assertEqual(pil.mode, "RGB")

    def test_does_not_mutate_input(self):
        arr = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        before = arr.copy()
        _bgr_to_pil(arr)
        np.testing.assert_array_equal(arr, before)


class TestFramesToTensor(unittest.TestCase):

    def test_output_shape(self):
        frames = _random_frames(n=3, h=16, w=24)
        t = _frames_to_tensor(frames, torch.device("cpu"), torch.float32)
        self.assertEqual(t.shape, (1, 3, 3, 16, 24))   # (B, C, T, H, W)

    def test_value_range(self):
        frames = _random_frames()
        t = _frames_to_tensor(frames, torch.device("cpu"), torch.float32)
        self.assertGreaterEqual(t.min().item(), -1.0 - 1e-5)
        self.assertLessEqual(   t.max().item(),  1.0 + 1e-5)

    def test_channel_order_rgb(self):
        """A pure-red BGR frame should have positive R (dim 0) and near-zero B (dim 2)."""
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :, 2] = 255   # red in BGR = channel index 2
        t = _frames_to_tensor([frame], torch.device("cpu"), torch.float32)
        # After BGR→RGB flip: channel 0 = R = 255/127.5-1 = 1.0
        self.assertAlmostEqual(t[0, 0, 0, 0, 0].item(),  1.0, places=3)  # R
        self.assertAlmostEqual(t[0, 2, 0, 0, 0].item(), -1.0, places=3)  # B

    def test_dtype_cast(self):
        frames = _random_frames(n=2)
        t16 = _frames_to_tensor(frames, torch.device("cpu"), torch.float16)
        self.assertEqual(t16.dtype, torch.float16)

    def test_single_frame(self):
        frames = _random_frames(n=1)
        t = _frames_to_tensor(frames, torch.device("cpu"), torch.float32)
        self.assertEqual(t.shape, (1, 3, 1, H, W))


class TestTensorToFrames(unittest.TestCase):

    def test_output_length(self):
        t = torch.zeros(1, 3, T, H, W)
        frames = _tensor_to_frames(t)
        self.assertEqual(len(frames), T)

    def test_output_dtype_shape(self):
        t = torch.zeros(1, 3, 2, H, W)
        frames = _tensor_to_frames(t)
        for f in frames:
            self.assertEqual(f.dtype, np.uint8)
            self.assertEqual(f.shape, (H, W, 3))

    def test_value_clipping(self):
        """Values outside [-1,1] must be clipped to [0,255]."""
        t = torch.full((1, 3, 1, 4, 4), fill_value=5.0)    # >> 1
        frames = _tensor_to_frames(t)
        self.assertTrue((frames[0] == 255).all())

        t2 = torch.full((1, 3, 1, 4, 4), fill_value=-5.0)  # << -1
        frames2 = _tensor_to_frames(t2)
        self.assertTrue((frames2[0] == 0).all())

    def test_channel_order_bgr(self):
        """Tensor channel 0 = R should end up in BGR index 2."""
        # Fill everything with -1.0 (→ 0 in uint8), then set channel 0 (R) to 1.0 (→ 255).
        t = torch.full((1, 3, 1, 4, 4), -1.0)
        t[0, 0] = 1.0    # R channel = 1.0 → pixel value 255
        frames = _tensor_to_frames(t)
        # After BGR reversal: BGR[2] = R = 255, BGR[0] = B = 0
        self.assertEqual(int(frames[0][0, 0, 2]), 255)
        self.assertEqual(int(frames[0][0, 0, 0]),   0)   # B


class TestRoundtrip(unittest.TestCase):

    def test_frames_tensor_frames_roundtrip(self):
        """Encode then decode should recover original frames within ±1 (uint8 rounding)."""
        frames = _random_frames(n=3)
        t = _frames_to_tensor(frames, torch.device("cpu"), torch.float32)
        recovered = _tensor_to_frames(t)
        for orig, rec in zip(frames, recovered):
            np.testing.assert_allclose(
                orig.astype(np.int32), rec.astype(np.int32), atol=1,
                err_msg="roundtrip should be lossless up to 1-level uint8 rounding",
            )


# =========================================================================== #
#  Tests: ModelAdapter shared concrete methods (via minimal subclass)           #
# =========================================================================== #

class _MinimalAdapter(ModelAdapter):
    """Concrete subclass that wires the shared methods to a mock scheduler."""

    def __init__(self, sched=None):
        self._sched = sched or _MockScheduler()

    @property
    def device(self): return torch.device("cpu")
    @property
    def dtype(self):  return torch.float32
    @property
    def scheduler(self): return self._sched

    def encode_video(self, f): raise NotImplementedError
    def decode_latents(self, l): raise NotImplementedError
    def encode_image_cond(self, f): raise NotImplementedError
    def encode_text(self, p): raise NotImplementedError

    def forward_transformer(self, noisy, t, text, img):
        # Returns constant velocity of 1s so we can verify guidance math exactly
        return torch.ones_like(noisy)


class TestAddNoise(unittest.TestCase):

    def setUp(self):
        self.adapter = _MinimalAdapter()

    def test_gamma_zero_returns_latents(self):
        x   = torch.ones(1, 4, 3, 4, 4) * 0.5
        eps = torch.ones(1, 4, 3, 4, 4) * 2.0
        out = self.adapter.add_noise(x, eps, gamma=0.0)
        torch.testing.assert_close(out, x)

    def test_gamma_one_returns_noise(self):
        x   = torch.ones(1, 4, 3, 4, 4) * 0.5
        eps = torch.ones(1, 4, 3, 4, 4) * 2.0
        out = self.adapter.add_noise(x, eps, gamma=1.0)
        torch.testing.assert_close(out, eps)

    def test_gamma_half_is_midpoint(self):
        x   = torch.zeros(1, 4, 3, 4, 4)
        eps = torch.ones(1, 4, 3, 4, 4) * 2.0
        out = self.adapter.add_noise(x, eps, gamma=0.5)
        expected = 0.5 * x + 0.5 * eps
        torch.testing.assert_close(out, expected)

    def test_output_shape(self):
        x   = _random_latents()
        eps = torch.randn_like(x)
        out = self.adapter.add_noise(x, eps, gamma=0.5)
        self.assertEqual(out.shape, x.shape)


class TestPredictWithGuidance(unittest.TestCase):

    def setUp(self):
        self.adapter = _MinimalAdapter()

    def _adapter_with_velocities(self, v_edited, v_original):
        """Return an adapter whose forward_transformer returns v_e or v_o based on cond."""
        class _A(_MinimalAdapter):
            def forward_transformer(self, noisy, t, text, img):
                return v_edited if img == "edited" else v_original
        return _A()

    def test_formula_numerically(self):
        """v̂ = (λ+1)*v_e - λ*v_o  with λ=8."""
        lam = 8.0
        v_e = torch.full((1, 4, 3, 4, 4), 2.0)
        v_o = torch.full((1, 4, 3, 4, 4), 1.0)
        adapter = self._adapter_with_velocities(v_e, v_o)

        noisy = _random_latents()
        t     = torch.tensor([500])
        out   = adapter.predict_with_guidance(noisy, t, "txt", "edited", "original", lam=lam)

        expected = (lam + 1) * v_e - lam * v_o
        torch.testing.assert_close(out, expected)

    def test_lam_zero_equals_v_edited(self):
        """When λ=0, guidance formula reduces to v_edited."""
        v_e = torch.randn(1, 4, 3, 4, 4)
        v_o = torch.randn(1, 4, 3, 4, 4)
        adapter = self._adapter_with_velocities(v_e, v_o)
        noisy   = _random_latents()
        out = adapter.predict_with_guidance(noisy, torch.tensor([1]), "txt", "edited", "original", lam=0.0)
        torch.testing.assert_close(out, v_e)

    def test_equal_velocities_returns_v(self):
        """When v_edited == v_original, guidance has no effect."""
        v = torch.ones(1, 4, 3, 4, 4) * 3.14
        adapter = self._adapter_with_velocities(v, v)
        noisy   = _random_latents()
        out = adapter.predict_with_guidance(noisy, torch.tensor([1]), "txt", "edited", "original", lam=8.0)
        torch.testing.assert_close(out, v)

    def test_output_shape_matches_latents(self):
        noisy   = _random_latents()          # shape (1, LAT_C, T, LAT_H, LAT_W)
        v = torch.zeros_like(noisy)
        adapter = self._adapter_with_velocities(v, v)
        out = adapter.predict_with_guidance(noisy, torch.tensor([1]), "txt", "edited", "original")
        self.assertEqual(out.shape, noisy.shape)


class TestSchedulerWrappers(unittest.TestCase):

    def test_set_timesteps(self):
        sched   = _MockScheduler(n=50)
        adapter = _MinimalAdapter(sched)
        adapter.set_timesteps(20)
        self.assertEqual(len(adapter.timesteps), 20)

    def test_timesteps_property(self):
        sched   = _MockScheduler(n=30)
        adapter = _MinimalAdapter(sched)
        self.assertEqual(len(adapter.timesteps), 30)
        self.assertIsInstance(adapter.timesteps, torch.Tensor)

    def test_scheduler_step(self):
        sched   = _MockScheduler()
        adapter = _MinimalAdapter(sched)
        lat     = torch.ones(1, 4, 3, 4, 4)
        vel     = torch.zeros(1, 4, 3, 4, 4)
        t       = torch.tensor(500)
        out     = adapter.scheduler_step(vel, t, lat)
        # Our mock: prev_sample = latents - 0.01*velocity = latents when vel=0
        torch.testing.assert_close(out, lat)


# =========================================================================== #
#  Tests: CogVideoXAdapter                                                      #
# =========================================================================== #

class TestCogVideoXAdapterProperties(unittest.TestCase):

    def setUp(self):
        self.pipe    = _cogvideox_pipe()
        self.adapter = CogVideoXAdapter(self.pipe)

    def test_device(self):
        self.assertEqual(self.adapter.device, torch.device("cpu"))

    def test_dtype(self):
        self.assertEqual(self.adapter.dtype, torch.float32)

    def test_scheduler_is_pipe_scheduler(self):
        self.assertIs(self.adapter.scheduler, self.pipe.scheduler)


class TestCogVideoXAdapterEncodeVideo(unittest.TestCase):

    def setUp(self):
        self.adapter = CogVideoXAdapter(_cogvideox_pipe())

    def test_output_shape(self):
        frames = _random_frames(n=T)
        lat    = self.adapter.encode_video(frames)
        # Mock VAE keeps spatial dims; shape should be (1, LAT_C, T, H, W)
        self.assertEqual(lat.shape, (B, LAT_C, T, H, W))

    def test_scaled_by_factor(self):
        """Raw latents (all zero from mock) scaled by SCALE_F should still be 0; check non-zero case."""
        # Patch the mock VAE to return a known non-zero latent
        adapter  = CogVideoXAdapter(_cogvideox_pipe())
        sentinel = torch.full((1, LAT_C, T, H, W), 2.0)
        adapter.pipe.vae.encode = lambda x: _VAEEncodeResult(sentinel)

        lat = adapter.encode_video(_random_frames())
        expected = sentinel * SCALE_F
        torch.testing.assert_close(lat, expected)


class TestCogVideoXAdapterDecodeLatents(unittest.TestCase):

    def setUp(self):
        self.adapter = CogVideoXAdapter(_cogvideox_pipe())

    def test_output_list_length(self):
        # Mock VAE decode returns (1, C_IMG=3, T, H, W) — standard (B,C,T,H,W) layout.
        lat = _random_latents(t=T)
        frames = self.adapter.decode_latents(lat)
        self.assertEqual(len(frames), T)

    def test_output_frame_dtype_shape(self):
        lat = _random_latents(t=3)
        frames = self.adapter.decode_latents(lat)
        for f in frames:
            self.assertEqual(f.dtype, np.uint8)
            self.assertEqual(f.shape[2], 3)     # BGR channels

    def test_unscales_before_decode(self):
        """The value passed to vae.decode should be lat / scaling_factor."""
        received = {}
        def _spy_decode(x):
            received["x"] = x.clone()
            return _VAEDecodeResult(torch.zeros(1, C_IMG, x.shape[2], H, W))
        adapter = CogVideoXAdapter(_cogvideox_pipe())
        adapter.pipe.vae.decode = _spy_decode

        lat = torch.full((1, LAT_C, T, H, W), SCALE_F)
        adapter.decode_latents(lat)
        expected_input = lat / SCALE_F   # should be all-ones
        torch.testing.assert_close(received["x"], expected_input)

    def test_handles_btchw_layout(self):
        """If VAE returns (1, T, C=3, H, W), the adapter must permute to (1, C, T, H, W)."""
        def _decode_btchw(x):
            T_ = x.shape[2]
            # Return (1, T, 3, H, W) — note: shape[1]=T ≠ 3, shape[2]=3
            return _VAEDecodeResult(torch.zeros(1, T_, C_IMG, H, W))
        adapter = CogVideoXAdapter(_cogvideox_pipe())
        adapter.pipe.vae.decode = _decode_btchw

        lat    = _random_latents(t=T)   # T=4, so mock returns (1,4,3,H,W)
        frames = adapter.decode_latents(lat)
        self.assertEqual(len(frames), T)
        for f in frames:
            self.assertEqual(f.shape, (H, W, 3))


class TestCogVideoXAdapterEncodeImageCond(unittest.TestCase):

    def setUp(self):
        self.adapter = CogVideoXAdapter(_cogvideox_pipe())

    def test_output_shape_single_frame(self):
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        cond  = self.adapter.encode_image_cond(frame)
        # Should be (1, LAT_C, 1, H, W)  — T=1 single conditioning frame
        self.assertEqual(cond.shape, (1, LAT_C, 1, H, W))

    def test_scaled_by_factor(self):
        sentinel = torch.full((1, LAT_C, 1, H, W), 3.0)
        adapter  = CogVideoXAdapter(_cogvideox_pipe())
        adapter.pipe.vae.encode = lambda x: _VAEEncodeResult(sentinel)

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cond  = adapter.encode_image_cond(frame)
        torch.testing.assert_close(cond, sentinel * SCALE_F)


class TestCogVideoXAdapterEncodeText(unittest.TestCase):

    def setUp(self):
        self.adapter = CogVideoXAdapter(_cogvideox_pipe())

    def test_output_is_tensor(self):
        emb = self.adapter.encode_text("a moving car")
        self.assertIsInstance(emb, torch.Tensor)

    def test_output_shape(self):
        emb = self.adapter.encode_text("")
        # (1, TEXT_LEN, TEXT_D) from our mock
        self.assertEqual(emb.shape, (1, TEXT_LEN, TEXT_D))


def _cogvx_spy_transformer(received: dict, out_shape):
    """Return a CogVideoX transformer class whose __call__ records kwargs."""
    class _Spy(_MockCogVideoXTransformer):
        def __call__(self, **kwargs):
            received.update(kwargs)
            return (torch.zeros(*out_shape),)
    return _Spy()


class TestCogVideoXAdapterForwardTransformer(unittest.TestCase):

    def setUp(self):
        self.adapter = CogVideoXAdapter(_cogvideox_pipe())

    def test_output_shape(self):
        noisy    = _random_latents()
        img_cond = torch.zeros(1, LAT_C, 1, LAT_H, LAT_W)
        text     = torch.zeros(1, TEXT_LEN, TEXT_D)
        t        = torch.tensor([500])
        out = self.adapter.forward_transformer(noisy, t, text, img_cond)
        self.assertEqual(out.shape, noisy.shape)   # (1, LAT_C, T, lH, lW)

    def test_transformer_receives_doubled_channels(self):
        """Transformer hidden_states should have 2*LAT_C channels."""
        received = {}
        adapter  = CogVideoXAdapter(_cogvideox_pipe())
        adapter.pipe.transformer = _cogvx_spy_transformer(received, (1, LAT_C, T, LAT_H, LAT_W))

        noisy = _random_latents()
        img   = torch.zeros(1, LAT_C, 1, LAT_H, LAT_W)
        adapter.forward_transformer(noisy, torch.tensor([1]), torch.zeros(1, TEXT_LEN, TEXT_D), img)
        self.assertEqual(received["hidden_states"].shape[1], 2 * LAT_C)

    def test_image_cond_expanded_over_all_frames(self):
        """Image latent should be expanded to T frames before concat."""
        received = {}
        adapter  = CogVideoXAdapter(_cogvideox_pipe())
        adapter.pipe.transformer = _cogvx_spy_transformer(received, (1, LAT_C, T, LAT_H, LAT_W))

        noisy    = torch.zeros(1, LAT_C, T, LAT_H, LAT_W)
        img_cond = torch.ones(1, LAT_C, 1, LAT_H, LAT_W)   # non-zero sentinel
        adapter.forward_transformer(noisy, torch.tensor([1]), torch.zeros(1, TEXT_LEN, TEXT_D), img_cond)

        hs = received["hidden_states"]
        # Channels [LAT_C:] = image latent (all-ones), all frames
        self.assertTrue((hs[0, LAT_C:, :, :, :] == 1.0).all())

    def test_scalar_timestep_gets_batch_dim(self):
        """Scalar timestep should be unsqueezed to (1,) before passing to transformer."""
        received = {}
        adapter  = CogVideoXAdapter(_cogvideox_pipe())
        adapter.pipe.transformer = _cogvx_spy_transformer(received, (1, LAT_C, T, LAT_H, LAT_W))

        noisy    = _random_latents()
        img      = torch.zeros(1, LAT_C, 1, LAT_H, LAT_W)
        scalar_t = torch.tensor(500)    # ndim == 0
        adapter.forward_transformer(noisy, scalar_t, torch.zeros(1, TEXT_LEN, TEXT_D), img)
        self.assertGreaterEqual(received["timestep"].ndim, 1)


# =========================================================================== #
#  Tests: WanAdapter                                                            #
# =========================================================================== #

class TestWanAdapterProperties(unittest.TestCase):

    def setUp(self):
        self.pipe    = _wan_pipe()
        self.adapter = WanAdapter(self.pipe)

    def test_device(self):
        self.assertEqual(self.adapter.device, torch.device("cpu"))

    def test_dtype(self):
        self.assertEqual(self.adapter.dtype, torch.float32)

    def test_default_img_lat_frames(self):
        self.assertEqual(self.adapter._img_lat_frames, 1)


class TestWanAdapterVaeScale(unittest.TestCase):

    def test_reads_config(self):
        pipe          = _wan_pipe()
        pipe.vae.config.scaling_factor = 0.5
        adapter       = WanAdapter(pipe)
        self.assertAlmostEqual(adapter._vae_scale(), 0.5)

    def test_default_fallback(self):
        pipe          = _wan_pipe()
        del pipe.vae.config.scaling_factor
        adapter       = WanAdapter(pipe)
        self.assertAlmostEqual(adapter._vae_scale(), 1.0)


class TestWanAdapterEncodeVideo(unittest.TestCase):

    def test_output_shape(self):
        adapter = WanAdapter(_wan_pipe())
        frames  = _random_frames(n=T)
        lat     = adapter.encode_video(frames)
        self.assertEqual(lat.shape, (B, LAT_C, T, H, W))


class TestWanAdapterDecodeLatents(unittest.TestCase):

    def test_strips_image_frame(self):
        """decode_latents receives T+1 frames (with prepended image) → outputs T frames."""
        adapter    = WanAdapter(_wan_pipe())
        lat_with_img = _random_latents(t=T + 1)  # 1 image frame + T video frames
        frames = adapter.decode_latents(lat_with_img)
        self.assertEqual(len(frames), T)

    def test_output_dtype_shape(self):
        adapter = WanAdapter(_wan_pipe())
        lat     = _random_latents(t=T + 1)
        frames  = adapter.decode_latents(lat)
        for f in frames:
            self.assertEqual(f.dtype, np.uint8)
            self.assertEqual(f.shape[2], 3)

    def test_unscales_video_latent_only(self):
        """Only the video portion (frames 1:) should be unscaled before decode."""
        received = {}
        def _spy_decode(x):
            received["x"] = x.clone()
            return _VAEDecodeResult(torch.zeros(1, C_IMG, x.shape[2], H, W))

        adapter = WanAdapter(_wan_pipe())
        adapter.pipe.vae.decode = _spy_decode
        adapter.pipe.vae.config.scaling_factor = 2.0

        lat_with_img = torch.full((1, LAT_C, T + 1, H, W), 2.0)
        adapter.decode_latents(lat_with_img)

        # Passed to decode: lat[1:] / 2.0 = 1.0
        self.assertAlmostEqual(received["x"].mean().item(), 1.0, places=5)


class TestWanAdapterEncodeImageCond(unittest.TestCase):

    def setUp(self):
        self.adapter = WanAdapter(_wan_pipe())

    def test_returns_dict_with_correct_keys(self):
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        cond  = self.adapter.encode_image_cond(frame)
        self.assertIn("clip_emb",   cond)
        self.assertIn("vae_latent", cond)

    def test_vae_latent_shape(self):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cond  = self.adapter.encode_image_cond(frame)
        self.assertEqual(cond["vae_latent"].shape, (1, LAT_C, 1, H, W))

    def test_clip_emb_shape_last_hidden_state(self):
        """When image_encoder returns last_hidden_state → (1, N_tokens, D)."""
        adapter = WanAdapter(_wan_pipe(use_lhs=True))
        frame   = np.zeros((H, W, 3), dtype=np.uint8)
        cond    = adapter.encode_image_cond(frame)
        self.assertEqual(cond["clip_emb"].shape, (1, CLIP_N, CLIP_D))

    def test_clip_emb_shape_image_embeds(self):
        """When image_encoder returns image_embeds → unsqueezed to (1, 1, D)."""
        adapter = WanAdapter(_wan_pipe(use_lhs=False))
        frame   = np.zeros((H, W, 3), dtype=np.uint8)
        cond    = adapter.encode_image_cond(frame)
        self.assertEqual(cond["clip_emb"].shape, (1, 1, CLIP_D))


class TestWanAdapterEncodeText(unittest.TestCase):

    def setUp(self):
        self.adapter = WanAdapter(_wan_pipe())

    def test_returns_dict_with_correct_keys(self):
        out = self.adapter.encode_text("")
        self.assertIn("embeds", out)
        self.assertIn("mask",   out)

    def test_embeds_shape(self):
        out = self.adapter.encode_text("a cat")
        self.assertEqual(out["embeds"].shape, (1, TEXT_LEN, TEXT_D))

    def test_mask_shape(self):
        out = self.adapter.encode_text("test")
        self.assertEqual(out["mask"].shape, (1, TEXT_LEN))


class TestWanAdapterPrependImageLatent(unittest.TestCase):

    def setUp(self):
        self.adapter = WanAdapter(_wan_pipe())

    def test_temporal_dim_increases_by_one(self):
        noisy    = _random_latents(t=T)
        img_cond = {"vae_latent": torch.zeros(1, LAT_C, 1, LAT_H, LAT_W)}
        out = self.adapter._prepend_image_latent(noisy, img_cond)
        self.assertEqual(out.shape[2], T + 1)

    def test_first_frame_is_image_latent(self):
        noisy    = torch.zeros(1, LAT_C, T, LAT_H, LAT_W)
        img_lat  = torch.ones(1, LAT_C, 1, LAT_H, LAT_W) * 7.0
        img_cond = {"vae_latent": img_lat}
        out = self.adapter._prepend_image_latent(noisy, img_cond)
        torch.testing.assert_close(out[:, :, 0:1, :, :], img_lat)
        torch.testing.assert_close(out[:, :, 1:,  :, :], noisy)

    def test_spatial_interpolation_when_sizes_differ(self):
        """If image latent has different spatial dims, it should be resized."""
        noisy    = torch.zeros(1, LAT_C, T, LAT_H, LAT_W)          # lH x lW
        img_lat  = torch.ones(1,  LAT_C, 1, LAT_H * 2, LAT_W * 2)  # 2x bigger
        img_cond = {"vae_latent": img_lat}
        out = self.adapter._prepend_image_latent(noisy, img_cond)
        # After resize, spatial dims should match noisy
        self.assertEqual(out.shape[3], LAT_H)
        self.assertEqual(out.shape[4], LAT_W)


class TestWanAdapterForwardTransformer(unittest.TestCase):

    def setUp(self):
        self.adapter = WanAdapter(_wan_pipe())

    def _make_cond(self):
        return {
            "clip_emb":   torch.zeros(1, CLIP_N, CLIP_D),
            "vae_latent": torch.zeros(1, LAT_C, 1, LAT_H, LAT_W),
        }

    def _make_text(self):
        return {
            "embeds": torch.zeros(1, TEXT_LEN, TEXT_D),
            "mask":   torch.ones(1, TEXT_LEN, dtype=torch.long),
        }

    def test_output_shape_strips_image_frame(self):
        noisy = _random_latents(t=T)
        out   = self.adapter.forward_transformer(
            noisy, torch.tensor([500]), self._make_text(), self._make_cond()
        )
        self.assertEqual(out.shape, noisy.shape)   # (1, LAT_C, T, lH, lW)

    def test_transformer_receives_T_plus_one_frames(self):
        """Transformer input should have T+1 temporal frames (prepended image)."""
        received = {}

        class _SpyWan(_MockWanTransformer):
            def __call__(self, **kwargs):
                received["T_in"] = kwargs["hidden_states"].shape[2]
                return (torch.zeros(1, LAT_C, T + 1, LAT_H, LAT_W),)

        adapter = WanAdapter(_wan_pipe())
        adapter.pipe.transformer = _SpyWan()

        noisy = _random_latents(t=T)
        adapter.forward_transformer(noisy, torch.tensor([1]), self._make_text(), self._make_cond())
        self.assertEqual(received["T_in"], T + 1)

    def test_image_embeds_passed_to_transformer(self):
        """clip_emb from image_cond must be forwarded as image_embeds."""
        received = {}

        class _SpyWan(_MockWanTransformer):
            def __call__(self, **kwargs):
                received["image_embeds"] = kwargs.get("image_embeds")
                return (torch.zeros(1, LAT_C, T + 1, LAT_H, LAT_W),)

        adapter  = WanAdapter(_wan_pipe())
        adapter.pipe.transformer = _SpyWan()

        clip_emb = torch.full((1, CLIP_N, CLIP_D), 3.14)
        cond = {"clip_emb": clip_emb, "vae_latent": torch.zeros(1, LAT_C, 1, LAT_H, LAT_W)}
        adapter.forward_transformer(_random_latents(), torch.tensor([1]), self._make_text(), cond)
        torch.testing.assert_close(received["image_embeds"], clip_emb)


# =========================================================================== #
#  Tests: WanVACEAdapter                                                       #
# =========================================================================== #

class TestWanVACEAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = WanVACEAdapter(_wan_vace_pipe())

    def _make_text(self):
        return {
            "embeds": torch.zeros(1, TEXT_LEN, TEXT_D),
            "mask":   torch.ones(1, TEXT_LEN, dtype=torch.long),
        }

    def test_encode_video_records_shape_metadata(self):
        frames = _random_frames(n=T)
        lat = self.adapter.encode_video(frames)
        self.assertEqual(lat.shape, (B, LAT_C, T, H, W))
        self.assertEqual(self.adapter._last_num_frames, T)
        self.assertEqual(self.adapter._last_height, H)
        self.assertEqual(self.adapter._last_width, W)
        self.assertEqual(self.adapter._last_latent_frames, T)

    def test_fallback_image_cond_uses_vace_control_key(self):
        self.adapter.encode_video(_random_frames(n=T))
        cond = self.adapter.encode_image_cond(np.zeros((H, W, 3), dtype=np.uint8))
        self.assertIn("control", cond)
        self.assertEqual(cond["control"].shape, (B, LAT_C + 1, T, H, W))
        mask = cond["control"][:, LAT_C:]
        self.assertEqual(mask[0, 0, 0].max().item(), 0.0)
        self.assertEqual(mask[0, 0, 1:].min().item(), 1.0)

    def test_forward_transformer_passes_control_hidden_states(self):
        received = {}

        class _SpyVACE(_MockWanVACETransformer):
            def __call__(self, **kwargs):
                received["control"] = kwargs.get("control_hidden_states")
                received["scale"] = kwargs.get("control_hidden_states_scale")
                return (torch.zeros_like(kwargs["hidden_states"]),)

        adapter = WanVACEAdapter(_wan_vace_pipe())
        adapter.pipe.transformer = _SpyVACE()
        control = torch.ones(1, LAT_C + 1, T, LAT_H, LAT_W)
        cond = {"control": control, "scale": 0.75}

        noisy = _random_latents(t=T)
        out = adapter.forward_transformer(noisy, torch.tensor([1]), self._make_text(), cond)

        self.assertEqual(out.shape, noisy.shape)
        torch.testing.assert_close(received["control"], control)
        self.assertEqual(received["scale"], 0.75)

    def test_encode_text_without_text_encoder_returns_zeros_for_empty_prompt(self):
        pipe = _wan_vace_pipe()
        pipe.text_encoder = None
        pipe.tokenizer = None
        adapter = WanVACEAdapter(pipe)

        out = adapter.encode_text("")

        self.assertEqual(out["embeds"].shape, (1, 512, TEXT_D))
        self.assertEqual(out["mask"].shape, (1, 512))
        self.assertEqual(out["embeds"].abs().sum().item(), 0.0)
        self.assertEqual(out["mask"].sum().item(), 0)

    def test_encode_text_without_text_encoder_rejects_non_empty_prompt(self):
        pipe = _wan_vace_pipe()
        pipe.text_encoder = None
        pipe.tokenizer = None
        adapter = WanVACEAdapter(pipe)

        with self.assertRaisesRegex(ValueError, "text encoder"):
            adapter.encode_text("a prompt")


# =========================================================================== #
#  Tests: create_adapter factory                                                #
# =========================================================================== #

class TestCreateAdapter(unittest.TestCase):

    def test_cogvideox_class_name(self):
        pipe    = _cogvideox_pipe()
        adapter = create_adapter(pipe)
        self.assertIsInstance(adapter, CogVideoXAdapter)

    def test_wan_class_name(self):
        pipe    = _wan_pipe()
        adapter = create_adapter(pipe)
        self.assertIsInstance(adapter, WanAdapter)

    def test_wan_vace_class_name(self):
        pipe    = _wan_vace_pipe()
        adapter = create_adapter(pipe)
        self.assertIsInstance(adapter, WanVACEAdapter)

    def test_heuristic_wan_signature(self):
        """Unknown class but Wan-like transformer signature → WanAdapter."""
        def _wan_fwd(hidden_states, timestep, encoder_hidden_states,
                     encoder_attention_mask=None, image_embeds=None, return_dict=True):
            pass

        class _WanLikeTransformer:
            forward = _wan_fwd
            dtype   = torch.float32

        class SomeUnknownPipeline:      # name ∉ _COGVIDEOX_CLASSES, _WAN_CLASSES
            transformer = _WanLikeTransformer()
            scheduler   = _MockScheduler()

        adapter = create_adapter(SomeUnknownPipeline())
        self.assertIsInstance(adapter, WanAdapter)

    def test_heuristic_wan_vace_signature(self):
        """Unknown class but VACE-like transformer signature → WanVACEAdapter."""
        def _vace_fwd(hidden_states, timestep, encoder_hidden_states,
                      control_hidden_states=None, control_hidden_states_scale=1.0,
                      return_dict=True):
            pass

        class _VaceLikeTransformer:
            forward = _vace_fwd
            dtype   = torch.float32

        class SomeUnknownPipeline:
            transformer = _VaceLikeTransformer()
            scheduler   = _MockScheduler()

        adapter = create_adapter(SomeUnknownPipeline())
        self.assertIsInstance(adapter, WanVACEAdapter)

    def test_heuristic_cogvideox_signature(self):
        """Unknown class with CogVideoX-like transformer signature → CogVideoXAdapter."""
        def _cogvx_fwd(hidden_states, encoder_hidden_states, timestep,
                       image_rotary_emb=None, return_dict=True):
            pass

        class _CogVXLikeTransformer:
            forward = _cogvx_fwd
            dtype   = torch.float32
            config  = SimpleNamespace(use_rotary_positional_embeddings=False)

        class AnotherUnknownPipeline:
            transformer = _CogVXLikeTransformer()
            scheduler   = _MockScheduler()

        adapter = create_adapter(AnotherUnknownPipeline())
        self.assertIsInstance(adapter, CogVideoXAdapter)

    def test_returns_model_adapter_subclass(self):
        for pipe in [_cogvideox_pipe(), _wan_pipe()]:
            adapter = create_adapter(pipe)
            self.assertIsInstance(adapter, ModelAdapter)

    def test_loads_vace_pipeline_class(self):
        class LoadableWanVACEPipeline(WanVACEPipeline):
            @classmethod
            def from_pretrained(cls, model_id, torch_dtype=None, **kwargs):
                pipe = cls()
                pipe.model_id = model_id
                pipe.torch_dtype = torch_dtype
                pipe.from_pretrained_kwargs = kwargs
                pipe.to_device = None
                return pipe

            def to(self, device):
                self.to_device = device
                return self

            def enable_model_cpu_offload(self):
                self.cpu_offload = True

        fake_diffusers = SimpleNamespace(WanVACEPipeline=LoadableWanVACEPipeline)
        with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            pipe = load_wan_pipe("Wan-AI/Wan2.1-VACE-1.3B-diffusers", device="cpu")

        self.assertIsInstance(pipe, LoadableWanVACEPipeline)
        self.assertEqual(pipe.model_id, "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
        self.assertIsNone(pipe.from_pretrained_kwargs["text_encoder"])
        self.assertIsNone(pipe.from_pretrained_kwargs["tokenizer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

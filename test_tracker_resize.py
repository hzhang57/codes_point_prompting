import unittest
from unittest.mock import patch

import numpy as np
import torch

from model_adapter import ModelAdapter
from tracker import PointPrompter, PointPrompterConfig, _aligned_size


class _DummyAdapter(ModelAdapter):
    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    @property
    def scheduler(self):
        return None

    def encode_video(self, frames_bgr):
        raise NotImplementedError

    def decode_latents(self, latents):
        raise NotImplementedError

    def encode_image_cond(self, frame_bgr):
        raise NotImplementedError

    def encode_text(self, prompt):
        raise NotImplementedError

    def forward_transformer(self, noisy_latents, timestep, text_cond, image_cond):
        raise NotImplementedError


class TestTrackerResize(unittest.TestCase):
    def test_aligned_size_fits_within_limit(self):
        width, height = _aligned_size(2314, 1270, 832, 480, 16)
        self.assertLessEqual(width, 832)
        self.assertLessEqual(height, 480)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_track_resizes_model_input_and_restores_track_coordinates(self):
        frames = [np.zeros((1270, 2314, 3), dtype=np.uint8) for _ in range(2)]
        cfg = PointPrompterConfig(
            do_refine=False,
            model_width=832,
            model_height=480,
            model_stride=16,
        )
        tracker = PointPrompter(_DummyAdapter(), cfg)

        received = {}

        def _fake_sdedit(adapter, frames_bgr_edited, frame_bgr_original, **kwargs):
            received["shape"] = frames_bgr_edited[0].shape[:2]
            return frames_bgr_edited

        with patch("tracker.run_sdedit", side_effect=_fake_sdedit):
            result = tracker.track(frames, (320.0, 240.0))

        self.assertEqual(received["shape"], (448, 832))
        np.testing.assert_allclose(result.tracks[0], np.array([320.0, 240.0]), atol=1e-4)


if __name__ == "__main__":
    unittest.main()

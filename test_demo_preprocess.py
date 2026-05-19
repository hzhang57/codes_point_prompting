import unittest
from unittest.mock import patch

import numpy as np

from demo import cuda_preflight_error, resize_video, scale_points


class TestDemoPreprocess(unittest.TestCase):
    def test_resize_video_to_fixed_resolution(self):
        frames = [np.zeros((1270, 2314, 3), dtype=np.uint8) for _ in range(2)]
        resized = resize_video(frames, 832, 480)
        self.assertEqual(len(resized), 2)
        self.assertEqual(resized[0].shape, (480, 832, 3))

    def test_scale_points_to_preprocessed_resolution(self):
        points = [(320.0, 240.0), (640.0, 360.0)]
        scaled = scale_points(points, src_size=(2314, 1270), dst_size=(832, 480))
        self.assertAlmostEqual(scaled[0][0], 320.0 * 832 / 2314)
        self.assertAlmostEqual(scaled[0][1], 240.0 * 480 / 1270)
        self.assertAlmostEqual(scaled[1][0], 640.0 * 832 / 2314)
        self.assertAlmostEqual(scaled[1][1], 360.0 * 480 / 1270)

    def test_cuda_preflight_skips_cpu(self):
        self.assertIsNone(cuda_preflight_error("cpu"))

    @patch("demo.torch.cuda.is_available", return_value=False)
    def test_cuda_preflight_reports_unavailable_cuda(self, _mock_available):
        msg = cuda_preflight_error("cuda")
        self.assertIn("torch.cuda.is_available", msg)

    @patch("demo.torch.cuda.is_available", return_value=True)
    @patch("demo.torch.cuda.current_device", return_value=0)
    @patch("demo.torch.cuda.get_device_name", return_value="Old GPU")
    @patch("demo.torch.cuda.get_device_capability", return_value=(6, 0))
    @patch("demo.torch.ones", side_effect=RuntimeError("no kernel image is available"))
    def test_cuda_preflight_reports_kernel_arch_error(
        self,
        _mock_ones,
        _mock_capability,
        _mock_name,
        _mock_current_device,
        _mock_available,
    ):
        msg = cuda_preflight_error("cuda")
        self.assertIn("不能在这张 GPU", msg)
        self.assertIn("no kernel image", msg)


if __name__ == "__main__":
    unittest.main()

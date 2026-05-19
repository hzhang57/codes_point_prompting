import unittest

import numpy as np

from demo import resize_video, scale_points


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


if __name__ == "__main__":
    unittest.main()

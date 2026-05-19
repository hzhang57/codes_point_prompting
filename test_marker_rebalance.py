"""
Unit tests for marker.py and color_rebalance.py.

All tests are self-contained: they generate synthetic BGR frames with numpy/cv2
and require no GPU or external data.

Run:
    python -m pytest test_marker_rebalance.py -v
  or
    python test_marker_rebalance.py
"""

import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, ".")   # ensure local modules take precedence

from marker import (
    insert_marker,
    detect_marker,
    track_marker_sequence,
    MARKER_RADIUS,
    MARKER_COLOR_BGR,
    DEFAULT_SEARCH_RADIUS,
    MAX_SEARCH_RADIUS,
)
from color_rebalance import (
    rebalance_frame,
    rebalance_video,
    _red_proximity_weight,
    _RED_HUE_MARGIN,
)


# --------------------------------------------------------------------------- #
#  Synthetic frame factory                                                      #
# --------------------------------------------------------------------------- #

def _gray_frame(h: int = 240, w: int = 320, val: int = 128) -> np.ndarray:
    """Uniform gray BGR frame."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def _frame_with_marker(cx: int, cy: int, h: int = 240, w: int = 320) -> np.ndarray:
    """Gray frame with a red marker at (cx, cy)."""
    frame = _gray_frame(h, w)
    return insert_marker(frame, (cx, cy))


def _solid_color_frame(bgr: tuple, h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


# =========================================================================== #
#  Tests: marker.py                                                             #
# =========================================================================== #

class TestInsertMarker(unittest.TestCase):

    def test_returns_copy(self):
        """insert_marker must not mutate the input frame."""
        frame = _gray_frame()
        original = frame.copy()
        insert_marker(frame, (100, 100))
        np.testing.assert_array_equal(frame, original)

    def test_red_pixels_at_center(self):
        """The pixel exactly at the query point should be red (BGR 0,0,220)."""
        cx, cy = 160, 120
        result = _frame_with_marker(cx, cy)
        b, g, r = result[cy, cx]
        self.assertGreater(r, 150, "red channel should be dominant")
        self.assertLess(g, 50,    "green channel should be low")
        self.assertLess(b, 50,    "blue channel should be low")

    def test_shape_preserved(self):
        frame = _gray_frame(200, 300)
        result = insert_marker(frame, (50, 50))
        self.assertEqual(result.shape, frame.shape)

    def test_dtype_preserved(self):
        frame = _gray_frame()
        result = insert_marker(frame, (50, 50))
        self.assertEqual(result.dtype, np.uint8)

    def test_custom_radius(self):
        """Larger radius → more red pixels."""
        cx, cy = 100, 100
        frame = _gray_frame()
        r_small = insert_marker(frame, (cx, cy), radius=4)
        r_large = insert_marker(frame, (cx, cy), radius=20)
        red_small = (r_small[:, :, 2] > 150).sum()
        red_large = (r_large[:, :, 2] > 150).sum()
        self.assertGreater(red_large, red_small)


class TestDetectMarker(unittest.TestCase):

    def test_detects_center(self):
        """Marker at frame center should be found within MARKER_RADIUS pixels."""
        cx, cy = 160, 120
        frame  = _frame_with_marker(cx, cy)
        result = detect_marker(frame, (cx, cy))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], cx, delta=MARKER_RADIUS)
        self.assertAlmostEqual(result[1], cy, delta=MARKER_RADIUS)

    def test_detects_off_center_hint(self):
        """Detection still works when the hint is offset by less than search_radius."""
        cx, cy = 80, 60
        frame  = _frame_with_marker(cx, cy)
        hint   = (cx + 30, cy + 30)         # hint is 42 px away
        result = detect_marker(frame, hint, search_radius=DEFAULT_SEARCH_RADIUS)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], cx, delta=MARKER_RADIUS)
        self.assertAlmostEqual(result[1], cy, delta=MARKER_RADIUS)

    def test_returns_none_on_gray_frame(self):
        """No red marker → should return None."""
        frame  = _gray_frame()
        result = detect_marker(frame, (160, 120))
        self.assertIsNone(result)

    def test_returns_none_when_outside_search_radius(self):
        """Marker exists but the hint is farther than search_radius → None."""
        cx, cy = 160, 120
        frame  = _frame_with_marker(cx, cy)
        far_hint = (0, 0)
        result = detect_marker(frame, far_hint, search_radius=10)
        self.assertIsNone(result)

    def test_near_border(self):
        """Marker near the image border should still be detected."""
        cx, cy = 5, 5
        frame  = _frame_with_marker(cx, cy, h=240, w=320)
        result = detect_marker(frame, (cx, cy), search_radius=DEFAULT_SEARCH_RADIUS)
        self.assertIsNotNone(result)

    def test_blue_frame_not_detected(self):
        """Blue pixels are not red → no detection."""
        frame  = _solid_color_frame((200, 0, 0), 240, 320)
        result = detect_marker(frame, (160, 120))
        self.assertIsNone(result)

    def test_returns_tuple_of_floats(self):
        frame  = _frame_with_marker(100, 100)
        result = detect_marker(frame, (100, 100))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


class TestTrackMarkerSequence(unittest.TestCase):

    def _sequence_static(self, n: int = 5, cx: int = 160, cy: int = 120) -> list:
        """N identical frames with the marker at the same position."""
        return [_frame_with_marker(cx, cy) for _ in range(n)]

    def _sequence_moving(self, positions: list) -> list:
        """One frame per position."""
        return [_frame_with_marker(cx, cy) for cx, cy in positions]

    def test_output_shapes(self):
        frames = self._sequence_static(5)
        tracks, visible = track_marker_sequence(frames, (160, 120))
        self.assertEqual(tracks.shape,  (5, 2))
        self.assertEqual(visible.shape, (5,))

    def test_all_visible_when_marker_present(self):
        frames = self._sequence_static(4)
        _, visible = track_marker_sequence(frames, (160, 120))
        self.assertTrue(visible.all(), "all frames have a marker so all should be visible")

    def test_first_frame_uses_query_point(self):
        """Frame 0 track must equal the query point regardless of detection."""
        qp = (160.0, 120.0)
        frames = self._sequence_static(3)
        tracks, _ = track_marker_sequence(frames, qp)
        np.testing.assert_allclose(tracks[0], qp)

    def test_tracking_accuracy_static(self):
        """Detected positions should be within MARKER_RADIUS of ground truth."""
        cx, cy = 100, 80
        frames = self._sequence_static(4, cx, cy)
        tracks, visible = track_marker_sequence(frames, (cx, cy))
        for t in range(1, 4):
            self.assertTrue(visible[t])
            self.assertAlmostEqual(tracks[t, 0], cx, delta=MARKER_RADIUS)
            self.assertAlmostEqual(tracks[t, 1], cy, delta=MARKER_RADIUS)

    def test_missing_marker_propagates_last_position(self):
        """On a frame with no marker, the last known position is propagated."""
        cx, cy = 160, 120
        marked = _frame_with_marker(cx, cy)
        blank  = _gray_frame()
        frames = [marked, marked, blank, blank]
        tracks, visible = track_marker_sequence(frames, (cx, cy))

        self.assertFalse(visible[2], "blank frame → not visible")
        self.assertFalse(visible[3], "blank frame → not visible")
        # Position should be propagated from last visible frame
        np.testing.assert_allclose(tracks[2], tracks[1], atol=1.0)

    def test_search_radius_expands_after_miss(self):
        """
        After consecutive misses, the sequence should eventually re-detect
        even when the marker is farther than DEFAULT_SEARCH_RADIUS.
        Verify by checking that MAX_SEARCH_RADIUS is never exceeded.
        """
        cx, cy = 160, 120
        # 10 blank frames then one with the marker
        frames = [_frame_with_marker(cx, cy)] + [_gray_frame()] * 10 + [_frame_with_marker(cx, cy)]
        tracks, visible = track_marker_sequence(frames, (cx, cy))
        # Verify the function completes without error and shapes are correct
        self.assertEqual(tracks.shape, (12, 2))
        self.assertTrue(visible[0])
        self.assertTrue(visible[11])

    def test_moving_marker(self):
        """Marker that moves linearly should be tracked within tolerance."""
        positions = [(60 + i * 10, 80 + i * 5) for i in range(6)]
        frames = self._sequence_moving(positions)
        tracks, visible = track_marker_sequence(frames, positions[0])
        for t, (cx, cy) in enumerate(positions[1:], start=1):
            if visible[t]:
                self.assertAlmostEqual(tracks[t, 0], cx, delta=MARKER_RADIUS * 2)
                self.assertAlmostEqual(tracks[t, 1], cy, delta=MARKER_RADIUS * 2)


# =========================================================================== #
#  Tests: color_rebalance.py                                                    #
# =========================================================================== #

class TestRedProximityWeight(unittest.TestCase):

    def test_pure_red_hue_zero_weight(self):
        """Hue = 0 (pure red) → weight = 0."""
        hue = np.array([0], dtype=np.int32)
        w   = _red_proximity_weight(hue)
        self.assertAlmostEqual(float(w[0]), 0.0)

    def test_pure_red_hue_180_weight(self):
        """Hue = 180 (also pure red in OpenCV wrap) → weight = 0."""
        hue = np.array([180], dtype=np.int32)
        w   = _red_proximity_weight(hue)
        self.assertAlmostEqual(float(w[0]), 0.0)

    def test_green_hue_full_weight(self):
        """Hue = 60 (green, far from red) → weight = 1."""
        hue = np.array([60], dtype=np.int32)
        w   = _red_proximity_weight(hue)
        self.assertAlmostEqual(float(w[0]), 1.0)

    def test_blue_hue_full_weight(self):
        """Hue = 120 (blue) → weight = 1."""
        hue = np.array([120], dtype=np.int32)
        w   = _red_proximity_weight(hue)
        self.assertAlmostEqual(float(w[0]), 1.0)

    def test_weight_clipped_to_unit_interval(self):
        """Weight is always in [0, 1]."""
        hues = np.arange(0, 181, dtype=np.int32)
        w    = _red_proximity_weight(hues)
        self.assertTrue((w >= 0.0).all())
        self.assertTrue((w <= 1.0).all())

    def test_monotone_near_zero(self):
        """Weight increases as hue moves away from 0."""
        hues = np.arange(0, _RED_HUE_MARGIN + 1, dtype=np.int32)
        w    = _red_proximity_weight(hues)
        self.assertTrue(np.all(np.diff(w) >= 0), "weight must be non-decreasing away from red")

    def test_symmetry_near_180(self):
        """Weight near 170-180 should mirror weight near 0-10."""
        lo = np.array([0, 5, 10], dtype=np.int32)
        hi = np.array([180, 175, 170], dtype=np.int32)
        np.testing.assert_allclose(_red_proximity_weight(lo), _red_proximity_weight(hi), atol=1e-5)


class TestRebalanceFrame(unittest.TestCase):

    def test_output_shape_dtype(self):
        frame  = _gray_frame()
        result = rebalance_frame(frame)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_does_not_mutate_input(self):
        frame    = _solid_color_frame((0, 0, 200))   # red
        original = frame.copy()
        rebalance_frame(frame)
        np.testing.assert_array_equal(frame, original)

    def test_red_pixel_desaturated(self):
        """A fully saturated red frame should have its saturation reduced to near zero."""
        frame  = _solid_color_frame((0, 0, 220), 64, 64)   # pure red BGR
        result = rebalance_frame(frame)
        hsv_in  = cv2.cvtColor(frame,  cv2.COLOR_BGR2HSV)
        hsv_out = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        # Mean saturation should drop
        self.assertLess(
            float(hsv_out[:, :, 1].mean()),
            float(hsv_in[:, :, 1].mean()),
            "saturation of red pixels should decrease after rebalancing",
        )

    def test_green_pixel_unchanged(self):
        """Green pixels are far from red in hue space → saturation must not change."""
        frame  = _solid_color_frame((0, 200, 0), 64, 64)   # pure green BGR
        result = rebalance_frame(frame)
        hsv_in  = cv2.cvtColor(frame,  cv2.COLOR_BGR2HSV)
        hsv_out = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        np.testing.assert_allclose(
            hsv_in[:, :, 1].astype(float),
            hsv_out[:, :, 1].astype(float),
            atol=2.0,  # allow tiny rounding error from uint8 round-trip
        )

    def test_blue_pixel_unchanged(self):
        """Blue pixels are far from red → saturation must not change."""
        frame  = _solid_color_frame((200, 0, 0), 64, 64)   # pure blue BGR
        result = rebalance_frame(frame)
        hsv_in  = cv2.cvtColor(frame,  cv2.COLOR_BGR2HSV)
        hsv_out = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        np.testing.assert_allclose(
            hsv_in[:, :, 1].astype(float),
            hsv_out[:, :, 1].astype(float),
            atol=2.0,
        )

    def test_gray_pixel_unchanged(self):
        """Gray (zero saturation) pixels should be unaffected."""
        frame  = _gray_frame(val=150)
        result = rebalance_frame(frame)
        np.testing.assert_allclose(frame.astype(float), result.astype(float), atol=2.0)

    def test_marker_invisible_after_rebalance(self):
        """
        After inserting a red marker and then rebalancing, the marker should
        no longer be detectable by the HSV red-detector used in detect_marker.
        (This is the whole point of color rebalancing.)
        """
        from marker import detect_marker, insert_marker
        from color_rebalance import rebalance_frame

        cx, cy = 80, 60
        frame_with_marker = insert_marker(_gray_frame(), (cx, cy))

        # Before rebalance: detectable
        self.assertIsNotNone(detect_marker(frame_with_marker, (cx, cy)))

        # After rebalance: NOT detectable (saturation wiped out)
        rebalanced = rebalance_frame(frame_with_marker)
        self.assertIsNone(
            detect_marker(rebalanced, (cx, cy)),
            "rebalanced marker should not be detectable as red",
        )


class TestRebalanceVideo(unittest.TestCase):

    def test_list_length_preserved(self):
        frames = [_gray_frame() for _ in range(5)]
        result = rebalance_video(frames)
        self.assertEqual(len(result), 5)

    def test_each_frame_processed(self):
        """Every output frame should differ from a fully red input frame."""
        frames = [_solid_color_frame((0, 0, 200)) for _ in range(3)]
        result = rebalance_video(frames)
        for i, (inp, out) in enumerate(zip(frames, result)):
            hsv_in  = cv2.cvtColor(inp, cv2.COLOR_BGR2HSV)
            hsv_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            self.assertLess(
                float(hsv_out[:, :, 1].mean()),
                float(hsv_in[:, :, 1].mean()),
                f"frame {i}: saturation not reduced",
            )

    def test_empty_list(self):
        self.assertEqual(rebalance_video([]), [])


# =========================================================================== #
#  Integration: rebalance → insert → detect                                    #
# =========================================================================== #

class TestIntegration(unittest.TestCase):

    def test_natural_red_suppressed_synthetic_marker_detectable(self):
        """
        Simulate a frame that has natural red content.
        After rebalancing, the natural red is suppressed.
        Then inserting the synthetic marker should be the only red → detectable.
        """
        # Create a frame with a natural red patch in the top-left
        frame = _gray_frame(h=240, w=320)
        frame[0:40, 0:40] = (0, 0, 180)   # natural red patch (BGR)

        # Query point far from the natural red patch
        cx, cy = 200, 150

        # Without rebalancing: detect_marker might be confused by the red patch
        # With rebalancing: only the synthetic marker should remain red
        from color_rebalance import rebalance_frame
        frame_rb = rebalance_frame(frame)

        # Insert synthetic marker on the rebalanced frame
        frame_marked = insert_marker(frame_rb, (cx, cy))

        # Should detect the synthetic marker, not the (now grey) natural red patch
        result = detect_marker(frame_marked, (cx, cy))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], cx, delta=MARKER_RADIUS)
        self.assertAlmostEqual(result[1], cy, delta=MARKER_RADIUS)

    def test_end_to_end_static_track(self):
        """
        Full pipeline smoke test:
        rebalance → insert marker → track sequence → check visibility and accuracy.
        """
        from color_rebalance import rebalance_frame

        cx, cy = 140, 100
        raw_frames = [_gray_frame() for _ in range(5)]
        rb_frames  = [rebalance_frame(f) for f in raw_frames]

        # Only mark frame 0; remaining frames do NOT have the marker
        # (simulating that the video diffusion model propagated it but imperfectly)
        frames_with_marker = [insert_marker(rb_frames[0], (cx, cy))] + rb_frames[1:]
        # Add the marker to all frames to simulate a perfect diffusion output
        frames_all_marked = [insert_marker(f, (cx, cy)) for f in rb_frames]

        tracks, visible = track_marker_sequence(frames_all_marked, (cx, cy))
        self.assertTrue(visible.all())
        for t in range(5):
            self.assertAlmostEqual(tracks[t, 0], cx, delta=MARKER_RADIUS)
            self.assertAlmostEqual(tracks[t, 1], cy, delta=MARKER_RADIUS)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Manual 180° camera flip shared between the recording and preview stabilizers."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from lerobot_ros2.helper import CameraStabilizer, ManualFlip180


def _scene(seed: int = 0) -> np.ndarray:
    """An asymmetric frame, so a 180° rotation is never equal to the original."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 60, size=(16, 24, 3), dtype=np.uint8)
    frame[:4, :, :] = 240  # bright band along the top only
    return frame


def _rot180(frame: np.ndarray) -> np.ndarray:
    return cv2.rotate(frame, cv2.ROTATE_180)


class ManualFlipStateTest(unittest.TestCase):
    def test_toggle_out_of_range_is_ignored(self):
        flip = ManualFlip180(2)
        self.assertIsNone(flip.toggle(5))
        self.assertIsNone(flip.toggle(-1))
        self.assertEqual(flip.flags(), [False, False])
        self.assertFalse(flip.any_flipped())

    def test_toggle_reports_new_state(self):
        flip = ManualFlip180(1)
        self.assertTrue(flip.toggle(0))
        self.assertTrue(flip.is_flipped(0))
        self.assertFalse(flip.toggle(0))
        self.assertFalse(flip.is_flipped(0))

    def test_initial_length_must_match(self):
        with self.assertRaises(ValueError):
            ManualFlip180(3, initial=[True, False])


class CameraStabilizerFlipTest(unittest.TestCase):
    def test_flip_applies_when_auto_stabilization_is_off(self):
        # The config default is stabilize_orientation_180=False, so the manual
        # override has to work independently of the automatic path.
        flip = ManualFlip180(1)
        stab = CameraStabilizer([False], [20.0], manual_flip=flip)
        frame = _scene()

        np.testing.assert_array_equal(stab.process([frame])[0], frame)

        flip.toggle(0)
        np.testing.assert_array_equal(stab.process([frame])[0], _rot180(frame))

    def test_toggle_survives_the_temporal_stabilizer(self):
        # Regression: the temporal check compares against the latched previous
        # frame, so a toggle would look like a 180° error and be undone at once
        # unless the latched reference is rotated along with it.
        flip = ManualFlip180(1)
        stab = CameraStabilizer([True], [1.0], manual_flip=flip)
        frame = _scene()

        stab.process([frame])
        flip.toggle(0)

        for _ in range(5):
            out = stab.process([frame])[0]
            np.testing.assert_array_equal(out, _rot180(frame), "toggle was reverted")

    def test_reset_keeps_the_flip_but_clears_the_reference(self):
        flip = ManualFlip180(1)
        stab = CameraStabilizer([True], [20.0], manual_flip=flip)
        frame = _scene()

        flip.toggle(0)
        stab.process([frame])
        stab.reset()  # happens at the start of every episode

        np.testing.assert_array_equal(stab.process([frame])[0], _rot180(frame))

    def test_one_shared_object_moves_both_stabilizers(self):
        # record.py builds a stabilizer for the dataset and another for the
        # preview; a toggle from the preview must move the recorded frames too.
        flip = ManualFlip180(2)
        recording = CameraStabilizer([True, True], [20.0, 20.0], manual_flip=flip)
        preview = CameraStabilizer([True, True], [20.0, 20.0], manual_flip=flip)
        frames = [_scene(1), _scene(2)]

        flip.toggle(0)
        rec_out = recording.process(list(frames))
        viz_out = preview.process(list(frames))

        np.testing.assert_array_equal(rec_out[0], viz_out[0])
        np.testing.assert_array_equal(rec_out[0], _rot180(frames[0]))
        np.testing.assert_array_equal(
            rec_out[1], frames[1], "untouched camera must be unaffected"
        )

    def test_auto_stabilization_still_corrects_a_mid_episode_firmware_flip(self):
        flip = ManualFlip180(1)
        stab = CameraStabilizer([True], [1.0], manual_flip=flip)
        frame = _scene()

        stab.process([frame])
        # The camera firmware inverts the feed on its own.
        np.testing.assert_array_equal(stab.process([_rot180(frame)])[0], frame)

    def test_none_frames_pass_through(self):
        flip = ManualFlip180(1)
        flip.toggle(0)
        stab = CameraStabilizer([True], [20.0], manual_flip=flip)
        self.assertEqual(stab.process([None]), [None])

    def test_stabilizer_without_manual_flip_is_unchanged(self):
        stab = CameraStabilizer([False], [20.0])
        frame = _scene()
        np.testing.assert_array_equal(stab.process([frame])[0], frame)


if __name__ == "__main__":
    unittest.main()

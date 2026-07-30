"""Unit tests for subtask labelling in ``lerobot-ros-record``.

Covers the pieces that are easy to get subtly wrong and hard to notice in a
recorded dataset:

* the :class:`RecordingState` subtask counter (clamping, debounce, reset),
* the config arm map and the "did the acting arm change?" predicate that
  decides whether a pedal press costs a GELLO switch,
* subtask names and the operator-facing label string,
* the ``subtask_index`` feature declaration,
* the numpy-2.x scalar coercion, without which every episode fails to save, and
* ``lerobot-ros-fix-subtasks``: boundary-to-label expansion and the diagnosis
  that tells a missed pedal press apart from an extra one.

Run::

    /opt/conda/envs/lerobot/bin/python -m pytest tests/test_subtask_labelling.py -q

or, since the repo doesn't ship a pytest config, simply::

    /opt/conda/envs/lerobot/bin/python tests/test_subtask_labelling.py
"""

from __future__ import annotations

import shutil
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from lerobot_ros2.cli import fix_subtasks, record


class SubtaskCounterTest(unittest.TestCase):
    """The counter state machine in :class:`record.RecordingState`."""

    def setUp(self) -> None:
        # The 0.5 s debounce exists for pedal echoes; stepping instantly is the
        # whole point of these tests.
        self._debounce = record._SUBTASK_DEBOUNCE_S
        record._SUBTASK_DEBOUNCE_S = 0.0

    def tearDown(self) -> None:
        record._SUBTASK_DEBOUNCE_S = self._debounce

    def test_disabled_by_default(self) -> None:
        state = record.RecordingState()
        self.assertEqual(state.subtask_count, 0)
        self.assertIsNone(state.advance_subtask())
        self.assertIsNone(state.retreat_subtask())
        self.assertEqual(state.subtask_index, 0)

    def test_advances_to_the_last_subtask_then_clamps(self) -> None:
        state = record.RecordingState(subtask_count=6)
        self.assertEqual(state.subtask_index, 0)
        for expected in range(1, 6):
            self.assertEqual(state.advance_subtask(), expected)
        self.assertEqual(state.subtask_index, 5)
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(state.advance_subtask())
        self.assertEqual(state.subtask_index, 5)

    def test_retreat_undoes_a_press_and_clamps_at_zero(self) -> None:
        state = record.RecordingState(subtask_count=3)
        state.advance_subtask()
        self.assertEqual(state.retreat_subtask(), 0)
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(state.retreat_subtask())
        self.assertEqual(state.subtask_index, 0)

    def test_debounce_drops_an_echoed_press(self) -> None:
        record._SUBTASK_DEBOUNCE_S = 60.0
        state = record.RecordingState(subtask_count=6)
        self.assertEqual(state.advance_subtask(), 1)
        self.assertIsNone(state.advance_subtask())
        self.assertEqual(state.subtask_index, 1)

    def test_episode_start_resets_to_the_first_subtask(self) -> None:
        state = record.RecordingState(subtask_count=6)
        self.assertTrue(state.start(label="test"))
        state.advance_subtask()
        state.advance_subtask()
        self.assertEqual(state.subtask_index, 2)
        state.stop(label="test")
        # Stand in for the 1 s gap the toggle debounce enforces between episodes.
        state._last_toggle_time = 0.0
        self.assertTrue(state.start(label="test"))
        self.assertEqual(state.subtask_index, 0)

    def test_snapshot_returns_index_and_count(self) -> None:
        state = record.RecordingState(subtask_count=4)
        state.advance_subtask()
        self.assertEqual(state.subtask_snapshot(), (1, 4))


class SubtaskArmMapTest(unittest.TestCase):
    """Parsing ``subtask_arms`` and deriving per-arm control modes from it."""

    IDLE = record.GelloControlModeClient.MODE_IDLE
    NORMAL = record.GelloControlModeClient.MODE_NORMAL

    def test_absent_config_means_legacy_behaviour(self) -> None:
        self.assertEqual(record.resolve_subtask_arms({}), [])
        self.assertEqual(record.resolve_subtask_arms({"subtask_arms": None}), [])

    def test_entries_are_normalised(self) -> None:
        self.assertEqual(
            record.resolve_subtask_arms({"subtask_arms": [" Left", "RIGHT"]}),
            ["left", "right"],
        )

    def test_rejects_non_list_and_empty_entries(self) -> None:
        with self.assertRaises(ValueError):
            record.resolve_subtask_arms({"subtask_arms": "left"})
        with self.assertRaises(ValueError):
            record.resolve_subtask_arms({"subtask_arms": ["left", "  "]})

    def test_validate_rejects_an_unrecorded_arm(self) -> None:
        record.validate_subtask_arms(["left", "right"], ["left", "right"])
        with self.assertRaises(ValueError):
            record.validate_subtask_arms(["left", "right"], ["left"])

    def test_acting_arm_is_none_outside_the_map(self) -> None:
        arms = ["left", "right"]
        self.assertEqual(record.acting_arm_for_subtask(arms, 0), "left")
        self.assertIsNone(record.acting_arm_for_subtask(arms, 2))
        self.assertIsNone(record.acting_arm_for_subtask(arms, -1))
        self.assertIsNone(record.acting_arm_for_subtask([], 0))

    def test_modes_put_one_arm_live_and_the_rest_idle(self) -> None:
        arm_keys = ["left", "right"]
        subtask_arms = ["left", "left", "right"]
        self.assertEqual(
            record.arm_modes_for_subtask(arm_keys, subtask_arms, 0),
            {"left": self.NORMAL, "right": self.IDLE},
        )
        self.assertEqual(
            record.arm_modes_for_subtask(arm_keys, subtask_arms, 2),
            {"left": self.IDLE, "right": self.NORMAL},
        )

    def test_unmapped_subtask_leaves_modes_alone(self) -> None:
        self.assertEqual(record.arm_modes_for_subtask(["left"], [], 0), {})
        self.assertEqual(record.arm_modes_for_subtask(["left"], ["left"], 5), {})

    def test_only_the_left_to_right_boundary_hands_over_gello(self) -> None:
        """The whole point of the predicate: one switch, not six.

        A press that keeps the same arm live must not pause capture, or the
        episode gets a settling transient spliced into the middle of it.
        """
        subtask_arms = ["left", "left", "right", "right", "right", "right"]
        changed = [
            record.acting_arm_for_subtask(subtask_arms, i)
            != record.acting_arm_for_subtask(subtask_arms, i + 1)
            for i in range(len(subtask_arms) - 1)
        ]
        self.assertEqual(changed, [False, True, False, False, False])


class SubtaskNameTest(unittest.TestCase):
    """The ``{arm, name}`` config form and the operator-facing label string."""

    def test_bare_arm_names_still_parse(self) -> None:
        specs = record.resolve_subtasks({"subtask_arms": ["left", " RIGHT"]})
        self.assertEqual(specs, [record.SubtaskSpec("left", ""), record.SubtaskSpec("right", "")])

    def test_mapping_entries_carry_a_name(self) -> None:
        specs = record.resolve_subtasks({
            "subtask_arms": [
                {"arm": "Left", "name": " stir bar "},
                {"arm": "right"},
            ]
        })
        self.assertEqual(specs[0], record.SubtaskSpec("left", "stir bar"))
        self.assertEqual(specs[1], record.SubtaskSpec("right", ""))

    def test_mixed_forms_are_allowed(self) -> None:
        arms = record.resolve_subtask_arms({
            "subtask_arms": ["left", {"arm": "right", "name": "solid"}]
        })
        self.assertEqual(arms, ["left", "right"])
        self.assertEqual(
            record.resolve_subtask_names({
                "subtask_arms": ["left", {"arm": "right", "name": "solid"}]
            }),
            ["", "solid"],
        )

    def test_rejects_a_mapping_without_an_arm_or_with_junk_keys(self) -> None:
        with self.assertRaises(ValueError):
            record.resolve_subtasks({"subtask_arms": [{"name": "solid"}]})
        with self.assertRaises(ValueError):
            record.resolve_subtasks({"subtask_arms": [{"arm": "left", "hand": "x"}]})
        with self.assertRaises(ValueError):
            record.resolve_subtasks({"subtask_arms": [{"arm": "  "}]})

    def test_label_falls_back_to_the_bare_counter(self) -> None:
        self.assertEqual(record.subtask_label([], 2, 6), "step 3/6")
        self.assertEqual(record.subtask_label(["a", "b"], 5, 6), "step 6/6")

    def test_label_includes_name_and_arm(self) -> None:
        names = ["stir bar", "funnel on", "solid"]
        self.assertEqual(
            record.subtask_label(names, 2, 6, "right"), "step 3/6 solid (right)"
        )
        self.assertEqual(record.subtask_label(names, 0, 6), "step 1/6 stir bar")


class SubtaskFlashTest(unittest.TestCase):
    """The press announcement that the corner status line cannot show."""

    def test_absent_until_set_and_expires(self) -> None:
        state = record.RecordingState(subtask_count=6)
        self.assertIsNone(state.get_subtask_flash())
        state.set_subtask_flash("STEP 2/6 FUNNEL ON", duration_s=5.0)
        self.assertEqual(state.get_subtask_flash(), "STEP 2/6 FUNNEL ON")
        state.set_subtask_flash("STEP 3/6", duration_s=0.0)
        self.assertIsNone(state.get_subtask_flash())


class BoundaryLabelTest(unittest.TestCase):
    """``labels_from_boundaries`` — the core of the repair tool."""

    def test_expands_boundaries_into_per_frame_labels(self) -> None:
        labels = fix_subtasks.labels_from_boundaries(10, [3, 5, 8], 4)
        self.assertEqual(list(labels), [0, 0, 0, 1, 1, 2, 2, 2, 3, 3])

    def test_single_subtask_needs_no_boundaries(self) -> None:
        self.assertEqual(list(fix_subtasks.labels_from_boundaries(3, [], 1)), [0, 0, 0])

    def test_rejects_wrong_boundary_count(self) -> None:
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(10, [3, 5], 4)
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(10, [2, 3, 5, 8], 4)

    def test_rejects_boundaries_that_would_empty_a_subtask(self) -> None:
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(10, [3, 3, 8], 4)  # repeated
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(10, [5, 3, 8], 4)  # out of order
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(10, [0, 5, 8], 4)  # empty first
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(10, [3, 5, 10], 4)  # past the end

    def test_rejects_an_episode_shorter_than_its_subtask_count(self) -> None:
        with self.assertRaises(ValueError):
            fix_subtasks.labels_from_boundaries(3, [1, 2, 3], 4)

    def test_seconds_convert_at_the_dataset_fps(self) -> None:
        self.assertEqual(fix_subtasks.seconds_to_frames([0.0, 1.04, 2.96], 10.0), [0, 10, 30])


class SegmentAndDiagnosisTest(unittest.TestCase):
    """Collapsing labels into runs, and spotting the two operator errors."""

    NAMES = ["stir bar", "funnel on", "solid", "liquid", "funnel off", "septum"]

    def _diagnose(self, labels):
        return fix_subtasks.diagnose(
            fix_subtasks.segments_from_labels(labels, self.NAMES), 6
        )

    def test_segments_collapse_runs_and_carry_names(self) -> None:
        segments = fix_subtasks.segments_from_labels([0, 0, 0, 1, 1], self.NAMES)
        self.assertEqual(
            [(s.label, s.start, s.end, s.n_frames, s.name) for s in segments],
            [(0, 0, 2, 3, "stir bar"), (1, 3, 4, 2, "funnel on")],
        )

    def test_a_complete_episode_is_clean(self) -> None:
        labels = [index for index in range(6) for _ in range(4)]
        self.assertEqual(self._diagnose(labels), [])

    def test_missed_press_reports_ending_early_only(self) -> None:
        """A trailing gap must not also be blamed on an extra press."""
        labels = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 8 + [4] * 4
        problems = self._diagnose(labels)
        self.assertEqual(len(problems), 1)
        self.assertIn("ends on 5/6", problems[0])
        self.assertIn("missed", problems[0])

    def test_extra_press_reports_the_skipped_subtask(self) -> None:
        labels = [0] * 4 + [1] * 4 + [3] * 4 + [4] * 4 + [5] * 8
        problems = self._diagnose(labels)
        self.assertEqual(len(problems), 1)
        self.assertIn("subtask 3/6 has no frames", problems[0])
        self.assertIn("extra", problems[0])

    def test_stepping_back_shows_up_as_a_repeated_run(self) -> None:
        labels = [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
        problems = self._diagnose(labels)
        self.assertTrue(any("more than one run" in problem for problem in problems))
        self.assertTrue(any("not monotonically increasing" in problem for problem in problems))

    def test_empty_episode(self) -> None:
        self.assertEqual(self._diagnose([]), ["no frames"])


class SubtaskFeatureTest(unittest.TestCase):
    """The dataset schema and the numpy-2.x save-path workaround."""

    JOINT_NAMES = {"left": ["left_shoulder_pan_joint", "left_gripper_joint"]}

    def test_feature_is_absent_when_labelling_is_off(self) -> None:
        features = record.build_features(["left"], self.JOINT_NAMES, [])
        self.assertNotIn("subtask_index", features)

    def test_feature_matches_the_action_source_declaration(self) -> None:
        features = record.build_features(
            ["left"], self.JOINT_NAMES, [], subtask_count=6
        )
        self.assertEqual(
            features["subtask_index"],
            {"dtype": "int64", "shape": (1,), "names": ["subtask_index"]},
        )

    def test_coercion_flattens_buffered_values_to_python_ints(self) -> None:
        buffered = [np.asarray([i], dtype=np.int64) for i in (0, 0, 1, 2)]
        dataset = types.SimpleNamespace(
            writer=types.SimpleNamespace(
                episode_buffer={
                    "subtask_index": list(buffered),
                    "action": [np.zeros(2, dtype=np.float32)] * 4,
                }
            )
        )
        record._coerce_scalar_episode_columns(dataset)

        column = dataset.writer.episode_buffer["subtask_index"]
        self.assertEqual(column, [0, 0, 1, 2])
        self.assertTrue(all(type(value) is int for value in column))
        # Other columns must be left exactly as they were.
        self.assertEqual(len(dataset.writer.episode_buffer["action"]), 4)

    def test_coercion_tolerates_a_missing_buffer(self) -> None:
        record._coerce_scalar_episode_columns(types.SimpleNamespace(writer=None))
        record._coerce_scalar_episode_columns(
            types.SimpleNamespace(writer=types.SimpleNamespace(episode_buffer=None))
        )
        record._coerce_scalar_episode_columns(
            types.SimpleNamespace(writer=types.SimpleNamespace(episode_buffer={}))
        )


class SubtaskRoundTripTest(unittest.TestCase):
    """Write a labelled episode through the real LeRobotDataset and read it back.

    The stubbed coercion test above can't notice LeRobot renaming
    ``writer.episode_buffer``, which would turn ``_coerce_scalar_episode_columns``
    into a silent no-op and make every episode fail to save. This one would.
    """

    SUBTASK_COUNT = 6
    FRAMES_PER_SUBTASK = 3

    def test_column_survives_save_and_reload(self) -> None:
        try:
            import pandas as pd
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - depends on the env
            self.skipTest(f"LeRobot/pandas unavailable: {exc}")

        features = record.build_features(
            ["left"],
            {"left": ["left_shoulder_pan_joint", "left_gripper_joint"]},
            [],
            subtask_count=self.SUBTASK_COUNT,
        )
        tmp_root = Path(tempfile.mkdtemp())
        saved_debounce = record._SUBTASK_DEBOUNCE_S
        record._SUBTASK_DEBOUNCE_S = 0.0
        try:
            dataset = LeRobotDataset.create(
                repo_id="ur_robotiq/subtask_roundtrip",
                fps=10,
                features=features,
                robot_type="ur3e_bimanual",
                root=tmp_root / "subtask_roundtrip",
                use_videos=False,
            )
            state = record.RecordingState(subtask_count=self.SUBTASK_COUNT)
            self.assertTrue(state.start(label="test"))

            expected = []
            for step in range(self.SUBTASK_COUNT):
                if step > 0:
                    self.assertEqual(state.advance_subtask(label="test"), step)
                for _ in range(self.FRAMES_PER_SUBTASK):
                    index, count = state.subtask_snapshot()
                    expected.append(index)
                    dataset.add_frame({
                        "task": "long horizon round trip",
                        "action": np.zeros(2, dtype=np.float32),
                        "observation.state": np.zeros(2, dtype=np.float32),
                        "subtask_index": np.asarray([index], dtype=np.int64),
                    })

            record._coerce_scalar_episode_columns(dataset)
            dataset.save_episode()
            dataset.finalize()

            parquet = next((tmp_root / "subtask_roundtrip" / "data").rglob("*.parquet"))
            frame = pd.read_parquet(parquet)
            self.assertEqual(len(frame), len(expected))
            self.assertEqual(frame["subtask_index"].tolist(), expected)
            self.assertEqual(frame["subtask_index"].dtype, np.int64)
        finally:
            record._SUBTASK_DEBOUNCE_S = saved_debounce
            shutil.rmtree(tmp_root, ignore_errors=True)


class IncompleteEpisodeWarningTest(unittest.TestCase):
    def test_warns_when_the_episode_stopped_short(self) -> None:
        state = record.RecordingState(subtask_count=6)
        state.episode_idx = 12
        with self.assertLogs(level="WARNING") as captured:
            record._warn_on_incomplete_subtasks(state)
        self.assertIn("subtask 1/6", "\n".join(captured.output))

    def test_silent_on_a_complete_episode(self) -> None:
        record._SUBTASK_DEBOUNCE_S, saved = 0.0, record._SUBTASK_DEBOUNCE_S
        try:
            state = record.RecordingState(subtask_count=3)
            for _ in range(2):
                state.advance_subtask()
            with self.assertNoLogs(level="WARNING"):
                record._warn_on_incomplete_subtasks(state)
        finally:
            record._SUBTASK_DEBOUNCE_S = saved

    def test_silent_when_labelling_is_off(self) -> None:
        with self.assertNoLogs(level="WARNING"):
            record._warn_on_incomplete_subtasks(record.RecordingState())


if __name__ == "__main__":
    unittest.main(verbosity=2)

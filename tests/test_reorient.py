"""Tests for the reorient CLI: episode selection, frame ranges, and rotation."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from lerobot_ros2.cli import reorient

FPS = 10
WIDTH = HEIGHT = 64
BAND = 8
CAMERAS = ("camera_00_wrist", "camera_01_other")


def _make_frames(count: int) -> List[np.ndarray]:
    """Frames with a bright band along the top, so a 180 rotation is detectable."""
    frames = []
    for i in range(count):
        frame = np.full((HEIGHT, WIDTH, 3), 10 + (i * 3) % 120, np.uint8)
        frame[:BAND] = 250
        frames.append(frame)
    return frames


def _encode(path: Path, frames: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [reorient.FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
         "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def _orientations(path: Path) -> List[str]:
    """Per-frame 'UP'/'DOWN' based on which horizontal band is bright."""
    import av

    out: List[str] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24").astype(np.int16)
            out.append("UP" if arr[:BAND].mean() > arr[-BAND:].mean() else "DOWN")
    return out


def _build_dataset(root: Path, spans: Sequence[Tuple[int, float, float, int, int]]) -> None:
    """Write a minimal v3.0 dataset.

    ``spans`` entries are ``(episode_index, from_ts, to_ts, length, file_index)``.
    ``length`` is stored as-is so a span longer than the row count can be
    represented, which is what the real datasets contain.
    """
    features = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": ["a"] * 7},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": HEIGHT,
                "video.width": WIDTH,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.fps": FPS,
            },
        }
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "total_episodes": len(spans),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    rows = []
    for episode, start, end, length, file_index in spans:
        row = {"episode_index": episode, "length": length}
        for cam in CAMERAS:
            key = f"videos/observation.images.{cam}"
            row[f"{key}/chunk_index"] = 0
            row[f"{key}/file_index"] = file_index
            row[f"{key}/from_timestamp"] = start
            row[f"{key}/to_timestamp"] = end
        rows.append(row)
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ep_dir / "file-000.parquet")

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_index": [s[0] for s in spans]}).to_parquet(
        data_dir / "file-000.parquet"
    )
    (root / "recording_config.yaml").write_text("cameras: {}\n")

    # One packed mp4 per camera per file_index, long enough to cover every span.
    by_file: dict[int, float] = {}
    for _, _, end, _, file_index in spans:
        by_file[file_index] = max(by_file.get(file_index, 0.0), end)
    for cam in CAMERAS:
        for file_index, end in by_file.items():
            path = (root / "videos" / f"observation.images.{cam}" / "chunk-000"
                    / f"file-{file_index:03d}.mp4")
            _encode(path, _make_frames(int(round(end * FPS))))


class EpisodeSpecTest(unittest.TestCase):
    valid = [0, 1, 2, 3, 4, 5, 10]

    def test_all(self):
        self.assertEqual(reorient.parse_episode_spec("all", self.valid), sorted(self.valid))
        self.assertEqual(reorient.parse_episode_spec(" ALL ", self.valid), sorted(self.valid))

    def test_single_and_list(self):
        self.assertEqual(reorient.parse_episode_spec("3", self.valid), [3])
        self.assertEqual(reorient.parse_episode_spec("3,1,5", self.valid), [1, 3, 5])

    def test_ranges_and_dedup(self):
        self.assertEqual(reorient.parse_episode_spec("1-4", self.valid), [1, 2, 3, 4])
        self.assertEqual(reorient.parse_episode_spec("1-3,2,3", self.valid), [1, 2, 3])
        self.assertEqual(reorient.parse_episode_spec("0, 2-3 , 10", self.valid), [0, 2, 3, 10])

    def test_rejects_unknown_episode(self):
        with self.assertRaises(SystemExit):
            reorient.parse_episode_spec("7", self.valid)
        with self.assertRaises(SystemExit):
            reorient.parse_episode_spec("4-11", self.valid)

    def test_rejects_malformed(self):
        for bad in ("", "   ", "abc", "3-", "-", "2-1", "1,,x"):
            with self.subTest(spec=bad), self.assertRaises(SystemExit):
                reorient.parse_episode_spec(bad, self.valid)


class EnableExpressionTest(unittest.TestCase):
    def _range(self, start, end, episode=0):
        return reorient.FrameRange(
            episode=episode, file_index=0, chunk_index=0, start=start, end=end
        )

    def test_single_range_is_inclusive_of_last_frame(self):
        # between() is inclusive at both ends, so a half-open [10,20) span must
        # be emitted as between(n,10,19) or one extra frame gets rotated.
        self.assertEqual(
            reorient.build_enable_expr([self._range(10, 20)]), "between(n,10,19)"
        )

    def test_multiple_ranges_are_ored(self):
        expr = reorient.build_enable_expr([self._range(0, 5), self._range(10, 12)])
        self.assertEqual(expr, "between(n,0,4)+between(n,10,11)")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            reorient.build_enable_expr([])


class FrameRangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _meta(self, spans):
        root = self.tmp / "ds"
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        _build_dataset_meta_only(root, spans)
        return reorient.load_dataset_meta(root)

    def test_span_longer_than_length_uses_the_span(self):
        # Real numbers from salt_20260725 episode 1: the video span covers 721
        # frames while only 576 rows reference it, because footage from a
        # discarded take was retained. The whole span must be rotated.
        meta = self._meta([
            (0, 0.0, 67.7, 677, 0),
            (1, 67.7, 139.8, 576, 0),
            (2, 139.8, 195.2, 554, 0),
        ])
        key = meta.camera_keys["wrist"]
        ranges = reorient.episode_frame_ranges(meta, key, [1])
        self.assertEqual(len(ranges), 1)
        self.assertEqual((ranges[0].start, ranges[0].end), (677, 1398))
        self.assertEqual(ranges[0].count, 721)
        self.assertGreater(ranges[0].count, 576)

    def test_ranges_are_contiguous_and_cover_the_file(self):
        meta = self._meta([
            (0, 0.0, 67.7, 677, 0),
            (1, 67.7, 139.8, 576, 0),
            (2, 139.8, 195.2, 554, 0),
        ])
        key = meta.camera_keys["wrist"]
        ranges = reorient.episode_frame_ranges(meta, key, meta.episode_ids)
        self.assertEqual([(r.start, r.end) for r in ranges],
                         [(0, 677), (677, 1398), (1398, 1952)])

    def test_grouping_splits_by_file(self):
        meta = self._meta([
            (0, 0.0, 10.0, 100, 0),
            (1, 10.0, 20.0, 100, 0),
            (2, 0.0, 10.0, 100, 1),
        ])
        key = meta.camera_keys["wrist"]
        grouped = reorient.group_by_file(
            reorient.episode_frame_ranges(meta, key, meta.episode_ids)
        )
        self.assertEqual(sorted(grouped), [(0, 0), (0, 1)])
        self.assertEqual([r.episode for r in grouped[(0, 0)]], [0, 1])
        self.assertEqual([r.episode for r in grouped[(0, 1)]], [2])

    def test_unknown_camera_is_rejected(self):
        meta = self._meta([(0, 0.0, 10.0, 100, 0)])
        with self.assertRaises(SystemExit):
            reorient.resolve_cameras(meta, ["nope"])

    def test_camera_short_name_handles_both_key_styles(self):
        self.assertEqual(
            reorient.camera_short_name("observation.images.camera_01_left_wrist_top"),
            "left_wrist_top",
        )
        self.assertEqual(
            reorient.camera_short_name("observation.images.left_wrist_top"),
            "left_wrist_top",
        )
        self.assertEqual(reorient.camera_short_name("left_wrist_top"), "left_wrist_top")


def _build_dataset_meta_only(root: Path, spans) -> None:
    """Metadata-only dataset (no videos), for the pure range/selection tests."""
    features = {
        "observation.images.camera_00_wrist": {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "info": {"video.codec": "av1"},
        }
    }
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    rows = []
    for episode, start, end, length, file_index in spans:
        key = "videos/observation.images.camera_00_wrist"
        rows.append({
            "episode_index": episode,
            "length": length,
            f"{key}/chunk_index": 0,
            f"{key}/file_index": file_index,
            f"{key}/from_timestamp": start,
            f"{key}/to_timestamp": end,
        })
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ep_dir / "file-000.parquet")


class RotateRoundTripTest(unittest.TestCase):
    """End-to-end: rotate one episode inside a packed mp4 and check the result."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.src = cls.tmp / "src"
        # Three 10-frame episodes packed into one file (30 frames total), plus a
        # second file so we can prove untouched files are copied not re-encoded.
        _build_dataset(cls.src, [
            (0, 0.0, 1.0, 10, 0),
            (1, 1.0, 2.0, 10, 0),
            (2, 2.0, 3.0, 10, 0),
            (3, 0.0, 1.0, 10, 1),
        ])
        cls.dst = cls.tmp / "dst"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = reorient.main([
                "--src", str(cls.src), "--dst", str(cls.dst),
                "--camera", "wrist", "--episodes", "1",
            ])
        assert rc == 0, "reorient returned nonzero"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _video(self, root: Path, cam: str, file_index: int) -> Path:
        return (root / "videos" / f"observation.images.{cam}" / "chunk-000"
                / f"file-{file_index:03d}.mp4")

    def test_only_the_selected_episode_is_rotated(self):
        got = _orientations(self._video(self.dst, "camera_00_wrist", 0))
        self.assertEqual(len(got), 30)
        self.assertEqual(got[:10], ["UP"] * 10, "episode 0 must be untouched")
        self.assertEqual(got[10:20], ["DOWN"] * 10, "episode 1 must be rotated")
        self.assertEqual(got[20:], ["UP"] * 10, "episode 2 must be untouched")

    def test_frame_count_is_preserved(self):
        src_n = reorient.probe_frame_count(self._video(self.src, "camera_00_wrist", 0))
        dst_n = reorient.probe_frame_count(self._video(self.dst, "camera_00_wrist", 0))
        self.assertEqual(src_n, dst_n)
        self.assertEqual(src_n, 30)

    def test_other_camera_is_copied_byte_for_byte(self):
        src = self._video(self.src, "camera_01_other", 0)
        dst = self._video(self.dst, "camera_01_other", 0)
        self.assertEqual(src.read_bytes(), dst.read_bytes())

    def test_file_without_a_selected_episode_is_copied_byte_for_byte(self):
        src = self._video(self.src, "camera_00_wrist", 1)
        dst = self._video(self.dst, "camera_00_wrist", 1)
        self.assertEqual(src.read_bytes(), dst.read_bytes())

    def test_source_is_untouched(self):
        got = _orientations(self._video(self.src, "camera_00_wrist", 0))
        self.assertEqual(got, ["UP"] * 30)

    def test_metadata_and_provenance(self):
        info = json.loads((self.dst / "meta" / "info.json").read_text())
        block = info["features"]["observation.images.camera_00_wrist"]["info"]
        self.assertEqual(block["video.codec"], "h264")
        self.assertEqual(block["reorient_180_episodes"], [1])
        untouched = info["features"]["observation.images.camera_01_other"]["info"]
        self.assertEqual(untouched["video.codec"], "av1")
        self.assertNotIn("reorient_180_episodes", untouched)

        manifest = json.loads((self.dst / "reorient_manifest.json").read_text())
        self.assertEqual(manifest["rotation_degrees"], 180)
        self.assertEqual(manifest["cameras"], ["wrist"])
        self.assertEqual(manifest["episodes"], [1])

    def test_data_and_sidecars_are_copied(self):
        self.assertTrue((self.dst / "data" / "chunk-000" / "file-000.parquet").is_file())
        self.assertTrue((self.dst / "recording_config.yaml").is_file())
        self.assertTrue((self.dst / "meta" / "episodes" / "chunk-000"
                         / "file-000.parquet").is_file())


class GuardrailTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self.tmp / "src"
        _build_dataset_meta_only(self.src, [(0, 0.0, 1.0, 10, 0)])

    def test_non_dataset_source_is_rejected(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(SystemExit):
            reorient.main(["--src", str(empty), "--report"])

    def test_same_src_and_dst_is_rejected(self):
        with self.assertRaises(SystemExit):
            reorient.main(["--src", str(self.src), "--dst", str(self.src),
                           "--camera", "wrist", "--episodes", "0"])

    def test_rewrite_requires_camera_and_episodes(self):
        with self.assertRaises(SystemExit):
            reorient.main(["--src", str(self.src), "--dst", str(self.tmp / "out"),
                           "--episodes", "0"])
        with self.assertRaises(SystemExit):
            reorient.main(["--src", str(self.src), "--dst", str(self.tmp / "out"),
                           "--camera", "wrist"])

    def test_dst_required_without_report(self):
        with self.assertRaises(SystemExit):
            reorient.main(["--src", str(self.src), "--camera", "wrist", "--episodes", "0"])

    def test_dry_run_writes_nothing(self):
        out = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = reorient.main(["--src", str(self.src), "--dst", str(out),
                               "--camera", "wrist", "--episodes", "0", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists(), "dry run must not create the destination")
        self.assertIn("Dry run", buf.getvalue())

    def test_non_empty_dst_needs_overwrite(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "stale.txt").write_text("x")
        with self.assertRaises(SystemExit):
            reorient.prepare_destination(out, overwrite=False)
        reorient.prepare_destination(out, overwrite=True)
        self.assertFalse((out / "stale.txt").exists())


class SummarizeIdsTest(unittest.TestCase):
    def test_collapses_runs(self):
        self.assertEqual(reorient._summarize_ids([]), "(none)")
        self.assertEqual(reorient._summarize_ids([3]), "3")
        self.assertEqual(reorient._summarize_ids([1, 2, 3]), "1-3")
        self.assertEqual(reorient._summarize_ids([1, 2, 3, 7, 9, 10]), "1-3,7,9-10")


if __name__ == "__main__":
    unittest.main()

"""Data loader for LeRobot v3 teleoperation datasets.

Handles reading parquet trajectory data and extracting synchronized
video frames from MP4 files. Uses PyAV for AV1 decoding with
JPEG-compressed in-memory frame cache for fast scrubbing.

Supports multi-file datasets where episodes are spread across
multiple parquet and video files.
"""

from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import pandas as pd
from PIL import Image
import yaml


@dataclass(frozen=True)
class ArmSpec:
    """Description of one recorded arm in the dataset."""

    name: str
    joint_names: tuple[str, ...]
    start: int
    stop: int

    @property
    def label(self) -> str:
        """Human-readable label for plots."""
        return self.name.replace("_", " ").title()


class TeleopDataset:
    """Loads and serves per-frame data from a LeRobot v3 dataset.

    On init, all video frames are decoded once and stored as compressed
    JPEG bytes in memory (~50 KB/frame) so that subsequent frame lookups
    are near-instant.

    Args:
        dataset_dir: Root directory of the dataset (contains data/, videos/, meta/).
        progress_callback: Optional callable(camera_key, current, total) for
            reporting video decode progress.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
        preload_frames: bool = True,
    ) -> None:
        self.root = Path(dataset_dir)
        self._meta = self._load_meta()
        self._experiment_config = self._load_experiment_config()
        self._df = self._load_parquet()
        self._episode_row_indices = self._compute_episode_row_indices()
        self._episode_ranges = self._compute_episode_ranges()
        self._episode_video_map = self._load_episode_video_map()
        self._arm_specs = self._build_arm_specs()

        self._preload_frames = preload_frames
        self._frame_store: dict[str, dict[int, list[bytes]]] = {}
        if preload_frames:
            self._preload_videos(progress_callback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        """Frames per second of the dataset."""
        return int(self._meta["fps"])

    @property
    def episode_ids(self) -> list[int]:
        """Sorted list of episode indices."""
        return sorted(self._episode_row_indices.keys())

    @property
    def joint_names(self) -> list[str]:
        """Names of the action/state dimensions."""
        return self._meta["features"]["observation.state"]["names"]

    @property
    def arm_specs(self) -> list[ArmSpec]:
        """Ordered arm layout inferred from experiment metadata."""
        return list(self._arm_specs)

    @property
    def num_arms(self) -> int:
        """Number of recorded arms in the dataset."""
        return len(self._arm_specs)

    @property
    def camera_keys(self) -> list[str]:
        """Available camera feature keys."""
        return [
            k
            for k, v in self._meta["features"].items()
            if isinstance(v, dict) and v.get("dtype") == "video"
        ]

    def episode_length(self, episode_id: int) -> int:
        """Return number of frames in a given episode.

        Args:
            episode_id: Episode index.

        Returns:
            Number of frames.
        """
        return len(self._episode_row_indices[int(episode_id)])

    def get_episode_timestamps(self, episode_id: int) -> list[float]:
        """Return timestamps for one episode, ordered by frame index."""
        row_positions = self._episode_row_indices[int(episode_id)]
        return self._df.iloc[row_positions]["timestamp"].astype(float).tolist()

    def get_frame_data(self, episode_id: int, frame_idx: int) -> dict[str, Any]:
        """Return all data for a single frame.

        Args:
            episode_id: Episode index.
            frame_idx: Frame index *within* the episode (0-based).

        Returns:
            Dict with keys: 'state', 'action', 'timestamp', 'frame_index',
            'episode_index', 'global_index'.
        """
        episode_id = int(episode_id)
        frame_idx = int(frame_idx)
        row_positions = self._episode_row_indices[episode_id]
        if frame_idx < 0 or frame_idx >= len(row_positions):
            raise IndexError(
                f"Frame index out of range for episode {episode_id}: {frame_idx}"
            )

        row = self._df.iloc[int(row_positions[frame_idx])]
        return {
            "state": np.array(row["observation.state"], dtype=np.float32),
            "action": np.array(row["action"], dtype=np.float32),
            "timestamp": float(row["timestamp"]),
            "frame_index": int(row["frame_index"]),
            "episode_index": int(row["episode_index"]),
            "global_index": int(row["index"]),
        }

    def get_video_frame(
        self, episode_id: int, frame_idx: int, camera_key: str
    ) -> np.ndarray:
        """Return a single RGB frame from the pre-decoded cache.

        Args:
            episode_id: Episode index.
            frame_idx: Frame index within the episode.
            camera_key: e.g. 'observation.images.cam_0'.

        Returns:
            RGB image as uint8 numpy array (H, W, 3).

        Raises:
            RuntimeError: If the camera key is unknown.
            IndexError: If the frame index is out of range.
        """
        if camera_key not in self.camera_keys:
            raise RuntimeError(f"Unknown camera key: {camera_key}")

        vmap = self._episode_video_map[episode_id][camera_key]
        file_idx = vmap["file_index"]
        start_frame = vmap["start_frame"]
        frame_in_file = start_frame + frame_idx

        if not self._preload_frames:
            return self._decode_single_frame(camera_key, file_idx, frame_in_file)

        jpeg_bytes = self._frame_store[camera_key][file_idx][frame_in_file]
        img = Image.open(io.BytesIO(jpeg_bytes))
        return np.array(img)

    def get_episode_video_source(self, episode_id: int, camera_key: str) -> dict[str, Any]:
        """Return video file path and timing metadata for one episode/camera.

        Args:
            episode_id: Episode index.
            camera_key: Feature key like 'observation.images.cam_0'.

        Returns:
            Dict with keys:
              - path: str path to source MP4 file
              - start_frame: int frame offset in that file where episode starts
              - start_seconds: float offset in seconds
              - fps: int dataset frame rate
        """
        if camera_key not in self.camera_keys:
            raise RuntimeError(f"Unknown camera key: {camera_key}")

        vmap = self._episode_video_map[episode_id][camera_key]
        file_idx = int(vmap["file_index"])
        start_frame = int(vmap["start_frame"])
        return {
            "path": str(self._get_video_path(camera_key, file_idx)),
            "start_frame": start_frame,
            "start_seconds": start_frame / float(self.fps),
            "fps": int(self.fps),
        }

    def get_episode_actions(self, episode_id: int) -> np.ndarray:
        """Return all action vectors for an episode as (N, 14) array.

        Args:
            episode_id: Episode index.

        Returns:
            numpy array of shape (episode_length, 14).
        """
        row_positions = self._episode_row_indices[int(episode_id)]
        actions = self._df.iloc[row_positions]["action"].tolist()
        return np.array(actions, dtype=np.float32)

    def get_episode_states(self, episode_id: int) -> np.ndarray:
        """Return all state vectors for an episode as (N, 14) array.

        Args:
            episode_id: Episode index.

        Returns:
            numpy array of shape (episode_length, 14).
        """
        row_positions = self._episode_row_indices[int(episode_id)]
        states = self._df.iloc[row_positions]["observation.state"].tolist()
        return np.array(states, dtype=np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        """Load meta/info.json."""
        info_path = self.root / "meta" / "info.json"
        with open(info_path) as f:
            return json.load(f)

    def _load_experiment_config(self) -> dict | None:
        """Load dataset experiment metadata when available."""
        for candidate in (
            self.root / "experiment_config.yaml",
            self.root / "recording_config.yaml",
            self.root / "meta" / "experiment_config.yaml",
            self.root / "meta" / "recording_config.yaml",
        ):
            if candidate.exists():
                with open(candidate) as f:
                    return yaml.safe_load(f)
        return None

    def _build_arm_specs(self) -> list[ArmSpec]:
        """Build ordered arm slices from experiment metadata or feature names."""
        names = list(self.joint_names)
        configs = self._extract_arm_joint_names(self._experiment_config)

        if configs:
            specs: list[ArmSpec] = []
            cursor = 0
            for arm_name, joint_names in configs:
                stop = cursor + len(joint_names)
                specs.append(
                    ArmSpec(
                        name=arm_name,
                        joint_names=tuple(joint_names),
                        start=cursor,
                        stop=stop,
                    )
                )
                cursor = stop

            if cursor < len(names):
                specs.append(
                    ArmSpec(
                        name="extra_joints",
                        joint_names=tuple(names[cursor:]),
                        start=cursor,
                        stop=len(names),
                    )
                )
            return specs

        if len(names) == 0:
            return []

        return [ArmSpec(name="arm_1", joint_names=tuple(names), start=0, stop=len(names))]

    @staticmethod
    def _extract_arm_joint_names(experiment_config: dict | None) -> list[tuple[str, list[str]]]:
        """Return arm joint names in config order for the available arms."""
        if not experiment_config:
            return []

        arm_names: list[tuple[str, list[str]]] = []
        for key, section in experiment_config.items():
            if not key.endswith("_arm"):
                continue
            params = section.get("ros__parameters") or {}
            joint_names = list(params.get("robot_joint_names") or [])
            if joint_names:
                arm_names.append((key.removesuffix("_arm"), joint_names))

        return arm_names

    def _load_parquet(self) -> pd.DataFrame:
        """Load and concatenate all trajectory parquet files."""
        data_dir = self.root / "data"
        pq_files = sorted(data_dir.rglob("*.parquet"))
        if not pq_files:
            pattern = self._meta["data_path"]
            pq_files = [self.root / pattern.format(chunk_index=0, file_index=0)]

        dfs = [pd.read_parquet(p) for p in pq_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("index").reset_index(drop=True)
        return df

    def _compute_episode_row_indices(self) -> dict[int, np.ndarray]:
        """Map episode_id -> ordered dataframe row positions."""
        # First, keep deterministic order by dataset index within each episode.
        order_cols = [col for col in ("episode_index", "index") if col in self._df.columns]
        ordered = self._df.sort_values(order_cols, kind="mergesort") if order_cols else self._df
        grouped = ordered.groupby("episode_index", sort=True)

        if "frame_index" not in self._df.columns:
            return {
                int(ep_id): positions.to_numpy(dtype=np.int64, copy=False)
                for ep_id, positions in grouped.groups.items()
            }

        episode_rows: dict[int, np.ndarray] = {}
        for ep_id, positions in grouped.groups.items():
            row_pos = positions.to_numpy(dtype=np.int64, copy=False)
            episode_frame = self._df.iloc[row_pos][["frame_index", "index"]].copy()
            episode_frame["__row_pos"] = row_pos

            # Some rewritten datasets can contain overlapping segments for an
            # episode (duplicate frame_index values). Keep first occurrence.
            episode_frame = episode_frame.sort_values(
                ["frame_index", "index"], kind="mergesort"
            ).drop_duplicates(subset=["frame_index"], keep="first")

            episode_rows[int(ep_id)] = episode_frame["__row_pos"].to_numpy(
                dtype=np.int64, copy=False
            )

        return episode_rows

    def _compute_episode_ranges(self) -> dict[int, tuple[int, int]]:
        """Map episode_id -> [min_row, max_row+1] for compatibility paths."""
        ranges: dict[int, tuple[int, int]] = {}
        for ep_id, positions in self._episode_row_indices.items():
            start = int(positions.min())
            end = int(positions.max()) + 1
            ranges[int(ep_id)] = (start, end)
        return ranges

    def _load_episode_video_map(self) -> dict[int, dict[str, dict[str, Any]]]:
        """Build per-episode video lookup from episode metadata.

        Returns:
            ``{episode_id: {cam_key: {"file_index": int, "start_frame": int}}}``
        """
        ep_dir = self.root / "meta" / "episodes"
        pq_files = sorted(ep_dir.rglob("*.parquet"))

        if not pq_files:
            return self._fallback_video_map()

        dfs = [pd.read_parquet(p) for p in pq_files]
        ep_meta = pd.concat(dfs, ignore_index=True)

        fps = self.fps
        result: dict[int, dict[str, dict[str, Any]]] = {}
        for _, row in ep_meta.iterrows():
            ep_id = int(row["episode_index"])
            cam_map: dict[str, dict[str, Any]] = {}
            for cam_key in self.camera_keys:
                prefix = f"videos/{cam_key}"
                file_idx = int(row[f"{prefix}/file_index"])
                from_ts = float(row[f"{prefix}/from_timestamp"])
                start_frame = round(from_ts * fps)
                cam_map[cam_key] = {
                    "file_index": file_idx,
                    "start_frame": start_frame,
                }
            result[ep_id] = cam_map
        return result

    def _fallback_video_map(self) -> dict[int, dict[str, dict[str, Any]]]:
        """Fallback when no episode metadata exists (single-file dataset)."""
        result: dict[int, dict[str, dict[str, Any]]] = {}
        for ep_id in self.episode_ids:
            start = self._episode_ranges[ep_id][0]
            cam_map = {
                cam: {"file_index": 0, "start_frame": start}
                for cam in self.camera_keys
            }
            result[ep_id] = cam_map
        return result

    def _get_video_path(self, camera_key: str, file_index: int = 0) -> Path:
        """Return the video file path for a camera key and file index.

        Args:
            camera_key: Feature key like 'observation.images.cam_0'.
            file_index: Which file within the chunk.

        Returns:
            Path to the MP4 file.

        Raises:
            RuntimeError: If the video file does not exist.
        """
        pattern = self._meta["video_path"]
        video_path = self.root / pattern.format(
            video_key=camera_key, chunk_index=0, file_index=file_index
        )
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path}")
        return video_path

    def _collect_needed_file_indices(self) -> dict[str, set[int]]:
        """Determine which video file indices are needed per camera."""
        needed: dict[str, set[int]] = {cam: set() for cam in self.camera_keys}
        for cam_map in self._episode_video_map.values():
            for cam_key, info in cam_map.items():
                needed[cam_key].add(info["file_index"])
        return needed

    def _decode_single_frame(self, camera_key: str, file_index: int, frame_idx: int) -> np.ndarray:
        """Decode one frame from a source video file on-demand."""
        video_path = self._get_video_path(camera_key, file_index)
        container = av.open(str(video_path))
        try:
            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_idx:
                    return frame.to_ndarray(format="rgb24")
        finally:
            container.close()

        raise IndexError(
            f"Frame index out of range for {camera_key} file {file_index}: {frame_idx}"
        )

    def _preload_videos(
        self, progress_callback: Callable[[str, int, int], None] | None
    ) -> None:
        """Decode all needed video files and store frames as JPEG bytes.

        Cameras are decoded in parallel threads (PyAV releases the GIL
        during decoding) so N cameras take roughly the same wall-time as 1.

        Args:
            progress_callback: Optional callable(camera_key, current, total).
        """
        total = int(self._meta["total_frames"])
        needed = self._collect_needed_file_indices()
        cb_lock = threading.Lock()

        def _decode_camera(cam_key: str) -> None:
            store: dict[int, list[bytes]] = {}
            decoded_so_far = 0

            for file_idx in sorted(needed[cam_key]):
                video_path = self._get_video_path(cam_key, file_idx)
                container = av.open(str(video_path))
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"

                frames: list[bytes] = []
                for i, frame in enumerate(container.decode(video=0)):
                    rgb = frame.to_ndarray(format="rgb24")
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
                    frames.append(buf.getvalue())
                    if progress_callback and (decoded_so_far + i) % 50 == 0:
                        with cb_lock:
                            progress_callback(cam_key, decoded_so_far + i, total)

                container.close()
                decoded_so_far += len(frames)
                store[file_idx] = frames

            self._frame_store[cam_key] = store
            if progress_callback:
                with cb_lock:
                    progress_callback(cam_key, decoded_so_far, decoded_so_far)

        with ThreadPoolExecutor(max_workers=len(self.camera_keys)) as pool:
            list(pool.map(_decode_camera, self.camera_keys))

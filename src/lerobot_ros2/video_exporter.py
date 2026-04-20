"""Export synchronized camera + timeline composite videos.

Renders all camera views in a grid (up to 3 per row) on top with the
action/state timeline (sweeping red line) on the bottom, frame-by-frame
into an MP4. Uses OpenCV for frame compositing and video encoding.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, cast

import av  # type: ignore[import-not-found]
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw

try:
    from PIL.Image import Resampling
except ImportError:  # pragma: no cover - older Pillow fallback
    Resampling = Image  # type: ignore[assignment]

matplotlib.use("Agg")

from .data_loader import ArmSpec, TeleopDataset

_CAM_WIDTH = 640
_MAX_COLS = 3
_VIDEO_CODECS = ("avc1", "H264", "mp4v", "XVID")


def compute_grid_shape(item_count: int) -> tuple[int, int]:
    """Return a near-square grid shape for *item_count* tiles.

    The layout is chosen to minimize the difference between rows and columns,
    then to minimize empty cells, and finally to prefer even dimensions.
    """
    if item_count <= 0:
        raise ValueError("At least one tile is required to compute a grid shape")

    best_rows = 1
    best_cols = item_count
    best_score = (
        abs(best_rows - best_cols),
        best_rows % 2 + best_cols % 2,
        best_rows * best_cols - item_count,
    )

    for cols in range(1, item_count + 1):
        rows = int(np.ceil(item_count / cols))
        score = (
            abs(rows - cols),
            rows % 2 + cols % 2,
            rows * cols - item_count,
        )
        if score < best_score:
            best_rows, best_cols = rows, cols
            best_score = score

    return best_rows, best_cols


def _build_camera_grid(
    images: list[np.ndarray], cam_width: int, max_cols: int
) -> np.ndarray:
    """Arrange *images* into a grid of up to *max_cols* columns.

    Each image is resized to *cam_width*; rows are height-padded so
    hstack works.  The last row is padded with black if it has fewer
    than *max_cols* images.
    """
    resized = [_resize_to_width(img, cam_width) for img in images]
    cols = min(len(resized), max_cols)

    rows: list[np.ndarray] = []
    for start in range(0, len(resized), cols):
        chunk = resized[start : start + cols]
        target_h = max(c.shape[0] for c in chunk)
        padded = []
        for c in chunk:
            if c.shape[0] < target_h:
                c = np.pad(c, ((0, target_h - c.shape[0]), (0, 0), (0, 0)))
            padded.append(c)
        while len(padded) < cols:
            padded.append(np.zeros((target_h, cam_width, 3), dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))

    return np.concatenate(rows, axis=0)


def _build_grid_from_tiles(tiles: list[np.ndarray], max_cols: int) -> np.ndarray:
    """Arrange already-sized tiles into a grid with black padding."""
    if not tiles:
        raise ValueError("At least one tile is required to build a grid")

    cols = min(len(tiles), max_cols)
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), cols):
        chunk = tiles[start : start + cols]
        target_h = max(tile.shape[0] for tile in chunk)
        padded = []
        for tile in chunk:
            if tile.shape[0] < target_h:
                tile = np.pad(tile, ((0, target_h - tile.shape[0]), (0, 0), (0, 0)))
            padded.append(tile)
        while len(padded) < cols:
            padded.append(np.zeros((target_h, chunk[0].shape[1], 3), dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))

    return np.concatenate(rows, axis=0)


def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Resize an RGB image to *target_w* while keeping aspect ratio."""
    h, w = img.shape[:2]
    target_h = int(h * target_w / w)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _make_even(arr: np.ndarray) -> np.ndarray:
    """Crop 1 pixel from right/bottom if dimensions are odd (H.264 requirement)."""
    h, w = arr.shape[:2]
    return arr[: h - h % 2, : w - w % 2]


def _timeline_label(data_type: str) -> str:
    return "Action" if data_type == "action" else "State"


def build_episode_timeline_figure(
    arr: np.ndarray,
    arm_specs: list[ArmSpec],
    episode_id: int,
    data_type: str,
    frame_idx: int | None = None,
) -> Figure:
    """Render one episode timeline figure for action or state data."""
    fig, axes = plt.subplots(
        len(arm_specs),
        1,
        figsize=(10, max(2.8, 1.9 * len(arm_specs))),
        sharex=True,
        dpi=100,
        squeeze=False,
    )
    n_frames = arr.shape[0]
    t = np.arange(n_frames)

    for ax, arm in zip(axes.ravel(), arm_specs):
        segment = arr[:, arm.start : arm.stop]
        for offset, name in enumerate(arm.joint_names):
            if offset < segment.shape[1]:
                ax.plot(t, segment[:, offset], linewidth=0.8, label=name)
        if frame_idx is not None:
            ax.axvline(frame_idx, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlim(0, max(0, n_frames - 1))
        ax.margins(x=0)
        ax.set_ylabel(arm.label)
        ax.legend(fontsize=6, loc="upper right", ncol=4)

    axes.ravel()[-1].set_xlabel("Frame")
    suffix = f" — Frame {frame_idx}" if frame_idx is not None else ""
    fig.suptitle(f"{_timeline_label(data_type)} Timeline — Episode {episode_id}{suffix}", fontsize=11)
    fig.tight_layout()
    return fig


def _fig_to_array(fig: Figure) -> np.ndarray:
    """Rasterize a matplotlib Figure to an RGB numpy array."""
    fig.canvas.draw()
    canvas = cast(Any, fig.canvas)
    buf = canvas.buffer_rgba()
    return np.asarray(buf)[..., :3].copy()


def _render_timeline_background(
    arr: np.ndarray,
    arm_specs: list[ArmSpec],
    episode_id: int,
    data_type: str,
    width_px: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    """Render the static timeline plot background once and return plot bounds and x mapping.

    The figure width is set so the rasterized image matches *width_px*
    at the configured DPI.
    """
    dpi = 100
    fig_w = width_px / dpi
    fig, axes = plt.subplots(
        len(arm_specs),
        1,
        figsize=(fig_w, max(2.8, 1.9 * len(arm_specs))),
        sharex=True,
        dpi=dpi,
        squeeze=False,
    )
    n_frames = arr.shape[0]
    t = np.arange(n_frames)

    for ax, arm in zip(axes.ravel(), arm_specs):
        segment = arr[:, arm.start : arm.stop]
        for offset, name in enumerate(arm.joint_names):
            if offset < segment.shape[1]:
                ax.plot(t, segment[:, offset], linewidth=0.8, label=name)
        ax.set_xlim(0, max(0, n_frames - 1))
        ax.margins(x=0)
        ax.set_ylabel(arm.label)
        ax.legend(fontsize=6, loc="upper right", ncol=4)

    axes.ravel()[-1].set_xlabel("Frame")

    label = _timeline_label(data_type)
    fig.suptitle(
        f"{label} Timeline \u2014 Episode {episode_id}",
        fontsize=11,
    )
    fig.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxs = [ax.get_window_extent(renderer=renderer) for ax in axes.ravel()]
    plot_x0 = int(min(b.x0 for b in bboxs))
    plot_x1 = int(max(b.x1 for b in bboxs))
    plot_y0 = int(min(b.y0 for b in bboxs))
    plot_y1 = int(max(b.y1 for b in bboxs))

    canvas_w, canvas_h = fig.canvas.get_width_height()
    bottom_ax = axes.ravel()[-1]
    x0_data = int(round(bottom_ax.transData.transform((0, 0))[0]))
    x1_data = int(round(bottom_ax.transData.transform((max(0, n_frames - 1), 0))[0]))
    plot_bounds = (
        max(0, plot_x0),
        max(0, canvas_h - plot_y1),
        min(canvas_w - 1, plot_x1),
        min(canvas_h - 1, canvas_h - plot_y0),
    )

    rgb = _fig_to_array(fig)
    plt.close(fig)
    return rgb, plot_bounds, (x0_data, x1_data)


def _render_timeline_cursor(
    background: np.ndarray,
    plot_bounds: tuple[int, int, int, int],
    x_bounds: tuple[int, int],
    frame_idx: int,
    n_frames: int,
) -> np.ndarray:
    """Overlay the current frame cursor on a cached timeline background."""
    image = background.copy()
    x0, y0, x1, y1 = plot_bounds
    x_data0, x_data1 = x_bounds
    if n_frames <= 1:
        x = x_data0
    else:
        x = int(round(x_data0 + (frame_idx / (n_frames - 1)) * max(1, x_data1 - x_data0)))
    cv2.line(image, (x, y0), (x, y1), (255, 0, 0), 2)
    return image


def _decode_jpeg_frame(frame_bytes: bytes) -> np.ndarray:
    """Decode a cached JPEG frame bytes object to a BGR image."""
    buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode cached camera frame")
    return image


def _load_episode_camera_frames(
    ds: TeleopDataset, episode_id: int, camera_key: str
) -> list[np.ndarray]:
    """Load all frames for one episode/camera into memory once.

    This avoids reopening and re-decoding the source video on every exported
    frame. The returned frames are BGR images suitable for OpenCV compositing.
    """
    frame_count = ds.episode_length(episode_id)
    vmap = ds._episode_video_map[episode_id][camera_key]
    file_idx = int(vmap["file_index"])
    start_frame = int(vmap["start_frame"])

    if ds._preload_frames:
        store = ds._frame_store[camera_key][file_idx]
        return [_decode_jpeg_frame(store[start_frame + offset]) for offset in range(frame_count)]

    video_path = ds._get_video_path(camera_key, file_idx)
    container = av.open(str(video_path))
    try:
        frames: list[np.ndarray] = []
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index < start_frame:
                continue
            if len(frames) >= frame_count:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()

    if len(frames) != frame_count:
        raise RuntimeError(
            f"Failed to load all frames for episode {episode_id}, camera {camera_key}: "
            f"expected {frame_count}, got {len(frames)}"
        )

    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]


def _iter_episode_camera_frames(
    ds: TeleopDataset, episode_id: int, camera_key: str
):
    """Yield frames for one episode/camera without buffering the whole episode."""
    frame_count = ds.episode_length(episode_id)
    vmap = ds._episode_video_map[episode_id][camera_key]
    file_idx = int(vmap["file_index"])
    start_frame = int(vmap["start_frame"])

    if ds._preload_frames:
        store = ds._frame_store[camera_key][file_idx]
        for offset in range(frame_count):
            yield _decode_jpeg_frame(store[start_frame + offset])
        return

    video_path = ds._get_video_path(camera_key, file_idx)
    container = av.open(str(video_path))
    try:
        frames_seen = 0
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index < start_frame:
                continue
            if frames_seen >= frame_count:
                break
            frames_seen += 1
            rgb = frame.to_ndarray(format="rgb24")
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        container.close()

    if frame_count > 0 and frames_seen != frame_count:
        raise RuntimeError(
            f"Failed to load all frames for episode {episode_id}, camera {camera_key}: "
            f"expected {frame_count}, got {frames_seen}"
        )


def _build_camera_grid_bgr(images: list[np.ndarray], cam_width: int, max_cols: int) -> np.ndarray:
    """Build a BGR camera mosaic using OpenCV-friendly operations."""
    resized = [_resize_to_width(img, cam_width) for img in images]
    cols = min(len(resized), max_cols)

    rows: list[np.ndarray] = []
    for start in range(0, len(resized), cols):
        chunk = resized[start : start + cols]
        target_h = max(c.shape[0] for c in chunk)
        padded: list[np.ndarray] = []
        for c in chunk:
            if c.shape[0] < target_h:
                bottom = target_h - c.shape[0]
                c = cv2.copyMakeBorder(c, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded.append(c)
        while len(padded) < cols:
            padded.append(np.zeros((target_h, cam_width, 3), dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))

    return np.concatenate(rows, axis=0)


def _add_tile_border(image: np.ndarray, border_px: int, color: tuple[int, int, int]) -> np.ndarray:
    """Pad an image with a solid border color."""
    return cv2.copyMakeBorder(
        image,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=color,
    )


def _annotate_episode_tile(
    image: np.ndarray,
    episode_id: int,
    frame_idx: int,
    total_frames: int,
    ended: bool,
) -> np.ndarray:
    """Overlay a small episode label and end-state marker on a tile."""
    label = f"Episode {episode_id}"
    status = "ended" if ended else f"{frame_idx + 1}/{total_frames}"
    text = f"{label} | {status}"
    origin = (10, 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def export_episode_video(
    ds: TeleopDataset,
    episode_id: int,
    output_path: str | Path | None = None,
    data_type: str = "action",
    fps: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Render a composite video for one episode and write it to disk.

    Layout (N cameras arranged in rows of up to 3)::

        +----------+----------+----------+
        |  Cam 0   |  Cam 1   |  Cam 2   |
        +----------+----------+----------+
        |  Cam 3   |  Cam 4   | (black)  |
        +----------+----------+----------+
        |     Action / State Timeline     |
        +---------------------------------+

    Args:
        ds: Loaded TeleopDataset.
        episode_id: Which episode to export.
        output_path: Destination .mp4 path.  If *None* a temp file is used.
        data_type: ``"action"`` or ``"state"`` -- selects which timeline.
        fps: Frames-per-second for the output video.  Defaults to ``ds.fps``.
        progress_callback: Optional ``callback(current_frame, total_frames)``.

    Returns:
        Path to the written MP4 file.
    """
    if output_path is None:
        output_path = Path(
            tempfile.mktemp(suffix=".mp4", prefix=f"teleop_ep{episode_id}_")
        )
    output_path = Path(output_path)

    if fps is None:
        fps = ds.fps

    n_frames = ds.episode_length(episode_id)
    cams = ds.camera_keys
    traj = (
        ds.get_episode_actions(episode_id)
        if data_type == "action"
        else ds.get_episode_states(episode_id)
    )
    arm_specs = ds.arm_specs or [
        ArmSpec(name="arm_1", joint_names=tuple(ds.joint_names), start=0, stop=len(ds.joint_names))
    ]
    cols = min(len(cams), _MAX_COLS)
    composite_w = _CAM_WIDTH * cols
    timeline_background, timeline_bounds, timeline_x_bounds = _render_timeline_background(
        traj, arm_specs, episode_id, data_type, composite_w
    )
    progress_stride = max(1, n_frames // 100)

    if n_frames <= 0:
        raise RuntimeError(f"Episode {episode_id} has no frames to export")

    episode_camera_frames = {
        cam: _load_episode_camera_frames(ds, episode_id, cam) for cam in cams
    }

    def _compose_frame(frame_index: int) -> np.ndarray:
        cam_images = [episode_camera_frames[cam][frame_index] for cam in cams]
        cam_grid = _build_camera_grid_bgr(cam_images, _CAM_WIDTH, _MAX_COLS)
        timeline_rgb = _render_timeline_cursor(
            timeline_background, timeline_bounds, timeline_x_bounds, frame_index, n_frames
        )
        timeline_bgr = cv2.cvtColor(timeline_rgb, cv2.COLOR_RGB2BGR)
        return _make_even(np.concatenate([cam_grid, timeline_bgr], axis=0))

    first_frame = _compose_frame(0)

    writer = None
    for codec_name in _VIDEO_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        candidate = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(fps),
            (first_frame.shape[1], first_frame.shape[0]),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(
            "No usable video encoder is available through OpenCV for MP4 export. "
            f"Tried: {', '.join(_VIDEO_CODECS)}"
        )

    writer.write(first_frame)
    if progress_callback:
        progress_callback(1, n_frames)

    for chunk_start in range(1, n_frames, 100):
        chunk_end = min(chunk_start + 100, n_frames)
        for idx in range(chunk_start, chunk_end):
            composite = _compose_frame(idx)
            writer.write(composite)

            if progress_callback and (
                idx == n_frames - 1 or (idx + 1) % progress_stride == 0
            ):
                progress_callback(idx + 1, n_frames)

    writer.release()

    return output_path


def export_episode_grid_video(
    ds: TeleopDataset,
    episode_ids: list[int],
    camera_key: str,
    output_path: str | Path | None = None,
    fps: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Render a grid of episodes for one camera and write it to disk.

    Each cell shows one episode from the same camera. Cells keep advancing until
    their episode ends, then freeze on the final frame and switch to a green border.
    The export continues until the longest episode in the grid finishes.
    """
    if not episode_ids:
        raise ValueError("At least one episode is required for grid export")
    if camera_key not in ds.camera_keys:
        raise RuntimeError(f"Unknown camera key: {camera_key}")

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".mp4", prefix="teleop_grid_"))
    output_path = Path(output_path)

    if fps is None:
        fps = ds.fps

    episode_lengths = {ep_id: ds.episode_length(ep_id) for ep_id in episode_ids}
    max_frames = max(episode_lengths.values())
    _, cols = compute_grid_shape(len(episode_ids))
    progress_stride = max(1, max_frames // 100)

    episode_iters = {
        ep_id: _iter_episode_camera_frames(ds, ep_id, camera_key) for ep_id in episode_ids
    }
    episode_current_frames: dict[int, np.ndarray] = {}
    episode_final_frames: dict[int, np.ndarray] = {}

    for ep_id in episode_ids:
        try:
            first_frame = next(episode_iters[ep_id])
        except StopIteration as exc:
            raise RuntimeError(f"Episode {ep_id} has no frames to export") from exc
        episode_current_frames[ep_id] = first_frame
        episode_final_frames[ep_id] = first_frame

    def _compose_frame(frame_index: int) -> np.ndarray:
        tiles: list[np.ndarray] = []
        for ep_id in episode_ids:
            total_frames = episode_lengths[ep_id]
            ended = frame_index >= total_frames - 1
            current_idx = min(frame_index, total_frames - 1)
            if not ended:
                if current_idx == 0:
                    frame = episode_current_frames[ep_id]
                else:
                    frame = episode_current_frames[ep_id]
            else:
                frame = episode_final_frames[ep_id]
            tile = _resize_to_width(frame, _CAM_WIDTH)
            border_color = (0, 255, 0) if ended else (40, 40, 40)
            tile = _add_tile_border(tile, 6, border_color)
            tile = _annotate_episode_tile(tile, ep_id, current_idx, total_frames, ended)
            tiles.append(tile)

        return _make_even(_build_grid_from_tiles(tiles, cols))

    first_frame = _compose_frame(0)

    writer = None
    for codec_name in _VIDEO_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        candidate = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(fps),
            (first_frame.shape[1], first_frame.shape[0]),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(
            "No usable video encoder is available through OpenCV for MP4 export. "
            f"Tried: {', '.join(_VIDEO_CODECS)}"
        )

    writer.write(first_frame)
    if progress_callback:
        progress_callback(1, max_frames)

    for chunk_start in range(1, max_frames, 100):
        chunk_end = min(chunk_start + 100, max_frames)
        for idx in range(chunk_start, chunk_end):
            for ep_id in episode_ids:
                total_frames = episode_lengths[ep_id]
                if idx < total_frames:
                    try:
                        episode_current_frames[ep_id] = next(episode_iters[ep_id])
                        episode_final_frames[ep_id] = episode_current_frames[ep_id]
                    except StopIteration:
                        episode_current_frames[ep_id] = episode_final_frames[ep_id]
            composite = _compose_frame(idx)
            writer.write(composite)

            if progress_callback and (
                idx == max_frames - 1 or (idx + 1) % progress_stride == 0
            ):
                progress_callback(idx + 1, max_frames)

    writer.release()

    return output_path

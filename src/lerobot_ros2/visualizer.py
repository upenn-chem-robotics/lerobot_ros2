"""Live OpenCV preview shared by the record and deploy scripts.

Owns the cv2 window lifecycle, frame tiling, display sizing, graceful
fallback when the GUI is unavailable, and keyboard polling. Callers
supply camera frames and an optional overlay callback; state-specific
overlays (REC/IDLE badges, mode dots, etc.) live in the caller.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .helper import quiet_stderr

__all__ = ["LivePreview", "tile_frames", "tile_frames_grid"]


def tile_frames(
    frames: Sequence[Optional[np.ndarray]],
    labels: Sequence[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    n = len(frames)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    cell_w = target_w
    cell_h = target_h // n
    canvas = np.zeros((n * cell_h, cell_w, 3), dtype=np.uint8)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        y0 = i * cell_h
        cell = _fit_frame_to_cell(frame, cell_w, cell_h)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y0:y0 + cell_h] = cell
    return canvas


def tile_frames_grid(
    frames: Sequence[Optional[np.ndarray]],
    labels: Sequence[str],
    target_w: int,
    target_h: int,
    columns: int,
) -> np.ndarray:
    columns = max(1, columns)
    n = len(frames)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    rows = int(np.ceil(n / columns))
    cell_w = max(1, target_w // columns)
    cell_h = max(1, target_h // rows)
    canvas = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)

    for i, (frame, label) in enumerate(zip(frames, labels)):
        row = i // columns
        col = i % columns
        y0 = row * cell_h
        x0 = col * cell_w
        cell = _fit_frame_to_cell(frame, cell_w, cell_h)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    return canvas


def _fit_frame_to_cell(frame: Optional[np.ndarray], cell_w: int, cell_h: int) -> np.ndarray:
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if frame is None:
        return cell

    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return cell

    scale = min(cell_w / frame_w, cell_h / frame_h)
    resized_w = max(1, int(round(frame_w * scale)))
    resized_h = max(1, int(round(frame_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h))

    x0 = (cell_w - resized_w) // 2
    y0 = (cell_h - resized_h) // 2
    cell[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return cell


class LivePreview:
    """cv2 window wrapper: tiling, overlay callback, keyboard polling, and graceful fallback.

    Use as a context manager:

        with LivePreview(name, labels, fullscreen=True) as preview:
            while preview.available and not stop_event.is_set():
                frames = [cam.get_frame() for cam in cameras]
                key = preview.render(frames, draw_overlay=my_overlay_fn)
                if key in (ord('q'), 27):
                    break
    """

    def __init__(
        self,
        window_name: str,
        cam_labels: Sequence[str],
        *,
        fullscreen: bool = False,
        initial_size: Optional[Tuple[int, int]] = None,
        grid_columns: Optional[int] = None,
        default_size: Tuple[int, int] = (1280, 720),
        wait_key_ms: int = 30,
    ) -> None:
        self._window_name = window_name
        self._cam_labels = list(cam_labels)
        self._fullscreen = fullscreen
        self._initial_size = initial_size
        self._grid_columns = grid_columns
        self._default_size = default_size
        self._wait_key_ms = max(1, int(wait_key_ms))
        self._available = False
        self._opened = False
        self._closed = False

    @property
    def available(self) -> bool:
        return self._available

    def __enter__(self) -> "LivePreview":
        with quiet_stderr():
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            if self._fullscreen:
                cv2.setWindowProperty(
                    self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
            else:
                if self._initial_size is not None:
                    cv2.setWindowProperty(
                        self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                    cv2.resizeWindow(
                        self._window_name,
                        int(self._initial_size[0]),
                        int(self._initial_size[1]),
                    )
        self._opened = True
        self._available = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _display_rect(self) -> Tuple[int, int]:
        default_w, default_h = self._default_size
        if not self._available:
            return default_w, default_h
        try:
            rect = cv2.getWindowImageRect(self._window_name)
            return max(rect[2], 640), max(rect[3], 480)
        except cv2.error as exc:
            self._disable("OpenCV preview unavailable; continuing without visualization: %s", exc)
            return default_w, default_h

    def _disable(self, msg: str, *args: object) -> None:
        if self._available:
            logging.warning(msg, *args)
        self._available = False

    def set_labels(self, cam_labels: Sequence[str]) -> None:
        """Replace the per-tile labels, e.g. to annotate a camera's live state."""
        self._cam_labels = list(cam_labels)

    def render(
        self,
        frames: Sequence[Optional[np.ndarray]],
        draw_overlay: Optional[Callable[[np.ndarray, int, int], None]] = None,
    ) -> int:
        if not self._available:
            time.sleep(self._wait_key_ms / 1000.0)
            return 255

        disp_w, disp_h = self._display_rect()

        if self._grid_columns is None:
            canvas = tile_frames(frames, self._cam_labels, disp_w, disp_h)
        else:
            canvas = tile_frames_grid(
                frames, self._cam_labels, disp_w, disp_h, self._grid_columns
            )

        if draw_overlay is not None:
            try:
                draw_overlay(canvas, disp_w, disp_h)
            except Exception:
                logging.exception("draw_overlay callback raised; skipping overlay this frame")

        try:
            cv2.imshow(self._window_name, canvas)
            return cv2.waitKey(self._wait_key_ms) & 0xFF
        except cv2.error as exc:
            self._disable("OpenCV preview disabled after GUI error: %s", exc)
            time.sleep(self._wait_key_ms / 1000.0)
            return 255

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._available = False
        if not self._opened:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
"""Interactive ROI picker for dataset videos.

Point it at one or more video files (or directories / globs of videos). For each
video a window opens where you scrub across all frames (trackbar or ``a``/``d`` =
-1/+1, ``j``/``l`` = -10/+10) and drag a crop box directly on the (upscaled)
image. It prints the crop spec ready to paste into
``dp64_cfg.json -> dataset.image_transforms.per_camera_crops``.

Keys: drag = draw box, ``r`` = clear box, ENTER/SPACE = accept, ``c``/ESC = skip
this video, ``q`` = quit.

Examples::

    python -m lerobot_ros2.cli.cut path/to/file-000.mp4
    python -m lerobot_ros2.cli.cut data/.../videos/*/chunk-000/file-000.mp4
    python -m lerobot_ros2.cli.cut data/.../videos --frame 30      # start on frame 30
    python -m lerobot_ros2.cli.cut data/.../videos --second 2.5    # start at t=2.5s
    python -m lerobot_ros2.cli.cut data/.../videos --frame 30 --no-scrub  # skip scrubber
"""

import argparse
import glob
import os

import cv2

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def collect_videos(inputs):
    """Expand files, directories, and glob patterns into a sorted video list."""
    videos = []
    for item in inputs:
        if os.path.isfile(item):
            videos.append(item)
        elif os.path.isdir(item):
            for ext in VIDEO_EXTS:
                videos.extend(glob.glob(os.path.join(item, "**", f"*{ext}"), recursive=True))
        else:
            videos.extend(glob.glob(item, recursive=True))
    # De-dup while keeping only real video files.
    seen = set()
    out = []
    for path in sorted(videos):
        real = os.path.realpath(path)
        if real in seen:
            continue
        if not path.lower().endswith(VIDEO_EXTS):
            continue
        seen.add(real)
        out.append(path)
    return out


def camera_key_for(video_path):
    """Derive an ``observation.images.<name>`` key from the video path.

    Datasets store videos under ``videos/observation.images.<name>/chunk-*/file-*.mp4``,
    so we walk up the path looking for that directory. Falls back to the file stem.
    """
    parts = os.path.normpath(video_path).split(os.sep)
    for part in reversed(parts):
        if part.startswith("observation.images."):
            return part
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"observation.images.{stem}"


def read_frame(video_path, frame_idx):
    """Return the requested frame (BGR) from ``video_path`` or ``None``.

    Decodes with PyAV (software, e.g. ``libdav1d``) rather than
    ``cv2.VideoCapture``, whose FFmpeg backend cannot decode the AV1-encoded
    dataset videos in many environments. This matches the decode path used
    throughout the rest of the codebase (see ``data_loader._decode_single_frame``).
    """
    import av  # type: ignore[import-not-found]

    target = max(0, int(frame_idx))
    frame_rgb = None
    last_rgb = None
    try:
        container = av.open(str(video_path))
        try:
            for i, frame in enumerate(container.decode(video=0)):
                last_rgb = frame.to_ndarray(format="rgb24")
                if i == target:
                    frame_rgb = last_rgb
                    break
        finally:
            container.close()
    except Exception:
        return None

    # If the requested index was past the end, fall back to the last frame seen.
    if frame_rgb is None:
        frame_rgb = last_rgb
    if frame_rgb is None:
        return None

    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


class LazyFrames:
    """On-demand frame reader that decodes forward and caches as it goes.

    The scrubber must not block for the whole clip before showing anything (that
    made every window appear frozen on long/full-res videos). This opens the
    container once, decodes the first frame immediately, and only decodes
    further when a later frame is actually requested. Already-seen frames are
    cached, so scrubbing backward is instant. Uses the same PyAV/AV1 decode path
    as :func:`read_frame`.
    """

    def __init__(self, video_path):
        import av  # type: ignore[import-not-found]

        self._container = av.open(str(video_path))
        stream = self._container.streams.video[0]
        self.fps = float(stream.average_rate) if stream.average_rate else None

        approx = 0
        if stream.frames:
            approx = int(stream.frames)
        elif stream.duration is not None and stream.time_base and self.fps:
            approx = int(float(stream.duration * stream.time_base) * self.fps)
        elif self._container.duration is not None and self.fps:
            approx = int((self._container.duration / av.time_base) * self.fps)
        self._approx_n = max(approx, 1)

        self._decoder = self._container.decode(video=0)
        self._cache: dict = {}
        self._next = 0
        self._exhausted = False
        self._decode_until(0)

    def _decode_until(self, target):
        while not self._exhausted and self._next <= target:
            try:
                frame = next(self._decoder)
            except StopIteration:
                self._exhausted = True
                break
            except Exception:
                self._exhausted = True
                break
            rgb = frame.to_ndarray(format="rgb24")
            self._cache[self._next] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._next += 1

    def __len__(self):
        if self._exhausted:
            return max(len(self._cache), 1)
        return max(self._approx_n, self._next)

    def get(self, i):
        i = max(0, int(i))
        if i not in self._cache:
            self._decode_until(i)
        if i in self._cache:
            return self._cache[i]
        if not self._cache:
            return None
        return self._cache[max(self._cache)]

    def close(self):
        try:
            self._container.close()
        except Exception:
            pass


def pick_frame_and_roi(frames, cam_key, start_idx, allow_scrub=True):
    """Scrub ``frames`` and drag a crop box, all in one window.

    Drawing happens in the *same* window as scrubbing (no window handoff, which
    is what broke mouse input on some GTK/Qt builds), and the frame is shown
    upscaled so a tiny 256x144 image is easy to draw on precisely.

    Controls: drag to draw the box, ``r`` clears it, ENTER/SPACE accepts,
    ``c``/ESC skips this video, ``q`` quits. When ``allow_scrub`` is set, the
    trackbar plus ``a``/``d`` (+/-1) and ``j``/``l`` (+/-10) move between frames.

    Returns ``(idx, (x, y, w, h))`` in image coordinates when accepted, ``None``
    to skip this video, or the string ``"quit"`` to stop everything.
    """
    fps = frames.fps
    n = len(frames)
    idx = [min(max(int(start_idx), 0), max(n - 1, 0))]

    first = frames.get(idx[0])
    if first is None:
        return None
    h, w = first.shape[:2]
    scale = max(1, min(8, round(1024 / max(w, 1))))

    # Box stored in display (upscaled) coordinates.
    box = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "drawing": False, "has": False}

    def _roi_image_coords():
        x0 = max(0, min(box["x0"], box["x1"])) // scale
        y0 = max(0, min(box["y0"], box["y1"])) // scale
        x1 = min(w, (max(box["x0"], box["x1"]) + scale - 1) // scale)
        y1 = min(h, (max(box["y0"], box["y1"]) + scale - 1) // scale)
        return int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0))

    window = f"{cam_key}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    if allow_scrub:
        def _on_track(v):
            idx[0] = v

        cv2.createTrackbar("frame", window, idx[0], max(n - 1, 1), _on_track)

    def _on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            box.update(x0=mx, y0=my, x1=mx, y1=my, drawing=True, has=False)
        elif event == cv2.EVENT_MOUSEMOVE and box["drawing"]:
            box["x1"], box["y1"] = mx, my
        elif event == cv2.EVENT_LBUTTONUP:
            box["x1"], box["y1"] = mx, my
            box["drawing"] = False
            box["has"] = True

    cv2.setMouseCallback(window, _on_mouse)

    try:
        while True:
            i = idx[0]
            clean = frames.get(i)
            if clean is None:
                return None
            total = len(frames)
            disp = cv2.resize(
                clean, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
            )

            if box["has"] or box["drawing"]:
                cv2.rectangle(
                    disp,
                    (min(box["x0"], box["x1"]), min(box["y0"], box["y1"])),
                    (max(box["x0"], box["x1"]), max(box["y0"], box["y1"])),
                    (0, 0, 255), 2,
                )

            rx, ry, rw, rh = _roi_image_coords()
            t = (i / fps) if fps else 0.0
            line1 = f"frame {i}/{total - 1}  t={t:.2f}s"
            line2 = (
                f"box top={ry} left={rx} h={rh} w={rw}"
                if (box["has"] and rw > 0 and rh > 0)
                else "drag to draw a box"
            )
            for row, text in ((22, line1), (46, line2)):
                cv2.putText(
                    disp, text, (8, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )
            cv2.imshow(window, disp)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):  # ENTER / SPACE
                if box["has"] and rw > 0 and rh > 0:
                    return i, (rx, ry, rw, rh)
                print("  draw a box first (drag on the image)")
                continue
            if key in (27, ord("c")):  # ESC / c
                return None
            if key == ord("q"):
                return "quit"
            if key == ord("r"):
                box["has"] = False
                continue
            if allow_scrub:
                if key == ord("a"):
                    idx[0] = max(0, i - 1)
                elif key == ord("d"):
                    idx[0] = min(total - 1, i + 1)
                elif key == ord("j"):
                    idx[0] = max(0, i - 10)
                elif key == ord("l"):
                    idx[0] = min(total - 1, i + 10)
                else:
                    continue
                cv2.setTrackbarPos("frame", window, idx[0])
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(
        description="Pick per-camera crop ROIs from dataset videos."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Video file(s), directory(ies), or glob pattern(s).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to start the scrubber on (default: 0).",
    )
    parser.add_argument(
        "--second",
        type=float,
        default=None,
        help="Timestamp (seconds) to start the scrubber on; overrides --frame.",
    )
    parser.add_argument(
        "--no-scrub",
        action="store_true",
        help="Hide the frame trackbar/keys; just draw a box on --frame/--second.",
    )
    args = parser.parse_args()

    videos = collect_videos(args.videos)
    if not videos:
        raise FileNotFoundError(f"No videos found for inputs: {args.videos}")

    print(f"found {len(videos)} video(s)\n")

    crops = {}
    for video_path in videos:
        cam_key = camera_key_for(video_path)
        print(f"video: {video_path}")
        print(f"camera: {cam_key}")

        try:
            frames = LazyFrames(video_path)
        except Exception as exc:
            print(f"  could not open video ({exc}); skipping\n")
            continue
        fps = frames.fps
        if args.second is not None:
            if not fps:
                print("  unknown fps; --second ignored, starting at --frame")
                start_idx = args.frame
            else:
                start_idx = round(args.second * fps)
        else:
            start_idx = args.frame

        try:
            chosen = pick_frame_and_roi(
                frames, cam_key, start_idx, allow_scrub=not args.no_scrub
            )
        finally:
            frames.close()

        if chosen == "quit":
            print("  quit requested; stopping\n")
            break
        if chosen is None:
            print("  skipped\n")
            continue

        chosen_idx, (x, y, w, h) = chosen
        t = (chosen_idx / fps) if fps else 0.0
        print(f"  picked frame {chosen_idx} (t={t:.2f}s)")
        if w == 0 or h == 0:
            print("  empty selection; skipping\n")
            continue

        crops[cam_key] = {"top": int(y), "left": int(x), "height": int(h), "width": int(w)}
        print(f"  top={y}, left={x}, height={h}, width={w}\n")

    if not crops:
        print("no ROIs selected")
        return

    print("paste into dp64_cfg.json -> dataset.image_transforms.per_camera_crops:")
    print("  {")
    items = list(crops.items())
    for i, (cam_key, spec) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        print(
            f'    "{cam_key}": {{ "top": {spec["top"]}, "left": {spec["left"]}, '
            f'"height": {spec["height"]}, "width": {spec["width"]} }}{comma}'
        )
    print("  }")


if __name__ == "__main__":
    main()

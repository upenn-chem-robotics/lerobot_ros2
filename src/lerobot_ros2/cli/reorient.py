#!/usr/bin/env python
"""Rotate chosen cameras 180 degrees for chosen episodes of a LeRobot v3.0 dataset.

The OBSBOT wrist cameras auto-rotate their own image when their internal gravity
sensor decides they are upside down, and the firmware exposes no way to lock the
orientation. When the arm rolls far enough during an episode the feed inverts and
stays inverted, so a recording session can end up with some episodes the right
way up and some upside down. A policy trained on that mixture sees the wrist view
in two different frames of reference.

This tool fixes it after the fact, on a dataset you name, writing to a new
directory. It never modifies the source.

The workflow is two steps, because deciding *which* episodes are inverted is a
judgement call and a wrong guess silently corrupts the training data:

1. ``--report`` renders a contact sheet of every episode's first frame, one PNG
   per camera, tiles labelled with their episode index. You read off the
   inverted episodes by eye.
2. Pass those episode numbers to a rewrite run.

The report also prints a boundary-continuity table (last frame of episode N
against the first frame of N+1, normal versus rotated). Treat it as advisory
only: the two scores can land within a few points of each other on a busy
scene, so the contact sheet is what decides.

Videos are packed -- one mp4 holds many episodes, addressed by timestamp -- so
rotating an episode means rotating a frame range inside a shared file. Episode E
of camera C occupies frames ``[round(from_timestamp * fps), round(to_timestamp *
fps))`` of its file. Some episodes span more frames than they have dataset rows
(footage from a discarded take is retained); rotating the whole span is correct,
since the surplus frames are never read.

The rotation is a single ffmpeg pass using timeline-gated ``hflip,vflip``, so the
frame count is preserved structurally and untouched episodes never make a round
trip through a Python frame buffer. Files with no selected episode are copied
byte for byte, and so are the videos of every camera you did not name, which
keeps the re-encode confined to what you asked for.

Examples::

    # 1. look at what you have (read-only, writes only the contact sheets)
    lerobot-ros-reorient \
        --src data/smrithi/salt_20260725/original_data \
        --report --report-dir /tmp/orient

    # 2. rotate the episodes that came out upside down
    lerobot-ros-reorient \
        --src data/smrithi/salt_20260725/original_data \
        --dst data/smrithi/salt_20260725/reoriented \
        --camera left_wrist_top --episodes 3,7,10-14

    # a whole session that was inverted end to end
    lerobot-ros-reorient --src <in> --dst <out> \
        --camera left_wrist_top --episodes all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from lerobot_ros2.visualizer import tile_frames_grid

FFMPEG = "/bin/ffmpeg"
FFPROBE = "/bin/ffprobe"
IMG_PREFIX = "observation.images."

_CAMERA_INDEX_RE = re.compile(r"^camera_\d+_(?P<name>.+)$")

# Dataset content this tool understands and reproduces. Anything else found in
# the source root (derived review media under exports/, PNG frame dumps under
# images/, stray clips) is deliberately left behind rather than duplicated: it
# is all regenerable, it can dwarf the dataset itself, and copying a *stale*
# unrotated frame dump next to rotated videos would be actively misleading.
_COPIED_DIRS = ("meta", "data")

# Contact sheet tile size. Small enough that 65 episodes stay inside one
# reasonable PNG, large enough to tell up from down at a glance.
_SHEET_CELL_W = 320
_SHEET_CELL_H = 180


# ── source inspection ────────────────────────────────────────────────────


def camera_short_name(feature_key: str) -> str:
    """Camera name from a video feature key, with any ``camera_NN_`` index dropped.

    Both raw record-time keys (``observation.images.camera_01_left_wrist_top``)
    and downsampled keys (``observation.images.left_wrist_top``) reduce to
    ``left_wrist_top``, so ``--camera`` names the same thing at every stage of
    the pipeline.
    """
    name = feature_key
    idx = name.rfind(IMG_PREFIX)
    if idx != -1:
        name = name[idx + len(IMG_PREFIX):]
    match = _CAMERA_INDEX_RE.match(name)
    return match.group("name") if match else name


@dataclass(frozen=True)
class DatasetMeta:
    root: Path
    info: dict
    episodes: pd.DataFrame
    fps: float
    video_path_template: str
    # short camera name -> full video feature key
    camera_keys: Dict[str, str]

    @property
    def episode_ids(self) -> List[int]:
        return sorted(int(v) for v in self.episodes["episode_index"].unique())


def load_dataset_meta(root: Path) -> DatasetMeta:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(
            f"{root} is not a dataset root (no meta/info.json). Point --src at the "
            "directory that holds meta/, data/ and videos/."
        )
    info = json.loads(info_path.read_text())

    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise SystemExit(f"No episode metadata under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)

    fps = float(info.get("fps") or 0.0)
    if fps <= 0:
        raise SystemExit(f"{info_path} has no usable fps (got {info.get('fps')!r})")

    template = info.get("video_path")
    if not template:
        raise SystemExit(f"{info_path} has no video_path template")

    camera_keys: Dict[str, str] = {}
    for key, feature in (info.get("features") or {}).items():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        short = camera_short_name(key)
        if short in camera_keys:
            raise SystemExit(
                f"Two video features reduce to the camera name {short!r}: "
                f"{camera_keys[short]!r} and {key!r}"
            )
        camera_keys[short] = key
    if not camera_keys:
        raise SystemExit(f"{info_path} declares no video features")

    return DatasetMeta(
        root=root,
        info=info,
        episodes=episodes,
        fps=fps,
        video_path_template=template,
        camera_keys=camera_keys,
    )


def resolve_cameras(meta: DatasetMeta, requested: Sequence[str]) -> List[str]:
    """Validate ``--camera`` names against the dataset, preserving the given order."""
    resolved: List[str] = []
    for name in requested:
        short = camera_short_name(name.strip())
        if short not in meta.camera_keys:
            raise SystemExit(
                f"--camera {name!r} is not a camera in this dataset. "
                f"Available: {sorted(meta.camera_keys)}"
            )
        if short not in resolved:
            resolved.append(short)
    return resolved


def parse_episode_spec(spec: str, valid_ids: Sequence[int]) -> List[int]:
    """Expand ``all`` / ``3`` / ``3,7`` / ``10-14`` into a sorted episode list.

    Every referenced episode must exist in the dataset; a typo that would
    silently rotate nothing (or the wrong take) is rejected instead.
    """
    text = (spec or "").strip()
    if not text:
        raise SystemExit("--episodes may not be empty (use 'all' or e.g. 3,7,10-14)")

    valid = set(int(v) for v in valid_ids)
    if text.lower() == "all":
        return sorted(valid)

    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token.lstrip("-"):
            lo_text, _, hi_text = token.partition("-")
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError:
                raise SystemExit(f"--episodes range {token!r} is not two integers")
            if lo > hi:
                raise SystemExit(f"--episodes range {token!r} is inverted (low > high)")
            selected.update(range(lo, hi + 1))
        else:
            try:
                selected.add(int(token))
            except ValueError:
                raise SystemExit(f"--episodes entry {token!r} is not an integer")

    if not selected:
        raise SystemExit(f"--episodes {spec!r} selected no episodes")
    missing = sorted(selected - valid)
    if missing:
        raise SystemExit(
            f"--episodes references episode(s) not in this dataset: {missing}. "
            f"Valid range is {min(valid)}-{max(valid)}."
        )
    return sorted(selected)


# ── episode to frame range ───────────────────────────────────────────────


@dataclass(frozen=True)
class FrameRange:
    """Half-open ``[start, end)`` frame span of one episode inside one mp4."""

    episode: int
    file_index: int
    chunk_index: int
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


def episode_frame_ranges(
    meta: DatasetMeta,
    feature_key: str,
    episodes: Sequence[int],
) -> List[FrameRange]:
    """Frame spans occupied by ``episodes`` in ``feature_key``'s packed videos.

    Frame indices come from the per-camera timestamps rather than from episode
    lengths: a few episodes retain trailing footage from a discarded take, so
    the span is the authoritative extent of what belongs to that episode in the
    file, and ``length`` undercounts it.
    """
    prefix = f"videos/{feature_key}/"
    needed = [f"{prefix}chunk_index", f"{prefix}file_index",
              f"{prefix}from_timestamp", f"{prefix}to_timestamp"]
    missing = [c for c in needed if c not in meta.episodes.columns]
    if missing:
        raise SystemExit(
            f"Episode metadata has no video columns for {feature_key!r} "
            f"(missing {missing}). Is this camera present in the dataset?"
        )

    wanted = set(int(e) for e in episodes)
    ranges: List[FrameRange] = []
    for _, row in meta.episodes.iterrows():
        episode = int(row["episode_index"])
        if episode not in wanted:
            continue
        start = int(round(float(row[f"{prefix}from_timestamp"]) * meta.fps))
        end = int(round(float(row[f"{prefix}to_timestamp"]) * meta.fps))
        if end <= start:
            raise SystemExit(
                f"Episode {episode} of {feature_key} has an empty video span "
                f"({start} -> {end}); refusing to guess."
            )
        ranges.append(
            FrameRange(
                episode=episode,
                file_index=int(row[f"{prefix}file_index"]),
                chunk_index=int(row[f"{prefix}chunk_index"]),
                start=start,
                end=end,
            )
        )
    return ranges


def group_by_file(ranges: Sequence[FrameRange]) -> Dict[Tuple[int, int], List[FrameRange]]:
    """Bucket frame ranges by the ``(chunk_index, file_index)`` mp4 they live in."""
    grouped: Dict[Tuple[int, int], List[FrameRange]] = {}
    for item in ranges:
        grouped.setdefault((item.chunk_index, item.file_index), []).append(item)
    for key in grouped:
        grouped[key].sort(key=lambda r: r.start)
    return grouped


def build_enable_expr(ranges: Sequence[FrameRange]) -> str:
    """ffmpeg timeline expression selecting every frame in ``ranges``.

    ``between`` is inclusive at both ends so the half-open ``end`` becomes
    ``end - 1``. Summing the terms is the idiomatic OR here: each ``between``
    yields 0 or 1, and any nonzero total enables the filter.
    """
    if not ranges:
        raise ValueError("build_enable_expr needs at least one range")
    terms = [f"between(n,{r.start},{r.end - 1})" for r in ranges]
    return "+".join(terms)


def video_file_path(meta: DatasetMeta, feature_key: str, chunk: int, file_index: int) -> Path:
    rel = meta.video_path_template.format(
        video_key=feature_key, chunk_index=chunk, file_index=file_index
    )
    return meta.root / rel


# ── video rewriting ──────────────────────────────────────────────────────


def probe_frame_count(path: Path) -> int:
    """Exact frame count of ``path``, by decoding it (slow but authoritative)."""
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    text = (result.stdout or "").strip().splitlines()
    if not text or not text[0].strip().isdigit():
        raise SystemExit(f"Could not count frames in {path} (ffprobe said {result.stdout!r})")
    return int(text[0].strip())


def rotate_video_file(
    src_mp4: Path,
    dst_mp4: Path,
    ranges: Sequence[FrameRange],
    crf: int,
    preset: str,
) -> None:
    """Re-encode ``src_mp4`` to ``dst_mp4`` with ``ranges`` rotated 180 degrees.

    Uses timeline-gated ``hflip,vflip`` so ffmpeg rotates exactly the selected
    frames in one pass and every other frame flows through the filter graph
    untouched. Frame count is preserved by construction and verified after.
    """
    expr = build_enable_expr(ranges)
    vf = f"hflip=enable='{expr}',vflip=enable='{expr}'"
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vf", vf,
        # -vsync 0 passes source timestamps through untouched; without it ffmpeg
        # may duplicate or drop frames to fit a constant rate, which would break
        # the timestamp-to-frame mapping every episode depends on.
        "-vsync", "0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(int(crf)),
        "-an",
        str(dst_mp4),
    ]
    subprocess.run(cmd, check=True)

    before, after = probe_frame_count(src_mp4), probe_frame_count(dst_mp4)
    if before != after:
        raise SystemExit(
            f"Frame count changed for {src_mp4.name}: {before} -> {after}. "
            "Refusing to leave a dataset whose episode timestamps no longer line up."
        )


# ── frame reading (report mode) ──────────────────────────────────────────


def decode_frames_at(path: Path, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Decode the requested frame ``indices`` from ``path``, returned as BGR.

    Decodes forward once and stops after the highest index. PyAV rather than
    ``cv2.VideoCapture`` because the dataset videos are AV1, which OpenCV's
    FFmpeg backend cannot decode in many environments -- the same reason
    ``cut.py`` and ``data_loader.py`` use PyAV.
    """
    import av  # imported lazily so --help works without the optional dep

    wanted = set(int(i) for i in indices)
    if not wanted:
        return {}
    ceiling = max(wanted)
    out: Dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                out[i] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if i >= ceiling:
                break
    return out


def _orientation_scores(reference: np.ndarray, candidate: np.ndarray) -> Tuple[float, float]:
    """Mean absolute difference of ``candidate`` against ``reference``, un/rotated."""
    ref = reference.astype(np.int16)
    cand = candidate.astype(np.int16)
    normal = float(np.abs(cand - ref).mean())
    rotated = float(np.abs(cv2.rotate(candidate, cv2.ROTATE_180).astype(np.int16) - ref).mean())
    return normal, rotated


def scan_camera_edges(
    meta: DatasetMeta,
    camera: str,
    episodes: Sequence[int],
) -> Tuple[Dict[Tuple[int, int], List[FrameRange]], Dict[Tuple[int, int], Dict[int, np.ndarray]]]:
    """Decode each episode's first and last frame, one pass per packed file.

    The contact sheet needs the first frames and the boundary table needs first
    and last frames, so both are gathered in a single decode: these files hold
    tens of thousands of frames and a second pass would double the wait for
    nothing.
    """
    feature_key = meta.camera_keys[camera]
    grouped = group_by_file(episode_frame_ranges(meta, feature_key, episodes))
    decoded: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
    for file_key, items in sorted(grouped.items()):
        chunk, file_index = file_key
        path = video_file_path(meta, feature_key, chunk, file_index)
        if not path.is_file():
            print(f"[warn] missing {path}; skipping its episodes")
            decoded[file_key] = {}
            continue
        indices: List[int] = []
        for item in items:
            indices.extend((item.start, item.end - 1))
        decoded[file_key] = decode_frames_at(path, indices)
    return grouped, decoded


def contact_sheet(
    grouped: Dict[Tuple[int, int], List[FrameRange]],
    decoded: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    columns: int,
) -> Optional[np.ndarray]:
    """Grid of each episode's first frame, tiles labelled by episode index."""
    pairs: List[Tuple[int, Optional[np.ndarray]]] = []
    for file_key, items in sorted(grouped.items()):
        frames = decoded.get(file_key, {})
        for item in items:
            pairs.append((item.episode, frames.get(item.start)))
    if not pairs:
        return None

    pairs.sort(key=lambda pair: pair[0])
    labels = [f"ep {episode}" for episode, _ in pairs]
    frames = [frame for _, frame in pairs]
    columns = max(1, columns)
    rows = int(np.ceil(len(frames) / columns))
    return tile_frames_grid(
        frames, labels, columns * _SHEET_CELL_W, rows * _SHEET_CELL_H, columns
    )


def boundary_report(
    camera: str,
    grouped: Dict[Tuple[int, int], List[FrameRange]],
    decoded: Dict[Tuple[int, int], Dict[int, np.ndarray]],
) -> List[int]:
    """Print the advisory normal-versus-rotated table across episode boundaries.

    Consecutive episodes inside one packed file are adjacent frames, so a
    firmware flip between them shows up as the rotated score beating the normal
    one. It is only a hint: on a scene that changed a lot during the reset the
    two scores can come within a few points of each other, which is why the
    contact sheet is what decides. Returns the episodes it suspects.
    """
    print(f"\n  boundary continuity for {camera} (advisory only):")
    flagged: List[int] = []
    for file_key, items in sorted(grouped.items()):
        frames = decoded.get(file_key, {})
        for prev, nxt in zip(items, items[1:]):
            last = frames.get(prev.end - 1)
            first = frames.get(nxt.start)
            if last is None or first is None:
                continue
            normal, rotated = _orientation_scores(last, first)
            mark = ""
            if rotated < normal:
                mark = "   <-- possible flip"
                flagged.append(nxt.episode)
            print(
                f"    ep{prev.episode:>3} -> ep{nxt.episode:<3} "
                f"normal={normal:7.2f} rotated={rotated:7.2f}{mark}"
            )
    if flagged:
        print(f"    possible flips at: {_summarize_ids(flagged)}")
    else:
        print("    no boundary suggests a flip; this camera looks uniform")
    return flagged


# ── destination writing ──────────────────────────────────────────────────


def prepare_destination(dst: Path, overwrite: bool) -> None:
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"{dst} already exists and is not empty. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def copy_passthrough(src: Path, dst: Path) -> List[str]:
    """Copy the dataset content that is not being rotated.

    Returns the names of source entries that were deliberately skipped, so the
    caller can say so out loud rather than let them go missing quietly.
    """
    skipped: List[str] = []
    for entry in sorted(src.iterdir()):
        if entry.name == "videos":
            continue
        if entry.is_dir():
            if entry.name in _COPIED_DIRS:
                shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
            else:
                skipped.append(entry.name + "/")
        elif entry.is_file():
            if entry.suffix.lower() in (".yaml", ".yml", ".json", ".md", ".txt"):
                shutil.copy2(entry, dst / entry.name)
            else:
                skipped.append(entry.name)
    return skipped


def update_info_json(
    dst: Path,
    rotations: Dict[str, List[int]],
    camera_keys: Dict[str, str],
) -> None:
    """Record the rotation in ``meta/info.json`` and correct the codec fields.

    Rewritten tracks come out as h264 regardless of the source codec, so the
    declared codec has to follow or downstream decoders are told the wrong
    thing. ``reorient_180_episodes`` is new provenance: nothing else in the
    dataset would otherwise show that these frames were rotated.
    """
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features") or {}
    for camera, episodes in rotations.items():
        key = camera_keys[camera]
        feature = features.get(key)
        if not isinstance(feature, dict):
            continue
        block = feature.setdefault("info", {})
        block["video.codec"] = "h264"
        block["video.pix_fmt"] = "yuv420p"
        block["reorient_180_episodes"] = list(episodes)
    info_path.write_text(json.dumps(info, indent=4))


def write_manifest(dst: Path, payload: dict) -> None:
    (dst / "reorient_manifest.json").write_text(json.dumps(payload, indent=2))


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate chosen cameras 180 degrees for chosen episodes of a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="Source dataset root (read-only; must contain meta/info.json).")
    parser.add_argument("--dst", type=Path, default=None,
                        help="Destination dataset root. Required unless --report.")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME",
                        help="Camera to rotate, e.g. left_wrist_top. Repeatable.")
    parser.add_argument("--episodes", type=str, default=None,
                        help="'all', or a comma list with ranges such as 3,7,10-14.")
    parser.add_argument("--report", action="store_true",
                        help="Read-only: write first-frame contact sheets and print the "
                             "advisory boundary table. Makes no dataset changes.")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Where --report writes its PNGs (default: ./reorient_report).")
    parser.add_argument("--report-columns", type=int, default=8,
                        help="Contact sheet columns (default: 8).")
    parser.add_argument("--crf", type=int, default=12,
                        help="x264 quality for rewritten videos; lower is better "
                             "(default: 12, visually transparent).")
    parser.add_argument("--preset", type=str, default="medium",
                        help="x264 preset for rewritten videos (default: medium).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be rotated and copied, then stop.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace a non-empty --dst.")
    return parser.parse_args(argv)


def run_report(meta: DatasetMeta, args: argparse.Namespace) -> int:
    cameras = resolve_cameras(meta, args.camera) if args.camera else sorted(meta.camera_keys)
    report_dir = args.report_dir or Path("reorient_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    episodes = meta.episode_ids

    print(f"Report for {meta.root}")
    print(f"  {len(episodes)} episodes, cameras: {sorted(meta.camera_keys)}")
    for camera in cameras:
        print(f"\n[{camera}] decoding episode edges for {len(episodes)} episodes...")
        grouped, decoded = scan_camera_edges(meta, camera, episodes)
        sheet = contact_sheet(grouped, decoded, args.report_columns)
        if sheet is None:
            print("  no frames decoded; skipping")
            continue
        out = report_dir / f"first_frames_{camera}.png"
        cv2.imwrite(str(out), sheet)
        print(f"  wrote {out}")
        boundary_report(camera, grouped, decoded)

    print(
        "\nLook at the contact sheets, note the upside-down episodes, then run:\n"
        f"  lerobot-ros-reorient --src {meta.root} --dst <out> "
        "--camera <name> --episodes <list>"
    )
    return 0


def run_rewrite(meta: DatasetMeta, args: argparse.Namespace) -> int:
    if not args.camera:
        raise SystemExit("--camera is required when rewriting (or use --report).")
    if not args.episodes:
        raise SystemExit("--episodes is required when rewriting (or use --report).")
    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("--src and --dst must be different paths")

    cameras = resolve_cameras(meta, args.camera)
    episodes = parse_episode_spec(args.episodes, meta.episode_ids)

    # Work out every file that needs rewriting before touching the destination,
    # so a bad selection fails before anything has been written.
    plan: Dict[str, Dict[Tuple[int, int], List[FrameRange]]] = {}
    for camera in cameras:
        ranges = episode_frame_ranges(meta, meta.camera_keys[camera], episodes)
        plan[camera] = group_by_file(ranges)

    print(f"Reorienting {meta.root} -> {args.dst}")
    print(f"  cameras: {cameras}")
    print(f"  episodes ({len(episodes)}): {_summarize_ids(episodes)}")
    total_files = 0
    for camera, grouped in plan.items():
        key = meta.camera_keys[camera]
        all_mp4s = sorted((meta.root / "videos" / key).rglob("*.mp4"))
        print(f"  [{camera}] {len(grouped)} of {len(all_mp4s)} video file(s) re-encoded")
        for (chunk, file_index), items in sorted(grouped.items()):
            spans = ", ".join(f"ep{r.episode}:[{r.start},{r.end})" for r in items)
            print(f"    chunk-{chunk:03d}/file-{file_index:03d}: {spans}")
            total_files += 1

    if args.dry_run:
        print(f"\nDry run: {total_files} file(s) would be re-encoded. Nothing written.")
        return 0

    prepare_destination(args.dst, args.overwrite)
    skipped = copy_passthrough(meta.root, args.dst)
    print(f"\n  copied meta/ and data/ to {args.dst}")
    if skipped:
        print(f"  [note] not copied (regenerable or derived): {', '.join(skipped)}")

    for camera_key in sorted(meta.camera_keys.values()):
        src_dir = meta.root / "videos" / camera_key
        if not src_dir.is_dir():
            continue
        camera = camera_short_name(camera_key)
        grouped = plan.get(camera, {})
        rewritten = {
            video_file_path(meta, camera_key, chunk, file_index).resolve(): items
            for (chunk, file_index), items in grouped.items()
        }
        for mp4 in sorted(src_dir.rglob("*.mp4")):
            out = args.dst / mp4.relative_to(meta.root)
            items = rewritten.get(mp4.resolve())
            if items is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, out)
                continue
            print(f"  [ffmpeg] rotate {mp4.relative_to(meta.root)}")
            rotate_video_file(mp4, out, items, args.crf, args.preset)

    rotations = {camera: episodes for camera in cameras}
    update_info_json(args.dst, rotations, meta.camera_keys)
    write_manifest(
        args.dst,
        {
            "source": str(meta.root),
            "rotation_degrees": 180,
            "cameras": cameras,
            "episodes": episodes,
            "encoder": {"codec": "libx264", "crf": int(args.crf), "preset": args.preset},
        },
    )
    print(f"\nDone: {total_files} file(s) re-encoded. Wrote {args.dst}")
    print("  verify with: lerobot-ros-reorient --src "
          f"{args.dst} --report --report-dir /tmp/orient_after")
    return 0


def _summarize_ids(ids: Sequence[int]) -> str:
    """Collapse a sorted id list into compact ranges for logging."""
    if not ids:
        return "(none)"
    parts: List[str] = []
    start = prev = ids[0]
    for value in ids[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.src.is_dir():
        raise SystemExit(f"--src {args.src} is not a directory")
    meta = load_dataset_meta(args.src)

    if args.report:
        return run_report(meta, args)
    if args.dst is None:
        raise SystemExit("--dst is required unless --report is passed")
    return run_rewrite(meta, args)


if __name__ == "__main__":
    sys.exit(main())

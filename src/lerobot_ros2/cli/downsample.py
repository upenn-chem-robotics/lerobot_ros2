#!/usr/bin/env python3
"""Rename image keys and downsample videos in a LeRobot v3.0 dataset.

Hard-coded mapping (per request):
    observation.images.camera_00_back            -> observation.images.back
    observation.images.camera_01_front           -> observation.images.front
    observation.images.camera_02_left_wrist_top  -> observation.images.left_wrist_top
    observation.images.camera_03_perspective     -> observation.images.perspective
    observation.images.camera_04_right_wrist_top -> observation.images.right_wrist_top

Videos are re-encoded at 1/DOWNSAMPLE resolution (default 5x: 720x1280 -> 144x256).

Optionally drop the first N frames from every episode when rewriting the
trajectory and episode metadata. When trimming, per-row timestamps are
rebased to start at 0 for each episode so they stay aligned with the
shifted video metadata.

Usage:
    lerobot-ros-downsample --src /path/to/src_dataset --dst /path/to/dst_dataset [--skip-first-frames N] [--delete-episodes E1 E2 ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEY_MAP = {
    "observation.images.camera_00_back": "observation.images.back",
    "observation.images.camera_01_front": "observation.images.front",
    "observation.images.camera_02_left_wrist_top": "observation.images.left_wrist_top",
    "observation.images.camera_03_perspective": "observation.images.perspective",
    "observation.images.camera_04_right_wrist_top": "observation.images.right_wrist_top",
}


def rename_in_string(s: str) -> str:
    for old, new in KEY_MAP.items():
        if old in s:
            s = s.replace(old, new)
    return s


def renamed_schema(schema: pa.Schema) -> pa.Schema:
    fields = [
        pa.field(
            rename_in_string(field.name),
            field.type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        for field in schema
    ]
    return pa.schema(fields, metadata=schema.metadata)


def load_data_index(src: Path) -> tuple[pd.DataFrame, dict[str, pa.Table]]:
    src_dir = src / "data"
    tables: dict[str, pa.Table] = {}
    frames: list[pd.DataFrame] = []

    for p in sorted(src_dir.rglob("*.parquet")):
        rel = str(p.relative_to(src))
        table = pq.read_table(p)
        tables[rel] = table

        df = table.select(["episode_index", "index"]).to_pandas()
        df["__source_relpath"] = rel
        df["__source_row"] = range(table.num_rows)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["episode_index", "index", "__source_relpath", "__source_row"]), tables

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("index").reset_index(drop=True), tables


def trim_data_index(data_index: pd.DataFrame, skip_first_frames: int) -> pd.DataFrame:
    if skip_first_frames <= 0:
        trimmed = data_index.copy()
    else:
        episode_positions = data_index.groupby("episode_index", sort=False).cumcount()
        trimmed = data_index.loc[episode_positions >= skip_first_frames].copy()

    if trimmed.empty:
        raise SystemExit("Skipping that many frames would remove the entire dataset")

    missing_episodes = set(data_index["episode_index"].unique()) - set(trimmed["episode_index"].unique())
    if missing_episodes:
        episodes = ", ".join(str(ep) for ep in sorted(int(ep) for ep in missing_episodes))
        raise SystemExit(
            f"Skipping {skip_first_frames} frame(s) would empty episode(s): {episodes}"
        )

    trimmed["frame_index"] = trimmed.groupby("episode_index", sort=False).cumcount()
    trimmed["index"] = range(len(trimmed))
    return trimmed


def delete_episodes(data_index: pd.DataFrame, episode_ids: set[int]) -> pd.DataFrame:
    if not episode_ids:
        return data_index.copy()

    available = {int(episode_id) for episode_id in data_index["episode_index"].unique()}
    missing = episode_ids - available
    if missing:
        episodes = ", ".join(str(ep) for ep in sorted(missing))
        raise SystemExit(f"Requested episode(s) do not exist in the source dataset: {episodes}")

    filtered = data_index.loc[~data_index["episode_index"].isin(episode_ids)].copy()
    if filtered.empty:
        raise SystemExit("Deleting those episode(s) would remove the entire dataset")
    return filtered


def remap_episode_indices(data_index: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    episode_ids = sorted(int(episode_id) for episode_id in data_index["episode_index"].unique())
    mapping = {episode_id: new_episode_id for new_episode_id, episode_id in enumerate(episode_ids)}
    remapped = data_index.copy()
    remapped["episode_index"] = remapped["episode_index"].map(mapping)
    return remapped, mapping


def summarize_trimmed_data(trimmed: pd.DataFrame) -> dict[int, dict[str, int]]:
    summary: dict[int, dict[str, int]] = {}
    for episode_id, group in trimmed.groupby("episode_index", sort=False):
        start_index = int(group["index"].min())
        end_index = int(group["index"].max()) + 1
        summary[int(episode_id)] = {
            "length": int(len(group)),
            "dataset_from_index": start_index,
            "dataset_to_index": end_index,
        }
    return summary


def write_trimmed_data(
    src: Path,
    dst: Path,
    trimmed: pd.DataFrame,
    source_tables: dict[str, pa.Table],
    skip_first_frames: int = 0,
    fps: int | None = None,
) -> None:
    frame_offset = 0.0
    if skip_first_frames > 0:
        if fps is None:
            raise ValueError("fps is required when skip_first_frames is set")
        frame_offset = skip_first_frames / float(fps)

    for rel, table in source_tables.items():
        group = trimmed.loc[trimmed["__source_relpath"] == rel].sort_values("__source_row")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        if group.empty:
            pq.write_table(table.slice(0, 0), out)
            continue

        source_rows = pa.array(group["__source_row"].tolist(), type=pa.int64())
        selected = table.take(source_rows)
        if "episode_index" in selected.schema.names:
            selected = selected.set_column(
                selected.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array(
                    group["episode_index"].tolist(),
                    type=selected.schema.field("episode_index").type,
                ),
            )
        if skip_first_frames > 0 and "timestamp" in selected.schema.names:
            timestamps = selected.column("timestamp").to_pylist()
            selected = selected.set_column(
                selected.schema.get_field_index("timestamp"),
                "timestamp",
                pa.array(
                    [float(ts) - frame_offset for ts in timestamps],
                    type=selected.schema.field("timestamp").type,
                ),
            )
        frame_index_field = selected.schema.get_field_index("frame_index")
        index_field = selected.schema.get_field_index("index")
        selected = selected.set_column(
            frame_index_field,
            "frame_index",
            pa.array(group["frame_index"].tolist(), type=selected.schema.field("frame_index").type),
        )
        selected = selected.set_column(
            index_field,
            "index",
            pa.array(group["index"].tolist(), type=selected.schema.field("index").type),
        )
        pq.write_table(selected, out)


def rewrite_info_json(src: Path, dst: Path, downsample: int, total_frames: int | None = None) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    new_features = {}
    for key, ft in info["features"].items():
        new_key = KEY_MAP.get(key, key)
        if ft.get("dtype") == "video":
            ft = json.loads(json.dumps(ft))  # deep copy
            h, w, c = ft["shape"]
            new_h, new_w = h // downsample, w // downsample
            ft["shape"] = [new_h, new_w, c]
            if "info" in ft:
                ft["info"]["video.height"] = new_h
                ft["info"]["video.width"] = new_w
                ft["info"]["video.codec"] = "h264"
                ft["info"]["video.pix_fmt"] = "yuv420p"
        new_features[new_key] = ft
    info["features"] = new_features
    if total_frames is not None:
        info["total_frames"] = total_frames
    out = dst / "meta" / "info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=4))


def rewrite_info_json(
    src: Path,
    dst: Path,
    downsample: int,
    total_frames: int | None = None,
    total_episodes: int | None = None,
) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    new_features = {}
    for key, ft in info["features"].items():
        new_key = KEY_MAP.get(key, key)
        if ft.get("dtype") == "video":
            ft = json.loads(json.dumps(ft))  # deep copy
            h, w, c = ft["shape"]
            new_h, new_w = h // downsample, w // downsample
            ft["shape"] = [new_h, new_w, c]
            if "info" in ft:
                ft["info"]["video.height"] = new_h
                ft["info"]["video.width"] = new_w
                ft["info"]["video.codec"] = "h264"
                ft["info"]["video.pix_fmt"] = "yuv420p"
        new_features[new_key] = ft
    info["features"] = new_features
    if total_frames is not None:
        info["total_frames"] = total_frames
    if total_episodes is not None:
        info["total_episodes"] = total_episodes
        splits = info.get("splits")
        if isinstance(splits, dict):
            new_splits = {}
            for split_name, split_range in splits.items():
                if isinstance(split_range, str) and ":" in split_range:
                    start_str, end_str = split_range.split(":", 1)
                    if start_str.isdigit() and end_str.isdigit() and int(end_str) > total_episodes:
                        new_splits[split_name] = f"{start_str}:{total_episodes}"
                    else:
                        new_splits[split_name] = split_range
                else:
                    new_splits[split_name] = split_range
            info["splits"] = new_splits
    out = dst / "meta" / "info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=4))


def _replace_count_leaves(value: object, count: int) -> object:
    if isinstance(value, list):
        return [_replace_count_leaves(item, count) for item in value]
    if isinstance(value, (int, float)):
        return count
    return value


def rewrite_stats_json(src: Path, dst: Path, total_frames: int | None = None) -> None:
    p = src / "meta" / "stats.json"
    if not p.exists():
        return
    stats = json.loads(p.read_text())
    new_stats = {KEY_MAP.get(k, k): v for k, v in stats.items()}
    if total_frames is not None:
        for key, value in list(new_stats.items()):
            if key.endswith("/count"):
                new_stats[key] = _replace_count_leaves(value, total_frames)
    (dst / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=4))


def rewrite_episodes_parquet(
    src: Path,
    dst: Path,
    episode_summary: dict[int, dict[str, int]] | None = None,
    episode_id_map: dict[int, int] | None = None,
    deleted_episodes: set[int] | None = None,
    skip_first_frames: int = 0,
    fps: int | None = None,
) -> None:
    src_dir = src / "meta" / "episodes"
    if not src_dir.exists():
        return
    for p in src_dir.rglob("*.parquet"):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(p)
        frame = table.to_pandas()
        frame = frame.rename(columns=rename_in_string)

        if deleted_episodes and "episode_index" in frame.columns:
            frame = frame.loc[~frame["episode_index"].isin(deleted_episodes)].copy()

        if episode_id_map and "episode_index" in frame.columns:
            frame["episode_index"] = frame["episode_index"].map(episode_id_map)

        if frame.empty:
            pq.write_table(table.slice(0, 0), out)
            continue

        if episode_summary is not None:
            frame["length"] = frame["episode_index"].map(
                lambda episode_id: episode_summary[int(episode_id)]["length"]
            )
            frame["dataset_from_index"] = frame["episode_index"].map(
                lambda episode_id: episode_summary[int(episode_id)]["dataset_from_index"]
            )
            frame["dataset_to_index"] = frame["episode_index"].map(
                lambda episode_id: episode_summary[int(episode_id)]["dataset_to_index"]
            )

            if skip_first_frames > 0:
                if fps is None:
                    raise ValueError("fps is required when skip_first_frames is set")
                frame_offset = skip_first_frames / float(fps)
                for new_key in KEY_MAP.values():
                    from_col = f"videos/{new_key}/from_timestamp"
                    to_col = f"videos/{new_key}/to_timestamp"
                    if from_col not in frame.columns or to_col not in frame.columns:
                        continue
                    frame[from_col] = frame[from_col] + frame_offset
                    frame[to_col] = frame[from_col] + frame["length"] / float(fps)

        renamed = renamed_schema(table.schema)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False, schema=renamed), out)


def copy_data_parquets(src: Path, dst: Path) -> None:
    src_dir = src / "data"
    if not src_dir.exists():
        return
    for p in src_dir.rglob("*.parquet"):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def copy_tasks(src: Path, dst: Path) -> None:
    for name in ("tasks.parquet", "subtasks.parquet"):
        p = src / "meta" / name
        if p.exists():
            shutil.copy2(p, dst / "meta" / name)


def downsample_videos(src: Path, dst: Path, downsample: int) -> None:
    src_videos = src / "videos"
    if not src_videos.exists():
        return
    for old_key, new_key in KEY_MAP.items():
        src_key_dir = src_videos / old_key
        if not src_key_dir.exists():
            print(f"[skip] {src_key_dir} not present")
            continue
        for mp4 in src_key_dir.rglob("*.mp4"):
            rel = mp4.relative_to(src_key_dir)
            out = dst / "videos" / new_key / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            # iw/downsample, ih/downsample, force even dims
            vf = (
                f"scale=trunc(iw/{downsample}/2)*2:trunc(ih/{downsample}/2)*2"
            )
            cmd = [
                "/bin/ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-an",
                str(out),
            ]
            print(f"[ffmpeg] {mp4} -> {out}")
            subprocess.run(cmd, check=True)


def copy_extras(src: Path, dst: Path) -> None:
    for name in ("experiment_config.yaml", "recording_config.yaml", "README.md"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def clean_destination(dst: Path) -> None:
    """Remove generated outputs so reruns don't accumulate stale files."""
    managed_dirs = [
        dst / "data",
        dst / "videos",
        dst / "meta" / "episodes",
    ]
    for directory in managed_dirs:
        if directory.exists():
            shutil.rmtree(directory)

    managed_files = [
        dst / "meta" / "info.json",
        dst / "meta" / "stats.json",
        dst / "meta" / "tasks.parquet",
        dst / "meta" / "subtasks.parquet",
    ]
    for file_path in managed_files:
        if file_path.exists():
            file_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Source LeRobot v3.0 dataset root")
    ap.add_argument("--dst", type=Path, required=True, help="Destination dataset root")
    ap.add_argument("--downsample", type=int, default=5, help="Spatial downsample factor (default 5)")
    ap.add_argument(
        "--skip-first-frames",
        "--discard-first-frames",
        dest="skip_first_frames",
        type=int,
        default=0,
        help="Discard this many frames from the start of every episode before rewriting",
    )
    ap.add_argument(
        "--delete-episodes",
        nargs="+",
        type=int,
        default=[],
        help="Remove these episode indices from the rewritten dataset",
    )
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"Not a LeRobot dataset: {src}")
    if src == dst:
        raise SystemExit("--src and --dst must be different paths")
    dst.mkdir(parents=True, exist_ok=True)
    clean_destination(dst)
    source_info = json.loads((src / "meta" / "info.json").read_text())
    source_fps = int(source_info["fps"])

    print(
        f"[info] src={src}\n[info] dst={dst}\n[info] downsample={args.downsample}x\n"
        f"[info] skip_first_frames={args.skip_first_frames}"
    )

    if args.skip_first_frames < 0:
        raise SystemExit("--skip-first-frames must be >= 0")

    delete_episode_ids = {int(episode_id) for episode_id in args.delete_episodes}

    episode_summary: dict[int, dict[str, int]] | None = None
    total_frames: int | None = None
    episode_id_map: dict[int, int] | None = None
    total_episodes: int | None = None

    if args.skip_first_frames > 0 or delete_episode_ids:
        data_index, source_tables = load_data_index(src)
        data_index = delete_episodes(data_index, delete_episode_ids)
        if delete_episode_ids:
            data_index, episode_id_map = remap_episode_indices(data_index)
        trimmed = trim_data_index(data_index, args.skip_first_frames)
        episode_summary = summarize_trimmed_data(trimmed)
        total_frames = len(trimmed)
        total_episodes = len(episode_summary)
        write_trimmed_data(
            src,
            dst,
            trimmed,
            source_tables,
            skip_first_frames=args.skip_first_frames,
            fps=source_fps,
        )
    else:
        copy_data_parquets(src, dst)

    rewrite_info_json(
        src,
        dst,
        args.downsample,
        total_frames=total_frames,
        total_episodes=total_episodes,
    )
    rewrite_stats_json(src, dst, total_frames=total_frames)
    copy_tasks(src, dst)
    rewrite_episodes_parquet(
        src,
        dst,
        episode_summary=episode_summary,
        episode_id_map=episode_id_map,
        deleted_episodes=delete_episode_ids,
        skip_first_frames=args.skip_first_frames,
        fps=source_fps,
    )
    downsample_videos(src, dst, args.downsample)
    copy_extras(src, dst)
    print("[done]")


if __name__ == "__main__":
    main()

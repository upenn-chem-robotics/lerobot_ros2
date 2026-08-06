"""Drop the trailing fraction of frames from every episode of a LeRobot dataset.

Diffusion policies tend to overfit to long stationary "completion" tails (gripper
fully closed, arm fully raised, no further motion). Trimming the last N% of frames
from every episode removes most of that signal while keeping the actual task
trajectory intact, including any in-the-middle DAgger correction frames.

Base demos (pure teleoperation) and DAgger episodes get different default trim
fractions, because DAgger episodes typically have a much longer post-correction
"recover" plateau than base demos do. Classification is purely from
``action_source``:

* ``base``   - every frame in the episode has ``action_source == 1`` (all-human)
* ``dagger`` - at least one frame has ``action_source == 0`` (policy)

The source dataset is read-only; output is written to a new directory.

Stats handling
--------------
* Numeric features (state, action, action_source, timestamp, frame_index, ...)
  are recomputed exactly from the kept rows.
* Image/video features keep their source mean/std/quantiles (these are sampled
  from a frame subset during dataset creation and barely change when 5-20% of
  the trailing frames are removed) but their ``count`` is clamped to the new
  episode length, so dataset-wide aggregation stays consistent.

Example
-------
    lerobot-ros-trim-tail \\
        --src /lerobot-ros/data/septum_white_full_train_20260511_ds5_clean \\
        --dst /lerobot-ros/data/septum_white_full_train_20260511_ds5_clean_trim_5_20 \\
        --base-fraction 0.05 \\
        --dagger-fraction 0.20
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.dataset_tools import _keep_episodes_from_video_with_av
from lerobot.datasets.io_utils import write_info, write_stats

from lerobot_ros2.dataset_rewrite import data_parquets, episode_parquets, read_info


def _recursive_to_array(value):
    """Recursively unwrap nested ndarray/object dtypes to a clean nested list of floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_recursive_to_array(v) for v in value]
    return value


def trim_tail(
    src: Path,
    dst: Path,
    base_fraction: float,
    dagger_fraction: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Drop the trailing tail of frames from every episode in ``src`` and write the trimmed
    dataset to ``dst``.

    Each episode is classified as either ``base`` (every ``action_source == 1``) or
    ``dagger`` (at least one ``action_source == 0`` frame) and trimmed by
    ``base_fraction`` or ``dagger_fraction`` of its length respectively.
    """
    for name, frac in [("base_fraction", base_fraction), ("dagger_fraction", dagger_fraction)]:
        if not (0.0 <= frac < 1.0):
            raise ValueError(f"{name} must be in [0, 1), got {frac}")

    if dst.exists():
        print(f"Destination exists, removing: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    info = read_info(src)
    fps = int(info["fps"])
    features = info["features"]
    video_path_fmt = info["video_path"]
    video_feats = [k for k, v in features.items() if v["dtype"] in ("video", "image")]
    numeric_feats = [
        k for k, v in features.items() if v["dtype"] not in ("video", "image", "string")
    ]
    print(f"video features:   {video_feats}")
    print(f"numeric features: {numeric_feats}")

    src_ep_path = episode_parquets(src)[0]
    src_ep_tbl = pq.read_table(src_ep_path)
    src_ep_df = src_ep_tbl.to_pandas().sort_values("episode_index").reset_index(drop=True)
    src_ep_schema = src_ep_tbl.schema

    src_data_path = data_parquets(src)[0]
    src_data_tbl = pq.read_table(src_data_path)
    src_data_df = (
        src_data_tbl.to_pandas()
        .sort_values(["episode_index", "frame_index"])
        .reset_index(drop=True)
    )
    src_data_schema = src_data_tbl.schema

    if "action_source" not in src_data_df.columns:
        raise RuntimeError(
            "Dataset has no 'action_source' column, so base/dagger classification is not "
            "possible. Use a uniform trim or add the column with lerobot-ros-add-action-source."
        )
    has_policy_frame = (
        src_data_df.assign(_is_policy=src_data_df["action_source"] == 0)
        .groupby("episode_index")["_is_policy"]
        .any()
    )
    is_dagger_by_ep = has_policy_frame.to_dict()

    def _drop_n(length: int, ep_idx: int) -> int:
        frac = dagger_fraction if is_dagger_by_ep.get(int(ep_idx), False) else base_fraction
        return int(round(length * frac))

    src_ep_df["is_dagger"] = src_ep_df["episode_index"].map(
        lambda i: bool(is_dagger_by_ep.get(int(i), False))
    )
    src_ep_df["drop_n"] = src_ep_df.apply(
        lambda r: _drop_n(int(r["length"]), int(r["episode_index"])), axis=1
    )
    src_ep_df["new_length"] = src_ep_df["length"] - src_ep_df["drop_n"]
    new_lengths = dict(
        zip(src_ep_df["episode_index"].astype(int), src_ep_df["new_length"].astype(int))
    )
    total_new = int(src_ep_df["new_length"].sum())

    n_base = int((~src_ep_df["is_dagger"]).sum())
    n_dagger = int(src_ep_df["is_dagger"].sum())
    base_rows = src_ep_df[~src_ep_df["is_dagger"]]
    dagger_rows = src_ep_df[src_ep_df["is_dagger"]]
    print(
        f"classification: {n_base} base episodes (trim {base_fraction:.0%}), "
        f"{n_dagger} dagger episodes (trim {dagger_fraction:.0%})"
    )
    if len(base_rows):
        print(
            f"  base:   drop {base_rows['drop_n'].mean():.1f} avg, "
            f"keep {base_rows['new_length'].mean():.1f} avg per ep"
        )
    if len(dagger_rows):
        print(
            f"  dagger: drop {dagger_rows['drop_n'].mean():.1f} avg, "
            f"keep {dagger_rows['new_length'].mean():.1f} avg per ep"
        )
    print(f"total frames: {info['total_frames']} -> {total_new}")

    keep_global_idx = []
    for ep_idx in sorted(new_lengths.keys()):
        new_len = new_lengths[ep_idx]
        ep_rows = src_data_df.index[src_data_df["episode_index"] == ep_idx].to_list()
        keep_global_idx.extend(ep_rows[:new_len])
    new_data_df = src_data_df.loc[keep_global_idx].reset_index(drop=True)
    new_data_df["index"] = np.arange(len(new_data_df), dtype=np.int64)
    assert len(new_data_df) == total_new, (len(new_data_df), total_new)

    (dst / "data" / "chunk-000").mkdir(parents=True)
    new_data_tbl = pa.Table.from_pandas(
        new_data_df, schema=src_data_schema, preserve_index=False
    )
    pq.write_table(new_data_tbl, dst / "data" / "chunk-000" / "file-000.parquet")
    print(f"wrote data parquet: {len(new_data_df)} rows")

    for vk in video_feats:
        chunk_col = f"videos/{vk}/chunk_index"
        file_col = f"videos/{vk}/file_index"
        for (chunk_idx, file_idx), eps in src_ep_df.groupby([chunk_col, file_col]):
            chunk_idx, file_idx = int(chunk_idx), int(file_idx)
            src_video = src / video_path_fmt.format(
                video_key=vk, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_video = dst / video_path_fmt.format(
                video_key=vk, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            eps_sorted = eps.sort_values(f"videos/{vk}/from_timestamp")
            ranges = []
            for _, ep in eps_sorted.iterrows():
                from_frame = round(ep[f"videos/{vk}/from_timestamp"] * fps)
                ranges.append((from_frame, from_frame + int(ep["new_length"])))
            print(f"  re-encode {vk} chunk={chunk_idx} file={file_idx}: {len(ranges)} eps")
            _keep_episodes_from_video_with_av(
                src_video, dst_video, ranges, fps=fps, vcodec=vcodec, pix_fmt=pix_fmt
            )

    new_video_ts = {vk: {} for vk in video_feats}
    for vk in video_feats:
        chunk_col = f"videos/{vk}/chunk_index"
        file_col = f"videos/{vk}/file_index"
        for (_chunk_idx, _file_idx), eps in src_ep_df.groupby([chunk_col, file_col]):
            cum = 0.0
            for _, ep in eps.sort_values(f"videos/{vk}/from_timestamp").iterrows():
                ep_idx = int(ep["episode_index"])
                new_len = int(ep["new_length"])
                new_video_ts[vk][ep_idx] = (cum, cum + new_len / fps)
                cum += new_len / fps

    print("computing per-episode numeric stats...")
    per_ep_numeric_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for ep_idx in sorted(new_lengths.keys()):
        ep_data = new_data_df[new_data_df["episode_index"] == ep_idx]
        feat_stats: dict[str, dict[str, np.ndarray]] = {}
        for feat in numeric_feats:
            arr = np.asarray(ep_data[feat].to_list())
            keepdims = arr.ndim == 1
            feat_stats[feat] = get_feature_stats(arr, axis=0, keepdims=keepdims)
        per_ep_numeric_stats[ep_idx] = feat_stats

    print("building new episodes parquet rows...")
    src_image_stats_cache: dict[int, dict[str, dict[str, list]]] = {}
    src_ep_rows_by_idx = {int(r["episode_index"]): r for _, r in src_ep_df.iterrows()}
    for ep_idx, row in src_ep_rows_by_idx.items():
        cache: dict[str, dict[str, list]] = {}
        for vk in video_feats:
            cache[vk] = {}
            for stat_name in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
                col = f"stats/{vk}/{stat_name}"
                if col in row:
                    cache[vk][stat_name] = row[col]
        src_image_stats_cache[ep_idx] = cache

    new_columns: dict[str, list] = {name: [] for name in src_ep_schema.names}
    cum_dataset_idx = 0
    for ep_idx in sorted(new_lengths.keys()):
        row = src_ep_rows_by_idx[ep_idx]
        new_len = int(new_lengths[ep_idx])
        from_idx = cum_dataset_idx
        to_idx = cum_dataset_idx + new_len
        cum_dataset_idx = to_idx

        for col_name in src_ep_schema.names:
            if col_name == "length":
                value = new_len
            elif col_name == "dataset_from_index":
                value = from_idx
            elif col_name == "dataset_to_index":
                value = to_idx
            elif col_name.startswith("videos/") and col_name.endswith("/from_timestamp"):
                vk = col_name[len("videos/") : -len("/from_timestamp")]
                value = float(new_video_ts[vk][ep_idx][0])
            elif col_name.startswith("videos/") and col_name.endswith("/to_timestamp"):
                vk = col_name[len("videos/") : -len("/to_timestamp")]
                value = float(new_video_ts[vk][ep_idx][1])
            elif col_name.startswith("stats/"):
                parts = col_name[len("stats/") :].split("/")
                feat_name = "/".join(parts[:-1])
                stat_name = parts[-1]
                if feat_name in numeric_feats:
                    arr = per_ep_numeric_stats[ep_idx][feat_name][stat_name]
                    value = arr.tolist() if hasattr(arr, "tolist") else list(arr)
                elif feat_name in video_feats:
                    base_value = src_image_stats_cache[ep_idx][feat_name].get(stat_name)
                    if stat_name == "count":
                        orig_count = (
                            base_value[0]
                            if isinstance(base_value, (list, tuple, np.ndarray))
                            else base_value
                        )
                        value = [int(min(int(orig_count), new_len))]
                    else:
                        if hasattr(base_value, "tolist"):
                            value = base_value.tolist()
                        elif isinstance(base_value, (list, tuple, np.ndarray)):
                            value = list(base_value)
                        else:
                            value = base_value
                else:
                    value = row[col_name]
                    if hasattr(value, "tolist"):
                        value = value.tolist()
            else:
                value = row[col_name]
                if hasattr(value, "tolist"):
                    value = value.tolist()
            new_columns[col_name].append(value)

    new_ep_arrays = {}
    for col_name in src_ep_schema.names:
        field_type = src_ep_schema.field(col_name).type
        try:
            new_ep_arrays[col_name] = pa.array(new_columns[col_name], type=field_type)
        except Exception as e:
            print(f"  failed building column {col_name}: {e}", file=sys.stderr)
            print(f"  sample values: {new_columns[col_name][:3]}", file=sys.stderr)
            raise
    new_ep_tbl = pa.Table.from_pydict(new_ep_arrays, schema=src_ep_schema)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        new_ep_tbl,
        dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        compression="snappy",
    )
    print(f"wrote episodes parquet: {len(new_columns['episode_index'])} rows")

    shutil.copy(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    new_info = read_info(src)
    new_info["total_frames"] = total_new
    new_info["splits"] = {"train": f"0:{new_info['total_episodes']}"}
    write_info(new_info, dst)

    print("aggregating dataset-wide stats...")
    all_ep_stats = []
    for ep_idx in sorted(new_lengths.keys()):
        ep_stats: dict[str, dict[str, np.ndarray]] = {}
        for feat, st in per_ep_numeric_stats[ep_idx].items():
            ep_stats[feat] = {k: np.asarray(v) for k, v in st.items()}
        for vk in video_feats:
            ep_stats[vk] = {}
            for stat_name in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
                base = src_image_stats_cache[ep_idx][vk].get(stat_name)
                if base is None:
                    continue
                if stat_name == "count":
                    orig_count = (
                        base[0]
                        if isinstance(base, (list, tuple, np.ndarray))
                        else base
                    )
                    ep_stats[vk][stat_name] = np.asarray(
                        [int(min(int(orig_count), new_lengths[ep_idx]))]
                    )
                else:
                    ep_stats[vk][stat_name] = np.asarray(
                        _recursive_to_array(base), dtype=np.float64
                    )
        all_ep_stats.append(ep_stats)
    aggregated = aggregate_stats(all_ep_stats)
    filtered = {k: v for k, v in aggregated.items() if k in features}
    write_stats(filtered, dst)
    print(f"wrote meta/stats.json with {len(filtered)} features")

    print(f"\nDONE\noutput: {dst}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drop the trailing fraction of frames from every episode of a LeRobot dataset, "
            "using separate fractions for base (all-human) and dagger (any-policy) episodes."
        )
    )
    parser.add_argument("--src", required=True, type=Path, help="Source dataset root directory.")
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination dataset root directory (will be wiped if it exists).",
    )
    parser.add_argument(
        "--base-fraction",
        type=float,
        default=0.05,
        help="Fraction of the tail to drop from base/teleoperation episodes (default: 0.05).",
    )
    parser.add_argument(
        "--dagger-fraction",
        type=float,
        default=0.20,
        help="Fraction of the tail to drop from DAgger episodes (default: 0.20).",
    )
    parser.add_argument(
        "--vcodec", default="libsvtav1", help="Video codec for re-encoding (default: libsvtav1)."
    )
    parser.add_argument(
        "--pix-fmt",
        dest="pix_fmt",
        default="yuv420p",
        help="Pixel format for re-encoding (default: yuv420p).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trim_tail(
        src=args.src,
        dst=args.dst,
        base_fraction=args.base_fraction,
        dagger_fraction=args.dagger_fraction,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

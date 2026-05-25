#!/usr/bin/env python3
"""Preview the image-transform pipeline that training will see.

Builds a ``LeRobotDataset`` exactly the way ``lerobot.datasets.factory.make_dataset``
would (same ``ImageTransformsConfig`` -> ``ImageTransforms`` chain, same
``video_backend``), pulls one or more random batches, and writes a tiled PNG
per (pass, camera) so you can eyeball what the policy is actually trained on.

Usage:
    lerobot-ros-preview-aug --config /path/to/dp64_cfg.json
    lerobot-ros-preview-aug --config /path/to/dp64_cfg.json --batch-size 16 --passes 4
    lerobot-ros-preview-aug --config /path/to/dp64_cfg.json --output-dir /tmp/aug

There is also a ``--first-frames`` mode that, with augmentation forced off,
fetches frame 0 of every episode at the dataset's training resolution and
tiles them into a single PNG per camera. Use this to verify that a small or
transparent object (e.g. a clear septum or stir bar) is actually visible to
the model at training resolution before debating augmentation tweaks. If the
object is invisible to your eye in those tiles, the model can't see it
either; fix data first (camera framing, lighting, contrast mat, or higher
training resolution) before tuning the augmentation recipe.

    lerobot-ros-preview-aug --config /path/to/dp64_cfg.json --first-frames
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple





import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)

import cv2

from lerobot_ros2.visualizer import tile_frames_grid


def _build_image_transforms_config(raw: Dict[str, Any]) -> ImageTransformsConfig:
    """Convert the dict from ``dp64_cfg.json`` into a typed ``ImageTransformsConfig``."""
    allowed = {f.name for f in fields(ImageTransformsConfig)}
    kwargs: Dict[str, Any] = {k: v for k, v in raw.items() if k in allowed}

    raw_tfs = raw.get("tfs") or {}
    tfs: Dict[str, ImageTransformConfig] = {}
    for name, entry in raw_tfs.items():
        if not isinstance(entry, dict):
            continue
        tfs[name] = ImageTransformConfig(
            weight=float(entry.get("weight", 1.0)),
            type=str(entry.get("type", "Identity")),
            kwargs=dict(entry.get("kwargs", {})),
        )
    if tfs:
        kwargs["tfs"] = tfs

    return ImageTransformsConfig(**kwargs)


def _isolate_single_transform(
    raw_image_transforms: Dict[str, Any], active_name: str
) -> Dict[str, Any]:
    """Return a copy of ``raw_image_transforms`` where only ``active_name`` is enabled.

    All other transforms have their ``weight`` zeroed (which causes
    ``ImageTransforms`` to skip them), and ``max_num_transforms`` is forced to 1
    so the surviving transform is applied with probability 1.
    """
    raw_tfs = raw_image_transforms.get("tfs") or {}
    new_tfs: Dict[str, Any] = {}
    for name, entry in raw_tfs.items():
        if not isinstance(entry, dict):
            continue
        new_entry = {
            "weight": float(entry.get("weight", 1.0)) if name == active_name else 0.0,
            "type": entry.get("type", "Identity"),
            "kwargs": dict(entry.get("kwargs", {})),
        }
        new_tfs[name] = new_entry
    return {
        **raw_image_transforms,
        "tfs": new_tfs,
        "max_num_transforms": 1,
        "enable": True,
    }


def _tensor_to_bgr_grid(
    images: torch.Tensor,
    columns: int,
    cell_w: int,
    cell_h: int,
    label_template: str,
) -> np.ndarray:
    """Tile ``images`` (N, C, H, W in [0, 1]) into a single labelled BGR canvas."""
    if images.ndim == 5:
        images = images[:, -1]
    if images.ndim != 4:
        raise ValueError(f"Expected (N, C, H, W) tensor, got shape {tuple(images.shape)}")

    rgb_uint8 = (images.clamp(0.0, 1.0) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    bgr_frames: List[np.ndarray] = [frame[..., ::-1].copy() for frame in rgb_uint8]
    labels = [label_template.format(i=i) for i in range(len(bgr_frames))]
    rows = max(1, int(np.ceil(len(bgr_frames) / max(1, columns))))
    return tile_frames_grid(
        bgr_frames,
        labels,
        cell_w * columns,
        cell_h * rows,
        columns,
    )


def _episode_first_indices(dataset: LeRobotDataset) -> List[Tuple[int, int]]:
    """Return ``[(episode_index, dataset_row_index), ...]`` sorted by episode_index.

    ``dataset.hf_dataset`` is the parquet-backed metadata table that
    ``LeRobotDataset.__getitem__`` is indexed against. The row whose
    ``frame_index == 0`` is the first frame of an episode, and the pandas row
    index is exactly the integer index ``__getitem__`` expects. This works
    after ``downsample`` deletes/renumbers episodes, because we read the
    rebuilt destination metadata.
    """
    hf = dataset.hf_dataset
    cols = hf.select_columns(["episode_index", "frame_index"]).to_pandas()
    firsts = cols[cols["frame_index"] == 0].copy()
    firsts["__row__"] = firsts.index.astype(int)
    firsts = firsts.sort_values("episode_index")
    return list(
        zip(
            firsts["episode_index"].astype(int).tolist(),
            firsts["__row__"].astype(int).tolist(),
        )
    )


def _render_first_frames(
    dataset: LeRobotDataset,
    image_keys: List[str],
    output_dir: Path,
    columns: int,
) -> None:
    """Render frame-0 of every episode for every camera into one PNG per camera."""
    pairs = _episode_first_indices(dataset)
    if not pairs:
        sys.exit("No frame_index==0 rows found; dataset has no episode boundaries?")

    if columns <= 4 and len(pairs) > 16:
        columns = int(np.ceil(np.sqrt(len(pairs))))

    per_cam: Dict[str, List[np.ndarray]] = {k: [] for k in image_keys}
    labels: List[str] = []
    for ep_idx, ds_idx in pairs:
        sample = dataset[ds_idx]
        for k in image_keys:
            tensor = sample.get(k)
            if tensor is None:
                continue
            if tensor.ndim == 4:
                tensor = tensor[-1]
            rgb = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            per_cam[k].append(rgb[..., ::-1].copy())
        labels.append(f"ep {ep_idx}")

    for k, frames in per_cam.items():
        if not frames:
            logging.warning("Camera key %s yielded no frames; skipping.", k)
            continue
        h, w = frames[0].shape[:2]
        rows = int(np.ceil(len(frames) / columns))
        canvas = tile_frames_grid(
            frames,
            labels,
            w * columns,
            h * rows,
            columns,
        )
        safe_key = k.replace(".", "_")
        out = output_dir / f"first_frames_{safe_key}.png"
        cv2.imwrite(str(out), canvas)
        logging.info(
            "wrote %s (%d episodes, %dx%d grid)",
            out,
            len(frames),
            columns,
            rows,
        )


def _sample_indices(n_dataset: int, k: int, seed: int) -> List[int]:
    """Pick ``k`` distinct dataset indices reproducibly from torch's RNG."""
    if k >= n_dataset:
        return list(range(n_dataset))
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_dataset, generator=g)[:k].tolist()


def _render_indices_through_dataset(
    dataset: LeRobotDataset,
    indices: List[int],
    image_keys: List[str],
    output_dir: Path,
    columns: int,
    file_label: str,
    grid_label_prefix: str,
) -> None:
    """Pull ``indices`` from ``dataset``, tile per camera, write PNGs.

    ``file_label`` becomes the leading token in the output filename
    (e.g. ``per_transform_color_temp``); ``grid_label_prefix`` is what's drawn
    on every tile (typically the transform name) followed by the tile index.
    """
    per_cam_tensors: Dict[str, List[torch.Tensor]] = {k: [] for k in image_keys}
    for idx in indices:
        sample = dataset[int(idx)]
        for k in image_keys:
            tensor = sample.get(k)
            if tensor is None:
                continue
            if tensor.ndim == 4:
                tensor = tensor[-1]
            per_cam_tensors[k].append(tensor)

    for cam_key, tensors in per_cam_tensors.items():
        if not tensors:
            logging.warning("No tensors collected for %s; skipping.", cam_key)
            continue
        stacked = torch.stack(tensors, dim=0)
        cell_h, cell_w = int(stacked.shape[-2]), int(stacked.shape[-1])
        grid_bgr = _tensor_to_bgr_grid(
            stacked,
            columns,
            cell_w,
            cell_h,
            label_template=f"{grid_label_prefix} #{{i}}",
        )
        safe_key = cam_key.replace(".", "_")
        out = output_dir / f"{file_label}_{safe_key}.png"
        cv2.imwrite(str(out), grid_bgr)
        logging.info("wrote %s", out)


def _run_per_transform_mode(
    *,
    raw_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    image_transforms_raw: Dict[str, Any],
    repo_id: str,
    root: str,
    output_dir: Path,
    batch_size: int,
    columns: int,
    seed: int,
    num_workers: int,
) -> None:
    """Render one PNG per (named transform, camera) with only that transform active.

    Writes a ``per_transform_RAW_<cam>.png`` baseline plus
    ``per_transform_<name>_<cam>.png`` for every entry under
    ``dataset.image_transforms.tfs``. The same dataset indices are used for
    every transform so visual comparison is direct.
    """
    raw_tfs = (image_transforms_raw or {}).get("tfs") or {}
    if not raw_tfs:
        sys.exit(
            "--per-transform requires dataset.image_transforms.tfs to be non-empty."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing per-transform previews to %s", output_dir)

    raw_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=ds_cfg.get("episodes"),
        image_transforms=None,
        revision=ds_cfg.get("revision"),
        video_backend=ds_cfg.get("video_backend"),
        tolerance_s=float(raw_cfg.get("tolerance_s", 1e-4)),
    )
    image_keys = sorted(getattr(raw_dataset.meta, "camera_keys", []))
    if not image_keys:
        sys.exit("Dataset reports no camera keys — nothing to preview.")
    logging.info("Found %d camera key(s): %s", len(image_keys), image_keys)

    indices = _sample_indices(len(raw_dataset), batch_size, seed)
    logging.info(
        "Per-transform mode: %d transform(s) x %d camera(s), batch_size=%d, indices=%s",
        len(raw_tfs),
        len(image_keys),
        len(indices),
        indices[:8] + (["..."] if len(indices) > 8 else []),
    )

    _render_indices_through_dataset(
        dataset=raw_dataset,
        indices=indices,
        image_keys=image_keys,
        output_dir=output_dir,
        columns=columns,
        file_label="per_transform_RAW",
        grid_label_prefix="raw",
    )

    for active_name in raw_tfs:
        isolated_raw = _isolate_single_transform(image_transforms_raw, active_name)
        cfg = _build_image_transforms_config(isolated_raw)
        img_tf = ImageTransforms(cfg)

        ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=ds_cfg.get("episodes"),
            image_transforms=img_tf,
            revision=ds_cfg.get("revision"),
            video_backend=ds_cfg.get("video_backend"),
            tolerance_s=float(raw_cfg.get("tolerance_s", 1e-4)),
        )

        torch.manual_seed(seed)
        np.random.seed(seed)

        _render_indices_through_dataset(
            dataset=ds,
            indices=indices,
            image_keys=image_keys,
            output_dir=output_dir,
            columns=columns,
            file_label=f"per_transform_{active_name}",
            grid_label_prefix=active_name,
        )

    logging.info(
        "Done. per-transform mode: %d transform(s) -> %s",
        len(raw_tfs),
        output_dir,
    )

    if num_workers:
        logging.debug("--num-workers=%d ignored in per-transform mode.", num_workers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render augmented training batches from a dp*_cfg.json so you can "
        "sanity-check the image transform pipeline before training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training JSON config (e.g. dp64_cfg.json) with a dataset.image_transforms block.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Frames per pass (>=1). Each frame is independently augmented. Default: 16.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help="Number of independent random batches to render. Default: 4.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Tile columns per camera grid. Default: 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed for reproducible passes. Default: 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the PNGs. Defaults to <dataset.root>/_aug_preview.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep at 0 for deterministic seeded augmentations.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also dump the same indices with image_transforms disabled, for side-by-side compare.",
    )
    parser.add_argument(
        "--first-frames",
        action="store_true",
        help=(
            "Instead of random augmented batches, dump frame 0 of EVERY episode at "
            "training resolution with image_transforms FORCED OFF, tiled into one "
            "PNG per camera. Use this to verify small / transparent objects are "
            "actually visible to the model at the dataset's training resolution "
            "(e.g. 144x256) before debating augmentation. When set, --passes / "
            "--batch-size / --include-raw are ignored."
        ),
    )
    parser.add_argument(
        "--per-transform",
        action="store_true",
        help=(
            "Render one PNG per (transform, camera) where ONLY that named transform "
            "is active (others have weight=0 and max_num_transforms=1). This makes "
            "it trivial to see what each individual augmentation does — e.g. "
            "color_temp on glassware. Uses the same random indices for every "
            "transform so cross-file comparison is meaningful. Also writes a "
            "per_transform_RAW_<cam>.png baseline. When set, --passes is ignored."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.batch_size < 1:
        sys.exit("--batch-size must be >= 1")
    if args.passes < 1:
        sys.exit("--passes must be >= 1")
    if args.columns < 1:
        sys.exit("--columns must be >= 1")

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        sys.exit(f"Config not found: {cfg_path}")

    raw_cfg = json.loads(cfg_path.read_text())
    ds_cfg = raw_cfg.get("dataset") or {}
    repo_id: Optional[str] = ds_cfg.get("repo_id")
    root: Optional[str] = ds_cfg.get("root")
    if not repo_id or not root:
        sys.exit("Config is missing dataset.repo_id or dataset.root")

    if args.first_frames:
        if args.include_raw:
            logging.warning("--include-raw is ignored when --first-frames is set.")
        if args.passes != 4:
            logging.warning("--passes is ignored when --first-frames is set.")
        if args.batch_size != 16:
            logging.warning("--batch-size is ignored when --first-frames is set.")

        logging.info(
            "Loading no-aug dataset for --first-frames: repo_id=%s root=%s",
            repo_id,
            root,
        )
        no_aug_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=ds_cfg.get("episodes"),
            image_transforms=None,
            revision=ds_cfg.get("revision"),
            video_backend=ds_cfg.get("video_backend"),
            tolerance_s=float(raw_cfg.get("tolerance_s", 1e-4)),
        )
        image_keys = sorted(getattr(no_aug_dataset.meta, "camera_keys", []))
        if not image_keys:
            sys.exit("Dataset reports no camera keys — nothing to preview.")
        logging.info("Found %d camera key(s): %s", len(image_keys), image_keys)

        output_dir = (
            args.output_dir or (Path(root) / "_first_frames_preview")
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Writing first-frame previews to %s", output_dir)

        _render_first_frames(no_aug_dataset, image_keys, output_dir, args.columns)
        logging.info(
            "Done. first-frames mode: %d camera(s) -> %s",
            len(image_keys),
            output_dir,
        )
        return

    image_transforms_raw = ds_cfg.get("image_transforms") or {}
    img_tf_cfg = _build_image_transforms_config(image_transforms_raw)
    if not img_tf_cfg.enable:
        logging.warning(
            "dataset.image_transforms.enable is false — preview will show raw frames "
            "(use --include-raw to confirm there's no transform pipeline)."
        )
    img_tf = ImageTransforms(img_tf_cfg) if img_tf_cfg.enable else None

    if args.per_transform:
        _run_per_transform_mode(
            raw_cfg=raw_cfg,
            ds_cfg=ds_cfg,
            image_transforms_raw=image_transforms_raw,
            repo_id=repo_id,
            root=root,
            output_dir=(args.output_dir or (Path(root) / "_aug_preview")).expanduser().resolve(),
            batch_size=args.batch_size,
            columns=args.columns,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        return

    logging.info("Loading dataset: repo_id=%s root=%s", repo_id, root)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=ds_cfg.get("episodes"),
        image_transforms=img_tf,
        revision=ds_cfg.get("revision"),
        video_backend=ds_cfg.get("video_backend"),
        tolerance_s=float(raw_cfg.get("tolerance_s", 1e-4)),
    )

    image_keys = sorted(getattr(dataset.meta, "camera_keys", []))
    if not image_keys:
        sys.exit("Dataset reports no camera keys — nothing to preview.")
    logging.info("Found %d camera key(s): %s", len(image_keys), image_keys)

    output_dir = (args.output_dir or (Path(root) / "_aug_preview")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing previews to %s", output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    iterator = iter(loader)

    raw_iterator = None
    if args.include_raw:
        raw_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=ds_cfg.get("episodes"),
            image_transforms=None,
            revision=ds_cfg.get("revision"),
            video_backend=ds_cfg.get("video_backend"),
            tolerance_s=float(raw_cfg.get("tolerance_s", 1e-4)),
        )
        raw_iterator = iter(
            DataLoader(
                raw_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=False,
            )
        )

    for pass_idx in range(args.passes):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        raw_batch = None
        if raw_iterator is not None:
            try:
                raw_batch = next(raw_iterator)
            except StopIteration:
                raw_iterator = iter(raw_iterator)  # type: ignore[arg-type]
                raw_batch = next(raw_iterator)

        for cam_key in image_keys:
            if cam_key not in batch:
                logging.warning("Camera key %s missing from batch; skipping.", cam_key)
                continue
            images = batch[cam_key]
            if images.ndim == 5:
                cell_h, cell_w = int(images.shape[-2]), int(images.shape[-1])
            else:
                cell_h, cell_w = int(images.shape[-2]), int(images.shape[-1])
            grid_bgr = _tensor_to_bgr_grid(
                images,
                args.columns,
                cell_w,
                cell_h,
                label_template=cam_key.split(".")[-1] + " #{i}",
            )
            safe_key = cam_key.replace(".", "_")
            out = output_dir / f"pass{pass_idx:02d}_{safe_key}_aug.png"
            cv2.imwrite(str(out), grid_bgr)
            logging.info("wrote %s", out)

            if raw_batch is not None and cam_key in raw_batch:
                raw_grid = _tensor_to_bgr_grid(
                    raw_batch[cam_key],
                    args.columns,
                    cell_w,
                    cell_h,
                    label_template=cam_key.split(".")[-1] + " RAW #{i}",
                )
                raw_out = output_dir / f"pass{pass_idx:02d}_{safe_key}_raw.png"
                cv2.imwrite(str(raw_out), raw_grid)
                logging.info("wrote %s", raw_out)

    logging.info("Done. %d pass(es) x %d camera(s) -> %s", args.passes, len(image_keys), output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the exact training-time view of a single episode as an MP4.

Replays one episode through the **same** image pipeline the diffusion policy
sees during training:

    raw frame in dataset (e.g. 144x256)
      -> dataset.image_transforms (ColorJitter, SharpnessJitter, RandomAffine, ...)
      -> torchvision.RandomCrop(policy.crop_shape) per call
                              (mirrors policy's training-time RandomCrop)
      -> output frame at policy.crop_shape (e.g. 134x246)

Both cameras are placed side-by-side. The MP4 is written at the dataset fps
with an integer nearest-neighbour upscale (default 3x) so the per-pixel
content the encoder sees is preserved while the video remains watchable.

Visual normalization (``VISUAL: MEAN_STD`` with ImageNet stats) is **not**
applied: it would only shift/scale pixel values, not the content the encoder
sees, and would make the MP4 unwatchable.

Usage:
    lerobot-ros-preview-episode \
        --config /path/to/dp64_cfg.json \
        --episode 250 \
        --output /path/to/episode_250_model_view.mp4

Common variants:
    # deterministic deployment view (CenterCrop instead of RandomCrop)
    lerobot-ros-preview-episode --config CFG --episode 250 --output OUT --crop-mode center

    # no augmentation, just the crop window (debug what framing the encoder sees)
    lerobot-ros-preview-episode --config CFG --episode 250 --output OUT --no-aug

    # render every episode in the dataset to OUTPUT_DIR/episode_<idx>_model_view.mp4
    lerobot-ros-preview-episode --config CFG --all-episodes --output-dir /path/to/out_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)

from lerobot_ros2.video_exporter import _open_video_writer


def _build_image_transforms_config(raw: Dict[str, Any]) -> ImageTransformsConfig:
    """Convert the ``dataset.image_transforms`` block from a training cfg into the typed config."""
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


def _annotate(bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    """Draw small labelled lines in the top-left corner of a BGR frame."""
    out = bgr.copy()
    y = 12
    for line in lines:
        cv2.putText(out, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        y += 12
    return out


def _episode_rows(dataset: LeRobotDataset, episode_index: int) -> tuple[list[int], list[int]]:
    """Return ``(dataset_row_indices, frame_indices)`` for the given episode, sorted by frame_index.

    Mirrors the trick used by :mod:`lerobot_ros2.cli.preview_aug._episode_first_indices`:
    the pandas row index of ``hf_dataset`` is exactly the integer index
    ``LeRobotDataset.__getitem__`` expects.
    """
    cols = dataset.hf_dataset.select_columns(["episode_index", "frame_index"]).to_pandas()
    sub = cols[cols["episode_index"] == episode_index].copy()
    if sub.empty:
        return [], []
    sub["__row__"] = sub.index.astype(int)
    sub = sub.sort_values("frame_index")
    return (
        sub["__row__"].astype(int).tolist(),
        sub["frame_index"].astype(int).tolist(),
    )


def _all_episode_indices(dataset: LeRobotDataset) -> list[int]:
    cols = dataset.hf_dataset.select_columns(["episode_index"]).to_pandas()
    return sorted(int(v) for v in cols["episode_index"].unique())


def _render_episode(
    dataset: LeRobotDataset,
    image_keys: list[str],
    episode_index: int,
    crop_module: torch.nn.Module,
    crop_h: int,
    crop_w: int,
    out_path: Path,
    fps: int,
    upscale: int,
    aug_on: bool,
    crop_label: str,
) -> int:
    rows, frame_idxs = _episode_rows(dataset, episode_index)
    if not rows:
        logging.warning("episode %d: no frames found, skipping", episode_index)
        return 0

    up = max(1, int(upscale))
    out_w = crop_w * len(image_keys) * up
    out_h = crop_h * up
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(out_path, out_w, out_h, fps)

    try:
        for i, (row, fidx) in enumerate(zip(rows, frame_idxs)):
            sample = dataset[row]
            tiles: List[np.ndarray] = []
            for k in image_keys:
                t = sample[k]
                if t.ndim == 4:
                    t = t[-1]
                t_cropped = crop_module(t)
                rgb = (t_cropped.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
                bgr = rgb[..., ::-1].copy()
                tiles.append(_annotate(bgr, [k.split(".")[-1]]))
            canvas = np.concatenate(tiles, axis=1)
            canvas = _annotate(
                canvas,
                [
                    f"ep {episode_index}  frame {fidx}",
                    f"aug={'on' if aug_on else 'off'}  crop={crop_h}x{crop_w} ({crop_label})",
                ],
            )
            if up > 1:
                canvas = cv2.resize(
                    canvas,
                    (canvas.shape[1] * up, canvas.shape[0] * up),
                    interpolation=cv2.INTER_NEAREST,
                )
            writer.write(np.ascontiguousarray(canvas))
            if (i + 1) % 50 == 0 or i == 0:
                logging.info("  ep %d: wrote %d/%d", episode_index, i + 1, len(rows))
    finally:
        writer.release()
    return len(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the exact training-time view (image_transforms + RandomCrop) "
            "of one or all episodes from a dp*_cfg.json as an MP4."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training JSON config (e.g. dp64_cfg.json) with dataset.image_transforms and policy.crop_shape.",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--episode", type=int, help="Single episode index to render.")
    target.add_argument(
        "--all-episodes",
        action="store_true",
        help="Render every episode in the dataset. Requires --output-dir (instead of --output).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="MP4 path. Required when --episode is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write one MP4 per episode. Required when --all-episodes is set.",
    )

    parser.add_argument(
        "--crop-mode",
        choices=("config", "random", "center"),
        default="config",
        help=(
            "Override the policy's crop sampling. 'config' uses policy.crop_is_random "
            "from the cfg (default). 'random' = training-time RandomCrop. "
            "'center' = deployment-time CenterCrop."
        ),
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable dataset.image_transforms (still applies the policy crop).",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=3,
        help="Integer nearest-neighbour upscale of the rendered MP4 (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch/numpy seed (default: 0).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.episode is not None and args.output is None:
        sys.exit("--output is required when --episode is set.")
    if args.all_episodes and args.output_dir is None:
        sys.exit("--output-dir is required when --all-episodes is set.")

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        sys.exit(f"Config not found: {cfg_path}")
    raw_cfg = json.loads(cfg_path.read_text())

    ds_cfg = raw_cfg.get("dataset") or {}
    pol_cfg = raw_cfg.get("policy") or {}
    repo_id = ds_cfg.get("repo_id")
    root = ds_cfg.get("root")
    if not repo_id or not root:
        sys.exit("Config is missing dataset.repo_id or dataset.root")

    crop_shape = pol_cfg.get("crop_shape")
    if not crop_shape or len(crop_shape) != 2:
        sys.exit("Config policy.crop_shape must be [H, W].")
    crop_h, crop_w = int(crop_shape[0]), int(crop_shape[1])

    cfg_crop_is_random = bool(pol_cfg.get("crop_is_random", True))
    if args.crop_mode == "config":
        use_random_crop = cfg_crop_is_random
    elif args.crop_mode == "random":
        use_random_crop = True
    else:
        use_random_crop = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_transforms_raw = ds_cfg.get("image_transforms") or {}
    img_tf_cfg = _build_image_transforms_config(image_transforms_raw)
    aug_on = bool(img_tf_cfg.enable) and not args.no_aug
    img_tf = ImageTransforms(img_tf_cfg) if aug_on else None

    logging.info(
        "Loading dataset: repo_id=%s root=%s  crop=%dx%d (%s)  aug=%s",
        repo_id,
        root,
        crop_h,
        crop_w,
        "random" if use_random_crop else "center",
        "on" if aug_on else "off",
    )
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
        sys.exit("Dataset reports no camera keys — nothing to render.")
    fps = int(getattr(dataset.meta, "fps", 10) or 10)

    crop_module = T.RandomCrop((crop_h, crop_w)) if use_random_crop else T.CenterCrop((crop_h, crop_w))
    crop_label = "random" if use_random_crop else "center"

    if args.episode is not None:
        out = args.output.expanduser().resolve()
        n = _render_episode(
            dataset=dataset,
            image_keys=image_keys,
            episode_index=int(args.episode),
            crop_module=crop_module,
            crop_h=crop_h,
            crop_w=crop_w,
            out_path=out,
            fps=fps,
            upscale=args.upscale,
            aug_on=aug_on,
            crop_label=crop_label,
        )
        if n == 0:
            sys.exit(f"Episode {args.episode} not found in dataset.")
        logging.info("Done. ep %d: %d frames -> %s", args.episode, n, out)
        return 0

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = _all_episode_indices(dataset)
    logging.info("Rendering all %d episodes -> %s", len(eps), out_dir)
    for ep in eps:
        out = out_dir / f"episode_{ep:04d}_model_view.mp4"
        _render_episode(
            dataset=dataset,
            image_keys=image_keys,
            episode_index=ep,
            crop_module=crop_module,
            crop_h=crop_h,
            crop_w=crop_w,
            out_path=out,
            fps=fps,
            upscale=args.upscale,
            aug_on=aug_on,
            crop_label=crop_label,
        )
    logging.info("Done. %d episodes -> %s", len(eps), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inference-time preprocessing helpers shared by deploy.py and dagger.py.

The intent is that whatever spatial preprocessing a policy was trained with is
also applied identically at inference time, keyed by the policy's camera
feature key (e.g. ``observation.images.front``). Today this is just a
deterministic per-camera crop + resize-back pipeline; the module exists so
training-side and deploy-side stay in lockstep through one definition.

The training-side counterpart lives in
``lerobot.datasets.transforms.ImageTransforms.apply_for_key`` and is fed by
``ImageTransformsConfig.per_camera_crops``. Deploy/dagger reach into the
training run's ``train_config.json`` to recover the same dictionary; see
``detect_per_camera_crops`` in ``lerobot_ros2.policy_runtime``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torchvision.transforms.v2 import functional as F  # noqa: N812

_REQUIRED_KEYS = ("top", "left", "height", "width")


def _coerce_spec(camera_key: str, raw: Any) -> Optional[Dict[str, int]]:
    """Coerce a raw JSON crop spec into ``{top, left, height, width}`` ints.

    Returns ``None`` (with a warning) when the spec is malformed. Inference
    paths must not crash on a misconfigured policy directory; missing or bad
    crops simply mean "skip the crop for this camera."
    """
    if not isinstance(raw, Mapping):
        logging.warning(
            "per_camera_crops[%r] is not a mapping (%s); ignoring.",
            camera_key,
            type(raw).__name__,
        )
        return None
    out: Dict[str, int] = {}
    for k in _REQUIRED_KEYS:
        if k not in raw:
            logging.warning(
                "per_camera_crops[%r] missing key %r; ignoring entry.",
                camera_key,
                k,
            )
            return None
        v = raw[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            logging.warning(
                "per_camera_crops[%r][%r] must be a number; got %r. Ignoring entry.",
                camera_key,
                k,
                v,
            )
            return None
        if isinstance(v, float) and not v.is_integer():
            logging.warning(
                "per_camera_crops[%r][%r] must be an integer; got %r. Ignoring entry.",
                camera_key,
                k,
                v,
            )
            return None
        out[k] = int(v)
    if out["height"] <= 0 or out["width"] <= 0 or out["top"] < 0 or out["left"] < 0:
        logging.warning(
            "per_camera_crops[%r] has invalid bounds %s; ignoring entry.",
            camera_key,
            out,
        )
        return None
    return out


def normalize_per_camera_crops(raw: Any) -> Dict[str, Dict[str, int]]:
    """Validate / coerce a deserialized ``per_camera_crops`` block.

    Bad entries are dropped with a warning rather than raising, so a stale
    or partially-edited ``train_config.json`` cannot block inference.
    """
    if not isinstance(raw, Mapping):
        if raw is not None:
            logging.warning(
                "per_camera_crops must be a mapping; got %s. Ignoring.",
                type(raw).__name__,
            )
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for camera_key, spec in raw.items():
        coerced = _coerce_spec(str(camera_key), spec)
        if coerced is not None:
            out[str(camera_key)] = coerced
    return out


def load_per_camera_crops_from_train_config(
    policy_path: str | Path,
) -> Dict[str, Dict[str, int]]:
    """Read ``per_camera_crops`` out of ``train_config.json`` next to ``policy_path``.

    Mirrors ``policy_runtime.detect_training_resize`` in spirit: tolerate a missing or
    malformed file and return an empty mapping so inference keeps working.
    """
    candidates = [
        Path(policy_path) / "train_config.json",
        Path(policy_path).parent / "train_config.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read %s: %s", p, exc)
            continue
        raw = (
            cfg.get("dataset", {})
            .get("image_transforms", {})
            .get("per_camera_crops")
        )
        return normalize_per_camera_crops(raw)
    return {}


def apply_per_camera_crop(
    tensor: torch.Tensor,
    camera_key: str,
    crops: Mapping[str, Mapping[str, int]],
) -> torch.Tensor:
    """Crop ``tensor`` for ``camera_key`` then resize back to its original H/W.

    ``tensor`` is expected to be ``[..., C, H, W]`` (typically ``[C, H, W]``
    for the deploy/dagger code paths that build single-frame observations).
    Cameras absent from ``crops`` are returned unchanged. The crop is in the
    input tensor's stored coordinates, matching the training-time semantics.
    """
    spec = crops.get(camera_key) if isinstance(crops, Mapping) else None
    if not spec:
        return tensor
    if tensor.ndim < 3:
        return tensor
    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    if spec["top"] + spec["height"] > h or spec["left"] + spec["width"] > w:
        logging.warning(
            "per_camera_crops[%r]=%s exceeds image (H=%d, W=%d); skipping crop.",
            camera_key,
            dict(spec),
            h,
            w,
        )
        return tensor
    cropped = F.crop(
        tensor,
        int(spec["top"]),
        int(spec["left"]),
        int(spec["height"]),
        int(spec["width"]),
    )
    return F.resize(cropped, [h, w], antialias=True)


def load_fullres_crops_from_train_config(
    policy_path: str | Path,
) -> Dict[str, Dict[str, int]]:
    """Recover the per-camera *full-resolution* crop ROIs used to build a dataset.

    Unlike :func:`load_per_camera_crops_from_train_config` (which reads a
    training-time augmentation crop applied to the already-downsampled frame),
    this reads the source ROI that ``downsample.py`` baked into the dataset by
    cropping the original recordings before resizing. Those ROIs live under each
    video feature in the dataset's ``meta/info.json`` as
    ``info.source_crop`` / ``info.source_shape``.

    We locate the dataset through ``train_config.json`` (``dataset.root``) next
    to ``policy_path`` and return ``{feature_key: {top,left,height,width,
    source_shape:[H,W]}}``, keeping only cameras whose ROI is smaller than the
    full source frame (i.e. real crops; full-frame entries are no-ops). Returns
    an empty mapping when anything is missing or malformed, so inference keeps
    working.
    """
    candidates = [
        Path(policy_path) / "train_config.json",
        Path(policy_path).parent / "train_config.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read %s: %s", p, exc)
            continue
        root = (cfg.get("dataset") or {}).get("root")
        if not root:
            return {}
        info_path = Path(root) / "meta" / "info.json"
        if not info_path.exists():
            logging.warning("fullres crops: dataset info.json missing at %s", info_path)
            return {}
        try:
            info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read %s: %s", info_path, exc)
            return {}
        out: Dict[str, Dict[str, int]] = {}
        for key, ft in (info.get("features") or {}).items():
            if not isinstance(ft, dict) or ft.get("dtype") != "video":
                continue
            meta = ft.get("info") or {}
            coerced = _coerce_spec(str(key), meta.get("source_crop"))
            if coerced is None:
                continue
            src_shape = meta.get("source_shape")
            if isinstance(src_shape, (list, tuple)) and len(src_shape) == 2:
                sh, sw = int(src_shape[0]), int(src_shape[1])
                # Drop full-frame (no-op) crops so we only touch real ROIs.
                if (
                    coerced["top"] == 0
                    and coerced["left"] == 0
                    and coerced["height"] >= sh
                    and coerced["width"] >= sw
                ):
                    continue
                coerced["source_shape"] = [sh, sw]
            out[str(key)] = coerced
        return out
    return {}


def apply_fullres_crop(
    bgr,
    camera_key: str,
    crops: Mapping[str, Mapping[str, int]],
):
    """Crop a raw ``H x W x C`` BGR frame to the training ROI (no resize).

    This mirrors the offline crop-then-resize: the caller resizes the returned
    region to the policy's visual size afterwards. When the live frame size
    differs from the recorded ``source_shape``, the ROI is scaled
    proportionally so the same physical region is selected. Cameras absent from
    ``crops`` (or a ``None`` frame) are returned unchanged.
    """
    spec = crops.get(camera_key) if isinstance(crops, Mapping) else None
    if not spec or bgr is None:
        return bgr
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    top, left = int(spec["top"]), int(spec["left"])
    ch, cw = int(spec["height"]), int(spec["width"])
    src_shape = spec.get("source_shape") if isinstance(spec, Mapping) else None
    if isinstance(src_shape, (list, tuple)) and len(src_shape) == 2:
        sh, sw = int(src_shape[0]), int(src_shape[1])
        if sh > 0 and sw > 0 and (sh != h or sw != w):
            sy, sx = h / sh, w / sw
            top, ch = int(round(top * sy)), int(round(ch * sy))
            left, cw = int(round(left * sx)), int(round(cw * sx))
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    bottom = min(h, top + ch)
    right = min(w, left + cw)
    if bottom <= top or right <= left:
        return bgr
    return bgr[top:bottom, left:right]

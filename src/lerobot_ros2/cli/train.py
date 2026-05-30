#!/usr/bin/env python
"""Thin wrapper around ``lerobot-train`` that mirrors checkpoints to Hugging Face.

Behaves exactly like stock ``lerobot-train`` (same draccus parser, same
``TrainPipelineConfig`` / CLI flags). After training finishes, the whole
``--output_dir`` (all saved checkpoints, including ``last/``) is mirrored to its
private HF repo via :mod:`lerobot_ros2.hub_sync`, following the mapping in
``config/hf_backup.yaml``.

Use this instead of ``lerobot-train`` when you want checkpoints auto-backed-up.
For action-source / DAgger datasets use ``lerobot-ros-train-dagger`` (which has
the same auto-push behavior).

Notes:
  * Pass ``--no-push`` to skip the upload, or set ``LEROBOT_HF_PUSH=0``.
  * For the backup mapping to work, point ``--output_dir`` somewhere under
    ``data/`` (e.g. ``--output_dir=data/<op>/<task>/.../deploy``). Out-of-tree
    paths (like ``outputs/...``) are skipped with a warning; back them up
    manually with ``lerobot-ros-backup <output_dir>``.

Usage::

    lerobot-ros-train \\
      --dataset.repo_id=local/pick_place_ds5 \\
      --dataset.root=data/smrithi/pick_vial_20260524/train/pick_vial_ds5 \\
      --policy.type=diffusion \\
      --output_dir=data/smrithi/pick_vial_20260524/deploy/pick_vial_ds5 \\
      --steps=40000
"""

import sys

import lerobot.scripts.lerobot_train as _lt
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_ros2.hub_sync import pop_no_push_flag, try_sync_to_hub

# Set in main() before the draccus parser runs.
_PUSH: bool | None = None


@parser.wrap()
def _train(cfg: TrainPipelineConfig) -> None:
    _lt.train(cfg)
    # NOTE: print()/logging here runs after lerobot's own logging setup.
    try_sync_to_hub(cfg.output_dir, push=_PUSH)


def main() -> None:
    global _PUSH
    # Strip --no-push before draccus sees it (it would reject the unknown flag).
    _PUSH = False if pop_no_push_flag(sys.argv) else None
    register_third_party_plugins()
    _train()


if __name__ == "__main__":
    main()

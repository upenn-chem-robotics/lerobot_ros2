#!/usr/bin/env python
"""Drop-in replacement for ``lerobot-train`` that uses an action-source-aware sampler.

Behaves exactly like ``lerobot-train``: same CLI flags, same draccus parser,
same ``TrainPipelineConfig``. The only differences are:

* Before training, ``action_source`` is loaded from
  ``cfg.dataset.root/data/**/*.parquet`` and validated against
  ``dataset.num_frames``.
* The :class:`EpisodeAwareSampler` reference inside
  :mod:`lerobot.scripts.lerobot_train` is monkey-patched with
  :class:`ActionSourceAwareEpisodeSampler` (closure-bound to the loaded
  ``action_source``), so policy-prefix frames remain in the dataset as
  observation context but are never picked as anchors.

Use this entry point any time you train on a dataset that contains an
``action_source`` column (e.g. the aggregated base + dagger dataset
produced by the workflow in ``src/lerobot_ros2/cli/dagger.py``). For plain
imitation datasets without ``action_source``, just use stock
``lerobot-train`` -- this script will refuse to run if the column is
missing, which is by design (silent fall-back would mask bugs).

Usage::

    conda run -n lerobot --no-capture-output \\
      python -m lerobot_ros2.cli.train_dagger \\
        --dataset.repo_id=local/cap_plus_dagger4_ds5_no_back \\
        --dataset.root=data/cap_plus_dagger4_ds5_no_back \\
        --policy.type=strided_diffusion \\
        --policy.fps=10 \\
        --policy.n_obs_steps=2 \\
        --policy.horizon=16 \\
        --policy.n_action_steps=8 \\
        --policy.drop_n_last_frames=7 \\
        --policy.resize_shape='[144,256]' \\
        --batch_size=32 --steps=200000 \\
        --output_dir=outputs/dp_dagger_round1
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import lerobot.scripts.lerobot_train as _lt
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_ros2.training.action_source_sampler import ActionSourceAwareEpisodeSampler


def _load_action_source(dataset_root: Path) -> np.ndarray:
    """Concatenate ``action_source`` across all parquets under ``<root>/data``."""
    parts = sorted((Path(dataset_root) / "data").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}/data")
    arrays = []
    for p in parts:
        schema = pq.read_schema(p)
        if "action_source" not in schema.names:
            raise SystemExit(
                f"Dataset at {dataset_root} is missing 'action_source' column "
                f"(found in parquet {p}). Use lerobot-train for plain imitation "
                f"datasets, or run dagger.py / add_features to add the column."
            )
        arrays.append(pq.read_table(p, columns=["action_source"])
                        .column("action_source").to_numpy(zero_copy_only=False))
    return np.concatenate(arrays).astype(np.int64).reshape(-1)


@parser.wrap()
def train_dagger(cfg: TrainPipelineConfig) -> None:
    """Like :func:`lerobot.scripts.lerobot_train.train` but with action-source filtering."""
    cfg.validate()

    if cfg.dataset.root is None:
        raise SystemExit(
            "train_dagger requires --dataset.root to point at the dataset folder "
            "(so the action_source column can be loaded for the sampler)."
        )

    action_source = _load_action_source(Path(cfg.dataset.root))
    n_total = int(action_source.shape[0])
    n_human = int((action_source == 1).sum())
    n_policy = int((action_source == 0).sum())
    # NOTE: print() (not logging) so the message survives lerobot's
    # init_logging() reconfiguration that runs *inside* _lt.train().
    print(
        f"[train_dagger] action_source loaded from {cfg.dataset.root}: "
        f"total={n_total}, human(=1)={n_human}, policy(=0)={n_policy} "
        f"(dropping {n_policy} frames as anchors)",
        flush=True,
    )

    class _BoundSampler(ActionSourceAwareEpisodeSampler):
        """``EpisodeAwareSampler``-shaped factory closing over ``action_source``."""

        _announced = False

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, action_source=action_source, **kwargs)
            if not _BoundSampler._announced:
                print(
                    f"[train_dagger] ActionSourceAwareEpisodeSampler active: "
                    f"{len(self.indices)} anchors after action_source==1 filter "
                    f"(dropped {self.dropped_by_action_source} policy-prefix frames)",
                    flush=True,
                )
                _BoundSampler._announced = True

    _orig = _lt.EpisodeAwareSampler
    _lt.EpisodeAwareSampler = _BoundSampler
    try:
        _lt.train(cfg)
    finally:
        _lt.EpisodeAwareSampler = _orig


def main() -> None:
    register_third_party_plugins()
    train_dagger()


if __name__ == "__main__":
    main()

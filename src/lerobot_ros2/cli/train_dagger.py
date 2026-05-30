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
* If ``--dagger_fraction`` is given (a value in ``[0, 1]``), the sampler is
  upgraded to :class:`WeightedDaggerEpisodeSampler` which draws anchors via
  weighted multinomial sampling so that, in expectation, ``dagger_fraction``
  of the per-epoch samples come from dagger episodes (episodes containing at
  least one ``action_source == 0`` frame). Useful when dagger episodes are
  longer than base demos and you don't want them to dominate the loss.

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
        --output_dir=outputs/dp_dagger_round1 \\
        --dagger_fraction=0.25
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import lerobot.scripts.lerobot_train as _lt
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_ros2.hub_sync import pop_no_push_flag, try_sync_to_hub
from lerobot_ros2.training.action_source_sampler import (
    ActionSourceAwareEpisodeSampler,
    WeightedDaggerEpisodeSampler,
)


def _pop_dagger_fraction_from_argv(argv: list[str]) -> float | None:
    """Strip ``--dagger_fraction`` / ``--dagger-fraction`` from ``argv`` and return its value.

    Done before :func:`parser.wrap` runs so the draccus parser doesn't see (and
    reject) this flag. Accepts both ``--dagger_fraction 0.25`` and
    ``--dagger_fraction=0.25`` forms.
    """
    aliases = ("--dagger_fraction", "--dagger-fraction")
    out: float | None = None
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        matched = False
        for name in aliases:
            if tok == name and i + 1 < len(argv):
                out = float(argv[i + 1])
                i += 2
                matched = True
                break
            if tok.startswith(name + "="):
                out = float(tok.split("=", 1)[1])
                i += 1
                matched = True
                break
        if not matched:
            cleaned.append(tok)
            i += 1
    argv[:] = cleaned
    return out


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


def _load_episode_index(dataset_root: Path) -> np.ndarray:
    """Concatenate ``episode_index`` across all parquets under ``<root>/data``."""
    parts = sorted((Path(dataset_root) / "data").rglob("*.parquet"))
    arrays = [
        pq.read_table(p, columns=["episode_index"])
        .column("episode_index")
        .to_numpy(zero_copy_only=False)
        for p in parts
    ]
    return np.concatenate(arrays).astype(np.int64).reshape(-1)


def _compute_is_dagger_per_frame(
    action_source: np.ndarray, episode_index: np.ndarray
) -> np.ndarray:
    """Return a per-frame bool array marking dagger-episode frames.

    A frame is "dagger" iff *its episode* contains at least one
    ``action_source == 0`` frame. Base/teleop episodes (where every frame is
    ``action_source == 1``) are marked ``False`` for every row.
    """
    if action_source.shape != episode_index.shape:
        raise ValueError(
            f"action_source shape {action_source.shape} != episode_index shape "
            f"{episode_index.shape}"
        )
    has_policy_per_ep: dict[int, bool] = {}
    for ep in np.unique(episode_index):
        ep = int(ep)
        mask = episode_index == ep
        has_policy_per_ep[ep] = bool((action_source[mask] == 0).any())
    return np.array(
        [has_policy_per_ep[int(ep)] for ep in episode_index], dtype=bool
    )


# Set at module import time (after _pop_dagger_fraction_from_argv runs in main()).
# Closed over by _BoundSampler / _BoundWeightedSampler below.
_DAGGER_FRACTION: float | None = None

# Set in main() from a popped --no-push flag (None => default/env decides).
_PUSH: bool | None = None


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

    is_dagger_per_frame: np.ndarray | None = None
    if _DAGGER_FRACTION is not None:
        episode_index = _load_episode_index(Path(cfg.dataset.root))
        is_dagger_per_frame = _compute_is_dagger_per_frame(action_source, episode_index)
        n_dagger_frames = int(is_dagger_per_frame.sum())
        n_base_frames = int((~is_dagger_per_frame).sum())
        print(
            f"[train_dagger] dagger-aware reweighting active: "
            f"target dagger_fraction={_DAGGER_FRACTION:.3f}; "
            f"base-episode frames={n_base_frames}, "
            f"dagger-episode frames={n_dagger_frames}",
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

    class _BoundWeightedSampler(WeightedDaggerEpisodeSampler):
        """:class:`WeightedDaggerEpisodeSampler` closed over the loaded arrays."""

        _announced = False

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(
                *args,
                action_source=action_source,
                is_dagger_per_frame=is_dagger_per_frame,
                dagger_fraction=_DAGGER_FRACTION,
                **kwargs,
            )
            if not _BoundWeightedSampler._announced:
                print(
                    f"[train_dagger] WeightedDaggerEpisodeSampler active: "
                    f"target dagger_fraction={self.dagger_fraction:.3f}; "
                    f"base anchors={self.n_base_anchors}, "
                    f"dagger anchors={self.n_dagger_anchors}, "
                    f"num_samples per epoch={self.num_samples}",
                    flush=True,
                )
                _BoundWeightedSampler._announced = True

    bound = _BoundWeightedSampler if _DAGGER_FRACTION is not None else _BoundSampler
    _orig = _lt.EpisodeAwareSampler
    _lt.EpisodeAwareSampler = bound
    try:
        _lt.train(cfg)
    finally:
        _lt.EpisodeAwareSampler = _orig

    # Mirror the checkpoints (output_dir) to Hugging Face. Best-effort: never
    # fails the run. Disable with --no-push or LEROBOT_HF_PUSH=0.
    try_sync_to_hub(cfg.output_dir, push=_PUSH)


def main() -> None:
    global _DAGGER_FRACTION, _PUSH
    _PUSH = False if pop_no_push_flag(sys.argv) else None
    _DAGGER_FRACTION = _pop_dagger_fraction_from_argv(sys.argv)
    register_third_party_plugins()
    train_dagger()


if __name__ == "__main__":
    main()

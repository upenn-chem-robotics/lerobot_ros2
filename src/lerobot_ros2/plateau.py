"""Detect plateau (no-op) frames in a LeRobot dataset.

A *plateau* is a contiguous run of frames where the operator's commanded
action barely changes -- typically a hesitation after pickup, mid-pour,
or any other moment where the demonstrator stayed still. Diffusion
policies trained on such frames learn "static observation -> static
action" as a confident behavior mode, which manifests at deploy time
as the policy pausing at the same semantic moments.

The algorithm:

1. Normalize each action joint to ``[0, 1]`` using the dataset's
   ``action.min`` / ``action.max`` (so the gripper joint, which lives
   on a totally different physical scale from arm joints, contributes
   on equal footing).
2. Compute a per-frame *speed* ``s_t``: the chosen norm (``Linf`` by
   default) of the normalized action delta between consecutive frames.
   ``Linf`` answers "did *any* joint move much?" which is the right
   question -- a slow pour driven by ``wrist_3`` rotation alone is
   still motion, not a plateau.
3. Mark every frame with ``s_t < tau`` as "low".
4. Find contiguous runs of low frames per episode. A run is a plateau
   iff its length is ``>= min_run``.
5. Inside each plateau run, keep the first ``margin`` and last
   ``margin`` frames as *non-plateau* (so the policy still has
   "decelerate into pause" and "accelerate out of pause" frames as
   trainable anchors). Mark only the interior frames as plateau.

The output ``plateau_mask`` has the same length as the dataset; it is
combined downstream with any existing ``action_source`` column via
``new_as = existing_as AND NOT plateau_mask`` so that anchor sampling
in :class:`ActionSourceAwareEpisodeSampler` skips both DAgger
policy-prefix frames *and* operator-hesitation frames.


The detection code is pure NumPy and has no dataset/parquet dependencies so
it can be reused from CLIs, notebooks, and unit tests alike. The argparse glue
at the bottom is shared by the three plateau CLIs so they cannot drift apart on
defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class PlateauParams:
    """Hyperparameters for plateau detection.

    Attributes:
        tau: Threshold on per-frame normalized action speed. Frames with
            ``s_t < tau`` are candidates for plateau membership.
        min_run: Minimum length (in frames) of a low-speed run to count
            as a plateau. Shorter runs (legitimate slow segments) are
            kept entirely.
        margin: Number of frames at each edge of a plateau run that are
            NOT marked plateau, so transition frames remain trainable
            anchors. Effective dropped length per plateau is
            ``max(0, run_length - 2 * margin)``.
        norm: ``"linf"`` (max joint delta) or ``"l2"`` (L2 over joints).
            ``"linf"`` is the safe default for heterogeneous joints.
        joint_indices: Optional subset of joint indices to consider when
            computing the speed. ``None`` = all joints. Useful for
            ignoring a noisy gripper signal during tuning, but not the
            recommended default.
    """

    tau: float = 0.005
    min_run: int = 5
    margin: int = 2
    norm: str = "linf"
    joint_indices: tuple[int, ...] | None = None


@dataclass
class PlateauResult:
    """Per-frame plateau detection output and aggregate stats."""

    plateau_mask: np.ndarray  # bool, shape (N,)
    speed: np.ndarray         # float64, shape (N,), normalized speed s_t
    is_low: np.ndarray        # bool, shape (N,), s_t < tau (no run/margin)
    action_norm: np.ndarray   # float64, shape (N, D), min-max normalized actions
    params: PlateauParams
    stats: dict = field(default_factory=dict)


def _normalize_actions(
    action: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> np.ndarray:
    """Min-max normalize each joint to ``[0, 1]``, with eps for constant joints."""
    span = action_max - action_min
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    return (action - action_min) / span


def _speed_per_frame(
    action_norm: np.ndarray,
    ep_idx: np.ndarray,
    norm: str,
    joint_indices: tuple[int, ...] | None,
) -> np.ndarray:
    """Per-frame normalized action speed, zeroed across episode boundaries.

    ``s_t = ||a_norm[t+1] - a_norm[t]||`` for ``t`` inside an episode,
    and ``s_{last} = s_{last-1}`` so the array has the same length as
    the dataset. At the last frame of any episode we copy the previous
    frame's speed (rather than crossing into the next episode's first
    action, which would produce a spurious huge delta).
    """
    n, d = action_norm.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if joint_indices is not None:
        sel = list(joint_indices)
        a = action_norm[:, sel]
    else:
        a = action_norm

    delta = np.zeros((n, a.shape[1]), dtype=np.float64)
    delta[:-1] = a[1:] - a[:-1]

    same_ep = np.zeros(n, dtype=bool)
    same_ep[:-1] = ep_idx[1:] == ep_idx[:-1]
    delta[~same_ep] = 0.0

    if norm.lower() == "l2":
        speed = np.linalg.norm(delta, ord=2, axis=1)
    elif norm.lower() == "linf":
        speed = np.max(np.abs(delta), axis=1)
    else:
        raise ValueError(f"Unknown norm {norm!r}; expected 'linf' or 'l2'")

    # The last frame of every episode has ``delta = 0`` by construction
    # (``~same_ep`` covered it above), so ``speed`` there is 0 and will
    # naturally fall below ``tau``. Short single-frame "runs" at episode
    # ends are filtered out by ``min_run`` anyway.
    return speed


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive), ...]`` runs of ``True`` in ``mask``."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(diff == 1) + 1)
    ends = list(np.flatnonzero(diff == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def detect_plateaus(
    action: np.ndarray,
    episode_index: np.ndarray,
    action_min: Iterable[float],
    action_max: Iterable[float],
    params: PlateauParams | None = None,
) -> PlateauResult:
    """Detect plateau (no-op) frames in a dataset's action sequence.

    Args:
        action: ``(N, D)`` float array of demonstrator actions.
        episode_index: ``(N,)`` int array; ``episode_index[t]`` is the
            episode that frame ``t`` belongs to. Must be contiguous per
            episode (i.e. all rows of episode ``e`` occupy a contiguous
            slice of ``[0, N)``).
        action_min: per-joint min of length ``D`` (from dataset stats).
        action_max: per-joint max of length ``D`` (from dataset stats).
        params: hyperparameters; default :class:`PlateauParams`.

    Returns:
        :class:`PlateauResult` with the bool ``plateau_mask`` (frames to
        flip to ``action_source=0``), the per-frame normalized
        ``speed``, the raw ``is_low`` flags, and a ``stats`` dict with
        aggregate counts and a per-episode breakdown.
    """
    params = params or PlateauParams()

    action = np.asarray(action, dtype=np.float64)
    ep_idx = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    a_min = np.asarray(list(action_min), dtype=np.float64)
    a_max = np.asarray(list(action_max), dtype=np.float64)

    if action.ndim != 2:
        raise ValueError(f"action must be (N, D); got shape {action.shape}")
    if action.shape[0] != ep_idx.shape[0]:
        raise ValueError(
            f"action rows ({action.shape[0]}) != episode_index length "
            f"({ep_idx.shape[0]})"
        )
    if a_min.shape != (action.shape[1],) or a_max.shape != (action.shape[1],):
        raise ValueError(
            f"action_min/max must be (D,) with D={action.shape[1]}; got "
            f"{a_min.shape}, {a_max.shape}"
        )

    action_norm = _normalize_actions(action, a_min, a_max)
    speed = _speed_per_frame(action_norm, ep_idx, params.norm, params.joint_indices)
    is_low = speed < params.tau

    plateau_mask = np.zeros_like(is_low)

    ep_stats: list[dict] = []
    unique_eps, ep_starts = np.unique(ep_idx, return_index=True)
    order = np.argsort(ep_starts)
    unique_eps = unique_eps[order]
    ep_starts = ep_starts[order]

    for i, ep in enumerate(unique_eps):
        start = int(ep_starts[i])
        end = int(ep_starts[i + 1]) if i + 1 < len(ep_starts) else action.shape[0]
        ep_low = is_low[start:end]
        ep_speed = speed[start:end]
        runs = _runs_of_true(ep_low)
        ep_dropped = 0
        plateau_runs = 0
        run_lengths: list[int] = []
        for s, e in runs:
            run_len = e - s
            if run_len < params.min_run:
                continue
            plateau_runs += 1
            run_lengths.append(run_len)
            interior_s = s + params.margin
            interior_e = e - params.margin
            if interior_e > interior_s:
                plateau_mask[start + interior_s:start + interior_e] = True
                ep_dropped += interior_e - interior_s
        ep_stats.append({
            "episode_index": int(ep),
            "length": int(end - start),
            "n_low_frames": int(ep_low.sum()),
            "n_plateau_runs": int(plateau_runs),
            "n_dropped_frames": int(ep_dropped),
            "max_run_length": int(max(run_lengths) if run_lengths else 0),
            "mean_speed": float(ep_speed.mean()),
            "p99_speed": float(np.quantile(ep_speed, 0.99)) if ep_speed.size else 0.0,
        })

    total = int(plateau_mask.size)
    dropped = int(plateau_mask.sum())
    stats = {
        "n_total_frames": total,
        "n_low_frames": int(is_low.sum()),
        "n_dropped_frames": dropped,
        "dropped_fraction": float(dropped / total) if total else 0.0,
        "n_episodes": int(len(unique_eps)),
        "speed_p50": float(np.quantile(speed, 0.50)),
        "speed_p90": float(np.quantile(speed, 0.90)),
        "speed_p99": float(np.quantile(speed, 0.99)),
        "speed_max": float(speed.max()) if speed.size else 0.0,
        "params": {
            "tau": float(params.tau),
            "min_run": int(params.min_run),
            "margin": int(params.margin),
            "norm": str(params.norm),
            "joint_indices": (
                list(params.joint_indices) if params.joint_indices is not None else None
            ),
        },
        "per_episode": ep_stats,
    }

    return PlateauResult(
        plateau_mask=plateau_mask,
        speed=speed,
        is_low=is_low,
        action_norm=action_norm,
        params=params,
        stats=stats,
    )


def load_action_min_max_from_stats(stats_json_path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``action.min``/``action.max`` from a LeRobot v3 ``stats.json``.

    LeRobot v3 datasets keep a separate per-episode stats block inside
    ``meta/episodes/...`` but ``meta/stats.json`` (when present) carries
    the dataset-level aggregate, which is what we want for normalization.
    """
    import json
    from pathlib import Path

    p = Path(stats_json_path)
    raw = json.loads(p.read_text())
    if "action" not in raw:
        raise KeyError(f"{p}: no 'action' block")
    a = raw["action"]
    return np.asarray(a["min"], dtype=np.float64), np.asarray(a["max"], dtype=np.float64)


# ── Shared CLI glue ──────────────────────────────────────────────────────
#
# ``plateau-stats``, ``plateau-visualize`` and ``add-action-source-with-plateau``
# must agree on these knobs: a threshold that means one thing when you tune it
# and another when you apply it would silently produce a mislabelled dataset.

def add_plateau_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the plateau detection knobs shared by every plateau CLI."""
    parser.add_argument(
        "--tau", type=float, default=PlateauParams.tau,
        help=f"Normalized speed threshold (default: {PlateauParams.tau}).",
    )
    parser.add_argument(
        "--min-run", type=int, default=PlateauParams.min_run,
        help="Minimum length (frames) of a low-speed run to count as a plateau "
             f"(default: {PlateauParams.min_run}).",
    )
    parser.add_argument(
        "--margin", type=int, default=PlateauParams.margin,
        help="Frames at each edge of a plateau kept as anchors "
             f"(default: {PlateauParams.margin}).",
    )
    parser.add_argument(
        "--norm", choices=("linf", "l2"), default=PlateauParams.norm,
        help=f"Norm for the action delta (default: {PlateauParams.norm}).",
    )
    parser.add_argument(
        "--joints", type=str, default=None,
        help="Optional comma-separated joint indices to consider "
             "(e.g. '0,1,2,3,4,5'). Default = all joints.",
    )
    return parser


def parse_joint_indices(spec: str | None) -> tuple[int, ...] | None:
    """Turn a ``--joints`` string like ``"0,1,5"`` into a tuple, or ``None``."""
    if not spec:
        return None
    return tuple(int(x) for x in spec.split(","))


def params_from_args(args: argparse.Namespace) -> PlateauParams:
    """Build :class:`PlateauParams` from a parser built with :func:`add_plateau_args`."""
    return PlateauParams(
        tau=float(args.tau),
        min_run=int(args.min_run),
        margin=int(args.margin),
        norm=args.norm,
        joint_indices=parse_joint_indices(args.joints),
    )

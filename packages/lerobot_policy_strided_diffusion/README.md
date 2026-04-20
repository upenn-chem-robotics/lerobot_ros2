# lerobot_policy_strided_diffusion

Third-party LeRobot policy plugin that adds a `strided_diffusion` policy type:
a `DiffusionPolicy` whose observation window is uniformly spaced in seconds
(e.g. `[obs_{t-2s}, obs_{t-1s}, obs_t]`) instead of stock consecutive frames.

The plugin is a single `StridedDiffusionConfig` dataclass that subclasses
`DiffusionConfig` and overrides `observation_delta_indices`. The stock
`lerobot-train` pipeline discovers it through
`lerobot.utils.import_utils.register_third_party_plugins()` (which enumerates
`importlib.metadata.distributions()` for names starting with
`lerobot_policy_`); no edits to the lerobot repo or to `lerobot_train.py` are
required.

## Install

Already baked into the image (see the top-level `Dockerfile`). For an ad-hoc
install outside the container:

```bash
pip install -e packages/lerobot_policy_strided_diffusion
```

## Train

```bash
conda run -n lerobot --no-capture-output lerobot-train \
  --dataset.repo_id=local/pick \
  --dataset.root=data/rama/pick \
  --policy.type=strided_diffusion \
  --policy.fps=30 \
  --policy.stride_seconds=1.0 \
  --policy.n_obs_steps=3 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.drop_n_last_frames=6 \
  --policy.resize_shape='[240,320]' \
  --batch_size=32 \
  --steps=200000 \
  --output_dir=outputs/dp_pick_strided
```

`--policy.fps` must match the dataset fps (the training pipeline divides our
integer frame offsets by `ds_meta.fps` to get seconds, which only round-trips
when the two agree). Early-episode anchors train on copy-padded history for
the first `(n_obs_steps - 1) * stride_seconds * fps` frames per episode; that
is semantically identical to `StridedHistoryRunner`'s inference-time warm-start
and is the intended behaviour.

## Deploy

`lerobot-ros-deploy` (`lerobot_ros2.cli.deploy`) auto-detects strided checkpoints
via `load_strided_config`, which reads `pretrained_model/config.json` and picks
up the runner iff `type == "strided_diffusion"`. No deploy flags change.

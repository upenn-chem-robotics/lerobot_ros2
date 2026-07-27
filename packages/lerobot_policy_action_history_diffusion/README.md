# lerobot_policy_action_history_diffusion

Third-party LeRobot policy plugin that adds an `action_history_diffusion`
policy type: a `DiffusionPolicy` whose FiLM global-conditioning vector is
extended with a small MLP encoding of the last `K` *commanded* actions.

The motivation comes from the empirical observation that the most common
on-robot failure for `DiffusionPolicy` trained from human demos is *phase
ambiguity*: the policy gets visually stuck in a state that looks similar to
a moment when the demonstrator hesitated (e.g. mid-pour), and freezes. Past
actions are a cheap, well-targeted feature for this — they tell the model
"I've already been commanding small deltas for the last second, time to
move on" — and unlike adding extra observation history they don't multiply
the vision-encoder cost.

The plugin is intentionally minimal:

* `ActionHistoryDiffusionConfig` subclasses `DiffusionConfig` and adds three
  fields: `n_action_history`, `action_history_embed_dim`,
  `action_history_dropout_prob`. It overrides `action_delta_indices` so the
  data loader pulls `K + horizon` action slots per anchor (the first `K`
  are conditioning, the remaining `horizon` are the stock denoising target).
* `ActionHistoryDiffusionPolicy` subclasses `DiffusionPolicy` and, when
  `n_action_history > 0`, swaps `self.diffusion` for an
  `ActionHistoryDiffusionModel` that owns the past-action MLP and a wider
  U-Net (FiLM cond dim grown by `action_history_embed_dim`). The policy
  maintains its own ring buffer of committed actions for inference; the
  buffer is cleared by `reset()` and updated by `select_action()`.

When `n_action_history == 0` the policy is numerically identical to stock
`DiffusionPolicy`, which makes A/B comparisons trivial.

## Install

Requires `lerobot` to already be installed in the same environment (see the
top-level [README](../../README.md) for the pinned commit). Then, from the
repository root:

```bash
pip install -e packages/lerobot_policy_action_history_diffusion
```

The stock `lerobot-train` pipeline discovers the plugin through
`lerobot.utils.import_utils.register_third_party_plugins()` (which
enumerates `importlib.metadata.distributions()` for names starting with
`lerobot_policy_`); no edits to the lerobot repo or to `lerobot_train.py` are
required.

## Train

```bash
conda run -n lerobot --no-capture-output lerobot-train \
  --dataset.repo_id=local/pour \
  --dataset.root=data/pour_20260514_merged_ds5 \
  --policy.type=action_history_diffusion \
  --policy.n_obs_steps=1 \
  --policy.n_action_history=8 \
  --policy.action_history_embed_dim=128 \
  --policy.action_history_dropout_prob=0.1 \
  --policy.horizon=64 \
  --policy.n_action_steps=8 \
  --policy.drop_n_last_frames=7 \
  --batch_size=128 \
  --steps=40000 \
  --output_dir=outputs/dp_pour_action_history
```

Or point at a `config.json` (e.g. the sibling
`data/pour_20260514_merged_ds5/dp64_cfg_action_history.json` in this repo)
via `--config_path=...`.

### Picking the hyperparameters

* `n_action_history`: the time window in frames. At 10 fps, `K=8` is ~0.8 s
  of action history (typical hesitation length on a pour); `K=16` is ~1.6 s.
  Costs near nothing in compute, so it's worth sweeping.
* `action_history_embed_dim`: the size of the per-step embedding added to
  the FiLM cond vector. 128 is a sensible default; bigger only helps if you
  also widen the U-Net. Smaller (32) is fine if you want to push the
  representational bottleneck on the conditioning side.
* `action_history_dropout_prob`: probability of zeroing the past-action
  input on a training forward pass. **This is the causal-confusion
  mitigation knob.** Higher values force the model to keep relying on vision
  (because past actions can't be trusted on every batch), lower values let
  the model lean on past actions more freely. Start at `0.1` and try `0.0`,
  `0.25` if you see deploy-time drift. See "Known failure modes" below.

### Delta-index semantics (training)

For `K = n_action_history` and stock target window
`[1 - n_obs_steps, ..., horizon - n_obs_steps]`, the extended window is::

    history = [-K + (1 - n_obs_steps), ..., -n_obs_steps]   # K indices
    target  = [1 - n_obs_steps,        ..., horizon - n_obs_steps]

`history` is always *strictly before* `target`, so the action target never
overlaps the history conditioning (no information leakage). With
`n_obs_steps = 1` (the most common case) this collapses to
`history = [-K, ..., -1]`, `target = [0, ..., horizon-1]`.

## Deploy

`lerobot-ros-deploy` (`lerobot_ros2.cli.deploy`) detects checkpoints whose
`config.json` advertises `type == "action_history_diffusion"` and dispatches
to `ActionHistoryDiffusionPolicy.from_pretrained` so the past-action encoder
weights are restored. The policy owns its own ring buffer of committed
actions; there is no deploy-side wrapper analogous to `StridedHistoryRunner`.

The plugin and the strided plugin are independent — you can stack them if
you want both strided observation history and past-action conditioning
(just remember the strided plugin requires `DATASET_FPS` / `STRIDE_SECONDS`
env vars at training time).

## Known failure modes

* **Causal confusion** (de Haan et al. 2019,
  [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)). Expert action
  sequences are highly autocorrelated, so the model can learn
  `a_t ≈ f(a_{t-1})` and largely ignore the visual stream. At deploy time
  this manifests as the policy converging on a fixed action and locking
  there ("I just commanded small deltas, so the right thing to do is
  command another small delta..."). Mitigations:

  * Set `action_history_dropout_prob > 0` (default in the sample config
    is `0.1`). This forces the model to keep its visual pathway useful.
  * Validate by training a `n_action_history=0` baseline (which collapses
    to stock `DiffusionPolicy`) and comparing on-robot rollouts — not just
    held-out loss, which will look identical or even better for the
    causally-confused model.
  * The chunk-replay mechanism in `DiffusionPolicy.select_action` already
    biases rollouts toward smoothness (it commits `n_action_steps` actions
    per inference); past-action conditioning amplifies that. If you see
    drift, try lowering `n_action_steps`.

* **Off-by-one at deploy** for `n_obs_steps > 1`. The inference-time queue
  drops the `n_obs_steps - 1` most recent committed actions before
  building the history tensor, so the K conditioning slots align with the
  training-time delta indices (which are strictly before the target). For
  `n_obs_steps = 1` this is a no-op. See
  `ActionHistoryDiffusionPolicy._build_action_history_tensor` for the math.

## Tests

```bash
/opt/conda/envs/lerobot/bin/python -m pytest tests/test_action_history_diffusion.py -q
```

Covers the `action_delta_indices` math, the train-time batch split, end-to-end
forward producing finite loss (with `K > 0`, `K = 0`, and dropout = 1.0),
inference-queue tail alignment, warm-start zero padding, and a sanity check
that the extra parameters are <5% of stock model size.

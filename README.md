# lerobot-ros2

Recording, deploy, export and visualization utilities for UR3e bimanual
teleoperation on top of [huggingface/lerobot](https://github.com/huggingface/lerobot).

## Layout

```
lerobot-ros2/
├── config/gello.yaml           Example teleop / camera config
├── src/lerobot_ros2/           Main package (import `lerobot_ros2`)
│   ├── helper.py               Cameras, arms, wrap joints, ROS helpers, ...
│   ├── visualizer.py           OpenCV live preview
│   ├── strided_history.py      Inference-side runner for strided-diffusion
│   ├── data_loader.py          Parquet + video dataset reader
│   ├── video_exporter.py       Timeline / grid video export
│   ├── config_paths.py         `--config` / $GELLO_CONFIG resolution
│   └── cli/                    Console entry points (see below)
├── packages/
│   ├── lerobot_policy_strided_diffusion/        Sibling distribution (lerobot plugin)
│   └── lerobot_policy_action_history_diffusion/ Sibling distribution (lerobot plugin)
└── tests/
```

## Install

```bash
# One-shot dev setup (after `pip install -e huggingface/lerobot`)
pip install -e .
pip install -e packages/lerobot_policy_strided_diffusion
pip install -e packages/lerobot_policy_action_history_diffusion
```

Policy plugins live in their own distributions because
`lerobot.utils.import_utils.register_third_party_plugins` discovers plugins
by enumerating installed distributions whose name starts with
`lerobot_policy_`. Two are shipped here:

* `lerobot_policy_strided_diffusion` — `DiffusionPolicy` with a uniformly-strided
  observation window (e.g. `[obs_{t-1s}, obs_t]` instead of `[obs_{t-0.1s}, obs_t]`
  at 10 fps); same VRAM as stock `n_obs_steps=2`, longer effective horizon.
* `lerobot_policy_action_history_diffusion` — `DiffusionPolicy` with a queue
  of prior commanded actions appended to the FiLM global conditioning vector
  (no extra image / proprio history); negligible extra compute, targets
  phase-ambiguity failures at low VRAM cost.

## Config

All CLIs require a teleop/camera config. Point them at one via either:

```bash
export GELLO_CONFIG=/path/to/gello.yaml
# or per-invocation:
lerobot-ros-record --config /path/to/gello.yaml ...
```

There is no filesystem default — the installed package does not know where
your config lives.

## Cameras (OBSBOT Meet SE — disable AI auto-framing)

Standard v4l2 controls (autofocus, `zoom_absolute`, exposure, white balance)
are pinned per camera in `config/gello.yaml` and applied + verified at record /
deploy time (see `apply_v4l2_settings` in `helper.py`). Snapshot the current
controls with `lerobot-ros-probe-cameras`.

If your cameras are **OBSBOT Meet SE** units, their **AI auto-framing / gesture
zoom is not a v4l2 control** — it drives an internal `zoom_continuous` and makes
the image zoom in/out as objects move near the lens, which quietly corrupts the
image geometry a diffusion policy needs for mm-accurate tasks. It must be turned
off through OBSBOT's SDK:

```bash
# 1. One-time: build the OBSBOT control CLI (needs Rust+libclang, or Nix)
scripts/setup_obsbot_cli.sh
export OBSBOT_CLI=third_party/obsbot-meetse-cli/target/release/obsbot-cli

# 2. At the start of every session (cameras plugged in), before record/deploy:
scripts/obsbot_lock_cameras.sh            # auto-framing off, HDR off, zoom 1.0x
scripts/obsbot_lock_cameras.sh --focus 35 # also pin manual focus on every cam
```

The lock script hits every detected OBSBOT device by serial, so it needs no
per-camera mapping. Verify a camera stays put while you move the gripper:

```bash
watch -n0.5 'v4l2-ctl -d /dev/video2 --get-ctrl=zoom_continuous,zoom_absolute'
```

## Console entry points

| Command                | Description                                 |
|------------------------|---------------------------------------------|
| `lerobot-ros-record`   | Record teleoperation datasets               |
| `lerobot-ros-deploy`   | Deploy a trained ACT / Diffusion policy     |
| `lerobot-ros-export`   | Export dataset grid videos / timelines      |
| `lerobot-ros-probe-cameras` | Snapshot v4l2 controls per camera     |
| `lerobot-ros-downsample` | Rewrite + downsample a LeRobot v3 dataset |
| `lerobot-ros-app`      | Gradio dataset visualizer                   |
| `lerobot-ros-train`    | `lerobot-train` + auto-mirror checkpoints to HF |
| `lerobot-ros-train-dagger` | DAgger-aware training + auto-mirror to HF |
| `lerobot-ros-backup`   | Mirror / verify `data/` folders on Hugging Face |

## Hugging Face backup

Everything under `data/` (recorded datasets and training checkpoints) can be
mirrored to **private** Hugging Face *dataset* repos so old folders can be
deleted locally and restored later. The strategy is a **raw mirror**: each
experiment folder is uploaded verbatim (its `meta/ data/ videos/ deploy/` tree
preserved) into its own repo. The mapping lives in
[`config/hf_backup.yaml`](config/hf_backup.yaml).

### One-time setup

```bash
# 1. Create a HF account + a *Write* token at huggingface.co/settings/tokens
# 2. Authenticate this machine once:
hf auth login            # or: export HF_TOKEN=hf_xxx
```

Note: `data/` is large (hundreds of GB). Private storage on Hugging Face is
metered, so this likely requires an HF PRO plan. Edit `hf_user` /
`backup_targets` in `config/hf_backup.yaml` if your username or folder layout
changes.

### Back up existing data

```bash
# Mirror every folder listed in config/hf_backup.yaml (resumable; safe to re-run):
lerobot-ros-backup --all

# Or a single folder:
lerobot-ros-backup data/rama/dose_solid

# Preview the repo mapping without uploading:
lerobot-ros-backup --all --dry-run
```

### Verify, then delete locally

```bash
# Confirm a repo holds every local file before you remove the folder:
lerobot-ros-backup --verify data/rama/dose_solid
# Only after [OK]:
rm -rf data/rama/dose_solid
```

### Automatic upload going forward

By default, **nothing** is uploaded automatically. Use `lerobot-ros-backup` for
manual backup, or pass `--push` (or set `LEROBOT_HF_PUSH=1`) on
`lerobot-ros-record`, `lerobot-ros-dagger`, `lerobot-ros-train` and
`lerobot-ros-train-dagger` to mirror output when a run finishes. The push is
best-effort (it never fails the run; on error it prints the
`lerobot-ros-backup ...` command to retry).

- Enable for a single run with `--push`.
- Enable globally with `LEROBOT_HF_PUSH=1`.
- For training auto-backup to work, point `--output_dir` somewhere under
  `data/` (e.g. `--output_dir=data/<op>/<task>/.../deploy`). Plain upstream
  `lerobot-train` does **not** auto-push; use `lerobot-ros-train` (same flags)
  or run `lerobot-ros-backup <output_dir>` afterwards.

### Restore a deleted folder

```bash
hf download RamaE/lerobot-data-rama-dose_solid \
  --repo-type dataset --local-dir data/rama/dose_solid
```

Restored datasets keep `meta/info.json` etc. intact and work with LeRobot as
before.

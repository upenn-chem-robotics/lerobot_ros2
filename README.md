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

### Prerequisites (not installed by pip)

| Requirement | Why | Notes |
|-------------|-----|-------|
| Python >= 3.10 | | |
| ROS 2 Humble (`rclpy`, `sensor_msgs`, `std_srvs`, `rcl_interfaces`) | Arm/camera I/O in `helper.py`, `record`, `deploy`, `dagger` | Installed as **conda** packages from the `robostack-staging` channel (see [`environment.yml`](environment.yml)) — not apt, not pip. Activating the conda env is what puts `rclpy` on the path. Dataset-only tooling (plateau, action-source, downsample, backup, app) does not need it. |
| `v4l-utils` (`v4l2-ctl`) | Camera control apply/verify, `lerobot-ros-probe-cameras` | System package (`apt install v4l-utils`) |
| Rust toolchain (`cargo`/`rustup`) or Nix | Only to build the OBSBOT Meet SE control CLI via `scripts/setup_obsbot_cli.sh` | Optional; see the OBSBOT section below |
| `huggingface/lerobot` | Core datasets/policies/training; imported by 17 modules | Installed separately and **deliberately not declared** as a dependency so pip never pulls a PyPI version over your editable checkout |

### Reproducible install (recommended)

The [`Dockerfile`](Dockerfile) builds the exact validated stack — conda env, ROS
2, CUDA torch wheels, `lerobot` at its pinned commit, this repo and both policy
plugins:

```bash
docker build -t lerobot-ros2 .
docker run --gpus all -it --rm -v /path/to/data:/lerobot-ros/data lerobot-ros2
```

Datasets are mounted at runtime, not baked in (`data/` is in
[`.dockerignore`](.dockerignore)).

Two lockfiles sit behind it:

| File | Covers | Notes |
|------|--------|-------|
| [`environment.yml`](environment.yml) | 537 conda + 81 pip packages | **Source of truth.** Includes ROS 2 Humble from `robostack-staging`. linux-64 specific (exact build strings). |
| [`constraints.txt`](constraints.txt) | the 81 pip packages only | For pip-only users; holds versions steady on a manual install. |

`environment.yml` is the authoritative one because ROS 2 here is **conda**
packages from the `robostack-staging` channel, not apt and not PyPI — a pip
requirements file structurally cannot describe this environment. Both files
deliberately exclude `lerobot` and this repo's three distributions, which the
Dockerfile layers on top from source.

To reproduce the env without Docker:

```bash
conda env create -f environment.yml
conda activate lerobot
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@d60a700d2b32590ed113d694fd87617e43506081"
pip install --no-deps -e . -e packages/lerobot_policy_strided_diffusion -e packages/lerobot_policy_action_history_diffusion
```

`--no-deps` is deliberate: `environment.yml` already pins the full tree, and
letting pip re-resolve fights the conda-provided packages.

### Manual install

```bash
# 1. lerobot first, pinned to the commit this repo is validated against:
pip install -e "git+https://github.com/huggingface/lerobot.git@d60a700d2b32590ed113d694fd87617e43506081#egg=lerobot"

# 2. This repo, picking the feature set you need (see matrix below):
pip install -e ".[all]" -c constraints.txt

# 3. Policy plugins (each is a standalone distribution, install only what you use):
pip install -e packages/lerobot_policy_strided_diffusion
pip install -e packages/lerobot_policy_action_history_diffusion
```

### What to install for which feature

| I want to... | Install | Adds |
|--------------|---------|------|
| Dataset prep, training and deploy — most CLIs | `pip install -e .` | base only |
| Visualize — `app`, `export`, `plateau-visualize` | `pip install -e ".[viz]"` | `gradio`, `matplotlib` |
| Use the USB foot pedal in `record`/`dagger` | `pip install -e ".[pedal]"` | `evdev` |
| Everything | `pip install -e ".[all]"` | both extras |
| Strided-observation policy | `pip install -e packages/lerobot_policy_strided_diffusion` | separate distribution |
| Action-history policy | `pip install -e packages/lerobot_policy_action_history_diffusion` | separate distribution |

Only two things are genuinely optional. `gradio` is a heavy tree (fastapi,
uvicorn, starlette) needed solely by the visualizers, and `evdev` only by the
foot pedal.

`torch` is **not** an extra, for two reasons: 10 modules import it directly, and
`lerobot` itself requires `torch>=2.7`, so no realistic install lacks it.
Notably `add-action-source*` and `trim-tail` pull torch at import even though
they are dataset-prep tools.

`record`, `deploy` and `dagger` additionally need ROS 2, which pip cannot
install (see Prerequisites above).

### Python dependency reference

Versions are the known-good set from the validated `lerobot` conda env, and are
what [`constraints.txt`](constraints.txt) pins.

Base (always installed):

| Package | Version | Used by |
|---------|---------|---------|
| `numpy` | 2.4.3 | everywhere (20 modules) |
| `opencv-python` (`cv2`) | 4.13.0.92 | camera capture, preview, video export (11 modules) |
| `pyyaml` | 6.0.3 | config loading (9 modules) |
| `pyarrow` | 23.0.1 | parquet dataset read/write (6 modules) |
| `pandas` | 3.0.2 | dataset frames / stats |
| `av` | 17.0.0 | video encode/decode |
| `pillow` (`PIL`) | 11.3.0 | image I/O |
| `huggingface_hub` | 1.15.0 | HF backup + mirror (`hf` CLI comes from here) |
| `torch` | 2.10.0 | deploy, dagger, preprocessing, `training/`, action-source CLIs, action-history plugin |
| `torchvision` | 0.25.0 | image transforms / augmentation preview |

Extras:

| Package | Version | Extra | Used by |
|---------|---------|-------|---------|
| `gradio` | 6.12.0 | `viz` | `lerobot-ros-app` visualizer |
| `matplotlib` | 3.10.8 | `viz` | timeline plots, `app`, `video_exporter` |
| `evdev` | 1.9.3 | `pedal` | USB foot pedal in record/dagger |

Installed transitively via `lerobot`, but version-critical for reproducing
training runs — `constraints.txt` pins these too:

| Package | Version | Why it matters |
|---------|---------|----------------|
| `lerobot` | 0.5.1 (commit `d60a700d`) | policy/dataset/training APIs |
| `diffusers` | 0.35.2 | diffusion policy scheduler/UNet |
| `torchcodec` | 0.10.0 | video decoding in the dataset loader |
| `datasets` | 4.8.4 | HF dataset backend |

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

The hardware CLIs (`record`, `deploy`, `dagger`, `probe-cameras`) require a
teleop/camera config; the dataset-only CLIs do not. Point them at one via either:

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

Record / deploy (needs ROS 2 + hardware):

| Command                | Description                                 |
|------------------------|---------------------------------------------|
| `lerobot-ros-record`   | Record teleoperation datasets               |
| `lerobot-ros-deploy`   | Deploy a trained ACT / Diffusion policy     |
| `lerobot-ros-dagger`   | DAgger-style human-correction recorder      |
| `lerobot-ros-probe-cameras` | Snapshot v4l2 controls per camera     |

Dataset prep (no ROS 2 needed):

| Command                | Description                                 |
|------------------------|---------------------------------------------|
| `lerobot-ros-downsample` | Rewrite + downsample a LeRobot v3 dataset |
| `lerobot-ros-add-action-source` | Add `action_source = 1` to a base teleop dataset so it is schema-compatible with DAgger datasets |
| `lerobot-ros-add-action-source-with-plateau` | Plateau-aware sibling: tags no-motion frames `0` so they are not sampled as anchors |
| `lerobot-ros-trim-tail` | Drop the trailing fraction of frames per episode (stationary "completion" tails) |
| `lerobot-ros-fix-subtasks` | Audit `subtask_index` labels per episode and repair a mislabelled one from subtask boundary times, instead of re-recording it |

Inspect / debug:

| Command                | Description                                 |
|------------------------|---------------------------------------------|
| `lerobot-ros-app`      | Gradio dataset visualizer                   |
| `lerobot-ros-export`   | Export dataset grid videos / timelines      |
| `lerobot-ros-plateau-stats` | Read-only plateau (no-op) frame statistics per dataset / episode |
| `lerobot-ros-plateau-visualize` | Render an MP4 overlaying plateau detection on episode video |
| `lerobot-ros-preview-aug` | Tiled PNG of the exact image-transform pipeline training sees |
| `lerobot-ros-preview-episode` | Render one episode through the exact training-time image pipeline |
| `lerobot-ros-cut`      | Interactive ROI picker; prints a crop spec for `per_camera_crops` |

Train:

| Command                | Description                                 |
|------------------------|---------------------------------------------|
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

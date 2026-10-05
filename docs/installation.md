# Installation

> **Page scope:** Use this page to prepare and verify the host and container environment. For the guided onboarding sequence, continue with [Your first run](first-run.md).

## Prerequisites

Use Linux x86-64 with:

- Docker Engine
- Docker Compose v2
- Git
- sufficient storage for Conda, PyTorch, ROS 2, images, datasets, and checkpoints
- NVIDIA Container Toolkit only when using the GPU service

The supported installation is containerized. Do not combine the release environment with host ROS, a host Conda environment, or a native pip installation.

## Clone and prepare local state

```bash
git clone <repository-url> lerobot-ros2
cd lerobot-ros2
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
export UID="$(id -u)" GID="$(id -g)"
```

`config.local/`, `compose.hardware.yaml`, and runtime data are intentionally excluded from version control.

## Images and services

The Compose file exposes four working modes:

- `tools`: CPU runtime for diagnostics and dataset operations
- `gpu`: runtime with GPU access for training and inference
- `robot`: host-networked runtime extended with explicit hardware mappings
- `dev`: source-mounted development image with test and lint tooling

Build the CPU tools image and run the default preflight:

```bash
docker compose build tools
docker compose run --rm tools
```

Build the development image when modifying source or documentation:

```bash
docker compose build dev
docker compose run --rm dev pytest
```

## Verify the environment

The default tools command checks configuration, imports, installed distributions, and the data directory without probing hardware or the ROS graph. Run the command directly when you need explicit options:

```bash
docker compose run --rm tools lerobot-ros-doctor --help
docker compose run --rm tools lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

For the repository-level checks, use the validation commands in [Release and validation](release-and-validation.md).

## Common installation failures

### Files are owned by root

Export the host identifiers before building or running Compose:

```bash
export UID="$(id -u)" GID="$(id -g)"
```

### GPU is unavailable

Confirm that the NVIDIA driver and NVIDIA Container Toolkit work with Docker before debugging the application. CPU workflows should use `tools`, not `gpu`.

### A dependency check fails

Treat `python -m pip check` as a release failure. Dependencies are intentionally split between Conda and pip to avoid crossing ABI boundaries for NumPy, PyAV, OpenCV, FFmpeg, ROS, and PyTorch.

### Local devices are missing

Do not add `privileged: true`. Create `compose.hardware.yaml` with explicit stable device mappings as described in [Hardware and safety](hardware-and-safety.md).

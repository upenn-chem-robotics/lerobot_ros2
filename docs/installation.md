# Installation

> **Page scope:** This page installs a published release image for normal use. Source builds, tests, and documentation tooling belong in [Development](development.md). Release verification belongs in [Development](development.md#quick-verification-flows).

## Prerequisites

Use Linux x86-64 with Docker Engine, Docker Compose v2, and enough storage for images, datasets, and checkpoints. NVIDIA Container Toolkit is required only for the `gpu` service.

The supported runtime is containerized. Do not combine the release image with host ROS, a host Conda environment, or a native pip installation.

## Obtain the release files

Use a sparse checkout of the released tag so normal users receive the Compose file, supported profiles, generic configuration templates, documentation, license, and notices without populating the application source directories. Replace `<release-tag>` with the tag named in the release notes.

```bash
git clone --filter=blob:none --sparse --no-checkout \
  --branch <release-tag> --single-branch \
  https://github.com/penzottimattia/lerobot_ros2.git lerobot-ros2
cd lerobot-ros2
git sparse-checkout set docs config profiles
git checkout
```

Cone mode includes repository-root files alongside the selected directories. Files such as `compose.yaml`, `LICENSE`, and `THIRD_PARTY_NOTICES.md` remain available, while `src/`, `tests/`, and `packages/` are not populated.

## Choose a configuration path

For the lab bimanual UR3 and Robotiq system, copy the supported profile:

```bash
mkdir -p config.local data
cp profiles/ur_robotiq_bimanual/gello.yaml config.local/gello.yaml
cp profiles/ur_robotiq_bimanual/compose.hardware.yaml compose.hardware.yaml
```

Replace every `REPLACE_*` device path in the copied files with the stable `/dev/v4l/by-id/` and `/dev/input/by-id/` paths for this host before using hardware.

For another robot or site, start from the generic templates:

```bash
mkdir -p config.local data
cp config/gello.example.yaml config.local/gello.yaml
cp config/compose.hardware.example.yaml compose.hardware.yaml
```

The generic templates deliberately contain `REPLACE_*` interface placeholders. Do not mix a profile with unrelated generic topic or service defaults.

## Select the released image

From the sparse checkout:

```bash
export UID="$(id -u)"
export GID="$(id -g)"
export LEROBOT_ROS_IMAGE="<registry>/<namespace>/lerobot-ros2"
export IMAGE_TAG="<release-tag>"
```

Use the image reference and immutable version or commit tag supplied in the release notes. Do not use `latest` as the only identifier.

`config.local/`, `compose.hardware.yaml`, datasets, and outputs are local state. Do not commit or redistribute them with credentials, device inventories, or participant data.

## Pull the runtime

The Compose file exposes three user-facing services:

- `tools`: CPU visualization and dataset operations
- `gpu`: GPU training and inference
- `robot`: host-networked runtime extended with explicit hardware mappings

Pull the service required by your workflow:

```bash
docker compose pull tools
```

Run the software-only preflight:

```bash
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

This does not validate a dataset, checkpoint, GPU, camera, or robot.

For GPU or robot workflows:

```bash
docker compose pull gpu
docker compose pull robot
```

A successful pull completes installation. There is no required diagnostic command afterward. Continue with [Your first run](first-run.md) or go directly to [Workflows](workflows.md).

The Compose file also contains a `dev` service and local build definitions. They are for contributors and maintainers, not required installation steps.

## Common failures

### Compose selects the wrong image

```bash
printf '%s:%s\n' "$LEROBOT_ROS_IMAGE" "$IMAGE_TAG"
```

The printed reference must match the release notes. Then rerun `docker compose pull <service>`.

### Registry authentication fails

Use the authentication method documented by the registry and retry. Do not put registry tokens in `compose.yaml`, shell history, or committed files.

### Files are owned by root

Export `UID` and `GID` before running Compose.

### GPU access fails

Confirm the host NVIDIA driver and NVIDIA Container Toolkit before using the `gpu` service.

### You need to modify code

Switch to [Development](development.md), clone the repository, and use the `dev` target.

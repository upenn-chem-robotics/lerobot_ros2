# Installation

> **Page scope:** This page installs a published release image for normal use. Source builds, editable mounts, tests, and documentation tooling belong in [Development](development.md). Release qualification belongs in [Release and validation](release-and-validation.md).

## Prerequisites

Use Linux x86-64 with Docker Engine, Docker Compose v2, and enough storage for images, datasets, and checkpoints. NVIDIA Container Toolkit is required only for the `gpu` service.

The supported runtime is containerized. Do not combine the release image with host ROS, a host Conda environment, or a native pip installation.

## Obtain the release bundle

Download and unpack the release bundle published with the image. It should contain at least `compose.yaml`, `examples/`, and the user documentation. A source checkout is not required for normal use. If only a source archive is published, use it as the Compose and configuration bundle without building it during onboarding.

## Select the released image

From the unpacked release directory:

```bash
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
export UID="$(id -u)"
export GID="$(id -g)"
export LEROBOT_ROS_IMAGE="<registry>/<namespace>/lerobot-ros2"
export IMAGE_TAG="<release-tag>"
```

Use the image reference and immutable version or commit tag supplied in the release notes. Do not use `latest` as the only identifier.

`config.local/`, `compose.hardware.yaml`, datasets, and outputs are local state. Do not commit or redistribute them with credentials, device inventories, or participant data.

## Pull the runtime

The Compose file exposes three user-facing services:

- `tools`: CPU diagnostics, visualization, and dataset operations
- `gpu`: GPU training and inference
- `robot`: host-networked runtime extended with explicit hardware mappings

Pull only what the current workflow needs:

```bash
docker compose pull tools
```

Later, as required:

```bash
docker compose pull gpu
docker compose pull robot
```

The Compose file also contains a `dev` service and local build definitions. They are for contributors and maintainers, not required installation steps.

## Verify the runtime

```bash
docker compose run --rm tools
```

The default command checks configuration, imports, installed distributions, and the data directory while skipping hardware and ROS-graph probing. To inspect options:

```bash
docker compose run --rm tools lerobot-ros-doctor --help
```

Continue with [Your first run](first-run.md). Repository-wide validation is not part of normal installation.

## Common failures

### Compose selects the wrong image

```bash
printf '%s:%s\n' "$LEROBOT_ROS_IMAGE" "$IMAGE_TAG"
```

The printed reference must match the release notes. Then rerun `docker compose pull tools`.

### Registry authentication fails

Use the authentication method documented by the registry and retry. Do not put registry tokens in `compose.yaml`, shell history, or committed files.

### Files are owned by root

Export `UID` and `GID` before running Compose.

### GPU access fails

Confirm the CPU-only `tools` service works first. GPU workflows additionally require a working host NVIDIA driver and NVIDIA Container Toolkit.

### You need to modify code

Switch to [Development](development.md), clone the repository, and use the `dev` target.

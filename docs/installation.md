# Installation

> **Scope:** Install a published release. Source development and release verification are covered in [Development](development.md).

A **Docker image** is the packaged software that Docker downloads and runs. An image **tag** identifies the published project version. A Compose **service** selects how that image is run for tools, GPU work, or robot access. A **mount** makes a host directory, such as `data/`, visible inside the container.

## Choose your path

- **Dataset inspection or transformation:** complete [Software-only installation](#software-only-installation). Do not create robot configuration.
- **Training:** complete the software-only steps, then [GPU addition](#gpu-addition).
- **Camera or robot use:** complete the software-only steps, then [Hardware addition](#hardware-addition).

## Prerequisites

Use Linux x86-64 with Git, Docker Engine, Docker Compose v2, and enough storage for images, datasets, and checkpoints. NVIDIA Container Toolkit is required only for the `gpu` service.

The supported runtime is containerized. Do not combine the released image with host ROS, a host Conda environment, or a native pip installation.

## Software-only installation

### 1. Obtain the release files

Replace `<release-tag>` with the tag shown in the release notes:

```bash
git clone --filter=blob:none --sparse --no-checkout \
  --branch <release-tag> --single-branch \
  https://github.com/upenn-chem-robotics/lerobot_ros2.git lerobot-ros2
cd lerobot-ros2
git sparse-checkout set docs config profiles
git checkout
```

This checkout supplies `compose.yaml`, documentation, profiles, templates, licenses, and notices. The application itself comes from the released image; this is not an editable source installation.

### 2. Prepare local directories and select the release

```bash
mkdir -p data
export UID="$(id -u)"
export GID="$(id -g)"
export LEROBOT_ROS_IMAGE="ghcr.io/upenn-chem-robotics/lerobot-ros2"
export IMAGE_TAG="v0.1.0"
```

This documentation is pinned to `ghcr.io/upenn-chem-robotics/lerobot-ros2:v0.1.0`. Keep the checked-out Git release and image tag aligned. Avoid `latest`, because its meaning can change.

Check what Docker will run:

```bash
docker compose config --images
```

**Continue when:** every displayed `lerobot-ros2` image uses the expected public image name and release tag.

### 3. Download and check the tools service

```bash
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

**Software preflight passed:** the image downloads, the doctor command exits successfully, and it does not report a failed packaged-application or mount check.

This does not validate a dataset, GPU, camera, ROS graph, checkpoint, or robot.

## GPU addition

Complete the software-only installation first, then download the GPU service:

```bash
docker compose pull gpu
```

Continue with [Train a policy](workflows.md#train-a-policy). A successful tools preflight confirms the basic container path, not GPU access.

## Hardware addition

Complete the software-only installation first. Use this path only for cameras or the documented robot integration.

For the supported lab system:

```bash
mkdir -p config.local
cp profiles/ur_robotiq_bimanual/gello.yaml config.local/gello.yaml
cp profiles/ur_robotiq_bimanual/compose.hardware.yaml compose.hardware.yaml
docker compose pull robot
```

Use this profile only for the documented two-UR3, two-Robotiq, two-GELLO integration. For another robot or site, start from the generic templates and treat the result as a separately reviewed integration:

```bash
mkdir -p config.local
cp config/gello.example.yaml config.local/gello.yaml
cp config/compose.hardware.example.yaml compose.hardware.yaml
```

Do not run a hardware workflow until every `REPLACE_*` value has been resolved and checked under [Configuration](configuration.md).

## When installation does not work

Released images are public and the release checkout and image tag are intended to match. Most version and dependency problems should therefore be corrected by returning to the released path rather than repairing a running container.

### Docker uses the wrong project version

**What you see:** `docker compose config --images` shows a different tag from the checked-out release, or the documented command is missing.

**Fix:** set `IMAGE_TAG` to the checkout's release tag, run `docker compose config --images` again, and pull the required service. Do not edit files inside the container.

### Docker cannot download the image

**What you see:** `docker compose pull tools` reports that the image or tag cannot be found.

**Check:** compare the public image name and tag with the release notes and the checked-out Git tag. Also confirm that Docker can access the network.

**Fix:** correct the image name or tag and retry. If the exact released public image is unavailable, stop and report the release issue rather than substituting another image.

### You cannot edit files created by Docker

**What you see:** outputs exist on the host but your normal user cannot edit or delete them.

**Fix:** set the host identity before running Compose:

```bash
export UID="$(id -u)"
export GID="$(id -g)"
```

Create a new test output before changing ownership of existing datasets.

### Docker cannot use the GPU

**What you see:** the `gpu` service fails to start or training reports no NVIDIA GPU.

**Check:** run the software-only preflight first. If it passes, the basic image and mounts work and the remaining issue is GPU access.

**Fix:** verify the host NVIDIA driver and NVIDIA Container Toolkit, then retry the released `gpu` service. Do not modify the image to work around a host GPU configuration problem.

### Source edits do not change the released container

The released image contains an installed package and does not run from the sparse checkout. Use the full-clone `dev` workflow in [Development](development.md) for source changes.

# Your first run

> **Page scope:** This is the normal-user onboarding path for a published image. It deliberately avoids source builds, tests, linting, security scans, and release verification.

This tutorial starts from an unpacked release bundle and ends with a successful software-only diagnostic. It does not command a robot.

## 1. Check Docker

```bash
docker --version
docker compose version
```

Both commands must succeed. NVIDIA Container Toolkit is not needed for this first run.

## 2. Prepare local state

```bash
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
export UID="$(id -u)"
export GID="$(id -g)"
```

The copied YAML is only a starting template. Do not connect hardware while placeholder values remain.

## 3. Select and pull the release image

Use the image reference and immutable tag supplied with the release:

```bash
export LEROBOT_ROS_IMAGE="<registry>/<namespace>/lerobot-ros2"
export IMAGE_TAG="<release-tag>"
docker compose pull tools
```

**Expected result:** Compose downloads the published image rather than compiling project source.

## 4. Run the software-only diagnostic

```bash
docker compose run --rm tools
```

The default command runs `lerobot-ros-doctor` with hardware and ROS-graph probing disabled.

**Expected result:** configuration, imports, installed distributions, and the data directory pass their checks.

If it fails, retain the first meaningful error. Check the image reference, registry access, Docker permissions, disk space, and local directory ownership before changing project code.

## 5. Inspect available commands

```bash
docker compose run --rm tools lerobot-ros-doctor --help
docker compose run --rm tools lerobot-ros-app --help
```

The normal installation is complete. Choose a task in [Workflows](workflows.md).

## 6. Stop before hardware

Before any physical workflow:

1. Read [Hardware and safety](hardware-and-safety.md).
2. Replace every `REPLACE_*` value.
3. Create `compose.hardware.yaml` from the supplied example.
4. Use stable `/dev/v4l/by-id` and `/dev/input/by-id` paths.
5. Verify workspace clearance, limits, and emergency-stop access.
6. Pull the same released tag for `robot` and run the full preflight without enabling actuation.

```bash
docker compose pull robot
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-doctor
```

Do not continue if a required check fails, a device path is ambiguous, the expected ROS graph differs, limits are unverified, or the emergency stop is unavailable.

## Advanced paths

To modify source, build images, run tests or linting, or edit documentation, use [Development](development.md). To qualify an artifact for publication, use [Release and validation](release-and-validation.md). Neither path is part of normal onboarding.

## Troubleshooting

### Compose builds instead of pulling

Run `docker compose pull <service>` before `docker compose run`. Confirm `LEROBOT_ROS_IMAGE` and `IMAGE_TAG` identify a published artifact.

### Docker permission denied

Confirm your account can run `docker info`. Follow your organization's Docker access policy instead of running the whole workflow as `root`.

### Configuration contains placeholders

```bash
grep -R "REPLACE_" config.local compose.hardware.yaml 2>/dev/null
```

Replace every result before hardware use.

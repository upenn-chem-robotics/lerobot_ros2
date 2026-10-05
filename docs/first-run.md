# Your first run

> **Page scope:** This is the linear onboarding tutorial. For prerequisite details and installation troubleshooting, use [Installation](installation.md); for later command recipes, use [Workflows](workflows.md).

This tutorial takes you from a fresh clone to a successful software-only diagnostic. It intentionally stops before commanding a robot. When complete, you will know that Docker can build the supported environment and that the project CLI starts correctly.

## 1. Understand the pieces

You will work with three local directories or files:

- `config.local/`: your machine-specific configuration;
- `data/`: datasets and other local data;
- `compose.hardware.yaml`: private device mappings for your workstation.

These are local inputs, not portable project defaults. Do not commit device inventories, credentials, datasets, or checkpoints.

## 2. Check the prerequisites

The supported release path is Linux x86-64 with Docker Engine and Docker Compose v2. Run:

```bash
docker --version
docker compose version
```

Both commands must complete successfully. GPU training additionally requires the NVIDIA Container Toolkit, but it is not needed for this first run.

## 3. Clone and create local directories

```bash
git clone <repository-url> lerobot-ros2
cd lerobot-ros2
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
```

The copied YAML is a starting template. Do not connect hardware using placeholder values.

## 4. Set your container user IDs

```bash
export UID="$(id -u)"
export GID="$(id -g)"
```

This lets container-created files use your host user and group rather than `root`.

## 5. Build the tools image

```bash
docker compose build tools
```

**Expected result:** Docker completes the image build without an error. If the build fails, keep the first meaningful error line and check network access, disk space, and Docker permissions before changing project code.

## 6. Run the safe, software-only diagnostic

```bash
docker compose run --rm tools
```

The default command runs `lerobot-ros-doctor` without hardware or ROS graph probes.

**Expected result:** the diagnostic prints checks for the packaged runtime and configuration. It may report that no hardware devices are declared. That is acceptable at this stage.

**Stop here if:** the command cannot import the project, cannot read the configuration, or exits with a required check failing. Fix this before adding hardware because hardware errors would otherwise hide the basic runtime problem.

## 7. Learn the command pattern

Every workflow has a CLI entry point. Ask a command for its current options instead of copying flags from an old experiment:

```bash
docker compose run --rm tools lerobot-ros-export --help
docker compose run --rm gpu lerobot-ros-train --help
```

For hardware commands, use both Compose files:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-record --help
```

This only displays help, but Docker still evaluates the hardware override. Create the override before using this form.

## 8. Prepare for hardware, without moving it

1. Replace every `REPLACE_*` value in your local configuration.
2. Create `compose.hardware.yaml` from the repository example.
3. Use stable `/dev/v4l/by-id` and `/dev/input/by-id` paths.
4. Verify workspace clearance, limits, and emergency-stop access.
5. Run the full preflight:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-doctor
```

**Expected result:** declared devices are present, the configuration loads, and required checks pass.

**Do not continue to actuation if:** a required check fails, a device path is ambiguous, the expected ROS graph differs, limits are unverified, or the emergency stop is unavailable.

## 9. Pick the next tutorial path

- To understand the available workflows, read [What you can do](capabilities.md).
- To bring up hardware in controlled stages, read [Hardware and safety](hardware-and-safety.md).
- To record, transform, train, deploy, or run DAgger, use [Workflows](workflows.md).
- To understand mounts and local overrides, use [Configuration](configuration.md).

## Troubleshooting this tutorial

### Docker permission denied

Confirm that your account can run `docker info`. Follow your organization’s Docker access policy rather than running the whole workflow as `root`.

### Files are owned by root

Start a new shell, export `UID` and `GID` again, and rerun the relevant container command.

### Configuration still contains placeholders

Search only your local configuration:

```bash
grep -R "REPLACE_" config.local compose.hardware.yaml 2>/dev/null
```

Replace every result before hardware use.

### A device disappears between reboots

Do not use changing names such as `/dev/video0` when a stable `/dev/v4l/by-id/...` path is available. Update the local configuration and hardware override together.

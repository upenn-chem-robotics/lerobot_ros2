# lerobot-ros2

Docker-first research tooling for ROS 2 bimanual teleoperation, LeRobot dataset recording and transformation, policy training, and deployment. ROS 2 Humble comes from RoboStack through Conda inside Docker. Host ROS, host Conda, and native pip installation are not supported release paths.

> **Physical safety:** this software can command real robots. Validate topics, services, limits, workspace clearance, emergency stops, and checkpoints without motion before actuation. This project is not safety-rated and provides no real-time guarantee.

## Supported platform

- Linux x86-64, Docker Engine, and Compose v2
- Optional NVIDIA Container Toolkit for GPU training and inference
- Explicit `/dev/v4l/by-id` and `/dev/input/by-id` mappings
- Wheel-installed runtime; editable source only in the `dev` target

macOS, Windows, Docker Desktop USB forwarding, host ROS integration, and arbitrary camera firmware are best-effort and outside the supported contract.

## Quick start

```bash
git clone <repository-url> lerobot-ros2
cd lerobot-ros2
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
export UID="$(id -u)" GID="$(id -g)"
docker compose build tools
docker compose run --rm tools
```

The final command runs `lerobot-ros-doctor` without hardware or ROS graph probes. Replace all `REPLACE_*` values, create `compose.hardware.yaml` from the example, then run the full preflight:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot lerobot-ros-doctor
```

## Workflows

```bash
# Dataset operations
docker compose run --rm tools lerobot-ros-export --help

# Recording with hardware and host ROS networking
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot lerobot-ros-record --help

# GPU training and deployment
docker compose run --rm gpu lerobot-ros-train --help
```

## Documentation

- [Getting started](docs/getting-started.md)
- [Docker runtime](docs/docker.md)
- [Configuration](docs/configuration.md)
- [Hardware](docs/hardware.md)
- [Recording](docs/recording.md)
- [Deployment](docs/deployment.md)
- [Training](docs/training.md)
- [Dataset operations](docs/datasets.md)
- [Policy plugins](docs/policy-plugins.md)
- [Backup](docs/backup.md)
- [Development](docs/development.md)
- [Architecture](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Release process](docs/release-process.md)

## Status, contribution, and security

Research software. Public APIs and configuration may change before 1.0. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [LICENSE](LICENSE).

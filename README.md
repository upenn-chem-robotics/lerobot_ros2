# lerobot-ros2

Docker-first research tooling for ROS 2 bimanual teleoperation, LeRobot dataset recording and transformation, policy training, and deployment. ROS 2 Humble comes from the project containers rather than the host.

> [!IMPORTANT]
> **Independent project.** This repository is not affiliated with, endorsed by, sponsored by, or officially connected with Hugging Face or the LeRobot project. LeRobot and Hugging Face are names used to identify the upstream projects and services with which this software interoperates.

> [!WARNING]
> This software can command real robots. Validate topics, services, limits, workspace clearance, emergency stops, and checkpoints without motion before actuation. This project is not safety-rated and provides no real-time guarantee.

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

This is a software-only diagnostic. Before using hardware, replace every `REPLACE_*` value, create `compose.hardware.yaml` from the example, and follow the safety checks in the documentation.

## Documentation

The documentation has one recommended path:

1. [Overview](docs/index.md)
2. [Concepts for first-time users](docs/concepts.md)
3. [Installation](docs/installation.md)
4. [Your first run](docs/first-run.md)
5. [Choose a capability](docs/capabilities.md)
6. [Run a workflow](docs/workflows.md)
7. [Hardware and safety](docs/hardware-and-safety.md)
8. [Configuration reference](docs/configuration.md)
9. [Development](docs/development.md)
10. [Release and validation](docs/release-and-validation.md)

Use **Choose a capability** when deciding what to do and **Run a workflow** when you need commands. Installation owns prerequisites and image setup; Your first run is the guided onboarding sequence.

## Status, contribution, and security

Research software. Public APIs and configuration may change before 1.0. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [LICENSE](LICENSE).

## License and third-party software

This project is licensed under the [Apache License 2.0](LICENSE). Third-party software, including LeRobot, remains subject to its own license and notices. See [Third-party notices](THIRD_PARTY_NOTICES.md) for dependency and attribution details.

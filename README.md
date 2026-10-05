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

Check out the user-facing files from the released tag without materializing the source tree. Replace `<release-tag>` with the tag named in the release notes.

```bash
git clone --filter=blob:none --sparse --no-checkout \
  --branch <release-tag> --single-branch \
  https://github.com/penzottimattia/lerobot_ros2.git lerobot-ros2
cd lerobot-ros2
git sparse-checkout set docs config profiles
git checkout

mkdir -p config.local data
cp profiles/ur_robotiq_bimanual/gello.yaml config.local/gello.yaml
cp profiles/ur_robotiq_bimanual/compose.hardware.yaml compose.hardware.yaml
export UID="$(id -u)" GID="$(id -g)"
export LEROBOT_ROS_IMAGE="<registry>/<namespace>/lerobot-ros2"
export IMAGE_TAG="<release-tag>"
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

Cone-mode sparse checkout includes the selected profile and template directories and repository-root files such as `compose.yaml`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`. Application source directories are not populated. The image coordinates still come from the release notes.

The final command is a **software-only preflight**. It checks the packaged application without accessing cameras, the ROS graph, or a robot; it does not validate a dataset, checkpoint, GPU, or physical system. Continue to [Start here](docs/index.md) and choose the path that matches the inputs you already have. Hardware workflows still require completed local configuration and the checks in [Hardware and safety](docs/hardware-and-safety.md).

Building from source is intentionally not part of onboarding. Contributors, advanced users, and maintainers should clone the repository and follow [Development](docs/development.md).

## Choose your path

- **Using the lab bimanual robot:** copy the [UR Robotiq profile](docs/robot-integration.md), verify the released robot runtime, then follow [Hardware and safety](docs/hardware-and-safety.md).
- **Adapting another robot:** start from the generic templates under `config/` and document its ROS interface before recording.
- **Existing LeRobot dataset:** follow [Visualize your first dataset](docs/first-run.md), then [Choose and run a workflow](docs/workflows.md).
- **Training or deployment:** use [Choose and run a workflow](docs/workflows.md) to check prerequisites and run the command in one place.
- **Contributing or releasing:** use [Development](docs/development.md), including its quick verification flows.

## Status, contribution, and security

Research software. Public APIs and configuration may change before 1.0. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [LICENSE](LICENSE).

## License and third-party software

This project is licensed under the [Apache License 2.0](LICENSE). Third-party software, including LeRobot, remains subject to its own license and notices. See [Third-party notices](THIRD_PARTY_NOTICES.md) for dependency and attribution details.

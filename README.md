# lerobot-ros2

Docker-first research tooling for ROS 2 bimanual teleoperation, LeRobot dataset recording and transformation, policy training, deployment, and DAgger workflows. ROS 2 Humble is supplied by the project containers rather than the host.

> [!IMPORTANT]
> **Independent project.** This repository is not affiliated with, endorsed by, sponsored by, or officially connected with Hugging Face or the LeRobot project. LeRobot and Hugging Face are names used to identify upstream projects and services with which this software interoperates.

> [!WARNING]
> This software can command real robots. Validate topics, services, limits, workspace clearance, emergency stops, and checkpoints without motion before actuation. This project is not safety-rated and provides no real-time guarantee.

## Start here

- **Existing dataset:** [install a released version](docs/installation.md), then [visualize the dataset](docs/first-run.md).
- **Training or dataset processing:** choose a command in [Choose and run a workflow](docs/workflows.md).
- **Supported lab robot:** review the [reference robot integration](docs/robot-integration.md), then complete [Hardware and safety](docs/hardware-and-safety.md).
- **Another robot or site:** start from the generic templates under `config/` and define a separately reviewed ROS, device, dataset, and safety contract.
- **Contributing:** use the Docker development target and follow the [Contributor guide](docs/development.md).

The [documentation overview](docs/index.md) helps you choose a workflow. Platform requirements and release support details are in [Installation](docs/installation.md).

## Minimal released-runtime check

A published release consists of a matching Git tag and container-image tag. The locally built development documentation instead uses repository branch `devel` with image tag `latest`. Follow [Install a released version](docs/installation.md) for the canonical installation procedure and image coordinates from the release notes. After installation, run the software-only preflight:

```bash
mkdir -p data config.local .cache/lerobot/{huggingface,torch}
cp config/gello.example.yaml config.local/gello.yaml
cat > .env <<EOF
LEROBOT_HOST_UID=$(id -u)
LEROBOT_HOST_GID=$(id -g)
EOF
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

This verifies the packaged application without accessing cameras, the ROS graph, or a robot. It does not validate a dataset, checkpoint, GPU, or physical system.

## Project status

Research software. Public APIs and configuration may change before 1.0. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [LICENSE](LICENSE).

The project is licensed under the [Apache License 2.0](LICENSE). Third-party software, including LeRobot, remains subject to its own license and notices. See [Third-party notices](THIRD_PARTY_NOTICES.md).

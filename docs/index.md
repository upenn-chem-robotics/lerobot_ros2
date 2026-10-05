# Overview

`lerobot_ros2` connects the stages of a robot-learning experiment: inspect sensors, record demonstrations, prepare data, train a policy, validate a checkpoint, deploy it, and collect corrective data.

Dataset inspection, transformation, and training are software workflows that can be used without the reference robot. Hardware workflows are documented against the supported `ur_robotiq_bimanual` profile; another robot or site requires its own reviewed interface, configuration, and acceptance work.

!!! important "Independent project"
    This project is not affiliated with, endorsed by, sponsored by, or officially connected with Hugging Face or the LeRobot project.

!!! warning "Physical safety"
    This software can command real robots. It is not safety-rated and provides no real-time guarantee. Follow [Hardware and safety](hardware-and-safety.md) before enabling cameras or actuation.

## Choose your path

Start with the path that matches what you need to do. Software-only work comes first; hardware-capable paths explicitly introduce the integration and safety prerequisites before recording or deployment.

### Inspect or transform an existing dataset

1. Complete [Install a released version](installation.md) and the software-only preflight.
2. Place the dataset below the host `data/` directory.
3. [Visualize an existing dataset](first-run.md).
4. Select a non-destructive operation in [Choose and run a workflow](workflows.md).

This path does not require robot hardware. Keep the source dataset unchanged and write repairs or transformations to a new destination.

### Train a policy

Complete [Install a released version](installation.md), verify that the dataset and policy configuration are compatible, and follow [Train a policy](workflows.md#train-a-policy). The documented GPU path requires NVIDIA Container Toolkit. Training does not require robot hardware.

### Prepare the supported bimanual system

[Reference robot integration](robot-integration.md) documents the UR3, Robotiq 2F-85, GELLO, camera, and foot-pedal profile. Review the [Configuration reference](configuration.md), then complete [Hardware and safety](hardware-and-safety.md) before recording, deployment, or DAgger.

### Record new demonstrations

1. Read [Robot-learning fundamentals](concepts.md).
2. Complete [Install a released version](installation.md) and its software-only preflight.
3. Copy and verify the released profile using [Reference robot integration](robot-integration.md).
4. Complete [Hardware and safety](hardware-and-safety.md).
5. Make the [minimum required configuration edits](configuration.md#minimum-required-edits).
6. [Record demonstrations](workflows.md#record-demonstrations).
7. Inspect the result with [Visualize an existing dataset](first-run.md).

!!! warning "Hardware workflow"
    Recording can access cameras, operator inputs, and robot hardware. Complete [Hardware and safety](hardware-and-safety.md) before recording.

### Validate or deploy a checkpoint

[Deploy a checkpoint](workflows.md#deploy-a-checkpoint) requires a compatible checkpoint, completed hardware configuration, the verified [robot interface](robot-integration.md), and controlled no-motion checks. Do not treat successful checkpoint loading as permission to actuate.

### Contribute or publish a release

[Contributor guide](development.md) covers source changes, documentation changes, tests, and releases. The release runtime is wheel-installed; editable source belongs in the `dev` target.

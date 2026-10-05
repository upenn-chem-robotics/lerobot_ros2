# Start here

`lerobot_ros2` connects the stages of a robot-learning experiment: inspect sensors, record demonstrations, prepare data, train a policy, validate a checkpoint, deploy it, and collect corrective data.

!!! important "Independent project"
    This project is not affiliated with, endorsed by, sponsored by, or officially connected with Hugging Face or the LeRobot project.

!!! warning "Physical safety"
    This software can command real robots. It is not safety-rated and provides no real-time guarantee. Follow [Hardware and safety](hardware-and-safety.md) before enabling cameras or actuation.

## Choose your path

### I am new and do not have a dataset

1. Read [Concepts](concepts.md).
2. Complete [Installation](installation.md) and its software-only preflight.
3. Copy and verify the released robot runtime profile with [Robot integration](robot-integration.md).
4. Read [Hardware and safety](hardware-and-safety.md).
5. Make the [minimum required configuration edits](configuration.md#minimum-required-edits).
6. [Record demonstrations](workflows.md#record-demonstrations).
7. Inspect the result with [Visualize your first dataset](first-run.md).

Recording is a hardware workflow. There is no bundled starter dataset and onboarding does not pretend that one exists.

### I already have a LeRobot dataset

1. Complete [Installation](installation.md).
2. Put the dataset below the host `data/` directory.
3. Follow [Visualize your first dataset](first-run.md).
4. Continue to [Choose and run a workflow](workflows.md).

### I want to train or deploy

Use [Choose and run a workflow](workflows.md) to select the task, confirm its prerequisites, and run the command. Training requires a compatible dataset; the documented GPU path requires NVIDIA Container Toolkit. Deployment additionally requires a compatible checkpoint, completed hardware configuration, the verified [robot interface](robot-integration.md), and controlled no-motion checks.


### I want to change code or publish a release

Use [Development](development.md), including its quick verification flows. This page is not part of normal-user onboarding.

## Supported platform

- Linux x86-64, Docker Engine, and Compose v2
- Optional NVIDIA Container Toolkit for GPU training and inference
- Explicit `/dev/v4l/by-id` and `/dev/input/by-id` mappings for hardware

macOS, Windows, Docker Desktop USB forwarding, host ROS integration, and arbitrary camera firmware are best-effort and outside the supported contract.

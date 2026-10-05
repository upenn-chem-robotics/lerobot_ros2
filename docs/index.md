# Start here

`lerobot_ros2` connects the stages of a robot-learning experiment: inspect data, record demonstrations, train a policy, validate a checkpoint, deploy it, and collect corrective data.

Choose the path that matches what you have now. Dataset inspection, transformation, and training do not require robot hardware. Recording, deployment, and DAgger can access physical devices and require the supported integration and local acceptance procedures.

!!! important "Independent project"
    This project is not affiliated with, endorsed by, sponsored by, or officially connected with Hugging Face or the LeRobot project.

!!! warning "Physical safety"
    This software can command real robots. It is not safety-rated and provides no real-time guarantee. Complete [Hardware and safety](hardware-and-safety.md) before enabling cameras or actuation.

## Readiness gates

The workflows use the following gates. Complete only those required by your path:

1. **Software preflight passed:** the released container starts and the project mounts are available.
2. **Dataset inspection passed:** episodes, cameras, timestamps, observations, and actions have been reviewed.
3. **Hardware configuration resolved:** every local device placeholder has been replaced and checked.
4. **ROS interface verified:** the expected state, command, and service interfaces match the supported profile.
5. **Checkpoint compatibility verified:** the checkpoint and current observation/action configuration agree without motion.
6. **Physical acceptance completed:** the site's controlled hardware and emergency-stop checks are complete.

A later gate never replaces an earlier one. In particular, successful software or checkpoint loading is not permission to actuate.

## I have an existing dataset

**Required:** a published release and a LeRobot dataset on the workstation.

1. Follow the **software-only** path in [Installation](installation.md#software-only-installation).
2. Place the dataset below the host `data/` directory.
3. [Visualize and inspect the dataset](first-run.md).
4. If a specific mismatch must be corrected, choose a non-destructive operation in [Dataset workflows](workflows.md#dataset-workflows).

**Finish:** the source remains unchanged and either passes inspection or has a separately inspected derived copy.

## I want to train a policy

**Required:** an inspected dataset and a working NVIDIA container runtime.

1. Complete the dataset path above.
2. Follow the GPU additions in [Installation](installation.md#gpu-addition).
3. Complete [Train a policy](workflows.md#train-a-policy), including the loading check before the full run.
4. Reload and record the resulting checkpoint before considering deployment.

**Finish:** a checkpoint with recorded dataset, configuration, release image, and output location. Training success does not make it deployable.

## I want to record demonstrations

**Required:** the supported bimanual system, an authorized operator or integrator, and a controlled workspace.

1. Read the recording sections on [observations and actions](concepts.md#observation-what-the-robot-can-currently-sense), [episodes and datasets](concepts.md#demonstration-episode-and-dataset), [teleoperation](concepts.md#teleoperation-inference-and-deployment), and [camera probing](concepts.md#camera-probing).
2. Complete the software-only installation and preflight.
3. Read [Hardware and safety](hardware-and-safety.md) through the no-motion boundary.
4. Choose the operator or commissioning route in [Robot integration](robot-integration.md#choose-your-role).
5. Complete [Minimum required configuration](configuration.md#minimum-required-edits) and probe the cameras.
6. Pass the recording readiness gate in [Record demonstrations](workflows.md#record-demonstrations).
7. Record, then inspect the result with [Visualize a dataset](first-run.md).

**Finish:** an inspected dataset whose camera roles, state/action fields, and episode boundaries are understood.

## I want to validate or deploy a checkpoint

Validation and deployment are separate steps:

1. [Validate checkpoint compatibility without motion](workflows.md#validate-a-checkpoint-without-motion).
2. Complete the ROS interface and physical acceptance gates.
3. Only then follow [Deploy a validated checkpoint](workflows.md#deploy-a-validated-checkpoint).

**Finish:** either a documented incompatibility that blocks deployment, or a validated checkpoint used under the site's controlled procedure.

## I want to commission or operate the supported bimanual system

Use [Robot integration](robot-integration.md). It separates ordinary operation of an already commissioned system from robot-side commissioning. Complete [Hardware and safety](hardware-and-safety.md) before recording, deployment, or DAgger.

## I want to develop the project or publish a release

Use [Development](development.md). A release checkout runs installed packages; editable source belongs in the `dev` service.

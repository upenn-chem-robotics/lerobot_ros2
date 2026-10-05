# Hardware and safety

> **Scope:** Physical-device access, controlled bring-up, and acceptance checks. Complete these checks before running any workflow that can command hardware.

!!! danger "Not safety-rated"
    This project is research software. It is not safety-rated, does not provide a real-time guarantee, and cannot replace a risk assessment, guarding, emergency-stop system, or trained operator.

## Hardware access contract

The base Compose file does not grant broad device access. Site-specific access belongs in the ignored `compose.hardware.yaml` file and should map only the required stable device paths. Do not use `privileged: true`.

Use:

- `/dev/v4l/by-id/...` for cameras
- `/dev/input/by-id/...` for pedals and operator inputs
- explicit network configuration for robot controllers
- host networking only for the service that requires the ROS graph

## Controlled bring-up

This page defines the safety order; Robot integration supplies the reference-system commands. Complete the no-motion stages before following any instruction that enables commands.

Complete these stages in order:

1. **Physical preparation:** clear the workspace, inspect fixtures and cables, establish exclusion zones, and confirm emergency stops.
2. **Configuration review:** verify robot addresses, namespaces, topics, services, joint order, limits, device mappings, and data paths.
3. **No-motion diagnostics:** run `lerobot-ros-doctor`, inspect the ROS graph, and validate observations and operator inputs without enabling actuation.
4. **Checkpoint review:** confirm policy class, observation schema, action dimensions, normalization metadata, and expected control frequency.
5. **No-motion gate:** record that device identity, current state, left/right association, checkpoint compatibility, and stop controls have passed without command publication.
6. **Limited actuation:** use controlled speed and a prepared stop path; begin with the smallest practical motion.
7. **Recording and shutdown:** finalize recordings, stop command publication, stop containers, and then power down using the local hardware procedure.

## Full preflight

```bash
docker compose -f compose.yaml -f compose.hardware.yaml config
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-doctor
```

!!! warning "Preflight boundary"
    A successful software preflight does not establish readiness for safe actuation. Continue only when the local configuration contains no unresolved placeholders and the doctor output matches the intended devices and ROS graph.

## Hardware acceptance record

For each supported setup, identify the released application image, robot-runtime image, profile or configuration revision, and checkpoint when one is being deployed. Then record:

- camera enumeration and stream validation
- pedal or operator-input behavior
- ROS publisher, subscriber, service, and namespace checks
- no-motion command-boundary checks
- arm and gripper direction, scaling, and limits
- emergency-stop response
- recording finalization
- orderly shutdown

Hardware acceptance is not covered by the automated hardware-free suite. Repeat acceptance after changing controllers, firmware, devices, calibration, networking, policy checkpoints, or safety configuration.

## Controlled bimanual bring-up

Complete these checks for the `ur_robotiq` stack before recording, deployment, or DAgger:

1. Start with `mode:=full_mock` and verify the interface contract in [Robot integration](robot-integration.md).
2. Confirm the left and right robot IP assignments at the site without committing them.
3. Confirm that left and right UR custom ports differ.
4. Confirm that left and right ToolComm TCP ports differ.
5. Confirm the selected calibration files and robot base poses.
6. Keep `run_setup_node:=false` during the first real-hardware graph check. The setup node can operate dashboard services, including power, brake, program-load, and play operations.
7. Keep both GELLO offset nodes in `control_mode=0` while inspecting state topics and controller status.
8. Confirm that the left state topic follows only the left arm and the right state topic follows only the right arm.
9. Confirm controller and hardware-component state with `ros2 control list_controllers` and `ros2 control list_hardware_components`.
10. Clear the workspace, verify limits and emergency-stop access, and appoint one operator to control the transition.
11. Enter `control_mode=1`, wait for both transition services, and observe the first commanded motion at a conservative site-approved condition.
12. Return both offset nodes to mode `0` before changing GELLO, controller, calibration, gripper, or topic parameters.

Modes `2` and `3` are specialized single-joint speed modes: mode `2` rotates the selected joint in the positive direction and mode `3` in the negative direction. They may be used for validated tasks requiring sustained rotation, including screwing. Before entering either mode, verify the selected joint, trigger, direction, maximum velocity, tool alignment, joint limits, workspace clearance, and stop path. Return to mode `0` before changing any of those settings.

## Robot-stack shutdown

Return both GELLO offset nodes to idle before stopping containers. Finalize dataset or deployment outputs, stop `lerobot_ros2`, stop the robot launch, and follow the site's robot program, brake, and power procedure. A container exit is not a substitute for a verified physical stop. If command publication cannot be confirmed stopped, use the site's physical stop procedure rather than relying on container state.

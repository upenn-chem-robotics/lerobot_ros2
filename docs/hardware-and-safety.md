# Hardware and safety

> **Page scope:** This page owns physical-device access, controlled bring-up, and acceptance checks. Complete it before running any workflow that can command hardware.

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

Complete these stages in order:

1. **Physical preparation:** clear the workspace, inspect fixtures and cables, establish exclusion zones, and confirm emergency stops.
2. **Configuration review:** verify robot addresses, namespaces, topics, services, joint order, limits, device mappings, and data paths.
3. **No-motion diagnostics:** run `lerobot-ros-doctor`, inspect the ROS graph, and validate observations and operator inputs without enabling actuation.
4. **Checkpoint review:** confirm policy class, observation schema, action dimensions, normalization metadata, and expected control frequency.
5. **Limited actuation:** use controlled speed and a prepared stop path; begin with the smallest practical motion.
6. **Recording and shutdown:** finalize recordings, stop command publication, stop containers, and then power down using the local hardware procedure.

## Full preflight

```bash
docker compose -f compose.yaml -f compose.hardware.yaml config
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-doctor
```

A successful software preflight is necessary but not sufficient for safe actuation.

## Hardware acceptance record

For each supported setup, record:

- camera enumeration and stream validation
- pedal or operator-input behavior
- ROS publisher, subscriber, service, and namespace checks
- no-motion command-boundary checks
- arm and gripper direction, scaling, and limits
- emergency-stop response
- recording finalization
- orderly shutdown

Hardware acceptance is not covered by the automated hardware-free suite. Repeat acceptance after changing controllers, firmware, devices, calibration, networking, policy checkpoints, or safety configuration.

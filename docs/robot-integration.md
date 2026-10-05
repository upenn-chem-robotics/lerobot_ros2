# Robot integration

## Supported reference integration

The supported reference integration connects `lerobot_ros2` to the released ROS 2 Humble `ur_robotiq` runtime through the `ur_robotiq_bimanual` profile. It is the hardware configuration used to develop and validate this project:

- two UR3 manipulators;
- two Robotiq 2F-85 grippers;
- two GELLO devices;
- the cameras and foot pedal mapped in the local hardware overlay.

The robot code remains an external runtime dependency. Operators start the published image and use its public ROS interfaces. They do not edit the robot repository, installed Python files, launch files, or controller configuration as part of a normal `lerobot_ros2` workflow.

!!! note "Reference hardware and independence"
    This page documents the hardware used by the project and the integration that the project currently supports as its reference implementation. The project is not affiliated with, endorsed by, sponsored by, or officially connected with Universal Robots, Robotiq, Hugging Face, or the LeRobot project. Product and project names are used only to identify the hardware and upstream software with which this project interoperates.

!!! danger "Actuation boundary"
    The robot runtime can power robots, release brakes, start controllers, and publish commands. Complete [Hardware and safety](hardware-and-safety.md), keep both GELLO offset nodes in `control_mode=0` during checks, and follow the site's robot operating procedure.

Generic templates remain available for adaptation to other robots and sites, but those configurations are not presented as supported integrations.

## What is committed

The profile separates reusable integration decisions from machine-local state:

| File | Purpose |
|---|---|
| `profiles/ur_robotiq_bimanual/gello.yaml` | Ready-to-copy `lerobot_ros2` topics, services, recording semantics, camera roles, and camera defaults. |
| `profiles/ur_robotiq_bimanual/compose.hardware.yaml` | Hardware overlay with explicit placeholders for host device identities. |
| `profiles/ur_robotiq_bimanual/compatibility.yaml` | Expected robot image, ROS distribution, and ROS interface contract. |
| `config/gello.example.yaml` | Generic annotated template for other robots and sites. |
| `config/compose.hardware.example.yaml` | Generic hardware-access template. |

The profile intentionally does not commit robot IP addresses, calibration files, physical device identifiers, GELLO serial identifiers, or site network details.

## 1. Copy the supported profile

```bash
mkdir -p config.local data
cp profiles/ur_robotiq_bimanual/gello.yaml config.local/gello.yaml
cp profiles/ur_robotiq_bimanual/compose.hardware.yaml compose.hardware.yaml
```

Replace machine-local device placeholders, then check that none remain:

```bash
grep -R "REPLACE_" config.local/gello.yaml compose.hardware.yaml
```

No output is expected before hardware access.

For a different robot or site, start from the generic files under `config/` instead of this profile.

## 2. Prepare the robot hardware

The robot-side setup follows the same upstream procedures referenced by the `ur_robotiq` project. Complete them for both robots before starting the runtime:

1. [Install and configure UR External Control](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/setup/robot_setup.html).
2. [Configure robot networking](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/setup/network_setup.html).
3. [Install and configure the Universal Robots ToolComm Forwarder URCap](https://github.com/UniversalRobots/Universal_Robots_ToolComm_Forwarder_URCap) for gripper communication.

!!! caution "Use distinct ports"
    The two robots must not use the same External Control custom port. The launch example below uses `50002` for the left robot and `50102` for the right robot.

### Configure ToolComm for two grippers

With two Robotiq grippers, the two ToolComm Forwarder instances must also use distinct TCP ports. The reference launch uses `54321` for the left robot and `54322` for the right robot.

After installing ToolComm Forwarder, select one robot and open a shell session using the [UR shell-access procedure](https://docs.universal-robots.com/tutorials/urscript-tutorials/ssh.html). On that robot, verify the running ToolComm process and edit the ToolComm Forwarder daemon configuration:

```bash
# Verify the running process and its TCP port.
top -c

# Edit the ToolComm Forwarder daemon configuration.
nano /ursim/GUI/felix-cache/bundle185/data/com/fzi/rs485/impl/daemon/daemon-rs485.py
```

Change the `socat` TCP port from `54321` to `54322`, save the file, and reboot that robot. After rebooting, use `top -c` again to verify that the configured ToolComm process uses the intended port.

The path contains the URCap bundle identifier used by the documented `ur_robotiq` setup. If the installed ToolComm Forwarder version uses a different bundle directory, locate the installed daemon rather than assuming that `bundle185` is universal. Perform robot shell changes under the site's robot-administration procedure.

### Verify calibration and physical configuration

Before real-hardware launch, verify that the robot deployment has the intended:

- left and right UR calibration files;
- left and right robot base poses;
- left and right robot IP assignments;
- distinct External Control custom ports;
- distinct ToolComm TCP ports and local virtual serial device names.

Keep these machine-local values in the robot launch arguments and deployment configuration. Do not commit robot credentials, IP addresses, calibration files, or site details. The committed `lerobot_ros2` profile records the ROS interface contract, not the site's physical configuration.

## 3. Start the released robot runtime

Create the long-lived container once, or start the existing container. Use host networking for the ROS and robot network path, but do not grant blanket privileged access. Add only reviewed device mappings, groups, or Linux capabilities when the driver actually requires them:

```bash
docker start ur_robotiq || \
  docker run -dit \
    --net host \
    --name ur_robotiq \
    --entrypoint bash \
    ghcr.io/penzottimattia/ur_robotiq:gello
```

Launch the bimanual stack with the site-specific values. Keep the dashboard setup node disabled for the first real-hardware graph check:

```bash
docker exec -it ur_robotiq bash -lc '  source /opt/ros/humble/ur_robotiq/setup.bash && \
  ros2 launch ur_robotiq control_bimanual_ur3_robotiq.launch.py \
    mode:=none \
    use_gello:=true \
    run_setup_node:=false \
    left_robot_ip:=<LEFT_ROBOT_IP> \
    right_robot_ip:=<RIGHT_ROBOT_IP> \
    left_custom_port:=50002 \
    right_custom_port:=50102 \
    left_tool_tcp_port:=54321 \
    right_tool_tcp_port:=54322'
```

Replace every placeholder and confirm that the selected ports match the teach-pendant and ToolComm configuration. Do not reuse example IP addresses from another installation.

Inspect the launch arguments exposed by the installed runtime when needed:

```bash
docker exec -it ur_robotiq bash -lc '  source /opt/ros/humble/ur_robotiq/setup.bash && \
  ros2 launch ur_robotiq control_bimanual_ur3_robotiq.launch.py -s'
```

The runtime uses host networking. Set the same `ROS_DOMAIN_ID` for the robot runtime and `lerobot_ros2`. Keep the robot launch in its own terminal and do not add it to this repository's Compose project.

## 4. Validate before enabling motion

Follow the staged checks in [Hardware and safety](hardware-and-safety.md). At minimum:

1. validate the profile first with `mode:=full_mock`;
2. keep `run_setup_node:=false` during the first real-hardware graph inspection;
3. keep both GELLO offset nodes in `control_mode=0`;
4. inspect controller and hardware-component state;
5. verify left and right state and command paths independently;
6. confirm workspace clearance, limits, and emergency-stop access before enabling commands.

## 5. Verify the interface contract

Both runtimes must use the same `ROS_DOMAIN_ID` and host ROS network.

| Interface | Expected name |
|---|---|
| Left state | `/left_state_broadcaster/joint_states` |
| Right state | `/right_state_broadcaster/joint_states` |
| Left action | `/left_arm_controller/commands` |
| Right action | `/right_arm_controller/commands` |
| Left mode service | `/left_gello_offset_node/set_parameters` |
| Right mode service | `/right_gello_offset_node/set_parameters` |
| Left transition service | `/left_gello_offset_node/wait_for_mode_transition` |
| Right transition service | `/right_gello_offset_node/wait_for_mode_transition` |

Inspect the live graph before recording or deployment:

```bash
ros2 control list_controllers
ros2 control list_hardware_components
ros2 topic info /left_state_broadcaster/joint_states
ros2 topic info /right_state_broadcaster/joint_states
ros2 topic info /left_arm_controller/commands
ros2 topic info /right_arm_controller/commands
ros2 service type /left_gello_offset_node/set_parameters
ros2 service type /right_gello_offset_node/set_parameters
ros2 service type /left_gello_offset_node/wait_for_mode_transition
ros2 service type /right_gello_offset_node/wait_for_mode_transition
```

Then run the learning-side graph check:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-doctor --skip-hardware
```

Do not pass `--skip-ros-graph` for this integration check.

## 6. GELLO operating contract

The normal workflow uses two modes:

- `0`: idle, with no command publishing;
- `1`: normal offset mode for teleoperation and recording.

Modes `2` and `3` are robot-runtime diagnostics and are not part of normal `lerobot_ros2` operation. The profile uses the transition services exposed by the bimanual launch. Return both offset nodes to mode `0` before changing hardware, controller, calibration, gripper, or topic settings.

## 7. Continue to recording or deployment

After the interface check:

1. [Probe configured cameras](workflows.md#probe-configured-cameras).
2. [Record demonstrations](workflows.md#record-demonstrations).
3. [Visualize the dataset](workflows.md#visualize-a-dataset).
4. Confirm camera roles, image orientation, joint ordering, action dimensions, gripper convention, and `recording.hz` before training.
5. Repeat the same compatibility checks before deploying a checkpoint.

## Troubleshooting

### The robot runtime starts but topics are missing

Confirm that both runtimes use the same `ROS_DOMAIN_ID`, host ROS networking is available, and the robot deployment reports active controllers. Do not patch the robot container from this guide.

### Service names differ from the profile

Stop the workflow and compare the deployed robot image with `profiles/ur_robotiq_bimanual/compatibility.yaml`. Update the runtime or submit a reviewed profile change; do not make an unrecorded local interface fork.

### One arm is associated with the wrong data

Check the left/right topic mapping in `config.local/gello.yaml` and ask the robot operator to verify the deployed robot configuration. Keep robot addresses and calibration details in the robot deployment, not in this repository.

### Dataset actions jump by approximately one revolution

Review `wrap_joints`, `unwrap.max_step`, joint ordering, and the physical wrist state before recording again. Preserve the original dataset and correct the integration before collecting replacement episodes.

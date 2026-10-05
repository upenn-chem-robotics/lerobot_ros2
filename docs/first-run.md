# Visualize a dataset

> **Scope:** Software-only inspection of an existing LeRobot dataset. This workflow does not access cameras or command a robot. Dataset recording is covered in [Record demonstrations](workflows.md#record-demonstrations).

## Prerequisites

Required inputs are the installed release image and an existing LeRobot dataset below the host `data/` directory. If you are unsure whether a directory is a usable dataset, place it below `data/` and use the viewer as the first identification check; a directory merely existing is not enough. No Hugging Face authentication is required unless the dataset must first be downloaded from a private or gated repository. For example, host `data/pick_place` appears in the container as `/data/pick_place`.

## 1. Run the software-only preflight

```bash
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

This checks the packaged application and local mounts. It does not validate the dataset itself.

**Success criterion:** the command exits successfully without reporting a failed packaged-application or mount check. Because hardware and ROS-graph checks are explicitly skipped, this result does not say that a camera, robot, checkpoint, GPU, or dataset is ready.

## 2. Open the dataset

```bash
docker compose run --rm --service-ports tools \
  lerobot-ros-app --dataset_dir /data/pick_place --port 7860
```

Replace `/data/pick_place` with your path below `/data`, then open `http://localhost:7860` on the same workstation.

**Expected outcome:** the application remains running and the browser opens the selected dataset.

If it does not:

- **The path is missing:** confirm that host `data/pick_place` exists and that the command uses `/data/pick_place`.
- **The page does not open:** confirm that the command is still running and open `http://localhost:7860` on the same workstation.
- **The page opens but no episodes appear:** the directory may not be a recognized or complete LeRobot dataset; stop and inspect its metadata and media rather than transforming it.
- **Metadata appears but images or videos do not:** keep the source unchanged and check that its referenced media files are present.

 This command reads the dataset for inspection; do not use the source directory as the destination of a later repair or transformation.

## 3. Inspect before processing or training

Check that episodes and frames load; camera identity and orientation are correct; timestamps and task boundaries are plausible; observations and actions have the expected dimensions; and partial episodes are identified. Typical warning signs include exchanged camera roles, rotated images, discontinuous action traces, long inactive intervals, implausible timestamp gaps, and an incomplete final episode.

**Dataset inspection passed:** the intended episodes load; camera identity and orientation are understood; timestamps are plausible; observation and action dimensions are consistent; and incomplete episodes are identified.

If an issue is found, determine whether it is a metadata problem, requires a non-destructive dataset transformation, or indicates that the recording interface itself was wrong. Keep the source unchanged and write repairs or transformations to a new destination; recollect data when the original observations or action semantics cannot be recovered reliably.

!!! warning "Hardware boundary"
    Camera probing, recording, deployment, and DAgger require the checks in [Hardware and safety](hardware-and-safety.md), resolved `REPLACE_*` values, a reviewed `compose.hardware.yaml`, and completed no-motion checks.

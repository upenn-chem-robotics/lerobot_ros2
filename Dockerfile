# Reproducible image for lerobot-ros2.
#
# Conda-based on purpose: ROS 2 Humble (rclpy, sensor_msgs, std_srvs,
# rcl_interfaces) comes from the robostack-staging conda channel, not apt and
# not PyPI. environment.yml is therefore the source of truth for the
# environment; constraints.txt only covers the pip-installed subset.
#
# Build (needs network: conda-forge, robostack-staging, PyPI, GitHub):
#   docker build -t lerobot-ros2 .
#
# Run with GPU (requires the NVIDIA container runtime on the host):
#   docker run --gpus all -it --rm -v /path/to/data:/lerobot-ros/data lerobot-ros2
#
# Cameras and the foot pedal need device access, e.g.:
#   --device /dev/video0 --device /dev/video2 --group-add video
FROM condaforge/miniforge3:26.3.2-3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# git: pip installs lerobot from a git commit.
# v4l-utils: v4l2-ctl, used to apply/verify camera controls at record time.
# libgl1 / libglib2.0-0: shared libs OpenCV needs at import.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      v4l-utils \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /lerobot-ros

# ── 1. Conda environment ────────────────────────────────────────────────────
# Copied alone so this layer is cached until the lockfiles actually change.
# Creates an env named "lerobot" (from the name: key in environment.yml).
COPY environment.yml constraints.txt ./
RUN conda env create -f environment.yml \
 && conda clean -afy

# ── 2. lerobot at the validated commit ──────────────────────────────────────
# --no-deps because environment.yml already pins lerobot's whole dependency
# tree; letting pip re-resolve here would fight the conda-provided packages
# (numpy, av and all of ROS) and silently change validated versions.
ARG LEROBOT_COMMIT=d60a700d2b32590ed113d694fd87617e43506081
RUN conda run -n lerobot pip install --no-deps \
      "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

# ── 3. This repo + both policy plugins ──────────────────────────────────────
# The plugins must stay separate distributions: lerobot discovers third-party
# policies by scanning installed distributions whose NAME starts with
# "lerobot_policy_", so folding them in would break --policy.type resolution.
COPY . .
RUN conda run -n lerobot pip install --no-deps -e . \
 && conda run -n lerobot pip install --no-deps -e packages/lerobot_policy_strided_diffusion \
 && conda run -n lerobot pip install --no-deps -e packages/lerobot_policy_action_history_diffusion

# Fail the build if anything is inconsistent or a policy plugin is undiscoverable.
RUN conda run -n lerobot pip check \
 && conda run -n lerobot python -c "\
from lerobot.utils.import_utils import register_third_party_plugins; \
register_third_party_plugins(); \
import lerobot_ros2, torch, rclpy; \
print('ok: lerobot_ros2 + torch', torch.__version__, '+ rclpy imported')"

ENV GELLO_CONFIG=/lerobot-ros/config/gello.yaml

# Every command must run inside the env so rclpy and the CLIs resolve.
ENTRYPOINT ["conda", "run", "-n", "lerobot", "--no-capture-output"]
CMD ["bash"]

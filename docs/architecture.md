# Architecture

Conda supplies Python, native scientific libraries, FFmpeg, and RoboStack ROS 2 Humble. Pip supplies pinned Python-only ML dependencies and one LeRobot commit. Wheels supply the main project and policy plugins. Compose supplies mounts, GPU, host ROS networking, and explicit devices.

Configuration is operator-owned and mounted at runtime. Datasets, checkpoints, hardware inventory, and caches are not baked into images.

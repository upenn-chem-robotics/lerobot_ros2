# Third-party notices

This file records the principal upstream software and services named by this repository. It is informational and does not replace the license files shipped with third-party components.

## Independence and names

This project is independent and is not affiliated with, endorsed by, sponsored by, or officially connected with Hugging Face or the LeRobot project. The names “LeRobot” and “Hugging Face” are used only to identify upstream projects, formats, APIs, and services with which this software interoperates. No ownership of those names or associated branding is claimed.

## LeRobot

- Upstream project: `huggingface/lerobot`
- Upstream license: Apache License 2.0
- Copyright notice in the upstream license: Copyright 2024 The Hugging Face team. All rights reserved.
- Version used by the container build: commit `d60a700d2b32590ed113d694fd87617e43506081` (the dependency lock identifies LeRobot distribution version `0.5.1`).

LeRobot is fetched and installed as a separate dependency. Its source code, binaries, modifications, and notices remain governed by the upstream Apache License 2.0 and any notices distributed with the relevant upstream version. If a release artifact or container image redistributes LeRobot, preserve the upstream license and copyright notices and record any modifications as required by Apache License 2.0.

## Other third-party components

Python, ROS 2, container base images, system packages, Python packages, hardware SDKs, and optional tools retain their respective licenses and notices. Lock files and container manifests identify versions but do not replace those licenses. Before distributing a container image or binary bundle, generate and review an artifact-specific software bill of materials and license report because the final dependency set depends on the selected build target and optional components.

The optional OBSBOT CLI bootstrap is separately licensed and is not covered by this project's Apache License 2.0 merely because a setup script is provided here.

## This project

Unless a file states otherwise, the original content of this repository is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).

# Release process

- [ ] Review one version across main package, plugins, tag, and image.
- [ ] Review `dependencies.env`; rebuild without cache.
- [ ] Reconcile Conda and pip; avoid duplicate ABI-sensitive libraries.
- [ ] Build and clean-install all wheels.
- [ ] Run Ruff, tests, `pip check`, image smoke tests, and `ros2 doctor --report`.
- [ ] Run Gitleaks over all history and Trivy over the image.
- [ ] Generate and retain an SPDX or CycloneDX SBOM.
- [ ] Verify no data, checkpoints, logs, addresses, tokens, or private inventories are tracked.
- [ ] Complete camera, pedal, ROS, no-motion, recording-finalization, and shutdown hardware acceptance tests.
- [ ] Publish immutable `MAJOR.MINOR.PATCH` and `sha-<commit>` images from the tagged commit.

Do not publish `latest` until the process has succeeded for multiple release candidates.

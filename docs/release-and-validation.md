# Release and validation

A release claim must match the validation level actually completed. Hardware-free success does not imply checkpoint, dataset, or hardware acceptance.

## Validation levels

### Level 0: static

Python, TOML, and YAML parsing; required files; Compose configuration; merge-marker checks; and forbidden-path checks.

### Level 1: container and packaging

Runtime and development image builds, `pip check`, package installation, ROS installation, CLI discovery, and policy-distribution discovery.

### Level 2: hardware-free integration

ROS 2 publisher/subscriber round trip, installed CLI parser execution, and policy tests in the built development image.

```bash
./sh/test_no_hardware.sh
```

### Level 3: synthetic dataset and checkpoint

Not currently claimed. This level requires opening a minimal valid LeRobot dataset, running a representative transformation, instantiating supported policies, loading a controlled checkpoint, and exercising the observation-to-command boundary.

### Level 4: hardware acceptance

Not covered by automated hardware-free validation. Cameras, operator inputs, dashboards, arms, grippers, recording finalization, shutdown, and actuation require controlled acceptance tests.

## Public-release verification

Run from the repository root:

```bash
./sh/verify_public_release.sh
```

The script writes a timestamped log and summary under `release-logs/` and continues after individual failures so one run captures the complete state. Do not publish from a different tree than the verified commit.

## Release checklist

- [ ] Reconcile the version across the main package, policy plugins, tag, and images.
- [ ] Review `dependencies.env` and perform a clean cold rebuild.
- [ ] Keep ABI-sensitive dependencies in their intended Conda or pip layer.
- [ ] Build and clean-install all wheels.
- [ ] Run Ruff, pytest, `pip check`, image smoke tests, and `ros2 doctor --report`.
- [ ] Run `./sh/test_no_hardware.sh`.
- [ ] Run Gitleaks against the complete Git history and document reviewed suppressions.
- [ ] Run Trivy against the release image.
- [ ] Generate and retain an SPDX or CycloneDX SBOM.
- [ ] Verify that no data, checkpoints, logs, addresses, tokens, or private inventories are tracked.
- [ ] Complete controlled hardware acceptance before making hardware claims.
- [ ] Publish immutable `MAJOR.MINOR.PATCH` and `sha-<commit>` image tags from the tagged commit.

Do not use `latest` as the sole release identifier.

## Release statement

Until Levels 3 and 4 are completed, describe the project as a hardware-free-validated research preview. State explicitly that external datasets, arbitrary checkpoints, and physical hardware are outside the automated validation claim.

# Runtime activation and Trivy correction

The images now build successfully. All runtime, ROS, plugin, pytest, and Ruff commands failed before execution because the entrypoint enabled Bash `nounset` before RoboStack's Conda activation hook referenced `CONDA_BUILD`.

The entrypoint now disables `nounset` only while Conda and RoboStack activation hooks run, then restores it before executing the requested command.

Trivy now scans vulnerabilities only, because Gitleaks already handles repository secrets, and uses a configurable 20-minute timeout:

```bash
TRIVY_TIMEOUT=30m ./verify_public_release.sh
```

The image remains cached and retained by default.

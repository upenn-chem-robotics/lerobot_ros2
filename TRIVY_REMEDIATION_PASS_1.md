# Trivy remediation, pass 1

All functional and secret-scanning checks pass. The first successful image scan found fixable Ubuntu and Python findings plus vulnerabilities in W&B's embedded `wandb-core` Go binary.

This pass updates directly controlled dependencies:

- upgrades Ubuntu packages during image construction, including the fixed `libssl3t64` available from the configured repository
- upgrades Diffusers from 0.35.2 to 0.38.0
- pins urllib3 2.8.0
- requires setuptools 78.1.1 through the LeRobot-compatible `<81` upper bound
- upgrades setuptools and urllib3 in the Miniforge base environment as well as the `lerobot` environment

The verifier now saves the complete Trivy result as `release-logs/trivy-<timestamp>.json`, avoiding a huge terminal table and making the remaining findings reviewable.

No exception is added for `wandb-core` yet. The next scan should establish exactly what remains after the controlled upgrades.

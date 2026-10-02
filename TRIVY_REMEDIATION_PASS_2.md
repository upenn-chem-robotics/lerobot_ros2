# Trivy remediation, pass 2

The full JSON report shows 55 remaining findings: 48 in W&B's embedded `wandb-core` Go binary and 7 Python findings.

This pass corrects base-environment package metadata and the remaining directly upgradable Python dependency:

- installs setuptools `>=78.1.1,<81` through Conda in the base environment
- installs urllib3 `>=2.8,<3` through Conda in the base environment
- installs `msgpack-python>=1.2.1,<2` in both base and runtime Conda environments

Using Conda rather than overlaying these packages with pip prevents stale Conda package records from continuing to appear as vulnerable installations.

Known residuals expected after this pass:

- Diffusers 0.35.2, because 0.38.0 requires an incompatible pre-release safetensors version
- vulnerabilities embedded in `wandb-core` from the newest W&B release allowed by LeRobot 0.5.1's `<0.25` constraint

Those residuals remain visible in the JSON report. They are not added to an ignore file in this delta.

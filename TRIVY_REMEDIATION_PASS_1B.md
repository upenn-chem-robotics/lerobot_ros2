# Trivy remediation, pass 1B

The Diffusers 0.38.0 upgrade is reverted because that release requires the pre-release `safetensors>=0.8.0-rc.0`, while this environment deliberately pins the stable `safetensors==0.7.0`. Pip correctly rejected the incompatible set before either image could build.

The other remediation changes remain:

- Ubuntu package upgrade during image construction
- setuptools `>=78.1.1,<81`
- urllib3 `2.8.0`
- JSON Trivy report output

Diffusers remains at 0.35.2 for compatibility. Its Trivy findings stay visible in the generated report rather than being hidden or bypassed.

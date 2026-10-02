# Dependency reconciliation

The verification log reached `pip check` with LeRobot 0.5.1 and exposed its installed distribution requirements. This update aligns the Conda-native ABI packages and pip-only dependencies with those requirements:

- NumPy `>=2.0,<2.3`
- PyAV `>=15,<16`
- packaging `>=24.2,<26`
- setuptools `>=75,<81`
- W&B `0.24.2`, satisfying `>=0.24,<0.25`
- added cmake, deepdiff, imageio, jsonlines, pynput, pyserial, rerun-sdk, and termcolor

The verifier continues to use Docker cache by default. If resolution changes one of these exact pip pins, retain the new log and update the pin from the solver result rather than disabling `pip check`.

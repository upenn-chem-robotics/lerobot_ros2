# Hardware

Before operation: verify the emergency stop and clear workspace; confirm persistent camera and input paths; confirm container group permissions; apply and read back V4L2 controls; disable firmware auto-framing, zoom, HDR, auto-exposure, auto white balance, and autofocus where required; run the doctor; observe ROS state before enabling commands.

Recording and deployment must use identical camera geometry and controls. The optional OBSBOT helper requires an explicitly reviewed repository and commit. Its proprietary SDK is not vendored here.

# rerun-sdk reconciliation

The pinned LeRobot 0.5.1 distribution requires `rerun-sdk>=0.24.0,<0.27.0`.
The prior lock selected 0.27.2, so `pip check` correctly rejected both images.
This delta pins `rerun-sdk==0.26.2` and leaves `pip check` enabled.

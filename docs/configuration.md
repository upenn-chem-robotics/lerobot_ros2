# Configuration

Provide configuration through `--config PATH` or `GELLO_CONFIG`. The image defaults to `/config/gello.yaml`; there is no source-tree fallback. Copy `examples/gello.yaml` to ignored `config.local/` and prefer `/dev/*/by-id` paths.

The schema covers arms and ROS services, camera groups and V4L2 controls, recording, visualization, joint unwrapping, homing, and pedal input. Run `lerobot-ros-doctor` after every topic or hardware change. Tracked `config/` files are sanitized compatibility examples, not production lab settings.

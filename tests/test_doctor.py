from pathlib import Path

from lerobot_ros2.cli.doctor import _config_check, _device_checks


def test_config_check_accepts_mapping(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  device: /dev/definitely-not-present\n")
    check, config = _config_check(path)
    assert check.ok
    assert config["camera"]["device"].startswith("/dev/")


def test_missing_device_fails():
    checks = _device_checks({"device": "/dev/definitely-not-present"})
    assert len(checks) == 1
    assert not checks[0].ok

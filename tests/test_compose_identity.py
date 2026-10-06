from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_runs_as_host_identity_and_uses_host_owned_bind_mounts():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    common = compose["x-common"]

    assert common["user"] == "${LEROBOT_HOST_UID:-1000}:${LEROBOT_HOST_GID:-1000}"
    assert common["environment"]["HOME"] == "/tmp"
    assert common["environment"]["LEROBOT_HOST_UID"] == "${LEROBOT_HOST_UID:-1000}"
    assert common["environment"]["LEROBOT_HOST_GID"] == "${LEROBOT_HOST_GID:-1000}"
    assert "${LEROBOT_DATA:-./data}:/data" in common["volumes"]
    assert "${LEROBOT_CACHE:-./.cache/lerobot}/huggingface:/cache/huggingface" in common["volumes"]
    assert "${LEROBOT_CACHE:-./.cache/lerobot}/torch:/cache/torch" in common["volumes"]

    build_args = common["build"].get("args", {})
    assert "APP_UID" not in build_args
    assert "APP_GID" not in build_args

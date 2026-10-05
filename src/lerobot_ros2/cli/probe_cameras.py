#!/usr/bin/env python3
"""
Probe every camera listed in ``config.local/gello.yaml`` and dump a structured
snapshot of every v4l2 control plus supported pixel formats to
``config/camera_probes/<camera_name>.yaml``.

Use this output as the source of truth when deciding which controls to pin
in ``camera_defaults`` / per-camera ``settings`` in ``gello.yaml``.

Requires ``v4l2-ctl`` (``sudo apt install v4l-utils``).

Usage:
    lerobot-ros-probe-cameras --config config.local/gello.yaml
    lerobot-ros-probe-cameras --config config.local/gello.yaml --camera left_wrist_top
    lerobot-ros-probe-cameras --config config.local/gello.yaml --raw
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from lerobot_ros2.config_paths import resolve_config_path
from lerobot_ros2.helper import CameraConfig, load_camera_configs, load_config

CONTROL_LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)        # control name
    \s+0x[0-9a-fA-F]+\s*                     # ioctl id
    \(\s*(?P<type>[a-zA-Z0-9_]+)\s*\)        # (int|bool|menu|...)
    \s*:\s*
    (?P<rest>.*)
    $
    """,
    re.VERBOSE,
)

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?\d+)")
MENU_ITEM_RE = re.compile(r"^\s+(\d+):\s+(.*)$")
MENU_LABEL_TAIL_RE = re.compile(r"\(([^)]+)\)\s*$")
FLAGS_RE = re.compile(r"flags=([A-Za-z0-9_,\-]+)")


# Controls whose driver name stabilised differently across kernels; we record
# both spellings in the snapshot summary so the user can search for either.
AUTO_CONTROLS = {
    "exposure_auto",
    "auto_exposure",
    "white_balance_automatic",
    "white_balance_temperature_auto",
    "focus_automatic_continuous",
    "focus_auto",
    "backlight_compensation",
    "exposure_dynamic_framerate",
    "hue_auto",
    "gain_automatic",
}


def _run_v4l2(port: str, args: List[str]) -> str:
    cmd = ["v4l2-ctl", "--device", port, *args]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"v4l2-ctl timed out on {port}: {' '.join(cmd)}") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"v4l2-ctl failed on {port} ({' '.join(cmd)}): {stderr}"
        )
    return result.stdout


def _parse_flags(rest: str) -> List[str]:
    match = FLAGS_RE.search(rest)
    if not match:
        return []
    return [flag for flag in match.group(1).split(",") if flag]


def parse_list_ctrls_menus(raw: str) -> Dict[str, Dict[str, Any]]:
    """Parse the output of ``v4l2-ctl --list-ctrls-menus`` into a dict.

    The parser deliberately preserves every numeric field it sees (``min``,
    ``max``, ``step``, ``default``, ``value``) so we never lose information.
    Menu items that follow a ``(menu)`` control are attached as ``options``.
    """

    controls: Dict[str, Dict[str, Any]] = {}
    current: Optional[str] = None

    for raw_line in raw.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            current = None
            continue

        ctrl_match = CONTROL_LINE_RE.match(line)
        if ctrl_match:
            name = ctrl_match.group("name")
            ctype = ctrl_match.group("type").lower()
            rest = ctrl_match.group("rest")

            entry: Dict[str, Any] = {"type": ctype}
            for key, value in KV_RE.findall(rest):
                entry[key] = int(value)

            flags = _parse_flags(rest)
            if flags:
                entry["flags"] = flags

            if ctype == "menu":
                tail = MENU_LABEL_TAIL_RE.search(rest)
                if tail and "value" in entry:
                    entry["current_label"] = tail.group(1).strip()
                entry["options"] = {}

            controls[name] = entry
            current = name if ctype == "menu" else None
            continue

        if current is not None:
            menu_match = MENU_ITEM_RE.match(raw_line)
            if menu_match:
                idx = int(menu_match.group(1))
                label = menu_match.group(2).strip()
                controls[current].setdefault("options", {})[idx] = label
                continue
            current = None

    return controls


def parse_list_formats_ext(raw: str) -> List[Dict[str, Any]]:
    """Parse ``v4l2-ctl --list-formats-ext`` into a list of format dicts."""

    formats: List[Dict[str, Any]] = []
    current_fmt: Optional[Dict[str, Any]] = None
    current_size: Optional[Dict[str, Any]] = None

    fmt_re = re.compile(r"\[\d+\]:\s*'([^']+)'\s*\((.*)\)")
    size_re = re.compile(r"Size:\s*Discrete\s+(\d+)x(\d+)")
    interval_re = re.compile(r"Interval:\s*Discrete\s+([0-9.]+)s\s*\(([0-9.]+)\s*fps\)")

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        fmt_match = fmt_re.search(line)
        if fmt_match:
            current_fmt = {
                "fourcc": fmt_match.group(1),
                "description": fmt_match.group(2),
                "sizes": [],
            }
            formats.append(current_fmt)
            current_size = None
            continue

        size_match = size_re.search(line)
        if size_match and current_fmt is not None:
            current_size = {
                "width": int(size_match.group(1)),
                "height": int(size_match.group(2)),
                "fps": [],
            }
            current_fmt["sizes"].append(current_size)
            continue

        interval_match = interval_re.search(line)
        if interval_match and current_size is not None:
            current_size["fps"].append(float(interval_match.group(2)))

    return formats


def probe_camera(camera: CameraConfig, include_raw: bool) -> Dict[str, Any]:
    ctrls_raw = _run_v4l2(camera.port, ["--list-ctrls-menus"])
    formats_raw = _run_v4l2(camera.port, ["--list-formats-ext"])
    all_raw = _run_v4l2(camera.port, ["--all"])

    snapshot: Dict[str, Any] = {
        "camera": camera.name,
        "port": camera.port,
        "index": camera.index,
        "controls": parse_list_ctrls_menus(ctrls_raw),
        "formats": parse_list_formats_ext(formats_raw),
    }
    if include_raw:
        snapshot["raw"] = {
            "list_ctrls_menus": ctrls_raw,
            "list_formats_ext": formats_raw,
            "all": all_raw,
        }
    return snapshot


def _is_auto_on(name: str, entry: Dict[str, Any]) -> Optional[bool]:
    """Return True if ``entry`` looks like an 'auto' knob currently enabled.

    Bool controls: auto when ``value == 1``.
    Menu controls: auto when the current option label mentions 'Auto'/'Aperture'
    (common vendor wording for 'auto exposure').
    Returns None when this doesn't look like an auto control.
    """
    if name not in AUTO_CONTROLS and not name.endswith("_auto"):
        return None

    ctype = entry.get("type")
    value = entry.get("value")
    if ctype == "bool":
        return bool(value)
    if ctype == "menu":
        label = entry.get("current_label") or ""
        label_lower = label.lower()
        if "manual" in label_lower:
            return False
        if any(token in label_lower for token in ("auto", "aperture", "priority")):
            return True
        return None
    if ctype == "int":
        return bool(value)
    return None


def _snapshot_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    autos_on: List[Tuple[str, Any]] = []
    for ctrl_name, entry in snapshot["controls"].items():
        status = _is_auto_on(ctrl_name, entry)
        if status:
            autos_on.append(
                (
                    ctrl_name,
                    entry.get("current_label") or entry.get("value"),
                )
            )
    return {
        "camera": snapshot["camera"],
        "port": snapshot["port"],
        "autos_on": autos_on,
        "control_count": len(snapshot["controls"]),
        "format_count": len(snapshot["formats"]),
    }


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)


def _print_cross_camera_diff(snapshots: List[Dict[str, Any]]) -> None:
    """For every control that appears on >=2 cameras, show which cameras disagree
    on the current value. This is the diff the user acts on when writing the
    shared ``camera_defaults`` block vs per-camera overrides."""

    by_control: Dict[str, Dict[str, Any]] = {}
    for snap in snapshots:
        for ctrl_name, entry in snap["controls"].items():
            by_control.setdefault(ctrl_name, {})[snap["camera"]] = entry.get("value")

    diffs: List[Tuple[str, Dict[str, Any]]] = []
    agrees: List[Tuple[str, Any]] = []
    for ctrl_name, values_by_cam in sorted(by_control.items()):
        distinct = {v for v in values_by_cam.values() if v is not None}
        if len(distinct) <= 1:
            only_value = next(iter(distinct), None)
            if len(values_by_cam) == len(snapshots):
                agrees.append((ctrl_name, only_value))
            continue
        diffs.append((ctrl_name, values_by_cam))

    print("\n=== Cross-camera diff (controls where current values disagree) ===")
    if not diffs:
        print("  (none — all cameras report identical values for every shared control)")
    else:
        for ctrl_name, values_by_cam in diffs:
            joined = ", ".join(
                f"{cam}={values_by_cam[cam]}" for cam in sorted(values_by_cam)
            )
            print(f"  {ctrl_name}: {joined}")

    print("\n=== Controls shared by every camera with identical current values ===")
    if not agrees:
        print("  (none)")
    else:
        preview = agrees[:25]
        for ctrl_name, value in preview:
            print(f"  {ctrl_name}={value}")
        if len(agrees) > len(preview):
            print(f"  ... and {len(agrees) - len(preview)} more")


def _print_autos_summary(summaries: List[Dict[str, Any]]) -> None:
    print("\n=== Auto controls currently ON (these are what's drifting on you) ===")
    any_on = False
    for summary in summaries:
        if not summary["autos_on"]:
            continue
        any_on = True
        parts = ", ".join(f"{name}={value}" for name, value in summary["autos_on"])
        print(f"  {summary['camera']} ({summary['port']}): {parts}")
    if not any_on:
        print("  (none — all autos already off)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the named camera / teleop config file. Defaults to $GELLO_CONFIG.",
    )
    parser.add_argument(
        "--camera",
        help="Probe only this camera (name as it appears under *_cameras in gello.yaml).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also embed verbatim v4l2-ctl output under a 'raw' key in each YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write per-camera YAMLs into. "
             "Defaults to <config-dir>/camera_probes next to the loaded config.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if shutil.which("v4l2-ctl") is None:
        print(
            "ERROR: v4l2-ctl not found on PATH. Install it with:\n"
            "    sudo apt install v4l-utils",
            file=sys.stderr,
        )
        return 2

    config_path = resolve_config_path(args.config)
    output_dir = args.output_dir or (config_path.parent / "camera_probes")
    cfg = load_config(config_path)
    cameras = load_camera_configs(cfg)

    if args.camera:
        cameras = [c for c in cameras if c.name == args.camera]
        if not cameras:
            print(
                f"ERROR: camera {args.camera!r} is not listed in gello.yaml",
                file=sys.stderr,
            )
            return 2

    snapshots: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for camera in cameras:
        print(f"\n--- Probing {camera.name} @ {camera.port} ---")
        try:
            snapshot = probe_camera(camera, include_raw=args.raw)
        except RuntimeError as exc:
            failures.append((camera.name, str(exc)))
            print(f"  FAILED: {exc}")
            continue

        out_path = output_dir / f"{camera.name}.yaml"
        _write_yaml(out_path, snapshot)
        snapshots.append(snapshot)
        summary = _snapshot_summary(snapshot)
        summaries.append(summary)
        print(
            f"  wrote {out_path} "
            f"({summary['control_count']} controls, {summary['format_count']} formats)"
        )

    if summaries:
        _print_autos_summary(summaries)
    if len(snapshots) >= 2:
        _print_cross_camera_diff(snapshots)

    if failures:
        print("\n=== Failures ===", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

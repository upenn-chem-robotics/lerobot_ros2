"""Gradio-based visualizer for LeRobot v3 teleoperation datasets.

Launch with:
    lerobot-ros-app --dataset_dir PATH
"""

from __future__ import annotations

import argparse
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from lerobot_ros2.data_loader import ArmSpec, TeleopDataset
from lerobot_ros2.video_exporter import (
    build_episode_timeline_figure,
    export_episode_grid_video,
    export_episode_video,
)


def build_app(dataset_dir: str | Path) -> gr.Blocks:
    """Construct the Gradio Blocks application.

    Args:
        dataset_dir: Path to the LeRobot v3 dataset root.

    Returns:
        A gr.Blocks instance ready to launch.
    """
    print(f"Loading dataset from {dataset_dir} ...", flush=True)
    ds = TeleopDataset(dataset_dir, preload_frames=False)
    print("Dataset loaded.\n", flush=True)

    ep_actions: dict[int, np.ndarray] = {}
    ep_states: dict[int, np.ndarray] = {}
    ep_times: dict[int, list[float]] = {}
    for ep in ds.episode_ids:
        ep_actions[ep] = ds.get_episode_actions(ep)
        ep_states[ep] = ds.get_episode_states(ep)
        ep_times[ep] = ds.get_episode_timestamps(ep)

    arm_specs = ds.arm_specs or [
        ArmSpec(name="arm_1", joint_names=tuple(ds.joint_names), start=0, stop=len(ds.joint_names))
    ]

    def _frame_offset_seconds(
        episode_id: int,
        frame_idx: int,
        episode_times: list[float] | None = None,
    ) -> float:
        """Return the episode-relative time for a frame."""
        frame_idx = max(0, int(frame_idx))
        times = episode_times if episode_times is not None else ep_times.get(int(episode_id), [])

        if times:
            clipped_idx = min(frame_idx, len(times) - 1)
            base_time = float(times[0])
            current_time = float(times[clipped_idx])
            delta = current_time - base_time
            if np.isfinite(delta):
                return max(0.0, float(delta))

        fps = max(float(ds.fps), 1.0)
        return max(0.0, frame_idx / fps)

    # ── Helper renderers ─────────────────────────────────────────────

    def _arm_label(arm: ArmSpec) -> str:
        return arm.label if arm.name.startswith("arm_") else arm.label

    def _vector_bar_plot(vector: np.ndarray, title: str, color: str) -> plt.Figure:
        """Horizontal bar charts that follow the recorded arm layout."""
        fig, axes = plt.subplots(1, len(arm_specs), figsize=(5.2 * len(arm_specs), 3.2), squeeze=False)
        for ax, arm in zip(axes.ravel(), arm_specs):
            segment = vector[arm.start : arm.stop]
            labels = list(arm.joint_names)
            ax.barh(labels, segment, color=color)
            ax.set_xlim(-1.0, 1.0)
            ax.set_title(f"{_arm_label(arm)} {title}", fontsize=10)
            ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def _timeline_plot(
        episode_id: int, frame_idx: int, data_type: str = "action"
    ) -> plt.Figure:
        """Line plot of all dimensions across the episode with a vertical marker."""
        arr = ep_actions[episode_id] if data_type == "action" else ep_states[episode_id]
        return build_episode_timeline_figure(arr, arm_specs, episode_id, data_type, frame_idx=frame_idx)

    # ── Callbacks ─────────────────────────────────────────────────────

    def on_episode_change(episode_id: int, cam_left: str, cam_right: str):
        """Reset slider and re-render when episode changes."""
        episode_id = int(episode_id)
        max_frame = max(0, ds.episode_length(episode_id) - 1)
        slider_update = gr.update(value=0, maximum=max_frame)
        render = on_frame_change(episode_id, 0, cam_left, cam_right)
        return (slider_update, *render)

    def _video_value(video_meta: dict[str, float | int | str]) -> str:
        """Return local source path for gr.Video."""
        return str(video_meta["path"])

    def _video_update(
        video_meta: dict[str, float | int | str], label: str, playback_position: float
    ) -> gr.Video:
        """Return a Video component update with a source path and seek position."""
        return gr.Video(
            value=_video_value(video_meta),
            label=label,
            playback_position=float(playback_position),
        )

    def _camera_slug(camera_key: str) -> str:
        """Return a filesystem-friendly camera identifier."""
        return camera_key.replace("/", "_").replace(".", "_")

    def on_export_mode_change(export_mode: str):
        """Toggle export controls based on the selected mode."""
        is_grid = export_mode == "Episode grid"
        return (
            gr.update(visible=not is_grid),
            gr.update(visible=is_grid),
            gr.update(visible=is_grid),
            gr.update(visible=is_grid),
        )

    def on_video_source_change(
        episode_id: int,
        cam_left: str,
        cam_right: str,
        frame_idx: int = 0,
        episode_times: list[float] | None = None,
    ):
        """Update both embedded video players for current episode/camera selection."""
        episode_id = int(episode_id)
        frame_idx = int(frame_idx)
        left_meta = ds.get_episode_video_source(episode_id, cam_left)
        right_meta = ds.get_episode_video_source(episode_id, cam_right)
        left_seek = float(left_meta["start_seconds"]) + _frame_offset_seconds(
            episode_id, frame_idx, episode_times
        )
        right_seek = float(right_meta["start_seconds"]) + _frame_offset_seconds(
            episode_id, frame_idx, episode_times
        )

        return (
            _video_update(left_meta, f"Left Camera ({cam_left})", left_seek),
            _video_update(right_meta, f"Right Camera ({cam_right})", right_seek),
        )

    def on_frame_change(
        episode_id: int,
        frame_idx: int,
        cam_left: str,
        cam_right: str,
    ):
        """Render everything for a given episode + frame."""
        frame_idx = int(frame_idx)
        episode_id = int(episode_id)
        frame_idx = max(0, min(frame_idx, ds.episode_length(episode_id) - 1))
        episode_times = ep_times.get(episode_id, [])

        left_meta = ds.get_episode_video_source(episode_id, cam_left)
        right_meta = ds.get_episode_video_source(episode_id, cam_right)
        left_seek = float(left_meta["start_seconds"]) + _frame_offset_seconds(
            episode_id, frame_idx, episode_times
        )
        right_seek = float(right_meta["start_seconds"]) + _frame_offset_seconds(
            episode_id, frame_idx, episode_times
        )

        data = ds.get_frame_data(episode_id, frame_idx)

        action_fig = _vector_bar_plot(data["action"], "Action", "#4C9AFF")
        state_fig = _vector_bar_plot(data["state"], "State", "#50E3C2")
        action_timeline = _timeline_plot(episode_id, frame_idx, "action")
        state_timeline = _timeline_plot(episode_id, frame_idx, "state")

        info_text = (
            f"**Episode** {data['episode_index']}  |  "
            f"**Frame** {data['frame_index']}  |  "
            f"**Time** {data['timestamp']:.2f}s  |  "
            f"**Global idx** {data['global_index']}"
        )

        plt.close("all")

        return (
            _video_update(left_meta, f"Left Camera ({cam_left})", left_seek),
            _video_update(right_meta, f"Right Camera ({cam_right})", right_seek),
            action_fig,
            state_fig,
            action_timeline,
            state_timeline,
            info_text,
        )

    def on_camera_change(episode_id: int, cam_left: str, cam_right: str, frame_idx: int):
        """Refresh video sources and plots when a camera selection changes."""
        return on_frame_change(episode_id, frame_idx, cam_left, cam_right)

    # ── Pre-render initial frame so the UI isn't blank on load ─────
    cams = ds.camera_keys
    default_cam_left = cams[0]
    default_cam_right = cams[1] if len(cams) > 1 else cams[0]

    init_left_meta = ds.get_episode_video_source(ds.episode_ids[0], default_cam_left)
    init_right_meta = ds.get_episode_video_source(ds.episode_ids[0], default_cam_right)
    init_left_seek = float(init_left_meta["start_seconds"])
    init_right_seek = float(init_right_meta["start_seconds"])

    init = on_frame_change(ds.episode_ids[0], 0, default_cam_left, default_cam_right)
    init_act, init_state = init[2], init[3]
    init_act_tl, init_state_tl = init[4], init[5]
    init_info = init[6]

    # ── Layout ────────────────────────────────────────────────────────

    with gr.Blocks(title="Teleop Trajectory Visualizer") as app:
        gr.Markdown("# Teleoperation Trajectory Visualizer")
        gr.Markdown(
            "Scrub through episodes to see synchronized camera views, "
            "action vectors, and observation states."
        )

        with gr.Row():
            episode_dd = gr.Dropdown(
                choices=ds.episode_ids,
                value=ds.episode_ids[0],
                label="Episode",
                type="value",
            )
            frame_slider = gr.Slider(
                minimum=0,
                maximum=ds.episode_length(ds.episode_ids[0]) - 1,
                step=1,
                value=0,
                label="Frame",
            )

        info_md = gr.Markdown(value=init_info)

        with gr.Row():
            cam_left_dd = gr.Dropdown(
                choices=cams,
                value=default_cam_left,
                label="Left Camera",
                type="value",
            )
            cam_right_dd = gr.Dropdown(
                choices=cams,
                value=default_cam_right,
                label="Right Camera",
                type="value",
            )

        with gr.Row():
            cam0_video = gr.Video(
                label="Left Camera",
                value=_video_value(init_left_meta),
                elem_id="teleop-video-left",
                playback_position=init_left_seek,
                autoplay=False,
                interactive=False,
            )
            cam1_video = gr.Video(
                label="Right Camera",
                value=_video_value(init_right_meta),
                elem_id="teleop-video-right",
                playback_position=init_right_seek,
                autoplay=False,
                interactive=False,
            )

        with gr.Tabs():
            with gr.TabItem("Action Vector"):
                action_plot = gr.Plot(label="Current Action", value=init_act)
            with gr.TabItem("State Vector"):
                state_plot = gr.Plot(label="Current State", value=init_state)
            with gr.TabItem("Action Timeline"):
                action_tl = gr.Plot(label="Action Over Episode", value=init_act_tl)
            with gr.TabItem("State Timeline"):
                state_tl = gr.Plot(label="State Over Episode", value=init_state_tl)

        # ── Video Export ──────────────────────────────────────────────

        gr.Markdown("---")
        gr.Markdown("### Export Synchronized Video")
        gr.Markdown(
            "Grid mode starts from a chosen episode and keeps later episodes "
            "advancing until each one ends."
        )

        with gr.Row():
            export_mode = gr.Radio(
                choices=["Single episode", "Episode grid"],
                value="Single episode",
                label="Export mode",
            )
            export_timeline_type = gr.Radio(
                choices=["action", "state"],
                value="action",
                label="Timeline to include",
            )
            export_grid_camera = gr.Dropdown(
                choices=cams,
                value=default_cam_left,
                label="Camera for grid export",
                type="value",
                visible=False,
            )
            export_grid_start_episode = gr.Number(
                value=0,
                precision=0,
                label="Start episode",
                interactive=True,
                visible=False,
            )
            export_grid_count = gr.Slider(
                minimum=1,
                maximum=len(ds.episode_ids),
                value=min(4, len(ds.episode_ids)),
                step=1,
                label="Episodes in grid",
                interactive=True,
                visible=False,
            )
            export_btn = gr.Button("Export Video", variant="primary")

        export_status = gr.Markdown(value="")
        export_progress = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=1,
            label="Export Progress",
            interactive=False,
        )
        export_file = gr.File(label="Download exported video", visible=False)

        def on_export(
            episode_id: int,
            export_mode_value: str,
            timeline_type: str,
            grid_camera: str,
            grid_start_episode: int,
            grid_count: int,
            progress=gr.Progress(),
        ):
            episode_id = int(episode_id)
            out_dir = Path(tempfile.mkdtemp(prefix="teleop_export_"))
            progress_state = {"value": 0.0, "text": "Starting export..."}
            finished = threading.Event()
            result: dict[str, Exception | None] = {"error": None}

            if export_mode_value == "Episode grid":
                start_episode = int(grid_start_episode)
                if start_episode not in ds.episode_ids:
                    raise ValueError(
                        f"Unknown start episode {start_episode}. Available episodes: {ds.episode_ids[0]} to {ds.episode_ids[-1]}"
                    )

                start_idx = ds.episode_ids.index(start_episode)
                count = max(1, int(grid_count))
                selected_episode_ids = ds.episode_ids[start_idx : start_idx + count]
                if not selected_episode_ids:
                    raise RuntimeError("No episodes are available for grid export")

                camera_slug = _camera_slug(grid_camera)
                out_path = out_dir / (
                    f"episodes_{selected_episode_ids[0]}_to_{selected_episode_ids[-1]}_{camera_slug}.mp4"
                )
                total_frames = max(ds.episode_length(ep_id) for ep_id in selected_episode_ids)

                def _run_export() -> None:
                    try:
                        export_episode_grid_video(
                            ds,
                            selected_episode_ids,
                            grid_camera,
                            output_path=out_path,
                            progress_callback=_cb,
                        )
                    except Exception as exc:  # pragma: no cover - surfaced to UI below
                        result["error"] = exc
                    finally:
                        finished.set()

                title = (
                    f"Exporting {len(selected_episode_ids)} episodes from camera {grid_camera}..."
                )
                complete_text = (
                    f"Export complete — **{len(selected_episode_ids)}** episodes written to `{out_path.name}`"
                )
            else:
                out_path = out_dir / f"episode_{episode_id}_{timeline_type}.mp4"
                total_frames = ds.episode_length(episode_id)

                def _run_export() -> None:
                    try:
                        export_episode_video(
                            ds,
                            episode_id,
                            output_path=out_path,
                            data_type=timeline_type,
                            progress_callback=_cb,
                        )
                    except Exception as exc:  # pragma: no cover - surfaced to UI below
                        result["error"] = exc
                    finally:
                        finished.set()

                title = f"Exporting episode {episode_id}..."
                complete_text = (
                    f"Export complete — **{total_frames}** frames written to `{out_path.name}`"
                )

            def _cb(cur: int, total: int) -> None:
                percent = 100.0 * (cur / max(total, 1))
                progress_state["value"] = percent
                progress_state["text"] = f"Rendering frame {cur}/{total}"
                progress(cur / total, desc=progress_state["text"])

            worker = threading.Thread(target=_run_export, daemon=True)
            worker.start()

            while not finished.is_set():
                yield (
                    gr.update(value=title if progress_state["text"] == "Starting export..." else progress_state["text"]),
                    gr.update(value=progress_state["value"]),
                    gr.update(visible=False),
                )
                time.sleep(0.1)

            worker.join()

            if result["error"] is not None:
                raise result["error"]

            yield (
                gr.update(value=complete_text),
                gr.update(value=100),
                gr.update(value=str(out_path), visible=True),
            )

        export_mode.change(
            fn=on_export_mode_change,
            inputs=[export_mode],
            outputs=[
                export_timeline_type,
                export_grid_camera,
                export_grid_start_episode,
                export_grid_count,
            ],
        )

        export_btn.click(
            fn=on_export,
            inputs=[
                episode_dd,
                export_mode,
                export_timeline_type,
                export_grid_camera,
                export_grid_start_episode,
                export_grid_count,
            ],
            outputs=[export_status, export_progress, export_file],
        )

        # ── Wiring ────────────────────────────────────────────────────

        video_outputs = [
            cam0_video,
            cam1_video,
        ]

        render_outputs = [
            action_plot,
            state_plot,
            action_tl,
            state_tl,
            info_md,
        ]

        episode_dd.change(
            fn=on_episode_change,
            inputs=[episode_dd, cam_left_dd, cam_right_dd],
            outputs=[frame_slider, *video_outputs, *render_outputs],
        )

        frame_slider.change(
            fn=on_frame_change,
            inputs=[episode_dd, frame_slider, cam_left_dd, cam_right_dd],
            outputs=[*video_outputs, *render_outputs],
        )

        cam_left_dd.change(
            fn=on_camera_change,
            inputs=[episode_dd, cam_left_dd, cam_right_dd, frame_slider],
            outputs=[*video_outputs, *render_outputs],
        )

        cam_right_dd.change(
            fn=on_camera_change,
            inputs=[episode_dd, cam_left_dd, cam_right_dd, frame_slider],
            outputs=[*video_outputs, *render_outputs],
        )

    return app


def main() -> None:
    """Parse args and launch the Gradio app."""
    parser = argparse.ArgumentParser(description="Teleop Trajectory Visualizer")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to LeRobot v3 dataset root.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.dataset_dir)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        ssr_mode=False,
        allowed_paths=[str(Path(args.dataset_dir).resolve())],
    )


if __name__ == "__main__":
    main()

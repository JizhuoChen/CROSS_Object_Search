#!/usr/bin/env python3
"""Play a Stretch recording with a synchronized top-down robot trajectory.

The viewer reads recordings produced by ``record_stretch_stream.py``.  RGB is
decoded only for the current frame, while the small pose arrays are loaded once
at startup so that the complete trajectory can remain visible during playback.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play Stretch RGB frames beside their synchronized planar trajectory."
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="Recording directory containing meta.json and frames/*.npz.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Displayed frames per second (default: record_fps from meta.json, otherwise 4).",
    )
    parser.add_argument("--start", type=int, default=0, help="First saved-frame index to show.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive saved-frame index at which to stop (default: end of recording).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Show every Nth saved frame while keeping the requested display FPS (default: 1).",
    )
    parser.add_argument(
        "--pose-source",
        choices=("auto", "gps", "camera-pose"),
        default="auto",
        help="Planar pose field (default: GPS/compass, with camera-pose fallback).",
    )
    parser.add_argument(
        "--rgb-convention",
        choices=("legacy-stretch", "standard-jpeg"),
        default="legacy-stretch",
        help=(
            "Use legacy-stretch for recordings whose bridge encoded an RGB array with OpenCV; "
            "use standard-jpeg for normally encoded JPEGs (default: legacy-stretch)."
        ),
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=None,
        help="Resize the RGB and trajectory panels to this height (default: recorded RGB height).",
    )
    parser.add_argument("--loop", action="store_true", help="Restart after the final frame.")
    return parser


def _load_metadata(recording: Path) -> dict:
    path = recording / "meta.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _default_fps(metadata: dict) -> float:
    args = metadata.get("args", {})
    candidates = (
        args.get("record_fps") if isinstance(args, dict) else None,
        metadata.get("saved_fps_effective"),
        4.0,
    )
    for value in candidates:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fps) and fps > 0.0:
            return fps
    return 4.0


def _gps_pose(frame: np.lib.npyio.NpzFile) -> tuple[float, float, float] | None:
    if "gps" not in frame or "compass" not in frame:
        return None
    gps = np.asarray(frame["gps"], dtype=np.float64).reshape(-1)
    compass = np.asarray(frame["compass"], dtype=np.float64).reshape(-1)
    if gps.size < 2 or compass.size < 1:
        return None
    pose = np.array((gps[0], gps[1], compass[0]), dtype=np.float64)
    if not np.all(np.isfinite(pose)):
        return None
    return float(pose[0]), float(pose[1]), float(pose[2])


def _camera_pose(frame: np.lib.npyio.NpzFile) -> tuple[float, float, float] | None:
    if "camera_pose" not in frame:
        return None
    matrix = np.asarray(frame["camera_pose"], dtype=np.float64)
    if matrix.size != 16:
        return None
    matrix = matrix.reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        return None
    # The Stretch optical frame points along +Z, so its projected +Z direction
    # gives the planar heading in the world frame.
    yaw = math.atan2(float(matrix[1, 2]), float(matrix[0, 2]))
    return float(matrix[0, 3]), float(matrix[1, 3]), yaw


def _read_pose(
    frame: np.lib.npyio.NpzFile, source: str
) -> tuple[tuple[float, float, float] | None, str | None]:
    if source in ("auto", "gps"):
        pose = _gps_pose(frame)
        if pose is not None:
            return pose, "gps"
        if source == "gps":
            return None, None
    pose = _camera_pose(frame)
    return (pose, "camera-pose") if pose is not None else (None, None)


def load_trajectory(
    frame_paths: list[Path], pose_source: str
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    positions = np.full((len(frame_paths), 2), np.nan, dtype=np.float64)
    yaws = np.full(len(frame_paths), np.nan, dtype=np.float64)
    sources: list[str | None] = []
    for index, path in enumerate(frame_paths):
        try:
            with np.load(path, allow_pickle=False) as frame:
                pose, used_source = _read_pose(frame, pose_source)
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not read pose from {path.name}: {exc}")
            pose, used_source = None, None
        sources.append(used_source)
        if pose is not None:
            positions[index] = pose[:2]
            yaws[index] = pose[2]
    return positions, yaws, sources


def _nice_distance(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10.0**exponent)
    nice_fraction = 1.0 if fraction < 1.5 else 2.0 if fraction < 3.5 else 5.0
    return nice_fraction * (10.0**exponent)


def _trajectory_transform(positions: np.ndarray, size: int):
    valid = np.all(np.isfinite(positions), axis=1)
    padding = max(60, int(size * 0.10))
    if not np.any(valid):
        center = np.zeros(2, dtype=np.float64)
        world_span = 1.0
    else:
        lower = np.min(positions[valid], axis=0)
        upper = np.max(positions[valid], axis=0)
        center = (lower + upper) / 2.0
        world_span = max(float(np.max(upper - lower)), 0.5)
    scale = max(1.0, (size - 2 * padding) / world_span)

    def project(point: np.ndarray) -> tuple[int, int]:
        pixel_x = size / 2.0 + (float(point[0]) - center[0]) * scale
        pixel_y = size / 2.0 - (float(point[1]) - center[1]) * scale
        return int(round(pixel_x)), int(round(pixel_y))

    return project, scale, valid


def _draw_segments(
    panel: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    project,
    last_index: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for index in range(1, min(last_index + 1, len(positions))):
        if valid[index - 1] and valid[index]:
            cv2.line(panel, project(positions[index - 1]), project(positions[index]), color, thickness, cv2.LINE_AA)


def render_trajectory(
    positions: np.ndarray,
    yaws: np.ndarray,
    current_index: int,
    size: int,
) -> np.ndarray:
    panel = np.full((size, size, 3), (29, 31, 35), dtype=np.uint8)
    project, pixels_per_meter, valid = _trajectory_transform(positions, size)

    # Full route, then the portion already traversed.
    _draw_segments(panel, positions, valid, project, len(positions) - 1, (100, 104, 110), 2)
    _draw_segments(panel, positions, valid, project, current_index, (0, 190, 255), 3)

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        start = int(valid_indices[0])
        finish = int(valid_indices[-1])
        cv2.circle(panel, project(positions[start]), 7, (80, 220, 80), -1, cv2.LINE_AA)
        cv2.circle(panel, project(positions[finish]), 7, (190, 190, 190), 2, cv2.LINE_AA)

    if 0 <= current_index < len(positions) and valid[current_index]:
        current = project(positions[current_index])
        cv2.circle(panel, current, 8, (30, 30, 240), -1, cv2.LINE_AA)
        if np.isfinite(yaws[current_index]):
            arrow_pixels = max(25.0, min(55.0, size * 0.08))
            yaw = float(yaws[current_index])
            tip = (
                int(round(current[0] + arrow_pixels * math.cos(yaw))),
                int(round(current[1] - arrow_pixels * math.sin(yaw))),
            )
            cv2.arrowedLine(panel, current, tip, (30, 30, 240), 3, cv2.LINE_AA, tipLength=0.28)

    cv2.putText(panel, "Robot trajectory", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, "full", (18, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 154, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "travelled", (78, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 1, cv2.LINE_AA)

    # A scale bar remains meaningful even when the route does not start at (0, 0).
    scale_m = _nice_distance((size * 0.18) / pixels_per_meter)
    scale_px = max(1, int(round(scale_m * pixels_per_meter)))
    left, baseline = 18, size - 26
    cv2.line(panel, (left, baseline), (left + scale_px, baseline), (225, 225, 225), 2, cv2.LINE_AA)
    cv2.line(panel, (left, baseline - 5), (left, baseline + 5), (225, 225, 225), 2)
    cv2.line(panel, (left + scale_px, baseline - 5), (left + scale_px, baseline + 5), (225, 225, 225), 2)
    label = f"{scale_m:g} m"
    cv2.putText(panel, label, (left, baseline - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)

    if 0 <= current_index < len(positions) and valid[current_index]:
        x, y = positions[current_index]
        yaw_text = "n/a" if not np.isfinite(yaws[current_index]) else f"{math.degrees(yaws[current_index]):.1f} deg"
        text = f"x {x:.2f} m   y {y:.2f} m   yaw {yaw_text}"
    else:
        text = "pose unavailable"
    cv2.putText(panel, text, (18, size - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
    return panel


def load_rgb(path: Path, convention: str) -> tuple[np.ndarray, float | None]:
    with np.load(path, allow_pickle=False) as frame:
        if "rgb_jpg" not in frame:
            raise KeyError(f"{path} contains no rgb_jpg")
        blob = np.asarray(frame["rgb_jpg"], dtype=np.uint8).reshape(-1)
        recv = np.asarray(frame["recv_time"], dtype=np.float64).reshape(-1) if "recv_time" in frame else np.empty(0)
    image = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode RGB payload in {path}")
    if convention == "legacy-stretch":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    recv_time = float(recv[0]) if recv.size and np.isfinite(recv[0]) else None
    return image, recv_time


def _overlay_frame_text(
    image: np.ndarray,
    frame_name: str,
    display_number: int,
    display_count: int,
    elapsed_sec: float | None,
    fps: float,
    source: str | None,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 67), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0, image)
    elapsed = "n/a" if elapsed_sec is None else f"{elapsed_sec:.2f} s"
    line_one = f"Frame {frame_name}  ({display_number + 1}/{display_count})"
    line_two = f"time {elapsed}   playback {fps:g} FPS   pose {source or 'unavailable'}"
    cv2.putText(image, line_one, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, line_two, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (220, 220, 220), 1, cv2.LINE_AA)


def _validate_args(args: argparse.Namespace, frame_count: int) -> None:
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0.0):
        raise ValueError("--fps must be a finite value greater than zero")
    if args.stride <= 0:
        raise ValueError("--stride must be greater than zero")
    if args.start < 0:
        raise ValueError("--start cannot be negative")
    if args.end is not None and args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if args.display_height is not None and args.display_height < 240:
        raise ValueError("--display-height must be at least 240 pixels")
    if args.start >= frame_count:
        raise ValueError(f"--start {args.start} is outside this {frame_count}-frame recording")


def main() -> int:
    args = build_parser().parse_args()
    recording = args.recording.expanduser().resolve()
    frames_dir = recording / "frames"
    if not frames_dir.is_dir():
        raise SystemExit(f"Recording has no frames directory: {frames_dir}")
    all_paths = sorted(frames_dir.glob("*.npz"))
    if not all_paths:
        raise SystemExit(f"No .npz frames found in {frames_dir}")

    try:
        _validate_args(args, len(all_paths))
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc
    stop = min(args.end if args.end is not None else len(all_paths), len(all_paths))
    frame_paths = all_paths[args.start : stop : args.stride]
    if not frame_paths:
        raise SystemExit("The selected frame range is empty")

    metadata = _load_metadata(recording)
    fps = float(args.fps) if args.fps is not None else _default_fps(metadata)
    positions, yaws, pose_sources = load_trajectory(frame_paths, args.pose_source)
    valid_pose_count = int(np.count_nonzero(np.all(np.isfinite(positions), axis=1)))
    if valid_pose_count == 0:
        print(f"[WARN] No valid {args.pose_source} poses found; the trajectory panel will be empty.")

    print(
        f"Playing {recording.name}: {len(frame_paths)} frames at {fps:g} FPS; "
        f"valid poses {valid_pose_count}/{len(frame_paths)}"
    )
    print("Controls: Space pause/resume | N/Right next | P/Left previous | Q/Esc quit")

    window_name = f"Stretch recording - {recording.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    index = 0
    paused = False
    first_recv_time: float | None = None
    redraw = True

    try:
        while True:
            if redraw:
                started = time.monotonic()
                try:
                    rgb, recv_time = load_rgb(frame_paths[index], args.rgb_convention)
                except (OSError, KeyError, ValueError) as exc:
                    print(f"[WARN] Skipping {frame_paths[index].name}: {exc}")
                    index += 1
                    if index >= len(frame_paths):
                        if args.loop:
                            index = 0
                        else:
                            break
                    continue
                if first_recv_time is None and recv_time is not None:
                    first_recv_time = recv_time
                elapsed = None if recv_time is None or first_recv_time is None else recv_time - first_recv_time

                display_height = args.display_height or int(rgb.shape[0])
                if rgb.shape[0] != display_height:
                    width = max(1, int(round(rgb.shape[1] * display_height / rgb.shape[0])))
                    interpolation = cv2.INTER_AREA if display_height < rgb.shape[0] else cv2.INTER_LINEAR
                    rgb = cv2.resize(rgb, (width, display_height), interpolation=interpolation)
                _overlay_frame_text(
                    rgb,
                    frame_paths[index].stem,
                    index,
                    len(frame_paths),
                    elapsed,
                    fps,
                    pose_sources[index],
                )
                trajectory = render_trajectory(positions, yaws, index, display_height)
                composite = np.concatenate((rgb, trajectory), axis=1)
                cv2.imshow(window_name, composite)
                redraw = False
            else:
                started = time.monotonic()

            if paused:
                delay_ms = 30
            else:
                remaining = (1.0 / fps) - (time.monotonic() - started)
                delay_ms = max(1, int(math.ceil(remaining * 1000.0)))
            key = cv2.waitKey(delay_ms)

            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key in (ord("n"), ord("N"), 83, 2555904):
                paused = True
                index = min(index + 1, len(frame_paths) - 1)
                redraw = True
                continue
            if key in (ord("p"), ord("P"), 81, 2424832):
                paused = True
                index = max(index - 1, 0)
                redraw = True
                continue
            if paused:
                continue

            index += 1
            if index >= len(frame_paths):
                if args.loop:
                    index = 0
                    first_recv_time = None
                else:
                    break
            redraw = True
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

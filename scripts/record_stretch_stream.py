#!/usr/bin/env python3
"""Record the Stretch ZMQ RGB-D stream for reproducible CROSS experiments.

The output is deliberately compatible with the existing FATSS recorder:

.. code-block:: text

    <out-dir>/meta.json
    <out-dir>/frames/00000000.npz

Frame archives contain ``rgb_jpg``, ``depth_jp2``, ``gps``, ``compass``,
``camera_K``, optional ``camera_pose``, local ``recv_time``, and the original
bridge ``step`` (the last action ID, not a frame counter). An optional independently supplied
``ground_truth_pose`` is preserved when present. Each frame is written to a temporary file and
atomically renamed, so an interrupted recording exposes either a complete
archive or no archive at that index.

PyZMQ is imported only when a live client is constructed.  Consequently,
``--help`` and imports used by offline tools work without robot dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_u8_blob(value: Any, name: str) -> np.ndarray:
    """Copy a bytes-like compressed payload into a one-dimensional uint8 array."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        array = np.frombuffer(value, dtype=np.uint8).copy()
    else:
        array = np.asarray(value, dtype=np.uint8).reshape(-1).copy()
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _as_pose_matrix(
    value: Any,
    *,
    field: str = "camera_pose",
) -> np.ndarray | None:
    """Normalize an optional streamed pose to a finite homogeneous matrix."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    if array.shape != (4, 4):
        if array.size != 16:
            raise ValueError(f"{field} must be 4x4; got shape {array.shape}")
        array = array.reshape(4, 4)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} contains non-finite values")
    if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{field} is not a homogeneous transform")
    return array.copy()


def _finite_vector(value: Any, name: str, minimum_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < minimum_size:
        raise ValueError(
            f"{name} needs at least {minimum_size} values; got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def _camera_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 9:
        array = array.reshape(3, 3)
    if array.shape != (3, 3):
        raise ValueError(f"camera_K must be 3x3; got shape {array.shape}")
    if (
        not np.all(np.isfinite(array))
        or array[0, 0] <= 0.0
        or array[1, 1] <= 0.0
    ):
        raise ValueError("camera_K contains invalid intrinsics")
    return array.copy()


def _validate_stream_message(message: dict[str, Any]) -> dict[str, Any]:
    """Decode and validate one bridge packet before an acquisition starts.

    ZMQ ``connect()`` is asynchronous, so constructing a SUB socket does not
    establish that a publisher is reachable.  This check also prevents a
    syntactically nonempty but undecodable image payload from creating an
    apparently valid recording directory.
    """

    if not isinstance(message, dict):
        raise ValueError(f"stream packet must be a dictionary, got {type(message).__name__}")
    if "rgb" not in message:
        raise ValueError("stream message has no rgb payload")
    if "depth" not in message:
        raise ValueError("stream message has no depth payload")
    if "camera_K" not in message:
        raise ValueError("stream message has no camera_K")

    rgb_blob = _to_u8_blob(message["rgb"], "rgb")
    depth_blob = _to_u8_blob(message["depth"], "depth")
    camera_k = _camera_matrix(message["camera_K"])
    _finite_vector(message.get("gps", []), "gps", 2)
    _finite_vector(message.get("compass", []), "compass", 1)
    camera_pose = _as_pose_matrix(message.get("camera_pose"))

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - deployment dependent
        raise RuntimeError(
            "Live Stretch recording preflight requires OpenCV. Install it with "
            "`python -m pip install opencv-python`."
        ) from exc

    rgb = cv2.imdecode(rgb_blob, cv2.IMREAD_COLOR)
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        shape = None if rgb is None else tuple(rgb.shape)
        raise ValueError(f"rgb payload did not decode to HxWx3; got {shape}")
    depth = cv2.imdecode(depth_blob, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        shape = None if depth is None else tuple(depth.shape)
        raise ValueError(f"depth payload did not decode to HxW; got {shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            "decoded RGB/depth dimensions differ: "
            f"RGB={tuple(rgb.shape)}, depth={tuple(depth.shape)}"
        )

    height, width = map(int, depth.shape)
    return {
        "message_keys": sorted(str(key) for key in message),
        "rgb_width": width,
        "rgb_height": height,
        "rgb_decoded_dtype": str(rgb.dtype),
        "depth_decoded_dtype": str(depth.dtype),
        "camera_K": camera_k.tolist(),
        "camera_pose_present": camera_pose is not None,
        "bridge_action_step": _publisher_step(message),
    }


def _receive_startup_packet(
    client: "StretchZmqRawClient",
    *,
    timeout_ms: int,
    startup_timeout_sec: float,
) -> tuple[dict[str, Any], float, float, dict[str, Any], int]:
    """Wait for one decodable packet without touching the output directory."""

    started = time.monotonic()
    deadline = started + float(startup_timeout_sec)
    rejected_count = 0
    last_error: Exception | None = None

    while True:
        remaining_sec = deadline - time.monotonic()
        if remaining_sec <= 0.0:
            detail = (
                ""
                if last_error is None
                else f" Last rejected packet: {type(last_error).__name__}: {last_error}"
            )
            raise TimeoutError(
                f"No valid Stretch RGB-D packet received from {client.addr} within "
                f"{float(startup_timeout_sec):.1f} seconds.{detail}"
            )

        poll_ms = max(1, min(int(timeout_ms), int(np.ceil(remaining_sec * 1000.0))))
        message = client.recv_raw(timeout_ms=poll_ms)
        if message is None:
            continue
        receive_wall = time.time()
        receive_monotonic = time.monotonic()
        try:
            probe = _validate_stream_message(message)
        except Exception as exc:
            rejected_count += 1
            last_error = exc
            print(
                "[WARN] rejected startup packet "
                f"{rejected_count}: {type(exc).__name__}: {exc}"
            )
            continue
        probe["waited_sec"] = float(receive_monotonic - started)
        return (
            message,
            receive_wall,
            receive_monotonic,
            probe,
            rejected_count,
        )


def _publisher_step(message: dict[str, Any]) -> int:
    value = message.get("step", message.get("publisher_step", -1))
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return -1
    return int(array[0])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON manifest and fsync its temporary file."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_frame(
    path: Path,
    *,
    message: dict[str, Any],
    recv_time: float,
) -> None:
    """Validate and atomically save one FATSS-compatible frame archive."""

    if "rgb" not in message:
        raise ValueError("stream message has no rgb payload")
    if "depth" not in message:
        raise ValueError("stream message has no depth payload")
    if "camera_K" not in message:
        raise ValueError("stream message has no camera_K")

    rgb_blob = _to_u8_blob(message["rgb"], "rgb")
    depth_blob = _to_u8_blob(message["depth"], "depth")
    gps = _finite_vector(message.get("gps", []), "gps", 2)
    compass = _finite_vector(message.get("compass", []), "compass", 1)
    camera_k = _camera_matrix(message["camera_K"])
    camera_pose = _as_pose_matrix(message.get("camera_pose"))
    if camera_pose is None:
        camera_pose = np.empty((0,), dtype=np.float64)
    ground_truth_pose = _as_pose_matrix(
        message.get("ground_truth_pose"),
        field="ground_truth_pose",
    )
    if ground_truth_pose is None:
        ground_truth_pose = np.empty((0,), dtype=np.float64)
    step = _publisher_step(message)

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                rgb_jpg=rgb_blob,
                depth_jp2=depth_blob,
                gps=gps,
                compass=compass,
                camera_K=camera_k,
                camera_pose=camera_pose,
                ground_truth_pose=ground_truth_pose,
                recv_time=np.array([float(recv_time)], dtype=np.float64),
                # ``step`` is the canonical FATSS key.  The explicit alias
                # makes its acquisition provenance self-documenting.
                step=np.array([step], dtype=np.int64),
                publisher_step=np.array([step], dtype=np.int64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class StretchZmqRawClient:
    """Minimal SUB client for the Stretch observation publisher.

    PyZMQ is required only when this constructor is called.  A dedicated
    context is used so closing the recorder cannot terminate a context owned by
    another application in the same process.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        recv_port: int = 4401,
        local: bool = False,
        conflate: bool = True,
    ) -> None:
        if not local and not str(robot_ip).strip():
            raise ValueError("robot_ip is required unless local=True")
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment dependent
            raise RuntimeError(
                "Live Stretch recording requires pyzmq. Install it with "
                "`python -m pip install pyzmq`."
            ) from exc

        host = "127.0.0.1" if local else str(robot_ip).strip()
        self.addr = f"tcp://{host}:{int(recv_port)}"
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.SNDHWM, 1000)
        self._socket.setsockopt(zmq.RCVHWM, 1000)
        self._socket.setsockopt(zmq.CONFLATE, 1 if conflate else 0)
        self._socket.connect(self.addr)
        self._closed = False

    def recv_raw(self, timeout_ms: int = 2000) -> dict[str, Any] | None:
        """Receive one dictionary, returning ``None`` on timeout/non-data input."""

        if self._closed:
            return None
        if self._socket.poll(timeout=int(timeout_ms)) == 0:
            return None
        message = self._socket.recv_pyobj()
        return message if isinstance(message, dict) else None

    def close(self) -> None:
        """Close the socket promptly and release its private context."""

        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        self._context.term()


def _prepare_output(out_dir: Path, *, overwrite: bool) -> Path:
    """Create the recording directory without silently replacing prior data."""

    out_dir = out_dir.expanduser().resolve()
    frames_dir = out_dir / "frames"
    meta_path = out_dir / "meta.json"

    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {out_dir}")

    existing_entries = list(out_dir.iterdir()) if out_dir.is_dir() else []
    if existing_entries and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. "
            "Pass --overwrite to replace prior recording data."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Delete only files owned by this recorder.  Unexpected files are left
        # untouched even when overwrite was explicitly requested.
        for pattern in ("*.npz", ".*.npz.tmp-*", "*.tmp"):
            for old_path in frames_dir.glob(pattern):
                if old_path.is_file():
                    old_path.unlink()
        if meta_path.is_file():
            meta_path.unlink()
        for temporary in out_dir.glob(".meta.json.tmp-*"):
            if temporary.is_file():
                temporary.unlink()
    return frames_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; separated for unit tests and reuse."""

    parser = argparse.ArgumentParser(
        description="Record a Stretch RGB-D ZMQ stream for CROSS offline replay"
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.0.102",
        help="Stretch robot IP (ignored with --local)",
    )
    parser.add_argument(
        "--recv-port",
        type=int,
        default=4401,
        help="Stretch observation publisher port",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Connect to 127.0.0.1 instead of --robot-ip",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="New recording directory",
    )
    parser.add_argument(
        "--sequence-id",
        default="",
        help="Stable sequence identifier (defaults to the output directory name)",
    )
    parser.add_argument(
        "--role",
        default="unspecified",
        help="Sequence role, for example 'mapping' or 'test'",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Stop after N saved frames; a negative value is unlimited",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=-1.0,
        help="Stop after N wall-clock seconds; a negative value is unlimited",
    )
    parser.add_argument(
        "--record-fps",
        "--fps",
        dest="record_fps",
        type=float,
        default=4.0,
        help="Maximum saved FPS; 0 saves every received message",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=2000,
        help="Receive polling timeout (also bounds graceful-stop latency)",
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=15.0,
        help=(
            "Fail before creating --out-dir unless a valid RGB-D packet is "
            "received and decoded within this many seconds"
        ),
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=50,
        help="Print counters every N saved frames",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help="Optional sentinel path whose existence requests a graceful stop",
    )
    parser.add_argument(
        "--conflate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the newest queued ZMQ observation (use --no-conflate to keep all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace prior recorder-owned data in --out-dir",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate recorder command-line arguments."""

    args = build_parser().parse_args(argv)
    if args.recv_port <= 0 or args.recv_port > 65535:
        raise ValueError("--recv-port must be between 1 and 65535")
    if args.max_frames == 0:
        raise ValueError("--max-frames must be positive or negative for unlimited")
    if args.duration_sec == 0:
        raise ValueError("--duration-sec must be positive or negative for unlimited")
    if args.record_fps < 0:
        raise ValueError("--record-fps must be >= 0")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be > 0")
    if args.startup_timeout_sec <= 0:
        raise ValueError("--startup-timeout-sec must be > 0")
    if args.status_every <= 0:
        raise ValueError("--status-every must be > 0")
    if not str(args.role).strip():
        raise ValueError("--role must not be empty")
    return args


def record_stream(
    args: argparse.Namespace,
    *,
    client_factory: Callable[..., StretchZmqRawClient] = StretchZmqRawClient,
) -> dict[str, Any]:
    """Run one recording session and return the final manifest.

    ``client_factory`` is injectable so format and shutdown behavior can be
    tested without a robot or PyZMQ.
    """

    out_dir = Path(args.out_dir).expanduser().resolve()
    meta_path = out_dir / "meta.json"
    stop_file = (
        Path(args.stop_file).expanduser().resolve() if str(args.stop_file) else None
    )
    sequence_id = (
        str(args.sequence_id).strip()
        if str(args.sequence_id).strip()
        else out_dir.name
    )
    role = str(args.role).strip()

    # ZMQ connect is asynchronous. Receive and decode a real packet before
    # creating or replacing any output data.
    client = client_factory(
        robot_ip=args.robot_ip,
        recv_port=int(args.recv_port),
        local=bool(args.local),
        conflate=bool(args.conflate),
    )
    try:
        (
            startup_message,
            startup_receive_wall,
            startup_receive_monotonic,
            startup_probe,
            startup_rejected_count,
        ) = _receive_startup_packet(
            client,
            timeout_ms=int(args.timeout_ms),
            startup_timeout_sec=float(args.startup_timeout_sec),
        )
        frames_dir = _prepare_output(out_dir, overwrite=bool(args.overwrite))
        start_wall = time.time()
        start_monotonic = time.monotonic()
        manifest: dict[str, Any] = {
            "format_version": 1,
            "producer": "CROSS/scripts/record_stretch_stream.py",
            "created_utc": _utc_now(),
            "source_addr": client.addr,
            "frame_files_pattern": "frames/%08d.npz",
            "sequence_id": sequence_id,
            "sequence_role": role,
            "role": role,
            "state": "recording",
            "args": {
                key: value
                for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
            "stream": {
                "conflate": bool(args.conflate),
                "record_fps": float(args.record_fps),
                "rgb_encoding": "jpeg",
                "depth_encoding": "jpeg2000",
                "depth_storage_unit": "millimetre",
                "recv_time_clock": "unix_wall_seconds_at_recorder",
                "publisher_step_key": "step",
                "publisher_step_semantics": "last_bridge_action_id_not_frame_counter",
            },
            "startup_probe": {
                **startup_probe,
                "rejected_packet_count": int(startup_rejected_count),
            },
        }
        _atomic_write_json(meta_path, manifest)
    except BaseException:
        client.close()
        raise

    print(f"Validated observation stream: {client.addr}")
    print(
        "First packet decoded: "
        f"{startup_probe['rgb_width']}x{startup_probe['rgb_height']} RGB-D, "
        f"camera_pose={'yes' if startup_probe['camera_pose_present'] else 'no'}"
    )
    print(f"Recording sequence {sequence_id!r} (role={role!r}) to: {out_dir}")
    if args.record_fps > 0:
        print(f"Save rate limit: {float(args.record_fps):.3f} FPS")
    else:
        print("Save rate limit: disabled; saving every received message")
    if stop_file is not None:
        print(f"Graceful stop file: {stop_file}")
    print("Press Ctrl+C once to stop gracefully.")

    frame_count = 0
    received_count = 0
    skipped_rate_count = 0
    dropped_count = 0
    last_saved_monotonic: float | None = None
    last_status_monotonic = start_monotonic
    stop_requested = False
    force_stop = False
    stop_reason = "completed"
    failure: BaseException | None = None
    completion_error: RuntimeError | None = None
    pending_message: tuple[dict[str, Any], float, float] | None = (
        startup_message,
        startup_receive_wall,
        startup_receive_monotonic,
    )

    def on_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, force_stop, stop_reason
        if not stop_requested:
            stop_requested = True
            stop_reason = f"signal_{signum}"
            print(f"\nReceived signal {signum}; requesting graceful stop...")
        else:
            force_stop = True
            print(f"\nReceived signal {signum} again; forcing stop.")

    install_signals = threading.current_thread() is threading.main_thread()
    old_sigint: Any = None
    old_sigterm: Any = None
    if install_signals:
        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

    try:
        while not force_stop and not stop_requested:
            elapsed_monotonic = time.monotonic() - start_monotonic
            if stop_file is not None and stop_file.exists():
                stop_requested = True
                stop_reason = "stop_file"
                print(f"Stop file detected: {stop_file}")
                break
            if args.max_frames > 0 and frame_count >= int(args.max_frames):
                stop_reason = "max_frames"
                break
            if args.duration_sec > 0 and elapsed_monotonic >= float(args.duration_sec):
                stop_reason = "duration"
                break

            if pending_message is not None:
                message, receive_wall, receive_monotonic = pending_message
                pending_message = None
            else:
                message = client.recv_raw(timeout_ms=int(args.timeout_ms))
                if message is None:
                    continue
                receive_wall = time.time()
                receive_monotonic = time.monotonic()
            received_count += 1

            if args.record_fps > 0 and last_saved_monotonic is not None:
                minimum_interval = 1.0 / float(args.record_fps)
                if receive_monotonic - last_saved_monotonic < minimum_interval:
                    skipped_rate_count += 1
                    continue

            frame_path = frames_dir / f"{frame_count:08d}.npz"
            try:
                _atomic_write_frame(
                    frame_path,
                    message=message,
                    recv_time=receive_wall,
                )
            except Exception as exc:
                dropped_count += 1
                print(
                    f"[WARN] dropped received message {received_count}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            frame_count += 1
            last_saved_monotonic = receive_monotonic
            if frame_count % int(args.status_every) == 0:
                now = time.monotonic()
                window_seconds = max(1e-9, now - last_status_monotonic)
                print(
                    f"Saved={frame_count} "
                    f"window_fps={float(args.status_every) / window_seconds:.2f} "
                    f"received={received_count} "
                    f"skipped_rate={skipped_rate_count} "
                    f"dropped={dropped_count}"
                )
                last_status_monotonic = now
    except KeyboardInterrupt:
        stop_requested = True
        stop_reason = "keyboard_interrupt"
        print("\nKeyboardInterrupt received; finalizing recording...")
    except BaseException as exc:
        failure = exc
        stop_reason = f"error:{type(exc).__name__}"
        raise
    finally:
        try:
            client.close()
        finally:
            if install_signals:
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)

            end_wall = time.time()
            elapsed = max(1e-9, end_wall - start_wall)
            if failure is not None:
                state = "failed"
            elif frame_count == 0:
                state = "failed"
                stop_reason = "error:no_frames_saved"
                completion_error = RuntimeError(
                    "The Stretch stream passed startup validation, but no frames "
                    "were saved; inspect recorder warnings and meta.json."
                )
            elif stop_reason.startswith("signal_") or stop_reason == "keyboard_interrupt":
                state = "interrupted"
            elif stop_reason == "stop_file":
                state = "stopped"
            else:
                state = "completed"
            manifest.update(
                {
                    "finished_utc": _utc_now(),
                    "state": state,
                    "stop_reason": stop_reason,
                    "frame_count": frame_count,
                    "received_count": received_count,
                    "skipped_rate_count": skipped_rate_count,
                    "dropped_count": dropped_count,
                    "duration_sec": elapsed,
                    "saved_fps_effective": float(frame_count) / elapsed,
                    "received_fps_effective": float(received_count) / elapsed,
                }
            )
            if failure is not None:
                manifest["error"] = f"{type(failure).__name__}: {failure}"
            elif completion_error is not None:
                manifest["error"] = (
                    f"{type(completion_error).__name__}: {completion_error}"
                )
            _atomic_write_json(meta_path, manifest)

    print(
        "Done. "
        f"Saved={frame_count}, Received={received_count}, "
        f"SkippedByRate={skipped_rate_count}, Dropped={dropped_count}, "
        f"SavedFPS={manifest['saved_fps_effective']:.2f}"
    )
    if completion_error is not None:
        raise completion_error
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    record_stream(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

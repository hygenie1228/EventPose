#!/usr/bin/env python3
"""Warp an RGB-D RGB frame into its paired Event-camera image plane.

This is a standalone example for a delivered Busan26 5 FPS H5 session.  It
reads compressed images, corrected timestamps, synchronization tables, and
calibration directly from the H5 files; it does not import DVS_SDK modules.

The delivered images are already undistorted.  Therefore this program uses
``calibration/*/intrinsic_matrix`` directly and deliberately does not apply a
second undistortion.

Example:

    python3 warp_rgbd_h5_to_event.py \
        --session /DVS/Dataset/20260720_131207_863008_200_b \
        --device 0 \
        --sync-index 0 \
        --output-dir /DVS/Dataset/rgbd_to_event_example

Requirements: Python 3, numpy, h5py, opencv-python (or system OpenCV Python).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


# BGR palettes: old Event frames are dark; recent frames are bright.
POSITIVE_PALETTE = np.asarray(
    [(0, 0, 90), (20, 20, 170), (20, 80, 235), (0, 155, 255), (0, 255, 255)],
    dtype=np.uint8,
)
NEGATIVE_PALETTE = np.asarray(
    [(90, 25, 0), (185, 65, 0), (255, 130, 0), (255, 205, 0), (255, 255, 0)],
    dtype=np.uint8,
)
BIN_OPACITY = np.asarray([0.28, 0.43, 0.60, 0.78, 0.96], dtype=np.float32)


def decode_frame(stream: h5py.Group, position: int) -> np.ndarray:
    """Decode one JPEG/PNG frame without loading the full H5 byte array."""
    count = len(stream["offsets"]) - 1
    if position < 0 or position >= count:
        raise IndexError(f"frame position {position} is outside [0, {count})")
    start = int(stream["offsets"][position])
    end = int(stream["offsets"][position + 1])
    encoded = np.asarray(stream["encoded_data"][start:end], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to decode H5 frame at position {position}")
    return image


def require_undistorted(stream: h5py.Group, label: str) -> None:
    if not bool(stream.attrs.get("images_are_undistorted", False)):
        raise RuntimeError(
            f"{label} is not marked as undistorted; this example must not be "
            "used on raw/distorted images"
        )


def find_sync_row(sync_group: h5py.Group, requested: int) -> tuple[int, int]:
    sync_indices = np.asarray(sync_group["sync_index"][:], dtype=np.int64)
    if len(sync_indices) == 0:
        raise RuntimeError("empty synchronization table")
    if requested < 0:
        row = len(sync_indices) + requested
        if row < 0:
            raise IndexError(f"negative sync index {requested} is out of range")
        return row, int(sync_indices[row])
    matches = np.flatnonzero(sync_indices == requested)
    if len(matches) != 1:
        raise KeyError(f"sync_index={requested} was not found exactly once")
    return int(matches[0]), requested


def event_positions_for_sync(
    event_h5: h5py.File, sync_index: int, event_count: int
) -> np.ndarray:
    """Return the exact Event positions associated with one sync anchor."""
    sync = event_h5["synchronization"]
    row, _ = find_sync_row(sync, sync_index)
    offsets = sync["anchor_event_offsets"]
    positions = sync["anchor_event_positions"]
    start = int(offsets[row])
    end = int(offsets[row + 1])
    selected = np.asarray(positions[start:end], dtype=np.int64)
    if len(selected) == 0:
        raise RuntimeError(f"sync_index={sync_index} has no associated Event frames")
    if event_count > 0:
        selected = selected[-event_count:]
    timestamps = event_h5["streams/images/timestamp_ns"]
    order = np.argsort(np.asarray(timestamps[selected], dtype=np.int64))
    return selected[order]


def fill_small_gaps(canvas: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return canvas
    mask = np.any(canvas != 0, axis=2).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(canvas, kernel)
    needs_fill = (closed > 0) & (mask == 0)
    canvas[needs_fill] = dilated[needs_fill]
    return canvas


def warp_rgbd_to_event(
    rgb: np.ndarray,
    depth: np.ndarray,
    K_rgb: np.ndarray,
    K_event: np.ndarray,
    R_rgb_to_event: np.ndarray,
    t_rgb_to_event_m: np.ndarray,
    event_shape: tuple[int, int],
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
    fill_kernel: int,
) -> tuple[np.ndarray, dict]:
    """Back-project RGB pixels with aligned depth, transform, and re-project."""
    if rgb.shape[:2] != depth.shape[:2]:
        raise RuntimeError(
            f"RGB/depth_align shape mismatch: {rgb.shape[:2]} vs {depth.shape[:2]}"
        )
    height, width = depth.shape[:2]
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    rows = rows.reshape(-1)
    columns = columns.reshape(-1)
    depth_m = depth[rows, columns].astype(np.float64) * depth_scale
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    rows, columns, depth_m = rows[valid], columns[valid], depth_m[valid]
    colors = rgb[rows, columns]

    fx, fy = float(K_rgb[0, 0]), float(K_rgb[1, 1])
    cx, cy = float(K_rgb[0, 2]), float(K_rgb[1, 2])
    points_rgb = np.column_stack(
        ((columns - cx) * depth_m / fx, (rows - cy) * depth_m / fy, depth_m)
    )
    points_event = points_rgb @ R_rgb_to_event.T + t_rgb_to_event_m.reshape(3)

    z = points_event[:, 2]
    forward = z > 1e-6
    points_event, colors, z = points_event[forward], colors[forward], z[forward]
    efx, efy = float(K_event[0, 0]), float(K_event[1, 1])
    ecx, ecy = float(K_event[0, 2]), float(K_event[1, 2])
    target_x = np.rint(efx * points_event[:, 0] / z + ecx).astype(np.int64)
    target_y = np.rint(efy * points_event[:, 1] / z + ecy).astype(np.int64)
    event_height, event_width = event_shape
    inside = (
        (target_x >= 0)
        & (target_x < event_width)
        & (target_y >= 0)
        & (target_y < event_height)
    )
    target_x, target_y, z, colors = (
        target_x[inside], target_y[inside], z[inside], colors[inside]
    )

    # Z-buffer: retain the nearest RGB-D point at each Event pixel.
    z_buffer = np.full((event_height, event_width), np.inf, dtype=np.float64)
    if len(z):
        np.minimum.at(z_buffer, (target_y, target_x), z)
    warped = np.zeros((event_height, event_width, 3), dtype=np.uint8)
    if len(z):
        nearest = z <= z_buffer[target_y, target_x] + 1e-9
        warped[target_y[nearest], target_x[nearest]] = colors[nearest]
    coverage = float(np.count_nonzero(np.isfinite(z_buffer)))
    coverage = coverage / float(event_height * event_width) * 100.0
    warped = fill_small_gaps(warped, fill_kernel)
    return warped, {
        "method": "undistorted_rgbd_depth_align_3d_to_undistorted_event",
        "depth_scale_m_per_unit": depth_scale,
        "depth_range_m": [min_depth_m, max_depth_m],
        "projection_stride": stride,
        "valid_rgbd_points": int(len(points_rgb)),
        "projected_event_points": int(len(z)),
        "event_plane_coverage_percent_before_gap_fill": coverage,
    }


def temporal_event_layers(
    event_frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Color Event polarity and age; later frames overwrite earlier events."""
    height, width = event_frames[0].shape[:2]
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    opacity = np.zeros((height, width), dtype=np.float32)
    for index, frame in enumerate(event_frames):
        if frame.shape[:2] != (height, width):
            raise RuntimeError("Event frame shapes are inconsistent")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        age_bin = min(4, index * 5 // len(event_frames))
        positive = frame > 128
        negative = frame < 128
        colors[negative] = NEGATIVE_PALETTE[age_bin]
        colors[positive] = POSITIVE_PALETTE[age_bin]
        active = positive | negative
        opacity[active] = BIN_OPACITY[age_bin]

    event_view = np.full((height, width, 3), 128, dtype=np.uint8)
    active = opacity > 0
    event_view[active] = colors[active]
    return colors, opacity, event_view


def overlay_events(
    warped_rgb: np.ndarray, event_colors: np.ndarray, opacity: np.ndarray
) -> np.ndarray:
    hsv = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.60
    hsv[:, :, 2] *= 0.75
    background = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
    ).astype(np.float32)
    alpha = opacity[:, :, None]
    return np.clip(
        background * (1.0 - alpha) + event_colors.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="H5 session root")
    parser.add_argument("--device", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--sync-index",
        type=int,
        default=0,
        help="Synchronization index; -1 selects the final synchronized frame",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=10,
        help="Most recent associated Event frames to display; 0 uses all",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=12.0)
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="RGB-D pixel sampling stride; 1 is densest and slowest",
    )
    parser.add_argument("--fill-kernel", type=int, default=3)
    args = parser.parse_args()

    if args.event_count < 0:
        parser.error("--event-count must be zero or positive")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        parser.error("invalid depth range")

    session = args.session.expanduser().resolve()
    rgbd_path = session / "RGBD" / f"device{args.device}.h5"
    event_path = session / "Event" / f"device{args.device}.h5"
    if not rgbd_path.is_file():
        raise FileNotFoundError(rgbd_path)
    if not event_path.is_file():
        raise FileNotFoundError(event_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else session / "rgbd_to_event_h5_example"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(rgbd_path, "r") as rgbd_h5, h5py.File(event_path, "r") as event_h5:
        rgb_stream = rgbd_h5["streams/rgb"]
        depth_stream = rgbd_h5["streams/depth_align"]
        event_stream = event_h5["streams/images"]
        require_undistorted(rgb_stream, "RGBD/rgb")
        require_undistorted(depth_stream, "RGBD/depth_align")
        require_undistorted(event_stream, "Event/images")

        rgb_row, sync_index = find_sync_row(
            rgbd_h5["synchronization"], args.sync_index
        )
        if rgb_row >= len(rgb_stream["timestamp_ns"]):
            raise RuntimeError("synchronization row exceeds RGB stream length")
        rgb = decode_frame(rgb_stream, rgb_row)
        depth = decode_frame(depth_stream, rgb_row)

        event_positions = event_positions_for_sync(
            event_h5, sync_index, args.event_count
        )
        event_frames = [decode_frame(event_stream, int(pos)) for pos in event_positions]
        event_shape = event_frames[0].shape[:2]

        K_rgb = np.asarray(
            rgbd_h5["calibration/rgb/intrinsic_matrix"][:], dtype=np.float64
        )
        K_event = np.asarray(
            rgbd_h5["calibration/event/intrinsic_matrix"][:], dtype=np.float64
        )
        R = np.asarray(
            rgbd_h5["calibration/rgb_to_event/rotation_matrix"][:],
            dtype=np.float64,
        )
        t = np.asarray(
            rgbd_h5["calibration/rgb_to_event/translation"][:], dtype=np.float64
        )
        for key, expected in (
            ("calibration/event/intrinsic_matrix", K_event),
            ("calibration/rgb_to_event/rotation_matrix", R),
            ("calibration/rgb_to_event/translation", t),
        ):
            actual = np.asarray(event_h5[key][:], dtype=np.float64)
            if not np.allclose(actual, expected, rtol=1e-9, atol=1e-12):
                raise RuntimeError(f"calibration mismatch between H5 files: {key}")

        warped, projection = warp_rgbd_to_event(
            rgb,
            depth,
            K_rgb,
            K_event,
            R,
            t,
            event_shape,
            args.depth_scale,
            args.min_depth_m,
            args.max_depth_m,
            args.stride,
            args.fill_kernel,
        )
        event_colors, opacity, event_view = temporal_event_layers(event_frames)
        overlay = overlay_events(warped, event_colors, opacity)

        prefix = f"device{args.device}_sync{sync_index:06d}"
        warped_path = output_dir / f"{prefix}_rgb_warped_to_event.jpg"
        event_view_path = output_dir / f"{prefix}_event_accumulation.png"
        overlay_path = output_dir / f"{prefix}_rgb_event_overlay.jpg"
        report_path = output_dir / f"{prefix}_report.json"
        write_image(warped_path, warped)
        write_image(event_view_path, event_view)
        write_image(overlay_path, overlay)

        event_timestamps = np.asarray(
            event_stream["timestamp_ns"][event_positions], dtype=np.int64
        )
        report = {
            "session": str(session),
            "device": args.device,
            "sync_index": sync_index,
            "rgb_h5_position": rgb_row,
            "rgb_frame_index": int(rgb_stream["frame_index"][rgb_row]),
            "rgb_timestamp_ns": int(rgb_stream["timestamp_ns"][rgb_row]),
            "timestamp_note": "timestamp_ns is corrected; original_filename is not used",
            "input_images_already_undistorted": True,
            "second_undistortion_applied": False,
            "event_h5_positions": event_positions.tolist(),
            "event_frame_indices": np.asarray(
                event_stream["frame_index"][event_positions], dtype=np.int64
            ).tolist(),
            "event_timestamp_ns": event_timestamps.tolist(),
            "event_count": int(len(event_positions)),
            "event_time_span_ms": float(
                (event_timestamps[-1] - event_timestamps[0]) / 1_000_000.0
            ),
            "projection": projection,
            "calibration_source": "RGBD H5 /calibration",
            "pose_convention": "X_event = R_rgb_to_event * X_rgb + t_rgb_to_event",
            "outputs": {
                "warped_rgb": warped_path.name,
                "event_accumulation": event_view_path.name,
                "overlay": overlay_path.name,
            },
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"[OK] warped RGB:       {warped_path}")
    print(f"[OK] Event view:       {event_view_path}")
    print(f"[OK] overlay:          {overlay_path}")
    print(f"[OK] report:           {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

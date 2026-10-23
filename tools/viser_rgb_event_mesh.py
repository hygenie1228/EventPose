#!/usr/bin/env python3
"""Interactive server-L RGB + Event + SMPL-X viewer for five sessions.

The dataset is opened read-only. No cache or output is ever written below
dataset/ or orig_dataset/.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import viser


# Keep the workspace-visible path instead of resolving the `exp/` symlink.
PROJECT = Path(__file__).absolute().parents[1]
DATASET = PROJECT / "dataset"
EXPERIMENT = PROJECT / "exp" / "261023" / "server_l_preview"
SESSIONS = (
    "20260720_125338_040324_200_f",
    "20260720_130303_568628_200_h",
    "20260720_131207_863008_200_b",
    "20260720_132039_045275_200_i_201",
    "20260720_132719_522887_201_f",
)
COLORS = ((70, 160, 255), (255, 125, 75), (110, 220, 140), (210, 115, 240))


def _assert_read_only_source(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(DATASET.resolve()):
        raise RuntimeError(f"source escaped server L root: {resolved}")


def _decode_rgb(session: Path, position: int) -> np.ndarray:
    path = session / "RGB" / "device0" / "images.h5"
    _assert_read_only_source(path)
    with h5py.File(path, "r") as f:
        stream = f["streams/images"]
        position = min(max(position, 0), len(stream["offsets"]) - 2)
        start, stop = map(int, stream["offsets"][position : position + 2])
        encoded = bytes(stream["encoded_data"][start:stop])
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"))


def _event_paths(session: Path, position: int) -> tuple[Path, Path, int]:
    manifest_path = session / "Event" / "device0" / "manifest.json"
    _assert_read_only_source(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for shard in manifest["shards"]:
        if int(shard["start"]) <= position < int(shard["stop"]):
            root = manifest_path.parent
            return root / shard["matrix"], root / shard["frames"], position - int(shard["start"])
    raise IndexError(f"Event position {position} not found in {manifest_path}")


@lru_cache(maxsize=2)
def _event_shard(matrix_path: Path, frames_path: Path):
    with np.load(frames_path) as meta, np.load(matrix_path) as sparse:
        return (tuple(map(int, meta['image_shape'])), int(meta['background_value']),
                sparse['indptr'], sparse['indices'], sparse['data'])


def _decode_event(session: Path, position: int) -> np.ndarray:
    matrix_path, frames_path, local = _event_paths(session, position)
    _assert_read_only_source(matrix_path)
    (height, width), background, indptr, all_indices, all_values = _event_shard(matrix_path, frames_path)
    start, stop = map(int, indptr[local:local + 2])
    indices, values = all_indices[start:stop], all_values[start:stop]
    frame = np.full(height * width, background, dtype=np.uint8)
    frame[indices] = np.where(values > 0, 255, 0).astype(np.uint8)
    # RGB output gives positive/negative events visible colors in the GUI.
    rgb = np.repeat(frame.reshape(height, width, 1), 3, axis=2)
    rgb[frame.reshape(height, width) > background] = (255, 70, 40)
    rgb[frame.reshape(height, width) < background] = (30, 130, 255)
    return rgb


@lru_cache(maxsize=5)
def _valid_rows(session_name: str):
    with h5py.File(DATASET / session_name / 'RGBD/device0_mamma.h5', 'r') as f:
        return np.flatnonzero(f['results/success'][:])


def _load_sample(session_name: str, fraction: float, frame_index: int | None = None) -> dict:
    session = DATASET / session_name
    gt_path = session / "RGBD" / "device0_mamma.h5"
    _assert_read_only_source(gt_path)
    with h5py.File(gt_path, "r") as f:
        valid_rows = _valid_rows(session_name)
        if not len(valid_rows):
            raise RuntimeError(f"no successful GT rows: {gt_path}")
        index = round(np.clip(fraction, 0.0, 1.0) * (len(valid_rows) - 1)) if frame_index is None else frame_index
        row = int(valid_rows[index])
        person_valid = np.asarray(f["results/person_valid"][row], dtype=bool)
        quantized = np.asarray(f["meshes/verts_cam"][row][person_valid])
        scale = float(f.attrs["vertex_scale_mm_per_unit"]) / 1000.0
        vertices = quantized.astype(np.float32) * scale
        faces = np.asarray(f["topology/faces"][:], dtype=np.int32)
        rgb_position = int(f["frames/source_rgb_position"][row])
        event_positions = np.asarray(f["synchronization/event/position"][row], dtype=np.int64)
        event_valid = np.asarray(f["synchronization/event/valid"][row], dtype=bool)
        if not event_valid[0]:
            raise RuntimeError(f"device0 Event is invalid at GT row {row}")
        event_position = int(event_positions[0])
        timestamp_ns = int(f["frames/source_rgb_timestamp_ns"][row])

    # Camera coordinates (x right, y down, z forward) -> Viser (z up).
    vertices = vertices[..., (0, 2, 1)]
    vertices[..., 2] *= -1.0
    if len(vertices):
        center_xy = np.median(vertices.reshape(-1, 3), axis=0)
        vertices[..., :2] -= center_xy[:2]
        vertices[..., 2] -= vertices[..., 2].min()
    return {
        "session": session_name,
        "row": row,
        "rgb_position": rgb_position,
        "event_position": event_position,
        "timestamp_ns": timestamp_ns,
        "vertices": vertices,
        "faces": faces,
        "rgb": _decode_rgb(session, rgb_position),
        "event": _decode_event(session, event_position),
    }


def validate() -> None:
    """Load one synchronized RGB/Event/mesh sample from every selected session."""
    results = []
    for name in SESSIONS:
        sample = _load_sample(name, 0.5)
        results.append({
            "session": name,
            "gt_row": sample["row"],
            "rgb_position": sample["rgb_position"],
            "event_position": sample["event_position"],
            "people": len(sample["vertices"]),
            "vertices_per_person": int(sample["vertices"].shape[1]),
            "faces": int(len(sample["faces"])),
            "rgb_shape": list(sample["rgb"].shape),
            "event_shape": list(sample["event"].shape),
        })
    output = EXPERIMENT / "validation.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def serve(host: str, port: int) -> None:
    server = viser.ViserServer(host=host, port=port, label="EventPose RGB | Event | SMPL-X")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=8, height=8, cell_size=0.25, section_size=1.0)

    server.gui.add_markdown(
        "# Server L viewer\nSynchronized **RGB | Event | GT SMPL-X mesh**. "
        "Drag in the 3D viewport to orbit around the mesh."
    )
    session_gui = server.gui.add_dropdown("Session", options=SESSIONS, initial_value=SESSIONS[0])
    frame_gui = server.gui.add_slider(
        "Timeline (%)", min=0, max=100, step=1, initial_value=50,
        hint="Position within successful GT frames",
    )
    play_gui = server.gui.add_checkbox('Auto play', initial_value=True)
    loop_gui = server.gui.add_checkbox('Loop', initial_value=True)
    fps_gui = server.gui.add_slider('Playback FPS', min=1, max=30, step=1, initial_value=15)
    status_gui = server.gui.add_markdown("Loading...")
    with server.gui.add_folder("Synchronized camera views", expand_by_default=True):
        rgb_gui = server.gui.add_image(np.zeros((32, 32, 3), np.uint8), label="RGB", format="jpeg")
        event_gui = server.gui.add_image(np.zeros((32, 32, 3), np.uint8), label="Event (+ red / - blue)", format="png")

    state: dict = {"meshes": [], "frame": 0}
    lock = threading.Lock()

    def refresh(frame_index=None) -> None:
        with lock:
            status_gui.content = "Loading synchronized sample..."
            try:
                name = str(session_gui.value)
                count = len(_valid_rows(name))
                index = round(float(frame_gui.value) / 100.0 * (count - 1)) if frame_index is None else frame_index
                sample = _load_sample(name, 0, frame_index=index)
                state['frame'] = index
                rgb_gui.image = sample["rgb"]
                event_gui.image = sample["event"]
                for handle in state["meshes"]:
                    handle.remove()
                state["meshes"] = []
                for person, verts in enumerate(sample["vertices"]):
                    state["meshes"].append(server.scene.add_mesh_simple(
                        f"/people/person_{person}", verts, sample["faces"],
                        color=COLORS[person % len(COLORS)], material="standard",
                        flat_shading=False, side="double",
                    ))
                status_gui.content = (
                    f"**Session:** `{sample['session']}`  \n"
                    f"**GT row:** {sample['row']:,} · **people:** {len(sample['vertices'])}  \n"
                    f"**Playback:** {index + 1:,} / {count:,} ({100 * index / max(count - 1, 1):.1f}%)  \n"
                    f"**RGB position:** {sample['rgb_position']:,} · "
                    f"**Event position:** {sample['event_position']:,}"
                )
            except Exception as exc:
                status_gui.content = f"**Load failed:** `{exc!r}`"
                raise

    @session_gui.on_update
    def _(_) -> None:
        refresh()

    @frame_gui.on_update
    def _(_) -> None:
        refresh()

    refresh()
    print(f"Serving five sessions from read-only {DATASET}")
    print(f"Open http://localhost:{port}")
    try:
        while True:
            start = time.monotonic()
            if play_gui.value:
                count = len(_valid_rows(str(session_gui.value)))
                next_frame = state['frame'] + 1
                if next_frame >= count:
                    if loop_gui.value:
                        next_frame = 0
                    else:
                        play_gui.value = False
                if play_gui.value:
                    refresh(next_frame)
            time.sleep(max(0.001, 1.0 / float(fps_gui.value) - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate()
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()

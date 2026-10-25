"""Read-only RGB/event comparison using corrected timestamps, without spatial warping."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np


def prefix_array(path, key, count):
    # Read only the needed prefix of large compressed CSR arrays.
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        assert not fortran and len(shape) == 1
        return np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    path = args.source / 'event.npz'
    with np.load(path, allow_pickle=False) as z:
        event_times = z['timestamp_ns'].astype(np.int64)
        rgb_times = z['rgb__timestamp_ns'].astype(np.int64)
        height, width = map(int, z['image_shape'])
        indptr = z['indptr']
    period = int(np.median(np.diff(rgb_times)))
    count = int(np.searchsorted(rgb_times, rgb_times[0] + 60_000_000_000))
    last_row = int(np.searchsorted(event_times, rgb_times[count - 1], side='right'))
    indices = prefix_array(path, 'indices', int(indptr[last_row]))
    values = prefix_array(path, 'data', int(indptr[last_row]))
    cap = cv2.VideoCapture(str(args.source / 'rgb.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt',
               'bgr24', '-s', '1920x400', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(count):
        ok, rgb = cap.read()
        assert ok, frame
        end = int(rgb_times[frame])
        start = int(rgb_times[frame - 1]) if frame else end - period
        lo, hi = np.searchsorted(event_times, [start, end], side='right')
        a, b = int(indptr[lo]), int(indptr[hi])
        activity = np.bincount(indices[a:b], weights=values[a:b],
                               minlength=height * width).reshape(height, width)
        color = np.zeros((height, width, 3), np.uint8)
        color[activity > 0] = (0, 70, 255)
        color[activity < 0] = (255, 180, 0)
        # Fit each native image into the same panel, preserving aspect ratio.
        event = np.zeros((360, 640, 3), np.uint8)
        scaled_w = round(width * 360 / height)
        left = (640 - scaled_w) // 2
        event[:, left:left + scaled_w] = cv2.resize(color, (scaled_w, 360), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        overlay = rgb.copy()
        active = event.any(axis=2)
        overlay[active] = (0.25 * rgb[active] + 0.75 * event[active]).astype(np.uint8)
        canvas = np.zeros((400, 1920, 3), np.uint8)
        canvas[40:] = np.concatenate([rgb, event, overlay], axis=1)
        for panel, label in enumerate(['RGB', 'EVENT | corrected time', 'OVERLAY | no spatial warp']):
            cv2.putText(canvas, label, (panel * 640 + 14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'{(end-int(rgb_times[0]))/1e9:05.2f}s', (530, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        encoder.stdin.write(canvas.tobytes())
    cap.release()
    encoder.stdin.close()
    assert encoder.wait() == 0
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        source=str(args.source), frames=count, fps=fps, duration_seconds=count/fps,
        event_window='previous RGB timestamp < corrected event timestamp <= current RGB timestamp',
        spatial_warp=False, display='aspect-preserving fit, centered; event positive red, negative cyan',
        note='No events are fabricated outside event timestamp coverage.'), indent=2))
    print(args.output, flush=True)


if __name__ == '__main__':
    main()

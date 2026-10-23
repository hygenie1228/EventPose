"""Read-only server L export: five synchronized 10-second, four-panel clips."""
import json
import subprocess
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from viser_rgb_event_mesh import DATASET, EXPERIMENT, SESSIONS, _decode_rgb, _decode_event


def render(image, vertices, faces, K, alpha=.65):
    out = image.copy()
    for person, v in enumerate(vertices):
        triangles = v[faces]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        brightness = .35 + .65 * np.abs(normal @ np.array([.3, -.4, -.866]))
        base = np.array([(245, 170, 65), (90, 145, 250)][person % 2])
        colors = np.clip(brightness[:, None] * base, 0, 255).astype(np.uint8)
        projected = v @ K.T
        uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-5)
        polys = np.clip(uv[faces], -10000, 10000).astype(np.int32)
        for idx in np.argsort(triangles[:, :, 2].mean(axis=1))[::-1]:
            if triangles[idx, :, 2].min() > .01:
                cv2.fillConvexPoly(out, polys[idx], tuple(map(int, colors[idx])), lineType=cv2.LINE_AA)
    return cv2.addWeighted(out, alpha, image, 1-alpha, 0)


def panel(image, title):
    height, width = image.shape[:2]
    scale = min(640 / width, 330 / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    out = np.full((360, 640, 3), 25, np.uint8)
    x = (640 - resized.shape[1]) // 2
    y = 30 + (330 - resized.shape[0]) // 2
    out[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    cv2.rectangle(out, (0, 0), (640, 30), (25, 25, 25), -1)
    cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
    return out


def export(name):
    session = DATASET / name
    path = EXPERIMENT / f'{name}_4panel.mp4'
    fps, duration = 15, 10
    records = []
    with h5py.File(session/'RGBD/device0_mamma.h5', 'r') as gt, h5py.File(session/'RGB/device0/images.h5', 'r') as rgbfile:
        times = gt['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
        good = gt['results/success'][:].astype(bool)
        start = int(times[len(times)//2])
        targets = start + (np.arange(fps*duration)*1e9/fps).astype(np.int64)
        rows = np.clip(np.searchsorted(times, targets), 0, len(times)-1)
        faces = gt['topology/faces'][:].astype(np.int32)
        Krgb = rgbfile['calibration/rgb/intrinsic_matrix'][:]
        Kev = rgbfile['calibration/event/intrinsic_matrix'][:]
        R = rgbfile['calibration/rgb_to_event/rotation_matrix'][:]
        t = rgbfile['calibration/rgb_to_event/translation'][:]
        scale = float(gt.attrs['vertex_scale_mm_per_unit'])/1000
        cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for index, row in enumerate(rows):
                row = int(row)
                valid = gt['results/person_valid'][row].astype(bool)
                v = gt['meshes/verts_cam'][row][valid].astype(np.float32)*scale
                rp = int(gt['frames/source_rgb_position'][row])
                ep = int(gt['synchronization/event/position'][row,0])
                if not good[row] or not gt['synchronization/event/valid'][row,0]:
                    raise RuntimeError(f'invalid synchronized row {row}')
                rgb = cv2.cvtColor(_decode_rgb(session,rp),cv2.COLOR_RGB2BGR)
                event = cv2.cvtColor(_decode_event(session,ep),cv2.COLOR_RGB2BGR)
                # Render directly at panel resolution, scaling each camera intrinsic.
                small_rgb = cv2.resize(rgb,(640,360))
                # Event is 4:3. Keep its native aspect ratio before letterboxing.
                small_ev = cv2.resize(event,(480,360))
                K1 = Krgb.copy(); K1[:2] *= .5
                K2 = Kev.copy(); K2[0] *= 480/event.shape[1]; K2[1] *= 360/event.shape[0]
                overlay_rgb = render(small_rgb,v,faces,K1)
                evverts = v @ R.T + t
                overlay_event = render(small_ev,evverts,faces,K2)
                frame = np.vstack([np.hstack([panel(small_rgb,'RGB'),panel(overlay_rgb,'RGB + SMPL-X')]),np.hstack([panel(small_ev,'Event'),panel(overlay_event,'Event + SMPL-X')])])
                process.stdin.write(frame.tobytes())
                records.append({'gt_row':row,'rgb_position':rp,'event_position':ep})
                if index % 30 == 0: print(name,index,'/',len(rows),flush=True)
        finally:
            process.stdin.close()
            code = process.wait()
        if code: raise RuntimeError(f'ffmpeg failed: {code}')
    return {'session':name,'file':path.name,'fps':fps,'duration_seconds':duration,'frames':len(rows),'source_frames':records,'panels':['RGB','RGB + SMPL-X','Event','Event + SMPL-X'],'aspect_ratio':'Preserved for every panel with letterboxing','event_transform':'X_event = R_rgb_to_event @ X_rgb + t_rgb_to_event_m'}


if __name__ == '__main__':
    results = []
    for name in SESSIONS:
        results.append(export(name))
        (EXPERIMENT/'video_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Finished all five MP4 files',flush=True)

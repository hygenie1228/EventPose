"""Read-only sensor references and real-mesh six-panel MP4 rendering."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import time
import cv2
import h5py
import numpy as np
from multihmr.video_gallery import read, write
from utils.event_encoding import fit_geometry
from utils.mesh_video_rendering import mesh, pointcloud, labelled_panel


def display_geometry(f, sensor):
    nh, nw, top, left = fit_geometry(f[sensor+'_native_shape'][:], 896)[0]
    sx, sy = 640/nw, 360/nh
    transform = np.array([[sx, 0, (sx-1)/2-sx*left], [0, sy, (sy-1)/2-sy*top], [0, 0, 1.]])
    return (top, left, nh, nw), transform @ f[sensor+'_K_primary'][:]


def prepare_depth(goal, clip, first_rgb, display_K):
    task = clip['task']; source = Path(goal['source_root'])
    dataset = Path(__file__).resolve().parents[2] / 'dataset' / task['parent_session'] / 'device3'
    path = dataset / 'depth.mp4'; work = Path(goal['work_root']) / 'references'
    receipt = dict(available=False, source=str(path), depth_scale_m_per_unit=goal['pointcloud_depth_scale_m_per_unit'],
                   scale_assumption=goal['depth_scale_provenance'], depth_protocol_version=2)
    try:
        metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format_tags', '-of', 'json', str(path)]))
        tags = metadata['format']['tags']; assert 'R=low byte,G=high byte,B=0' in tags['depth_encoding']
        stamps = np.asarray(json.loads(tags['depth_timestamps_ns']), np.int64)
        with h5py.File(dataset / 'label.h5', 'r', locking=False) as f:
            positions = f['frames/source_depth_position'][:]; mapped_times = f['frames/source_depth_timestamp_ns'][:]
        # The archived depth is a timestamp-selected 5 Hz subset of the
        # original 30 Hz stream. Its MP4 position is not the label position.
        original_times = np.unique(mapped_times[mapped_times > 0])
        matches = np.searchsorted(original_times, stamps)
        assert len(original_times) >= 10 and np.all(matches < len(original_times)), 'Depth timestamp range mismatch'
        differences = original_times[matches]-stamps
        maximum_clock_difference = int(np.max(np.abs(differences)))
        assert maximum_clock_difference == 0, 'Archived timestamps must exactly match provided corrected source timestamps'
        offset = 0
        with h5py.File(source / 'cache' / (task['parent_session']+'.h5'), 'r', locking=False) as f:
            i = int(np.searchsorted(f['frames'][:], task['start_frame'])); timestamp = int(f['timestamp_ns'][i])
        position = int(np.searchsorted(stamps, timestamp, side='right')-1)
        assert position >= 0 and 0 <= timestamp-int(stamps[position]) <= 250_000_000
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        assert cap.isOpened() and cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, packed = cap.read(); cap.release(); assert ok
        assert np.count_nonzero(packed[:, :, 0]) == 0
        depth = packed[:, :, 2].astype(np.uint16)+(packed[:, :, 1].astype(np.uint16)<<8)
        with np.load(dataset / 'event.npz') as z:
            K = z['rgb_calibration__rgb__pre_resize_intrinsic_matrix'].astype(np.float64)
            pixel = z['rgb_calibration__rgb__resize_pixel_transform'].astype(np.float64)
            native_K = z['rgb_calibration__rgb__intrinsic_matrix'].astype(np.float64)
        assert np.allclose(pixel @ K, native_K, atol=1e-4)
        yy, xx = np.mgrid[0:depth.shape[0]:8, 0:depth.shape[1]:8]
        z = depth[yy, xx].reshape(-1).astype(np.float64)*goal['pointcloud_depth_scale_m_per_unit']
        pixels = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(xx.size)], axis=1)
        points = (pixels @ np.linalg.inv(K).T)*z[:, None]
        valid = (z > .2) & (z < 10) & np.isfinite(points).all(1)
        points = points[valid]; raw_pixels = pixels[valid, :2]
        projection = points @ K.T; back = projection[:, :2]/projection[:, 2:]
        error = float(np.max(np.abs(back-raw_pixels))); assert error < 1e-5
        projection = points @ display_K.T
        uv = np.rint(projection[:, :2]/projection[:, 2:]).astype(int)
        valid = (uv[:, 0]>=0)&(uv[:, 0]<640)&(uv[:, 1]>=0)&(uv[:, 1]<360)
        points, uv = points[valid], uv[valid]; colors = first_rgb[uv[:, 1], uv[:, 0]]
        assert len(points) > 1000
        np.savez_compressed(work / (clip['clip']+'_cloud.npz'), points=points.astype(np.float32), colors=colors,
                            K_depth=K, source_depth_position=position, timestamp_ns=int(stamps[position]))
        receipt.update(available=True, points=len(points), raw_depth_shape=list(depth.shape), source_depth_position=position,
                       source_depth_timestamp_ns=int(stamps[position]), reference_timestamp_ns=timestamp,
                       age_ns=timestamp-int(stamps[position]), clock_offset_ns=offset,
                       clock_comparison_maximum_residual_ns=maximum_clock_difference,
                       all_archive_timestamps_exactly_matched_to_original_source=True,
                       label_source_depth_position_for_selected_frame=int(positions[np.flatnonzero(mapped_times == stamps[position])[0]]),
                       backprojection_reprojection_maximum_error_px=error,
                       depth_frame_sha256=hashlib.sha256(depth.astype('<u2').tobytes()).hexdigest(),
                       scene_camera='device3 rectified RGB camera; aligned depth plane',
                       meaning='Measured first-past depth cloud, static per clip, reference RGB colors; .001 scale assumed from provided example')
    except Exception as error:
        receipt['unavailable_reason'] = type(error).__name__ + ': ' + str(error)
    write(work / (clip['clip']+'_depth.json'), receipt)
    return receipt


def prepare_references(goal):
    cv2.setNumThreads(1)
    work = Path(goal['work_root']) / 'references'; work.mkdir(parents=True, exist_ok=True)
    source = Path(goal['source_root']); records = []
    for clip in goal['clips']:
        path = work / (clip['clip']+'.h5'); receipt = path.with_suffix('.json')
        if receipt.exists():
            value = read(receipt)
            if value['depth'].get('depth_protocol_version') != 2:
                with h5py.File(path, 'r', locking=False) as f:
                    value['depth'] = prepare_depth(goal, clip, f['rgb3'][0], f['rgb3_K'][:])
                write(receipt, value)
            records.append(value); continue
        task = clip['task']; temporary = path.with_suffix('.partial.h5')
        with h5py.File(source / 'cache' / (task['parent_session']+'.h5'), 'r', locking=False) as f, h5py.File(temporary, 'w', locking=False) as out:
            frames = np.arange(task['start_frame'], task['end_frame'], dtype=np.int64)
            indices = np.searchsorted(f['frames'][:], frames); assert np.array_equal(f['frames'][indices], frames)
            out.create_dataset('source_frames', data=frames); out.create_dataset('timestamp_ns', data=f['timestamp_ns'][indices])
            geometry = {}
            for sensor in ('rgb3', 'rgb4'):
                geometry[sensor], K = display_geometry(f, sensor)
                out.create_dataset(sensor+'_K', data=K)
                out.create_dataset(sensor, (len(frames), 360, 640, 3), dtype='u1', chunks=(1, 360, 640, 3), compression='lzf')
            for sensor in ('event3', 'event4'):
                out.create_dataset(sensor, (len(frames), 2, 224, 224), dtype='u1', chunks=(1, 2, 224, 224), compression='lzf')
                latest = f[sensor+'_latest_timestamp_ns'][indices]
                assert np.all((latest <= f['timestamp_ns'][indices]) | ~f[sensor+'_available'][indices])
                out.create_dataset(sensor+'_latest_timestamp_ns', data=latest)
            for i, row in enumerate(indices):
                for sensor in ('rgb3', 'rgb4'):
                    top, left, nh, nw = geometry[sensor]
                    rgb = f[sensor][row].transpose(1, 2, 0)[top:top+nh, left:left+nw, ::-1]
                    out[sensor][i] = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
                for sensor in ('event3', 'event4'): out[sensor][i] = f[sensor][row]
            first = out['rgb3'][0]; display_K = out['rgb3_K'][:]
            out.attrs['complete'] = True
        temporary.replace(path)
        depth = prepare_depth(goal, clip, first, display_K)
        value = dict(status='complete', clip=clip['clip'], frames=len(frames), path=str(path),
                     event_display='count33 actual model input, red positive / blue negative, no future events', depth=depth)
        write(receipt, value); records.append(value)
        print(json.dumps(dict(references_complete=clip['clip'], depth_available=depth['available'])), flush=True)
    write(Path(goal['output_root']) / 'references_complete.json', dict(status='complete', clips=records))


def view_settings(goal, clip):
    work = Path(goal['work_root']); path = work / (clip+'_view.json')
    with (work / (clip+'_view.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists(): return read(path)
        with h5py.File(Path(goal['output_root']) / 'meshes/rgb3_short90' / (clip+'.h5'), 'r', locking=False) as f:
            vertices = f['vertices_common_m'][:]; joints = f['joints_common_m'][:]
        assert len(vertices)
        center = np.median(joints[:, 0], axis=0)+np.array([0, -.35, 0])
        eye = center+np.array([2.8, -1.2, -4.5])
        forward = center-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0., -1, 0])); right /= np.linalg.norm(right)
        down = np.cross(forward, right); rotation = np.stack([right, down, forward])
        K = np.array([[550., 0, 320], [0, 550, 180], [0, 0, 1]])
        value = dict(eye=eye.tolist(), rotation=rotation.tolist(), K=K.tolist(),
                     foreground_min=(vertices.min((0, 1))-.05).tolist(), foreground_max=(vertices.max((0, 1))+.05).tolist(),
                     camera_source='Frozen fine-tuned RGB3 trajectory, same fixed novel camera for all nine conditions; no GT alignment')
        write(path, value); return value


def events_panel(a, b):
    result = np.full((360, 640, 3), 24, np.uint8)
    for x, data in ((0, a), (320, b)):
        positive, negative = np.clip(data.astype(np.float32)/255*4, 0, 1)
        image = np.full((224, 224, 3), 235., np.float32)
        strength = np.maximum(positive, negative)
        image[:, :, 0] -= 210*positive; image[:, :, 1] -= 210*strength; image[:, :, 2] -= 210*negative
        image = cv2.resize(np.clip(image, 0, 255).astype(np.uint8), (310, 310), interpolation=cv2.INTER_NEAREST)
        result[25:335, x+5:x+315] = image
    return result


def render_video(goal, name, clip_name):
    cv2.setNumThreads(1); started = time.time()
    clip = next(c for c in goal['clips'] if c['clip'] == clip_name); task = clip['task']
    out = Path(goal['output_root']); work = Path(goal['work_root'])
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = out / 'mp4' / name / (clip_name+'.mp4'); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = work / (name+'_'+clip_name+'.partial.mp4')
    settings = view_settings(goal, clip_name)
    eye, rotation, Kview = (np.array(settings[k]) for k in ('eye', 'rotation', 'K'))
    background = np.full((360, 640, 3), 24, np.uint8); background_depth = None
    depth_receipt = read(work / 'references' / (clip_name+'_depth.json'))
    if depth_receipt['available']:
        with np.load(work / 'references' / (clip_name+'_cloud.npz')) as z:
            points, colors = z['points'], z['colors']
        human = ((points >= settings['foreground_min']) & (points <= settings['foreground_max'])).all(1)
        background, background_depth = pointcloud(points[~human], colors[~human], eye, rotation, Kview)
    config = read(goal['models'][name]['config']); used = [s for s in (config['primary_sensor'], config['auxiliary_sensor']) if s]
    faces = np.load(out / 'meshes' / name / 'faces.npy')
    frame_count = task['video_frames']; fps = task['fps']; pixels = []; detections = []
    encoder = subprocess.Popen(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                '-s', '1920x864', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
                                '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
    contacts = out / 'contacts' / name; contacts.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(out / 'meshes' / name / (clip_name+'.h5'), 'r', locking=False) as pred, h5py.File(work / 'references' / (clip_name+'.h5'), 'r', locking=False) as ref:
            offsets = pred['frame_offsets'][:]; stamps = ref['timestamp_ns'][:]
            assert np.array_equal(pred['timestamp_ns'][:], stamps)
            K = ref['rgb3_K'][:]
            for i in range(frame_count):
                vertices = pred['vertices_common_m'][offsets[i]:offsets[i+1]]
                rgb3, rgb4 = ref['rgb3'][i], ref['rgb4'][i]
                overlay, _, painted = mesh(rgb3, vertices, faces, K, alpha=.65)
                novel = (vertices-eye) @ rotation.T
                alone, _, _ = mesh(np.full_like(rgb3, 28), novel, faces, Kview)
                scene, _, _ = mesh(background, novel, faces, Kview, background_depth=background_depth)
                panels = [labelled_panel(rgb3, 'RGB3 | INPUT' if 'rgb3' in used else 'RGB3 | reference only'),
                          labelled_panel(rgb4, 'RGB4 | INPUT' if 'rgb4' in used else 'RGB4 | reference only'),
                          labelled_panel(events_panel(ref['event3'][i], ref['event4'][i]), 'Event3 / Event4 | count33 display x4 | red + / blue -'),
                          labelled_panel(alone, 'Predicted SMPL-X | fixed novel view | unaligned'),
                          labelled_panel(overlay, 'Mesh overlay | common RGB3 camera | unaligned'),
                          labelled_panel(scene, 'Measured depth scene | scale .001 assumed' if depth_receipt['available'] else 'No verified depth scene | mesh only')]
                canvas = np.full((864, 1920, 3), 20, np.uint8)
                canvas[64:464] = np.concatenate(panels[:3], axis=1); canvas[464:] = np.concatenate(panels[3:], axis=1)
                label = name+' | inputs: '+', '.join(used)+' | '+clip_name+f' | t={(int(stamps[i])-int(stamps[0]))/1e9:.3f}s | frame {i+1}/{frame_count}'
                cv2.putText(canvas, label, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
                note = 'All raw detections; no GT boxes / no pose alignment. Reference RGB/event/depth panels never enter an unused model branch.'
                if not len(vertices): note = 'NO MODEL DETECTION | '+note
                cv2.putText(canvas, note, (14, 51), cv2.FONT_HERSHEY_SIMPLEX, .49, (220, 220, 220), 1, cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes()); pixels.append(painted); detections.append(len(vertices))
                if i in (0, frame_count//2, frame_count-1): cv2.imwrite(str(contacts / f'{clip_name}_{i:04d}.jpg'), canvas)
    finally:
        encoder.stdin.close(); code = encoder.wait()
    assert code == 0 and len(pixels) == frame_count
    import shutil
    shutil.copyfile(temporary, destination.with_suffix('.copy.mp4')); destination.with_suffix('.copy.mp4').replace(destination)
    temporary.unlink()
    receipt = dict(status='rendered_full_decode_verification_pending', name=name, clip=clip_name, path=str(destination),
                   render_version=2, event_display_gain=4, render_implementation_sha256=implementation_hash,
                   frames=frame_count, fps=fps, width=1920, height=864, detections=detections,
                   overlay_mesh_pixels=pixels, scene_depth=depth_receipt, fixed_view=settings,
                   input_sensors=used, reference_sensors_never_added_to_model=True, no_GT_pose_alignment=True,
                   source_timestamp_span_seconds=float((int(stamps[-1])-int(stamps[0]))/1e9),
                   CFR_frame_timestamp_maximum_difference_seconds=float(np.max(np.abs((stamps-stamps[0])/1e9-np.arange(frame_count)/fps))),
                   elapsed_seconds=time.time()-started, sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == implementation_hash
    write(destination.with_suffix('.json'), receipt)
    print(json.dumps(dict(video_rendered=str(destination), frames=frame_count, seconds=receipt['elapsed_seconds'])), flush=True)

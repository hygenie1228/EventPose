"""Fully decode every produced MP4 and verify its mesh/time provenance."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from goal_runtime_control import read, write, register_job
from report_device34_video_gallery import update


def check(goal, name, clip):
    out = Path(goal['output_root']); video = out / 'mp4' / name / (clip['clip']+'.mp4')
    receipt = read(video.with_suffix('.json'))
    assert receipt['render_version']==2 and receipt['event_display_gain']==4
    assert receipt['render_implementation_sha256']==hashlib.sha256((ROOT/'src/multihmr/video_gallery_render.py').read_bytes()).hexdigest()
    assert hashlib.sha256(video.read_bytes()).hexdigest() == receipt['sha256']
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                      '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,avg_frame_rate', '-of', 'json', str(video)]))['streams'][0]
    assert info['codec_name'] == 'h264' and info['pix_fmt'] == 'yuv420p'
    assert info['width'] == 1920 and info['height'] == 864
    rate_a, rate_b = map(int, info['avg_frame_rate'].split('/'))
    assert abs(rate_a/rate_b-receipt['fps']) < 1e-4
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]); assert cap.isOpened()
    count = 0; samples = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        assert frame.shape == (864, 1920, 3)
        if count in (0, receipt['frames']//2, receipt['frames']-1):
            expected = cv2.imread(str(out / 'contacts' / name / f"{clip['clip']}_{count:04d}.jpg")); assert expected is not None
            difference = float(np.abs(frame.astype(np.float32)-expected.astype(np.float32)).mean())
            assert difference < 15 and frame.std() > 5
            samples.append(dict(frame=count, decoded_contact_mean_absolute_difference=difference))
        count += 1
    cap.release(); assert count == receipt['frames'] == clip['task']['video_frames'] == 301
    mesh_path = out / 'meshes' / name / (clip['clip']+'.h5')
    mesh_receipt = read(mesh_path.with_suffix('.json')); config = read(goal['models'][name]['config'])
    assert mesh_receipt['checkpoint_sha256'] == goal['models'][name]['sha256']
    assert mesh_receipt['old_matched_instances_missing'] == 0
    assert mesh_receipt['maximum_joint_difference_from_published_Test_m'] < .002
    assert mesh_receipt['GT_used_as_input'] is False
    assert set(mesh_receipt['pixel_datasets_read']) == {s for s in (config['primary_sensor'], config['auxiliary_sensor']) if s}
    with h5py.File(mesh_path, 'r', locking=False) as f, h5py.File(Path(goal['work_root']) / 'references' / (clip['clip']+'.h5'), 'r', locking=False) as reference:
        offsets = f['frame_offsets'][:]; vertices = f['vertices_common_m'][:]
        assert len(offsets) == 302 and offsets[0] == 0 and offsets[-1] == len(vertices)
        assert np.all(np.diff(offsets) >= 0) and vertices.shape[1:] == (10475, 3) and np.isfinite(vertices).all()
        assert np.array_equal(f['source_frames'][:], np.arange(clip['task']['start_frame'], clip['task']['end_frame']))
        assert np.array_equal(f['timestamp_ns'][:], reference['timestamp_ns'][:])
        assert np.array_equal(np.diff(offsets), receipt['detections'])
        assert not f.attrs['GT_input'] and f.attrs['checkpoint_sha256'] == goal['models'][name]['sha256']
    return dict(path=str(video), name=name, clip=clip['clip'], status='passed', frames=count, codec=info,
                decoded_contacts=samples, sha256=receipt['sha256'],
                maximum_joint_difference_from_published_Test_m=mesh_receipt['maximum_joint_difference_from_published_Test_m'],
                no_GT_inference_input=True, input_sensors=mesh_receipt['pixel_datasets_read'],
                source_frame_timestamp_mapping_verified=True, no_pose_alignment_for_visualization=True,
                depth_scene_available=receipt['scene_depth']['available'],
                depth_scene_scale_assumption=receipt['scene_depth']['scale_assumption'],
                CFR_maximum_timestamp_difference_seconds=receipt['CFR_frame_timestamp_maximum_difference_seconds'])


def main():
    cv2.setNumThreads(1)
    goal = read(ROOT / '_agents/research/261028/device34_test_video_goal.json')
    register_job(goal, os.getpid(), 'video_gallery_video_verifier')
    out = Path(goal['output_root']); state_path = out / 'video_verification.json'
    existing = read(state_path) if state_path.exists() else dict(records=[], verified_paths=[])
    records = existing['records']; done = {r['path'] for r in records}
    while len(records) < 45:
        changed = False
        for name in goal['models']:
            for clip in goal['clips']:
                video = out / 'mp4' / name / (clip['clip']+'.mp4')
                if str(video) in done or not video.with_suffix('.json').exists(): continue
                row = check(goal, name, clip); records.append(row); done.add(str(video)); changed = True
                write(state_path, dict(status='verification_in_progress', records=records, verified_paths=sorted(done), updated_unix=time.time()))
                update(goal)
                print(json.dumps(dict(verified=len(records), name=name, clip=clip['clip'])), flush=True)
        if not changed: time.sleep(30)
    from multihmr.multisensor_repair import implementation_provenance
    approved = read(Path(goal['source_root']) / 'source_revision_scope_extension.json')['approved_implementations']
    assert implementation_provenance() == approved[-1]
    assert read(out / 'renderer_CPU_geometry_check.json')['status'] == 'passed'
    assert len(records) == 45 and len({r['path'] for r in records}) == 45
    assert len({r['clip'] for r in records}) == 5
    write(state_path, dict(status='passed_full_decode_and_mesh_provenance', records=records, verified_paths=sorted(done), updated_unix=time.time(),
                           goal_complete=False, manual_visual_review_required=True))
    update(goal)
    write(out / 'ready_for_agent_review.json', dict(status='ready', video_count=45, frames=45*301,
                                                   actual_inference_replay_verified=True, full_decode_verified=True, goal_complete=False))


if __name__ == '__main__': main()

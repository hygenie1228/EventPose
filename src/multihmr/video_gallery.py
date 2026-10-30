"""Reproduce frozen Test predictions as real meshes for synchronized videos."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temporary.replace(path)


def infer_meshes(goal, name):
    import torch
    from multihmr.multisensor_inference import CachedSensors, load_model
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    from utils.pose_evaluation import configure_torch
    configure_torch(); torch.set_num_threads(2)
    assert int(os.environ['EVENTPOSE_PHYSICAL_GPU']) in goal['allowed_gpus']
    row = goal['models'][name]; config = read(row['config']); checkpoint = Path(row['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['sha256']
    model = load_model(config, checkpoint)
    reg = torch.from_numpy(np.load(Path(goal['source_root']) / 'joint_regressor.npy')).cuda()
    folder = Path(goal['output_root']) / 'meshes' / name; folder.mkdir(parents=True, exist_ok=True)
    faces = next(iter(model.base.smpl_layer.values())).bm_x.faces
    np.save(folder / 'faces.npy', np.asarray(faces, np.int32))
    detection_ids = []; original = model.base.detection
    def detection(*args, **kwargs):
        value = original(*args, **kwargs)
        detection_ids[:] = value[2][0].detach().cpu().tolist()
        return value
    model.base.detection = detection
    try:
        for clip in goal['clips']:
            target = folder / (clip['clip'] + '.h5'); receipt = target.with_suffix('.json')
            if receipt.exists():
                assert read(receipt)['checkpoint_sha256'] == row['sha256']; continue
            task = clip['task']; started = time.time()
            source = CachedSensors(task['parent_session'], config['condition'], sensor_layout=REPAIR_LAYOUTS[config['condition']])
            positions = np.arange(task['start_frame'], task['end_frame'], dtype=np.int64)
            indices = np.searchsorted(source.frames, positions); assert np.array_equal(source.frames[indices], positions)
            timestamps = source.f['timestamp_ns'][indices]
            with np.load(Path(goal['source_root']) / 'results' / name / task['session'] / 'result.npz') as previous:
                old_joints = previous['matched_joints_cam_m']; old_indices = previous['matched_pred_indices']
                assert previous['seen_gt_rows'].all() and np.array_equal(previous['source_rgb_positions'], np.arange(len(positions)))
            offsets = [0]; max_difference = 0.; missing_old = 0; total = 0
            temporary = target.with_suffix('.partial.h5')
            with h5py.File(temporary, 'w', locking=False) as out:
                vertices = out.create_dataset('vertices_common_m', shape=(0, 10475, 3), maxshape=(None, 10475, 3),
                                              chunks=(1, 10475, 3), compression='lzf', dtype='f4')
                joints = out.create_dataset('joints_common_m', shape=(0, 55, 3), maxshape=(None, 55, 3),
                                            chunks=(1, 55, 3), compression='lzf', dtype='f4')
                for start in range(0, len(positions), 16):
                    samples = [source.sample(int(frame)) for frame in positions[start:start + 16]]
                    selected = [(i, sample) for i, sample in enumerate(samples) if sample is not None]
                    grouped = [[] for _ in samples]
                    if selected:
                        inputs = {key: torch.stack([sample[key] for _, sample in selected]).cuda() for key in selected[0][1]}
                        assert not set(inputs) & {'joints', 'valid', 'centers', 'depth', 'betas'}
                        if name == 'event3_event4': assert 'image' not in inputs
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                            people = model(**inputs, is_training=False, det_thresh=.3, nms_kernel_size=3)
                        assert len(people) == len(detection_ids)
                        for image, person in zip(detection_ids, people):
                            v = person['v3d'].float()
                            j = (reg @ v).cpu().numpy()
                            v = v.cpu().numpy()
                            v = (v - source.t) @ source.R; j = (j - source.t) @ source.R
                            assert v.shape == (10475, 3) and np.isfinite(v).all() and np.isfinite(j).all()
                            grouped[selected[image][0]].append((v, j))
                    for local, values in enumerate(grouped):
                        frame = start + local
                        for person, pred in enumerate(old_indices[frame]):
                            if pred < 0: continue
                            if pred >= len(values): missing_old += 1; continue
                            max_difference = max(max_difference, float(np.max(np.abs(values[pred][1] - old_joints[frame, person]))))
                        if values:
                            vertices.resize(total + len(values), axis=0); joints.resize(total + len(values), axis=0)
                            vertices[total:] = np.stack([v for v, _ in values]); joints[total:] = np.stack([j for _, j in values])
                            total += len(values)
                        offsets.append(total)
                    write(folder / 'progress.json', dict(name=name, clip=clip['clip'], frames=min(start + 16, len(positions)),
                                                          total_frames=len(positions), updated_unix=time.time()))
                assert missing_old == 0 and max_difference < .002, (name, clip['clip'], missing_old, max_difference)
                out.create_dataset('frame_offsets', data=np.asarray(offsets, np.int64))
                out.create_dataset('source_frames', data=positions); out.create_dataset('timestamp_ns', data=timestamps)
                out.attrs.update(complete=True, checkpoint_sha256=row['sha256'], primary_sensor=source.primary,
                                 auxiliary_sensor=source.aux or '', common_camera='device3 RGB', GT_input=False)
            source.close(); temporary.replace(target)
            write(receipt, dict(status='complete', name=name, clip=clip['clip'], frames=len(positions), people=total,
                                checkpoint_sha256=row['sha256'], gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                                physical_gpu=int(os.environ['EVENTPOSE_PHYSICAL_GPU']), pid=os.getpid(),
                                pixel_datasets_read=sorted(source.pixel_datasets_read), GT_used_as_input=False,
                                old_matched_instances_missing=missing_old, maximum_joint_difference_from_published_Test_m=max_difference,
                                elapsed_seconds=time.time() - started))
            print(json.dumps(read(receipt)), flush=True)
    finally:
        model.base.detection = original
    write(folder / 'complete.json', dict(status='complete', name=name, clips=5, checkpoint_sha256=row['sha256']))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['infer', 'references', 'render'])
    parser.add_argument('--goal', required=True); parser.add_argument('--name'); parser.add_argument('--clip')
    args = parser.parse_args(); goal = read(args.goal)
    os.environ['EVENTPOSE_MULTISENSOR_OUT'] = goal['source_root']
    sys.path.insert(0, str(ROOT / 'tools'))
    from goal_runtime_control import register_job
    register_job(goal, os.getpid(), 'video_gallery:' + args.stage + ':' + str(args.name or args.clip or 'all'))
    if args.stage == 'infer': infer_meshes(goal, args.name)
    else:
        from multihmr.video_gallery_render import prepare_references, render_video
        if args.stage == 'references': prepare_references(goal)
        else: render_video(goal, args.name, args.clip)


if __name__ == '__main__':
    main()

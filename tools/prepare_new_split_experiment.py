"""Independent, read-only preparation for the user-frozen 2026-10-26 split."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'exp/261026/rgb_event_new_split_24h'
WORK = OUT / 'work'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def test_gt():
    import cv2
    import h5py
    import numpy as np
    import torch
    import smplx
    from utils.pose_evaluation import atomic_npz
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    raw = (OUT / 'split_snapshot.json').read_bytes()
    assert raw == (ROOT / 'dataset/split.json').read_bytes(), 'Frozen split changed'
    split = json.loads(raw)
    target = OUT / 'test'
    target.mkdir(exist_ok=True)
    model = smplx.create(str(ROOT / 'data/human_model_files'), model_type='smplx',
        gender='neutral', num_betas=16, use_pca=False, flat_hand_mean=False).cuda().eval()
    np.save(target / 'joint_regressor.npy', model.J_regressor.cpu().numpy())
    save(OUT / 'gt_gpu_identity.json', dict(pid=os.getpid(),
        visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'], physical_gpu=os.environ['EVENTPOSE_PHYSICAL_GPU'],
        cuda_name=torch.cuda.get_device_name(), started_unix=time.time(),
        split_sha256=hashlib.sha256(raw).hexdigest()))
    tasks = []
    for name, intervals in split['test'].items():
        source = ROOT / 'dataset' / name / 'device0'
        cap = cv2.VideoCapture(str(source / 'rgb.mp4'))
        assert cap.isOpened()
        count = int(cap.get(7)); fps = cap.get(5)
        width, height = int(cap.get(3)), int(cap.get(4)); cap.release()
        with h5py.File(source / 'label.h5', 'r') as f:
            assert f.attrs['complete'] and f.attrs['coordinate_unit'] == 'metre'
            positions = f['frames/source_rgb_position'][:].astype(np.int64)
            stamps = f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
            valid = f['params/person_valid'][:].astype(bool)
            poses = f['params/smplx_pose_axis_angle'][:].astype(np.float32)
            betas = f['params/smplx_betas'][:].astype(np.float32)
            transl = f['params/translation_world_m'][:].astype(np.float32)
        with np.load(source / 'event.npz', allow_pickle=False) as z:
            rgb_times = z['rgb__timestamp_ns']
            assert np.array_equal(stamps, rgb_times[positions]), name
            K = z['rgb_calibration__rgb__intrinsic_matrix'].astype(np.float32)
            R = z['rgb_calibration__rgb__rotation_matrix'].astype(np.float32)
            t = z['rgb_calibration__rgb__translation'].astype(np.float32).reshape(3)
        valid &= np.isfinite(poses).all(-1) & np.isfinite(betas).all(-1) & np.isfinite(transl).all(-1)
        for index, (start, end) in enumerate(intervals):
            assert 0 <= start < end <= count
            key = name + '__segment' + str(index)
            destination = target / 'gt' / (key + '.npz')
            rows = np.flatnonzero((positions >= start) & (positions < end))
            assert len(rows) and positions[rows[0]] == start, (key, 'exact frame0 missing')
            vmask = valid[rows]
            if not destination.exists():
                joints = np.full((*vmask.shape, 55, 3), np.nan, np.float32)
                ids = np.argwhere(vmask)
                for begin in range(0, len(ids), 64):
                    fi, pi = ids[begin:begin+64].T
                    original = rows[fi]
                    p = torch.from_numpy(poses[original, pi]).cuda()
                    b = torch.from_numpy(betas[original, pi]).cuda()
                    tr = torch.from_numpy(transl[original, pi]).cuda()
                    with torch.no_grad():
                        vertices = model(betas=b, global_orient=p[:, :3], body_pose=p[:, 3:66],
                            jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
                            left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165], transl=tr,
                            expression=torch.zeros((len(p), 10), device='cuda')).vertices
                        j = (model.J_regressor @ vertices).cpu().numpy()
                    joints[fi, pi] = j @ R.T + t
                atomic_npz(destination, joints_cam_m=joints, valid=vmask,
                    source_rgb_positions=positions[rows]-start, timestamps_ns=stamps[rows],
                    absolute_source_rgb_positions=positions[rows], original_source_rgb_positions=positions[rows], K=K, R_w2c=R, t_w2c=t,
                    image_shape=[height, width])
            tasks.append(dict(session=key, device='device0', video=str(source/'rgb.mp4'),
                source_video=str(source/'rgb.mp4'), gt_file=str(destination), gt_frames=len(rows),
                valid_gt_instances=int(vmask.sum()), video_frames=end-start, fps=fps,
                width=width, height=height, parent_session=name, segment=index,
                start_frame=start, end_frame=end,
                timestamp_span_seconds=float((rgb_times[end-1]-rgb_times[start])/1e9)))
        save(target / 'preparation_progress.json', dict(prepared_segments=len(tasks),
            last_session=name, updated_unix=time.time()))
        print(json.dumps(dict(session=name, prepared_segments=len(tasks))), flush=True)
    save(target / 'tasks.json', tasks)
    save(target / 'preparation_complete.json', dict(segments=len(tasks),
        frames=sum(t['video_frames'] for t in tasks), split_sha256=hashlib.sha256(raw).hexdigest(),
        interval_semantics='RGB source positions [start,end), exactly as frozen split',
        completed_unix=time.time()))


def train_plan():
    import numpy as np
    raw = (OUT / 'split_snapshot.json').read_bytes()
    assert raw == (ROOT / 'dataset/split.json').read_bytes()
    split = json.loads(raw)
    def ids(name):
        fields = name.split('_')
        return {int(fields[3])} | ({int(fields[5])} if len(fields) == 6 else set())
    parent = {}
    def find(i):
        parent.setdefault(i, i)
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    for name in split['train']:
        people = sorted(ids(name))
        for person in people[1:]:
            parent[find(person)] = find(people[0])
    groups = {}
    for name in split['train']:
        groups.setdefault(find(min(ids(name))), []).append(name)
    values = list(groups.values())
    np.random.default_rng(261026).shuffle(values)
    validation = []; val_ids = set()
    for group in values:
        if len(val_ids) >= 10:
            break
        validation.extend(group)
        val_ids.update(set().union(*(ids(n) for n in group)))
    training = sorted(set(split['train']) - set(validation))
    train_ids = set().union(*(ids(n) for n in training))
    test_ids = set().union(*(ids(n) for n in split['test']))
    assert not train_ids & val_ids and not (train_ids | val_ids) & test_ids
    manifest = dict(seed=261026, model='Multi-HMR 896 L pretrained', device='device0',
        training_sessions=training, validation_sessions=sorted(validation),
        training_subject_ids=sorted(train_ids), validation_subject_ids=sorted(val_ids),
        test_subject_ids=sorted(test_ids), test_sessions=list(split['test']),
        test_split_sha256=hashlib.sha256(raw).hexdigest(),
        training_sampling='maximum 128 equally spaced valid rows per session after 2fps candidate sampling; shared causal-event availability filter',
        validation_sampling='16 seeded valid frames per held-out session; shared causal-event availability filter',
        old_weights_caches_results_reused=False, work_root=str(WORK),
        deadline_unix=json.loads((ROOT/'_agents/research/261026/new_split_agent_goal.json').read_text())['deadline_unix'])
    path = OUT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        save(path, manifest)
    return manifest


def train_cache():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    import numpy as np
    import torch
    import smplx
    import multihmr.data as data
    assert data.WORK.resolve() == WORK.resolve()
    manifest = train_plan()
    for folder in ['images','targets','raw','logs','tmp','validation']:
        (WORK/folder).mkdir(parents=True, exist_ok=True)
    items = [(name, role, 261026+i) for role, key in
        [('validation','validation_sessions'), ('training','training_sessions')]
        for i, name in enumerate(manifest[key])]
    missing = [item for item in items if not (ROOT/'dataset'/item[0]/'device0/label.h5').exists()]
    save(OUT/'missing_training_sources.json', dict(missing=missing,
        policy='Preparation continues on available sessions; missing sessions cannot silently be omitted from final experiment'))
    available = [item for item in items if item not in missing]
    torch.set_num_threads(2)
    model = smplx.create(str(ROOT/'data/human_model_files'), model_type='smplx', gender='neutral',
        num_betas=16, use_pca=False, flat_hand_mean=False).cuda().eval()
    np.save(WORK/'joint_regressor.npy', model.J_regressor.cpu().numpy())
    records = []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(data.extract, item) for item in available]
        for i, future in enumerate(as_completed(futures)):
            name, role, path = future.result()
            records.extend(data.reconstruct(model, name, role, path))
            save(OUT/'train_preparation_progress.json', dict(processed_sessions=i+1,
                available_sessions=len(available), missing_sessions=len(missing), frames=len(records),
                last_session=name, updated_unix=time.time()))
            print(json.dumps(dict(prepared=i+1, session=name, frames=len(records))), flush=True)
    for role in ('training','validation'):
        selected = sorted([r for r in records if r['role']==role], key=lambda r:(r['session'],r['frame']))
        save(WORK/(role+'_index.json'), selected)
    save(OUT/'train_cache_available_complete.json', dict(sessions=len(available),
        missing=missing, records=len(records), completed_unix=time.time()))


def configure_environment():
    os.environ.update(EVENTPOSE_PREPARATION_OUT=str(OUT), EVENTPOSE_EVENT_OUT=str(OUT),
        EVENTPOSE_RGB_CACHE=str(WORK), EVENTPOSE_MAX_TRAINING_FRAMES='128',
        EVENTPOSE_MAX_VALIDATION_FRAMES='16', EVENTPOSE_LOSSLESS_RGB_CACHE='1')


def event_cache():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from multihmr.event_data import prepare_controlled_indices
    from utils.event_encoding import ENCODINGS, self_check
    from tools.prepare_event_cache import cache_session
    started=time.time()
    manifest=json.loads((OUT/'manifest.json').read_text())
    prepare_controlled_indices()
    sessions=[]
    for role in ('validation','training'):
        rows=json.loads((OUT/('controlled_'+role+'_index.json')).read_text())['records']
        for name in sorted({r['session'] for r in rows}):
            sessions.append(dict(session=name,role=role,frames=sorted({r['frame'] for r in rows if r['session']==name})))
    tasks=json.loads((OUT/'test/tasks.json').read_text())
    for name in manifest['test_sessions']:
        frames=sorted({frame for task in tasks if task['parent_session']==name
                       for frame in range(task['start_frame'],task['end_frame'])})
        sessions.append(dict(session=name,role='test',frames=frames))
    config=dict(version=3,size=224,count_cap=256,encodings=ENCODINGS,
        split_sha256=manifest['test_split_sha256'],
        frame_plan_sha256=hashlib.sha256(json.dumps(sessions,sort_keys=True).encode()).hexdigest(),
        quantization='round(255*x), uint8; reader divides by 255',
        windows='corrected integer-nanosecond timestamps, causal (t-window,t]',
        coordinate_frame='rectified event camera; sensor-specific intrinsics and calibrated extrinsics',
        observation_semantics='counts and times describe stored sparse frames, not recovered raw individual events')
    config['sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    root=OUT/'event_cache';root.mkdir(exist_ok=True)
    plan=dict(config=config,sessions=sessions,cache_root=str(root),dataset_read_only=True,
        frames_by_role={role:sum(len(s['frames']) for s in sessions if s['role']==role)
                        for role in ('training','validation','test')})
    save(OUT/'event_cache_plan.json',plan)
    save(OUT/'event_encoding_self_checks.json',self_check())
    results=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(cache_session,(item,config,str(root),manifest['deadline_unix'])) for item in sessions]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            save(OUT/'event_build_status.json',dict(complete_sessions=len(results),total_sessions=len(sessions),
                last_session=result['session'],updated_unix=time.time()))
            print(json.dumps(result),flush=True)
    save(OUT/'event_build_summary.json',dict(status='complete',config=config,results=results,
        sessions=len(results),started_unix=started,completed_unix=time.time(),frames_by_role=plan['frames_by_role']))


def audit_sources():
    import h5py
    import numpy as np
    from utils.event_encoding import calibration
    from utils.paired_statistics import participant_components
    raw=(OUT/'split_snapshot.json').read_bytes()
    assert raw==(ROOT/'dataset/split.json').read_bytes()
    split=json.loads(raw);rows=[];missing=[]
    for role,names in split.items():
        for name in names:
            source=ROOT/'dataset'/name/'device0'
            if not (source/'label.h5').exists():
                missing.append(dict(role=role,session=name));continue
            with h5py.File(source/'label.h5','r') as f:
                assert f.attrs['complete'] and f.attrs['coordinate_unit']=='metre'
                pos=f['frames/source_rgb_position'][:]
                stamps=f['frames/source_rgb_timestamp_ns'][:]
                valid=f['params/person_valid'][:].astype(bool)
                assert np.all(np.diff(pos)>0)
                expected=int(f.attrs['expected_person_count'])
            with np.load(source/'event.npz',allow_pickle=False) as z:
                times=z['timestamp_ns'];rgb=z['rgb__timestamp_ns'];cameras=calibration(z)
                assert np.all(np.diff(times)>=0) and np.all(np.diff(rgb)>0)
                assert np.array_equal(stamps,rgb[pos]),name
                chosen=np.searchsorted(times,rgb[pos],side='right')-1
                covered=(chosen>=0)&(rgb[pos]-times[np.maximum(chosen,0)]<=20_000_000)
                assert np.all(times[chosen[chosen>=0]]<=rgb[pos][chosen>=0])
                rows.append(dict(role=role,session=name,gt_rows=len(pos),valid_gt_frames=int(valid.any(1).sum()),
                    valid_person_frames=int(valid.sum()),rgb_frames=len(rgb),event_frames=len(times),
                    expected_person_count=expected,causal_event_covered_gt_frames=int(covered.sum()),
                    frame_mapping_sha256=hashlib.sha256(pos.tobytes()+stamps.tobytes()).hexdigest(),
                    calibration={k:v.tolist() for k,v in cameras.items()}))
            save(OUT/'source_audit_progress.json',dict(processed=len(rows),last_session=name,updated_unix=time.time()))
    checkpoint=ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'
    digest=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    save(OUT/'source_audit.json',dict(split_sha256=hashlib.sha256(raw).hexdigest(),sessions=rows,
        missing=missing,frame_mapping_verified=True,old_cache_reused=False,
        original_checkpoint=dict(path=str(checkpoint),sha256=digest.hexdigest(),bytes=checkpoint.stat().st_size),
        completed_unix=time.time()))
    metadata=[]
    for name in split['test']:
        fields=name.split('_')
        metadata.append(dict(session=name,participant_ids=[int(fields[3])]+([int(fields[5])] if len(fields)==6 else [])))
    mapping,clusters=participant_components(metadata,split['test'])
    save(OUT/'statistical_analysis_plan.json',dict(frozen_unix=time.time(),reference='rgb_baseline',
        variants=['fusion_'+e for e in ('count33','bins100','surface20','visualize_gray')],
        primary_metric='frame_aligned_pa_mpjpe_mm on paired exact-frame0 eligible common matches',
        primary_family_correction='Holm across four fusion contrasts',cluster_unit='connected participant components preserving repeated/co-occurring sessions',
        clusters=clusters,session_to_cluster=mapping,bootstrap_replicates=10000,
        main_metrics=['first_frame_aligned_mpjpe_mm','first_frame_aligned_pa_mpjpe_mm',
            'frame_aligned_mpjpe_mm','frame_aligned_pa_mpjpe_mm','root_depth_error_mm'],
        root_depth_definition='1000*mean(abs(predicted_camera_pelvis_Z-GT_camera_pelvis_Z)), without alignment',
        root_3d_definition='1000*mean(norm(predicted_camera_pelvis-GT_camera_pelvis)), supplementary',
        temporal_clock='corrected source RGB timestamp_ns, no missing-frame bridging',
        missing_policy='No detection is not zero error; full GT denominator reported separately',
        motion_bins=dict(still_max_mps=.05,fast_min_mps=.5),seed_replication=False,
        seed_limitation='Cluster bootstrap does not measure training-seed uncertainty'))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('stage', choices=['test','train','plan','events','audit'], default='test', nargs='?')
    args=p.parse_args()
    configure_environment()
    {'test':test_gt, 'train':train_cache, 'plan':train_plan,'events':event_cache,'audit':audit_sources}[args.stage]()

"""Audit every frozen input mapping and independently reconstruct sampled GT on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import h5py
import numpy as np
from tools.prepare_device34_body import OUT,save


def main():
    started=time.time();manifest=json.loads((OUT/'manifest.json').read_text())
    sha=hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest();assert sha==manifest['split_sha256']
    indices={r:json.loads((OUT/(r+'_index.json')).read_text()) for r in ('training','validation')}
    assert len(indices['training'])==6501 and len(indices['validation'])==288
    for role,key in [('training','training_sessions'),('validation','validation_sessions')]:
        assert {r['session'] for r in indices[role]}==set(manifest[key])
        assert len({(r['session'],r['frame']) for r in indices[role]})==len(indices[role])
    records=[];gt_samples=[];total_cached=0
    for name in manifest['training_sessions']+manifest['validation_sessions']+manifest['test_sessions']:
        with np.load(OUT/'source_maps'/(name+'.npz')) as z:maps={k:z[k] for k in z.files}
        with np.load(OUT/'targets'/(name+'.npz')) as z:target={k:z[k] for k in z.files}
        with h5py.File(OUT/'cache'/(name+'.h5'),'r',locking=False) as f:
            assert f.attrs['complete'] and f.attrs['split_sha256']==sha
            frames=f['frames'][:];times=f['timestamp_ns'][:]
            assert np.array_equal(times,maps['canonical_rgb_timestamp_ns'][frames])
            assert np.array_equal(frames,target['positions']) and np.all(np.diff(frames)>0)
            assert np.isfinite(target['joints'][target['valid']]).all()
            for sensor in ('rgb3','rgb4','event3','event4'):
                event=sensor.startswith('event');kind='event' if event else 'rgb';prefix='device'+sensor[-1]+'_'
                shape=(len(frames),2,224,224) if event else (len(frames),3,896,896)
                assert f[sensor].shape==shape and f[sensor].dtype==np.uint8
                assert np.array_equal(f[sensor+'_available'][:],maps[prefix+kind+'_available'][frames])
                if event:
                    mask=f[sensor+'_available'][:];assert np.all(f[sensor+'_latest_timestamp_ns'][:][mask]<=times[mask])
                R=maps[prefix+kind+'_R']@maps['device3_rgb_R'].T
                t=maps[prefix+kind+'_t']-R@maps['device3_rgb_t']
                assert np.allclose(f[sensor+'_R_common_to_sensor'][:],R,atol=1e-8)
                assert np.allclose(f[sensor+'_t_common_to_sensor'][:],t,atol=1e-8)
                assert np.allclose(R@R.T,np.eye(3),atol=1e-5) and np.isclose(np.linalg.det(R),1,atol=1e-5)
            for role in indices:
                for row in [r for r in indices[role] if r['session']==name]:
                    i=row['cache_row'];assert frames[i]==row['frame'] and target['valid'][i].any()
                    assert all(bool(f[s+'_available'][i]) for s in ('rgb3','rgb4','event3','event4'))
            total_cached+=len(frames)
            records.append(dict(session=name,frames=len(frames),frame_mapping_sha256=hashlib.sha256(frames.tobytes()+times.tobytes()).hexdigest()))
        if name in manifest['test_sessions']:
            valid=np.argwhere(target['valid'])
            selected=np.linspace(0,len(valid)-1,min(3,len(valid)),dtype=int)
            gt_samples.extend((name,int(valid[i,0]),int(valid[i,1]),target['joints'][tuple(valid[i])].copy()) for i in selected)
    tasks=json.loads((OUT/'test/tasks.json').read_text());split=json.loads((OUT/'split_snapshot.json').read_text())
    assert len(tasks)==80 and sum(t['video_frames'] for t in tasks)==24080
    expected={(n,i,a,b) for n in manifest['test_sessions'] for i,(a,b) in enumerate(split['test'][n])}
    assert {(t['parent_session'],t['segment'],t['start_frame'],t['end_frame']) for t in tasks}==expected
    for task in tasks:
        with np.load(task['gt_file']) as z:
            pos=z['source_rgb_positions'];assert np.array_equal(pos,np.arange(task['video_frames']))
            assert np.array_equal(z['original_source_rgb_positions'],pos+task['start_frame'])
            with np.load(OUT/'source_maps'/(task['parent_session']+'.npz')) as m:
                assert np.array_equal(z['timestamps_ns'],m['canonical_rgb_timestamp_ns'][pos+task['start_frame']])
    import torch
    import smplx
    torch.set_num_threads(2)
    model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',
        num_betas=16,use_pca=False,flat_hand_mean=False).cpu().eval()
    assert np.array_equal(np.load(OUT/'joint_regressor.npy'),model.J_regressor.numpy())
    errors=[]
    for name,row,person,expected_joints in gt_samples:
        with np.load(OUT/'raw'/(name+'.npz')) as z:
            p=torch.from_numpy(z['poses'][row,person][None]);b=torch.from_numpy(z['betas'][row,person][None])
            tr=torch.from_numpy(z['transl'][row,person][None]);R=z['R'];t=z['t']
        with torch.no_grad():
            v=model(betas=b,global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],leye_pose=p[:,69:72],
                reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],right_hand_pose=p[:,120:165],transl=tr,
                expression=torch.zeros((1,10))).vertices
            actual=(model.J_regressor@v)[0].numpy()@R.T+t
        error=float(np.max(np.abs(actual-expected_joints)));assert error<2e-5,(name,row,error)
        errors.append(error)
    save(OUT/'actual_input_audit.json',dict(status='passed',split_sha256=sha,sessions=len(records),
        cached_frames=total_cached,training_frames=len(indices['training']),validation_frames=len(indices['validation']),
        test_segments=len(tasks),test_frames=sum(t['video_frames'] for t in tasks),
        all_frame_mappings_verified=True,all_event_causality_verified=True,all_relative_calibrations_verified=True,
        independent_cpu_GT_samples=len(errors),max_cpu_GPU_GT_difference_m=max(errors),
        records=records,started_unix=started,completed_unix=time.time()))
    print(json.dumps(dict(status='passed',sessions=len(records),GT_samples=len(errors),max_GT_difference_m=max(errors))),flush=True)


if __name__=='__main__':main()

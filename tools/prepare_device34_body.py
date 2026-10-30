"""Read-only source audit and preparation for the six-condition body study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261026/device34_body_28h'
GOAL=ROOT/'_agents/research/261026/device34_body_goal_28h.json'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def audit(wait=False):
    import h5py
    import numpy as np
    import cv2
    from utils.event_encoding import calibration
    from utils.pose_evaluation import atomic_npz
    source_split=ROOT/'dataset/split.json'
    deadline=json.loads(GOAL.read_text())['deadline_unix']
    if wait and not source_split.is_file():
        save(OUT/'preparation_state.json',dict(status='waiting_for_actual_dataset_mount',
            path=str(source_split),started_unix=time.time(),deadline_unix=deadline))
    while wait and not source_split.is_file() and time.time()<deadline-300:time.sleep(30)
    raw=source_split.read_bytes();split=json.loads(raw)
    body={role:sorted(n for n in split[role] if n.endswith('_b')) for role in ('train','test')}
    assert all(body.values())
    def ids(names):return {int(n.split('_')[3]) for n in names}
    assert not ids(body['train'])&ids(body['test'])
    snapshot=OUT/'split_snapshot.json'
    if snapshot.exists():assert snapshot.read_bytes()==raw
    else:snapshot.write_bytes(raw)
    missing=[dict(role=role,session=n,device=d,file=f) for role,names in body.items() for n in names
        for d in (0,3,4) for f in ('label.h5','event.npz','rgb.mp4')
        if not (ROOT/'dataset'/n/('device'+str(d))/f).is_file()]
    save(OUT/'scope_audit.json',dict(source=str(source_split.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
        body=body,train_subjects=sorted(ids(body['train'])),test_subjects=sorted(ids(body['test'])),
        overlap=[],missing=missing,frozen_unix=time.time()))
    assert not missing,missing
    rng=np.random.default_rng(261026);subjects=np.array(sorted(ids(body['train'])))
    rng.shuffle(subjects);holdout=set(map(int,subjects[:max(4,round(len(subjects)*.15))]))
    manifest=dict(split_sha256=hashlib.sha256(raw).hexdigest(),seed=261026,
        training_sessions=[n for n in body['train'] if int(n.split('_')[3]) not in holdout],
        validation_sessions=[n for n in body['train'] if int(n.split('_')[3]) in holdout],
        test_sessions=body['test'],train_subjects=sorted(ids(body['train'])-holdout),
        validation_subjects=sorted(holdout),test_subjects=sorted(ids(body['test'])),
        common_coordinates='device3 rectified RGB camera; all primary sensor predictions transformed into it',
        common_clock='corrected device0 RGB timestamp_ns at the split source frame position',
        rgb_mapping='latest corrected sensor RGB timestamp <= common timestamp, max age 50ms',
        event_mapping='causal encoding ending at common timestamp; max latest stored observation age 20ms',
        event_encoding='count33 polarity-separated 2 channels, fixed before any new Test result',
        event_semantics='stored sparse frame observations, not recovered individual raw events',
        training_sampling='128 equally spaced valid common-available rows per session',
        validation_sampling='32 fixed seeded valid common-available rows per held-out subject session',
        trial_hours=3,maximum_trial_hours=4,selection='Train holdout only',
        old_weights_caches_results_reused=False,created_unix=time.time())
    manifest_path=OUT/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        assert all(previous[k]==v for k,v in manifest.items() if k!='created_unix')
        manifest=previous
    else:save(manifest_path,manifest)
    for role,names in body.items():
        for name in names:
            folder=ROOT/'dataset'/name
            with h5py.File(folder/'device0/label.h5','r',locking=False) as f:
                assert f.attrs['complete'] and f.attrs['coordinate_unit']=='metre'
                positions=f['frames/source_rgb_position'][:].astype(np.int64)
                stamps=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
                valid=f['params/person_valid'][:].astype(bool)
            with np.load(folder/'device0/event.npz',allow_pickle=False) as z:
                clock=z['rgb__timestamp_ns'].astype(np.int64)
            assert np.array_equal(stamps,clock[positions]),name
            assert np.all(np.diff(positions)>0) and np.all(np.diff(clock)>0)
            arrays=dict(canonical_rgb_timestamp_ns=clock,gt_source_positions=positions,
                gt_timestamp_ns=stamps,gt_valid=valid)
            summaries={}
            available=np.ones(len(clock),bool)
            for device in (3,4):
                src=folder/('device'+str(device))
                with np.load(src/'event.npz',allow_pickle=False) as z:
                    times=z['timestamp_ns'].astype(np.int64);rgb=z['rgb__timestamp_ns'].astype(np.int64)
                    cameras=calibration(z)
                    rr=z['rgb_calibration__rgb__rotation_matrix'].astype(np.float64)
                    tr=z['rgb_calibration__rgb__translation'].astype(np.float64).reshape(3)
                    re=z['rgb_calibration__event__rotation_matrix'].astype(np.float64)
                    te=z['rgb_calibration__event__translation'].astype(np.float64).reshape(3)
                assert np.all(np.diff(rgb)>0) and np.all(np.diff(times)>=0)
                with h5py.File(src/'label.h5','r',locking=False) as f:
                    assert f.attrs['complete'] and f.attrs['coordinate_unit']=='metre'
                    local_pos=f['frames/source_rgb_position'][:]
                    assert np.array_equal(f['frames/source_rgb_timestamp_ns'][:],rgb[local_pos])
                    assert str(f.attrs['world_coordinate_frame'])=='Busan26 world == logical device0 RGB camera'
                rp=np.searchsorted(rgb,clock,side='right')-1
                ep=np.searchsorted(times,clock,side='right')-1
                ra=(rp>=0)&(clock-rgb[np.maximum(rp,0)]<=50_000_000)
                ea=(ep>=0)&(clock-times[np.maximum(ep,0)]<=20_000_000)
                assert np.all(rgb[rp[rp>=0]]<=clock[rp>=0])
                assert np.all(times[ep[ep>=0]]<=clock[ep>=0])
                available&=ra&ea
                cap=cv2.VideoCapture(str(src/'rgb.mp4'));assert cap.isOpened()
                shape=np.array([int(cap.get(4)),int(cap.get(3))]);count=int(cap.get(7));cap.release()
                assert count==len(rgb),(name,device,count,len(rgb))
                prefix='device'+str(device)+'_'
                for key,value in dict(rgb_positions=rp,rgb_available=ra,event_available=ea,
                    rgb_timestamp_ns=rgb,event_timestamp_ns=times,rgb_R=rr,rgb_t=tr,event_R=re,event_t=te,
                    rgb_K=cameras['K_rgb'],event_K=cameras['K_event'],rgb_shape=shape).items():arrays[prefix+key]=value
                summaries[str(device)]=dict(rgb_frames=len(rgb),event_frames=len(times),rgb_available=int(ra.sum()),event_available=int(ea.sum()),
                    label_clock_verified=True,baseline_m=float(np.linalg.norm(cameras['t_rgb_to_event'])))
            arrays['all_sensors_available']=available
            atomic_npz(OUT/'source_maps'/(name+'.npz'),**arrays)
            save(OUT/'source_summaries'/(name+'.json'),dict(role=role,session=name,gt_rows=len(positions),
                valid_gt_frames=int(valid.any(1).sum()),valid_person_frames=int(valid.sum()),
                gt_mapping_sha256=hashlib.sha256(positions.tobytes()+stamps.tobytes()).hexdigest(),sensors=summaries))
            print(json.dumps(dict(audited=name,role=role)),flush=True)
    save(OUT/'source_audit_complete.json',dict(completed_unix=time.time(),split_sha256=manifest['split_sha256'],
        train_sessions=len(body['train']),test_sessions=len(body['test']),
        test_segments=sum(len(split['test'][n]) for n in body['test'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','cache'],default='audit',nargs='?')
    p.add_argument('--wait-for-mount',action='store_true');a=p.parse_args()
    if a.stage=='audit':audit(a.wait_for_mount)
    else:
        from multihmr.multisensor_preparation import prepare_all
        prepare_all()

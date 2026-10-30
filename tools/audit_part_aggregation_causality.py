"""Read-only timestamp/label audit of the frozen aggregation cache."""
import hashlib
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(path.read_text())

def main():
    out=ROOT/'exp/261029/part_aware_dynamic_aggregation'
    config=read(out/'configs/v3_part_dynamic.json')
    source=Path(config['source_root']);manifest=read(source/'manifest.json')
    splitsha=hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()
    assert splitsha==config['split_sha256']
    names=sorted(set(sum([manifest[role+'_sessions'] for role in ['training','validation','test']],[])))
    counts=0;maximum_age=0;source_preserved=0;delay_by_frame={};baselines=[];rotation_differences=[]
    for name in names:
        with h5py.File(source/'cache'/(name+'.h5'),'r',locking=False) as original:
            assert original.attrs['common_camera']=='device3 RGB'
            R=original['rgb3_R_common_to_sensor'][:];t=original['rgb3_t_common_to_sensor'][:]
            assert np.allclose(R,np.eye(3),atol=1e-6) and np.allclose(t,0,atol=1e-6)
            Ra=original['event3_R_common_to_sensor'][:];ta=original['event3_t_common_to_sensor'][:]
            relative=Ra@R.T
            baselines.append(float(np.linalg.norm(ta-relative@t)))
            rotation_differences.append(float(np.max(np.abs(relative-np.eye(3)))))
        with h5py.File(out/'memory_cache'/(name+'.h5'),'r',locking=False) as f:
            assert f.attrs['complete'] and f.attrs['split_sha256']==splitsha
            query=f['event_query_timestamp_ns'][:];clock=f['timestamp_ns'][:]
            assert np.array_equal(query,f['rgb3_timestamp_ns'][:])
            assert (query<=clock).all() and (clock-query<=50_000_000).all()
            ages=f['bin_age_bounds_ms'][:];assert np.array_equal(ages,[[0,33],[33,100],[100,200]])
            earliest=f['bin_earliest_timestamp_ns'][:];latest=f['bin_latest_timestamp_ns'][:]
            for i,(near,far) in enumerate(ages):
                present=latest[:,i]>=0
                assert (earliest[present,i]>query[present]-int(far)*1_000_000).all()
                assert (earliest[present,i]<=latest[present,i]).all()
                assert (latest[present,i]<=query[present]-int(near)*1_000_000).all()
            maximum_age=max(maximum_age,int((clock-query).max()));counts+=len(query)
            delay_by_frame.update({(name,int(frame)):int(delay) for frame,delay in zip(f['frames'][:],clock-query)})
            archive=Path(f.attrs['source_event_path']);stat=archive.stat()
            assert stat.st_size==f.attrs['source_event_bytes'] and stat.st_mtime_ns==f.attrs['source_event_mtime_ns']
            source_preserved+=1
    labels={};planned_delays={}
    for role,filename in [('training','training_part_motion_targets.npz'),('validation','validation_part_motion_training_targets.npz')]:
        with np.load(out/'diagnostics'/filename,allow_pickle=False) as f:
            assert not bool(f['inference_input_allowed'])
            assert bool(f['training_supervision_allowed'])==(role=='training')
            keys=list(zip(f['sessions'].tolist(),f['frames'].tolist()))
            assert len(set(keys))==len(keys)
            values=[delay_by_frame[key] for key in keys]
            planned_delays[role]=dict(queries=len(values),maximum_ms=max(values)/1e6)
            assert set(f['sessions'].tolist())<=set(manifest[role+'_sessions'])
            for name in set(f['sessions'].tolist()):
                indices=np.flatnonzero(f['sessions']==name)
                with h5py.File(out/'memory_cache'/(name+'.h5'),'r',locking=False) as cache:
                    mapping=dict(zip(cache['frames'][:].tolist(),cache['event_query_timestamp_ns'][:].tolist()))
                    assert all(int(f['query_timestamp_ns'][i])==mapping[int(f['frames'][i])] for i in indices)
            labels[role]=dict(queries=len(keys),valid_queries=int(f['valid'].sum()),
                labelled_part_cells=int((f['valid'][:,None]&f['part_valid']).sum()),
                inference_input_allowed=False,optimizer_supervision_allowed=role=='training')
    tasks=read(source/'test/tasks.json')
    values=[delay_by_frame[(task['parent_session'],task['start_frame']+position)]
        for task in tasks for position in range(task['video_frames'])]
    planned_delays['test']=dict(queries=len(values),maximum_ms=max(values)/1e6)
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        cache_sessions=len(names),cache_queries=counts,maximum_RGB_to_GT_delay_ms=maximum_age/1e6,
        planned_query_RGB_to_GT_delay=planned_delays,
        root_depth_coordinate_frame='Common device3 RGB camera Z; RGB3 common-to-sensor transform is identity within 1e-6 for all sessions',
        RGB_event_camera_baseline_mm=[1000*min(baselines),1000*max(baselines)],
        RGB_event_rotation_identity_max_deviation_range=[min(rotation_differences),max(rotation_differences)],
        original_event_archive_stat_preserved=source_preserved,split_sha256=splitsha,motion_labels=labels,
        scope='Read-only cached timestamps, label joins, original event size/mtime; no training/evaluation progress or GPU inspection',
        limitation='Archive stat equality is not a cryptographic proof of original archive byte equality; cached event pixels were not regenerated')
    (out/'single_owner_causality_and_motion_label_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

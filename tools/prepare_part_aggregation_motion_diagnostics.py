"""CPU-only GT motion labels and diagnostics for temporal event aggregation.

GT never enters inference inputs. ``--role training`` exports auxiliary
supervision labels for a separately configured model; validation and the
original holdout diagnostic stay excluded from optimizer updates. The native
30 Hz GT trajectory is linearly interpolated at the window start, using only
labels no later than the actual RGB3 input time. The latest past GT is held
to that input time; its age is recorded rather than hidden by extrapolation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import h5py
import numpy as np
import torch
import smplx
from smplx.lbs import batch_rodrigues,batch_rigid_transform
from multihmr.part_aggregation_model import PART_NAMES,PART_JOINTS


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def source_stat(path):
    s=Path(path).stat()
    return dict(path=str(Path(path).resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


class FastBodyJoints:
    """Exact SMPL-X rest-joint shape blend followed by official rigid FK."""
    def __init__(self,model):
        self.model=model
        with torch.no_grad():
            self.rest_template=model.J_regressor@model.v_template
            self.rest_shapedirs=torch.einsum('jv,vcl->jcl',model.J_regressor,model.shapedirs)

    def __call__(self,pose,betas,transl,batch_size=128):
        result=[]
        with torch.no_grad():
            for start in range(0,len(pose),batch_size):
                p=torch.as_tensor(pose[start:start+batch_size],dtype=torch.float32)
                b=torch.as_tensor(betas[start:start+batch_size],dtype=torch.float32)
                t=torch.as_tensor(transl[start:start+batch_size],dtype=torch.float32)
                rest=self.rest_template[None]+torch.einsum('bl,jcl->bjc',b,self.rest_shapedirs)
                rotation=batch_rodrigues((p+self.model.pose_mean).reshape(-1,3)).reshape(-1,55,3,3)
                joints,_=batch_rigid_transform(rotation,rest,self.model.parents)
                result.append((joints[:,:22]+t[:,None]).numpy())
        return np.concatenate(result)


def official_joints(model,pose,betas,transl):
    with torch.no_grad():
        p=torch.as_tensor(pose,dtype=torch.float32);b=torch.as_tensor(betas,dtype=torch.float32)
        return model(betas=b,global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],
            leye_pose=p[:,69:72],reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],
            right_hand_pose=p[:,120:165],transl=torch.as_tensor(transl,dtype=torch.float32),
            expression=torch.zeros((len(p),10))).joints[:,:22].numpy()


def interpolate_start_and_hold_end(times,points,start,end):
    """All times are past-only. No interpolation uses a GT newer than end."""
    times=np.asarray(times,np.int64)
    assert len(times) and times[-1]<=end and np.all(np.diff(times)>0)
    lo=int(np.searchsorted(times,start,side='right'))-1
    coverage=lo>=0
    if lo<0:
        first=points[0]
    elif lo+1<len(times):
        fraction=(int(start)-int(times[lo]))/(int(times[lo+1])-int(times[lo]))
        first=points[lo]*(1-fraction)+points[lo+1]*fraction
    else:first=points[lo]
    keep=np.flatnonzero((times>start)&(times<=end))
    trajectory=np.concatenate([first[None],points[keep],points[-1:]],axis=0)
    knots=np.concatenate([[int(start)],times[keep],[int(end)]])
    assert len(knots)==len(trajectory) and np.all(np.diff(knots)>=0)
    return trajectory,knots,bool(coverage)


def projected(points,R,t,K):
    camera=points@R.T+t
    hom=camera@K.T
    uv=hom[...,:2]/np.maximum(hom[...,2:],1e-8)
    return uv,camera


def grouped_path(points):
    joint_path=np.linalg.norm(np.diff(points,axis=0),axis=-1).sum(0)
    joint_displacement=np.linalg.norm(points[-1]-points[0],axis=-1)
    return np.array([joint_path[list(ids)].mean() for ids in PART_JOINTS]),np.array([
        joint_displacement[list(ids)].mean() for ids in PART_JOINTS])


def grouped_displacement_vector(points):
    """Mean signed end-minus-start joint displacement, in the supplied axes."""
    joint_vector=points[-1]-points[0]
    return np.stack([joint_vector[list(ids)].mean(0) for ids in PART_JOINTS])


def self_check():
    # A return motion has a nonzero path and zero endpoint displacement.
    q=1_700_000_000_000_000_000
    times=q+np.array([0,10,20],np.int64)*1_000_000
    points=np.zeros((3,22,3));points[1,:,0]=1
    trace,knots,coverage=interpolate_start_and_hold_end(times,points,q,q+25_000_000)
    path,displacement=grouped_path(trace)
    assert coverage and np.allclose(path,2) and np.allclose(displacement,0)
    trace,_,_=interpolate_start_and_hold_end(times,points,q+5_000_000,q+25_000_000)
    assert np.allclose(trace[0,:,0],.5)
    assert np.allclose(trace[-1],points[-1])
    vector=grouped_displacement_vector(trace)
    assert np.allclose(vector[:,0],-.5) and np.allclose(vector[:,1:],0)
    return dict(path_not_endpoint_displacement=True,start_linear_interpolation=True,
        end_latest_past_hold=True,integer_ns_clock=True,signed_displacement_vector=True)


def main(args):
    torch.set_num_threads(2)
    source=Path(args.source_root);out=Path(args.output_root);dest=out/'diagnostics'
    dest.mkdir(parents=True,exist_ok=True)
    started=time.time();checks=self_check()
    role=args.role
    index_role='training' if role=='training' else 'validation'
    prefix={'holdout':'holdout_part_motion_targets','training':'training_part_motion_targets',
        'validation':'validation_part_motion_training_targets'}[role]
    target=dest/(prefix+'.npz');summary_path=dest/(prefix+'.json')
    # Never silently replace an earlier diagnostic or a completed label export.
    if target.exists() or summary_path.exists():
        raise FileExistsError('Preserve completed motion targets: '+str(target))
    protected={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (dest/'holdout_part_motion_targets.npz',dest/'holdout_part_motion_targets.json')
        if path.exists()}
    manifest=json.loads((source/'manifest.json').read_text())
    records=json.loads((source/(index_role+'_index.json')).read_text())
    sessions=sorted({r['session'] for r in records})
    assert sessions==sorted(manifest[index_role+'_sessions'])
    assert len(records)==(6501 if index_role=='training' else 288), 'Frozen Train/holdout query plan changed'
    assert not set(sessions)&set(manifest['test_sessions'])
    other_role='validation' if index_role=='training' else 'training'
    assert not set(sessions)&set(manifest[other_role+'_sessions'])
    splitsha=hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()
    assert splitsha==manifest['split_sha256']
    model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',
        num_betas=16,use_pca=False,flat_hand_mean=False).cpu().eval()
    fast=FastBodyJoints(model)
    scalar_keys=['query_timestamp_ns','common_GT_timestamp_ns','window_start_timestamp_ns',
        'latest_used_GT_timestamp_ns','latest_GT_hold_age_ms','RGB_to_common_GT_lag_ms',
        'max_GT_knot_gap_ms','causal_GT_rows_used','start_covered_by_past_GT','diagnostic_valid']
    array_keys=['event3_part_path_px','event3_part_displacement_px',
        'rgb3_part_path_px','rgb3_part_displacement_px','camera3D_part_path_m',
        'camera3D_part_displacement_m','pelvis_relative3D_part_path_m',
        'pelvis_relative3D_part_displacement_m','event3_part_visible_fraction',
        'rgb3_part_visible_fraction','event3_joint_path_px','rgb3_joint_path_px',
        'event3_end_joint_uv','rgb3_end_joint_uv',
        'event3_part_displacement_uv_px','rgb3_part_displacement_uv_px']
    output={key:[None]*len(records) for key in scalar_keys+array_keys}
    provenance=[]
    for session in sessions:
        if args.deadline_unix and time.time()>args.deadline_unix-120:
            raise TimeoutError('Diagnostic preparation reached fixed deadline')
        labelpath=ROOT/'dataset'/session/'device0/label.h5';before=source_stat(labelpath)
        ids=[i for i,r in enumerate(records) if r['session']==session]
        local=[records[i] for i in ids]
        with np.load(source/'source_maps'/(session+'.npz'),allow_pickle=False) as z:
            R=z['device3_rgb_R'].astype(np.float64);t=z['device3_rgb_t'].astype(np.float64)
            K=z['device3_rgb_K'].astype(np.float64)
            Re=z['device3_event_R'].astype(np.float64);te=z['device3_event_t'].astype(np.float64)
            Ke=z['device3_event_K'].astype(np.float64)
            common=np.array([z['canonical_rgb_timestamp_ns'][r['frame']] for r in local],np.int64)
            rgb_positions=np.array([z['device3_rgb_positions'][r['frame']] for r in local],np.int64)
            query=z['device3_rgb_timestamp_ns'][rgb_positions].astype(np.int64)
            gt_clock=z['gt_timestamp_ns'].astype(np.int64)
            rgb_shape=z['device3_rgb_shape'].astype(np.int64)
        assert np.all(query<=common) and np.all(common-query<=50_000_000)
        with h5py.File(source/'cache'/(session+'.h5'),'r',locking=False) as f:
            event_shape=f['event3_native_shape'][:]
            for r,c in zip(local,common):
                assert int(f['frames'][r['cache_row']])==r['frame']
                assert int(f['timestamp_ns'][r['cache_row']])==int(c)
        memory=out/'memory_cache'/(session+'.h5')
        if memory.exists():
            with h5py.File(memory,'r',locking=False) as f:
                assert np.array_equal(f['event_query_timestamp_ns'][[r['cache_row'] for r in local]],query)
        plans=[];needed=set()
        with h5py.File(labelpath,'r',locking=False) as f:
            times=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
            assert np.array_equal(times,gt_clock)
            assert f['params/person_valid'].shape[1]==1, 'Explicit multi-person diagnostic mapping required'
            valid=f['params/person_valid'][:,0].astype(bool)
            valid_rows=np.flatnonzero(valid);valid_times=times[valid_rows]
            assert np.all(np.diff(valid_times)>0)
            for q in query:
                start=int(q)-args.window_ms*1_000_000
                lo=max(0,int(np.searchsorted(valid_times,start,side='right'))-1)
                hi=int(np.searchsorted(valid_times,int(q),side='right'))
                rows=valid_rows[lo:hi] if hi>lo else np.array([],np.int64)
                assert np.all(times[rows]<=q)
                plans.append(rows);needed.update(rows.tolist())
            selected=np.array(sorted(needed),np.int64)
            pose=f['params/smplx_pose_axis_angle'][selected,0]
            betas=f['params/smplx_betas'][selected,0]
            transl=f['params/translation_world_m'][selected,0]
        assert np.isfinite(pose).all() and np.isfinite(betas).all() and np.isfinite(transl).all()
        joints=fast(pose,betas,transl)
        reference=official_joints(model,pose[:2],betas[:2],transl[:2])
        parity=float(np.abs(reference-joints[:2]).max());assert parity<2e-5,parity
        for output_i,rows,q,c in zip(ids,plans,query,common):
            if not len(rows):
                # A latest RGB can precede the first valid GT. Retain the query
                # and mask its supervision rather than borrowing future GT.
                values={}
                for key in scalar_keys:values[key]=0
                for key in array_keys:
                    shape=(22,2) if key.endswith('_end_joint_uv') else (22,) if key.endswith('_joint_path_px') else (6,2) if key.endswith('_displacement_uv_px') else (6,)
                    values[key]=np.zeros(shape,np.float64)
                values.update(query_timestamp_ns=int(q),common_GT_timestamp_ns=int(c),
                    window_start_timestamp_ns=int(q)-args.window_ms*1_000_000,
                    latest_used_GT_timestamp_ns=-1,latest_GT_hold_age_ms=-1,
                    RGB_to_common_GT_lag_ms=float((int(c)-int(q))/1e6),
                    start_covered_by_past_GT=False,diagnostic_valid=False)
                for key in output:output[key][output_i]=values[key]
                continue
            row_positions=np.searchsorted(selected,rows)
            assert np.array_equal(selected[row_positions],rows)
            start=int(q)-args.window_ms*1_000_000
            trace,knots,covered=interpolate_start_and_hold_end(times[rows],joints[row_positions],start,int(q))
            uv,cam=projected(trace,R,t,K);uve,came=projected(trace,Re,te,Ke)
            relative=cam-cam[:,:1]
            px,disp=grouped_path(uv);pxe,dispe=grouped_path(uve)
            m,mdisp=grouped_path(cam);rm,rdisp=grouped_path(relative)
            gap=float(np.max(np.diff(times[rows]))/1e6) if len(rows)>1 else 0.
            endage=float((int(q)-int(times[rows[-1]]))/1e6)
            values=dict(query_timestamp_ns=int(q),common_GT_timestamp_ns=int(c),window_start_timestamp_ns=start,
                latest_used_GT_timestamp_ns=int(times[rows[-1]]),latest_GT_hold_age_ms=endage,
                RGB_to_common_GT_lag_ms=float((int(c)-int(q))/1e6),max_GT_knot_gap_ms=gap,
                causal_GT_rows_used=len(rows),start_covered_by_past_GT=covered,
                diagnostic_valid=bool(covered and endage<=67 and gap<=67 and (cam[...,2]>.05).all() and (came[...,2]>.05).all()),
                event3_part_path_px=pxe,event3_part_displacement_px=dispe,
                rgb3_part_path_px=px,rgb3_part_displacement_px=disp,
                camera3D_part_path_m=m,camera3D_part_displacement_m=mdisp,
                pelvis_relative3D_part_path_m=rm,pelvis_relative3D_part_displacement_m=rdisp,
                event3_joint_path_px=np.linalg.norm(np.diff(uve,axis=0),axis=-1).sum(0),
                rgb3_joint_path_px=np.linalg.norm(np.diff(uv,axis=0),axis=-1).sum(0),
                event3_end_joint_uv=uve[-1],rgb3_end_joint_uv=uv[-1])
            values.update(event3_part_displacement_uv_px=grouped_displacement_vector(uve),
                rgb3_part_displacement_uv_px=grouped_displacement_vector(uv))
            for prefix,coordinates,camera,shape in [('rgb3',uv,cam,rgb_shape),('event3',uve,came,event_shape)]:
                h,w=map(int,shape)
                visible=(camera[...,2]>.05)&(coordinates[...,0]>=0)&(coordinates[...,0]<w)&(coordinates[...,1]>=0)&(coordinates[...,1]<h)
                values[prefix+'_part_visible_fraction']=np.array([visible[:,list(group)].mean() for group in PART_JOINTS])
            for key in output:output[key][output_i]=values[key]
        after=source_stat(labelpath);assert after==before
        provenance.append(dict(session=session,source=before,source_stat_preserved=True,
            reconstructed_GT_rows=len(selected),queries=len(ids),official_SMPLX_parity_max_m=parity,
            RGB3_K=K.tolist(),event3_K=Ke.tolist(),native_RGB3_shape=rgb_shape.tolist(),native_event3_shape=event_shape.tolist()))
        print(json.dumps(dict(event=role+'_motion_session_ready',session=session,queries=len(ids),
            GT_rows=len(selected),official_parity_m=parity,elapsed_seconds=time.time()-started)),flush=True)
    arrays={key:np.asarray(value) for key,value in output.items()}
    assert all(len(value)==len(records) for value in arrays.values())
    assert all(np.isfinite(value).all() for value in arrays.values())
    arrays.update(session=np.array([r['session'] for r in records]),
        sessions=np.array([r['session'] for r in records]),
        subject_id=np.array([int(r['session'].split('_')[3]) for r in records],np.int64),
        global_frame=np.array([r['frame'] for r in records],np.int64),
        frames=np.array([r['frame'] for r in records],np.int64),
        cache_row=np.array([r['cache_row'] for r in records],np.int64),
        record_index=np.arange(len(records)),part_names=np.array(PART_NAMES),
        window_ms=np.array(args.window_ms),diagnostic_GT_only=np.array(role=='holdout'),
        inference_input_allowed=np.array(False),training_supervision_allowed=np.array(role=='training'),
        validation_only=np.array(index_role=='validation'))
    if index_role=='validation':arrays['validation_index']=np.arange(len(records))
    arrays['valid']=arrays['diagnostic_valid'].astype(bool)
    arrays['part_valid']=np.broadcast_to(arrays['valid'][:,None],(len(records),len(PART_NAMES))).copy()
    for path,digest in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    tmp=target.with_name(target.stem+'.%d.tmp.npz'%os.getpid())
    np.savez_compressed(tmp,**arrays);tmp.replace(target)
    summary=dict(status='passed',path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        role=role,index_role=index_role,queries=len(records),sessions=len(sessions),subject_clusters=len(set(arrays['subject_id'].tolist())),
        part_names=list(PART_NAMES),part_joint_indices=[list(group) for group in PART_JOINTS],
        primary_motion=f'event3_part_path_px: native event-camera mean joint projected path length over{args.window_ms}ms',
        signed_vector='event3_part_displacement_uv_px: mean joint end-minus-start vector in native event3 (x,y) pixels',
        secondary_motion=['rgb3_part_path_px','camera3D_part_path_m','pelvis_relative3D_part_path_m'],
        endpoint_displacement_separately_stored=True,path_not_speed_or_uncertainty=True,
        GT_native_rate_approx_hz=30,interpolation='3D joint linear interpolation at start within available past GT; latest past GT held to actual RGB3 end',
        future_GT_used=False,future_event_used=False,inference_input_allowed=False,
        training_supervision_allowed=role=='training',validation_only=index_role=='validation',
        approximate_diagnostic_not_motion_ground_truth_sensor_rate=True,
        independence='Do not treat queries x6parts as independent samples; preserve subject/session clusters',
        valid_diagnostic_queries=int(arrays['diagnostic_valid'].sum()),
        end_GT_hold_age_ms_quantiles=np.quantile(arrays['latest_GT_hold_age_ms'],[0,.5,.9,1]).tolist(),
        RGB_to_common_GT_lag_ms_quantiles=np.quantile(arrays['RGB_to_common_GT_lag_ms'],[0,.5,.9,1]).tolist(),
        event3_part_path_px_median=np.median(arrays['event3_part_path_px'],axis=0).tolist(),
        self_check=checks,source_provenance=provenance,elapsed_seconds=time.time()-started,
        completed_unix=time.time(),gpu_used=False,split_sha256=splitsha,
        preserved_original_holdout_sha256=protected,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(summary_path,summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',default=str(ROOT/'exp/261028/device3_encoding_12h'))
    parser.add_argument('--output-root',default=str(ROOT/'exp/261029/part_aware_dynamic_aggregation'))
    parser.add_argument('--window-ms',type=int,default=200)
    parser.add_argument('--role',choices=['holdout','training','validation'],default='holdout')
    parser.add_argument('--deadline-unix',type=float,default=1793321712)
    main(parser.parse_args())

"""Common-clock cache construction; dataset inputs are opened read-only."""
import json
import os
from pathlib import Path
import time
import hashlib
import shutil

import numpy as np
from .multisensor_data import ROOT,OUT


def stage_inputs(source,name):
    """Sequential original-file copies limit remote seek contention; copies are temporary."""
    import fcntl
    root=OUT/'input_stage'/name;root.mkdir(parents=True,exist_ok=True)
    lock_root=ROOT/'.experiment_storage/device34_body_28h/copy_locks';lock_root.mkdir(parents=True,exist_ok=True)
    acquired=None
    while acquired is None:
        for i in range(2):
            handle=(lock_root/str(i)).open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=handle;break
            except BlockingIOError:handle.close()
        if acquired is None:time.sleep(.2)
    records=[]
    try:
        for device in (3,4):
            folder=root/('device'+str(device));folder.mkdir(exist_ok=True)
            for filename in ('rgb.mp4','event.npz'):
                src=source/('device'+str(device))/filename;dst=folder/filename
                started=time.time();digest=hashlib.sha256();temporary=dst.with_name(dst.name+'.partial')
                before=src.stat()
                with src.open('rb') as reader,temporary.open('wb') as writer:
                    for block in iter(lambda:reader.read(4*1024*1024),b''):digest.update(block);writer.write(block)
                after=src.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
                assert temporary.stat().st_size==before.st_size
                temporary.replace(dst)
                records.append(dict(source=str(src),bytes=before.st_size,mtime_ns=before.st_mtime_ns,
                    sha256=digest.hexdigest(),seconds=time.time()-started))
    finally:acquired.close()
    record_path=OUT/'input_stage_records'/(name+'.json');record_path.parent.mkdir(exist_ok=True)
    record_path.write_text(json.dumps(records,indent=2))
    return root


def raw_plan(item):
    import h5py
    name,role,seed=item
    source=ROOT/'dataset'/name
    with np.load(OUT/'source_maps'/(name+'.npz'),allow_pickle=False) as z:maps={k:z[k] for k in z.files}
    valid=maps['gt_valid'].astype(bool)
    if role=='test':
        split=json.loads((OUT/'split_snapshot.json').read_text())
        frames=np.array(sorted({i for a,b in split['test'][name] for i in range(a,b)}),np.int64)
    else:
        eligible=valid.any(1)&maps['all_sensors_available'][maps['gt_source_positions']]
        candidates=maps['gt_source_positions'][eligible]
        assert len(candidates),name
        maximum=32 if role=='validation' else 128
        if role=='validation':
            frames=np.sort(np.random.default_rng(seed).choice(candidates,min(maximum,len(candidates)),replace=False))
        else:frames=candidates[np.linspace(0,len(candidates)-1,min(maximum,len(candidates)),dtype=int)]
    assert len(frames) and frames.max()<len(maps['canonical_rgb_timestamp_ns'])
    clock=maps['canonical_rgb_timestamp_ns'][frames]
    rows=np.searchsorted(maps['gt_source_positions'],frames);safe=np.minimum(rows,len(valid)-1)
    mapped=(rows<len(valid))&(maps['gt_source_positions'][safe]==frames)
    raw=OUT/'raw'/(name+'.npz');raw.parent.mkdir(exist_ok=True)
    values=dict(valid=valid[safe]&mapped[:,None],
        positions=frames,K=maps['device3_rgb_K'].astype(np.float32),R=maps['device3_rgb_R'].astype(np.float32),
        t=maps['device3_rgb_t'].astype(np.float32),image_shape=maps['device3_rgb_shape'],
        fps=1e9/np.median(np.diff(maps['canonical_rgb_timestamp_ns'])))
    reusable=False
    if raw.exists():
        try:
            with np.load(raw) as z:
                reusable=all(np.array_equal(z[k],v,equal_nan=True) for k,v in values.items())
                reusable&=z['poses'].shape==(*values['valid'].shape,165) and z['betas'].shape==(*values['valid'].shape,16)
        except Exception:pass
    if not reusable:
        with h5py.File(source/'device0/label.h5','r',locking=False) as f:
            assert np.array_equal(f['params/person_valid'][:].astype(bool),valid)
            values.update(poses=f['params/smplx_pose_axis_angle'][:][safe].astype(np.float32),
                betas=f['params/smplx_betas'][:][safe].astype(np.float32),
                transl=f['params/translation_world_m'][:][safe].astype(np.float32))
        temporary=raw.with_name(raw.stem+'.'+str(os.getpid())+'.tmp.npz')
        np.savez_compressed(temporary,**values);temporary.replace(raw)
    result=dict(session=name,role=role,raw=str(raw),frames=len(frames),total_gt_rows=len(valid),
        total_valid_frames=int(valid.any(1).sum()))
    return source,maps,frames,clock,result


def prepare_session(item):
    import cv2
    import h5py
    from PIL import Image
    from utils.event_encoding import EventFrames,fit_geometry
    cv2.setNumThreads(1)
    started=time.time();name,role,seed=item
    source,maps,frames,clock,result=raw_plan(item)
    manifest=json.loads((OUT/'manifest.json').read_text())
    destination=OUT/'cache'/(name+'.h5');destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        with h5py.File(destination,'r',locking=False) as f:
            assert f.attrs['complete'] and f.attrs['split_sha256']==manifest['split_sha256']
            assert f.attrs['event_encoding']=='count33'
            assert np.array_equal(f['frames'][:],frames) and np.array_equal(f['timestamp_ns'][:],clock)
            elapsed=float(f.attrs['total_seconds'])
        return dict(result,bytes=destination.stat().st_size,seconds=elapsed,reused_verified=True)
    original_source=source
    if (OUT/'input_stage').exists():source=stage_inputs(source,name)
    partial=destination.with_suffix('.partial.h5')
    Rc=maps['device3_rgb_R'];tc=maps['device3_rgb_t'];native=maps['device3_rgb_shape']
    with h5py.File(partial,'w',locking=False) as f:
        f.attrs.update(complete=False,session=name,role=role,split_sha256=manifest['split_sha256'],
            event_encoding='count33',common_camera='device3 RGB',source=str(original_source),started_unix=started)
        f.create_dataset('frames',data=frames);f.create_dataset('timestamp_ns',data=clock)
        for device in (3,4):
            prefix='device'+str(device)+'_';sensor='rgb'+str(device)
            shape=maps[prefix+'rgb_shape'];geometry,pixel=fit_geometry(shape,896)
            nh,nw,top,left=geometry
            R=maps[prefix+'rgb_R']@Rc.T;t=maps[prefix+'rgb_t']-R@tc
            K=maps[prefix+'rgb_K'];scale=np.array([[.25,0,-.375],[0,.25,-.375],[0,0,1.]])
            for key,value in dict(K_native=K,K_primary=pixel@K,K_auxiliary=scale@pixel@K,
                native_shape=shape,R_common_to_sensor=R,t_common_to_sensor=t).items():f.create_dataset(sensor+'_'+key,data=value)
            available=maps[prefix+'rgb_available'][frames]
            f.create_dataset(sensor+'_available',data=available)
            ds=f.create_dataset(sensor,shape=(len(frames),3,896,896),dtype=np.uint8,
                chunks=(1,3,896,896),compression='lzf',fillvalue=0)
            selected=maps[prefix+'rgb_positions'][frames]
            requested={int(pos):np.flatnonzero((selected==pos)&available).tolist() for pos in np.unique(selected[available])}
            assert requested,name
            video=str(source/('device'+str(device))/'rgb.mp4')
            cap=(cv2.VideoCapture(video,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
                 if hasattr(cv2,'CAP_PROP_N_THREADS') else cv2.VideoCapture(video))
            assert cap.isOpened()
            for pos in range(max(requested)+1):
                assert cap.grab(),(name,device,pos)
                if pos not in requested:continue
                ok,bgr=cap.retrieve();assert ok
                image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((nw,nh),Image.Resampling.LANCZOS)
                canvas=Image.new('RGB',(896,896));canvas.paste(image,(left,top))
                value=np.asarray(canvas).transpose(2,0,1)
                for index in requested[pos]:ds[index]=value
            cap.release()
            sensor='event'+str(device)
            events=EventFrames.load(source/('device'+str(device))/'event.npz',size=224)
            R=maps[prefix+'event_R']@Rc.T;t=maps[prefix+'event_t']-R@tc
            K=maps[prefix+'event_K'];Ke=events.pixel_transform@K
            upsample=np.array([[4,0,1.5],[0,4,1.5],[0,0,1.]])
            for key,value in dict(K_native=K,K_primary=upsample@Ke,K_auxiliary=Ke,
                native_shape=events.native_shape,R_common_to_sensor=R,t_common_to_sensor=t).items():f.create_dataset(sensor+'_'+key,data=value)
            ds=f.create_dataset(sensor,shape=(len(frames),2,224,224),dtype=np.uint8,
                chunks=(1,2,224,224),compression='lzf')
            available=f.create_dataset(sensor+'_available',shape=(len(frames),),dtype=np.bool_)
            latest=f.create_dataset(sensor+'_latest_timestamp_ns',shape=(len(frames),),dtype=np.int64)
            for i,stamp in enumerate(clock):
                encoded,meta=events.encode(int(stamp),encodings=['count33'])
                ds[i]=np.rint(encoded['count33']*255).astype(np.uint8)
                available[i]=meta['event_available'];latest[i]=meta['latest_event_timestamp_ns']
                assert not meta['event_available'] or meta['latest_event_timestamp_ns']<=stamp
            assert np.array_equal(available[:],maps[prefix+'event_available'][frames])
            del events
        f.attrs.update(complete=True,completed_unix=time.time(),total_seconds=time.time()-started)
        f.flush()
    partial.replace(destination)
    if source!=original_source:shutil.rmtree(source)
    return dict(result,bytes=destination.stat().st_size,seconds=time.time()-started,reused_verified=False)


def prepare_all():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    import torch
    import smplx
    import multihmr.data as data
    from utils.pose_evaluation import atomic_npz
    from tools.prepare_device34_body import save
    torch.set_num_threads(2)
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['split_sha256']==__import__('hashlib').sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()
    for folder in ('targets','raw','test/gt','cache'): (OUT/folder).mkdir(parents=True,exist_ok=True)
    model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',
        num_betas=16,use_pca=False,flat_hand_mean=False).cuda().eval()
    np.save(OUT/'joint_regressor.npy',model.J_regressor.cpu().numpy())
    data.WORK=OUT
    workers=int(os.environ.get('EVENTPOSE_PREP_WORKERS','12'))
    old_identity=json.loads((OUT/'preparation_gpu_identity.json').read_text()) if (OUT/'preparation_gpu_identity.json').exists() else {}
    save(OUT/'preparation_gpu_identity.json',dict(pid=os.getpid(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        physical_gpu=os.environ.get('EVENTPOSE_PHYSICAL_GPU'),cuda_name=torch.cuda.get_device_name(),
        started_unix=old_identity.get('started_unix',time.time()),resumed_unix=time.time(),cpu_workers=workers))
    items=[(name,role,261026+i) for role,key in [('validation','validation_sessions'),('training','training_sessions'),('test','test_sessions')]
        for i,name in enumerate(manifest[key])]
    results=[];indices={'training':[],'validation':[]};prepared=set();tasks=[]
    split=json.loads((OUT/'split_snapshot.json').read_text())
    records_by_session={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[]
        # Reconstruct actual GT on GPU immediately, while CPU workers prepare
        # the same frozen RGB/event frame plans in parallel.
        for item in items:
            source,maps,frames,clock,result=raw_plan(item)
            name=result['session'];role=result['role']
            target_path=OUT/'targets'/(name+'.npz');reused_target=False
            if target_path.exists():
                try:
                    with np.load(target_path) as z:
                        reused_target=np.array_equal(z['positions'],frames) and np.array_equal(z['K'],maps['device3_rgb_K'].astype(np.float32))
                        reused_target&=np.isfinite(z['joints'][z['valid']]).all() and np.isfinite(z['rotations'][z['valid']]).all()
                        if reused_target:records=[dict(session=name,row=i,frame=int(frame)) for i,frame in enumerate(frames) if role!='training' or z['valid'][i].any()]
                except Exception:reused_target=False
            if not reused_target:records=data.reconstruct(model,name,'validation',result['raw'])
            if role=='training' and not reused_target:
                import h5py
                with np.load(OUT/'targets'/(name+'.npz')) as z:targets={k:z[k] for k in z.files}
                visible=targets['valid'].copy()
                Rc=maps['device3_rgb_R'];tc=maps['device3_rgb_t']
                for sensor in ('rgb3','rgb4','event3','event4'):
                    kind='event' if sensor.startswith('event') else 'rgb';prefix='device'+sensor[-1]+'_'
                    R=maps[prefix+kind+'_R']@Rc.T;t=maps[prefix+kind+'_t']-R@tc
                    centers=targets['centers']@R.T+t;hom=centers@maps[prefix+kind+'_K'].T
                    uv=hom[...,:2]/np.maximum(hom[...,2:],1e-8)
                    if kind=='rgb':h,w=maps[prefix+'rgb_shape']
                    else:
                        with np.load(source/('device'+sensor[-1])/'event.npz') as z:h,w=z['image_shape']
                    visible&=(centers[...,2]>.05)&(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)
                targets['valid']=visible
                np.savez_compressed(OUT/'targets'/(name+'.npz'),**targets)
                records=[r for r in records if visible[r['row']].any()]
                result['excluded_not_common_visible_frames']=int((~visible.any(1)).sum())
                assert records,(name,'No common-visible training frame')
            records_by_session[name]=records
            futures.append(pool.submit(prepare_session,item))
            save(OUT/'gt_precompute_progress.json',dict(precomputed_sessions=len(records_by_session),
                total_sessions=len(items),last_session=name,updated_unix=time.time()))
            print(json.dumps(dict(event='GPU_GT_precomputed',session=name,frames=len(frames))),flush=True)
        for future in as_completed(futures):
            result=future.result();name=result['session'];role=result['role'];records=records_by_session[name]
            if role in indices:
                indices[role].extend(dict(session=name,cache_row=r['row'],frame=r['frame']) for r in records)
            else:
                with np.load(OUT/'targets'/(name+'.npz')) as z:
                    joints=z['joints'];valid=z['valid'];positions=z['positions'];K=z['K'];shape=z['image_shape']
                with np.load(OUT/'source_maps'/(name+'.npz')) as z:clock=z['canonical_rgb_timestamp_ns']
                for i,(start,end) in enumerate(split['test'][name]):
                    key=name+'__segment'+str(i);keep=(positions>=start)&(positions<end)
                    assert keep.sum()==end-start and positions[keep][0]==start
                    dest=OUT/'test/gt'/(key+'.npz')
                    atomic_npz(dest,joints_cam_m=joints[keep],valid=valid[keep],source_rgb_positions=positions[keep]-start,
                        original_source_rgb_positions=positions[keep],timestamps_ns=clock[positions[keep]],K=K,image_shape=shape)
                    tasks.append(dict(session=key,parent_session=name,segment=i,start_frame=start,end_frame=end,
                        video_frames=end-start,gt_file=str(dest),gt_frames=int(keep.sum()),valid_gt_instances=int(valid[keep].sum()),
                        fps=float(1e9/np.median(np.diff(clock))),width=int(shape[1]),height=int(shape[0])))
            results.append(result);prepared.add(name)
            if set(manifest['training_sessions']+manifest['validation_sessions'])<=prepared and not (OUT/'training_cache_complete.json').exists():
                for r,records in indices.items():save(OUT/(r+'_index.json'),sorted(records,key=lambda v:(v['session'],v['frame'])))
                save(OUT/'training_cache_complete.json',dict(completed_unix=time.time(),
                    training_frames=len(indices['training']),validation_frames=len(indices['validation']),
                    split_sha256=manifest['split_sha256']))
            save(OUT/'cache_progress.json',dict(sessions=len(results),total_sessions=len(items),last_session=name,updated_unix=time.time()))
            print(json.dumps(result),flush=True)
    tasks.sort(key=lambda t:t['session'])
    save(OUT/'test/tasks.json',tasks)
    save(OUT/'cache_complete.json',dict(completed_unix=time.time(),results=results,
        test_segments=len(tasks),test_frames=sum(t['video_frames'] for t in tasks),
        bytes=sum(r['bytes'] for r in results),summed_session_seconds=sum(r['seconds'] for r in results)))

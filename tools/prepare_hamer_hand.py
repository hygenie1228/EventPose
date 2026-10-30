"""Read-only hand source audit and GT-crop preparation for the GPU0 study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261028/hamer_hand'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def audit():
    import numpy as np
    splitpath=ROOT/'dataset/split.json';raw=splitpath.read_bytes();split=json.loads(raw)
    hand={r:sorted(n for n in split[r] if n.endswith('_h')) for r in ('train','test')}
    subjects=lambda names:{int(n.split('_')[3]) for n in names}
    assert hand['train'] and hand['test'] and not subjects(hand['train'])&subjects(hand['test'])
    body=json.loads((ROOT/'exp/261028/device3_encoding_12h/manifest.json').read_text())
    holdout=set(body['validation_subjects'])&subjects(hand['train'])
    assert holdout and subjects(hand['train'])-holdout
    missing=[str(ROOT/'dataset'/n/('device'+str(d))/f) for names in hand.values() for n in names
             for d,files in [(0,['label.h5','event.npz']),(3,['label.h5','event.npz','rgb.mp4']),(4,['event.npz'])]
             for f in files if not (ROOT/'dataset'/n/('device'+str(d))/f).is_file()]
    assert not missing,missing
    snapshot=OUT/'split_snapshot.json'
    if snapshot.exists():assert snapshot.read_bytes()==raw,'Split changed; use a new experiment'
    else:snapshot.write_bytes(raw)
    manifest=dict(seed=261028,split_sha256=hashlib.sha256(raw).hexdigest(),
        training_sessions=[n for n in hand['train'] if int(n.split('_')[3]) not in holdout],
        validation_sessions=[n for n in hand['train'] if int(n.split('_')[3]) in holdout],test_sessions=hand['test'],
        train_subjects=sorted(subjects(hand['train'])-holdout),validation_subjects=sorted(holdout),
        test_subjects=sorted(subjects(hand['test'])),
        body_reference='exp/261028/device3_encoding_12h/manifest.json',
        crop_protocol='Oracle GT hand boxes from device3 label.h5; anatomical left/right; HaMeR rescale2.5',
        gt='Neutral SMPL-X,16 betas,flat_hand_mean=False; 21 anatomical joints including fingertips',
        common_camera='device3 RGB; wrist index0; coordinates in metres',
        clock='device0 corrected RGB timestamp_ns; latest device3 RGB <= clock with maximum age50ms',
        events='device4 stored sparse observations, causal body encoders at224; no recovered raw events',
        selection='Train subject holdout only; no Test checkpoint selection',allowed_gpus=[0],
        training_sampling='128 uniformly spaced valid RGB-available person frames per session',
        validation_sampling='32 seeded valid RGB-available person frames per session',
        test_sampling='Every original frame in frozen Test intervals; no temporal subsampling',
        root_metrics=False,metrics=['first_frame_mpjpe','first_frame_pa_mpjpe','frame_mpjpe','frame_pa_mpjpe'])
    old=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else None
    assert old is None or old==manifest
    write(OUT/'manifest.json',manifest)
    write(OUT/'scope_audit.json',dict(hand_counts={r:len(v) for r,v in hand.items()},
        body_counts={r:sum(n.endswith('_b') for n in split[r]) for r in hand},missing=missing,
        train_test_subject_overlap=[],validation_source='same subject IDs as body',source_split=str(splitpath.resolve()),
        split_sha256=manifest['split_sha256'],checked_unix=time.time()))
    configs=[]
    for name,condition,encoding,hours in [('rgb3','rgb3','count33',1.5)]+[
            ('rgb3_event4_'+e,'rgb3_event4',e,3.) for e in ('count33','bins100','surface20')]:
        config=dict(name=name,condition=condition,encoding=encoding,hours=hours,seed=261028,
            output_root=str(OUT),split_sha256=manifest['split_sha256'],allowed_gpus=[0],last_blocks=8,
            batch_size=16,workers=2,head_lr=1e-5,backbone_lr=1e-6,fusion_lr=.001,
            validation_minutes=30,backbone_warmup_seconds=1800,warmup_steps=100,
            residual_regularization=.01,augment=True,crop_protocol='oracle_gt',root_metrics=False)
        write(OUT/'configs'/(name+'.json'),config);configs.append(config)
    write(OUT/'experiment_manifest.json',dict(configs=configs,pretrained_baseline=True,
        test_used_for_selection=False,persistent_goal_created=False))
    print(json.dumps(dict(training=len(manifest['training_sessions']),validation=len(manifest['validation_sessions']),
        test=len(manifest['test_sessions']),scope='4 trials, GPU0, no RPE')),flush=True)


_body=None
def prepare_session(item):
    global _body
    import cv2
    import h5py
    import numpy as np
    import torch
    import smplx
    from hamer.hand_geometry import joint_indices,crop_transform,apply_crop,rectify_boxes
    from utils.event_encoding import EventFrames,calibration
    torch.set_num_threads(2);cv2.setNumThreads(1)
    name,role,seed=item;started=time.time();source=ROOT/'dataset'/name
    sources=[source/('device'+str(d))/file for d,files in
        [(0,['label.h5','event.npz']),(3,['label.h5','event.npz','rgb.mp4']),(4,['event.npz'])] for file in files]
    source_stats={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sources}
    manifest=json.loads((OUT/'manifest.json').read_text());dest=OUT/'cache'/(name+'.h5')
    if dest.exists():
        with h5py.File(dest,'r',locking=False) as f:
            assert f.attrs['complete'] and f.attrs['split_sha256']==manifest['split_sha256']
            assert f.attrs['cache_version']==2,'Old source-pixel boxes were not rectified'
            return dict(session=name,role=role,samples=len(f['frame']),frames=len(f['frames']),
                bytes=dest.stat().st_size,reused=True)
    with np.load(source/'device0/event.npz',allow_pickle=False) as z:clock=z['rgb__timestamp_ns'].astype(np.int64)
    with np.load(source/'device3/event.npz',allow_pickle=False) as z:
        rgbclock=z['rgb__timestamp_ns'].astype(np.int64);K=calibration(z)['K_rgb'].astype(np.float32)
        R=z['rgb_calibration__rgb__rotation_matrix'].astype(np.float32)
        t=z['rgb_calibration__rgb__translation'].astype(np.float32).reshape(3)
        source_K=z['rgb_calibration__rgb__source_intrinsic_matrix'].astype(np.float64)
        source_distortion=z['rgb_calibration__rgb__source_distortion_coeffs'].astype(np.float64)
        pixel_transform=z['rgb_calibration__rgb__resize_pixel_transform'].astype(np.float64)
        assert np.allclose(pixel_transform@z['rgb_calibration__rgb__pre_resize_intrinsic_matrix'],K,atol=1e-3)
    with h5py.File(source/'device0/label.h5','r',locking=False) as f:
        assert f.attrs['complete'] and f.attrs['coordinate_unit']=='metre'
        positions=f['frames/source_rgb_position'][:].astype(np.int64)
        valid=f['params/person_valid'][:].astype(bool)
        poses=f['params/smplx_pose_axis_angle'][:];betas=f['params/smplx_betas'][:]
        transl=f['params/translation_world_m'][:]
        assert np.array_equal(f['frames/source_rgb_timestamp_ns'][:],clock[positions])
    rgbpos=np.searchsorted(rgbclock,clock,side='right')-1
    rgbavailable=(rgbpos>=0)&(clock-rgbclock[np.maximum(rgbpos,0)]<=50_000_000)
    if role=='test':
        intervals=json.loads((OUT/'split_snapshot.json').read_text())['test'][name]
        frames=np.array(sorted({i for a,b in intervals for i in range(a,b)}),np.int64)
    else:
        candidates=positions[valid.any(1)&rgbavailable[positions]]
        assert len(candidates),('No valid training frames',name)
        count=min(32 if role=='validation' else 128,len(candidates))
        frames=(np.sort(np.random.default_rng(seed).choice(candidates,count,replace=False)) if role=='validation'
                else candidates[np.linspace(0,len(candidates)-1,count,dtype=int)])
    rows=np.searchsorted(positions,frames);safe=np.minimum(rows,len(positions)-1)
    mapped=(rows<len(positions))&(positions[safe]==frames)
    valid=valid[safe]&mapped[:,None]&rgbavailable[frames,None]
    with h5py.File(source/'device3/label.h5','r',locking=False) as f:
        labels=f['frames/source_rgb_position'][:].astype(np.int64)
        assert np.array_equal(f['frames/source_rgb_timestamp_ns'][:],rgbclock[labels]),('RGB3 label clock mismatch',name)
        ids=np.searchsorted(labels,rgbpos[frames]);sid=np.minimum(ids,len(labels)-1)
        parts=list(f['bboxes'].attrs['part_names']);assert parts[2:4]==['left_hand','right_hand']
        # h5py fancy indexing requires unique monotonically increasing rows.
        unique,inverse=np.unique(sid,return_inverse=True)
        boxes=f['bboxes/xyxy'][unique][:,:,2:4][inverse].astype(np.float32)
        boxvalid=f['bboxes/valid'][unique][:,:,2:4][inverse]
        visible=f['bboxes/visible'][unique][:,:,2:4][inverse]
        boxvalid&=((ids<len(labels))&(labels[sid]==rgbpos[frames]))[:,None,None]
        label_image_shape=np.array([f['bboxes'].attrs['height'],f['bboxes'].attrs['width']],np.int64)
    source_boxes=boxes.copy()
    boxes=rectify_boxes(boxes,source_K,source_distortion,K)
    cap=cv2.VideoCapture(str(source/'device3/rgb.mp4'));assert cap.isOpened()
    image_shape=np.array([round(cap.get(4)),round(cap.get(3))],np.int64);cap.release()
    expected_shape=label_image_shape*np.array([pixel_transform[1,1],pixel_transform[0,0]])
    assert np.allclose(image_shape,expected_shape,atol=1),(name,image_shape,expected_shape)
    eligible=valid[:,:,None]&boxvalid&visible
    eligible&=(boxes[...,2:]-boxes[...,:2]).min(-1)>1
    ids=np.argwhere(eligible);assert len(ids),('No GT-visible hand samples',name)
    # Reconstruct GT independently from the HaMeR MANO layer; do not copy
    # SMPL-X betas/axis angles into MANO's incompatible shape/pose basis.
    if _body is None:
        _body=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',
            num_betas=16,use_pca=False,flat_hand_mean=False).eval()
    pairs=np.unique(ids[:,:2],axis=0);hands={};jidx=[joint_indices('left'),joint_indices('right')]
    with torch.inference_mode():
        for offset in range(0,len(pairs),16):
            selected=pairs[offset:offset+16];fi,pi=selected.T;ri=safe[fi]
            p=torch.from_numpy(poses[ri,pi]);b=torch.from_numpy(betas[ri,pi]);tr=torch.from_numpy(transl[ri,pi])
            out=_body(betas=b,global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],
                leye_pose=p[:,69:72],reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],
                right_hand_pose=p[:,120:165],transl=tr,expression=torch.zeros(len(p),10))
            js=out.joints.numpy()@R.T+t
            for i,key in enumerate(map(tuple,selected)):hands[key]=js[i,jidx]
    joints=np.stack([hands[(fi,pi)][side] for fi,pi,side in ids]).astype(np.float32)
    assert np.isfinite(joints).all()
    N=len(ids);F=len(frames);dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_suffix('.partial.h5')
    with h5py.File(partial,'w',locking=False) as f:
        f.attrs.update(complete=False,cache_version=2,session=name,role=role,split_sha256=manifest['split_sha256'],
            crop_protocol='oracle_gt_device3_hand_boxes',common_camera='device3 RGB',gt_joint_count=21)
        data=dict(frames=frames,frame=frames[ids[:,0]],frame_row=ids[:,0],person=ids[:,1],right=ids[:,2],
            timestamp_ns=clock[frames],joints=joints,native_K=K,image_shape=image_shape,
            rgb3_positions=rgbpos[frames],gt_boxes=boxes[tuple(ids.T)],gt_boxes_source=source_boxes[tuple(ids.T)],
            source_K=source_K,source_distortion=source_distortion,label_image_shape=label_image_shape,
            excluded_hand_slots=int(eligible.size-len(ids)))
        for key,value in data.items():f.create_dataset(key,data=value)
        transforms=[];sizes=[]
        for box,(_,_,side) in zip(data['gt_boxes'],ids):
            A,size=crop_transform(box,bool(side));transforms.append(A);sizes.append(size)
        transforms=np.asarray(transforms,np.float32)
        f.create_dataset('crop_transform',data=transforms);f.create_dataset('K',data=transforms@K)
        f.create_dataset('box_center',data=(data['gt_boxes'][:,:2]+data['gt_boxes'][:,2:])/2)
        f.create_dataset('box_size',data=np.array(sizes,np.float32))
        ds=f.create_dataset('rgb3',shape=(N,3,256,256),dtype=np.uint8,chunks=(1,3,256,256),compression='lzf')
        # Decode only selected RGB frames, sequentially; never stage or write source data.
        requests={int(p):[] for p in rgbpos[frames[ids[:,0]]]}
        for i,(fi,_,_) in enumerate(ids):requests[int(rgbpos[frames[fi]])].append(i)
        cap=(cv2.VideoCapture(str(source/'device3/rgb.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            if hasattr(cv2,'CAP_PROP_N_THREADS') else cv2.VideoCapture(str(source/'device3/rgb.mp4')))
        assert cap.isOpened()
        for pos in range(max(requests)+1):
            assert cap.grab(),(name,pos)
            if pos not in requests:continue
            ok,bgr=cap.retrieve();assert ok
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            for i in requests[pos]:
                edge=sizes[i];factor=edge/256/2;image=rgb
                if factor>1.1:
                    # Official demo uses skimage Gaussian antialias before affine.
                    from skimage.filters import gaussian
                    image=gaussian(rgb,sigma=(factor-1)/2,channel_axis=2,preserve_range=True)
                ds[i]=np.rint(apply_crop(image.transpose(2,0,1),transforms[i])).clip(0,255).astype(np.uint8)
        # Reject a source/crop coordinate mismatch before writing any event cache.
        for i in np.linspace(0,N-1,min(N,8),dtype=int):
            assert ds[i].mean()>1 and ds[i].std()>1,('Empty GT hand crop',name,int(i))
        cap.release()
        events=EventFrames.load(source/'device4/event.npz',size=224)
        with np.load(source/'device4/event.npz',allow_pickle=False) as z:
            Re=z['rgb_calibration__event__rotation_matrix'].astype(np.float32)
            te=z['rgb_calibration__event__translation'].astype(np.float32).reshape(3)
        relative=Re@R.T
        f.create_dataset('rotation',data=relative);f.create_dataset('translation',data=te-relative@t)
        f.create_dataset('auxiliary_K',data=(events.pixel_transform@events.cameras['K_event']).astype(np.float32))
        from utils.event_encoding import ENCODINGS
        eds={e:f.create_dataset('event4_'+e,shape=(F,ENCODINGS[e]['channels'],224,224),dtype=np.uint8,
             chunks=(1,ENCODINGS[e]['channels'],224,224),compression='lzf') for e in ('count33','bins100','surface20')}
        available=f.create_dataset('auxiliary_available',shape=(F,),dtype=np.bool_)
        latest=f.create_dataset('latest_event_timestamp_ns',shape=(F,),dtype=np.int64)
        for i,stamp in enumerate(clock[frames]):
            values,meta=events.encode(int(stamp),encodings=list(eds))
            for e,ds in eds.items():ds[i]=np.rint(values[e]*255).astype(np.uint8)
            available[i]=meta['event_available'];latest[i]=meta['latest_event_timestamp_ns']
            assert latest[i]<=stamp or not available[i]
        f.attrs.update(complete=True,completed_unix=time.time(),seconds=time.time()-started,
            event_available_frames=int(available[:].sum()),gt_hand_slots=eligible.size,samples=N)
        for path,stat in source_stats.items():
            actual=Path(path).stat();assert actual.st_size==stat['bytes'] and actual.st_mtime_ns==stat['mtime_ns']
        f.attrs['source_stats_json']=json.dumps(source_stats)
        f.flush()
    partial.replace(dest)
    result=dict(session=name,role=role,samples=N,frames=F,bytes=dest.stat().st_size,seconds=time.time()-started)
    write(OUT/'cache_records'/(name+'.json'),result)
    return result


def cache(workers=4):
    import h5py
    m=json.loads((OUT/'manifest.json').read_text());results=[];indices={r:[] for r in ('training','validation')}
    items=[(n,role,261026+i) for role in ('validation','training','test')
           for i,n in enumerate(m[role+'_sessions'])]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(prepare_session,item):item for item in items}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            write(OUT/'preparation_state.json',dict(completed=len(results),total=len(items),last=result,updated_unix=time.time()))
            print(json.dumps(result),flush=True)
            for role in ('training','validation'):
                if indices[role] or not set(m[role+'_sessions'])<={r['session'] for r in results}:continue
                for name in m[role+'_sessions']:
                    with h5py.File(OUT/'cache'/(name+'.h5'),'r',locking=False) as f:
                        indices[role]+=[dict(session=name,row=i,frame=int(frame)) for i,frame in enumerate(f['frame'][:])]
                write(OUT/(role+'_index.json'),indices[role])
    write(OUT/'cache_complete.json',dict(results=results,completed_unix=time.time(),
        bytes=sum(r['bytes'] for r in results),samples=sum(r['samples'] for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','cache','sample']);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.stage=='audit':audit()
    elif a.stage=='cache':
        try:cache(a.workers)
        except Exception as error:
            write(OUT/'preparation_failure.json',dict(error=repr(error),failed_unix=time.time()))
            raise
    else:
        m=json.loads((OUT/'manifest.json').read_text());print(prepare_session((m['validation_sessions'][0],'validation',261026)))

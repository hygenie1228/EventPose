from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Read-only ETRI Train preparation with participant-disjoint validation and SSD cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('EVENTPOSE_PREPARATION_OUT', ROOT/'exp/261023/multihmr_finetune_14h'))
WORK=Path(os.environ.get('EVENTPOSE_RGB_CACHE', ROOT/'.experiment_storage/multihmr_finetune_14h'))
ASSETS=OUT/'checkpoints'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def plan():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    for folder in ['images','targets','raw','logs','tmp','validation']:(WORK/folder).mkdir(exist_ok=True)
    if not (OUT/'work').exists():(OUT/'work').symlink_to(WORK,target_is_directory=True)
    split_raw=(ROOT/'dataset/split.json').read_bytes();split=json.loads(split_raw)
    metadata=json.loads((ROOT/'exp/261023/subject_split/split_proposal.json').read_text())['sessions_metadata']
    rows=[r for r in metadata if r['session'] in split['train']]
    assert len(rows)==len(split['train'])==219
    parent={}
    def find(i):
        parent.setdefault(i,i)
        if parent[i]!=i:parent[i]=find(parent[i])
        return parent[i]
    for row in rows:
        for i in row['participant_ids']:parent[find(i)]=find(row['primary_id'])
    groups={}
    for row in rows:groups.setdefault(find(row['primary_id']),[]).append(row)
    groups=list(groups.values());rng=np.random.default_rng(261023)
    rng.shuffle(groups);validation=[];ids=set()
    for group in groups:
        if len(ids)>=10:break
        validation.extend(r['session'] for r in group)
        ids.update(i for r in group for i in r['participant_ids'])
    training=[s for s in split['train'] if s not in validation]
    train_ids={i for r in rows if r['session'] in training for i in r['participant_ids']}
    test_ids={i for r in metadata if r['session'] in split['test'] for i in r['participant_ids']}
    assert not train_ids&ids and not (train_ids|ids)&test_ids
    assert set(training)|set(validation)==set(split['train'])
    manifest=dict(seed=261023,model='Multi-HMR 896 L pretrained',device='device0',
        training_sessions=training,validation_sessions=sorted(validation),
        training_subject_ids=sorted(train_ids),validation_subject_ids=sorted(ids),test_subject_ids=sorted(test_ids),
        test_sessions=list(split['test']),test_split_sha256=hashlib.sha256(split_raw).hexdigest(),
        training_sampling='approximately 2 fps across entire Train videos',validation_sampling='64 fixed random GT frames per held-out session',
        selection_metric='body22 per-person per-frame root-aligned PA-MPJPE, mm; actual detection with coverage reported',
        test_policy='frozen Test used only for final evaluation, never for training or checkpoint selection',
        gpu_pairs=[[0,1],[2,3],[4,5]],goal_start_unix=1792773208,deadline_unix=1792773208+14*3600,
        checkpoint_root=str(ASSETS),work_root=str(WORK))
    save(OUT/'manifest.json',manifest)
    (OUT/'split_snapshot.json').write_bytes(split_raw)
    print(json.dumps({k:len(manifest[k]) for k in ['training_sessions','validation_sessions','training_subject_ids','validation_subject_ids']},indent=2))
    return manifest


def extract(item):
    import cv2
    import h5py
    from PIL import Image
    cv2.setNumThreads(1)
    name,role,seed=item;source=ROOT/'dataset'/name/'device0'
    done=WORK/'raw'/f'{name}.npz'
    if done.exists():return name,role,str(done)
    assert all((source/f).exists() for f in ['rgb.mp4','label.h5','event.npz'])
    cap=cv2.VideoCapture(str(source/'rgb.mp4'));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(cap.get(3)),int(cap.get(4))
    with h5py.File(source/'label.h5','r') as f:
        assert f.attrs['complete'] and f.attrs['coordinate_unit']=='metre'
        positions=f['frames/source_rgb_position'][:]
        valid=f['params/person_valid'][:].astype(bool)
        available=np.flatnonzero(valid.any(1))
        if role=='validation':
            rng=np.random.default_rng(seed);selected=np.sort(rng.choice(available,min(64,len(available)),replace=False))
        else:
            step=max(1,int(round(fps/2)))
            selected=available[::step]
        maximum=int(os.environ.get('EVENTPOSE_MAX_'+role.upper()+'_FRAMES', '0'))
        if maximum and len(selected)>maximum:
            selected=selected[np.linspace(0,len(selected)-1,maximum,dtype=int)]
        poses=f['params/smplx_pose_axis_angle'][:][selected].astype(np.float32)
        betas=f['params/smplx_betas'][:][selected].astype(np.float32)
        transl=f['params/translation_world_m'][:][selected].astype(np.float32)
        valid=valid[selected]
    wanted=positions[selected].astype(np.int64);assert len(wanted) and wanted[-1]<count
    image_dir=WORK/'images'/name;image_dir.mkdir(exist_ok=True)
    cursor=0
    # Sequential local decoder traversal; retrieve/resize/write only chosen frames.
    for pos in range(int(wanted[-1])+1):
        assert cap.grab(),(name,pos)
        if pos!=wanted[cursor]:continue
        ok,bgr=cap.retrieve();assert ok
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        # Preserve native pixels and aspect ratio; contain/pad is applied in loader.
        image=Image.fromarray(rgb)
        if os.environ.get('EVENTPOSE_LOSSLESS_RGB_CACHE')=='1':
            image.save(image_dir/f'{pos:06d}.png')
        else:
            image.save(image_dir/f'{pos:06d}.jpg',quality=95,subsampling=0)
        cursor+=1
        if cursor==len(wanted):break
    cap.release();assert cursor==len(wanted)
    with np.load(source/'event.npz') as z:
        K=z['rgb_calibration__rgb__intrinsic_matrix'].astype(np.float32)
        R=z['rgb_calibration__rgb__rotation_matrix'].astype(np.float32)
        t=z['rgb_calibration__rgb__translation'].astype(np.float32).reshape(3)
    temp=done.with_suffix('.tmp.npz')
    np.savez_compressed(temp,poses=poses,betas=betas,transl=transl,valid=valid,positions=wanted,
                        K=K,R=R,t=t,image_shape=[height,width],fps=fps)
    temp.replace(done)
    return name,role,str(done)


def reconstruct(model,name,role,path):
    import torch
    import roma
    with np.load(path) as z:raw={k:z[k] for k in z.files}
    valid=raw['valid']&np.isfinite(raw['poses']).all(-1)&np.isfinite(raw['betas']).all(-1)&np.isfinite(raw['transl']).all(-1)
    joints=np.full((*valid.shape,55,3),np.nan,np.float32)
    centers=np.full((*valid.shape,3),np.nan,np.float32)
    rotations=np.full((*valid.shape,53,3,3),np.nan,np.float32)
    ids=np.argwhere(valid)
    R=torch.from_numpy(raw['R']).cuda();t=torch.from_numpy(raw['t']).cuda()
    for begin in range(0,len(ids),32):
        fi,pi=ids[begin:begin+32].T
        p=torch.from_numpy(raw['poses'][fi,pi]).cuda();b=torch.from_numpy(raw['betas'][fi,pi]).cuda()
        tr=torch.from_numpy(raw['transl'][fi,pi]).cuda()
        with torch.no_grad():
            output=model(betas=b,global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],
                leye_pose=p[:,69:72],reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],right_hand_pose=p[:,120:165],
                transl=tr,expression=torch.zeros((len(p),10),device=p.device))
            js=(model.J_regressor@output.vertices)@R.T+t
            head=output.joints[:,15]@R.T+t
            absolute_pose=p+model.pose_mean
            rotvec=torch.cat([absolute_pose[:,:66].reshape(-1,22,3),absolute_pose[:,75:165].reshape(-1,30,3),absolute_pose[:,66:69,None].transpose(1,2)],1)
            rot=roma.rotvec_to_rotmat(rotvec);rot[:,0]=R@rot[:,0]
        joints[fi,pi]=js.cpu().numpy();centers[fi,pi]=head.cpu().numpy();rotations[fi,pi]=rot.cpu().numpy()
    dest=WORK/'targets'/f'{name}.npz'
    np.savez_compressed(dest,joints=joints,centers=centers,rotations=rotations,betas=raw['betas'],
                        valid=valid,positions=raw['positions'],K=raw['K'],image_shape=raw['image_shape'])
    assert np.isfinite(joints[valid]).all() and np.isfinite(rotations[valid]).all()
    h,w=raw['image_shape'];hom=centers@raw['K'].T;uv=hom[...,:2]/np.maximum(hom[...,2:],1e-8)
    visible=valid&(centers[...,2]>.05)&(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)
    keep=np.ones(len(valid),bool) if role=='validation' else visible.any(1)
    return [dict(session=name,row=int(i),frame=int(raw['positions'][i]),role=role) for i in np.flatnonzero(keep)]


def main(workers):
    manifest=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else plan()
    items=[(name,role,261023+i) for role,key in [('validation','validation_sessions'),('training','training_sessions')]
           for i,name in enumerate(manifest[key])]
    context=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        futures=[pool.submit(extract,item) for item in items]
        import torch
        import smplx
        torch.set_num_threads(2)
        model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',
                           num_betas=16,use_pca=False,flat_hand_mean=False).cuda().eval()
        np.save(WORK/'joint_regressor.npy',model.J_regressor.cpu().numpy())
        records=[];processed=0
        for future in as_completed(futures):
            name,role,path=future.result();records.extend(reconstruct(model,name,role,path));processed+=1
            save(OUT/'preparation_status.json',dict(processed_sessions=processed,total_sessions=len(items),
                 cached_frames=len(records),last_session=name,updated_unix=time.time()))
            print('Prepared',processed,len(items),name,role,len(records),flush=True)
    for role in ['training','validation']:
        selected=[r for r in records if r['role']==role];selected.sort(key=lambda r:(r['session'],r['frame']))
        save(WORK/f'{role}_index.json',selected)
    save(OUT/'preparation_complete.json',dict(training_frames=sum(r['role']=='training' for r in records),
         validation_frames=sum(r['role']=='validation' for r in records),sessions=len(items),completed_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6)
    p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    if args.plan_only:plan()
    else:main(args.workers)

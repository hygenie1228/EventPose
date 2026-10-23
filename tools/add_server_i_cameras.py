"""Add server-I RGB-only cameras 5/6, preserving primary frames and matching GT anchors."""
from pathlib import Path
import concurrent.futures
import json
import subprocess
import sys
import zipfile

import cv2
import h5py
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
NAME='20260720_125338_040324_200_f'
SOURCE=ROOT/'orig_dataset/data1'/NAME
DEST=ROOT/'etri_processed'/NAME
WORK=ROOT/'exp/261023/server_i_cameras'


def copy_video(i,device):
    with h5py.File(SOURCE/f'RGB/device{i}.h5','r') as f:
        g=f['streams/images'];off=g['offsets'][:];n=len(off)-1
        p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','image2pipe','-framerate','5','-vcodec','mjpeg','-i','-',
            '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(device/'video.mp4')],stdin=subprocess.PIPE)
        try:
            for j in range(0,n,128):
                end=min(j+128,n);p.stdin.write(g['encoded_data'][int(off[j]):int(off[end])].tobytes())
        finally:p.stdin.close();code=p.wait()
        assert code==0
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames,avg_frame_rate,width,height','-of','json',str(device/'video.mp4')]))
    assert probe['streams'][0]['nb_frames']==str(n) and probe['streams'][0]['avg_frame_rate']=='5/1'
    print(i,'RGB primary stream video verified',n,flush=True)
    return probe


def copy_events(i,device):
    with h5py.File(SOURCE/f'Event/device{i}.h5','r') as f:
        g=f['streams/images'];off=g['offsets'][:];n=len(off)-1;encoded=g['encoded_data'][:]
        indptr=np.zeros(n+1,np.int64);idxs=[];vals=[]
        h,w=int(g.attrs['height']),int(g.attrs['width'])
        for j in range(n):
            image=cv2.imdecode(encoded[int(off[j]):int(off[j+1])],cv2.IMREAD_GRAYSCALE).reshape(-1)
            idx=np.flatnonzero(image!=128).astype(np.int32);v=image[idx]
            assert np.isin(v,(0,255)).all()
            idxs.append(idx);vals.append(np.where(v>128,127,-128).astype(np.int8));indptr[j+1]=indptr[j]+len(idx)
            if j%5000==0:print(i,'Event decoded',j,'/',n,flush=True)
        indices=np.concatenate(idxs);data=np.concatenate(vals);meta={}
        for key in ['timestamp_ns','raw_timestamp_ns','applied_delta_ns','frame_index']:
            if key in g:meta[key]=g[key][:]
        meta['original_filename']=np.asarray(g['original_filename'].asstr()[:],dtype=str)
        meta['source_position']=np.arange(n,dtype=np.int64)
        meta['image_shape']=np.array([h,w]);meta['background_value']=np.array(128)
    with h5py.File(SOURCE/f'RGB/device{i}.h5','r') as f:
        for key in ['timestamp_ns','raw_timestamp_ns','applied_delta_ns','frame_index']:
            if key in f['streams/images']:meta['rgb__'+key]=f['streams/images/'+key][:]
        meta['rgb__source_position']=np.arange(len(f['streams/images/timestamp_ns']),dtype=np.int64)
        def collect(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.dtype.kind!='O':meta['rgb_calibration__'+name.replace('/','__')]=obj[()]
        f['calibration'].visititems(collect)
    arrays=dict(format=np.array(b'csr'),shape=np.array([n,h*w]),indptr=indptr,indices=indices,data=data,
        frame_height=np.array(h),frame_width=np.array(w),**meta)
    with zipfile.ZipFile(device/'event.npz','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for key,value in arrays.items():
            with z.open(key+'.npy','w',force_zip64=True) as member:np.lib.format.write_array(member,np.asarray(value),allow_pickle=False)
    with np.load(device/'event.npz',allow_pickle=False) as z:
        for key,value in arrays.items():assert np.array_equal(z[key],value),key
    print(i,'All Event frames and metadata verified',n,flush=True)
    return {'frames':n,'nnz':len(indices)}


def labels(i,device):
    from visualize_processed_session import body_model,vertices
    model=body_model()
    with h5py.File(DEST/'device0/label.h5','r') as src,h5py.File(SOURCE/f'RGB/device{i}.h5','r') as rgb:
        anchor=rgb['synchronization/master_timestamp_ns'][:].astype(np.int64)
        gt_times=src['frames/reference_timestamp_ns'][:].astype(np.int64)
        candidates=np.searchsorted(gt_times,anchor)
        inside=candidates<len(gt_times)
        matching=np.zeros(len(anchor),bool)
        matching[inside]=gt_times[candidates[inside]]==anchor[inside]
        rgb_positions=np.flatnonzero(matching)
        rows=candidates[matching]
        if not len(rows):raise RuntimeError('No exact server-I anchors match available world-space GT')
        params={key:src['params/'+key][rows] for key in src['params']}
        shape_ids=src['bboxes/smplx_shape_id'][rows]
        R=rgb['calibration/rgb/rotation_matrix'][:];t=rgb['calibration/rgb/translation'][:];K=rgb['calibration/rgb/intrinsic_matrix'][:]
        g=rgb['streams/images'];h,w=int(g.attrs['height']),int(g.attrs['width'])
        full_rgb_count=len(g['timestamp_ns'])
        rgb_times=g['timestamp_ns'][rgb_positions];rgb_idx=g['frame_index'][rgb_positions]
        frames={key:src['frames/'+key][rows] for key in ['reference_position','reference_frame_index','reference_timestamp_ns']}
    n,people=params['smplx_betas'].shape[:2]
    boxes=np.zeros((n,people,6,4),np.int32);valid=np.zeros((n,people,6),bool);visible=valid.copy()
    with h5py.File(ROOT/'dataset'/NAME/'RGBD/device0_mamma.h5','r') as f:
        parts=[f['topology/part_vertex_indices/'+key][:] for key in ['full_body','head','left_hand','right_hand','left_foot','right_foot']]
    for start in range(0,n,16):
        end=min(start+16,n);batch=end-start
        pose=params['smplx_pose_axis_angle'][start:end].reshape(-1,165)
        beta=params['smplx_betas'][start:end].reshape(-1,16)
        trans=params['translation_world_m'][start:end].reshape(-1,3)
        present=np.isfinite(pose).all(axis=1)&np.isfinite(beta).all(axis=1)&np.isfinite(trans).all(axis=1)
        if 'person_valid' in params:present &= params['person_valid'][start:end].reshape(-1)
        pose=np.nan_to_num(pose,nan=0,posinf=0,neginf=0)
        beta=np.nan_to_num(beta,nan=0,posinf=0,neginf=0)
        trans=np.nan_to_num(trans,nan=0,posinf=0,neginf=0)
        v=vertices(model,pose,beta,trans).reshape(batch,people,10475,3)@R.T+t
        projected=v@K.T;uv=projected[...,:2]/np.maximum(projected[...,2:],1e-6)
        for k,part in enumerate(parts):
            lo=uv[:,:,part].min(axis=2);hi=uv[:,:,part].max(axis=2)
            margin=np.max(hi-lo,axis=-1,keepdims=True)*.05;lo-=margin;hi+=margin
            boxes[start:end,:,k]=np.concatenate([np.floor(lo),np.ceil(hi)],axis=-1).astype(np.int32)
            valid[start:end,:,k]=(v[:,:,part,2].min(axis=-1)>0)&np.isfinite(lo).all(axis=-1)&present.reshape(batch,people)
            overlap=np.maximum(np.minimum(hi,[w,h])-np.maximum(lo,0),0).prod(axis=-1)
            area=np.maximum((hi-lo).prod(axis=-1),1)
            visible[start:end,:,k]=valid[start:end,:,k]&(overlap/area>=.4)
        if start%160==0:print(i,'projected label frames',start,'/',n,flush=True)
    boxes[~valid]=0
    with h5py.File(device/'label.h5','w') as f:
        for key,value in params.items():f.create_dataset('params/'+key,data=value,compression='gzip',shuffle=True)
        for key,value in frames.items():f.create_dataset('frames/'+key,data=value)
        inverse=np.full(full_rgb_count,-1,np.int64);inverse[rgb_positions]=np.arange(n)
        for key,value in {'source_rgb_position':rgb_positions,'source_rgb_frame_index':rgb_idx,'source_rgb_timestamp_ns':rgb_times,
            'result_index':np.arange(n),'result_index_by_source_rgb_position':inverse}.items():f.create_dataset('frames/'+key,data=value)
        f.create_dataset('bboxes/xyxy',data=boxes,compression='gzip');f.create_dataset('bboxes/valid',data=valid,compression='gzip')
        f.create_dataset('bboxes/smplx_shape_id',data=shape_ids,compression='gzip')
        f.create_dataset('bboxes/visible',data=visible,compression='gzip');f.create_dataset('bboxes/source',data=valid.astype(np.uint8),compression='gzip')
        f['bboxes'].attrs['source_codes']='0=unavailable,1=projected SMPL-X; no source detection/mask boxes exist for these cameras'
        f['bboxes'].attrs['part_order']='full_body,head,left_hand,right_hand,left_foot,right_foot'
        f.attrs['logical_device_id']=i;f.attrs['person_masks_stored']=False
        f.attrs['pose_provenance']='Exact timestamp-matched device0 world-space SMPL-X GT; camera boxes reprojected using local neutral SMPL-X model'
    print(i,'Label GT anchors matched exactly',n,flush=True)
    return {'frames':n,'exact_master_timestamp_matches':n,'bbox_source':'derived SMPL-X projection, not original detector GT'}


def worker(i):
    device=DEST/f'device{i}';device.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        video=pool.submit(copy_video,i,device);event=pool.submit(copy_events,i,device)
        info={'device':i,'video':video.result(),'event':event.result()}
    info['labels']=labels(i,device)
    for kind in ['preview','event']:
        cmd=[sys.executable,str(ROOT/'tools/processed_event_videos.py'),'--kind',kind,'--session',str(DEST),'--device',f'device{i}',
            '--gpus',str(i-5),'--cpu-preview']
        if kind=='preview':cmd+=['--max-seconds','60']
        subprocess.run(cmd,check=True)
    (WORK/f'device{i}_validation.json').write_text(json.dumps(info,indent=2)+'\n')
    print(i,'CAMERA COMPLETE',flush=True)


if __name__=='__main__':
    WORK.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(worker,[5,6]):pass

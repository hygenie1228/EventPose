"""Native-resolution event occupancy pilot; reuses read-only source/FK helpers."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import smplx
from idea_event_motion_probe import ROOT, prefix, load_gt, save

OUT=ROOT/'exp/261026/idea_motion_native_probe'


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    sp=json.loads((ROOT/'dataset/split.json').read_text())
    names=['20260720_125338_040324_200_f','20260720_132719_522887_201_f',
           '20260720_131207_863008_200_b','20260720_134353_764469_201_b']
    assert set(names)<=set(sp['train']) and not set(names)&set(sp['test'])
    model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',num_betas=16,use_pca=False)
    start=time.time();audit=[]
    for sn,name in enumerate(names):
        times,joints,fk=load_gt(model,name,21,10)
        for device in ([0,1,2] if sn==2 else [0]):
            path=ROOT/'dataset'/name/f'device{device}'/'event.npz'
            stat=(path.stat().st_size,path.stat().st_mtime_ns)
            print('BEGIN',name,device,flush=True)
            with np.load(path,allow_pickle=False) as z:
                et=z['timestamp_ns'].astype(np.int64);ptr=z['indptr'];h,w=map(int,z['image_shape'])
                K=z['rgb_calibration__event__intrinsic_matrix'].astype(float)
                R=z['rgb_calibration__event__rotation_matrix'].astype(float)
                t=z['rgb_calibration__event__translation'].astype(float)
            hi=int(np.searchsorted(et,times[-1],side='right'))
            idx=prefix(path,'indices',int(ptr[hi]));pol=prefix(path,'data',int(ptr[hi]))>0
            pts=joints@R.T+t;pr=pts@K.T;uv=pr[...,:2]/pr[...,2:]
            visible=(pts[...,2]>.1)&(uv[...,0]>=21)&(uv[...,0]<w-21)&(uv[...,1]>=21)&(uv[...,1]<h-21)
            support=np.full((len(times),4,22),np.nan);bg=np.full((len(times),4),np.nan)
            observations=np.zeros((len(times),4));pixels=np.zeros_like(observations)
            old_fraction=np.zeros_like(observations);cancel_fraction=np.zeros_like(observations)
            for i,end in enumerate(times):
                cx=np.clip(np.rint(uv[i,:,0]).astype(int),21,w-22)
                cy=np.clip(np.rint(uv[i,:,1]).astype(int),21,h-22)
                bx=np.array([30,w-31,30,w-31]);by=np.array([30,30,h-31,h-31])
                for wi,ms in enumerate([10,33,100,200]):
                    lo,hi=np.searchsorted(et,[int(end)-ms*1_000_000,int(end)],side='right')
                    a,b=map(int,ptr[[lo,hi]])
                    mask=np.zeros(h*w,np.uint8);mask[idx[a:b]]=1
                    integral=cv2.integral(mask.reshape(h,w))
                    def rect(x,y):return (integral[y+21,x+21]-integral[y-20,x+21]-integral[y+21,x-20]+integral[y-20,x-20])/1681
                    support[i,wi]=np.where(visible[i],rect(cx,cy),np.nan)
                    distances=np.linalg.norm(np.stack([bx,by],1)[:,None]-uv[i][None],axis=-1)
                    legal=np.min(distances,axis=1)>80
                    bg[i,wi]=np.mean(rect(bx,by)[legal]) if legal.any() else np.nan
                    observations[i,wi]=b-a;pixels[i,wi]=mask.sum()
                    recent=int(np.searchsorted(et,int(end)-10_000_000,side='right'))
                    m2=np.zeros(h*w,np.uint8);m2[idx[int(ptr[recent]):b]]=1
                    old_fraction[i,wi]=float(np.sum((mask!=0)&(m2==0))/max(mask.sum(),1))
                    pos=np.bincount(idx[a:b][pol[a:b]],minlength=h*w)
                    neg=np.bincount(idx[a:b][~pol[a:b]],minlength=h*w)
                    cancel_fraction[i,wi]=float(np.sum((pos==neg)&(pos>0))/max(mask.sum(),1))
            assert stat==(path.stat().st_size,path.stat().st_mtime_ns)
            np.savez_compressed(OUT/f'{name}_device{device}.npz',times=times,joints=joints,uv=uv,visible=visible,
                support=support,background_support=bg,observations=observations,pixels=pixels,
                old_fraction=old_fraction,cancel_fraction=cancel_fraction)
            audit.append(dict(session=name,device=device,frames=len(times),native_shape=[h,w],patch_width_px=41,
                fk_max_difference_m=fk,readonly_stat_unchanged=True,source_size=stat[0],source_mtime_ns=stat[1],
                support_median=np.nanmedian(support,axis=(0,2)).tolist(),bg_median=np.nanmedian(bg,axis=0).tolist(),
                old_only_fraction_median=np.median(old_fraction,axis=0).tolist()))
            save(OUT/'source_audit.json',audit)
            print('DONE',name,device,'elapsed',round(time.time()-start,1),flush=True)
    save(OUT/'completion.json',dict(start_unix=start,end_unix=time.time(),success=True,session_views=len(audit)))

if __name__=='__main__':main()

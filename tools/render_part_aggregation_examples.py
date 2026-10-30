"""Real held-out RGB/events, learned RGB part maps and measured time weights.

Cases are fixed using GT motion alone, independently of model accuracy and
attention values. GT crops are display-only. The same cases apply to v1/v2/v3.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import zipfile

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261029/part_aware_dynamic_aggregation'
PRIOR=ROOT/'exp/261028/device3_encoding_12h'
PARTS=['Torso','Left arm','Right arm','Left leg','Right leg','Head']


def stat(p):
    s=Path(p).stat()
    return dict(path=str(Path(p).resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def array_range(path,name,start,end):
    """Read only the requested NPY range; ZIP seeking uses bounded buffers."""
    with zipfile.ZipFile(path) as archive,archive.open(name+'.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        assert len(shape)==1 and not fortran and 0<=start<=end<=shape[0]
        f.seek(start*dtype.itemsize,1)
        raw=f.read((end-start)*dtype.itemsize)
        assert len(raw)==(end-start)*dtype.itemsize
        return np.frombuffer(raw,dtype=dtype).copy()


def event_display(p,n):
    active=(p+n)>0
    support=cv2.boxFilter(active.astype(np.float32),-1,(3,3),normalize=False,
                         borderType=cv2.BORDER_CONSTANT)>=2
    a=np.sqrt(np.minimum(np.log1p(p)/np.log(9),1))*support
    b=np.sqrt(np.minimum(np.log1p(n)/np.log(9),1))*support
    return np.clip(np.stack([a,.25*a+.65*b,b],-1),0,1).astype(np.float32)


def bounds(uv,shape,margin):
    h,w=shape;uv=uv[np.isfinite(uv).all(1)]
    b=np.r_[np.floor(uv.min(0)-margin),np.ceil(uv.max(0)+margin)].astype(int)
    b[[0,2]]=np.clip(b[[0,2]],0,w);b[[1,3]]=np.clip(b[[1,3]],0,h)
    assert b[2]>b[0] and b[3]>b[1]
    return b.tolist()


def crop(im,b):return im[b[1]:b[3],b[0]:b[2]]


def selected_cases(gt):
    path=OUT/'figures/example_cases.json'
    if path.exists():return json.loads(path.read_text())
    motion=gt['event3_part_path_px'];arms=motion[:,1:3].mean(1);legs=motion[:,3:5].mean(1)
    # Bound source decoding cost, explicitly before looking at model outputs.
    early=gt['frames']<=1800
    a=int(np.argmax(np.where(early&(arms>=20),arms/(legs+5),-1)))
    b=int(np.argmax(np.where(early&(gt['sessions']!=gt['sessions'][a]),np.minimum(arms,legs),-1)))
    cases=[dict(case=label,gt_index=i,session=str(gt['sessions'][i]),frame=int(gt['frames'][i]),
                mean_arm_path_px=float(arms[i]),mean_leg_path_px=float(legs[i]))
           for label,i in [('A',a),('B',b)]]
    value=dict(cases=cases,selected_unix=time.time(),selection='GT-only, source frames<=1800; A maximum arms/(legs+5) with arms>=20px; B maximum min(arms,legs) in a different session',
               role='Train holdout; NOT Test',accuracy_or_attention_used_for_selection=False,
               v2_attention_existed_at_selection=(OUT/'diagnostics/v2_part_dynamic_attention.npz').exists())
    path.write_text(json.dumps(value,indent=2)+'\n');return value


def observations(case,gt):
    folder=OUT/'figures/example_sources';folder.mkdir(exist_ok=True)
    dest=folder/(case['case']+'.npz');meta=folder/(case['case']+'.json')
    if dest.exists():
        with np.load(dest) as z:arrays={k:z[k] for k in z.files}
        return arrays,json.loads(meta.read_text())
    i=case['gt_index'];session=case['session'];row=int(gt['cache_row'][i]);query=int(gt['query_timestamp_ns'][i])
    eventpath=ROOT/'dataset'/session/'device3/event.npz'
    cachepath=PRIOR/'cache'/(session+'.h5')
    before=[stat(eventpath),stat(cachepath)]
    with h5py.File(cachepath,'r',locking=False) as f:
        assert int(f['frames'][row])==case['frame']
        rgb=f['rgb3'][row].transpose(1,2,0)
        transform=f['rgb3_K_primary'][:]@np.linalg.inv(f['rgb3_K_native'][:])
    native=gt['rgb3_end_joint_uv'][i]
    uv=np.c_[native,np.ones(len(native))]@transform.T
    rb=bounds(uv[:,:2]/uv[:,2:],rgb.shape[:2],45)
    with np.load(eventpath,allow_pickle=False) as z:
        times=z['timestamp_ns'].astype(np.int64);ptr=z['indptr'];h,w=map(int,z['image_shape'])
    lo,hi=np.searchsorted(times,[query-200_000_000,query],side='right');start,end=map(int,ptr[[lo,hi]])
    ids=array_range(eventpath,'indices',start,end)
    positive=array_range(eventpath,'data',start,end)>0
    eb=bounds(gt['event3_end_joint_uv'][i],(h,w),45)
    arrays={'rgb':crop(rgb,rb)}
    for k,(near,far) in enumerate([(0,33),(33,100),(100,200)]):
        l,r=np.searchsorted(times,[query-far*1_000_000,query-near*1_000_000],side='right')
        a,b=int(ptr[l])-start,int(ptr[r])-start
        pix=ids[a:b];pol=positive[a:b]
        p=np.bincount(pix[pol],minlength=h*w).reshape(h,w)
        n=np.bincount(pix[~pol],minlength=h*w).reshape(h,w)
        arrays['event'+str(k)]=crop(event_display(p,n),eb)
        assert l==r or int(times[r-1])<=query-near*1_000_000
    assert before==[stat(eventpath),stat(cachepath)]
    details=dict(case=case,query_timestamp_ns=query,rgb_crop=rb,event_crop=eb,
                 rgb_canvas_shape=list(rgb.shape[:2]),source_stat=before,
                 GT_crop_for_display_only=True,model_input_uses_GT=False,
                 pixel_overlay_assumed=False,future_event_used=False,
                 event_display='Fixed sqrt(min(log1p(N)/log9,1)); positive orange/red, negative cyan; weak3x3 support>=2 for all bins; display only')
    np.savez_compressed(dest,**arrays);meta.write_text(json.dumps(details,indent=2)+'\n')
    print('Real sample extracted:',case['case'],session,flush=True)
    return arrays,details


def render(name):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    with np.load(OUT/'diagnostics/holdout_part_motion_targets.npz') as f:g={k:f[k] for k in f.files}
    selected=selected_cases(g)
    with np.load(OUT/'diagnostics'/(name+'_attention.npz')) as f:a={k:f[k] for k in f.files}
    lookup={(str(s),int(t)):i for i,(s,t) in enumerate(zip(a['sessions'],a['frames']))}
    rows=[];allmaps=[]
    for case in selected['cases']:
        idx=lookup[(case['session'],case['frame'])]
        obs,meta=observations(case,g)
        grid=math.isqrt(a['part_maps'].shape[-1]);assert grid*grid==a['part_maps'].shape[-1]
        maps=a['part_maps'][idx].reshape(6,grid,grid)
        density=np.stack([maps[1:3].sum(0),maps[3:5].sum(0)])
        h,w=meta['rgb_canvas_shape']
        up=np.stack([crop(cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR),meta['rgb_crop']) for m in density])
        allmaps.append(up)
        weights=np.column_stack([a['time_weights'][idx],a['null_weights'][idx]])
        assert weights.shape==(6,4) and np.allclose(weights.sum(1),1,atol=2e-6)
        rows.append((case,idx,obs,meta,up,weights))
    vmax=max(float(np.quantile(x,.995)) for x in allmaps)
    fig,axes=plt.subplots(2,6,figsize=(19.5,9),gridspec_kw={'width_ratios':[1.65,1.1,1.1,1.1,1.65,3.0]})
    fig.subplots_adjust(left=.025,right=.985,top=.89,bottom=.115,wspace=.16,hspace=.30)
    display_name='V1 part-aware' if name=='part_dynamic' else name.replace('_',' ')
    fig.text(.025,.965,display_name+': actual inputs and learned part/time choices',fontsize=20,weight='bold')
    fig.text(.025,.925,'Fixed held-out cases selected by GT motion only. RGB part maps are learned attention, not segmentation.',fontsize=11,color='#596b79')
    for row,(case,idx,obs,meta,maps,weights) in enumerate(rows):
        rgb=obs['rgb'].astype(np.float32)/255
        arms=np.clip(maps[0]/max(vmax,1e-10),0,1);legs=np.clip(maps[1]/max(vmax,1e-10),0,1)
        alpha=np.maximum(arms,legs)*.65
        color=(arms[...,None]*np.array([1.,.32,.08])+legs[...,None]*np.array([.05,.7,1.]))/np.maximum((arms+legs)[...,None],1e-10)
        overlay=rgb*(1-alpha[...,None])+color*alpha[...,None]
        imgs=[obs['rgb'],obs['event0'],obs['event1'],obs['event2'],overlay]
        labels=['RGB3 query','0–33 ms','33–100 ms','100–200 ms','RGB part attention']
        for ax,img,label in zip(axes[row,:5],imgs,labels):
            ax.imshow(img);ax.axis('off');ax.set_title(label,fontsize=11,pad=7)
        axes[row,4].text(.5,-.035,'Orange: arms | Cyan: legs',transform=axes[row,4].transAxes,ha='center',fontsize=9,color='#556674')
        ax=axes[row,5];ax.imshow(weights,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_title('Measured aggregation weights',fontsize=11,pad=7)
        ax.set_xticks(range(4),['0–33','33–100','100–200','Null'],fontsize=9)
        ax.set_yticks(range(6),PARTS,fontsize=9);ax.tick_params(length=0)
        for y in range(6):
            for x in range(4):
                val=float(weights[y,x]);label=f'{val:.1e}' if 0<val<.001 else f'{val:.2f}'
                ax.text(x,y,label,ha='center',va='center',fontsize=8,color='white' if val>.55 else '#253f55')
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xlabel('History age (ms); shared probability scale 0–1',fontsize=9)
        axes[row,0].text(0,-.07,f"{case['case']} | subject {case['session'].split('_')[3]} | frame {case['frame']}",transform=axes[row,0].transAxes,fontsize=9)
        fig.text(.025,.50 if row==0 else .065,f"{case['case']}: GT 200 ms path — arms {case['mean_arm_path_px']:.1f}px, legs {case['mean_leg_path_px']:.1f}px. P(event) = {weights[:,:3].sum(1).mean():.3g}.",fontsize=10,color='#465d70')
    fig.text(.025,.025,'Real device3 RGB + native event observations. GT defines display crops and case selection only. No pose prediction or synthetic attention shown.',fontsize=10,color='#596b79')
    stem=folder/(name+'_actual_examples')
    for ext in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=200,facecolor='white')
    plt.close(fig)
    manifest=dict(model=name,selection=selected,learned_attention_source=str(OUT/'diagnostics'/(name+'_attention.npz')),
                  attention_sha256=hashlib.sha256((OUT/'diagnostics'/(name+'_attention.npz')).read_bytes()).hexdigest(),
                  shared_map_display_vmax=vmax,attention_is_segmentation=False,weights_are_measured=True,
                  samples=[r[3] for r in rows],GT_model_input=False,Test_used=False)
    stem.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(stem.with_suffix('.png'))


if __name__=='__main__':
    cv2.setNumThreads(1)
    p=argparse.ArgumentParser();p.add_argument('--name',default='part_dynamic');args=p.parse_args()
    render(args.name)

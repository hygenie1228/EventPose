"""English paper figure for the implemented part/time aggregation architecture.

Actual Training RGB3 and native CSR observations; no invented learned weights.
GT is used only to choose readable display crops, never as a model input.
"""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from visualize_rgb_event_alignment import prefix_array

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation/figures'
PRIOR = ROOT/'exp/261028/device3_encoding_12h'
SESSION = '20260720_131207_863008_200_b'
PARTS = ['Torso', 'Left arm', 'Right arm', 'Left leg', 'Right leg', 'Head']
COLORS = ['#718096', '#e29540', '#a975c9', '#58a779', '#5199c8', '#c66e81']


def stat(path):
    s = path.stat()
    return dict(path=str(path.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def display(positive, negative):
    active=(positive+negative)>0
    support=cv2.boxFilter(active.astype(np.float32), -1, (3,3), normalize=False,
                         borderType=cv2.BORDER_CONSTANT)>=2
    p=np.sqrt(np.minimum(np.log1p(positive)/np.log(9),1))*support
    n=np.sqrt(np.minimum(np.log1p(negative)/np.log(9),1))*support
    return np.clip(np.stack([p,.25*p+.65*n,n],-1),0,1)


def crop_box(points, width, height, margin):
    points=points[np.isfinite(points).all(-1)]
    low=np.floor(points.min(0)-margin).astype(int)
    high=np.ceil(points.max(0)+margin).astype(int)
    b=np.r_[low,high]
    b[[0,2]]=np.clip(b[[0,2]],0,width)
    b[[1,3]]=np.clip(b[[1,3]],0,height)
    assert b[2]>b[0] and b[3]>b[1]
    return b.tolist()


def sample():
    cached=OUT/'architecture_sample.npz'
    meta=OUT/'architecture_sample.json'
    if cached.exists():
        with np.load(cached) as z:
            return {k:z[k] for k in z.files},json.loads(meta.read_text())
    src=ROOT/'dataset'/SESSION/'device3'
    paths=[src/n for n in ['rgb.mp4','event.npz','label.h5']]
    before=[stat(p) for p in paths]
    with h5py.File(OUT.parent/'memory_cache'/(SESSION+'.h5'),'r',locking=False) as f:
        stamps=f['rgb3_timestamp_ns'][:]
        i=int(np.argmin(np.abs(stamps-(int(stamps[0])+12_000_000_000))))
        query=int(stamps[i]); frame=int(f['rgb3_source_positions'][i])
        gt_frame=int(f['frames'][i])
    with h5py.File(PRIOR/'cache'/(SESSION+'.h5'),'r',locking=False) as f:
        Kr=f['rgb3_K_native'][:];Ke=f['event3_K_native'][:]
        R=f['event3_R_common_to_sensor'][:];t=f['event3_t_common_to_sensor'][:]
    with np.load(PRIOR/'targets'/(SESSION+'.npz')) as z:
        person=int(np.flatnonzero(z['valid'][i])[0]);joints=z['joints'][i,person,:22]
    project=lambda p,K:(p@K.T)[:,:2]/(p@K.T)[:,2:]
    cap=cv2.VideoCapture(str(src/'rgb.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release()
    assert ok
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    rb=crop_box(project(joints,Kr),rgb.shape[1],rgb.shape[0],65)
    with np.load(src/'event.npz',allow_pickle=False) as z:
        times=z['timestamp_ns'].astype(np.int64);ptr=z['indptr'];h,w=map(int,z['image_shape'])
    left,right=np.searchsorted(times,[query-200_000_000,query],side='right')
    a,b=map(int,ptr[[left,right]])
    idx=prefix_array(src/'event.npz','indices',b)[a:b]
    positive=prefix_array(src/'event.npz','data',b)[a:b]>0
    eb=crop_box(project(joints@R.T+t,Ke),w,h,50)
    arrays={'rgb':rgb[rb[1]:rb[3],rb[0]:rb[2]]}
    for k,(near,far) in enumerate([(0,33),(33,100),(100,200)]):
        lo,hi=np.searchsorted(times,[query-far*1_000_000,query-near*1_000_000],side='right')
        l,r=int(ptr[lo])-a,int(ptr[hi])-a
        ids=idx[l:r];pol=positive[l:r]
        p=np.bincount(ids[pol],minlength=h*w).reshape(h,w)
        n=np.bincount(ids[~pol],minlength=h*w).reshape(h,w)
        arrays['bin'+str(k)]=display(p,n)[eb[1]:eb[3],eb[0]:eb[2]]
        assert hi==lo or int(times[hi-1])<=query-near*1_000_000
    assert before==[stat(p) for p in paths]
    metadata=dict(session=SESSION,device=3,role='Training',native_rgb_frame=frame,
        common_gt_frame=gt_frame,memory_cache_row=i,query_timestamp_ns=query,
        source_stat=before,rgb_crop=rb,event_crop=eb,
        crop_gt_for_display_only=True,model_uses_GT=False,learned_weights_shown=False,
        event_bins_ms=[[0,33],[33,100],[100,200]],source_readonly=True,
        event_display='Black; fixed sqrt(min(log1p(N)/log(9),1)); positive orange/red and negative cyan; identical weak3x3 support>=2 for all bins; visualization only',
        architecture_source_sha256=hashlib.sha256((ROOT/'src/multihmr/part_aggregation_model.py').read_bytes()).hexdigest())
    np.savez_compressed(cached,**arrays)
    meta.write_text(json.dumps(metadata,indent=2)+'\n')
    return arrays,metadata


def draw(data,meta):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
    fig=plt.figure(figsize=(20,8))
    ax=fig.add_axes([.015,.035,.97,.94]);ax.set(xlim=(0,20),ylim=(0,8));ax.axis('off')
    ink='#243443'; muted='#667785'
    def text(x,y,s,size=11,bold=False,color=ink,ha='center'):
        ax.text(x,y,s,fontsize=size,weight='bold' if bold else 'normal',color=color,
                ha=ha,va='center',linespacing=1.25,zorder=6)
    def box(x,y,w,h,s,color='#e6edf3',size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.07',
                                  fc=color,ec='#81909d',lw=1.1,zorder=3))
        text(x+w/2,y+h/2,s,size,True)
    def arrow(x1,y1,x2,y2,color='#425a6d'):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                                   mutation_scale=13,lw=1.4,color=color,zorder=5))
    def photo(im,x,y,w,h):
        ih,iw=im.shape[:2];scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
        ax.imshow(im,extent=(x+(w-ww)/2,x+(w+ww)/2,y+(h-hh)/2,y+(h+hh)/2),zorder=4)
    # Quiet panels leave the mechanism visually dominant.
    for x,w,c,title in [(0,5.1,'#f0f5fa','1. RGB-conditioned part queries'),
                         (5.25,5.3,'#f0f7f2','2. Calibrated event memory'),
                         (10.7,5.2,'#fff7e8','3. Part-aware temporal aggregation'),
                         (16.05,3.95,'#f4eff9','4. RGB residual + pose head')]:
        ax.add_patch(FancyBboxPatch((x,.55),w,6.9,boxstyle='round,pad=.02,rounding_size=.1',fc=c,ec='none'))
        text(x+w/2,7.12,title,12,True)
    text(10,7.8,'Part-aware Dynamic Aggregation',21,True)
    photo(data['rgb'],.2,3.9,1.2,2.35)
    text(.8,3.62,'RGB3 at t',10,True)
    box(1.65,4.65,1.35,.95,'RGB ViT',size=11)
    arrow(1.4,5.12,1.65,5.12)
    box(3.3,4.55,1.5,1.15,'RGB spatial\ntokens',size=11)
    arrow(3,5.12,3.3,5.12)
    arrow(4.05,4.55,4.05,3.9)
    box(2.25,2.95,2.55,.9,'6 learned part queries',color='#dae9f7',size=11)
    text(1.05,2.55,'No GT at inference',9,color=muted)
    for k,p in enumerate(PARTS):
        yy=2.42-k*.25
        ax.add_patch(Rectangle((2.35,yy-.075),.12,.15,fc=COLORS[k],ec='none'))
        text(2.55,yy,p,9,ha='left')
    text(2.55,.88,'Train-only weak part supervision',9,color=muted)
    # Real observed bins, newest first; no synthetic attention matrix.
    for k,label in enumerate(['0–33 ms','33–100 ms','100–200 ms']):
        x=5.5+k*1.56
        photo(data['bin'+str(k)],x,4.8,1.34,1.65)
        text(x+.67,4.55,label,10,True)
        arrow(x+.67,4.35,x+.67,3.97)
    box(5.6,3.05,4.65,.85,'Shared event patch + spatial Transformer',color='#deeee3',size=10)
    arrow(7.92,3.05,7.92,2.65)
    box(5.6,1.65,4.65,.95,'Calibrated ray / depth matching',color='#deeee3',size=11)
    text(7.92,1.22,'Separate RGB/event optics • K, R, t',9,color=muted)
    text(7.92,.88,'Past-only, non-overlapping bins',9,color=muted)
    # Dominant core: each part chooses time, not a shared global window.
    box(11.1,4.95,4.4,1.12,'Part RGB + part event + time',color='#f8e8be',size=12)
    ax.plot([4.8,5.05,5.05,10.9,10.9],[3.4,3.4,6.6,6.6,5.65],color='#425a6d',lw=1.3,zorder=5)
    arrow(10.9,5.65,11.1,5.65)
    text(10.35,6.43,'Part RGB',9,color=muted)
    arrow(10.25,2.12,10.85,2.12)
    arrow(10.85,2.12,10.85,5.2)
    arrow(10.85,5.2,11.1,5.2)
    arrow(13.3,4.95,13.3,4.48)
    box(11.1,3.42,4.4,.98,'Time-score MLP + null choice',color='#f6dda4',size=12)
    arrow(13.3,3.42,13.3,2.93)
    box(11.1,1.75,4.4,1.1,'6 part-specific time distributions\n→ spatial event messages',color='#f8e8be',size=11)
    text(13.3,1.27,'Weights are learned from the pose task',9,color=muted)
    text(13.3,.9,'No fixed “fast = short” rule or GT motion input',9,color=muted)
    # Feature skip and residual, faithful to the actual backbone hook.
    arrow(15.5,2.3,16.52,2.3)
    box(16.65,1.8,2.7,1.,'Project event residual\n+ RGB spatial tokens',color='#e7def1',size=11)
    arrow(19.5,6.8,19.5,2.3)
    arrow(19.5,2.3,19.35,2.3)
    ax.plot([4.8,19.5],[6.8,6.8],color='#506b83',lw=1.3,zorder=4)
    arrow(4.8,5.55,4.8,6.8)
    text(17.9,6.57,'RGB skip connection',9,color=muted)
    arrow(18,2.8,18,3.32)
    box(16.65,3.4,2.7,1.1,'Multi-HMR\nSMPL-X pose head',color='#e7def1',size=11)
    arrow(18,4.5,18,4.98)
    box(16.65,5.05,2.7,.86,'3D body pose',color='#e7def1',size=12)
    text(18,.95,'Output projection starts at zero',9,color=muted)
    text(.05,.19,'Actual Training samples: device3 RGB + device3 event. Architecture schematic; no learned weights or pose predictions shown.',9,color=muted,ha='left')
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/('part_aware_architecture.'+ext),dpi=220,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    data,meta=sample();draw(data,meta)
    print(OUT/'part_aware_architecture.png')

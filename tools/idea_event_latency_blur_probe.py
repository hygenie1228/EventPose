"""Small CPU preprocessing benchmark and explicitly synthetic RGB blur controls."""
import json
import sys
import time
from pathlib import Path
import cv2
import h5py
import numpy as np
from idea_event_motion_probe import ROOT, OUT, prefix, save
sys.path.insert(0,str(ROOT/'src'))
from utils.event_encoding import EventFrames


def main():
    cv2.setNumThreads(1)
    manifest=json.loads((OUT/'manifest.json').read_text());name=manifest['sessions'][0]
    path=ROOT/'dataset'/name/'device0/event.npz'
    with np.load(OUT/f'{name}_device0.npz') as z:queries=z['times'][::10]
    with np.load(path,allow_pickle=False) as z:
        et=z['timestamp_ns'];ptr=z['indptr'];shape=z['image_shape']
    hi=int(np.searchsorted(et,queries[-1],side='right'));size=int(ptr[hi])
    frames=EventFrames(et[:hi],ptr[:hi+1],prefix(path,'indices',size),prefix(path,'data',size),shape,size=224)
    stats={}
    for encoding in ['visualize_gray','count33','bins100','surface20']:
        frames.encode(int(queries[0]),encodings=[encoding]);timings=[]
        for q in queries:
            t0=time.perf_counter();enc,meta=frames.encode(int(q),encodings=[encoding]);timings.append((time.perf_counter()-t0)*1000)
            assert meta['latest_event_timestamp_ns']<=int(q)
        stats[encoding]=dict(n=len(timings),median_ms=float(np.median(timings)),p95_ms=float(np.quantile(timings,.95)),maximum_ms=float(max(timings)))
    save(OUT/'preprocessing_latency.json',dict(session=name,device='CPU',output_size=224,opencv_threads=1,
        encodings=stats,included='Only in-memory representation construction, no disk read, HMR, CUDA, sensor response, transfer or output scheduling.',
        policy='One warmup per encoding then 20 distinct 1-second-spaced queries; not a full end-to-end benchmark.',
        window_note='A trailing causal W-ms window does not necessarily require W-ms extra wait at steady state. Query cadence, startup, lookahead and stale evidence must be measured separately.'))
    # Synthetic line blur is an image stress-test artifact, not a natural-blur detector.
    image_rows=[];blur_records=[]
    for name in manifest['sessions'][:2]:
        p=ROOT/'dataset'/name/'device0'
        with h5py.File(p/'label.h5','r',locking=False) as f:
            stamps=f['frames/source_rgb_timestamp_ns'][:]
            row=int(np.searchsorted(stamps,int(stamps[0])+12_000_000_000))
            position=int(f['frames/source_rgb_position'][row]);box=f['bboxes/xyxy'][row,0,0].astype(float)
            bbox_shape=[int(f['bboxes'].attrs['height']),int(f['bboxes'].attrs['width'])]
        cap=cv2.VideoCapture(str(p/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,position);ok,im=cap.read();cap.release();assert ok
        h,w=im.shape[:2]
        box*=np.array([w/bbox_shape[1],h/bbox_shape[0],w/bbox_shape[1],h/bbox_shape[0]])
        x0,y0,x1,y1=map(int,box)
        x0=max(0,x0-25);y0=max(0,y0-25);x1=min(w,x1+25);y1=min(h,y1+25)
        assert x1>x0 and y1>y0
        versions=[];records=[]
        for length in [1,15,31]:
            kernel=np.zeros((length,length),np.float32);kernel[length//2,:]=1/length
            blurred=im if length==1 else cv2.filter2D(im,-1,kernel)
            crop=blurred[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            records.append(dict(horizontal_kernel_pixels=length,laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var())))
            scale=min(480/crop.shape[1],480/crop.shape[0]);new=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
            panel=np.full((480,480,3),245,np.uint8);top=(480-new.shape[0])//2;left=(480-new.shape[1])//2
            panel[top:top+new.shape[0],left:left+new.shape[1]]=new;versions.append(panel)
        image_rows.append(np.concatenate(versions,axis=1))
        blur_records.append(dict(session=name,frame=position,crop_xyxy=[x0,y0,x1,y1],conditions=records))
    cv2.imwrite(str(OUT/'synthetic_blur_controls.jpg'),np.concatenate(image_rows,axis=0))
    save(OUT/'synthetic_blur_controls.json',dict(records=blur_records,
        interpretation='Original / horizontal 15px / horizontal 31px. Artificial image-space blur, not measured exposure integration or natural motion blur; no pose model was evaluated. GT crop for visualization only.'))

if __name__=='__main__':main()

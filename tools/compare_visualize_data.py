"""Compare the unchanged visualize_data CSR renderer on the existing RGB timeline."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from compare_event_encodings import encoder, fit
from visualize_data import convert_event_npz_to_mp4
from visualize_rgb_event_alignment import prefix_array

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261025/event_encoding_visualize_data'


def register():
    config = ROOT/'_agents/research/261024/rgb_event_goal_36h.json'
    if not config.exists():
        return None
    settings = json.loads(config.read_text())
    assert time.time() < settings['deadline_unix']
    if os.getpid() != os.getpgrp():
        os.setsid()
    pid = os.getpid()
    stat = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    job = dict(pid=pid, pgid=os.getpgrp(), start_ticks=stat[19],
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),
        role='event_encoding_visualize_data_cpu_comparison', deadline_unix=settings['deadline_unix'])
    registry = Path(settings['managed_jobs_file'])
    # Use a host-local lock: NFS advisory locking may wait indefinitely.
    with open('/tmp/eventpose-managed-jobs.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with registry.open('r+') as file:
            data = json.load(file)
            data.setdefault('jobs',[]).append(job)
            file.seek(0)
            json.dump(data,file,ensure_ascii=False,indent=2)
            file.truncate()
    return settings['deadline_unix']


def render(session, role, deadline):
    source = ROOT/'dataset'/session/'device0'
    old = ROOT/'exp/261024/event_encoding_comparison'
    old = old/'body_refined' if role=='body' else old
    out = OUT/role
    out.mkdir(parents=True,exist_ok=True)
    temp = out/'tmp'
    temp.mkdir(exist_ok=True)
    with np.load(source/'event.npz',allow_pickle=False) as z:
        ts=z['timestamp_ns'].astype(np.int64)
        rgb_ts=z['rgb__timestamp_ns'].astype(np.int64)
        ptr=z['indptr'];h,w=map(int,z['image_shape'])
    targets=int(rgb_ts[0])+np.arange(600,dtype=np.int64)*100_000_000
    rgb_ids=np.minimum(np.searchsorted(rgb_ts,targets),len(rgb_ts)-1)
    previous=np.maximum(rgb_ids-1,0)
    rgb_ids=np.where(abs(rgb_ts[previous]-targets)<abs(rgb_ts[rgb_ids]-targets),previous,rgb_ids)
    ends=rgb_ts[rgb_ids]
    last=np.searchsorted(ts,ends[-1],side='right')
    indices=prefix_array(source/'event.npz','indices',int(ptr[last]))
    values=prefix_array(source/'event.npz','data',int(ptr[last]))
    rows=np.searchsorted(ts,ends,side='right')-1
    selected_i=[];selected_v=[];sample_ptr=[0]
    for row in rows:
        a,b=(int(ptr[row]),int(ptr[row+1])) if row>=0 else (0,0)
        selected_i.append(indices[a:b]);selected_v.append(values[a:b]);sample_ptr.append(sample_ptr[-1]+b-a)
    sampled=temp/'timestamp_selected_frames.npz'
    np.savez(sampled,frame_height=h,frame_width=w,indptr=np.array(sample_ptr),
             indices=np.concatenate(selected_i),data=np.concatenate(selected_v))
    # Call the user's original function without changing its implementation.
    raw_path=out/'visualize_data_single_frame.partial.mp4'
    convert_event_npz_to_mp4(str(sampled),str(raw_path),fps=10,crf=18)
    raw_path.rename(out/'visualize_data_single_frame.mp4')
    cap=cv2.VideoCapture(str(source/'rgb.mp4'))
    raw=cv2.VideoCapture(str(out/'visualize_data_single_frame.mp4'))
    contour=cv2.VideoCapture(str(old/('contour_supported_33ms.mp4' if role=='body' else 'contour_supported_100ms.mp4')))
    recent=cv2.VideoCapture(str(old/('time_surface_20ms.mp4' if role=='body' else 'comparison_60s.mp4')))
    motion=cv2.VideoCapture(str(old/'motion_3bins_100ms.mp4'))
    assert all(c.isOpened() for c in [cap,raw,contour,recent,motion])
    video=encoder(out/'comparison_60s.mp4',1440,784,10)
    gray_video=encoder(out/'gray_signed_33ms.mp4',w,h,10)
    labels=['RGB reference','visualize_data | ONE stored frame','Gray signed sum | 33 ms (adapted)',
            'Supported count | '+('33 ms' if role=='body' else '100 ms'),
            'Time surface | tau '+('20 ms' if role=='body' else '50 ms'),'3 time bins | 100 ms']
    stats=[];cursor=-1
    for step,(rgb_id,end) in enumerate(zip(rgb_ids,ends)):
        if deadline is not None:
            assert time.time()<deadline,'User deadline reached'
        while cursor<rgb_id:
            assert cap.grab();cursor+=1
        ok,rgb=cap.retrieve();assert ok
        decoded=[]
        for c in [raw,contour,recent,motion]:
            ok,im=c.read();assert ok;decoded.append(im)
        single,clean,surface,temporal=decoded
        if role=='face':
            # The existing 3x3 grid stores tau=50ms in panel 5; reuse that rendering.
            surface=surface[392+32:784,960:1440]
        lo,hi=np.searchsorted(ts,[int(end)-33_000_000,int(end)],side='right')
        a,b=int(ptr[lo]),int(ptr[hi]);ids=indices[a:b];pos=values[a:b]>0
        p=np.bincount(ids[pos],minlength=h*w)
        m=np.bincount(ids[~pos],minlength=h*w)
        gray=np.full(h*w,128,np.uint8);gray[p>m]=255;gray[m>p]=0
        gray=cv2.cvtColor(gray.reshape(h,w),cv2.COLOR_GRAY2BGR)
        images=[rgb,single,gray,clean,surface,temporal]
        canvas=np.zeros((784,1440,3),np.uint8)
        for j,(label,image) in enumerate(zip(labels,images)):
            r,c=divmod(j,3);x,y=c*480,r*392
            canvas[y+32:y+392,x:x+480]=fit(image)
            cv2.putText(canvas,label,(x+7,y+22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas,f'{(int(end)-int(rgb_ts[0]))/1e9:.1f}s',(372,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        video.stdin.write(canvas.tobytes());gray_video.stdin.write(gray.tobytes())
        stats.append(dict(single_active_pixels=len(selected_i[step]),
            window33_active_pixels=int(np.count_nonzero(p+m)),
            window33_cancelled_pixels=int(np.count_nonzero((p+m>0)&(p==m)))))
        if step in [50,120,250,400,550]:
            cv2.imwrite(str(out/f'comparison_{step//10:02d}s.jpg'),canvas)
        if step%200==0: print(role,step,'/600',flush=True)
    for c in [cap,raw,contour,recent,motion]: c.release()
    for proc in [video,gray_video]:
        proc.stdin.close();assert proc.wait()==0;proc.partial_path.replace(proc.output_path)
    source_count=int(np.searchsorted(ts,int(rgb_ts[0])+60_000_000_000,side='left')-np.searchsorted(ts,rgb_ts[0],side='left'))
    valid=[s for s,end in zip(stats,ends) if end>=ts[0]+33_000_000]
    medians={k:float(np.median([s[k] for s in valid])) for k in valid[0]}
    settings=dict(source=str(source),frames=600,fps=10,seconds=60,labels=labels,
        renderer='tools/visualize_data.py:convert_event_npz_to_mp4 (called unchanged)',
        renderer_sha256=hashlib.sha256((ROOT/'tools/visualize_data.py').read_bytes()).hexdigest(),
        timeline='latest stored event frame timestamp <= nearest RGB timestamp; causal, no future event',
        comparison_adaptation='select 600 CSR rows by corrected timestamps, render those rows at 10fps; native resolution and CRF18',
        original_defaults='all CSR rows, fixed 250fps, default small resolution480x360 CRF28; no timestamp retiming',
        gray_window33='ADAPTED baseline: sign(unit_positive_count - unit_negative_count), background128; not an original accumulation mode',
        source_frames_first60s=source_count,source_frames_at_original250fps_seconds=source_count/250,
        native_pixels=h*w,median_pixels_after_event_warmup=medians,
        spatial_alignment=False,raw_frame_indices=rows.tolist(),rgb_frame_indices=rgb_ids.tolist(),
        frame_statistics=stats)
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    print(role,'completed',medians,'original250fps_duration',source_count/250,flush=True)


if __name__=='__main__':
    cv2.setNumThreads(2)
    deadline=register()
    render('20260720_131207_863008_200_b','body',deadline)
    render('20260720_125338_040324_200_f','face',deadline)

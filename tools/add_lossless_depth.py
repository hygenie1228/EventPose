"""Losslessly pack every uint16 depth frame into H.264 RGB MP4 and verify all pixels."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess
import argparse

import cv2
import h5py
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
NAME='20260720_125338_040324_200_f'
SOURCE=ROOT/'orig_dataset/data1'/NAME/'RGBD'
DEST=ROOT/'etri_processed'/NAME
WORK=ROOT/'exp/261023/lossless_depth'


def read_depth_mp4(path,index):
    """Read one archived depth frame as the original uint16 raw values."""
    cap=cv2.VideoCapture(str(path))
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,p=cap.read();cap.release()
    if not ok:raise IndexError(index)
    return p[:,:,2].astype(np.uint16)+(p[:,:,1].astype(np.uint16)<<8)


def depth_frame(group,index):
    a,b=map(int,group['offsets'][index:index+2])
    d=cv2.imdecode(group['encoded_data'][a:b],cv2.IMREAD_UNCHANGED)
    assert d.dtype==np.uint16
    return d


def export(i):
    device=DEST/f'device{i}';path=device/'depth.mp4';records=[]
    with h5py.File(SOURCE/f'device{i}.h5','r') as f:
        g=f['streams/depth_align'];n=len(g['offsets'])-1
        times=g['timestamp_ns'][:].astype(np.int64)
        fps=1e9/float(np.median(np.diff(times)))
        h,w=int(g.attrs['height']),int(g.attrs['width'])
        command=['ffmpeg','-y','-loglevel','error','-filter_threads','1','-f','rawvideo',
            '-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an',
            '-c:v','libx264rgb','-threads','4','-preset','veryfast','-qp','0','-g','1',
            '-metadata','depth_encoding=uint16: R=low byte,G=high byte,B=0; depth=R+(G<<8)',
            '-metadata','depth_unit=source_raw_unit','-metadata','depth_timestamps_ns='+json.dumps(times.tolist(),separators=(',',':')),
            '-movflags','+faststart+use_metadata_tags',str(path)]
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
        packed=np.zeros((h,w,3),np.uint8)
        try:
            for j in range(n):
                d=depth_frame(g,j)
                packed[:,:,0]=(d&255).astype(np.uint8);packed[:,:,1]=(d>>8).astype(np.uint8)
                records.append(hashlib.sha256(d.astype('<u2',copy=False).tobytes()).hexdigest())
                encoder.stdin.write(memoryview(packed))
                if j%200==0:print(i,'depth encoded',j,'/',n,flush=True)
        finally:
            encoder.stdin.close();code=encoder.wait()
        assert code==0
    decoder=subprocess.Popen(['ffmpeg','-loglevel','error','-threads','4','-filter_threads','1','-i',str(path),
        '-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    try:
        for j,expected in enumerate(records):
            data=decoder.stdout.read(h*w*3);assert len(data)==h*w*3
            p=np.frombuffer(data,np.uint8).reshape(h,w,3)
            d=p[:,:,0].astype(np.uint16)+(p[:,:,1].astype(np.uint16)<<8)
            assert np.count_nonzero(p[:,:,2])==0
            assert hashlib.sha256(d.astype('<u2',copy=False).tobytes()).hexdigest()==expected,(i,j)
            if j%200==0:print(i,'depth pixel-exact verified',j,'/',n,flush=True)
        assert decoder.stdout.read(1)==b''
    finally:
        decoder.stdout.close();code=decoder.wait()
    assert code==0
    report={'device':i,'depth_frames':n,'width':w,'height':h,'dtype':'uint16','lossless_all_frames_verified':True,
        'encoding':'libx264rgb QP=0, R=low8 G=high8 B=0, all intra frames',
        'timestamps_ns':times.tolist(),'frame_sha256':records,'file_bytes':path.stat().st_size}
    (WORK/f'device{i}_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(i,'ALL DEPTH FRAMES VERIFIED',flush=True)
    return report


def preview(i):
    device=DEST/f'device{i}'
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags','-of','json',str(device/'depth.mp4')]))
    depth_times=np.array(json.loads(meta['format']['tags']['depth_timestamps_ns']),np.int64)
    with np.load(device/'event.npz',allow_pickle=False) as z:
        ptr=z['indptr'];idx=z['indices'];val=z['data']
        et=z['timestamp_ns'].astype(np.int64);rt=z['rgb__timestamp_ns'].astype(np.int64)
        h,w=int(z['frame_height']),int(z['frame_width'])
    fps=15;start=max(int(rt[0]),int(et[0]),int(depth_times[0]))
    end=min(start+60_000_000_000,int(rt[-1]),int(et[-1]),int(depth_times[-1]))
    count=min(900,int((end-start)/1e9*fps))
    options=[cv2.CAP_PROP_N_THREADS,2]
    rgb=cv2.VideoCapture(str(device/'rgb.mp4'),cv2.CAP_FFMPEG,options)
    depth=cv2.VideoCapture(str(device/'depth.mp4'),cv2.CAP_FFMPEG,options)
    assert rgb.isOpened() and depth.isOpened()
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s','1152x240','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4','-preset','veryfast',
        '-crf','20','-profile:v','baseline','-level','3.0','-tag:v','avc1',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(device/'preview.mp4')],stdin=subprocess.PIPE)
    canvas=np.zeros((240,1152,3),np.uint8);image=np.full(h*w,128,np.uint8)
    last_d=-1;color=None;mapping=[]
    next_rgb=0;next_depth=0
    def nearest(ts,t):
        pos=int(np.searchsorted(ts,t));return min((j for j in [pos-1,pos] if 0<=j<len(ts)),key=lambda j:abs(int(ts[j])-t))
    try:
        for j in range(count):
            t=start+round(j*1e9/fps);rp=nearest(rt,t);ep=nearest(et,t);dp=nearest(depth_times,t)
            if rp<next_rgb or rp-next_rgb>64:
                rgb.set(cv2.CAP_PROP_POS_FRAMES,rp);next_rgb=rp
            while next_rgb<=rp:
                ok,r=rgb.read();assert ok
                next_rgb+=1
            if dp!=last_d:
                if dp!=next_depth:depth.set(cv2.CAP_PROP_POS_FRAMES,dp)
                ok,p=depth.read();assert ok
                next_depth=dp+1
                # OpenCV returns BGR: original R is channel 2, original G channel 1.
                d=p[:,:,2].astype(np.uint16)+(p[:,:,1].astype(np.uint16)<<8)
                color=cv2.applyColorMap(np.clip(d.astype(np.float32)/5000*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
                color[d==0]=0;last_d=dp
                color=cv2.resize(color,(384,216),interpolation=cv2.INTER_AREA)
            a,b=int(ptr[ep]),int(ptr[ep+1]);image.fill(128);image[idx[a:b]]=np.where(val[a:b]>0,255,0).astype(np.uint8)
            e=cv2.resize(image.reshape(h,w),(320,240),interpolation=cv2.INTER_AREA)
            r=cv2.resize(r,(384,216),interpolation=cv2.INTER_AREA)
            canvas.fill(0);canvas[:,32:352]=e[:,:,None];canvas[12:228,384:768]=r;canvas[12:228,768:1152]=color
            encoder.stdin.write(memoryview(canvas))
            mapping.append([rp,ep,dp])
            if j%150==0:print(i,'preview',j,'/',count,flush=True)
    finally:
        rgb.release();depth.release();encoder.stdin.close();code=encoder.wait()
    assert code==0
    result={'device':i,'frames':count,'fps':fps,'duration_seconds':count/fps,'panels':['Event','RGB','Depth'],
        'resolution':[1152,240],'text_overlay':False,'source_positions_rgb_event_depth':mapping}
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,pix_fmt,nb_frames',
        '-show_entries','format=duration','-of','json',str(device/'preview.mp4')]))
    assert int(probe['streams'][0]['nb_frames'])==count and float(probe['format']['duration'])<=60
    (WORK/f'device{i}_preview_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(i,'PREVIEW VERIFIED',flush=True)


def worker(i):
    record=WORK/f'device{i}_validation.json'
    depth_path=DEST/f'device{i}/depth.mp4'
    skip=False
    if record.is_file() and depth_path.is_file():
        r=json.loads(record.read_text())
        skip=r.get('lossless_all_frames_verified') and depth_path.stat().st_size==r.get('file_bytes')
    if skip:print(i,'reusing previously pixel-verified depth',flush=True)
    else:export(i)
    preview_record=WORK/f'device{i}_preview_validation.json'
    if preview_record.is_file():
        p=DEST/f'device{i}/preview.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames,width,height','-of','json',str(p)]))['streams'][0]
        r=json.loads(preview_record.read_text())
        if int(probe['nb_frames'])==r['frames'] and (probe['width'],probe['height'])==(1152,240):
            print(i,'reusing validated three-panel preview',flush=True);return
    preview(i)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--devices',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(worker,args.devices):pass

"""Event video and text-free Event|RGB preview from compact processed data.

Event reconstruction and INTER_AREA resizing follow tools/visualize_data.py.
"""
import argparse
import json
import subprocess
import time
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
DEVICE=ROOT/'etri_processed/20260720_125338_040324_200_f/device0'
OUT=ROOT/'exp/261023/processed_visualization'


def encoder(path,width,height,fps,gray=False,gpu=0,cpu=False):
    # NVENC is used only to encode H.264; pixel reconstruction matches the reference.
    codec=(['-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-profile:v','baseline','-level','3.0','-tag:v','avc1'] if cpu else
           ['-c:v','h264_nvenc','-gpu',str(gpu),'-preset','fast','-rc','constqp','-qp','18','-bf','0'])
    return subprocess.Popen(['ffmpeg','-y','-loglevel','error','-threads','2',
        '-f','rawvideo','-pix_fmt','gray' if gray else 'bgr24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-an',*codec,'-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


def probe(path,frames,width,height):
    value=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,codec_name,pix_fmt,width,height,avg_frame_rate',
        '-show_entries','format=duration','-of','json',str(path)]))
    stream=value['streams'][0]
    assert int(stream['nb_frames'])==frames
    assert (stream['width'],stream['height'])==(width,height)
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    return value


EVENT_ARRAYS=None


def event_part(bounds):
    number,start,stop,width,height,gpu=bounds
    ptr,indices,values=EVENT_ARRAYS
    path=OUT/f'event_part_{number}.mp4'
    image=np.full(height*width,128,np.uint8)
    process=encoder(path,width,height,250,gray=True,gpu=gpu)
    try:
        for i in range(start,stop):
            a,b=int(ptr[i]),int(ptr[i+1])
            image.fill(128);image[indices[a:b]]=values[a:b]
            process.stdin.write(memoryview(image))
            if (i-start)%20000==0:print('Event segment',number,i-start,'/',stop-start,flush=True)
    finally:
        process.stdin.close();code=process.wait()
    assert code==0
    probe(path,stop-start,width,height)
    return str(path)


def main():
    global DEVICE,OUT,EVENT_ARRAYS
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['event','preview'],required=True)
    parser.add_argument('--session',type=Path,default=DEVICE.parent)
    parser.add_argument('--device',choices=[f'device{i}' for i in range(7)],default='device0')
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    parser.add_argument('--cpu-preview',action='store_true')
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--work-root',type=Path)
    parser.add_argument('--cleanup-parts',action='store_true')
    args=parser.parse_args()
    DEVICE=args.session/args.device
    assert DEVICE.resolve().is_relative_to((ROOT/'etri_processed').resolve())
    OUT=(args.work_root or ROOT/'exp/261023/device_videos')/args.session.name/args.device
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(DEVICE/'event.npz',allow_pickle=False) as z:
        ptr=z['indptr'];indices=z['indices'];data=z['data']
        values=np.where(data>0,255,0).astype(np.uint8);del data
        height,width=int(z['frame_height']),int(z['frame_width'])
        event_times=z['timestamp_ns'].astype(np.int64)
        rgb_times=z['rgb__timestamp_ns'].astype(np.int64)
    image=np.full(height*width,128,np.uint8)
    start=time.monotonic()
    if args.kind=='event':
        count=len(ptr)-1;fps=250  # Same Event playback FPS as visualize_data.py.
        EVENT_ARRAYS=(ptr,indices,values)
        boundaries=np.linspace(0,count,len(args.gpus)+1,dtype=int)
        jobs=[(i,int(boundaries[i]),int(boundaries[i+1]),width,height,gpu) for i,gpu in enumerate(args.gpus)]
        with multiprocessing.get_context('fork').Pool(len(jobs)) as pool:
            parts=pool.map(event_part,jobs)
        concat=OUT/'event_concat.txt'
        concat.write_text(''.join("file '"+p+"'\n" for p in parts))
        path=DEVICE/'event.mp4'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),
            '-c','copy','-movflags','+faststart',str(path)],check=True)
        info=probe(path,count,width,height)
        if args.cleanup_parts:
            for part in parts:
                p=Path(part)
                assert p.parent==OUT and p.name.startswith('event_part_')
                p.unlink()
            concat.unlink()
        info.update({'all_event_frames_included':True,'source_frame_count':count,'text_overlay':False,
            'reference':'tools/visualize_data.py: background=128, positive=255, negative=0; Event FPS=250'})
    else:
        rgb_path=DEVICE/'rgb.mp4'
        if not rgb_path.exists():rgb_path=DEVICE/'video.mp4'
        capture=cv2.VideoCapture(str(rgb_path));assert capture.isOpened()
        fps=float(capture.get(cv2.CAP_PROP_FPS));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count==len(rgb_times)
        full_rgb_count=count
        if args.max_seconds is not None:count=min(count,int(args.max_seconds*fps))
        path=DEVICE/'preview.mp4';process=encoder(path,768,240,fps,gpu=args.gpus[0],cpu=args.cpu_preview)
        canvas=np.zeros((240,768,3),np.uint8)
        try:
            for i in range(count):
                ok,rgb=capture.read();assert ok
                candidate=int(np.searchsorted(event_times,rgb_times[i]))
                ep=min((j for j in (candidate-1,candidate) if 0<=j<len(event_times)),key=lambda j:abs(int(event_times[j])-int(rgb_times[i])))
                a,b=int(ptr[ep]),int(ptr[ep+1]);image.fill(128);image[indices[a:b]]=values[a:b]
                gray=cv2.resize(image.reshape(height,width),(320,240),interpolation=cv2.INTER_AREA)
                scale=min(384/rgb.shape[1],240/rgb.shape[0])
                rw,rh=round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)
                rgb=cv2.resize(rgb,(rw,rh),interpolation=cv2.INTER_AREA)
                x,y=384+(384-rw)//2,(240-rh)//2
                canvas.fill(0);canvas[:,32:352]=gray[...,None];canvas[y:y+rh,x:x+rw]=rgb
                process.stdin.write(memoryview(canvas))
                if i%1000==0:print('Preview',i,'/',count,flush=True)
        finally:
            capture.release();process.stdin.close();code=process.wait()
        assert code==0
        info=probe(path,count,768,240)
        info.update({'all_rgb_frames_included':count==full_rgb_count,'text_overlay':False,
            'panels':['Event grayscale','RGB'],'event_selection':'nearest corrected timestamp for each RGB frame',
            'resize':'INTER_AREA, aspect ratio preserved'})
    (OUT/f'{args.kind}_validation.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Verified',path,flush=True)


if __name__=='__main__':main()

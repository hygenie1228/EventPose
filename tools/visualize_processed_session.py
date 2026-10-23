"""Render all five devices using only compact processed files + SMPL-X assets.

No dataset/ or orig_dataset/ reads. Model assets are loaded from data/.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import smplx
import torch

# Shared rasterizer does not read any dataset; only these two pure functions are used.
from mesh_rendering import render, panel

ROOT=Path(__file__).absolute().parents[1]
OUT=ROOT/'exp/261023/processed_visualization'


def body_model():
    torch.set_num_threads(2)
    return smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',
        gender='neutral',num_betas=16,use_pca=False,flat_hand_mean=False)


def vertices(model,pose,betas,translation):
    p=torch.from_numpy(np.asarray(pose,np.float32))
    with torch.no_grad():
        result=model(betas=torch.from_numpy(np.asarray(betas,np.float32)),
            expression=torch.zeros((len(p),model.num_expression_coeffs),dtype=p.dtype),
            global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],
            leye_pose=p[:,69:72],reye_pose=p[:,72:75],
            left_hand_pose=p[:,75:120],right_hand_pose=p[:,120:165],
            transl=torch.from_numpy(np.asarray(translation,np.float32)))
    return result.vertices.numpy()


def visualize(session,device_name,model):
    device=session/device_name
    output=OUT/f'{session.name}_{device_name}_4panel.mp4'
    with np.load(device/'event.npz',allow_pickle=False) as z:
        pointers=z['indptr'];indices=z['indices'];values=z['data']
        event_times=z['timestamp_ns'].astype(np.int64)
        height,width=map(int,z['image_shape'])
        background=int(z['background_value'])
        def calibration(key): return z['rgb_calibration__'+key.replace('/','__')]
        Krgb=calibration('rgb/intrinsic_matrix')
        Kev=calibration('event/intrinsic_matrix')
        Rworld=calibration('rgb/rotation_matrix')
        tworld=calibration('rgb/translation')
        Revent=calibration('rgb_to_event/rotation_matrix')
        tevent=calibration('rgb_to_event/translation')
    fps,duration=15,10
    capture=cv2.VideoCapture(str(device/'rgb.mp4'))
    assert capture.isOpened()
    output_frames=[]
    writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4',
        '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    try:
        with h5py.File(device/'label.h5','r') as f:
            assert set(f.keys())=={'params','bboxes','frames'}
            times=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
            targets=times[len(times)//2]+(np.arange(fps*duration)*1e9/fps).astype(np.int64)
            rows=np.clip(np.searchsorted(times,targets),0,len(times)-1)
            for i,row in enumerate(rows):
                row=int(row)
                pose=f['params/smplx_pose_axis_angle'][row]
                beta=f['params/smplx_betas'][row]
                translation=f['params/translation_world_m'][row]
                valid=np.isfinite(pose).all(axis=1)&np.isfinite(beta).all(axis=1)&np.isfinite(translation).all(axis=1)
                v=vertices(model,pose[valid],beta[valid],translation[valid])
                v=v@Rworld.T+tworld
                rgb_position=int(f['frames/source_rgb_position'][row])
                capture.set(cv2.CAP_PROP_POS_FRAMES,rgb_position)
                ok,rgb=capture.read()
                assert ok, (device_name,rgb_position)
                # Nearest corrected Event timestamp replaces removed sync tables.
                candidate=int(np.searchsorted(event_times,times[row]))
                choices=[j for j in (candidate-1,candidate) if 0<=j<len(event_times)]
                ep=min(choices,key=lambda j:abs(int(event_times[j])-int(times[row])))
                assert abs(int(event_times[ep])-int(times[row]))<=3_000_000
                a,b=map(int,pointers[ep:ep+2])
                event=np.full(height*width,background,np.uint8)
                # Exact polarity reconstruction from visualize_data.py.
                event[indices[a:b]]=np.where(values[a:b]>0,255,0).astype(np.uint8)
                event=event.reshape(height,width)
                color=np.repeat(event[...,None],3,axis=2)
                small_rgb=cv2.resize(rgb,(640,360),interpolation=cv2.INTER_AREA)
                small_event=cv2.resize(color,(480,360),interpolation=cv2.INTER_AREA)
                K1=Krgb.copy();K1[0]*=640/rgb.shape[1];K1[1]*=360/rgb.shape[0]
                K2=Kev.copy();K2[0]*=480/width;K2[1]*=360/height
                rgb_mesh=render(small_rgb,v,model.faces,K1)
                event_mesh=render(small_event,v@Revent.T+tevent,model.faces,K2)
                frame=np.vstack([np.hstack([panel(small_rgb,'RGB'),panel(rgb_mesh,'RGB + SMPL-X')]),
                    np.hstack([panel(small_event,'Event'),panel(event_mesh,'Event + SMPL-X')])])
                writer.stdin.write(frame.tobytes())
                output_frames.append({'gt_row':row,'rgb_position':rgb_position,'event_position':ep,
                    'event_residual_ns':int(event_times[ep])-int(times[row])})
                if i%30==0:print(device_name,i,'/',len(rows),flush=True)
    finally:
        capture.release();writer.stdin.close();returncode=writer.wait()
    assert returncode==0
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'stream=codec_name,pix_fmt,width,height,nb_frames','-show_entries','format=duration','-of','json',str(output)]))
    assert probe['streams'][0]['nb_frames']==str(fps*duration)
    return {'device':device_name,'file':output.name,'fps':fps,'duration_seconds':duration,
        'inputs':[str(device/k) for k in ('event.npz','rgb.mp4','label.h5')],
        'mesh':'Reconstructed from SMPL-X pose, betas and translation using local neutral model; no stored mesh read',
        'frames':output_frames,'probe':probe}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,default=ROOT/'etri_processed/20260720_125338_040324_200_f')
    parser.add_argument('--wait-ready',action='store_true',help='Wait for compact conversion validation before each device')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    model=body_model();results=[]
    for i in range(5):
        if args.wait_ready:
            deadline=time.monotonic()+3600
            while True:
                ready_file=ROOT/'exp/261023/compact_processed/validation.json'
                try:
                    ready={d['device'] for d in json.loads(ready_file.read_text())}
                except (FileNotFoundError,json.JSONDecodeError):
                    ready=set()
                if f'device{i}' in ready:break
                if time.monotonic()>deadline:raise TimeoutError(f'device{i} conversion not ready')
                time.sleep(2)
        results.append(visualize(args.session,f'device{i}',model))
        (OUT/'manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    print('All five processed-only visualizations verified',flush=True)


if __name__=='__main__':main()

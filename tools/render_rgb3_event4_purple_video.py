"""Render five verified RGB3+Event4 clips with native Event4 and purple meshes."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import cv2
import h5py
from utils.mesh_video_rendering import _triangles
from goal_runtime_control import register_job

def main():
    goal=json.loads((ROOT/'_agents/research/261028/device3_encoding_goal_12h.json').read_text())
    register_job(goal,os.getpid(),'RGB3_Event4_purple_video_CPU')
    gallery=ROOT/'exp/261028/device34_test_video_gallery'
    manifest=json.loads((gallery/'manifest.json').read_text())
    out=ROOT/'exp/261029/device34_compact_video/rgb3_event4_purple';out.mkdir(exist_ok=True)
    work=Path(manifest['work_root']);faces=np.load(gallery/'meshes/rgb3_event4/faces.npy')
    video=out/'Device3_RGB_Device4_Event_PurpleMesh_Clips1to5.mp4'
    partial=video.with_suffix('.partial.mp4')
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x400','-r','30','-i','-','-an','-c:v','libx264','-threads','3','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE)
    # The encoder inherits this registered job's group and exits on closed stdin.
    cv2.setNumThreads(1);count=0;records=[]
    for clip in manifest['clips']:
        name=clip['clip'];n=clip['task']['video_frames']
        with h5py.File(gallery/f'meshes/rgb3_event4/{name}.h5','r') as pred,h5py.File(work/f'references/{name}.h5','r') as ref,h5py.File(out.parent/f'event_cache/{name}.h5','r') as ev:
            assert np.array_equal(pred['timestamp_ns'][:],ref['timestamp_ns'][:])
            assert np.array_equal(ev['timestamp_ns'][:],ref['timestamp_ns'][:])
            assert np.array_equal(ev['source_frames'][:],ref['source_frames'][:])
            offsets=pred['frame_offsets'][:];K=ref['rgb3_K'][:];painted=[]
            for i in range(n):
                rgb=ref['rgb3'][i];event=cv2.cvtColor(ev['event4_rgb'][i],cv2.COLOR_RGB2BGR)
                overlay=rgb.copy();depth=np.full(rgb.shape[:2],np.inf,np.float32)
                vertices=pred['vertices_common_m'][offsets[i]:offsets[i+1]]
                for v in vertices:
                    projected=v@K.T;uv=projected[:,:2]/np.maximum(projected[:,2:],1e-8)
                    t=v[faces];normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
                    shade=.3+.7*np.abs(normal@np.array([.25,-.35,-.902]))
                    # PPT material RGB (178,145,216), converted to OpenCV BGR.
                    colors=np.clip(shade[:,None]*np.array([216,145,178])[None],0,255).astype(np.uint8)
                    _triangles(overlay,depth,uv,v[:,2],faces.astype(np.int32),colors)
                painted.append(int(np.isfinite(depth).sum()))
                canvas=np.full((400,1920,3),20,np.uint8)
                for j,(image,label) in enumerate(zip((rgb,event,overlay),('device3 RGB','device4 Event','Mesh overlay'))):
                    canvas[40:,640*j:640*(j+1)]=image
                    cv2.putText(canvas,label,(640*j+14,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes());count+=1
            records.append(dict(clip=name,frames=n,start_frame=count-n,painted_pixels_min=min(painted),painted_pixels_max=max(painted)))
        print(name,'complete',flush=True)
    encoder.stdin.close();assert encoder.wait()==0;assert count==1505;partial.replace(video)
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(video)]))['streams'][0]
    assert int(stream['nb_read_frames'])==count
    cap=cv2.VideoCapture(str(video));reviews=[]
    for j in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES,j*301+150);ok,image=cap.read();assert ok
        path=out/f'decoded_clip{j+1:02d}.png';cv2.imwrite(str(path),image);reviews.append(str(path))
    cap.release()
    proof=dict(status='full_decode_passed_pending_visual_review',video=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frames=count,duration_seconds=count/30,conditions=['rgb3_event4'],clips=records,stream=stream,reviews=reviews,material_RGB=[178,145,216],mesh_geometry_unchanged=True,GT_pose_alignment=False,event4_spatial_warp=False,event4_display='native 960x720 aspect-preserved to480x360, letterboxed640x360',labels=['device3 RGB','device4 Event','Mesh overlay'],source_checkpoint=manifest['models']['rgb3_event4'])
    (out/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(video,flush=True)

if __name__=='__main__':main()

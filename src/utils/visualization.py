from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Three-panel, metric scene demo movies from saved model predictions."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from utils.mesh_rendering import render

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'exp/261023/human3r_multihmr_test'


def project(points,rotation,eye,K):
    p=(points-eye)@rotation.T
    q=p@K.T
    return q[:,:2]/np.maximum(q[:,2:],1e-6),p[:,2]


def scene_setup(predictions):
    candidates=[]
    for path in predictions[:30]:
        with np.load(path) as z:
            v=z['vertices_cam_m'];t=z['camera_to_world']
            if len(v):candidates.append(v.reshape(-1,3)@t[:3,:3].T+t[:3,3])
    assert candidates,'No predicted humans in first 30 frames'
    v=np.concatenate(candidates)
    center=np.median(v,axis=0)
    floor=float(np.percentile(v[:,1],99.5))
    # A fixed oblique camera makes translation and size changes visible.
    eye=center+np.array([4,-2.0,-5.5])
    forward=center-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,np.array([0,-1,0.]));right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.stack([right,down,forward])
    K=np.array([[570,0,320],[0,570,185],[0,0,1]],float)
    return center,floor,eye,rotation,K


def scene(z,faces,settings,model_name):
    center,floor,eye,rotation,K=settings
    image=np.full((360,640,3),[31,28,26],np.uint8)
    def line(a,b,color,width=1):
        uv,depth=project(np.array([a,b]),rotation,eye,K)
        if np.all(depth>.1):
            xy=np.clip(uv,-10000,10000).astype(int)
            cv2.line(image,tuple(xy[0]),tuple(xy[1]),color,width,cv2.LINE_AA)
    # Ground is a visualization reference estimated from feet, not measured floor.
    cx,cz=np.round(center[[0,2]]).astype(int)
    for x in range(cx-4,cx+5):line([x,floor,cz-4],[x,floor,cz+4],(66,61,56))
    for zz in range(cz-4,cz+5):line([cx-4,floor,zz],[cx+4,floor,zz],(66,61,56))
    c2w=z['camera_to_world']
    if 'points_cam_m' in z:
        p=z['points_cam_m'][::4,::4].reshape(-1,3)
        colors=z['color_rgb'][::4,::4].reshape(-1,3)[:,::-1]
        conf=z['confidence'][::4,::4].reshape(-1)
        p=p@c2w[:3,:3].T+c2w[:3,3]
        keep=np.isfinite(p).all(1)&(conf>1.5)&(np.linalg.norm(p-center,axis=1)<12)
        p=p[keep];colors=colors[keep]
        uv,depth=project(p,rotation,eye,K)
        keep=(depth>.1)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<360)
        uv=uv[keep].astype(int);depth=depth[keep];colors=colors[keep]
        for j in np.argsort(depth)[::-1]:
            u,v=uv[j];cv2.circle(image,(u,v),1,tuple(map(int,colors[j])),-1)
    vertices=z['vertices_cam_m']
    if len(vertices):
        world=vertices@c2w[:3,:3].T+c2w[:3,3]
        novel=(world-eye)@rotation.T
        image=render(image,novel,faces,K,alpha=1.)
    # Input-camera frustum, shown in the same metric coordinate system.
    cam=c2w[:3,3]
    corners=np.array([[-.4,-.225,1],[.4,-.225,1],[.4,.225,1],[-.4,.225,1]])
    corners=corners@c2w[:3,:3].T+cam
    for p in corners:line(cam,p,(230,190,70),2)
    for i in range(4):line(corners[i],corners[(i+1)%4],(230,190,70),1)
    label='Predicted scene + SMPL-X' if model_name=='human3r' else 'SMPL-X + camera (no scene reconstruction)'
    cv2.putText(image,label,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(image,'Grid: 1 m | floor reference from predicted feet',(10,345),
                cv2.FONT_HERSHEY_SIMPLEX,.4,(215,215,215),1,cv2.LINE_AA)
    return image


def panel(image,title):
    result=np.full((400,640,3),20,np.uint8)
    h,w=image.shape[:2];scale=min(640/w,360/h)
    image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    y=40+(360-image.shape[0])//2;x=(640-image.shape[1])//2
    result[y:y+image.shape[0],x:x+image.shape[1]]=image
    cv2.putText(result,title,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
    return result


def run(model_name,limit):
    dest=OUT/model_name;files=sorted((dest/'predictions').glob('*.npz'))
    manifest=json.loads((OUT/'inputs/manifest.json').read_text())
    if limit:files=files[:limit]
    else:assert len(files)==manifest['frames'],(len(files),manifest['frames'])
    faces=np.load(dest/'faces.npy');settings=scene_setup(files)
    output=dest/('preview.mp4' if limit else f'{model_name}_video_overlay_scene.mp4')
    proc=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo',
        '-pix_fmt','bgr24','-s','1920x400','-r',str(manifest['output_fps']),'-i','-',
        '-an','-c:v','libx264','-threads','4','-crf','19','-preset','fast','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    people=[]
    try:
        for i,p in enumerate(files):
            with np.load(p) as z:
                rgb=cv2.imread(str(OUT/'inputs/frames'/f'{i:06d}.jpg'));assert rgb is not None
                v=z['vertices_cam_m'];assert np.isfinite(v).all()
                if model_name=='human3r':
                    h,w=z['color_rgb'].shape[:2]
                    base=cv2.resize(rgb,(w,h));K=z['K']
                else:base=rgb;K=z['source_K']
                overlay=render(base,v,faces,K,alpha=.72)
                third=scene(z,faces,settings,model_name)
                t=manifest['source_times_s'][i]
                result=np.concatenate([panel(rgb,f'Video | {t:05.1f} s'),
                     panel(overlay,f'{model_name} | mesh overlay'),panel(third,'Scene vis | metric coordinates')],axis=1)
                proc.stdin.write(result.tobytes());people.append(len(v))
                if i in [0,len(files)//2,len(files)-1]:cv2.imwrite(str(dest/f'contact_{i:06d}.jpg'),result)
                if i%25==0:print(model_name,'render',i+1,'/',len(files),flush=True)
    finally:
        proc.stdin.close()
        rc=proc.wait()
    assert rc==0,rc
    summary=dict(model=model_name,frames=len(files),output=str(output),fps=5,
                 min_people=min(people),max_people=max(people),empty_frames=people.count(0),
                 scene_floor='reference estimated from predicted feet, not GT',
                 coordinate_frame='model reconstruction world' if model_name=='human3r' else 'input camera',
                 scene_geometry='predicted point map' if model_name=='human3r' else 'none; mesh/grid/camera only')
    (dest/('preview_summary.json' if limit else 'summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['human3r','multi_hmr'])
    parser.add_argument('--limit',type=int,default=0);args=parser.parse_args();run(args.model,args.limit)

"""CPU-only actual current-checkpoint mesh and reference-style event previews."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job,write

def main():
    g=json.loads((ROOT/'_agents/research/261028/device3_encoding_goal_12h.json').read_text());register_job(g,os.getpid(),'editable_stereo_actual_assets_CPU')
    os.environ.update(EVENTPOSE_MULTISENSOR_OUT=g['output_root'],CUDA_VISIBLE_DEVICES='',XFORMERS_DISABLED='1')
    import numpy as np
    import torch
    import h5py
    import cv2
    from PIL import Image
    from utils.event_visualization import load_window,render
    out=Path(g['output_root'])/'encoding_explanation/editable';out.mkdir(parents=True,exist_ok=True)
    ref=json.loads((out.parent/'설명_그림_검증자료.json').read_text());name=ref['session'];frame=ref['source_frame'];stamp=ref['timestamp_ns']
    if (out/'dense_example_selection.json').exists():
        selected=json.loads((out/'dense_example_selection.json').read_text());name=selected['session'];frame=selected['source_frame'];stamp=selected['timestamp_ns']
    with h5py.File(Path(g['output_root'])/'cache'/(name+'.h5'),'r',locking=False) as f:
        i=int(np.searchsorted(f['frames'][:],frame));assert int(f['frames'][i])==frame and int(f['timestamp_ns'][i])==stamp
        rgb=f['rgb3'][i].transpose(1,2,0)
    Image.fromarray(rgb).save(out/'rgb3_input.png')
    event_proof=[]
    for device in (3,4):
        source=ROOT/'dataset'/name/('device'+str(device))/'event.npz'
        window=load_window(source,stamp-200_000_000,stamp)
        for encoding in ('count33','bins100','surface20'):
            Image.fromarray(render(window,stamp,encoding,supported=True)).save(out/f'event{device}_{encoding}_display.png')
        event_proof.append(dict(device=device,native_shape=list(window['shape']),source=str(source.resolve()),source_timestamp_ns=int(stamp),
            latest_past_timestamp_ns=int(window['times'][-1])))
        del window
    if '--events-only' in sys.argv:
        proof=json.loads((out/'actual_assets_provenance.json').read_text());proof.update(event_source=event_proof,
            event_visualization_signed_timestamp_verified=True,display_support_filter=True,
            display_timestamp_fix='Cast raw uint64 timestamps to int64 before subtraction; causal-age assertion passed; no training encoding change')
        write(out/'actual_assets_provenance.json',proof);print('Corrected signed-time event displays ready',flush=True);return
    torch.set_num_threads(3)
    from utils.pose_evaluation import load_multihmr
    from multihmr.multisensor_repair import build_repair_model,load_verified_initialization,REPAIR_LAYOUTS
    from multihmr.multisensor_inference import CachedSensors
    cfg=json.loads((Path(g['output_root'])/'configs/rgb3_event4_surface20.json').read_text())
    checkpoint=Path(g['output_root'])/'checkpoints/rgb3_event4_surface20/best.pt'
    assert json.loads((Path(g['output_root'])/'runs/rgb3_event4_surface20/complete.json').read_text())['status']=='complete'
    torch.manual_seed(cfg['seed']);model=build_repair_model(load_multihmr(device='cpu'),cfg).eval();load_verified_initialization(model,cfg)
    saved=torch.load(checkpoint,map_location='cpu');assert saved['config']==cfg and not saved['test_used_for_selection']
    assert saved['metrics_apply_to_checkpoint'] and saved['metrics_step']==saved['step'];model.load_state_dict(saved['model_delta'],strict=False);step=saved['step'];del saved
    source=CachedSensors(name,cfg['condition'],sensor_layout=REPAIR_LAYOUTS[cfg['condition']],encoding=cfg['encoding'])
    sample=source.sample(frame);assert sample is not None and not set(sample)&{'joints','centers','betas','valid','depth'}
    with torch.inference_mode():people=model(**{k:v[None] for k,v in sample.items()},is_training=False,det_thresh=.3,nms_kernel_size=3)
    assert people;person=max(people,key=lambda p:float(p['scores']));v=person['v3d'].numpy();v=(v-source.t)@source.R
    faces=np.asarray(next(iter(model.base.smpl_layer.values())).bm_x.faces,np.int32);source.close();assert v.shape==(10475,3) and np.isfinite(v).all()
    np.savez_compressed(out/'actual_prediction_mesh.npz',vertices_common_camera_m=v,faces=faces)
    # CPU z-buffer, prediction-centred display camera; no GT pose or alignment.
    from utils.mesh_video_rendering import _triangles
    height,width=1000,760;points=v.astype(np.float64).copy();center=(points.min(0)+points.max(0))/2;points-=center;points[:,2]+=3.
    span=np.ptp(points,axis=0);focal=.72*min(width*points[:,2].min()/span[0],height*points[:,2].min()/span[1]);K=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
    hom=points@K.T;uv=hom[:,:2]/hom[:,2:];tri=points[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
    shade=.3+.7*np.abs(normal@np.array([.25,-.35,-.902]));colors=np.clip(shade[:,None]*np.array([178,145,216])[None],0,255).astype(np.uint8)
    image=np.full((height,width,3),255,np.uint8);depth=np.full((height,width),np.inf,np.float32);_triangles(image,depth,uv,points[:,2],faces,colors)
    mask=np.isfinite(depth);assert mask.sum()>10000;y,x=np.where(mask);rgba=np.dstack([image,mask.astype(np.uint8)*255]);pad=16
    rgba=rgba[max(0,y.min()-pad):min(height,y.max()+pad+1),max(0,x.min()-pad):min(width,x.max()+pad+1)];Image.fromarray(rgba).save(out/'actual_smplx_mesh.png')
    digest=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    write(out/'actual_assets_provenance.json',dict(status='passed',checkpoint=str(checkpoint),checkpoint_sha256=digest.hexdigest(),checkpoint_step=step,
        condition=cfg['condition'],encoding=cfg['encoding'],session=name,source_frame=frame,timestamp_ns=stamp,device='CPU float32',
        actual_native_inference=True,GT_used_as_inference_input=False,vertices=len(v),faces=len(faces),detections=len(people),
        display_camera='centred on predicted mesh only; native orientation retained; no GT alignment',event_source=event_proof,
        event_style_reference=str(ROOT/'_agents/report/261024_인체윤곽과_자세변화를_위한_이벤트_인코딩_비교.md'),
        event_display='Black background; split polarity count sqrt(log1p(N)/log9); orange-red positive/cyan negative; surface tau20; bins old red/middle green/recent blue; display only, training unchanged',
        display_support_filter=True,event_visualization_signed_timestamp_verified=True,
        sample_selection=json.loads((out/'dense_example_selection.json').read_text()) if (out/'dense_example_selection.json').exists() else None,
        experiment_goal_complete=False))
    print(out,flush=True)

if __name__=='__main__':main()

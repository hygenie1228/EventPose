"""Check native single-frame FP32/FP16 inference against saved batched predictions."""
from pathlib import Path
import json,sys,time
import cv2,numpy as np,torch
import torchvision.transforms as T
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json
from multihmr.target_crop import prepare_target_crop,target_acceptance
from utils.pose_evaluation import configure_torch,load_multihmr
configure_torch();torch.set_num_threads(2);model=load_multihmr();reg=model.smpl_layer[f'neutral_{model.num_betas}'].bm_x.J_regressor
norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);manifest=json.loads((OUTPUT/'manifest.json').read_text());results={};started=time.time()
for sample in ['baseball_bat-rotate-05','toy_stroller-back-forth-ground-04','spray_bottle-left-desk-ground-02']:
 record=manifest['samples'][sample];folder=OUTPUT/'predictions/original'/sample
 with np.load(folder/'smplx_predictions.npz') as z:old=z['joints_cam_m'][:,:22]
 cap=cv2.VideoCapture(str(SOURCE/sample/'video.mp4'));masks=cv2.VideoCapture(str(SOURCE/sample/'processed/human_mask.mp4'));rows=[]
 for frame in [0,record['frames']//2,record['frames']-1]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame);masks.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();ok2,mask=masks.read();assert ok and ok2
  K=np.asarray(record['K'],np.float32);image,camera,support,_=prepare_target_crop(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),mask,K)
  tensor=norm(image)[None].cuda();cam=torch.tensor(camera)[None].cuda();modes={}
  for label,half in [('single_FP32',False),('single_FP16',True)]:
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=half):people=model(tensor,K=cam,is_training=False,det_thresh=.3,nms_kernel_size=3)
   candidates=[]
   for person in people:
    j=(reg@person['v3d'].float()).detach().cpu().numpy()[:22];accepted,overlap,torso=target_acceptance(j,K,support)
    if accepted:candidates.append(dict(joints=j,score=float(person['scores']),rank=4*overlap+np.log(max(float(person['scores']),1e-6))))
   if candidates:
    chosen=max(candidates,key=lambda x:x['rank']);closest=min(candidates,key=lambda x:np.linalg.norm(x['joints']-old[frame],axis=1).mean())
    modes[label]=dict(accepted_candidates=len(candidates),selected_root_cam_m=chosen['joints'][0].tolist(),
     selected_mean_joint_difference_to_saved_cm=float(np.linalg.norm(chosen['joints']-old[frame],axis=1).mean()*100),
     closest_mean_joint_difference_to_saved_cm=float(np.linalg.norm(closest['joints']-old[frame],axis=1).mean()*100))
   else:modes[label]=dict(accepted_candidates=0)
  row=dict(frame=frame,saved_root_cam_m=old[frame,0].tolist(),modes=modes);rows.append(row);print(sample,json.dumps(row),flush=True)
 cap.release();masks.release();results[sample]=rows
 save_json(OUTPUT/'original_inference_contract_checks.json',dict(samples=results,GT3D_read=False,weights_unchanged=True,gpu=6,elapsed_seconds=time.time()-started))

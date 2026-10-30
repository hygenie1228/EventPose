"""Inspect target candidates without GT3D input on the ambiguous stroller frame."""
from pathlib import Path
import sys,json,cv2,numpy as np,torch
import torchvision.transforms as T
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json
from multihmr.target_crop import prepare_target_crop,target_acceptance
from utils.pose_evaluation import configure_torch,load_multihmr
from visualize_track2_validation import draw
configure_torch();torch.set_num_threads(2);sample='toy_stroller-back-forth-ground-04';record=json.loads((OUTPUT/'manifest.json').read_text())['samples'][sample];frame=record['frames']//2
cap=cv2.VideoCapture(str(SOURCE/sample/'video.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release();m=cv2.VideoCapture(str(SOURCE/sample/'processed/human_mask.mp4'));m.set(cv2.CAP_PROP_POS_FRAMES,frame);ok2,mask=m.read();m.release();assert ok and ok2
K=np.asarray(record['K'],np.float32);image,camera,support,info=prepare_target_crop(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),mask,K)
cv2.imwrite(str(OUTPUT/'target_foreground_input.jpg'),cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
model=load_multihmr();reg=model.smpl_layer[f'neutral_{model.num_betas}'].bm_x.J_regressor
norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
with torch.no_grad():people=model(norm(image)[None].cuda(),K=torch.tensor(camera)[None].cuda(),is_training=False,det_thresh=.3,nms_kernel_size=3)
panels=[];rows=[]
for index,person in enumerate(people):
 j=(reg@person['v3d'].float()).detach().cpu().numpy()[:22];accepted,overlap,torso=target_acceptance(j,K,support)
 loc=person['loc'].detach().float().cpu().numpy();xy=(loc-np.asarray(info['padding']))/np.asarray(info['scale'])+np.asarray(info['box'][:2]);x,y=np.rint(xy).astype(int)
 head_inside=bool(0<=x<support.shape[1] and 0<=y<support.shape[0] and support[y,x]);row=dict(index=index,score=float(person['scores']),body_gate=accepted,head_in_mask=head_inside,body_overlap=overlap,torso_overlap=torso,head_pixel=xy.tolist(),root_camera_m=j[0].tolist(),body_rank=4*overlap+np.log(max(float(person['scores']),1e-6)));rows.append(row)
 if accepted or head_inside:
  scale=640/record['width'];p=cv2.resize(bgr,(640,round(record['height']*scale)));kk=K.copy();kk[:2]*=scale;draw(p,j,kk,(30,30,245));cv2.circle(p,tuple(np.rint(xy*scale).astype(int)),7,(0,255,0),2)
  p=cv2.copyMakeBorder(p,40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(p,f'#{index} conf{row["score"]:.3f} headMask{head_inside} overlap{overlap:.2f}',(5,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);panels.append(p)
print(json.dumps(rows,indent=2),flush=True)
save_json(OUTPUT/'target_candidate_diagnostic.json',dict(sample=sample,frame=frame,candidates=rows,GT3D_read=False))
if panels:
 while len(panels)%2:panels.append(np.full_like(panels[0],255))
 cv2.imwrite(str(OUTPUT/'target_candidate_diagnostic.jpg'),np.concatenate([np.concatenate(panels[i:i+2],axis=1) for i in range(0,len(panels),2)],axis=0))

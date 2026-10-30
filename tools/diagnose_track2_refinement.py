"""Prediction-only conversion refinement experiment on a Train holdout clip."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch,smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,ASSETS,decode_mhr,mapping,transfer,save_json
from utils import v2d_official_metric as metric

torch.set_num_threads(2)
folder=OUTPUT/'predictions/original/black_platform-turn-05'
with np.load(folder/'human_params_mhr.npz') as z:
 pose=z['pose'];shape=z['shape'];scales=z['scales'];errors=z['per_frame_vertex_error_mm']
frames=np.unique(np.r_[0,np.linspace(0,len(pose)-1,8,dtype=int),np.argsort(errors)[-5:]])
lbfgs_mode='--lbfgs' in sys.argv
lm_fast='--lm-fast' in sys.argv
lm_mode='--lm' in sys.argv or lm_fast
if lbfgs_mode:
 with np.load(OUTPUT/'conversion_refinement_diagnostic.npz') as z:pose[frames]=z['pose']
body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
ids,weights=mapping('smplx2mhr',body.faces,'cuda')
with np.load(folder/'smplx_predictions.npz') as z:
 target=transfer(torch.tensor(z['vertices_cam_m'][frames],device='cuda'),ids,weights).detach()
model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cuda').eval()
for p in model.parameters():p.requires_grad_(False)
p=torch.tensor(pose[frames],device='cuda',requires_grad=True);s=torch.tensor(shape,device='cuda');sc=torch.tensor(scales,device='cuda')
optimizer=torch.optim.Adam([p],lr=.01);best=p.detach().clone();best_error=torch.full((len(p),),float('inf'),device='cuda');history=[];started=time.time()
for step in range(1 if lbfgs_mode or lm_mode else 601):
 optimizer.zero_grad(set_to_none=True);v,_=decode_mhr(model,p,s,sc);squared=(v-target).square().mean((1,2))
 with torch.no_grad():
  better=squared<best_error;best[better]=p[better];best_error[better]=squared[better]
 if step%100==0:
  err=(v-target).norm(dim=-1).mean(1)*1000;item=dict(step=step,mean_fit_mm=err.mean().item(),max_fit_mm=err.max().item(),elapsed_seconds=time.time()-started);history.append(item);print(json.dumps(item),flush=True)
 if step==600 or lbfgs_mode or lm_mode:break
 loss=squared.mean();loss.backward();torch.nn.utils.clip_grad_norm_([p],1.)
 for group in optimizer.param_groups:group['lr']=.01 if step<300 else .003
 optimizer.step()
if lbfgs_mode:
 optimizer=torch.optim.LBFGS([p],lr=1,max_iter=200,history_size=20,tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
 calls=[0]
 def closure():
  optimizer.zero_grad(set_to_none=True);v,_=decode_mhr(model,p,s,sc);squared=(v-target).square().mean((1,2));loss=squared.mean()
  with torch.no_grad():
   better=squared<best_error;best[better]=p[better];best_error[better]=squared[better]
  loss.backward();calls[0]+=1
  if calls[0]%25==0:print(json.dumps(dict(optimizer='LBFGS',calls=calls[0],mean_fit_mm=(v-target).norm(dim=-1).mean().item()*1000)),flush=True)
  return loss
 optimizer.step(closure)
if lm_mode:
 from utils import mhr_reference_fitting as reference
 wrapped=reference.MHR(str(ASSETS/'mhr/mhr_model.pt'),'cuda:0',chunk=256 if lm_fast else 64,precision='float32')
 identity=torch.cat([sc,s]).double()[None]
 solved,fit=reference.lm_pose(wrapped,target.double().reshape(len(target),-1)*1000,p.detach().double(),identity,iters=40 if lm_fast else 20,frames_per_batch=8 if lm_fast else 2)
 with torch.no_grad():
  solved=solved.float();candidate,_=decode_mhr(model,solved,s,sc);squared=(candidate-target).square().mean((1,2));better=squared<best_error;best[better]=solved[better]
 print(json.dumps(dict(optimizer='reference_LM20',mean_fit_mm=fit.mean().item(),elapsed_seconds=time.time()-started)),flush=True)
with torch.no_grad():v,_=decode_mhr(model,best,s,sc);err=(v-target).norm(dim=-1).mean(1).cpu().numpy()*1000
roles=json.loads((OUTPUT/'body_roles.json').read_text())['roles'];a=roles['alignment'];b=roles['body']
v=v.cpu().numpy().astype(np.float64);target=target.cpu().numpy().astype(np.float64)
aligned=metric.apply_similarity_transform(v[:,b],metric.fit_similarity_transform(v[0,a],target[0,a]))
cd=float(metric.symmetric_chamfer_distance(aligned,target[:,b]).mean()*100)
report=dict(Train_validation_sample='black_platform-turn-05',trial='original',frames=frames.tolist(),baseline_fit_mm=errors[frames].tolist(),refined_fit_mm=err.tolist(),history=history,GT_used=False,test_used=False,conversion_only_body_chamfer_frame0_aligned_cm=cd,elapsed_seconds=time.time()-started)
report['optimizer']='reference_LM40 batch8/chunk256 from v1' if lm_fast else 'reference_LM20 from v1' if lm_mode else 'LBFGS200 after Adam600' if lbfgs_mode else 'Adam600'
prefix='conversion_lmfast_diagnostic' if lm_fast else 'conversion_lm_diagnostic' if lm_mode else 'conversion_lbfgs_diagnostic' if lbfgs_mode else 'conversion_refinement_diagnostic'
save_json(OUTPUT/(prefix+'.json'),report)
np.savez_compressed(OUTPUT/(prefix+'.npz'),pose=best.cpu().numpy(),frames=frames,shape=shape,scales=scales)
print(json.dumps(report),flush=True)

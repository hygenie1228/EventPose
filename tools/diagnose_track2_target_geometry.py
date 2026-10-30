"""Independent CPU decomposition of target-only model versus MHR fitting error."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np,torch,smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,decode_mhr,mapping,transfer,save_json,sha256
from multihmr.target_crop import BODY_PROTOCOLS
from utils import v2d_official_metric as official

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trial',default='original')
parser.add_argument('--input-view',choices=['target_foreground','native_rgb','native_rgb_static_crop'],default='target_foreground')
parser.add_argument('--body-target',action='store_true',help='Validate the frozen body-v3 protocol without changing predictions')
parser.add_argument('--motion-diagnostics',action='store_true');args=parser.parse_args()
PROTOCOL='native_rgb_target_head_v1' if args.input_view=='native_rgb' else 'target_foreground_head_v1'
if args.body_target:
 assert args.input_view in BODY_PROTOCOLS
 PROTOCOL=BODY_PROTOCOLS[args.input_view]
elif args.input_view=='native_rgb_static_crop':
 raise ValueError('Static-crop diagnosis requires --body-target; do not silently accept a different protocol')
TRIAL=args.trial
DESTINATION=OUTPUT/'target_geometry_diagnostic.json' if TRIAL=='original' else OUTPUT/'selected_geometry_diagnostics'/(TRIAL+'.json')
torch.set_num_threads(2);manifest=json.loads((OUTPUT/'manifest.json').read_text());model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10);all_ids,all_weights=mapping('smplx2mhr',body.faces,'cpu')
results={};started=time.time();root_index=official.MHR_TABLE3_BODY_JOINT_INDICES[0]

def decode(path,selected):
 with np.load(path) as z:pose=z['pose'].astype(np.float32);shape=z['shape'].astype(np.float32);scales=z['scales'].astype(np.float32)
 points=[];roots=[]
 with torch.no_grad():
  for start in range(0,len(pose),8):
   v,j=decode_mhr(model,torch.tensor(pose[start:start+8]),torch.tensor(shape),torch.tensor(scales));points.append(v[:,selected].numpy());roots.append(j[:,root_index].numpy())
 return np.concatenate(points).astype(np.float64),np.concatenate(roots).astype(np.float64)

def score(pred,target):
 transform=official.fit_similarity_transform(pred[0,:27],target[0,:27]);aligned=official.apply_similarity_transform(pred[:,27:],transform)
 return float(official.symmetric_chamfer_distance(aligned,target[:,27:]).mean()*100),transform

provenance={}
for sample in manifest['test']:
 folder=OUTPUT/'predictions'/TRIAL/sample;metric=json.loads((folder/'metrics.json').read_text());info=json.loads((folder/'inference.json').read_text());assert metric['evaluation_protocol']==info['evaluation_protocol']==PROTOCOL;assert info['target_foreground_only_input']==(args.input_view=='target_foreground') and info['all_accepted_heads_in_target_mask'];roles=metric['protocol']['roles'];selected=roles['alignment']+roles['body']
 if args.body_target:
  assert info['target_body_overlap'] and info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
  assert info['body_box_rule_sha256']==sha256(OUTPUT/'body_target_gate_Train_rule.json')
 files=dict(raw=folder/'smplx_predictions.npz',native=folder/'human_params_mhr_v2.npz',GT=SOURCE/sample/'human/human_params_gt.npz',init=SOURCE/sample/'human/human_params_init.npz')
 before={key:sha256(path) for key,path in files.items()}
 provenance[sample]=dict(input_sha256=before,checkpoint_sha256=info['checkpoint_sha256'],evaluation_protocol=PROTOCOL)
 target,target_roots=decode(SOURCE/sample/'human/human_params_gt.npz',selected)
 converted,roots=decode(folder/'human_params_mhr_v2.npz',selected)
 init,init_roots=decode(SOURCE/sample/'human/human_params_init.npz',selected)
 ids=all_ids[selected];weights=all_weights[selected]
 with np.load(folder/'smplx_predictions.npz') as z:vertices=z['vertices_cam_m']
 mapped=[]
 with torch.no_grad():
  for start in range(0,len(vertices),16):mapped.append(transfer(torch.tensor(vertices[start:start+16]),ids,weights).numpy())
 raw=np.concatenate(mapped).astype(np.float64)
 assert len(raw)==len(target)==len(converted)==len(init)==metric['frames']==manifest['samples'][sample]['frames']
 native_cd,transform=score(converted,target);raw_cd,_=score(raw,target);init_cd,init_transform=score(init,target)
 assert abs(native_cd-metric['CD_H_cm'])<.02,(sample,native_cd,metric['CD_H_cm'])
 chosen=[0,len(roots)//2,len(roots)-1];aligned_roots=official.apply_similarity_transform(roots,transform);aligned_init=official.apply_similarity_transform(init_roots,init_transform)
 results[sample]=dict(frames=len(roots),raw_mapped_SMPLX_CD_proxy_cm=raw_cd,converted_MHR_CD_H_cm=native_cd,init_v6_CD_H_cm=init_cd,
  sampled_frames=chosen,original_frame0_aligned_root_errors_cm=(np.linalg.norm(aligned_roots[chosen]-target_roots[chosen],axis=1)*100).tolist(),
  init_frame0_aligned_root_errors_cm=(np.linalg.norm(aligned_init[chosen]-target_roots[chosen],axis=1)*100).tolist(),
  first_frame_sim3_scale=float(transform[0]),official_score_reproduced_on_CPU=True)
 if args.motion_diagnostics:
  conditions={}
  for label,pred,fixed in [('original_MHR',converted,transform),('init_v6_MHR',init,init_transform)]:
   aligned=official.apply_similarity_transform(pred,fixed);shift=target[:,:27].mean(1)-aligned[:,:27].mean(1)
   centered=aligned+shift[:,None]
   per_frame=np.stack([official.apply_similarity_transform(pred[i],official.fit_similarity_transform(pred[i,:27],target[i,:27])) for i in range(len(pred))])
   conditions[label]=dict(center_corrected_CD_diagnostic_cm=float(official.symmetric_chamfer_distance(centered[:,27:],target[:,27:]).mean()*100),
    per_frame_Sim3_CD_diagnostic_cm=float(official.symmetric_chamfer_distance(per_frame[:,27:],target[:,27:]).mean()*100),
    centroid_trajectory_RMSE_cm=float(np.sqrt(np.mean(np.sum(shift**2,axis=1)))*100))
  results[sample]['motion_diagnostics']=conditions
 assert before=={key:sha256(path) for key,path in files.items()},'Diagnostic inputs changed during CPU evaluation'
 save_json(DESTINATION,dict(samples=results,provenance=provenance,evaluation_protocol=PROTOCOL,full_frame_scope=True,model=TRIAL,GT_read_for_evaluation_only=True,gpu_used=False,test_used_for_selection=False,
  units='cm; raw mapped SMPLX is a mesh proxy, converted MHR is the requested metric',elapsed_seconds=time.time()-started))
 print(sample,'raw SMPLX',round(raw_cd,3),'MHR',round(native_cd,3),'init',round(init_cd,3),flush=True)
summary={key:float(np.mean([v[key] for v in results.values()])) for key in ['raw_mapped_SMPLX_CD_proxy_cm','converted_MHR_CD_H_cm','init_v6_CD_H_cm']}
motion_summary={label:{key:float(np.mean([v['motion_diagnostics'][label][key] for v in results.values()])) for key in next(iter(results.values()))['motion_diagnostics'][label]} for label in ['original_MHR','init_v6_MHR']} if args.motion_diagnostics else None
save_json(DESTINATION,dict(samples=results,provenance=provenance,summary=summary,motion_summary=motion_summary,evaluation_protocol=PROTOCOL,full_frame_scope=True,model=TRIAL,GT_read_for_evaluation_only=True,gpu_used=False,test_used_for_selection=False,elapsed_seconds=time.time()-started,
 diagnostic_caveat='Per-frame corrections are counterfactual diagnostics, not official CD-H, additive components, or a basis for model selection.'))
print(json.dumps(summary),flush=True)

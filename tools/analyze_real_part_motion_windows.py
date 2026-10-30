"""GT-only diagnostic of motion-budget windows on the rendered real examples."""
from pathlib import Path
import json
import numpy as np
import h5py
import torch,smplx
from idea_event_motion_probe import joint_positions
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/261028/dynamic_aggregation_real_examples'
torch.set_num_threads(2)
meta=json.loads((OUT/'실제예시_메타데이터.json').read_text())
model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',num_betas=16,use_pca=False)
result=[]
for r in meta:
 end=r['timestamp_ns'];src=Path(r['source'])
 with h5py.File(src/'label.h5','r',locking=False) as f:
  times=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
  selected=np.flatnonzero((times>=end-600_000_000)&(times<=end)&f['params/person_valid'][:,0])
  assert len(selected)>5
  p=f['params/smplx_pose_axis_angle'][selected,0];b=f['params/smplx_betas'][selected,0];tr=f['params/translation_world_m'][selected,0];t=times[selected]
 js=joint_positions(model,p,b,tr)
 with np.load(src/'event.npz') as z:
  R=z['rgb_calibration__event__rotation_matrix'];trans=z['rgb_calibration__event__translation'];K=z['rgb_calibration__event__intrinsic_matrix']
 pts=js@R.T+trans;pr=pts@K.T;uv=pr[...,:2]/pr[...,2:]
 groups=[('양팔·손목 전체',[18,19,20,21]),('양쪽 다리 전체',[1,2,4,5,7,8,10,11])] if r['kind']=='전신' else [('머리 주변',[15]),('가슴 주변',[9])]
 rr=dict(session=r['session'],kind=r['kind'],timestamp_ns=end,gt_latest_age_ms=(end-int(t[-1]))/1e6,gt_samples=len(t),groups=[])
 for label,ids in groups:
  coords=uv[:,ids]
  displacement=[]
  for ms in [10,33,100,200]:
   start=end-ms*1_000_000
   loc=np.searchsorted(t,start,side='right');points=[]
   if loc>0 and loc<len(t):
    w=(start-int(t[loc-1]))/(int(t[loc])-int(t[loc-1]));points.append(coords[loc-1]*(1-w)+coords[loc]*w)
   elif loc==0:points.append(coords[0])
   else:points.append(coords[-1])
   points.extend(coords[loc:]);points.append(coords[-1])
   distance=float(np.linalg.norm(np.diff(np.array(points),axis=0),axis=-1).sum(0).mean())
   displacement.append(distance)
  allowed=[ms for ms,d in zip([10,33,100,200],displacement) if d<=5]
  chosen=max(allowed) if allowed else 10
  rr['groups'].append(dict(region=label,joints=ids,path_length_px=displacement,windows_ms=[10,33,100,200],motion_budget_px=5,selected_window_ms=chosen))
 result.append(rr)
record=dict(protocol='GT-only illustration, not a learned selector or a Test performance result; mean joint image-plane path length, linear interpolation within available past GT timestamps, hold latest GT until query time.',motion_budget_px=5,threshold_status='Illustrative fixed budget, not optimized or validated; native 960x720 coordinates.',rows=result)
(OUT/'부위별_이동량.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2))

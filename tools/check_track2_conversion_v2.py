"""Independent CPU validation of the full 700-frame v2 Train pilot."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch,smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,ASSETS,decode_mhr,mapping,transfer,save_json,sha256

torch.set_num_threads(2);folder=OUTPUT/'predictions/original/black_platform-turn-05'
receipt=json.loads((folder/'conversion_v2.json').read_text())
with np.load(folder/'human_params_mhr.npz') as z:old={key:z[key] for key in ['pose','shape','scales','per_frame_vertex_error_mm']}
with np.load(folder/'human_params_mhr_v2.npz') as z:new={key:z[key] for key in ['pose','shape','scales','per_frame_vertex_error_mm']}
assert old['pose'].shape==new['pose'].shape==(700,136)
assert np.array_equal(old['shape'],new['shape']) and np.array_equal(old['scales'],new['scales'])
selected=old['per_frame_vertex_error_mm']>20
assert np.array_equal(old['pose'][~selected],new['pose'][~selected])
assert all(np.isfinite(value).all() for value in new.values())
assert sha256(folder/'human_params_mhr.npz')==receipt['initial_sha256']
assert sha256(folder/'smplx_predictions.npz')==receipt['source_sha256']
with np.load(folder/'smplx_predictions.npz') as z:vertices=z['vertices_cam_m']
body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
ids,weights=mapping('smplx2mhr',body.faces,'cpu');model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
before=[];after=[];norm=[];started=time.time()
with torch.no_grad():
 for start in range(0,700,8):
  target=transfer(torch.tensor(vertices[start:start+8]),ids,weights)
  values=[]
  for arrays in [old,new]:
   v,_=decode_mhr(model,torch.tensor(arrays['pose'][start:start+8]),torch.tensor(arrays['shape']),torch.tensor(arrays['scales']))
   values.append(v)
  before.extend((values[0]-target).square().mean((1,2)).numpy())
  after.extend((values[1]-target).square().mean((1,2)).numpy())
  norm.extend((values[1]-target).norm(dim=-1).mean(1).numpy()*1000)
before=np.array(before);after=np.array(after);norm=np.array(norm)
assert np.all(after<=before+2e-9)
assert np.allclose(norm,new['per_frame_vertex_error_mm'],atol=.05,rtol=1e-4)
save_json(OUTPUT/'conversion_v2_pilot_independent_audit.json',dict(frames=700,selected_frames=int(selected.sum()),
 all_frame_MSE_nonincreasing=True,unselected_poses_exactly_unchanged=True,shape_and_scales_exactly_unchanged=True,
 independently_decoded_errors_match=True,max_error_receipt_discrepancy_mm=float(np.max(np.abs(norm-new['per_frame_vertex_error_mm']))),
 before_mean_fit_mm=float(old['per_frame_vertex_error_mm'].mean()),after_mean_fit_mm=float(norm.mean()),
 finite_native_MHR_arrays=True,source_hashes_preserved=True,GT_read=False,gpu_used=False,
 elapsed_seconds=time.time()-started,checked_unix=time.time()))
print('Independent CPU audit passed: 700 frames, every-frame MSE nonincreasing',flush=True)

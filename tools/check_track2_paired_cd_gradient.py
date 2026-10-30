"""Check paired CD-H values and full anchor derivatives independently on CPU."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json,sha256
from utils.track2_metrics import paired_frame0_chamfer
from utils import v2d_official_metric as official

torch.set_num_threads(1);torch.manual_seed(261025)
target=torch.randn(4,91,3,dtype=torch.float64)
prediction=(target+torch.randn_like(target)*.13).requires_grad_()
loss=paired_frame0_chamfer(prediction,target,2)
expected=[]
for i in range(2):
    transform=official.fit_similarity_transform(prediction.detach().numpy()[i+2,:27],target.numpy()[i+2,:27])
    aligned=official.apply_similarity_transform(prediction.detach().numpy()[i,27:],transform)
    expected.append(float(official.symmetric_chamfer_distance(aligned[None],target.numpy()[i:i+1,27:]).mean()))
assert abs(loss.item()-np.mean(expected))<1e-10
gradient,=torch.autograd.grad(loss,prediction)
errors=[]
for mode in ['all','anchor_rotation','anchor_scale','anchor_translation']:
    for _ in range(3):
        direction=torch.randn_like(prediction)
        if mode!='all':
            direction[:2]=0;direction[2:]=0
            anchors=prediction.detach()[2:,:27];center=anchors.mean(1,keepdim=True)
            if mode=='anchor_rotation':
                axis=torch.randn(2,1,3,dtype=torch.float64)
                direction[2:,:27]=torch.linalg.cross(axis.expand_as(anchors),anchors-center)
            elif mode=='anchor_scale':direction[2:,:27]=anchors-center
            else:direction[2:,:27]=torch.randn(2,1,3,dtype=torch.float64)
        direction/=direction.norm();eps=1e-6
        finite=(paired_frame0_chamfer(prediction.detach()+eps*direction,target,2)-paired_frame0_chamfer(prediction.detach()-eps*direction,target,2))/(2*eps)
        analytic=(gradient*direction).sum();error=abs(float(finite-analytic))
        assert error<1e-7,(mode,float(finite),float(analytic),error)
        errors.append(dict(mode=mode,finite_difference=float(finite),analytic=float(analytic),absolute_error=error))
assert torch.autograd.gradcheck(lambda x:paired_frame0_chamfer(x,target,2),(prediction,),fast_mode=True,atol=1e-5,rtol=1e-4)
save_json(OUTPUT/'paired_cd_full_gradient_checks.json',dict(passed=True,full_anchor_sim3_gradient=True,
    official_value_difference_m=abs(loss.item()-np.mean(expected)),directional_checks=errors,
    gradcheck_fast_mode_passed=True,implementation_sha256=sha256(ROOT/'src/utils/track2_metrics.py'),
    GPU_used=False,GT_dataset_read=False,checked_unix=time.time()))
print('Paired CD-H reference value and full Sim3 gradient: PASS',flush=True)

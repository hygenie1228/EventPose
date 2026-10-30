"""Actual hand crop, geometry and released-model integration checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261028/hamer_hand'


def main(stage):
    import h5py
    import numpy as np
    import torch
    from hamer.hand_geometry import crop_transform
    from hamer.hand_data import HandFrames,to_device,model_inputs
    from utils.hand_metrics import self_check
    from utils.event_encoding import self_check as event_check
    from prepare_hamer_hand import write
    m=json.loads((OUT/'manifest.json').read_text());config=json.loads((OUT/'configs/rgb3.json').read_text())
    session=m['validation_sessions'][0]
    d=HandFrames(config,'validation',records=[dict(session=session,row=i) for i in range(8)])
    samples=[d[i] for i in range(8)];pixel_means=[];projection=[]
    for s in samples:
        rgb=s['image'].numpy()*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
        pixel_means.append(float(rgb.mean()));assert rgb.mean()>.005 and rgb.std()>.005,'Empty hand crop'
        p=s['joints'].numpy()@s['K'].numpy().T;uv=p[:,:2]/p[:,2:]
        inside=((uv>=0)&(uv<256)).all(1).mean();projection.append(float(inside));assert inside>.8
    # The two anatomical sides must map to positive vs negative crop focal X.
    for right in (0,1):
        A,size=crop_transform([30,40,80,110],right)
        center=A@np.array([55,75,1]);assert np.allclose(center[:2],[128,128],atol=1e-4)
        assert np.sign(A[0,0])==(2*right-1)
    result=dict(status='passed',metric_checks=self_check(),event_checks=event_check(),
        actual_validation_session=session,crop_pixel_means=pixel_means,joints_inside_crop_fraction=projection,
        root_metrics=False,checked_unix=time.time())
    from hamer.event_model import HandEpipolarFusion
    torch.set_num_threads(2);fusion_checks={}
    for encoding in ('count33','bins100','surface20'):
        c=dict(config,condition='rgb3_event4',encoding=encoding)
        ds=HandFrames(c,'validation',records=[dict(session=session,row=0)])
        s=ds[0];channels=s['auxiliary'].shape[0]
        fusion=HandEpipolarFusion(channels,1280,normalize=True,input_gain=4.)
        feature=torch.randn(1,1280,16,12)*.1
        context=[s[k][None] for k in ('auxiliary','K','auxiliary_K','rotation','translation','auxiliary_available')]
        original=fusion(feature,*context);assert torch.equal(original,feature)
        opt=torch.optim.AdamW(fusion.parameters(),lr=.001)
        original.square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        updated=fusion(feature,*context);updated.square().mean().backward()
        gradient=sum(float(p.grad.square().sum()) for p in fusion.encoder.parameters() if p.grad is not None)**.5
        assert gradient>0,('Actual Event4 pixels receive no gradient',encoding)
        context[-1]=torch.zeros_like(context[-1]);missing=fusion(feature,*context)
        assert torch.equal(missing,feature),'Unavailable Event4 changed RGB features'
        fusion_checks[encoding]=dict(zero_initial_residual=True,encoder_gradient_l2=gradient,missing_event_bypass=True)
    result['actual_calibrated_fusion_checks']=fusion_checks
    if stage=='gpu':
        from hamer.event_model import load_model
        torch.set_num_threads(2);assert int(os.environ['EVENTPOSE_PHYSICAL_GPU'])==0
        assert torch.cuda.device_count()==1
        device=torch.device('cuda',0);batch={k:torch.stack([s[k] for s in samples[:2]]) for k in samples[0]}
        batch=to_device(batch,device)
        base=load_model(config,device).eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            expected=base(model_inputs(batch))['joints'].cpu()
        assert torch.isfinite(expected).all();del base;torch.cuda.empty_cache()
        differences={}
        for encoding in ('count33','bins100','surface20'):
            c=json.loads((OUT/'configs'/('rgb3_event4_'+encoding+'.json')).read_text())
            ds=HandFrames(c,'validation',records=[dict(session=session,row=i) for i in range(2)])
            items=[ds[i] for i in range(2)];b=to_device({k:torch.stack([s[k] for s in items]) for k in items[0]},device)
            model=load_model(c,device).eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                got=model(model_inputs(b))['joints'].cpu()
            error=float((got-expected).abs().max());assert error<2e-4,('Zero residual changed original HaMeR',encoding,error)
            differences[encoding]=error;del model;torch.cuda.empty_cache()
        result.update(cuda_name=torch.cuda.get_device_name(0),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
            physical_gpu=0,pretrained_full_forward=True,zero_residual_max_joint_difference_m=differences)
    write(OUT/('gpu_preflight.json' if stage=='gpu' else 'cpu_preflight.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cpu','gpu']);main(p.parse_args().stage)

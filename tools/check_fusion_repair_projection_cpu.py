"""Independent projection check on frozen Train calibration, without pixels/GT."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_independent_Train_projection_CPU')
    import h5py
    import numpy as np
    import torch
    from multihmr.multisensor_repair import EpipolarEventFusion,REPAIR_LAYOUTS
    torch.set_num_threads(2)
    out=Path(goal['output_root']);manifest=json.loads((out/'manifest.json').read_text())
    fusion=EpipolarEventFusion(2,16,hidden=16,grid=16,normalize=True,input_gain=4.)
    axis=(np.arange(16,dtype=np.float64)+.5)*896/16-.5
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    pixels=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3)
    depths=fusion.depths_m.numpy().astype(np.float64)
    records=[]
    for session in manifest['training_sessions']:
        with h5py.File(out/'cache'/(session+'.h5'),'r',locking=False) as f:
            for condition in ['rgb3_event4','event3_rgb4','event3_event4']:
                p,a=REPAIR_LAYOUTS[condition]
                K=f[p+'_K_primary'][:].astype(np.float32)
                Ka=f[a+'_K_auxiliary'][:].astype(np.float32)
                Rp=f[p+'_R_common_to_sensor'][:].astype(np.float32)
                Ra=f[a+'_R_common_to_sensor'][:].astype(np.float32)
                tp=f[p+'_t_common_to_sensor'][:].astype(np.float32)
                ta=f[a+'_t_common_to_sensor'][:].astype(np.float32)
                R=Ra@Rp.T;t=ta-R@tp
                # Independent column-vector camera projection in float64.
                rays=np.linalg.solve(K.astype(np.float64),pixels.T).T
                rays/=rays[:,2:]
                xyz=(rays[:,None,:]*depths[None,:,None]).reshape(-1,3)
                aux=(R.astype(np.float64)@xyz.T+t.astype(np.float64)[:,None]).T
                hom=(Ka.astype(np.float64)@aux.T).T
                expected=(hom[:,:2]/np.maximum(hom[:,2:],1e-8)).reshape(256,11,2)
                args=[torch.from_numpy(x)[None] for x in [K,Ka,R,t]]
                with torch.no_grad():grid,valid=fusion.projection(*args,896,224)
                actual=(grid[0].numpy()+1)*223/2
                front=aux[:,2].reshape(256,11)>.05
                error=float(np.max(np.abs(actual[front]-expected[front])))
                assert error<.01,(session,condition,error)
                valid64=(aux[:,2].reshape(256,11)>0)&(expected>=0).all(-1)&(expected<=223).all(-1)
                # Near a boundary a float32/64 comparison can differ; examine
                # only samples farther than .01 pixel from every image edge.
                boundary=np.minimum(np.abs(expected),np.abs(expected-223)).min(-1)>.01
                assert np.array_equal(valid[0].numpy()[boundary],valid64[boundary])
                records.append(dict(session=session,condition=condition,max_error_pixels=error,
                                    rays=256,candidates=11))
    # A valid positive primary ray points behind the auxiliary camera here.
    K=torch.tensor([[[100.,0,111.5],[0,100.,111.5],[0,0,1.]]])
    reverse=torch.diag(torch.tensor([-1.,1.,-1.]))[None]
    with torch.no_grad():
        fusion.output.weight.fill_(.1);fusion.output.bias.fill_(.2)
        x=torch.randn(1,4096,16)
        context=(torch.rand(1,2,224,224),K,K,reverse,torch.zeros(1,3),torch.ones(1),896)
        y=fusion(x,context)
        assert torch.isfinite(y).all() and torch.equal(x,y)
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                sessions=len(manifest['training_sessions']),records=records,
                maximum_error_pixels=max(r['max_error_pixels'] for r in records),
                all_behind_camera_finite_exact_fallback=True,
                inputs='Frozen Train camera calibration only; no sensor pixels, GT or Test',
                independent_reference='NumPy float64 column-vector projection / solve',
                training_or_configuration_changed=False)
    (out/'independent_Train_projection_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()

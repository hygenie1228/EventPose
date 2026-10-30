"""Real-Train gradient/coordinate sanity checks; weights are always discarded."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_multihmr import configure_model,targets,losses
from multihmr.multisensor_model import CONDITIONS,MultiSensorMultiHMR,checkpoint_frozen_blocks
from multihmr.multisensor_data import Frames,collate,input_keys
from utils.pose_evaluation import configure_torch,load_multihmr
from tools.prepare_device34_body import OUT,save


def main():
    configure_torch();torch.cuda.set_device(0)
    manifest=json.loads((OUT/'manifest.json').read_text());records=[];sessions=[]
    for name in manifest['training_sessions']:
        cache=OUT/'cache'/(name+'.h5');target=OUT/'targets'/(name+'.npz')
        if not cache.exists() or not target.exists():continue
        with h5py.File(cache,'r',locking=False) as f:
            assert f.attrs['complete'] and f.attrs['split_sha256']==manifest['split_sha256']
            frames=f['frames'][:]
        with np.load(target) as z:
            assert np.array_equal(frames,z['positions']);available=np.flatnonzero(z['valid'].any(1))
        if not len(available):continue
        chosen=available[np.linspace(0,len(available)-1,min(16,len(available)),dtype=int)]
        records.extend(dict(session=name,cache_row=int(i),frame=int(frames[i])) for i in chosen)
        sessions.append(name)
        if len(sessions)==3:break
    assert len(sessions)==3 and len(records)==48
    assert not set(sessions)&set(manifest['test_sessions'])
    index=OUT/'probe_training_index.json';save(index,records)
    reg=torch.from_numpy(np.load(OUT/'joint_regressor.npy')).cuda();results=[]
    for condition in CONDITIONS:
        torch.manual_seed(261026);np.random.seed(261026)
        base=load_multihmr();configure_model(base,8);checkpoint_frozen_blocks(base,8)
        model=MultiSensorMultiHMR(base,condition).cuda().train()
        parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        optimizer=torch.optim.AdamW([dict(params=[p for n,p in parameters.items() if not n.startswith('base.backbone.')],lr=1e-5),
            dict(params=[p for n,p in parameters.items() if n.startswith('base.backbone.')],lr=1e-6)],weight_decay=.01)
        scaler=torch.cuda.amp.GradScaler(init_scale=512)
        dataset=Frames('training',condition=condition,index_path=index,augment=True)
        loader=DataLoader(dataset,batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(261026),
            num_workers=0,collate_fn=collate,drop_last=True)
        iterator=iter(loader);success=0;rows=[];started=time.time()
        for step in range(20):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            for name,p in parameters.items():
                if name.startswith('base.backbone.'):p.requires_grad_(step>=4)
            b,gt=targets(batch,torch.device('cuda'))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):output=model(**input_keys(b),idx=gt['idx'],is_training=True)
            loss,parts=losses(output,gt,reg,dict(pa_weight=100,root_weight=25))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();success+=int(scaler.get_scale()>=before)
            row=dict(step=step+1,loss=float(loss.detach()),finite_grad=bool(torch.isfinite(norm)),amp_scale=scaler.get_scale(),successful_updates=success)
            rows.append(row);print(json.dumps(dict(condition=condition,**row)),flush=True)
            if step>=7 and success>=2:break
        assert success>0,(condition,'No successful real-Train update')
        results.append(dict(condition=condition,steps=len(rows),successful_updates=success,elapsed_seconds=time.time()-started,rows=rows))
        save(OUT/'real_train_gradient_probe_progress.json',dict(completed_conditions=len(results),results=results))
        for handle in dataset.handles.values():handle.close()
        if hasattr(model,'_hook_handle'):model._hook_handle.remove()
        del output,loss,model,base,parameters,optimizer,scaler,dataset,loader,b,batch,gt
        torch.cuda.empty_cache()
    save(OUT/'real_train_gradient_probe.json',dict(status='passed',sessions=sessions,unique_frames=48,
        results=results,split_sha256=manifest['split_sha256'],gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        description='실제 Train만 사용한 짧은 좌표·유한 loss·AMP gradient 검사. 모든 probe 가중치는 폐기하며 본실험은 원본에서 독립 초기화.',
        final_checkpoints_produced=False,Test_used=False,completed_unix=time.time()))


if __name__=='__main__':main()

from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Two-GPU NLF adaptation; holdout selection stays inside the Train split."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import torchvision
from nlf.model import load_model
from torch.utils.data import DataLoader,DistributedSampler,Subset
from nlf.data import PersonCrops,pa_loss,ROOT

OUT=ROOT/'exp/261024/nlf_finetune_12h'
ASSETS=OUT/'checkpoints'

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main(a):
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(261024);torch.cuda.manual_seed_all(261024)
    full=load_model(device=f'cuda:{rank}')
    model=full.crop_model.float().eval()
    canonical=full.cano_all['smplx'][-55:].detach().float().cuda()
    assert canonical.shape==(55,3)
    for p in model.parameters():p.requires_grad_(False)
    for name,p in model.named_parameters():
        if name.startswith('heatmap_head.layer.') or (a.backbone and name.startswith('backbone.1.') and int(name.split('.')[2])>=a.backbone_start):
            p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad]
    assert params
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    data=PersonCrops(resolution=int(model.input_resolution))
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=261024)
    loader=DataLoader(data,batch_size=a.batch,sampler=sampler,num_workers=4,pin_memory=True,drop_last=True)
    val=PersonCrops(role='validation',resolution=int(model.input_resolution),augment=False)
    ids=torch.linspace(0,len(val)-1,min(512,len(val))).long().tolist()[rank::world]
    val_loader=DataLoader(Subset(val,ids),batch_size=a.batch,num_workers=2,pin_memory=True)
    folder=OUT/'runs'/a.name;asset=ASSETS/a.name
    folder.mkdir(parents=True,exist_ok=True);asset.mkdir(parents=True,exist_ok=True)
    if rank==0:atomic_json(folder/'configuration.json',dict(vars(a),world_size=world,people=len(data),
                  excluded_behind_camera=data.excluded_behind_camera,
                  trainable_parameters=sum(p.numel() for p in params),selection='Train internal holdout crop PA; Test never used'))
    def predict(batch):
        x=batch['image'].cuda(non_blocking=True);K=batch['K'].cuda(non_blocking=True)
        coords,_=model.predict_multi_same_canonicals(x,K,canonical)
        return coords.float()/1000,batch['joints'].cuda(non_blocking=True)
    def validate(step):
        total=torch.zeros(2,device='cuda',dtype=torch.float64)
        with torch.no_grad():
            for batch in val_loader:
                p,t=predict(batch);loss=pa_loss(p,t)
                assert torch.isfinite(loss)
                total+=torch.tensor([loss.item()*len(p),len(p)],device='cuda')
        if world>1:dist.all_reduce(total)
        value=float(total[0]/total[1])*1000
        if rank==0:
            with (folder/'selection_history.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,pa_mm=value,time=time.time()))+'\n')
        return value
    start=time.time();step=0;successful=0;epoch=0;best=validate(0);last_val=time.time();last_save=time.time()
    resumed=False;resumed_improvement=False
    if a.resume:
        resume=asset/'last.pt'
        if resume.exists():
            checkpoint_data=torch.load(resume,map_location='cpu')
            model.load_state_dict(checkpoint_data['model_state_dict'],strict=False)
            if 'optimizer' in checkpoint_data:optimizer.load_state_dict(checkpoint_data['optimizer'])
            step=checkpoint_data['step'];successful=checkpoint_data['successful_steps']
            start-=checkpoint_data['elapsed_seconds'];current=validate(step)
            previous=json.loads((folder/'best.json').read_text())['pa_mm']
            best=min(previous,current);resumed=True;resumed_improvement=current<previous
    def checkpoint(label,value):
        if rank!=0:return
        state={name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
        path=asset/f'{label}.pt';temp=path.with_suffix('.tmp.pt')
        torch.save(dict(model_state_dict=state,step=step,successful_steps=successful,
                        elapsed_seconds=time.time()-start,selection_pa_mm=value,config=vars(a),
                        optimizer=optimizer.state_dict()),temp);temp.replace(path)
        atomic_json(folder/f'{label}.json',dict(step=step,pa_mm=value,successful_steps=successful,elapsed_seconds=time.time()-start))
    if not resumed or resumed_improvement:checkpoint('best',best)
    stopped=False
    while not stopped:
        sampler.set_epoch(epoch)
        for batch in loader:
            stop=torch.tensor(int(time.time()>=a.end_unix or (a.max_steps and step>=a.max_steps)),device='cuda')
            if world>1:dist.broadcast(stop,src=0)
            if stop.item():
                stopped=True
                break
            optimizer.zero_grad(set_to_none=True)
            p,t=predict(batch)
            pa=pa_loss(p,t);root=torch.linalg.vector_norm((p-p[:,:1])-(t-t[:,:1]),dim=-1).mean()
            absolute=torch.nn.functional.smooth_l1_loss(p,t,beta=.1)
            loss=a.pa_weight*pa+25*root+5*absolute
            loss.backward()
            finite=torch.tensor(int(torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in params)),device='cuda')
            if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if finite.item():
                if world>1:
                    for q in params:
                        if q.grad is None:q.grad=torch.zeros_like(q)
                        dist.all_reduce(q.grad);q.grad.div_(world)
                torch.nn.utils.clip_grad_norm_(params,1.)
                optimizer.step();successful+=1
            step+=1
            save_now=torch.tensor(int(time.time()-last_save>300),device='cuda')
            if world>1:dist.broadcast(save_now,src=0)
            if save_now.item():checkpoint('last',best);last_save=time.time()
            if rank==0 and step%25==0:
                atomic_json(folder/'progress.json',dict(step=step,successful_steps=successful,loss=float(loss.detach()),
                    pa_mm=float(pa.detach())*1000,elapsed_seconds=time.time()-start,updated_unix=time.time()))
                print(json.dumps(dict(step=step,successful=successful,loss=float(loss.detach()))),flush=True)
            check=torch.tensor(int(time.time()-last_val>1800),device='cuda')
            if world>1:dist.broadcast(check,src=0)
            if check.item():
                value=validate(step);last_val=time.time();checkpoint('last',value)
                if value<best:best=value;checkpoint('best',best)
        epoch+=1
    value=validate(step);checkpoint('last',value)
    if value<best:best=value;checkpoint('best',best)
    if rank==0:
        assert successful>0
        atomic_json(folder/'complete.json',dict(step=step,successful_steps=successful,elapsed_seconds=time.time()-start,best_pa_mm=best))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--backbone',action='store_true')
    p.add_argument('--backbone-start',type=int,default=7);p.add_argument('--pa-weight',type=float,default=100.)
    p.add_argument('--end-unix',type=float,required=True);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    main(p.parse_args())

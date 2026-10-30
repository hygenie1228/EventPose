"""Train the four GT-hand-crop trials with the body experiment convention."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from torch.utils.data import DataLoader
from hamer.hand_data import HandFrames,to_device,model_inputs
from hamer.event_model import load_model
from utils.humanpose_metrics import joint_errors


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def pose_loss(pred,target):
    """Identical proper Sim(3) and root-relative losses used by the body study."""
    x=pred-pred[:,:1];y=target-target[:,:1]
    xc=x-x.mean(1,keepdim=True);yc=y-y.mean(1,keepdim=True)
    with torch.no_grad():
        u,s,vh=torch.linalg.svd(xc.detach().transpose(1,2)@yc)
        d=torch.ones_like(s);d[:,-1]=torch.det(u@vh).sign()
        R=(u*d[:,None,:])@vh
        scale=(s*d).sum(1)/xc.detach().square().sum((1,2)).clamp_min(1e-10)
    aligned=scale[:,None,None]*(xc@R)+y.mean(1,keepdim=True)
    pa=torch.linalg.vector_norm(aligned-y,dim=-1).mean()
    relative=torch.linalg.vector_norm(x-y,dim=-1).mean()
    # Absolute wrist is supervised to retain motion under fixed first-frame
    # alignment; no RPE is computed or reported as an evaluation metric.
    absolute=torch.nn.functional.smooth_l1_loss(pred[:,0],target[:,0],beta=.1)
    return 100*pa+25*relative+absolute


@torch.inference_mode()
def validate(model,loader,device):
    model.eval();total=np.zeros(2,np.float64);count=0
    for batch in loader:
        target=batch['joints'].numpy()
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            pred=model(model_inputs(to_device(batch,device)))['joints'].cpu().numpy()
        assert np.isfinite(pred).all()
        for p,t in zip(pred,target):total+=joint_errors(p,t)[1:];count+=1
    assert count
    return dict(frame_mpjpe_mm=float(total[0]/count*1000),frame_pa_mpjpe_mm=float(total[1]/count*1000),
                samples=count,source='Train subject holdout',test_used_for_selection=False)


def main(config_path):
    config=json.loads(Path(config_path).read_text());out=Path(config['output_root']);name=config['name']
    torch.set_num_threads(2);assert int(os.environ['EVENTPOSE_PHYSICAL_GPU'])==0
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and torch.cuda.device_count()==1
    run=out/'runs'/name;assets=out/'checkpoints'/name
    run.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    # NFS lock-manager RPCs can hang even with LOCK_NB. Keep ownership locks local.
    lock_path=ROOT/'_agents/research/261028/hamer_run_locks'/name
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'complete.json').exists(),'Refusing duplicate completed training'
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=torch.device('cuda',0)
    model=load_model(config,device)
    # Constructing different fusion modules must not change RGB augmentation/shuffle.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    groups=[]
    for key,prefix in [('backbone','backbone.'),('head','mano_head.'),('fusion','fusion.')]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix)]
        if params:groups.append(dict(params=params,lr=config[key+'_lr'],initial_lr=config[key+'_lr'],group_name=key))
    opt=torch.optim.AdamW(groups,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=512)
    pilot=config.get('pilot_steps',0)
    def pilot_records(role):
        path=config.get('pilot_'+role+'_index') if pilot else None
        return json.loads(Path(path).read_text()) if path else None
    train=HandFrames(config,'training',augment=config['augment'],records=pilot_records('training'),limit=64 if pilot else 0)
    val=HandFrames(config,'validation',records=pilot_records('validation'),limit=32 if pilot else 0)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=config['batch_size'],shuffle=True,num_workers=0 if pilot else config['workers'],
        pin_memory=True,drop_last=True,generator=generator)
    vloader=DataLoader(val,batch_size=config['batch_size'],num_workers=0 if pilot else config['workers'],pin_memory=True)
    selected={n for n,p in model.named_parameters() if p.requires_grad}
    started=time.time();step=successful=epoch=seen=0;best=float('inf');metrics_step=0;unique=set()
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def checkpoint(tag,metrics):
        state=model.state_dict();path=assets/(tag+'.pt');tmp=path.with_suffix('.tmp.pt')
        torch.save(dict(config=config,step=step,successful_steps=successful,epoch=epoch,metrics=metrics,metrics_step=metrics_step,
            started_unix=started,model_delta={k:v.detach().cpu() for k,v in state.items() if k in selected},
            optimizer=opt.state_dict(),amp_scaler=scaler.state_dict(),test_used_for_selection=False,
            unique_indices=sorted(unique),samples_seen=seen,torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state(),
            loader_rng=generator.get_state()),tmp)
        tmp.replace(path)
    if (assets/'last.pt').exists():
        old=torch.load(assets/'last.pt',map_location='cpu',weights_only=False);assert old['config']==config
        assert not model.load_state_dict(old['model_delta'],strict=False).unexpected_keys
        opt.load_state_dict(old['optimizer']);scaler.load_state_dict(old['amp_scaler'])
        started=old['started_unix'];step=old['step'];successful=old['successful_steps'];epoch=old['epoch']
        unique=set(old['unique_indices']);seen=old['samples_seen']
        if (run/'best.json').exists():best=json.loads((run/'best.json').read_text())['frame_pa_mpjpe_mm']
        torch.set_rng_state(old['torch_rng']);np.random.set_state(old['numpy_rng']);random.setstate(old['python_rng'])
        torch.cuda.set_rng_state(old['cuda_rng'])
        generator.set_state(old['loader_rng'])
    finish=started+config['hours']*3600
    metrics=validate(model,vloader,device);metrics_step=step
    if step>0 and metrics['frame_pa_mpjpe_mm']<best:
        best=metrics['frame_pa_mpjpe_mm'];checkpoint('best',metrics);write(run/'best.json',dict(metrics,step=step))
    write(run/'initial_validation.json',dict(metrics,step=step,started_unix=started))
    last_val=time.time();last_save=time.time();model.train()
    while time.time()<finish and not stop[0] and (not pilot or step<pilot):
        batches=0
        for batch in loader:
            if time.time()>=finish or stop[0] or (pilot and step>=pilot):break
            step+=1;indices=batch['index'].tolist();batch=to_device(batch,device);opt.zero_grad(set_to_none=True)
            fraction=min(1,step/config['warmup_steps'])
            for group in opt.param_groups:
                group['lr']=group['initial_lr']*fraction
                if group['group_name']=='backbone' and time.time()-started<config['backbone_warmup_seconds']:group['lr']=0.
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(model_inputs(batch))
            with torch.autocast('cuda',enabled=False):
                loss=pose_loss(pred['joints'].float(),batch['joints'].float())
                if hasattr(model,'fusion'):loss=loss+config['residual_regularization']*model.fusion.residual_energy
            assert torch.isfinite(loss),('Non-finite hand loss',step)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            grads={}
            if pilot:
                for label,prefix in [('fusion_encoder','fusion.encoder.'),('fusion_output','fusion.output.'),('head','mano_head.')]:
                    grads[label+'_gradient_l2']=float(sum(p.grad.float().square().sum().item() for n,p in model.named_parameters()
                        if n.startswith(prefix) and p.grad is not None)**.5)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            before=scaler.get_scale();scaler.step(opt);scaler.update()
            if torch.isfinite(norm) and scaler.get_scale()>=before:successful+=1
            seen+=len(indices);unique.update(indices)
            batches+=1
            if pilot or step%20==0:
                record=dict(step=step,successful_steps=successful,loss=float(loss.detach()),elapsed_seconds=time.time()-started,**grads)
                with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if not pilot and time.time()-last_val>=config['validation_minutes']*60:
                metrics=validate(model,vloader,device);metrics_step=step
                with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(metrics,step=step))+'\n')
                if metrics['frame_pa_mpjpe_mm']<best:
                    best=metrics['frame_pa_mpjpe_mm'];checkpoint('best',metrics);write(run/'best.json',dict(metrics,step=step))
                checkpoint('last',metrics);last_val=last_save=time.time();model.train()
            elif time.time()-last_save>600:
                checkpoint('last',metrics);last_save=time.time()
        if batches==len(loader):epoch+=1
    assert successful>0,'No successful parameter update'
    checkpoint('last',metrics)
    metrics=validate(model,vloader,device);metrics_step=step
    # Always use an actually updated checkpoint for the 90min/3h comparison.
    # Zero-update pretrained HaMeR is evaluated separately as its own baseline.
    if metrics['frame_pa_mpjpe_mm']<best:
        checkpoint('best',metrics);write(run/'best.json',dict(metrics,step=step))
    checkpoint('last',metrics)
    write(run/'complete.json',dict(status='interrupted' if stop[0] else 'pilot_complete' if pilot else 'complete',
        step=step,successful_steps=successful,elapsed_seconds=time.time()-started,subset_samples=len(train),
        samples_seen=seen,unique_samples_seen=len(unique),subset_epochs=epoch,checkpoint=str(assets/'best.pt'),
        test_used_for_selection=False,completed_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)

from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Two-GPU DDP Multi-HMR adaptation with held-out-subject PA-MPJPE selection."""
import argparse
from collections import OrderedDict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler,Subset
from PIL import Image,ImageOps
import torchvision.transforms as T

from multihmr.data import ROOT,OUT,WORK,ASSETS,save
from utils.pose_evaluation import configure_torch,load_multihmr
from utils.humanpose_metrics import joint_errors,match_people

BODY_MIRROR=[0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20]
JOINT_MIRROR=BODY_MIRROR+[22,24,23]+list(range(40,55))+list(range(25,40))
ROTATION_MIRROR=BODY_MIRROR+list(range(37,52))+list(range(22,37))+[52]


def mirror_targets(joints,centers,rotations,K,width):
    """Reflect camera X, swap anatomical sides, conjugate proper joint rotations."""
    j=joints[:,JOINT_MIRROR].copy();j[...,0]*=-1
    c=centers.copy();c[...,0]*=-1
    reflection=np.diag([-1,1,1]).astype(rotations.dtype)
    r=reflection@rotations[:,ROTATION_MIRROR]@reflection
    camera=K.copy();camera[0,2]=width-1-camera[0,2]
    return j,c,r,camera


class Frames(Dataset):
    def __init__(self,role,augment=False,limit=0,flip_probability=0):
        self.records=json.loads((WORK/f'{role}_index.json').read_text())
        if limit:
            indices=np.linspace(0,len(self.records)-1,min(limit,len(self.records)),dtype=int)
            self.records=[self.records[int(i)] for i in indices]
        self.augment=augment;self.flip_probability=flip_probability;self.cache=OrderedDict()
        self.norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.jitter=T.ColorJitter(.2,.2,.15,.03)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index];name=record['session'];row=record['row']
        if name not in self.cache:
            with np.load(WORK/'targets'/f'{name}.npz') as z:self.cache[name]={k:z[k] for k in z.files}
            if len(self.cache)>12:self.cache.popitem(last=False)
        z=self.cache[name]
        with Image.open(WORK/'images'/name/f'{record["frame"]:06d}.jpg') as img:image=img.convert('RGB')
        w,h=image.size;assert (h,w)==tuple(z['image_shape'])
        joints=z['joints'][row];centers=z['centers'][row];rotations=z['rotations'][row];original_K=z['K']
        if self.augment and self.flip_probability and random.random()<self.flip_probability:
            image=ImageOps.mirror(image)
            joints,centers,rotations,original_K=mirror_targets(joints,centers,rotations,original_K,w)
        image=ImageOps.contain(image,(896,896),method=Image.Resampling.LANCZOS)
        nw,nh=image.size;l,t=(896-nw)//2,(896-nh)//2
        image=ImageOps.expand(image,border=(l,t,896-nw-l,896-nh-t))
        if self.augment:image=self.jitter(image)
        K=original_K.copy();K[0]*=nw/w;K[1]*=nh/h;K[0,2]+=l;K[1,2]+=t
        return dict(image=self.norm(image),K=K,original_K=original_K,image_shape=z['image_shape'],
                    joints=joints,centers=centers,valid=z['valid'][row],
                    rotations=rotations,betas=z['betas'][row],index=index)


def collate(items):
    result={k:torch.stack([torch.as_tensor(x[k]) for x in items]) for k in ['image','K','original_K','image_shape']}
    people=max(len(x['valid']) for x in items)
    for key in ['joints','centers','rotations','betas','valid']:
        shape=(len(items),people,*items[0][key].shape[1:])
        a=np.zeros(shape,dtype=items[0][key].dtype)
        for i,x in enumerate(items):a[i,:len(x[key])]=np.nan_to_num(x[key])
        result[key]=torch.from_numpy(a)
    result['index']=[x['index'] for x in items]
    return result


def configure_model(model,last_blocks):
    for p in model.parameters():p.requires_grad_(False)
    for name,p in model.named_parameters():
        if name.startswith(('mlp_classif.','mlp_offset.','x_attention_head.')):p.requires_grad_(True)
    if last_blocks:
        from torch.utils.checkpoint import checkpoint
        encoder=model.backbone.encoder
        for block in list(encoder.blocks)[-last_blocks:]:
            for p in block.parameters():p.requires_grad_(True)
            original=block.forward
            def forward(*args,_original=original,**kwargs):
                return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            block.forward=forward
        for p in encoder.norm.parameters():p.requires_grad_(True)


def load_delta(model,path):
    value=torch.load(path,map_location='cpu')
    state=value['model_delta']
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    return value


def targets(batch,device):
    b={k:v.to(device,non_blocking=True) for k,v in batch.items() if isinstance(v,torch.Tensor)}
    valid=b['valid'].bool();center=b['centers'];K=b['K']
    hom=center@K.transpose(-1,-2)
    uv=hom[...,:2]/hom[...,2:].clamp_min(1e-8)
    original=center@b['original_K'].transpose(-1,-2)
    original_uv=original[...,:2]/original[...,2:].clamp_min(1e-8)
    h,w=b['image_shape'][:,0,None],b['image_shape'][:,1,None]
    valid=valid&(center[...,2]>.05)&(original_uv[...,0]>=0)&(original_uv[...,0]<w)&(original_uv[...,1]>=0)&(original_uv[...,1]<h)
    cells=(uv/14).floor().long().clamp(0,63)
    selected=[];occupied=set()
    for bi,pi in valid.nonzero().tolist():
        x,y=cells[bi,pi].tolist();key=(bi,y,x)
        if key in occupied:continue
        selected.append((bi,pi));occupied.add(key)
    assert selected,'Training batch has no visible GT person centers'
    bi,pi=torch.tensor(selected,device=device).T
    xy=cells[bi,pi];scores=torch.zeros((len(valid),64,64,1),device=device)
    scores[bi,xy[:,1],xy[:,0],0]=1
    return b,dict(idx=(bi,xy[:,1],xy[:,0],torch.zeros_like(bi)),scores=scores,
                  offset=(uv[bi,pi]-(xy+.5)*14)/14,joints=b['joints'][bi,pi],
                  rotations=b['rotations'][bi,pi],betas=b['betas'][bi,pi])


def aligned_errors(pred,target):
    """Independent per-person root centering and proper similarity alignment."""
    x=pred-pred[:,0:1];y=target-target[:,0:1]
    xc=x-x.mean(1,keepdim=True);yc=y-y.mean(1,keepdim=True)
    # Solve alignment without differentiating the SVD; pose gradients flow through
    # aligned predicted joints. This avoids unstable SVD derivatives near symmetry.
    with torch.no_grad():
        u,s,vh=torch.linalg.svd(xc.detach().transpose(1,2)@yc)
        d=torch.ones_like(s);d[:,-1]=torch.det(u@vh).sign()
        R=(u*d[:,None,:])@vh
        scale=(s*d).sum(1)/xc.detach().square().sum((1,2)).clamp_min(1e-10)
    aligned=scale[:,None,None]*(xc@R)+y.mean(1,keepdim=True)
    return torch.linalg.vector_norm(aligned-y,dim=-1).mean(),torch.linalg.vector_norm(x-y,dim=-1).mean()


def losses(output,gt,reg,config):
    with torch.autocast('cuda',enabled=False):
        joints=reg@output['v3d'].float();target=gt['joints'].float()
        pa,root=aligned_errors(joints[:,:22],target[:,:22])
        scores=output['scores'].float().clamp(1e-6,1-1e-6);truth=gt['scores']
        positive=truth.eq(1);negative=~positive
        detection=-(scores.log()*(1-scores).square()*positive+(1-scores).log()*scores.square()*negative).sum()/positive.sum().clamp_min(1)
        offset=torch.nn.functional.smooth_l1_loss(output['offset'].float(),gt['offset'])
        rotation=torch.nn.functional.smooth_l1_loss(output['rotmat'].float()[:,:22],gt['rotations'][:,:22])
        shape=torch.nn.functional.smooth_l1_loss(output['shape'].float(),gt['betas'][:,:output['shape'].shape[1]])
        absolute=torch.nn.functional.smooth_l1_loss(joints[:,0],target[:,0],beta=.1)
        total=config.get('pa_weight',100)*pa+config.get('root_weight',25)*root+detection+offset+.2*rotation+.02*shape+absolute
    return total,dict(loss=total,pa_mm=1000*pa,root_mm=1000*root,detection=detection,offset=offset,rotation=rotation,shape=shape,absolute=absolute)


@torch.no_grad()
def validate(model,loader,reg,device):
    model.eval();original=model.detection;image_ids=[]
    def detection(*args,**kwargs):
        out=original(*args,**kwargs);image_ids[:]=out[2][0].detach().cpu().tolist();return out
    model.detection=detection
    values=np.zeros(6,np.float64)
    try:
        for batch in loader:
            with torch.autocast('cuda',dtype=torch.float16):
                people=model(batch['image'].to(device,non_blocking=True),K=batch['K'].to(device),is_training=False,det_thresh=.3,nms_kernel_size=3)
            assert len(people)==len(image_ids)
            grouped=[[] for _ in batch['index']]
            for index,person in zip(image_ids,people):grouped[index].append((reg@person['v3d'].float()).cpu().numpy())
            for i,persons in enumerate(grouped):
                pred=np.stack(persons) if persons else np.empty((0,55,3),np.float32)
                target=batch['joints'][i].numpy();valid=batch['valid'][i].numpy().astype(bool)
                K=batch['original_K'][i].numpy();shape=batch['image_shape'][i].tolist()
                mapping=match_people(pred,target,valid,K,K,shape)
                values[4]+=valid.sum();values[5]+=len(pred)
                for gi,pi in mapping.items():
                    values[:3]+=joint_errors(pred[pi,:22],target[gi,:22]);values[3]+=1
    finally:model.detection=original
    tensor=torch.tensor(values,dtype=torch.float64,device=device);dist.all_reduce(tensor)
    values=tensor.cpu().numpy();assert values[3]>0
    return dict(mpjpe_mm=1000*values[0]/values[3],root_mpjpe_mm=1000*values[1]/values[3],
                pa_mpjpe_mm=1000*values[2]/values[3],matched_instances=int(values[3]),valid_instances=int(values[4]),
                coverage=float(values[3]/values[4]),predicted_instances=int(values[5]))


def checkpoint(path,model,optimizer,config,step,epoch,metrics,baseline,started,full=True,scaler=None,successful_steps=0):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    value=dict(model_delta={n:v.detach().cpu() for n,v in model.state_dict().items() if n in names},
               base_checkpoint=str(ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'),
               config=config,step=step,epoch=epoch,metrics=metrics,baseline=baseline,started_unix=started,
               saved_unix=time.time(),trainable_names=sorted(names))
    if full:
        value['optimizer']=optimizer.state_dict()
        value['successful_steps']=successful_steps
        if scaler is not None:value['amp_scaler']=scaler.state_dict()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.pt');torch.save(value,temp);temp.replace(path)


def main(config_path):
    config=json.loads(Path(config_path).read_text())
    overrides=OUT/'training_overrides.json'
    if overrides.exists():config.update(json.loads(overrides.read_text()).get(config['name'],{}))
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20));assert dist.get_world_size()==2
    configure_torch();seed=config.get('seed',261023)
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed)
    run=config['name'];run_dir=OUT/'runs'/run;run_dir.mkdir(parents=True,exist_ok=True)
    ckpt_dir=ASSETS/run;ckpt_dir.mkdir(parents=True,exist_ok=True)
    model=load_multihmr();configure_model(model,config.get('last_blocks',0))
    if config.get('init_checkpoint'):load_delta(model,config['init_checkpoint'])
    params=[p for p in model.parameters() if p.requires_grad]
    groups=[]
    for backbone in [False,True]:
        selected=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')==backbone]
        if selected:groups.append(dict(params=selected,lr=config['backbone_lr'] if backbone else config['head_lr'],
                                       initial_lr=config['backbone_lr'] if backbone else config['head_lr']))
    optimizer=torch.optim.AdamW(groups,weight_decay=config.get('weight_decay',.01))
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    reg=torch.from_numpy(np.load(WORK/'joint_regressor.npy')).to(device)
    pilot=config.get('pilot_steps',0)
    train=Frames('training',augment=config.get('augment',True),limit=128 if pilot else 0,
                 flip_probability=config.get('horizontal_flip_probability',0))
    validation=Frames('validation',limit=32 if pilot else 0)
    sampler=DistributedSampler(train,num_replicas=2,rank=rank,shuffle=True,seed=seed,drop_last=True)
    workers=config.get('workers',2)
    train_loader=DataLoader(train,batch_size=config.get('batch_size',4),sampler=sampler,collate_fn=collate,
                            num_workers=workers,pin_memory=True,drop_last=True,persistent_workers=workers>0)
    val_loader=DataLoader(Subset(validation,list(range(rank,len(validation),2))),batch_size=16,collate_fn=collate,
                          num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    scaler=torch.cuda.amp.GradScaler(init_scale=512);step=0;epoch=0;started=time.time();best=float('inf');baseline=None;successful=0
    resume=ckpt_dir/'last.pt'
    if resume.exists() and not pilot:
        restored=load_delta(model,resume);optimizer.load_state_dict(restored['optimizer'])
        step=restored['step'];epoch=restored['epoch'];started=restored['started_unix'];baseline=restored['baseline']
        if 'amp_scaler' in restored:scaler.load_state_dict(restored['amp_scaler'])
        # Legacy checkpoints still contain exact Adam per-parameter update counters.
        counters=[int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
        successful=restored.get('successful_steps',min(counters,default=0))
        if (run_dir/'best.json').exists():best=json.loads((run_dir/'best.json').read_text())['pa_mpjpe_mm']
    duration=config.get('hours',6)*3600;finish=min(started+duration,config.get('deadline_unix',float('inf')))
    if rank==0:
        save(run_dir/'configuration.json',config)
        save(run_dir/'started.json',dict(pid=os.getpid(),started_unix=started,finish_unix=finish,
             trainable_parameters=sum(p.numel() for p in params),batch_size_per_gpu=config.get('batch_size',4),world_size=2))
    next_eval=time.time()+config.get('validation_minutes',45)*60
    metrics=validate(model,val_loader,reg,device)
    if baseline is None:baseline=metrics
    if rank==0:
        save(run_dir/'baseline.json',baseline)
        print(json.dumps(dict(event='baseline',run=run,**metrics)),flush=True)
        if best==float('inf'):
            best=metrics['pa_mpjpe_mm'];save(run_dir/'best.json',dict(metrics,step=step))
            checkpoint(ckpt_dir/'best.pt',model,optimizer,config,step,epoch,metrics,baseline,started,full=False)
    dist.barrier();sampler.set_epoch(epoch);iterator=iter(train_loader)
    moving=[];train_started=time.perf_counter();start_step=step;model.train()
    while True:
        control=torch.zeros(3,dtype=torch.float64,device=device)
        if rank==0:control[:]=torch.tensor([float(time.time()>=finish or (pilot and step>=pilot)),float(time.time()>=next_eval),time.time()-started],device=device)
        dist.broadcast(control,src=0);stop,evaluate,elapsed=control.tolist()
        if stop or evaluate:
            metrics=validate(model,val_loader,reg,device)
            if rank==0:
                record=dict(event='validation',run=run,step=step,epoch=epoch,elapsed_seconds=elapsed,**metrics)
                with (run_dir/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                eligible=metrics['coverage']>=config.get('minimum_coverage',baseline['coverage']-.005)
                if eligible and metrics['pa_mpjpe_mm']<best:
                    best=metrics['pa_mpjpe_mm'];save(run_dir/'best.json',dict(metrics,step=step))
                    checkpoint(ckpt_dir/'best.pt',model,optimizer,config,step,epoch,metrics,baseline,started,full=False)
                checkpoint(ckpt_dir/'last.pt',model,optimizer,config,step,epoch,metrics,baseline,started,
                           scaler=scaler,successful_steps=successful)
                next_eval=time.time()+config.get('validation_minutes',45)*60
            dist.barrier()
            if stop:break
            model.train();continue
        try:batch=next(iterator)
        except StopIteration:
            epoch+=1;sampler.set_epoch(epoch);iterator=iter(train_loader);batch=next(iterator)
        b,gt=targets(batch,device)
        warmup=min(1.,(step+1)/config.get('warmup_steps',100))
        cosine=.1+.9*.5*(1+math.cos(math.pi*min(1.,elapsed/duration)))
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*warmup*cosine
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            output=ddp(b['image'],idx=gt['idx'],K=b['K'],is_training=True)
        loss,components=losses(output,gt,reg,config)
        assert torch.isfinite(loss).item(),{k:float(v.detach()) for k,v in components.items()}
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        successful+=int(scaler.get_scale()>=before);step+=1
        moving.append({k:float(v.detach()) for k,v in components.items()})
        if step%25==0 or pilot:
            if rank==0:
                record=dict(event='training',run=run,step=step,epoch=epoch,elapsed_seconds=time.time()-started,
                    images_per_second=((step-start_step)*config.get('batch_size',4)*2)/max(time.perf_counter()-train_started,1e-6),
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,grad_norm=float(norm) if torch.isfinite(norm) else None,amp_scale=scaler.get_scale(),
                    successful_steps=successful,**{k:float(np.mean([v[k] for v in moving])) for k in moving[0]})
                print(json.dumps(record),flush=True);save(run_dir/'progress.json',record)
            moving=[]
    if rank==0:
        assert successful>0,'No optimizer steps succeeded'
        save(run_dir/'complete.json',dict(status='pilot_complete' if pilot else 'complete',run=run,step=step,
             successful_steps=successful,elapsed_seconds=time.time()-started,best_pa_mpjpe_mm=best,
             last_metrics=metrics,checkpoint=str(ckpt_dir/'best.pt'),completed_unix=time.time()))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)

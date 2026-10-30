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
    def __init__(self,role,augment=False,limit=0,flip_probability=0,preserve_frame0=False,target_foreground=False,fixed_view_plan=None,native_surface_target=False):
        self.target_foreground=target_foreground
        self.native_surface_target=native_surface_target
        if native_surface_target:
            from multihmr.track2_data import sha256
            assert not flip_probability,'Native vertex correspondence requires an explicitly implemented mirror map'
            self.native_metric_roles_sha256=sha256(WORK.parent/'body_roles.json')
        self.fixed_view_plan=fixed_view_plan
        self.fixed_view_specs=json.loads(Path(fixed_view_plan).read_text())['samples'] if fixed_view_plan else None
        if fixed_view_plan:assert not target_foreground and not flip_probability
        if target_foreground:assert (WORK/'target_foreground_preparation.json').exists(),'Prepare target-only RGB cache first'
        self.records=json.loads((WORK/f'{role}_index.json').read_text())
        if fixed_view_plan:assert {r['session'] for r in self.records}<=set(self.fixed_view_specs)
        if limit:
            count=min(limit,len(self.records))
            if preserve_frame0:
                mandatory=[i for i,r in enumerate(self.records) if r['frame']==0]
                assert len(mandatory)<=count,'Pilot must retain frame0 of every validation sequence'
                remaining=[i for i in range(len(self.records)) if i not in set(mandatory)]
                chosen=np.linspace(0,len(remaining)-1,count-len(mandatory),dtype=int)
                indices=sorted(mandatory+[remaining[int(i)] for i in chosen])
            else:indices=np.linspace(0,len(self.records)-1,count,dtype=int)
            self.records=[self.records[int(i)] for i in indices]
        self.augment=augment;self.flip_probability=flip_probability;self.cache=OrderedDict()
        self.norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.jitter=T.ColorJitter(.2,.2,.15,.03)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index];name=record['session'];row=record['row']
        if name not in self.cache:
            with np.load(WORK/'targets'/f'{name}.npz') as z:self.cache[name]={k:z[k] for k in z.files}
            if self.native_surface_target:
                from multihmr.track2_data import sha256
                with np.load(WORK/'native_surface_targets'/f'{name}.npz') as exact:
                    assert np.array_equal(exact['frames'],self.cache[name]['frames'])
                    assert exact['source_gt_sha256'].item()==self.cache[name]['source_sha256'].item()
                    assert exact['fitted_label_sha256'].item()==sha256(WORK/'targets'/f'{name}.npz')
                    assert exact['metric_roles_sha256'].item()==self.native_metric_roles_sha256
                    assert exact['coordinate_frame'].item()=='opencv_camera'
                    points=exact['surface_points_cam_m']
                    assert points.shape==(len(exact['frames']),91,3) and np.isfinite(points).all()
                    self.cache[name]['native_surface_points']=points
            if len(self.cache)>12:self.cache.popitem(last=False)
        z=self.cache[name]
        image_path=WORK/'target_foreground_images'/name/f'{record["frame"]:06d}.png' if self.target_foreground else WORK/'images'/name/f'{record["frame"]:06d}.jpg'
        with Image.open(image_path) as img:image=img.convert('RGB')
        w,h=image.size;assert (h,w)==tuple(z['image_shape'])
        joints=z['joints'][row];centers=z['centers'][row];rotations=z['rotations'][row];original_K=z['K']
        if self.augment and self.flip_probability and random.random()<self.flip_probability:
            image=ImageOps.mirror(image)
            joints,centers,rotations,original_K=mirror_targets(joints,centers,rotations,original_K,w)
        if self.fixed_view_specs:
            from multihmr.target_crop import prepare_native_rgb_box
            image,K,_=prepare_native_rgb_box(np.asarray(image),original_K,self.fixed_view_specs[name]['box'])
        else:
            image=ImageOps.contain(image,(896,896),method=Image.Resampling.LANCZOS)
            nw,nh=image.size;l,t=(896-nw)//2,(896-nh)//2
            image=ImageOps.expand(image,border=(l,t,896-nw-l,896-nh-t))
            K=original_K.copy();K[0]*=nw/w;K[1]*=nh/h;K[0,2]+=l;K[1,2]+=t
        if self.augment:image=self.jitter(image)
        item=dict(image=self.norm(image),K=K,original_K=original_K,image_shape=z['image_shape'],
                    joints=joints,centers=centers,valid=z['valid'][row],
                    rotations=rotations,betas=z['betas'][row],index=index)
        if 'transl' in z:item['smplx_translation']=z['transl'][row:row+1]
        if self.native_surface_target:item['native_surface_points']=z['native_surface_points'][row:row+1]
        return item


class AnchorFrames(Frames):
    """Train-only current/frame0 RGB pairs, with independent color jitter."""
    def __init__(self,role,augment=False,limit=0,flip_probability=0,target_foreground=False,fixed_view_plan=None,native_surface_target=False):
        super().__init__(role,augment=augment,limit=0,flip_probability=flip_probability,target_foreground=target_foreground,fixed_view_plan=fixed_view_plan,native_surface_target=native_surface_target)
        self.anchor_indices={r['session']:i for i,r in enumerate(self.records) if r['frame']==0}
        self.time_indices=list(range(len(self.records)))
        if limit:self.time_indices=np.linspace(0,len(self.records)-1,min(limit,len(self.records)),dtype=int).tolist()
        assert {r['session'] for r in self.records}==set(self.anchor_indices)

    def __len__(self):return len(self.time_indices)

    def __getitem__(self,index):
        row=self.time_indices[index];record=self.records[row]
        return super().__getitem__(row),super().__getitem__(self.anchor_indices[record['session']])


def collate(items):
    result={k:torch.stack([torch.as_tensor(x[k]) for x in items]) for k in ['image','K','original_K','image_shape']}
    people=max(len(x['valid']) for x in items)
    keys=['joints','centers','rotations','betas','valid']
    if 'smplx_translation' in items[0]:keys.append('smplx_translation')
    if 'native_surface_points' in items[0]:keys.append('native_surface_points')
    for key in keys:
        shape=(len(items),people,*items[0][key].shape[1:])
        a=np.zeros(shape,dtype=items[0][key].dtype)
        for i,x in enumerate(items):a[i,:len(x[key])]=np.nan_to_num(x[key])
        result[key]=torch.from_numpy(a)
    result['index']=[x['index'] for x in items]
    return result


def collate_anchors(items):
    result=collate([item[0] for item in items]+[item[1] for item in items])
    result['pairs']=len(items)
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
        total=config.get('pa_weight',100)*pa+config.get('root_weight',25)*root+detection+offset+.2*rotation+.02*shape+config.get('absolute_weight',1)*absolute
    return total,dict(loss=total,pa_mm=1000*pa,root_mm=1000*root,detection=detection,offset=offset,rotation=rotation,shape=shape,absolute=absolute)


def relative_root_motion_loss(joints,target,pairs):
    """Match camera-space root displacement from each pair's original frame0."""
    assert len(joints)==len(target)==2*pairs
    predicted_delta=joints[:pairs,0]-joints[pairs:,0]
    target_delta=target[:pairs,0]-target[pairs:,0]
    return torch.nn.functional.smooth_l1_loss(predicted_delta,target_delta,beta=.1)


@torch.no_grad()
def validate(model,loader,reg,device,cd_proxy=False,target_head_gate=False,strict_target_mask=False,target_body_overlap=False):
    model.eval();original=model.detection;image_ids=[]
    proxy=None;proxy_records=[];reconstruction_error=0.;mask_readers={}
    dataset=loader.dataset.dataset if isinstance(loader.dataset,Subset) else loader.dataset
    body_rule=None
    if target_body_overlap:
        assert target_head_gate and strict_target_mask
        body_rule=json.loads((OUT/'body_target_gate_Train_rule.json').read_text())
    if cd_proxy:
        from utils.track2_metrics import SmplxCDHProxy
        roles=json.loads((OUT/'body_roles.json').read_text())['roles']
        proxy=SmplxCDHProxy(model.smpl_layer[f'neutral_{model.num_betas}'].bm_x,device,roles)
    def detection(*args,**kwargs):
        out=original(*args,**kwargs);image_ids[:]=out[2][0].detach().cpu().tolist();return out
    model.detection=detection
    values=np.zeros(6,np.float64)
    try:
        for batch in loader:
            if proxy is not None:
                gt_points,error=proxy.targets(batch);reconstruction_error=max(reconstruction_error,error)
                if dataset.native_surface_target:gt_points=batch['native_surface_points'][:,0].numpy()
            with torch.autocast('cuda',dtype=torch.float16):
                people=model(batch['image'].to(device,non_blocking=True),K=batch['K'].to(device),is_training=False,det_thresh=.3,nms_kernel_size=3)
            assert len(people)==len(image_ids)
            selected_image_ids=image_ids
            if target_head_gate:
                import cv2
                from multihmr.track2_data import open_track2_target_mask
                from multihmr.target_crop import select_target_head
                candidates=[[] for _ in batch['index']]
                for index,person in zip(image_ids,people):candidates[index].append(person)
                selected_people=[];selected_image_ids=[]
                for index,persons in enumerate(candidates):
                    record=dataset.records[batch['index'][index]];session=record['session']
                    if session not in mask_readers:
                        mask_readers[session]=open_track2_target_mask(session)
                    cap=mask_readers[session];cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame'])
                    ok,mask=cap.read();assert ok,(session,record['frame'])
                    chosen=select_target_head(persons,batch['original_K'][index].numpy(),
                        batch['K'][index].numpy(),batch['image_shape'][index].tolist(),mask,strict_target_mask=strict_target_mask,
                        body_box_min_iou=body_rule['minimum_body_box_iou'] if body_rule else None,
                        joint_regressor=model.smpl_layer[f'neutral_{model.num_betas}'].bm_x.J_regressor)
                    if chosen is not None:selected_people.append(chosen);selected_image_ids.append(index)
                people=selected_people
            grouped=[[] for _ in batch['index']]
            surfaces=[[] for _ in batch['index']]
            for index,person in zip(selected_image_ids,people):
                grouped[index].append((reg@person['v3d'].float()).cpu().numpy())
                if proxy is not None:surfaces[index].append(proxy.points(person['v3d'].float()[None])[0].cpu().numpy())
            for i,persons in enumerate(grouped):
                pred=np.stack(persons) if persons else np.empty((0,55,3),np.float32)
                target=batch['joints'][i].numpy();valid=batch['valid'][i].numpy().astype(bool)
                K=batch['original_K'][i].numpy();shape=batch['image_shape'][i].tolist()
                mapping=({0:0} if len(pred) and valid[0] else {}) if target_head_gate else match_people(pred,target,valid,K,K,shape)
                values[4]+=valid.sum();values[5]+=len(pred)
                for gi,pi in mapping.items():
                    values[:3]+=joint_errors(pred[pi,:22],target[gi,:22]);values[3]+=1
                    if proxy is not None:
                        assert gi==0,'Track2 has one supervised person'
                        record=dataset.records[batch['index'][i]]
                        proxy_records.append(dict(session=record['session'],frame=record['frame'],
                            pred=surfaces[i][pi],target=gt_points[i]))
    finally:
        model.detection=original
        for cap in mask_readers.values():cap.release()
    tensor=torch.tensor(values,dtype=torch.float64,device=device);dist.all_reduce(tensor)
    values=tensor.cpu().numpy();assert values[3]>0
    metrics=dict(mpjpe_mm=1000*values[0]/values[3],root_mpjpe_mm=1000*values[1]/values[3],
                pa_mpjpe_mm=1000*values[2]/values[3],matched_instances=int(values[3]),valid_instances=int(values[4]),
                coverage=float(values[3]/values[4]),predicted_instances=int(values[5]))
    if target_head_gate:
        metrics.update(detection_selection='target_mask_head_highest_confidence',GT_used_for_detection_selection=False,
            evaluation_protocol='native_rgb_static_crop_target_head_v1' if dataset.fixed_view_plan else 'target_foreground_head_v1' if dataset.target_foreground else 'native_rgb_target_head_v1')
        if strict_target_mask:
            from multihmr.target_crop import STRICT_PROTOCOLS
            assert not dataset.target_foreground
            metrics.update(evaluation_protocol=STRICT_PROTOCOLS['native_rgb_static_crop' if dataset.fixed_view_plan else 'native_rgb'],
                strict_target_mask=True,target_mask_dilation_kernel=0)
        if target_body_overlap:
            from multihmr.target_crop import BODY_PROTOCOLS
            from multihmr.track2_data import sha256
            metrics.update(evaluation_protocol=BODY_PROTOCOLS['native_rgb_static_crop' if dataset.fixed_view_plan else 'native_rgb'],
                target_body_overlap=True,body_box_min_iou=body_rule['minimum_body_box_iou'],
                body_box_rule_sha256=sha256(OUT/'body_target_gate_Train_rule.json'))
    if proxy is not None:
        gathered=[None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered,proxy_records)
        metrics.update(proxy.summarize([item for records in gathered for item in records]))
        metrics['cd_proxy_target_space']='native_MHR_GT' if dataset.native_surface_target else 'fitted_SMPLX_GT'
        error=torch.tensor(reconstruction_error,device=device)
        dist.all_reduce(error,op=dist.ReduceOp.MAX)
        metrics['max_gt_reconstruction_error_m']=float(error)
    return metrics


def checkpoint(path,model,optimizer,config,step,epoch,metrics,baseline,started,full=True,scaler=None,successful_steps=0):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    value=dict(model_delta={n:v.detach().cpu() for n,v in model.state_dict().items() if n in names},
               base_checkpoint=str(ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'),
               config=config,step=step,epoch=epoch,metrics=metrics,baseline=baseline,started_unix=started,
               metrics_step=config.get('last_validation_step',step),
               saved_unix=time.time(),trainable_names=sorted(names))
    if full:
        value['optimizer']=optimizer.state_dict()
        value['successful_steps']=successful_steps
        if scaler is not None:value['amp_scaler']=scaler.state_dict()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.pt');torch.save(value,temp);temp.replace(path)


def main(config_path):
    global OUT,WORK,ASSETS
    config=json.loads(Path(config_path).read_text())
    if config.get('experiment')=='part_aware_dynamic_aggregation_20h':
        from multihmr.part_aggregation_training import train,diagnose
        return diagnose(config) if config.get('task')=='diagnose' else train(config)
    if config.get('experiment') in ('rgb_event_36h','rgb_event_new_split_24h','device34_body_28h','device34_fusion_repair_15h','device3_encoding_12h'):
        return run_event_trial(config)
    if config.get('experiment')=='track2_multihmr_12h':
        OUT=Path(config['output_root'])
        WORK=OUT/'work'
        ASSETS=OUT/'checkpoints'
        config['target_foreground']=config.get('target_foreground',True)
        assert (WORK/'complete.json').is_file(),'Track 2 preparation must finish first'
        resources=OUT/'gpu_resources.json'
        if resources.exists():
            config['distributed_backend']=json.loads(resources.read_text()).get('distributed_backend','nccl')
    overrides=OUT/'training_overrides.json'
    if overrides.exists():config.update(json.loads(overrides.read_text()).get(config['name'],{}))
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group(config.get('distributed_backend','nccl'),timeout=timedelta(minutes=20));assert dist.get_world_size()==2
    configure_torch();seed=config.get('seed',261023)
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed)
    run=config['name'];run_dir=OUT/'runs'/run;run_dir.mkdir(parents=True,exist_ok=True)
    ckpt_dir=ASSETS/run;ckpt_dir.mkdir(parents=True,exist_ok=True)
    model=load_multihmr();configure_model(model,config.get('last_blocks',0))
    if config.get('init_checkpoint'):load_delta(model,config['init_checkpoint'])
    params=[p for p in model.parameters() if p.requires_grad]
    anchors={n:p.detach().clone() for n,p in model.named_parameters()
             if p.requires_grad} if config.get('anchor_l2_weight',0) else {}
    anchor_count=sum(v.numel() for v in anchors.values())
    groups=[]
    for backbone in [False,True]:
        selected=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')==backbone]
        if selected:groups.append(dict(params=selected,lr=config['backbone_lr'] if backbone else config['head_lr'],
                                       initial_lr=config['backbone_lr'] if backbone else config['head_lr']))
    optimizer=torch.optim.AdamW(groups,weight_decay=config.get('weight_decay',.01))
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    reg=torch.from_numpy(np.load(WORK/'joint_regressor.npy')).to(device)
    pilot=config.get('pilot_steps',0)
    frame_class=AnchorFrames if config.get('pair_first_frame') else Frames
    train=frame_class(config.get('training_role','training'),augment=config.get('augment',True),limit=128 if pilot else 0,
                 flip_probability=config.get('horizontal_flip_probability',0),target_foreground=config.get('target_foreground',False),fixed_view_plan=config.get('fixed_view_plan'),native_surface_target=config.get('native_surface_target',False))
    validation=Frames('validation',limit=32 if pilot else 0,preserve_frame0=config.get('selection_metric')=='cd_h_smplx_proxy_cm',target_foreground=config.get('target_foreground',False),fixed_view_plan=config.get('fixed_view_plan'),native_surface_target=config.get('native_surface_target',False))
    sampler=DistributedSampler(train,num_replicas=2,rank=rank,shuffle=True,seed=seed,drop_last=True)
    workers=config.get('workers',2)
    train_loader=DataLoader(train,batch_size=config.get('batch_size',4),sampler=sampler,collate_fn=collate_anchors if config.get('pair_first_frame') else collate,
                            num_workers=workers,pin_memory=True,drop_last=True,persistent_workers=workers>0)
    val_loader=DataLoader(Subset(validation,list(range(rank,len(validation),2))),batch_size=16,collate_fn=collate,
                          num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    scaler=torch.cuda.amp.GradScaler(init_scale=config.get('amp_init_scale',512));step=0;epoch=0;started=time.time();best=float('inf');baseline=None;successful=0
    selection_metric=config.get('selection_metric','pa_mpjpe_mm')
    resume=ckpt_dir/'last.pt'
    if resume.exists() and not pilot:
        restored=load_delta(model,resume);optimizer.load_state_dict(restored['optimizer'])
        step=restored['step'];epoch=restored['epoch'];started=restored['started_unix'];baseline=restored['baseline']
        if 'amp_scaler' in restored:scaler.load_state_dict(restored['amp_scaler'])
        # Legacy checkpoints still contain exact Adam per-parameter update counters.
        counters=[int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
        successful=restored.get('successful_steps',min(counters,default=0))
        if (run_dir/'best.json').exists():best=json.loads((run_dir/'best.json').read_text())[selection_metric]
    duration=config.get('hours',6)*3600;finish=min(started+duration,config.get('deadline_unix',float('inf')))
    if rank==0:
        save(run_dir/'configuration.json',config)
        save(run_dir/'started.json',dict(pid=os.getpid(),started_unix=started,finish_unix=finish,
             trainable_parameters=sum(p.numel() for p in params),batch_size_per_gpu=config.get('batch_size',4),world_size=2))
    next_eval=time.time()+config.get('validation_minutes',45)*60
    fixed_full_train=config.get('fixed_full_train',False)
    train_cd_proxy=None
    if config.get('pair_first_frame'):
        from utils.track2_metrics import SmplxCDHProxy
        roles=json.loads((OUT/'body_roles.json').read_text())['roles']
        train_cd_proxy=SmplxCDHProxy(model.smpl_layer[f'neutral_{model.num_betas}'].bm_x,device,roles)
    if fixed_full_train:next_eval=finish+3600
    def measure():
        values=validate(model,val_loader,reg,device,cd_proxy=selection_metric=='cd_h_smplx_proxy_cm',
            target_head_gate=config.get('validation_target_head_gate',False),strict_target_mask=config.get('strict_target_mask',False),
            target_body_overlap=config.get('target_body_overlap',False))
        if selection_metric!='cd_h_smplx_proxy_cm':return values
        if rank==0:
            from multihmr.track2_data import sha256
            config['last_validation_step']=step
            candidate=ckpt_dir/'validation'/f'step_{step:06d}.pt'
            checkpoint(candidate,model,optimizer,config,step,epoch,values,baseline,started,full=False)
            tag=f'{run}_step_{step:06d}'
            save(OUT/'validation_smplx'/tag/'metrics.json',dict(values,checkpoint=str(candidate),
                checkpoint_sha256=sha256(candidate),step=step,evaluated_unix=time.time(),test_used=False))
        return values
    metrics=measure()
    config['last_validation_step']=step
    if baseline is None:baseline=metrics
    if rank==0:
        save(run_dir/'baseline.json',baseline)
        print(json.dumps(dict(event='baseline',run=run,**metrics)),flush=True)
        if best==float('inf'):
            best=metrics[selection_metric];save(run_dir/'best.json',dict(metrics,step=step))
            checkpoint(ckpt_dir/'best.pt',model,optimizer,config,step,epoch,metrics,baseline,started,full=False)
    dist.barrier();sampler.set_epoch(epoch);iterator=iter(train_loader)
    moving=[];train_started=time.perf_counter();start_step=step;model.train()
    last_save=time.time();stale_validations=0
    while True:
        control=torch.zeros(3,dtype=torch.float64,device=device)
        if rank==0:control[:]=torch.tensor([float(time.time()>=finish or (pilot and step>=pilot)
            or (config.get('max_steps',0) and step>=config['max_steps'])
            or (config.get('max_epochs',0) and epoch>=config['max_epochs'])
            or (config.get('early_stop_patience',0) and stale_validations>=config['early_stop_patience'])),
            float(time.time()>=next_eval),time.time()-started],device=device)
        dist.broadcast(control,src=0);stop,evaluate,elapsed=control.tolist()
        if fixed_full_train and stop:
            metrics={}
            if rank==0:
                # The internal holdout chose the recipe/update budget before
                # this full-Train refit. Its final model has no holdout score.
                checkpoint(ckpt_dir/'last.pt',model,optimizer,config,step,epoch,metrics,baseline,started,
                           scaler=scaler,successful_steps=successful)
                checkpoint(ckpt_dir/'best.pt',model,optimizer,config,step,epoch,metrics,baseline,started,full=False)
                save(run_dir/'best.json',dict(step=step,selection='fixed full-Train refit budget',
                    heldout_metrics=None,recipe_selection=config['recipe_selection']))
            dist.barrier();break
        if stop or evaluate:
            metrics=measure()
            config['last_validation_step']=step
            if rank==0:
                record=dict(event='validation',run=run,step=step,epoch=epoch,elapsed_seconds=elapsed,**metrics)
                with (run_dir/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                eligible=metrics['coverage']>=config.get('minimum_coverage',baseline['coverage']-.005)
                if eligible and metrics[selection_metric]<best:
                    stale_validations=0
                    best=metrics[selection_metric];save(run_dir/'best.json',dict(metrics,step=step))
                    checkpoint(ckpt_dir/'best.pt',model,optimizer,config,step,epoch,metrics,baseline,started,full=False)
                else:stale_validations+=1
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
        if train_cd_proxy is not None:
            from utils.track2_metrics import paired_frame0_chamfer
            with torch.autocast('cuda',enabled=False):
                with torch.no_grad():
                    gt_points=b['native_surface_points'][:,0] if config.get('native_surface_target',False) else train_cd_proxy.targets(batch,return_torch=True)[0]
                pred_points=train_cd_proxy.points(output['v3d'].float())
                assert len(pred_points)==2*batch['pairs'],'Both supervised pair frames must have visible centers'
                cd=paired_frame0_chamfer(pred_points,gt_points,batch['pairs'])
                loss=loss+config.get('paired_cd_weight',10)*cd
                components['paired_cd_cm']=cd*100;components['loss']=loss
                if config.get('relative_translation_weight',0):
                    joints=reg@output['v3d'].float();n=batch['pairs']
                    relative=relative_root_motion_loss(joints,gt['joints'],n)
                    loss=loss+config['relative_translation_weight']*relative
                    components['relative_translation']=relative;components['loss']=loss
        if anchors:
            penalty=sum((p-anchors[n]).square().sum() for n,p in model.named_parameters()
                if n in anchors)/anchor_count
            loss=loss+config['anchor_l2_weight']*penalty
            components['anchor_l2']=penalty
            components['loss']=loss
        assert torch.isfinite(loss).item(),{k:float(v.detach()) for k,v in components.items()}
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        successful+=int(scaler.get_scale()>=before);step+=1
        if rank==0 and time.time()-last_save>=300:
            # This periodic checkpoint deliberately records which validation step
            # its metrics describe, rather than presenting them as fresh metrics.
            checkpoint(ckpt_dir/'last.pt',model,optimizer,config,step,epoch,metrics,baseline,started,
                       scaler=scaler,successful_steps=successful)
            last_save=time.time()
        moving.append({k:float(v.detach()) for k,v in components.items()})
        if step%25==0 or pilot:
            if rank==0:
                record=dict(event='training',run=run,step=step,epoch=epoch,elapsed_seconds=time.time()-started,
                    images_per_second=((step-start_step)*config.get('batch_size',4)*2*(2 if config.get('pair_first_frame') else 1))/max(time.perf_counter()-train_started,1e-6),
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,grad_norm=float(norm) if torch.isfinite(norm) else None,amp_scale=scaler.get_scale(),
                    successful_steps=successful,**{k:float(np.mean([v[k] for v in moving])) for k in moving[0]})
                print(json.dumps(record),flush=True);save(run_dir/'progress.json',record)
            moving=[]
    if rank==0:
        assert successful>0,'No optimizer steps succeeded'
        save(run_dir/'complete.json',dict(status='pilot_complete' if pilot else 'complete',run=run,step=step,
             successful_steps=successful,elapsed_seconds=time.time()-started,best_pa_mpjpe_mm=None if fixed_full_train else json.loads((run_dir/'best.json').read_text())['pa_mpjpe_mm'],
             selection_metric=selection_metric,best_selection_value=None if fixed_full_train else best,
             validation_used_for_selection=not fixed_full_train,training_role=config.get('training_role','training'),
             last_metrics=metrics,checkpoint=str(ckpt_dir/'best.pt'),completed_unix=time.time()))
    dist.barrier();dist.destroy_process_group()


@torch.no_grad()
def validate_event(model,loader,reg,device):
    """Actual detections on Train holdout; no GT boxes/indices enter inference."""
    from multihmr.event_data import model_inputs
    if hasattr(model,'input_keys'):
        from multihmr.multisensor_data import input_keys
        model_inputs=lambda batch,mode:input_keys(batch)
    model.eval();original=model.base.detection;image_ids=[]
    def detection(*args,**kwargs):
        value=original(*args,**kwargs);image_ids[:]=value[2][0].detach().cpu().tolist();return value
    model.base.detection=detection
    total=np.zeros(6,np.float64)
    try:
        for batch in loader:
            inputs={k:v.to(device,non_blocking=True) for k,v in model_inputs(batch,model.mode).items()}
            with torch.autocast('cuda',dtype=torch.float16):
                people=model(**inputs,is_training=False,det_thresh=.3,nms_kernel_size=3)
            assert len(people)==len(image_ids)
            grouped=[[] for _ in batch['index']]
            with torch.autocast('cuda',enabled=False):
                for image,person in zip(image_ids,people):
                    grouped[image].append((reg@person['v3d'].float()).cpu().numpy())
            for image,predictions in enumerate(grouped):
                target=batch['joints'][image].numpy();valid=batch['valid'][image].numpy().astype(bool)
                K=batch['original_K'][image].numpy();shape=batch['image_shape'][image].tolist()
                predictions=np.stack(predictions) if predictions else np.empty((0,55,3),np.float32)
                mapping=match_people(predictions,target,valid,K,K,shape)
                for gt_id,pred_id in mapping.items():
                    total[:3]+=joint_errors(predictions[pred_id,:22],target[gt_id,:22])
                total[3]+=len(mapping);total[4]+=valid.sum();total[5]+=len(predictions)
    finally:
        model.base.detection=original
    return dict(zip(['mpjpe_mm','root_mpjpe_mm','pa_mpjpe_mm'],
                    (1000*total[:3]/total[3]).tolist() if total[3] else [None]*3),
                matched_instances=int(total[3]),valid_instances=int(total[4]),
                predicted_instances=int(total[5]),coverage=float(total[3]/total[4]) if total[4] else 0.)


def event_selection_rank(metrics,minimum_coverage=.95):
    if metrics['pa_mpjpe_mm'] is None:
        return (-1,metrics['coverage'],0)
    if metrics['coverage']>=minimum_coverage:
        return (1,-metrics['pa_mpjpe_mm'],metrics['coverage'])
    return (0,metrics['coverage'],-metrics['pa_mpjpe_mm'])


def run_event_trial(config):
    """Single-GPU, half-day controlled RGB/event adaptation with safe resume."""
    global WORK
    import fcntl
    import hashlib
    import signal
    from multihmr.event_model import EventMultiHMR
    from multihmr.event_data import Frames as EventFrames,collate as event_collate,model_inputs,OUT as EVENT_OUT
    from utils.event_encoding import ENCODINGS
    repair=config.get('experiment') in ('device34_fusion_repair_15h','device3_encoding_12h')
    if config.get('experiment')=='device34_body_28h' or repair:
        from functools import partial
        from multihmr.multisensor_data import Frames as SensorFrames,collate as event_collate,OUT as EVENT_OUT,input_keys
        from multihmr.multisensor_model import MultiSensorMultiHMR
        EventFrames=partial(SensorFrames,condition=config['condition'])
        if repair:
            from multihmr.multisensor_repair import build_repair_model,REPAIR_LAYOUTS,load_verified_initialization,implementation_provenance
            MultiSensorMultiHMR=lambda base,condition:build_repair_model(base,config)
            layout=REPAIR_LAYOUTS[config['condition']]
            assert layout==(config['primary_sensor'],config['auxiliary_sensor'])
            EventFrames=partial(SensorFrames,condition=config['condition'],sensor_layout=layout)
        model_inputs=lambda batch,mode:input_keys(batch)
        WORK=Path(config['output_root'])
        assert EVENT_OUT.resolve()==WORK.resolve()
        assert (config.get('experiment')=='device3_encoding_12h' and config['condition']=='rgb3' and config['hours']==1.5) or 3<=config['hours']<=4
        assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==config['split_sha256']
    if config.get('experiment') == 'rgb_event_new_split_24h':
        import hashlib
        WORK=Path(config['rgb_cache_root'])
        assert EVENT_OUT.resolve()==Path(config['output_root']).resolve()
        assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==config['split_sha256']
        assert 3 <= config['hours'] <= 4
    visible=[value for value in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if value]
    physical_gpu=int(os.environ.get('EVENTPOSE_PHYSICAL_GPU',visible[0] if visible and visible[0].isdigit() else '-1'))
    allowed=config['allowed_gpus'] if repair else list(range(2,7))
    assert len(visible)==1 and physical_gpu in allowed,'Set exactly one authorized physical GPU'
    assert config['mode'] in ('rgb','event','fusion')
    run_dir=EVENT_OUT/'runs'/config['name'];run_dir.mkdir(parents=True,exist_ok=True)
    assets=EVENT_OUT/'checkpoints'/config['name'];assets.mkdir(parents=True,exist_ok=True)
    import hashlib
    lock_path=ROOT/'.experiment_storage/run_locks'/hashlib.sha256(str(EVENT_OUT).encode()).hexdigest()[:16]/config['name']
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run_dir/'complete.json').exists(),'This trial already completed; refusing duplicate training'
    configure_torch();torch.cuda.set_device(0);device=torch.device('cuda',0)
    seed=config.get('seed',261024);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    started=config.get('trial_started_unix',time.time());pilot=config.get('pilot_steps',0)
    finish=min(started+config.get('hours',12)*3600,config['deadline_unix']-300)
    assert finish>started,'No training budget remains'
    base=load_multihmr();configure_model(base,config.get('last_blocks',8))
    if config.get('experiment')=='device34_body_28h' or repair:
        from multihmr.multisensor_model import checkpoint_frozen_blocks
        checkpoint_frozen_blocks(base,config.get('last_blocks',8))
    channels=ENCODINGS[config['encoding']]['channels'] if config['mode']!='rgb' else 0
    model=(MultiSensorMultiHMR(base,config['condition']) if config.get('experiment')=='device34_body_28h' or repair
           else EventMultiHMR(base,config['mode'],channels)).to(device)
    initialization=load_verified_initialization(model,config) if repair else None
    # Modality-specific module initialization must not change the shared RGB
    # augmentation/random stream (especially pilot runs with zero data workers).
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    selected={name:parameter for name,parameter in model.named_parameters() if parameter.requires_grad}
    groups=[]
    def parameter_group(name):
        if name.startswith('base.backbone.'):return 'backbone'
        if repair and name.startswith('adapter.'):return 'adapter'
        if repair and name.startswith('fusion.'):return 'fusion'
        return 'head'
    for group_name in ('head','backbone','adapter','fusion') if repair else ('head','backbone'):
        values=[parameter for name,parameter in selected.items() if parameter_group(name)==group_name]
        if values:
            rate=config.get(group_name+'_lr',1e-6 if group_name=='backbone' else 1e-5)
            groups.append(dict(params=values,lr=rate,initial_lr=rate,group_name=group_name))
    optimizer=torch.optim.AdamW(groups,weight_decay=config.get('weight_decay',.01))
    scaler=torch.cuda.amp.GradScaler(init_scale=512)
    train=EventFrames('training',config['mode'],config.get('encoding'),augment=config.get('augment',True),
                      limit=config.get('pilot_train_limit',128) if pilot else 0,
                      sessions=config.get('pilot_training_sessions') if pilot else None)
    validation=EventFrames('validation',config['mode'],config.get('encoding'),
                           limit=config.get('pilot_validation_limit',32) if pilot else 0,
                           sessions=config.get('pilot_validation_sessions') if pilot else None)
    workers=config.get('workers',2);batch_size=config.get('batch_size',16)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=batch_size,shuffle=True,generator=generator,collate_fn=event_collate,
                      num_workers=workers,pin_memory=True,drop_last=True,persistent_workers=workers>0)
    val_loader=DataLoader(validation,batch_size=batch_size,collate_fn=event_collate,num_workers=workers,
                         pin_memory=True,persistent_workers=workers>0)
    reg=torch.from_numpy(np.load(WORK/'joint_regressor.npy')).to(device)
    step=successful=epoch=0;best_metrics=None;best_rank=(-2,0,0);baseline=None
    seen_indices=set(); samples_seen=0
    resume=assets/'last.pt'
    if resume.exists() and not pilot:
        restored=torch.load(resume,map_location='cpu')
        from utils.training_resume import resume_clock
        started,finish=resume_clock(config,restored,time.time())
        unexpected=model.load_state_dict(restored['model_delta'],strict=False).unexpected_keys
        assert not unexpected
        optimizer.load_state_dict(restored['optimizer']);scaler.load_state_dict(restored['amp_scaler'])
        step=restored['step'];successful=restored['successful_steps'];epoch=restored['epoch']
        seen_indices=set(restored.get('seen_indices',[]));samples_seen=restored.get('samples_seen',step*batch_size)
        # resume_clock preserves pre-pause work and the newly authorized wall limit.
        baseline=restored['baseline'];best_metrics=restored['best_metrics']
        if best_metrics is not None:best_rank=event_selection_rank(best_metrics,config.get('minimum_coverage',.95))
        random.setstate(restored['random_state']);np.random.set_state(restored['numpy_state'])
        torch.set_rng_state(restored['torch_rng_state']);torch.cuda.set_rng_state(restored['cuda_rng_state'])
        generator.set_state(restored['sampler_generator_state'])
    save(run_dir/'configuration.json',config)
    save(run_dir/'started.json',dict(pid=os.getpid(),started_unix=started,finish_unix=finish,
        physical_gpu=physical_gpu,mode=config['mode'],encoding=config.get('encoding'),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        training_frames=len(train),validation_frames=len(validation),batch_size=batch_size,
        condition=config.get('condition'),total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in selected.values()),
        initialization=initialization or 'original pretrained Multi-HMR, not the previous RGB-finetuned model',
        implementation_sha256=implementation_provenance(WORK/'source_snapshots') if repair else None,
        backbone_warmup_seconds=config.get('backbone_warmup_seconds',1800),test_used_for_selection=False))
    def checkpoint(label,metrics):
        value=dict(model_delta={name:parameter.detach().cpu() for name,parameter in selected.items()},
            config=config,step=step,successful_steps=successful,epoch=epoch,started_unix=started,
            samples_seen=samples_seen,unique_frames_seen=len(seen_indices),seen_indices=sorted(seen_indices),
            training_elapsed_seconds=time.time()-started,
            resume_generation=config.get('resume_authorization',{}).get('generation'),
            metrics=metrics,metrics_step=metrics_step,metrics_apply_to_checkpoint=metrics_step==step,
            best_metrics=best_metrics,baseline=baseline,optimizer=optimizer.state_dict() if not pilot else None,
            amp_scaler=scaler.state_dict(),random_state=random.getstate(),numpy_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
            sampler_generator_state=generator.get_state(),test_used_for_selection=False)
        destination=assets/(label+'.pt');temporary=destination.with_suffix('.tmp.pt')
        torch.save(value,temporary);temporary.replace(destination)
    interrupted=[False]
    def stop_signal(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    metrics=validate_event(model,val_loader,reg,device);metrics_step=step
    if baseline is None:baseline=metrics
    save(run_dir/'baseline.json',baseline)
    last_validation=last_save=time.time();model.train()
    while time.time()<finish and not interrupted[0] and not (pilot and step>=pilot):
        epoch_batches=0
        for batch in loader:
            elapsed=time.time()-started
            if time.time()>=finish or interrupted[0] or (pilot and step>=pilot):break
            for name,parameter in selected.items():
                if name.startswith('base.backbone.'):
                    parameter.requires_grad_(elapsed>=config.get('backbone_warmup_seconds',1800))
            b,gt=targets(batch,device)
            samples_seen+=len(batch['index']);seen_indices.update(batch['index']);epoch_batches+=1
            fraction=min(1.,elapsed/(config.get('hours',12)*3600))
            rate_scale=min(1.,(step+1)/config.get('warmup_steps',100))*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*rate_scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                output=model(**model_inputs(b,model.mode),idx=gt['idx'],is_training=True)
            loss,components=losses(output,gt,reg,config)
            if repair and hasattr(model,'fusion'):
                penalty=model.fusion.residual_energy*config.get('residual_regularization',0.)
                loss=loss+penalty;components['fusion_residual_penalty']=penalty
            assert torch.isfinite(loss),{key:float(value.detach()) for key,value in components.items()}
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(list(selected.values()),1.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            successful+=int(scaler.get_scale()>=before);step+=1
            if step%25==0 or pilot:
                gradient_diagnostics={}
                if repair:
                    for label,prefix in [('adapter','adapter.'),('fusion_encoder','fusion.encoder.'),('fusion_output','fusion.output.')]:
                        pieces=[p.grad.detach().float().square().sum() for n,p in selected.items()
                                if n.startswith(prefix) and p.grad is not None]
                        gradient_diagnostics[label+'_gradient_l2']=float(torch.stack(pieces).sum().sqrt()) if pieces else None
                progress=dict(step=step,successful_steps=successful,elapsed_seconds=time.time()-started,
                    samples_seen=samples_seen,unique_frames_seen=len(seen_indices),completed_subset_epochs=epoch,
                    images_per_second=step*batch_size/max(time.time()-started,1e-6),
                    grad_norm=float(norm) if torch.isfinite(norm) else None,amp_scale=scaler.get_scale(),
                    peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,updated_unix=time.time(),
                    **{key:float(value.detach()) for key,value in components.items()},**gradient_diagnostics)
                save(run_dir/'progress.json',progress);print(json.dumps(progress),flush=True)
                if repair and pilot:
                    with (run_dir/'pilot_gradient_diagnostics.jsonl').open('a') as stream:
                        stream.write(json.dumps(progress)+'\n')
            if time.time()-last_save>=300:
                checkpoint('last',metrics);last_save=time.time()
            if time.time()-last_validation>=config.get('validation_minutes',45)*60:
                metrics=validate_event(model,val_loader,reg,device);metrics_step=step;last_validation=time.time()
                record=dict(metrics,step=step,successful_steps=successful,time=time.time())
                with (run_dir/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                rank=event_selection_rank(metrics,config.get('minimum_coverage',.95))
                if rank>best_rank:
                    best_rank=rank;best_metrics=metrics;checkpoint('best',metrics);save(run_dir/'best.json',record)
                checkpoint('last',metrics);last_save=time.time();model.train()
        if epoch_batches==len(loader):epoch+=1
    assert successful>0,'No optimizer update succeeded'
    # Save before a final holdout pass so deadline interruption cannot lose progress.
    checkpoint('last',metrics)
    if time.time()<config['deadline_unix']-120 and not interrupted[0]:
        metrics=validate_event(model,val_loader,reg,device);metrics_step=step
        rank=event_selection_rank(metrics,config.get('minimum_coverage',.95))
        record=dict(metrics,step=step,successful_steps=successful,time=time.time())
        if rank>best_rank:
            best_rank=rank;best_metrics=metrics;checkpoint('best',metrics);save(run_dir/'best.json',record)
        checkpoint('last',metrics)
    save(run_dir/'complete.json',dict(status='pilot_complete' if pilot else 'interrupted' if interrupted[0] else 'complete',
        step=step,successful_steps=successful,elapsed_seconds=time.time()-started,best_metrics=best_metrics,
        samples_seen=samples_seen,unique_frames_seen=len(seen_indices),completed_subset_epochs=epoch,
        subset_frames=len(train),drop_last_frames_per_epoch=len(train)%batch_size,
        completed_unix=time.time(),checkpoint=str(assets/'best.pt'),test_used_for_selection=False))
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)

"""Controlled single-GPU training for calibrated part-aware event memory."""
import json,math,os,random,signal,time,fcntl,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .part_aggregation_model import model_inputs,PART_NAMES
from .part_aggregation_factory import build_part_model

ROOT=Path(__file__).resolve().parents[2]
_TRAINING_SOURCE=Path(__file__).read_bytes()
_TRAINING_SHA256=hashlib.sha256(_TRAINING_SOURCE).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

@torch.no_grad()
def validate(model,loader,reg,device):
    from utils.humanpose_metrics import match_people,joint_errors
    model.eval();original=model.base.detection;ids=[];values=np.zeros(6,np.float64)
    time_weights=[];null_weights=[];motion_errors=np.zeros(3,np.float64)
    def detection(*a,**kw):
        out=original(*a,**kw);ids[:]=out[2][0].detach().cpu().tolist();return out
    model.base.detection=detection
    try:
        for batch in loader:
            inputs={k:v.to(device,non_blocking=True) for k,v in model_inputs(batch).items()}
            with torch.autocast('cuda',dtype=torch.float16):
                people=model(**inputs,is_training=False,det_thresh=.3,nms_kernel_size=3)
            assert len(people)==len(ids)
            grouped=[[] for _ in batch['index']]
            for i,p in zip(ids,people):grouped[i].append((reg@p['v3d'].float()).cpu().numpy())
            for i,pp in enumerate(grouped):
                target=batch['joints'][i].numpy();valid=batch['valid'][i].numpy().astype(bool)
                K=batch['original_K'][i].numpy();shape=batch['image_shape'][i].tolist()
                pp=np.stack(pp) if pp else np.empty((0,55,3),np.float32)
                matches=match_people(pp,target,valid,K,K,shape)
                for gt_id,pred_id in matches.items():values[:3]+=joint_errors(pp[pred_id,:22],target[gt_id,:22])
                values[3]+=len(matches);values[4]+=valid.sum();values[5]+=len(pp)
            if model.fusion.last_time_weights is not None:
                time_weights.append(model.fusion.last_time_weights.detach().float().cpu().numpy())
                null_weights.append(model.fusion.last_null_weights.detach().float().cpu().numpy())
            if getattr(model.fusion.time_score,'last_predicted_path',None) is not None and 'motion_gt_valid' in batch:
                path=model.fusion.time_score.last_predicted_path.detach().float().cpu().numpy()
                uv=model.fusion.time_score.last_predicted_uv.detach().float().cpu().numpy()
                target=batch['motion_gt_path_px'].numpy();target_uv=batch['motion_gt_uv_px'].numpy()
                mask=batch['motion_gt_valid'].numpy().astype(bool)&np.isfinite(target)&np.isfinite(target_uv).all(-1)
                motion_errors+=np.array([np.abs(path[mask]-target[mask]).sum(),np.linalg.norm(uv[mask]-target_uv[mask],axis=-1).sum(),mask.sum()])
    finally:model.base.detection=original
    return dict(zip(['mpjpe_mm','root_mpjpe_mm','pa_mpjpe_mm'],
                    (values[:3]*1000/values[3]).tolist() if values[3] else [None]*3),
                coverage=float(values[3]/max(values[4],1)),matched_instances=int(values[3]),
                valid_instances=int(values[4]),predicted_instances=int(values[5]),
                part_names=PART_NAMES,mean_time_weights=np.concatenate(time_weights).mean(0).tolist() if time_weights else None,
                mean_null_weights=np.concatenate(null_weights).mean(0).tolist() if null_weights else None,
                motion_path_mae_px=float(motion_errors[0]/motion_errors[2]) if motion_errors[2] else None,
                motion_uv_endpoint_error_px=float(motion_errors[1]/motion_errors[2]) if motion_errors[2] else None,
                motion_labelled_part_cells=int(motion_errors[2]))

def rank(metrics):
    if metrics['pa_mpjpe_mm'] is None:return(-1,metrics['coverage'],0)
    if metrics['coverage']<.95:return(0,metrics['coverage'],-metrics['pa_mpjpe_mm'])
    return(1,-metrics['pa_mpjpe_mm'],metrics['coverage'])

def diagnose(config):
    """Frozen-checkpoint Train-holdout interventions; never a Test headline."""
    from .part_aggregation_data import PartAggregationFrames
    if config.get('architecture_version') in ('calibrated_part_motion_horizon_v3','calibrated_part_frozen_rgb_v4','calibrated_part_dense_frozen_rgb_v5','calibrated_part_semantic_frozen_rgb_v6'):
        from .part_aggregation_data_v3 import PartMotionFrames as PartAggregationFrames
    from .multisensor_data import collate
    torch.set_num_threads(2);torch.cuda.set_device(0)
    out=Path(config['output_root']);checkpoint=out/'checkpoints'/config['name']/'best.pt'
    model=build_part_model(config,'cuda',checkpoint).eval()
    if hasattr(model.fusion,'collect_support_diagnostics_in_eval'):model.fusion.collect_support_diagnostics_in_eval=True
    valset=PartAggregationFrames('validation',source_root=config['source_root'],cache_root=out)
    loader=DataLoader(valset,batch_size=config.get('batch_size',8),collate_fn=collate,num_workers=2,pin_memory=True)
    reg=torch.from_numpy(np.load(Path(config['source_root'])/'joint_regressor.npy')).cuda()
    class Intervention:
        def __init__(self,control):self.control=control
        def __iter__(self):
            for batch in loader:
                if time.time()>=config['deadline_unix']-90:raise TimeoutError('Holdout diagnosis fixed deadline')
                if self.control in ('zero_event','missing_event'):
                    batch['event_bins']=torch.zeros_like(batch['event_bins'])
                    if self.control=='missing_event':
                        batch['event_bin_available']=torch.zeros_like(batch['event_bin_available']);batch['auxiliary_available']=torch.zeros_like(batch['auxiliary_available'])
                elif self.control=='shuffle_event':batch['event_bins']=batch['event_bins'].roll(1,0)
                elif self.control=='reverse_time':batch['event_bins']=batch['event_bins'].flip(1)
                yield batch
    result={}
    for control in ('intact','zero_event','missing_event','shuffle_event','reverse_time'):
        result[control]=validate(model,Intervention(control),reg,'cuda')
    # Preserve actual per-query maps/weights for a separate GT-motion diagnosis.
    per_time=[];per_null=[];per_maps=[];indices=[];support_diagnostics=[];support_maps=[];motion_path=[];motion_uv=[];horizons=[]
    with torch.no_grad():
        for batch in loader:
            if time.time()>=config['deadline_unix']-90:raise TimeoutError('Attention export fixed deadline')
            inputs={k:v.cuda(non_blocking=True) for k,v in model_inputs(batch).items()}
            with torch.autocast('cuda',dtype=torch.float16):model(**inputs,is_training=False,det_thresh=.3,nms_kernel_size=3)
            per_time.append(model.fusion.last_time_weights.detach().float().cpu().numpy())
            per_null.append(model.fusion.last_null_weights.detach().float().cpu().numpy())
            per_maps.append(model.fusion.last_part_maps.detach().float().cpu().numpy());indices.extend(batch['index'])
            if getattr(model.fusion,'last_spatial_support',None) is not None:
                support_maps.append(model.fusion.last_spatial_support.detach().float().cpu().numpy())
                support_diagnostics.append(model.fusion.last_support_diagnostics)
            if getattr(model.fusion.time_score,'last_predicted_path',None) is not None:
                motion_path.append(model.fusion.time_score.last_predicted_path.detach().float().cpu().numpy())
                motion_uv.append(model.fusion.time_score.last_predicted_uv.detach().float().cpu().numpy())
                horizons.append(model.fusion.time_score.last_horizon_ms.detach().float().cpu().numpy())
    dest=out/'diagnostics';dest.mkdir(exist_ok=True)
    tw=np.concatenate(per_time);nw=np.concatenate(per_null);mass=tw.sum(-1)
    numerator=(tw*np.array([16.5,66.5,150.],np.float32)).sum(-1)
    age=np.divide(numerator,mass,out=np.full_like(numerator,np.nan),where=mass>=.05)
    records=[valset.records[i] for i in indices]
    np.savez_compressed(dest/(config['name']+'_attention.npz'),time_weights=tw,null_weights=nw,
        event_probability=mass,conditional_bin_midpoint_age_ms=age,age_score_numerator_ms=numerator,
        part_maps=np.concatenate(per_maps),dataset_indices=np.array(indices),
        predicted_spatial_support=np.concatenate(support_maps) if support_maps else np.empty((0,)),
        predicted_motion_path_px=np.concatenate(motion_path) if motion_path else np.empty((0,)),
        predicted_motion_uv_px=np.concatenate(motion_uv) if motion_uv else np.empty((0,)),
        predicted_horizon_ms=np.concatenate(horizons) if horizons else np.empty((0,)),
        sessions=np.array([r['session'] for r in records]),frames=np.array([r['frame'] for r in records]),
        part_names=np.array(PART_NAMES),bin_midpoints_ms=np.array([16.5,66.5,150.]),
        minimum_event_probability_for_age=.05,
        interpretation='Bin-centre proxy conditional on event selection; null has no age; not actual mean event timestamp')
    write(out/'diagnostics'/(config['name']+'.json'),dict(name=config['name'],checkpoint=str(checkpoint),split='Train internal holdout; NOT Test',frames=len(valset),controls=result,predicted_density_residual_diagnostics=support_diagnostics,Test_used=False,completed_unix=time.time()))
    return result

def train(config):
    from .part_aggregation_data import PartAggregationFrames
    if config.get('architecture_version') in ('calibrated_part_motion_horizon_v3','calibrated_part_frozen_rgb_v4','calibrated_part_dense_frozen_rgb_v5','calibrated_part_semantic_frozen_rgb_v6'):
        from .part_aggregation_data_v3 import PartMotionFrames as PartAggregationFrames
    from .multisensor_data import collate
    from scripts.train_multihmr import configure_model,targets,losses
    config=dict(config,runtime_training_source_sha256=_TRAINING_SHA256)
    sourcefolder=Path(config['output_root'])/'source_snapshots';sourcefolder.mkdir(exist_ok=True)
    snapshot=sourcefolder/(_TRAINING_SHA256+'.txt')
    if not snapshot.exists():snapshot.write_bytes(_TRAINING_SOURCE)
    torch.set_num_threads(2);torch.cuda.set_device(0);device='cuda'
    physical=int(os.environ['EVENTPOSE_PHYSICAL_GPU']);assert physical in config['allowed_gpus']
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    out=Path(config['output_root']);run=out/'runs'/config['name'];assets=out/'checkpoints'/config['name']
    run.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    lockpath=ROOT/'.experiment_storage/part_aggregation_locks'/config['name'];lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'complete.json').exists()
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==config['split_sha256']
    model=build_part_model(config,device)
    if config.get('freeze_RGB_base',False):model.base.requires_grad_(False)
    else:configure_model(model.base,config.get('last_blocks',4))
    selected={n:p for n,p in model.named_parameters() if p.requires_grad}
    groups=[]
    gate_prefix=('fusion.time_score.','fusion.null_score.')
    for label,prefix,lr in [('backbone','base.backbone.',config.get('backbone_lr',1e-6)),
                            ('fusion','fusion.',config.get('fusion_lr',3e-4)),
                            ('gate',gate_prefix,config.get('gate_lr',config.get('fusion_lr',3e-4))),
                            ('head',None,config.get('head_lr',1e-5))]:
        if label=='gate' and 'gate_lr' not in config:continue
        params=[p for n,p in selected.items() if n.startswith(prefix)] if prefix else [p for n,p in selected.items() if not n.startswith(('base.backbone.','fusion.'))]
        if label=='fusion' and 'gate_lr' in config:params=[p for n,p in selected.items() if n.startswith('fusion.') and not n.startswith(gate_prefix)]
        if params:groups.append(dict(params=params,lr=lr,initial_lr=lr,group_name=label))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    pilot=config.get('pilot_steps',0)
    trainset=PartAggregationFrames('training',source_root=config['source_root'],cache_root=out,
        augment=True,limit=config.get('pilot_train_limit',32) if pilot else 0,
        sessions=config.get('pilot_training_sessions') if pilot else None,use_events=config['variant']!='rgb_only')
    valset=PartAggregationFrames('validation',source_root=config['source_root'],cache_root=out,
        limit=config.get('pilot_validation_limit',16) if pilot else 0,
        sessions=config.get('pilot_validation_sessions') if pilot else None,use_events=config['variant']!='rgb_only')
    # Reset the data/augmentation stream after architecture initialization.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed);workers=config.get('workers',2);bs=config.get('batch_size',8)
    trainloader=DataLoader(trainset,batch_size=bs,shuffle=True,generator=generator,collate_fn=collate,num_workers=workers,pin_memory=True,drop_last=True,persistent_workers=workers>0)
    valloader=DataLoader(valset,batch_size=bs,collate_fn=collate,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    reg=torch.from_numpy(np.load(Path(config['source_root'])/'joint_regressor.npy')).to(device)
    started=time.time();finish=min(started+config.get('hours',3)*3600,config['deadline_unix']-300)
    target_steps=pilot or config.get('max_successful_steps',3000)
    interrupted=[False];signal.signal(signal.SIGTERM,lambda *_:interrupted.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:interrupted.__setitem__(0,True))
    step=successful=epoch=0;seen=set();metrics_step=0;best=None;best_rank=(-2,0,0)
    metrics=validate(model,valloader,reg,device);baseline=metrics
    write(run/'configuration.json',config)
    write(run/'started.json',dict(pid=os.getpid(),physical_gpu=physical,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],started_unix=started,finish_unix=finish,global_deadline_unix=config['deadline_unix'],training_frames=len(trainset),validation_frames=len(valset),trainable_parameters=sum(p.numel() for p in selected.values()),initialization=config['rgb_initialization_checkpoint'],same_device=True,test_used_for_selection=False))
    write(run/'baseline.json',baseline)
    def checkpoint(label):
        dest=assets/(label+'.pt');tmp=dest.with_suffix('.tmp.pt')
        torch.save(dict(model_delta={n:p.detach().cpu() for n,p in selected.items()},optimizer=optimizer.state_dict(),amp_scaler=scaler.state_dict(),config=config,step=step,successful_steps=successful,epoch=epoch,seen_indices=sorted(seen),training_elapsed_seconds=time.time()-started,metrics=metrics,metrics_step=metrics_step,metrics_apply_to_checkpoint=metrics_step==step,baseline=baseline,best_metrics=best,uniform_aggregation_warmup_steps=config.get('uniform_aggregation_warmup_steps',0),checkpoint_after_uniform_warmup=successful>=config.get('uniform_aggregation_warmup_steps',0),inference_uniform_warmup_disabled=True,test_used_for_selection=False),tmp);tmp.replace(dest)
    def measured():
        nonlocal metrics,metrics_step,best,best_rank
        metrics=validate(model,valloader,reg,device);metrics_step=step
        record=dict(metrics,step=step,successful_steps=successful,time=time.time())
        with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if rank(metrics)>best_rank and successful>=config.get('uniform_aggregation_warmup_steps',0):
            best_rank=rank(metrics);best=metrics;checkpoint('best');write(run/'best.json',record)
        checkpoint('last');model.train()
    last_validation=last_save=time.time();model.train()
    while time.time()<finish and successful<target_steps and not interrupted[0]:
        batches=0
        for batch in trainloader:
            if time.time()>=finish or successful>=target_steps or interrupted[0]:break
            b,gt=targets(batch,device);optimizer.zero_grad(set_to_none=True)
            if hasattr(model.fusion,'set_training_step'):model.fusion.set_training_step(successful)
            fraction=min(1,successful/target_steps)
            scale=min(1,(step+1)/config.get('warmup_steps',100))*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*scale
            with torch.autocast('cuda',dtype=torch.float16):
                output=model(**model_inputs(b),idx=gt['idx'],is_training=True)
            loss,components=losses(output,gt,reg,config)
            if config.get('part_supervision_weight',.1)>0:
                partloss=model.fusion.part_supervision(b['joints'][:,:,:22],b['K'],b['valid'].bool())
                loss=loss+config.get('part_supervision_weight',.1)*partloss;components['part_map_loss']=partloss
            loss=loss+config.get('residual_regularization',.01)*model.fusion.residual_energy
            if config.get('motion_aux_weight',0)>0:
                motion_loss,motion_components=model.fusion.motion_supervision(b)
                loss=loss+config['motion_aux_weight']*motion_loss;components.update(motion_components)
            assert torch.isfinite(loss),{k:float(v.detach()) for k,v in components.items()}
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(list(selected.values()),1.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            successful+=int(scaler.get_scale()>=before);step+=1;seen.update(batch['index']);batches+=1
            if pilot or step%25==0:
                diagnostics={}
                for label,prefix in [('event_encoder','fusion.encoder.'),('spatial_transformer','fusion.spatial_transformer.'),('time_score','fusion.time_score.'),('output','fusion.output.')]:
                    pieces=[p.grad.detach().float().square().sum() for n,p in selected.items() if n.startswith(prefix) and p.grad is not None]
                    diagnostics[label+'_gradient_l2']=float(torch.stack(pieces).sum().sqrt()) if pieces else 0.
                if config.get('architecture_version') in ('calibrated_part_motion_horizon_v3','calibrated_part_frozen_rgb_v4','calibrated_part_dense_frozen_rgb_v5','calibrated_part_semantic_frozen_rgb_v6'):
                    for label,prefix in [('motion_head','fusion.time_score.motion_head.'),('event_use_head','fusion.time_score.quality_head.')]:
                        pieces=[p.grad.detach().float().square().sum() for n,p in selected.items() if n.startswith(prefix) and p.grad is not None]
                        diagnostics[label+'_gradient_l2']=float(torch.stack(pieces).sum().sqrt()) if pieces else 0.
                weights=model.fusion.last_time_weights.detach().float()
                progress=dict(step=step,successful_steps=successful,elapsed_seconds=time.time()-started,unique_frames_seen=len(seen),grad_norm=float(norm) if torch.isfinite(norm) else None,amp_scale=scaler.get_scale(),peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,part_time_weight_std=float(weights.std(1).mean()),time_weights_mean=weights.mean(0).cpu().tolist(),updated_unix=time.time(),**{k:float(v.detach()) for k,v in components.items()},**diagnostics)
                if hasattr(model.fusion,'last_support_diagnostics'):progress.update(model.fusion.last_support_diagnostics)
                if hasattr(model.fusion,'last_warmup_active'):progress['uniform_warmup_active']=model.fusion.last_warmup_active
                write(run/'progress.json',progress);print(json.dumps(progress),flush=True)
                if pilot:
                    with (run/'pilot_gradient_diagnostics.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            if time.time()-last_save>300:checkpoint('last');last_save=time.time()
            if time.time()-last_validation>config.get('validation_minutes',30)*60:
                measured();last_validation=last_save=time.time()
        if batches==len(trainloader):epoch+=1
    checkpoint('last')
    if not interrupted[0] and time.time()<config['deadline_unix']-120:measured()
    assert successful>0
    write(run/'complete.json',dict(status='pilot_complete' if pilot else 'interrupted' if interrupted[0] else 'complete',step=step,successful_steps=successful,target_successful_steps=target_steps,matched_update_target_reached=successful>=target_steps,elapsed_seconds=time.time()-started,best_metrics=best,unique_frames_seen=len(seen),subset_frames=len(trainset),checkpoint=str(assets/'best.pt'),completed_unix=time.time(),test_used_for_selection=False))


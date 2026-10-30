"""Measure learned auxiliary sensitivity on one fixed Train frame, CPU only."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_30min_snapshot_Train_ablation_CPU')
    os.environ.update(EVENTPOSE_MULTISENSOR_OUT=goal['output_root'],XFORMERS_DISABLED='1',CUDA_VISIBLE_DEVICES='')
    import fcntl
    lock=(ROOT/'.experiment_storage/device34_fusion_repair_15h/snapshot_cpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np
    import torch
    from utils.pose_evaluation import load_multihmr
    from utils.humanpose_metrics import match_people,joint_errors
    from multihmr.multisensor_data import Frames,collate,input_keys
    from multihmr.multisensor_repair import build_repair_model,load_verified_initialization,REPAIR_LAYOUTS
    torch.set_num_threads(2)
    out=Path(goal['output_root']);folder=out/'diagnostics/first_holdout_snapshot';folder.mkdir(exist_ok=True)
    names=['rgb3_event4','event3_rgb4','event3_event4']
    # Training writes temporary files and atomically replaces best.pt. A link
    # freezes the old immutable inode without changing or locking the trainer.
    for name in names:
        source=out/'checkpoints'/name/'best.pt';snapshot=folder/(name+'.pt')
        assert source.exists()
        if not snapshot.exists():
            try:os.link(source,snapshot)
            except OSError:shutil.copyfile(source,snapshot)
    reg=torch.from_numpy(np.load(out/'joint_regressor.npy'));records=[]
    for name in names:
        print(json.dumps(dict(phase='loading_snapshot',condition=name)),flush=True)
        path=folder/(name+'.pt');digest=sha(path);saved=torch.load(path,map_location='cpu')
        config=saved['config'];assert config['name']==name and saved['test_used_for_selection'] is False
        assert saved['metrics_apply_to_checkpoint'] and saved['metrics_step']==saved['step']
        assert all(torch.isfinite(v).all() for v in saved['model_delta'].values())
        torch.manual_seed(config['seed']);base=load_multihmr(device='cpu')
        model=build_repair_model(base,config).eval();initialization=load_verified_initialization(model,config)
        assert not model.load_state_dict(saved['model_delta'],strict=False).unexpected_keys
        step=saved['step'];updates=saved['successful_steps'];elapsed=saved['training_elapsed_seconds'];del saved
        data=Frames('training',condition=name,sensor_layout=REPAIR_LAYOUTS[name])
        batch=collate([data[0]]);inputs=input_keys(batch)
        assert not set(inputs)&{'joints','centers','rotations','betas','valid'}
        if name=='event3_event4':assert 'image' not in inputs
        with torch.inference_mode():
            image=inputs['image'] if model.mode=='rgb' else (model.adapter(inputs['event'],896)-model.mean)/model.std
            raw=base.backbone(image)
            ctx=(inputs['auxiliary'],inputs['K'],inputs['auxiliary_K'],inputs['primary_to_aux_rotation'],
                 inputs['primary_to_aux_translation'],inputs['auxiliary_available'],896)
            captured=[]
            handle=model.fusion.output.register_forward_hook(lambda module,args,value:captured.append((args[0],value)))
            fused=model.fusion(raw,ctx);ratio=float((fused-raw).norm()/raw.norm())
            message,linear=captured[-1]
            weighted=torch.nn.functional.linear(message,model.fusion.output.weight,None)
            bias=model.fusion.output.bias.expand_as(linear)
            components=dict(linear_output_L2=float(linear.norm()),weighted_message_L2=float(weighted.norm()),
                            output_bias_L2=float(bias.norm()),
                            bias_vs_full_linear_L2=float(bias.norm()/linear.norm()))
            captured.clear()
            empty_ctx=(torch.zeros_like(ctx[0]),*ctx[1:])
            zero_features=model.fusion(raw,empty_ctx)
            components['zero_aux_residual_relative_L2']=float((zero_features-raw).norm()/raw.norm())
            components['content_changed_residual_relative_L2']=float((zero_features-fused).norm()/raw.norm())
            handle.remove();del zero_features,weighted,bias,linear,message,captured
            original=base.backbone.forward;base.backbone.forward=lambda *_args,**_kwargs:raw
            predictions={};errors={};detections={}
            try:
                for mode in ['intact','missing_aux','zero_aux','swap_aux_polarity']:
                    b=dict(inputs)
                    if mode=='missing_aux':b['auxiliary_available']=torch.zeros_like(b['auxiliary_available'])
                    if mode=='zero_aux':b['auxiliary']=torch.zeros_like(b['auxiliary'])
                    if mode=='swap_aux_polarity':b['auxiliary']=b['auxiliary'].flip(1)
                    people=model(**b,is_training=False,det_thresh=.3,nms_kernel_size=3)
                    joints=np.stack([(reg@p['v3d']).numpy() for p in people]) if people else np.empty((0,55,3))
                    target=batch['joints'][0].numpy();valid=batch['valid'][0].numpy().astype(bool)
                    mapping=match_people(joints,target,valid,batch['original_K'][0].numpy(),
                                         batch['original_K'][0].numpy(),batch['image_shape'][0].tolist())
                    errors[mode]={str(g):[float(v)*1000 for v in joint_errors(joints[k,:22],target[g,:22])]
                                  +[float(np.linalg.norm(joints[k,0]-target[g,0]))*1000] for g,k in mapping.items()}
                    predictions[mode]=joints;detections[mode]=len(people)
            finally:base.backbone.forward=original
        dest=folder/(name+'_Train_predictions.npz')
        np.savez_compressed(dest,**{mode+'_joints_primary_camera_m':j for mode,j in predictions.items()})
        records.append(dict(condition=name,primary=model.primary_sensor,auxiliary=model.auxiliary_sensor,
            checkpoint=str(path),checkpoint_sha256=digest,step=step,successful_updates=updates,
            training_elapsed_seconds=elapsed,row=data.records[0],initialization=initialization,
            auxiliary_residual_relative_L2=ratio,linear_components=components,native_detections=detections,
            single_Train_frame_matched_errors_mm=errors,predictions=str(dest),
            error_order=['absolute_MPJPE','pelvis_aligned_MPJPE','PA_MPJPE','full3D_root']))
        print(json.dumps(dict(condition=name,step=step,residual_relative_L2=ratio,detections=detections)),flush=True)
        for h in data.handles.values():h.close()
        model._hook_handle.remove();del model,base,data,batch,inputs,raw,fused,image,predictions,people
        gc.collect()
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),records=records,
        scope='One fixed Train frame per condition at first holdout checkpoint, not a population or Test comparison',
        inference_GT_input=False,Test_used=False,main_training_changed=False,device='CPU')
    (out/'first_holdout_snapshot_Train_ablation_components.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',conditions=3,Test_used=False)),flush=True)


if __name__=='__main__':main()

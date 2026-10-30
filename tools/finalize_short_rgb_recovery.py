"""Validate a preserved RGB checkpoint after the user's shortened-budget stop."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from goal_runtime_control import register_job,write


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_short_RGB_checkpoint_recovery')
    os.environ['EVENTPOSE_MULTISENSOR_OUT']=goal['output_root']
    import pynvml as nv
    nv.nvmlInit();uuid=os.environ['CUDA_VISIBLE_DEVICES'];physical=int(os.environ['EVENTPOSE_PHYSICAL_GPU'])
    h=nv.nvmlDeviceGetHandleByUUID(uuid);assert nv.nvmlDeviceGetMinorNumber(h)==physical==5
    assert physical in goal['allowed_gpus'] and not nv.nvmlDeviceGetComputeRunningProcesses(h)
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from multihmr.multisensor_repair import build_repair_model,load_verified_initialization,REPAIR_LAYOUTS,implementation_provenance
    from multihmr.multisensor_data import Frames,collate
    from utils.pose_evaluation import load_multihmr
    from scripts.train_multihmr import validate_event,event_selection_rank
    torch.set_num_threads(2);out=Path(goal['output_root']);run=out/'runs/rgb3';checkpoint=out/'checkpoints/rgb3'
    assert not (run/'complete.json').exists(),'Never overwrite a terminal completion record'
    value=torch.load(checkpoint/'last.pt',map_location='cpu');config=json.loads((out/'configs/rgb3.json').read_text())
    assert value['config']==config and value['successful_steps']>0 and value['test_used_for_selection'] is False
    assert all(torch.isfinite(v).all() for v in value['model_delta'].values())
    model=build_repair_model(load_multihmr(),config).cuda().eval();load_verified_initialization(model,config)
    assert not model.load_state_dict(value['model_delta'],strict=False).unexpected_keys
    data=Frames('validation',condition='rgb3',sensor_layout=REPAIR_LAYOUTS['rgb3']);assert len(data)==288
    loader=DataLoader(data,batch_size=16,collate_fn=collate,num_workers=0)
    metrics=validate_event(model,loader,torch.from_numpy(np.load(out/'joint_regressor.npy')).cuda(),torch.device('cuda'))
    prior=json.loads((run/'best.json').read_text())
    record=dict(metrics,step=value['step'],successful_steps=value['successful_steps'],time=time.time(),recovery_validation=True)
    backup=checkpoint/'last_before_user_stop_recovery.pt'
    if not backup.exists():os.link(checkpoint/'last.pt',backup)
    value.update(metrics=metrics,metrics_step=value['step'],metrics_apply_to_checkpoint=True)
    if event_selection_rank(metrics,.95)>event_selection_rank(prior,.95):
        oldbest=checkpoint/'best_before_user_stop_recovery.pt'
        if not oldbest.exists():os.link(checkpoint/'best.pt',oldbest)
        value['best_metrics']=metrics;tmp=checkpoint/'recovery_best.tmp.pt';torch.save(value,tmp);tmp.replace(checkpoint/'best.pt')
        write(run/'best.json',record);prior=record
    else:value['best_metrics']={k:prior[k] for k in metrics}
    tmp=checkpoint/'recovery_last.tmp.pt';torch.save(value,tmp);tmp.replace(checkpoint/'last.pt')
    with (run/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    amendment=json.loads((out/'user_budget_amendment.json').read_text())
    history=json.loads((out/'pipeline_state.json').read_text())['history']
    terminated=next(j for j in history if j['role']=='train:rgb3')
    elapsed=terminated['ended_unix']-value['started_unix']
    progress=json.loads((run/'progress.json').read_text())
    complete=dict(status='complete_user_shortened_with_checkpoint_recovery',
        step=value['step'],successful_steps=value['successful_steps'],elapsed_seconds=elapsed,
        samples_seen=value['samples_seen'],unique_frames_seen=value['unique_frames_seen'],
        completed_subset_epochs=value['epoch'],subset_frames=6501,drop_last_frames_per_epoch=5,
        best_metrics=value['best_metrics'],checkpoint=str(checkpoint/'best.pt'),completed_unix=time.time(),
        training_elapsed_seconds_at_checkpoint=value['training_elapsed_seconds'],
        latest_observed_step_before_stop=progress['step'],
        unrecovered_update_lower_bound=max(0,progress['step']-value['step']),
        stop_reason='Latest user requests shorter RGB-only budget; group stop also terminated a loader worker',
        actual_trainer_end_observed_unix=terminated['ended_unix'],
        recovery_validation_only_no_optimizer_updates=True,test_used_for_selection=False)
    write(run/'complete.json',complete)
    write(out/'short_rgb_recovery.json',dict(status='passed',condition='rgb3',metrics=metrics,
        stored_step=value['step'],actual_elapsed_seconds=elapsed,
        checkpoint_training_elapsed_seconds=value['training_elapsed_seconds'],
        unrecovered_update_lower_bound=complete['unrecovered_update_lower_bound'],
        GPU_UUID=uuid,physical_gpu=physical,pid=os.getpid(),
        implementation_sha256=implementation_provenance(out/'source_snapshots'),
        no_new_optimizer_updates=True,Test_used=False))
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()

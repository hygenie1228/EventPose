"""Exercise the real holdout diagnostic on one real frame at initialization."""
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
    register_job(goal,os.getpid(),'fusion_repair_ablation_helper_real_cpu')
    os.environ['EVENTPOSE_MULTISENSOR_OUT']=goal['output_root'];os.environ['XFORMERS_DISABLED']='1'
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from utils.pose_evaluation import load_multihmr
    from multihmr.multisensor_repair import build_repair_model,REPAIR_LAYOUTS,load_verified_initialization
    from multihmr.multisensor_data import Frames,collate
    from multihmr.multisensor_diagnostics import holdout_modality_errors
    torch.set_num_threads(2);out=Path(goal['output_root']);name='event3_rgb4'
    config=json.loads((out/'configs'/(name+'.json')).read_text());torch.manual_seed(config['seed'])
    base=load_multihmr(device='cpu');model=build_repair_model(base,config).eval()
    initialization=load_verified_initialization(model,config)
    data=Frames('validation',condition=name,sensor_layout=REPAIR_LAYOUTS[name],limit=1)
    loader=DataLoader(data,batch_size=1,collate_fn=collate,num_workers=0,shuffle=False)
    reg=torch.from_numpy(np.load(out/'joint_regressor.npy'))
    with torch.inference_mode():
        raw=base.backbone(data[0]['image'][None]);original_forward=base.backbone.forward
        base.backbone.forward=lambda *_args,**_kwargs:raw
        try:
            rows={mode:holdout_modality_errors(model,loader,reg,mode)
                  for mode in ('intact','missing_aux','zero_aux','swap_aux_polarity')}
        finally:base.backbone.forward=original_forward
    reference=rows['intact'];assert reference['total_gt_instances']==1 and reference['matched_instances']==1
    for row in rows.values():
        assert row['gt_keys']==reference['gt_keys'] and row['errors_mm']==reference['errors_mm']
        assert row['GT_used_as_inference_input'] is False
    write(out/'ablation_cpu_preflight.json',dict(status='passed',checked_unix=time.time(),condition=name,
          row=data.records[0],initialization=initialization,modes=list(rows),
          real_holdout_frames=1,matched_instances=1,all_initial_diagnostic_predictions_equal=True,
          source_checkpoint_only=True,new_finetuning_completed=False,Test_evaluation=False,
          scope='Actual full model CPU initialization, native features cached across modes; one-frame diagnostic correctness, not full holdout or Test performance'))
    for handle in data.handles.values():handle.close()
    print(json.dumps(dict(status='passed',modes=4,real_holdout_frames=1,performance_claim=False)),flush=True)


if __name__=='__main__':main()

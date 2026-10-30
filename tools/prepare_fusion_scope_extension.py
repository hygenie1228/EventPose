"""Freeze same-device trials and zero-update released baselines before Test."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job,write
from fusion_repair_scope import TRAIN_NAMES,PRETRAIN_NAMES


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_scope_extension_prepare')
    import torch
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    out=Path(goal['output_root'])
    assert not list((out/'results').glob('*/*/complete.json')),'Freeze new protocol before any new Test result'
    original=json.loads((out/'original_pretrained_manifest.json').read_text())
    for name,source in [('rgb3_event3','rgb3_event4'),('rgb4_event4','event3_rgb4')]:
        config=json.loads((out/'configs'/(source+'.json')).read_text())
        config.pop('trial_started_unix',None)
        config.update(name=name,condition=name,primary_sensor=REPAIR_LAYOUTS[name][0],auxiliary_sensor=REPAIR_LAYOUTS[name][1],hours=3)
        f=out/'configs'/(name+'.json')
        if f.exists():assert json.loads(f.read_text())==config
        else:write(f,config)
    for name in PRETRAIN_NAMES:
        condition=name.removeprefix('pre_')
        config=json.loads((out/'configs'/(condition+'.json')).read_text());config.pop('trial_started_unix',None)
        config.update(name=name,hours=0,deadline_unix=goal['deadline_unix'],weight_stage='released_pretrained',
                      initialization='Original released Multi-HMR; zero ETRI fine-tuning updates',
                      initialization_checkpoint=original['path'],initialization_sha256=original['sha256'])
        config.pop('initialization_manifest',None)
        f=out/'configs'/(name+'.json')
        if f.exists():assert json.loads(f.read_text())==config
        else:write(f,config)
        target=out/'checkpoints'/name/'best.pt';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            torch.save(dict(config=config,step=0,successful_steps=0,model_delta={},
                test_used_for_selection=False,released_checkpoint=original['path'],
                released_checkpoint_sha256=original['sha256'],
                kind='Source manifest only; weights remain at the read-only original path'),target)
    manifest_path=out/'manifest.json';manifest=json.loads(manifest_path.read_text())
    backup=out/'manifest_before_scope_extension.json'
    if not backup.exists():backup.write_text(manifest_path.read_text())
    manifest.update(layouts=REPAIR_LAYOUTS,scope_extended_unix=time.time());write(manifest_path,manifest)
    write(out/'experiment_manifest.json',dict(configs=[json.loads((out/'configs'/(n+'.json')).read_text()) for n in TRAIN_NAMES],
        pretrained_configs=[json.loads((out/'configs'/(n+'.json')).read_text()) for n in PRETRAIN_NAMES],
        selection='Train holdout only',total_wall_hours=15,
        budget_interpretation='User requests shorter RGB-only vs longer fusion; not isolated modality-effect comparison'))
    print(json.dumps(dict(training_conditions=TRAIN_NAMES,zero_update_conditions=PRETRAIN_NAMES)),flush=True)


if __name__=='__main__':main()

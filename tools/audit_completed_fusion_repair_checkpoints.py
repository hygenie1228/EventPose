"""Early actual CPU checkpoint audit for terminal trials; never infer from presence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from goal_runtime_control import register_job,write


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+')
    parser.add_argument('--output-name',default='completed_checkpoint_CPU_audit_first_wave.json')
    args=parser.parse_args()
    assert Path(args.output_name).name==args.output_name and args.output_name.endswith('.json')
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_completed_checkpoints_CPU')
    import torch
    torch.set_num_threads(2);out=Path(goal['output_root']);records=[]
    approved=json.loads((out/'source_revision_scope_extension.json').read_text())['approved_implementations']
    for name in args.names:
        run=out/'runs'/name;complete=json.loads((run/'complete.json').read_text());assert complete['status']=='complete'
        config=json.loads((out/'configs'/(name+'.json')).read_text());started=json.loads((run/'started.json').read_text())
        selected=json.loads((run/'best.json').read_text());assert started['implementation_sha256'] in approved
        assert complete['samples_seen']==16*complete['step'] and 0<complete['successful_steps']<=complete['step']
        for kind in ['best','last']:
            path=out/'checkpoints'/name/(kind+'.pt');before=path.stat();sha=digest(path)
            value=torch.load(path,map_location='cpu');assert value['config']==config and value['test_used_for_selection'] is False
            assert value['model_delta'] and all(torch.isfinite(x).all() for x in value['model_delta'].values())
            assert sum(x.numel() for x in value['model_delta'].values())==started['trainable_parameters']
            if kind=='best':
                assert value['metrics_apply_to_checkpoint'] and value['metrics_step']==value['step']==selected['step']
                assert all(selected[k]==v for k,v in value['metrics'].items())
            else:
                for k in ['step','successful_steps','samples_seen','unique_frames_seen']:assert value[k]==complete[k]
                assert len(set(value['seen_indices']))==value['unique_frames_seen']<=6501
                assert all(0<=i<6501 for i in value['seen_indices'])
            after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
            records.append(dict(condition=name,kind=kind,path=str(path),sha256=sha,step=value['step'],
                                successful_updates=value['successful_steps'],tensors_finite=True,
                                configuration_equal=True,metrics_apply_to_checkpoint=value['metrics_apply_to_checkpoint'],
                                checkpoint_training_elapsed_seconds=value['training_elapsed_seconds'],Test_used=False))
            del value
        print(json.dumps(dict(condition=name,status='passed_actual_CPU_load')),flush=True)
    result=dict(status='passed',checked_unix=time.time(),records=records,
                scope='Specified terminal trials only; full Test / speed / final report remain required',goal_complete=False)
    write(out/args.output_name,result)
    print(json.dumps(dict(status='passed',checkpoints=len(records),goal_complete=False)),flush=True)


if __name__=='__main__':main()

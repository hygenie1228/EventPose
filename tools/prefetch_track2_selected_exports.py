"""Fill a terminal Test shard's GPU with its own pending Train export shard."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import time

import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command, process_identity
from multihmr.track2_data import OUTPUT, save_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def main(gpu, parent):
    import pynvml
    prefix=f'selected_export_prefetch_gpu{gpu}'
    with (OUTPUT/'locks'/(prefix+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME=prefix+'_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children)
        signal.signal(signal.SIGINT,runtime.stop_children)
        identity=process_identity(parent)
        assert identity is not None
        manifest=read(OUTPUT/'manifest.json');gpus=runtime.healthy_gpus()
        assert gpu in gpus and gpu!=0
        index=gpus.index(gpu)
        plan=read(OUTPUT/'target_pooled_followup_plan.json')
        trials=['original']+[cfg['name'] for cfg in plan['trials']]
        test_tasks=[(trial,sample) for trial in trials for sample in manifest['test']][index::len(gpus)]
        selected=read(OUTPUT/'selected.json');trial=selected['trial']
        assert not selected['test_used_for_selection']
        mapping=read(OUTPUT/'evaluation_checkpoints.json')[trial]
        checkpoint=Path(mapping['checkpoint']);checkpoint_hash=sha256(checkpoint)
        assert checkpoint_hash==mapping['checkpoint_sha256']
        tasks=manifest['train'][index::len(gpus)]
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByUUID(read(OUTPUT/'gpu_resources.json')['uuids'][str(gpu)])
        def status(value,**extra):
            save_json(OUTPUT/(prefix+'_status.json'),dict(status=value,pid=os.getpid(),gpu=gpu,
                parent_pid=parent,parent_start_ticks=identity,trial=trial,tasks=tasks,
                checkpoint_sha256=checkpoint_hash,source_exports_written=False,
                test_used_for_selection=False,updated_unix=time.time(),**extra))
        status('waiting_for_own_Test_shard_and_idle_GPU')
        while process_identity(parent)==identity and time.time()<deadline-60:
            if read(OUTPUT/'supervisor.json')['status']!='evaluating_test':
                status('not_needed_main_left_Test');return
            done=all((OUTPUT/'predictions'/name/sample/'metrics.json').exists() for name,sample in test_tasks)
            registry=read(OUTPUT/'target_contenders_jobs.json')['jobs']
            active=any(gpu in job['gpus'] and process_identity(job['pid'])==job['start_ticks'] for job in registry)
            if done and not active and not pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                break
            time.sleep(30)
        else:
            status('parent_terminal_or_deadline');return
        status('running_on_verified_idle_GPU')
        for sample in tasks:
            if process_identity(parent)!=identity or read(OUTPUT/'supervisor.json')['status']!='evaluating_test':
                break
            folder=OUTPUT/'predictions'/trial/sample
            for stage in ['track2-infer','track2-convert']:
                command=evaluation_command(stage,trial,sample)
                if stage=='track2-infer':command.extend(['--checkpoint',str(checkpoint)])
                runtime.run(command,[gpu],prefix+'_'+stage+'_'+sample,deadline)
            info=read(folder/'inference.json');conversion=read(folder/'conversion_v2.json')
            assert info['checkpoint_sha256']==checkpoint_hash
            assert info['evaluation_protocol']=='target_foreground_head_v1'
            assert info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert info['frames']==manifest['samples'][sample]['frames']
            assert (folder/'human_params_mhr_v2.npz').exists() and conversion
            status('running_on_verified_idle_GPU',completed_sample=sample)
        status('complete_or_handed_back_to_main')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--parent',type=int,required=True);args=parser.parse_args()
    try:
        main(args.gpu,args.parent)
    except BaseException as error:
        save_json(OUTPUT/f'selected_export_prefetch_gpu{args.gpu}_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

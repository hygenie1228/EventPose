"""Use an idle healthy GPU for pending exports without restarting live shards."""
from pathlib import Path
import fcntl
import json
import os
import signal
import time

import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command, process_identity
from multihmr.track2_data import OUTPUT, save_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import pynvml
    with (OUTPUT/'locks/export_backfill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='export_backfill_jobs.json'
        signal.signal(signal.SIGTERM, runtime.stop_children)
        signal.signal(signal.SIGINT, runtime.stop_children)
        state=read(OUTPUT/'target_eval_status.json')
        assert state['status']=='target_winner_all30_export'
        parent=state['pid'];identity=process_identity(parent)
        assert identity is not None
        resources=read(OUTPUT/'gpu_resources.json');gpu=6
        assert gpu in resources['healthy_gpus']
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(gpu)])
        assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        manifest=read(OUTPUT/'manifest.json');trial=state['selected_trial']
        # All of the original GPU6 static shard must already be complete.
        workers=runtime.healthy_gpus();index=workers.index(gpu)
        assert all((OUTPUT/'predictions'/trial/s/'conversion_v2.json').exists()
                   for s in manifest['train'][index::len(workers)])
        active=read(OUTPUT/'target_eval_jobs.json')['jobs']
        active_samples={j['command'][j['command'].index('--sample')+1]
                        for j in active if process_identity(j['pid'])==j['start_ticks']}
        tasks=[s for s in reversed(manifest['train']) if s not in active_samples
               and not (OUTPUT/'predictions'/trial/s/'inference.json').exists()]
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        checkpoint=read(OUTPUT/'evaluation_checkpoints.json')[trial]['checkpoint']
        def status(value, **extra):
            save_json(OUTPUT/'export_backfill_status.json',dict(status=value,pid=os.getpid(),
                main_pid=parent,main_start_ticks=identity,gpu=gpu,trial=trial,tasks=tasks,
                changes_to_evaluation_protocol=False,updated_unix=time.time(),**extra))
        status('running')
        for sample in tasks:
            if process_identity(parent)!=identity:break
            if (OUTPUT/'predictions'/trial/sample/'conversion_v2.json').exists():continue
            for stage in ['track2-infer','track2-convert']:
                command=evaluation_command(stage,trial,sample)
                if stage=='track2-infer':command.extend(['--checkpoint',checkpoint])
                # The normal CLI owns the same stage lock/cache as live shards.
                runtime.run(command,[gpu],'export_backfill_'+stage+'_'+sample,deadline)
            status('running',completed_sample=sample)
        status('complete')


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save_json(OUTPUT/'export_backfill_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children()
        raise

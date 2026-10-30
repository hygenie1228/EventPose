"""Complete saved-checkpoint evaluation/export on the five authorized healthy GPUs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import fcntl
import json
import os
import queue
import signal
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
import run_track2_experiments as runtime
runtime.REGISTRY_NAME='healthy_eval_jobs.json'


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'healthy_eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        resources=read(OUTPUT/'gpu_resources.json');gpus=resources['healthy_gpus']
        manifest=read(OUTPUT/'manifest.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        selected=read(OUTPUT/'selected.json')['trial'];mappings=read(OUTPUT/'evaluation_checkpoints.json')
        names=['original','head_strong','last2_strong','last4_strong']
        names.remove(selected);names.insert(0,selected)
        jobs=queue.Queue()
        for name in names:
            for sample in sorted(manifest['test'],key=lambda n:-manifest['samples'][n]['frames']):
                if not (OUTPUT/'predictions'/name/sample/'metrics.json').exists():jobs.put((name,sample,True))
        for sample in sorted(manifest['train'],key=lambda n:-manifest['samples'][n]['frames']):
            if not (OUTPUT/'predictions'/selected/sample/'conversion.json').exists():jobs.put((selected,sample,False))
        save_json(OUTPUT/'healthy_eval_status.json',dict(status='running',pid=os.getpid(),gpus=gpus,
            uuid_masks=True,queued=jobs.qsize(),started_unix=time.time()))
        save_json(OUTPUT/'supervisor.json',dict(status='evaluating_saved_checkpoints_healthy5',pid=os.getpid(),
            gpus=gpus,excluded_gpu=3,updated_unix=time.time()))
        def worker(gpu):
            while True:
                try:trial,sample,score=jobs.get_nowait()
                except queue.Empty:return
                stages=['track2-infer','track2-convert']+(['track2-score'] if score else [])
                for stage in stages:
                    command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,'--trial',trial,'--sample',sample]
                    if stage=='track2-infer' and trial!='original':command.extend(['--checkpoint',mappings[trial]['checkpoint']])
                    runtime.run(command,[gpu],'healthy_'+stage+'_'+trial+'_'+sample,deadline)
                jobs.task_done()
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,gpu) for gpu in gpus]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                runtime.stop_children();raise
        models={}
        for name in names:
            values={s:read(OUTPUT/'predictions'/name/s/'metrics.json') for s in manifest['test']}
            for s,v in values.items():assert v['gt_sha256']==manifest['samples'][s]['gt_sha256']
            models[name]=dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in values.values()])),
                ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in values.values()])),
                sequences=10,frames=sum(v['frames'] for v in values.values()),per_sequence=values)
        save_json(OUTPUT/'comparison.json',dict(models=models,aggregation='equal sequence weights',
            units=dict(CD_H='cm',ACC_H='cm/frame^2'),test_used_for_selection=False,
            source_manifest_sha256=sha256(OUTPUT/'manifest.json'),official_leaderboard_parity=False,
            hardware_degraded_to=gpus,completed_unix=time.time()))
        import finalize_track2_experiments as finish
        receipts={}
        for sample in sorted(manifest['samples']):
            receipts[sample]=finish.export_sample(selected,sample,manifest)
            save_json(OUTPUT/'export_receipts.json',dict(trial=selected,samples=receipts,updated_unix=time.time()))
        finish.audit(manifest,selected,receipts)
        save_json(OUTPUT/'healthy_eval_status.json',dict(status='artifacts_ready_for_final_audit',pid=os.getpid(),gpus=gpus,completed_unix=time.time()))
        finish.status('artifacts_ready_for_final_audit',selected_recipe=selected,exports=30,gpus=gpus,completed_unix=time.time())


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'healthy_eval_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

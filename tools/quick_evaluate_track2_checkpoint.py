"""Immediate Test of an already saved checkpoint, using a shared six-GPU queue."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import os
import queue
import signal
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json,sha256
import run_track2_experiments as runtime
runtime.REGISTRY_NAME='quick_test_jobs.json'


def read(path):return json.loads(Path(path).read_text())


def main():
    import fcntl
    with (OUTPUT/'quick_test.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        chosen=read(OUTPUT/'selected.json')['trial']
        checkpoint=OUTPUT/'checkpoints'/chosen/'best.pt' if chosen!='original' else None
        trial='quick_'+chosen if checkpoint else 'original'
        step=read(OUTPUT/'runs'/chosen/'best.json')['step'] if checkpoint else 0
        jobs=queue.Queue()
        # Longer sequences first; every free GPU takes the next remaining clip.
        for sample in sorted(manifest['test'],key=lambda n:-manifest['samples'][n]['frames']):jobs.put(sample)
        save_json(OUTPUT/'quick_test_status.json',dict(status='running',pid=os.getpid(),trial=chosen,
            checkpoint=str(checkpoint) if checkpoint else None,physical_gpus=[1,2,3,4,5,6],started_unix=time.time()))
        def worker(gpu):
            while True:
                try:sample=jobs.get_nowait()
                except queue.Empty:return
                for stage in ['track2-infer','track2-convert','track2-score']:
                    command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,
                        '--trial',trial,'--sample',sample]
                    if stage=='track2-infer' and checkpoint:command.extend(['--checkpoint',str(checkpoint)])
                    runtime.run(command,[gpu],'quick_'+stage+'_'+sample,deadline)
                jobs.task_done()
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(worker,gpu) for gpu in range(1,7)]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                runtime.stop_children();raise
        samples={name:read(OUTPUT/'predictions'/trial/name/'metrics.json') for name in manifest['test']}
        for name,value in samples.items():
            assert value['frames']==manifest['samples'][name]['frames']
            assert value['gt_sha256']==manifest['samples'][name]['gt_sha256']
        metrics=dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in samples.values()])),
            ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in samples.values()])),
            sequences=10,frames=sum(v['frames'] for v in samples.values()),per_sequence=samples)
        save_json(OUTPUT/'quick_comparison.json',dict(models={chosen:metrics},checkpoint=str(checkpoint) if checkpoint else None,
            checkpoint_sha256=sha256(checkpoint) if checkpoint else None,step=step,training_sequences=16 if checkpoint else 0,
            internal_validation_sequences=4,test_used_for_selection=False,completed_unix=time.time()))
        save_json(OUTPUT/'quick_test_status.json',dict(status='complete',pid=os.getpid(),trial=chosen,completed_unix=time.time()))
        import subprocess
        subprocess.run([str(runtime.PYTHON),str(ROOT/'tools/report_track2_experiments.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'quick_test_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

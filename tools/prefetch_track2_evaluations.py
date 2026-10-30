"""Use genuinely free GPUs1–6 for ready evaluation work across phase boundaries.

Respects live process identities and NVML allocations; never uses GPU0, trains a
duplicate model, changes a checkpoint, or chooses a model from Test results.
Primary supervisors reuse these exact per-stage locked prediction caches.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import json
import os
import signal
import sys
import time

import numpy as np
import pynvml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json
import run_track2_experiments as runtime
runtime.REGISTRY_NAME='prefetch_jobs.json'


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def live(job):
    try:
        fields=Path(f'/proc/{job["pid"]}/stat').read_text().split(') ',1)[1].split()
        return fields[0]!='Z' and fields[19]==str(job['start_ticks'])
    except FileNotFoundError:return False


def reserved(handles):
    busy=set();cases=set()
    for gpu,handle in handles.items():
        try:
            if pynvml.nvmlDeviceGetComputeRunningProcesses(handle):busy.add(gpu)
        except pynvml.NVMLError:busy.add(gpu)
    for path in OUTPUT.glob('*jobs.json'):
        for job in read(path,{}).get('jobs',[]):
            if not live(job):continue
            busy.update(job.get('gpus',[]))
            command=job.get('command',[])
            if '--trial' in command and '--sample' in command:
                cases.add((command[command.index('--trial')+1],command[command.index('--sample')+1]))
    return busy,cases


def ready_tasks(manifest):
    selected=read(OUTPUT/'selected.json',{}).get('trial')
    if not selected:return []
    names=['head_strong','last2_strong','last4_strong']
    names.extend(cfg['name'] for cfg in read(OUTPUT/'cd_wave2_plan.json',{}).get('trials',[]))
    names.append('original')
    if selected in names:names.remove(selected);names.insert(0,selected)
    mappings=read(OUTPUT/'evaluation_checkpoints.json',{})
    tests=[];exports=[];baselines=[]
    for name in names:
        checkpoint=None
        if name!='original':
            complete=read(OUTPUT/'runs'/('refit_'+name)/'complete.json')
            if complete and complete.get('status')=='complete':
                checkpoint=OUTPUT/'checkpoints'/('refit_'+name)/'best.pt'
            elif mappings.get(name,{}).get('full_train_refit'):
                checkpoint=Path(mappings[name]['checkpoint'])
            else:continue
            if not checkpoint.is_file():continue
        samples=list(manifest['test'])
        if name==selected:samples+=manifest['train']
        samples.sort(key=lambda sample:(sample not in manifest['test'],-manifest['samples'][sample]['frames']))
        for sample in samples:
            folder=OUTPUT/'predictions'/name/sample
            if sample in manifest['test']:
                if (folder/'metrics.json').exists():continue
                stages=['track2-infer','track2-convert','track2-score']
            else:
                if (folder/'conversion.json').exists():continue
                stages=['track2-infer','track2-convert']
            task=(name,sample,checkpoint,stages)
            (tests if sample in manifest['test'] else exports).append(task)
    # The existing init predictions on the same Train-validation clips give an
    # apples-to-apples internal reference, distinct from the frozen Test10.
    for name in ['init_v6','init_v0']:
        for sample in manifest['validation']:
            if not (OUTPUT/'predictions'/name/sample/'metrics.json').exists():
                baselines.append((name,sample,None,['track2-score']))
    return tests+exports+baselines


def baseline_validation_summary(manifest):
    models={}
    for name in ['init_v6','init_v0']:
        values=[read(OUTPUT/'predictions'/name/sample/'metrics.json') for sample in manifest['validation']]
        if not all(values):return
        models[name]=dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in values])),
            ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in values])),
            sequences=4,per_sequence=dict(zip(manifest['validation'],values)))
    save_json(OUTPUT/'init_validation_comparison.json',dict(models=models,split='four Train-only validation sequences',
        test_used=False,updated_unix=time.time()))


def main():
    with (OUTPUT/'prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        manifest=read(OUTPUT/'manifest.json')
        pynvml.nvmlInit()
        handles={gpu:pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in range(1,7)}
        active={};recent_errors=[];failed=set()
        def execute(gpu,task):
            trial,sample,checkpoint,stages=task
            for stage in stages:
                command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,
                    '--trial',trial,'--sample',sample]
                if stage=='track2-infer' and checkpoint:command.extend(['--checkpoint',str(checkpoint)])
                runtime.run(command,[gpu],'prefetch_'+stage+'_'+trial+'_'+sample,deadline)
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                while time.time()<deadline:
                    for gpu,(future,task) in list(active.items()):
                        if not future.done():continue
                        try:future.result()
                        except Exception as error:
                            recent_errors.append(dict(task=task[:2],error=repr(error),time=time.time()))
                            failed.add(task[:2])
                        del active[gpu]
                    if read(OUTPUT/'priority_resume_status.json',{}).get('status')=='artifacts_ready_for_final_audit' and not active:break
                    busy,cases=reserved(handles)
                    cases.update((task[0],task[1]) for _,task in active.values())
                    tasks=[task for task in ready_tasks(manifest) if task[:2] not in failed]
                    for gpu in range(1,7):
                        if gpu in active or gpu in busy:continue
                        task=next((t for t in tasks if (t[0],t[1]) not in cases),None)
                        if task is None:break
                        active[gpu]=(pool.submit(execute,gpu,task),task);cases.add(task[:2])
                    baseline_validation_summary(manifest)
                    save_json(OUTPUT/'prefetch_status.json',dict(pid=os.getpid(),status='active',
                        active={str(g):dict(trial=t[0],sample=t[1]) for g,(_,t) in active.items()},
                        reserved_gpus=sorted(busy),ready_tasks=len(tasks),recent_errors=recent_errors[-6:],
                        gpu0_used=False,updated_unix=time.time()))
                    time.sleep(min(5,max(1,deadline-time.time())))
        finally:
            runtime.stop_children();pynvml.nvmlShutdown()
        save_json(OUTPUT/'prefetch_status.json',dict(pid=os.getpid(),status='stopped',gpu0_used=False,updated_unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'prefetch_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

"""Controlled single-GPU full-pipeline comparison after training/Test finish."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
OUT=ROOT/'exp/261024/nlf_finetune_12h/runtime_benchmark'
WORK=ROOT/'.experiment_storage/nlf_runtime_benchmark'
SOURCE=ROOT/'.experiment_storage/humanpose_gt_segments'

def plan():
    tasks=json.loads((SOURCE/'tasks.json').read_text())
    selected=[]
    for desired in [1,2,3]:
        available=[t for t in tasks if t['parent_session'] not in {s['parent_session'] for s in selected}]
        selected.append(min(available,key=lambda t:abs(t['valid_gt_instances']/t['gt_frames']-desired)))
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE/'joint_regressor.npy',WORK/'joint_regressor.npy')
    (OUT/'tasks.json').write_text(json.dumps(selected,indent=2))
    return selected

def worker(method):
    import torch
    import utils.pose_evaluation as base
    from utils.segment_inference import multihmr
    from scripts.evaluate_nlf import infer
    from nlf.model import load_model
    base.configure_torch();base.OUT=WORK
    tasks=json.loads((OUT/'tasks.json').read_text())
    load_start=time.perf_counter()
    if method.startswith('multihmr'):
        checkpoint=None
        if method.endswith('finetuned'):
            checkpoint=json.loads((ROOT/'exp/261023/multihmr_finetune_14h/selected.json').read_text())['checkpoint']
        model=base.load_multihmr(checkpoint)
    else:
        checkpoint=None
        if method.endswith('finetuned'):checkpoint=json.loads((OUT.parent/'selected.json').read_text())['checkpoint']
        model=load_model(checkpoint)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    records=[]
    for raw in tasks:
        started=time.perf_counter();cached=WORK/f'{method}_{raw["parent_session"]}.mp4'
        shutil.copyfile(raw['source_video'],cached);stage_seconds=time.perf_counter()-started
        try:
            task=dict(raw,video=str(cached),video_frames=100,smoke=True)
            c=base.Collector(task,method)
            torch.cuda.reset_peak_memory_stats()
            if method.startswith('multihmr'):multihmr(model,c)
            else:infer(model,c,16)
            c.save();torch.cuda.synchronize()
            records.append(dict(session=raw['parent_session'],frames=100,
                measured_seconds=time.perf_counter()-started,stage_seconds=stage_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                mean_gt_people=raw['valid_gt_instances']/raw['gt_frames']))
        finally:cached.unlink(missing_ok=True)
    result=dict(method=method,load_seconds=load_seconds,segments=records,gpu=torch.cuda.get_device_name(),
                includes='NFS video staging, segment seek/decode, preprocessing, detection, model, SMPL-X recovery, GT matching, result save',
                frames=sum(r['frames'] for r in records),seconds=sum(r['measured_seconds'] for r in records))
    result['fps']=result['frames']/result['seconds']
    (OUT/f'{method}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

def supervise(gpu):
    assert gpu in (2,3,4,5,6), 'Use physical GPUs 2–6 only'
    plan()
    identity=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    methods=['multihmr_pretrained','nlf_pretrained','multihmr_finetuned','nlf_finetuned']
    for method in methods:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
        started_unix=time.time()
        before=time.perf_counter()
        subprocess.run([sys.executable,'-u',__file__,'worker','--method',method],cwd=ROOT,env=env,check=True)
        result=json.loads((OUT/f'{method}.json').read_text());result['cold_process_wall_seconds']=time.perf_counter()-before
        result.update(gpu_identity=identity,cuda_visible_devices=str(gpu),
                      process_started_unix=started_unix,process_finished_unix=time.time())
        (OUT/f'{method}.json').write_text(json.dumps(result,indent=2))
    results={m:json.loads((OUT/f'{m}.json').read_text()) for m in methods}
    (OUT/'summary.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','supervise'])
    p.add_argument('--method');p.add_argument('--gpu',type=int,choices=[2,3,4,5,6],default=3);a=p.parse_args()
    if a.stage=='worker':worker(a.method)
    else:supervise(a.gpu)

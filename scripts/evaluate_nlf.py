from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Evaluate released or adapted NLF on the unchanged SMPL-X Test segments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torchvision
from nlf.model import load_model,regressed_joints_m
import utils.pose_evaluation as base
from utils.segment_inference import frames

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261024/nlf_finetune_12h'
WORK=ROOT/'.experiment_storage/nlf_finetune_12h'
SOURCE=ROOT/'.experiment_storage/humanpose_gt_segments'

def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    for name in ['rgb','logs']:(WORK/name).mkdir(exist_ok=True)
    shutil.copyfile(SOURCE/'joint_regressor.npy',WORK/'joint_regressor.npy')
    tasks=json.loads((SOURCE/'tasks.json').read_text())
    assert len(tasks)==320
    split=json.loads((ROOT/'dataset/split.json').read_text())
    assert set(split['test'])=={t['parent_session'] for t in tasks}
    base.atomic_json(WORK/'tasks.json',tasks)
    return tasks

def infer(model,c,batch):
    pending=[];gpu_seconds=0.
    def flush():
        nonlocal gpu_seconds
        x=torch.from_numpy(np.stack([rgb for _,rgb in pending])).permute(0,3,1,2).cuda()
        K=torch.from_numpy(c.K).float().cuda()[None].expand(len(pending),-1,-1)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            result=model.detect_smpl_batched(x,intrinsic_matrix=K,model_name='smplx',
                      num_aug=1,internal_batch_size=64,detector_threshold=.3)
            joints=[regressed_joints_m(v,c.reg).cpu().numpy() for v in result['vertices3d']]
        torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-start
        for (pos,_),j in zip(pending,joints):
            assert j.shape[1:]==(55,3) and np.isfinite(j).all()
            c.current_pos=pos;c.collect(j,c.K)
        pending.clear()
    for pos,rgb in frames(c.task):
        pending.append((pos,rgb))
        if len(pending)==batch:flush()
    if pending:flush()
    return gpu_seconds

def worker(a):
    tasks=prepare();base.OUT=WORK;base.configure_torch()
    method=a.method
    model=load_model(a.checkpoint)
    names=sorted({t['parent_session'] for t in tasks})[a.rank::a.workers]
    if a.smoke:names=names[:1]
    for name in names:
        clips=[t for t in tasks if t['parent_session']==name]
        if a.smoke:clips=clips[:1]
        if not a.smoke and all((WORK/method/t['session']/'complete.json').exists() for t in clips):continue
        cached=WORK/'rgb'/f'{method}_{name}.mp4'
        before=time.perf_counter();shutil.copyfile(clips[0]['source_video'],cached)
        print(json.dumps(dict(event='stage_rgb',session=name,seconds=time.perf_counter()-before)),flush=True)
        try:
            for raw in clips:
                if not a.smoke and (WORK/method/raw['session']/'complete.json').exists():continue
                task=dict(raw,video=str(cached))
                if a.smoke:task.update(video_frames=32,smoke=True)
                c=base.Collector(task,method)
                try:
                    seconds=infer(model,c,a.batch);c.save()
                    base.atomic_json(c.folder/'timing.json',dict(model_pipeline_seconds=seconds,
                        frames=task['video_frames'],batch=a.batch,checkpoint=str(a.checkpoint),
                        includes='detector, crop, NLF, SMPL-X fitting, joint regression, device transfer'))
                except Exception:
                    c.save(partial=True);raise
                shutil.copytree(c.folder,OUT/('smoke_results' if a.smoke else 'results')/method/task['session'],dirs_exist_ok=True)
        finally:cached.unlink(missing_ok=True)

def supervise(a):
    if a.method=='nlf_finetuned':
        metadata=json.loads(a.checkpoint.with_suffix('.json').read_text())
        assert metadata['test_used_for_selection'] is False
        subprocess.run([sys.executable,str(ROOT/'src/nlf/checkpoint.py'),
            '--delta',metadata['source_delta'],'--checkpoint',str(a.checkpoint)],cwd=ROOT,check=True)
    tasks=prepare();jobs=[]
    for rank in range(a.workers):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2')
        log=(WORK/'logs'/f'{a.method}_{rank}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--rank',str(rank),'--workers',str(a.workers),
            '--method',a.method,'--checkpoint',str(a.checkpoint),'--batch',str(a.batch)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log))
    while any(p.poll() is None for p,_ in jobs):
        base.atomic_json(OUT/f'{a.method}_status.json',dict(status='running',updated_unix=time.time(),
            completed_segments=sum((WORK/a.method/t['session']/'complete.json').exists() for t in tasks),
            workers=[dict(pid=p.pid,exit_code=p.poll()) for p,_ in jobs]))
        time.sleep(15)
    for p,log in jobs:log.close();assert p.returncode==0,(p.pid,p.returncode)
    assert all((WORK/a.method/t['session']/'complete.json').exists() for t in tasks)
    base.atomic_json(OUT/f'{a.method}_status.json',dict(status='inference_complete',completed_segments=320,updated_unix=time.time()))
    if a.method=='nlf_finetuned':
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,'-m','utils.compare_methods'],cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,str(ROOT/'src/utils/runtime_benchmark.py'),'supervise','--gpu','0'],
                       cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,'-m','utils.compare_methods'],cwd=ROOT,env=env,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','supervise'])
    p.add_argument('--checkpoint',type=Path,default=ROOT/'data/pretrained/nlf/nlf_l_multi.torchscript')
    p.add_argument('--method',default='nlf_pretrained');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.stage=='worker':worker(a)
    else:supervise(a)

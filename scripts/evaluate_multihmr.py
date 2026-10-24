from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""One final frozen Test evaluation after validation-only checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import utils.pose_evaluation as base
from utils.segment_inference import multihmr
from multihmr.data import ROOT,OUT,WORK,save

SNAPSHOT=os.environ.get('MULTIHMR_TEST_SNAPSHOT','')
TEST=WORK/'test_snapshots'/SNAPSHOT if SNAPSHOT else WORK/'final_test'
RESULT_OUT=OUT/'test_snapshots'/SNAPSHOT if SNAPSHOT else OUT
METHOD='multihmr_finetuned'


def prepare():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['test_split_sha256']
    source=ROOT/'.experiment_storage/humanpose_gt_segments'
    tasks=json.loads((source/'tasks.json').read_text())
    assert len(tasks)==320 and {t['parent_session'] for t in tasks}==set(manifest['test_sessions'])
    TEST.mkdir(parents=True,exist_ok=True)
    for folder in ['rgb','logs']:(TEST/folder).mkdir(exist_ok=True)
    shutil.copyfile(source/'joint_regressor.npy',TEST/'joint_regressor.npy')
    save(TEST/'tasks.json',tasks)
    return tasks


def worker(rank):
    base.OUT=TEST;base.configure_torch()
    tasks=json.loads((TEST/'tasks.json').read_text())
    chosen=json.loads((RESULT_OUT/'selected.json').read_text())
    model=base.load_multihmr(chosen['checkpoint'])
    names=sorted({t['parent_session'] for t in tasks})[rank::6]
    for name in names:
        clips=[t for t in tasks if t['parent_session']==name]
        if all((TEST/METHOD/t['session']/'complete.json').exists() for t in clips):
            for t in clips:
                shutil.copytree(TEST/METHOD/t['session'],RESULT_OUT/'final_test_results'/t['session'],dirs_exist_ok=True)
            continue
        cached=TEST/'rgb'/f'{name}.mp4';shutil.copyfile(clips[0]['source_video'],cached)
        try:
            for raw in clips:
                if (TEST/METHOD/raw['session']/'complete.json').exists():continue
                task=dict(raw,video=str(cached))
                collector=base.Collector(task,METHOD)
                try:multihmr(model,collector);collector.save()
                except Exception:
                    collector.save(partial=True);raise
            for t in clips:
                shutil.copytree(TEST/METHOD/t['session'],RESULT_OUT/'final_test_results'/t['session'],dirs_exist_ok=True)
        finally:cached.unlink(missing_ok=True)


def supervise():
    tasks=prepare();running=[]
    for rank in range(6):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',TMPDIR=str(WORK/'tmp'))
        log=(TEST/'logs'/f'worker_{rank}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--rank',str(rank)],
                           env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        running.append((p,log))
    while any(p.poll() is None for p,_ in running):
        done=sum((TEST/METHOD/t['session']/'complete.json').exists() for t in tasks)
        save(RESULT_OUT/'final_test_status.json',dict(status='running',completed_segments=done,total_segments=320,
             updated_unix=time.time(),workers=[dict(pid=p.pid,exit_code=p.poll()) for p,_ in running]))
        time.sleep(15)
    for p,log in running:log.close();assert p.returncode==0,(p.pid,p.returncode)
    assert all((TEST/METHOD/t['session']/'complete.json').exists() for t in tasks)
    subprocess.run([sys.executable,str(ROOT/'src/utils/report_multihmr.py')],check=True,cwd=ROOT)
    save(RESULT_OUT/'final_test_status.json',dict(status='complete',completed_segments=320,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['supervise','worker']);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args()
    if a.stage=='supervise':supervise()
    else:worker(a.rank)

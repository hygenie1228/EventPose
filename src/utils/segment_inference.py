from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Frozen frame-interval evaluation: Multi-HMR first, then Human3R, GPUs 0..4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import utils.pose_evaluation as base

ROOT=base.ROOT
ARCHIVE=ROOT/'exp/261023/humanpose_gt_segments'
WORK=ROOT/'.experiment_storage/humanpose_gt_segments'
OLD=ROOT/'exp/261023/humanpose_gt_eval'
base.OUT=WORK


def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    for folder in ['gt','logs','rgb','tmp']:(WORK/folder).mkdir(exist_ok=True)
    link=ARCHIVE/'work'
    if not link.exists():link.symlink_to(WORK,target_is_directory=True)
    split=json.loads((ROOT/'dataset/split.json').read_text())
    frozen=json.loads((ARCHIVE/'split_snapshot.json').read_text())
    assert split==frozen
    manifest=json.loads((ARCHIVE/'sampling_manifest.json').read_text())
    assert manifest['unit']=='native RGB frame number'
    old={t['session']:t for t in json.loads((OLD/'tasks.json').read_text())}
    assert set(split['test'])==set(old) and len(old)==64
    shutil.copyfile(OLD/'joint_regressor.npy',WORK/'joint_regressor.npy')
    tasks=[]
    for name,intervals in split['test'].items():
        task=old[name]
        with np.load(task['gt_file']) as z:
            arrays={k:z[k] for k in z.files}
        for i,(start,end) in enumerate(intervals):
            keep=(arrays['source_rgb_positions']>=start)&(arrays['source_rgb_positions']<end)
            assert keep.any()
            key=f'{name}__segment{i}'
            gt=WORK/'gt'/f'{key}.npz'
            values=dict(arrays)
            for field in ['joints_cam_m','valid','source_rgb_positions','timestamps_ns']:values[field]=arrays[field][keep]
            values['original_source_rgb_positions']=values['source_rgb_positions'].copy()
            values['source_rgb_positions']=values['source_rgb_positions']-start
            base.atomic_npz(gt,**values)
            tasks.append(dict(task,session=key,parent_session=name,segment=i,start_frame=start,end_frame=end,
                              source_video=task['video'],gt_file=str(gt),video_frames=end-start,
                              gt_frames=int(keep.sum()),valid_gt_instances=int(values['valid'].sum())))
    base.atomic_json(WORK/'tasks.json',tasks)
    base.atomic_json(WORK/'protocol.json',dict(sampling_manifest=manifest,sessions=64,clips=len(tasks),
        frames=sum(t['video_frames'] for t in tasks),device='device0',gpus=list(range(5)),
        order=['multi_hmr','human3r'],precision='AMP FP16 network, FP32 joint regression and metric geometry',
        multi_hmr_batch=32,human3r_state='reset each segment; sequential within segment',
        population='same GT instances matched by both methods',unit='m',joint_set='SMPL-X body 22; supplement 55'))
    base.atomic_json(OLD/'interrupted_for_segments.json',dict(reason='User selected frozen random frame segments',
                                                           new_experiment=str(ARCHIVE),time=time.time()))
    print('Prepared',len(tasks),'segments',sum(t['video_frames'] for t in tasks),'frames',flush=True)


def frames(task):
    import cv2
    cap=cv2.VideoCapture(task['video']);assert cap.isOpened()
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,task['start_frame'])
    assert round(cap.get(cv2.CAP_PROP_POS_FRAMES))==task['start_frame']
    try:
        for pos in range(task['video_frames']):
            ok,image=cap.read();assert ok
            assert round(cap.get(cv2.CAP_PROP_POS_FRAMES))==task['start_frame']+pos+1
            yield pos,cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    finally:cap.release()


def multihmr(model,c):
    import torch
    from PIL import Image,ImageOps
    import torchvision.transforms as T
    norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    original=model.detection;ids=[];pending=[]
    def detection(*a,**kw):
        out=original(*a,**kw);ids[:]=out[2][0].detach().cpu().tolist();return out
    model.detection=detection
    def flush():
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            people=model(torch.stack([v[1] for v in pending]).cuda(),is_training=False,nms_kernel_size=3,
                         det_thresh=.3,K=torch.from_numpy(np.stack([v[2] for v in pending])).cuda())
        assert len(people)==len(ids)
        grouped=[[] for _ in pending]
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            for idx,p in zip(ids,people):grouped[idx].append((c.reg@p['v3d'].float()).cpu().numpy())
        for (pos,_,_),j in zip(pending,grouped):
            c.current_pos=pos;c.collect(np.stack(j) if j else np.empty((0,55,3),np.float32),c.K)
        pending.clear()
    try:
        for pos,rgb in frames(c.task):
            h,w=rgb.shape[:2];size=model.img_size
            image=ImageOps.contain(Image.fromarray(rgb),(size,size),method=Image.Resampling.LANCZOS)
            nw,nh=image.size;l,t=(size-nw)//2,(size-nh)//2
            image=ImageOps.expand(image,border=(l,t,size-nw-l,size-nh-t))
            K=c.K.copy();K[0]*=nw/w;K[1]*=nh/h;K[0,2]+=l;K[1,2]+=t
            pending.append((pos,norm(image),K))
            if len(pending)==32:flush()
        if pending:flush()
    finally:model.detection=original


def worker(method,rank,smoke=False):
    base.configure_torch();base.rgb_frames=frames
    tasks=json.loads((WORK/'tasks.json').read_text())
    names=sorted({t['parent_session'] for t in tasks})[rank::5]
    if smoke:names=names[:1]
    model=base.load_multihmr() if method=='multi_hmr' else base.load_human3r()
    for name in names:
        clips=[t for t in tasks if t['parent_session']==name]
        if smoke:clips=clips[:1]
        if all((WORK/method/t['session']/'complete.json').exists() for t in clips) and not smoke:continue
        cached=WORK/'rgb'/f'{method}_{name}.mp4'
        before=time.perf_counter();shutil.copyfile(clips[0]['source_video'],cached)
        print(json.dumps(dict(event='stage_rgb',session=name,seconds=time.perf_counter()-before)),flush=True)
        try:
            for raw in clips:
                if not smoke and (WORK/method/raw['session']/'complete.json').exists():continue
                task=dict(raw,video=str(cached))
                if smoke:task.update(video_frames=32,smoke=True)
                c=base.Collector(task,method)
                try:
                    if method=='multi_hmr':multihmr(model,c)
                    else:base.infer_human3r(model,c,amp=True)
                    c.save()
                    assert np.isfinite(c.matched[c.indices>=0]).all()
                except Exception as e:
                    c.save(partial=True);base.atomic_json(c.folder/'error.json',dict(error=repr(e)));raise
            if not smoke:
                for task in clips:
                    shutil.copytree(WORK/method/task['session'],ARCHIVE/'results'/method/task['session'],dirs_exist_ok=True)
        finally:cached.unlink(missing_ok=True)


def supervise():
    for method in ['multi_hmr','human3r']:
        # A failure in the first method prevents starting the second method.
        running=[]
        for rank in range(5):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',TMPDIR=str(WORK/'tmp'))
            log=(WORK/'logs'/f'{method}_{rank}.log').open('ab',buffering=0)
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--method',method,'--rank',str(rank)],
                                  env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            running.append((proc,log))
        while any(p.poll() is None for p,_ in running):
            base.atomic_json(WORK/'status.json',dict(stage=method,status='running',updated_unix=time.time(),
                                                   workers=[dict(pid=p.pid,exit_code=p.poll()) for p,_ in running]))
            time.sleep(10)
        for p,log in running:log.close();assert p.returncode==0,(method,p.pid,p.returncode)
        tasks=json.loads((WORK/'tasks.json').read_text())
        assert all((WORK/method/t['session']/'complete.json').exists() for t in tasks)
    subprocess.run([sys.executable,str(ROOT/'src/utils/report_segments.py')],check=True,cwd=ROOT)
    base.atomic_json(WORK/'status.json',dict(status='complete',updated_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','worker','smoke','supervise'])
    parser.add_argument('--method',choices=['human3r','multi_hmr']);parser.add_argument('--rank',type=int,default=0)
    a=parser.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='supervise':supervise()
    else:worker(a.method,a.rank,smoke=a.stage=='smoke')

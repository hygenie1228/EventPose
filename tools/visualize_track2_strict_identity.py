"""Prepare RGB/mask/head-query evidence for every flagged identity transition.

CPU only; never reads GT poses, changes predictions, excludes metric frames, or
marks a case manually reviewed. Whole-frame checks are separate from renders.
"""
import argparse
import fcntl
import json
import os
import runpy
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
from audit_track2_strict_results import audit_heads,read


def render(folder,sample,manifest,audit):
    raw_sha=audit['raw_sha256'];out=OUTPUT/'strict_identity_visuals'/folder.parent.name/sample
    receipt=out/'receipt.json'
    if receipt.exists():
        old=read(receipt)
        if old['raw_sha256']==raw_sha and old['mask_sha256']==audit['mask_sha256']:
            if all(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in old['images']):return old
    out.mkdir(parents=True,exist_ok=True)
    with np.load(folder/'smplx_predictions.npz') as z:
        xy=z['head_query_native_xy'];valid=z['detection_valid'];roots=z['joints_cam_m'][:,0]
        confidence=z['confidence']
    cap=cv2.VideoCapture(str(SOURCE/sample/'video.mp4'))
    maskcap=cv2.VideoCapture(str(SOURCE/sample/'processed/human_mask.mp4'))
    assert cap.isOpened() and maskcap.isOpened()
    images=[]
    try:
        for flag in audit['temporal_flags']:
            a,b=flag['frames'];frames=sorted(set([max(0,a-1),a,b,min(len(valid)-1,b+1)]))
            panels=[]
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=cap.read();assert ok
                maskcap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,mask=maskcap.read();assert ok
                gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                if gray.shape!=rgb.shape[:2]:gray=cv2.resize(gray,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
                foreground=(gray>127).astype(np.uint8)
                contours,_=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                annotated=rgb.copy();cv2.drawContours(annotated,contours,-1,(30,230,30),2)
                if valid[frame]:
                    center=tuple(np.rint(xy[frame]).astype(int));cv2.circle(annotated,center,8,(0,0,255),2)
                    cv2.drawMarker(annotated,center,(0,0,255),cv2.MARKER_CROSS,18,2)
                else:
                    ys,xs=np.where(foreground)
                    center=(int(np.median(xs)),int(np.median(ys))) if len(xs) else (rgb.shape[1]//2,rgb.shape[0]//2)
                h,w=rgb.shape[:2];scale=640/w
                full=cv2.resize(annotated,(640,round(h*scale)))
                # Close-up preserves the RGB background; the mask is an outline.
                size=min(320,h,w);x0=max(0,min(w-size,center[0]-size//2));y0=max(0,min(h-size,center[1]-size//2))
                close=cv2.resize(annotated[y0:y0+size,x0:x0+size],(full.shape[0],full.shape[0]))
                panel=np.concatenate([full,close],axis=1)
                panel=cv2.copyMakeBorder(panel,56,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                cv2.putText(panel,f'{sample} / frame {frame}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(panel,f'detected={bool(valid[frame])} score={float(confidence[frame]):.4f} pelvis Z={float(roots[frame,2]):.3f}m',
                    (8,36),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(panel,'original target mask: green outline / accepted head query: red',(8,51),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
                panels.append(panel)
            path=out/f'frames_{a}_{b}.jpg';assert cv2.imwrite(str(path),np.concatenate(panels,axis=0))
            images.append(dict(path=str(path),sha256=sha256(path),flag_frames=[a,b],rendered_frames=frames))
    finally:
        cap.release();maskcap.release()
    value=dict(trial=folder.parent.name,sample=sample,raw_sha256=raw_sha,mask_sha256=audit['mask_sha256'],
        frames=[f['frames'] for f in audit['temporal_flags']],images=images,manual_review_pending=bool(images),
        GT_pose_read=False,RGB_background_preserved=True,gpu_used=False,created_unix=time.time())
    save_json(receipt,value);return value


def update():
    cv2.setNumThreads(1)
    from track2_strict_scope import scope
    plan,prefix=scope()
    manifest=read(OUTPUT/'manifest.json');models=plan['models']
    selection_path=OUTPUT/(prefix+'_selected.json')
    selected=read(selection_path)['trial'] if selection_path.exists() else None
    cases={}
    def save_status(in_progress):
        save_json(OUTPUT/'strict_identity_visual_status.json',dict(pid=os.getpid(),cases=cases,
            fully_checked_clips=len(cases),flagged_transitions=sum(v['flags'] for v in cases.values()),
            evaluation_scope_prefix=prefix,body_target_overlap_checked=prefix.startswith('body_target'),
            scan_in_progress=in_progress,rendering_does_not_imply_manual_review=True,
            GT_pose_read=False,updated_unix=time.time()))
    for model in models:
        full_samples=manifest['test']+(manifest['train'] if model['name']==selected else [])
        for full,samples in [(False,manifest['validation']),(True,full_samples)]:
            ckpt=model['full_checkpoint' if full else 'search_checkpoint']
            trial=model['name'] if full or ckpt is None else 'search_'+model['name']
            for sample in samples:
                folder=OUTPUT/'predictions'/trial/sample
                if not (folder/'inference.json').exists():continue
                info=read(folder/'inference.json')
                assert info['evaluation_protocol']==model['evaluation_protocol']
                assert info['checkpoint_sha256']==model['full_checkpoint_sha256' if full else 'search_checkpoint_sha256']
                audit=audit_heads(folder,sample,manifest,info)
                visual=render(folder,sample,manifest,audit)
                cases[trial+'/'+sample]=dict(raw_sha256=audit['raw_sha256'],frames=audit['frames'],
                    flags=len(audit['temporal_flags']),visual_receipt=str(OUTPUT/'strict_identity_visuals'/trial/sample/'receipt.json'),
                    manual_review_pending=visual['manual_review_pending'])
                save_status(True)
    save_status(False)
    print(json.dumps(dict(clips=len(cases),flags=sum(v['flags'] for v in cases.values()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    with (OUTPUT/'locks/strict_identity_visual.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if args.watch:
                runpy.run_path(str(Path(__file__).resolve()))['update']()
            else:update()
            if not args.watch or time.time()>=read(OUTPUT/'budget.json')['deadline_unix']:break
            for _ in range(10):time.sleep(60)

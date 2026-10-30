"""CPU-only Train-validation camera overlays for pose/coordinate diagnostics."""
from pathlib import Path
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import SOURCE,OUTPUT,ASSETS,decode_mhr,save_json
from utils.v2d_official_metric import MHR_TABLE3_BODY_JOINT_INDICES

PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]


def joints(model,path,frames):
    with np.load(path) as z:
        pose=torch.as_tensor(z['pose'][frames],dtype=torch.float32)
        shape=torch.as_tensor(z['shape'],dtype=torch.float32)
        scales=torch.as_tensor(z['scales'],dtype=torch.float32)
    with torch.no_grad():_,value=decode_mhr(model,pose,shape,scales)
    return value[:,list(MHR_TABLE3_BODY_JOINT_INDICES)].numpy()


def draw(image,points,K,color):
    projected=points@K.T
    uv=projected[:,:2]/np.maximum(projected[:,2:],1e-6)
    valid=(points[:,2]>.05)&np.isfinite(uv).all(1)
    for child,parent in enumerate(PARENTS):
        if parent>=0 and valid[child] and valid[parent]:
            cv2.line(image,tuple(np.rint(uv[child]).astype(int)),tuple(np.rint(uv[parent]).astype(int)),color,2,cv2.LINE_AA)
    for point,ok in zip(uv,valid):
        if ok:cv2.circle(image,tuple(np.rint(point).astype(int)),3,color,-1,cv2.LINE_AA)


def main(split='validation',samples=None):
    torch.set_num_threads(2)
    model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    out=OUTPUT/('test_visuals' if split=='test' else 'validation_visuals');out.mkdir(exist_ok=True)
    chosen=samples or manifest[split]
    assert set(chosen)<=set(manifest[split])
    receipt={}
    for sample in chosen:
        record=manifest['samples'][sample];frames=[0,record['frames']//2,record['frames']-1]
        E=np.asarray(record['world_to_camera'],np.float32)
        gt=joints(model,SOURCE/sample/'human/human_params_gt.npz',frames)
        gt=gt@E[:3,:3].T+E[:3,3]
        cases={'v6 init':SOURCE/sample/'human/human_params_init.npz',
            'Multi-HMR':OUTPUT/'predictions/original'/sample/'human_params_mhr.npz',
            'Multi-HMR Last2':OUTPUT/'predictions'/('last2_strong' if split=='test' else 'search_last2_strong')/sample/'human_params_mhr.npz'}
        predictions={name:joints(model,path,frames) for name,path in cases.items()}
        cap=cv2.VideoCapture(str(SOURCE/sample/'video.mp4'));assert cap.isOpened()
        scale=480/record['width'];height=round(record['height']*scale)
        K=np.asarray(record['K'],np.float32).copy();K[:2]*=scale
        rows=[]
        for index,frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=cap.read();assert ok
            panels=[]
            for name,values in predictions.items():
                panel=cv2.resize(rgb,(480,height))
                draw(panel,gt[index],K,(40,230,40));draw(panel,values[index],K,(30,30,245))
                panel=cv2.copyMakeBorder(panel,45,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                cv2.putText(panel,name+f' / frame {frame}',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(panel,f'GT pelvis Z {gt[index,0,2]:.2f}m / prediction {values[index,0,2]:.2f}m',
                    (8,35),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
                panels.append(panel)
            rows.append(np.concatenate(panels,axis=1))
        cap.release();image=np.concatenate(rows,axis=0)
        destination=out/(sample+'.jpg');assert cv2.imwrite(str(destination),image)
        receipt[sample]=dict(path=str(destination),frames=frames,
            root_depth_errors_m={name:(values[:,0,2]-gt[:,0,2]).tolist() for name,values in predictions.items()},
            gt_color='green',prediction_color='red',GT_used_only_for_visual_evaluation=True)
        print(sample,receipt[sample]['root_depth_errors_m'],flush=True)
    save_json(out/'receipt.json',dict(samples=receipt,split=split,gpu_used=False,created_unix=time.time(),
        note='Unaligned camera overlays; GT world-to-camera applied exactly once; predictions remain camera-space'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],default='validation');parser.add_argument('--samples',nargs='*');args=parser.parse_args();main(args.split,args.samples)

"""Freeze a body-extent gate from Train4 predictions and masks only."""
import cv2,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256


def main():
    cv2.setNumThreads(1)
    read=lambda p:json.loads(p.read_text())
    manifest=read(OUTPUT/'manifest.json');plan=read(OUTPUT/'strict_evaluation_plan.json');boxes={}
    for sample in manifest['validation']:
        record=manifest['samples'][sample];cap=cv2.VideoCapture(str(SOURCE/sample/'processed/human_mask.mp4'));assert cap.isOpened()
        rows=[]
        for _ in range(record['frames']):
            ok,mask=cap.read();assert ok
            gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            if gray.shape!=(record['height'],record['width']):gray=cv2.resize(gray,(record['width'],record['height']),interpolation=cv2.INTER_NEAREST)
            y,x=np.where(gray>127);rows.append([x.min(),y.min(),x.max()+1,y.max()+1] if len(x) else [0,0,0,0])
        cap.release();boxes[sample]=np.asarray(rows,np.float64)
    models={}
    for model in plan['models']:
        trial=model['name'] if model['search_checkpoint'] is None else 'search_'+model['name'];values=[];per_sequence={}
        for sample in manifest['validation']:
            folder=OUTPUT/'predictions'/trial/sample
            with np.load(folder/'smplx_predictions.npz') as z:
                joints=z['joints_cam_m'][:,:22].astype(np.float64);valid=z['detection_valid']
            record=manifest['samples'][sample];xyz=joints@np.asarray(record['K'],np.float64).T
            xy=xyz[:,:,:2]/np.maximum(xyz[:,:,2:],1e-8)
            lo=np.maximum(xy.min(1),[0,0]);hi=np.minimum(xy.max(1),[record['width'],record['height']])
            target=boxes[sample];inter=np.maximum(np.minimum(hi,target[:,2:])-np.maximum(lo,target[:,:2]),0).prod(1)
            union=np.maximum(hi-lo,0).prod(1)+(target[:,2:]-target[:,:2]).prod(1)-inter
            iou=inter/np.maximum(union,1e-8);iou[(xyz[:,:,2]<=.05).any(1)]=0
            values.extend(iou[valid].tolist());per_sequence[sample]=dict(frames=len(valid),observed=int(valid.sum()),
                q01=float(np.quantile(iou[valid],.01)),median=float(np.median(iou[valid])),raw_sha256=sha256(folder/'smplx_predictions.npz'))
        models[model['name']]=dict(q01=float(np.quantile(values,.01)),median=float(np.median(values)),per_sequence=per_sequence)
    threshold=min(v['q01'] for v in models.values())
    result=dict(models=models,minimum_body_box_iou=threshold,threshold_rule='Minimum of per-model Train4 observed-prediction 1st-percentile body/mask box IoU',
        ranking='Highest body-mask box IoU among strict-head-mask candidates passing threshold; confidence tie-break',
        Test_frames_read=False,GT_pose_read=False,Test_metrics_read=False,selection_uses_2D_target_mask_and_predicted_body_only=True,frozen_unix=time.time())
    destination=OUTPUT/'body_target_gate_Train_rule.json'
    if destination.exists():assert read(destination)['models']==models and read(destination)['minimum_body_box_iou']==threshold
    else:save_json(destination,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

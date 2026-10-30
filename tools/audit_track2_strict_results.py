"""Check complete strict-mask model results and flag possible identity jumps.

Flags are diagnostic only: never filter metric frames or choose model outputs.
"""
import argparse
import cv2
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256


def read(path):return json.loads(path.read_text())


def audit_case(model,sample,full,manifest,reference):
    name=model['name'];checkpoint=model['full_checkpoint' if full else 'search_checkpoint']
    trial=name if full or checkpoint is None else 'search_'+name
    folder=OUTPUT/'predictions'/trial/sample
    value=read(folder/'metrics.json');info=read(folder/'inference.json')
    record=manifest['samples'][sample]
    assert value['evaluation_protocol']==info['evaluation_protocol']==model['evaluation_protocol']
    assert info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
    assert not info['gt_used_for_detection_or_tracking'] and not info['target_foreground_only_input']
    assert info.get('depth_decoder','published')=='published'
    assert info['checkpoint_sha256']==(sha256(Path(checkpoint)) if checkpoint else None)
    assert info['checkpoint_sha256']==model['full_checkpoint_sha256' if full else 'search_checkpoint_sha256']
    assert value['frames']==record['frames'] and value['full_frame_roster']
    assert value['conversion_revision']==2 and value['protocol']==reference
    assert value['gt_sha256']==record['gt_sha256']==sha256(SOURCE/sample/'human/human_params_gt.npz')
    assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
    assert all(math.isfinite(value[k]) for k in ['CD_H_cm','ACC_H_cm_per_frame2'])
    return value,info,audit_heads(folder,sample,manifest,info)


def audit_heads(folder,sample,manifest,info=None):
    """Independently check every accepted head, including unscored Train exports."""
    info=info or read(folder/'inference.json')
    record=manifest['samples'][sample]
    assert info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
    assert not info['gt_used_for_detection_or_tracking'] and not info['target_foreground_only_input']
    raw_sha=sha256(folder/'smplx_predictions.npz');mask_path=SOURCE/sample/'processed/human_mask.mp4'
    mask_sha=sha256(mask_path);assert mask_sha==info['target_mask_sha256']
    body_rule=read(OUTPUT/'body_target_gate_Train_rule.json') if info.get('target_body_overlap') else None
    body_rule_sha=sha256(OUTPUT/'body_target_gate_Train_rule.json') if body_rule else None
    if body_rule:assert info['body_box_rule_sha256']==body_rule_sha and info['body_box_min_iou']==body_rule['minimum_body_box_iou']
    pixel_audit=folder/'strict_independent_head_audit.json'
    if pixel_audit.exists():
        cached=read(pixel_audit)
        if cached['raw_sha256']==raw_sha and cached['mask_sha256']==mask_sha and cached.get('body_rule_sha256')==body_rule_sha:
            assert cached['outside_original_mask']==0
            return cached
    with np.load(folder/'smplx_predictions.npz') as z:
        valid=z['detection_valid'];queries=z['head_query_native_xy']
        if 'head_query_native_pixel' in z:pixels=z['head_query_native_pixel']
        else:
            pixels=np.full((len(valid),2),-1,np.int32);pixels[valid]=np.rint(queries[valid]).astype(np.int32)
        joints=z['joints_cam_m'];roots=joints[:,0];camera=z['K']
        if body_rule:stored_body_iou=z['target_body_box_iou']
        assert np.array_equal(z['frames'],np.arange(record['frames']))
        assert np.array_equal(z['K'],np.asarray(record['K'],np.float32))
    assert np.isfinite(queries[valid]).all() and np.isfinite(roots).all()
    assert info['imputed_frames']==np.flatnonzero(~valid).tolist()
    cap=cv2.VideoCapture(str(mask_path));assert cap.isOpened()
    outside=[]
    for i in range(len(valid)):
        ok,mask=cap.read();assert ok
        if not valid[i]:continue
        gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        if gray.shape!=(record['height'],record['width']):
            gray=cv2.resize(gray,(record['width'],record['height']),interpolation=cv2.INTER_NEAREST)
        x,y=pixels[i]
        if not (0<=x<record['width'] and 0<=y<record['height'] and gray[y,x]>127):outside.append(i)
        if body_rule:
            from multihmr.target_crop import target_body_box_iou
            actual=target_body_box_iou(joints[i],camera,gray>127)
            assert abs(actual-float(stored_body_iou[i]))<1e-6
            assert actual>=body_rule['minimum_body_box_iou']-1e-8,(sample,i,actual)
    cap.release();assert not outside,(folder.parent.name,sample,outside[:20])
    root_step=np.linalg.norm(np.diff(roots,axis=0),axis=1)*100
    head_step=np.linalg.norm(np.diff(queries,axis=0),axis=1)
    observed_pairs=valid[:-1]&valid[1:]
    flagged=np.flatnonzero((root_step>50)|(observed_pairs&(head_step>75)))
    flags=[dict(frames=[int(i),int(i+1)],root_step_cm=float(root_step[i]),
        head_step_px=float(head_step[i]) if observed_pairs[i] else None,both_detected=bool(observed_pairs[i])) for i in flagged]
    audit=dict(frames=len(valid),accepted_heads=int(valid.sum()),outside_original_mask=0,
        raw_sha256=raw_sha,mask_sha256=mask_sha,GT_pose_read=False,verified_unix=time.time(),
        temporal_flags=flags,temporal_flags_manual_review_pending=bool(flags),
        flag_thresholds=dict(root_step_cm=50,observed_head_step_px=75),
        flags_do_not_change_prediction_selection_or_metrics=True)
    if body_rule:audit.update(body_rule_sha256=body_rule_sha,body_box_violations=0,target_body_overlap=True)
    save_json(pixel_audit,audit)
    return audit


def main(split,new_only):
    cv2.setNumThreads(1)
    from track2_strict_scope import scope
    manifest=read(OUTPUT/'manifest.json');plan,prefix=scope();models=plan['models']
    samples=manifest[split];full=split=='test';count,frames=(10,7610) if full else (4,2714)
    assert len(samples)==count
    reference=read(OUTPUT/'init_comparison.json')['models']['init_v6']['per_sequence'][manifest['test'][0]]['protocol']
    independent_prefix='body_target' if prefix.startswith('body_target') else 'strict'
    destination=OUTPUT/(independent_prefix+('_independent_Test_models.json' if full else '_independent_Train4_models.json'))
    result=read(destination) if destination.exists() else dict(models={})
    result.update(evaluation_split=split,whole_roster_only=True,strict_target_mask=True)
    added=[]
    for model in models:
        name=model['name'];checkpoint=model['full_checkpoint' if full else 'search_checkpoint']
        trial=name if full or checkpoint is None else 'search_'+name
        if new_only and name in result['models']:continue
        if not all((OUTPUT/'predictions'/trial/s/'metrics.json').exists() for s in samples):continue
        values={};coverage=[];flag_count=0
        for sample in samples:
            value,info,audit=audit_case(model,sample,full,manifest,reference)
            if 'body_box_v3' in model['evaluation_protocol']:
                assert info['target_body_overlap'] and audit['target_body_overlap'] and audit['body_box_violations']==0
            values[sample]=value;coverage.append(info['detection_frame_rate']);flag_count+=len(audit['temporal_flags'])
        assert sum(v['frames'] for v in values.values())==frames
        summary={k:sum(v[k] for v in values.values())/count for k in ['CD_H_cm','ACC_H_cm_per_frame2']}
        summary.update(sequences=count,frames=frames,per_sequence=values,target_coverage=sum(coverage)/count,
            strict_target_mask=True,accepted_heads_outside_original_mask=0,temporal_flag_count=flag_count,
            temporal_flags_manual_review_pending=bool(flag_count),test_used_for_selection=False,checked_unix=time.time())
        if 'body_box_v3' in model['evaluation_protocol']:summary.update(target_body_overlap=True,body_box_violations=0)
        result['models'][name]=summary;added.append(name)
        result['updated_unix']=time.time();save_json(destination,result)
        print(name,json.dumps({k:v for k,v in summary.items() if k!='per_sequence'}),flush=True)
    result.update(evaluation_split=split,whole_roster_only=True,strict_target_mask=True,updated_unix=time.time())
    save_json(destination,result);print(json.dumps(dict(newly_checked=added,models=list(result['models']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['validation','test'],default='test');parser.add_argument('--new-only',action='store_true')
    args=parser.parse_args();main(args.split,args.new_only)

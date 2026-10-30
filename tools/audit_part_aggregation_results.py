"""Verify completed Test artifacts without new inference or metric estimation."""
import argparse
import hashlib
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261029/part_aware_dynamic_aggregation'
def read(path):return json.loads(path.read_text())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def training_completion(config,run):
    """Distinguish the update target from a fully consumed authorized time cap."""
    assert run['status']=='complete'
    actual=run['successful_steps'];target=config['max_successful_steps']
    assert 0<actual<=target
    assert run.get('matched_update_target_reached',actual==target)==(actual==target)
    elapsed=run['elapsed_seconds'];ended=run['completed_unix'];started=ended-elapsed
    assert elapsed>0 and started<config['deadline_unix']-300 and ended<=config['deadline_unix']
    reason='update_target_reached'
    if actual<target:
        # The worker reports elapsed time over this same started/completed
        # interval; a non-interrupted short run must have reached a real cap.
        local_finish=started+config['hours']*3600
        global_finish=config['deadline_unix']-300
        expected=min(local_finish,global_finish)
        assert elapsed>0 and ended>=expected-1,'Incomplete target without consuming its authorized cap'
        reason='fixed_global_deadline_cap' if global_finish<local_finish else 'configured_training_time_cap'
    return dict(actual_successful_updates=actual,configured_maximum_updates=target,
                update_target_reached=actual==target,completion_reason=reason,
                actual_elapsed_seconds=run['elapsed_seconds'])

def audit(names,label,comparison):
    configs={name:read(OUT/'configs'/(name+'.json')) for name in names}
    first=configs[names[0]];source=Path(first['source_root'])
    tasks=read(source/'test/tasks.json');assert len(tasks)==80
    summary=read(OUT/comparison)
    assert summary['all_complete'] and set(summary['completed'])==set(names)
    checkpoint_hashes={};training={};counts=dict.fromkeys(names,0);hashes=[];common_count=0;GT_count=0
    for name in names:
        c=configs[name];run=read(OUT/'runs'/name/'complete.json')
        training[name]=training_completion(c,run)
        assert c['split_sha256']==first['split_sha256'] and c['deadline_unix']==1793321712
        checkpoint=OUT/'checkpoints'/name/'best.pt';checkpoint_hashes[name]=sha(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert saved['test_used_for_selection'] is False and saved['successful_steps']>0
        assert saved['metrics_apply_to_checkpoint'] and saved['metrics_step']==saved['step']
        best=read(OUT/'runs'/name/'best.json');assert saved['step']==best['step']
        assert saved['config']['name']==name and saved['config']['split_sha256']==c['split_sha256']
        del saved
    for task in tasks:
        assert not task.get('smoke')
        with np.load(task['gt_file'],allow_pickle=False) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
        assert positions[0]==0 and len(positions)==task['video_frames']
        GT_count+=int(valid.sum());common=valid.copy()
        for name in names:
            folder=OUT/'results'/name/task['session'];receipt=read(folder/'complete.json');record=receipt['task']
            assert not receipt.get('partial') and not record.get('smoke')
            assert record['split_sha256']==first['split_sha256'] and record['checkpoint_sha256']==checkpoint_hashes[name]
            for key in ['parent_session','start_frame','end_frame','video_frames']:assert record[key]==task[key]
            path=folder/'result.npz'
            with np.load(path,allow_pickle=False) as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                mask=valid&(z['matched_pred_indices']>=0)
                assert z['matched_joints_cam_m'].shape==gt.shape and np.isfinite(z['matched_joints_cam_m'][mask]).all()
            counts[name]+=int(mask.sum());common&=mask
            hashes.append(dict(condition=name,segment=task['session'],sha256=sha(path)))
        common_count+=int((common&common[0][None]).sum())
    assert GT_count==summary['GT_instances']
    for name in names:
        assert counts[name]==summary['coverage'][name]['matched']
        assert common_count==summary['metrics'][name]['samples']
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        conditions=names,segments_each=len(tasks),prediction_files=len(hashes),GT_instances=GT_count,
        common_metric_samples=common_count,own_matched_instances=counts,
        training_completion=training,
        actual_checkpoint_sha256=checkpoint_hashes,raw_prediction_sha256=hashes,
        scope='Completed artifacts only: all scheduled frames, shapes, finite matches, exact frame0/common population, checkpoint bytes and recorded Train-selected step; no inference or metric/statistical re-estimation',
        limitation='Recorded Train selection flags and history provenance do not alone prove absence of every possible external Test-dependent decision')
    folder=OUT/'artifact_audits';folder.mkdir(exist_ok=True)
    (folder/(label+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_prediction_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names',nargs='+',required=True);parser.add_argument('--label',required=True)
    parser.add_argument('--comparison',required=True);args=parser.parse_args()
    audit(args.names,args.label,args.comparison)

"""Recompute raw common-GT metrics, prespecified effects, and coverage."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from device3_encoding_scope import GOAL,EVAL_NAMES,PAIRS
from goal_runtime_control import write,register_job
import numpy as np
from utils.compare_event_methods import segment_metrics,summarize
from utils.paired_statistics import paired_cluster,holm_adjust,motion_strata
from finalize_device34_body import independent_errors
def read(p):return json.loads(Path(p).read_text())

def analyze(g):
    out=Path(g['output_root']);tasks=read(out/'test/tasks.json');manifest=read(out/'manifest.json')
    plan=read(out/'statistical_analysis_plan.json')
    assert plan['conditions']==EVAL_NAMES and [tuple(p) for p in plan['pairs']]==PAIRS
    assert plan['Test_metrics_examined_before_plan'] is False
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['split_sha256']
    rows={n:[] for n in EVAL_NAMES};own={n:[] for n in EVAL_NAMES};coverage=dict.fromkeys(EVAL_NAMES,0);total=0;maximum=0.;hashes=[]
    motion={n:{k:dict(count=0,sum_mm=0.) for k in ('still','slow','fast','unknown')} for n in EVAL_NAMES}
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions'];stamps=z['timestamps_ns']
            assert positions[0]==0 and len(positions)==task['video_frames']
            assert np.array_equal(z['original_source_rgb_positions'],positions+task['start_frame'])
        total+=int(valid.sum());predictions={};masks={}
        for n in EVAL_NAMES:
            folder=out/'results'/n/task['session'];meta=read(folder/'complete.json')['task'];cfg=read(out/'configs'/(n+'.json'))
            assert meta['split_sha256']==manifest['split_sha256'] and meta['condition']==cfg['condition']
            assert meta['checkpoint']==str(out/'checkpoints'/n/'best.pt')
            assert meta['matching_protocol']=='primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1'
            for k in ('parent_session','start_frame','end_frame','video_frames'):assert meta[k]==task[k]
            p=folder/'result.npz'
            with np.load(p) as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                pred=z['matched_joints_cam_m'].astype(np.float64);mask=valid&(z['matched_pred_indices']>=0)
                assert pred.shape==gt.shape and np.isfinite(pred[mask]).all() and np.isnan(pred[valid&~mask]).all()
            predictions[n]=pred;masks[n]=mask;coverage[n]+=int(mask.sum())
            hashes.append(dict(name=n,segment=task['session'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            v,_,_=segment_metrics(pred,gt,mask,positions,task['fps'],timestamps_ns=stamps);own[n].append(v)
        common=np.logical_and.reduce(list(masks.values()));strata=motion_strata(gt,valid,positions,task['fps'],timestamps_ns=stamps)
        for n in EVAL_NAMES:
            row,errors,eligible=segment_metrics(predictions[n],gt,common,positions,task['fps'],timestamps_ns=stamps)
            sums,count=independent_errors(predictions[n],gt,common,positions);expected=np.array(row['alignment_sums_m']+[row['pelvis_sum_m']])
            assert count==row['count'] and np.allclose(sums,expected,rtol=1e-6,atol=1e-7)
            maximum=max(maximum,float(np.max(np.abs(sums-expected))))
            row.update(session=task['session'],parent_session=task['parent_session'],cluster=int(task['parent_session'].split('_')[3]));rows[n].append(row)
            for label,code in [('still',0),('slow',1),('fast',2),('unknown',-1)]:
                keep=eligible&(strata==code);motion[n][label]['count']+=int(keep.sum());motion[n][label]['sum_mm']+=float(errors[keep,3].sum()*1000)
    tests={};clusters=sorted({r['cluster'] for r in rows['rgb3']})
    for a,b in PAIRS:
        for label,index in [('pa',3),('root_position',4)]:
            sums={};counts=None
            for n in (a,b):
                selected=[[r for r in rows[n] if r['cluster']==c] for c in clusters]
                sums[n]=[1000*sum(r['alignment_sums_m'][index] if index<4 else r['pelvis_sum_m'] for r in rr) for rr in selected]
                current=[sum(r['count'] for r in rr) for rr in selected]
                if counts is not None:assert current==counts
                counts=current
            tests[a+'__'+b+'_'+label]=paired_cluster(sums[a],sums[b],counts,seed=261028)
    for suffix in ('_pa','_root_position'):
        for k,v in holm_adjust({k:v['p_value'] for k,v in tests.items() if k.endswith(suffix)}).items():tests[k]['holm_p_value']=v
    result=dict(conditions=EVAL_NAMES,segments=len(tasks),frames=sum(t['video_frames'] for t in tasks),gt_instances=total,
        metrics={n:summarize(rows[n]) for n in EVAL_NAMES},own_metrics={n:summarize(own[n]) for n in EVAL_NAMES},tests=tests,motion_strata=motion,
        coverage={n:dict(matched=coverage[n],total=total,fraction=coverage[n]/total) for n in EVAL_NAMES},clusters=clusters,
        independent_maximum_sum_difference_m=maximum,used_for_checkpoint_or_encoding_selection=False,updated_unix=time.time())
    for v in result['metrics'].values():
        if v['samples']:assert v['frame_aligned_pa_mpjpe_mm']<=v['frame_aligned_mpjpe_mm']+1e-5
    write(out/'segment_sums.json',rows);write(out/'raw_prediction_hashes.json',hashes);write(out/'comparison.json',result);return result
if __name__=='__main__':
    g=read(GOAL);register_job(g,os.getpid(),'encoding_full_raw_analysis');analyze(g)

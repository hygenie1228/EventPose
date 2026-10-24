"""First-frame-fixed versus per-frame alignment on the exact 10-second clips."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .humanpose_metrics import joint_errors

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'exp/261023/multihmr_finetune_14h'
WORK=ROOT/'.experiment_storage/multihmr_finetune_14h'
SOURCE=ROOT/'.experiment_storage/humanpose_gt_segments'
METRICS=['first_frame_aligned_mpjpe_mm','first_frame_aligned_pa_mpjpe_mm',
         'frame_aligned_mpjpe_mm','frame_aligned_pa_mpjpe_mm']

def similarity(pred,target):
    """Fit a proper row-vector Sim(3), optionally batched over skeletons."""
    p=np.asarray(pred,np.float64);t=np.asarray(target,np.float64)
    pm=p.mean(-2,keepdims=True);tm=t.mean(-2,keepdims=True)
    x=p-pm;y=t-tm
    u,s,vh=np.linalg.svd(np.swapaxes(x,-1,-2)@y)
    d=np.ones_like(s);d[...,-1]=np.linalg.det(u@vh)
    r=(u*d[...,None,:])@vh
    scale=(s*d).sum(-1)/np.square(x).sum((-2,-1))
    assert np.all(scale>0)
    shift=tm-scale[...,None,None]*(pm@r)
    return r,scale,shift

def aligned_errors(pred,target,anchor_pred,anchor_target):
    """Return four per-person-frame errors in metres, in METRICS order."""
    p=np.asarray(pred,np.float64);t=np.asarray(target,np.float64)
    ap=np.asarray(anchor_pred,np.float64);at=np.asarray(anchor_target,np.float64)
    first_root=p+(at[0]-ap[0])
    r,s,shift=similarity(ap,at)
    first_pa=s*(p@r)+shift
    frame_root=(p-p[:,:1])-(t-t[:,:1])
    fr,fs,ft=similarity(p,t)
    frame_pa=fs[:,None,None]*(p@fr)+ft
    return np.stack([np.linalg.norm(first_root-t,axis=-1).mean(-1),
                     np.linalg.norm(first_pa-t,axis=-1).mean(-1),
                     np.linalg.norm(frame_root,axis=-1).mean(-1),
                     np.linalg.norm(frame_pa-t,axis=-1).mean(-1)],axis=-1)

def main():
    tasks=json.loads((SOURCE/'tasks.json').read_text());assert len(tasks)==320
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['test_split_sha256']
    final=WORK/'final_test/multihmr_finetuned'
    final_ready=all((final/t['session']/'complete.json').exists() for t in tasks)
    if final_ready:
        chosen=OUT;finetuned=final
    else:
        chosen=OUT/'test_snapshots/20261024_040315'
        finetuned=WORK/'test_snapshots/20261024_040315/multihmr_finetuned'
    label='Multi-HMR'
    selected=json.loads((chosen/'selected.json').read_text());assert selected['test_used_for_selection'] is False
    paths={'pretrained':SOURCE/'multi_hmr','finetuned':finetuned}
    totals={k:np.zeros(4) for k in paths};rows=[];count=anchors=skipped=checks=0
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
            assert np.array_equal(z['original_source_rgb_positions'],positions+task['start_frame'])
        predictions={};masks=[]
        for method,path in paths.items():
            folder=path/task['session'];complete=json.loads((folder/'complete.json').read_text())
            assert complete['task']['video_frames']==task['end_frame']-task['start_frame']
            assert not complete['task'].get('smoke')
            with np.load(folder/'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                p=z['matched_joints_cam_m'];assert p.shape==gt.shape
                mask=valid&(z['matched_pred_indices']>=0)&np.isfinite(p).all((2,3))
                predictions[method]=p;masks.append(mask)
        common=np.logical_and.reduce(masks)
        first=np.flatnonzero(positions==0)
        eligible=common[first[0]] if len(first) else np.zeros(gt.shape[1],bool)
        anchors+=int(eligible.sum());skipped+=int((valid.any(0)&~eligible).sum())
        sums={k:np.zeros(4) for k in paths};chunk_count=0
        for person in np.flatnonzero(eligible):
            keep=common[:,person];t=gt[keep,person,:22].astype(np.float64)
            anchor_t=gt[first[0],person,:22].astype(np.float64)
            chunk_count+=len(t)
            for method,prediction in predictions.items():
                p=prediction[keep,person,:22].astype(np.float64)
                anchor_p=prediction[first[0],person,:22].astype(np.float64)
                errors=aligned_errors(p,t,anchor_p,anchor_t);assert np.isfinite(errors).all()
                # Cross-check per-frame metrics against the previously audited routine.
                for i in {0,len(p)-1}:
                    old=joint_errors(p[i],t[i]);assert np.allclose(errors[i,2:],old[1:],atol=1e-10)
                    checks+=1
                # Independent first-frame Sim(3) check using SciPy rotation fitting.
                x=anchor_p-anchor_p.mean(0);y=anchor_t-anchor_t.mean(0)
                rotation,_=Rotation.align_vectors(y,x);rx=rotation.apply(x)
                scale=(rx*y).sum()/np.square(x).sum()
                independent=scale*rotation.apply((p-anchor_p.mean(0)).reshape(-1,3)).reshape(p.shape)+anchor_t.mean(0)
                assert np.allclose(np.linalg.norm(independent-t,axis=-1).mean(-1),errors[:,1],atol=1e-8)
                sums[method]+=errors.sum(0)
        count+=chunk_count
        for method in paths:
            totals[method]+=sums[method]
            row=dict(session=task['parent_session'],segment=task['segment'],start_frame=task['start_frame'],
                end_frame=task['end_frame'],method=method,person_frames=chunk_count,anchor_people=int(eligible.sum()))
            row.update(dict(zip(METRICS,(1000*sums[method]/chunk_count).tolist() if chunk_count else [None]*4)))
            rows.append(row)
    assert count>0
    result=dict(status='complete',updated_unix=time.time(),segments=320,test_sessions=64,
        rgb_frames=sum(t['video_frames'] for t in tasks),paired_person_frames=count,
        eligible_person_chunks=anchors,excluded_person_chunks=skipped,independent_checks=checks,
        checkpoint=selected['checkpoint'],model_label=label,final_test_complete=final_ready,
        metrics={k:dict(zip(METRICS,(1000*v/count).tolist())) for k,v in totals.items()},
        protocol='Exact clip frame 0 only; root translation for MPJPE, proper Sim(3) for PA; same eligible paired population for all four metrics.')
    (chosen/'chunk_alignment_metrics.json').write_text(json.dumps(result,indent=2))
    with (chosen/'chunk_alignment_per_segment.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Multi-HMR fine-tuning — Test 10초 chunk 평가','',
        '## Test 성능','',
        f'Test 64세션·device0, 영상당 10초 × 5개 고정 구간: 총 320 chunk·{result["rgb_frames"]:,} RGB 프레임. '
        'dataset/split.json의 원본 프레임 번호 [start,end)를 그대로 사용했다. 신체 22관절 기준이며 모든 오차는 **mm**다.','',
        '| 모델 | First frame aligned MPJPE ↓ | First frame aligned PA-MPJPE ↓ | Frame aligned MPJPE ↓ | Frame aligned PA-MPJPE ↓ |',
        '|---|---:|---:|---:|---:|']
    for method,name in [('pretrained','Baseline · Multi-HMR'),('finetuned','Fine-tuned Multi-HMR')]:
        if method not in result['metrics']:
            lines.append('| '+name+' | — | — | — | — |');continue
        v=result['metrics'][method];cells=[]
        for k in METRICS:
            cell=f'{v[k]:.2f}'
            if np.isclose(v[k],min(m[k] for m in result['metrics'].values()),rtol=0,atol=.005):cell=f'**{cell}**'
            cells.append(cell)
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['',f'네 지표 모두 동일한 **{count:,} 사람-프레임**을 사용했다. '
        f'각 chunk의 정확한 첫 프레임에서 평가 대상 모델에 공통 매칭된 사람-chunk {anchors:,}개가 대상이다. '
        f'첫 프레임에 유효 GT나 공통 검출이 없는 사람-chunk {skipped:,}개는 제외했다. '
        '나중의 첫 매칭 프레임으로 대체하지 않았다. 미검출을 0 오차로 처리하지 않았다.','',
        '## 정렬 정의','',
        '- **First frame aligned MPJPE:** 각 사람·10초 chunk의 첫 프레임에서 예측 pelvis를 GT pelvis로 옮기는 translation 하나를 구하고, 구간 전체에 고정 적용한다. 회전·스케일은 바꾸지 않는다.',
        '- **First frame aligned PA-MPJPE:** 첫 프레임의 22관절만으로 proper Sim(3)의 회전·양의 등방 스케일·translation을 구하고, 같은 변환을 구간 전체에 고정 적용한다.',
        '- **Frame aligned MPJPE:** 매 프레임·각 사람마다 pelvis translation을 새로 맞춘다. 기존 Root-aligned MPJPE와 같은 정렬이다.',
        '- **Frame aligned PA-MPJPE:** 매 프레임·각 사람마다 proper Sim(3)를 새로 맞춘다. Reflection은 허용하지 않는다.','',
        '첫 프레임 이후 GT로 변환을 다시 추정하는 것은 frame aligned 두 지표뿐이다. '
        'First-frame PA도 첫 프레임에서 scale을 제거하므로 절대 metric scale 정확도 자체를 검증하는 지표는 아니다.','',
        'First-frame PA는 첫 프레임에서만 최적화한 변환이므로, 이후 프레임까지 평균하면 first-frame MPJPE보다 오차가 커질 수도 있다.','',
        '## 모델과 검증','',f'사용한 fine-tuned checkpoint: `{selected["checkpoint"]}`. '
        '학습 내부 holdout으로 모델을 선택했으며 Test는 가중치 업데이트나 모델 선택에 사용하지 않았다. '
        '기존 저장 추론의 camera-space 관절을 재사용해 정렬 지표만 재계산했다.','',
        f'전체 chunk 완료·GT 프레임 대응을 확인했고, per-frame 지표 {checks:,}건을 기존 구현과 대조했다. '
        'First-frame PA는 각 사람-chunk에서 SciPy 회전 정렬로 독립 검산했다.','',
        f'원시 집계: `{chosen.relative_to(ROOT)}/chunk_alignment_metrics.json`, '
        f'chunk별 결과: `{chosen.relative_to(ROOT)}/chunk_alignment_per_segment.csv`.','',
        'Train 219세션·68명, Test 64세션·20명은 피험자가 겹치지 않는다. '
        'Train 내부 187세션·58명은 가중치 학습, 32세션·10명은 모델 선택에 사용했다.','',
        '학습: `scripts/train_multihmr.py`. Test 추론: `scripts/evaluate_multihmr.py`. '
        '정렬 평가: `src/utils/chunk_alignment.py`.']
    report=ROOT/'_agents/report/multihmr_finetuning_14h.md';report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

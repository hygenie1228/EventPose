from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Validate every frozen clip and report paired camera-space metric errors."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.humanpose_metrics import joint_errors,self_check
from utils.segment_inference import ROOT,WORK,ARCHIVE


def main():
    self_check()
    from verify_humanpose_segments import verify
    verify(require_all=True)
    raw=(ROOT/'dataset/split.json').read_bytes()
    manifest=json.loads((ARCHIVE/'sampling_manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['split_sha256']
    split=json.loads(raw);before=json.loads((ARCHIVE/'split_before_segments.json').read_text())
    assert split['train']==before['train'] and set(split['test'])==set(before['test'])
    assert len(split['test'])==64 and not set(split['test'])&set(split['train'])
    tasks=json.loads((WORK/'tasks.json').read_text());assert len(tasks)==320
    assert {(t['parent_session'],t['segment']) for t in tasks}=={(s,i) for s in split['test'] for i in range(5)}
    methods=['human3r','multi_hmr'];totals={m:{22:[],55:[]} for m in methods}
    matched={m:0 for m in methods};valid_total=0;paired=0;rows=[];pa_checks=0
    for task in tasks:
        s,e=split['test'][task['parent_session']][task['segment']]
        assert (s,e)==(task['start_frame'],task['end_frame'])
        assert task['device']=='device0' and task['video_frames']==e-s
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
            assert np.array_equal(z['original_source_rgb_positions'],positions+s)
            assert (positions>=0).all() and (positions<e-s).all() and (np.diff(positions)>0).all()
        preds={};masks={}
        for m in methods:
            folder=WORK/m/task['session']
            completion=json.loads((folder/'complete.json').read_text())
            assert completion['task']['video_frames']==e-s
            assert not completion['task'].get('smoke')
            progress=json.loads((folder/'progress.json').read_text())
            assert progress['status']=='complete' and progress['rgb_position']==e-s-1
            with np.load(folder/'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                pred=z['matched_joints_cam_m'];index=z['matched_pred_indices']
                assert pred.shape==gt.shape
                mask=np.isfinite(pred).all((2,3))&valid
                assert np.array_equal(mask,(index>=0)&valid)
                preds[m]=pred;masks[m]=mask;matched[m]+=int(mask.sum())
        common=masks[methods[0]]&masks[methods[1]]
        ids=np.argwhere(common);valid_total+=int(valid.sum());paired+=len(ids)
        for m in methods:
            row=dict(session=task['parent_session'],segment=task['segment'],method=m,start_frame=s,end_frame=e,
                     valid_gt_instances=int(valid.sum()),matched_instances=int(masks[m].sum()),paired_instances=len(ids))
            for count in [22,55]:
                pred=preds[m][common][:,:count].astype(np.float64);target=gt[common][:,:count].astype(np.float64)
                errors=np.array([joint_errors(p,t) for p,t in zip(pred,target)]).reshape(-1,3)
                assert np.isfinite(errors).all()
                if len(errors):
                    # Independent vectorized checks of absolute and root-relative distances.
                    assert np.allclose(errors[:,0],np.linalg.norm(pred-target,axis=-1).mean(-1),atol=1e-10)
                    p=pred-pred[:,0:1];t=target-target[:,0:1]
                    assert np.allclose(errors[:,1],np.linalg.norm(p-t,axis=-1).mean(-1),atol=1e-10)
                    for i in {0,len(errors)-1}:
                        x=p[i]-p[i].mean(0);y=t[i]-t[i].mean(0)
                        rot,_=Rotation.align_vectors(y,x)
                        rx=rot.apply(x);scale=np.sum(rx*y)/np.sum(x*x)
                        independent=np.linalg.norm(scale*rx-y,axis=-1).mean()
                        assert np.isclose(independent,errors[i,2],atol=1e-8)
                        pa_checks+=1
                totals[m][count].append(errors)
                mean=errors.mean(0) if len(errors) else np.full(3,np.nan)
                for metric,v in zip(['mpjpe','root_aligned_mpjpe','root_aligned_pa_mpjpe'],mean):row[f'{metric}_{count}_m']=float(v)
            rows.append(row)
    assert paired>0
    summary=dict(status='complete',sessions=64,segments=320,frames=sum(t['video_frames'] for t in tasks),
                 unit='m',valid_gt_instances=valid_total,paired_instances=paired,matched_instances=matched,
                 independent_pa_checks=pa_checks,metrics={})
    for m in methods:
        summary['metrics'][m]={}
        for count in [22,55]:
            values=np.concatenate(totals[m][count]);assert len(values)==paired
            summary['metrics'][m][str(count)]=dict(zip(['mpjpe_m','root_aligned_mpjpe_m','root_aligned_pa_mpjpe_m'],map(float,values.mean(0))))
    for folder in [WORK,ARCHIVE]:
        (folder/'metrics_summary.json').write_text(json.dumps(summary,indent=2))
        with (folder/'per_segment_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Human3R / Multi-HMR — Test 고정 랜덤 구간 GT 평가','',
           'Test 64개 세션·device0, 영상마다 랜덤 10초 × 5개 구간. `dataset/split.json`에 저장된 0-based 원본 RGB frame number `[start,end)`를 그대로 사용했다. '
           f'총 {summary["frames"]:,}개 RGB 프레임을 원본 약 30fps 그대로 처리했다. Human3R state는 구간마다 초기화했다.','',
           '네트워크 AMP FP16, Multi-HMR 배치 32. GPU 0–4로 Multi-HMR 전체 완료 후 Human3R을 실행했다. 학습 없이 공개 pretrained 모델을 사용했다.','',
           'GT SMPL-X mesh와 예측 mesh에 같은 55×10475 joint regressor를 적용했다. 아래 주 표는 pelvis를 포함한 body 22관절, 표시 단위 mm이며 작을수록 좋다. 내부 좌표와 원시 _m 지표는 m로 보존하고 표시할 때 1000을 곱한다.','']
    for count in [22,55]:
        lines += [f'## {count}관절','', '| Method | MPJPE (mm) | Root-aligned MPJPE (mm) | Root-aligned PA-MPJPE (mm) |','|---|---:|---:|---:|']
        for m in methods:
            v=summary['metrics'][m][str(count)]
            lines.append(f'| {"Human3R" if m=="human3r" else "Multi-HMR"} | {1000*v["mpjpe_m"]:.2f} | {1000*v["root_aligned_mpjpe_m"]:.2f} | {1000*v["root_aligned_pa_mpjpe_m"]:.2f} |')
        lines.append('')
        if count==22:
            best=min(methods,key=lambda m:summary['metrics'][m]['22']['mpjpe_m'])
            label='Human3R' if best=='human3r' else 'Multi-HMR'
            lines += [f'이 평가의 metric-scale 위치 정확도(MPJPE)를 우선하면 **{label}을 추천**한다. 카메라 입력 정보 조건은 아래에 명시했다.','']
    lines += [f'유효 GT 사람-프레임 {valid_total:,}개 중 두 방법 모두 매칭된 {paired:,}개를 동일하게 사용한 instance-weighted 평균이다. 미검출은 0 오차로 넣지 않았다.','',
              '| Method | 매칭된 GT 사람-프레임 | 유효 GT 대비 |','|---|---:|---:|']
    for m in methods:lines.append(f'| {m} | {matched[m]:,} | {100*matched[m]/valid_total:.3f}% |')
    lines += ['', '정렬은 모든 프레임의 각 사람에 대해 독립적으로 적용한다. MPJPE는 원본 camera 좌표에서 정렬 없이 계산한다. Root-aligned는 예측과 GT 각각에서 해당 프레임의 pelvis(index 0)를 뺀다. PA는 이 pelvis centering 후 프레임·사람마다 proper Sim(3) Procrustes(rotation, positive scale, translation)를 따로 최적화하고 reflection은 허용하지 않는다. 시퀀스 전체에 하나의 변환을 공유하지 않는다.','',
              'GT는 neutral SMPL-X, 16 betas, non-PCA hands, flat_hand_mean=False, zero expression으로 복원하고 calibration으로 camera 좌표에 옮겼다. GT 대응은 label의 source_rgb_position을 정확히 사용하며 보간하지 않았다.','',
              '사람 매칭은 영상 내·카메라 앞 GT body 관절의 2D reprojection 거리로 Hungarian assignment(영상 대각선 25% gate)를 사용했다. Human3R은 자체 focal 추정, Multi-HMR은 보정된 GT camera intrinsic을 입력하므로 카메라 정보 조건은 다르다. GT pose/depth는 추론 입력이 아니다.','',
              f'검증: 모든 320구간×2방법 완료 및 전체 GT row 처리 확인, 독립 MPJPE/root MPJPE 검산, SciPy 기반 PA {pa_checks}건 교차 검산.','',
              '원시 결과와 sampling manifest: `exp/261023/humanpose_gt_segments/`. 전체 영상 FP32 실험은 중단해 별도로 보존했으며 이 결과에 합치지 않았다.','']
    report=ROOT/'_agents/report/human3r_multihmr_gt_evaluation.md';report.write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

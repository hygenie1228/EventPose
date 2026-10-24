from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Summarize only complete, paired, full-Test inference results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from utils.humanpose_metrics import joint_errors,self_check

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'exp/261023/humanpose_gt_eval'
REPORT=ROOT/'_agents/report/human3r_multihmr_gt_evaluation.md'
METHODS=['human3r','multi_hmr']


def summarize(smoke=False):
    tasks=json.loads((OUT/'tasks.json').read_text())
    base=OUT/'smoke' if smoke else OUT
    if smoke:tasks=[t for t in tasks if t['session']=='20260720_161542_272630_205_b']
    missing=[f'{m}/{t["session"]}' for t in tasks for m in METHODS
             if not (base/m/t['session']/'complete.json').exists()]
    assert not missing,{'incomplete_jobs':len(missing),'first':missing[:8]}
    totals={m:{22:[],55:[]} for m in METHODS};per_session=[];coverage={m:0 for m in METHODS}
    gt_total=0;paired_total=0
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
        if smoke:
            selected=positions<20;gt=gt[selected];valid=valid[selected];positions=positions[selected]
        predictions={};present={}
        for method in METHODS:
            with np.load(base/method/task['session']/'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                pred=z['matched_joints_cam_m'];assert pred.shape==gt.shape
                predictions[method]=pred;present[method]=np.isfinite(pred).all((2,3))&valid
                assert np.array_equal(present[method],(z['matched_pred_indices']>=0)&valid)
                coverage[method]+=int(present[method].sum())
        common=present['human3r']&present['multi_hmr']
        ids=np.argwhere(common);gt_total+=int(valid.sum());paired_total+=len(ids)
        for method in METHODS:
            pred=predictions[method]
            row=dict(session=task['session'],method=method,valid_gt_instances=int(valid.sum()),
                     method_matched_instances=int(present[method].sum()),paired_instances=len(ids))
            for count in [22,55]:
                errors=np.array([joint_errors(pred[i,j,:count],gt[i,j,:count]) for i,j in ids]).reshape(-1,3)
                assert errors.ndim==2 and errors.shape[1]==3 and np.isfinite(errors).all()
                totals[method][count].append(errors)
                mean=errors.mean(0) if len(errors) else np.full(3,np.nan)
                for metric,value in zip(['mpjpe','root_aligned_mpjpe','root_aligned_pa_mpjpe'],mean):
                    row[f'{metric}_{count}_m']=float(value)
            per_session.append(row)
    summary=dict(scope='SMOKE ONLY' if smoke else 'all snapshot-available test sessions, complete videos, device0',
                 sessions=len(tasks),valid_gt_instances=gt_total,paired_instances=paired_total,
                 matched_instances=coverage,metric_self_checks=self_check(),metrics={})
    for method in METHODS:
        summary['metrics'][method]={}
        for count in [22,55]:
            all_errors=np.concatenate(totals[method][count],axis=0)
            assert len(all_errors)==paired_total
            summary['metrics'][method][str(count)]={name:float(value) for name,value in zip(
                ['mpjpe_m','root_aligned_mpjpe_m','root_aligned_pa_mpjpe_m'],all_errors.mean(0))}
    (base/'metrics_summary.json').write_text(json.dumps(summary,indent=2))
    with (base/'per_session_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_session[0]));writer.writeheader();writer.writerows(per_session)
    from verify_humanpose_evaluation import verify
    verify(smoke)
    if not smoke:write_report(summary,per_session)
    print(json.dumps(summary,indent=2),flush=True)


def write_report(summary,rows):
    inventory=json.loads((OUT/'session_inventory.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    def table(count):
        lines=['| Method | MPJPE (m) ↓ | Root-aligned MPJPE (m) ↓ | Root-aligned PA-MPJPE (m) ↓ |',
               '|---|---:|---:|---:|']
        for method in METHODS:
            m=summary['metrics'][method][str(count)]
            lines.append(f'| {"Human3R" if method=="human3r" else "Multi-HMR"} | {m["mpjpe_m"]:.6f} | {m["root_aligned_mpjpe_m"]:.6f} | {m["root_aligned_pa_mpjpe_m"]:.6f} |')
        return '\n'.join(lines)
    lines=['# Human3R / Multi-HMR — available Test 전체 영상 GT 평가',
        '', '평가일: 2026-10-23 시작. 모든 결과 단위는 **m**이며 낮을수록 좋다.',
        '', '## 결과: SMPL-X body 22관절', '',table(22),'',
        f'실제 존재하는 Test {summary["sessions"]}개 세션의 device0 전체 RGB 영상을 사용했다. '
        f'GT 대응 프레임은 {protocol["gt_frames"]:,}개이며 시간 샘플링을 하지 않았다. '
        f'유효 GT 사람-프레임 {summary["valid_gt_instances"]:,}개 중 두 모델 모두 대응된 '
        f'**{summary["paired_instances"]:,}개**를 동일하게 사용해 instance-weighted 평균을 계산했다.',
        '', '| Method | GT와 대응된 사람-프레임 | 유효 GT 대비 비율 |', '|---|---:|---:|']
    for method in METHODS:
        n=summary['matched_instances'][method]
        lines.append(f'| {method} | {n:,} | {100*n/summary["valid_gt_instances"]:.3f}% |')
    lines += ['', '미검출·매칭 실패를 0 오차로 넣지 않았다. 따라서 표는 검출 성공 조건부 pose 정확도이며, 전체 검출 성능을 대신하지 않는다.',
        '', '## 평가 규약', '',
        '- 범위: `dataset/split.json`의 Test와 시작 시점에 실제 존재하는 세션의 교집합. Train은 처리하지 않았다.',
        '- 카메라: 세션마다 canonical `device0` 한 개. 다른 카메라를 합산해 같은 타임라인을 중복 가중하지 않았다.',
        '- 전체 영상: RGB 원본의 모든 프레임을 순서대로 추론했다. Human3R state도 영상 전체에서 이어진다. GT가 없는 시작/끝 프레임은 추론하되 metric에서는 제외했다.',
        '- 시간 대응: `frames/source_rgb_position`과 원본 MP4 frame index를 정확히 연결. nearest-frame GT 보간을 하지 않았다.',
        '- GT: 저장된 165차원 pose·16차원 betas·world translation으로 neutral SMPL-X mesh를 재구성하고, calibration으로 camera space에 옮겼다.',
        '- 관절: GT와 두 예측 mesh에 동일한 SMPL-X `J_regressor`를 적용. 주 표는 pelvis를 포함한 관절 0..21, 추가 표는 0..54다.',
        '- 사람 대응: GT의 카메라 앞·영상 안으로 투영되는 body 관절과 예측 관절의 2D reprojection 거리로 Hungarian matching. Occlusion visibility annotation을 사용한 것은 아니다. 3D 오차를 최소화하는 방식으로 사람 ID를 선택하지 않았다. 0.25 image-diagonal gate를 적용했다.',
        '- MPJPE: 카메라 좌표의 절대 xyz를 그대로 비교. root·회전·스케일을 맞추지 않는다.',
        '- Root-aligned MPJPE: 각 skeleton의 pelvis를 빼고 비교. 회전·스케일은 유지한다.',
        '- Root-aligned PA-MPJPE: pelvis 정렬 후 conventional Procrustes Sim(3), 즉 회전·양의 단일 스케일·이동을 최적화. Reflection은 허용하지 않는다. 이 값은 절대 metric-scale 정확도를 입증하지 않는다.',
        '- 모델: 사전학습 checkpoint 그대로 FP32 추론. Fine-tuning 없음. Human3R은 native scene-derived camera/depth와 TTT3R, Multi-HMR은 제공된 calibration intrinsics를 사용한다. 두 방법의 camera 정보 조건이 같다는 주장은 하지 않는다.',
        '', '## 추가 결과: SMPL-X 전체 55관절', '',table(55),'',
        '## 세션별 body 22관절 결과', '',
        '| Session | Method | Paired instances | MPJPE (m) | Root-aligned MPJPE (m) | Root-aligned PA-MPJPE (m) |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        values=[r[k] for k in ['mpjpe_22_m','root_aligned_mpjpe_22_m','root_aligned_pa_mpjpe_22_m']]
        cells=' | '.join(f'{x:.6f}' if np.isfinite(x) else '—' for x in values)
        lines.append(f'| {r["session"]} | {r["method"]} | {r["paired_instances"]:,} | {cells} |')
    lines += ['', '## 포함/제외 및 재현', '',
        (f'- Split의 Test 총 {inventory["test_total"]}개 중 없는 {len(inventory["missing_sessions"])}개 세션은 추론·평가하지 않았다. 목록은 inventory에 보존했다.'
         if inventory['missing_sessions'] else f'- Split의 Test {inventory["test_total"]}개 세션의 device0을 모두 평가했다. 제외된 Test 세션은 없다.'),
        '- GT의 `person_valid` 및 pose/shape/translation의 finite 조건을 적용했다. 영상·GT 읽기 실패를 조용히 누락하지 않고 준비/실행 실패로 처리했다.',
        '- GT는 MAMMA-derived label이므로 독립 계측 GT에 대한 정확도와 동일시하지 않는다. 얼굴/상체 close-up 세션도 제외하지 않아 보이지 않는 신체 부위 추정이 포함된다.',
        '- 원본 데이터셋에는 쓰기 작업을 하지 않았다.',
        '- 검증: 모든 세션의 RGB 끝 프레임·GT row 처리 및 Test membership을 확인했다. MPJPE와 root-aligned MPJPE 전체 평균을 독립적으로 재계산하고, PA는 별도 SciPy rotation alignment 구현과 표본 대조했다. `evaluation_verification.json` 참조.',
        '', '실험 파일: `exp/261023/humanpose_gt_eval/`의 `protocol.json`, `session_inventory.json`, `tasks.json`, `metrics_summary.json`, `per_session_metrics.csv`, 모델별 세션 `result.npz`와 `complete.json`.',
        '', '구현: `tools/evaluate_humanpose_gt.py`, `tools/humanpose_metrics.py`, `tools/report_humanpose_gt.py`.',
        '', '재계산: `.venv-humanpose/bin/python tools/report_humanpose_gt.py`.', '']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    summarize(a.smoke)

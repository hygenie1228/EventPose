from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Audit the frozen final Test and compare pretrained versus validation-selected FT."""
import csv
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from multihmr.data import ROOT,OUT,WORK,save
from scripts.evaluate_multihmr import TEST,METHOD,RESULT_OUT,SNAPSHOT
from utils.humanpose_metrics import joint_errors,self_check


def main():
    self_check();manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['test_split_sha256']
    selected=json.loads((RESULT_OUT/'selected.json').read_text())
    assert selected['test_used_for_selection'] is False
    actual_config=json.loads((OUT/'runs'/selected['config']['name']/'configuration.json').read_text())
    selected['config']=actual_config
    save(RESULT_OUT/'selected.json',selected)
    # Keep the exported full-weight metadata consistent with explicit runtime overrides.
    exported=torch.load(selected['checkpoint'],map_location='cpu')
    if exported['fine_tuning']['config']!=actual_config:
        exported['fine_tuning']['config']=actual_config
        from pathlib import Path
        path=Path(selected['checkpoint']);temporary=path.with_suffix('.metadata.tmp.pt')
        torch.save(exported,temporary);temporary.replace(path)
    del exported
    assert not set(manifest['training_subject_ids'])&set(manifest['validation_subject_ids'])
    assert not (set(manifest['training_subject_ids'])|set(manifest['validation_subject_ids']))&set(manifest['test_subject_ids'])
    tasks=json.loads((TEST/'tasks.json').read_text());assert len(tasks)==320
    assert len({t['parent_session'] for t in tasks})==64
    baseline=ROOT/'.experiment_storage/humanpose_gt_segments/multi_hmr'
    sums={m:{n:np.zeros(3) for n in [22,55]} for m in ['pretrained','finetuned']}
    coverage={m:0 for m in sums};valid_count=0;paired_count=0;rows=[];checks=0
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
            assert np.array_equal(z['original_source_rgb_positions'],positions+task['start_frame'])
        predictions={};present={}
        for method,folder in [('pretrained',baseline/task['session']),('finetuned',TEST/METHOD/task['session'])]:
            completion=json.loads((folder/'complete.json').read_text())
            assert completion['task']['video_frames']==task['end_frame']-task['start_frame']
            assert not completion['task'].get('smoke')
            with np.load(folder/'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                pred=z['matched_joints_cam_m'];assert pred.shape==gt.shape
                mask=np.isfinite(pred).all((2,3))&valid
                assert np.array_equal(mask,(z['matched_pred_indices']>=0)&valid)
                predictions[method]=pred;present[method]=mask;coverage[method]+=int(mask.sum())
            if method=='finetuned':
                archive=RESULT_OUT/'final_test_results'/task['session']/'result.npz'
                assert hashlib.sha256(archive.read_bytes()).digest()==hashlib.sha256((folder/'result.npz').read_bytes()).digest()
        common=present['pretrained']&present['finetuned'];count=int(common.sum())
        valid_count+=int(valid.sum());paired_count+=count
        for method in sums:
            row=dict(session=task['parent_session'],segment=task['segment'],method=method,
                     valid_instances=int(valid.sum()),matched_instances=int(present[method].sum()),paired_instances=count)
            for n in [22,55]:
                p=predictions[method][common][:,:n].astype(np.float64);t=gt[common][:,:n].astype(np.float64)
                errors=np.array([joint_errors(a,b) for a,b in zip(p,t)]).reshape(-1,3)
                assert np.isfinite(errors).all()
                if len(errors):
                    assert np.allclose(errors[:,0],np.linalg.norm(p-t,axis=-1).mean(-1))
                    x=p-p[:,0:1];y=t-t[:,0:1]
                    assert np.allclose(errors[:,1],np.linalg.norm(x-y,axis=-1).mean(-1))
                    for i in {0,len(errors)-1}:
                        xc=x[i]-x[i].mean(0);yc=y[i]-y[i].mean(0)
                        rot,_=Rotation.align_vectors(yc,xc);rx=rot.apply(xc)
                        scale=np.sum(rx*yc)/np.sum(xc*xc)
                        assert np.isclose(np.linalg.norm(scale*rx-yc,axis=-1).mean(),errors[i,2],atol=1e-8)
                        checks+=1
                sums[method][n]+=errors.sum(0)
                for metric,value in zip(['mpjpe','root_mpjpe','pa_mpjpe'],errors.mean(0) if count else [np.nan]*3):
                    row[f'{metric}_{n}_mm']=float(1000*value)
            rows.append(row)
    assert paired_count>0
    summary=dict(status='complete',test_sessions=64,segments=320,rgb_frames=sum(t['video_frames'] for t in tasks),
                 valid_instances=valid_count,paired_instances=paired_count,matched_instances=coverage,
                 unit='mm',independent_pa_checks=checks,selected_checkpoint=selected['checkpoint'],metrics={})
    for method in sums:
        summary['metrics'][method]={str(n):dict(zip(['mpjpe_mm','root_mpjpe_mm','pa_mpjpe_mm'],
                              map(float,1000*sums[method][n]/paired_count))) for n in [22,55]}
    save(RESULT_OUT/'final_test_metrics.json',summary)
    with (RESULT_OUT/'final_test_per_segment.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if SNAPSHOT:
        lines=['# Multi-HMR 중간 Test 평가','',f'체크포인트: {selected["checkpoint"]}',
               '', 'Test 64세션·device0·320구간 전체 평가. 모델 선택은 Validation 기준을 유지한다.',
               f'동일 매칭 모집단: {paired_count:,} 사람-프레임. 단위 mm.',
               '', '| 모델 | MPJPE | Root-aligned MPJPE | PA-MPJPE |', '|---|---:|---:|---:|']
        for name in ['pretrained','finetuned']:
            m=summary['metrics'][name]['22']
            lines.append(f'| {name} | {m["mpjpe_mm"]:.2f} | {m["root_mpjpe_mm"]:.2f} | {m["pa_mpjpe_mm"]:.2f} |')
        lines+=['',f'독립 PA 검산 {checks:,}건 통과. 최종 학습 종료 결과와 구분한 중간 Test 평가다.']
        (ROOT/'_agents/report/multihmr_interim_test.md').write_text('\n'.join(lines)+'\n')
        return
    experiments=[]
    for wave in ['wave1.json','wave2.json']:
        for config in json.loads((OUT/wave).read_text()):
            folder=OUT/'runs'/config['name']
            config=json.loads((folder/'configuration.json').read_text())
            complete=json.loads((folder/'complete.json').read_text());best=json.loads((folder/'best.json').read_text())
            assert 4*3600<=complete['elapsed_seconds']<=8*3600+15*60
            assert complete['successful_steps']>0
            experiments.append(dict(name=config['name'],last_blocks=config['last_blocks'],
                 hours=complete['elapsed_seconds']/3600,steps=complete['step'],best_pa_mm=best['pa_mpjpe_mm'],
                 coverage=best['coverage'],flip_probability=config.get('horizontal_flip_probability',0)))
    save(OUT/'experiment_summary.json',experiments)
    lines=['# Multi-HMR Train fine-tuning — 14시간 실험','',
        '메인 지표는 프레임·사람별 Root-aligned PA-MPJPE, 단위 mm이며 낮을수록 좋다. '
        '예측·GT 각각 pelvis centering 후 proper Sim(3) 정렬을 적용한다. Reflection은 허용하지 않는다.','',
        '데이터 구분은 Train 219세션·68명, Test 64세션·20명이다. '
        f'Train 내부에서 {len(manifest["training_sessions"])}세션·{len(manifest["training_subject_ids"])}명은 가중치 업데이트, '
        f'{len(manifest["validation_sessions"])}세션·{len(manifest["validation_subject_ids"])}명은 모델 선택에 사용했다. '
        '기존 Test 20명과 모두 분리됐고 데이터셋 파일은 수정하지 않았다. 체크포인트·후속 설정 선택에는 Test를 사용하지 않았다.','',
        '학습 캐시는 Train 전체 시간 구간에서 약 2fps로 뽑은 116,146프레임, validation은 고정 2,048프레임이다. '
        'Train GT SMPL-X 16 betas·non-PCA hands·flat_hand_mean=False·zero expression으로 mesh를 복원한 뒤 동일한 SMPL-X joint regressor를 적용했다. '
        'Camera calibration을 적용했고 head 중심 GT로 학습 위치를 지정했다. 검증과 Test에서는 GT 위치를 제공하지 않고 실제 검출을 사용했다.','',
        'GPU 0·1 / 2·3 / 4·5, 실험당 2-GPU DDP·GPU당 배치 16을 사용했다. AMP와 backbone gradient checkpointing을 적용했다. '
        'Loss는 body22 PA 오차를 중심으로 root-relative 관절, 검출, offset, pose, shape, 절대 pelvis 보조 항을 사용했다. '
        'PA loss의 정렬 변환은 SVD로 풀어 detach하고, 정렬된 예측 관절을 통해 gradient를 전달했다.','',
        '증강은 기본 ColorJitter와 설정별 수평 반전을 사용했다. 수평 반전은 camera X·좌우 관절·proper rotation·intrinsic을 함께 변환한다. '
        '반전 투영 관계, 두 번 반전 시 복원, PA 지표 불변성을 검증했다. Validation과 Test에는 증강을 적용하지 않았다. '
        '표의 설정은 실제 프로세스에 적용된 configuration.json을 기준으로 하며 training_overrides.json을 반영한다.','',
        '## 실험과 validation','',
        '| 실험 | 학습 backbone blocks | 수평 반전 확률 | 실행 시간 (h) | step | Best validation PA-MPJPE (mm) | 검출·매칭률 |','|---|---:|---:|---:|---:|---:|---:|']
    for e in experiments:lines.append(f'| {e["name"]} | {e["last_blocks"]} | {e["flip_probability"]:.2f} | {e["hours"]:.2f} | {e["steps"]:,} | {e["best_pa_mm"]:.2f} | {100*e["coverage"]:.2f}% |')
    lines += ['',f'사전학습 validation PA-MPJPE: **{selected["baseline_validation"]["pa_mpjpe_mm"]:.2f} mm**. '
               f'선택 실험: **{selected["config"]["name"]}**, validation **{selected["validation"]["pa_mpjpe_mm"]:.2f} mm**. '
               '검출·매칭률이 사전학습 대비 0.5 percentage point 이상 낮아지는 후보는 best 선택에서 제외했다.','',
               '## 최종 Test','',
               f'고정 Test 64세션·device0, 영상당 10초 × 5구간, 총 {summary["rgb_frames"]:,} RGB 프레임을 원본 약 30fps로 평가했다. '
               f'두 모델 모두 매칭된 동일 GT **{paired_count:,} 사람-프레임** 기준이다. 비교 모집단이 이전 Human3R 비교와 달라 사전학습 점수도 다시 계산했다.','']
    for n in [22,55]:
        lines += [f'### {n}관절','', '| 모델 | MPJPE (mm) | Root-aligned MPJPE (mm) | Root-aligned PA-MPJPE (mm) |','|---|---:|---:|---:|']
        for method in sums:
            v=summary['metrics'][method][str(n)]
            lines.append(f'| {method} | {v["mpjpe_mm"]:.2f} | {v["root_mpjpe_mm"]:.2f} | {v["pa_mpjpe_mm"]:.2f} |')
        lines.append('')
    for method in coverage:lines.append(f'- {method}: GT 검출·매칭 {coverage[method]:,}/{valid_count:,} ({100*coverage[method]/valid_count:.2f}%).')
    lines += ['', '미검출에 0 오차를 넣지 않았다. 두 모델 모두 같은 calibrated camera intrinsic과 같은 RGB 전처리를 사용했다.',
        '',f'검증: 전체 320구간×2모델 완료·GT frame mapping·저장 보관본 hash 확인, 독립 MPJPE/root MPJPE 검산, SciPy PA 교차 검산 {checks}건 통과.','',
        f'최종 체크포인트: `{selected["checkpoint"]}`. 공식 pretrained args/model_state_dict 형식의 full weights이며 SMPL-X 자산은 기존 data/human_model_files를 사용한다.','',
        '실험 설정·validation 로그·원시 결과: `exp/261023/multihmr_finetune_14h/`. 재현 코드: `tools/prepare_multihmr_finetune.py`, '
        '`tools/train_multihmr_etr.py`, `tools/orchestrate_multihmr_finetune.py`, `tools/evaluate_multihmr_finetuned.py`.','']
    from utils.training_plots import main as plot_progress
    plot_progress()
    # User-facing performance reporting is Test-only; keep selection diagnostics in exp/.
    begin=lines.index('## 실험과 validation')
    end=lines.index('## 최종 Test')
    lines[begin:end]=['## 학습 실험','',
        '| 실험 | Backbone blocks | 수평 반전 확률 | 실행 시간 (h) | step |',
        '|---|---:|---:|---:|---:|']+[
        f'| {e["name"]} | {e["last_blocks"]} | {e["flip_probability"]:.2f} | {e["hours"]:.2f} | {e["steps"]:,} |'
        for e in experiments]+['', '모델 선택은 학습 내부 Validation으로 수행했다. 성능 비교는 아래 Test 결과를 사용한다.','']
    (ROOT/'_agents/report/multihmr_finetuning_14h.md').write_text('\n'.join(lines))
    from utils.chunk_alignment import main as report_chunk_alignment
    report_chunk_alignment()
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

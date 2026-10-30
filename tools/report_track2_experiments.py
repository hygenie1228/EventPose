"""Update exactly one Korean report and one research Markdown in place."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import fcntl
import json
import subprocess
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json,sha256

REPORT=ROOT/'_agents/report/261025_트랙2_MultiHMR_미세조정_진행.md'
RESEARCH=ROOT/'_agents/research/261025/track2_multihmr_progress.md'


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def utc(t):
    return datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial.md');temporary.write_text(text);temporary.replace(path)


def _update():
    manifest=read(OUTPUT/'manifest.json',{})
    budget=read(OUTPUT/'budget.json',{})
    supervisor=read(OUTPUT/'supervisor.json',{})
    receipts=[read(p,{}) for p in (OUTPUT/'work/receipts').glob('*.json')]
    label_checks=read(OUTPUT/'label_quality_checks.json',{})
    asset_checks=read(OUTPUT/'model_assets.json',{})
    prepared=len(receipts)
    selection=read(OUTPUT/'selected.json',{})
    selected_prediction_review=read(OUTPUT/'target_foreground_visuals'/selection.get('trial','original')/'receipt.json',{}).get('visual_review',{})
    selected_geometry=read(OUTPUT/'selected_geometry_diagnostics'/(selection.get('trial','original')+'.json'),{})
    full_mask_audit=read(OUTPUT/'full_frame_target_mask_audit/audit.json',{})
    mhr_validation=read(OUTPUT/'validation_mhr_comparison.json',{})
    cd_selection=read(OUTPUT/'internal_cd_h_selection.json',{})
    init_comparison=read(OUTPUT/'init_comparison.json',{})
    quick=read(OUTPUT/'quick_comparison.json',{})
    quick_state=read(OUTPUT/'quick_test_status.json',{})
    priority_state=read(OUTPUT/'priority_resume_status.json',{})
    wave2_plan=read(OUTPUT/'cd_wave2_plan.json',{})
    wave2_state=read(OUTPUT/'cd_wave2_status.json',{})
    guardian_state=read(OUTPUT/'target_training_guardian_status.json',{})
    if guardian_state and guardian_state.get('status')!='target_followups_started':wave2_state=guardian_state
    target_completed=read(OUTPUT/'target_completed_test_models.json',{})
    target_eval_state=read(OUTPUT/'target_eval_status.json',{})
    target_geometry=read(OUTPUT/'target_geometry_diagnostic.json',{})
    followup_order=read(OUTPUT/'target_followup_order_revision.json',{})
    timing_profile=read(OUTPUT/'target_pipeline_timing_profile.json',{})
    foreground_visual_review=read(OUTPUT/'target_foreground_visuals/receipt.json',{}).get('visual_review',{})
    exported_visual_review=read(OUTPUT/'target_foreground_visuals/head_strong_exported/receipt.json',{}).get('visual_review',{})
    train_export_review=read(OUTPUT/'target_foreground_visuals/head_strong_exported/all/visual_review_progress.json',{})
    current_proxy_baseline=read(OUTPUT/'validation_smplx/original/metrics.json',{})
    wave3_plan=read(OUTPUT/'paired_wave3_plan.json',{})
    wave3_state=read(OUTPUT/'paired_wave3_status.json',{})
    paired_pretraining=read(OUTPUT/'paired_foreground_search_status.json',{})
    if paired_pretraining and not wave3_state:wave3_state=paired_pretraining
    pooled_state=read(OUTPUT/'target_contenders_status.json',{})
    refit_prefetch_state=read(OUTPUT/'target_refit_prefetch_status.json',{})
    export_backfill=read(OUTPUT/'export_backfill_status.json',{})
    evaluation_backfill=read(OUTPUT/'evaluation_backfill_status.json',{})
    if pooled_state:wave2_state=pooled_state;wave3_state=pooled_state
    pair_pilot=read(OUTPUT/'paired_cd_pilot_status.json',{})
    current_pilot=read(OUTPUT/'runs'/wave3_plan.get('pilot_name','pair_cd_pilot')/'complete.json',{})
    if current_pilot:
        pair_pilot=dict(current_pilot,attempted_steps=current_pilot['step'],skipped_steps=current_pilot['step']-current_pilot['successful_steps'])
    conversion_diagnostic=read(OUTPUT/'conversion_metric_diagnostic.json',{})
    v2_pilot=read(OUTPUT/'conversion_v2_pilot_status.json',{})
    v2_prefetch=read(OUTPUT/'conversion_v2_prefetch_status.json',{})
    v2_final=read(OUTPUT/'conversion_v2_final_status.json',{})
    v2_audit=read(OUTPUT/'conversion_v2_pilot_independent_audit.json',{})
    v2_train_score=read(OUTPUT/'conversion_v2_train_score_smoke.json',{})
    outlier_diagnostic=read(OUTPUT/'test_outlier_diagnostic.json',{})
    test_visual_review=read(OUTPUT/'test_visuals/receipt.json',{}).get('visual_review',{})
    refinement_trials={name:read(OUTPUT/(name+'.json')) for name in ['conversion_refinement_diagnostic','conversion_lbfgs_diagnostic','conversion_lm_diagnostic','conversion_lmfast_diagnostic']}
    refinement_trials={name:value for name,value in refinement_trials.items() if value}
    followup_trials=wave2_plan.get('trials',[])+wave3_plan.get('trials',[])
    comparison=read(OUTPUT/'comparison.json',{})
    evaluation_validity=read(OUTPUT/'evaluation_validity.json',{})
    target_recheck=evaluation_validity.get('re_evaluation_required',False)
    if target_recheck:
        comparison={};mhr_validation={};quick={}
    if not mhr_validation:
        ready_validation=read(OUTPUT/'target_train_validation_completed.json',{})
        if ready_validation.get('evaluation_protocol')=='target_foreground_head_v1' and ready_validation.get('comparison_complete'):
            mhr_validation=ready_validation
    exports=read(OUTPUT/'export_receipts.json',{})
    available_exports=read(OUTPUT/'available_export_receipts.json',{})
    priority_exports=read(OUTPUT/'selected_priority_export_receipts.json',{})
    saved_exports=exports if exports.get('samples') else available_exports
    error=dict(training=read(OUTPUT/'supervisor_error.json',{}),finalization=read(OUTPUT/'finalizer_error.json',{}),
        cd_wave2=read(OUTPUT/'cd_wave2_error.json',{}),init_comparison=read(OUTPUT/'init_comparison_error.json',{}),
        quick_test=read(OUTPUT/'quick_test_error.json',{}),priority_resume=read(OUTPUT/'priority_resume_error.json',{}))
    error.update(conversion_v2_prefetch=read(OUTPUT/'conversion_v2_prefetch_error.json',{}),conversion_v2_final=read(OUTPUT/'conversion_v2_final_error.json',{}))
    error.update(target_evaluation=read(OUTPUT/'target_eval_error.json',{}),
        target_followup_guardian=read(OUTPUT/'target_followup_guardian_error.json',{}),
        foreground_training=read(OUTPUT/'paired_foreground_search_error.json',{}),
        paired_target_phase=read(OUTPUT/'paired_wave3_error.json',{}),
        target_contenders=read(OUTPUT/'target_contenders_error.json',{}),
        target_refit_prefetch=read(OUTPUT/'target_refit_prefetch_error.json',{}))
    for prefix in ['native_training','native_refits','native_finalization','native_baseline_Test',
        'fixed_training','fixed_finalization','native_test_prefetch','exact_depth_order_Train_control',
        'root_smoothing_Train','native_surface_target_preparation','native_surface_pilot','native_surface_training','native_surface_finalization','fixed_baseline_backfill','fixed_proxy_backfill','fixed_validation_backfill','fixed_validation_backfill_head','fixed_validation_backfill_last2']:
        error[prefix]=read(OUTPUT/(prefix+'_error.json'),{})
    error={k:v for k,v in error.items() if v}
    conversion_reports=[read(p,{}) for p in (OUTPUT/'predictions').glob('*/*/conversion.json')]
    conversion_means=[x['mean_vertex_fit_mm'] for x in conversion_reports if 'mean_vertex_fit_mm' in x]
    healthy_state=target_eval_state if evaluation_validity else read(OUTPUT/'healthy_eval_status.json',{})
    hardware=read(OUTPUT/'hardware_incident.json',{})
    hardware_pilot=read(OUTPUT/'hardware_pilot_status.json',{})
    now=utc(time.time())
    lines=[f'# 트랙2 Multi-HMR 미세조정 진행',f'\n갱신: {now}',
        '\n이 파일과 연구 기록 파일을 각각 하나씩 유지하며 덮어써서 갱신한다.',
        f'\n실험 폴더: `{OUTPUT.relative_to(ROOT)}/`',
        f'\n실험 시간: {utc(budget["started_unix"])} ~ {utc(budget["deadline_unix"])} (총 {budget.get("total_hours",12)}시간 한도).',
        '\n## 현재 상태',
        f'\n실행기 상태: `{supervisor.get("status","초기화 중")}`. Train MHR→SMPL-X 변환 {prepared}/20개 완료.',
        '\nGPU 0은 보존한다. GPU 3은 Xid79 / fallen off the bus로 제외했으며, 사용자 승인에 따라 정상 물리 GPU 1·2·4·5·6에서 계속한다. CUDA UUID로 장치를 지정한다. 기존 namhj 학습·실행기·worker는 종료했다. '
        'GPU 독점 모드 설정은 관리자 권한 부족으로 실패했으므로 다른 계정 실행을 시스템 차원에서 차단했다고 주장하지 않는다.',
        f'\n정상 5 GPU 평가 실행기 상태: `{healthy_state.get("status","확인 중")}`. '
        '사용자의 빠른 평가 요청에 따라 저장된 optimizer checkpoint를 보존하고 refit을 중단해 '
        '즉시 평가를 수행했다. 이후 세 설정 모두 전체 Train 20 refit을 완료했으며, 저장된 refit checkpoint 평가와 30개 export를 진행한다.',
        '\n## 분할 및 검증',
        '\n시퀀스 30개를 seed 261025로 Train 20 / Test 10으로 고정했다. Train 내부에서 '
        '먼저 16개를 학습, 4개를 시퀀스 단위 검증에 사용한다. 이후 검증으로 고정한 설정과 update 예산으로 '
        '20개 전체를 다시 학습한다. 저장된 best/last 후보의 선택은 SMPL-X 기준 CD-H로 하고, '
        '설정 선택은 이 4개의 연속 전체 프레임 MHR CD-H를 우선하며 ACC-H를 동률 판정에 사용한다. '
        '전체 Train refit에서는 검증으로 체크포인트를 고르지 않는다. Test 점수로 체크포인트나 설정을 선택하지 않는다. '
        '동일 인물 여부 메타데이터가 없어 인물 단위 분리라고 주장하지 않는다.',
        '\n## Trial',
        '\n| Trial | GPU | 학습 범위 | Head LR | Backbone LR | Weight decay | L2-SP 가중치 | 검증 PA-MPJPE (mm) | 검출 매칭 비율 | 성공 update |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for name,pair,blocks,anchor in [('head_strong','1,2',0,'2e5'),('last2_strong','3,4',2,'1e5'),('last4_strong','5,6',4,'5e4')]:
        best=read(OUTPUT/'runs'/name/'best.json',{})
        progress=read(OUTPUT/'runs'/name/'progress.json',{})
        complete=read(OUTPUT/'runs'/name/'complete.json',{})
        pa=best.get('pa_mpjpe_mm');cov=best.get('coverage')
        lines.append(f'| {name} | {pair} | 마지막 {blocks} blocks + heads | 3e-6 | {"0" if blocks==0 else "2e-7"} | 0.1 | {anchor} | {pa:.2f} | {cov:.4f} | {complete.get("successful_steps",progress.get("successful_steps","대기"))} |' if pa is not None else
            f'| {name} | {pair} | 마지막 {blocks} blocks + heads | 3e-6 | {"0" if blocks==0 else "2e-7"} | 0.1 | {anchor} | 미측정 | 미측정 | {progress.get("successful_steps","대기")} |')
    lines.extend(['\n초기 Trial은 2 GPU DDP, GPU당 batch 8, 최대 2시간이다. 전체 Train refit은 최대 1시간을 추가하고 '
        '초기 학습과 refit의 합계가 Trial당 3시간 이내가 되도록 제한한다. 위 검증 PA-MPJPE는 refit 전 내부 검증에서 측정한 값이다. '
        '최대 2,500 update 시도 또는 20 epoch, 검증 4회 연속 미개선 시 조기 종료한다. '
        '작은 데이터에 대해 ColorJitter·낮은 학습률·weight decay·원본 가중치 L2-SP를 적용한다.',
        '\n## 내부 검증 SMPL-X CD-H',
        '\n초기 실행은 PA-MPJPE로 best와 마지막 last를 보존했다. 사용자 요청에 따라 저장된 두 후보를 '
        'SMPL-X 기준 CD-H로 재평가해 선택한다. 원본 frame0 고정 Sim(3)·공식 topology map·동일 FPS64 body 표본을 사용한다. '
        'GT는 fitting한 SMPL-X mesh이므로 아래 값은 최종 MHR CD-H와 구분한다.'])
    if evaluation_validity:
        current_initial_proxy=all(cd_selection.get('trials',{}).get(n,{}).get('metrics',{}).get('detection_selection')=='target_mask_head_highest_confidence' for n in ['head_strong','last2_strong','last4_strong'])
        if current_initial_proxy:
            lines.append('\n아래 초기 세 Trial의 저장된 best/last를 현재의 target foreground/head 최고 confidence 기준으로 재평가했다. GT로 검출을 선택하지 않으며 원본과 같은 Train validation4×64 프레임을 사용한다. 최종 MHR 내부 검증·Test 점수와는 mesh 공간과 프레임 표본이 다르다.')
        else:
            lines.append('\n아래 초기 세 Trial의 proxy 값은 이전 fullRGB·GT 매칭 방식의 기록이다. 현재 target foreground/head 방식으로 best/last를 재평가 중이며 완료 후 교체한다. 이전 값과 현재 target 방식 수치를 직접 비교하지 않는다.')
    if current_proxy_baseline.get('detection_selection')=='target_mask_head_highest_confidence':
        lines.append(f'\n현재 target 방식 원본 SMPL-X proxy: {current_proxy_baseline["cd_h_smplx_proxy_cm"]:.4f}cm, Train 내부 검증 {current_proxy_baseline["matched_instances"]}/{current_proxy_baseline["valid_instances"]}개 검출. GT 재구성 오차는 {current_proxy_baseline["max_gt_reconstruction_error_m"]*1000:.4f}mm. Test 결과가 아니다.')
    lines.extend(['\n| Trial | 선택 후보 | Step | SMPL-X CD-H (cm) | 매칭 비율 |','|---|---|---|---|---|'])
    proxy_set=[current_proxy_baseline]+[cd_selection.get('trials',{}).get(n,{}).get('metrics',{}) for n in ['head_strong','last2_strong','last4_strong']]
    proxy_complete=all(v.get('detection_selection')=='target_mask_head_highest_confidence' for v in proxy_set)
    proxy_min=min((v['cd_h_smplx_proxy_cm'] for v in proxy_set),default=None) if proxy_complete else None
    if current_proxy_baseline.get('detection_selection')=='target_mask_head_highest_confidence':
        lines.append(f'| original | 원본 | 0 | {current_proxy_baseline["cd_h_smplx_proxy_cm"]:.4f} | {current_proxy_baseline["coverage"]:.4f} |')
    for name in ['head_strong','last2_strong','last4_strong']:
        selected=cd_selection.get('trials',{}).get(name)
        if selected:
            v=selected['metrics'];cd_text=f'{v["cd_h_smplx_proxy_cm"]:.4f}'
            if proxy_complete and v['cd_h_smplx_proxy_cm']==proxy_min:cd_text='**'+cd_text+'**'
            lines.append(f'| {name} | {selected["selected_candidate"]} | {v["step"]} | {cd_text} | {v["coverage"]:.4f} |')
        else:lines.append(f'| {name} | 평가 대기 | — | 미측정 | — |')
    if followup_order.get('execution_mode')=='pooled_contenders_with_Train20_refit_prefetch':
        lines.append(f'\n후속6개 후보를 같은 target/head 기준으로 비교한다. Train-only SMPL-X 검증으로 고정한 update 예산의 전체Train20 refit을 현재 평가와 병행한다. 이후 동일4개 Train MHR CD-H/ACC-H로 초기 모델과6개 후보를 함께 선택하고, 고정 Test10 비교와 최종30개 저장을 한 단계로 묶는다. 원래12시간·후보당3시간·2-GPU pair lease는 유지한다. Refitting 상태: `{refit_prefetch_state.get("status","준비")}`. `target_pooled_followup_plan.json`.')
    elif followup_order:
        lines.append('\n후속 실행 순서는 target foreground로 학습한 paired CD-H 세 후보를 먼저 검증·전체Train refit·Test·export하고, 이후 예산이 남으면 기존 absolute-loss 세 후보를 처리한다. 원래12시간 마감과 후보당3시간 한도는 유지한다. 새 단계는 전체 검증·refit·Test·export 예산이3.5시간 이상 남았을 때 진입하며, 부족하면 이전에 검증·저장한30개 결과를 유지하고 추가 단계 미실행을 기록한다. `target_followup_order_revision.json`.')
    if wave2_plan:
        lines.extend(['\n## CD-H 중심 후속 병렬 실험',
            '\n초기 내부 CD-H 결과에서 발견한 절대 위치·궤적 오차를 줄이기 위한 후속 후보다. '
            '설정은 미세조정 모델의 Test 추론 전에 고정했다. 기존 평가·export와 병행해 정상 두 GPU 쌍(1·2 / 4·5)에서 최대 두 후보를 동시에 학습하고 세 번째는 쌍이 비면 시작한다. '
            '학습 도중 10분 간격으로 CD-H를 측정해 checkpoint를 선택한다.',
            f'\n후속 상태: `{wave2_state.get("status","설정 고정")}`.',
            '\n아래 후속 학습 중 proxy는 기존 fullRGB/GT-matched 방식이다. 최종 후보 선택 전 target foreground/head 조건으로 best/last를 재평가하므로 위의 새 target 방식 proxy와 직접 비교하지 않는다.',
            '\n| Trial | GPU | Blocks | 절대 pelvis loss 가중치 | 방향 보존 body loss 가중치 | 내부 SMPL-X CD-H (cm) | 성공 update |',
            '|---|---|---|---|---|---|---|'])
        for index,cfg in enumerate(wave2_plan['trials']):
            name=cfg['name'];best=read(OUTPUT/'runs'/name/'best.json',{});progress=read(OUTPUT/'runs'/name/'progress.json',{})
            complete=read(OUTPUT/'runs'/name/'complete.json',{})
            metric=best.get('cd_h_smplx_proxy_cm');cd=f'{metric:.4f}' if metric is not None else '미측정'
            lines.append(f'| {name} | 두 정상 GPU 쌍 대기열 | {cfg["last_blocks"]} | {cfg["absolute_weight"]} | {cfg["root_weight"]} | {cd} | {complete.get("successful_steps",progress.get("successful_steps","대기"))} |')
    lines.extend(['\n## 내부 검증 MHR 결과',
        '\n원본과 초기 세 Trial을 Train 내부 검증 4개 전체에서 평가한 뒤, '
        'MHR CD-H가 낮은 모델을 선택하고 동률일 때 ACC-H를 사용한다. '
        '새 target/head 입력 조건에서는 대상 직접 검출률95% 이상을 후보 기준으로 사용한다. 이전 fullRGB/GT-matching의 원본 대비0.005 기준과 구분한다. 최종 MHR 점수는 미검출 보완을 포함한 전체 프레임을 사용한다. 원본이 더 좋으면 원본을 선택하고 미세조정 개선을 주장하지 않는다.'])
    if mhr_validation:
        lines.extend(['\n| 모델 | CD-H (cm) | ACC-H (cm/frame²) |',
            '|---|---|---|'])
        minimum_cd=min(v['CD_H_cm'] for v in mhr_validation['models'].values())
        minimum_acc=min(v['ACC_H_cm_per_frame2'] for v in mhr_validation['models'].values())
        for name,v in mhr_validation['models'].items():
            cd=f'{v["CD_H_cm"]:.4f}';acc=f'{v["ACC_H_cm_per_frame2"]:.4f}'
            if v['CD_H_cm']==minimum_cd:cd='**'+cd+'**'
            if v['ACC_H_cm_per_frame2']==minimum_acc:acc='**'+acc+'**'
            lines.append(f'| {name} | {cd} | {acc} |')
        lines.append('\n현재 Train 내부 CD-H 최솟값 모델: `'+mhr_validation.get('selected_trial','미선택')+'`. Test 점수는 이 선택에 사용하지 않는다. CD-H와 ACC-H의 개선 여부는 각각 해석한다.')
    else:
        lines.extend(['\n현재 target 조건으로 내부 검증을 진행 중이다. 모델별 4개 영상 전체가 완료되는 즉시 평균을 표시한다. 모델 선택은 모든 후보의 동일 검증이 끝난 후 수행한다.',
            '\n| 모델 | 완료 검증 시퀀스 | MHR CD-H cm | ACC-H cm/frame² |','|---|---|---|---|'])
        aliases=read(OUTPUT/'evaluation_checkpoints.json',{})
        for name in ['original','head_strong','last2_strong','last4_strong']:
            alias='original' if name=='original' else 'search_'+name;measured=[]
            for sample in manifest.get('validation',[]):
                folder=OUTPUT/'predictions'/alias/sample
                metric=read(folder/'metrics.json',{});info=read(folder/'inference.json',{})
                if metric.get('evaluation_protocol')!='target_foreground_head_v1' or metric.get('conversion_revision')!=2:continue
                if not info.get('target_foreground_only_input') or not info.get('all_accepted_heads_in_target_mask'):continue
                if metric.get('gt_sha256')!=manifest['samples'][sample]['gt_sha256']:continue
                if metric.get('frames')!=manifest['samples'][sample]['frames'] or not metric.get('full_frame_roster'):continue
                if name!='original' and info.get('checkpoint_sha256')!=aliases.get(alias,{}).get('checkpoint_sha256'):continue
                if not (folder/'human_params_mhr_v2.npz').exists() or metric.get('prediction_sha256')!=sha256(folder/'human_params_mhr_v2.npz'):continue
                measured.append(metric)
            if len(measured)==4:
                cd=sum(v['CD_H_cm'] for v in measured)/4;acc=sum(v['ACC_H_cm_per_frame2'] for v in measured)/4
                lines.append(f'| {name} | 4/4 | {cd:.4f} | {acc:.4f} |')
            else:lines.append(f'| {name} | {len(measured)}/4 | 전체 평균 대기 | 전체 평균 대기 |')
    followup_validation_lines=[]
    if followup_trials and evaluation_validity:
        followup_validation_start=len(lines)
        lines.extend(['\n### 후속 후보 Train MHR 검증 진행',
            '\n앞의 SMPL-X proxy와 구분되는 native MHR v2 결과다. 모델별 동일4개 검증 영상의 모든 프레임이 완료돼야 평균을 표시한다. 후보 전체 검증 전에는 새 모델을 선택하지 않으며 기존 저장본을 유지한다.',
            '\n| 모델 | 완료 검증 영상 | MHR CD-H cm | ACC-H cm/frame² |','|---|---|---|---|'])
        validation_aliases=read(OUTPUT/'evaluation_checkpoints.json',{})
        for cfg in followup_trials:
            name=cfg['name'];alias='search_'+name;measured=[]
            expected=validation_aliases.get(alias,{}).get('checkpoint_sha256')
            for sample in manifest.get('validation',[]):
                folder=OUTPUT/'predictions'/alias/sample
                metric=read(folder/'metrics.json',{});info=read(folder/'inference.json',{})
                if metric.get('evaluation_protocol')!='target_foreground_head_v1' or metric.get('conversion_revision')!=2:continue
                if not info.get('target_foreground_only_input') or not info.get('all_accepted_heads_in_target_mask'):continue
                if not expected or info.get('checkpoint_sha256')!=expected:continue
                if metric.get('gt_sha256')!=manifest['samples'][sample]['gt_sha256']:continue
                if metric.get('frames')!=manifest['samples'][sample]['frames'] or not metric.get('full_frame_roster'):continue
                if not (folder/'human_params_mhr_v2.npz').exists() or metric.get('prediction_sha256')!=sha256(folder/'human_params_mhr_v2.npz'):continue
                measured.append(metric)
            if len(measured)==4:
                cd=sum(v['CD_H_cm'] for v in measured)/4;acc=sum(v['ACC_H_cm_per_frame2'] for v in measured)/4
                lines.append(f'| {name} | 4/4 | {cd:.4f} | {acc:.4f} |')
            else:lines.append(f'| {name} | {len(measured)}/4 | 전체 평균 대기 | 전체 평균 대기 |')
        followup_validation_lines=lines[followup_validation_start:]
    lines.extend(['\n## CD-H / ACC-H Test 비교',
        '\n아직 측정하지 않은 값은 결과로 채우지 않는다. 원본 모델과 세 Trial을 동일한 Test 10개, '
        '원본 연속 프레임 전체에서 비교한다. CD-H는 cm, ACC-H는 cm/frame²이다. '
        '시퀀스 최초 프레임에서 구한 하나의 Sim(3)을 전체에 고정하고 시퀀스별 동일 가중 평균을 낸다.'])
    # Publish each complete, fixed-Test model immediately; do not wait for
    # other models or Train exports to finish before reporting measured scores.
    completed_models={}
    checkpoint_map=read(OUTPUT/'evaluation_checkpoints.json',{})
    test_names=manifest.get('test',[])
    for model_name in ['original','head_strong','last2_strong','last4_strong']+[cfg['name'] for cfg in followup_trials]:
        values={sample:read(OUTPUT/'predictions'/model_name/sample/'metrics.json') for sample in test_names}
        if len(values)!=10 or not all(values.values()):continue
        valid=True
        for sample,value in values.items():
            info=read(OUTPUT/'predictions'/model_name/sample/'inference.json',{})
            valid=valid and value.get('evaluation_protocol')=='target_foreground_head_v1'
            valid=valid and info.get('target_foreground_only_input',False) and info.get('all_accepted_heads_in_target_mask',False)
            expected=checkpoint_map.get(model_name,{}).get('checkpoint_sha256')
            valid=valid and value.get('gt_sha256')==manifest['samples'][sample]['gt_sha256']
            valid=valid and value.get('frames')==manifest['samples'][sample]['frames'] and value.get('full_frame_roster',False)
            if model_name!='original':valid=valid and bool(expected) and info.get('checkpoint_sha256')==expected
        if not valid:continue
        completed_models[model_name]=dict(CD_H_cm=sum(x['CD_H_cm'] for x in values.values())/10,
            ACC_H_cm_per_frame2=sum(x['ACC_H_cm_per_frame2'] for x in values.values())/10,
            sequences=10,frames=sum(x['frames'] for x in values.values()),per_sequence=values,
            checkpoint=checkpoint_map.get(model_name,{}).get('checkpoint'),
            checkpoint_sha256=checkpoint_map.get(model_name,{}).get('checkpoint_sha256'),
            full_training_refit=checkpoint_map.get(model_name,{}).get('full_train_refit',False))
    save_json(OUTPUT/'completed_test_models.json',dict(models=completed_models,test_used_for_selection=False,
        aggregation='equal sequence weights',updated_unix=time.time()))
    model_values=dict(init_comparison.get('models',{}))
    model_values.update(quick.get('models',{}))
    model_values.update(completed_models)
    model_values.update(comparison.get('models',{}))
    if target_recheck:model_values={name:value for name,value in model_values.items() if name in ['init_v6','init_v0']}
    if target_completed.get('evaluation_protocol')=='target_foreground_head_v1':
        model_values.update(target_completed.get('models',{}))
    lines.extend(['\n| 모델 | 완료 Test 시퀀스 | CD-H (cm) | ACC-H (cm/frame²) |',
        '|---|---|---|---|'])
    minimum_cd=min((v['CD_H_cm'] for v in model_values.values()),default=None)
    minimum_acc=min((v['ACC_H_cm_per_frame2'] for v in model_values.values()),default=None)
    names=['init_v6','init_v0','original','head_strong','last2_strong','last4_strong']+[cfg['name'] for cfg in followup_trials]
    for name in names:
        if name in model_values:
            value=model_values[name];cd=f'{value["CD_H_cm"]:.4f}';acc=f'{value["ACC_H_cm_per_frame2"]:.4f}'
            if len(model_values)==len(names) and value['CD_H_cm']==minimum_cd:cd='**'+cd+'**'
            if len(model_values)==len(names) and value['ACC_H_cm_per_frame2']==minimum_acc:acc='**'+acc+'**'
            lines.append(f'| {name} | 10/10 | {cd} | {acc} |')
        else:
            measured=[read(OUTPUT/'predictions'/name/sample/'metrics.json') for sample in manifest.get('test',[])
                if (OUTPUT/'predictions'/name/sample/'metrics.json').exists()]
            if quick_state.get('trial')==name:
                measured=[read(OUTPUT/'predictions'/('quick_'+name)/sample/'metrics.json') for sample in manifest.get('test',[])
                    if (OUTPUT/'predictions'/('quick_'+name)/sample/'metrics.json').exists()]
            lines.append(f'| {name} | {len(measured)}/10 | 최종 미측정 | 최종 미측정 |')
    if completed_models:
        lines.append('\n전체 Train refit 모델은 고정 Test 10개 평가가 끝나는 즉시 반영한다. 다른 모델이나 Train export가 끝날 때까지 기다리지 않는다. checkpoint 해시와 시퀀스별 수치는 `completed_test_models.json`에 기록한다.')
    if comparison:lines.append('\n재계산 가능한 시퀀스별 수치 및 입력 해시: `exp/261025/track2_multihmr_12h/comparison.json`.')
    quick_only=set(quick.get('models',{}))-set(comparison.get('models',{}))-set(completed_models)
    if quick_only:
        lines.append('\n'+', '.join(sorted(quick_only))+f' 행은 사용자가 요청한 즉시 평가 결과다. '
            f'정확한 checkpoint step은 {quick.get("step")}, 학습 시퀀스는 {quick.get("training_sequences")}개이며 '
            '동일한 Test 10개 전체를 MHR로 평가했다. 원본 checkpoint 경로·해시·시퀀스별 결과는 `quick_comparison.json`에 보존한다. '
            '전체 Train refit 모델의 Test 측정이 끝나면 해당 모델 행은 그 측정치로 갱신한다.')
    elif quick_state.get('status')=='running':
        lines.append('\n저장된 checkpoint 즉시 평가가 GPU 1–6의 공용 작업 큐에서 진행 중이다. '
            '비어 있는 worker가 다음 Test 시퀀스를 바로 가져가며 전체 Train refit 완료를 기다리지 않는다.')
    lines.extend(['\n`init_v6`: `exp2/hj/challenge_track2_v6/*/human/human_params_init.npz`. '
        '`init_v0`: `exp2/hj/challenge_track2_v0/*/human/human_params_init.npz`. '
        '두 baseline 모두 동일한 v6 GT와 고정 Test 10개로 평가한다. `original`은 원본 Multi-HMR RGB 추론으로, 기존 init NPZ와 다른 비교 대상이다.',
        '\n## init 대비 개선량',
        '\n감소량 = init − 모델, 개선율 = (init − 모델)/init × 100. 음수는 성능 악화다. '
        '각 셀은 절대 감소량과 개선율을 함께 표시한다.',
        '\n| 모델 | v6 init 대비 CD-H 감소 cm (%) | v6 init 대비 ACC-H 감소 cm/frame² (%) | v0 init 대비 CD-H 감소 cm (%) | v0 init 대비 ACC-H 감소 cm/frame² (%) |',
        '|---|---|---|---|---|'])
    for name in names[2:]:
        cells=[]
        for baseline in ['init_v6','init_v0']:
            for metric in ['CD_H_cm','ACC_H_cm_per_frame2']:
                if name in model_values and baseline in model_values:
                    delta=model_values[baseline][metric]-model_values[name][metric]
                    percent=100*delta/max(model_values[baseline][metric],1e-12)
                    cells.append(f'{delta:+.4f} ({percent:+.2f}%)')
                else:cells.append('미측정')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    if init_comparison:
        identical=init_comparison.get('v0_v6_identical_parameters',{})
        if identical and all(identical.values()):lines.append('\n확인 결과 v0/v6 init의 pose·shape·scales가 Test 10개 모두 동일하다. 따라서 두 baseline 점수와 각각의 개선율도 동일해야 한다.')
        lines.append('\ninit source 해시·시퀀스별 점수: `init_input_inventory.json`, `init_comparison.json`.')
    lines.extend(['\n## 이전 배경 제거 조건의 출력 기록',
        '\n이전 조건의 Train 검증 선택 모델: '+selection.get('trial',mhr_validation.get('selected_trial','미선택'))+'.',
        '\n30개 시퀀스 각각의 `exp2/hj/challenge_track2_v6/<시퀀스>/human/human_params_uranus.npz`에 '
        'MHR `pose(T,136)`, `shape(45)`, `scales(68)`를 저장한다. GT 파일은 수정하지 않는다.',
        f'\n이전 조건에서 저장한 receipt: {len(saved_exports.get("samples",{}))}/30개. 모델: `{saved_exports.get("trial","아직 없음")}`. 현재 source 파일 수는 앞의 원본 RGB 진행표와 실제 SHA 일치 수로 확인한다.',
        '\n## 해석상 주의사항',
        '\n공개 참가자 배포판에는 공식 채점 서버의 MHR asset/vertex roles가 없다. '
        '`ChallengeHOR/scripts/run_evaluation.py`의 로컬 평가 경로와 동일한 '
        'MHR TorchScript·FPS27 alignment·FPS64 body·body22 acceleration 및 공식 수식을 사용한다. '
        '공식 리더보드와 완전 동일한 점수라고 주장하지 않는다.',
        '\n카메라 calibration은 이미 존재하는 GT-assisted 추정값이다. '
        'GT의 world→camera 변환을 학습 라벨에 한 번 적용하고, 추론 mesh는 카메라 좌표를 유지한 채 MHR로 변환한다. '
        '이는 기존 init/final NPZ 및 native renderer와 같은 좌표계다. GT world 좌표와의 고정 차이는 공식 최초 프레임 Sim(3)으로 처리한다. '
        '추론과 MHR fitting에는 GT3D pose/shape를 입력하지 않는다. 제공된 target human mask로 대상을 확인하며 배경 인물을 평가에 넣지 않는다.'])
    current_trial=selection.get('trial',mhr_validation.get('selected_trial'))
    if current_trial:
        inferred=converted=0;checkpoint_map_now=read(OUTPUT/'evaluation_checkpoints.json',{})
        for sample in manifest.get('samples',{}):
            folder=OUTPUT/'predictions'/current_trial/sample;info=read(folder/'inference.json',{})
            if info.get('evaluation_protocol')!='target_foreground_head_v1' or not info.get('full_original_frame_rate'):continue
            if info.get('frames')!=manifest['samples'][sample]['frames']:continue
            if current_trial!='original' and info.get('checkpoint_sha256')!=checkpoint_map_now.get(current_trial,{}).get('checkpoint_sha256'):continue
            inferred+=1
            report=read(folder/'conversion_v2.json',{})
            if report.get('evaluation_protocol')=='target_foreground_head_v1' and (folder/'human_params_mhr_v2.npz').exists():converted+=1
        lines.append(f'\n현재 선택 모델 `{current_trial}`의 전체30개 생성 진행: RGB 추론 {inferred}/30, MHR revision2 변환 {converted}/30. 이 수치는 처리 완료된 중간 산출물 수이며 위의 source NPZ 저장·검증 수와 구분한다.')
    if error:lines.extend(['\n## 실행 오류','\n```json\n'+json.dumps(error,ensure_ascii=False,indent=2)+'\n```'])
    if label_checks:
        lines.extend(['\n## 라벨 변환 검증',
            f'\nTrain 20개의 모든 표본에서 head 중심이 영상 안에 있고 회전행렬·수치가 정상이다. '
            f'프레임별 평균 표면 fitting 오차의 최댓값은 {label_checks["max_fit_frame_error_mm"]:.2f} mm다. '
            '이 값은 네트워크 예측 성능이 아닌 MHR→SMPL-X 변환 잔차다.'])
    lines.extend(['\n## 하드웨어 복구 및 시각 점검', '\nGPU3 PCI 61:00.0 장애 후 CUDA 번호 변경으로 실패한 실행기를 종료하고 정상 5개 UUID로 평가를 재개했다. 하드웨어 복구에는 관리자 조치가 필요하다. `hardware_incident.json`, `gpu_resources.json`에 기록했다.', '\n내부 검증 4개 시퀀스의 최초·중간·마지막 총 12프레임 overlay를 확인했다. 확인한 표본에서는 사람 전환이나 좌표축 오류가 보이지 않았으나, 시간에 따라 달라지는 깊이 편향이 남았다. 전체 프레임 무오류를 의미하지 않는다. `validation_visuals/receipt.json` 및 이미지에 보존했다.'])
    lines.append(f'\n정상 GPU의 Gloo 통신 시험은 두 GPU 쌍 모두 통과했다. 실제 Multi-HMR 2-GPU/CD-H pilot 상태: `{hardware_pilot.get("status","대기")}`. 첫 pilot에서 검증 축소 표본이 일부 시퀀스 frame0을 빠뜨린 문제를 수정했으며, 32개 표본 안에 4개 frame0이 보존됨을 CPU에서 확인했다.')
    if conversion_means:
        lines.append(f'\n현재 완료된 변환 {len(conversion_means)}건의 전체 mesh 평균 잔차 범위는 {min(conversion_means):.2f}–{max(conversion_means):.2f} mm다. 일부 프레임은 100mm를 넘으므로, 변환을 무손실로 해석하지 않는다. 이 잔차는 GT 예측 오차와 다르며 시퀀스별 `conversion.json`에 보존한다.')
    integrity=read(OUTPUT/'refit_checkpoint_integrity.json',{})
    if integrity:
        lines.append('\n전체 Train refit 체크포인트 3개를 CPU에서 직접 로드했다. 각 1,380회 성공 update, 가중치 유한값, search 대비 가중치 변경, Train20/Test10 분리, 원본 GT 30개 SHA256 보존을 확인했다. 증거: `refit_checkpoint_integrity.json`.')
    followup_integrity=read(OUTPUT/'target_followup_refit_integrity.json',{})
    if followup_integrity:
        values=list(followup_integrity['trials'].values())
        hours=[v['bootstrap_search_refit_total_seconds']/3600 for v in values]
        lines.append(f'\n후속 전체Train20 재학습 체크포인트 {len(values)}개도 CPU에서 직접 로드했다. 모든 저장 가중치가 유한하고 초기 가중치에서 실제로 변경됐으며, 검증으로 refit checkpoint를 선택하지 않았음을 확인했다. 각 search와 refit의 RGB/foreground 입력이 일치하고 bootstrap+search+refit 합계는 {min(hours):.3f}–{max(hours):.3f}시간으로 모두3시간 미만이다. Train20/Test10 분리와 원본 GT30개 SHA도 확인했다. `target_followup_refit_integrity.json`.')
    if wave3_plan:
        lines.extend(['\n## 최초 프레임 쌍 CD-H 탐색',
            '\n독립 프레임 PA loss가 직접 다루지 않는 시간별 상대 궤적 오차를 줄이기 위해, 같은 Train 시퀀스의 현재 프레임과 최초 프레임을 함께 학습한다. FPS27 최초 프레임 Sim(3)을 고정한 뒤 FPS64 body의 양방향 거리 평균 합을 loss로 쓴다. GT는 학습 supervision에만 쓴다.',
            '\n초기 Test 집계 결과를 이미 본 뒤 추가한 탐색이다. Test 관측 전 사전 고정된 실험으로 주장하지 않는다. 모델과 checkpoint는 동일한 Train 내부 검증 CD-H로만 선택한다.',
            f'\n후속 상태: `{wave3_state.get("status","대기")}`. 최초 12시간 마감과 후보당3시간 한도를 유지한다. 검증된 foreground 입력으로 현재 평가와 초기 학습을 병행하고, refit·비교·export는 이전 wave 종료 후 순차 처리한다.',
            f'\n실제 DDP pilot: {pair_pilot.get("successful_steps","확인 중")}회 성공 / {pair_pilot.get("attempted_steps","확인 중")}회 시도, AMP overflow {pair_pilot.get("skipped_steps","확인 중")}회. 현재 pilot은 대상 foreground 입력과 정렬 전체 기울기를 사용하며 본 후보는 초기 scale128을 쓴다.',
            '\n| Trial | Blocks | Paired CD-H loss 가중치 | 내부 SMPL-X CD-H cm | 성공 update |', '|---|---|---|---|---|'])
        for cfg in wave3_plan['trials']:
            name=cfg['name'];best=read(OUTPUT/'runs'/name/'best.json',{});progress=read(OUTPUT/'runs'/name/'progress.json',{});complete=read(OUTPUT/'runs'/name/'complete.json',{})
            metric=best.get('cd_h_smplx_proxy_cm');value=f'{metric:.4f}' if metric is not None else '미측정'
            lines.append(f'| {name} | {cfg["last_blocks"]} | {cfg["paired_cd_weight"]} | {value} | {complete.get("successful_steps",progress.get("successful_steps","대기"))} |')
    if conversion_diagnostic and not evaluation_validity:
        lines.extend(['\n## 변환만 비교한 body 진단',
            '\nGT를 읽지 않고 SMPL-X 예측 자체와 그 MHR 변환을 비교했다. Train 검증 4개에서 균일 32프레임 및 변환 잔차가 큰 5프레임을 표본으로 삼았으므로 전체 프레임 평균이 아니다. 최초 프레임 정렬 후 양방향 body Chamfer이며, 사람 예측 CD-H 점수나 최종 점수에서 뺄 수 있는 오차 항목이 아니다.',
            '\n| 모델 | 표본의 변환 전후 body Chamfer cm |', '|---|---|'])
        for name,value in conversion_diagnostic.get('models',{}).items():
            lines.append(f'| {name} | {value["conversion_body_chamfer_frame0_aligned_cm"]:.4f} |')
        lines.append('\n상세 표본과 원본 해시: `conversion_metric_diagnostic.json`. 변환 잔차가 큰 구간에 대한 추가 pose fitting은 Train-only 별도 진단 중이며 기존 평가 결과를 조용히 바꾸지 않는다.')
    if refinement_trials and not evaluation_validity:
        lines.extend(['\n## Train 표본 변환 수렴 시험', '\n동일한 black_platform 검증 영상의 13개 표본으로 한정한 GT-free 실험이다. 전체 시퀀스나 Test 개선으로 주장하지 않으며, 운영 중인 변환은 아직 변경하지 않았다.', '\n| 추가 최적화 | 이전 평균 vertex 오차 mm | 이후 평균 vertex 오차 mm | 소요 초 |', '|---|---|---|---|'])
        for value in refinement_trials.values():
            before=sum(value['baseline_fit_mm'])/len(value['frames']);after=sum(value['refined_fit_mm'])/len(value['frames'])
            lines.append(f'| {value.get("optimizer","Adam600")} | {before:.3f} | {after:.3f} | {value["elapsed_seconds"]:.1f} |')
    if v2_pilot and not evaluation_validity:
        receipt=v2_pilot.get('receipt',{})
        lines.extend(['\n## 공통 MHR 변환 버전2',
            '\n기존 변환에서 프레임별 평균 residual >20mm인 경우에만 동일 identity에서 reference LM10을 추가한다. 전체 mesh MSE가 줄어드는 pose만 채택하며 GT/시간 smoothing을 사용하지 않는다. 버전1 NPZ·점수는 보존한다.',
            f'\n전체 Train 검증 영상 pilot: {receipt.get("total_frames","진행 중")}프레임, 평균 mesh 잔차 {receipt.get("baseline_mean_vertex_fit_mm","미측정")} → {receipt.get("mean_vertex_fit_mm","미측정")} mm. 소요 {receipt.get("elapsed_seconds","미측정")}초.',
            f'\n재평가 상태: `{v2_prefetch.get("status","대기")}` / 최종 집계·export: `{v2_final.get("status","대기")}`. 학습 중 GPU6 및 남는 정상 GPU에서 평가를 병행하고, 학습 종료 후 정상5개로 마무리한다.',
            f'\n현재 성능표의 공통 변환 버전: **{comparison.get("conversion_revision",1)}**. 버전2에서 모든 모델의 같은 Test10 비교와 Train-only 재선택, export30개 검증이 끝나기 전에는 표를 섞어 갱신하지 않는다.',
            '\n독립 CPU 검증: '+('700프레임 모두 MSE 비증가, 미선택 pose·공유 shape/scales 동일, 저장 잔차와 재복원 잔차 일치 확인.' if v2_audit else '실행 중 또는 아직 미완료.')])
    if v2_train_score and not evaluation_validity:
        a=v2_train_score['v1'];b=v2_train_score['v2']
        lines.append(f'\n동일한 Train pilot 700프레임의 공식식 점검: CD-H {a["CD_H_cm"]:.4f} → {b["CD_H_cm"]:.4f} cm, ACC-H {a["ACC_H_cm_per_frame2"]:.4f} → {b["ACC_H_cm_per_frame2"]:.4f} cm/frame². 변환 잔차 감소가 ACC-H 개선을 보장하지 않으며 전체 Test 결과로 일반화하지 않는다.')
    concurrent=read(OUTPUT/'concurrent_scheduling_change.json',{})
    if refit_prefetch_state:
        lines.extend(['\n## 후속 후보 공동 검증 및 전체Train20 refit',
            '\n동일 target foreground/head 최고 confidence 조건의 posthoc SMPL-X proxy다. 검출된 표본만 채점하며 검출률을 함께 표시한다. recipe 최종 선택은 네 검증 영상 전체 프레임 MHR CD-H/ACC-H로 한다. 각 refit의 학습 입력은 자기 search와 동일하게 유지한다(absolute: fullRGB, paired: foreground). 평가 입력은 모두 동일한 foreground 조건이다.',
            '\n| 후보 | posthoc 선택 | SMPL-X CD-H cm | 대상 검출률 | 전체Train20 성공 update |', '|---|---|---|---|---|'])
        for cfg in followup_trials:
            name=cfg['name'];choice=read(OUTPUT/'refit_prefetch_choices'/(name+'.json'),{});value=choice.get('metrics',{})
            completed=read(OUTPUT/'runs'/('refit_'+name)/'complete.json',{});progress=read(OUTPUT/'runs'/('refit_'+name)/'progress.json',{})
            cd=f'{value["cd_h_smplx_proxy_cm"]:.4f}' if value else '검증 대기';coverage=f'{100*value["coverage"]:.2f}%' if value else '—'
            lines.append(f'| {name} | {choice.get("selected_candidate","대기")} | {cd} | {coverage} | {completed.get("successful_steps",progress.get("successful_steps","GPU 쌍 대기"))} |')
    if concurrent:
        lines.append('\n사용자 추가 지시로 학습과 평가를 병행하도록 변경했다. 기존 첫 평가 프로세스는 유지하고 후속 두 2-GPU Trial을 즉시 시작했다. 다음 wave의 target foreground 학습도 실제 pilot 검증 후 현재 평가와 겹치며, pair lease로 학습 GPU 쌍의 중복 배정을 막는다. GPU0 보존. 변경 기록: `concurrent_scheduling_change.json`.')
    if outlier_diagnostic and not evaluation_validity:
        lines.extend(['\n## 초기 네 모델의 Test 분포 진단',
            '\n아래는 공통 변환 v1으로 완료한 Test10의 보조 진단이다. 공식 집계는 계속 시퀀스별 동일 가중 평균이며, 중앙값이나 outlier 제거 점수로 대체하지 않는다. 모델 선택에도 사용하지 않는다.',
            '\n| 모델 | 평균 CD-H cm | 시퀀스 중앙값 cm | 최대 시퀀스 CD-H cm | 최악2개가 전체 CD 합에서 차지하는 비율 |', '|---|---|---|---|---|'])
        for name,value in outlier_diagnostic['models'].items():
            lines.append(f'| {name} | {value["mean_CD_H_cm"]:.4f} | {value["median_sequence_CD_H_cm"]:.4f} | {value["max_sequence_CD_H_cm"]:.4f} | {value["top2_share_of_CD_sum_percent"]:.2f}% |')
        lines.append('\n원본은 blue_easter_basket·toy_stroller 두 영상의 큰 오차에 평균이 민감하다. 미세조정 모델은 중앙값도 낮아졌지만, 원본 대비 평균 개선율만으로 일반화 성능을 해석하지 않는다. 원시 시퀀스별 값: `test_outlier_diagnostic.json`.')
    if test_visual_review:
        lines.append('\n무효화한 이전 unmasked 결과의 사후 영상 점검: 3개 영상의 최초·중간·마지막 총9프레임을 확인했다. 원본 Multi-HMR는 blue_easter_basket의 중간·마지막, toy_stroller의 세 표본에서 배우 대신 배경의 앉아 있는 사람을 선택했다. Last2는 확인한9프레임에서 대상 배우를 유지했다. 따라서 원본 대비 CD-H 개선에는 대상 선택 안정화가 포함되며 순수 pose 정확도 개선으로만 해석할 수 없다. 전체 프레임 오류율로 일반화하지 않으며 이전 설정의 진단 기록이며, 이후 사용자 지시에 따라 native 화면을 유지한 target mask 선택으로 교체했다. 이미지 및 기록: `test_visuals/receipt.json`.')
    if available_exports and not exports.get('samples'):
        lines.append(f'\n완료된 Test {len(available_exports.get("samples",{}))}개의 native MHR NPZ를 각 `human/human_params_uranus.npz` 경로에 먼저 저장했다. 모델은 `{available_exports.get("trial")}`의 전체Train20 refit이며 baseball_bat도 포함한다. Train-only 내부 MHR CD-H로 선택했고 배경 인물 선택을 차단한 입력·revision2 변환·전체 프레임 채점을 확인했다. 전체30개 및 후속 최종 모델 저장을 이어간다. 개별 해시·checkpoint: `available_export_receipts.json`.')
    if export_backfill:
        lines.append(f'\nGPU6 추가 export 작업 상태: `{export_backfill["status"]}`. 기존 GPU6 할당4개 영상의 변환 완료와 장치의 실행 프로세스 없음·사용률0%를 확인한 뒤 남은 Train 영상 추론·변환을 추가 배정했다. 기존 실행기는 유지하며 동일 checkpoint·대상 선택·변환·stage lock/cache를 사용한다. 중복 실행 요청은 같은 stage lock에서 직렬화한 뒤 완료 cache를 재사용한다. `export_backfill_status.json`, `export_backfill_jobs.json`.')
    front=['\n## init_v6 / init_v0 기준 비교 — Test10 전체',
        '\nCD-H: cm, ACC-H: cm/frame². 개선율 양수는 오차 감소, 음수는 악화다. 원본/미세조정 모델은 target human mask로 대상을 고정해 재평가하며, 이전 배경 인물 혼입 점수는 유효 비교에서 제외한다.',
        '\n| 모델 | 완료 Test | CD-H ↓ | ACC-H ↓ | v6 대비 CD-H 개선율 | v0 대비 CD-H 개선율 | original 대비 CD-H 개선율 |', '|---|---|---|---|---|---|---|']
    for name in names[:6]+[n for n in names[6:] if n in model_values]:
        value=model_values.get(name)
        done_count=10 if value else sum(1 for sample in manifest.get('test',[]) if read(OUTPUT/'predictions'/name/sample/'metrics.json',{}).get('evaluation_protocol')=='target_foreground_head_v1')
        if value:
            v6=model_values.get('init_v6',{}).get('CD_H_cm');v0=model_values.get('init_v0',{}).get('CD_H_cm')
            d6=f'{100*(v6-value["CD_H_cm"])/v6:+.2f}%' if v6 else '—'
            d0=f'{100*(v0-value["CD_H_cm"])/v0:+.2f}%' if v0 else '—'
            original_cd=model_values.get('original',{}).get('CD_H_cm')
            delta_original=f'{100*(original_cd-value["CD_H_cm"])/original_cd:+.2f}%' if original_cd else '—'
            cd=f'{value["CD_H_cm"]:.4f}';acc=f'{value["ACC_H_cm_per_frame2"]:.4f}'
            if value['CD_H_cm']==minimum_cd:cd='**'+cd+'**'
            if value['ACC_H_cm_per_frame2']==minimum_acc:acc='**'+acc+'**'
            front.append(f'| {name} | 10/10 | {cd} | {acc} | {d6} | {d0} | {delta_original} |')
        else:front.append(f'| {name} | {done_count}/10 | target 재평가 중 | target 재평가 중 | 미측정 | 미측정 | 미측정 |')
    front.append('\n굵은 값은 Test10 전체 평가가 완료된 모델 중 해당 지표의 최솟값이다. 각 모델은 동일 Test10의 총7,610프레임을 평가하고 시퀀스별 동일 가중치로 평균한다. 후속 후보의 Train SMPL-X proxy는 이 최종 MHR Test 표에 넣지 않는다.')
    front.append('\n**입력 조건 정정:** 이 표의 original 및 미세조정 모델 점수는 배경 RGB를 제거한 조건이다. 이는 원래 Multi-HMR의 native RGB 추론과 다르며, 배경 제거가 성능에 미치는 영향을 분리하기 전에는 native 원본 성능으로 해석하면 안 된다. 원본 RGB를 보존하고 출력 head gate로 대상만 선택하는 native_rgb_target_head_v1 조건을 별도로 재평가한다. init NPZ 입력 경로와도 구분한다.')
    native_control=read(OUTPUT/'native_rgb_control_status.json',{})
    export_recovery=read(OUTPUT/'selected_export_recovery_status.json',{})
    if export_recovery:
        front.append(f'\n기존 배경 제거 조건의 저장 복구: `{export_recovery.get("status")}`. 시간 연장 budget의 한글 문구를 LANG=C 작업이 읽지 못해 종료된 원인을 확인하고 ASCII-escaped JSON으로 수정했다. 종료가 확인된 기존 실행기의 남은 export만 cache에서 재개했으며 학습·Test를 재시작하지 않았다. 후속 GPU subprocess에는 PYTHONUTF8=1도 지정했다. `extension_encoding_recovery.json`, `selected_export_recovery_status.json`. 원본 RGB 원인 분리 실험은 GPU6에서 별도로 유지한다.')
    if native_control:
        front.append(f'\n배경 제거 영향 분리 실험: `{native_control.get("status")}`, Train 검증 {len(native_control.get("samples",[])) if native_control.get("status")=="complete" else len(native_control.get("completed",[]))}/4개 완료. 동일 원본 가중치·동일 native K/화각·동일 출력 head gate·동일 MHR 변환에서 RGB 배경 보존 여부만 바꾼다. Test로 이 설정을 고르지 않는다. `native_rgb_control_plan.json`, `native_rgb_control_status.json`.')
        backfills=[read(p,{}) for p in sorted(OUTPUT.glob('native_rgb_backfill_gpu*_status.json'))]
        if backfills:
            front.append('\n추가 GPU 통제 실험: '+', '.join(f'GPU{x["gpu"]} `{x["status"]}`' for x in backfills)+'. 각 GPU의 기존 Train export 할당 완료와 실제 비점유를 확인한 뒤 고정된 Train 검증 영상을 같은 stage lock/cache로 처리한다. global 모델 선택·source 저장은 변경하지 않는다.')
        completed_native=[]
        for sample in native_control.get('samples',[]):
            value=read(OUTPUT/'predictions/diagnostic_original_native_rgb'/sample/'metrics.json',{})
            old=read(OUTPUT/'predictions/original'/sample/'metrics.json',{})
            if value.get('evaluation_protocol')=='native_rgb_target_head_v1' and value.get('full_frame_roster') and old:
                assert value['frames']==old['frames']==manifest['samples'][sample]['frames']
                completed_native.append((sample,old,value))
        if completed_native:
            front.extend(['\nTrain 영상별 실제 MHR CD-H — 해당 영상의 전체 프레임, 일부 영상 평균 없음:',
                '\n| Train 영상 | 프레임 | 배경 제거 CD-H cm | 원본 RGB CD-H cm |','|---|---|---|---|'])
            for sample,old,value in completed_native:
                front.append(f'| {sample} | {value["frames"]} | {old["CD_H_cm"]:.4f} | {value["CD_H_cm"]:.4f} |')
        native_comparison=read(OUTPUT/'native_rgb_control_comparison.json',{})
        if native_comparison:
            front.extend(['\nTrain 검증4개 전체의 입력 조건 비교(위 Test 표와 다른 범위):','\n| 입력 조건 | CD-H cm ↓ | ACC-H cm/frame² ↓ |','|---|---|---|'])
            for condition,value in native_comparison['summary'].items():
                front.append(f'| {condition} | {value["CD_H_cm"]:.4f} | {value["ACC_H_cm_per_frame2"]:.4f} |')
    control_geometry=read(OUTPUT/'native_rgb_control_geometry.json',{})
    if control_geometry.get('samples'):
        front.extend(['\n원인 분리용 Train 영상별 변환 전 진단. 각 행은 해당 영상의 모든 프레임이며 일부 영상 평균은 내지 않는다. 표면은 mapped SMPL-X proxy다. 프레임별 정렬 점수는 공식 CD-H가 아니고, 두 정렬 점수 차이를 가산 가능한 오차 성분으로 해석하지 않는다.',
            '\n| Train 영상 | 입력 | frame0 고정 CD proxy cm | 매 프레임 Sim3 진단 cm | 고정 정렬 후 중심 궤적 RMSE cm |','|---|---|---|---|---|'])
        for sample,value in control_geometry['samples'].items():
            for condition,metrics in value['conditions'].items():
                front.append(f'| {sample} | {condition} | {metrics["raw_SMPLX_fixed_frame0_CD_proxy_cm"]:.4f} | {metrics["diagnostic_per_frame_Sim3_CD_cm"]:.4f} | {metrics["fixed_alignment_centroid_trajectory_RMSE_cm"]:.4f} |')
        front.append('\n근거: `native_rgb_control_geometry.json`. 최초 black_platform의 native RGB700프레임은 head gate를 모두 통과했고 처음·중간·마지막3표본에서 대상 배우를 직접 확인했다. 해당 영상은 배경을 남겨도 고정 정렬 proxy가 낮아지지 않았으며 프레임별 정렬에서는 오차가 크게 줄었다. 이 영상에서 시간별 전역 위치·크기 변화가 큰 영향임을 시사하지만 카메라 또는 네트워크의 구체적인 원인이 확정된 것은 아니다. 전체4개 결과는 아래에 별도로 표시한다.')
        if control_geometry.get('all_four_complete'):
            assert set(control_geometry['samples'])==set(manifest['validation'])
            front.extend(['\nTrain 검증4개 전체·2714프레임의 변환 전 진단 평균(시퀀스 동일 가중치):',
                '\n| 입력 | frame0 고정 CD proxy cm | 매 프레임 중심 위치 보정 진단 cm | 매 프레임 Sim3 진단 cm | 중심 궤적 RMSE cm |','|---|---|---|---|---|'])
            for condition,value in control_geometry['summary'].items():
                front.append(f'| {condition} | {value["raw_SMPLX_fixed_frame0_CD_proxy_cm"]:.4f} | {value["diagnostic_per_frame_center_corrected_CD_cm"]:.4f} | {value["diagnostic_per_frame_Sim3_CD_cm"]:.4f} | {value["fixed_alignment_centroid_trajectory_RMSE_cm"]:.4f} |')
            front.append('\n4개 모두 원본 RGB의 프레임별 자세 진단이 배경 제거보다 낮았다. 그러나 원본 RGB의 고정 정렬 proxy38.90cm는 위치만 매 프레임 보정한 진단7.89cm보다 훨씬 높아, 이 범위에서는 시간별 전역 궤적 오차가 크게 작용한다. 이는 진단적 개입 결과이며 차이를 오차 기여율로 더하거나 빼지 않는다. 카메라 설정 또는 네트워크의 구체적 원인까지 증명한 것은 아니다.')
    init_trajectory=read(OUTPUT/'init_v6_Train_trajectory_diagnostic.json',{})
    if init_trajectory.get('all_four_complete'):
        value=init_trajectory['summary']
        front.append(f'\n동일 Train4개 init_v6 native MHR 진단: 고정 frame0 CD-H {value["MHR_fixed_frame0_CD_H_cm"]:.4f}cm, 매 프레임 중심 위치 보정 {value["diagnostic_per_frame_center_corrected_CD_cm"]:.4f}cm, 매 프레임 Sim3 {value["diagnostic_per_frame_Sim3_CD_cm"]:.4f}cm, 중심 궤적 RMSE {value["fixed_alignment_centroid_trajectory_RMSE_cm"]:.4f}cm. 원본의 변환 전 surface proxy와 MHR 지표를 동일 공간 점수로 섞지 않는다. `init_v6_Train_trajectory_diagnostic.json`.')
    if (OUTPUT/'native_trajectory_diagnostic/analysis.json').exists():
        front.append('\n[Train 깊이·이동 궤적 그래프](/home/namhj/EventPose/exp/261025/track2_multihmr_12h/native_trajectory_diagnostic/trajectory.png): 동일 MHR pelvis joint를 같은 camera 좌표계로 비교한 GT·init_v6·원본 RGB 결과다. GT만 world→camera를 한 번 적용했고 영상별64개 원본 프레임을 표시했다. Sim3를 적용하지 않은 진단이며 공식 CD-H나 전체 프레임 지표를 대신하지 않는다. 원시 좌표·범위: `native_trajectory_diagnostic/analysis.json` 및 영상별 NPZ.')
    depth_order=read(OUTPUT/'checkpoint_depth_compatibility/depth_order_sampled_control.json',{})
    if depth_order.get('all_four_complete'):
        d=depth_order['summary']
        front.append(f'\n깊이 복원 순서의 추가 Train 진단: 영상별64개·총256개 고정 표본에서 같은 네트워크 출력/포즈/인물 선택을 유지하고 focal scaling과 exp의 순서만 바꿨다. 표면 proxy는 공식 순서 {d["published_order"]:.4f}cm → 실험 순서 {d["experimental_exp_then_focal"]:.4f}cm였다. 이 값은 최종 MHR CD-H가 아니며 전체 프레임 결과도 아니다. fixed K에서 f/fn=0.63447이다. primary head translation은 저장된 SMPL-X와 zero-expression 재복원에서 추정했고 body median 재복원 잔차는 약0.31mm였다. GT로 계수를 fitting하지 않았지만 완전한 새 inference 실행과 MHR 전수 변환으로 확인하기 전에는 확정 개선으로 보고하지 않는다. `checkpoint_depth_compatibility/depth_order_sampled_control.json`.')
        front.append('\n현재 깊이 복원 함수의 AST는 [공식 2024 소스](https://github.com/naver/multi-hmr/blob/beed12c685e2/model.py)와 동일했다. 따라서 우리 구현만의 버그로 증명된 것은 아니다. 기존 학습·평가는 공식 순서를 유지하며 이 실험은 별도 진단이다. `checkpoint_depth_compatibility/source_comparison.json`.')
        front.append('\n[공식 학습 코드](https://github.com/naver/multi-hmr/blob/beed12c685e2/train.py)도 함께 확인했다. 학습 target은 q=log(z)×fn/f이고 공식 decoder exp(q×f/fn)는 그 정확한 역변환이다. 수치 round-trip 오차는 1.8e-15m였다. 따라서 순서 변경이 개선되더라도 곧바로 구현 오류의 증거는 아니며, 이 데이터에서의 깊이 보정 효과로 해석해야 한다. 수식·원본 소스·round-trip 근거를 `checkpoint_depth_compatibility/source_comparison.json`에 기록했다.')
    static_control=read(OUTPUT/'static_crop_Train_control_status.json',{})
    batch_check=read(OUTPUT/'native_batch_consistency.json',{})
    if batch_check:
        front.append(f'\nGT 없는 배치 일관성 검사: basket 고정6프레임을 개별·tail4·mixed6로 추론한 관절 최대 차이는 {batch_check["max_joint_difference_cm"]*10:.3f}mm였다. 마지막 프레임의 약4.22m 깊이는 개별 추론에서도 재현됐다. 검사한 프레임의 큰 깊이 오차를 이미지 간 배치 혼선으로 설명하는 증거는 없었다. 이 검사는 잠재적인 인물 특징 혼동이나 다른 프레임의 문제를 모두 배제하지 않는다. `native_batch_consistency.json`.')
    exact_depth=read(OUTPUT/'exact_depth_order_Train_control_status.json',{})
    if exact_depth:
        front.append(f'\n깊이 복원 순서의 정확한 전수 통제 실험: `{exact_depth.get("status")}`, 완료 {len(exact_depth.get("completed",[]))}/4개. 원본 RGB·원본 가중치·대상 head 선택·고정 K를 유지하고 decoder 순서만 변경한 별도 Train 실험이다. 네트워크 재추론과 동일 MHR-v2 변환으로 앞선256표본 근사 진단을 확인한다. 기존 공식 decoder·후속 학습·Test 선택 설정은 유지한다. `exact_depth_order_Train_control_plan.json`.')
        exact_result=read(OUTPUT/'exact_depth_order_Train_control_comparison.json',{})
        if exact_result:
            assert set(exact_result['per_sequence'])==set(manifest['validation'])
            value=exact_result['summary']
            front.append(f'\n실험 decoder의 Train4 전체 MHR 평균: CD-H {value["CD_H_cm"]:.4f}cm, ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame². 원본 RGB의 공식 decoder Train4 기준값은 CD-H38.9521cm다. Test 결과·공식 원본 결과로 대체하지 않는다. `exact_depth_order_Train_control_comparison.json`.')
    if static_control:
        front.append(f'\n고정 viewport Train 통제 실험: `{static_control.get("status")}`, 완료 {len(static_control.get("completed",[]))}/4개. 원본 RGB를 지우지 않고 제공된 마스크의 영상 전체 union을 포함하는 고정 crop을 쓴다. 매 프레임 zoom은 바꾸지 않고 K를 정확히 변환한다. viewport/FOV와 인물 해상도가 함께 바뀌는 별도 입력 진단이며 focal만 바꾼 실험으로 해석하지 않는다. 실제 유휴2-GPU pair lease를 얻은 뒤 실행하므로 진행 중인 DDP를 중단하거나 겹치지 않는다. 현재 전체 RGB 기준값·학습·선택은 유지한다. `static_crop_Train_control_plan.json`, `static_crop_projection_check.json`.')
        completed_crop=[]
        for sample in manifest['validation']:
            folder=OUTPUT/'predictions/diagnostic_original_static_crop'/sample
            value=read(folder/'metrics.json',{})
            if value.get('evaluation_protocol')=='native_rgb_static_crop_target_head_v1':
                assert value['full_frame_roster'] and value['frames']==manifest['samples'][sample]['frames']
                assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                completed_crop.append((sample,value))
        if completed_crop:
            front.extend(['\n고정 crop 완료 영상별 전체 프레임 MHR 비교(일부 영상 평균 없음):',
                '\n| Train 영상 | 프레임 | init_v6 CD-H cm | 전체 RGB CD-H cm | 고정 crop CD-H cm |','|---|---|---|---|---|'])
            for sample,value in completed_crop:
                original=read(OUTPUT/'predictions/diagnostic_original_native_rgb'/sample/'metrics.json')
                init=init_trajectory.get('samples',{}).get(sample,{})
                init_cd=f'{init["MHR_fixed_frame0_CD_H_cm"]:.4f}' if init else '—'
                front.append(f'| {sample} | {value["frames"]} | {init_cd} | {original["CD_H_cm"]:.4f} | {value["CD_H_cm"]:.4f} |')
        if (OUTPUT/'checkpoint_depth_compatibility/viewport_focal_comparison.json').exists():
            front.append('\nTrain4의 전체 영상 수평 화각은84.60°, 고정 crop은60.02°다. 896 resize 후 fx는492.32 →775.60px이며 모델 정규화 기준 화각60°의 focal ratio는0.63447 →0.99953이다. crop은 인물 해상도와 배경 문맥도 바꾸므로 화각만의 원인으로 단정하지 않는다. `checkpoint_depth_compatibility/checkpoint_args.json`, `viewport_focal_comparison.json`.')
        if (OUTPUT/'checkpoint_depth_compatibility/author_FOV_confirmation.json').exists():
            front.append('\n저자의 공개 답변을 확인했다: [Multi-HMR 저자 FOV 설명](https://github.com/naver/multi-hmr/issues/37#issuecomment-2345518900)에서 주 학습 화각50–70°, 평균60°, 약80%가60°였고 범위를 벗어나는 화각에는 취약할 수 있다고 설명한다. [다른 카메라 답변](https://github.com/naver/multi-hmr/issues/28#issuecomment-2245378769)도50–70°를 명시한다. 현재84.60° 입력은 그 범위를 벗어난다. 고정 crop과 별도의 깊이 보정 전수 실험에서 오차가 크게 줄어든 결과는 이 카메라·깊이 대응 취약성과 일치한다. 단, 개별 checkpoint의 정확한 학습 histogram이나 화각만의 기여율은 확인되지 않았다. `checkpoint_depth_compatibility/author_FOV_confirmation.json`.')
        head_target_audit=read(OUTPUT/'checkpoint_depth_compatibility/head_pelvis_depth_target_audit.json',{})
        if head_target_audit:
            front.append('\n추가 공개 코드 점검: [저자가 인정한 학습 depth target 불일치](https://github.com/naver/multi-hmr/issues/63#issuecomment-3792233538)는 head 기준 SMPL layer에 pelvis depth를 supervision하는 문제다. 현재 사용 checkpoint가 정확히 어떤 비공개 학습 target으로 만들어졌는지는 증명되지 않아 이 결과의 확정 원인으로 단정하지 않는다. Train64표본에서 head/pelvis 깊이 차이 최대치는 black49.97cm·traffic53.89cm여서 자세에 따라 영향 가능한 크기다. 우리 미세조정은 이 scalar target 코드를 복사하지 않고 실제 예측 mesh의 root와 GT root의3D 위치·상대 이동을 직접 supervision한다. `checkpoint_depth_compatibility/head_pelvis_depth_target_audit.json`, `issue_63_comments.json`.')
        head_reinterpret=read(OUTPUT/'head_pelvis_Train_counterfactual_geometry.json',{})
        if head_reinterpret.get('all_four_complete'):
            values=head_reinterpret['summary'];original=values['native_RGB'];crop=values['static_crop']
            front.append(f'\nhead/pelvis 의미를 추론에서 단순히 바꾸는 대안은 Train4 전수 CPU 통제에서 기각했다. 같은 포즈·머리 투영을 유지하고 기존 head 깊이를 pelvis 깊이로 재해석하자 변환 전 CD proxy가 전체 RGB {original["raw_SMPLX_fixed_frame0_CD_proxy_cm"]:.4f} →{original["pelvis_depth_reinterpretation_CD_proxy_cm"]:.4f}cm, crop {crop["raw_SMPLX_fixed_frame0_CD_proxy_cm"]:.4f} →{crop["pelvis_depth_reinterpretation_CD_proxy_cm"]:.4f}cm로 악화됐다. GT로 보정값을 fitting하지 않았고 기존 출력·학습·선택은 바꾸지 않았다. 이 결과는 공개 학습 target 불일치의 존재를 반박하는 것이 아니라, 현재 checkpoint에 대한 단순 추론 재해석을 개선책으로 채택할 근거가 없다는 뜻이다. `head_pelvis_Train_counterfactual_geometry.json`.')
        static_result=read(OUTPUT/'static_crop_Train_control_comparison.json',{})
        if static_result:
            assert set(static_result['per_sequence'])==set(manifest['validation'])
            value=static_result['summary']
            front.append(f'\n고정 crop 통제 실험의 Train4 전체 MHR 평균: CD-H {value["CD_H_cm"]:.4f}cm, ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame². 기존 전체 RGB와 같은 GT·변환·metric이고 입력 viewport만 다르다. Test 점수나 기존 원본 RGB 후속 모델의 결과가 아니다. `static_crop_Train_control_comparison.json`.')
        crop_visual=read(OUTPUT/'target_foreground_visuals/diagnostic_original_static_crop/validation/receipt.json',{})
        if crop_visual and not crop_visual.get('manual_review_pending',True):
            front.append(f'\n고정 crop 실제 입력·투영 시각 검토: {crop_visual["frames_rendered"]}표본을 직접 열었다. crop 안의 배경 RGB가 보존되고 대상 배우가 포함됨을 확인했다. cyan 첫 프레임의 raw root 깊이는 전체 RGB 약3.03m → crop3.92m, GT4.02m였다. 다른 표본에서도 깊이 오차가 줄어드는 단서가 있으나 일부 표본을 전체 점수로 일반화하지 않는다. `target_foreground_visuals/diagnostic_original_static_crop/validation/receipt.json`.')
            if crop_visual.get('manual_reviewed_samples') and all(v.get('MHR_v2_available') for v in crop_visual['samples'].values()):
                front.append('\n변환 완료 뒤 Train4의 최초·중간·마지막12표본을 다시 렌더링해 raw SMPL-X와 MHR-v2 양쪽을 직접 확인했다. 모두 물체를 조작하는 대상 배우를 따라가며 앉아 있는 배경 인물을 선택하지 않았다. black의 굽힘 자세와 traffic의 몸통·팔 등에는 잔여 오차가 보인다. JPG·예측 NPZ SHA를 묶어 검토 시점을 기록했으며 전체2714프레임 인물 선택을 전수 검증했다는 의미는 아니다.')
        crop_geometry=read(OUTPUT/'static_crop_Train_control_geometry.json',{})
        if crop_geometry.get('all_four_complete'):
            front.extend(['\n고정 crop 원인 분리: Train4 전체2714프레임 변환 전 동일 surface proxy 진단:',
                '\n| 입력 | frame0 고정 CD proxy cm | 매 프레임 중심 보정 진단 cm | 중심 궤적 RMSE cm |','|---|---|---|---|'])
            for name,value in crop_geometry['summary'].items():
                front.append(f'| {name} | {value["raw_SMPLX_fixed_frame0_CD_proxy_cm"]:.4f} | {value["diagnostic_per_frame_center_corrected_CD_cm"]:.4f} | {value["fixed_alignment_centroid_trajectory_RMSE_cm"]:.4f} |')
            front.append('\n원본 가중치·대상 선택·변환 전 mesh 정의를 유지한 입력 비교다. crop은 화각·인물 해상도·camera embedding·focal ratio를 함께 바꾸므로 한 요소만의 원인으로 단정하지 않는다. 프레임별 보정은 진단이고 가산 가능한 오차 분해가 아니다. `static_crop_Train_control_geometry.json`.')
    fixed_plan=read(OUTPUT/'fixed_view_adaptation_plan.json',{})
    if fixed_plan:
        preparation=read(OUTPUT/'fixed_view_training_preparation_status.json',{})
        dataset_checks=read(OUTPUT/'fixed_view_dataset_checks.json',{})
        fixed_state=read(OUTPUT/'fixed_training_status.json',{})
        front.append(f'\n고정 crop 후속 미세조정: `{fixed_state.get("status",fixed_plan.get("status"))}`. Train20 viewport 준비 `{preparation.get("status")}`; 실제 validation Dataset {dataset_checks.get("validation_records",0)}개와 Train20 전체 {dataset_checks.get("full_Train20_records",0)}개에서 crop K·GT 중심의 입력 영역 포함을 확인했다. 원본 RGB 기본 경로 보존과16개 current/frame0 anchor의 CPU target 구성도 확인했다. 원본 pretrained부터 head/last2/last4 세 조건을 비교하도록 설정했다. 대기 상태는 학습 시작으로 보고하지 않는다. 현재 원본 RGB wave의 GPU 평가와30개 저장이 끝난 뒤 실제2-GPU pilot을 통과해야 시작한다. nominal search1시간+refit0.75시간이며 실제 전체 학습 비용은 후보당3시간 미만으로 검사한다. ColorJitter·AdamW decay0.1·L2-SP, Train-only 선택을 유지한다. `fixed_view_adaptation_plan.json`, `fixed_view_training_plan.json`, `fixed_view_dataset_checks.json`, `fixed_training_jobs.json`.')
        fixed_finalization=read(OUTPUT/'fixed_finalization_status.json',{})
        if fixed_finalization:
            front.append(f'\n고정 crop 후속 검증·최종 저장 실행기: `{fixed_finalization.get("status")}`. 전체 RGB5개와 고정 crop4개를 같은 Train4 MHR CD-H·coverage95%·ACC tie-break로 비교한 뒤 선택을 고정한다. 고정 crop4개 Test10 전체 비교와 선택 모델의30개 저장을 이어가며, 전체 RGB가 Train에서 더 좋으면 그 기존 저장본을 보존한다. `fixed_finalization_jobs.json`, `final_selected.json`(선택 완료 후), `final_export_receipts.json`(저장 후).')
        fixed_export_check=read(OUTPUT/'fixed_view_export_format_check.json',{})
        if fixed_export_check:
            front.append('\n고정 crop의 실제 MHR-v2 예측을 별도 staging 경로로 export해 pose/shape/scales 배열 일치·원본 카메라 투영 metadata·고정 viewport provenance를 확인했다. source Uranus는 이 형식 검사로 변경하지 않았다. `fixed_view_export_format_check.json`.')
        fixed_geometry_audit=read(OUTPUT/'fixed_view_all30_geometry_audit.json',{})
        if fixed_geometry_audit:
            front.append(f'\n전체 입력 기하 확인: {fixed_geometry_audit["clips"]}/30개·{fixed_geometry_audit["total_frames"]}프레임의 고정 viewport는 모두975×975px이고 효과 화각60.02°로 저자가 보고한50–70° 범위 안에 든다. 영상 전체 target mask union 포함·원본 경계·frame count·mask SHA를 확인했다. 이 확인에는 Test GT pose/identity나 Test 점수를 읽지 않았다. 성능 개선 여부는 후속 MHR 평가로 판정한다. `fixed_view_all30_geometry_audit.json`.')
        if (OUTPUT/'final_completion_scope_audit.json').exists():
            scope_checks=read(OUTPUT/'final_completion_scope_audit.json')['checks']
            confirmed=sum(v['passed'] for v in scope_checks.values())
            front.append(f'\n전체 목표의 독립 완료 감사: {confirmed}개 항목 확인 / {len(scope_checks)-confirmed}개 완료 미입증. 작업 진행률 백분율을 뜻하지 않는다. `final_completion_scope_audit.json`에 전체 RGB5개+고정 crop4개 Test, Train-only 선택, 실제30개 source 배열·metadata 재로딩, 최종90개 시각 표본, 두 보고서, 모델 asset 보존, init_v6 개선, 승인된28시간 작업 창을 각각 확인한다. 학습 검사는 실제 checkpoint tensor의 유한값·원본 대비 변경과 search/refit 각각의 실제2-GPU·정규화3종·총 학습시간도 직접 검사한다. step0 선택을 재학습 개선으로 처리하지 않는다. 파일이 없거나 미완료인 항목은 통과로 처리하지 않으며 감사 실행만으로 Goal을 종료하지 않는다.')
        final_visual_state=read(OUTPUT/'final_visual_preparation_status.json',{})
        if final_visual_state:
            front.append(f'\n최종 source 시각 검토 준비 상태: `{final_visual_state["status"]}`. 최종30개 Uranus 저장·SHA 일치 후 CPU에서 각 영상의 최초·중간·마지막90표본을 실제 source NPZ로 렌더링한다. 입력 RGB·init_v6·raw SMPL-X·저장된 MHR 투영을 함께 만든다. 이미지 생성만으로 수동 검토를 통과시키지 않으며 직접 확인 전 receipt의 manual_review_pending을 유지한다. `final_visual_preparation_status.json`, `final_visual_preparation_launch.json`.')
    native_original_visual=read(OUTPUT/'target_foreground_visuals/diagnostic_original_native_rgb/receipt.json',{})
    if (not native_original_visual.get('manual_review_pending',True)
        and len(native_original_visual.get('manual_reviewed_samples',{}))==10):
        front.append('\n배경을 유지한 원본 RGB 모델도 Test10의 최초·중간·마지막30표본을 새로 렌더링해 raw SMPL-X와 MHR-v2 양쪽을 직접 확인했다. 검토 표본에서는 모두 대상 배우를 따라가고 앉아 있는 배경 인물을 선택하지 않았다. basket 마지막 표본은 GT depth2.60m에 raw/MHR 모두4.22m였으며 팔 자세도 어긋났다. 여러 직립 표본의 다리 투영과 watering의 무릎 꿇은 자세에서도 오차가 보인다. JPG·실제 MHR 예측 SHA를 확인해 기록했다. 전체7610프레임의 인물 선택 무오류 증명은 아니며 Test 표본을 학습이나 checkpoint 선택에 사용하지 않았다. `target_foreground_visuals/diagnostic_original_native_rgb/receipt.json`.')
    native_targets=read(OUTPUT/'native_surface_target_preparation.json',{})
    if native_targets:
        discrepancy=native_targets['validation_fitted_target_CD_discrepancy_cm']
        front.append(f'\n추가 학습 목표 점검: Train4의 기존 fitted SMPL-X 타깃과 native MHR GT 사이 CD proxy는 {discrepancy:.4f}cm였다. 이 값은 가산 가능한 모델 오차나 회복 가능한 개선량이 아니다. Train20의 기존1107개 cache frame에 대해 같은 FPS27+64 MHR 표면을 원래 카메라 좌표로 준비했다. Test 타깃은 생성하지 않았으며 현재9개 후보의 설정·점수·선택은 바꾸지 않았다. 선택적으로 정확한 MHR 타깃을 쓰는 추가 손실 경로는 별도 검증 중이며 실제2-GPU pilot 통과 전에는 새 학습을 시작하지 않는다. `native_surface_target_preparation.json`.')
        native_target_check=read(OUTPUT/'native_surface_dataset_checks.json',{})
        if native_target_check:
            front.append('\n선택적 MHR 타깃 Dataset의 CPU 확인: Train851 / validation256 / fullTrain1107개 전수에서 frame·GT/label/role SHA·카메라 좌표 타깃 일치와 유한값, 실제 target center 유효성을 확인했다. 기본 Dataset 출력 보존 및 같은 영상 frame0 anchor 연결, paired CD의 time/anchor 양쪽 유한·비영 gradient를 검사했다. 이는 실제 GPU 학습 성공의 증거는 아니다. `native_surface_dataset_checks.json`.')
        native_surface_pilot=read(OUTPUT/'native_surface_pilot_status.json',{})
        if native_surface_pilot:
            front.append(f'\n정확한 MHR 타깃 경로의 실제2-GPU pilot 상태: `{native_surface_pilot["status"]}`. 현재9개 후보의 최종30개 저장 완료 후 pair lease와 NVML의 실제 GPU 비점유를 확인하고20-step pilot을 실행한다. Test 타깃을 읽지 않고 pilot 가중치를 폐기하며 source SHA 보존을 검사한다. 이 pilot은 최종 모델이나 추가 성능 개선의 증거가 아니다. `native_surface_pilot_plan.json`, `native_surface_pilot_jobs.json`, `native_surface_pilot_status.json`.')
        native_surface_plan=read(OUTPUT/'native_surface_adaptation_plan.json',{})
        if native_surface_plan:
            surface_training_state=read(OUTPUT/'native_surface_training_status.json',{})
            front.append(f'\n후속 native MHR 표면 타깃3조건 상태: `{surface_training_state.get("status","준비; 미실행")}`. head/last2/last4 각각 원본 가중치에서 시작해 search30분+전체Train20 refit30분 이내로 제한하고 기존 ColorJitter·AdamW decay0.1·L2-SP를 유지한다. 현재9개 전체 비교·저장 및 실제2-GPU pilot 검증 후에도 init_v6 목표 미달이고 마감까지5시간 이상 남을 때만 실행 대상이다. 실행 시 기존9개와 신규3개 전체를 공통 MHR Train/Test 비교·최종 감사 범위에 포함한다. 기존 Test 노출을 명시했으며 신규 crop 학습모델의 Test 결과로 설정을 고르지 않았다. `native_surface_adaptation_plan.json`.')
            if native_surface_plan.get('orchestration_preparation'):
                preparation=native_surface_plan['orchestration_preparation']
                front.append(f'\n후속 실행기 준비: 학습 `{preparation["training"]}`, 전체12개 비교·저장 `{preparation["finalization"]}`, 12개 보고서 연결 `{preparation["reporting_full_twelve_scope"]}`, 12개 완료 감사 연결 `{preparation["completion_audit_full_twelve_scope"]}`. 현재 launch 허용 상태는 `{native_surface_plan.get("ready_to_launch",False)}`이며 실제 새 학습을 시작했다는 뜻이 아니다. 두 실행기는 통합 준비 완료와 기존9개 완료·실제 pilot 통과를 별도로 검사한다. `tools/run_track2_native_surface_training.py`, `tools/finalize_track2_native_surface_wave.py`.')
    root_filter=read(OUTPUT/'diagnostic_root_smoothing/comparison.json',{})
    if root_filter.get('all_four_complete'):
        front.extend(['\n추가 Train4 전수 CPU 후처리 진단 — 기존 학습·Test·source 저장 유지:',
            '\n| 조건 | Train CD-H cm ↓ | Train ACC-H cm/frame² ↓ |','|---|---|---|'])
        init_train=read(OUTPUT/'init_v6_Train4_MHR_baseline.json',{})
        if init_train:
            value=init_train['summary'];front.append(f'| init_v6 | {value["CD_H_cm"]:.4f} | {value["ACC_H_cm_per_frame2"]:.4f} |')
        for name,value in root_filter['summary'].items():
            front.append(f'| crop 원본 {name} | {value["CD_H_cm"]:.4f} | {value["ACC_H_cm_per_frame2"]:.4f} |')
        front.append('\n고정 sigma1·길이5의 Gaussian을 MHR root translation에만 적용한 GT-free 후처리다. 실제 mesh와 채점 body22 관절의 이동을 확인했으며, 채점에서 제외되는 고정 origin joint0는 이 검사에서 제외했다. ACC는 낮아졌지만 CD는 조금 높아져 현재 CD 우선 선택에 채택하지 않는다. 학습 설정이나 원본 결과를 바꾸지 않은 별도 진단이다. `diagnostic_root_smoothing/comparison.json`, `root_parameter_semantics_check.json`, `init_v6_Train4_MHR_baseline.json`.')
    native_plan=read(OUTPUT/'native_trajectory_plan.json',{})
    if native_plan:
        front.append(f'\n원본 RGB 후속 학습 상태: `{read(OUTPUT/"native_training_status.json",{}).get("status","prepared_not_started")}`. 같은 paired CD100 조건에서 상대 root 이동 supervision 없음/있음과 마지막2개 backbone block 학습을 비교한다. 입력은 모두 원본 RGB, 검증은 출력 target head gate를 적용한다. 현재4개 Train 통제 실험·기존 저장 복구·새 검증 경로·실제2-GPU pilot 검증을 끝낸 뒤 실행한다. 상대 이동 loss의 zero/shared-translation invariance·current/anchor gradient·double gradcheck는 통과했다. Test로 설정을 선택하지 않으며 기존 foreground Test를 이미 본 탐색이라는 한계는 남는다. `native_trajectory_plan.json`, `native_relative_motion_loss_checks.json`.')
        native_refits=read(OUTPUT/'native_refits_status.json',{})
        if native_refits:
            ready_refits=[c['name'] for c in native_plan['trials'] if (OUTPUT/'native_refit_choices'/(c['name']+'.json')).exists()]
            registered_refits=[j['label'] for j in read(OUTPUT/'native_refit_jobs.json',{}).get('jobs',[]) if j['label'].startswith('refit_')]
            front.append(f'\n원본 RGB checkpoint 검증·전체Train20 refit: 전체Train20 대응 checkpoint 준비 {len(ready_refits)}/3개; 등록된 refit 작업 `{", ".join(registered_refits) or "없음"}`. GPU6에서 완료된 search의 best/last를 같은 native RGB head gate로 재검증하고 coverage95% 이상에서 Train proxy 최솟값을 고른다. 전체Train20 refit은 고정 최대 update와 시간 한도를 사용하며 Test로 checkpoint를 고르지 않는다. 공유 bootstrap 학습·bootstrap Train20 refit·현재 search·현재 refit을 합쳐 후보당3시간 미만인지 검사한다. 실행기 원시 상태 `{native_refits.get("status")}`는 전체 작업 종료 때 갱신되므로 준비 수·등록 작업과 함께 해석한다. `native_refits_status.json`, `native_checkpoint_choices/`, `native_refit_choices/`.')
            parameter_audits=read(OUTPUT/'completed_native_refit_parameter_audits.json',{})
            if parameter_audits:
                front.extend(['\n완료된 refit checkpoint의 독립 CPU 재로딩:',
                    '\n| 모델 | 저장 step | 실제 성공 update | 공유 초기 학습 포함 시간 h | 초기화 대비 파라미터 L2 변화 |','|---|---|---|---|---|'])
                for name,value in parameter_audits.items():
                    front.append(f'| {name} | {value["step"]} | {value["successful_steps"]} | {value["entire_training_lineage_hours"]:.3f} | {value["L2_parameter_change"]:.6f} |')
                front.append('\n원본 체크포인트+delta 형식으로 저장되며 실제 delta tensor가 초기화와 달라졌고 유한값인지 확인했다. 이 검사는 성능 개선의 증거와 구분한다. `completed_native_refit_parameter_audits.json`.')
        front.extend(['\n원본 RGB 학습의 내부 검증(고정 Train4개 표본, SMPL-X proxy이며 최종 MHR Test 아님):',
            '\n| 후보 | 선택 checkpoint step | Train CD-H proxy cm ↓ | 검출 coverage |','|---|---|---|---|'])
        for cfg in native_plan['trials']:
            value=read(OUTPUT/'runs'/cfg['name']/'best.json',{})
            metric=value.get('cd_h_smplx_proxy_cm')
            front.append(f'| {cfg["name"]} | {value.get("step","—")} | '+(f'{metric:.4f}' if metric is not None else '—')+' | '+(f'{value["coverage"]:.2%}' if value.get('coverage') is not None else '—')+' |')
    front.append(f'\n사용자 승인으로 기존 마감에16시간을 추가했다. 현재 마감: {utc(budget["deadline_unix"])}; 총 {budget.get("total_hours",12)}시간. trial당3시간 한도와 GPU0 보존은 유지한다. `user_time_extension.json`.')
    front.append('\n사용자 지정 성능 목표는 init_v6의 Test10 CD-H보다 낮은 오차다. original 대비 개선만으로 목표 달성으로 보고하지 않는다. Test는 최종 비교에 사용하며 checkpoint 선택은 Train 검증으로 유지한다.')
    front.append('\n원본 열화 원인 재점검: 이전 배경 인물 선택은 실제 확인되어 해당 점수를 폐기했다. 현재 target-only 원본의 baseball·blue_easter_basket·tall_bar_stool 최초/중간/마지막9표본을 다시 직접 열어 대상 배우를 따르는 것을 확인했다. 전체 프레임 인물 오류가 없다는 증명은 아니다. Test10 전체의 변환 전 raw SMPL-X mapped surface CD proxy도41.9174cm로 높으므로 MHR 변환만을 원인으로 단정할 수 없다. 표본에서 자세·깊이 오차가 보이나 고정 frame0 Sim3로 제거되는 일정한 scale/translation bias만으로 CD를 설명할 수는 없다. init_v6의30개 NPZ 내부 report가 각 `_sam3d_body/diagnostics.json`과 모두 정확히 일치하여 SAM 3D Body 생성 결과임을 확인했다. 현재 original은 Multi-HMR이므로 서로 다른 모델·추론 경로 비교다. SAM3D 기록은 supplied static K, mask bbox, temporal_smoothing=false를 명시한다. 이것만으로 전체 오차 차이의 원인을 확정하지는 않는다. 근거: `init_v6_generation_provenance_audit.json`. RGB는 BGR→RGB 변환을 거치며 정규화를 원 구현 normalize_rgb와 비교한 최대 차이는2.384e-7이었다. 이 검사 범위에서는 채널/정규화 차이를 찾지 못했으며 전체 추론의 무오류 증명은 아니다. `original_preprocessing_reference_check.json`. 비교 영상: `target_foreground_visuals/`; 전수 geometry 진단: `target_geometry_diagnostic.json`.')
    chosen_name=selection.get('trial')
    chosen_value=model_values.get(chosen_name,{})
    if chosen_value and model_values.get('init_v6') and model_values.get('original'):
        chosen_cd=chosen_value['CD_H_cm'];original_cd=model_values['original']['CD_H_cm'];init_cd=model_values['init_v6']['CD_H_cm']
        front.append(f'\n이전 배경 제거 조건에서 Train 검증으로 선택한 모델은 `{chosen_name}`다. 해당 조건의 Test CD-H는 {chosen_cd:.4f}cm로 같은 입력의 original보다 {100*(original_cd-chosen_cd)/original_cd:.2f}% 낮지만, init_v6보다 {100*(chosen_cd-init_cd)/init_cd:.2f}% 높다. 이 결과는 init_v6 대비 개선이 아니다.')
    elif chosen_name and chosen_name in mhr_validation.get('models',{}):
        value=mhr_validation['models'][chosen_name]
        front.append(f'\n현재 Train 검증 선택 모델은 `{chosen_name}`이며 Train MHR CD-H {value["CD_H_cm"]:.4f}cm, ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame²다. 이 모델의 Test10 전체 평균은 아직 대기 중이다. 위 Test 표의 완료 모델 점수와 구분한다.')
    if chosen_name:
        for prefetch_path in sorted(OUTPUT.glob('selected_export_prefetch_gpu*_status.json')):
            prefetch=read(prefetch_path,{})
            if prefetch.get('trial')==chosen_name:
                front.append(f'\n선택 모델 export 사전 처리 GPU{prefetch["gpu"]}: `{prefetch["status"]}`. 해당 GPU의 전체 Test 할당 완료와 실제 프로세스 비점유를 확인한 후, 같은 GPU에 예정된 Train 영상만 동일 stage lock/cache로 추론·변환한다. source NPZ 저장은 기존 실행기가 담당한다. 근거: `{prefetch_path.name}`.')
        selected_folders=[OUTPUT/'predictions'/chosen_name/s for s in manifest.get('samples',{})]
        inferred=sum((p/'inference.json').exists() and (p/'smplx_predictions.npz').exists() for p in selected_folders)
        converted=sum((p/'conversion_v2.json').exists() and (p/'human_params_mhr_v2.npz').exists() for p in selected_folders)
        source_receipts={}
        for receipt_set in [available_exports,exports,priority_exports]:
            if receipt_set.get('trial')==chosen_name:source_receipts.update(receipt_set.get('samples',{}))
        verified_saved=sum(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in source_receipts.values())
        if priority_exports.get('trial')==chosen_name and priority_exports.get('samples'):
            front.append(f'\n선택 모델의 완료 Test 결과를 최종 경로에 우선 저장했다. 개별 source SHA·checkpoint·GT 불변 확인: `selected_priority_export_receipts.json`. 최종30개 통합 receipt는 기존 실행기가 갱신한다.')
            source_visual=read(OUTPUT/'target_foreground_visuals'/(chosen_name+'_exported')/'receipt.json',{})
            if source_visual.get('visual_review') and not source_visual.get('manual_review_pending',True):
                front.append(f'\n선택 모델의 실제 저장 Test10 NPZ를 카메라에 투영한 최초·중간·마지막30표본을 직접 확인했다. 검토 표본은 대상 배우를 유지했고 배경 인물 전환·좌표축 반전은 보이지 않았으나 자세·깊이 오차는 남아 있다. 전체 프레임의 무오류 증명은 아니다. source 파일과 JPG SHA 및 범위: `target_foreground_visuals/{chosen_name}_exported/receipt.json`.')
        front.extend(['\n이전 배경 제거 조건 모델의 처리 기록(변환 완료와 현재 source SHA 일치는 별도 단계):',
            '\n| 단계 | 완료 |', '|---|---|',
            f'| SMPL-X 추론 파일 생성 | {inferred}/30 |',
            f'| MHR v2 변환 파일 생성 | {converted}/30 |',
            f'| human_params_uranus.npz 저장·receipt SHA 일치 | {verified_saved}/30 |'])
        if saved_exports.get('trial') and saved_exports['trial']!=chosen_name:
            retained_count=sum(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in saved_exports.get('samples',{}).values())
            front.append(f'\n현재 source 경로에 보존된 기존 저장 모델은 `{saved_exports["trial"]}`이며 {retained_count}/30개가 해당 receipt SHA와 일치한다. 위 진행표는 새 선택 모델 `{chosen_name}`의 처리 수다. 새 결과가 준비될 때까지 기존 저장본을 유지한다.')
    if evaluation_validity:
        front.append('\noriginal은 원본 Multi-HMR 가중치의 재추론 결과다. init_v6는30개 NPZ의 내장 report와 SAM3D 진단 기록이 일치하는 SAM 3D Body의 native MHR 결과이며 Multi-HMR의 초기 출력이 아니다. init_v0의30개 NPZ는 모두 report.method=`native_SAM_fixed_K_no_depth_raw`, body_model=`MHR`를 명시한다. 따라서 init_v0 역시 현재 Multi-HMR 재추론과 같은 생성 경로가 아니다. 근거: `init_v0_generation_provenance_audit.json`. 현재 공통 입력은 제공된 target human mask 밖 RGB를 중립색으로 제거한 영상이다. 원래 화면 크기·화각·카메라 K는 유지한다. 마스크 안에 머리 검출점이 있는 후보 중 검출 신뢰도가 가장 높은 대상만 선택하며 배경 사람으로 fallback하지 않는다. 이전 unmasked 점수와 몸통 overlap만으로 선택했던 original CD-H 44.2949cm는 배경 인물 혼입으로 무효다. 각각 `invalid_unmasked/`, `diagnostic_body_overlap_gate/`에 보존하고 유효 표에서 제외했다.')
        front.append(f'\n현재 후속 단계: `{supervisor.get("status","준비 중")}`. 초기 target 재평가: `{target_eval_state.get("status","준비 중")}`. 후속 전체Train20 재학습: `{refit_prefetch_state.get("status","대기")}`. 기존 실행 중인 GPU 작업은 유지한다.')
    if evaluation_backfill:
        front.append(f'\n유휴 GPU 추가 평가 상태: `{evaluation_backfill["status"]}`. GPU5·6의 실행 프로세스 없음·사용률0%를 확인한 후 GPU6에 대기 중 Train 검증 영상1개, GPU5에 이미 재학습이 끝난 후보의 고정 Test 영상1개를 배정했다. 같은 checkpoint·stage lock/cache·변환·지표를 사용하며 global 모델 선택·export는 기존 실행기가 담당한다. Test 점수는 선택에 사용하지 않는다. `evaluation_backfill_status.json`, `evaluation_backfill_jobs.json`.')
    if evaluation_validity:
        front.append('\n검증: 30개 영상의 최초·중간·마지막 90개 표본에서 대상 마스크를 시각 확인했다. 별도 CPU 검사로 배경 RGB 제거와 카메라 K 보존(차이 0)을 확인했다. 전체 프레임 수동 검토를 의미하지 않는다. 근거: `target_mask_audit/audit.json`, `target_native_input_checks.json`. 평가 프로토콜: `target_foreground_head_v1`.')
    if full_mask_audit:
        front.append(f'\n전체 프레임 마스크/좌표 진단: `{full_mask_audit["status"]}`, {full_mask_audit["clips_checked"]}/30개 영상·{full_mask_audit["frames_checked"]}프레임 확인. GT 머리의 camera 투영점이 제공된 마스크의 같은 ellipse15 지지 영역에 드는지 CPU에서 점검한다. GT는 이 사후 진단에만 쓰며 대상 선택·학습 설정·추론·점수는 바꾸지 않는다. 물체가 얼굴을 가리면 GT점이 마스크 밖에 나올 수 있고, 이 통계만으로 분할 또는 인물 identity의 완벽함을 증명하지 않는다. `full_frame_target_mask_audit/audit.json`.')
        if full_mask_audit.get('summary'):
            audit_summary=full_mask_audit['summary']
            front.append(f'\n마스크 전수 진단 집계: 빈 마스크 {audit_summary["mask_empty_frames"]}프레임, GT 머리점 지지 영역 밖 {audit_summary["GT_head_outside_support_frames"]}/{audit_summary["total_frames"]}프레임({audit_summary["flagged_clips"]}개 영상). 표시된 모든 연속 구간에서 최초·중간·마지막 총{audit_summary["manually_reviewed_flagged_samples"]}표본을 직접 확인했고 bat·bottle·table에 의한 얼굴 가림이 보였다.102개 모두를 수동 검토했다는 뜻은 아니며 이 프레임을 점수에서 제외하지 않았다. `full_frame_target_mask_audit/flagged_visual_reviews.json`.')
        sequence_values={}
        for name in names:
            values={}
            for sample in manifest.get('test',[]):
                if name in ['init_v6','init_v0']:
                    value=init_comparison.get('models',{}).get(name,{}).get('per_sequence',{}).get(sample,{})
                else:
                    folder=OUTPUT/'predictions'/name/sample;value=read(folder/'metrics.json',{});info=read(folder/'inference.json',{})
                    if value.get('evaluation_protocol')!='target_foreground_head_v1' or value.get('conversion_revision')!=2:continue
                    if not info.get('target_foreground_only_input') or not info.get('all_accepted_heads_in_target_mask'):continue
                    if name!='original' and info.get('checkpoint_sha256')!=checkpoint_map.get(name,{}).get('checkpoint_sha256'):continue
                    if not (folder/'human_params_mhr_v2.npz').exists() or value.get('prediction_sha256')!=sha256(folder/'human_params_mhr_v2.npz'):continue
                if value.get('gt_sha256')!=manifest['samples'][sample]['gt_sha256']:continue
                if value.get('frames')!=manifest['samples'][sample]['frames'] or not value.get('full_frame_roster'):continue
                values[sample]=value
            if values:sequence_values[name]=values
        if sequence_values:
            front.extend(['\n### Target Test — 시퀀스별 완료분',
                '\n각 셀은 해당 영상의 모든 원본 프레임 결과다. 미검출 프레임도 같은 target의 관측값으로 보완해 포함한다. 빈 셀은 평가 대기이며 완료된 일부 영상만으로 Test10 평균을 만들지 않는다. 모델별 영상 전체가 완료되는 즉시 두 지표를 함께 갱신한다.'])
            columns=list(sequence_values)
            for metric,title in [('CD_H_cm','CD-H (cm)'),('ACC_H_cm_per_frame2','ACC-H (cm/frame²)')]:
                front.extend(['\n'+title,'\n| 시퀀스 | '+' | '.join(columns)+' |','|---|'+'---|'*len(columns)])
                for sample in manifest['test']:
                    cells=[f'{sequence_values[n][sample][metric]:.4f}' if sample in sequence_values[n] else '—' for n in columns]
                    front.append('| '+sample+' | '+' | '.join(cells)+' |')
    if target_geometry.get('samples') and target_geometry.get('evaluation_protocol')=='target_foreground_head_v1':
        front.extend(['\n### 원본 CD-H 독립 CPU 검증 — 완료 시퀀스',
            '\nraw SMPL-X는 공식 map으로 사상한 surface proxy이고 MHR v2가 요청한 최종 지표다. 각 공간에서 최초 프레임 Sim3를 따로 구하므로 두 값의 차이를 가산 가능한 오차 성분으로 해석하지 않는다. 아래는 각 영상의 모든 프레임 결과이며 모델 선택에 사용하지 않는다.',
            '\n| 시퀀스 | 전체 프레임 | raw SMPL-X CD proxy cm | MHR v2 CD-H cm | init_v6 CD-H cm |', '|---|---|---|---|---|'])
        for sample,value in target_geometry['samples'].items():
            front.append(f'| {sample} | {value["frames"]} | {value["raw_mapped_SMPLX_CD_proxy_cm"]:.4f} | {value["converted_MHR_CD_H_cm"]:.4f} | {value["init_v6_CD_H_cm"]:.4f} |')
    if foreground_visual_review:
        front.append('\n새 예측 영상 점검: Test10의 최초·중간·마지막30프레임을 직접 확인했다. 표본의 foreground 입력에는 대상 배우만 있었고 raw SMPL-X 예측도 배우를 따라갔다. 배경 인물 선택은 보이지 않았지만 다리 자세·깊이 오차가 남았다. 전체 프레임 수동 무오류 확인은 아니다. `target_foreground_visuals/receipt.json`과 시퀀스별 JPG에 기록했다.')
    if exported_visual_review:
        front.append('\n실제 저장 NPZ 점검: Head의 `human_params_uranus.npz` Test10개를 직접 불러 최초·중간·마지막30표본을 카메라에 투영했다. 확인한 표본에서는 대상 배우를 유지했고 배경 인물 선택이나 좌표축 반전은 보이지 않았다. 깊이 오차는 남아 있으며2D 투영 일치가3D 정확도를 뜻하지 않는다. 전체 프레임 수동 검토는 아니다. 원본 파일 SHA와 검토 범위: `target_foreground_visuals/head_strong_exported/receipt.json`.')
    if train_export_review:
        front.append(f'\nHead 저장본 추가 Train 시각 점검: {len(train_export_review["samples"])}개 영상의 최초·중간·마지막 {train_export_review["manual_review_frames"]}표본 이미지를 직접 확인했다. 확인 범위에서 대상 배우를 유지했고 배경 사람으로 전환되거나 좌표축이 반전된 표본은 관찰되지 않았다. 자세·깊이 오차는 남으며 전체 프레임 검토나3D 정확도 보증은 아니다. 이미지·저장 NPZ SHA와 검토 목록: `target_foreground_visuals/head_strong_exported/all/visual_review_progress.json`.')
    if selected_prediction_review:
        front.append(f'\n현재 선택 모델 `{chosen_name}` 예측 시각 점검: Test10의 최초·중간·마지막30표본 JPG를 직접 열어 raw SMPL-X와 native MHR v2가 대상 배우를 유지함을 확인했다. 확인한 표본에서 배경 인물 전환·좌표축 반전은 보이지 않았으며 자세·깊이 오차는 남는다. source Uranus 교체 전 예측 파일의 검토이며 전체 프레임 수동 검토가 아니다. 이 Test 사후 진단으로 모델을 선택하지 않았다. `target_foreground_visuals/{chosen_name}/receipt.json`.')
    if selected_geometry:
        front.append(f'\n현재 선택 모델 독립 CPU CD-H 검증: {len(selected_geometry.get("samples",{}))}/10개 영상의 모든 원본 프레임을 확인했다. 각 native MHR CD-H가 기존 GPU 값과0.02cm 이내인지 검사하며 모델 선택에는 쓰지 않는다. `selected_geometry_diagnostics/{chosen_name}.json`.')
        if selected_geometry.get('summary'):
            geo=selected_geometry['summary']
            front.append(f'\n선택 모델 CPU Test10 집계: raw mapped SMPL-X CD proxy {geo["raw_mapped_SMPLX_CD_proxy_cm"]:.4f}cm, native MHR CD-H {geo["converted_MHR_CD_H_cm"]:.4f}cm. 각각 최초 프레임 Sim3를 따로 fitting하므로 두 값의 차이를 가산 가능한 변환 오차 성분으로 해석하지 않는다.')
    if target_geometry.get('summary') and target_geometry.get('evaluation_protocol')=='target_foreground_head_v1':
        summary=target_geometry['summary']
        front.append(f'\n원본 Test10 독립 CPU 검증: raw SMPL-X mapped-surface CD proxy {summary["raw_mapped_SMPLX_CD_proxy_cm"]:.4f}cm, MHR v2 CD-H {summary["converted_MHR_CD_H_cm"]:.4f}cm. MHR CD-H는 GPU 값과0.02cm 이내로 독립 재현했다. 원시 값과 시퀀스별 root 궤적: `target_geometry_diagnostic.json`. 이 진단으로 모델을 선택하지 않는다.')
    if (OUTPUT/'native_baseline_Test_plan.json').exists():
        native_names=[('original_RGB','diagnostic_original_native_rgb')]+[(cfg['name'],cfg['name']) for cfg in native_plan.get('trials',[])]
        native_evaluation=read(OUTPUT/'native_evaluation_plan.json',{})
        if native_evaluation:
            legacy_name=native_evaluation['legacy_model']
            native_names.insert(1,(legacy_name,legacy_name))
        native_values={};native_counts={};native_completed_cases={}
        for label,trial in native_names:
            samples={}
            for sample in manifest['test']:
                folder=OUTPUT/'predictions'/trial/sample;metric=read(folder/'metrics.json',{})
                if metric.get('evaluation_protocol')!='native_rgb_target_head_v1':continue
                assert metric['full_frame_roster'] and metric['frames']==manifest['samples'][sample]['frames']
                assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
                assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                samples[sample]=metric
            native_counts[label]=len(samples)
            native_completed_cases[trial]=samples
            if len(samples)==10:
                native_values[label]={key:sum(v[key] for v in samples.values())/10 for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
        table_values={name:model_values[name] for name in ['init_v6','init_v0']}
        table_values.update(native_values)
        best_cd=min(v['CD_H_cm'] for v in table_values.values());best_acc=min(v['ACC_H_cm_per_frame2'] for v in table_values.values())
        native_front=['\n## init_v6 / init_v0 비교 — 원본 RGB Test10',
            '\n현재 기준은 원본 RGB를 보존하고 출력 target head gate로 대상만 선택하는 `native_rgb_target_head_v1`이다. 각 모델은 동일 Test10 전체7610프레임·MHR revision2·시퀀스 동일 가중치로 평가한다. 전체10개가 끝나기 전에는 평균을 표시하지 않는다. init_v6/init_v0는 제공된 native MHR NPZ이며 재추론 입력을 바꾼 결과가 아니다.',
            '\n| 모델 | 완료 Test | CD-H cm ↓ | ACC-H cm/frame² ↓ | init_v6 대비 CD-H 개선율 |','|---|---|---|---|---|']
        for label in ['init_v6','init_v0']+[name for name,_ in native_names]:
            value=table_values.get(label);count=10 if value else native_counts.get(label,0)
            if value:
                cd=f'{value["CD_H_cm"]:.4f}';acc=f'{value["ACC_H_cm_per_frame2"]:.4f}'
                if value['CD_H_cm']==best_cd:cd='**'+cd+'**'
                if value['ACC_H_cm_per_frame2']==best_acc:acc='**'+acc+'**'
                delta=100*(table_values['init_v6']['CD_H_cm']-value['CD_H_cm'])/table_values['init_v6']['CD_H_cm']
                native_front.append(f'| {label} | {count}/10 | {cd} | {acc} | {delta:+.2f}% |')
            else:native_front.append(f'| {label} | {count}/10 | — | — | 대기 |')
        native_learning_status=read(OUTPUT/'native_training_status.json',{}).get('status','준비')
        if len(read(OUTPUT/'native_refit_mappings.json',{}))==3:
            native_learning_status='세 후보의 search 및 전체 Train20 재학습 완료'
        native_front.append('\n원본 RGB와 배경 제거의 점수는 입력 조건이 달라 개선율을 서로 섞지 않는다. 아래 이전 조건의16.4182cm는 원본 RGB 후속 모델의 점수가 아니다. source Uranus는 별도 검증 후 순차 갱신하며 아래 저장 진행표는 실제 SHA 일치 수다. 후속 학습 상태: '+native_learning_status+'.')
        native_final=read(OUTPUT/'native_finalization_status.json',{})
        if native_final:
            native_front.append(f'\n원본 RGB MHR 선택·Test·전체30개 저장 상태: `{native_final.get("status")}`. 기존 paired 모델도 search16-Train checkpoint를 같은 원본 RGB로 검증해 대조한다. 새 세 조건과 원본·기존 모델을 같은 Train4개 MHR CD-H로 고른 후 선택을 고정한다. Test와 Train20 추론은 정상5개 GPU의 동적 작업 큐로 처리한다. `native_evaluation_plan.json`, `native_finalization_status.json`.')
        native_choice=read(OUTPUT/'native_selected.json',{})
        if native_choice:
            name=native_choice['trial'];value=native_choice['models'][name]
            original=native_choice['models']['diagnostic_original_native_rgb']
            gain=100*(original['CD_H_cm']-value['CD_H_cm'])/original['CD_H_cm']
            native_front.append(f'\n원본 RGB5개 후보의 Train-only 선택 완료: `{name}`. 같은 Train4 전체2714프레임 MHR CD-H {value["CD_H_cm"]:.4f}cm / ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame², target detection coverage {100*native_choice["target_coverage"][name]:.2f}%. 같은 입력 원본보다 Train CD-H {gain:.2f}% 감소했으나 ACC-H는 높아졌다. 이 선택은 Test 점수로 하지 않았으며 Test 개선이나 init_v6 초과 달성의 증거는 아니다. 고정 crop 후속 후보와의 전체 비교·최종 선택은 별도로 남아 있다. `native_selected.json`.')
            native_complete=read(OUTPUT/'native_test_comparison.json',{})
            if native_complete:
                measured=native_complete['models'][name]
                base=native_complete['models']['diagnostic_original_native_rgb']
                init=init_comparison['models']['init_v6']
                gain=100*(base['CD_H_cm']-measured['CD_H_cm'])/base['CD_H_cm']
                gap=100*(measured['CD_H_cm']-init['CD_H_cm'])/init['CD_H_cm']
                native_front.append(f'\n원본 RGB Test10 전체7610프레임 검증 완료: Train에서 고른 `{name}`의 CD-H {measured["CD_H_cm"]:.4f}cm / ACC-H {measured["ACC_H_cm_per_frame2"]:.4f}cm/frame². 같은 입력 원본 대비 CD-H {gain:.2f}% 감소, init_v6 대비 CD-H 차이 {gap:+.2f}%다. 양수 차이는 init_v6보다 나쁜 결과이며, original 대비 개선만으로 목표를 달성한 것으로 처리하지 않는다. 이 단계의 전체 Test 비교는 완료했으며 고정 crop 후속 학습·전체 후보 최종 선택은 계속 진행한다. `native_test_comparison.json`, `native_completed_Test_models.json`.')
            selected_native_visual=read(OUTPUT/'target_foreground_visuals'/name/'receipt.json',{})
            if (not selected_native_visual.get('manual_review_pending',True)
                and len(selected_native_visual.get('manual_reviewed_samples',{}))==10):
                native_front.append(f'\n선택 모델 `{name}`의 Test10 최초·중간·마지막30표본도 raw SMPL-X/MHR-v2 양쪽을 직접 열어 확인했다. 검토 표본 모두 배경 인물이 아닌 대상 배우를 따라가며 원본 배경 RGB를 유지했다. basket 마지막 raw depth는 원본4.22m에서2.79m로 바뀌었고 GT는2.60m였다. watering의 무릎 꿇은 자세·절대 depth bias 등 잔여 오차는 보인다. 해당 표본의 정성 관찰이며 전체 프레임 인물 선택 무오류나 최종 source90표본 검토 완료를 뜻하지 않는다. JPG·예측 SHA를 확인했으며 Test로 모델을 다시 고르지 않았다. `target_foreground_visuals/{name}/receipt.json`.')
            learned_geometry=read(OUTPUT/(name+'_Train_geometry.json'),{})
            if learned_geometry.get('all_four_complete'):
                native_front.extend(['\nTrain4 전체2714프레임의 미세조정 잔여 오차 진단 — 변환 전 mapped SMPL-X 공간이며 공식 MHR 점수와 구분:',
                    '\n| 모델 | frame0 고정 CD proxy cm | 매 프레임 중심 보정 CD 진단 cm | 매 프레임 Sim3 CD 진단 cm | 중심 궤적 RMSE cm |',
                    '|---|---:|---:|---:|---:|'])
                for label,value in learned_geometry['summary'].items():
                    native_front.append(f'| {label} | {value["raw_SMPLX_fixed_frame0_CD_proxy_cm"]:.4f} | {value["diagnostic_per_frame_center_corrected_CD_cm"]:.4f} | {value["diagnostic_per_frame_Sim3_CD_cm"]:.4f} | {value["fixed_alignment_centroid_trajectory_RMSE_cm"]:.4f} |')
                native_front.append(f'\n이동 경로 진단은 좋아졌지만 중심 또는 Sim3를 프레임별로 보정한 몸 자세·형상·방향 진단은 개선되지 않았다. 이는 가산 가능한 오차 분해나 회복 가능한 개선량의 추정이 아니며 기존 학습·선택·Test 결과를 바꾸지 않는다. Train의 잔여 오차와 fitted SMPL-X/native MHR 타깃 차이를 함께 점검할 근거다. `{name}_Train_geometry.json`.')
            cases=native_completed_cases.get(name,{})
            if cases:
                native_front.extend([f'\n선택 모델 `{name}`의 Test 영상별 완료 결과 — 각 행은 해당 영상 전체 프레임이며, 미완료 영상은 생략한다:',
                    '\n| Test 영상 | 프레임 | init_v6 CD | init_v0 CD | 선택 모델 CD | init_v6 ACC | init_v0 ACC | 선택 모델 ACC |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|'])
                for sample in manifest['test']:
                    if sample not in cases:continue
                    values=[init_comparison['models'][base]['per_sequence'][sample] for base in ['init_v6','init_v0']]+[cases[sample]]
                    cells=[]
                    for key in ['CD_H_cm','ACC_H_cm_per_frame2']:
                        best=min(v[key] for v in values)
                        cells.extend(('**' if v[key]==best else '')+f'{v[key]:.4f}'+('**' if v[key]==best else '') for v in values)
                    native_front.append(f'| {sample} | {cases[sample]["frames"]} | '+ ' | '.join(cells)+' |')
                native_front.append('\nCD 단위는 cm, ACC 단위는 cm/frame²다. 이 표의 완료된 일부 영상만 평균내 전체 Test 성능으로 표시하지 않는다. 전체10개 평균은 맨 위 비교표가 완성됐을 때 판단하며 이 Test 결과로 모델을 다시 선택하지 않는다.')
        learned_geometry_state=read(OUTPUT/'native_selected_geometry_audit_status.json',{})
        if learned_geometry_state:
            learned_trial=learned_geometry_state['trial']
            learned_geometry=read(OUTPUT/'selected_geometry_diagnostics'/(learned_trial+'.json'),{})
            native_front.append(f'\n전체 RGB 미세조정 모델의 별도 CPU 진단: `{learned_geometry_state["status"]}`. GPU 학습·평가를 점유하지 않고 Test10 전체7610프레임의 raw mapped SMPL-X와 MHR-v2, init_v6를 같은 geometry 진단으로 비교한다. 부분 영상 평균이나 이 진단에 따른 checkpoint 선택은 하지 않는다. `native_selected_geometry_audit_status.json`.')
            if learned_geometry.get('summary') and learned_geometry.get('motion_summary'):
                assert set(learned_geometry['samples'])==set(manifest['test'])
                assert sum(v['frames'] for v in learned_geometry['samples'].values())==7610
                value=learned_geometry['summary']
                native_front.append(f'\n`{learned_trial}` Test10 CPU 재현: raw mapped SMPL-X proxy {value["raw_mapped_SMPLX_CD_proxy_cm"]:.4f}cm / MHR-v2 CD-H {value["converted_MHR_CD_H_cm"]:.4f}cm. 두 값은 별도 최초 프레임 Sim3를 쓰므로 차이를 가산 가능한 변환 오차로 해석하지 않는다.')
                comparisons={}
                original_geometry_reference=read(OUTPUT/'selected_geometry_diagnostics/diagnostic_original_native_rgb.json',{})
                if original_geometry_reference.get('motion_summary'):
                    comparisons['original_RGB']=original_geometry_reference['motion_summary']['original_MHR']
                comparisons[learned_trial]=learned_geometry['motion_summary']['original_MHR']
                comparisons['init_v6']=learned_geometry['motion_summary']['init_v6_MHR']
                native_front.extend(['\nTest10 전수 전역 궤적·프레임별 보정 진단:',
                    '\n| 모델 | 매 프레임 중심 보정 CD 진단 cm ↓ | 매 프레임 Sim3 CD 진단 cm ↓ | 중심 궤적 RMSE cm ↓ |', '|---|---:|---:|---:|'])
                keys=['center_corrected_CD_diagnostic_cm','per_frame_Sim3_CD_diagnostic_cm','centroid_trajectory_RMSE_cm']
                minima={key:min(v[key] for v in comparisons.values()) for key in keys}
                for name,metrics in comparisons.items():
                    cells=[('**'+f'{metrics[key]:.4f}'+'**') if metrics[key]==minima[key] else f'{metrics[key]:.4f}' for key in keys]
                    native_front.append('| '+name+' | '+' | '.join(cells)+' |')
                native_front.append(f'\n모든 값은 Test 사후 원인 진단이며 공식 CD-H·오차 성분의 가산 분해·미래 모델 선택 기준이 아니다. `selected_geometry_diagnostics/{learned_trial}.json`.')
        native_audit=read(OUTPUT/'native_completion_scope_audit.json',{})
        if native_audit:
            checks=native_audit['checks'];passed=sum(v['passed'] for v in checks.values())
            pending=', '.join(n for n,v in checks.items() if not v['passed'])
            native_front.append(f'\n원본 RGB 단계 독립 증거 점검: {passed}/{len(checks)}항목 확인. 실제 실행 configuration·world size2·GPU 시작 기록·세 regularizer·학습 시간·checkpoint SHA 및 Train 선택을 확인하며, 아직 확인되지 않은 항목은 `{pending or "없음"}`이다. 이 점검은 현재 원본 RGB5개 후보 범위이며 고정 crop 후속 단계까지 포함한 전체 목표 완료 감사와 다르다. `native_completion_scope_audit.json`.')
        native_test_prefetch=read(OUTPUT/'native_test_prefetch_status.json',{})
        if native_test_prefetch:
            native_front.append(f'\n유휴 GPU 쌍의 고정 Test 선행 평가: `{native_test_prefetch.get("status")}`, 완료 모델·영상 {len(native_test_prefetch.get("completed",[]))}건. 두 refit 완료 및 나머지 refit 실행을 확인하고 실제 비어 있는 GPU 쌍에만 배정한다. 기존 고정 후보·checkpoint·Test roster·전체 프레임·MHR-v2를 유지하며 source 저장이나 Train 선택은 바꾸지 않는다. main5-GPU Test 단계가 시작되면 새 선행 작업 배정을 멈추고 진행 중인 건만 같은 stage lock으로 마무리한다. `native_test_prefetch_status.json`, `native_test_prefetch_jobs.json`.')
        native_validation_progress=read(OUTPUT/'native_validation_mhr_progress.json',{})
        fixed_validation_progress=read(OUTPUT/'fixed_validation_mhr_progress.json',{})
        independently_completed_fixed=read(OUTPUT/'fixed_completed_Train4_models.json',{})
        if independently_completed_fixed.get('whole_Train4_only'):
            for checked in independently_completed_fixed['models'].values():
                assert checked['sequences']==4 and checked['frames']==2714
                assert set(checked['per_sequence'])==set(manifest['validation'])
            fixed_validation_progress=dict(fixed_validation_progress,models=dict(
                fixed_validation_progress.get('models',{}),**independently_completed_fixed['models']))
        fixed_last4_audit=read(OUTPUT/'fixed_last4_Train4_independent_audit.json',{})
        if fixed_last4_audit.get('status')=='passed':
            assert fixed_last4_audit['sequences']==4 and fixed_last4_audit['frames']==2714
            assert set(fixed_last4_audit['per_sequence'])==set(manifest['validation'])
            checked=dict(fixed_last4_audit['summary'],sequences=4,frames=2714,per_sequence=fixed_last4_audit['per_sequence'])
            fixed_validation_progress=dict(fixed_validation_progress,models=dict(
                fixed_validation_progress.get('models',{}),**{fixed_last4_audit['trial']:checked}))
        completed_validation=dict(native_validation_progress.get('models',{}),**fixed_validation_progress.get('models',{}))
        init_train_reference=read(OUTPUT/'init_v6_Train4_MHR_baseline.json',{})
        if init_train_reference.get('sequences')==4 and init_train_reference.get('frames')==2714:
            assert set(init_train_reference['per_sequence'])==set(manifest['validation'])
            completed_validation={'init_v6':dict(init_train_reference['summary'],sequences=4,frames=2714),**completed_validation}
        if completed_validation:
            native_front.extend(['\n동일 Train 검증4개·2714프레임 MHR 평가 완료분(Test 표와 다른 모집단):',
                '\n| 모델 | 입력 | Train CD-H cm ↓ | Train ACC-H cm/frame² ↓ |','|---|---|---|---|'])
            min_train_cd=min(v['CD_H_cm'] for v in completed_validation.values())
            min_train_acc=min(v['ACC_H_cm_per_frame2'] for v in completed_validation.values())
            for name,value in completed_validation.items():
                assert value['sequences']==4 and value['frames']==2714
                view='제공 init_v6' if name=='init_v6' else '고정 crop RGB' if name in fixed_validation_progress.get('models',{}) else '전체 RGB'
                cd=f'{value["CD_H_cm"]:.4f}';acc=f'{value["ACC_H_cm_per_frame2"]:.4f}'
                if value['CD_H_cm']==min_train_cd:cd='**'+cd+'**'
                if value['ACC_H_cm_per_frame2']==min_train_acc:acc='**'+acc+'**'
                native_front.append(f'| {name} | {view} | {cd} | {acc} |')
            native_front.append('\n각 행은 네 영상 전체가 완료된 평균이며 굵은 값은 현재 표에 완료된 행 사이의 최솟값이다. init_v6는 같은 Train4에 대한 제공 결과의 비교 기준이며 미세조정 후보 선택 대상에 넣지 않는다. 전체 RGB와 고정 crop 후보 전체의 검증이 끝나기 전에는 전체 후보 최종 선택으로 해석하지 않는다. crop의 독립 검사 완료분도 포함하며 이는 Train16/4 search checkpoint의 평가다. Test는 Train20 refit으로 별도 평가한다. `native_validation_mhr_progress.json`, `fixed_validation_mhr_progress.json`, `fixed_completed_Train4_models.json`, `init_v6_Train4_MHR_baseline.json`.')
            if fixed_last4_audit.get('status')=='passed':
                original_fixed=completed_validation.get('diagnostic_original_static_crop')
                value=fixed_last4_audit['summary']
                gain=100*(original_fixed['CD_H_cm']-value['CD_H_cm'])/original_fixed['CD_H_cm'] if original_fixed else None
                gain_text=f'동일 crop 원본 대비 Train CD-H {gain:.2f}% 감소. ' if gain is not None else ''
                native_front.append(f'\nlast4의 독립 검증 완료: Train4 전체2714프레임 MHR CD-H {value["CD_H_cm"]:.4f}cm / ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame², target coverage {fixed_last4_audit["target_coverage"]*100:.2f}%. {gain_text}이는 Train16/4에서 고른 search checkpoint의 평가이며 전체Train20 refit의 Test 결과가 아니다. source·GT·mask·예측·checkpoint SHA와 동일 metric protocol을 검증했다. 집계 실행기의 순차 반영을 기다리지 않고 완료분만 표에 표시하며 최종 후보 선택은 기존 실행기에 맡긴다. `fixed_last4_Train4_independent_audit.json`.')
        geometry_state=read(OUTPUT/'native_baseline_geometry_audit_status.json',{})
        if geometry_state:
            native_front.append(f'\n원본 RGB Test10 독립 CPU geometry 검증: `{geometry_state.get("status")}`. GPU 기준값이 모두 완료되면 동일 native MHR 공간에서 원본·init_v6의 전체 프레임을 다시 복원한다. Test 사후 원인 진단이며 모델·학습 설정 선택에 쓰지 않는다.')
        native_geometry=read(OUTPUT/'selected_geometry_diagnostics/diagnostic_original_native_rgb.json',{})
        if native_geometry.get('summary'):
            assert set(native_geometry['samples'])==set(manifest['test'])
            geometry=native_geometry['summary']
            native_front.append(f'\n독립 CPU 재현 Test10: 원본 RGB MHR CD-H {geometry["converted_MHR_CD_H_cm"]:.4f}cm, 변환 전 mapped SMPL-X proxy {geometry["raw_mapped_SMPLX_CD_proxy_cm"]:.4f}cm, init_v6 MHR CD-H {geometry["init_v6_CD_H_cm"]:.4f}cm. 원본 MHR 결과는 각 영상의 GPU 점수와0.02cm 이내인지 확인했다.')
            if native_geometry.get('motion_summary'):
                native_front.extend(['\nTest10 전수 native MHR 위치·자세 진단(공식 점수 아님):',
                    '\n| 모델 | 매 프레임 중심 보정 CD 진단 cm | 매 프레임 Sim3 CD 진단 cm | 중심 궤적 RMSE cm |','|---|---|---|---|'])
                for name,value in native_geometry['motion_summary'].items():
                    native_front.append(f'| {name} | {value["center_corrected_CD_diagnostic_cm"]:.4f} | {value["per_frame_Sim3_CD_diagnostic_cm"]:.4f} | {value["centroid_trajectory_RMSE_cm"]:.4f} |')
                native_front.append('\n프레임별 보정은 원인 분리를 위한 사후 개입이다. 공식 CD-H를 대체하거나 점수 차이를 가산 가능한 오차 기여율로 해석하지 않는다. `selected_geometry_diagnostics/diagnostic_original_native_rgb.json`.')
        native_audit=read(OUTPUT/'native_completion_scope_audit.json',{})
        if native_audit:
            passed=sum(v['passed'] for v in native_audit['checks'].values())
            native_front.append(f'\n원본 RGB 최종 범위 감사: {passed}/{len(native_audit["checks"])}개 확인. 미완료 항목은 완료로 처리하지 않으며 실제 source30개 재로딩·Train 선택·Test5모델·init_v6 개선을 각각 검사한다. 시각 검토·보고서·전체 작업 시간은 별도로 남는다. `native_completion_scope_audit.json`.')
        native_selected=read(OUTPUT/'native_selected.json',{})
        if native_selected:
            winner=native_selected['trial'];value=native_selected['models'][winner]
            native_front.append(f'\n원본 RGB Train 검증 선택: `{winner}`, Train4 MHR CD-H {value["CD_H_cm"]:.4f}cm / ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame². Test로 선택하지 않았다. `native_selected.json`.')
            folders=[OUTPUT/'predictions'/winner/s for s in manifest['samples']]
            receipt=read(OUTPUT/'native_export_receipts.json',{})
            saved=sum(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in receipt.get('samples',{}).values()) if receipt.get('trial')==winner else 0
            native_front.extend(['\n| 원본 RGB 선택 모델 처리 | 완료 |','|---|---|',
                f'| 추론 | {sum((f/"inference.json").exists() for f in folders)}/30 |',
                f'| MHR v2 변환 | {sum((f/"conversion_v2.json").exists() for f in folders)}/30 |',
                f'| source Uranus 저장·SHA 일치 | {saved}/30 |'])
        fixed_stage=read(OUTPUT/'fixed_training_status.json',{})
        if fixed_stage.get('status') not in [None,'waiting_for_native_wave_all30_and_frozen_input_preparation']:
            fixed_names=[('original_fixed_crop','diagnostic_original_static_crop')]+[(c['name'],c['name']) for c in fixed_plan.get('trials',[])]
            fixed_values={};fixed_counts={}
            verified_fixed_test=read(OUTPUT/'fixed_completed_Test_models.json',{}).get('models',{})
            for label,trial in fixed_names:
                values={}
                for sample in manifest['test']:
                    folder=OUTPUT/'predictions'/trial/sample;value=read(folder/'metrics.json',{})
                    if value.get('evaluation_protocol')!='native_rgb_static_crop_target_head_v1':continue
                    assert value['full_frame_roster'] and value['frames']==manifest['samples'][sample]['frames']
                    assert value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
                    assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                    values[sample]=value
                fixed_counts[label]=len(values)
                if len(values)==10 and trial in verified_fixed_test:
                    checked=verified_fixed_test[trial]
                    assert checked['sequences']==10 and checked['frames']==7610
                    fixed_values[label]={key:sum(v[key] for v in values.values())/10 for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
                    assert all(abs(fixed_values[label][key]-checked[key])<1e-9 for key in ['CD_H_cm','ACC_H_cm_per_frame2'])
            fixed_table={name:model_values[name] for name in ['init_v6','init_v0']};fixed_table.update(fixed_values)
            best_fixed={key:min(v[key] for v in fixed_table.values()) for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
            fixed_front=['\n## init_v6 / init_v0 비교 — 배경 보존 고정 crop Test10',
                '\n입력은 영상 전체 mask union을 포함하는 고정 RGB crop이며 배경을 지우지 않는다. K를 정확히 변환하고 출력 target head gate로 대상만 선택한다. 같은 Test10·7610프레임·MHR-v2·시퀀스 동일 가중치다. crop은 전체 영상의 제공된 마스크를 사용하는 오프라인 입력 처리다. Test로 crop 규칙이나 checkpoint를 고르지 않았다. init NPZ는 재추론하지 않은 제공 결과다.',
                '\n| 모델 | 완료 Test | CD-H cm ↓ | ACC-H cm/frame² ↓ | init_v6 대비 CD-H 개선율 | 같은 crop 원본 대비 CD-H 개선율 |','|---|---|---|---|---|---|']
            for label in ['init_v6','init_v0']+[n for n,_ in fixed_names]:
                value=fixed_table.get(label);count=10 if value else fixed_counts.get(label,0)
                if value:
                    cd=f'{value["CD_H_cm"]:.4f}';acc=f'{value["ACC_H_cm_per_frame2"]:.4f}'
                    if value['CD_H_cm']==best_fixed['CD_H_cm']:cd='**'+cd+'**'
                    if value['ACC_H_cm_per_frame2']==best_fixed['ACC_H_cm_per_frame2']:acc='**'+acc+'**'
                    gain=100*(fixed_table['init_v6']['CD_H_cm']-value['CD_H_cm'])/fixed_table['init_v6']['CD_H_cm']
                    baseline=fixed_values.get('original_fixed_crop');weight_gain=f'{100*(baseline["CD_H_cm"]-value["CD_H_cm"])/baseline["CD_H_cm"]:+.2f}%' if baseline else '대기'
                    fixed_front.append(f'| {label} | {count}/10 | {cd} | {acc} | {gain:+.2f}% | {weight_gain} |')
                else:fixed_front.append(f'| {label} | {count}/10 | — | — | 대기 | 대기 |')
            fixed_final=read(OUTPUT/'fixed_finalization_status.json',{})
            fixed_front.append('\n각 모델의 Test10·7610프레임이 모두 완료되면 보고서 갱신 전에 CPU 독립 검사를 수행한다. 실제 선택 checkpoint SHA·target mask·예측·GT SHA·crop box와 입력 조건·공통 metric을 검증한 평균만 표시한다. step0 원본과 실제 Train20 재학습은 구분한다. 10/10이어도 독립 검증 반영 전에는 수치를 보류할 수 있다. `fixed_completed_Test_models.json`.')
            completed_fixed_refit=read(OUTPUT/'completed_fixed_last2_refit_audit.json',{})
            fixed_parameter_audits=read(OUTPUT/'completed_fixed_refit_parameter_audits.json',{})
            if fixed_parameter_audits.get('status')=='passed':
                fixed_front.extend(['\n고정 crop 선택 checkpoint 및 전체 Train20 재학습 독립 검증:',
                    '\n| 후보 | Train20 성공 update | 원본과 달라진 저장 tensor | search+refit 분 | 선택 결과 |',
                    '|---|---:|---:|---:|---|'])
                for name,value in fixed_parameter_audits['models'].items():
                    label='Train20 재학습' if value['full_Train20_refit_performed'] else '원본 step0 선택; 미세조정 개선 아님'
                    fixed_front.append(f'| {name} | {value["successful_refit_updates"]} | {value["changed_parameter_tensors"]}/{value["parameter_tensors"]} | {value["total_learning_seconds"]/60:.2f} | {label} |')
                fixed_front.append('\n실제 world size2·탐색 정규화3종·재학습에서 validation 선택 안 함·모든 저장 tensor 유한값·원본 가중치와의 실제 차이·checkpoint SHA를 확인했다. 각 후보의 전체 학습 비용은3시간 미만이다. head는57개 tensor 모두 원본과 같고 step0이어서 학습 효과로 계산하지 않는다. 성능 판정은 별도 전체 프레임 MHR 비교로 한다. `completed_fixed_refit_parameter_audits.json`.')
                repeatability=read(OUTPUT/'step0_conversion_repeatability_diagnostic.json',{})
                if repeatability.get('all_raw_prediction_npz_byte_identical'):
                    value=repeatability['summary']
                    max_clip_delta=max(abs(v['CD_H_difference_cm']) for v in repeatability['per_sequence'].values())
                    fixed_front.append(f'\nstep0 대조의 변환 일관성 점검: Train4 전체2714프레임의 변환 전 NPZ는 원본 crop과 byte 단위로 동일했다. 하지만 MHR fitting 결과는 달라 전체 평균 CD-H가 {value["original_CD_H_cm"]:.4f}→{value["step0_CD_H_cm"]:.4f}cm, ACC-H가 {value["original_ACC_H"]:.4f}→{value["step0_ACC_H"]:.4f}였다. 영상별 CD-H 차이의 최대 절댓값은 {max_clip_delta:.4f}cm다. 이 차이는 네트워크 개선이 아니라 변환 fitting 단계에서 생겼다. 같은 수학적 변환 설정을 확인했지만 정확한 수치적 원인은 반복 GPU 재실행으로 분리하지 않았으므로 특정 RNG·GPU 원인이나 통계적 오차 한계로 단정하지 않는다. 큰 원본 오차는 변환 전에도 존재한다는 별도 진단과 구분한다. `step0_conversion_repeatability_diagnostic.json`.')
                    numeric_ops=read(OUTPUT/'conversion_numeric_operator_inspection.json',{})
                    if numeric_ops:
                        fixed_front.append('\n수치적 원인 후보를 추가 확인했다: 실제 MHR TorchScript에 `aten::index_add`와 sparse tensor 연산이 포함돼 있고 설치 버전은 PyTorch2.3.0+CUDA12.1이다. [같은 PyTorch2.3의 재현성 문서](https://docs.pytorch.org/docs/2.3/notes/randomness.html)는 CUDA indexed addition 등의 비결정성과 결정적 실행 검증 방법을 설명한다. 따라서 합산 순서에 따른 수치 차이는 근거 있는 가설이지만, 이번 fitting 차이의 원인으로 분리 입증한 것은 아니다. MHR 모델 SHA와 변환 코드 SHA·수정시각도 기록했다. 현재 비교 중인 converter 설정은 바꾸지 않는다. `conversion_numeric_operator_inspection.json`.')
            elif completed_fixed_refit.get('status')=='passed':
                fixed_front.append(f'\nlast2의 전체 Train20 재학습 완료를 독립 확인했다: 실제 {completed_fixed_refit["world_size"]}-GPU, 성공 update {completed_fixed_refit["successful_steps"]}회, search+refit 합계 {completed_fixed_refit["total_training_seconds"]/60:.2f}분. 정규화3종·refit에서 validation 선택 안 함·저장 파라미터 {completed_fixed_refit["parameter_tensors"]}개 tensor의 유한값과 checkpoint SHA를 확인했다. 성능 개선 판정은 전체 프레임 MHR 비교 이후다. `completed_fixed_last2_refit_audit.json`.')
            temporal=read(OUTPUT/'raw_temporal_stability_diagnostic.json',{})
            if temporal.get('models'):
                fixed_front.extend(['\n변환 전 예측의 시간적 흔들림 진단 — Test10 전체7610프레임:',
                    '\n| 입력·가중치 | raw body22 가속도 크기 cm/frame² ↓ |', '|---|---:|'])
                for name,value in temporal['models'].items():
                    fixed_front.append(f'| {name} | {value["equal_sequence_mean_raw_acceleration_magnitude"]:.4f} |')
                basket=temporal['models']['diagnostic_original_static_crop']['per_sequence']['blue_easter_basket-place-10']['largest_root_steps'][0]
                roots=basket['raw_root_camera_m']
                fixed_front.append(f'\nGT를 읽지 않고 raw 관절의 연속 원본 프레임 2차 차분 크기를 계산한 진단이며 ACC-H 오차나 정렬 후 지표가 아니다. basket crop 원본의 frame{basket["frames"][0]}→{basket["frames"][1]}에서 raw pelvis Z가 {roots[0][2]:.3f}→{roots[1][2]:.3f}m로 급변했다. 두 프레임 모두 검출 성공: {basket["both_detected"]}. 따라서 시간적 급변은 MHR 변환 전에 이미 존재한다. 급변만으로 인물 전환을 증명하거나 변환 오차가 없다고 단정하지 않는다. Test 사후 진단으로 모델·설정을 선택하지 않는다. `raw_temporal_stability_diagnostic.json`.')
                jump_review=read(OUTPUT/'target_foreground_visuals/diagnostic_original_static_crop/frames_898_899_900/receipt.json',{})
                if jump_review and jump_review.get('manual_review_pending') is False:
                    fixed_front.append('\n해당 basket898·899·900프레임의 RGB/raw/MHR를 직접 확인했다. 배우의 영상상 위치는 거의 같지만899부터 raw 골격의 크기·형태가 급변하고 MHR도 이를 따른다. 이3표본에서는 착석한 배경 인물로 넘어간 모습은 없었다. target mask 안의 잘못된 중복 head query 가능성까지 배제한 것은 아니다. 별도로 Test10×최초/중간/마지막30표본도 직접 확인했으며 전 프레임 수동 검증은 아니다. 근거: `target_foreground_visuals/diagnostic_original_static_crop/receipt.json`, `frames_898_899_900/receipt.json`.')
            baseline_audit=read(OUTPUT/'fixed_baseline_Test_independent_audit.json',{})
            native_original=native_values.get('original_RGB',{})
            crop_original=fixed_values.get('original_fixed_crop')
            if baseline_audit.get('status')=='passed' and crop_original and native_original:
                cd_gain=100*(native_original['CD_H_cm']-crop_original['CD_H_cm'])/native_original['CD_H_cm']
                acc_increase=100*(crop_original['ACC_H_cm_per_frame2']-native_original['ACC_H_cm_per_frame2'])/native_original['ACC_H_cm_per_frame2']
                fixed_front.append(f'\n원본 가중치 입력 대조의 Test10 전체 결과: 전체 RGB → 고정 crop의 CD-H는 {native_original["CD_H_cm"]:.4f} → {crop_original["CD_H_cm"]:.4f}cm({cd_gain:.2f}% 감소), ACC-H는 {native_original["ACC_H_cm_per_frame2"]:.4f} → {crop_original["ACC_H_cm_per_frame2"]:.4f}cm/frame²({acc_increase:.2f}% 증가)다. 깊이·궤적 민감성과 일치하지만 crop은 화각·인물 해상도·문맥을 함께 바꾸므로 화각만의 인과 효과로 단정하지 않는다. CD 개선이 시간적 안정성 개선을 뜻하지 않는다. 동일 원본 가중치·배경 보존·target gate·전체7610프레임·GT/예측 SHA·공통 채점 protocol을 독립 검증했다. 이 Test 사후 대조로 checkpoint나 학습 설정을 선택하지 않는다. `fixed_baseline_Test_independent_audit.json`.')
            fixed_front.append(f'\n고정 crop 학습·refit: `{fixed_stage.get("status")}`; Train MHR 선택·Test·최종 저장: `{fixed_final.get("status","대기")}`. 같은 crop 원본과의 비교는 미세조정 효과이며 전체 RGB 원본과의 비교에는 입력 변경 효과도 포함된다.')
            fixed_backfill=read(OUTPUT/'fixed_baseline_backfill_status.json',{})
            proxy_backfill=read(OUTPUT/'fixed_proxy_backfill_status.json',{})
            for validation_prefix in ['fixed_validation_backfill','fixed_validation_backfill_head','fixed_validation_backfill_last2']:
                validation_backfill=read(OUTPUT/(validation_prefix+'_status.json'),{})
                if validation_backfill:
                    fixed_front.append(f'\n전체 프레임 Train-MHR 평가 병행({validation_backfill.get("trial","준비")}): `{validation_backfill["status"]}`. 완료된 Train16/4 checkpoint 선택을 그대로 사용해 Train 검증2영상만 유휴 GPU 쌍에서 기존 stage lock·MHR-v2 변환·채점 경로로 처리한다. Test나 source 저장을 바꾸지 않으며 최종4영상 평균 집계·모델 선택은 기존 실행기에 맡긴다. `{validation_prefix}_jobs.json`, `{validation_prefix}_status.json`.')
            if proxy_backfill:
                fixed_front.append(f'\n완료 checkpoint 검증 병행: `{proxy_backfill["status"]}`. 모든 fixed search 종료를 확인한 후 유휴 GPU 쌍의 lease·NVML 비점유를 확인해 best·last 검증을 병렬 처리한다. 기존 실행기와 같은 validation tag·stage lock·입력·metric을 사용하고 후보 선택은 기존 실행기에 맡긴다. 새 학습 trial이나 Test 기반 선택은 추가하지 않는다. `fixed_proxy_backfill_jobs.json`, `fixed_proxy_backfill_status.json`.')
            if fixed_backfill:
                fixed_front.append(f'\n유휴 GPU 쌍의 원본 crop 평가 보충: `{fixed_backfill["status"]}`, 완료 영상 {len(fixed_backfill.get("completed",[]))}개. 세 search가 모두 배정된 뒤 pair4·5 lease와 NVML 비점유를 확인하고 기존 고정 Test 중 최대4개만 동일 stage lock·입력·MHR-v2로 처리한다. 재학습 요청이 생기면 새 배정을 멈춘다. 학습 설정·선택·Test roster·source 파일을 바꾸지 않는다. `fixed_baseline_backfill_jobs.json`, `fixed_baseline_backfill_status.json`, `fixed_baseline_backfill_plan.json`(실제 배정 후).')
            fixed_training_rows=[]
            for config in fixed_plan.get('trials',[]):
                run=OUTPUT/'runs'/config['name']
                fixed_training_rows.append((config['name'],read(run/'baseline.json',{}),
                    read(run/'best.json',{}),read(run/'complete.json',{}) or read(run/'progress.json',{})))
            proxy_values=[b['cd_h_smplx_proxy_cm'] for _,_,b,_ in fixed_training_rows if b.get('cd_h_smplx_proxy_cm') is not None]
            coverage_values=[b['coverage'] for _,_,b,_ in fixed_training_rows if b.get('coverage') is not None]
            fixed_front.extend(['\n고정 crop 학습 내부 검증 — fitted SMPL-X GT 공간 CD proxy이며 위 Test MHR CD-H와 다른 지표:',
                '\n| 후보 | 성공 update / 시도 | 학습 전 proxy cm ↓ | best proxy cm ↓ | best step | best coverage ↑ |',
                '|---|---:|---:|---:|---:|---:|'])
            for label,baseline,best,progress in fixed_training_rows:
                success=f'{progress["successful_steps"]} / {progress["step"]}' if 'successful_steps' in progress else '대기'
                baseline_cd=f'{baseline["cd_h_smplx_proxy_cm"]:.4f}' if baseline.get('cd_h_smplx_proxy_cm') is not None else '대기'
                best_cd=f'{best["cd_h_smplx_proxy_cm"]:.4f}' if best.get('cd_h_smplx_proxy_cm') is not None else '대기'
                coverage=f'{100*best["coverage"]:.2f}%' if best.get('coverage') is not None else '대기'
                if proxy_values and best.get('cd_h_smplx_proxy_cm')==min(proxy_values):best_cd='**'+best_cd+'**'
                if coverage_values and best.get('coverage')==max(coverage_values):coverage='**'+coverage+'**'
                fixed_front.append(f'| {label} | {success} | {baseline_cd} | {best_cd} | {best.get("step","대기")} | {coverage} |')
            fixed_front.append('\nTrain 내부 validation4개·고정64프레임씩 총256표본의 검출 기반 proxy다. unmatched 표본은 proxy에서 제외될 수 있어 coverage를 함께 표시하며, best step0은 원본 초기화이므로 미세조정 개선으로 보고하지 않는다. 굵은 값은 현재 기록이 있는 후보 사이의 최솟값/최대 coverage다. 전체 프레임 MHR 검증·Test 개선·init_v6 초과 달성을 뜻하지 않는다. 자동 보고서는10분 간격으로 이 표를 갱신하고 수동 학습 진행률 점검은30분 간격을 유지한다. `runs/fixed_*/baseline.json`, `best.json`, `progress.json`.')
            final_choice=read(OUTPUT/'final_selected.json',{})
            if final_choice:
                winner=final_choice['trial'];value=final_choice['models'][winner]
                scope='전체 RGB5개·고정 crop4개'+('·native MHR 타깃3개' if len(final_choice['models'])==12 else '')
                fixed_front.append(f'\n{scope}를 동일 Train4 MHR 기준으로 비교한 선택은 `{winner}`다. Train CD-H {value["CD_H_cm"]:.4f}cm / ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame². 입력 조건 `{final_choice["evaluation_protocol"]}`. Test 점수로 선택하지 않았다. `final_selected.json`.')
                final_receipts=read(OUTPUT/'final_export_receipts.json',{})
                saved=sum(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in final_receipts.get('samples',{}).values()) if final_receipts.get('trial')==winner else 0
                fixed_front.append(f'\n현재 최종 선택 모델의 source Uranus 저장·receipt SHA 일치: {saved}/30개. `final_export_receipts.json`.')
            native_front=fixed_front+native_front
        surface_plan=read(OUTPUT/'native_surface_adaptation_plan.json',{})
        surface_state=read(OUTPUT/'native_surface_training_status.json',{})
        surface_active=(surface_state.get('status') in ['training_native_surface_trials','all_native_surface_searches_and_Train20_refits_complete']
            or any((OUTPUT/'runs'/c['name']/'started.json').exists() for c in surface_plan.get('trials',[])))
        if surface_active:
            surface_names=[c['name'] for c in surface_plan['trials']]
            surface_checked=read(OUTPUT/'native_surface_completed_Test_models.json',{}).get('models',{})
            surface_values={n:surface_checked[n] for n in surface_names if n in surface_checked}
            table={n:model_values[n] for n in ['init_v6','init_v0']}
            previous=read(OUTPUT/'native_surface_prior_selected.json',{})
            previous_test=read(OUTPUT/'native_surface_prior_test.json',{}).get('models',{})
            if previous.get('trial') in previous_test:table[previous['trial']]=previous_test[previous['trial']]
            table.update(surface_values)
            best={k:min(v[k] for v in table.values()) for k in ['CD_H_cm','ACC_H_cm_per_frame2']}
            surface_front=['\n## init_v6 / init_v0 비교 — 정확한 MHR 표면 타깃 학습 Test10',
                '\n학습 CD의 표면 타깃을 native MHR GT로 바꾼 추가3조건이다. SMPL-X 공간에서 추론한 뒤 동일 MHR-v2 변환·공통 CD-H/ACC-H로 Test10 전체7610프레임을 평가한다. Test 타깃은 학습에 사용하지 않는다. 이전9개에서 Train 기준으로 선택한 모델을 비교행에 함께 표시하며, 이전9개 전체 Test 표는 아래에 유지한다. 굵은 값은 이 표에 완료된 행 사이의 최솟값이다.',
                '\n| 모델 | 완료 Test | CD-H cm ↓ | ACC-H cm/frame² ↓ | init_v6 대비 CD-H 개선율 |','|---|---|---|---|---|']
            for name in list(table)+[n for n in surface_names if n not in table]:
                if name in table:
                    value=table[name];cells=[]
                    for key in ['CD_H_cm','ACC_H_cm_per_frame2']:
                        text=f'{value[key]:.4f}';cells.append('**'+text+'**' if value[key]==best[key] else text)
                    gain=100*(model_values['init_v6']['CD_H_cm']-value['CD_H_cm'])/model_values['init_v6']['CD_H_cm']
                    surface_front.append(f'| {name} | 10/10 | {cells[0]} | {cells[1]} | {gain:+.2f}% |')
                else:
                    count=sum((OUTPUT/'predictions'/name/s/'metrics.json').exists() for s in manifest['test'])
                    surface_front.append(f'| {name} | {count}/10 | — | — | 대기 |')
            surface_final=read(OUTPUT/'native_surface_finalization_status.json',{})
            surface_front.append(f'\n학습 상태 `{surface_state.get("status")}`; 12개 전체 비교·Test·저장 상태 `{surface_final.get("status","대기")}`. 각 신규 평균은 전체 Test10이 끝난 후 checkpoint·예측·GT·mask SHA와 공통 입력/채점을 독립 검증해야 표시한다. `native_surface_completed_Test_models.json`.')
            surface_choice=read(OUTPUT/'native_surface_selected.json',{})
            if surface_choice:
                selected=surface_choice['trial'];value=surface_choice['models'][selected]
                surface_front.append(f'\n전체12개 Train4 MHR 선택: `{selected}`, CD-H {value["CD_H_cm"]:.4f}cm / ACC-H {value["ACC_H_cm_per_frame2"]:.4f}cm/frame². 새 Test 추론 시작 전에 선택을 고정하며 재개 시 최초 선택시각을 보존한다. `native_surface_selected.json`.')
            surface_mapping=read(OUTPUT/'native_surface_refit_mappings.json',{})
            if surface_mapping:
                surface_front.extend(['\n실제 탐색·전체 Train20 재학습 기록:',
                    '\n| 후보 | Train20 재학습 | 성공 update | 전체 학습 분 |','|---|---|---:|---:|'])
                for name,value in surface_mapping.items():
                    refit=value['refit'];performed=refit['refit_performed_this_wave']
                    updates=refit['complete']['successful_steps'] if performed else 0
                    label='수행' if performed else '원본 step0 선택; 개선 아님'
                    minutes=(value['search_seconds']+value['refit_seconds']+value.get('training_lineage_seconds',0))/60
                    surface_front.append(f'| {name} | {label} | {updates} | {minutes:.2f} |')
            surface_validation=read(OUTPUT/'native_surface_validation_mhr_progress.json',{}).get('models',{})
            if surface_validation:
                surface_front.extend(['\n추가3조건의 Train4 전체 MHR 완료분:',
                    '\n| 모델 | Train CD-H cm ↓ | Train ACC-H cm/frame² ↓ |','|---|---:|---:|'])
                surface_minima={key:min(v[key] for v in surface_validation.values()) for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
                for name,value in surface_validation.items():
                    assert value['sequences']==4 and value['frames']==2714 and set(value['per_sequence'])==set(manifest['validation'])
                    cells=[('**'+f'{value[key]:.4f}'+'**') if value[key]==surface_minima[key] else f'{value[key]:.4f}' for key in ['CD_H_cm','ACC_H_cm_per_frame2']]
                    surface_front.append(f'| {name} | {cells[0]} | {cells[1]} |')
                surface_front.append('\n네 영상 전체가 완료된 search checkpoint의 평가이며 Test 결과나 Train20 refit의 성능으로 해석하지 않는다. 내부 native MHR-GT proxy256표본과도 구분한다.')
            surface_exports=read(OUTPUT/'native_surface_export_receipts.json',{})
            if surface_exports:
                saved=sum(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in surface_exports['samples'].values())
                surface_front.append(f'\n후속 선택 모델의 실제 source Uranus·SHA 일치: {saved}/30개. 전체 비교 후 순차 저장하고 global 결과는30개 검증 뒤 갱신한다. `native_surface_export_receipts.json`.')
            native_front=surface_front+native_front
        front[0]='\n## 이전 배경 제거 조건 — init_v6 / init_v0 비교'
        front=native_front+front
    lines[2:2]=front
    write(REPORT,'\n'.join(lines)+'\n')
    research=[f'# Track2 Multi-HMR 연구 기록',f'\n갱신: {now}',
        '\n## 요청과 고정 조건',
        '\n기존 ETRI 실험의 일시정지 문서는 기법 참고다. 새 사용자 요청에 따라 이전 작업을 종료하고 '
        '독립 Track2 실험을 실행한다. 초기 세 Trial은 GPU1–6을 2개씩 동시 사용했다. GPU3 하드웨어 장애 이후 사용자가 정상 5개 사용을 승인했다. 현재 GPU0 보존, GPU1·2·4·5·6 UUID 지정, 후속 학습은 두 GPU 쌍 대기열로 수행한다. '
        '사용자가 연장한 전체 실험 시간과 Trial당 3시간 이내 한도를 따른다. 최종 MHR 산출물 이름은 `human_params_uranus.npz`이다.',
        '\n## 구현 및 근거',
        '\n- `src/multihmr/track2_data.py`: 고정 분할, 공식 topology map, MHR GT decode, camera 변환, SMPL-X surface fitting 및 RGB cache.',
        '- `scripts/train_multihmr.py`: 기존 DDP loss/validation/checkpoint 경로 재사용, 독립 실험 경로, L2-SP 및 조기 종료 추가.',
        '- `scripts/evaluate_multihmr.py`: RGB 검출 기반 추론, GT-free 단일 사람 추적, MHR 변환, 인간 지표 평가 진입점.',
        '- `src/utils/mhr_conversion.py`: 공식 V2D 세 anchor frame shared-identity fitting 후 독립 프레임별 미분 가능한 pose fitting. temporal smoothing 없음. 실행 검증 상태는 아래 smoke 결과로 구분한다.',
        '- `src/utils/v2d_official_metric.py`, `mhr_reference_fitting.py`, `mhr_evaluation_body.py`: 참조 소스에서 복사해 ChallengeHOR runtime 의존성을 제거했다. SHA256과 CD-H/ACC-H helper AST 동일성을 확인했다.',
        '- Meta mapping revision: `877cba5ee61a5f77d1457bf4de93a66695132bd9`. [공식 변환 문서](https://github.com/facebookresearch/MHR/blob/877cba5ee61a5f77d1457bf4de93a66695132bd9/tools/mhr_smpl_conversion/README.md).',
        '\n## 좌표계 검증과 수정',
        '\nGT raw NPZ는 `pose(T,136)`, `shape(45)`, `scales(68)`이며 metadata의 오래된 SOMA-X 표기와 다르다. '
        'MHR reference decoder로 복원한 world mesh를 camera 변환 없이 투영했을 때 RGB와 불일치했다. '
        'metadata의 world-to-camera를 적용한 뒤 관절 overlay가 사람과 일치함을 시각 확인했다. '
        '`smoke/fit_overlay.jpg`는 수정 전 진단이고 `smoke_camera/fit_overlay.jpg`는 수정 후 진단이다.',
        '\n기존 `human_params_init.npz`는 camera frame, `human_params_final.npz`는 human3r_first_camera이며 GT만 source world이다. '
        '따라서 최종 uranus NPZ는 `opencv_camera`를 유지하고 `intrinsics(T,4)`를 포함해 기존 native renderer와 호환한다. '
        'MHR fitting 전에 camera→world 변환을 추가하지 않는다. 공개 scorer의 원본 최초 프레임 고정 Sim(3)이 GT world와의 차이를 처리하며 '
        '이러한 전역 강체 변환에 대한 지표 불변성은 합성 수치 시험에서 검증했다.',
        '\nSMPL-X label fitting: full-surface MSE + 0.1 body22 joint MSE + 1e-7 shape prior, '
        '350 Adam iterations, batch내 shared betas, neutral/10 betas/flat_hand_mean=True. '
        '원본 MHR identity는 sequence shared지만 SMPL-X 변환 beta는 fitting batch간 달라질 수 있어 원본 recipe와의 차이로 공개한다.',
        '\n## 준비 결과',
        f'\n완료 {prepared}/20개.']
    if receipts:
        research.append(f'\n시퀀스별 평균 mesh fitting 오차 범위: {min(x["mean_fit_mm"] for x in receipts):.2f}–{max(x["mean_fit_mm"] for x in receipts):.2f} mm. '
            '이는 네트워크 예측 오차가 아닌 MHR→SMPL-X 표현 변환 잔차다.')
    if label_checks:
        research.append(f'\n추가 Train 라벨 검증: head 영상 내 비율의 시퀀스 최솟값 {label_checks["min_visible_head_fraction"]:.4f}, '
            f'프레임별 평균 fitting 오차 최댓값 {label_checks["max_fit_frame_error_mm"]:.2f} mm. '
            '전 프레임 양의 깊이, 유한한 joint/beta, 회전행렬 직교성 및 determinant +1을 확인했다. '
            '상세 수치: `label_quality_checks.json`.')
    if asset_checks:
        research.append('\nMHR official v1.0.1 asset SHA256 일치 확인: '
            '`'+asset_checks['assets']['mhr_model']['sha256']+'`. '
            '원본 Multi-HMR·neutral SMPL-X asset 해시도 `model_assets.json`에 보존했다.')
    research.extend(['\n## 전체 Train refit',
        '\n초기 실행이 보존한 PA-selected best 및 final last 후보를 4개 내부 검증 SMPL-X CD-H로 재평가해 '
        'checkpoint와 update 수를 선택한다. 과거 모든 checkpoint가 남아 있는 것은 아니므로 전체 학습 궤적을 CD-H로 선택했다고 주장하지 않는다. '
        '모델 설정은 4개 내부 검증 전체 프레임 MHR CD-H 우선, ACC-H 동률 판정으로 선택한다. '
        '세 Trial 각각의 선택된 update 수를 데이터 수 비율로 조정해 20개 Train 전체에 고정 예산 refit을 수행한다. '
        '원본에서 재시작하고, 초기 search 최대 2시간 + refit 최대 1시간(합계 3시간 이내)을 지킨다. '
        'refit 최종 체크포인트에는 이전 내부 검증 점수를 현재 모델의 측정치로 붙이지 않는다. '
        '검증이 step 0 초기화를 선택한 Trial은 원본 체크포인트를 유지하며 이를 명시한다.',
        '\n## CD-H 중심 후속 실험',
        '\n초기 세 Trial은 모두 20 epoch / 1,061회 성공 update로 끝났다. PA-MPJPE와 최초 프레임 고정 CD-H의 '
        '순위가 달라 Head-only 초기화에서 절대 pelvis 위치 loss와 방향 보존 root-relative body loss를 강화한 '
        '3개 후보를 미세조정 모델의 Test 추론 전에 고정했다. 새 학습은 두 DDP rank가 각각 검증 표본 절반의 '
        'SMPL-X 표면점을 계산한 뒤 gather하여 원본 frame0 고정 CD-H를 구한다. 10분마다 측정하고 이를 checkpoint 선택 기준으로 사용한다. '
        '검증에도 각 Trial의 두 GPU를 모두 사용하며 추가 모델을 별도 로딩하지 않는다. '
        '초기 공통 bootstrap 시간 + 후속 학습 + 전체 Train refit 시간 합계가 3시간 미만인지 확인한다. '
        '후속 최종 모델 역시 Train 내부 MHR CD-H로 선택하며 Test 점수는 선택에 사용하지 않는다.',
        '\n후속 상태:\n```json\n'+json.dumps(wave2_state,indent=2,ensure_ascii=False)+'\n```',
        '\n## 평가 smoke 검증',
        '\n```json\n'+json.dumps(read(OUTPUT/'evaluation_smoke.json',{'status':'아직 실행 전'}),indent=2,ensure_ascii=False)+'\n```',
        '\n## 고정 분할','\n```json\n'+json.dumps({k:manifest.get(k,[]) for k in ['train','training','validation','test']},indent=2)+'\n```',
        '\n## 현재 실행 및 오류','\n```json\n'+json.dumps(dict(supervisor=supervisor,error=error),indent=2,ensure_ascii=False)+'\n```',
        '\n## 남은 검증',
        '\n학습의 실제 optimizer 성공 update, checkpoint/검증 step 일치, 세 Trial 원본 대비 동일 Test 점수, '
        '추론 중 GT 비사용, MHR 변환 잔차, 30개 export의 shape/유한값/전체 frame수/provenance, 원본 GT 해시 보존을 확인한다. '
        '완료 증거가 없는 항목은 완료로 취급하지 않는다.'])
    research.extend(['\n## GPU 장애 후 실행', '\nGPU3 PCI61:00.0 Xid79로 CUDA ordinal이 재배치되어 숫자 mask 실행이 실패했다. 기존 CPU 실행기를 종료하고 사용자가 승인한 GPU1·2·4·5·6 UUID로 다시 시작했다. 후속 2 GPU 학습에는 파일 잠금으로 쌍을 배정해 동시 중복을 막는다. NCCL도 nvmlDeviceGetHandleByIndex(3) 오류로 실패해 Gloo로 전환했다. 정상 GPU1·2의 CUDA all-reduce, 3회 DDP gradient update 및 all_gather_object를 통과했다. gloo_healthy_pair_check.json에 기록했으며 실제 Multi-HMR pilot은 별도 통과해야 한다.', '\n## 내부 검증 영상 점검', '\n4개 검증 영상의 frame0/중간/마지막 12프레임을 직접 확인했다. 표본상 사람 전환·좌표축 오류는 보이지 않았다. 예를 들어 black_platform Last2의 깊이 오차는 +0.307/−0.015/+0.133m로 변해 frame0 고정 정렬 후 궤적 오차가 남는다. 전체 프레임 검증으로 확대 해석하지 않는다. 원본 수치와 overlay: `validation_visuals/receipt.json`.'])
    research.append('\n실제 Multi-HMR 통신 pilot 상태:\n```json\n'+json.dumps(hardware_pilot,indent=2,ensure_ascii=False)+'\n```')
    if conversion_means:
        research.append(f'\nMHR prediction 변환 품질: 완료 {len(conversion_means)}건 평균 vertex 잔차 {min(conversion_means):.2f}–{max(conversion_means):.2f} mm. 3개 anchor 사이의 pose 보간 초기화 후 Adam160회는 일부 큰 회전 구간에서 높은 잔차를 남긴다. 모든 모델에 동일 변환을 적용했지만 변환 자체가 결과에 미치는 영향은 남아 있다. GT 기반 보정은 하지 않았다. 학습 라벨 변환 잔차와 구분한다.')
    research.extend(['\n## 고정 Test 최신 완료 결과', '\n각 모델의 10개 전체 평가가 완료된 경우에만 평균을 게시한다. Test는 선택에 사용하지 않는다.', '\n```json\n'+json.dumps({name:{key:value[key] for key in ['CD_H_cm','ACC_H_cm_per_frame2']} for name,value in model_values.items()},indent=2,ensure_ascii=False)+'\n```', '\n전체 Train refit 체크포인트 무결성: `refit_checkpoint_integrity.json`. 학습 성공 update와 가중치 변경, 유한값, Train/Test 분리 및 GT30개 해시를 CPU에서 검증했다.'])
    if wave3_plan:
        research.extend(['\n## Paired CD-H 후속 탐색', '\n동일 시퀀스 frame0/current RGB 쌍에 공식식 FPS27 Sim(3), FPS64 body Chamfer(양방향 합, half 없음)를 직접 loss로 준다. 초기 pilot의 SVD/scale/rotation detach를 수정했다. anchor와 채점 time frame의 목적함수가 다르므로 정렬 회전·scale·translation 전체를 미분한다. 별도 CPU 공식 수식 값 비교, anchor 회전/scale/translation 12방향 수치 미분, autograd gradcheck를 통과했다(`paired_cd_full_gradient_checks.json`). 대상 foreground 입력을 사용한 새 실제 2-GPU pilot은20/20 update 성공, AMP128에서 skip0이다. 입력1107개 존재 및20개 시퀀스의 독립 mask/RGB 검사에서 target RGB·라벨·K 보존과 배경 제거도 확인했다(`target_training_input_checks.json`). 초기 fullRGB/detached pilot18/20과 구분한다.', '\n초기 Test 집계를 본 뒤 추가한 탐색이므로 pre-Test 고정 실험이라고 주장하지 않는다. 선택은 Train-only 4개 validation CD-H, global12h deadline과 후보당3h 한도를 그대로 적용한다.', '\n상태:\n```json\n'+json.dumps(wave3_state,indent=2,ensure_ascii=False)+'\n```'])
    if conversion_diagnostic:
        research.append('\nGT-free 변환 진단: `conversion_metric_diagnostic.json`. 4 Train-validation clips에서 균일32+잔차최대5 표본을 비교했다. 표본 worst-frame 편향이 있으며 full-frame 또는 GT 예측 정확도 수치로 사용할 수 없다. 현재 변환의 시간별 fitting 오차와 frame0 정렬 오차가 남으므로 별도 Train-only 600회 추가 Adam 진단을 수행한다. 지표는 비선형이므로 이 수치를 최종 CD-H에서 빼지 않는다.')
    if refinement_trials:
        research.append('\n변환 수렴 최적화 실험 결과: `conversion_refinement_diagnostic.json`(Adam600), `conversion_lbfgs_diagnostic.json`(Adam600 후 LBFGS200), `conversion_lm_diagnostic.json`(기존 v1 pose에서 reference LM20). 동일 Train13표본이며 운영 파이프라인에 미적용이다. 추가 최적화만으로 잔차가 줄어 기존 프레임별 Adam160 fitting이 충분히 수렴하지 않았다는 증거다. 모델 추정 오차와 분리해야 하며 모든 모델을 같은 새 변환으로 재평가하기 전 결과를 섞지 않는다.')
    if v2_pilot:
        research.extend(['\n## 공통 변환 revision2', '\nTrain 진단을 근거로 LM10 / residual20mm 임계값을 고정했다. 모델 선택/optimizer 학습 설정과 독립적인 변환 수렴 개선이다. 해당 v2 Test 점수를 보기 전에 설정을 고정했고, 원본 MHR NPZ는 건드리지 않는다. v2 파일과 점수는 `_v2` 접미사로 분리한다. 최종 선택은 모든 후보의 동일4개 Train-validation v2 CD-H로 다시 고르고, Test10 전체 및 export30개 검증 후에만 primary JSON을 교체한다.', '\n상태:\n```json\n'+json.dumps(dict(pilot=v2_pilot,prefetch=v2_prefetch,final=v2_final,independent_CPU_audit=v2_audit),indent=2,ensure_ascii=False)+'\n```'])
    if v2_train_score:
        research.append('\n동일 Train700프레임 v1/v2 metric smoke:\n```json\n'+json.dumps(v2_train_score,indent=2,ensure_ascii=False)+'\n```\n변환은 표면 MSE 수렴을 목표로 하며 GT metric을 loss로 쓰지 않는다. 해당 표본의 CD-H는 낮아졌지만 ACC-H는 증가했다. 기존 느슨한 fitting이 입력 SMPL-X의 프레임별 변동을 충분히 따라가지 못했던 영향도 있어, 수렴 개선을 시간 지표 개선으로 주장하지 않는다.')
    if concurrent:
        research.append('\n## 학습·평가 병행 스케줄\n\n사용자의 항상 학습·평가 병행 지시에 따라 후속 초기 학습은 이전 평가/export 종료를 기다리지 않는다. 두 DDP 후보는 GPU1·2 /4·5 pair lease로 실행하고 기존 MHR 평가와 VRAM을 공유한다. 후속 wave3의 target foreground 후보도 실제 pilot 검증 후 현재 평가와 선행 학습하며, 전역 checkpoint mapping·선택·export 변경은 이전 wave 종료 후 순차 처리해 metadata 경합을 막는다. 평가는 정상5개 GPU worker에서 실행하고 DDP 학습과 VRAM을 공유한다. 활성 GPU 작업은 종료하지 않았고 대기 중 CPU 관리자만 교체했다.')
    if outlier_diagnostic and not evaluation_validity:
        research.append('\n## 초기 v1 Test 분포 진단\n\n원본 Test 평균 CD-H70.392cm, 시퀀스 중앙값42.781cm, 최대233.077cm. 가장 큰 두 영상이 CD 합의52.79%를 차지한다. Head 모델은 평균20.211cm, 중앙값19.462cm로 평균 감소가 두 outlier에만 한정되지는 않는다. 단, v6 init12.426cm보다 여전히 크다. 공식 equal-sequence 평균과 Train-only 모델 선택을 그대로 유지한다. 개별 값: `test_outlier_diagnostic.json`.')
    if test_visual_review:
        research.append('\n## 무효화한 이전 unmasked association 점검\n\n현재 유효 점수의 근거가 아닌 이전 입력 방식의 역사적 진단이다. 3 Test 영상의9표본을 직접 확인했다. original의 blue_easter_basket frame490/979 및 toy_stroller frame0/473/945는 배경의 앉은 사람을 선택했다. Last2는 확인한9프레임에서 배우에 남았다. 최초 프레임 오선택은 고정 Sim(3) 기준 대상 자체를 바꾸고, 이후 오선택은 actor 궤적과 배경 person 궤적을 비교하게 만든다. 원본 대비 점수 개선은 검출·대상 선택과 pose/깊이 개선이 함께 반영된 end-to-end 결과다. tracker나 학습 후보 설정을 Test를 보고 수정하지 않았다. 특정 표본의 확인이며 전체 빈도로 확대하지 않는다. `test_visuals/receipt.json` 및 JPG3개에 보존했다.')
    if available_exports and not exports.get('samples'):
        research.append(f'\n## 완료된 Test 산출물 우선 저장\n\nTrain 내부4개 MHR CD-H 최솟값으로 선택한 `{available_exports.get("trial")}` 전체Train20 refit의 Test {len(available_exports.get("samples",{}))}개 native MHR 출력을 source의 `human/human_params_uranus.npz`에 저장했다. `target_train_validation_completed.json`의 실제 argmin과 checkpoint SHA를 확인했고, 각 영상의 target foreground/head 입력·revision2 변환·GT/prediction hash·전체 frame roster를 확인한 뒤 저장했다. GT/init은 변경하지 않았다. Test로 모델을 선택하지 않았다. 현재 저장본은 전체30개 및 추가 후보 검증이 끝나면 최종 선택 모델로 갱신한다. `available_export_receipts.json`, `available_export_independent_audit.json`.')
    if evaluation_validity:
        research.append('\n## 사용자 지시: target human만 평가\n\n이전 largest-person/continuity 결과와 body-overlap-only 결과를 모두 무효화했다. 특히 original44.2949cm도 배경 머리 검출점이 마스크 밖인데 몸통 겹침으로 통과한 결과여서 유효 점수가 아니다. 이전 우리 export10개는 보존 후 source에서 제거했고 GT/init은 변경하지 않았다. 현재 `target_foreground_head_v1`은 마스크 밖 RGB를 [124,116,104]로 제거하고 native 화면·K를 유지한다. head query가 대상 마스크(15px dilation) 안인 후보 중 최고 confidence만 허용한다. body overlap은 진단값이며 대상 pose 정확도에 따른 선택 편향을 막기 위해 순위/탈락 기준으로 쓰지 않는다. 미검출은 같은 target의 관측값으로만 보완하고 전체 프레임을 채점한다. 모든 후보에 동일 입력·MHR revision2를 적용한다.')
    if evaluation_validity:
        research.append('\n공통 입력 검증: 30개 clip의 최초/중간/마지막90표본 마스크를 시각 확인했다. 독립 CPU 검사로 대상 RGB 보존·비대상 RGB 제거·원본 K 변환 보존을 확인했다. toy_stroller 중간473의 실제 원본 모델 preflight는 대상 배우 한 명만 검출했고 head mask 통과, confidence0.712였다. 근거 `target_native_input_checks.json`, `target_candidate_diagnostic.json`, `target_foreground_input.jpg`. 전체 프레임 사람이 직접 검토한 것은 아니다. Train20의 후속 학습/검증용 foreground cache1107프레임도 준비했다. 이미 실행 중이던 학습은 기존 fullRGB 입력이며 후속 실행부터 target_foreground 입력을 사용한다.')
    if target_geometry.get('summary') and target_geometry.get('evaluation_protocol')=='target_foreground_head_v1':
        summary=target_geometry['summary']
        research.append(f'\n## Target-only 원본 CD-H의 독립 CPU 검증\n\nTest10 전체 프레임에서 공식 map으로 사상한 raw SMPL-X surface proxy {summary["raw_mapped_SMPLX_CD_proxy_cm"]:.4f}cm, native MHR v2 CD-H {summary["converted_MHR_CD_H_cm"]:.4f}cm, init_v6 CD-H {summary["init_v6_CD_H_cm"]:.4f}cm를 계산했다. 실제 MHR CD-H는 GPU 결과와0.02cm 이내로 CPU에서 재현했다. 두 prediction 공간은 각각 최초 프레임 Sim3를 fitting하므로 점수 차이를 가산 가능한 오차 성분으로 해석하지 않는다. GT는 진단 평가에만 사용했으며 모델/파라미터 선택에는 사용하지 않았다. 시퀀스별 root 궤적 표본: `target_geometry_diagnostic.json`.')
    if current_proxy_baseline.get('detection_selection')=='target_mask_head_highest_confidence':
        research.append(f'\n## 내부 checkpoint proxy의 대상 선택 통일\n\n후속 posthoc SMPL-X checkpoint 검증은 target foreground 입력 + head mask 통과 최고 confidence를 사용하며 GT pose로 검출을 선택하지 않는다. 원본의 Train validation4×64=256개 실제 검증을 완료했다: target 검출 {current_proxy_baseline["matched_instances"]}/{current_proxy_baseline["valid_instances"]}, SMPL-X proxy {current_proxy_baseline["cd_h_smplx_proxy_cm"]:.4f}cm. GT SMPL-X 재구성 최대 오차 {current_proxy_baseline["max_gt_reconstruction_error_m"]*1000:.4f}mm. proxy는 검출 프레임만 채점하며 원래 frame0 검출을 필수로 한다. 최종 MHR 점수는 미검출 보완을 포함한 모든 원본 프레임을 채점한다. 이전 fullRGB/GT-matched proxy 수치와 직접 비교하지 않는다. `validation_smplx/original/metrics.json`.')
    if evaluation_validity:
        research.append('\nTarget 검출 coverage 조건: 새 foreground/head protocol에서는 최소95%를 사용한다. 참조 RGB control의 minimum_coverage=.95를 따른다. 초기 fullRGB/GT-matched 기준의 원본 대비0.005 허용치는 대상 머리 query의 실제 검출률과 같은 통계량이 아니므로 새 input protocol에 재사용하지 않는다. Train 내부 256표본에서 원본100%, 초기 fine-tuned253/256(98.828%)를 관측한 뒤 Train-only protocol 선택으로 명시했다. Test로 정한 threshold가 아니며, 최종 MHR CD-H는 미검출을 같은 target의 nearest observation으로 보완한 전체 원본 프레임을 사용한다. `target_foreground_protocol.json`.')
    if foreground_visual_review:
        research.append('\n## 새 foreground 예측의 실제 영상 검토\n\n`target_foreground_visuals/`의 Test10×최초/중간/마지막30프레임 JPG를 직접 열어 확인했다. 대상 RGB 입력·init_v6·원본 raw SMPL-X·사용 가능한 MHRv2를 같은 camera에서 비교했다. 확인한30표본의 원본 예측은 target actor를 따랐고 배경 사람으로 바뀐 표본은 없었다. toy_stroller frame473의 잘못 길어진 다리는 foreground actor에 대한 pose 오차이며 뒤의 직원 선택과 다르다. 초기 standing frames의 raw pelvis Z가 GT보다 작은 경향, 일부 영상의 시간별 depth bias도 관찰했다. 전체 프레임 수동 검토가 아니며 원인이나 빈도는 전체 데이터 CPU 진단 없이 단정하지 않는다. 이 Test 사후 진단으로 checkpoint/recipe를 고르지 않는다. Watering의 MHRv2는 snapshot 당시 미완료였고 raw30표본은 모두 검토했다. 범위와 이미지 SHA는 `target_foreground_visuals/receipt.json`.')
    if followup_order.get('execution_mode')=='pooled_contenders_with_Train20_refit_prefetch':
        research.append('\n## 현재 실행: 후속6개 공동 비교와 Train20 refit 선행\n\n절대위치 후보를 현재 foreground/head 입력으로 재검증하니 약20.36–21.97cm proxy로 paired 후보와 비슷했다. 같은 Train 검증으로 비교하기 위해6개 후보를 한 단계에서 MHR 평가·선택한다. 각 후보의 best/last는 고정4개 Train SMPL-X proxy/coverage95%로 고르고 update 예산을 고정한 뒤, 전체Train20 refit을 초기 모델 Test 평가와 병행한다. Refitting은 validation으로 checkpoint를 고르지 않고 모든 Train20 데이터를 사용한다. 학습 입력은 각 search의 실제 configuration과 일치시킨다: absolute 후보는 fullRGB, paired 후보는 foreground다. 모든 모델의 평가 입력은 동일한 target foreground/head 조건이다. 대기 중인 absolute refit이 시작되기 전에 입력 일치를 확인했다(`refit_input_consistency_audit.json`). recipe 선택은 항상16-Train search checkpoint의 held-out4 MHR 점수로만 하며 refit 모델의 Train 점수로 고르지 않는다. 이미 시작된 초기 평가/학습 프로세스를 재시작하지 않았고 자식 없는 대기 CPU guardian만 교체했다. 모델 hyperparameter는 실제 완료된 search configuration과 일치함을 확인했다. Joint 실행계획은 search 완료 후 만들어졌으므로 pre-search 사전 등록이라고 주장하지 않는다. 각 후보 설정이 자기 학습 전에 고정됐다는 provenance와 Test 사후 탐색 한계를 구분한다. Global checkpoint mapping·선택·exports는 초기30개 저장 후 단일 finalizer에서 순차 처리한다. Prefetch는 별도 config/run/choice만 쓰고 동일 GPU pair lease를 사용하며, lease를 얻은 뒤 complete를 재확인해 중복 학습을 막는다. 초기 목표3개 비교는 유지하고 추가6개 Test를 공동 비교해 중복된 winner Train20 추론/export를 줄인다. 원래 마감/3h 한도/같은 metrics·변환을 유지한다. `target_pooled_followup_plan.json`, `target_refit_prefetch_status.json`, `refit_prefetch_choices/`, `target_followup_start.json`.')
    elif followup_order:
        research.append('\n## 남은 예산에서의 후속 실행 우선순위\n\nTarget foreground 입력과 full-anchor-Sim3 gradient를 적용한 paired 세 후보를 기존 absolute-loss 세 후보보다 먼저 처리하도록 후속 순서를 바꿨다. 현재 foreground online Train proxy가 약20–21cm까지 내려왔으나 다른 입력/선택 방식의 이전 proxy와 직접 성능 비교하지 않는다. 입력 일관성 및 전체Train refit·Test·export에 필요한 예산 확보를 위한 순서 변경이며 Test 성능으로 모델을 고르지 않는다. 모델 선택은 동일4개 Train MHR CD-H/ACC-H로 수행한다. 초기 목표의 세 Trial 비교/30개 export가 끝난 후 foreground 후보를 처리하고, absolute 후속 단계는3.5시간 이상 남았을 때만 진입한다. Background search 완료를 이유로 refit/Test/export 시간 검사를 생략하던 조건도 수정했다. live DDP와 초기 평가 프로세스는 유지하고 자식이 없는 대기 CPU guardian만 교체했다. 원래 실행계획은 `followup_plan_history/`, 변경 근거·원래 마감·실행기 PID는 `target_followup_order_revision.json`, `target_followup_guardian_start.json`에 보존했다.')
    if timing_profile.get('stages'):
        research.append('\n## 현재 target pipeline 실측 처리 시간\n\n`job_history.jsonl`의 현 foreground 실행 시작 이후 성공한 stage의 실제 시작/종료 시간을 모았다. 진행 중인 job은 평균에 넣지 않았고 stage마다 완료 모집단이 다르다. Training과5개 평가 worker가 GPU를 공유한 wall time이며 정확도에 따른 후보 선택에 쓰지 않는다. 이 처리 시간을 근거로 전체 검증·refit·Test·30개 저장에 필요한 후속 단계 예산을 확보했다. 변환 solver·iteration·sampling·acceptance는 그대로 유지한다. `target_pipeline_timing_profile.json`:\n```json\n'+json.dumps(timing_profile['stages'],indent=2,ensure_ascii=False)+'\n```')
    if mhr_validation.get('models'):
        research.append('\n## 완료된 Train 내부 MHR 검증\n\n현재 target foreground/head 조건에서 각 모델의 동일4개 Train-validation 전체 프레임을 평가했다. CD-H primary / ACC-H tie-break로 선택하며 Test 값은 선택에 사용하지 않는다. 초기 각 search checkpoint가 posthoc SMPL-X 기준으로 선택된 SHA와 같은지 확인했고 source GT·native MHR prediction hash·revision2·전체 frame roster도 확인했다. 초기16개 sequence/models가 끝난 시점에는 `target_train_validation_completed.json`으로 먼저 공개한다. 이후 후속 단계가 완료되면 공통 `validation_mhr_comparison.json`을 사용한다.\n```json\n'+json.dumps({n:{k:v[k] for k in ['CD_H_cm','ACC_H_cm_per_frame2']} for n,v in mhr_validation['models'].items()},indent=2,ensure_ascii=False)+'\n```\nCD-H 선택 모델: '+mhr_validation.get('selected_trial','미선택')+'. 이는 Test 또는 init 대비 개선의 증거가 아니며 각 지표를 별도로 비교한다.')
    if exported_visual_review:
        research.append('\n## 실제 저장 파일의 영상 투영 검토\n\n`target_foreground_visuals/head_strong_exported/`의10개 JPG를 직접 열어30개 frame0/mid/last 표본을 확인했다. raw SMPL-X와source의 실제 `human_params_uranus.npz`를 GT 및init과 같이 투영했고 저장본 SHA를 export receipt와 대조했다. 표본의 저장본은 대상 배우를 유지했고 배경 인물 선택·좌표축 반전은 관찰되지 않았다. gt world→camera는 투영에만 한 번 적용하며 예측은camera로 유지한다. 이 표는official frame0 Sim3를 적용한3D metric 그림이 아니므로2D가 가까워도 depth/trajectory CD-H가 클 수 있다. 모델/체크포인트는 이 Test 사후 시각 결과로 선택하지 않는다. scope는30개 표본이며 전체영상 수동 검토가 아니다. `target_foreground_visuals/head_strong_exported/receipt.json`.')
    # Keep the same reviewed comparison at the front of both persistent notes.
    # Detailed per-sequence tables and visual audits remain in the report.
    research[2:2]=front[:front.index('\n### Target Test — 시퀀스별 완료분')] if '\n### Target Test — 시퀀스별 완료분' in front else front
    if export_backfill:
        research.append('\n## GPU6 export 추가 배정\n\n기존 GPU6 static shard의4개 변환 완료와 UUID 기준 NVML compute processes=[]·utilization0을 확인했다. 기존 평가20761과 DDP/refit은 재시작하지 않았다. 별도 관리 registry를 사용해 아직 시작하지 않은 Train 영상의 추론·변환을 GPU6에 배정하고 정상 CLI의 동일 stage lock/cache로 기존 shard와 중복 계산을 방지한다. checkpoint·전처리·solver·평가 방식·선택 기준은 그대로다. source export 및 global 선택은 계속 기존 실행기가 담당한다. `export_backfill_status.json`, `export_backfill_jobs.json`.')
    if train_export_review:
        research.append(f'\n## 추가 Train 저장본 수동 투영 점검\n\nHead 전체Train20 refit으로 저장한 실제 Uranus NPZ의 {len(train_export_review["samples"])}개 Train 영상/{train_export_review["manual_review_frames"]}표본 이미지를 직접 열어 검토했다. 표본에서 target actor 유지, 배경 사람 전환 없음, 좌표축 반전 없음이 관찰됐다. 자세·깊이 오차는 남고 전체 프레임 수동 검토가 아니다. 이 진단으로 모델/체크포인트를 선택하지 않는다. 생성 완료와 수동 검토 범위를 구분한다. 실제 NPZ·이미지SHA/표본 목록: `target_foreground_visuals/head_strong_exported/all/visual_review_progress.json`.')
    research.extend(followup_validation_lines)
    if evaluation_backfill:
        research.append('\n## 유휴 GPU의 고정 평가 추가 배정\n\nGPU5·6 UUID의 NVML compute processes=[]·utilization0을 확인했다. GPU6은 아직 추론하지 않은 `search_cd_last2_abs100/wet_floor_sign-pick-place-ground-08`의 Train 검증을, GPU5는 완료된 Train20 refit `pair_head_cd50/baseball_bat-rotate-05`의 고정 Test 평가를 수행한다. 정상 stage CLI/lock/cache를 사용하고 각각 실제 checkpoint SHA를 고정했다. Global 선택/mapping/comparison/source export는 수정하지 않으며 기존 실행기도 재시작하지 않았다. 후속6개 Test 비교는 이미 선언된 범위이고 모델 선택은 동일4개 Train MHR 결과로만 한다. 이 Test 한 영상의 점수로 모델·threshold·하이퍼파라미터를 선택하지 않는다. `evaluation_backfill_status.json`, `evaluation_backfill_jobs.json`.')
    if followup_integrity:
        research.append('\n## 후속6개 실제 재학습 체크포인트 독립 감사\n\n완료된6개 best.pt를 CPU에서 직접 로드했다. 실제 step과 complete.json 일치, 모든 model_delta 유한값, 전체Train 초기 checkpoint 대비 가중치 변경, full_training/fixed_full_train 및 refit metrics 비어 있음, 실제 search와 refit의 foreground 입력 일치를 확인했다. bootstrap+search+refit 실제 시간을 합산해 모든 후보가3시간 미만임을 확인했고, full_training_index의 Train20 범위·Test10 배제 및 원본GT30개 SHA를 직접 검증했다. 새 모델 선택이나 Test 품질 평가에 쓰지 않는 무결성 감사다. `target_followup_refit_integrity.json`.')
    if selected_prediction_review:
        research.append(f'\n## 현재 선택 모델 Test 예측의 수동 투영 점검\n\n`{chosen_name}`의 Test10×3=30개 표본 JPG를 직접 열어 raw SMPL-X/native MHR v2의 대상 배우 유지를 확인했다. 확인 표본에서 배경 인물 선택·좌표축 반전은 없었으며 자세·깊이 오차는 남아 있다. Watering은 camera root depth bias가 시각적으로 남으며2D 투영만으로3D 정확도를 주장하지 않는다. source Uranus export 이전의 prediction NPZ를 검토했으며 이미지·native NPZ SHA를 확인했다. 모델 선택은 앞선 Train 검증으로 고정했고 이 Test 사후 진단은 선택에 사용하지 않았다. 전체 프레임 수동 검토가 아니다. `target_foreground_visuals/{chosen_name}/receipt.json`.')
    if full_mask_audit:
        research.append(f'\n## 전체 프레임 GT 머리/마스크 일관성 진단\n\n현재 `{full_mask_audit["status"]}`: {full_mask_audit["clips_checked"]}/30개 영상, {full_mask_audit["frames_checked"]}프레임. 원본GT를 CPU MHR로 복원하고 world→camera를 정확히 한 번 적용해 head joint를 native K로 투영한다. 기존 inference와 같은 ellipse15 지지 영역과의 포함 관계·빈 마스크를 전수 집계하며 GT는 이 진단에만 사용한다. 추론/대상 선택/모델 선택/metric/export는 바꾸지 않는다. GT 머리점의 지지 비율은 완전한 segmentation 또는 identity 보증이 아니다. baseball의566–568번3표본을 직접 열어, bat가 아랫얼굴의 GT head 투영점을 가리고 person mask가 bat를 제외한 상황임을 확인했다. 대상 배우의 보이는 얼굴/몸은 입력에 유지되며 이 flag 자체가 배경 사람 전환을 뜻하지 않는다. 이3프레임을 metric에서 제외하지 않았다. `full_frame_target_mask_audit/audit.json`, `flagged_visual_reviews.json`.')
        if full_mask_audit.get('summary'):
            research.append('\n전수 진단의30개 per-frame NPZ를 다시 읽어 집계를 독립 재계산했다. GT 머리점이 화면 안에 있는22,990프레임 중22,888개가 ellipse15 mask support 안에 있고102개가 밖이었다. 빈 마스크0개. 표시 구간은 baseball3·cyan bottle17·light blue book82프레임이다. 모든 연속 구간에서 최초/중간/마지막을 택한28표본 JPG를 직접 열었다. 각각 bat·bottle·table에 GT head 투영점이 가려진 상황이었고 확인 표본의 visible target actor는 마스크에 남아 있었다. 모든102개를 수동 검토하거나 identity의 완벽함을 증명한 것은 아니며, 추론/점수/선택은 수정하지 않았다. 집계·이미지·검토 SHA: `full_frame_target_mask_audit/audit.json`, `flagged_visual_reviews.json`.')
    if selected_geometry:
        research.append(f'\n## 현재 선택 모델의 독립 CPU CD-H 재현\n\n`{chosen_name}`의 {len(selected_geometry.get("samples",{}))}/10개 Test 영상에 대해 모든 프레임의 native GT·예측·init을 CPU MHR decoder로 복원했다. 원본 진단과 같은 FPS27 최초 프레임 Sim3와 FPS64 비손 body 공개 CD 식을 사용하며 각 native MHR score가 GPU와0.02cm 이내인지 확인한다. 원시 SMPL-X도 공식 map으로 변환해 별도의 frame0 Sim3로 proxy를 계산한다. 공간별 score 차이는 가산 가능한 변환 오차 성분이 아니며 이 진단으로 모델을 선택하지 않는다. 기존 original 진단은 보존했다. `selected_geometry_diagnostics/{chosen_name}.json`.')
    write(RESEARCH,'\n'.join(research)+'\n')
    save_json(OUTPUT/'report_update.json',dict(updated_unix=time.time(),report=str(REPORT),research=str(RESEARCH)))


def update_strict_repair(repair):
    from track2_strict_scope import scope
    initial=read(OUTPUT/'init_comparison.json')['models']
    plan,prefix=scope()
    body_target=prefix.startswith('body_target')
    independent_prefix='body_target' if prefix.startswith('body_target') else 'strict'
    manifest=read(OUTPUT/'manifest.json')
    checked=read(OUTPUT/(independent_prefix+'_independent_Test_models.json'),{}).get('models',{})
    rejected=set(repair.get('rejected_strict_models',[]))
    checked={n:v for n,v in checked.items() if n not in rejected}
    values={n:initial[n] for n in ['init_v6','init_v0']};values.update(checked)
    names=['init_v6','init_v0']+[c['name'] for c in plan.get('models',[])]
    best={k:min(v[k] for v in values.values()) for k in ['CD_H_cm','ACC_H_cm_per_frame2']}
    lines=['# 트랙2 대상 인물 선택 수정 및 재평가',f'\n갱신: {utc(time.time())}',
        '\n## init_v6 / init_v0 비교 — 대상 인물 선택 재검증',
        '\n이전 마스크 확장 gate에서 배경 사람의 머리가 선택된 것을 확인했다. 이전 모델 점수·개선율은 유효한 target-only 비교에서 제외하고 다시 평가한다. 제공된 init_v6/init_v0 기준값은 유지한다. 새 모델 평균은 동일 Test10 전체7610프레임의 strict 선택·공통 MHR 채점을 검증한 뒤에만 표시한다.',
        '\n| 모델 | 완료 Test | CD-H cm ↓ | ACC-H cm/frame² ↓ | init_v6 대비 CD-H 개선율 |','|---|---:|---:|---:|---:|']
    for name in names:
        if name in values:
            value=values[name];cells=[]
            for key in ['CD_H_cm','ACC_H_cm_per_frame2']:
                cell=f'{value[key]:.4f}';cells.append('**'+cell+'**' if value[key]==best[key] else cell)
            gain=100*(initial['init_v6']['CD_H_cm']-value['CD_H_cm'])/initial['init_v6']['CD_H_cm']
            lines.append(f'| {name} | 10/10 | {cells[0]} | {cells[1]} | {gain:+.2f}% |')
        else:
            count=sum((OUTPUT/'predictions'/name/s/'metrics.json').exists() for s in manifest['test'])
            lines.append(f'| {name} | {count}/10 | — | — | '+('인물 선택 오류 확인' if name in rejected else '대기')+' |')
    if body_target and 'body_original_RGB' in checked:
        original_body=checked['body_original_RGB']
        lines.append(f'\n수정v3 원본 전체RGB의 Test10 전체7610프레임을 독립 검증했다: CD-H {original_body["CD_H_cm"]:.4f}cm, ACC-H {original_body["ACC_H_cm_per_frame2"]:.4f}cm/frame². 따라서 확인된 배경 인물 선택 오류를 고쳤다는 사실만으로 큰 원본 오차가 해결되거나 성능 차이 전체가 설명되지는 않는다. Train의 동일 원본 가중치 전체RGB/crop 비교는 입력 방식 영향의 증거이고, stroller raw SMPL-X의 허벅지 오검출은 변환 전 모델 오차의 직접 증거다. init_v6가 SAM3D Body이고 여기 원본이 Multi-HMR이라는 출처 차이도 구분한다. 각 요인의 전체 Test 오차 기여율을 분리했다고 주장하지 않는다.')
    if repair.get('body_overlap_repair_pending'):
        lines.append('\n원본 마스크 안으로 배경 사람의 head query가 투영되는 가림 사례가 추가 확인됐다. 원본 전체RGB의40.4357cm는 target-only 유효 점수에서 제외한다. 이전v2 나머지 모델 점수도 머리 점 검사 후의 잠정값으로 보존하고, 이 표는 몸의 투영 범위까지 검사하는v3 결과로 다시 채운다. 다음 학습은 CPU 대기 상태에서 중단해 잘못된 선택 규칙을 이어받지 않게 했다. [직접 확인한 raw SMPL-X/MHR 배경 인물 전환 영상]('+repair['body_failure_image']+').')
        rule=read(OUTPUT/'body_target_gate_Train_rule.json',{})
        if rule:lines.append(f'\n몸 검사 기준은 GT pose나 Test 점수를 읽지 않고 Train4의 기존9개 예측과 원본 mask만으로 고정했다. 각 모델 body box IoU의 하위1% 값 중 최솟값 {rule["minimum_body_box_iou"]:.6f}를 공통 최소값으로 쓴다. strict head gate와 body gate를 모두 통과한 후보 중 body IoU, confidence 순으로 선택한다. 저장된 spray 예측에서 대상 배우는 IoU 약0.70~0.77, 잘못 선택된 배경 사람은0.14~0.19로 분리됐다. 실제 같은 forward 후보의 재실행 확인과 전수 재평가를 진행한다. `body_target_gate_Train_rule.json`.')
        body_replay=read(OUTPUT/'body_target_detector_replay_check.json',{})
        if body_replay.get('status')=='passed':
            lines.extend(['\n실제 GPU 재실행 확인 완료: 동일 원본 가중치·RGB·K·같은 forward 후보에서 기존 선택을 저장 score/pelvis Z와1e-6 이내로 재현했다. 몸 검사만 추가해 배경 후보가 제외되고 대상 배우가 선택됐다. GT pose나 Test 점수는 읽지 않았다.',
                '\n| spray 프레임 | 기존 선택 깊이 m | 기존 body IoU | 새 선택 깊이 m | 새 body IoU |','|---|---:|---:|---:|---:|'])
            for frame,value in body_replay['frames'].items():
                old=value['old_head_only'];new=value['new_head_and_body']
                lines.append(f'| {frame} | {old["pelvis_Z_m"]:.4f} | {old["body_box_iou"]:.4f} | {new["pelvis_Z_m"]:.4f} | {new["body_box_iou"]:.4f} |')
            lines.append('\n306·319의 기존 골격은 배경에 앉은 사람을 따랐고 새 선택은 대상 배우다. 가중치 학습 효과나 전체 Test CD-H 개선 수치가 아니라 선택 오류의 인과 확인이다. 같은9개 checkpoint를 새 기준으로 다시 평가한다. `body_target_detector_replay_check.json`.')
        body_exports=read(OUTPUT/'body_target_export_receipts.json',{}).get('samples',{})
        body_source_ready=len(body_exports)==30 and all(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in body_exports.values())
        lines.append('\n현재 source Uranus는 '+('v3 경로로30개 저장된 중간 결과다.' if body_source_ready else '이전v2 경로의 중간 출력으로 보존한다.')+' 몸 일치 검사v3의 전체 비교·실제30개 저장·시각 확인·init_v6 성능 조건이 충족되기 전에는 최종 성공 결과로 간주하지 않는다.')
    state=read(OUTPUT/'strict_evaluation_status.json',{})
    if prefix.startswith('body_target'):
        state=read(OUTPUT/'body_target_evaluation_status.json',read(OUTPUT/'body_target_pilot_status.json',state))
    if prefix=='strict_final':state=read(OUTPUT/'native_surface_finalization_status.json',state)
    lines.extend(['\n## 확인된 오류와 수정',
        '\nblue_easter_basket-place-10의898프레임에서는 대상 배우의 머리가 선택됐지만899·900프레임에서는 뒤에 앉은 사람의 머리 위치가 선택됐다. 저장 pose/shape와 vertices에서 복원한 head ray의 위치가 약110px 이동했고, 잘못 선택된 점은 원래 대상 마스크 밖5.39px였다. 기존 ellipse15 확장이 그 점을 허용했다. RGB 확대 영상에서 배경 머리 위치임을 직접 확인했다. 앞선 “배경 인물 전환이 아니다”라는 판단을 정정한다.',
        f'\n증거: [마스크와 선택점 확대 영상]({repair["image"]}), `head_query_projection_diagnostic.json`. 복원 vertex의 median 잔차는 약0.3mm다.',
        ('\n현재 v3는 원본 binary target mask 안의 head query와 예측 몸 범위의 대상 box 일치 검사를 함께 요구한다. 머리 점만 검사하는 v2는 가림 상황에서 배경 인물을 허용하므로 충분하지 않다.' if body_target else '\n원본 binary target mask 안의 head query만 허용한 v2 결과다. 머리 점이 대상 어깨와 겹치는 배경 인물을 허용하는 추가 오류가 확인돼, 이 기준만으로 target-only 유효성을 확정할 수 없다.') + ' RGB 배경은 유지하고 마스크를 RGB에 씌우지 않는다. GT 3D·init pose로 후보를 고르지 않는다. 미검출은 관측된 선택 예측으로 보완하고 모든 원본 프레임을 평가한다. 학습 checkpoint는 보존하며 선택/평가만 먼저 바로잡는다.',
        f'\n실행 상태: `{state.get("status",repair["status"])}`. 정상 GPU1·2·4·5·6을 사용하고 GPU0은 보존한다.',
        '\n이전 score·보고서는 `invalid_dilated_head_gate/`와 기존 prediction 폴더에 진단 기록으로 보존한다. 기존 source Uranus도 strict 재검증 전에는 최종 유효 결과로 간주하지 않는다. GT와 init 파일은 바꾸지 않는다.'])
    replay=read(OUTPUT/'strict_detector_replay_check.json',{})
    if replay.get('status')=='passed':
        lines.extend(['\n실제 GPU detector 재실행으로 원인을 분리 확인했다. 같은 원본 가중치·RGB·K·동일 forward 후보에서 gate만 비교했다. 기존 선택은 저장된 prediction의 score·pelvis Z와1e-6 이내로 일치했다. Test GT나 init pose를 읽어 후보를 고르지 않았다.',
            '\n| 프레임 | 기존 확장 gate 선택 깊이 m | 수정 원본 mask 선택 깊이 m |','|---|---:|---:|'])
        for frame,value in replay['frames'].items():
            lines.append(f'| {frame} | {value["previous_expanded_mask_candidate"]["pelvis_Z_m"]:.4f} | {value["strict_original_mask_candidate"]["pelvis_Z_m"]:.4f} |')
        lines.append('\n899·900에서는 기존 gate의 선택이 배경 후보였고 수정 gate는 대상 후보를 선택했다. 즉 이 급변은 원본 모델의 depth 회귀 실패로 설명할 일이 아니라 인물 선택 오류였다. 전체 점수의 영향을 확정하려면9개 후보 전수 재평가가 필요하다. `strict_detector_replay_check.json`.')
    pilot=read(OUTPUT/'strict_pilot_status.json',{})
    if pilot:lines.append(f'\n실제 strict 추론 검증: `{pilot["status"]}`. 원본 및 미세조정5개 모델의 동일 basket 영상 전체980프레임을 재추론했다. 이는 전체 Test10 평균이 아니다.')
    temporal=read(OUTPUT/'strict_pilot_temporal_diagnostic.json',{})
    old_temporal=read(OUTPUT/'raw_temporal_stability_diagnostic.json',{}).get('models',{})
    if temporal and old_temporal:
        lines.extend(['\n동일 basket980프레임·원본 가중치의 raw pelvis 최대 연속 프레임 이동 진단:',
            '\n| 입력 | 기존 확장 gate cm | 수정 원본 mask gate cm |','|---|---:|---:|'])
        for label,old,new in [('전체 RGB','diagnostic_original_native_rgb','strict_original_RGB'),('고정 crop RGB','diagnostic_original_static_crop','strict_original_fixed_crop')]:
            before=old_temporal[old]['per_sequence'][temporal['sample']]['raw_root_step_max_cm']
            after=temporal['models'][new]['max_root_step_cm']
            lines.append(f'| {label} | {before:.4f} | {after:.4f} |')
        lines.append('\n원본 네트워크 가중치를 유지하고 선택 gate만 수정한 결과다. 이 값은 해당 한 영상의 raw 시간적 급변 진단이며 CD-H·ACC-H나 전체 Test 평균이 아니다. `strict_pilot_temporal_diagnostic.json`.')
    error=read(OUTPUT/(independent_prefix+'_evaluation_error.json'),{})
    if error:lines.append('\n재평가 실행 오류: `'+str(error.get('error'))+'`. 완료로 처리하지 않는다.')
    validation=read(OUTPUT/(independent_prefix+'_independent_Train4_models.json'),{}).get('models',{})
    geometry=read(OUTPUT/'selected_geometry_diagnostics/body_original_fixed_crop.json',{})
    if body_target and geometry.get('summary'):
        assert len(geometry['samples'])==10 and sum(v['frames'] for v in geometry['samples'].values())==7610
        geo=geometry['summary'];motion=geometry['motion_summary']
        lines.extend(['\n## 원본 crop의 변환 전후 및 이동·자세 진단 — 전체 Test10',
            f'\nCPU에서 같은7610프레임을 재계산했다. raw mapped SMPL-X surface proxy {geo["raw_mapped_SMPLX_CD_proxy_cm"]:.4f}cm, 변환 후 MHR CD-H {geo["converted_MHR_CD_H_cm"]:.4f}cm, init_v6 MHR CD-H {geo["init_v6_CD_H_cm"]:.4f}cm다. 큰 오차가 변환 전부터 존재하므로 MHR 변환만으로 전체 격차를 설명할 수 없다. 공간마다 frame0 Sim3를 따로 fitting하므로 두 값 차이를 가산 가능한 변환 오차로 해석하지 않는다.',
            '\n| 전체 Test10 진단 | 원본 crop MHR | init_v6 MHR |','|---|---:|---:|'])
        for label,key in [('프레임별 중심 위치 보정 후 CD cm','center_corrected_CD_diagnostic_cm'),('프레임별 Sim3 보정 후 CD cm','per_frame_Sim3_CD_diagnostic_cm'),('중심 이동 궤적 RMSE cm','centroid_trajectory_RMSE_cm')]:
            lines.append(f'| {label} | {motion["original_MHR"][key]:.4f} | {motion["init_v6_MHR"][key]:.4f} |')
        lines.append('\n위 보정은 GT를 읽는 사후 원인 진단이며 공식 점수·실제 예측·export에는 적용하지 않는다. 고정 frame0 정렬 이후 이동 오차와, 위치/Sim3를 보정해도 남는 자세·표면 오차가 모두 확인된다. 신경망 내부의 원인이나 요인별 가산 기여율을 확정한 결과는 아니다. 모든 입력 SHA와 프레임 수, CPU MHR 채점 재현을 확인했다. `selected_geometry_diagnostics/body_original_fixed_crop.json`.')
        lines.extend(['\n### 영상별 원본 crop / init_v6 격차',
            '\n전체 프레임 진단을 MHR CD 격차순으로 정리했다. 전부 cm이며 궤적 RMSE는 별도 진단 지표다. Test 진단을 모델 선택이나 학습 recipe에 적용하지 않았다.',
            '\n| 영상 | 프레임 | raw SMPL-X proxy | 원본 MHR CD | init_v6 CD | CD 격차 | 중심 궤적 RMSE 원본 / init |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for sample,value in sorted(geometry['samples'].items(),key=lambda item:item[1]['converted_MHR_CD_H_cm']-item[1]['init_v6_CD_H_cm'],reverse=True):
            original=value['motion_diagnostics']['original_MHR']['centroid_trajectory_RMSE_cm']
            initial=value['motion_diagnostics']['init_v6_MHR']['centroid_trajectory_RMSE_cm']
            gap=value['converted_MHR_CD_H_cm']-value['init_v6_CD_H_cm']
            lines.append(f'| {sample} | {value["frames"]} | {value["raw_mapped_SMPLX_CD_proxy_cm"]:.3f} | {value["converted_MHR_CD_H_cm"]:.3f} | {value["init_v6_CD_H_cm"]:.3f} | {gap:+.3f} | {original:.3f} / {initial:.3f} |')
    native_state=read(OUTPUT/'native_surface_training_status.json',{})
    surface_refits=read(OUTPUT/'native_surface_refit_mappings.json',{})
    if body_target and native_state.get('status')=='all_native_surface_searches_and_Train20_refits_complete' and surface_refits:
        lines.extend(['\n## 후속 native-MHR surface 세 후보 실제 학습 완료',
            '\n세 후보의 검색과 Train20 refit이 완료됐다. 복구 시 보존한 학습 시간과 별도 실패 시도 시간을 합산했다. 아래는 실제 완료 기록이며 새 Test 성능은 전체 MHR 평가가 끝난 뒤 별도로 비교한다.',
            '\n| 모델 | search+refit+복구 시간 h | refit step | 성공 update |','|---|---:|---:|---:|'])
        for name,value in surface_refits.items():
            total=value['search_seconds']+value['refit_seconds']+value.get('training_lineage_seconds',0)+value.get('recovery_attempt_seconds',0)
            assert total<10800 and value['refit']['refit_performed_this_wave']
            done=value['refit']['complete']
            lines.append(f'| {name} | {total/3600:.4f} | {done["step"]} | {done["successful_steps"]} |')
        lines.append('\n모든 refit은2GPU DDP, ColorJitter·AdamW decay0.1·L2-SP를 사용한다. 가중치 변경/유한값 독립 감사는 `completed_native_surface_refit_parameter_audits.json`에 완료 시 기록한다. 모델 선택은 새3개를 포함한 전체12개 Train4 MHR 비교가 끝나야 고정한다.')
    recovery=read(OUTPUT/'native_surface_mask_decode_recovery.json',{})
    if body_target and recovery:
        lines.append('\n후속 학습의 실제 validation mask-video open timeout을 확인해 복구했다. 원본30개 마스크를 SHA가 같은 SSD 캐시로 복사하고 검증 디코더를1thread로 제한했다. barrier로16개 reader를 동시에 연 CPU 검사와 source/cache 픽셀 일치 검사를 통과했다. 완료된 head 검색은 보존하고 last2는 optimizer/step/원래 시작시각을 복원하며, baseline 전 실패한 last4는 같은 recipe로 다시 실행한다. 실패 로그·기존 checkpoint·이전 시도 시간을 보존했고 전체 마감·채점·인물 선택 기준은 바꾸지 않았다. `native_surface_mask_decode_recovery.json`, `target_mask_decoder_recovery_preflight.json`.')
    source_review=read(OUTPUT/'body_target_export_visual_manual_review.json',{})
    if body_target and source_review:
        assert source_review['clips_manually_reviewed']==30 and source_review['frames_manually_reviewed']==90
        lines.append('\n현재 body-v3 선택 모델의 실제 source Uranus30개에 대해 최초·중간·마지막90표본을 모두 직접 열어 확인했고, image/source/raw SHA와 개별 관찰을 기록했다. 확인 표본의 raw SMPL-X와 exported MHR는 대상 배우를 따른다. watering의 약0.4~0.6m 양의 깊이 편향, 굽힌 자세·가려진 팔다리 오차 등은 남는다. 이는90표본 검토이며22990프레임 전수 수동 identity 확인이나 init_v6 성능 초과를 뜻하지 않는다. 새 후보가 선택돼 source가 바뀌면 새 파일을 다시 검토해야 한다. `body_target_export_visual_manual_review.json`.')
    if body_target and native_state.get('status')=='training_native_surface_trials':
        lines.append('\n후속 native-MHR surface 학습이 실제 시작됐다. body-v3 선택30개 source 생성 후 실제2GPU20step pilot(19회 성공 update), GPU6 last.pt posthoc 검증을 통과했다. 두 후보는 GPU[1,2]와[4,5]에서2GPU DDP로 실행하고 세 번째는 pair를 기다린다. 아직 새 후보의 전체 Test 성능은 없다. 실제90개 source 시각화 표본의 직접 수동 검토 상태는 위 source 검토 기록을 따른다. `strict_surface_posthoc_preflight.json`, `native_surface_training_jobs.json`.')
    admission=read(OUTPUT/'body_target_surface_budget_admission_update.json',{})
    if body_target and admission:
        lines.append('\n후속 native-MHR surface 학습의 시작 여유시간은 기존5시간에서4시간으로 조정했다. 실제 완료된 subprocess 소요시간을 근거로 한 자원 배치 추정이며 완료 보장은 아니다. 후보별 search+refit 최대1시간, 실제2GPU DDP, 전체28시간 마감과 후보당3시간 제한 및 사전 고정 recipe는 유지한다. 자식 작업 없이 기다리던 CPU 관리자 두 개만 다시 실행했고 GPU 재평가는 유지했다. `body_target_surface_budget_admission_update.json`, `body_target_historical_runtime_estimate.json`.')
    if validation:
        lines.extend(['\n## Train4 전체 MHR 비교', '\n| 모델 | CD-H cm ↓ | ACC-H cm/frame² ↓ |','|---|---:|---:|'])
        baseline=read(OUTPUT/'init_v6_Train4_MHR_baseline.json',{})
        rows={'init_v6':baseline['summary']} if baseline else {};rows.update(validation)
        minima={k:min(v[k] for v in rows.values()) for k in ['CD_H_cm','ACC_H_cm_per_frame2']}
        for name,value in rows.items():
            if name!='init_v6':assert value['sequences']==4 and value['frames']==2714
            cells=[('**'+f'{value[k]:.4f}'+'**') if value[k]==minima[k] else f'{value[k]:.4f}' for k in ['CD_H_cm','ACC_H_cm_per_frame2']]
            lines.append(f'| {name} | {cells[0]} | {cells[1]} |')
        lines.append('\n각 행은 동일4영상 전체2714프레임이다. 일부 영상 평균을 표시하지 않는다. Test와 다른 모집단이며 모델 선택은 이 Train 검증만 사용한다.')
    selection=read(OUTPUT/(prefix+'_selected.json'),{})
    if selection:
        lines.append(f'\nTrain 기준 선택: `{selection["trial"]}`. 새 Test 점수를 보기 전에 고정한다.')
        definitions={m['name']:m for m in plan['models']};winner=selection['trial']
        control=('body_' if prefix.startswith('body_target') else 'strict_')+('original_fixed_crop' if definitions[winner]['input_view']=='native_rgb_static_crop' else 'original_RGB')
        if winner in validation and control in validation:
            original=validation[control]['CD_H_cm'];learned=validation[winner]['CD_H_cm']
            gain=100*(original-learned)/original
            lines.append(f'\n선택 모델의 동일 입력 방식 원본 대조군 대비 Train CD-H: {original:.4f}→{learned:.4f}cm, {gain:+.2f}% 개선. 전체 RGB와 crop 사이의 차이를 미세조정 효과로 합산하지 않는다. Test 개선율과는 별개다.')
    control_prefix='body_' if body_target else 'strict_'
    if all(control_prefix+n in validation for n in ['original_RGB','original_fixed_crop']):
        full=validation[control_prefix+'original_RGB']['CD_H_cm'];crop=validation[control_prefix+'original_fixed_crop']['CD_H_cm']
        lines.append(f'\n확인된 입력 방식 영향: 같은 원본 가중치·원본 mask 선택 기준에서 전체 RGB {full:.4f}cm, 고정 crop RGB {crop:.4f}cm로 Train CD-H가 {100*(full-crop)/full:.2f}% 낮아졌다. crop은 RGB 배경을 지우지 않으며 K도 함께 변환한다. 이 비교는 crop으로 함께 바뀌는 인물 크기·입력 시야각 등의 효과를 분리하지 않으므로 FOV 하나만 원인이라고 단정하지 않는다.')
    if (OUTPUT/'init_v6_generation_provenance_audit.json').exists():
        lines.append('\n비교 모델의 출처: 제공된 init_v6는30개 NPZ 내부 report가 각 SAM3D Body diagnostics와 일치하는 생성 결과다. 여기서 원본 대조군은 Multi-HMR 사전학습 가중치다. 따라서 init_v6와 원본은 같은 모델의 학습 전후가 아니다. 이 출처 차이만으로 정확도 차이 전체를 설명하지는 않는다. `init_v6_generation_provenance_audit.json`.')
    review_file=independent_prefix+'_Train4_selected_visual_review.json'
    body_stroller_review=read(OUTPUT/'strict_temporal_manual_reviews.json',{}).get('cases',{}).get('body_original_RGB/toy_stroller-back-forth-ground-04',{})
    if body_target and body_stroller_review.get('reviewed_by_agent'):
        lines.append('\n몸 범위 검사v3의 원본 전체RGB stroller 경고2개(433→434,434→435)를 직접 확인했다. 432~436의 RGB/mask/head query 창과433/434/435의 raw SMPL-X/MHR 골격에서,434의 query가 대상 배우 허벅지에 놓이고 작은 잘못된 골격을 예측하는 것을 확인했다. 배경의 앉은 사람 전환은 아니지만 정상 머리 검출도 아니다. raw 깊이2.383→3.035→2.360m의 급변은 MHR 변환 전에 존재한다. 모든 프레임을 유지하며 이 Test 진단으로 모델 선택·threshold·recipe를 바꾸지 않는다. `strict_temporal_manual_reviews.json`의 body_original_RGB/toy_stroller-back-forth-ground-04 항목.')
    reviewed=read(OUTPUT/review_file,{})
    if reviewed:
        lines.append(f'\n직접 시각 확인: `{reviewed["trial"]}`의 Train 검증4영상 최초·중간·마지막 총{reviewed["frames_manually_reviewed"]}프레임을 열어 봤다. 해당12표본에서는 물체를 조작하는 대상 배우를 따르고 배경의 앉은 사람을 선택하지 않았다. 팔·다리 자세와 카메라 깊이 오차는 남아 있다. 전체 프레임 인물 일치나 최종 source 파일 확인을 대신하지 않으며 모델 선택도 바꾸지 않았다. 이미지·raw·MHR SHA와 개별 관찰은 `{review_file}`, 영상은 `target_foreground_visuals/search_{reviewed["trial"]}/validation/`에 있다.')
    flags=sum(v.get('temporal_flag_count',0) for v in checked.values())
    flag_text=f'{len(checked)}개 모델에서 {flags}개' if checked else '전체 Test 검증 완료 모델이 없어 아직 집계하지 않음'
    lines.append(f'\n독립 검증 완료 모델의 Test 큰 급변 표시 구간: {flag_text}. 원본 mask 픽셀 밖 head가0개인지 전체 프레임에서 재검사한다. root 이동50cm 또는 관측 head 이동75px를 넘는 구간은 추가 수동 확인 대상으로 기록하며 예측 선택이나 채점 프레임을 바꾸지 않는다. `strict_independent_Test_models.json`, 각 영상의 `strict_independent_head_audit.json`.')
    stroller=read(OUTPUT/'strict_stroller_false_head_diagnostic.json',{})
    if stroller:
        lines.extend(['\n## 추가 확인: 대상 몸 안의 잘못된 머리 검출',
            '\nstroller 원본 전체RGB에서 표시된6개 급변 구간을 모두 직접 열어 확인했다. 배우가 몸을 숙였을 때 검출기가 같은 대상 배우의 허벅지를 머리 후보로 내고 상체 후보와 번갈아 선택했다. 앞선 basket의 배경 사람 선택과는 다른 검출 오류다. 원본 mask 안이라는 검사만으로 해부학적 머리 위치가 맞다고 보증할 수 없다. 이 잘못된 예측도 그대로 채점하며 프레임이나 점수를 제거·수정하지 않는다.',
            '\n| 동일 stroller946프레임 입력/모델 | raw pelvis 최대 프레임 이동 cm | 관측 query 최대 프레임 이동 px |','|---|---:|---:|'])
        for name,value in stroller['models'].items():
            lines.append(f'| {name} | {value["max_root_step_cm"]:.4f} | {value["max_observed_query_step_px"]:.4f} |')
        lines.append('\n원본 가중치 crop 대조군에서는 해당 큰 급변이 줄었다. 이는 한 영상의 검출·시간 진단이며 CD-H나 Test10 평균이 아니다. 선택 모델의438·439·455프레임 raw SMPL-X/MHR도 직접 확인했다. 대상 배우를 따르지만455의 팔 뻗기 등 자세 오차는 변환 전부터 남는다. 모델·예측·채점·사전 고정 추가 학습 recipe는 바꾸지 않았다. `strict_stroller_false_head_diagnostic.json`, `strict_temporal_manual_reviews.json`.')
    audit_error=read(OUTPUT/(independent_prefix+'_independent_audit_error.json'),{})
    if audit_error:lines.append('\n독립 검증 오류 확인 필요: `'+str(audit_error.get('error'))+'`. 해당 미검증 평균은 표시하지 않는다.')
    exports=read(OUTPUT/(prefix+'_export_receipts.json'),{})
    if exports:
        count=sum(Path(v['path']).exists() and sha256(Path(v['path']))==v['sha256'] for v in exports.get('samples',{}).values())
        lines.append(f'\nstrict 선택 모델의 source Uranus 저장·SHA 일치: {count}/30개.')
    visual=read(OUTPUT/('native_surface_visual_preparation_status.json' if prefix=='body_target_final' else independent_prefix+'_visual_preparation_status.json'),{})
    if visual:
        lines.append(f'\n실제 최종 source 파일 시각화 준비: `{visual["status"]}`. strict 30개 저장과 원본 mask 검증이 끝난 뒤 CPU에서 최초·중간·마지막 총90프레임을 렌더링한다. 렌더링 완료와 직접 확인 완료는 구분한다.')
    identity=read(OUTPUT/'strict_identity_visual_status.json',{})
    if identity:
        clips=identity['fully_checked_clips'];frames=sum(v['frames'] for v in identity['cases'].values())
        lines.append(f'\n별도 CPU 인물 선택 검사: 현재 {clips}개 예측 영상의 전체 {frames}프레임에서 수락된 head가 원본 mask 안에 있는지 검사했다. 큰 급변 표시는 {identity["flagged_transitions"]}개다. 표시된 구간은 배경을 유지한 RGB·원본 mask 윤곽·head 선택점을 확대 렌더링하며 직접 확인은 별도로 기록한다. 이 검사는 CD-H 평균이나 전체 인물 일치의 증명이 아니다. `strict_identity_visual_status.json`, `strict_identity_visuals/`.')
    audit=read(OUTPUT/(prefix+'_completion_scope_audit.json'),{})
    if audit:
        checks=audit['checks'];passed=sum(v['passed'] for v in checks.values())
        lines.append(f'\n수정 경로의 완료 조건 독립 점검: {passed}/{len(checks)}개 충족. 데이터·학습·선택·Test·최종 저장·성능·시각 확인·보고서·자산·수행시간을 각각 검사한다. 미충족 항목이 있으므로 최종 성공으로 보고하지 않는다. `{prefix}_completion_scope_audit.json`.')
    surface_pilot=read(OUTPUT/'native_surface_pilot_status.json',{})
    if surface_pilot.get('trial') in ['strict_native_surface_two_GPU_pilot','body_native_surface_two_GPU_pilot']:
        training=read(OUTPUT/'native_surface_training_status.json',{})
        learning_status=training.get('status','아직 시작하지 않음')
        lines.append(f'\n추가 학습 전 native MHR-GT surface 정합성 pilot: `{surface_pilot["status"]}`. 기존9개 strict 재평가·30개 저장이 끝난 뒤 유휴 GPU2개에서20step만 검사하며 pilot 가중치는 결과로 사용하지 않는다. 입력 방식은 사전 고정 규칙으로 두 원본 대조군의 전체 Train4 CD-H/ACC-H만 비교해 정한다. RGB 배경을 유지하고 내부 검증도 확장하지 않은 원본 mask를 쓴다. 추가3개 본 학습 상태: `{learning_status}`.')
        posthoc=read(OUTPUT/'strict_surface_posthoc_preflight.json',{})
        posthoc_state=posthoc.get('status','실제 GPU 검사 대기')
        lines.append(f'\n추가 학습 전 별도 Train 검증 명령 검사: `{posthoc_state}`. pilot의 실제 마지막 업데이트 가중치로 전체 Train4 proxy를 GPU6에서 검사하고, strict mask·원본 dilation0·native MHR-GT target·checkpoint SHA를 확인한 뒤3개 검색 학습을 시작한다. 내부 검증 결과의 strict metadata 누락은 사전에 수정했다. 이 GPU 검사는 아직 통과하지 않았다면 완료로 표시하지 않는다.')
        if (OUTPUT/'strict_surface_Test_dispatch_policy.json').exists():
            lines.append('\n추가 후보의 Test 작업은 전체 Train4 CD-H/ACC-H 순위가 높은 모델부터 해당 Test10을 먼저 완료하도록 배치한다. 첫 전체 평균을 일찍 확인하기 위한 실행 순서이며, 나머지 후보나 프레임을 제외하지 않고 Test 점수로 순위를 바꾸지도 않는다. 현재 실행 중인 기존9개 재평가의 순서는 바꾸지 않았다.')
    native_training=read(OUTPUT/'native_refit_mappings.json',{})
    fixed_training=read(OUTPUT/'fixed_refit_mappings.json',{})
    native_parameters=read(OUTPUT/'completed_native_refit_parameter_audits.json',{})
    fixed_parameters=read(OUTPUT/'completed_fixed_refit_parameter_audits.json',{}).get('models',{})
    if native_training and fixed_training:
        lines.extend(['\n## 실제 학습과 데이터',
            '\n30개 영상을 seed261025로 Train20/Test10에 고정했다. Train 안에서는16개 학습/4개 내부 검증을 분리했고, 캐시는851개 학습 프레임·256개 검증 프레임이며 Train20 refit은1107개다. Train20의 MHR GT만 SMPL-X 감독 신호로 변환했다. 최종 비교는 예측 SMPL-X를 MHR로 변환한 뒤 수행한다. Test GT는 학습이나 후보 선택에 쓰지 않는다.',
            '\n아래6개 학습은 실제 GPU2개 DDP로 완료됐다. ColorJitter·AdamW decay0.1·원본 가중치에 대한 L2-SP를 적용했다. 총 학습 시간은 bootstrap/부모 학습·검색·Train20 refit을 합산한 비용이다. 현재v3 재평가는 이 가중치를 보존하고 인물 선택을 다시 검사하는 단계다.',
            '\n| 학습 trial | GPU 수 | 누적 실제 학습 시간 h | 평가용 Train20 checkpoint step | 파라미터 변화 |','|---|---:|---:|---:|---|'])
        for name,run in dict(native_training,**fixed_training).items():
            seconds=sum(run.get(k,0) or 0 for k in ['training_lineage_seconds','search_seconds','refit_seconds'])
            parameters=native_parameters.get(name,fixed_parameters.get(name,{}))
            if not parameters:continue
            changed=('있음' if parameters.get('L2_parameter_change',0)>0 else str(parameters.get('changed_parameter_tensors',0))+'개 tensor')
            lines.append(f'| {name} | 2 | {seconds/3600:.4f} | {parameters["step"]} | {changed} |')
        lines.append('\nfixed_head는 검색 중 업데이트했지만 최종 선택이step0이어서 평가용 가중치가 원본과 동일하다. 이를 학습 개선으로 세지 않는다. 원본 비교 가중치는 Multi-HMR이며 제공 init_v6는 SAM3D Body 결과다. 근거는 실제 runs/configuration·complete와 `completed_native_refit_parameter_audits.json`, `completed_fixed_refit_parameter_audits.json`이다.')
    lines.extend(['\n## 평가 기준',
        '\n`ChallengeHOR/scripts/run_evaluation.py`가 사용하는 문서화된 local protocol과 동일한 공개 MHR human 지표 함수를 사용한다. 원본 frame0의 손 제외 FPS27 정점으로 proper Sim(3)을 한 번 구해 영상 전체에 고정한다. CD-H는 손 제외 FPS64 표면점의 양방향 최근접 유클리드 거리 평균의 합(cm), ACC-H는 body22 관절의 연속 원본 프레임 2차 차분 오차(cm/frame²)이며 fps²를 곱하지 않는다. 누락·미검출 프레임도 제거하지 않는다. 각 영상의 평균을 구한 뒤 영상별 동일 가중치로 합산한다.',
        '\n주최 측 host 정점 역할과 quantized 모델 자산이 공개되지 않아, 참조 평가 스크립트와 같은 고정 local 역할을 쓴다. 공식 리더보드 수치와 완전히 같다고 주장하지 않는다. init_v6·init_v0·모든 모델에는 동일한 지표·역할·프레임 목록을 적용한다. 내부 SMPL-X CD proxy와 최종 MHR CD-H를 혼동하지 않는다.'])
    train_analysis=read(OUTPUT/'Train_only_post_delivery_failure_analysis.json',{})
    if body_target and train_analysis:
        lines.extend(['\n## 추가 학습 전 Train-only 기록 분석',
            '\n기존 search checkpoint의 전체 held-out Train4 결과다. Test나 Train20 refit 점수와 혼합하지 않는다.',
            '\n| Train4 영상 | init_v6 CD cm | 원본 crop CD cm | surface-last4 CD cm |',
            '|---|---:|---:|---:|'])
        for sample,value in train_analysis['per_sequence_Train4'].items():
            lines.append(f'| {sample} | {value["init_v6_CD_H_cm"]:.4f} | {value["body_original_fixed_crop"]:.4f} | {value["native_surface_last4_cd100_rel100"]:.4f} |')
        lines.extend(['\n| search step | MPJPE mm | root-aligned MPJPE mm | PA-MPJPE mm | 내부 surface CD proxy cm |',
                      '|---|---:|---:|---:|---:|'])
        for value in train_analysis['search_validation_trajectory']:
            lines.append(f'| {value["step"]} | {value["mpjpe_mm"]:.3f} | {value["root_mpjpe_mm"]:.3f} | {value["pa_mpjpe_mm"]:.3f} | {value["cd_h_smplx_proxy_cm"]:.4f} |')
        lines.append('\n내부 proxy는256표본 중 검출253표본의 별도 지표다. 관절 지표가 좋아지는 동안 proxy가 악화됐다. 이것은 특정 loss의 인과 효과를 분리한 증거는 아니다. 추가 GPU 학습은 시작하지 않았으며 추가 학습 시간 답변을 기다린다. `Train_only_post_delivery_failure_analysis.json`.')
    recovery=read(OUTPUT/'native_surface_completion_recovery.json',{})
    if body_target and recovery:
        pid=recovery.get('supervisor_pid');process=Path('/proc')/str(pid)
        try:
            live=process.joinpath('stat').read_text().split()[21]==str(recovery.get('supervisor_start_ticks'))
            live=live and b'finalize_track2_native_surface_wave.py' in process.joinpath('cmdline').read_bytes()
        except (FileNotFoundError,ProcessLookupError):live=False
        resources=read(OUTPUT/'gpu_resources.json',{})
        lines.extend(['\n## 재부팅 후 미완료 평가·출력 복구',
            f'\n원래 실험 마감은 {utc(recovery["original_budget_deadline_unix"])}로 보존했다. 사용자의 Goal 재개 후 이미 학습된 후보의 미완료 평가·출력만 별도 {recovery["completion_timeout_seconds"]/3600:.1f}시간 실행 제한으로 복구한다. 새 학습은 시작하지 않았다. 완료 작업 제한은 추론·MHR 변환·보정까지 전달하며 budget.json과 Train 선택은 변경하지 않는다.',
            f'\n현재 UUID 기준 GPU 배치는 {resources.get("healthy_gpus",[])}이고 GPU0은 보존한다. 복구 관리자 PID {pid}의 명령/start tick 일치 및 생존 확인: {live}. 별도 실행 제한 종료: {utc(recovery["completion_deadline_unix"])}. `native_surface_completion_recovery.json`, `post_reboot_evaluation_recovery_GPU_verified.json`.'])
        recovery_error=read(OUTPUT/'native_surface_finalization_error.json',{})
        if recovery_error:lines.append('\n복구 실행의 실제 종료 오류: `'+str(recovery_error.get('error'))+'`. 작업이 완료됐다고 처리하지 않는다.')
        heap_check=read(OUTPUT/'post_reboot_double_free_completed_output_verification.json',{})
        if heap_check:
            lines.append(f'\n변환 프로세스의 native heap 종료 오류는 보존했다. 해당 {heap_check["sample"]}의 실제 완료 파일/source 해시와 전체 {heap_check["frames"]}프레임을 CPU 복원으로 확인했으며, 저장 잔차와 재계산 잔차 최대 차이는 {heap_check["max_stored_vs_CPU_residual_difference_mm"]:.8f}mm다. 검증된 완료 결과는 보존하고 누락 작업만 재개한다. 재시도 시 완료 작업의 기존 종료 시각을 유지한다. `post_reboot_double_free_completed_output_verification.json`.')
    text='\n'.join(lines)+'\n';write(REPORT,text);write(RESEARCH,text.replace('# 트랙2 대상 인물 선택 수정 및 재평가','# 트랙2 대상 선택 오류 수정 연구',1))
    save_json(OUTPUT/'report_update.json',dict(updated_unix=time.time(),report=str(REPORT),research=str(RESEARCH),strict_target_repair=True))


def update():
    with (OUTPUT/'report_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        repair=read(OUTPUT/'strict_target_repair.json',{})
        if repair.get('active'):
            from track2_strict_scope import scope
            plan,scope_prefix=scope()
            independent_prefix='body_target' if scope_prefix.startswith('body_target') else 'strict'
            manifest=read(OUTPUT/'manifest.json',{})
            for split,label in [('validation','Train4'),('test','Test')]:
                checked=read(OUTPUT/(independent_prefix+'_independent_'+label+'_models.json'),{}).get('models',{})
                def complete(model):
                    checkpoint=model['search_checkpoint' if split=='validation' else 'full_checkpoint']
                    trial=model['name'] if split=='test' or checkpoint is None else 'search_'+model['name']
                    return all((OUTPUT/'predictions'/trial/s/'metrics.json').exists() for s in manifest.get(split,[]))
                if any(m['name'] not in checked and complete(m) for m in plan.get('models',[])):
                    result=subprocess.run([sys.executable,str(ROOT/'tools/audit_track2_strict_results.py'),
                        '--split',split,'--new-only'],capture_output=True,text=True)
                    if result.returncode:
                        save_json(OUTPUT/'strict_independent_audit_error.json',dict(error=result.stderr[-3000:],split=split,updated_unix=time.time()))
            update_strict_repair(repair);return
        manifest=read(OUTPUT/'manifest.json',{})
        roster=manifest.get('test',[])
        for family,prefix,plan_path in [('fixed','fixed','fixed_view_adaptation_plan.json'),
                ('native-surface','native_surface','native_surface_adaptation_plan.json')]:
            plan=read(OUTPUT/plan_path,{})
            checked=read(OUTPUT/(prefix+'_completed_Test_models.json'),{}).get('models',{})
            names=(['diagnostic_original_static_crop'] if family=='fixed' else [])+[c['name'] for c in plan.get('trials',[])]
            if len(roster)==10 and any(name not in checked and all(
                    (OUTPUT/'predictions'/name/sample/'metrics.json').exists() for sample in roster) for name in names):
                subprocess.run([sys.executable,str(ROOT/'tools/audit_track2_completed_fixed_validation.py'),
                    '--split','test','--new-only','--family',family],check=True,capture_output=True,text=True)
        _update()


def watch_deadline():
    deadline=read(OUTPUT/'budget.json',{}).get('deadline_unix',time.time())
    recovery=read(OUTPUT/'native_surface_completion_recovery.json',{})
    pid=recovery.get('supervisor_pid')
    if pid is not None:
        process=Path('/proc')/str(pid)
        try:
            same=process.joinpath('stat').read_text().split()[21]==str(recovery['supervisor_start_ticks'])
            command=process.joinpath('cmdline').read_bytes()
            if same and b'finalize_track2_native_surface_wave.py' in command:
                deadline=max(deadline,recovery['completion_deadline_unix'])
        except (FileNotFoundError,KeyError,ProcessLookupError):
            pass
    return deadline


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        if not args.watch:
            update();break
        # Execute the current renderer every tick so reviewed template changes
        # take effect without restarting this long-lived CPU-only watcher.
        subprocess.run([sys.executable,str(Path(__file__).resolve())],check=False)
        deadline=watch_deadline()
        if time.time()>=deadline:break
        time.sleep(min(600,max(1,deadline-time.time())))

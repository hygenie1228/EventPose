"""Korean primary Test table first, refreshed every thirty minutes."""
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import time
from device3_encoding_scope import ROOT,GOAL,EVAL_NAMES,TRAIN_NAMES,LABELS,METRICS
from goal_runtime_control import write,register_job
KST=timezone(timedelta(hours=9))
def read(p):return json.loads(Path(p).read_text())
def stamp(t):return datetime.fromtimestamp(t,KST).strftime('%Y-%m-%d %H:%M:%S KST')

def snapshot(g):
    out=Path(g['output_root']);report=Path(g['report'])
    if (out/'final_artifact_audit.json').exists():return
    comparison=None
    if (out/'source_reuse_audit.json').exists():
        from fusion_repair_test_progress import completed_test_snapshot
        comparison=completed_test_snapshot(out,EVAL_NAMES)
    state=read(out/'pipeline_state.json') if (out/'pipeline_state.json').exists() else {}
    lines=['# 이벤트 인코딩 세 가지와 센서융합 비교','',
        '| 조건 / 인코딩 | first frame-aligned MPJPE (mm) | first frame-aligned PA-MPJPE (mm) | frame-aligned MPJPE (mm) | frame-aligned PA-MPJPE (mm) | Root position error (mm) |',
        '|---|---:|---:|---:|---:|---:|']
    for n in EVAL_NAMES:
        values=comparison['metrics'].get(n,{}) if comparison else {};cells=[]
        for k in METRICS:
            v=values.get(k);s='미평가' if v is None else f'{v:.2f}'
            if comparison and len(comparison['conditions'])==len(EVAL_NAMES):
                other=[comparison['metrics'][a][k] for a in EVAL_NAMES]
                if v is not None and all(z is not None for z in other) and v==min(other):s='**'+s+'**'
            cells.append(s)
        lines.append('| '+LABELS[n]+' | '+' | '.join(cells)+' |')
    common=next(iter(comparison['metrics'].values()))['samples'] if comparison else 0
    lines+=['',f'갱신: {stamp(time.time())}. 전체 Test 완료 {len(comparison["conditions"]) if comparison else 0}/11조건. '
        f'현재 완료 조건들 동일 공통 GT {common:,}개를 재집계한다. 완료 조건 추가 시 공통 표본이 바뀐다.',
        'Test는 body 18피험자·80개의 동일10초 구간·24,080프레임 전체다. 미완료·미평가 값은 채우지 않는다.',
        'Root position error는 비정렬 공통 device3 RGB 카메라좌표의 3D pelvis 거리다. 깊이 절댓값 오차 RPE-Z는 아래 보조표로 구분한다.', '',
        '## 실행 범위와 공정 비교','',
        f'고정 기간: {stamp(g["started_unix"])} ~ {stamp(g["deadline_unix"])}. 전체 최대12시간에 검증·캐시·학습·전체 Test·속도·통계·보고서 검증을 포함한다.',
        '원본 device3 RGB는 ETRI 추가학습0시간, device3 RGB는90분, event 조합9조건은 각각3시간이다. '
        '세 인코딩 간에는 같은 예산·원본 초기화·seed·Train subset·증강·batch를 사용한다. '
        'RGB 대비 차이는 학습량과 모달리티 효과가 섞이므로 event만의 인과적 효과로 주장하지 않는다.',
        '장치가 대칭이라는 이번 범위에 따라 device4 RGB 기준선·역방향 조합은 반복하지 않는다. 물리 장착·보정의 수치적 대칭을 가정해 좌표를 생략하지 않는다.',
        '이전 미세조정 가중치와 Test 예측은 이번 실험 결과로 재사용하지 않는다. 같은 split 캐시는 실제 SHA·인과적 프레임 매핑·피험자 분리를 검증한 뒤 읽기전용으로 참조한다.', '',
        '| 인코딩 | 채널 | 시간 범위 / 의미 |','|---|---:|---|',
        '| count33 | 2 | 최근33ms 극성별 저장 활성픽셀 관측 로그합 |',
        '| bins100 | 6 | 최근100ms를3구간으로 나눈 극성별 로그합 |',
        '| surface20 | 2 | 최근200ms 이내 마지막 극성별 관측의20ms 지수감쇠 |',
        '저장 희소 event 프레임으로 계산하며 개별 원시event 시각·횟수를 복원했다고 주장하지 않는다. '
        '보정 timestamp의 과거 관측만 사용한다. MP4를 event 입력으로 사용하지 않는다. '
        'RGB896·event224 별도 encoder/adapter와 실제 내외부 보정 ray fusion을 유지한다. bins100은 실제6채널로 학습한다.',
        '모든 조건은 원본 Multi-HMR에서 새로 시작한다. AdamW·AMP·유한 gradient 검사·초반 adapter/head,30분 이후 마지막8개 backbone block 적응·주기 checkpoint를 사용한다. '
        '검증은 Train 피험자 holdout만 checkpoint 선택에 쓰며 Test로 encoding·가중치를 선택하지 않는다.', '',
        '## 학습량과 진행','',
        '| 조건 | 학습 상태 | 실제 경과(분) | 성공 update | 본 샘플 | 고유 프레임 | subset 완료 epoch | 전체 Test 구간 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for n in TRAIN_NAMES:
        run=out/'runs'/n;p=read(run/'complete.json') if (run/'complete.json').exists() else (read(run/'progress.json') if (run/'progress.json').exists() else {})
        count=sum((out/'results'/n/t['session']/'complete.json').exists() for t in read(out/'test/tasks.json')) if (out/'test/tasks.json').exists() else 0
        status=p.get('status','학습 중' if p else '대기');elapsed=p.get('elapsed_seconds')
        lines.append('| '+LABELS[n]+' | '+status+' | '+('—' if elapsed is None else f'{elapsed/60:.1f}')+' | '+
            ' | '.join(str(p.get(k,'—')) for k in ('successful_steps','samples_seen','unique_frames_seen','completed_subset_epochs'))+f' | {count}/80 |')
    lines+=['','전체 Train body 원프레임과 고정 subset을 구분한다. subset6,501프레임·Train holdout288프레임이며 전체 Train1epoch 완료로 표기하지 않는다. '
        '완료 epoch는 drop_last를 적용한 subset 기준이다. 원본 초기화 출처·SHA는 original_pretrained_manifest.json에 별도 보존한다.', '',
        '## 보조 Test 지표','',
        '| 조건 | RPE-Z (mm) | 절대 MPJPE (mm) | pelvis 속도오차 (mm/s) | pelvis상대 관절 가속도오차 (mm/s²) |',
        '|---|---:|---:|---:|---:|']
    for n in EVAL_NAMES:
        v=comparison['metrics'].get(n,{}) if comparison else {}
        cells=['미평가' if v.get(k) is None else f'{v[k]:.2f}' for k in ('root_depth_error_mm','absolute_mpjpe_mm','pelvis_velocity_error_mmps','root_acceleration_error_mmps2')]
        lines.append('| '+LABELS[n]+' | '+' | '.join(cells)+' |')
    proof=out/'coverage_and_GT_provenance.json'
    if proof.exists():
        cov=read(proof);a=cov['roles']['training'];b=cov['roles']['validation']
        full=a['GT_valid_frames']+b['GT_valid_frames']
        lines+=['','## 실제 Train coverage와 checkpoint 출처','',
            f'분할 Train body 전체 유효 GT는 {full:,}프레임이다. 그중 최적화용51피험자·51세션은 유효GT '
            f'{a["GT_valid_frames"]:,}프레임(기준 clock {a["full_clock_frames"]:,}프레임), '
            f'Train holdout9피험자·9세션은 유효GT {b["GT_valid_frames"]:,}프레임이다. '
            f'고정 학습 subset {a["selected_frames"]:,}프레임은 전체 유효Train의{100*a["selected_frames"]/full:.4f}%, '
            f'최적화용 유효GT의{100*a["selected_frames"]/a["GT_valid_frames"]:.4f}%다. '
            f'실제로 본 고유프레임 수와 성공 update는 위의 각 학습 행으로 따로 기록한다.',
            '원본 released checkpoint는 original_pretrained_manifest.json의 실제 SHA로 동결했다. '
            '모든 새 학습은 동일 원본에서 시작하며 이전 ETRI fine-tuned delta는 적재하지 않는다. '
            '각 결과 checkpoint는 해당 조건 checkpoints/<조건>/best.pt이고, Train holdout으로 선택한다. '
            'best/last를 별도로 보존하며 최종 실제 CPU load·SHA·optimizer update 검증을 한다.']
    lines+=['','## 검증과 해석','',
        '첫 프레임 정렬은 정확한 frame0에서만 변환을 구해 구간 전체에 고정한다. frame-aligned는 매프레임 다시 계산한다. '
        'MPJPE는 pelvis 평행이동, PA-MPJPE는 반사 없는 proper Sim3다. 다섯 주지표는 같은 검출·공통 GT와 frame0 유효 표본을 사용하며 미검출을0오차로 넣지 않는다.',
        '센서별 검출·좌표변환·매칭을 유지한다. event-only는 학습·검출·추론에 RGB 픽셀을 읽지 않는다. GTbox·pose·depth는 추론입력이 아니다.',
        '전체11조건 완료 후 동일 GPU 순차 온라인/캐시 전체파이프라인 속도, 피험자 cluster bootstrap95%CI·Holm·효과크기와 동작속도별 결과를 추가한다. '
        '12시간 내 미완료 항목은 그대로 미평가로 남기고 본인 작업만 checkpoint 보존 중단 후 Goal을 pause한다. 실제 개선을 보장하거나 결과를 만들지 않는다.']
    if (out/'final_report_addendum.md').exists():lines+=['',(out/'final_report_addendum.md').read_text()]
    report.parent.mkdir(parents=True,exist_ok=True);tmp=report.with_suffix('.tmp.md');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(report)
    write(out/'progress_table.json',dict(updated_unix=time.time(),test_complete=comparison['conditions'] if comparison else [],
        common_GT_samples=common,pipeline=state,report=str(report)))

def main():
    g=read(GOAL);register_job(g,os.getpid(),'encoding_report_30min')
    while time.time()<g['deadline_unix']:
        snapshot(g)
        if (Path(g['output_root'])/'final_artifact_audit.json').exists():return
        end=min(time.time()+1800,g['deadline_unix'])
        while time.time()<end:time.sleep(min(30,end-time.time()))
if __name__=='__main__':main()

"""Thirty-minute Korean report updates; never fit or choose on Test."""
from datetime import datetime, timezone, timedelta
import json
import os
import re
from pathlib import Path
import time
from goal_runtime_control import register_job, write
ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json'
KST = timezone(timedelta(hours=9))
MARKER = '<!-- FUSION_REPAIR_PROGRESS_START -->'
MAIN_START = '<!-- FUSION_REPAIR_MAIN_TABLE_START -->'
MAIN_END = '<!-- FUSION_REPAIR_MAIN_TABLE_END -->'
from fusion_repair_scope import LABELS as SCOPE_LABELS,EVAL_NAMES,TRAIN_NAMES,PRETRAIN_NAMES,PRIMARY_POPULATION
from fusion_repair_test_progress import completed_test_snapshot
LABELS={n:SCOPE_LABELS[n] for n in EVAL_NAMES}
METRICS = ['first_frame_aligned_mpjpe_mm', 'first_frame_aligned_pa_mpjpe_mm',
           'frame_aligned_mpjpe_mm', 'frame_aligned_pa_mpjpe_mm', 'pelvis_error_mm']


def read(path):
    return json.loads(path.read_text())


def snapshot(goal):
    out = Path(goal['output_root']); report = Path(goal['report'])
    if (out / 'final_artifact_audit.json').exists():
        return
    pipeline = read(out / 'pipeline_state.json') if (out / 'pipeline_state.json').exists() else {}
    gpu = read(out / 'gpu_preflight.json')
    comparison = read(out / 'comparison.json') if (out / 'comparison.json').exists() else None
    interim = None if comparison else completed_test_snapshot(out)
    rows = []
    for name in LABELS:
        run = out / 'runs' / name
        complete = read(run / 'complete.json') if (run / 'complete.json').exists() else None
        progress = read(run / 'progress.json') if (run / 'progress.json').exists() else {}
        validation = None; source = None
        if complete and complete['status'].startswith('complete') and (run / 'best.json').exists():
            validation = read(run / 'best.json'); source = 'Train-selected checkpoint holdout'
        elif (run / 'validation.jsonl').exists():
            values = (run / 'validation.jsonl').read_text().splitlines()
            if values:
                validation = json.loads(values[-1]); source = 'latest Train holdout'
        rows.append(dict(name=name, complete=complete, progress=progress, validation=validation,
                         validation_source=source, test_segments_completed=sum(
                             (out / 'results' / name / t['session'] / 'complete.json').exists()
                             for t in read(out / 'test/tasks.json'))))
    stamp = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')
    lines = ['## 현재 실행 및 평가 표', '', f'갱신: {stamp}. 30분 간격.', '',
             'GPU 실제 CUDA 검사를 통과한 장치: ' + (', '.join(map(str, gpu['usable_gpus'])) or '없음. 드라이버 초기화 실패로 새 학습 대기 중.'),
             '본학습 완료: ' + str(len(pipeline.get('training_complete', []))) + '/7, 전체 Test 완료: ' + str(len(pipeline.get('test_complete', []))) + '/9.', '',
             '최신 사용자 지시: 원본 RGB 두 조건(ETRI 추가학습0), 단축 RGB-only 두 조건,3시간 융합 다섯 조건. 학습량 차이를 포함하는 비교이며 실제 학습량·checkpoint 출처를 별도로 보고한다.','',
             '아래 Test 표는 새 방법의 전체 Test 공통 GT 재계산을 마친 값만 표시한다. 기존 결과와 내부 검증 수치로 채우지 않는다.', '',
             '| 조건 | first frame-aligned MPJPE (mm) | first frame-aligned PA-MPJPE (mm) | frame-aligned MPJPE (mm) | frame-aligned PA-MPJPE (mm) | Root position error (mm) |',
             '|---|---:|---:|---:|---:|---:|']
    for name, label in LABELS.items():
        values = comparison['populations'][PRIMARY_POPULATION][name] if comparison else (interim['metrics'].get(name, {}) if interim else {})
        cells = []
        for key in METRICS:
            value = values.get(key); text = '미평가' if value is None else f'{value:.2f}'
            if comparison or (interim and set(interim['conditions']) == set(LABELS)):
                others = [comparison['populations'][PRIMARY_POPULATION][n][key] if comparison else interim['metrics'][n][key] for n in LABELS]
                if all(v is not None for v in others) and value == min(others):
                    text = '**' + text + '**'
            cells.append(text)
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Root position error는 정렬 전 공통 카메라의 full3D pelvis/root 오차이며 RPE-Z가 아니다.', '',
              '### Train 분리 검증', '', 'Test 성능이 아니다. 학습에 사용하지 않은 Train holdout 288프레임에서 측정한다. MPJPE는 pelvis 정렬, PA-MPJPE는 proper Sim3 정렬이다. 학습을 마친 조건은 holdout 선택 checkpoint의 값이며 학습 중인 조건은 최신 정기 검증값이다. 검출된 GT 표본이 조건마다 다를 수 있어 이 표만으로 공통 GT 성능 순위를 단정하지 않는다.', '',
              '| 조건 | 검증 MPJPE (mm) | 검증 PA-MPJPE (mm) | 검출/GT | 검증 시각 (KST) | 성공 update | 전체 Test 구간 |',
              '|---|---:|---:|---:|---|---:|---:|']
    for row in rows:
        if row['name'] in PRETRAIN_NAMES:continue
        v = row['validation']; p = row['complete'] or row['progress']
        mp = '미평가' if not v or v['root_mpjpe_mm'] is None else f"{v['root_mpjpe_mm']:.2f}"
        pa = '미평가' if not v or v['pa_mpjpe_mm'] is None else f"{v['pa_mpjpe_mm']:.2f}"
        stamp_v = datetime.fromtimestamp(v['time'], KST).strftime('%m-%d %H:%M:%S') if v else '미평가'
        coverage = f"{v['matched_instances']}/{v['valid_instances']}" if v else '미평가'
        counts = [x['successful_steps'] for x in (p, v or {}) if 'successful_steps' in x]
        updates = max(counts) if counts else '대기'
        lines.append(f"| {LABELS[row['name']]} | {mp} | {pa} | {coverage} | {stamp_v} | {updates} | {row['test_segments_completed']}/80 |")
    # Keep the primary Test table first, including during automatic updates.
    table_start = next(i for i, line in enumerate(lines) if line.startswith('| 조건 | first frame-aligned'))
    table_end = table_start + 2 + len(LABELS)
    main_table = lines[table_start:table_end]
    del lines[table_start:table_end]
    original = report.read_text().split(MARKER)[0].rstrip()
    original = re.sub(re.escape(MAIN_START) + r'.*?' + re.escape(MAIN_END), '', original, flags=re.S)
    title, _, body = original.partition('\n')
    front = [MAIN_START, '## 전체 Test 주요 평가', '', *main_table, '',
             f'갱신: {stamp}. 전체 Test 완료 {len(pipeline.get("test_complete", []))}/9조건. '
             '전체 Test80구간을 마친 조건의 실측값부터 표시하고 나머지는 미평가로 둔다.', '',
             ('전체9조건 공통 GT 집계다. 속도·통계·최종 보고서 감사는 별도 진행한다.' if comparison or (interim and len(interim['conditions']) == len(LABELS)) else
              ('현재 완료 조건: ' + ', '.join(LABELS[n] for n in interim['conditions']) +
               f". 이 조건들 공통 GT {next(iter(interim['metrics'].values()))['samples']:,}개 기준이다. "
               '완료 조건이 추가되면 동일 공통 GT로 다시 계산하므로 이전 표와 표본이 달라질 수 있다.' if interim else
               '전체 Test를 마친 조건이 아직 없다.')), '',
             '단위는 mm. Root position error는 비정렬 full3D root 오차다. '
             'RGB-only90분·융합3시간으로 학습량 차이가 포함된다.', MAIN_END]
    temporary = report.with_suffix('.progress.tmp')
    temporary.write_text(title + '\n\n' + '\n'.join(front) + '\n\n' + body.strip() +
                         '\n\n' + MARKER + '\n\n' + '\n'.join(lines) + '\n')
    temporary.replace(report)
    write(out / 'progress_table.json', dict(updated_unix=time.time(), rows=rows, gpu=gpu,
                                          report=str(report), deadline_unix=goal['deadline_unix']))
    print(json.dumps(dict(updated=stamp, report=str(report))), flush=True)


def main():
    goal = read(GOAL)
    register_job(goal, os.getpid(), 'fusion_repair_report_30min')
    while time.time() < goal['deadline_unix']:
        snapshot(goal)
        if (Path(goal['output_root']) / 'final_artifact_audit.json').exists():
            return
        end = min(time.time() + 1800, goal['deadline_unix'])
        while time.time() < end:
            time.sleep(min(30, end - time.time()))


if __name__ == '__main__':
    main()

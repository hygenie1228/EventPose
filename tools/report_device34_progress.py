"""Read-only 30-minute experiment observer; never evaluates or selects models."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261026/device34_body_28h'
GOAL=ROOT/'_agents/research/261026/device34_body_goal_28h.json'
REPORT=ROOT/'_agents/report/261026_신체동작_두장치_여섯조건_비교.md'
PROGRESS_MARKER='<!-- DEVICE34_PROGRESS_START -->'
LABELS={'rgb3':'device3 RGB','rgb4':'device4 RGB','rgb3_rgb4':'device3 RGB + device4 RGB',
        'rgb3_event4':'device3 RGB + device4 event','event3_rgb4':'device3 event + device4 RGB',
        'event3_event4':'device3 event + device4 event'}
KEYS=['first_frame_aligned_mpjpe_mm','first_frame_aligned_pa_mpjpe_mm',
      'frame_aligned_mpjpe_mm','frame_aligned_pa_mpjpe_mm','pelvis_error_mm']


def read(path):return json.loads(path.read_text())


def collect_rows(names,tasks):
    rows=[]
    for name,label in names.items():
        run=OUT/'runs'/name;progress=run/'progress.json';validation=run/'validation.jsonl'
        p=read(progress) if progress.exists() else {}
        values=validation.read_text().splitlines() if validation.exists() else []
        v=json.loads(values[-1]) if values else None
        complete=read(run/'complete.json') if (run/'complete.json').exists() else None
        validation_source='latest_periodic_holdout'
        if complete and complete.get('status')=='complete':
            p={**p,**complete}
            # Final validation need not be appended to validation.jsonl.
            # Use the existing Train-selected record, never select on Test.
            if (run/'best.json').exists():
                v=read(run/'best.json')
                validation_source='completed_training_selected_checkpoint_holdout'
        completed=sum((OUT/'results'/name/t['session']/'complete.json').exists() for t in tasks)
        rows.append(dict(name=name,label=label,progress=p,validation=v,
            validation_source=validation_source,test_segments_completed=completed))
    return rows


def snapshot():
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    tasks=read(OUT/'test/tasks.json') if (OUT/'test/tasks.json').exists() else []
    comparison=read(OUT/'comparison.json') if (OUT/'comparison.json').exists() else None
    rows=collect_rows(LABELS,tasks)
    repeats=collect_rows({name+'_seed261027':label for name,label in LABELS.items()},tasks)
    lines=['# 신체 동작 여섯 조건 중간 평가','',f'갱신: {stamp}. 30분 간격으로 읽기 전용 집계한다.',
        '', '## 전체 Test 성능','',
        '여섯 조건의 전체 Test와 공통 표본 재계산을 마친 값만 기입한다. 미평가를 0으로 채우지 않는다.', '',
        '| 조건 | first frame-aligned MPJPE (mm) | first frame-aligned PA-MPJPE (mm) | frame-aligned MPJPE (mm) | frame-aligned PA-MPJPE (mm) | Root position error (mm) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[]
        for key in KEYS:
            value=comparison['populations']['common6'][row['name']][key] if comparison else None
            cell='미평가' if value is None else f'{value:.2f}'
            others=[comparison['populations']['common6'][n][key] for n in LABELS] if comparison else []
            if others and all(v is not None for v in others) and value==min(others):cell='**'+cell+'**'
            values.append(cell)
        lines.append('| '+row['label']+' | '+' | '.join(values)+' |')
    lines+=['','Root position error는 정렬 전 공통 카메라의 full 3D root 위치 오차이며 RPE-Z와 다르다.',
        '', '## Train 내부 holdout 학습 점검','',
        'Test 성능이 아니다. 같은 고정 288프레임 holdout의 실제 검출 평가다. '
        '학습 중인 조건은 최신 정기 평가, 학습을 마친 조건은 Train holdout으로 선택한 checkpoint 평가를 표시한다. '
        '학습 경과와 검출 표본이 달라 현재 최우수 조건을 선언하지 않는다. '
        'MPJPE는 pelvis 평행이동, PA-MPJPE는 프레임별 proper Sim3다. 정확 frame0 정렬 지표와 root 위치 오차는 이 평가에서 계산하지 않는다.', '',
        '| 조건 | 표시 holdout 평가 update | MPJPE (mm) | PA-MPJPE (mm) | 검출률 (%) | 현재 성공 update | Test 완료 구간 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        v=row['validation'];p=row['progress']
        cells=['미평가']*4 if v is None else [str(v['step'])]+[
            '미평가' if v[k] is None else f'{v[k]*scale:.2f}'
            for k,scale in [('root_mpjpe_mm',1),('pa_mpjpe_mm',1),('coverage',100)]]
        lines.append('| '+row['label']+' | '+' | '.join(cells)+f" | {p.get('successful_steps','대기')} | {row['test_segments_completed']}/{len(tasks)} |")
    lines+=['','## 추가 seed261027: Train 검증 및 Test 진행','',
        '첫 실험과 독립적으로 원본 가중치에서 다시 학습한다. 아래 수치는 Train 내부 holdout이며 전체 Test 주표와 구분한다. 첫 검증 전에는 미평가로 표시한다.','',
        '| 조건 | 표시 holdout 평가 update | MPJPE (mm) | PA-MPJPE (mm) | 검출률 (%) | 현재 성공 update | Test 완료 구간 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in repeats:
        v=row['validation'];p=row['progress']
        cells=['미평가']*4 if v is None else [str(v['step'])]+[
            '미평가' if v[k] is None else f'{v[k]*scale:.2f}'
            for k,scale in [('root_mpjpe_mm',1),('pa_mpjpe_mm',1),('coverage',100)]]
        lines.append('| '+row['label']+' | '+' | '.join(cells)+f" | {p.get('successful_steps','대기')} | {row['test_segments_completed']}/{len(tasks)} |")
    text='\n'.join(lines)+'\n'
    if comparison and REPORT.exists():
        # Preserve the full audited primary report while repeat trials run.
        # The final audit hashes this same file; never modify it afterwards.
        original=REPORT.read_text().split(PROGRESS_MARKER)[0].rstrip()
        holdout=lines.index('## Train 내부 holdout 학습 점검')
        text=original+'\n\n'+PROGRESS_MARKER+'\n\n## 진행 상황 갱신\n\n'+stamp+'\n\n'+'\n'.join(lines[holdout:])+'\n'
    if not (OUT/'final_artifact_audit.json').exists():
        temporary=REPORT.with_suffix('.progress.tmp');temporary.write_text(text);temporary.replace(REPORT)
    status=OUT/'progress_table.json';temporary=status.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(updated_unix=time.time(),report=str(REPORT),conditions=rows,replications=repeats),ensure_ascii=False,indent=2)+'\n');temporary.replace(status)
    print(json.dumps(dict(updated=stamp,report=str(REPORT)),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    deadline=read(GOAL)['deadline_unix']
    while True:
        if (OUT/'final_artifact_audit.json').exists():return
        snapshot()
        if not args.watch or (OUT/'final_artifact_audit.json').exists() or time.time()>=deadline:return
        next_check=min(time.time()+1800,deadline)
        while time.time()<next_check:time.sleep(min(30,next_check-time.time()))


if __name__=='__main__':main()

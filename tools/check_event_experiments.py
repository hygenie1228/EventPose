"""One manual 30-minute progress observation; never launches or stops jobs."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.goal_runtime_control import identity, write

OUT = ROOT / 'exp/261024/rgb_event_36h'
NOTE = ROOT / '_agents/research/261024'


def read(path):
    return json.loads(path.read_text())


def utc(value):
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def main():
    now = time.time()
    previous = NOTE / 'rgb_event_last_progress_check.json'
    if previous.exists():
        assert now >= read(previous)['observed_unix'] + 1800, '30-minute observation interval has not elapsed'
    state = read(OUT / 'pipeline_state.json')
    rows = []
    for job in state['jobs']:
        current = identity(job['pid'])
        row = {key: job[key] for key in ('kind', 'name', 'gpu', 'pid')}
        row['live_identity_match'] = bool(current and all(current[key] == job[key]
                                                          for key in ('start_ticks', 'command', 'pgid')))
        if job['kind'] == 'train':
            for name in ('progress.json', 'best.json'):
                path = OUT / 'runs' / job['name'] / name
                if path.exists():
                    row[name] = read(path)
        else:
            row['completed_segments'] = len(list((OUT / 'results' / job['name']).glob('*/complete.json')))
        rows.append(row)
    summary_path = OUT / 'event_build_summary.json'
    cache = read(summary_path if summary_path.exists() else OUT / 'event_build_status.json')
    cache = {key: value for key, value in cache.items() if key not in ('results', 'config')}
    managers = {str(pid): identity(pid) for pid in (71605, 86503, 15088)}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                   '--format=csv,noheader'], text=True)
    record = dict(observed_unix=now, observed_utc=utc(now), next_check_not_before_utc=utc(now + 1800),
                  jobs=rows, failed=state['failed'], cache=cache, managers=managers, gpu_observation=gpu,
                  training_complete=state['training_complete'], test_complete=state['test_complete'])
    write(previous, record)
    handoff = NOTE / 'rgb_event_implementation_handoff.json'
    value = read(handoff)
    value.update(updated_unix=now, last_progress_check_utc=utc(now),
                 next_progress_check_not_before_utc=utc(now + 1800), last_observed_jobs=rows,
                 cache_last_observed=cache)
    write(handoff, value)
    report = ROOT / '_agents/report/261024_학습과_추론_속도_중간_비교.md'
    marker = '<!-- RGB_EVENT_PROGRESS -->'
    text = report.read_text().split(marker)[0].rstrip() + '\n\n' + marker
    text += '\n\n## RGB·이벤트 확장 실험 진행\n\n' + utc(now)
    text += ' 실제 프로세스 확인 기준이다. 기존4모델과 별개의 통제 실험이다. 아래 내부 holdout은 Test가 아니며, 학습 경과가 달라 최종 우열을 뜻하지 않는다.\n\n'
    text += '| GPU | 조건 | 작업 | 성공 업데이트 | 내부 holdout PA (mm) |\n|---|---|---|---:|---:|\n'
    brief = []
    for row in rows:
        best = row.get('best.json', {})
        pa = f"{best['pa_mpjpe_mm']:.2f}" if best else '—'
        steps = row.get('progress.json', {}).get('successful_steps', '—')
        task = '학습' if row['kind'] == 'train' else str(row['completed_segments']) + '/320구간 Test'
        text += f"| {row['gpu']} | {row['name']} | {task} | {steps} | {pa} |\n"
        brief.append(dict(name=row['name'], gpu=row['gpu'], live=row['live_identity_match'],
                          successful_steps=steps, holdout_pa_mm=best.get('pa_mpjpe_mm'), task=task))
    for name in state['test_complete']:
        if not any(row['name'] == name for row in rows):
            text += f'| — | {name} | 전체 Test 320/320구간 완료 | — | — |\n'
    text += f"\n캐시 {cache.get('sessions', cache.get('complete_sessions'))}/283세션. "
    if (OUT / 'cache_final_audit.json').exists():
        text += '283세션 원본 매핑·타임스탬프·인과성·보정값 검증 통과. '
    text += f"실패 기록: {state['failed'] or '없음'}. 다음 정기 확인은 {utc(now + 1800)} 이후다. 고정 마감은 2026-10-26 05:17:17 UTC다.\n"
    report.write_text(text)
    print(json.dumps(dict(observed=utc(now), next_check=utc(now + 1800), jobs=brief,
                          failed=state['failed'], cache=cache, managers_live={key: bool(value) for key, value in managers.items()},
                          training_complete=state['training_complete'], test_complete=state['test_complete']), ensure_ascii=False))


if __name__ == '__main__':
    main()

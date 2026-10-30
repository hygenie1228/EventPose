"""Korean report generated only from completed event comparison artifacts."""
import json
import datetime
from pathlib import Path
from .report_format import highlight_table_bests
from .chunk_alignment import METRICS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'exp/261024/rgb_event_36h'
REPORT = ROOT / '_agents/report/261024_RGB와_이벤트_최종_비교.md'


def number(value, digits=2):
    return 'NA' if value is None else f'{value:.{digits}f}'


def render(result, runtime, training, prior=None, cache=None):
    labels = {'rgb_original': 'RGB 원본', 'rgb_baseline': 'RGB 미세조정'}
    for mode, label in [('event', '이벤트 단독'), ('fusion', 'RGB+이벤트')]:
        for encoding in ['count33', 'bins100', 'surface20', 'visualize_gray']:
            labels[mode + '_' + encoding] = label + ' ' + encoding
    names = result['conditions']
    population = result['populations']['all10']
    lines = ['# RGB·이벤트 기반 3D 인체 움직임 최종 비교', '',
        'device0만 사용했다. Train 219세션·68명 중 187세션은 학습, 32세션은 내부 검증이며, '
        'Test 64세션·20명은 체크포인트·인코딩 선택에 사용하지 않았다. '
        '기존 RGB Test 결과를 알고 Multi-HMR 계열을 선택했으므로 완전히 새로운 비공개 Test 검증은 아니다.', '',
        '고정 작업창은 2026-10-24 17:17:17~2026-10-26 05:17:17 UTC다. '
        '이전 NLF 비교·오류 복구·구현·캐시 준비도 이 36시간 예산에 포함했다.', '',
        '## 기존 모델 계열 비교', '',
        '기존 4모델의 전체 Test·동일 GPU 결과는 [학습과 추론 속도 비교]'
        '(/home/namhj/EventPose/_agents/report/261024_학습과_추론_속도_중간_비교.md)와 '
        '[전체 모델 비교](/home/namhj/EventPose/_agents/report/261024_인체자세추정_모델_성능_비교.md)에 보존했다. '
        '자세·절대 위치·검출률과 속도를 근거로 Multi-HMR을 선택했다. '
        '본 실험은 기존 학습 가중치를 재활용하지 않고 모두 같은 원본 가중치에서 시작했다.', '',
        '## 10조건의 동일 표본 자세 비교', '',
        f"전체 Test는 320구간·{result['rgb_frames']:,} RGB 시각, 유효 GT {result['gt_instances']:,}명-프레임이다. "
        f"모든 10조건에서 공통 매칭되고 정확한 구간 frame0에 검출된 표본은 {population[names[0]]['samples']:,}개다. "
        '검출률은 이 교집합이 아닌 전체 유효 GT를 분모로 한다. 미검출은 0오차가 아니다. '
        '교집합이 비면 자세 오차를 NA로 표시하고 아래 RGB/fusion 쌍별 결과로 판단한다.', '',
        '| 조건 | first frame-aligned MPJPE (mm) | first frame-aligned PA-MPJPE (mm) | frame-aligned MPJPE (mm) | frame-aligned PA-MPJPE (mm) | Root absolute depth error (RPE-Z, mm) |',
        '|---|---:|---:|---:|---:|---:|']
    if prior is not None:
        old_labels = {'Multi-HMR pretrained': 'Multi-HMR 기본', 'NLF pretrained': 'NLF 기본',
                      'Multi-HMR fine-tuned': 'Multi-HMR 학습', 'NLF fine-tuned': 'NLF 학습'}
        old_runtime = {'Multi-HMR pretrained': 'multihmr_pretrained', 'NLF pretrained': 'nlf_pretrained',
                       'Multi-HMR fine-tuned': 'multihmr_finetuned', 'NLF fine-tuned': 'nlf_finetuned'}
        block = [f"기존 비교는 네 모델 공통 {prior['common_matched_instances']:,}명-프레임이다. "
                 '새 10조건과 평가 교집합·학습 예산이 달라 표 사이의 직접 순위 비교는 하지 않는다.', '',
                 '| 모델 | 프레임별 PA (mm) | 절대 MPJPE (mm) | GT 매칭률 (%) | 전체 fps |',
                 '|---|---:|---:|---:|---:|']
        for name, row in prior['methods'].items():
            block.append(f"| {old_labels[name]} | {number(row['pa_mpjpe_mm'])} | {number(row['mpjpe_mm'])} | "
                         f"{number(100*row['coverage'],3)} | {number(prior['controlled_runtime'][old_runtime[name]]['fps'],3)} |")
        index = lines.index('## 10조건의 동일 표본 자세 비교')
        lines[index:index] = block + ['']
    for name in names:
        lines.append('| ' + labels[name] + ' | ' + ' | '.join(number(population[name].get(key)) for key in METRICS+['root_depth_error_mm']) + ' |')
    lines += ['', '첫 frame0 정렬 변환은 10초 구간 전체에 고정한다. MPJPE는 pelvis 평행이동, '
        'PA는 반사를 허용하지 않는 양의 등방 scale·회전·이동이다. 실제 metric scale 위치는 다음 정렬 전 오차로 확인한다.', '',
        '| 조건 | 절대 MPJPE (mm) | Pelvis 위치 (mm) | GT 매칭률 (%) | Pelvis 속도 (mm/s) | Root 가속도 (mm/s²) |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        row = population[name]
        lines.append('| ' + labels[name] + ' | ' + ' | '.join([
            number(row['absolute_mpjpe_mm']), number(row['pelvis_error_mm']),
            number(100 * result['coverage'][name]['fraction'], 3),
            number(row['pelvis_velocity_error_mmps']), number(row['root_acceleration_error_mmps2'])]) + ' |')
    lines += ['', '시간적 오차는 동일 인물의 연속 원본 프레임만 사용하며 결측·구간 사이를 연결하지 않는다. '
        '평활화는 적용하지 않았다. GT 잡음의 영향은 남는다.', '', '## 모델별 검출 표본의 보조 오차', '',
        '다음 표는 각 모델 자체의 frame0 적격 검출 표본이다. 서로 다른 쉬운/어려운 표본이 섞이므로 '
        '이 표만으로 모델 순위를 정하지 않는다. 공통 표본이 작거나 없는 경우에도 실제 관측 성능을 숨기지 않기 위한 보조 자료다.', '',
        '| 조건 | 자체 표본 | 프레임별 PA-MPJPE (mm) | 절대 MPJPE (mm) | Pelvis (mm) |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        row = result['populations']['own_' + name][name]
        lines.append(f"| {labels[name]} | {row['samples']} | {number(row['frame_aligned_pa_mpjpe_mm'])} | "
                     f"{number(row['absolute_mpjpe_mm'])} | {number(row['pelvis_error_mm'])} |")
    lines += ['', '## RGB baseline과 RGB+이벤트의 대응 비교', '',
        '| 인코딩 | 공통 표본 | 독립 cluster | RGB PA (mm) | Fusion PA (mm) | 차이 (mm) | 차이 (%) | 95% CI (mm) | Holm p |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for name, test in result['primary_tests'].items():
        interval = 'NA' if test['ci95'] is None else '[' + ', '.join(number(v) for v in test['ci95']) + ']'
        lines.append('| ' + labels[name] + ' | ' + ' | '.join([str(test['samples']), str(test['clusters']),
            number(test.get('reference_mean')), number(test.get('variant_mean')), number(test['effect']),
            number(test.get('relative_percent')), interval, number(test['holm_p_value'], 5)]) + ' |')
    lines += ['', '음수 차이는 이벤트 조건의 오차 감소다. 인코딩마다 해당 RGB/fusion 쌍의 공통 매칭·frame0 적격 표본을 사용하므로 '
        '위 10조건 교집합 수치와 직접 같지 않다. 공동 출현·반복 피험자를 연결한 최대 11개 독립 성분을 통째로 '
        '10,000회 bootstrap하고 양측 cluster sign-flip 검정을 했다. 4개 주 자세 비교에 Holm 보정을 적용했다. '
        '프레임 수를 독립 표본 수로 사용하지 않았다.', '',
        '| 시간적 보조 비교 | 차이 | 95% CI | Holm p |', '|---|---:|---|---:|']
    for key, test in result['secondary_tests'].items():
        interval = 'NA' if test['ci95'] is None else '[' + ', '.join(number(v) for v in test['ci95']) + ']'
        lines.append(f"| {key} | {number(test['effect'])} | {interval} | {number(test['holm_p_value'],5)} |")
    lines += ['', '시간적 보조 검정 8개는 별도 Holm family다. 속도 차이 단위는 mm/s, 가속도는 mm/s²다.', '',
        '## 정지·느린동작·빠른동작', '',
        '| Fusion 조건 | 구간 | 공통 표본 | RGB PA (mm) | Fusion PA (mm) |', '|---|---|---:|---:|---:|']
    for variant, bins in result['motion_strata'].items():
        for label, members in bins.items():
            counts = [sum(row['samples'] for row in members[name]) for name in ['rgb_baseline', variant]]
            assert counts[0] == counts[1]
            means = [sum(row['pa_sum_mm'] for row in members[name]) / counts[0] if counts[0] else None
                     for name in ['rgb_baseline', variant]]
            lines.append(f"| {labels[variant]} | {label} | {counts[0]} | {number(means[0])} | {number(means[1])} |")
    lines += ['', 'GT 신체22관절 평균 절대 이동 속도의 중앙차분으로 정지 ≤0.05m/s, 빠른동작 ≥0.5m/s를 사전 고정했다. '
        'unknown은 경계·누락으로 속도를 정의하지 못한 표본이다. 사람의 행동 주석이 아닌 GT 기반 보조 분석이다.', '',
        '## 동일 GPU 전체 처리 속도', '',
        '| 조건 | 캐시 fps | 온라인 fps | 모델 로딩 (초) | 최대 할당 VRAM (GiB) |', '|---|---:|---:|---:|---:|']
    identities = set()
    for name in names:
        cached = runtime[name + '_cached']; online = runtime.get(name + '_online')
        identities.add(cached['gpu_identity'])
        if online:
            identities.add(online['gpu_identity'])
        memory = max(row['peak_allocated_bytes'] for row in cached['segments']) / 1024**3
        lines.append(f"| {labels[name]} | {number(cached['fps'],3)} | {number(online['fps'] if online else None,3)} | "
                     f"{number(cached['load_seconds'])} | {number(memory)} |")
    assert len(identities) == 1
    lines += ['', 'GPU: `' + next(iter(identities)) + '`. 모든 조건은 동일 100프레임 구간 3개·batch16을 순차 처리했다. '
        '영상 staging·디코드·이벤트 파일 열기·인코딩 또는 캐시 I/O·검출·인체 복원·GT 매칭·저장을 포함한다. '
        'event-only는 RGB 파일을 복사·디코드하지 않는다. 로딩은 별도이며 프로세스 전체 시간과 온라인 인코딩 세부 시간은 원시 JSON에 보존했다. '
        'OS 캐시는 강제 비우지 않았고 짧은 측정의 초기 실행 영향이 남는다.', '', '## 학습 예산과 출처', '',
        '| 조건 | 실제 경과 (시간) | 성공 업데이트 | 선택 checkpoint |', '|---|---:|---:|---|']
    for name in names:
        row = training.get(name)
        lines.append(f"| {labels[name]} | {number(row['elapsed_seconds']/3600) if row else '—'} | "
                     f"{row['successful_steps'] if row else '—'} | `{row['checkpoint'] if row else '원본 multiHMR_896_L.pt'}` |")
    lines += ['', '모든 학습 조건은 같은 원본 초기화, seed261024, 동일 115,838 Train 프레임·2,045 holdout 프레임, '
        'batch16·RGB 색상 증강·AdamW·AMP를 사용했다. Head/adapter를 먼저 학습하고 30분 후 후반8 backbone을 '
        '작은 학습률로 업데이트했다. 약12시간의 학습 경과 예산을 맞췄으며 처리량 차이로 업데이트 수는 다르다. '
        '이 시간에는 검증·저장·로딩이 포함되며 실제 순수 GPU 연산시간과는 다르다. '
        'holdout만으로 checkpoint를 골랐다. 이벤트 입력은 4가지 동결된 인코딩이며 RGB와 동일 픽셀로 겹치지 않고 '
        '센서 내외부 파라미터를 반영한 분리 encoder/fusion을 사용했다. 이벤트 단독은 RGB 검출·crop·GT 입력을 사용하지 않는다.', '',
        '저장된 희소 이벤트 프레임을 사용했으며 원시 이벤트의 개수·세부 시각을 복원한 것은 아니다. '
        '미래 이벤트를 제외하고 극성 채널을 분리했다. visualize_gray는 배경128·양255·음0의 최근 과거 프레임이며 '
        '손실 MP4나 frame_idx×50 동기화를 사용하지 않았다. 캐시는 exp 하위에 두고 dataset/data는 수정하지 않았다.', '',
        '반복 seed 실시 여부·최종 검증은 아래 별도 검증 자료로 확정해야 한다. '
        'cluster CI는 학습 seed 변동을 측정하지 않는다.', '', '## 추천 판단', '',
        '이 보고서의 자동 집계만으로 최종 추천을 확정하지 않는다. 유의한 paired 자세 개선, 전체 GT 검출률, '
        '절대 위치·시간적 일관성, 온라인/캐시 속도 비용과 반복 seed 근거를 대조해 한 모델을 최종 검증 보고서에서 확정한다.', '',
        '원시 근거: `exp/261024/rgb_event_36h/comparison.json`, `comparison_segment_sums.json`, '
        '`runtime/summary.json`, `statistical_analysis_plan.json`. 최종 원시 재계산·완료 감사 전에는 Goal 완료를 의미하지 않는다.']
    if cache is not None:
        index = lines.index('## 추천 판단')
        lines[index:index] = ['## 이벤트 캐시 비용', '',
            '| 캐시 세션 | 용량 (GiB) | 생성 경과 (시간) | 세션 작업 경과 합 (시간) |',
            '|---:|---:|---:|---:|',
            f"| {cache['sessions']} | {number(cache['bytes']/1024**3)} | {number(cache['wall_seconds']/3600)} | {number(cache['summed_task_seconds']/3600)} |", '',
            '4개 CPU worker를 사용했다. 세션 작업 경과 합은 병렬 작업별 wall time의 합이며 CPU 코어 시간 측정값은 아니다. '
            '이 표는 수정된 v2 최종 캐시의 생성 비용이다. 폐기 보존한 v1 준비 비용은 전체36시간 경과에 별도로 포함된다. '
            '캐시 fps에는 이 일회성 생성시간이 제외되므로 온라인 fps와 함께 판단해야 한다.', '']
        extension = cache.get('execution_extension')
        if extension:
            utc = lambda value: datetime.datetime.fromtimestamp(value, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            index = lines.index('## 학습 예산과 출처')
            block = ['## 중단 후 추가 실행 구간', '',
                f"사용자 승인에 따른 추가 구간은 **{utc(extension['started_unix'])}~{utc(extension['deadline_unix'])}**다. "
                '원래36시간 마감을 임의로 재설정한 것이 아니라, 사용자24시간 추가승인으로 대체했다. '
                '원래 결과와 중단 전 checkpoint를 보존하고 optimizer·AMP·RNG를 복원했다. '
                '사용자가 중단한 대기시간은 각 모델의12시간 학습량에 포함하지 않았다.', '',
                '| 조건 | 중단 전 경과 (시간) | 재개 후 경과 (시간) | 총 학습 경과 (시간) |',
                '|---|---:|---:|---:|']
            for name, prior_seconds in extension['prior_training_seconds'].items():
                total = training[name]['elapsed_seconds']
                assert total >= prior_seconds
                block.append(f"| {labels[name]} | {number(prior_seconds/3600)} | {number((total-prior_seconds)/3600)} | {number(total/3600)} |")
            block += ['', '저장 checkpoint 이후 중단까지의 짧은 미저장 업데이트는 복구되지 않을 수 있다. '
                '이 구간의 경과시간도 보수적으로 기존 예산에 포함했으므로 추가학습시간을 늘리는 데 사용하지 않았다. '
                '모델·optimizer·AMP·RNG를 복원했지만 DataLoader의 중간 epoch 프레임 순서를 bit 단위로 재현했다고 주장하지 않는다. '
                '물리GPU3의장치오류로 정상GPU2·4·5·6을 사용했다. CUDA정수순번과물리번호의차이를 실제UUID로 대조하고 새작업은UUID로지정했다.', '']
            lines[index:index] = block
    # Model-specific populations are not a fair ranking population.
    own_start = lines.index('## 모델별 검출 표본의 보조 오차')
    own_end = lines.index('## RGB baseline과 RGB+이벤트의 대응 비교')
    formatted = highlight_table_bests(lines[:own_start]) + lines[own_start:own_end] + highlight_table_bests(lines[own_end:])
    return '\n'.join(formatted) + '\n'


def main():
    read = lambda path: json.loads(path.read_text())
    result = read(OUT / 'comparison.json')
    runtime = read(OUT / 'runtime/summary.json')
    training = {name: read(OUT / 'runs' / name / 'complete.json') for name in result['conditions'] if name != 'rgb_original'}
    prior, cache = supporting_evidence()
    REPORT.write_text(render(result, runtime, training, prior, cache))


def supporting_evidence():
    read = lambda path: json.loads(path.read_text())
    prior = read(ROOT / 'exp/261024/nlf_finetune_12h/comparison.json')
    summary = read(OUT / 'event_build_summary.json')
    launch = read(ROOT / '_agents/research/261024/event_cache_launch.json')
    assert summary['sessions'] == 283 and summary['limited'] is False
    cache = dict(sessions=283, bytes=sum(row['bytes'] for row in summary['results']),
                 wall_seconds=summary['completed_unix'] - launch['started_unix'],
                 summed_task_seconds=sum(row.get('total_seconds', 0) for row in summary['results']))
    goal = read(ROOT / '_agents/research/261024/rgb_event_goal_36h.json')
    if goal.get('extension_started_unix'):
        conditions = read(OUT / 'experiment_manifest.json')['conditions']
        cache['execution_extension'] = dict(started_unix=goal['extension_started_unix'],
            deadline_unix=goal['deadline_unix'], prior_training_seconds={
                value['name']: value['resume_authorization']['prior_elapsed_seconds']
                for value in conditions if value.get('resume_authorization')})
    return prior, cache


if __name__ == '__main__':
    main()

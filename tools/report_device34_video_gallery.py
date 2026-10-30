"""Append the requested MP4 list without changing verified performance tables."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import fcntl
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from goal_runtime_control import read, write, register_job
START = '<!-- DEVICE34_TEST_VIDEO_LIST_START -->'
END = '<!-- DEVICE34_TEST_VIDEO_LIST_END -->'


def _update(goal):
    out = Path(goal['output_root']); stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    verification = read(out / 'video_verification.json') if (out / 'video_verification.json').exists() else {}
    verified = set(verification.get('verified_paths', [])); rendered = 0
    lines = ['## 대표 테스트 5구간의 MP4 목록', '', f'갱신: {stamp}. 같은5구간을9조건에서 비교한다.', '',
             '| 조건 | 구간1 | 구간2 | 구간3 | 구간4 | 구간5 |', '|---|---|---|---|---|---|']
    for name, model in goal['models'].items():
        cells = []
        for clip in goal['clips']:
            path = out / 'mp4' / name / (clip['clip']+'.mp4'); receipt = path.with_suffix('.json')
            if receipt.exists() and path.exists():
                rendered += 1
                label = 'MP4' if str(path) in verified else '생성본'
                cells.append(f'[{label}]({path})')
            else: cells.append('제작 중')
        lines.append('| '+model['label']+' | '+' | '.join(cells)+' |')
    lines += ['', f'생성 {rendered}/45개, 전체 디코딩 검증 {len(verified)}/45개. 생성본은 검증을 마치기 전까지 완료 영상으로 간주하지 않는다.', '',
              '각 영상은 RGB3·RGB4·event3/event4·실제 예측 SMPL-X mesh·RGB3 overlay·depth scene의6패널이다. '
              '모델이 사용하지 않은 RGB/event와 depth는 참고 화면이며 추론 입력에 추가하지 않는다. '
              'event는 실제 count33 입력을 극성별로 표시하고 가독성을 위한 표시 gain4만 적용한다. '
              '정량 표에 사용한 GT 정렬을 영상의 mesh에 적용하지 않는다.', '',
              '| 구간 | 피험자 | 세션·구간 | 동작 속도 중앙값 (m/s) | 선정 분위수 | 배경 point cloud |', '|---|---:|---|---:|---:|---|']
    for clip in goal['clips']:
        depth_path=Path(goal['work_root'])/'references'/(clip['clip']+'_depth.json')
        depth=read(depth_path) if depth_path.exists() else {}
        scene='실측 depth·단위 가정 표시' if depth.get('available') else ('없음: 원본 depth 미제공' if depth.get('source') and not Path(depth['source']).exists() else '확인 중')
        lines.append(f"| {clip['clip']} | {clip['participant']} | {clip['task']['session']} | {clip['motion_median_mps']:.4f} | {clip['motion_quantile_target']:.1f} | {scene} |")
    lines += ['', '선정은 전체 Test80구간의 GT body22 동작 속도 분위수와 피험자 다양성만 사용했다. 모델 오차가 좋은 사례를 고르지 않았다. '
              '동일 시각·동일 raw 예측을 비교하며 미검출 프레임은 숨기거나 GT mesh로 대체하지 않는다.', '',
              'Point cloud는 실제 depth의 첫 과거 프레임을 사용한 정적 장면이다. 프레임별 GT로 scene을 맞추지 않으며 '
              'RGB3 단독 예측의 이동 범위만 배경 분리와 공통 시점 설정에 사용한다. '
              'depth의 uint16 값은 lossless RGB byte packing을 풀어 복원한다. archive 단위 태그는 source_raw_unit이므로 '
              '제공된 예제의0.001 m/unit을 사용하는 스케일 가정을 명시한다. 독립 센서 단위 인증이나 완전한 다중시점 장면 복원이 아니다. '
              '유효한 depth·시간·보정을 확인하지 못한 구간에는 배경 point cloud를 만들었다고 표시하지 않는다.', '',
              '보관 depth 영상은 원본30Hz depth의5Hz 부분집합이다. MP4 프레임 번호와 라벨의 원본 프레임 번호를 같다고 가정하지 않고, '
              '제공된 보정 timestamp가 원본 timestamp와 정확히 일치하는지 대조한 뒤 첫 과거 depth를 선택했다. '
              '배경은 한 depth 시점의 부분 관측이므로 가림·깊이 결측으로 빈 영역이 남는다.', '',
              'Point cloud 색상은 클립 첫 RGB 참고 화면으로 입혔다. Depth는 그보다 앞선 시점일 수 있으므로 '
              '정적 배경을 가정한 표시이며, 예측 범위로 제거되지 않은 첫 시점의 인물 잔상이 일부 남을 수 있다.', '',
              '정량 평가표는 기존 검증 결과 그대로다. 영상 목록 추가 전 검증본은 '
              f"[보존본]({goal['source_report_snapshot']})에 있으며, 이번 영상 제작은 학습·checkpoint 선택·Test 성능표를 변경하지 않는다.",'']
    if verification.get('records'):
        drift=max(r['CFR_maximum_timestamp_difference_seconds'] for r in verification['records'])
        replay=max(r['maximum_joint_difference_from_published_Test_m'] for r in verification['records'])
        lines += [f'현재 검증한 영상의 고정 fps 재생 시각과 실제 source timestamp 사이 최대 차이는 {drift*1000:.3f}ms다. '
                  '영상 상단의 t는 실제 보정 timestamp를 사용하며, 각 패널은 같은 source 프레임에 대응한다. '
                  f'새 mesh 재추론에서 기존 Test의 매칭된 예측 joints와의 최대 좌표 차이는 {replay*1000:.6f}mm다. '
                  '이는 재현성 대조 오차이며 pose 정확도 지표가 아니다.','']
    body = '\n'.join(lines)
    report = Path(goal['original_report']); existing = report.read_text()
    prefix = existing.split(START)[0].rstrip()
    if '<!-- VIDEO_GALLERY_NAV -->' not in prefix:
        navigation='\n<!-- VIDEO_GALLERY_NAV -->\n실제 영상은 이 보고서 끝의 **대표 테스트 5구간의 MP4 목록**에서 확인할 수 있다. '
        navigation+=f"[영상 목록 바로 보기]({goal['gallery_report']}).\n"
        assert '\n고정15시간 범위:' in prefix
        prefix=prefix.replace('\n고정15시간 범위:',navigation+'\n고정15시간 범위:',1)
    text = prefix+'\n\n'+START+'\n\n'+body+'\n'+END+'\n'
    from render_fusion_repair import verify_tables
    previous = read(Path(goal['source_root']) / 'rendered_table_proof.json')
    verify_tables(text, previous['tables'])
    temporary = report.with_suffix('.videos.tmp'); temporary.write_text(text); temporary.replace(report)
    Path(goal['gallery_report']).write_text('# 센서융합 대표 테스트 구간 영상 목록\n\n'+body)
    write(out / 'report_progress.json', dict(updated_kst=stamp, rendered=rendered, verified=len(verified), expected=45,
                                            original_report=str(report), performance_tables_unchanged=True))


def update(goal):
    with (Path(goal['work_root']) / 'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _update(goal)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true'); args = parser.parse_args()
    goal = read(ROOT / '_agents/research/261028/device34_test_video_goal.json')
    if not args.once: register_job(goal, os.getpid(), 'video_gallery_report')
    while True:
        update(goal)
        if args.once or (Path(goal['output_root']) / 'final_verification.json').exists(): return
        time.sleep(1800)


if __name__ == '__main__': main()

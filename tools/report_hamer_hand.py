"""Korean report with the four measured hand metrics immediately after title."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261028/hamer_hand'
REPORT=ROOT/'_agents/report/261028_손_복원_하머_네조건_비교.md'
NAMES=['pre_rgb3','rgb3','rgb3_event4_count33','rgb3_event4_bins100','rgb3_event4_surface20']
LABELS=['HaMeR 원본 RGB3','HaMeR RGB3','HaMeR RGB3 + Event4 · count33',
        'HaMeR RGB3 + Event4 · bins100','HaMeR RGB3 + Event4 · surface20']


def main():
    m=json.loads((OUT/'manifest.json').read_text());complete=[];arrays={};keys=None
    for name in NAMES:
        if not (OUT/'results'/name/'complete.json').exists():continue
        assert json.loads((OUT/'results'/name/'complete.json').read_text())['sessions']==len(m['test_sessions'])
        records=[]
        for session in m['test_sessions']:
            with np.load(OUT/'results'/name/session/'predictions.npz') as z:
                this=[(session,*map(int,key)) for key in z['metric_keys']]
                records.extend(zip(this,z['errors_m']))
        assert len(records)==len({k for k,_ in records})
        arrays[name]=dict(records);complete.append(name)
    common=sorted(set.intersection(*[set(arrays[n]) for n in complete])) if complete else []
    measured={n:np.array([arrays[n][k] for k in common]) for n in complete}
    finished=len(complete)==len(NAMES)
    best=np.min([a.mean(0) for a in measured.values()],axis=0) if finished and common else None
    lines=['# GT 손 박스 기반 HaMeR 네 조건 복원 성능 비교','',
        '| 모델 | 첫 프레임 정렬 MPJPE (mm) | 첫 프레임 정렬 PA-MPJPE (mm) | 프레임 정렬 MPJPE (mm) | 프레임 정렬 PA-MPJPE (mm) |',
        '|---|---:|---:|---:|---:|']
    for name,label in zip(NAMES,LABELS):
        values=[]
        if name in measured and common:
            means=measured[name].mean(0)
            values=[f'**{v*1000:.3f}**' if best is not None and np.isclose(v,best[i],atol=1e-12,rtol=0) else f'{v*1000:.3f}' for i,v in enumerate(means)]
        else:values=['미평가']*4
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    lines+=['',f'갱신 시각: {now}. 전체 Test 완료 조건 {len(complete)}/{len(NAMES)}. 비교 공통 손 표본 {len(common):,}개.',
        '원본 pretrained 행은 별도 학습 실험이 아닌 초기 기준선이며, 학습 실험은 요청한 네 개다.',
        '평가 전 수치는 만들지 않으며 전체 조건이 끝나기 전에는 최우수값을 표시하지 않는다.','',
        '## 데이터와 평가 기준','',
        f'- `_h`: Train 63개 세션 중 학습 {len(m["training_sessions"])}개·검증 {len(m["validation_sessions"])}개, Test {len(m["test_sessions"])}개. 피험자는 body와 같은 기준으로 분리했다.',
        '- 사용자가 지정한 GT 손 박스 crop 기반 복원 평가다. 손 검출 성능은 포함하지 않는다. 실제 `device3/label.h5`의 좌우 손 박스와 가시성 플래그를 사용한다.',
        '- GT 박스는 원본 RGB 좌표이므로 source intrinsic·왜곡계수와 rectified intrinsic을 사용해 처리된 RGB 영상 좌표로 옮긴다. 단순히 원본 좌표를 축소 영상에 적용하지 않는다.',
        '- HaMeR 공식 ViT-H·MANO decoder pretrained 초기화. 양손 모두 입력하며 왼손은 공식 방식으로 반전하고 출력을 원래 센서 좌표로 복원한다.',
        '- GT는 SMPL-X 16개 shape 계수와 원래 hand mean으로 독립 재구성한 wrist·다섯 손가락·끝점 21관절이다. MANO shape와 SMPL-X shape 계수를 동일하다고 가정하지 않는다.',
        '- Body와 같은 count33·bins100·surface20, Event4 224 해상도, 고정 gain4, 보정된 광선 기반 depth attention과 영 초기화 잔차를 사용한다. HaMeR 입력은 공식 256 crop 및 ViT의 192×256 영역이다.',
        '- Event4는 공통 RGB 시각 이후 관측을 사용하지 않는다. 저장된 희소 관측 인코딩이며 개별 원시 event를 복원했다고 주장하지 않는다.',
        '- Body의 고정 Test 구간 전체 프레임에 대해 처리한다. 첫 프레임 정렬은 원래 구간 frame0의 변환을 고정하며, frame0 손 표본이 없으면 해당 손 구간을 명시적으로 제외한다. 네 지표는 같은 표본으로 계산한다.',
        '- MPJPE는 wrist 평행이동 정렬, PA-MPJPE는 반사를 금지한 proper Sim(3). RPE와 RPE-Z는 계산·보고하지 않는다.',
        '- GPU0만 사용한다. RGB는 90분, 세 융합은 각각 3시간이며 학습 예산 차이를 센서 효과와 혼동하지 않는다. 마지막 ViT 8개 블록·MANO head·융합을 학습하며 30분마다 Train holdout으로 checkpoint를 선택한다. Test로 checkpoint나 방법을 선택하지 않는다.','',
        '## 실행 기록','',
        '| 학습 조건 | 성공 업데이트 | 실제 경과 (분) | 고유 학습 손 표본 | 전체 Test 완료 |',
        '|---|---:|---:|---:|---:|']
    for n,label in zip(NAMES[1:],LABELS[1:]):
        p=OUT/'runs'/n/'complete.json'
        if p.exists():
            r=json.loads(p.read_text());lines.append(f'| {label} | {r["successful_steps"]} | {r["elapsed_seconds"]/60:.1f} | {r["unique_samples_seen"]:,} | {"완료" if n in complete else "미완료"} |')
        else:lines.append(f'| {label} | 미확인 | 미확인 | 미확인 | 미완료 |')
    selections=[]
    for n in NAMES[1:]:
        p=OUT/'runs'/n/'best.json'
        if p.exists():
            b=json.loads(p.read_text());selections.append(f'`{n}`: Train holdout 선택 step {b["step"]}, PA-MPJPE {b["frame_pa_mpjpe_mm"]:.3f} mm')
    if selections:lines+=['','선택 checkpoint 출처: '+'; '.join(selections)+'.']
    pilots=[n for n in NAMES[1:] if (OUT/'pilot_checks'/('pilot_'+n+'.json')).exists()]
    lines+=['',f'실제 8-step 학습·gradient 검증 통과: {len(pilots)}/4 조건. 짧은 검증 가중치는 본학습에 재사용하지 않는다.']
    if (OUT/'gpu_preflight.json').exists():
        lines+=['','공식 pretrained의 GPU0 전체 forward 검증을 통과했다. 초기 융합 세 조건은 원본 RGB와 관절 출력이 정확히 같았으며, 실제 Event4 입력의 gradient 전달과 센서 unavailable 시 RGB 보존 검증도 통과했다. 이는 성능 평가 결과가 아닌 구현 검사다.']
    if (OUT/'gt_crop_preflight.png').exists():
        lines+=['','### 실제 GT 손 crop 입력 검사','',
            '원본 박스를 처리된 RGB 좌표로 보정한 뒤 양손 crop과 GT 관절을 확인했다. 초록색은 GT 관절이며 예측 그림이 아니다.','',
            f'![GT 손 crop과 관절 확인]({OUT}/gt_crop_preflight.png)']
    if (OUT/'pipeline_state.json').exists():
        state=json.loads((OUT/'pipeline_state.json').read_text());lines+=['',f'실행 관리 상태: `{state.get("status","unknown")}`.']
        if state.get('failure'):lines+=['',f'해결이 필요한 실행 오류: `{state["failure"]}`.']
    if complete:
        rng=np.random.default_rng(261028);sessions=m['test_sessions'];stats={}
        for n in complete:
            sums=np.stack([measured[n][[k[0]==s for k in common]].sum(0) for s in sessions])
            counts=np.array([sum(k[0]==s for k in common) for s in sessions])
            draws=rng.integers(0,len(sessions),size=(2000,len(sessions)))
            boot=sums[draws].sum(1)/counts[draws].sum(1)[:,None]
            stats[n]=dict(mean_mm=(measured[n].mean(0)*1000).tolist(),
                bootstrap_95_ci_mm=(np.percentile(boot,[2.5,97.5],axis=0)*1000).tolist(),samples=len(common))
            for s in sessions:
                r=json.loads((OUT/'results'/n/s/'complete.json').read_text())
                assert r['root_position_error_computed'] is False
        (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
        lines+=['','## 평가 표본과 속도','',
            '| 조건 | 처리한 GT 손 crop | 원래 첫 프레임이 없어 제외한 손 구간 | 처리 속도 (손 crop/초) |',
            '|---|---:|---:|---:|']
        for n in complete:
            rows=[json.loads((OUT/'results'/n/s/'complete.json').read_text()) for s in sessions]
            total=sum(r['processed_samples'] for r in rows);seconds=sum(r['inference_seconds'] for r in rows)
            excluded=sum(len(r['excluded_no_frame0']) for r in rows)
            lines.append(f'| {LABELS[NAMES.index(n)]} | {total:,} | {excluded} | {total/seconds:.2f} |')
        lines+=['','속도는 동일 GPU0에서 캐시 crop을 읽고 모델 추론하는 구간이다. 원본 영상 decoding과 event 인코딩은 포함하지 않는다. 실제 센서 입력부터 출력까지의 지연으로 해석하지 않는다. 세션 단위 2,000회 bootstrap 신뢰구간은 `statistics.json`에 보존한다.']
        lines+=['','## 실제 추론 시각화','', '초록색은 GT, 빨간색은 예측이다.']
        for n,label in zip(NAMES,LABELS):
            p=OUT/'visualizations'/(n+'.png')
            if p.exists():lines+=['',f'### {label}','',f'![{label}]({p})']
    lines+=['','## 출처와 산출물','',
        '- [HaMeR 공식 저장소](https://github.com/geopavlakos/hamer), 공식 pretrained 다운로드 주소와 체크섬은 `pretrained_manifest.json`에 보존한다.',
        f'- 실험 출력: `{OUT}`. 원시 예측·오차는 `results/`, 미세조정 가중치는 `checkpoints/`, 분할은 `split_snapshot.json`에 보존한다.',
        '- `dataset/`과 `data/`는 읽기 전용으로 유지한다. 이번에 받은 pretrained 자산은 해당 실험의 `assets/`에 저장한다.',
        '- 전체 학습·전체 Test·원시 예측·공통 표본·실제 그림을 검증하기 전에는 실험 완료로 표시하지 않는다.']
    REPORT.write_text('\n'.join(lines)+'\n')
    return dict(complete=finished,conditions=complete,common_samples=len(common))


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))

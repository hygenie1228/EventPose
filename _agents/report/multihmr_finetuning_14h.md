# Multi-HMR fine-tuning — Test 10초 chunk 평가

## Test 성능

Test 64세션·device0, 영상당 10초 × 5개 고정 구간: 총 320 chunk·96,124 RGB 프레임. dataset/split.json의 원본 프레임 번호 [start,end)를 그대로 사용했다. 신체 22관절 기준이며 모든 오차는 **mm**다.

| 모델 | First frame aligned MPJPE ↓ | First frame aligned PA-MPJPE ↓ | Frame aligned MPJPE ↓ | Frame aligned PA-MPJPE ↓ |
|---|---:|---:|---:|---:|
| Baseline · Multi-HMR | 160.25 | 146.88 | 94.94 | 58.19 |
| Fine-tuned Multi-HMR | **71.54** | **75.93** | **48.41** | **32.57** |

네 지표 모두 동일한 **105,072 사람-프레임**을 사용했다. 각 chunk의 정확한 첫 프레임에서 평가 대상 모델에 공통 매칭된 사람-chunk 357개가 대상이다. 첫 프레임에 유효 GT나 공통 검출이 없는 사람-chunk 8개는 제외했다. 나중의 첫 매칭 프레임으로 대체하지 않았다. 미검출을 0 오차로 처리하지 않았다.

## 정렬 정의

- **First frame aligned MPJPE:** 각 사람·10초 chunk의 첫 프레임에서 예측 pelvis를 GT pelvis로 옮기는 translation 하나를 구하고, 구간 전체에 고정 적용한다. 회전·스케일은 바꾸지 않는다.
- **First frame aligned PA-MPJPE:** 첫 프레임의 22관절만으로 proper Sim(3)의 회전·양의 등방 스케일·translation을 구하고, 같은 변환을 구간 전체에 고정 적용한다.
- **Frame aligned MPJPE:** 매 프레임·각 사람마다 pelvis translation을 새로 맞춘다. 기존 Root-aligned MPJPE와 같은 정렬이다.
- **Frame aligned PA-MPJPE:** 매 프레임·각 사람마다 proper Sim(3)를 새로 맞춘다. Reflection은 허용하지 않는다.

첫 프레임 이후 GT로 변환을 다시 추정하는 것은 frame aligned 두 지표뿐이다. First-frame PA도 첫 프레임에서 scale을 제거하므로 절대 metric scale 정확도 자체를 검증하는 지표는 아니다.

First-frame PA는 첫 프레임에서만 최적화한 변환이므로, 이후 프레임까지 평균하면 first-frame MPJPE보다 오차가 커질 수도 있다.

## 모델과 검증

사용한 fine-tuned checkpoint: `/home/namhj/EventPose/data/finetuned/multihmr_261023_14h/test_snapshots/20261024_040315/model.pt`. 학습 내부 holdout으로 모델을 선택했으며 Test는 가중치 업데이트나 모델 선택에 사용하지 않았다. 기존 저장 추론의 camera-space 관절을 재사용해 정렬 지표만 재계산했다.

전체 chunk 완료·GT 프레임 대응을 확인했고, per-frame 지표 1,428건을 기존 구현과 대조했다. First-frame PA는 각 사람-chunk에서 SciPy 회전 정렬로 독립 검산했다.

원시 집계: `exp/261023/multihmr_finetune_14h/test_snapshots/20261024_040315/chunk_alignment_metrics.json`, chunk별 결과: `exp/261023/multihmr_finetune_14h/test_snapshots/20261024_040315/chunk_alignment_per_segment.csv`.

Train 219세션·68명, Test 64세션·20명은 피험자가 겹치지 않는다. Train 내부 187세션·58명은 가중치 학습, 32세션·10명은 모델 선택에 사용했다.

학습: `scripts/train_multihmr.py`. Test 추론: `scripts/evaluate_multihmr.py`. 정렬 평가: `src/utils/chunk_alignment.py`.

> 최신 상태: 아래는 오전 이관 당시의 역사 기록이다. NLF 학습·4모델 전체Test·동일GPU속도·원시결과검증은 모두 완료됐다. 현재는4인코딩/10조건 RGB-event 36시간 Goal이며 GPU2–6을 사용한다. 실행 중인 새 작업과 다음 점검 시각은 `_agents/research/261024/rgb_event_implementation_handoff.json` 및 `exp/261024/rgb_event_36h/pipeline_state.json`을 따른다. 아래의 과거 학습 PID/관리 프로세스를 재시작하지 않는다.

# NLF / Multi-HMR 목표 이관

사용자가 현재 진행 중인 목표를 웹 UI의 **AI Agent 탭**으로 이관하라고 명시했다. 이 문서는 재시작 계획이 아니라 실행 중인 작업을 이어받기 위한 상태 기록이다.

## 원래 목표와 시간 범위

원문: “지금 평가끝나면 NLF기반으로도 평가 & finetuning 진행해서 표로 비교하고 뭐가 더 좋은지 복합적으로 비교하는 표만들어줘 이것도 12시간 사용”.

원래 12시간 작업 시작: 2026-10-24 05:19:25 UTC. 원래 종료 목표: **2026-10-24 17:19:25 UTC**. 웹 목표 시작 시점부터 새 12시간을 계산하지 않는다. NLF 학습 종료는 **13:47:05 UTC** (`1792849625.8986695`)로 고정되어 있고, 이후 전체 Test와 동일 GPU 속도 측정, 최종 표·추천·검증을 마친다. 필요한 작업이 남아 있는데 완료 처리하지 않는다.

## 가장 중요한 운영 규칙

- `_agents/instruct.md`를 읽는다. 사용자 질문 없이 합리적으로 진행한다.
- 학습·평가 상태는 **30분마다 한 번만** 확인한다. 모델 토큰을 쓰는 잦은 조회·상태 반복을 금한다. 구체적인 오류가 있을 때만 더 자주 개입한다. 가벼운 백그라운드 관리 프로세스는 유지한다.
- 이미 실행 중인 GPU 학습을 다시 띄우지 않는다. 프로세스 번호는 아래 참고 정보이며 필요할 때 실제 명령행과 상태로 확인한다. 관찰 timeout을 종료로 판단하지 않는다.
- `dataset/`, `orig_dataset/`, `etri_processed/`, `data/`는 읽기 전용이다. GPU 0–5만 사용하고 6은 비워 둔다.
- 보고서는 `_agents/report/YYMMDD_이름.md`의 한글 문서로 작성한다. 연구·작업 기록은 `_agents/research/YYMMDD/` 바로 아래에 쓴다. `agent_sessions` 하위 폴더를 만들지 않는다.
- 오차 단위는 mm, 내부 관절 좌표는 m. 실제 Test만 성능 표에 보고한다. Train 내부 holdout은 모델 선택용이며 Test로 표시하지 않는다.
- 학습 중간 체크포인트의 Test 결과는 허용하지만 표 모델명에 단계·상태 표현을 붙이지 않는다. 출처는 별도 문단에 쓴다. 좋은 지표는 굵게 표시한다.

## 실행 중인 NLF 작업

실험: `exp/261024/nlf_finetune_12h/`, SSD 작업: `.experiment_storage/nlf_finetune_12h/`.

학습은 3설정, 각 2 GPU로 실행 중이다.

- head: GPU 0/1, 주 학습 프로세스 57051/57052.
- last_stage: GPU 2/3, 주 학습 프로세스 57056/57058.
- last_two_stages: GPU 4/5, 주 학습 프로세스 57055/57057.

원래 관리 부모가 종료되어 주 학습 프로세스들이 PID 1의 자식이 되었지만 가중치 업데이트는 계속됐다. **학습은 재시작하지 않고**, 새 detached 관리 프로세스 PID **33924**가 살아 있는 rank 0 프로세스들을 인계받았다. `tools/recover_nlf_training.py`와 `exp/261024/nlf_finetune_12h/training_adoption.json`이 이를 구현한다. 현재 상태는 `training_recovery_status.json`, 로그는 `logs/training_recovery_supervisor.log`. 이 프로세스가 학습 완료 후 자동으로 모델 선택·내보내기·전체 NLF Test·속도 비교를 수행한다. 중복 관리 프로세스를 만들지 않는다.

체크포인트는 실험의 `checkpoints/{head,last_stage,last_two_stages}/`에 있으며 5분마다 optimizer 포함 `last.pt`, 내부 선택 지표가 좋아지면 `best.pt`를 저장한다. `runs/{설정}/best.json`, `progress.json`, 최종 `complete.json`을 참고한다. 실패를 실제로 확인했을 때만 보존 체크포인트로 복구한다. 원래 학습 로그와 재시작 이력은 보존한다.

## 완료된 Test

Test는 64세션·device0, 세션별 10초×5개의 고정 구간, 총 320구간·96,124 RGB 프레임이다. split과 GT를 바꾸지 않는다. Train219세션68명/Test64세션20명 피험자 분리. Train 내부187학습+32holdout이므로 219개 모두 가중치 업데이트했다고 쓰지 않는다.

Multi-HMR baseline과 최종 학습 모델 Test는 모두 320/320 완료됐다. NLF baseline도 320/320 완료됐다. NLF fine-tuned는 학습 후 평가가 남았다.

최종 Multi-HMR: `exp/261023/multihmr_finetune_14h/selected.json`, `checkpoints/best_full.pt`, SSD `.experiment_storage/multihmr_finetune_14h/final_test/multihmr_finetuned/`. baseline Multi 결과 `.experiment_storage/humanpose_gt_segments/multi_hmr/`, NLF baseline `.experiment_storage/nlf_finetune_12h/nlf_pretrained/`. 동일 GT·실제 검출·2D Hungarian 매칭이며 Test에서 GT box를 쓰지 않는다.

현재 세 모델 공통 104,863 사람-프레임에서:

| 모델 | 첫 프레임 MPJPE | 첫 프레임 PA | 프레임별 MPJPE | 프레임별 PA | 절대 MPJPE |
|---|---:|---:|---:|---:|---:|
| Multi-HMR 기본 |160.04|146.65|94.94|58.19|272.30|
| NLF 기본 |116.61|111.94|78.83|53.31|127.63|
| Multi-HMR 학습 |68.53|73.26|46.68|32.54|70.58|

첫 프레임 정렬은 정확한 구간 frame0에서 구한 변환을 10초간 고정한다. MPJPE는 pelvis translation만, PA는 proper Sim(3). 프레임별은 매 프레임 재정렬한다. 모든 지표는 첫 프레임에 공통 매칭된 동일 사람·프레임 집합으로 계산한다. 4개 모델로 비교할 때는 공통 집합이 바뀌므로 다시 집계한다.

## 완료까지 남은 절차

1. NLF 세 설정의 고정 종료 시각까지 학습을 유지하고 완료 여부를 확인한다. Test를 선택 지표로 쓰지 않는다.
2. 관리 프로세스가 Train 내부 holdout의 최선 모델을 `checkpoints/best_full.torchscript`로 내보내고 `selected.json`을 기록한다. 내보내기 검증 도구는 모든 weight와 metadata를 확인한다.
3. `scripts/evaluate_nlf.py supervise --method nlf_finetuned --checkpoint ... --batch 16`이 GPU0–5 전체 320구간 Test를 수행한다. 각 구간의 complete/result와 프레임 대응을 확인한다.
4. `src/utils/compare_methods.py`가 네 모델 공통 집합에서 요청한 4지표, 절대 위치, 검출률, 속도, 움직임 오차, 55관절 보조 결과를 계산한다. 결과는 실험 `comparison.json`과 한글 보고서에 기록한다.
5. `src/utils/runtime_benchmark.py supervise --gpu 0`가 나머지 작업 종료 후 같은 세 100프레임 구간에서 네 모델을 순차 측정한다. 영상 복사·디코드·추론·SMPL-X 복원·GT 매칭·저장을 포함한 시간, 모델 로딩, 프로세스 전체 시간, VRAM을 비교한다. 현재 동시 실행 관측 fps를 공정한 속도 배수로 단정하지 않는다.
6. 비교 표와 단 하나의 우선 추천, 절대 위치·검출률·시간적 일관성·속도의 장단점을 함께 작성한다. PA는 스케일을 제거하므로 metric scale 판단에는 절대 오차를 사용한다. 임의의 가중 합산 점수를 만들지 않는다.
7. 각 요구사항의 실제 파일·실행 결과로 완료를 검증한다. NLF 학습·최종 Test·속도 측정이 끝나기 전에 완료 처리하지 않는다.

## 코드와 문서

학습 `scripts/train_nlf.py`, 평가 `scripts/evaluate_nlf.py`, 모델 `src/nlf/`, Multi-HMR `src/multihmr/`, 공통 지표·시각화 `src/utils/`. NLF v0.2.0의 mm 출력과 centered translation 보정은 `src/nlf/model.py`에 구현돼 있고 SMPL-X mesh 검산을 통과했다. 빈 검출과 큰 projected crop의 PIL 할당 오류도 해결했다.

보고서 현재 파일은 사용자 웹 작업에서 한글 이름으로 정리됐다:

- `_agents/report/261024_다중인체복원_14시간_미세조정.md`
- `_agents/report/261024_인체자세추정_모델_성능_비교.md`

생성 코드도 해당 경로로 수정했다. 웹 앱은 `src/apps/research-workbench/`이며 `eventpose-workbench.service`로 실행된다. Chat과 AI Agent 모두 쓰기가 가능하고 탭 목록은 분리된다. Goal은 실제 `codex app-server`의 thread/goal API를 사용하며, 소규모 파일 생성 목표의 native complete 상태가 검증됐다. 이 Goal은 AI Agent 영역에 생성된다.

Git main에 이전 구조·웹 기능 커밋들이 있다. origin push는 이전에 인증 401로 실패해 사용자에게 재로그인을 안내했다. 비밀번호·토큰을 채팅에 요구하지 않는다. 현재 사용자 소유 변경과 연구 기록을 보존하며, 무관한 `thirdparties/Human3R`의 untracked 모델 링크를 정리하거나 커밋하지 않는다.

# Workspace Instructions

- 사용자에게 안내하는 시각과 새 보고서의 표시 시각은 한국시간(KST, UTC+9)을 사용한다. 내부 deadline_unix 등 실행 기준 시각은 유지하고, 기존 UTC 원시 기록은 변경하지 않는다.

## Current GPU allocation — latest user instruction

- 2026-10-29 최신 단일 agent 지시: 사용자가 기존 대화와 새 탭 agent를 동시에 실행할 필요가 없다고 요청했다. **새 AI Agent 탭 `부위별 동적 집계 · 20시간 실험` 하나가 연구·구현·관리·평가·보고·최종검증을 전부 소유한다.** session `7286d3b2-b160-4baf-80f9-013bc86888a6`, thread `01a186d0-36f7-71b3-be07-479d06337b31`. 이전 root/하위 `/root/part_aware_aggregation`은 인계를 마치고 추가 작업을 중단한다. 최종 인계 `_agents/research/261029/부위별_동적집계_단일에이전트_최종인계.md`; 기존 관리기·worker·20시간 마감은 그대로이며 중복 agent·관리기·학습을 시작하지 않는다. 새 탭의 이전 모니터 전용 제한은 최신 실행 소유권 이관 지시로 대체되었다. 실험 user-control watch는 이미 이 새 탭 Goal에 연결됐다. 원래 고정 마감 **2026-10-30 09:55:12 KST**를 연장하거나 새20시간으로 재설정하지 않는다.

- 2026-10-29 13:55 KST: 사용자가 **Part-aware Dynamic Aggregation 전담 agent와 20시간 Goal**을 명시적으로 요청했다. 고정 범위는 **2026-10-29 13:55:12~2026-10-30 09:55:12 KST**이며 설정 `_agents/research/261029/part_aware_dynamic_aggregation_goal_20h.json`을 따른다. 담당 `/root/part_aware_aggregation`, 관리기 `tools/run_part_aggregation.py`, 출력 `exp/261029/part_aware_dynamic_aggregation/`, 보고서 `_agents/report/261029_부위별_동적_이벤트_집계_설계와_실험.md`. 같은 **device3 RGB+event3**, Transformer 기반 부위별 과거 시간 집계를 구현하고 RGB-only·고정 구간 균등 집계·전신 공통 동적 집계와 비교한다. GPU는 기존 허용 물리 **1·2·4·5 중 실제 유휴 장치만**, GPU0·3·6과 타인 작업을 보존한다. 기존 Train-selected RGB3 초기화는 네 조건에 공통이며 출처를 명시한다. 학습 forward는 해시/버전으로 동결하고 Test로 방법·checkpoint를 고르지 않는다. 4조건 실제 CUDA8-step pilot 및 checkpoint 재로드 검증을 완료했으며 본학습·전체 Test 완료를 뜻하지 않는다. 장기 진행 직접 조회는30분 간격, RPE-Z의 실제 키는 `root_depth_error_mm`다. 마감에는 본인 등록 작업만 checkpoint 보존 종료하며 개선 실패·미완료를 숨기거나 Goal 완료로 표시하지 않는다. 다른 대화의 hand 및 기존 결과는 변경하지 않는다.

- 2026-10-29 최신 시각화 지시: 모든 새 event 표시를 `_agents/report/261024_인체윤곽과_자세변화를_위한_이벤트_인코딩_비교.md`의 검정 배경·고정 sqrt(log1p(N)/log9) 밝기·양극성 빨강/주황/음극성 cyan으로 통일한다. count33 윤곽은 약한3×3지지(자신포함2개이상), bins100은 과거R/중간G/최근B, surface20은 극성별20ms감쇠다. 표시만 바꾸며 학습입력/가중치를 바꾸지 않는다. 사용자는 스테레오 개요도/PPTX의 모든 visible text를 영어로 쓰고 실제 mesh rendering을 쓰라고 명시했으므로 해당 문서/발표물의 영어는 승인된 예외다. 편집 가능한 native-shape PPTX와 기존9조건×5구간의 RGB3|Event4|Mesh overlay 통합MP4를 요청했다. hand GPU0 작업은 보존한다.

- 2026-10-28 hand 신규 지시: `_h`가 손 데이터이며 HaMeR 공식 pretrained로 진행한다. 손 작업은 **GPU0만** 사용한다. 조건은 device3 RGB 단독, device3 RGB + device4 Event의 count33·bins100·surface20으로 총 네 학습 실험이다. Body와 같은 컨벤션을 따르되 사용자가 **GT 손 박스 crop 기반 복원 평가**를 선택했고 **RPE는 측정하지 말라**고 지정했다. 네 MPJPE/PA-MPJPE 정렬 지표만 주표에 둔다. 출력 `exp/261028/hamer_hand`, 보고서 `_agents/report/261028_손_복원_하머_네조건_비교.md`. checkpoint의 SSD backing `.experiment_storage/hamer_hand_261028/`는 활성 작업 동안 보존한다. 기존 body 작업의 GPU1·2·4·5는 보존한다. 이 대화에서 persistent Goal 생성은 요청되지 않았다.
  - 2026-10-29 00:13 KST: 네 조건의8-step pilot이 모두8/8업데이트·실제융합gradient·checkpoint 검증을 통과했고 GPU0 RGB90분 본학습을 시작했다. `tools/run_hamer_hand.py`가 이후 세 융합 각3시간·전체Test·30분보고·최종검증을 관리한다. `pipeline_state.json`, `managed_jobs.json`, `implementation_handoff.json`을 먼저 확인하고 중복 실행하지 않는다. 현재 전체 실험 완료가 아니며 Test 실측값은 아직 없다.

- 2026-10-28 15:56:29 KST: 사용자 요청으로 새12시간 이벤트 인코딩 Goal을 시작했다. 설정 `_agents/research/261028/device3_encoding_goal_12h.json`, 출력 `exp/261028/device3_encoding_12h`, 보고서 `_agents/report/261028_이벤트_인코딩_세가지_센서융합_비교.md`. 고정마감2026-10-29 03:56:29 KST. body만, 원본RGB3·RGB3단축90분·RGB3+event4/event3+event4/RGB3+event3 각각count33·bins100·surface20 3시간으로 총11행. 기존fine-tuned가중치/결과 재사용 없이 원본에서 시작; 검증된 같은split 캐시만 읽기참조. GPU1·2·4·5의 현재UUID 검증값 사용. 직접진행확인은30분 간격, report는메인5지표표 먼저. 최종실제검증 후에만 complete; 미완료마감 본인작업만보존중단하고 pause.

- 2026-10-28 14:11 KST: 새 영상 Goal 완료. `exp/261028/device34_test_video_gallery`에 기존9조건×같은5개 Test구간45개 MP4를 실제추론·전체디코딩 검증했다. 기존 `_agents/report/261027_센서융합_열화원인_분석과_재실험.md`에 목록을 추가했고 주표154행은 유지했다. Point cloud는 원본 depth가 있는3구간에만 포함하며0.001m/unit 가정을 명시했다. 증거 `final_verification.json`; 초기판은 `initial_render_v1`이며 최종본은 `mp4/`의 render_version2다. 완료 작업을 자동 재시작하지 않는다.

- 2026-10-28 00:45 KST: `device34_fusion_repair_15h`의7학습+원본2조건 전체 Test, 동일GPU18속도실행, 통계·checkpoint·원시5지표·보고서·실제그림 검증을 마치고 Native Goal을 **complete**로 전환했다. 결과는 `_agents/report/261027_센서융합_열화원인_분석과_재실험.md`, 최종 증거는 실험출력의 `agent_completion_review.json`과 `final_artifact_audit.json`이다. 완료 실험을 자동으로 재시작하지 않는다. RGB3 본 비교는 별도90분 trial이며 옛103분 중단 run은 제외 참고다.

- 최신 추가 지시: 같은 장치 RGB3+event3, RGB4+event4도 추가하고 원본 RGB3/RGB4 미세조정 전 전체Test를 표에 포함한다. RGB-only는 더 짧게, 융합은3시간 학습한다. RGB3는 지시 당시 이미103분 진행되어 checkpoint 보존 중단했으며 실제 시간을 보고하고, 아직 시작 전이던 RGB4는90분으로 설정했다. 학습량 차이와 센서 효과를 혼동하지 않고 예산별 경향을 실측값으로 제시한다. 고정마감은 한국시간2026-10-28 04:15:31 그대로다.

- 2026-10-27 최신 사용자 요청은 body 센서융합 세 조건의 열화 원인 분석·수정·재학습·전체 Test를 위한 새 15시간 Goal이다. 고정 범위는 2026-10-27 04:15:31~19:15:31 UTC이며 설정 `_agents/research/261027/device34_fusion_repair_goal_15h.json`을 따른다. 사용 GPU는 **물리 1,2,4,5만**이며 이전 GPU2–6 지정보다 우선한다. UUID·버스·유휴·실제 CUDA 동작을 검증하고 GPU3 및 타인작업은 조작하지 않는다. 이전 28시간 Goal의 본인 관리 프로세스는 새 실험과 충돌하지 않도록 신원 검증 후 중단하며 결과는 보존한다. 개별 본학습 동일3시간·최대4시간 원칙과 dataset/data 읽기전용 원칙은 유지한다.

- 2026-10-26 사용자가 추가24시간 실행과 Goal 재설정을 승인했다. 추가 고정 범위는2026-10-26 06:50:02~2026-10-27 06:50:02 UTC다. 이 승인으로 아래 원래36시간 마감만 대체한다. 기존 중단 전 학습량을 보존하고 남은 예산만 이어가며, 완료되지 않으면 새 마감에 자동 pause한다. GPU2–6 중 정상·유휴 장치만 사용하고 다른 agent 작업은 보존한다.

- 최신 확장 목표에서 새 GPU 작업은 물리 GPU 2, 3, 4, 5, 6만 사용한다. 이전 GPU 3–6 및 0–5 지정보다 우선한다. 실행 중인 학습·관리·평가 프로세스는 중복 실행하거나 재시작하지 않는다.
- 이미 재개한 NLF 최종 평가는 GPU 3–6의 4개 worker로 유지하고 동일 GPU 속도 비교는 GPU 3에서 순차 수행한다. 이후 RGB/event 10조건(원본 RGB·미세조정 RGB 및 event-only/RGB+event 각각 4인코딩) 실험까지 완료·검증해야 새 Goal을 완료 종료한다. 인코딩에는 `tools/visualize_data.py`의 배경128·양극성255·음극성0 회색조 표현을 반드시 포함한다. 새 고정 36시간 범위는 2026-10-24 17:17:17~2026-10-26 05:17:17 UTC이며 미완료면 사용자 요청에 따라 자동 pause한다. 상세 목표와 타이머 설정은 `_agents/research/261024/rgb_event_goal_36h.json`을 따른다. 모든 새 실험 프로세스는 해당 설정의 managed_jobs_file에 PID·PGID·명령·시작 tick을 등록하고, 학습 자체에도 동일 마감을 전달해 종료 전 체크포인트를 보존한다.

## Code organization — explicit user requirements

- Use `/home/namhj/EventPose/ChallengeHOR` as an architectural reference only; it is not a runtime dependency.
- `scripts/` contains only user-facing launchers for training, demo/inference, evaluation, and visualization. Do not put temporary experiment scripts, benchmarks, migration scripts, inspections, audits, monitors, or one-off helpers there; these belong in `tools/`.
- `src/` is model-first: Multi-HMR code and its data code belong in `src/multihmr/`; NLF code and its data code belong in `src/nlf/`. Common evaluation, metrics, and visualization implementations belong in `src/utils/`. Its top level contains only `README.md` and directories, never loose Python files. Do not create generic `src/data`, `src/evaluation`, `src/inference`, `src/training`, or `src/visualization` directories.
- Put the executable training/evaluation implementation in the respective `scripts/train_multihmr.py`, `scripts/train_nlf.py`, `scripts/evaluate_multihmr.py`, and `scripts/evaluate_nlf.py`; do not create separate generic training/evaluation source folders. Reusable model/data/metric helpers remain in the model directory or `src/utils/`.
- Put Multi-HMR implementation directly in `src/multihmr/`, copied substantially unchanged from upstream. Preserve upstream licenses/notices; limit integration edits to necessary imports and paths. Include the backbone source needed to construct the model, so cloning this Git repository does not require a separate Multi-HMR checkout or a developer-only torch.hub cache.
- Reserve `thirdparties/` for external mask/image-processing components; do not put the main Multi-HMR implementation there.
- Manage remaining external repositories under `thirdparties/` as pinned Git submodules, following ChallengeHOR. Multi-HMR and NLF are copied source under `src/`, never submodules. `tools/` must contain files only, not source directories; remove the obsolete `tools/third_party/` layout after preserving dependencies.
- Keep temporary/auxiliary code in `tools/`. Compatibility wrappers for live processes may remain temporarily, but do not duplicate the core implementation there.
- Store each experiment's fine-tuned checkpoints in its own `exp/YYMMDD/{experiment}/checkpoints/`, not `data/finetuned/`. Pretrained weights and licensed SMPL-X assets remain in `data/` and must have documented setup steps.

## Pose metric reporting

- 최신 사용자 지시: 성능 비교 보고서는 제목 바로 다음에 메인 평가표를 배치한다. 배경·방법·실행 이력·보조표보다 앞에 두며 자동 진행 갱신과 최종 보고서 생성에서도 유지한다. 미평가 값은 만들지 않는다.
- Test 평가 결과를30분 간격 보고의 중심으로 삼는다. 전체 Test를 마친 조건의 실측5지표를 주표에 즉시 표시하고 전체9조건 완료까지 숨기지 않는다. 완료 조건이 추가될 때 그 조건들 공통 GT로 재계산하고 현재 비교 집합·표본 수를 명시한다. 학습이 끝난 RGB·융합 조건의 Test를 우선 배정하되 실행 중인 학습/평가는 끊지 않는다. Train holdout은 checkpoint 선택용 보조표이며 Test 수치를 대체하지 않는다. Test 결과로 학습/방법/checkpoint를 선택하지 않는다.

- 2026-10-26 최신 사용자 요청: 주표는 기존 네 정렬 지표에 **Root absolute depth error (RPE-Z, mm)**를 더한 다섯 지표다. RPE-Z는 정렬 전 카메라좌표 pelvis/root의 `mean(abs(Z_pred-Z_GT))*1000`이며, full3D RPE `mean(norm(root_pred-root_GT))*1000`와 혼동하지 않는다. 다섯 지표는 동일 공통 GT/검출 표본으로 계산한다.

- Do not append checkpoint-stage/status labels such as "frozen interim model" to model names in performance tables. The user allows results from training-intermediate checkpoints; they objected to the label, not the evaluation. Keep checkpoint provenance in a separate supporting paragraph, and never misrepresent an intermediate checkpoint as a final one.
- Bold the best value in each metric column (lower error, higher coverage/throughput). Do not claim a best value when the comparison is still incomplete.
- The Multi-HMR main table uses 10-second chunks and four mm-valued metrics: first-frame-aligned MPJPE, first-frame-aligned PA-MPJPE, frame-aligned MPJPE, frame-aligned PA-MPJPE. First-frame alignment uses exactly frame 0 and stays fixed across the chunk; frame alignment is recomputed per frame. MPJPE alignment is pelvis translation; PA is proper Sim(3). Omit the status column.
- Archive unused temporary Python utilities under `exp/_code_backup/`. Keep only currently needed experiment support in `tools/`; normal training/demo/evaluation/visualization must work without `tools/` source dependencies.

- Display pose estimation errors (MPJPE and PA-MPJPE variants) in millimetres (mm) in user-facing tables and reports. Preserve metre-valued geometric coordinates and explicitly named `_m` raw metric fields internally; multiply by 1000 for display.

## Immutable datasets

- Treat `dataset/` (server L) and `orig_dataset/` (server I) as strictly read-only.
- Never create, modify, rename, move, or delete any file or directory under either dataset root.
- Do not place caches, lock files, logs, temporary files, converted assets, or visualization outputs in either dataset.
- The user canceled the temporary RGBD copy into server L. Copy authorization is revoked. The only authorized cleanup exception is removing the newly copied `dataset/20260720_125338_040324_200_f/RGBD/device0.h5` through `device4.h5`, including partial copies. Preserve every `device*_mamma.h5` GT file. All other dataset writes remain prohibited; server I remains read-only.

## Experiments and outputs

- Run every experiment in a named subdirectory of a date-scoped directory: `exp/YYMMDD/{exp_name}/` (for example, `exp/261023/processed_visualization/`). Never place experiment results directly under the date directory.
- Store results, derived samples, visualization assets, logs, and manifests inside that dated experiment directory.
- Put runnable experiment entry points in `scripts/` and learning/evaluation implementations in `src/`. Keep `tools/` compatibility links only where live jobs still need old paths. Never implement Python files under `exp/`.
- Use a temporary directory inside the dated experiment directory when temporary files are required.

## Persistent model assets

- Store pretrained weights and human body model files under `data/`. Store experiment fine-tuned checkpoints inside that experiment's `exp/YYMMDD/{exp_name}/checkpoints/` (user instruction). Existing `data/finetuned` links are only live-job compatibility paths.
- Reuse the SMPL/SMPL-X assets in `data/human_model_files/`; do not copy them into a dataset or experiment directory unless an isolated experiment explicitly requires a derived artifact.
- Keep downloaded reusable assets in `data/`; experiment-specific fine-tuned checkpoints belong to the corresponding experiment directory.

## GT dataset processing

- Process all 283 server-L sessions with GT into `etri_processed/`, matching reference session `20260720_125338_040324_200_f` for available modalities.
- If corresponding server-I originals are absent, omit unavailable Depth and device5/6; do not fabricate data or wait for missing originals.
- Each session contains device directories only. Do not copy conversion logs, manifests, lock files, or top-level labels/RGB/Event directories.
- Device0–4 store `label.h5`, `event.npz`, `rgb.mp4`, and available lossless `depth.mp4`; device5/6 use server-I inputs and store RGB as `video.mp4` at 5 fps.
- Labels contain only `params`, `bboxes`, and `frames`, including required validity and frame mappings. Exclude meshes and person masks.
- Finish core data for every session before generating remaining `event.mp4` and `preview.mp4`. Preserve existing finished videos.
- Event visualization follows `tools/visualize_data.py`, includes every stored frame, and retains native resolution. Previews contain no text, last at most 60 seconds, and use reduced resolution with VS Code compatible encoding.
- Keep depth numerically lossless and validate decoded values against original uint16 frames.
- `etri_processed/split.json` contains exactly `{"train": [session names], "test": [session names]}`. Current split: Train 68 participant IDs / 219 sessions; Test 20 IDs / 64 sessions. Keep IDs disjoint, including co-occurring IDs. Store split metadata separately under the experiment directory.
- Prioritize Test sessions within each processing stage; generating Test visualization must still wait until all core data is complete.
- Run long conversion jobs in the background and check every 30 minutes. Keep the goal active until all required outputs have been verified.
- GPU jobs may use all eight GPUs, subject to actual encoder session limits.

## Repository and local storage

- Store research notes under `_agents/research/YYMMDD/` (for example `261023/` and `261024/`). Write all documents in `_agents/research/261023/` in English, as explicitly requested by the user. Keep `src/README.md` in English.

- Repository remote: `https://github.com/hygenie1228/EventPose`.
- Ignore external storage symlinks (`data`, `dataset`, `orig_dataset`, `etri_processed`, `exp`), `.experiment_storage/`, and Python caches in Git. Do not delete symlinks when asked to ignore them.
- `.experiment_storage/` is local SSD backing for selected experiment directories linked from `exp/261023/`, used to avoid slow NFS intermediate I/O. Active background jobs depend on it; do not remove it while in use.

## Fine-tuning monitoring frequency

- During fine-tuning, inspect progress at most once every 30 minutes unless the user asks for an update or a concrete failure requires intervention. Avoid frequent model/tool polling and repetitive status messages to conserve tokens. Lightweight background recovery supervisors may continue running.

## Browser workspace

- Keep the browser research workspace under `src/apps/research-workbench/`, as explicitly requested by the user; keep it separate from model code and motion scripts.
- Provide left-to-right file explorer, file/code/media viewer, and multiple independent AI chats, using the currently signed-in Codex CLI.
- Bind the remote workspace to the server network interface and a browser-compatible port >= 10000; the current deployment uses `0.0.0.0:18080`.
- Implement routine choices without asking clarification questions. Long training/evaluation progress checks remain limited to once every 30 minutes unless intervention or a user-requested update is needed.

- Remote workbench reachability: UFW is enabled on this server. A server-local HTTP 200 does not prove access from the user's computer. Do not claim external access was verified without an external client check. Opening TCP 18080 requires an administrator if passwordless sudo is unavailable.

- Persist all browser Agent user/assistant/tool events under `_agents/research/YYMMDD/` as complete JSONL and readable Markdown. Record this agent's implementation/research notes under the same date-scoped research directory. Put deliverable reports under `_agents/report/`.
- Workbench defaults: white/light theme, readable 15–16px text, prominent 30px close buttons, one Codex panel on the right and a second below the viewer, and top-bar one/two-panel buttons and light/dark theme buttons. The bottom panel must resize up to 50% of the workspace height. Each viewer/chat must scroll independently and keep the chat composer visible; show file/chat loading states.
- Default the right Chat pane to 30% of the viewport on every refresh; do not impose the old 750px drag-resize limit. Show animated spinners for loading and animated progress while Codex is working.

- Both Chat and AI Agent must have workspace write permissions. Their chat tabs/lists are independent (`chat` and `agents` pane ownership). The Chat empty state lists the five newest reports; omit generic welcome/suggestion text.
- Hide dot-prefixed files/directories in the browser explorer. Provide right-click Rename/Delete for files and nonempty directories; preserve dataset/read-only protection and move deletions to recoverable server trash.
- `_agents/report/`의 모든 보고서는 본문과 파일명 모두 한글로 작성한다. 파일명은 `YYMMDD_한글_보고서명.md` 형식을 사용한다(예: `261024_인체자세추정_모델_성능_비교.md`). 날짜 접두사와 파일 확장자는 유지한다.

- Telemetry: display GPU VRAM used percentage in a larger 4×2 grid, red when in use and green when idle. Label the compact account quota row `token limit`; omit the separate Codex quota heading. Refresh GPU telemetry without model tokens.

- Remove the explorer filename-filter input; the user wants only the file tree.

- Keep Goal status compact: a small icon immediately left of the Goal checkbox opens a popup containing objective/status/usage and pause/resume/clear controls. Do not reserve a large persistent status panel above the composer.

- Support initial web login from `?key=...` as requested; exchange the supplied login secret for the session cookie and remove it from the browser address, preserving other URL parameters.

- Scale browser UI density down on laptop/small windows: keep the top bar compact (32–40px), reduce tabs/toolbars/padding and text at smaller viewport widths/heights, and preserve visible composers and independent scrolling.

- On viewports <=767px wide, show one conversation panel with Chat / AI Agent tabs: hide the top bar, explorer, viewer and footer. Background goals keep running. Use compact laptop chrome/fonts at smaller desktop widths/heights.

- In one-panel view, place Chat and AI Agent mode tabs together in the panel heading, including on phones. Keep their tabs/conversations independent and preserve unsent drafts across mode/layout changes. Remove the Research partner/Ready/current CLI login info banner.
- Active Goal sessions accept additional chat messages through native turn/steer without a duplicate turn or goal reset; keep both Send and Stop available. AI Agent message/composer font sizes must match Chat, independent of iframe width.
- Workspace report/file links in Chat or AI Agent messages must open the shared Viewer, including links from the embedded Agent pane. On phones use a full-screen viewer with a return-to-chat control. Preserve external web links as external links.

- Remove the bottom Workspace connected/host/UTF-8 status bar entirely and give its space to the workspace.

- Remove the viewer file-size/UTF-8 status strip as well; do not display redundant metadata bars.
- On chat selection or page refresh, show the most recent message by scrolling restored history to the bottom in both Chat and AI Agent. Preserve deliberate scrolling back while reading older messages.
- Both ordinary Chat and AI Agent conversations accept messages during a running response using native turn steering, without requiring Goal. Keep Send and Stop available. Native clients execute on the authorized GPU host so CUDA devices remain visible; retain the immutable-dataset instructions and use `.venv-humanpose/bin/python` for scientific code.

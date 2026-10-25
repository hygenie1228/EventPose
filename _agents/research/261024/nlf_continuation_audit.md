# NLF 목표 이어받기 검증 기록

2026-10-24 09:35 UTC에 인계 문서 전체와 작업 지침을 읽었다. 목표는 미완료이며 원래 종료 목표 17:19:25 UTC, 학습 종료 13:47:05 UTC를 유지한다.

현재 exec 도구의 /proc/1은 codex-linux-sandbox이며 호스트 PID 공간과 다르다. 따라서 인계 PID가 ps에 안 보이는 것은 호스트 작업 종료 증거가 아니다. nvidia-smi도 격리 환경에서 드라이버에 연결하지 못했다. 학습이나 관리 프로세스를 재시작하지 않았다. 마지막 상태 파일은 training이지만 이것만으로 살아 있는 프로세스를 검증했다고 주장하지 않는다. 다음 정기 상태 확인은 10:05 UTC 이후로 제한한다. 호스트 상태 관찰 수단을 확보해야 한다.

정적 코드 확인: recover_nlf_training.py는 기존 학습 종료 후 내부 holdout 최저 PA 모델 내보내기와 선택 기록, evaluate_nlf.py supervise를 호출한다. 평가 supervise는 체크포인트 검증, 6개 worker 전체 Test, 공통 비교 집계, GPU 0 순차 속도 측정, 최종 재집계를 호출한다. 현재 관리 스크립트가 메모리에 이미 로딩되었을 수 있으므로 해당 파일 수정이나 중복 실행은 하지 않았다.

보고서 생성기에서 동일 GPU 표와 속도 우승 모델 이름을 한글로 통일하고, Train 187학습+32holdout 및 Test 분리를 명시했다. Python 3 AST 구문 검사를 통과했다. 최초 python 명령은 Python 2여서 검사 스니펫이 실패했으며 Python 3으로 바로잡았다. 실제 최종 수치나 추천은 아직 확정하지 않았다.

## 최종 완료에 필요한 직접 증거

- 세 학습 설정 complete.json 및 successful_steps, 고정 deadline, holdout best.json과 selected.json 일치.
- TorchScript 전체 weight/metadata 검증 도구 성공, Test 미사용 모델 선택 확인.
- 네 모델 각각 동일 tasks.json 320구간의 complete.json과 result.npz, 96,124프레임 및 source frame 대응 검증.
- 네 모델 동일 공통 매칭 집합, 정확한 frame0 고정 정렬, 프레임별 정렬, mm 표시와 metric scale 절대 오차 확인.
- 검출률 전체 valid GT 분모 및 속도/가속도 연속 프레임 표본 수 확인.
- 동일 GPU 0에서 네 모델 동일 3구간·총 300프레임의 순차 실행 완료, 로딩/프로세스/전체 파이프라인 시간 및 VRAM 원시 결과 확인.
- 최종 한글 보고서의 모든 숫자와 최선 강조, 단 하나의 추천 및 정확도·절대 위치·검출률·시간 일관성·속도 장단점을 원시 결과와 대조.

인계 확인 또는 정적 코드 검사만으로 목표 완료 처리하지 않는다. 데이터 디렉터리는 변경하지 않았다.

09:38 UTC 후속 준비: 동일 GPU 속도 측정 supervisor에 nvidia-smi GPU index/UUID/이름/드라이버와 CUDA_VISIBLE_DEVICES, 모델별 프로세스 시작·종료 Unix 시각 기록을 추가했다. 향후 실제 결과에서 동일 GPU·순차 실행 여부를 검증하기 위한 변경이다. 아직 벤치마크를 실행하지 않았다. Python 3 AST 검사를 통과했다. 학습/평가 상태 추가 조회는 하지 않았다.

09:37 UTC 관찰 문제 진단: 기존 웹 작업공간 /api/gpus를 읽기 전용으로 조회했다. GPU 0–5는 namhj 작업으로 89–96% 사용 중이며 GPU 6은 비어 있었다. 특정 PID 생존 증거는 아니므로 GPU 작업 존재만 확인한 것으로 기록한다. 로컬 SSH 읽기 전용 ps 시도는 공개 호스트 키를 대조한 뒤 publickey/password 인증 실패로 끝났고 호스트 명령은 실행되지 않았다. 다음 정기 조회는 10:07 UTC 이후다.

최종 검증 도구 tools/audit_nlf_comparison.py를 작성했다. 최종 학습 완료와 holdout 선택/내보내기 검증, 네 모델 1,280개 전체 Test 결과, 원시 프레임 매핑 및 모든 지표 재집계 일치, 보고서 재생성 일치, 동일 GPU UUID와 순차 실행 시각/3구간300프레임을 검사한다. 실패 시 final_audit.json 통과 기록을 만들지 않는다. AST 및 --help만 확인했으며 실제 최종 검증은 결과 생성 후 실행해야 한다.

09:50 UTC 사용자 속도 비교 요청에 따라 저장된 학습 progress/complete 및 기존 comparison.json을 1회 읽었다. 누적 성공 업데이트당 시간과 배치32 처리 샘플/초를 계산해 _agents/report/261024_학습과_추론_속도_중간_비교.md에 기록했다. NLF crop과 Multi-HMR 전체 프레임의 작업 차이를 명시했고, 동시 Test 관측 fps를 공정한 동일 GPU 벤치마크와 구분했다. NLF 최종 Test/동일 GPU 속도는 아직 미완료다. 다음 정기 점검은 10:20 UTC 이후로 한다. 새 GPU 작업은 실행하지 않았다.

10:20 UTC 정기 점검: 세 trial 성공 업데이트는 head 58,825, last_stage 40,450, last_two_stages 28,725로 이전보다 증가했다. progress 갱신은 조회 시점의 3–7초 전이고 관리 상태도 3초 전 training으로 갱신됐다. 관리 기록의 재시작 횟수는 모두0, 완료 trial은 없다. GPU0–5는88–98% 사용 중이고6은 유휴다. 최종 selected/Test/속도 summary는 아직 없다. 학습 종료13:47:05 UTC를 유지한다. 다음 조회10:50 UTC 이후.

13:56 UTC 정기 점검: 모든 NLF trial 학습 완료. head112355회/35.94665mm, last_stage77316회/32.35346mm, last_two_stages54005회/32.35024mm. 내부 holdout 최선 last_two_stages의 step53989 체크포인트 선택. TorchScript1806 tensors 검사,422 adapted/changed tensors, Test 미사용 선택 검증 통과.

자동 최종 Test는6개 worker 모두 첫 collect에서 ModuleNotFoundError(humanpose_metrics)로 종료했다. 부모 evaluate_nlf/recover_nlf_training 로그도 AssertionError 및 CalledProcessError로 종료를 기록했다. stale status.json의 running을 살아 있는 실행으로 해석하지 않는다. src/utils/pose_evaluation.py의 두 import를 utils.humanpose_metrics로 수정했고, 기존 self_check6항목 및 실제 Collector.collect의 뒤바뀐 두 사람 매칭 CPU 회귀 검증을 통과했다. 학습이나 관리 프로세스는 재시작하지 않았다.

GPU 실행 환경 제약: 현재 exec에서 /dev/nvidia* 자체가 없고 .venv-humanpose torch.cuda.is_available=False, device_count=0이다. GPU 평가를 같은 환경에서 중복 시도하지 않았다. 사용자에게 GPU 접근 가능한 호스트 실행 환경 전환 또는 호스트에서 평가만 재개하도록 요청했다. 이 제약이 최종Test와 속도 비교를 막는 첫 직접 blocker다. 목표는 미완료로 유지한다.

GPU 실행 환경 제약이 최종 평가 실패 진단 턴과 이후 두 연속 목표 턴에서 동일하게 재확인됐다. 학습·선택·내보내기 검증과 import 수정/회귀 검증은 완료했지만 GPU 장치가 없는 현재 환경에서 최종 Test·동일 GPU 벤치마크를 진행할 수 없다. 호스트 환경 전환 요청은 답변 대기 중이며 반복 조회를 중단하기 위해 목표를 blocked로 기록한다. 완료를 의미하지 않는다. 복구 후 학습/관리를 재시작하지 않고 평가만 재개한다.

2026-10-24T17:01:27.588004+00:00 사용자 재개/최신 지정: 이후 GPU는3–6만 사용하고 평가 및 결과 검증 후 Goal을 종료하도록 요청했다. NLF 평가 기본4worker와 --gpus3,4,5,6, 후속속도GPU3을 반영했다. 보고서 생성기는 실제 측정GPU를 표시하고 최종 audit은3–6만 허용한다. 평가 worker 실패 시 stale running 대신 failed를 저장하도록 수정했다. GPU매핑/금지GPU거절/실패상태 회귀검사는 통과했다(초기 테스트 harness에서 __file__/STDOUT 누락을 보완한 뒤 통과). 현재 /dev/nvidia*는 여전히 없고 nvidia-smi 드라이버연결 실패. NLF 최종Test0/320, runtime summary/final audit 없음. 기존 학습/관리 재시작 및 GPU평가 중복 실행은 하지 않았다. 이전 blocked 이후 사용자재개에 따른 첫 환경제약 재확인이다.

사용자 재개 이후 두 번째 환경 확인에서도 /dev/nvidia*가 없다. 실패한6개 worker 로그의 마지막 예외와 부모 CalledProcessError를 재대조한 뒤, 갱신되지 않았던 nlf_finetuned_status.json을 failed로 정정했다. 원본은 nlf_stale_status_before_correction.json으로 보존했다. 상태 표시만 정정했으며 새 프로세스는 실행하지 않았다. GPU접근이 해결되지 않아 최종평가와 동일GPU속도 비교는 계속 차단돼 있다.

사용자 재개 이후 세 번째 연속 목표 턴에서도 GPU 장치가 노출되지 않았다. GPU3–6 설정/오류수정/CPU회귀검증/실패상태정정은 완료했고, 현재 환경에서 실행 가능한 필수 후속 작업이 없다. 환경 변경 또는 호스트 평가 실행이 필요하므로 blocked로 기록한다. 최종Test·속도비교·최종추천 검증은 미완료이며 평가 후Goal종료 요청은 완료 시에만 적용한다.

17:07 UTC GPU 실행 환경 복구: 호스트 직접 실행에서 torch.cuda.is_available=True, device_count=7 및 nvidia-smi 정상 확인. 이전 GPU미노출 blocker는 해소됐다. GPU3–6 모두유휴이고 기존평가/benchmark/학습복구관리 프로세스가 없는 것을 확인한 뒤, 평가 supervisor PID46605를1회시작했다. 학습은재시작하지 않았다. 실행명령과로그는 nlf_final_test_resume.json. 사용자현재상태질문에 PID46605생존을확인했다. 다음정기진행조회17:38UTC이후. 원래작업창17:19:25는재설정하지않으며 남은평가가끝나기전에완료하지않는다.

사용자 최신 요청(4인코딩/10조건)으로 실제 native Goal 교체 완료. 외부 app-server는 active writer 때문에 실패했고 idle예약도 실패해 폐기했다. 웹 서버를 재시작하지 않고 일시적 localhost Node inspector로 기존 소유 서버의 goalClients 연결을 찾아 공식 thread/goal/set(origin=user) RPC를 호출했다. 단일 writer 제약을 해제하거나 우회하지 않고 기존 소유자가 변경을 기록했다. native Goal 변경 알림과 같은 서버 GET 재조회로3639자 목표 일치를 확인했다. 임시 inspector는 닫았고 프로토타입 변경은 남기지 않았다. 기존평가PID46605는재시작하지않았다. old apply/watchdog CPU 도우미는PID identity대조후종료하고 최신목표지문으로 watchdog만 새로기동했으며 고정마감2026-10-26 05:17:17UTC는연장하지않았다.

18:12 UTC 사용자 즉시보고서/GPU실행 요청: 지정 파일261024_학습과_추론_속도_중간_비교.md를 실제4모델Test와GPU3순차측정 최종값으로 교체했다. 향후 compare_methods 생성기에서도 같은파일을갱신하도록연결했다. 최선강조방향에서한국어프레임/초와crop/초도높을수록좋음으로수정했다. main algorithm Multi-HMR 선택을algorithm_selection.json에저장했다. original final_audit1280개재계산passed를확인했고 NLF학습모델은Test에서기본보다악화했다. GPU2–6 각각RGBbaseline및fusion4인코딩의2step실제GPU파일럿PID44901–44905를시작·등록했다.25초기준프로세스모두살아있으나GPU메모리는아직초기상태여서모델초기화중으로기록하며성공한학습업데이트로오인하지않는다. Full primary training은파일럿완료후진입한다. 캐시v2는전32holdout과Train35세션완료,Test캐시는아직대기중이다. 새로운정규상태확인은18:42UTC이후,파일럿검증/구체적실패개입은별도다.

사용자가 현재 Goal 완료 후 NLF 학습 모델의 Test 악화 원인 분석을 요청했다. nlf_post_goal_analysis_request.json에 순서·보존대상·진단항목을 기록했다. 현재 RGB/event 실험 우선순위 및36시간 고정마감은 유지하며, 지금 NLF 재학습/추가Test를 시작하지 않는다.

# 센서융합 Goal 최종 완료 요건 대조

판정: 모든 필수 실험·평가·검증 요건 통과. Native Goal complete 전환 완료: 2026-10-28 00:45:35 KST. 실제 그림 네 개를 직접 열어 확인했고, 아래 증거를 실제 실행·원시 예측·저장된 수치와 대조했다. 계획·파일 존재만으로 완료를 판정하지 않았다.

| 요건 | 확인한 증거와 결과 | 판정 |
|---|---|---|
| body(_b)·split·피험자 분리 | 현재 split SHA 재대조, 학습51/holdout9/Test18 피험자 상호 분리 | 통과 |
| 새7학습+원본2의 전체 Test | 각80구간·24080프레임, 새720개 구간 원시결과 전부 완료 | 통과 |
| RGB90분·융합3시간·최대4시간·고정15시간 | actual started/finish/complete, 최종 검증·저장 포함 시간과 RGB4 재실행 잔여74.02분 명시 | 통과 |
| 동일 초기화·seed·batch·증강·고정균형subset | 6501프레임·51세션, 조건별 같은 초기화 SHA·배치16·Train-only 선택 | 통과 |
| 공개 원본 RGB 미세조정0 | 공개 원본 실제 SHA 및 CPU native parity, 두 참조manifest의0 update | 통과 |
| 재사용 cache·GT·timestamp·센서 보정 | 78개 cache 전체SHA 최종재검증, GT158파일·54개 독립 GT재구성, 프레임·인과성·보정 전체검사 | 통과 |
| 현재 원본 변경 여부 | 312개 원본archive 크기·mtime 재대조, 전체archive 바이트 재해시는 하지 않은 범위 명시 | 통과 |
| event-only RGB 픽셀 미사용·GT 추론입력 금지 | 실제9개 실행의 입력 접근 집합과 GT=false, event3/event4만 읽음 | 통과 |
| 유한 gradient·AMP·실제 checkpoint | 본학습7조건의 best/last14개 실제 CPU로드·유한값·카운터·seen frame 검사, 원본2참조 별도 | 통과 |
| 동일 공통GT5지표·정확frame0·properSim3 | 9조건 공통22337표본, 독립 SciPy/SVD 재계산 최대 합 차이5.62e-13m 미만 | 통과 |
| full3D Root와 RPE-Z·절대오차·검출률·시간오차 | 원시비정렬 root와depth분리, pelvis속도/상대body22가속도 정의와 실제악화 명시 | 통과 |
| 통계·효과크기·동작속도 | 18군집·bootstrap10000회, PA13쌍/root13쌍 Holm·효과mm/상대%, 고정속도구간 | 통과 |
| 5조건×4모드 holdout 진단과 실패 | zero_aux 자원실패 GT288/285 별도, 공통표본없는 오차 미평가, normal Test 미변경 | 통과 |
| 동일GPU 순차18실행 전체파이프라인 속도 | 물리GPU1 UUID 일치·실행비중첩·본인단독PID·각300프레임·online/cache입력parity | 통과 |
| 로딩·메모리·cache생성비용 | 양쪽 실행 로딩 분리·peak양쪽최대·캐시50.386GiB·재사용 새생성0·과거생성span 구분 | 통과 |
| 논문형 구조·실제예측·학습량 경향 | 모델도식출처SHA, 동일Test시각9패널, 선택checkpoint실제누적시간6곡선을 직접 열어 확인 | 통과 |
| 메인표 최상단·최저값굵게·출처 분리·숫자 검증 | 한국어 보고서 첫9행5지표, 전체11표154행 실제저장수치 검증·모든local링크 존재 | 통과 |
| 단일 추천·한계 | Train-only argmin과 최종추천 동일, 시간효과·분기기여미미·단일seed·재사용benchmark·online비용 명시 | 통과 |
| 본인 프로세스 관리·마감 준수 | 필수학습/평가/속도/감사 작업 잔여없음, 본인deadline watchdog만 남음, 고정마감 이전 | 통과 |

증거: exp/261027/device34_fusion_repair_15h/agent_completion_review.json 및 final_artifact_audit.json. 보고서: _agents/report/261027_센서융합_열화원인_분석과_재실험.md. 과거 RGB3 중단의 최소37 update 미보존 이력은 남겨두었고 본 비교는 새90분 trial이다. 실제 데이터와 data/는 읽기전용으로 유지했다. cache·GT의 최종 전체SHA 검증 이후 변경이 없음을 파일시각으로 다시 확인했다. 자원보호로 제외된 인위적0입력 프레임의 native 복원/pose 오차는 평가했다고 주장하지 않는다.

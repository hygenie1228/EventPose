# 반복 RGB 두 조건 전체 Test 원시 검증

2026-10-26 22:16 UTC. seed261027의 rgb3와 rgb4 전체 Test가 각각 80구간·24,080프레임을 완료한 뒤 CPU로 각 원시 result.npz를 전부 읽어 검증했다. GPU 학습·추론을 재실행하지 않았다.

검증 도구: tools/audit_device34_completed_repeat_test.py. 각 실행은 이번 Goal managed_jobs.json에 등록했으며 CUDA_VISIBLE_DEVICES는 빈 문자열이다. rgb3 PID73542, rgb4 PID74729는 정상 종료했다.

두 조건 모두 split 해시·checkpoint 출처·조건·시작/끝 프레임·parent session·매칭 프로토콜 일치, 정확 frame0, 원본/구간 프레임 매핑, 모든 GT row 처리, 검출 예측 유한값, 미검출 NaN 보존을 확인했다. 원시 관절에서 다섯 지표를 SciPy proper Sim3로 독립 재계산하고 기존 segment_metrics의 합계 및 표본 수와 대조했다.

| 조건 | 전체 GT | 검출 | 자체 frame0 적격 표본 | 독립 합계 최대 차이(m) |
|---|---:|---:|---:|---:|
| device3 RGB | 24080 | 24059 | 24059 | 5.684341886080801e-13 |
| device4 RGB | 24080 | 24080 | 24080 | 1.4921397450962104e-13 |

증거: exp/261026/device34_body_28h/rgb3_seed261027_raw_test_cpu_audit.json 및 rgb4_seed261027_raw_test_cpu_audit.json. 둘 다 status=passed다. 자체 조건 표본 검증이며 6조건 공통 GT 성능표나 최종 전체 Goal 완료로 해석하지 않는다. 반복 fusion 두 조건의 전체 Test 및 event 기반 두 조건의 학습/Test, 동일 GPU 속도, 최종 감사가 남아 있다.

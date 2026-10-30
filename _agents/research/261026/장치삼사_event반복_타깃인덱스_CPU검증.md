# event-only 반복 실험 타깃 인덱스 CPU 검증

2026-10-26 23:59 UTC에 `tools/audit_device34_target_bounds_cpu.py`를 실제 실행하고 정상 종료를 확인했다.

- 대상은 `event3_event4`의 실제 학습 6,501프레임과 내부 검증 288프레임이다. 동결된 index의 프레임과 캐시 프레임 일치를 확인하고 실제 event3 내외부 보정으로 타깃 좌표를 변환했다.
- 실제 `multisensor_data.collate`와 `scripts/train_multihmr.py::targets`를 CPU에서 호출했다. 역할별 다섯 순열, batch 16과 마지막 잔여 batch를 포함하여 학습 2,035배치와 검증 90배치를 검사했다.
- 모든 배치의 사람/중심/카메라/영상 크기 첫 차원이 일치했고, 출력 배치 인덱스와 64×64 셀 인덱스가 범위 안이었다. 동일 IndexError는 재현되지 않았다.
- 센서 영상과 모델 추론은 실행하지 않았다. 이 검사는 GPU 실행, CUDA 비동기 오류, 하드웨어 메모리 오류를 검증하지 않으며 장애 원인을 확정하지 않는다. GPU3 장치 오류와 CUDA 초기화 실패는 여전히 별개의 실제 관측이다.
- 산출물: `exp/261026/device34_body_28h/target_bounds_cpu_audit.json`. 실행은 본 Goal 관리 프로세스로 등록했다. 데이터와 학습 설정은 수정하지 않았다.

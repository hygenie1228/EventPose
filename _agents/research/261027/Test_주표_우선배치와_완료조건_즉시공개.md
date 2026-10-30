# Test 주표 우선 배치와 완료 조건 공개

한국시간2026-10-27 20:25부터 보고서 제목 바로 다음에 실제 Test5지표 표를 배치했다. 자동 갱신과 최종 생성에 같은 규칙을 적용했다. Train holdout은 checkpoint 선택용 보조 정보다. 전체9조건 종료까지 Test 표를 비워두던 문제를 수정해80구간 전체 평가를 마친 조건부터 독립 재계산 및 frame0/원시매핑 확인 후 표시한다. 완료된 조건 집합의 공통 GT 표본 수와 변경 가능성을 함께 표시한다. 미완료 조건의 수치를 만들지 않는다.

학습이 완료된 RGB와 융합 조건의 다음 Test 배정을 우선하도록 scheduler만 교체했다. 기존 GPU 학습/평가 PID는 보존했다. Test 결과를 Train 선택이나 학습 조정에 쓰지 않는다. 관리자와 보고서 갱신기만 신원 확인 후 재기동했다.

증거: main_table_first_actual_Test_numeric_proof.json, interim_full_test_comparison.json, test_first_reporting_reload.json. 현재 원본 RGB3의 전체80구간·24080프레임과 공통적격24000표본에 대한5지표를 실제 원시 예측에서 재계산했다. 아직 새7학습조건의 전체 Test 결과가 아니며 Goal 완료를 의미하지 않는다.

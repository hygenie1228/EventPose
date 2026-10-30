# 단축 RGB와 장기 융합의 실제 학습 예산 검증

네 RGB 대 융합 비교쌍의 설정과 실제 started.json의 종료 시각을 독립 확인했다. RGB-only는90분, RGB+event는180분이며 seed·batch16·6501 Train/288 holdout·증강·초기화 SHA·학습률·warmup·선택 규칙이 동일하다. 설정의 hours=3은 동일 optimizer 스케줄을 유지하며 RGB의 실제 finish_unix가90분 종료를 적용한다. 최종 저장과 검증 시간, 실제 성공 update 및 선택 checkpoint의 학습량은 종료 후 별도로 확인한다.

학습 시간 효과가 포함되므로 센서 추가만의 효과로 해석하지 않는다. Test 결과나 학습 완료를 이 검증으로 주장하지 않는다.

증거: exp/261027/device34_fusion_repair_15h/short_rgb_long_fusion_actual_budget_audit.json

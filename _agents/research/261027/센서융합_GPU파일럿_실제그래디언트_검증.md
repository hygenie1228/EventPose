# GPU 파일럿 실제 그래디언트 검증

## 실제 GPU 학습 전 검사 결과

2026-10-27 16:17 KST에 종료된 다섯 GPU pilot의 JSONL을 별도로 대조했다. 각8회, 총40회 optimizer update가 모두 성공했고 loss가 유한했다. 세 융합 조건의 event encoder는 각각8회 중7회에 양의 gradient를 보였고 event-only 입력 adapter는8회 모두 양의 gradient를 보였다. 첫 update의 보조 encoder gradient0은 zero-initialized output이 연결된 구조에서 예상되는 동작이다. 두 RGB 조합의 학습 전 holdout 검출 수·MPJPE·PA-MPJPE는 해당 RGB 단독과 정확히 같았다. Test는 사용하지 않았다. 근거: `gpu_pilot_independent_review.json`. 이는 작은 pilot의 연결·gradient 검사이며 본학습 수렴이나 Test 개선을 입증하지 않는다.

2026-10-27 16:16 KST 사용자 요청 점검에서 실제 본학습 네 작업의 GPU 사용률은 모두100%였다. 장치 전체 메모리는 GPU1 5.72GiB, GPU2 5.72GiB, GPU4 31.99GiB, GPU5 5.70GiB였다. RGB 기반은 첫30분 backbone 동결 상태이고, event-only는 입력 adapter 학습을 위해 backbone을 통과하는 입력 gradient를 유지하므로 메모리 요구량이 다르다. 이 순간 관측은 최종 peak 메모리나 공정한 동일GPU 속도 벤치마크를 대체하지 않는다.


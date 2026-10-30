# 완료된 네 조건의 두 seed 공통 GT 점검

## 전체 Test가 끝난 네 조건의 반복 seed 점검

첫 네 조건의 두 seed, 총 여덟 실행에서 검출과 정확 frame0 적격 조건을 동시에 만족한 동일 GT 24,058개로 재계산했다. Test 80구간·24,080프레임을 모두 읽고 독립 SciPy 회전 적합과 기존 집계의 다섯 지표 합을 대조했다. 주표의 여섯 조건 공통 GT 21,659개와 모집단이 다르므로 이 표와 주표의 수치를 직접 비교하지 않는다. 뒤의 event 주센서 두 조건은 아직 반복 Test가 없어 포함하지 않았다.

| 완료 조건 | seed261026 PA-MPJPE (mm) | seed261027 PA-MPJPE (mm) | seed261026 3D Root error (mm) | seed261027 3D Root error (mm) |
|---|---:|---:|---:|---:|
| device3 RGB | 24.64 | 24.59 | 55.49 | 54.58 |
| device4 RGB | 24.13 | 24.12 | 50.09 | 51.10 |
| device3 RGB + device4 RGB | 24.65 | 24.54 | 53.98 | 51.19 |
| device3 RGB + device4 event | 24.62 | 24.50 | 55.49 | 51.46 |

위 PA-MPJPE는 frame-aligned 지표다. 조건별 seed 간 PA-MPJPE 변화는 최대 0.12 mm, 3D Root error 변화는 최대 4.03 mm였다. 두 seed로 학습 변동성의 분포를 확정할 수 없으며, root 개선은 seed 민감성을 함께 고려해야 한다. 이 부분 점검은 전체 여섯 조건의 반복 비교나 최종 추천을 대체하지 않는다. 근거: `completed_four_seed_pairs_common_gt.json`.


실행 PID87506을 Goal에 등록하고 실제 원시 예측을 처리했다. 종료 후 산출물과 독립 합의 최대 차이 5.684341886080801e-13 m를 확인했다. 보고서 실제 렌더링 표의 16개 수치를 JSON과 대조했다. dataset/data와 학습 설정은 수정하지 않았다. GPU2 남은 반복 학습은 보존했다.

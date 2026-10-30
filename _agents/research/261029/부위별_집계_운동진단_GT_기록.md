# 부위별 집계의 운동 진단 GT 생성

2026-10-29 한국시간. 이 기록은 학습된 attention을 평가하기 전의 진단 자료 생성에 관한 것이다.

구현은 `tools/prepare_part_aggregation_motion_diagnostics.py`, 결과는 `exp/261029/part_aware_dynamic_aggregation/diagnostics/holdout_part_motion_targets.npz` 및 같은 이름의 JSON이다. 고정 Train holdout 9개 세션·9명·288개 질의를 사용했다. Test는 사용하지 않았다.

실제 device3 RGB 시각 이전 200ms에 해당하는 원본 device0 SMPL-X GT를 CPU에서 재구성했다. GT 관절의 약 30Hz 궤적을 사용하며, 시작점은 관측된 과거 GT 사이에서 3D 선형 보간하고 끝점은 최신 과거 GT를 유지했다. RGB보다 미래인 GT는 사용하지 않았다. 끝점 유지 시간은 최대33.279ms이며 입력 RGB와 평가 GT common clock의 차이는 최대33.9ms다. 이는 센서의 고속 운동 정답을 복원한 것이 아닌 진단용 근사이다.

주 진단 값은 native event3 카메라에서의 200ms 평균 관절 경로 길이(px)이다. RGB3 카메라 경로, 카메라 3D 경로(m), pelvis를 뺀 3D 경로(m), 두 끝점의 변위도 따로 저장했다. 왕복 운동을 놓치지 않도록 경로 길이를 끝점 변위와 구분한다. `sessions`·`frames`로 모델의 attention export와 같은 질의를 연결할 수 있다.

여섯 부위는 torso, left_arm, right_arm, left_leg, right_leg, head이며 모델의 PART_JOINTS 정의와 같다. 288×6을 독립 표본1728개로 취급하지 않고 세션·피험자 군집을 유지해야 한다. 관측 나이와의 관계를 조사할 때 null 선택과 이벤트 선택 조건의 구간 중심 나이를 분리해야 한다.

공식 SMPL-X forward와 축약 FK의 최대 좌표 차이는2.384185791015625e−7m였다. 288개 질의 모두 과거 GT의 충분한 범위와 정상 깊이를 만족했다. 경로 길이가 끝점 변위 이상인지, 질의가 유일하고 원래 holdout 순서와 일치하는지, 모든 사용 GT의 시각이 RGB 이하인지 검증했다. 원본9개 label 파일의 크기와 수정 시각도 보존되었다. GPU를 사용하지 않았고 계산은 약3.3초였다.

운동 진단 GT는 모델 입력이나 학습 손실로 제공하지 않는다. 자료의 팔 경로 중앙값이 다리보다 크다는 관찰만으로 학습형 집계가 성공했다고 해석할 수 없다. 정확도 비교와 실제 attention의 부위별 선택을 함께 검증해야 한다.

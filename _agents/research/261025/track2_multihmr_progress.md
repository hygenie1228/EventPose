# 트랙2 대상 선택 오류 수정 연구

갱신: 2026-10-26 10:13:41 UTC

## init_v6 / init_v0 비교 — 대상 인물 선택 재검증

이전 마스크 확장 gate에서 배경 사람의 머리가 선택된 것을 확인했다. 이전 모델 점수·개선율은 유효한 target-only 비교에서 제외하고 다시 평가한다. 제공된 init_v6/init_v0 기준값은 유지한다. 새 모델 평균은 동일 Test10 전체7610프레임의 strict 선택·공통 MHR 채점을 검증한 뒤에만 표시한다.

| 모델 | 완료 Test | CD-H cm ↓ | ACC-H cm/frame² ↓ | init_v6 대비 CD-H 개선율 |
|---|---:|---:|---:|---:|
| init_v6 | 10/10 | **12.4260** | 2.9196 | +0.00% |
| init_v0 | 10/10 | 12.8131 | 2.4191 | -3.12% |
| body_original_RGB | 10/10 | 39.9698 | **1.8755** | -221.66% |
| body_legacy_pair_head_cd100_native | 10/10 | 15.8738 | 2.3378 | -27.75% |
| body_native_head_cd100 | 10/10 | 15.4441 | 2.3195 | -24.29% |
| body_native_head_cd100_rel100 | 10/10 | 15.2625 | 2.2830 | -22.83% |
| body_native_last2_cd100_rel100 | 10/10 | 14.9887 | 2.2418 | -20.62% |
| body_original_fixed_crop | 10/10 | 23.1649 | 2.5477 | -86.42% |
| body_fixed_head_cd100_rel100 | 10/10 | 23.1649 | 2.5477 | -86.42% |
| body_fixed_last2_cd100_rel100 | 10/10 | 18.1401 | 2.7577 | -45.98% |
| body_fixed_last4_cd100_rel100 | 10/10 | 17.8866 | 2.6923 | -43.94% |
| native_surface_head_cd100_rel100 | 10/10 | 18.0707 | 2.6507 | -45.43% |
| native_surface_last2_cd100_rel100 | 10/10 | 17.7938 | 2.7387 | -43.20% |
| native_surface_last4_cd100_rel100 | 10/10 | 18.2120 | 2.7069 | -46.56% |

수정v3 원본 전체RGB의 Test10 전체7610프레임을 독립 검증했다: CD-H 39.9698cm, ACC-H 1.8755cm/frame². 따라서 확인된 배경 인물 선택 오류를 고쳤다는 사실만으로 큰 원본 오차가 해결되거나 성능 차이 전체가 설명되지는 않는다. Train의 동일 원본 가중치 전체RGB/crop 비교는 입력 방식 영향의 증거이고, stroller raw SMPL-X의 허벅지 오검출은 변환 전 모델 오차의 직접 증거다. init_v6가 SAM3D Body이고 여기 원본이 Multi-HMR이라는 출처 차이도 구분한다. 각 요인의 전체 Test 오차 기여율을 분리했다고 주장하지 않는다.

원본 마스크 안으로 배경 사람의 head query가 투영되는 가림 사례가 추가 확인됐다. 원본 전체RGB의40.4357cm는 target-only 유효 점수에서 제외한다. 이전v2 나머지 모델 점수도 머리 점 검사 후의 잠정값으로 보존하고, 이 표는 몸의 투영 범위까지 검사하는v3 결과로 다시 채운다. 다음 학습은 CPU 대기 상태에서 중단해 잘못된 선택 규칙을 이어받지 않게 했다. [직접 확인한 raw SMPL-X/MHR 배경 인물 전환 영상](/home/namhj/EventPose/.experiment_storage/track2_multihmr_12h/target_foreground_visuals/strict_original_RGB/frames_305_306_319/spray_bottle-left-desk-ground-02.jpg).

몸 검사 기준은 GT pose나 Test 점수를 읽지 않고 Train4의 기존9개 예측과 원본 mask만으로 고정했다. 각 모델 body box IoU의 하위1% 값 중 최솟값 0.334506를 공통 최소값으로 쓴다. strict head gate와 body gate를 모두 통과한 후보 중 body IoU, confidence 순으로 선택한다. 저장된 spray 예측에서 대상 배우는 IoU 약0.70~0.77, 잘못 선택된 배경 사람은0.14~0.19로 분리됐다. 실제 같은 forward 후보의 재실행 확인과 전수 재평가를 진행한다. `body_target_gate_Train_rule.json`.

실제 GPU 재실행 확인 완료: 동일 원본 가중치·RGB·K·같은 forward 후보에서 기존 선택을 저장 score/pelvis Z와1e-6 이내로 재현했다. 몸 검사만 추가해 배경 후보가 제외되고 대상 배우가 선택됐다. GT pose나 Test 점수는 읽지 않았다.

| spray 프레임 | 기존 선택 깊이 m | 기존 body IoU | 새 선택 깊이 m | 새 body IoU |
|---|---:|---:|---:|---:|
| 305 | 2.1885 | 0.7275 | 2.1885 | 0.7275 |
| 306 | 3.6858 | 0.1928 | 2.2109 | 0.7077 |
| 319 | 3.6944 | 0.1633 | 2.3088 | 0.7769 |

306·319의 기존 골격은 배경에 앉은 사람을 따랐고 새 선택은 대상 배우다. 가중치 학습 효과나 전체 Test CD-H 개선 수치가 아니라 선택 오류의 인과 확인이다. 같은9개 checkpoint를 새 기준으로 다시 평가한다. `body_target_detector_replay_check.json`.

현재 source Uranus는 이전v2 경로의 중간 출력으로 보존한다. 몸 일치 검사v3의 전체 비교·실제30개 저장·시각 확인·init_v6 성능 조건이 충족되기 전에는 최종 성공 결과로 간주하지 않는다.

## 확인된 오류와 수정

blue_easter_basket-place-10의898프레임에서는 대상 배우의 머리가 선택됐지만899·900프레임에서는 뒤에 앉은 사람의 머리 위치가 선택됐다. 저장 pose/shape와 vertices에서 복원한 head ray의 위치가 약110px 이동했고, 잘못 선택된 점은 원래 대상 마스크 밖5.39px였다. 기존 ellipse15 확장이 그 점을 허용했다. RGB 확대 영상에서 배경 머리 위치임을 직접 확인했다. 앞선 “배경 인물 전환이 아니다”라는 판단을 정정한다.

증거: [마스크와 선택점 확대 영상](/home/namhj/EventPose/exp/261025/track2_multihmr_12h/target_foreground_visuals/diagnostic_original_static_crop/frames_898_899_900/query_mask_closeup.png), `head_query_projection_diagnostic.json`. 복원 vertex의 median 잔차는 약0.3mm다.

현재 v3는 원본 binary target mask 안의 head query와 예측 몸 범위의 대상 box 일치 검사를 함께 요구한다. 머리 점만 검사하는 v2는 가림 상황에서 배경 인물을 허용하므로 충분하지 않다. RGB 배경은 유지하고 마스크를 RGB에 씌우지 않는다. GT 3D·init pose로 후보를 고르지 않는다. 미검출은 관측된 선택 예측으로 보완하고 모든 원본 프레임을 평가한다. 학습 checkpoint는 보존하며 선택/평가만 먼저 바로잡는다.

실행 상태: `all30_ready_for_independent_strict_audit`. 정상 GPU1·2·4·5·6을 사용하고 GPU0은 보존한다.

이전 score·보고서는 `invalid_dilated_head_gate/`와 기존 prediction 폴더에 진단 기록으로 보존한다. 기존 source Uranus도 strict 재검증 전에는 최종 유효 결과로 간주하지 않는다. GT와 init 파일은 바꾸지 않는다.

실제 GPU detector 재실행으로 원인을 분리 확인했다. 같은 원본 가중치·RGB·K·동일 forward 후보에서 gate만 비교했다. 기존 선택은 저장된 prediction의 score·pelvis Z와1e-6 이내로 일치했다. Test GT나 init pose를 읽어 후보를 고르지 않았다.

| 프레임 | 기존 확장 gate 선택 깊이 m | 수정 원본 mask 선택 깊이 m |
|---|---:|---:|
| 898 | 2.7066 | 2.7066 |
| 899 | 6.3386 | 2.7071 |
| 900 | 6.2770 | 2.7256 |

899·900에서는 기존 gate의 선택이 배경 후보였고 수정 gate는 대상 후보를 선택했다. 즉 이 급변은 원본 모델의 depth 회귀 실패로 설명할 일이 아니라 인물 선택 오류였다. 전체 점수의 영향을 확정하려면9개 후보 전수 재평가가 필요하다. `strict_detector_replay_check.json`.

실제 strict 추론 검증: `complete_actual_strict_head_inference`. 원본 및 미세조정5개 모델의 동일 basket 영상 전체980프레임을 재추론했다. 이는 전체 Test10 평균이 아니다.

동일 basket980프레임·원본 가중치의 raw pelvis 최대 연속 프레임 이동 진단:

| 입력 | 기존 확장 gate cm | 수정 원본 mask gate cm |
|---|---:|---:|
| 전체 RGB | 205.9448 | 7.5951 |
| 고정 crop RGB | 365.5031 | 12.3595 |

원본 네트워크 가중치를 유지하고 선택 gate만 수정한 결과다. 이 값은 해당 한 영상의 raw 시간적 급변 진단이며 CD-H·ACC-H나 전체 Test 평균이 아니다. `strict_pilot_temporal_diagnostic.json`.

## 원본 crop의 변환 전후 및 이동·자세 진단 — 전체 Test10

CPU에서 같은7610프레임을 재계산했다. raw mapped SMPL-X surface proxy 22.7668cm, 변환 후 MHR CD-H 23.1649cm, init_v6 MHR CD-H 12.4260cm다. 큰 오차가 변환 전부터 존재하므로 MHR 변환만으로 전체 격차를 설명할 수 없다. 공간마다 frame0 Sim3를 따로 fitting하므로 두 값 차이를 가산 가능한 변환 오차로 해석하지 않는다.

| 전체 Test10 진단 | 원본 crop MHR | init_v6 MHR |
|---|---:|---:|
| 프레임별 중심 위치 보정 후 CD cm | 9.3840 | 5.4416 |
| 프레임별 Sim3 보정 후 CD cm | 7.0144 | 4.3532 |
| 중심 이동 궤적 RMSE cm | 18.1536 | 7.8860 |

위 보정은 GT를 읽는 사후 원인 진단이며 공식 점수·실제 예측·export에는 적용하지 않는다. 고정 frame0 정렬 이후 이동 오차와, 위치/Sim3를 보정해도 남는 자세·표면 오차가 모두 확인된다. 신경망 내부의 원인이나 요인별 가산 기여율을 확정한 결과는 아니다. 모든 입력 SHA와 프레임 수, CPU MHR 채점 재현을 확인했다. `selected_geometry_diagnostics/body_original_fixed_crop.json`.

### 영상별 원본 crop / init_v6 격차

전체 프레임 진단을 MHR CD 격차순으로 정리했다. 전부 cm이며 궤적 RMSE는 별도 진단 지표다. Test 진단을 모델 선택이나 학습 recipe에 적용하지 않았다.

| 영상 | 프레임 | raw SMPL-X proxy | 원본 MHR CD | init_v6 CD | CD 격차 | 중심 궤적 RMSE 원본 / init |
|---|---:|---:|---:|---:|---:|---:|
| baseball_bat-rotate-05 | 803 | 39.727 | 41.278 | 10.042 | +31.236 | 33.908 / 4.828 |
| toy_stroller-back-forth-ground-04 | 946 | 27.390 | 28.568 | 12.385 | +16.184 | 25.742 / 7.821 |
| rice_cooker-clockwise-04 | 553 | 28.850 | 29.130 | 13.646 | +15.484 | 23.516 / 8.858 |
| round_table-flip-08 | 993 | 23.690 | 23.144 | 10.305 | +12.839 | 17.677 / 5.692 |
| blue_easter_basket-place-10 | 980 | 20.222 | 19.905 | 11.585 | +8.320 | 18.458 / 7.287 |
| spray_bottle-left-desk-ground-02 | 477 | 24.292 | 25.109 | 18.163 | +6.947 | 22.004 / 14.170 |
| cane-swing-05 | 886 | 15.294 | 15.881 | 10.345 | +5.536 | 9.021 / 5.247 |
| tall_bar_stool-pull-08 | 678 | 16.556 | 16.862 | 11.516 | +5.346 | 11.047 / 5.985 |
| watering_can-pour-right-02 | 748 | 13.971 | 13.944 | 10.613 | +3.331 | 7.451 / 6.052 |
| monitor-rotate-05 | 546 | 17.676 | 17.827 | 15.661 | +2.167 | 12.711 / 12.919 |

## 후속 native-MHR surface 세 후보 실제 학습 완료

세 후보의 검색과 Train20 refit이 완료됐다. 복구 시 보존한 학습 시간과 별도 실패 시도 시간을 합산했다. 아래는 실제 완료 기록이며 새 Test 성능은 전체 MHR 평가가 끝난 뒤 별도로 비교한다.

| 모델 | search+refit+복구 시간 h | refit step | 성공 update |
|---|---:|---:|---:|
| native_surface_head_cd100_rel100 | 0.3874 | 341 | 340 |
| native_surface_last2_cd100_rel100 | 0.8618 | 555 | 554 |
| native_surface_last4_cd100_rel100 | 0.5675 | 219 | 218 |

모든 refit은2GPU DDP, ColorJitter·AdamW decay0.1·L2-SP를 사용한다. 가중치 변경/유한값 독립 감사는 `completed_native_surface_refit_parameter_audits.json`에 완료 시 기록한다. 모델 선택은 새3개를 포함한 전체12개 Train4 MHR 비교가 끝나야 고정한다.

후속 학습의 실제 validation mask-video open timeout을 확인해 복구했다. 원본30개 마스크를 SHA가 같은 SSD 캐시로 복사하고 검증 디코더를1thread로 제한했다. barrier로16개 reader를 동시에 연 CPU 검사와 source/cache 픽셀 일치 검사를 통과했다. 완료된 head 검색은 보존하고 last2는 optimizer/step/원래 시작시각을 복원하며, baseline 전 실패한 last4는 같은 recipe로 다시 실행한다. 실패 로그·기존 checkpoint·이전 시도 시간을 보존했고 전체 마감·채점·인물 선택 기준은 바꾸지 않았다. `native_surface_mask_decode_recovery.json`, `target_mask_decoder_recovery_preflight.json`.

현재 body-v3 선택 모델의 실제 source Uranus30개에 대해 최초·중간·마지막90표본을 모두 직접 열어 확인했고, image/source/raw SHA와 개별 관찰을 기록했다. 확인 표본의 raw SMPL-X와 exported MHR는 대상 배우를 따른다. watering의 약0.4~0.6m 양의 깊이 편향, 굽힌 자세·가려진 팔다리 오차 등은 남는다. 이는90표본 검토이며22990프레임 전수 수동 identity 확인이나 init_v6 성능 초과를 뜻하지 않는다. 새 후보가 선택돼 source가 바뀌면 새 파일을 다시 검토해야 한다. `body_target_export_visual_manual_review.json`.

후속 native-MHR surface 학습의 시작 여유시간은 기존5시간에서4시간으로 조정했다. 실제 완료된 subprocess 소요시간을 근거로 한 자원 배치 추정이며 완료 보장은 아니다. 후보별 search+refit 최대1시간, 실제2GPU DDP, 전체28시간 마감과 후보당3시간 제한 및 사전 고정 recipe는 유지한다. 자식 작업 없이 기다리던 CPU 관리자 두 개만 다시 실행했고 GPU 재평가는 유지했다. `body_target_surface_budget_admission_update.json`, `body_target_historical_runtime_estimate.json`.

## Train4 전체 MHR 비교

| 모델 | CD-H cm ↓ | ACC-H cm/frame² ↓ |
|---|---:|---:|
| init_v6 | **15.1588** | 2.7290 |
| body_original_RGB | 38.9376 | **1.9532** |
| body_legacy_pair_head_cd100_native | 22.7091 | 2.4335 |
| body_native_head_cd100 | 22.3785 | 2.3229 |
| body_native_head_cd100_rel100 | 22.2192 | 2.3520 |
| body_native_last2_cd100_rel100 | 21.7826 | 2.3143 |
| body_original_fixed_crop | 19.3090 | 2.8169 |
| body_fixed_head_cd100_rel100 | 19.3090 | 2.8169 |
| body_fixed_last2_cd100_rel100 | 18.9713 | 2.9031 |
| body_fixed_last4_cd100_rel100 | 18.8586 | 2.9187 |
| native_surface_head_cd100_rel100 | 18.7185 | 2.8971 |
| native_surface_last2_cd100_rel100 | 18.2688 | 2.9189 |
| native_surface_last4_cd100_rel100 | 17.4393 | 2.8798 |

각 행은 동일4영상 전체2714프레임이다. 일부 영상 평균을 표시하지 않는다. Test와 다른 모집단이며 모델 선택은 이 Train 검증만 사용한다.

Train 기준 선택: `native_surface_last4_cd100_rel100`. 새 Test 점수를 보기 전에 고정한다.

선택 모델의 동일 입력 방식 원본 대조군 대비 Train CD-H: 19.3090→17.4393cm, +9.68% 개선. 전체 RGB와 crop 사이의 차이를 미세조정 효과로 합산하지 않는다. Test 개선율과는 별개다.

확인된 입력 방식 영향: 같은 원본 가중치·원본 mask 선택 기준에서 전체 RGB 38.9376cm, 고정 crop RGB 19.3090cm로 Train CD-H가 50.41% 낮아졌다. crop은 RGB 배경을 지우지 않으며 K도 함께 변환한다. 이 비교는 crop으로 함께 바뀌는 인물 크기·입력 시야각 등의 효과를 분리하지 않으므로 FOV 하나만 원인이라고 단정하지 않는다.

비교 모델의 출처: 제공된 init_v6는30개 NPZ 내부 report가 각 SAM3D Body diagnostics와 일치하는 생성 결과다. 여기서 원본 대조군은 Multi-HMR 사전학습 가중치다. 따라서 init_v6와 원본은 같은 모델의 학습 전후가 아니다. 이 출처 차이만으로 정확도 차이 전체를 설명하지는 않는다. `init_v6_generation_provenance_audit.json`.

몸 범위 검사v3의 원본 전체RGB stroller 경고2개(433→434,434→435)를 직접 확인했다. 432~436의 RGB/mask/head query 창과433/434/435의 raw SMPL-X/MHR 골격에서,434의 query가 대상 배우 허벅지에 놓이고 작은 잘못된 골격을 예측하는 것을 확인했다. 배경의 앉은 사람 전환은 아니지만 정상 머리 검출도 아니다. raw 깊이2.383→3.035→2.360m의 급변은 MHR 변환 전에 존재한다. 모든 프레임을 유지하며 이 Test 진단으로 모델 선택·threshold·recipe를 바꾸지 않는다. `strict_temporal_manual_reviews.json`의 body_original_RGB/toy_stroller-back-forth-ground-04 항목.

직접 시각 확인: `body_fixed_last4_cd100_rel100`의 Train 검증4영상 최초·중간·마지막 총12프레임을 열어 봤다. 해당12표본에서는 물체를 조작하는 대상 배우를 따르고 배경의 앉은 사람을 선택하지 않았다. 팔·다리 자세와 카메라 깊이 오차는 남아 있다. 전체 프레임 인물 일치나 최종 source 파일 확인을 대신하지 않으며 모델 선택도 바꾸지 않았다. 이미지·raw·MHR SHA와 개별 관찰은 `body_target_Train4_selected_visual_review.json`, 영상은 `target_foreground_visuals/search_body_fixed_last4_cd100_rel100/validation/`에 있다.

독립 검증 완료 모델의 Test 큰 급변 표시 구간: 12개 모델에서 2개. 원본 mask 픽셀 밖 head가0개인지 전체 프레임에서 재검사한다. root 이동50cm 또는 관측 head 이동75px를 넘는 구간은 추가 수동 확인 대상으로 기록하며 예측 선택이나 채점 프레임을 바꾸지 않는다. `strict_independent_Test_models.json`, 각 영상의 `strict_independent_head_audit.json`.

## 추가 확인: 대상 몸 안의 잘못된 머리 검출

stroller 원본 전체RGB에서 표시된6개 급변 구간을 모두 직접 열어 확인했다. 배우가 몸을 숙였을 때 검출기가 같은 대상 배우의 허벅지를 머리 후보로 내고 상체 후보와 번갈아 선택했다. 앞선 basket의 배경 사람 선택과는 다른 검출 오류다. 원본 mask 안이라는 검사만으로 해부학적 머리 위치가 맞다고 보증할 수 없다. 이 잘못된 예측도 그대로 채점하며 프레임이나 점수를 제거·수정하지 않는다.

| 동일 stroller946프레임 입력/모델 | raw pelvis 최대 프레임 이동 cm | 관측 query 최대 프레임 이동 px |
|---|---:|---:|
| strict_original_RGB | 98.5271 | 149.4235 |
| strict_original_fixed_crop | 9.2455 | 15.6384 |
| strict_fixed_last4_cd100_rel100 | 14.1133 | 15.1865 |

원본 가중치 crop 대조군에서는 해당 큰 급변이 줄었다. 이는 한 영상의 검출·시간 진단이며 CD-H나 Test10 평균이 아니다. 선택 모델의438·439·455프레임 raw SMPL-X/MHR도 직접 확인했다. 대상 배우를 따르지만455의 팔 뻗기 등 자세 오차는 변환 전부터 남는다. 모델·예측·채점·사전 고정 추가 학습 recipe는 바꾸지 않았다. `strict_stroller_false_head_diagnostic.json`, `strict_temporal_manual_reviews.json`.

strict 선택 모델의 source Uranus 저장·SHA 일치: 30/30개.

실제 최종 source 파일 시각화 준비: `actual_source90_samples_manually_reviewed`. strict 30개 저장과 원본 mask 검증이 끝난 뒤 CPU에서 최초·중간·마지막 총90프레임을 렌더링한다. 렌더링 완료와 직접 확인 완료는 구분한다.

별도 CPU 인물 선택 검사: 현재 148개 예측 영상의 전체 108668프레임에서 수락된 head가 원본 mask 안에 있는지 검사했다. 큰 급변 표시는 2개다. 표시된 구간은 배경을 유지한 RGB·원본 mask 윤곽·head 선택점을 확대 렌더링하며 직접 확인은 별도로 기록한다. 이 검사는 CD-H 평균이나 전체 인물 일치의 증명이 아니다. `strict_identity_visual_status.json`, `strict_identity_visuals/`.

수정 경로의 완료 조건 독립 점검: 9/10개 충족. 데이터·학습·선택·Test·최종 저장·성능·시각 확인·보고서·자산·수행시간을 각각 검사한다. 미충족 항목이 있으므로 최종 성공으로 보고하지 않는다. `body_target_final_completion_scope_audit.json`.

추가 학습 전 native MHR-GT surface 정합성 pilot: `complete_actual_native_surface_two_GPU_pilot`. 기존9개 strict 재평가·30개 저장이 끝난 뒤 유휴 GPU2개에서20step만 검사하며 pilot 가중치는 결과로 사용하지 않는다. 입력 방식은 사전 고정 규칙으로 두 원본 대조군의 전체 Train4 CD-H/ACC-H만 비교해 정한다. RGB 배경을 유지하고 내부 검증도 확장하지 않은 원본 mask를 쓴다. 추가3개 본 학습 상태: `all_native_surface_searches_and_Train20_refits_complete`.

추가 학습 전 별도 Train 검증 명령 검사: `passed_actual_GPU_validation`. pilot의 실제 마지막 업데이트 가중치로 전체 Train4 proxy를 GPU6에서 검사하고, strict mask·원본 dilation0·native MHR-GT target·checkpoint SHA를 확인한 뒤3개 검색 학습을 시작한다. 내부 검증 결과의 strict metadata 누락은 사전에 수정했다. 이 GPU 검사는 아직 통과하지 않았다면 완료로 표시하지 않는다.

추가 후보의 Test 작업은 전체 Train4 CD-H/ACC-H 순위가 높은 모델부터 해당 Test10을 먼저 완료하도록 배치한다. 첫 전체 평균을 일찍 확인하기 위한 실행 순서이며, 나머지 후보나 프레임을 제외하지 않고 Test 점수로 순위를 바꾸지도 않는다. 현재 실행 중인 기존9개 재평가의 순서는 바꾸지 않았다.

## 실제 학습과 데이터

30개 영상을 seed261025로 Train20/Test10에 고정했다. Train 안에서는16개 학습/4개 내부 검증을 분리했고, 캐시는851개 학습 프레임·256개 검증 프레임이며 Train20 refit은1107개다. Train20의 MHR GT만 SMPL-X 감독 신호로 변환했다. 최종 비교는 예측 SMPL-X를 MHR로 변환한 뒤 수행한다. Test GT는 학습이나 후보 선택에 쓰지 않는다.

아래6개 학습은 실제 GPU2개 DDP로 완료됐다. ColorJitter·AdamW decay0.1·원본 가중치에 대한 L2-SP를 적용했다. 총 학습 시간은 bootstrap/부모 학습·검색·Train20 refit을 합산한 비용이다. 현재v3 재평가는 이 가중치를 보존하고 인물 선택을 다시 검사하는 단계다.

| 학습 trial | GPU 수 | 누적 실제 학습 시간 h | 평가용 Train20 checkpoint step | 파라미터 변화 |
|---|---:|---:|---:|---|
| native_head_cd100 | 2 | 2.7119 | 2354 | 있음 |
| native_head_cd100_rel100 | 2 | 2.7121 | 2517 | 있음 |
| native_last2_cd100_rel100 | 2 | 2.7133 | 1990 | 있음 |
| fixed_head_cd100_rel100 | 2 | 0.3578 | 0 | 0개 tensor |
| fixed_last2_cd100_rel100 | 2 | 0.7415 | 553 | 87개 tensor |
| fixed_last4_cd100_rel100 | 2 | 0.7459 | 446 | 115개 tensor |

fixed_head는 검색 중 업데이트했지만 최종 선택이step0이어서 평가용 가중치가 원본과 동일하다. 이를 학습 개선으로 세지 않는다. 원본 비교 가중치는 Multi-HMR이며 제공 init_v6는 SAM3D Body 결과다. 근거는 실제 runs/configuration·complete와 `completed_native_refit_parameter_audits.json`, `completed_fixed_refit_parameter_audits.json`이다.

## 평가 기준

`ChallengeHOR/scripts/run_evaluation.py`가 사용하는 문서화된 local protocol과 동일한 공개 MHR human 지표 함수를 사용한다. 원본 frame0의 손 제외 FPS27 정점으로 proper Sim(3)을 한 번 구해 영상 전체에 고정한다. CD-H는 손 제외 FPS64 표면점의 양방향 최근접 유클리드 거리 평균의 합(cm), ACC-H는 body22 관절의 연속 원본 프레임 2차 차분 오차(cm/frame²)이며 fps²를 곱하지 않는다. 누락·미검출 프레임도 제거하지 않는다. 각 영상의 평균을 구한 뒤 영상별 동일 가중치로 합산한다.

주최 측 host 정점 역할과 quantized 모델 자산이 공개되지 않아, 참조 평가 스크립트와 같은 고정 local 역할을 쓴다. 공식 리더보드 수치와 완전히 같다고 주장하지 않는다. init_v6·init_v0·모든 모델에는 동일한 지표·역할·프레임 목록을 적용한다. 내부 SMPL-X CD proxy와 최종 MHR CD-H를 혼동하지 않는다.

## 추가 학습 전 Train-only 기록 분석

기존 search checkpoint의 전체 held-out Train4 결과다. Test나 Train20 refit 점수와 혼합하지 않는다.

| Train4 영상 | init_v6 CD cm | 원본 crop CD cm | surface-last4 CD cm |
|---|---:|---:|---:|
| black_platform-turn-05 | 14.0465 | 29.6962 | 22.8058 |
| cyan_water_bottle-rotate-01 | 20.3189 | 14.4659 | 17.6584 |
| traffic_cone-ground-table-02 | 11.4873 | 17.5431 | 14.2394 |
| wet_floor_sign-pick-place-ground-08 | 14.7827 | 15.5309 | 15.0537 |

| search step | MPJPE mm | root-aligned MPJPE mm | PA-MPJPE mm | 내부 surface CD proxy cm |
|---|---:|---:|---:|---:|
| 168 | 150.959 | 44.339 | 28.918 | 17.2314 |
| 355 | 134.994 | 42.823 | 27.610 | 18.4673 |
| 541 | 134.775 | 42.310 | 27.001 | 19.4434 |

내부 proxy는256표본 중 검출253표본의 별도 지표다. 관절 지표가 좋아지는 동안 proxy가 악화됐다. 이것은 특정 loss의 인과 효과를 분리한 증거는 아니다. 추가 GPU 학습은 시작하지 않았으며 추가 학습 시간 답변을 기다린다. `Train_only_post_delivery_failure_analysis.json`.

## 재부팅 후 미완료 평가·출력 복구

원래 실험 마감은 2026-10-26 08:24:13 UTC로 보존했다. 사용자의 Goal 재개 후 이미 학습된 후보의 미완료 평가·출력만 별도 3.0시간 실행 제한으로 복구한다. 새 학습은 시작하지 않았다. 완료 작업 제한은 추론·MHR 변환·보정까지 전달하며 budget.json과 Train 선택은 변경하지 않는다.

현재 UUID 기준 GPU 배치는 [1, 2, 3, 4, 5]이고 GPU0은 보존한다. 복구 관리자 PID 83020의 명령/start tick 일치 및 생존 확인: False. 별도 실행 제한 종료: 2026-10-26 11:44:10 UTC. `native_surface_completion_recovery.json`, `post_reboot_evaluation_recovery_GPU_verified.json`.

변환 프로세스의 native heap 종료 오류는 보존했다. 해당 blue_trash_can-drag-sideways-08의 실제 완료 파일/source 해시와 전체 858프레임을 CPU 복원으로 확인했으며, 저장 잔차와 재계산 잔차 최대 차이는 0.00003147mm다. 검증된 완료 결과는 보존하고 누락 작업만 재개한다. 재시도 시 완료 작업의 기존 종료 시각을 유지한다. `post_reboot_double_free_completed_output_verification.json`.

## 추가 학습 시간 답변 대기: 2026-10-26 10:14:45 UTC

기존28시간 실행과 복구 평가·30개 최종 출력·90표본 직접 검토를 마쳤다. 최종 감사9/10이며 init_v6 초과 성능만 미충족이다. 추가 학습 시간(지금부터3시간 또는6시간) 질문의 답변이 없어 새 GPU 학습은 시작하지 않았다. 기존 Train-only 기록 분석까지 완료했고, 다음 학습 실행은 시간 지정에 의존한다. Goal은 성공 완료가 아니다.

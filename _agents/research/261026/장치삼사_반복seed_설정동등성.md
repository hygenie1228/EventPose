# 반복 seed 설정 동등성 점검

주실험 seed261026과 반복 seed261027의 생성된 조건별 설정을 대조했다. 허용 차이는 실험 이름·seed·시작 시각뿐이며, 나머지 설정의 동일성을 assert로 확인했다. 학습 예산은 모두 3시간이다. 아직 생성되지 않은 반복 조건은 검증 완료로 취급하지 않는다.

근거: `exp/261026/device34_body_28h/repeat_configuration_parity_audit.json`. 이 검사는 설정 동등성만 증명하며 실제 반복 Test·원시 지표·checkpoint 검증을 대신하지 않는다.

## 최종 감사 연결

`tools/finalize_device34_body.py`의 최종 단계에서 여섯 조건의 주/반복 설정을 다시 읽어 seed·이름·시작 시각 외 차이가 없는지 확인한다. 두 seed는 각각 261026/261027, 학습 창은 모두 3시간이어야 한다. 12개 실험 각각의 실제 best 및 last checkpoint에 저장된 config 전체를 해당 실행 config 파일과 비교하고, 결과를 최종 감사의 `primary_repeat_configuration_parity`에 남긴다. 현재 완료된 RGB3의 best/last를 CPU로 직접 읽어 전체 config 일치가 통과함을 확인했으며, 나머지 checkpoint의 새 검사는 최종 감사에서 실행한다. Python 문법 검사는 통과했다.

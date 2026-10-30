# V4: 강한 RGB를 고정하고 이벤트 집계만 학습

작성: 2026-10-29 20:17 KST. 기존20시간과2026-10-30 09:55:12 KST 마감을 유지한다. 새 agent·새 중복 관리기·별도 임의 학습을 시작하지 않는다.

## Train 근거

20:07 직접 검토에서 V3 부위 모델은3,000업데이트 및 frozen holdout 진단을 마쳤다. 실제 motion path MAE6.54px, query별 horizon 표준편차45.2~60.2ms와 부위 간 시간분포 차이는 확인됐다. 그러나 intact PA24.791709mm가 zero24.757732·missing24.753235·shuffle24.728359mm보다 높았다. 시간 선택의 동적 성질과 자세 향상을 구분한다. V3 Test를 읽거나 후속 선택 근거로 사용하지 않았다.

V3의 입력·실제 부위지도·시간가중치를 같은 두 고정 holdout사례로 렌더하고 화면에서 실제 영문그림을 확인했다. motion relation 분석은 원래 immutable holdout NPZ에valid필드가 없어서 처음 중단됐고, 보존한 원본을 수정하는 대신 별도 validation motion supervision 파일의 마스크를 표본키로 연결하도록 수정해 통과했다. 추론 입력에는 어느 GT도 추가하지 않았다.

## 별도 동결 버전

- 모델 `src/multihmr/part_aggregation_v4.py`, SHA5488dc0454db12764b841fe62c9f0db81a895e590d703981860f025f86d7ced7.
- 강한RGB는Train-selected v2_rgb_only best, SHA bbcde2dac6b2e6133adc5beb7d31445a3b3905728c47626051f6bfd1906b6b31. V4가중치·동작모드 모두 고정한다.
- 공통집계초기화는Train-selected v3_part_dynamic best, SHA2d281915de25fcd802368f08cb7fb2b97a52ff9b9bcf3dba38764f9223e2a264. 네 event조건에 같은집계초기화를 쓰고 출력projection만0으로초기화한다.
- 고정, global, part, part-static 네 조건. motion aux0.5, support공통, gate/encoder학습률공통, mainwarmup100, 최대3,000성공업데이트·3시간상한. Pilot은각8업데이트·warmup2.
- RGB 기준은기존v2_rgb_only 전체Test를참고로재사용한다. 새V4RGB학습으로표시하지않으며, 실제5조건 비교의공통GT표본은조건이완료될때다시계산한다.
- 원래공통RGB1,651업데이트→강한RGB추가3,000업데이트 및집계V3추가3,000업데이트를사용한후속이다. 처음V1/V2/V3예산과같은단일학습으로표현하지않는다.

## 검증과 실행

20:15 CPU검증: 실제fullmodel구성과RGB115선택파라미터가checkpoint와같음, V3집계가중치초기화, RGB전파가중치requires_grad금지와train호출후eval유지, synthetic5조건초기identity/실제optimizer뒤base불변/adapter-only재로딩/정적horizon의query불변/누락eventidentity를확인했다. 단일담당본인의감사이며독립agent감사나실제CUDA성공은아니다.

공통factory는V4분기만추가했고V1/V2/V3dispatcher선택을CPU에서확인했다. 공통training은V4에기존motiondataset·gradient기록을사용하게하고freeze_RGB_base=True일때만configure_model을건너뛴다. V1/V2/V3모델소스및기존분기학습정책은보존했다. 새training SHA ad9e05545aaab39f222c3976b01da5a1bd984ad5a40b09fb1473c3f119a05049를V4설정과스냅샷에동결했다. 기존진행worker는로드된이전코드를계속쓴다.

기존관리기에V4단계와네pilot·실제adapter-onlycheckpoint검사를추가하는수정안을준비했다. CPU/Train-only승인은완료됐지만현재실행중인관리기는아직이수정안을로드하지않았다. 다음허용점검20:37:42KST이후기존worker를보존하며단독교체한다. V3전체Test→V4→V3ablations→V2ablations순서이며유휴시승인된V4pilot을먼저준비한다. 중복실행하지않는다.

소스동결계획에는준비당시controller스냅샷이들어있다. 이후V4스케줄·표시통합만한controller차이는 `v4_controller_integration.json`과실제관리기source SHA로별도기록하며모델/학습/데이터/지표소스의동결해시는일치한다. V4계획을덮어쓰지않는다.

검증한설계가개선을보장하지않는다. 실제본학습·전체Test·고정/global/static비교·이벤트개입·통계와마감보존종료가남아있다.

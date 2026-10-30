"""Create code-grounded Korean encoding/stereo explanations and real input figures."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job,write

def main():
    import numpy as np
    import h5py
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    from utils.event_encoding import EventFrames,ENCODINGS
    torch.set_num_threads(2)
    g=json.loads((ROOT/'_agents/research/261028/device3_encoding_goal_12h.json').read_text());register_job(g,os.getpid(),'encoding_and_stereo_documentation')
    out=Path(g['output_root'])/'encoding_explanation';out.mkdir(exist_ok=True)
    report1=ROOT/'_agents/report/261028_이벤트_세가지_인코딩_구현과_차이.md'
    report2=ROOT/'_agents/report/261028_스테레오_센서융합_모델_개요와_입출력.md'
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf')
    font=font_manager.FontProperties(fname='/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11})
    gallery=json.loads((ROOT/'exp/261028/device34_test_video_gallery/manifest.json').read_text());task=gallery['clips'][2]['task']
    session=task['parent_session'];frame=task['start_frame']+150;values={};metadata={}
    fig,axes=plt.subplots(6,6,figsize=(16,13),layout='constrained')
    for ax in axes.flat:ax.axis('off')
    labels={'count33':['양극성','음극성'],'bins100':['과거1 양','과거1 음','과거2 양','과거2 음','최근3 양','최근3 음'],'surface20':['양극성','음극성']}
    for device in (3,4):
        for j,e in enumerate(('count33','bins100','surface20')):
            with h5py.File(Path(g['output_root'])/'encoding_cache'/e/(session+'.h5'),'r',locking=False) as f:
                i=int(np.searchsorted(f['frames'][:],frame));assert int(f['frames'][i])==frame
                v=f['event'+str(device)][i].astype(float)/255;stamp=int(f['timestamp_ns'][i]);latest=int(f['event'+str(device)+'_latest_timestamp_ns'][i])
            assert latest<=stamp
            values[(device,e)]=v
            row=(device-3)*3+j
            for k,a in enumerate(v):
                ax=axes[row,k];ax.imshow(a,vmin=0,vmax=1,cmap='magma');ax.set_title(labels[e][k]);ax.axis('off')
                if k==0:ax.text(-.12,.5,f'device{device}\n{e}',transform=ax.transAxes,ha='right',va='center',fontweight='bold')
            metadata[f'event{device}_{e}']=dict(channels=len(v),nonzero=int(np.count_nonzero(v)),minimum=float(v.min()),maximum=float(v.max()),latest_timestamp_ns=latest)
    fig.suptitle('동일 시각의 실제 저장 입력 채널 — 0~1 공통 색척도, 표시 증폭 없음\n'+session+' | source frame '+str(frame)+' | timestamp_ns '+str(stamp),fontsize=14)
    fig.savefig(out/'실제_인코딩_채널_비교.png',dpi=150);plt.close(fig)
    # Exact synthetic example: one encoded pixel, five positive/negative observations.
    origin=1_700_000_000_000_000_000
    times=origin+np.array([-80,-45,-25,-15,-5],np.int64)*1_000_000
    sample=EventFrames(times,np.arange(6),np.zeros(5,dtype=np.int64),np.array([127,127,127,-128,127],np.int8),(1,1),size=1)
    example,_=sample.encode(origin,encodings=['count33','bins100','surface20'])
    example={e:v[:,0,0].tolist() for e,v in example.items()}
    if '--reuse-checkpoint-proof' in sys.argv:
        prior=json.loads((out/'설명_그림_검증자료.json').read_text());backbone=prior['backbone'];dim=prior['feature_dim'];ps=prior['patch_size']
    else:
        weights=torch.load(ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt',map_location='cpu')
        args=vars(weights['args']);patch=weights['model_state_dict']['backbone.encoder.patch_embed.proj.weight'];dim=int(patch.shape[0]);ps=int(patch.shape[-1]);backbone=args['backbone'];del weights
    assert dim==1024 and ps==14
    # Paper-style two-branch overview: primary row, auxiliary row, separate geometry.
    fig,ax=plt.subplots(figsize=(18,9));fig.patch.set_facecolor('white');ax.set(xlim=(0,1800),ylim=(1000,0));ax.axis('off')
    blue='#e8f1fc';orange='#fff0df';green='#e6f5ed';ink='#21364a'
    def box(x,y,w,h,title,detail,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=14',facecolor=color,edgecolor=ink,linewidth=1.4))
        ax.text(x+w/2,y+32,title,ha='center',va='center',fontsize=14,fontweight='bold',color=ink)
        ax.text(x+w/2,y+h/2+18,detail,ha='center',va='center',fontsize=11,color=ink,linespacing=1.55)
    def arrow(a,b,label=None,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color=ink,lw=1.6,linestyle='--' if dashed else '-'))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2-14,label,ha='center',va='bottom',fontsize=10,color=ink)
    ax.text(80,48,'현재 스테레오 센서융합 모델',fontsize=23,fontweight='bold',color=ink)
    ax.text(80,88,'대표 경로: device3 RGB + device4 event  |  Multi-HMR 원형 위에 보정 기반 event 잔차를 추가',fontsize=13,color=ink)
    box(60,155,220,150,'device3 RGB','과거 대응 RGB 프레임\nB × 3 × 896 × 896',blue)
    box(355,155,270,150,'주센서 encoder','DINOv2 ViT-L/14\n4096개 토큰 × 1024차원',blue)
    box(780,155,300,150,'보정 기반 융합','깊이 후보별 event 표본 추출\n주의 가중합 → 잔차 더하기',orange)
    box(1160,155,265,150,'검출 + HPH','사람 검출·중심 위치\n자세·체형·거리 회귀',blue)
    box(1495,155,245,150,'SMPL-X 출력','사람별 3D mesh / root\n55관절 → body22 평가',green)
    arrow((280,230),(355,230));arrow((625,230),(780,230),'주센서 특징 F')
    arrow((1080,230),(1160,230),'F + ΔF');arrow((1425,230),(1495,230))
    box(60,455,220,150,'device4 event','보정 timestamp의\n과거 희소 관측',orange)
    box(355,455,270,150,'인코딩 3종','count33 / bins100\nsurface20\nB × C × 224 × 224\nC = 2 / 6 / 2',orange)
    box(780,455,300,150,'보조 event encoder','별도 CNN · 입력 gain 4\nB × 128 × 28 × 28',orange)
    arrow((280,530),(355,530));arrow((625,530),(780,530))
    arrow((930,455),(930,305));ax.text(958,380,'16 × 16개의 ray\n각 ray의 11개 깊이 후보\n유효 투영만 사용',va='center',fontsize=11,color=ink)
    box(355,690,330,125,'센서 보정과 공통 시각','K주센서, Kevent, R, t\n공통 시각 이전 event만 사용',orange)
    ax.plot([685,725,725,825,825],[752,752,360,360,305],color=ink,lw=1.5,linestyle='--');arrow((825,335),(825,305),dashed=True)
    ax.text(1115,520,'기존 Multi-HMR 경로',fontsize=12,color='#245a8c')
    ax.text(1115,555,'이번 구현의 별도 event 경로·융합',fontsize=12,color='#ad6415')
    ax.text(1115,600,'GT box / pose / depth는 추론 입력 아님',fontsize=11,color=ink)
    box(60,865,1680,110,'같은 모듈을 공유하는 세 실험 조합','RGB3 + event4: 위 그림 그대로     |     event3 + event4: 주센서 입력을 event3 adapter의 3채널 표현으로 교체\nRGB3 + event3: 보조 센서만 event3로 교체     |     모든 예측은 공통 device3 RGB 카메라 좌표로 변환', '#f5f7fa')
    for ext in ('png','svg','pdf'):fig.savefig(out/('스테레오_모델_입출력_개요.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
    codepaths=['src/utils/event_encoding.py','src/multihmr/multisensor_data.py','src/multihmr/multisensor_repair.py','src/multihmr/multisensor_model.py','src/multihmr/multisensor_inference.py','src/multihmr/model.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in codepaths}
    receipt=dict(session=session,source_frame=frame,timestamp_ns=stamp,channels=metadata,numeric_example=example,backbone=backbone,feature_dim=dim,patch_size=ps,source_sha256=hashes,figures=['실제_인코딩_채널_비교.png','스테레오_모델_입출력_개요.png','스테레오_모델_입출력_개요.svg','스테레오_모델_입출력_개요.pdf'])
    write(out/'설명_그림_검증자료.json',receipt)
    report1.write_text(encoding_report(out,report2,example,receipt))
    report2.write_text(stereo_report(out,report1,receipt))
    write(out/'documentation_complete.json',dict(reports=[str(report1),str(report2)],source_sha256=hashes,real_inputs=True,scientific_experiment_goal_complete=False))
    print(json.dumps(dict(reports=[str(report1),str(report2)]),ensure_ascii=False),flush=True)

def encoding_report(out,other,example,receipt):
    return f'''# 이벤트 세 가지 인코딩의 구현과 차이

| 방법 | 출력 채널 | 사용하는 과거 관측 | 픽셀 값의 의미 |
|---|---:|---|---|
| count33 | 2 | 최근33ms | 양·음극성별 관측이 얼마나 쌓였는가 |
| bins100 | 6 | 최근100ms를3구간으로 분리 | 어느 시간 구간에 양·음 관측이 쌓였는가 |
| surface20 | 2 | 최근200ms에서 마지막 관측, 감쇠상수20ms | 양·음 관측이 얼마나 최근인가 |

**count33은 양, bins100은 시간대별 양, surface20은 최근성을 표현한다.** 세 방법 모두 같은 공통 시각 t에서 과거만 사용한다. surface20의20은 수집창20ms가 아니라 **감쇠 시간상수20ms**다. 실제 탐색 범위는200ms다.

## 실제 입력은 무엇인가

원본 입력은 각 센서의 `event.npz`에 저장된 희소 프레임이다. `timestamp_ns`는 저장 프레임의 보정 시각, `indptr`는 각 프레임의 희소 배열 범위, `indices`는 활성 픽셀 위치, `data`는 극성을 담는다. 구현은 `data > 0`을 양극성, 나머지를 음극성으로 분리한다. 원시 개별 event의 정밀 시각·발생 횟수를 복원한 데이터라고 주장하지 않는다. 한 저장 프레임 안의 관측은 그 프레임의 timestamp를 공유한다. `data`의 절댓값을 횟수로 더하지 않는다.

공간적으로는 센서 고유의 보정된 event 평면을 종횡비에 맞춰224×224 안에 배치한다. 원래 픽셀 중심을 축소 격자로 매핑하고 여백을0으로 채운다. 여러 원래 픽셀이 같은 출력 픽셀에 들어올 수 있다. 아래 count는 이 출력 격자에 누적한 **저장 활성픽셀 관측 수**다. RGB 영상 좌표에 event를 단순히 겹쳐 만들지 않는다.

## count33: 최근33ms의 극성별 관측량

시간창은 **(t−33ms, t]**다. 시작 경계는 제외하고 현재 시각은 포함한다. 출력은 `[양극성, 음극성]`, 크기는 `2×224×224`다.

출력 픽셀 u, 극성 p에 들어온 관측 수를 N(u,p)라 하면 정규화 값은 다음과 같다.

```text
count33(u,p) = min(log(1 + N(u,p)) / log(257), 1)
```

관측0개는0, 1개는약0.125, 2개는약0.198이며 256개 이상이면1이다. 큰 관측량의 범위를 로그로 압축한다. 같은33ms 안에서 일어난 순서는 합쳐진다. 따라서 이 표현만으로 먼저 관측된 위치와 나중에 관측된 위치를 구분할 수는 없다.

## bins100: 최근100ms를 시간 순서대로 나눈6채널

최근100ms를 오래된 구간부터 약33.33ms씩 나눈다. 각 구간에 count33과 같은 로그 관측량 계산을 **독립적으로** 적용한다. 시간창 길이는 약33.33ms이므로 count33의 정확한33ms와 완전히 같지는 않다.

```text
1번: (t−100ms,       t−66.666667ms] → 양, 음
2번: (t−66.666667ms, t−33.333334ms] → 양, 음
3번: (t−33.333334ms, t]            → 양, 음

채널 순서 = [1양, 1음, 2양, 2음, 3양, 3음]
```

실제 경계는 정수 ns로 `t−100000000 + floor(k×100000000/3)`을 계산한다. 정수 ns의 반올림 오차 때문에 미래 관측이 들어가지 않도록 처리한다. 서로 다른 구간의 관측은 다른 채널에 남지만, 한 구간 내부 순서는 합쳐진다. 출력은 `6×224×224`이며 모델도 실제6채널을 입력받는다.

## surface20: 마지막 관측으로부터 경과한 시간

**(t−200ms, t]** 범위에서 출력 픽셀·극성별 가장 최근 관측 시각 `t_last`를 찾는다. 출력은 `[양극성, 음극성]`, 크기는 `2×224×224`다.

```text
surface20(u,p) = exp(−(t − t_last(u,p)) / 20ms)
관측이 없으면 0
```

| 마지막 관측의 나이 | 정규화 값 |
|---:|---:|
| 0ms | 1.0000 |
| 5ms | 0.7788 |
| 20ms | 0.3679 |
| 40ms | 0.1353 |
| 100ms | 0.0067 |

같은 픽셀에 많이 관측됐어도 가장 최근 관측만 남는다. 여러 원래 픽셀이 같은224격자 픽셀에 들어오면 그중 가장 최근 관측을 사용한다. 200ms 이전 관측은 검색하지 않는다. 큰 절대 ns 시각은 정수로 먼저 빼고 상대시간을 float64로 변환해 감쇠값을 계산한다.

## 한 픽셀의 수치 예시

현재를0ms라고 하자. 같은 출력 픽셀에 양극성이−80, −45, −25, −5ms, 음극성이−15ms에 저장돼 있다고 가정한다. 이는 원시 event 횟수 예제가 아니라 저장 관측을 설명하기 위한 합성 예제다.

| 방법 | 계산 결과: 8bit 저장 전 |
|---|---|
| count33 | 양2개·음1개 → `{[round(v, 6) for v in example['count33']]}` |
| bins100 | 과거→최근 극성 쌍 → `{[round(v, 6) for v in example['bins100']]}` |
| surface20 | 양5ms·음15ms 경과 → `{[round(v, 6) for v in example['surface20']]}` |

count33은 최근 구간의 총량, bins100은 관측의 시간대 변화, surface20은 마지막 관측의 최근성을 담는 차이가 드러난다.

## 저장·학습 입력·색상 표시의 차이

인코딩 계산값은0~1 float다. 현재 캐시는 `round(255×값)`을 uint8로 저장하고 학습·평가에서255로 나눈다. HDF5 LZF 압축은 저장한 uint8 값을 보존하지만, float를 uint8로 바꾸는 단계에는 양자화가 있다. 극히 작은 surface 값은0으로 저장될 수 있다. 온라인 평가도 같은 양자화를 거쳐 캐시 입력과 맞춘다.

학습 증강은 기존 고정 규칙대로 입력에0.9~1.1의 무작위 gain을 주고0~1로 제한한다. 그 뒤 보조 event CNN의 첫 입력에는 세 방법 모두 고정 gain4가 적용된다. event-primary의 경우3채널 projection은 원래 값을 받고, 별도 학습 refiner 경로가 gain4 값을 받는다. 이 gain은 인코딩 공식의 일부나 uint8 저장값이 아니다. Test를 보고 인코딩별 gain을 맞추지 않았다.

시각화에서 빨강·파랑으로 합치거나 보기 좋게 증폭하는 것은 표시 방법이다. 학습 입력은 위의 극성별2/6/2채널이다. 아래 그림은 **증폭 없이0~1 공통 색척도**로 실제 저장 채널을 보여준다.

![같은 시각의 실제 세 인코딩 채널]({out}/실제_인코딩_채널_비교.png)

그림: `{receipt['session']}`, source frame `{receipt['source_frame']}`, 공통 시각 `{receipt['timestamp_ns']}ns`. 동일 시각의 device3·4 event를 각각 표시했다. bins100의 여섯 채널은 시간 순서와 극성 순서를 모두 유지한다.

## 인과성·결측·검증

공통 시각은 보정된 기준 RGB timestamp이며, event도 보정 timestamp를 사용한다. `frame_idx×50`으로 시간을 만들지 않는다. 현재보다 미래인 관측은 포함하지 않는다. 최신 저장 event 프레임이 없거나 그 나이가20ms를 넘으면 센서 unavailable로 표시한다. 이20ms 결측 판정은 surface20의 감쇠상수와 별도의 규칙이다. event-primary unavailable이면 해당 프레임에서 추론을 건너뛰고 미검출로 남기며, 보조 event unavailable이면 융합 잔차를0으로 마스킹한다.

현재 실험은78세션·두 센서의 실제 archive에서 각3개 시각을 골라 세 인코딩을 다시 계산해 캐시값의 정확한 일치를 확인했다. 신규 캐시 전체 SHA, 프레임·시각·채널 수도 검증했다. 이 표본 재계산을 모든 픽셀의 원시 개별 event 복원 검증으로 확대 해석하지 않는다. 성능 우열은 아직 진행 중인 동일 Test 비교와 통계 결과로 판단한다.

구현: [인코딩 계산]({ROOT}/src/utils/event_encoding.py), [캐시 생성]({ROOT}/tools/prepare_device3_encoding.py), [실제 캐시 검증]({ROOT}/exp/261028/device3_encoding_12h/actual_encoding_cache_audit.json), [스테레오 모델 개요]({other}), [현재 성능표]({ROOT}/_agents/report/261028_이벤트_인코딩_세가지_센서융합_비교.md).
'''

def stereo_report(out,other,receipt):
    return f'''# 현재 스테레오 센서융합 모델의 개요와 입출력

| 실험 조합 | 주센서: 검출·복원의 기본 경로 | 보조 센서 | event 채널 |
|---|---|---|---:|
| device3 RGB + device4 event | RGB3 → Multi-HMR | event4 → 별도 CNN | 2 / 6 / 2 |
| device3 event + device4 event | event3 → 학습 adapter → Multi-HMR | event4 → 별도 CNN | 2 / 6 / 2 |
| device3 RGB + device3 event | RGB3 → Multi-HMR | event3 → 별도 CNN | 2 / 6 / 2 |

현재 모델은 **주센서의 Multi-HMR 특징에 다른 event 센서의 특징을 보정된 광선 위치에서 읽어 더하는 구조**다. 두 영상에서 각각 인체를 복원해 삼각측량하는 방식이나 두 개의 동일 ViT를 사용하는 구조로 해석하면 안 된다. 같은 장치 조합도 RGB/event 센서의 내부·외부 보정을 따로 사용한다.

![현재 실제 구현의 스테레오 모델 개요]({out}/스테레오_모델_입출력_개요.png)

발표용 원본: [SVG]({out}/스테레오_모델_입출력_개요.svg), [PDF]({out}/스테레오_모델_입출력_개요.pdf). 파랑은 기존 Multi-HMR 경로, 주황은 이번 구현의 event 처리·융합 경로, 초록은 출력이다. 대표 그림은 사용자가 우선 요청한 RGB3+event4 경로이며, 나머지 두 조합의 변경점은 하단에 표시했다.

## 1. 입력과 시간 대응

주센서 RGB는 `B×3×896×896`, 각 event 인코딩은 `B×C×224×224`다. C는 count33=2, bins100=6, surface20=2다. 센서별 보정 intrinsic K, 두 센서 사이의 rotation R·translation t, auxiliary available mask를 함께 받는다. RGB는 기준 공통 시각 이전의 대응 프레임을 사용하고, event는 그 시각 이전의 희소 관측으로 인코딩한다.

RGB와 event의 픽셀·해상도·카메라 중심을 같다고 가정하지 않는다. event를 RGB 위에 픽셀 단위로 바로 합성해 입력하지 않는다. GT box·GT pose·GT depth는 추론 입력에 없다.

## 2. 주센서의 Multi-HMR 경로

원본 released `multiHMR_896_L.pt`의 실제 args와 patch projection 가중치를 확인했다. backbone은 `{receipt['backbone']}`, patch 크기는{receipt['patch_size']}이며 특징 차원은{receipt['feature_dim']}이다. 896÷14=64이므로 공간 토큰은64×64=4096개다.

RGB-primary에서는 정규화한 RGB를 기존 encoder에 넣는다. event-only에서는 실제 event2/6채널을 `1×1 projection + 학습 refiner`로3채널 표현으로 바꾸고896으로 확대해 같은 encoder에 넣는다. 이3채널은 event로부터 만든 표현이며 RGB 픽셀이 아니다. event-only 학습·검출·추론에서 RGB 픽셀을 읽지 않도록 분리했다.

## 3. 별도 event encoder

보조 event는 주센서 ViT와 별도인 작은 CNN으로 처리한다.

```text
C × 224 × 224
 → Conv5×5, stride2: 32 × 112 × 112
 → Conv3×3, stride2: 64 × 56 × 56
 → Conv3×3, stride2: 128 × 28 × 28
```

현재 `epipolar_repair_norm_v2`에서는 각 convolution 뒤에 GroupNorm과 GELU가 있다. 입력 gain은 고정4다. bins100은 첫 convolution부터6채널을 받는다. 채널 차이로 첫 층 파라미터 수가 달라지는 점과 실제 본 샘플·update 수는 학습시간 고정 비교와 구분한다.

## 4. 센서 보정에 기반한 특징 융합

주센서 특징을16×16 지점으로 pooling해 query를 만든다. 각 지점의 주센서 픽셀 q에서 `K_primary⁻¹ q`로 광선을 얻고, 카메라 Z 기준의 고정 깊이 후보11개를 놓는다.

```text
깊이 후보(m): 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8
X_primary = depth × ray / ray_z
X_aux     = R × X_primary + t
q_aux     = project(K_aux × X_aux)
```

이 후보는 **GT depth 입력이 아니다.** 보정만으로 가능한 대응 위치를 만들기 위한 가설이다. 각 위치에서 event CNN 특징을 bilinear sampling하고, query와의 내적으로11개 후보의 가중치를 계산한다. 카메라 뒤쪽·화면 밖·CNN 유효 범위 밖 후보는 마스킹한다. 세 convolution의 실제 feature 중심 간격8픽셀도 투영 좌표에 반영한다.

```text
가중치 = masked softmax(query · sampled_event / sqrt(128))
message = 깊이 후보별 event 특징의 가중합
ΔF = Linear(128→1024) → 64×64로 확대
융합 특징 = 원래 주센서 특징 F + available_mask × ΔF
```

잔차의 마지막 Linear는0으로 초기화한다. 따라서 학습 시작 시 RGB-primary의 기존 특징 경로가 그대로 유지되며, 학습을 통해 event 보조 정보를 더할 수 있게 한다. 이것은 원본 Multi-HMR 자체의 공식 stereo 모듈이 아니라 **이번 실험에서 추가한 보정 기반 fusion 구현**이다. 새로운 이론적 기여를 검증했다는 주장과 구분한다.

## 5. 검출·SMPL-X 복원·출력 좌표

융합은 backbone 출력 직후, 기존 사람 검출기보다 앞에서 적용된다. 이후 Multi-HMR의 검출·중심 offset과 HPH head가 사람별 자세·체형·거리 등을 회귀한다. SMPL-X layer가 사람별 mesh와3D 위치를 만든다. 추론은 검출 threshold0.3, NMS kernel3을 사용하며 사람이 없으면 빈 검출을 반환한다.

실제 native SMPL-X mesh는10,475정점이다. 고정 joint regressor로55관절을 얻고 주요 자세 지표는 body22관절로 계산한다. 주센서 카메라 좌표의 예측은 보정된 R·t를 이용해 공통 device3 RGB 카메라 좌표로 변환한다. 이후 평가 단계에서만 GT 매칭과 오차 계산을 한다. 미검출은0오차로 넣지 않는다.

## 6. 현재 학습과 평가 범위

원본 RGB3 1조건, 90분 RGB3 fine-tuning 1조건, 위 세 센서 조합×세 인코딩9조건으로 총11행이다. event조건은각3시간, 모든 새 학습은 같은 원본 checkpoint에서 시작한다. Train의 고정 subset6501프레임, 피험자 분리 holdout288프레임, batch16·seed261028·같은 증강을 사용한다. AdamW와 AMP, 유한 gradient 검사, 초반30분 head/adapter 우선 후 마지막8개 backbone block의 작은 LR 적응을 적용한다. checkpoint는 Train holdout으로만 선택한다.

현재 비교 Test 범위는 동결된 body18피험자·80개의10초 구간·24,080프레임이다. 동일 공통GT에서 첫 프레임 고정 pelvis/Sim3 정렬2지표, 매프레임 pelvis/Sim3 정렬2지표, 비정렬3D Root position error를 계산한다. RPE-Z는 별도 보조값이다. 전체11조건 완료 전의 표는 완료 조건들이 공유하는 GT로 재집계하므로 비교 집합이 늘면 숫자도 조금 달라진다.

RGB90분과 event조합3시간의 차이에는 학습량 효과가 포함된다. 현재 구조만으로 event 정보의 독립적인 개선 효과를 입증했다고 주장할 수 없다. 1seed 결과이며, 반복 공개 Test 평가의 한계도 있다. 현재 평가 숫자는 계속 갱신되는 [성능 보고서]({ROOT}/_agents/report/261028_이벤트_인코딩_세가지_센서융합_비교.md)에서 확인한다. 이 개요도 문서의 완성은 전체 실험 Goal의 완료를 의미하지 않는다.

인코딩 설명: [count33·bins100·surface20 구현과 실제 채널 그림]({other}). 구현 근거: [융합]({ROOT}/src/multihmr/multisensor_repair.py), [모델 연결]({ROOT}/src/multihmr/multisensor_model.py), [기존 검출·HPH·SMPL-X]({ROOT}/src/multihmr/model.py), [입력·추론·좌표변환]({ROOT}/src/multihmr/multisensor_inference.py). 소스 SHA와 원본 checkpoint 구조 확인값은 [그림 검증자료]({out}/설명_그림_검증자료.json)에 보존했다.
'''

if __name__=='__main__':main()

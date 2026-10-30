"""Paper-style proposed motion-conditioned causal Transformer diagram."""
from pathlib import Path
import sys
RELIABILITY="--reliability" in sys.argv
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/261028/dynamic_aggregation_diagram'
fp=subprocess.check_output(['fc-match','UnDotum','-f','%{file}'],text=True).strip();font_manager.fontManager.addfont(fp)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=fp).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
fig,ax=plt.subplots(figsize=(22,12));fig.subplots_adjust(left=.012,right=.99,bottom=.025,top=.98);ax.set(xlim=(0,28),ylim=(0,15));ax.axis('off')
blue='#d3e6f1';purple='#e4daf1';green='#d7e9dc';orange='#ffdfae'

def box(x,y,w,h,text,c=blue,fs=12):
 if RELIABILITY:
  mapping={
   'motion head\n임시 위치 / 영상 속도\n이전 예측 궤적과 결합\n추론 시 GT 입력 없음':'신뢰도 head\n시간·공간 문맥·관측 나이\n이 관측을 얼마나 쓸지 예측\nnoise 정답을 아는 것이 아님',
   '관절별 누적 이동량 D(T)\nT*: D(T) ≤ 허용 이동량\n\n다리: 작은 변화 → 긴 창\n손목: 큰 변화 → 짧은 창':'학습된 관측 가중치 r(i)\n0~1 신뢰도 score\n\n고정 count / 나이 제한 내\n유효한 과거에 더 큰 가중치',
   '운동량 조건부 cross-attention\nQ: 22 joint queries\nK,V: 시간별 token memory\n관절별 temporal gate g(j,b)\n+ 예측 부위 위치의 spatial bias':'신뢰도 조건부 cross-attention\nQ: RGB/이전 상태의 관절 queries\nK,V: 시간별 후보 token memory\nattention logit += log r(i)\n+ age bias / 읽지 않음 token',
   '학습에만 GT 사용: 3D 관절·속도·뼈 길이 손실 / motion head 지도\nattention logit += log g(j,b) · hard gate는 허용 밖 구간을 차단\n같은 Transformer에서 고정 gate와 motion gate를 비교해 집계 효과를 분리':'학습에만 GT 사용: 3D/속도 손실 + 관측 교란 대조 / 실제 noise 정답은 없음\n전부 신뢰도가 낮으면 읽지 않음 token 또는 이전 상태로 fallback\n동일 후보·용량에서 단순 N개 누적 vs 학습 가중치 vs 나이 제한을 비교',
   '중요한 제약\n\n시간 gate 전에 전체 시간을 섞으면\n짧은 창 query에도 오래된 정보가 유입됨\n\n따라서 temporal attention은\n아래 관절별 gate 뒤에서 수행':'후보 제한과 학습 집계는 별개\n\n패치별 최신 Nmax개 + 최대 이력 Tmax\n→ 계산량과 관측 나이의 상한\n\n그 안의 관측은 학습된 신뢰도로 가중\nN개를 채웠다는 이유로 전부 사용하지 않음'
  }
  text=mapping.get(text,text)
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.1',fc=c,ec='#526b7b',lw=1.4));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.45)

def arrow(a,b,ls='-',color='#486276',rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.5,color=color,ls=ls,connectionstyle=f'arc3,rad={rad}'))

def tokens(x,y,n=5,rows=4):
 for j in range(rows):
  for i in range(n):ax.add_patch(Rectangle((x+i*.25,y+j*.28),.2,.2,fc=[blue,'#c5dfed','#b4d2e7','#a5c9e0'][j],ec='#567b92',lw=.7))

ax.text(.2,14.45,'신뢰도 조건부 Transformer: N개를 채우는 대신, 유효한 관측을 골라 모은다' if RELIABILITY else '운동량 조건부 Transformer: 느린 부위는 긴 시간, 빠른 부위는 짧은 시간',fontsize=23,weight='bold')
ax.text(.2,13.85,'제안 구조도 · 특징맵/attention/출력은 도식이며 실제 학습 결과가 아님 · FPN top-down/lateral 결합 없음',fontsize=12,color='#59636e')
# A. Encoder is strictly per-bin spatial: no hidden old-time mixing before the joint gate.
ax.text(.25,13.15,'(a) 시간 구간을 섞지 않는 공간 token encoder',fontsize=17,weight='bold')
for k,label in enumerate(['B0: (t-10ms, t]','B1: (t-33ms, t-10ms]','B2: (t-100ms, t-33ms]','B3: (t-200ms, t-100ms]']):
 box(.35,11.7-k*.95,3.6,.7,label,'#f2dcb9',11)
ax.text(.35,7.8,'4구간 × 2극성 × 256 × 256\n서로 겹치지 않는 과거 관측',fontsize=11,linespacing=1.5)
box(4.95,9.6,3.0,2.0,'공유 patch embedding\nConv 8×8 / stride 8\n2 → 192채널\n+ 위치 / 관측 나이 embedding',blue,12)
for y in [12.05,11.1,10.15,9.2]:arrow((3.95,y),(4.95,10.6))
tokens(8.8,10.,5,4);ax.text(8.65,9.4,'각 구간: 1,024 tokens\n전체: 4 × 1,024 × 192',fontsize=10,linespacing=1.5)
arrow((7.95,10.6),(8.65,10.6))
box(11.,9.25,4.1,2.7,'공유 spatial Transformer × 4\n각 구간은 독립 처리\nwindow self-attention + FFN\n32×32 grid, 8×8 token window\n구간 간 temporal mixing 없음',blue,12)
arrow((10.15,10.6),(11,10.6))
box(16.2,9.55,4.0,2.1,'시간별 token memory\nZ0 / Z1 / Z2 / Z3\n각각 1,024 × 192\n각 token의 원본 시간 범위 유지',purple,12)
arrow((15.1,10.6),(16.2,10.6))
box(21.45,9.15,5.9,2.9,'중요한 제약\n\n시간 gate 전에 전체 시간을 섞으면\n짧은 창 query에도 오래된 정보가 유입됨\n\n따라서 temporal attention은\n아래 관절별 gate 뒤에서 수행',c='#fae2dd',fs=12)
ax.plot([.2,27.7],[7.3,7.3],c='#bdcbd4',lw=1.2)
ax.text(.25,6.8,'(b) 후보 관측의 신뢰도 추정 → 관절 query별 가중 temporal cross-attention' if RELIABILITY else '(b) 관절 query별 motion gate → masked temporal cross-attention',fontsize=17,weight='bold')
box(.4,4.2,3.65,1.8,'22 joint queries\n관절 ID embedding\n+ 이전 예측 상태\n+ 최근 Z0의 coarse pose',green,12)
box(.4,1.4,3.65,1.6,'motion head\n임시 위치 / 영상 속도\n이전 예측 궤적과 결합\n추론 시 GT 입력 없음',orange,12)
arrow((2.2,4.2),(2.2,3.0))
box(5.1,2.55,3.75,2.55,'관절별 누적 이동량 D(T)\nT*: D(T) ≤ 허용 이동량\n\n다리: 작은 변화 → 긴 창\n손목: 큰 변화 → 짧은 창',orange,12)
arrow((4.05,2.2),(5.1,3.3));arrow((4.05,5.1),(9.9,5.1),ls='--')
box(9.9,3.8,5.0,2.25,'운동량 조건부 cross-attention\nQ: 22 joint queries\nK,V: 시간별 token memory\n관절별 temporal gate g(j,b)\n+ 예측 부위 위치의 spatial bias',purple,12)
arrow((8.85,4.1),(9.9,4.6));arrow((18.2,9.55),(14.2,6.05),rad=.12)
ax.text(15.05,7.85,'Z0~Z3를 시간별로 유지한 채 전달',fontsize=10,color='#536779')
box(16.2,3.8,4.15,2.25,'joint self-attention + FFN\n22관절 관계 학습\n\ndecoder × 2\n출력: 22 × 192',green,12)
arrow((14.9,4.9),(16.2,4.9))
box(21.7,3.8,5.35,2.25,'shared MLP pose head\nbody22 절대 3D 위치\npelvis를 root로 사용\n\n출력: 22 × 3',green,13)
arrow((20.35,4.9),(21.7,4.9))
# Concrete mask example, explicitly not measured output.
if RELIABILITY:
 ax.text(9.95,3.2,'최종 attention 가중치의 가상 예시 — 실제 모델 출력 아님',fontsize=10,color='#687781')
 for r,(label,vals) in enumerate([('유효 관측',[.65,.20,.10,.05,.00]),('잡음 우세',[.03,.02,.02,.03,.90])]):
  ax.text(9.95,2.55-r*.52,label,fontsize=11,va='center')
  for c,val in enumerate(vals):
   ax.add_patch(Rectangle((12.+c*.6,2.36-r*.52),.48,.35,fc=purple if val>.1 else '#efefef',ec='#637589',lw=.7))
   ax.text(12.24+c*.6,2.535-r*.52,f'{val:.2f}',ha='center',va='center',fontsize=8)
 ax.text(14.64,2.98,'null',ha='center',fontsize=8)
 # Candidate-context branch into reliability head, with query conditioning kept.
 for aa,bb in [((16.2,9.65),(15.7,7.65)),((15.7,7.65),(.15,7.65)),((.15,7.65),(.15,2.2)),((.15,2.2),(.4,2.2))]:
  arrow(aa,bb,ls='--',color='#6c8594')
 ax.text(6.,7.8,'후보 token의 공간·시간 문맥 → 신뢰도 head',fontsize=10,color='#536779')
else:
 ax.text(9.95,3.2,'관절별 허용 구간 예시 — 설계 설명용, 실제 모델 출력 아님',fontsize=10,color='#687781')
 for r,label in enumerate(['손목: 짧은 창','다리: 긴 창']):
  ax.text(9.95,2.55-r*.52,label,fontsize=11,va='center')
  for c in range(4):
   on=(r==1 or c==0);ax.add_patch(Rectangle((12.0+c*.6,2.36-r*.52),.48,.35,fc=purple if on else '#efefef',ec='#637589',lw=.7))
   ax.text(12.24+c*.6,2.535-r*.52,'읽음' if on else '차단',ha='center',va='center',fontsize=8)
box(16.2,.7,10.85,1.35,'학습에만 GT 사용: 3D 관절·속도·뼈 길이 손실 / motion head 지도\nattention logit += log g(j,b) · hard gate는 허용 밖 구간을 차단\n같은 Transformer에서 고정 gate와 motion gate를 비교해 집계 효과를 분리',c='#f8e4de',fs=12)
ax.text(.4,.45,'모든 token을 계산하는 첫 구현은 자동 가속을 보장하지 않음 · 새 관측이 없으면 기억과 관측 부족을 구분',fontsize=11,color='#596571')
stem='신뢰도학습_이벤트집계_트랜스포머_제안' if RELIABILITY else '운동량조건부_트랜스포머_네트워크_제안'
for ext in ['png','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=155,facecolor='white')
plt.close(fig)

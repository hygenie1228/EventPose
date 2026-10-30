"""Publication-style proposed spatial FPN and per-joint temporal aggregation diagram."""
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Polygon,Circle
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/261028/dynamic_aggregation_diagram'
fp=subprocess.check_output(['fc-match','UnDotum','-f','%{file}'],text=True).strip();font_manager.fontManager.addfont(fp)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=fp).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
fig,ax=plt.subplots(figsize=(21,13));fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.02);ax.set(xlim=(0,25),ylim=(0,15.5));ax.axis('off')
colors={'input':'#f1d8b3','c':'#c8d5e8','p':'#b9dcf0','warp':'#d7e8db','temporal':'#e2d9ef','gain':'#ffe2aa'}

def box(x,y,w,h,s,c='#eef2f6',fs=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=.08',fc=c,ec='#5c6e7a',lw=1.25));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=fs,linespacing=1.45)

def arrow(a,b,c='#566877',ls='-',rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.4,color=c,ls=ls,connectionstyle=f'arc3,rad={rad}'))

def tensor(x,y,w,h,s,c):
 for k in [2,1,0]:ax.add_patch(Rectangle((x+.07*k,y+.07*k),w,h,fc=c,ec='#617583',lw=1))
 ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=11)

ax.text(.25,15.05,'제안 네트워크: 공간 FPN × 관절별 시간 피라미드 집계',fontsize=23,weight='bold')
ax.text(.25,14.5,'구조 설계도 · 아래 특징맵과 출력은 도식이며 실제 학습/추론 결과가 아님 · 배치 차원 생략',fontsize=12,color='#59636c')
ax.text(.25,13.85,'(a) 공간 스케일: 같은 encoder/FPN을 시간 조각마다 공유',fontsize=17,weight='bold')
# Three disjoint input observations.
for i,(lab,interval) in enumerate([('B0','(t-10ms, t]'),('B1','(t-33ms, t-10ms]'),('B2','(t-100ms, t-33ms]'),('B3','(t-200ms, t-100ms]')]):
 y=12.35-i*1.05;tensor(.4,y,3.1,.9,lab+'  '+interval,colors['input'])
ax.text(.4,8.35,'각 조각: 극성 2채널 × 256 × 256\n종횡비 유지 resize/pad\n서로 겹치지 않는 과거 구간',fontsize=11,linespacing=1.5)
box(4.2,10.5,2.3,1.55,'공유 ResNet-18\n입력층: 2채널\n4개 조각을 배치 처리',colors['c'],12)
for y in [12.8,11.75,10.7,9.65]:arrow((3.65,y),(4.2,11.3))
# Standard bottom-up C and top-down P pyramid.
ys=[8.5,9.8,11.1,12.4];names=['C2','C3','C4','C5'];shapes=['64×64×64','128×32×32','256×16×16','512×8×8']
for i,(y,n,sh) in enumerate(zip(ys,names,shapes)):
 w=2.5-.36*i;tensor(7.35+(2.5-w)/2,y,w,.65,n+'\n'+sh,colors['c'])
 if i<3:arrow((8.6,y+.8),(8.6,ys[i+1]))
arrow((6.5,11.1),(7.35,8.85),rad=.12)
ax.text(7.55,7.98,'bottom-up 공간',fontsize=11)
for i,y in enumerate(ys):
 w=2.5-.36*i;tensor(11.0+(2.5-w)/2,y,w,.65,f'P{i+2}\n128×{64//(2**i)}×{64//(2**i)}',colors['p'])
 arrow((10.0-.18*i,y+.33),(11.0+(2.5-w)/2,y+.33))
 if i>0:arrow((12.2,y),(12.2,ys[i-1]+.8))
ax.text(10.6,7.5,'1×1 lateral + upsample + 3×3\nP2: 작은 부위 / P5: 넓은 문맥\n신체 부위를 특정 레벨에 고정하지 않음',fontsize=11,linespacing=1.5)
box(15.0,9.65,4.0,2.25,'시간별 공간 특징\n{P2, P3, P4, P5} × 4조각\n\n공간 축: 해상도 4개\n시간 축: 관측 구간 4개',colors['p'],13)
arrow((13.7,10.8),(15.,10.8))
box(20.1,9.65,4.3,2.25,'현재 질의 시각 t까지의 관측만 사용\n\nRGB 사용 여부는 비교 조건에서 고정\n초기 버전: event + 이전 예측 상태\n초기 상태/track 초기화는 별도 명시',fs=12)
ax.plot([.2,24.7],[7.05,7.05],c='#bbc7d0',lw=1.2)
ax.text(.25,6.55,'(b) 시간 스케일: 정렬된 관측에서 관절마다 유효한 이력을 선택',fontsize=17,weight='bold')
# Evidence matrix: temporal x spatial, then alignment.
for r in range(4):
 for c in range(4):
  x=.4+c*.65;y=3.55+r*.47;ax.add_patch(Rectangle((x,y),.55,.36,fc=colors['p'],ec='#678ba3',lw=.8))
for c,s in enumerate(['B0','B1','B2','B3']):ax.text(.67+c*.65,5.6,s,ha='center',fontsize=11)
for r,s in enumerate(['P2','P3','P4','P5']):ax.text(3.1,3.7+r*.47,s,va='center',fontsize=10)
ax.text(.5,2.9,'(a)의 시간×공간 특징\n같은 관측을 다시 읽어 중복 count하지 않음',fontsize=10,linespacing=1.5)
box(4.0,3.5,3.35,1.9,'부위별 특징 정렬\n이전/현재 임시 예측 골격으로\n과거 관측 위치를 현재에 맞춤\n+ 정렬 신뢰도',colors['warp'],12)
arrow((3.45,4.45),(4.,4.45))
# Queries below, avoiding ground-truth leakage.
box(4.05,1.5,3.25,1.15,'이전 예측 + 최근 특징\n현재 임시 자세/query\nGT crop·GT pose 입력 없음',fs=11)
arrow((5.65,2.65),(5.65,3.5),ls='--')
# Nested history candidates.
for k,(y,t) in enumerate([(5.12,'A10 : B0'),(4.3,'A33 : B0 + 정렬 B1'),(3.48,'A100 : B0 + 정렬 B1,B2'),(2.66,'A200 : B0 + 정렬 B1,B2,B3')]):
 box(8.5,y,4.4,.65,t,colors['temporal'],12)

for y in [5.445,4.625,3.805,2.985]:arrow((7.35,4.45),(8.5,y))
ax.text(8.55,1.95,'각 후보: 22관절 × 128차원 특징\n공간 레벨 attention으로 joint pooling\n이 그림의 +는 특징 집계, 원시 이벤트 복제 아님',fontsize=10,linespacing=1.5)
box(14.,3.5,3.35,1.95,'부위별 motion gate\n입력: 예측 부위 궤적,\n관측 나이, 유효성\n누적 이동량 → 시간창',colors['gain'],12)
for y in [5.445,4.625,3.805,2.985]:arrow((12.9,y),(14.,4.45))
box(18.4,3.5,2.55,1.95,'작은 변화 → 긴 창\n큰 변화 → 짧은 창\nsoft / hard gate 비교\n→ 22 × 128',colors['temporal'],12)
arrow((17.35,4.45),(18.4,4.45))
box(22.,3.5,2.6,1.95,'관절 graph + MLP\nbody22 절대 3D 위치\npelvis를 root로 사용\n→ 22 × 3',colors['warp'],12)
arrow((20.95,4.45),(22.,4.45))
# Training only branch.
box(14.,.65,6.95,1.35,'학습 때만: GT 위치·속도로 motion head 지도\n부위별 누적 이동량 D(T) ≤ 허용 이동량 d\n3D/속도/뼈 길이 손실 · 후보별 오차는 검증에 사용',c='#f9dddd',fs=12)
arrow((15.65,2.),(15.65,3.5),c='#ac5555',ls='--')
ax.text(21.5,1.2,'목표: 부위마다 비슷한 이동량 범위 누적\n추론의 motion gate에 GT 입력 없음\n모든 후보를 계산하면 자동 가속되지 않음',fontsize=10,linespacing=1.6,color='#684c4c')
for ext in ['png','svg']:fig.savefig(OUT/f'공간시간_피라미드_네트워크_제안.{ext}',dpi=155,facecolor='white')
plt.close(fig)

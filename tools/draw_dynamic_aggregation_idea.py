"""Draw training/inference boundaries for a proposed event aggregation gate."""
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261028/dynamic_aggregation_diagram'
OUT.mkdir(parents=True,exist_ok=True)
font_path=subprocess.check_output(['fc-match','UnDotum','-f','%{file}'],text=True).strip()
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
fig,ax=plt.subplots(figsize=(17,10));fig.subplots_adjust(left=.02,right=.98,top=.96,bottom=.03)
ax.set(xlim=(0,17),ylim=(0,10));ax.axis('off')

def box(x,y,w,h,text,color='#e8f1fb',size=14):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=.12',facecolor=color,edgecolor='#57718a',linewidth=1.4))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.5)

def arrow(a,b,label=None,color='#395873'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,linewidth=1.8,color=color,connectionstyle='arc3'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=11,color=color)

ax.text(.2,9.7,'모호성을 안다고 가정하지 않는다 — 추가 관측의 이득을 학습해 예측한다',fontsize=22,weight='bold')
ax.text(.2,9.15,'① 학습: 같은 현재 시각의 정답으로, 과거 관측을 더 넣은 효과를 계산',fontsize=17,weight='bold',color='#245e89')
box(.3,6.7,2.4,1.65,'현재까지의 이벤트\n+ 이전 예측 상태')
box(3.4,7.55,3.1,1.0,'짧은 이력 → 자세 A\n예: 과거 33ms',size=13)
box(3.4,5.9,3.1,1.0,'긴 이력 → 자세 B\n예: 과거 100ms',size=13)
arrow((2.7,7.8),(3.4,8.05));arrow((2.7,7.15),(3.4,6.4))
ax.text(4.95,7.18,'동일한 pose head 사용',ha='center',fontsize=11,color='#51606b')
box(7.3,6.7,2.6,1.5,'각 관절의 GT 오차\neA와 eB 계산','#fff0d8')
arrow((6.5,8.05),(7.3,7.8));arrow((6.5,6.4),(7.3,7.0))
box(7.45,8.5,2.3,.5,'GT: 학습에만 사용','#ffe5e5',size=12)
arrow((8.6,8.5),(8.6,8.2),color='#a84848')
box(10.8,6.7,2.4,1.5,'이득의 지도값\nΔ = eA - eB','#e7f3e8')
arrow((9.9,7.45),(10.8,7.45))
box(14.0,6.7,2.55,1.5,'이득 예측기 학습\n예측 Δ가 지도값을\n맞히도록 학습',size=13)
arrow((13.2,7.45),(14,7.45))
ax.text(10.75,6.05,'Δ > 0: 과거 관측 추가가 도움\nΔ < 0: 오히려 자세 오차 증가',fontsize=12,linespacing=1.4)
ax.plot([.2,16.8],[5.45,5.45],color='#cbd4dc',lw=1.5)
ax.text(.2,5.05,'② 추론: GT 없이 관절별 이득을 예측하고, 사용할 이력을 선택',fontsize=17,weight='bold',color='#245e89')
box(.3,2.9,3.0,1.5,'현재까지의 이벤트 특징\n+ 이전 예측 상태\n+ 관측 나이 / 정렬 상태',size=13)
box(4.25,2.9,2.7,1.5,'학습된 이득 예측기\n관절별 예측 Δ 출력')
arrow((3.3,3.65),(4.25,3.65))
box(7.95,2.9,3.25,1.5,'후보 이력 비교·선택\n도움 예상 → 추가 사용\n도움 없음 → 짧은 이력 유지',size=13)
arrow((6.95,3.65),(7.95,3.65))
box(12.3,2.9,3.65,1.5,'선택된 특징으로 자세 복원\n손목과 몸통은 다른 이력 사용\n추론에서는 GT 접근 없음','#e7f3e8',size=13)
arrow((11.2,3.65),(12.3,3.65))
ax.text(.35,2.0,'중요: 예측 Δ는 실제 이득이 아니다. 새 피험자 검증에서 얼마나 맞는지 확인해야 한다.',fontsize=15,weight='bold',color='#a33b36')
ax.text(.35,1.25,'처음에는 여러 후보를 함께 비교한다. 33→100ms에서 손해여도 200ms에서 이득일 수 있으므로,\n한 번 손해라고 예측했다고 모든 더 긴 과거를 잘라내는 규칙은 보장되지 않는다.',fontsize=13,linespacing=1.5)
ax.text(.35,.4,'제안 개념도 · 아직 구현·성능 검증 전 · 시간창 값은 예시이며 실제 측정 결과가 아님',fontsize=12,color='#59666e')
for ext in ['png','svg']:fig.savefig(OUT/f'동적누적_학습과_추론.{ext}',dpi=150,facecolor='white')
plt.close(fig)

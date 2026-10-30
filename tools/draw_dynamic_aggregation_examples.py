"""Illustrative scenes for body-part event aggregation; not measured results."""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle,Rectangle

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/261028/dynamic_aggregation_diagram';OUT.mkdir(parents=True,exist_ok=True)
fp=subprocess.check_output(['fc-match','UnDotum','-f','%{file}'],text=True).strip();font_manager.fontManager.addfont(fp)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=fp).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path','font.size':12})
rng=np.random.default_rng(28);recent='#ee8b24';old='#2796b7';gray='#8c939a';green='#21774a'

def points_on(a,b,n,noise=.04):
 t=rng.uniform(0,1,n);return np.asarray(a)[None]*(1-t[:,None])+np.asarray(b)[None]*t[:,None]+rng.normal(0,noise,(n,2))
# Shared evidence across all comparison panels; synthetic drawing, not sensor output.
tr=np.concatenate([points_on((3.6,6.7),(3.5,3.7),5),points_on((5.2,6.7),(5.2,3.7),5),points_on((3.5,3.7),(5.2,3.7),3)])
to=np.concatenate([points_on((3.6,6.7),(3.5,3.7),65),points_on((5.2,6.7),(5.2,3.7),65),points_on((3.5,3.7),(5.2,3.7),24)])
wr=points_on((6.6,5.9),(8.25,6.65),68)
wo=[points_on((6.6,5.9),w,48) for w in [(8.24,6.88),(8.18,7.10),(8.09,7.29)]]
noise=rng.uniform([.8,2.4],[9.3,9.4],(60,2))
base=np.concatenate([points_on((3.6,6.7),(3.0,5.1),15),points_on((3,5.1),(2.4,4.6),12),points_on((5.2,6.7),(6.6,5.9),26),points_on((3.7,3.7),(3.2,1.5),15),points_on((4.9,3.7),(5.5,1.5),15)])
ang=rng.uniform(0,2*np.pi,20);head=np.c_[4.4+.68*np.cos(ang),8.05+.68*np.sin(ang)]

def setup(ax,title):
 ax.set(xlim=(0,10),ylim=(.6,10));ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=16,pad=14,weight='bold')

def scatter(ax,p,c,size=8,alpha=.9):ax.scatter(p[:,0],p[:,1],s=size,c=c,alpha=alpha,linewidths=0)

def highlight(ax,mode):
 scatter(ax,base,recent,8);scatter(ax,head,recent,8);scatter(ax,tr,recent,11)
 if mode in ['long','dynamic']:scatter(ax,to,old,7,.7)
 if mode=='long':
  for q in wo:scatter(ax,q,old,8,.7)
 if mode=='dynamic':
  # Small subset of retained past evidence illustrated after successful alignment.
  p=wr[:24].copy();p[:,0]+=.025;p[:,1]-=.025;scatter(ax,p,old,9,.6)
 scatter(ax,wr,recent,10)
 scatter(ax,noise if mode=='long' else noise[:8],gray,5,.35)

fig,axes=plt.subplots(1,4,figsize=(18,7.2));fig.subplots_adjust(left=.02,right=.98,top=.79,bottom=.23,wspace=.1)
fig.suptitle('같은 장면인데, 몸통과 팔에 필요한 관측 범위가 다르다',fontsize=23,weight='bold',y=.97)
fig.text(.5,.895,'설명용 가상 장면: 사람이 팔을 보통 속도로 움직이고, 몸통은 작은 움직임만 보이는 경우',ha='center',fontsize=14)
setup(axes[0],'장면: 현재 팔 + 과거 팔')
a=axes[0]
a.add_patch(Circle((4.4,8.05),.68,facecolor='#f3f5f6',edgecolor=gray,lw=2))
for p,q in [((3.6,6.7),(5.2,6.7)),((3.6,6.7),(3.5,3.7)),((5.2,6.7),(5.2,3.7)),((3.5,3.7),(5.2,3.7)),((3.6,6.7),(3,5.1)),((3,5.1),(2.4,4.6)),((5.2,6.7),(6.6,5.9)),((3.7,3.7),(3.2,1.5)),((4.9,3.7),(5.5,1.5))]:a.plot([p[0],q[0]],[p[1],q[1]],color=gray,lw=2.2)
for w in [(8.24,6.88),(8.18,7.10),(8.09,7.29)]:a.plot([6.6,w[0]],[5.9,w[1]],'--',c=old,lw=1.5,alpha=.55)
a.plot([6.6,8.25],[5.9,6.65],c=recent,lw=4)
a.text(.3,9.2,'주황: 현재',color=recent,fontsize=12);a.text(.3,8.65,'파랑: 과거 위치',color=old,fontsize=12)
setup(axes[1],'고정 짧은 누적: 33ms');highlight(axes[1],'short')
axes[1].add_patch(Rectangle((3.15,3.35),2.45,3.65,fill=False,edgecolor='#c24949',lw=2,ls='--'))
setup(axes[2],'고정 긴 누적: 200ms');highlight(axes[2],'long')
axes[2].add_patch(Rectangle((6.2,5.5),2.8,3.25,fill=False,edgecolor='#c24949',lw=2,ls='--'))
setup(axes[3],'부위별 동적 누적');highlight(axes[3],'dynamic')
axes[3].text(.6,9.1,'몸통: 긴 이력 사용',c=old,fontsize=12);axes[3].text(5.6,8.25,'팔: 최근 이력\n+ 정렬된 과거 일부',c=green,fontsize=11)
notes=['작은 몸통 움직임과\n팔의 움직임이 공존','팔 위치는 비교적 최근이지만\n몸통 관측이 부족할 수 있음','몸통 관측은 늘지만\n과거 팔 위치까지 섞일 수 있음','현재 부위에 맞는 관측만 유지\n필요한 이력은 부위마다 다름']
for ax,n in zip(axes,notes):ax.text(.5,-.045,n,transform=ax.transAxes,ha='center',va='top',fontsize=13,linespacing=1.4)
fig.text(.5,.065,'주황 = 최근 관측 · 파랑 = 과거 관측 · 회색 점 = 배경 예시 | 제안의 기대 동작을 그린 삽화이며 실제 이벤트·추정 결과가 아님',ha='center',fontsize=12,color='#515c66')
for ext in ['png','svg']:fig.savefig(OUT/f'부위별_동적누적_장면예시.{ext}',dpi=160,facecolor='white')
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(17,5.6),gridspec_kw={'width_ratios':[1,1,1.25]});fig.subplots_adjust(left=.05,right=.97,top=.72,bottom=.26,wspace=.35)
fig.suptitle('“몸통은 더 모으고, 팔은 덜 모은다”를 무엇으로 배우는가?',fontsize=22,weight='bold',y=.96)
fig.text(.5,.86,'학습 시의 가상 수치 예시 — 실제 측정값이 아니며, 고정 pose head의 관절별 GT 오차를 가정',ha='center',fontsize=13,color='#a33b36')
windows=['33ms','100ms','200ms'];values=[[18,29,40],[42,25,20]]
for ax,name,v in zip(axes[:2],['손목','몸통'],values):
 colors=[green if i==int(np.argmin(v)) else '#a8bac9' for i in range(3)]
 ax.bar(windows,v,color=colors,width=.6);ax.set_ylim(0,50);ax.set_ylabel('관절 오차 (mm)');ax.set_title(name,fontsize=17,weight='bold');ax.spines[['top','right']].set_visible(False)
 for i,x in enumerate(v):ax.text(i,x+1,str(x),ha='center',fontsize=13)
axes[0].text(.5,-.27,'100ms 추가의 이득 = 18 - 29 = -11mm\n이 예시에서는 최근 33ms 선택',transform=axes[0].transAxes,ha='center',fontsize=12,linespacing=1.5)
axes[1].text(.5,-.27,'100ms 추가의 이득 = 42 - 25 = +17mm\n이 예시에서는 더 긴 이력이 유리',transform=axes[1].transAxes,ha='center',fontsize=12,linespacing=1.5)
axes[2].axis('off');axes[2].text(.02,.98,'추론에서는 GT 오차를 모른다',fontsize=17,weight='bold',va='top')
axes[2].text(.02,.77,'학습에서 배운 선택기가\n관측 특징·관측 나이·정렬 상태로\n이득을 예측한다.\n\n예측이 실제 이득과 맞는지는\n새 피험자에서 검증해야 한다.\n\nGT로 최선의 창을 골라도\n이득이 없다면 이 아이디어를 중단한다.',fontsize=14,va='top',linespacing=1.6)
for ext in ['png','svg']:fig.savefig(OUT/f'부위별_선택이유_가상수치예시.{ext}',dpi=160,facecolor='white')
plt.close(fig)

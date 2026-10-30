"""Create Korean-labelled standalone plots from completed idea diagnostics."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import subprocess
font_path=subprocess.check_output(['fc-match','UnDotum','-f','%{file}'],text=True).strip()
font_manager.fontManager.addfont(font_path)
korean_font=font_manager.FontProperties(fname=font_path).get_name()
from idea_event_motion_probe import ROOT,OUT

plt.rcParams.update({'font.family':korean_font,'axes.unicode_minus':False,'font.size':11})
s=json.loads((ROOT/'exp/261026/idea_motion_native_probe/support_summary.json').read_text())
rows=[r for r in s['rows'] if r['device']==0]
fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for r in rows:
 label=r['session'].split('_')[3]+(' 상체' if r['session'].endswith('_f') else ' 전신')
 ax[0].plot(s['window_ms'],np.array(r['support_median'])*100,'o-',label=label)
 ax[1].plot(s['window_ms'],np.array(r['old_only_pixel_fraction_median'])*100,'o-',label=label)
ax[0].set(xlabel='과거 시간창 (ms)',ylabel='관절 주변 활성 픽셀 중앙값 (%)',title='긴 시간창의 관측량 증가');ax[0].legend(fontsize=8)
ax[1].set(xlabel='과거 시간창 (ms)',ylabel='최근 10ms 미관측 픽셀 비율 (%)',title='관측량과 최신성은 다르다')
labels=['상체 200','상체 201','전신 200','전신 201']
x=np.arange(4)
ax[2].bar(x-.18,[r['support_median'][1]*100 for r in rows],.36,label='관절 주변')
ax[2].bar(x+.18,[r['background_median'][1]*100 for r in rows],.36,label='배경 구석')
ax[2].set(xticks=x,xticklabels=labels,ylabel='33ms 활성 픽셀 중앙값 (%)',title='장면에 따라 관측 대비가 달라진다');ax[2].legend()
fig.savefig(OUT/'관측량과_최신성.png',dpi=160);plt.close(fig)
m=json.loads((OUT/'motion_linear_probe.json').read_text())
fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
keys=['hold','constant_velocity','history_ridge','history_event_ridge','history_shuffled_event_ridge']
labels=['이전 자세 유지','등속 외삽','과거 자세 회귀','과거 자세+이벤트','이벤트 시각 섞기']
values=[m['pooled'][k]['root_relative_mpjpe_mm'] for k in keys]
ax.bar(labels,values,color=['#888','#888','#3577b4','#e38f34','#888'])
for i,v in enumerate(values):ax.text(i,v+.4,f'{v:.2f}',ha='center')
ax.set(ylabel='골반 정렬 관절 오차 (mm)',title='GT 앵커를 제공한 1단계 진단: 이벤트 추가의 일관된 이득은 미확인',ylim=(0,46))
fig.savefig(OUT/'선형_모션_진단.png',dpi=160)

"""English, large real-data region comparisons for the paper-figure report."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_event_aggregation_paper_figure import ROOT,OUT,CASES,load_case,crop,WINDOWS
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':12})
data,meta=load_case(*CASES[0])
regions=json.loads((ROOT/'exp/261028/dynamic_aggregation_real_examples/실제예시_메타데이터.json').read_text())[0]['metrics'][0]['regions']
for i,(title,stem) in enumerate([('Moving arms: shorter history avoids mixing past positions','actual_arms_detail'),('Low-motion legs: longer history gathers more observations','actual_legs_detail')]):
 roi=regions[i]['box_xyxy'];fig,axes=plt.subplots(1,4,figsize=(16,5.1 if i==0 else 8.6))
 fig.subplots_adjust(left=.025,right=.99,top=.77 if i==0 else .83,bottom=.10,wspace=.05)
 fig.text(.025,.965,title,fontsize=19,weight='bold',va='top')
 fig.text(.025,.87 if i==0 else .90,f"Actual dataset | subject 200 | t = {meta['relative_seconds']:.3f} s | same native ROI {roi}",fontsize=11,color='#647180',va='top')
 for ax,ms in zip(axes,WINDOWS):
  ax.imshow(crop(data[f'event{ms}'],roi),interpolation='nearest');ax.set_title(f'{ms} ms history',fontsize=14,pad=8);ax.axis('off')
 fig.text(.025,.035,'Real observations only. Fixed polarity/color scale and weak 3x3 support across windows. No learned aggregation result.',fontsize=10,color='#647180')
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=220,facecolor='white')
 plt.close(fig)
print('ENGLISH DETAILS COMPLETE')

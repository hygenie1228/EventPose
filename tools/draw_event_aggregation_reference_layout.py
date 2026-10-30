"""User-reference-inspired four-panel figure with actual dataset inputs and GT mesh.

Only the layout grammar is borrowed; no reference-image pixels are reused.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon,Rectangle
from build_event_aggregation_paper_figure import ROOT,OUT,CASES,load_case,crop,display

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.unicode_minus':False})
(data,meta),(weak,wm)=[load_case(*CASES[k]) for k in [0,3]]
mesh=plt.imread(OUT/'gt_mesh_reference.png')
fig=plt.figure(figsize=(23.0,9.2));ax=fig.add_axes([.008,.045,.984,.92]);ax.set(xlim=(0,31),ylim=(0,12));ax.axis('off')
ink='#13212d';blue='#278bfa';colors=['#a46bde','#72bc60','#f5aa47','#338ffe']

def text(x,y,s,fs=12,weight='normal',ha='center',color=ink):
 ax.text(x,y,s,fontsize=fs,weight=weight,ha=ha,va='center',color=color,linespacing=1.22,zorder=8)

def panel(x,w,title,c):
 ax.add_patch(FancyBboxPatch((x,.65),w,11.0,boxstyle='round,pad=.05,rounding_size=.13',fc=c,ec='none',zorder=0))
 text(x+w/2,11.12,title,16,'bold')

def box(x,y,w,h,s,c='#fff5d8',fs=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.12',fc=c,ec='#6a7781',lw=1.0,zorder=3));text(x+w/2,y+h/2,s,fs,'bold')

def arrow(a,b,c='#233645',style='-'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.35,color=c,ls=style,zorder=5))

def photo(im,x,y,w,h,z=4):
 ih,iw=im.shape[:2];scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
 ax.imshow(im,extent=(x+(w-ww)/2,x+(w+ww)/2,y+(h-hh)/2,y+(h+hh)/2),interpolation='nearest',zorder=z)

def token_grid(x,y,c,cols=4,rows=2,size=.26):
 for r in range(rows):
  for k in range(cols):ax.add_patch(Rectangle((x+k*(size+.05),y+r*(size+.05)),size,size,fc=c,ec=c,lw=.8,alpha=.65,zorder=4))

def encoder(x,y,w,h,label,c):
 ax.add_patch(Polygon([[x,y],[x+w,y+.28],[x+w,y+h-.28],[x,y+h]],closed=True,fc=c,ec='#5385aa',lw=1.1,zorder=3));text(x+w/2,y+h/2,label,12,'bold')

panel(3.2,9.65,'1. Tokenization & Encoding','#f0f5f9')
panel(13.05,7.25,'2. Learn What to Accumulate','#eff8f1')
panel(20.5,6.15,'3. Joint-wise Cross-Attention','#edf6fd')
panel(26.85,3.85,'4. Pose Output','#f5effa')
# Actual input RGB and event history, from the same query time.
text(1.45,10.8,'RGB at time t',14,'bold');photo(crop(data['rgb'],meta['rgb_display_crop']),.12,7.55,2.65,2.85)
text(1.45,6.9,'Past event observations',12,'bold');text(1.45,6.5,'(t - 200 ms, t]',11)
# Each card is a real observed history image, not a generated event frame.
for k,ms in enumerate([200,100,33,10]):
 x=.28+.18*k;y=2.5+.26*k
 ax.add_patch(Rectangle((x-.04,y-.04),1.85,2.8,fc='black',ec='#8c969d',lw=.6,zorder=3.+k*.2))
 photo(crop(data[f'event{ms}'],meta['event_display_crop']),x,y,1.85,2.8,z=3.1+k*.2)
text(1.45,1.86,'Actual dataset samples',10,color='#586775')
text(1.45,1.5,'Subject 200',10,color='#586775')
encoder(3.65,7.8,1.55,2.0,'RGB ViT\nEncoder','#d8ebff');arrow((2.8,8.85),(3.65,8.85))
for x in [6.15,8.4]:token_grid(x,8.28,blue,4,3,.3)
text(7.85,9.85,'RGB tokens',13,'bold');text(7.85,9.4,'N_r x D',11);text(8.0,8.75,'...',17)
arrow((5.2,8.8),(6.0,8.8));text(10.5,9.25,'Query features',10)
# This line supplies queries to stage 3; it is not an event warp.
arrow((10.15,8.8),(14.05,9.62),c='#325c8e')
encoder(3.55,2.1,1.4,2.2,'Event\nPatch\nEncoder','#f9d9d6');arrow((2.72,3.85),(3.55,3.85))
text(8.9,7.68,'Past-only event bins',13,'bold')
# Non-overlapping bin counts derived from the cached raw CSR counts.
windows=[(100,200),(33,100),(10,33),(0,10)]
for k,(left,right) in enumerate(windows):
 x=5.25+k*1.85;label=3-k
 p=data[f'p{right}']-(data[f'p{left}'] if left else 0);n=data[f'n{right}']-(data[f'n{left}'] if left else 0)
 assert p.min()>=0 and n.min()>=0
 im,_,_=display(p,n);photo(crop(im,meta['event_display_crop']),x,5.05,1.5,1.65)
 text(x+.75,7.24,f'B{label}',12,'bold');text(x+.75,6.9,f'{left}-{right} ms',10)
 arrow((x+.75,5.02),(x+.75,4.45))
 box(x,3.53,1.5,.88,'Spatial\nTransformer',c='#e1e3e8',fs=9)
 text(x+.75,3.25,'shared',9,color='#65727d');arrow((x+.75,3.0),(x+.75,2.6))
 token_grid(x+.16,1.78,colors[k],4,2,.25);text(x+.75,1.43,f'Z{label}',11,'bold')
# The encoder fans out into temporal bins; macro link kept simple.
arrow((4.95,3.1),(5.22,3.95))
text(8.95,.96,'4 x 1,024 event tokens x D',11)
# Stage 2: query identity and event reliability, with real weak-response evidence.
text(16.6,10.25,'RGB-conditioned joint queries',12)
for k in range(6):ax.add_patch(Rectangle((14.18+k*.68,9.42),.4,.4,fc='#70b1fc',ec='#2875c0',lw=.7,zorder=4))
text(16.65,9.03,'22 joints',10,color='#687c70')
box(14.02,7.18,5.2,1.0,'Reliability Head',c='#dbeecf',fs=15)
arrow((16.65,9.3),(16.65,8.2))
text(16.65,6.65,'Coherence  |  Recency  |  Query relevance',9)
# Candidate event features enter reliability head: branch from encoder memory.
ax.plot([5.3,5.3,12.65,12.65],[1.48,1.24,1.24,1.48],c='#6a7781',lw=.8)
arrow((12.65,1.75),(13.55,1.75));arrow((13.55,1.75),(13.55,7.68));arrow((13.55,7.68),(14.02,7.68))
# Joint/patch group scores are schematic, not a trained attention map.
text(16.65,5.95,'Evidence weights r(j, b)',12,'bold')
labels=['Arms','Legs','Weak evidence'];score=np.array([[.12,.2,.55,.9],[.8,.78,.8,.72],[.08,.07,.06,.08]])
for r,lab in enumerate(labels):
 text(14.88,5.25-r*.55,lab,9,ha='right')
 for k in range(4):
  val=score[r,k];c=plt.cm.Blues(.12+.65*val)
  ax.add_patch(Rectangle((15.1+k*.8,5.-r*.55),.74,.48,fc=c,ec='white',lw=.6,zorder=3))
for k,lab in enumerate(['B3','B2','B1','B0']):text(15.47+k*.8,3.62,lab,9)
text(16.65,3.17,'Schematic weights, not measured',9,color='#6d7b73')
arrow((16.65,7.18),(16.65,6.22))
box(14.1,1.8,5.1,.85,'Event memory + scores',c='#dbeecf',fs=13)
arrow((16.65,3.0),(16.65,2.7))
text(16.65,1.2,'Unreliable evidence -> null / memory',9)
# Queries and reliability scores are routed separately into cross-attention.
arrow((18.2,9.62),(20.78,9.62),c='#325c8e');arrow((20.78,9.62),(20.78,7.55),c='#325c8e');arrow((20.78,7.55),(21.05,7.55),c='#325c8e');text(20.6,9.95,'Q',11,'bold',color='#325c8e')
# Stage 3: compact weighted cross-attention, no fictitious learned heatmaps.
for k,c in enumerate(colors):
 x=21.0+k*1.32;token_grid(x,8.65,c,3,2,.25);text(x+.42,9.6,f'Z{3-k}',11,'bold')
arrow((19.2,2.22),(20.7,2.22));arrow((20.7,2.22),(20.7,7.0));arrow((20.7,7.0),(21.0,7.0))
box(21.05,5.65,5.0,2.15,'Reliability-weighted\nCross-Attention',c='#fff0c9',fs=16)
for x in [21.42,22.74,24.06,25.38]:arrow((x,8.6),(x,7.85))
text(23.55,6.05,'Q K^T / sqrt(D) + log r(j,b)',10)
box(21.45,3.4,4.2,1.0,'Joint Decoder\n2 Transformer blocks',c='#f5ddd9',fs=13)
arrow((23.55,5.65),(23.55,4.45));box(21.45,1.67,4.2,.9,'3D Pose Head',c='#e0eee5',fs=13);arrow((23.55,3.4),(23.55,2.62))
# Output format uses a real GT mesh, explicitly labelled as reference.
text(28.78,10.25,'3D Human Pose',14,'bold');text(28.78,9.8,'22 joints',11)
photo(mesh,27.0,2.0,3.55,7.4)
arrow((25.65,2.15),(27.0,2.15))
text(28.78,1.65,'Dataset GT reference',10,'bold');text(28.78,1.2,'not a model prediction',9,color='#71677d')
# Global disclosure belongs in the caption; keep the figure itself short.
text(.15,.22,'Real RGB/event inputs and dataset GT mesh. Encoders, tokens and learned weights are schematic; aggregation is not trained yet.',10,ha='left',color='#5c6b78')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'paper_pipeline_reference_style.{ext}',dpi=240,facecolor='white')
plt.close(fig)
(OUT/'reference_layout_manifest.json').write_text(json.dumps(dict(user_reference='https://drive.google.com/file/d/1n8DTleXd0a1IjOEKSz4z5IL013sQB62-/view',reference_used_for='Four-panel layout and visual hierarchy only; all scene imagery from dataset',rgb_event_case=meta['session'],reference_pixels_reused=False,gt_mesh_actual=True,gt_mesh_is_prediction=False,weight_matrix_is_schematic=True,visible_text_language='English'),indent=2))
print('REFERENCE STYLE FIGURE COMPLETE')

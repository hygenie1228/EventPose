"""Concise English paper figures with actual, traceable RGB/CSR event samples.

No learned aggregation maps or predictions are fabricated. All outputs are design.
"""
from pathlib import Path
import json,datetime,hashlib
import cv2
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from visualize_rgb_event_alignment import prefix_array

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261029/dynamic_aggregation_paper_figure';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.unicode_minus':False})
cv2.setNumThreads(1)
WINDOWS=[10,33,100,200]
CASES=[('20260720_131207_863008_200_b',12.,'A'),('20260720_134353_764469_201_b',12.,'B'),('20260720_141545_676840_202_b',12.,'C'),('20260720_125338_040324_200_f',5.,'D')]


def stat(p):
 v=p.stat();return [v.st_size,v.st_mtime_ns]


def display(p,n):
 active=(p+n)>0
 support=cv2.boxFilter(active.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)>=2
 a=np.sqrt(np.minimum(np.log1p(p)/np.log(9),1))*support
 b=np.sqrt(np.minimum(np.log1p(n)/np.log(9),1))*support
 image=np.clip(np.stack([a,.25*a+.65*b,b],axis=-1),0,1)
 return image,int(np.count_nonzero(active)),int(np.count_nonzero(active&support))


def clipbox(box,w,h):
 b=np.array(box,dtype=int);b[0::2]=np.clip(b[0::2],0,w);b[1::2]=np.clip(b[1::2],0,h)
 assert b[2]>b[0] and b[3]>b[1];return b.tolist()


def crop(a,box):
 x0,y0,x1,y1=box;return a[y0:y1,x0:x1]


def load_case(session,seconds,case):
 src=ROOT/'dataset'/session/'device0';sources={n:stat(src/n) for n in ['rgb.mp4','event.npz','label.h5']}
 existing=OUT/f'{case}_observations.npz'
 print('BEGIN',case,session,flush=True)
 if existing.exists():
  with np.load(existing) as z:
   arrays={k:z[k] for k in z.files}
  previous=json.loads((OUT/f'{case}_source.json').read_text());assert previous['source_stat']==sources
  print('REUSE',case,flush=True);return arrays,previous
 with np.load(src/'event.npz',allow_pickle=False) as z:
  et=z['timestamp_ns'].astype(np.int64);rt=z['rgb__timestamp_ns'].astype(np.int64);ptr=z['indptr'];h,w=map(int,z['image_shape'])
 target=int(rt[0])+round(seconds*1e9);rgb_id=max(0,int(np.searchsorted(rt,target,side='right'))-1);end=int(rt[rgb_id])
 cached=ROOT/'exp/261028/dynamic_aggregation_real_examples'/('전신_누적수치.npz' if case=='A' else '상체_누적수치.npz')
 arrays={};from_previous=False
 if case in ['A','D'] and cached.is_file():
  with np.load(cached) as z:
   assert int(z['timestamp_ns'])==end
   for ms in WINDOWS:
    arrays[f'p{ms}']=z[f'p{ms}'];arrays[f'n{ms}']=z[f'n{ms}']
  from_previous=True
 else:
  lo,hi=np.searchsorted(et,[end-200_000_000,end],side='right');a,b=map(int,ptr[[lo,hi]])
  indices=prefix_array(src/'event.npz','indices',b)[a:b];polarity=prefix_array(src/'event.npz','data',b)[a:b]>0
  for ms in WINDOWS:
   row=int(np.searchsorted(et,end-ms*1_000_000,side='right'));start=int(ptr[row])-a
   idx=indices[start:];pos=polarity[start:]
   arrays[f'p{ms}']=np.bincount(idx[pos],minlength=h*w).reshape(h,w)
   arrays[f'n{ms}']=np.bincount(idx[~pos],minlength=h*w).reshape(h,w)
 cap=cv2.VideoCapture(str(src/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,rgb_id);ok,bgr=cap.read();cap.release();assert ok
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);arrays['rgb']=rgb;arrays['timestamp_ns']=np.array(end)
 with h5py.File(src/'label.h5','r',locking=False) as f:
  gt=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64);row=min(max(0,int(np.searchsorted(gt,end,side='right'))-1),len(gt)-1)
  box=f['bboxes/xyxy'][row,0,0].astype(float)
  sh=[int(f['bboxes'].attrs['height']),int(f['bboxes'].attrs['width'])]
 box*=np.array([rgb.shape[1]/sh[1],rgb.shape[0]/sh[0],rgb.shape[1]/sh[1],rgb.shape[0]/sh[0]])
 box+=np.array([-40,-40,40,40]);rgbbox=clipbox(box,rgb.shape[1],rgb.shape[0])
 cache=ROOT/'exp/261026/idea_motion_native_probe'/f'{session}_device0.npz'
 if not cache.exists():cache=ROOT/'exp/261026/idea_motion_probe'/f'{session}_device0.npz'
 with np.load(cache) as z:
  k=int(np.argmin(np.abs(z['times']-end)));uv=z['uv'][k];tgt=int(z['times'][k])
 if case=='D':eventbox=clipbox(np.r_[uv[15]-[165,140],uv[15]+[165,190]],w,h)
 else:
  points=uv[np.isfinite(uv).all(1)];eventbox=clipbox(np.r_[points.min(0)-[60,50],points.max(0)+[60,50]],w,h)
 msmeta={}
 for ms in WINDOWS:
  im,before,after=display(arrays[f'p{ms}'],arrays[f'n{ms}']);arrays[f'event{ms}']=im.astype(np.float32)
  msmeta[str(ms)]=dict(raw_active_pixels=before,displayed_supported_pixels=after)
 metadata=dict(case=case,session=session,device=0,rgb_frame=rgb_id,timestamp_ns=end,relative_seconds=(end-int(rt[0]))/1e9,
  source_stat=sources,source=str(src),rgb_display_crop=rgbbox,event_display_crop=eventbox,
  crop_provenance='GT bbox / projected joints for figure readability only; not network inputs',gt_crop_timestamp_ns=tgt,
  rgb_event_spatial_overlay=False,causal=True,windows_ms=WINDOWS,window_rule='(t-W,t]',
  last_event_ns=int(et[np.searchsorted(et,end,side='right')-1]),derived_count_cache_reused=from_previous,
  display='Black background; positive orange/red, negative cyan; fixed sqrt(min(log1p(N)/log(9),1)); same weak 3x3 support >=2 occupied pixels for ALL windows; visualization only.',metrics=msmeta)
 assert metadata['last_event_ns']<=end
 assert sources=={n:stat(src/n) for n in sources}
 np.savez_compressed(existing,**arrays);(OUT/f'{case}_source.json').write_text(json.dumps(metadata,indent=2))
 print('DONE',case,flush=True);return arrays,metadata


def gallery(cases):
 # Four subjects/scenes rather than repeatedly showing one hand-picked crop.
 fig,axes=plt.subplots(4,5,figsize=(13.8,14.5),squeeze=False)
 fig.subplots_adjust(left=.025,right=.985,top=.93,bottom=.035,wspace=.035,hspace=.24)
 fig.text(.025,.975,'Actual observations: one query time, different history lengths',fontsize=19,weight='bold',va='top')
 fig.text(.025,.95,'Real RGB and event samples. Fixed display scale across all event windows; no learned aggregation output.',fontsize=10,color='#56616c')
 for i,(data,meta) in enumerate(cases):
  items=[crop(data['rgb'],meta['rgb_display_crop'])]+[crop(data[f'event{ms}'],meta['event_display_crop']) for ms in WINDOWS]
  labels=['RGB query']+[f'{ms} ms history' for ms in WINDOWS]
  for j,(ax,im,lab) in enumerate(zip(axes[i],items,labels)):
   ax.imshow(im,interpolation='nearest');ax.set_title(lab,fontsize=11,pad=5);ax.axis('off')
  axes[i,0].text(0,-.045,f"{meta['case']} | subject {meta['session'].split('_')[3]} | {meta['relative_seconds']:.3f} s",transform=axes[i,0].transAxes,fontsize=9,va='top')
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'actual_dataset_gallery.{ext}',dpi=220,facecolor='white')
 plt.close(fig)


def pipeline(cases):
 data,meta=cases[0];weak,weakmeta=cases[-1]
 fig=plt.figure(figsize=(17.8,8.2));ax=fig.add_axes([.015,.04,.97,.92]);ax.set(xlim=(0,18),ylim=(0,8.4));ax.axis('off')
 ink='#263747';blue='#e1edf6';orange='#fbe7ca';green='#e2efea';purple='#eae4f2'
 def txt(x,y,text,size=12,weight='normal',color=ink,ha='left'):
  ax.text(x,y,text,fontsize=size,weight=weight,color=color,ha=ha,va='center',linespacing=1.25)
 def box(x,y,w,h,text,color,size=12):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.07',fc=color,ec='#6a7d8c',lw=1.1));txt(x+w/2,y+h/2,text,size,'bold',ha='center')
 def arrow(a,b,color='#596e80',style='-'):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.4,color=color,ls=style,connectionstyle='arc3'))
 def photo(im,x,y,w,h,label=None):
  # Letterbox to preserve scene geometry.
  ih,iw=im.shape[:2];ratio=min(w/iw,h/ih);ww=iw*ratio;hh=ih*ratio
  ax.imshow(im,extent=(x+(w-ww)/2,x+(w+ww)/2,y+(h-hh)/2,y+(h+hh)/2),interpolation='nearest',zorder=3)
  if label:txt(x+w/2,y-.17,label,10,ha='center')
 txt(.1,8.12,'Learning What to Accumulate',22,'bold')
 txt(.1,7.73,'RGB-guided, reliability-aware aggregation of past event observations',12,color='#667783')
 # Main flow: actual inputs, short labels, one visually dominant learned module.
 rgb=crop(data['rgb'],meta['rgb_display_crop']);photo(rgb,.15,4.25,1.6,2.8,'RGB at query time t')
 box(2.05,5.05,1.65,1.0,'RGB\nencoder',blue,12);arrow((1.8,5.55),(2.05,5.55))
 # Real event crop thumbnails are observed inputs, never simulated weighted output.
 for k,ms in enumerate([33,100,200]):
  x=.1+k*.76;photo(crop(data[f'event{ms}'],meta['event_display_crop']),x,1.9,.7,1.65,f'{ms} ms')
 txt(1.2,1.43,'Past event observations',10,ha='center');txt(1.2,1.14,'t_i <= t; bounded history',9,color='#6a7580',ha='center')
 box(2.5,2.15,1.7,1.0,'Event\ntokens',blue,12);arrow((2.25,2.65),(2.5,2.65))
 # Core mechanism.
 ax.add_patch(FancyBboxPatch((4.9,1.65),7.5,5.4,boxstyle='round,pad=.09,rounding_size=.15',fc='#fbfaf7',ec='#b69054',lw=1.7))
 txt(8.65,6.68,'Reliability-aware aggregation',16,'bold',ha='center')
 txt(8.65,6.25,'Learn which observations to use, not just how many to collect.',10,color='#6b644f',ha='center')
 box(5.35,4.9,2.1,.78,'Joint queries Q',blue,12);arrow((3.7,5.55),(5.35,5.28))
 box(5.35,2.3,2.1,.9,'Event memory\nK, V',blue,12);arrow((4.2,2.65),(5.35,2.75))
 box(8.05,2.25,2.35,.95,'Learn reliability\nr_i in [0,1]',orange,12);arrow((7.45,2.75),(8.05,2.75))
 box(8.05,4.6,3.72,1.05,'Weighted cross-attention',purple,13)
 arrow((7.45,5.28),(8.05,5.13));arrow((9.22,3.2),(9.22,4.6))
 arrow((7.45,3.0),(8.05,4.86),style='--')
 txt(9.9,4.17,'attention logits + log(r_i + epsilon)',10,ha='center')
 txt(8.65,3.67,'Use coherent history; down-weight unreliable observations.',9,color='#786a52',ha='center')
 txt(8.65,1.98,'No useful evidence?  Read memory / null token.',10,ha='center')
 # Prediction is mathematical / schematic, avoiding a fake mesh output.
 box(13.05,4.6,2.0,1.05,'Transformer\ndecoder',purple,12);arrow((11.77,5.13),(13.05,5.13))
 box(15.72,4.6,1.85,1.05,'3D pose\nhead',green,12);arrow((15.05,5.13),(15.72,5.13))
 txt(16.65,4.18,'J_t: 22 x 3',12,'bold',ha='center')
 txt(14.95,3.3,'GT supervision during training only',9,color='#7a8790',ha='center')
 txt(14.95,2.87,'Architecture proposal',11,'bold',ha='center');txt(14.95,2.51,'No trained weighting or pose result shown',9,color='#788590',ha='center')
 # Bottom evidence strip, made entirely from actual observed samples.
 # Place in a separate figure region to keep the pipeline clean and examples legible.
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'paper_pipeline.{ext}',dpi=240,facecolor='white')
 plt.close(fig)


def evidence(cases):
 data,meta=cases[0];weak,wm=cases[-1]
 source_meta=json.loads((ROOT/'exp/261028/dynamic_aggregation_real_examples/실제예시_메타데이터.json').read_text())
 arms=source_meta[0]['metrics'][0]['regions'][0]['box_xyxy'];legs=source_meta[0]['metrics'][0]['regions'][1]['box_xyxy'];head=source_meta[1]['metrics'][0]['regions'][0]['box_xyxy']
 entries=[('Moving arms',data,arms,'Longer history mixes past positions.'),('Low-motion legs',data,legs,'Longer history reveals more observations.'),('Weak response',weak,head,'More observations do not guarantee clear structure.')]
 fig,axes=plt.subplots(1,6,figsize=(17,5.3));fig.subplots_adjust(left=.02,right=.99,top=.78,bottom=.13,wspace=.06)
 fig.text(.02,.965,'Why adaptive, learned aggregation?',fontsize=21,weight='bold',va='top')
 for k,(name,d,roi,note) in enumerate(entries):
  for j,ms in enumerate([33,200]):
   ax=axes[k*2+j];ax.imshow(crop(d[f'event{ms}'],roi),interpolation='nearest');ax.set_title(f'{ms} ms',fontsize=12,pad=6);ax.axis('off')
  a=axes[k*2].get_position();b=axes[k*2+1].get_position();cx=(a.x0+b.x1)/2
  fig.text(cx,.865,name,fontsize=14,weight='bold',ha='center')
  fig.text(cx,.055,note,fontsize=10,ha='center',va='center')
 fig.text(.99,.985,'Actual dataset observations; not learned outputs',fontsize=9,ha='right',va='top',color='#677681')
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'actual_evidence_insets.{ext}',dpi=220,facecolor='white')
 plt.close(fig)


def main():
 cases=[load_case(*spec) for spec in CASES]
 gallery(cases);pipeline(cases);evidence(cases)
 manifest=dict(created_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),cases=[m for _,m in cases],
  real_rgb_event_samples=4,learned_aggregation_results=False,visible_text_language='English',
  design_outputs_schematic=True,source_dataset_readonly=True,formats=['png','svg','pdf'])
 (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2));print('ALL FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()

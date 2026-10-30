"""Render measured causal windows and diagnostic crops from immutable CSR data."""
from pathlib import Path
import json,hashlib,subprocess,datetime
import zipfile
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from visualize_rgb_event_alignment import prefix_array

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/261028/dynamic_aggregation_real_examples';OUT.mkdir(parents=True,exist_ok=True)
fp=subprocess.check_output(['fc-match','UnDotum','-f','%{file}'],text=True).strip();font_manager.fontManager.addfont(fp)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=fp).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11})
cv2.setNumThreads(1)

def intensity(p,n):
 a=np.clip(np.log1p(p)/np.log(9),0,1)**.5;b=np.clip(np.log1p(n)/np.log(9),0,1)**.5
 return np.clip(np.stack([a,.25*a+.65*b,b],-1),0,1)

def main():
 allmeta=[]
 for session,seconds,kind in [('20260720_131207_863008_200_b',12.,'전신'),('20260720_125338_040324_200_f',5.,'상체')]:
  source=ROOT/'dataset'/session/'device0';path=source/'event.npz';before={f:[(source/f).stat().st_size,(source/f).stat().st_mtime_ns] for f in ['event.npz','rgb.mp4','label.h5']}
  print('BEGIN',session,flush=True)
  with np.load(path,allow_pickle=False) as z:
   et=z['timestamp_ns'].astype(np.int64);rt=z['rgb__timestamp_ns'].astype(np.int64);ptr=z['indptr'];h,w=map(int,z['image_shape'])
  target=int(rt[0])+round(seconds*1e9);rgb_id=int(np.searchsorted(rt,target,side='right'))-1;end=int(rt[rgb_id])
  lo,hi=np.searchsorted(et,[end-200_000_000,end],side='right');a,b=map(int,ptr[[lo,hi]])
  # Decompress a prefix but retain exactly the causal 200ms observation interval.
  idx=prefix_array(path,'indices',b)[a:b];pol=prefix_array(path,'data',b)[a:b]>0
  rowptr=ptr[lo:hi+1]-a;localtimes=et[lo:hi]
  cap=cv2.VideoCapture(str(source/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,rgb_id);ok,bgr=cap.read();cap.release();assert ok
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  # Diagnostic boxes from previously reconstructed GT projected into event camera.
  cache=ROOT/'exp/261026/idea_motion_native_probe'/f'{session}_device0.npz'
  with np.load(cache) as z:
   k=int(np.argmin(np.abs(z['times']-end)));uv=z['uv'][k];gtstamp=int(z['times'][k])
   past=(z['times']>=end-200_000_000)&(z['times']<=end)
   history_uv=z['uv'][past] if past.any() else uv[None]
  if kind=='전신':
   # Include both limbs and their recent positions, with visible surrounding context.
   arm=history_uv[:,[16,17,18,19,20,21]].reshape(-1,2)
   x0,y0=np.floor(arm.min(0)-35).astype(int);x1,y1=np.ceil(arm.max(0)+35).astype(int)
   leg=history_uv[:,[0,1,2,4,5,7,8,10,11]].reshape(-1,2)
   lx0,ly0=np.floor(leg.min(0)-45).astype(int);lx1,ly1=np.ceil(leg.max(0)+45).astype(int)
   # Avoid a narrow strip: retain width at least 70% of the crop height.
   min_width=.7*(ly1-ly0)
   if lx1-lx0<min_width:
    cx=(lx0+lx1)/2;lx0=int(np.floor(cx-min_width/2));lx1=int(np.ceil(cx+min_width/2))
   rois=[('양팔·손목 전체',np.array([x0,y0,x1,y1])),('양쪽 다리 전체',np.array([lx0,ly0,lx1,ly1]))]
  else:
   center=uv[15];rois=[('머리 주변',np.r_[center-[55,60],center+[55,60]].astype(int)),('가슴 주변',np.r_[uv[9]-[45,35],uv[9]+[45,35]].astype(int))]
  legal=[]
  for label,box in rois:
   box[0::2]=np.clip(box[0::2],0,w);box[1::2]=np.clip(box[1::2],0,h)
   if box[2]-box[0]>10 and box[3]-box[1]>10:legal.append((label,box))
  assert legal
  images=[];metrics=[];count_arrays={}
  for ms in [10,33,100,200]:
   start=int(np.searchsorted(localtimes,end-ms*1_000_000,side='right'));aa=int(rowptr[start]);ids=idx[aa:];pos=pol[aa:]
   p=np.bincount(ids[pos],minlength=h*w).reshape(h,w);n=np.bincount(ids[~pos],minlength=h*w).reshape(h,w)
   image=intensity(p,n);images.append(image);count_arrays[f'p{ms}']=p;count_arrays[f'n{ms}']=n
   m=dict(window_ms=ms,observations=len(ids),unique_active_pixels=int(np.count_nonzero(p+n)),regions=[])
   for label,box in legal:
    x0,y0,x1,y1=box;active=(p+n)[y0:y1,x0:x1]>0
    m['regions'].append(dict(name=label,box_xyxy=box.tolist(),active_pixels=int(active.sum()),area_pixels=int(active.size),active_fraction=float(active.mean())))
   metrics.append(m)
  np.savez_compressed(OUT/f'{kind}_누적수치.npz',**count_arrays,timestamp_ns=np.array(end))
  rows=1+len(legal);fig,axes=plt.subplots(rows,5,figsize=(19,4+3.0*len(legal)),squeeze=False,gridspec_kw={'height_ratios':[1.7]+[1]*len(legal)})
  fig.subplots_adjust(left=.025,right=.99,top=.87,bottom=.075,wspace=.06,hspace=.35)
  fig.suptitle(f'실제 {kind} 장면 | {session} / device0 / RGB 시작 후 {(end-int(rt[0]))/1e9:.3f}초',fontsize=17,weight='bold',y=.97)
  fig.text(.5,.92,'같은 현재 시각, 과거 구간만 사용 · 원본 희소 이벤트에서 직접 누적 · 동적 집계 모델 출력은 아직 없음',ha='center',fontsize=12)
  axes[0,0].imshow(rgb);axes[0,0].set_title(f'실제 RGB · frame {rgb_id}\nRGB/event는 다른 카메라')
  for j,(im,ms) in enumerate(zip(images,[10,33,100,200]),1):
   axes[0,j].imshow(im);axes[0,j].set_title(f'실제 이벤트 {ms}ms 누적')
   for ri,(label,box) in enumerate(legal):
    x0,y0,x1,y1=box;axes[0,j].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=['#ffe35b','#61e17a'][ri],linewidth=1.1))
  for ri,(label,box) in enumerate(legal,1):
   x0,y0,x1,y1=box;axes[ri,0].axis('off');axes[ri,0].text(.08,.75,label,fontsize=17,weight='bold',color=['#957c00','#217d34'][ri-1])
   axes[ri,0].text(.08,.48,f'이벤트 좌표 ROI\n({x0}, {y0})-({x1}, {y1})\nGT 투영은 진단 위치 선정에만 사용',fontsize=10,linespacing=1.6)
   for j,im in enumerate(images,1):
    axes[ri,j].imshow(im[y0:y1,x0:x1],interpolation='nearest');v=metrics[j-1]['regions'][ri-1]
    axes[ri,j].set_title(f"활성 {v['active_pixels']:,} / {v['area_pixels']:,} px\n{100*v['active_fraction']:.2f}%",fontsize=11)
  for ax in axes.ravel():ax.axis('off')
  fig.text(.5,.026,'양극성 주황 / 음극성 청록 · 모든 창에 같은 로그 밝기 스케일 적용 · ROI 활성도는 자세 정확도·가시성·정보량의 정답이 아님',ha='center',fontsize=11)
  fig.savefig(OUT/f'{kind}_실제누적_비교.png',dpi=145,facecolor='white')
  if kind=='전신':fig.savefig(OUT/'전신_넓은영역_실제누적_비교.png',dpi=145,facecolor='white')
  plt.close(fig)
  if kind=='전신':
   for label,box in legal:
    x0,y0,x1,y1=box
    fig2,axs=plt.subplots(1,4,figsize=(18,8.5 if '다리' in label else 5.0))
    fig2.subplots_adjust(left=.025,right=.99,top=.83,bottom=.12,wspace=.08)
    fig2.suptitle(f'실제 {label} 확대 | {session} / device0 / {(end-int(rt[0]))/1e9:.3f}초',fontsize=19,weight='bold',y=.965)
    fig2.text(.5,.91,f'같은 ROI ({x0}, {y0})-({x1}, {y1}), 원본 {x1-x0}×{y1-y0}px · 원본 CSR 누적 · 임의 보완/생성 없음',ha='center',fontsize=12)
    for j,(axis,im,ms) in enumerate(zip(axs,images,[10,33,100,200])):
     axis.imshow(im[y0:y1,x0:x1],interpolation='nearest')
     v=next(q for q in metrics[j]['regions'] if q['name']==label)
     axis.set_title(f"{ms}ms\n활성 {v['active_pixels']:,} / {v['area_pixels']:,} px ({100*v['active_fraction']:.2f}%)",fontsize=14)
     axis.axis('off')
    fig2.text(.5,.045,'과거 (t-W,t]만 사용 · 주황/청록은 극성 · 모든 창에서 동일한 밝기 스케일 · 아직 학습된 동적 집계 출력은 아님',ha='center',fontsize=12)
    stem='양쪽다리_전체영역_실제비교' if '다리' in label else '양팔_전체영역_실제비교'
    fig2.savefig(OUT/f'{stem}.png',dpi=160,facecolor='white');plt.close(fig2)

  after={f:[(source/f).stat().st_size,(source/f).stat().st_mtime_ns] for f in before};assert before==after
  m=dict(session=session,device=0,kind=kind,source=str(source),rgb_id=rgb_id,rgb_origin_ns=int(rt[0]),timestamp_ns=end,relative_seconds=(end-int(rt[0]))/1e9,
   gt_roi_timestamp_ns=gtstamp,gt_roi_age_difference_ms=(gtstamp-end)/1e6,roi_gt_diagnostic_only=True,
   rgb_event_spatially_overlaid=False,causal=True,window_interval='(t-W,t]',
   min_read_event_timestamp_ns=int(localtimes[0]),max_read_event_timestamp_ns=int(localtimes[-1]),metrics=metrics,
   normalization='sqrt(clip(log1p(count)/log(9),0,1)), polarity separated; fixed across windows',
   source_stat_unchanged=True,source_stat=before,dynamic_aggregation_output=False)
  assert m['max_read_event_timestamp_ns']<=end
  allmeta.append(m);(OUT/'실제예시_메타데이터.json').write_text(json.dumps(allmeta,ensure_ascii=False,indent=2));print('DONE',kind,flush=True)
 (OUT/'완료.json').write_text(json.dumps(dict(completed_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),scenes=len(allmeta),success=True),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

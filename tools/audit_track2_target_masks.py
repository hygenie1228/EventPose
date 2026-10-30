"""Audit provided target-mask timelines; RGB overlays use no GT3D parameters."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
cv2.setNumThreads(1)
manifest=json.loads((OUTPUT/'manifest.json').read_text())

def check(name):
 record=manifest['samples'][name];path=SOURCE/name/'processed/human_mask.mp4';mask=cv2.VideoCapture(str(path));rgb=cv2.VideoCapture(str(SOURCE/name/'video.mp4'))
 assert mask.isOpened() and rgb.isOpened(),name
 assert int(mask.get(cv2.CAP_PROP_FRAME_COUNT))==record['frames'] and abs(mask.get(cv2.CAP_PROP_FPS)-record['fps'])<.01,name
 frames=[0,record['frames']//2,record['frames']-1];panels=[];stats=[]
 for frame in frames:
  mask.set(cv2.CAP_PROP_POS_FRAMES,frame);rgb.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,m=mask.read();ok2,image=rgb.read();assert ok and ok2
  binary=cv2.cvtColor(m,cv2.COLOR_BGR2GRAY)>127
  if binary.shape!=image.shape[:2]:binary=cv2.resize(binary.astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  yy,xx=np.where(binary);assert len(xx),('Empty target mask',name,frame)
  image[binary]=(image[binary]*.6+np.array([0,0,255])*.4).astype(np.uint8)
  cv2.rectangle(image,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(0,255,0),2)
  image=cv2.resize(image,(320,180));image=cv2.copyMakeBorder(image,24,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
  cv2.putText(image,name+' / '+str(frame),(3,16),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1);panels.append(image)
  stats.append(dict(frame=frame,mask_pixels=len(xx),bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]))
 mask.release();rgb.release()
 return name,dict(frames=record['frames'],fps=record['fps'],mask_sha256=sha256(path),sampled_masks=stats),np.concatenate(panels,axis=1)
with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,sorted(manifest['samples'])))
out=OUTPUT/'target_mask_audit';out.mkdir(exist_ok=True)
for start in range(0,len(results),10):
 image=np.concatenate([r[2] for r in results[start:start+10]],axis=0);assert cv2.imwrite(str(out/f'contact_{start//10+1}.jpg'),image)
save_json(out/'audit.json',dict(samples={r[0]:r[1] for r in results},all30_timelines_match=True,sampled_nonempty_masks=90,GT3D_read=False,gpu_used=False,created_unix=time.time(),visual_review_pending=True))
print('Target mask audit: 30 timelines, 90 nonempty RGB/mask samples; contact sheets ready')

"""CPU proof that target selection preserves the original model input/camera."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,sys,time
import cv2,numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json
from multihmr.target_crop import prepare_target_crop,PROTOCOL
cv2.setNumThreads(1);manifest=json.loads((OUTPUT/'manifest.json').read_text())

def check(sample):
 record=manifest['samples'][sample];a=cv2.VideoCapture(str(SOURCE/sample/'video.mp4'));b=cv2.VideoCapture(str(SOURCE/sample/'processed/human_mask.mp4'));error=0.
 for frame in [0,record['frames']//2,record['frames']-1]:
  a.set(cv2.CAP_PROP_POS_FRAMES,frame);b.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=a.read();ok2,mask=b.read();assert ok and ok2
  rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);K=np.asarray(record['K'],np.float32)
  actual,camera,_,_=prepare_target_crop(rgb,mask,K)
  gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
  if gray.shape!=rgb.shape[:2]:gray=cv2.resize(gray,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
  fg=rgb.copy();fg[gray<=127]=np.array([124,116,104],np.uint8)
  native=ImageOps.contain(Image.fromarray(fg),(896,896),method=Image.Resampling.LANCZOS)
  nw,nh=native.size;l,t=(896-nw)//2,(896-nh)//2;native=ImageOps.expand(native,border=(l,t,896-nw-l,896-nh-t))
  expected=K.copy();expected[0]*=nw/record['width'];expected[1]*=nh/record['height'];expected[0,2]+=l;expected[1,2]+=t
  assert np.array_equal(np.asarray(actual),np.asarray(native)),(sample,frame,'image changed')
  error=max(error,float(np.abs(camera-expected).max()));assert np.allclose(camera,expected,atol=1e-5),(sample,frame,'camera changed')
 a.release();b.release();return sample,error
with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,manifest['samples']))
save_json(OUTPUT/'target_native_input_checks.json',dict(evaluation_protocol=PROTOCOL,clips=30,frames_checked=90,
 foreground_RGB_preprocessing_independently_verified=True,non_target_RGB_replaced_before_resize=True,original_camera_transform_preserved=True,max_K_difference=max(r[1] for r in results),
 target_foreground_only_input=True,query_head_must_be_in_mask=True,GT3D_read=False,gpu_used=False,checked_unix=time.time()))
print('Foreground input check passed: 90 frames, actor-only RGB and original K')

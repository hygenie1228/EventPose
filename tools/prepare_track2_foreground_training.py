"""Cache target-only training RGB, using supplied 2D masks and unchanged K."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,sys,time
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
cv2.setNumThreads(1);work=OUTPUT/'work';records=json.loads((work/'training_index.json').read_text())+json.loads((work/'validation_index.json').read_text());groups={}
for r in records:groups.setdefault(r['session'],set()).add(r['frame'])

def prepare(item):
 sample,frames=item;mask_path=SOURCE/sample/'processed/human_mask.mp4';cap=cv2.VideoCapture(str(mask_path));assert cap.isOpened();out=work/'target_foreground_images'/sample;out.mkdir(parents=True,exist_ok=True)
 for frame in sorted(frames):
  destination=out/f'{frame:06d}.png'
  if destination.exists():continue
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,mask=cap.read();assert ok,(sample,frame)
  with Image.open(work/'images'/sample/f'{frame:06d}.jpg') as image:rgb=np.asarray(image.convert('RGB')).copy()
  gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
  if gray.shape!=rgb.shape[:2]:gray=cv2.resize(gray,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
  assert (gray>127).any(),('Missing target mask at training frame',sample,frame)
  rgb[gray<=127]=np.array([124,116,104],np.uint8)
  temporary=destination.with_name(destination.stem+'.partial.png');Image.fromarray(rgb).save(temporary,compress_level=3);temporary.replace(destination)
 cap.release();return sample,dict(frames=len(frames),mask_sha256=sha256(mask_path))
with ThreadPoolExecutor(max_workers=6) as pool:results=dict(pool.map(prepare,groups.items()))
assert len(results)==20
save_json(work/'target_foreground_preparation.json',dict(complete=True,samples=results,frames=sum(v['frames'] for v in results.values()),protocol='target_foreground_head_v1',K_unchanged=True,GT3D_used_for_masking=False,gpu_used=False,completed_unix=time.time()))
print('Target-only RGB cache ready:',sum(v['frames'] for v in results.values()),'frames /20 Train sequences',flush=True)

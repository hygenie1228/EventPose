"""Visualize server I aligned depth without writing to either dataset."""
from pathlib import Path
import json
import subprocess
import cv2
import h5py
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
SOURCE=ROOT/'orig_dataset/data1/20260720_125338_040324_200_f/RGBD'
OUT=ROOT/'exp/261023/depth_visualization'


def decode(group,index):
    a,b=map(int,group['offsets'][index:index+2])
    image=cv2.imdecode(group['encoded_data'][a:b],cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def comparison(rgb,depth):
    # Same 0..5000 raw-value/Turbo visualization as visualize_data.py.
    scaled=np.clip(depth.astype(np.float32)/5000*255,0,255).astype(np.uint8)
    color=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
    color[depth==0]=0
    canvas=np.zeros((400,1280,3),np.uint8)
    canvas[40:,:640]=cv2.resize(rgb,(640,360),interpolation=cv2.INTER_AREA)
    canvas[40:,640:]=cv2.resize(color,(640,360),interpolation=cv2.INTER_AREA)
    cv2.putText(canvas,'RGB',(10,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(canvas,'Aligned depth: 0..5000 raw units; invalid=black',(650,27),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    return canvas


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[]
    for i in range(5):
        with h5py.File(SOURCE/f'device{i}.h5','r') as f:
            g=f['streams/depth_align'];n=len(g['offsets'])-1;index=n//2
            depth=decode(g,index);rgb=decode(f['streams/rgb'],index)
            assert depth.dtype==np.uint16
            assert cv2.imwrite(str(OUT/f'device{i}_depth_raw.png'),depth)
            assert cv2.imwrite(str(OUT/f'device{i}_rgb_depth.jpg'),comparison(rgb,depth))
            valid=depth[depth>0]
            records.append({'device':i,'frames':n,'sample_index':index,'shape':list(depth.shape),
                'dtype':str(depth.dtype),'valid_percent':float(np.count_nonzero(depth)/depth.size*100),
                'valid_raw_percentiles':np.percentile(valid,[1,50,99]).tolist(),
                'timestamp_ns':int(g['timestamp_ns'][index])})
            print(records[-1],flush=True)
            if i==0:
                ts=g['timestamp_ns'][:].astype(np.int64)
                fps=1e9/float(np.median(np.diff(ts)))
                count=min(round(fps*10),n-index)
                path=OUT/'device0_rgb_depth.mp4'
                process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
                    '-s','1280x400','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4','-preset','veryfast',
                    '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
                try:
                    for j in range(index,index+count):
                        process.stdin.write(comparison(decode(f['streams/rgb'],j),decode(g,j)).tobytes())
                finally:
                    process.stdin.close();code=process.wait()
                assert code==0
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                    'stream=codec_name,pix_fmt,nb_frames,width,height','-of','json',str(path)]))
                assert int(probe['streams'][0]['nb_frames'])==count
                records[-1]['video']=probe
    (OUT/'validation.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Depth visualizations verified',flush=True)


if __name__=='__main__':main()

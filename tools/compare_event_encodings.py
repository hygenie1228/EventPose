"""Render causal event representations from stored CSR event-frame observations."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from visualize_rgb_event_alignment import prefix_array


def encoder(path, width, height, fps):
    partial = path.with_name(path.stem + '.partial.mp4')
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-threads', '3', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE)
    process.output_path = path
    process.partial_path = partial
    return process


def intensity(count):
    return np.sqrt(np.minimum(np.log1p(count) / np.log(9), 1)).astype(np.float32)


def polarity_image(positive, negative):
    # Positive orange/red, negative cyan; overlap remains visible, without cancellation.
    p, n = intensity(positive), intensity(negative)
    return np.uint8(np.clip(np.stack([n, .65*n + .25*p, p], -1), 0, 1)*255)


def fit(image, width=480, height=360):
    scale = min(width/image.shape[1], height/image.shape[0])
    w, h = round(image.shape[1]*scale), round(image.shape[0]*scale)
    result = np.zeros((height, width, 3), np.uint8)
    result[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w] = cv2.resize(image, (w,h), interpolation=cv2.INTER_AREA)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    path = args.source/'event.npz'
    with np.load(path, allow_pickle=False) as z:
        times = z['timestamp_ns'].astype(np.int64)
        rgb_times = z['rgb__timestamp_ns'].astype(np.int64)
        height, width = map(int, z['image_shape'])
        ptr = z['indptr']
    origin = int(rgb_times[0])
    targets = origin + np.arange(round(args.seconds*args.fps))*int(1e9/args.fps)
    rgb_ids = np.searchsorted(rgb_times, targets)
    rgb_ids = np.minimum(rgb_ids, len(rgb_times)-1)
    previous = np.maximum(rgb_ids-1, 0)
    rgb_ids = np.where(abs(rgb_times[previous]-targets) < abs(rgb_times[rgb_ids]-targets), previous, rgb_ids)
    end_row = np.searchsorted(times, rgb_times[rgb_ids[-1]], side='right')
    indices = prefix_array(path, 'indices', int(ptr[end_row]))
    positive = prefix_array(path, 'data', int(ptr[end_row])) > 0
    cap = cv2.VideoCapture(str(args.source/'rgb.mp4'))
    names = ['RGB reference', 'Signed binary | 33 ms', 'Split log-count | 33 ms',
             'Split log-count | 100 ms', 'Split log-count | 200 ms',
             'Time surface | tau 50 ms', '3 time bins | 100 ms',
             'Supported log-count | 100 ms', 'Multi-scale | 33/100/200 ms']
    support_ms = 33 if args.refine else 100
    if args.refine:
        names[4] = 'Time surface | tau 20 ms'
        names[7] = 'Supported log-count | 33 ms'
    videos = {'grid': encoder(args.output/'comparison_60s.mp4',1440,1176,args.fps),
              'contour': encoder(args.output/f'contour_supported_{support_ms}ms.mp4',960,720,args.fps),
              'motion': encoder(args.output/'motion_3bins_100ms.mp4',960,720,args.fps)}
    if args.refine:
        videos['recent'] = encoder(args.output/'time_surface_20ms.mp4',960,720,args.fps)
    n = height*width
    snapshots = {5, 12, 25, 40, 55}
    cursor = -1
    for step, rgb_id in enumerate(rgb_ids):
        while cursor < rgb_id:
            assert cap.grab()
            cursor += 1
        ok, rgb = cap.retrieve()
        assert ok
        end = int(rgb_times[rgb_id])

        def bounds(ms):
            lo, hi = np.searchsorted(times, [end-round(ms*1e6), end], side='right')
            return int(lo), int(hi), int(ptr[lo]), int(ptr[hi])

        counts = {}
        for ms in [33,100,200]:
            lo, hi, a, b = bounds(ms)
            ids, pos = indices[a:b], positive[a:b]
            p = np.bincount(ids[pos], minlength=n).reshape(height,width)
            m = np.bincount(ids[~pos], minlength=n).reshape(height,width)
            counts[ms] = (p,m)
        p,m = counts[33]
        signed = np.zeros((height,width,3),np.uint8)
        signed[p>m] = (0,64,255)
        signed[m>p] = (255,166,0)
        panels = [rgb, signed, polarity_image(p,m),
                  polarity_image(*counts[100]), polarity_image(*counts[200])]

        lo,hi,a,b = bounds(200)
        # Frame timestamps approximate observation times; sub-frame raw event times are unavailable.
        obs_times = np.repeat((times[lo:hi]-end)/1e9, np.diff(ptr[lo:hi+1]))
        last_p, last_m = np.full(n,-np.inf),np.full(n,-np.inf)
        ids,pos = indices[a:b],positive[a:b]
        np.maximum.at(last_p,ids[pos],obs_times[pos])
        np.maximum.at(last_m,ids[~pos],obs_times[~pos])
        sp,sm = np.exp(last_p/.05).reshape(height,width),np.exp(last_m/.05).reshape(height,width)
        panels.append(np.uint8(np.clip(np.stack([sm,.65*sm+.25*sp,sp],-1),0,1)*255))
        if args.refine:
            rp,rm = np.exp(last_p/.02).reshape(height,width),np.exp(last_m/.02).reshape(height,width)
            recent = np.uint8(np.clip(np.stack([rm,.65*rm+.25*rp,rp],-1),0,1)*255)
            panels[4] = recent

        edges = np.linspace(end-100_000_000,end,4).astype(np.int64)
        rows = np.searchsorted(times,edges,side='right')
        temporal = []
        for j in range(3):
            x,y = int(ptr[rows[j]]),int(ptr[rows[j+1]])
            temporal.append(intensity(np.bincount(indices[x:y],minlength=n).reshape(height,width)))
        # Old=red, middle=green, recent=blue. This display intentionally collapses polarity.
        motion = np.uint8(np.stack(temporal[::-1],-1)*255)
        panels.append(motion)
        p,m = counts[support_ms]
        total = p+m
        support = cv2.boxFilter((total>0).astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)
        mask = (total>0)&(support>=2)
        contour = polarity_image(p*mask,m*mask)
        panels.append(contour)
        panels.append(np.uint8(np.stack([intensity(sum(counts[33])),
                         intensity(sum(counts[100])),intensity(sum(counts[200]))],-1)*255))
        canvas = np.zeros((1176,1440,3),np.uint8)
        for j,(name,im) in enumerate(zip(names,panels)):
            row,col = divmod(j,3)
            x,y = col*480,row*392
            canvas[y+32:y+392,x:x+480] = fit(im)
            cv2.putText(canvas,name,(x+8,y+22),cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas,f'{(end-origin)/1e9:.1f}s',(370,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        videos['grid'].stdin.write(canvas.tobytes())
        videos['contour'].stdin.write(contour.tobytes())
        videos['motion'].stdin.write(motion.tobytes())
        if args.refine:
            videos['recent'].stdin.write(recent.tobytes())
        if step in {s*args.fps for s in snapshots}:
            cv2.imwrite(str(args.output/f'comparison_{step//args.fps:02d}s.jpg'),canvas)
        if step%100==0:
            print(f'{step}/{len(rgb_ids)} frames',flush=True)
    cap.release()
    for proc in videos.values():
        proc.stdin.close()
        assert proc.wait()==0
        proc.partial_path.replace(proc.output_path)
    manifest = dict(source=str(args.source), seconds=args.seconds, fps=args.fps,
        frames=len(rgb_ids), native_event_shape=[height,width], panels=names,
        timestamp='corrected timestamp_ns; all windows causal (t-window,t]',
        count_definition='number of stored nonzero pixel observations, NOT recovered raw event count',
        polarity='sign(data); -128 and +127 have equal unit weight',
        count_normalization='sqrt(clip(log1p(count)/log(9),0,1)), fixed across all frames; sqrt is display gamma',
        support_filter=f'{support_ms}ms total>0 AND >=2 occupied pixels in 3x3 neighborhood; no minimum recurrence',
        time_surface='exp((last observation timestamp-t)/50ms), lookback 200ms, split polarity',
        refined_time_surface='tau=20ms, lookback 200ms, split polarity' if args.refine else None,
        time_bins='three equal 100/3 ms bins, unsigned counts; old red, middle green, recent blue',
        multiscale='R=200ms, G=100ms, B=33ms unsigned log counts',
        spatial_alignment=False, labels_or_segmentation_used=False,
        limitation='Stored event-frame timestamps approximate observation times; within-frame event multiplicity/timing cannot be recovered.')
    (args.output/'settings.json').write_text(json.dumps(manifest,indent=2))
    print('Completed',flush=True)


if __name__=='__main__':
    main()

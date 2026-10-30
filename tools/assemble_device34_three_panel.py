"""Concatenate all published nine conditions x five clips into one three-panel MP4."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import write,register_job
GOAL=ROOT/'_agents/research/261028/device3_encoding_goal_12h.json'
OUT=ROOT/'exp/261029/device34_compact_video'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def prepare_event(clip):
    import cv2
    import numpy as np
    import h5py
    from utils.event_visualization import load_window,render
    cv2.setNumThreads(1)
    manifest=read(ROOT/'exp/261028/device34_test_video_gallery/manifest.json');source=Path(manifest['source_root']);task=clip['task'];name=task['parent_session']
    path=OUT/'event_cache'/(clip['clip']+'.h5');path.parent.mkdir(exist_ok=True)
    with h5py.File(source/'cache'/(name+'.h5'),'r',locking=False) as f:
        positions=np.arange(task['start_frame'],task['end_frame']);indices=np.searchsorted(f['frames'][:],positions)
        assert np.array_equal(f['frames'][indices],positions);stamps=f['timestamp_ns'][indices].astype(np.int64)
    archive=ROOT/'dataset'/name/'device4/event.npz';window=load_window(archive,int(stamps[0])-33_000_000,int(stamps[-1]));h,w=window['shape'];scale=min(640/w,360/h);nw,nh=round(w*scale),round(h*scale);left,top=(640-nw)//2,(360-nh)//2
    with h5py.File(path.with_suffix('.partial.h5'),'w',locking=False) as f:
        f.create_dataset('source_frames',data=positions);f.create_dataset('timestamp_ns',data=stamps)
        ds=f.create_dataset('event4_rgb',(len(stamps),360,640,3),dtype='u1',chunks=(1,360,640,3),compression='lzf')
        for i,t in enumerate(stamps):
            image=render(window,int(t),'count33',supported=True);canvas=np.zeros((360,640,3),np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA);ds[i]=canvas
        f.attrs.update(complete=True,causal=True,encoding='count33 support3x3 display only',native_height=h,native_width=w)
    path.with_suffix('.partial.h5').replace(path)
    return dict(clip=clip['clip'],frames=len(stamps),path=str(path),sha256=sha(path),source=str(archive.resolve()),native_shape=[h,w],all_timestamps_int64=True)

def main():
    import cv2
    import numpy as np
    import h5py
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor,as_completed
    g=read(GOAL);register_job(g,os.getpid(),'three_panel_video_assembly_CPU');OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);(OUT/'event_cache').mkdir(exist_ok=True)
    gallery=ROOT/'exp/261028/device34_test_video_gallery';manifest=read(gallery/'manifest.json');clips=manifest['clips'];models=list(manifest['models'])
    assert len(models)==9 and len(clips)==5
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(prepare_event,c) for c in clips]):records.append(f.result());write(OUT/'event_preparation_progress.json',dict(completed=len(records),total=5,records=records))
    write(OUT/'event_preparation_complete.json',dict(status='passed',records=records))
    ppt_review=Path(g['output_root'])/'encoding_explanation/editable/pptx_visual_review.json'
    while not ppt_review.exists():
        assert time.time()<g['deadline_unix']-300;time.sleep(5)
    cv2.setNumThreads(2);fps=30;destination=OUT/'Device3_RGB_Device4_Event_Mesh_All_Conditions_Clips1to5.mp4';partial=destination.with_suffix('.partial.mp4')
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x464','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-threads','3','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,start_new_session=True)
    register_job(g,encoder.pid,'three_panel_video_ffmpeg_encode');count=0;chapters=[];snapshots=[];proof=[]
    for model in models:
        config=read(manifest['models'][model]['config']);condition=config['condition']
        from multihmr.multisensor_repair import REPAIR_LAYOUTS
        sensors=[v for v in REPAIR_LAYOUTS[condition] if v]
        for clip in clips:
            task=clip['task'];source=gallery/'mp4'/model/(clip['clip']+'.mp4');assert source.exists();cap=cv2.VideoCapture(str(source));assert cap.isOpened()
            start=count;chapters.append(dict(model=model,clip=clip['clip'],start_frame=start,start_seconds=start/fps,source=str(source),source_sha256=sha(source)))
            with h5py.File(OUT/'event_cache'/(clip['clip']+'.h5'),'r',locking=False) as events:
                for i in range(task['video_frames']):
                    assert time.time()<g['deadline_unix']-120
                    ok,frame=cap.read();assert ok and frame.shape==(864,1920,3)
                    rgb=frame[98:458,:640];overlay=frame[498:858,640:1280];event=cv2.cvtColor(events['event4_rgb'][i],cv2.COLOR_RGB2BGR)
                    canvas=np.full((464,1920,3),20,np.uint8)
                    labels=['device3 RGB | '+('INPUT' if 'rgb3' in sensors else 'REFERENCE ONLY'),
                            'device4 Event | '+('INPUT' if 'event4' in sensors else 'REFERENCE ONLY'),
                            'Actual predicted mesh overlay | common RGB3 camera']
                    for j,(image,label) in enumerate(zip((rgb,event,overlay),labels)):
                        x=j*640;canvas[98:458,x:x+640]=image
                        cv2.putText(canvas,label,(x+10,88),cv2.FONT_HERSHEY_SIMPLEX,.49,(245,245,245),1,cv2.LINE_AA)
                    rel=(int(events['timestamp_ns'][i])-int(events['timestamp_ns'][0]))/1e9
                    cv2.putText(canvas,f'{model} | {clip["clip"]} / 5 | source frame {task["start_frame"]+i} | t={rel:.3f}s',(14,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1,cv2.LINE_AA)
                    cv2.putText(canvas,'Published 9-condition Test predictions | Event display: supported count33, red-orange (+), cyan (-) | No GT alignment',(14,51),cv2.FONT_HERSHEY_SIMPLEX,.47,(220,220,220),1,cv2.LINE_AA)
                    encoder.stdin.write(canvas.tobytes());count+=1
                    if i==150 and (model in (models[0],models[-1],'event3_event4','rgb3_event4')):
                        p=OUT/(model+'_'+clip['clip']+'_contact.png');cv2.imwrite(str(p),canvas);snapshots.append(dict(frame=count-1,path=str(p)))
                ok,_=cap.read();assert not ok,'Unexpected extra source frames'
            cap.release();proof.append(dict(model=model,clip=clip['clip'],frames=count-start))
            write(OUT/'assembly_progress.json',dict(segments=len(proof),total_segments=45,frames=count,last=proof[-1]))
    encoder.stdin.close();assert encoder.wait()==0
    assert count==9*5*301==13545;partial.replace(destination)
    verify=subprocess.Popen(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);register_job(g,verify.pid,'three_panel_video_full_decode_verification');_,errors=verify.communicate();assert verify.returncode==0,errors.decode()
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,pix_fmt','-of','json',str(destination)]))['streams'][0]
    assert int(info['nb_read_frames'])==count and info['width']==1920 and info['height']==464
    # Review images are decoded from the final MP4, never just pre-encode contacts.
    cap=cv2.VideoCapture(str(destination));reviews=[]
    for k,snap in enumerate(snapshots):
        if k not in (0,7,12,len(snapshots)-1):continue
        assert cap.set(cv2.CAP_PROP_POS_FRAMES,snap['frame']);ok,image=cap.read();assert ok
        p=OUT/('decoded_review_'+str(snap['frame'])+'.png');cv2.imwrite(str(p),image);reviews.append(str(p))
    cap.release()
    write(OUT/'verification.json',dict(status='full_decode_passed_pending_agent_visual_review',video=str(destination),sha256=sha(destination),
        frames=count,segments=45,conditions=models,clips=5,fps=fps,duration_seconds=count/fps,stream=info,chapters=chapters,source_proof=proof,
        all_published_predictions_preserved=True,mesh_overlay_cropped_from_verified_source=True,events_recomputed_from_causal_archive=True,
        event_display_reference=str(ROOT/'_agents/report/261024_인체윤곽과_자세변화를_위한_이벤트_인코딩_비교.md'),
        fixed_playback_note='30fps CFR; 301 source frames per segment retained; displayed t uses actual corrected timestamp',decoded_review_images=reviews,
        no_new_Test_performance_claim=True,agent_visual_review_required=True,experiment_goal_complete=False))
    report=ROOT/'_agents/report/261028_대표_테스트_구간별_센서융합_영상목록.md';body=report.read_text();title,_,rest=body.partition('\n')
    section=['','## RGB3·Event4·Mesh overlay 통합 영상','',f'[9조건 × 5구간 전체를 이어 붙인 MP4]({destination})', '',
        '레이아웃은 device3 RGB | device4 Event | 실제 예측 Mesh overlay다. 각 조건에서 구간1~5를 이어 붙이고,9조건을 같은 순서로 연결했다. '
        '총45구간·13,545프레임·30fps·7분31.5초이며 모든 source 프레임을 유지했다. 상단 t는 실제 보정 timestamp다.',
        '이벤트 표시는 지정한 기존 보고서와 같은 검정 배경, 고정 로그·제곱근 밝기, 양극성 빨강/주황·음극성 cyan, count33 약한3×3 이웃 지지를 사용한다. '
        '표시만 변경했고 학습 입력·기존 예측·정량값은 변경하지 않았다. 사용하지 않은 센서는 REFERENCE ONLY로 표시한다.', '',
        '| 조건 | 통합 영상 시작(분:초) |','|---|---|']
    for model in models:
        t=next(c['start_seconds'] for c in chapters if c['model']==model);section.append(f'| {model} | {int(t)//60:02d}:{t%60:04.1f} |')
    report.write_text(title+'\n'+'\n'.join(section)+'\n\n'+rest)
    print(destination,flush=True)

if __name__=='__main__':main()

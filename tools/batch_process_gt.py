"""Durable, resumable full-schema GT processing and 30-minute status checks.

Source datasets are strictly read-only. The user-facing schema is identical to
the reference session. Missing server-I inputs are NOT represented by fake files.
"""
import argparse
import concurrent.futures
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
EXP=Path(os.environ.get('EVENTPOSE_BATCH_WORK',str(ROOT/'exp/261023/all_gt_processed')))
OUTPUT=ROOT/'etri_processed'
GPU_IDS=(0,2,4,6)  # Four concurrent NVENC sessions; driver previously rejected more.
GROUPS={'params','bboxes','frames'}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def stat(path):
    s=Path(path).stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def expected_files(job):
    rgb='rgb.mp4' if job['device']<5 else 'video.mp4'
    names={'event.npz','event.mp4',rgb,'preview.mp4','label.h5'}
    if job['device']<5 and job['source_I']:names.add('depth.mp4')
    return names


def probe(path):
    r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,pix_fmt,width,height,nb_frames,avg_frame_rate','-show_entries','format=duration,tags',
        '-of','json',str(path)],capture_output=True,text=True,timeout=90)
    return json.loads(r.stdout) if r.returncode==0 else None


def valid_video(path,frames=None,width=None,height=None,max_seconds=None,strict=False):
    if not path.is_file():return False
    p=probe(path)
    if not p or not p.get('streams'):return False
    s=p['streams'][0]
    valid=(s.get('codec_name')=='h264' and (frames is None or int(s.get('nb_frames',-1))==frames)
        and (width is None or s['width']==width) and (height is None or s['height']==height)
        and (max_seconds is None or float(p['format']['duration'])<=max_seconds))
    if not valid or not strict:return valid
    r=subprocess.run(['ffprobe','-v','error','-count_packets','-select_streams','v:0','-show_entries',
        'stream=nb_frames,nb_read_packets','-of','json',str(path)],capture_output=True,text=True,timeout=900)
    if r.returncode or r.stderr.strip():return False
    d=json.loads(r.stdout)['streams'][0]
    return int(d.get('nb_frames',-1))==int(d.get('nb_read_packets',-2))


def source_I_lookup():
    roots=[ROOT/'orig_dataset']
    config=EXP/'extra_source_roots.json'
    if config.exists():roots += [Path(p) for p in json.loads(config.read_text())]
    result={}
    for root in roots:
        if not root.is_dir():continue
        if root.name.startswith('2026') and (root/'RGBD').is_dir():result.setdefault(root.name,str(root.absolute()))
        buckets=[root]+sorted(p for p in root.glob('data*') if p.is_dir())
        for bucket in buckets:
            for s in bucket.iterdir():
                if s.is_dir() and s.name.startswith('2026'):
                    result.setdefault(s.name,str(s.absolute()))
    return result


def inventory():
    if not (ROOT/'orig_dataset').is_dir() or not list((ROOT/'orig_dataset').glob('data*')):
        raise RuntimeError('Server I mount unavailable; do not confuse an unavailable mount with missing individual sessions')
    original=source_I_lookup();jobs=[];sessions=[]
    for source in sorted((ROOT/'dataset').iterdir()):
        if not source.is_dir():continue
        gt=list((source/'RGBD').glob('device*_mamma.h5'))
        if not gt:continue
        source_i=original.get(source.name)
        requirements=[]
        if source_i:
            si=Path(source_i)
            requirements=[si/'RGBD'/f'device{i}.h5' for i in range(5)]
            requirements += [si/m/f'device{i}.h5' for m in ['RGB','Event'] for i in [5,6]]
            if not all(p.is_file() for p in requirements):source_i=None
        sessions.append({'session':source.name,'source_I':source_i})
        for i in range(7 if source_i else 5):
            jobs.append({'id':source.name+f'/device{i}','session':source.name,'device':i,
                'source_L':str(source.absolute()),'source_I':source_i,
                'status':'pending' if i<5 or source_i else 'waiting_source_I','attempts':0})
    split_path=OUTPUT/'split.json'
    if split_path.exists():
        split=json.loads(split_path.read_text())
        test=set(split['test']);train=set(split['train'])
        assert not test&train and test|train=={j['session'] for j in jobs},'Split must cover every GT session exactly once'
        for j in jobs:j['split']='test' if j['session'] in test else 'train'
        jobs.sort(key=lambda j:(j['split']!='test',j['session'],j['device']))
    save(EXP/'session_inputs.json',sessions)
    return jobs


def phase(job,name,outputs,action,inputs):
    directory=EXP/'jobs'/job['session']/f"device{job['device']}";directory.mkdir(parents=True,exist_ok=True)
    record=directory/(name+'.json')
    signatures={str(p):stat(p) for p in inputs}
    if record.exists():
        old=json.loads(record.read_text())
        if old.get('inputs')==signatures and all(p.is_file() and old.get('outputs',{}).get(str(p))==stat(p) for p in outputs):
            print('REUSE',job['id'],name,flush=True);return old
    save(directory/'current_phase.json',{'phase':name,'started':now(),'pid':os.getpid()})
    print('START',job['id'],name,flush=True)
    result=action()
    assert signatures=={str(p):stat(p) for p in inputs},'Input changed during processing; retry from a stable snapshot'
    assert all(p.is_file() and p.stat().st_size>0 for p in outputs)
    value={'phase':name,'finished':now(),'inputs':signatures,'outputs':{str(p):stat(p) for p in outputs},'result':result}
    save(record,value);print('DONE',job['id'],name,flush=True);return value


def audit_event_L(path,source,i):
    manifest=json.loads((source/'Event'/f'device{i}'/'manifest.json').read_text())
    with np.load(path,allow_pickle=False) as z:
        ptr=z['indptr'];idx=z['indices'];data=z['data'];n=manifest['frames']
        assert len(ptr)==n+1 and int(z['shape'][0])==n and ptr[-1]==len(idx)==len(data)
        for shard in manifest['shards']:
            a,b=shard['start'],shard['stop'];start,stop=int(ptr[a]),int(ptr[b])
            base=source/'Event'/f'device{i}'
            with np.load(base/shard['matrix'],allow_pickle=False) as original:
                assert np.array_equal(ptr[a:b+1]-start,original['indptr'])
                assert np.array_equal(idx[start:stop],original['indices'])
                assert np.array_equal(data[start:stop],original['data'])
            with np.load(base/shard['frames'],allow_pickle=False) as original:
                for k in original.files:
                    expected=original[k]
                    assert np.array_equal(z[k] if k in ('image_shape','background_value') else z[k][a:b],expected),k
        with h5py.File(source/'RGB'/f'device{i}'/'images.h5','r') as original:
            assert np.array_equal(z['rgb__timestamp_ns'],original['streams/images/timestamp_ns'][:])
        return {'frames':n,'nnz':len(idx),'full_source_shard_comparison':True}


def prepare(job,target,directory):
    import compact_processed_session as compact
    source=Path(job['source_L']);i=job['device']
    compact.SOURCE=source;compact.SESSION=target.parent;compact.WORK=directory
    if i<5:
        gt=source/'RGBD'/f'device{i}_mamma.h5'
        def label():
            # Repack a compact three-group label. Retain person_valid inside params.
            (target/'label.h5').unlink(missing_ok=True)
            compact.label(target,target/'label.h5')
            with h5py.File(gt,'r') as src,h5py.File(target/'label.h5','r') as dst:
                assert set(dst)==GROUPS
                for group in GROUPS:
                    def check(name,obj):
                        if isinstance(obj,h5py.Dataset):
                            value=dst[group+'/'+name][:]
                            assert obj.shape==value.shape and np.array_equal(obj[:],value,equal_nan=True)
                    src[group].visititems(check)
            return {'groups':sorted(GROUPS),'person_valid_preserved':True}
        phase(job,'label',[target/'label.h5'],label,[gt])
        rgb=source/'RGB'/f'device{i}'/'images.h5'
        with h5py.File(rgb,'r') as f:n=len(f['streams/images/timestamp_ns'])
        def video():
            if valid_video(target/'rgb.mp4',n,strict=True):return {'reused_existing':True,'frames':n}
            (target/'rgb.mp4').unlink(missing_ok=True)
            return compact.video(target,target/'rgb.mp4')
        phase(job,'rgb',[target/'rgb.mp4'],video,[rgb])
        manifest=source/'Event'/f'device{i}'/'manifest.json'
        def event():
            if (target/'event.npz').exists():
                try:return audit_event_L(target/'event.npz',source,i)
                except Exception:traceback.print_exc()
            (target/'event.npz').unlink(missing_ok=True)
            return compact.events(target,target/'event.npz')
        phase(job,'event_npz',[target/'event.npz'],event,[manifest,rgb])
    else:
        import add_server_i_cameras as extra
        extra.NAME=job['session'];extra.SOURCE=Path(job['source_I']);extra.DEST=target.parent;extra.WORK=directory
        rgb=extra.SOURCE/'RGB'/f'device{i}.h5';event_source=extra.SOURCE/'Event'/f'device{i}.h5'
        with h5py.File(rgb,'r') as f:n=len(f['streams/images/timestamp_ns'])
        def video():
            if valid_video(target/'video.mp4',n,strict=True):return {'reused_existing':True,'frames':n}
            (target/'video.mp4').unlink(missing_ok=True)
            return extra.copy_video(i,target)
        phase(job,'rgb',[target/'video.mp4'],video,[rgb])
        def event():
            (target/'event.npz').unlink(missing_ok=True)
            return extra.copy_events(i,target)
        phase(job,'event_npz',[target/'event.npz'],event,[event_source,rgb])
        canonical=target.parent/'device0/label.h5'
        canonical_record=EXP/'jobs'/job['session']/'device0/label.json'
        deadline=time.monotonic()+3600
        while True:
            if canonical_record.exists() and canonical.exists():
                r=json.loads(canonical_record.read_text())
                if r.get('outputs',{}).get(str(canonical))==stat(canonical):break
            if time.monotonic()>deadline:raise TimeoutError('canonical device0 GT is not ready')
            time.sleep(2)
        phase(job,'label',[target/'label.h5'],lambda:extra.labels(i,target),
            [canonical,rgb,ROOT/'data/human_model_files/smplx/SMPLX_NEUTRAL.npz'])


@contextlib.contextmanager
def gpu_slot():
    locks=EXP/'encoder_locks';locks.mkdir(exist_ok=True)
    while True:
        for gpu in GPU_IDS:
            f=(locks/f'gpu{gpu}.lock').open('a+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:f.close();continue
            try:yield gpu
            finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()
            return
        time.sleep(2)


def worker(job):
    i=job['device'];target=OUTPUT/job['session']/f'device{i}'
    assert target.resolve().is_relative_to(OUTPUT.resolve())
    if i>=5 and not job['source_I']:return {'status':'waiting_source_I'}
    target.mkdir(parents=True,exist_ok=True)
    directory=EXP/'jobs'/job['session']/f'device{i}';directory.mkdir(parents=True,exist_ok=True)
    prepare(job,target,directory)
    with np.load(target/'event.npz',allow_pickle=False) as z:
        event_count=int(z['shape'][0]);h,w=int(z['frame_height']),int(z['frame_width'])
    def event_video():
        if valid_video(target/'event.mp4',event_count,w,h,strict=True):return {'reused_existing':True,'frames':event_count}
        (target/'event.mp4').unlink(missing_ok=True)
        with gpu_slot() as gpu:
            subprocess.run([sys.executable,str(ROOT/'tools/processed_event_videos.py'),'--kind','event',
                '--session',str(target.parent),'--device',target.name,'--gpus',str(gpu),
                '--work-root',str(EXP/'video_work'),'--cleanup-parts'],check=True)
        assert valid_video(target/'event.mp4',event_count,w,h)
        return {'frames':event_count,'all_stored_event_frames':True}
    if i<5 and job['source_I']:
        import add_lossless_depth as depth
        depth.SOURCE=Path(job['source_I'])/'RGBD';depth.DEST=target.parent;depth.WORK=directory/'depth'
        depth.WORK.mkdir(exist_ok=True)
        raw=depth.SOURCE/f'device{i}.h5'
        def archive():
            # Import the reference session's previous full-pixel verification if unchanged.
            old=ROOT/'exp/261023/lossless_depth'/f'device{i}_validation.json'
            if job['session']=='20260720_125338_040324_200_f' and old.exists() and (target/'depth.mp4').exists():
                r=json.loads(old.read_text())
                if r.get('lossless_all_frames_verified') and r['file_bytes']==(target/'depth.mp4').stat().st_size and (target/'depth.mp4').stat().st_mtime_ns<=old.stat().st_mtime_ns:
                    save(depth.WORK/f'device{i}_validation.json',r)
                    return {'reused_full_pixel_validation':str(old),'frames':r['depth_frames']}
            (target/'depth.mp4').unlink(missing_ok=True)
            r=depth.export(i);return {'frames':r['depth_frames'],'full_pixel_equality':True}
        phase(job,'depth',[target/'depth.mp4'],archive,[raw])
    if job.get('stage','core')=='core':
        required=expected_files(job)-{'event.mp4','preview.mp4'}
        with h5py.File(target/'label.h5','r') as f:assert set(f)==GROUPS
        result={'status':'core_done','finished':now(),'files':{name:stat(target/name) for name in sorted(required)}}
        save(directory/'core_complete.json',result);return result
    phase(job,'event_video',[target/'event.mp4'],event_video,[target/'event.npz'])
    if i<5 and job['source_I']:
        def preview():
            if valid_video(target/'preview.mp4',width=1152,height=240,max_seconds=60):return {'reused_existing':True}
            depth.preview(i);return {'panels':['Event','RGB','Depth'],'text':False,'maximum_seconds':60}
        phase(job,'preview',[target/'preview.mp4'],preview,[target/'event.npz',target/'rgb.mp4',target/'depth.mp4'])
        expected={'event.npz','event.mp4','rgb.mp4','depth.mp4','preview.mp4','label.h5'}
    else:
        rgb_name='rgb.mp4' if i<5 else 'video.mp4'
        def preview():
            if valid_video(target/'preview.mp4',width=768,height=240,max_seconds=60):return {'reused_existing':True}
            subprocess.run([sys.executable,str(ROOT/'tools/processed_event_videos.py'),'--kind','preview','--session',str(target.parent),
                '--device',target.name,'--cpu-preview','--max-seconds','60','--work-root',str(EXP/'video_work')],check=True)
            return {'panels':['Event','RGB'],'text':False,'maximum_seconds':60}
        phase(job,'preview',[target/'preview.mp4'],preview,[target/'event.npz',target/rgb_name])
        expected={'event.npz','event.mp4',rgb_name,'preview.mp4','label.h5'}
    assert {p.name for p in target.iterdir()}==expected
    with h5py.File(target/'label.h5','r') as f:assert set(f)==GROUPS
    result={'status':'done','finished':now(),'source_I_absent_authorized_skip':not bool(job['source_I']),
        'files':{name:stat(target/name) for name in sorted(expected)}}
    save(directory/'complete.json',result);return result


def busy_external_writer(target):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if not args or Path(os.fsdecode(args[0])).name!='ffmpeg':continue
        try:
            status=(p/'status').read_text().splitlines()
            pending=[int(line.split()[1],16) for line in status if line.startswith(('SigPnd:','ShdPnd:'))]
            if any(mask&(1<<8) for mask in pending) and 'read' in (p/'wchan').read_text():continue
        except (OSError,ValueError):pass
        for arg in args:
            if arg.endswith(b'.mp4'):
                try:
                    if Path(os.fsdecode(arg)).resolve().parent==target.resolve():return True
                except OSError:pass
    return False


def summarize(jobs,running):
    counts={}
    for j in jobs:counts[j['status']]=counts.get(j['status'],0)+1
    names=sorted({j['session'] for j in jobs})
    complete=[]
    for s in names:
        subset=[j for j in jobs if j['session']==s]
        if not all(j['status']=='done' for j in subset):continue
        root=OUTPUT/s
        if {p.name for p in root.iterdir()}!={f"device{j['device']}" for j in subset}:continue
        if all({p.name for p in (root/f"device{j['device']}").iterdir()}==expected_files(j) for j in subset):complete.append(s)
    return {'time':now(),'pid':os.getpid(),'total_sessions':len(names),'completed_sessions':len(complete),
        'completed_session_names':complete,'total_device_jobs':len(jobs),'device_status':counts,
        'core_completed_device_jobs':sum(j['status'] in ('done','core_done') or (j['status']=='running' and j.get('stage')=='visualization') for j in jobs),
        'processing_order':'all core data first, then event.mp4 and preview.mp4',
        'split_priority':'test first within each stage',
        'split_progress':{name:{'total':sum(j.get('split')==name for j in jobs),'core_done':sum(j.get('split')==name and (j['status'] in ('done','core_done') or (j['status']=='running' and j.get('stage')=='visualization')) for j in jobs),'done':sum(j.get('split')==name and j['status']=='done' for j in jobs)} for name in ('test','train')},
        'running_workers':[{'id':j['id'],'pid':p.pid,'started':started} for p,j,log,started in running.values()],
        'missing_source_I_sessions':sorted({j['session'] for j in jobs if not j['source_I']}),
        'missing_source_I_is_authorized_skip':True,
        'free_output_bytes':shutil.disk_usage(OUTPUT).free,'complete':len(complete)==len(names)}


def supervise():
    EXP.mkdir(parents=True,exist_ok=True)
    lock=(EXP/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=inventory();old_path=EXP/'jobs_state.json'
    if old_path.exists():
        old={j['id']:j for j in json.loads(old_path.read_text())}
        for j in jobs:
            if j['id'] in old:
                previous=old[j['id']]
                j['attempts']=previous.get('attempts',0)
                j['status']=previous['status'] if previous['status'] in ('done','core_done','failed','waiting_source_I') else 'pending'
                if j['status']=='waiting_source_I' and j['source_I']:j['status']='pending'
                if j['status']=='core_done':
                    record=EXP/'jobs'/j['session']/f"device{j['device']}"/'core_complete.json'
                    try:
                        r=json.loads(record.read_text());target=OUTPUT/j['session']/f"device{j['device']}"
                        if set(r['files'])!=expected_files(j)-{'event.mp4','preview.mp4'} or any(not (target/name).exists() or stat(target/name)!=v for name,v in r['files'].items()):j['status']='pending'
                    except (FileNotFoundError,KeyError):j['status']='pending'
                if j['status']=='done':
                    record=EXP/'jobs'/j['session']/f"device{j['device']}"/'complete.json'
                    try:
                        r=json.loads(record.read_text());target=OUTPUT/j['session']/f"device{j['device']}"
                        if set(r['files'])!=expected_files(j) or any(not (target/name).exists() or stat(target/name)!=s for name,s in r['files'].items()):j['status']='pending'
                    except (FileNotFoundError,KeyError):j['status']='pending'
    running={};last_refresh=time.monotonic()
    while True:
        for key,(process,j,log,started) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();running.pop(key)
            result_path=EXP/'jobs'/j['session']/f"device{j['device']}"/'worker_result.json'
            if code==0 and result_path.exists():
                r=json.loads(result_path.read_text());j['status']=r['status'];j['last_error']=None
            else:
                j['attempts']+=1;j['status']='pending' if j['attempts']<3 else 'failed'
                j['retry_after']=time.time()+60*(2**j['attempts'])
                j['last_error']=f'worker exit {code}; see {key}/worker.log'
        if time.monotonic()-last_refresh>=1800:
            original=source_I_lookup()
            for j in jobs:
                if j['status']=='waiting_source_I' and j['session'] in original:
                    j['source_I']=original[j['session']];j['status']='pending'
            last_refresh=time.monotonic()
        core_finished=all(j['status'] in ('done','core_done') or (j['status']=='running' and j.get('stage')=='visualization') for j in jobs)
        if shutil.disk_usage(OUTPUT).free>20*1024**3:
            for j in jobs:
                if len(running)>=8:break
                eligible=('core_done','pending') if core_finished else ('pending',)
                if j['status'] not in eligible or time.time()<j.get('retry_after',0):continue
                j['stage']='visualization' if core_finished else 'core'
                target=OUTPUT/j['session']/f"device{j['device']}"
                if busy_external_writer(target):continue
                directory=EXP/'jobs'/j['session']/f"device{j['device']}";directory.mkdir(parents=True,exist_ok=True)
                jobfile=directory/'job.json';save(jobfile,j)
                log=(directory/'worker.log').open('a')
                env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                process=subprocess.Popen([sys.executable,str(Path(__file__).absolute()),'--worker',str(jobfile)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                j['status']='running';running[j['id']]=(process,j,log,now())
        summary=summarize(jobs,running);save(EXP/'status.json',summary);save(old_path,jobs)
        if summary['complete']:
            print('ALL GT SESSIONS FULLY VERIFIED',now(),flush=True);return
        time.sleep(15)


def monitor():
    path=EXP/'status.json'
    status=json.loads(path.read_text()) if path.exists() else {'error':'no controller status yet'}
    service=subprocess.run(['systemctl','--user','is-active','eventpose-all-gt.service'],capture_output=True,text=True).stdout.strip()
    failed=[]
    state=EXP/'jobs_state.json'
    if state.exists():
        for j in json.loads(state.read_text()):
            if j.get('status')=='failed':
                log=EXP/'jobs'/j['session']/f"device{j['device']}"/'worker.log'
                failed.append({'id':j['id'],'error':j.get('last_error'),'log':str(log),
                    'last_lines':log.read_text(errors='replace').splitlines()[-15:] if log.exists() else []})
    report={'checked_at':now(),'service':service,'summary':status,'failed_count':len(failed),'failed_examples':failed[:10]}
    save(EXP/'monitor_latest.json',report)
    with (EXP/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    print(json.dumps(report),flush=True)
    if status.get('complete'):
        subprocess.run(['systemctl','--user','disable','--now','eventpose-all-gt-monitor.timer'],check=False)
        subprocess.run(['systemctl','--user','disable','eventpose-all-gt.service'],check=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--monitor',action='store_true');args=parser.parse_args()
    if args.monitor:monitor();return
    if args.worker:
        job=json.loads(args.worker.read_text())
        result=worker(job);save(args.worker.parent/'worker_result.json',result);return
    supervise()


if __name__=='__main__':main()

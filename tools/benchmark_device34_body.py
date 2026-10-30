"""Sequential same-UUID cached/online end-to-end timings for all six inputs."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.prepare_device34_body import OUT,GOAL,save
from tools.goal_runtime_control import register_job


def read(path):return json.loads(path.read_text())


def worker(name,kind):
    import torch
    from multihmr.multisensor_inference import load_model,CachedSensors,OnlineSensors,infer
    import utils.pose_evaluation as base
    import shutil
    import numpy as np
    config=read(OUT/'configs'/(name+'.json'));base.configure_torch()
    work=OUT/'runtime/work'/(name+'_'+kind);work.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OUT/'joint_regressor.npy',work/'joint_regressor.npy');base.OUT=work
    started=time.perf_counter();model=load_model(config,OUT/'checkpoints'/name/'best.pt')
    torch.cuda.synchronize();loading=time.perf_counter()-started
    current=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert {int(p) for p in current.split()}=={os.getpid()},'Concurrent compute work invalidates fair speed timing'
    records=[];cls=CachedSensors if kind=='cached' else OnlineSensors
    tasks=read(OUT/'runtime/tasks.json')
    for raw in tasks:
        task=dict(raw,video_frames=100,smoke=True,condition=config['condition'])
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
        source=cls(raw['parent_session'],config['condition']);opened=time.perf_counter()-begin
        collector=base.Collector(task,name);infer(model,collector,source,16);collector.save()
        encoding=getattr(source,'encoding_seconds',None);decode=getattr(source,'decode_seconds',None)
        pixels=sorted(source.pixel_datasets_read);source.close();torch.cuda.synchronize()
        records.append(dict(task=raw['session'],frames=100,seconds=time.perf_counter()-begin,
            source_open_seconds=opened,encoding_seconds=encoding,decode_seconds=decode,pixel_datasets_read=pixels,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    # Input parity is separately timed and excluded from pipeline fps/RSS above.
    begin=time.perf_counter();cached=CachedSensors(tasks[0]['parent_session'],config['condition'])
    online=OnlineSensors(tasks[0]['parent_session'],config['condition'])
    frame=tasks[0]['start_frame'];a=cached.sample(frame);b=online.sample(frame)
    assert (a is None)==(b is None)
    if a is not None:
        assert set(a)==set(b)
        assert all(torch.allclose(a[k],b[k],atol=1e-6,rtol=1e-6) for k in a),'Online/cache input mismatch'
    cached.close();online.close();parity_seconds=time.perf_counter()-begin
    active=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert {int(p) for p in active.split()}=={os.getpid()}
    seconds=sum(r['seconds'] for r in records)
    save(OUT/'runtime'/(name+'_'+kind+'.json'),dict(name=name,source=kind,frames=300,seconds=seconds,
        fps=300/seconds,loading_seconds=loading,host_peak_rss_bytes=rss,
        total_parameters=sum(p.numel() for p in model.parameters()),segments=records,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],physical_gpu=os.environ['EVENTPOSE_PHYSICAL_GPU'],
        cuda_name=torch.cuda.get_device_name(),pid=os.getpid(),input_parity_passed=True,
        separate_input_parity_seconds=parity_seconds,
        included='source open; original RGB decode/resize or HDF5 I/O; causal event encoding online; detection; reconstruction; camera transform; matching; save; source close',
        cache_policy='OS cache not forcibly evicted; model loading separated; each condition starts a new process'))


def supervise(gpu):
    from multihmr.multisensor_model import CONDITIONS
    goal=read(GOAL);assert time.time()<goal['deadline_unix']-600
    identity=subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],text=True).strip()
    uuid=identity.split(',')[1].strip();tasks=read(OUT/'test/tasks.json')
    parents=sorted({t['parent_session'] for t in tasks})
    chosen=[next(t for t in tasks if t['parent_session']==parents[i]) for i in (0,len(parents)//2,len(parents)-1)]
    save(OUT/'runtime/tasks.json',chosen)
    for name in CONDITIONS:
        for kind in ('cached','online'):
            path=OUT/'runtime'/(name+'_'+kind+'.json')
            if path.exists():assert read(path)['gpu_uuid']==uuid;continue
            assert time.time()<goal['deadline_unix']-300
            active=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            assert not active.strip(),'Other jobs must be preserved; wait for an idle GPU'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,EVENTPOSE_PHYSICAL_GPU=str(gpu),
                EVENTPOSE_MULTISENSOR_OUT=str(OUT),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            started=time.time()
            with (OUT/'runtime'/(name+'_'+kind+'.log')).open('a') as log:
                process=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',name,'--source',kind],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                register_job(goal,process.pid,'device34_runtime:'+name+':'+kind)
                assert process.wait()==0,(name,kind)
            result=read(path);result.update(gpu_identity=identity,process_started_unix=started,
                process_finished_unix=time.time(),cold_process_wall_seconds=time.time()-started)
            save(path,result)
    save(OUT/'runtime/summary.json',{name+'_'+kind:read(OUT/'runtime'/(name+'_'+kind+'.json'))
        for name in CONDITIONS for kind in ('cached','online')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['worker','supervise'])
    parser.add_argument('--gpu',type=int);parser.add_argument('--name');parser.add_argument('--source',choices=['cached','online'])
    args=parser.parse_args()
    if args.stage=='worker':worker(args.name,args.source)
    else:supervise(args.gpu)

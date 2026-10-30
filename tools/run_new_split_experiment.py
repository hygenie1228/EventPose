"""Registered, deadline-bounded orchestration for the independent new split."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.goal_runtime_control import identity,register_job,write

OUT=ROOT/'exp/261026/rgb_event_new_split_24h'
GOAL=ROOT/'_agents/research/261026/new_split_agent_goal.json'


def read(path):
    return json.loads(path.read_text())


def environment():
    return dict(os.environ,EVENTPOSE_EVENT_OUT=str(OUT),EVENTPOSE_PREPARATION_OUT=str(OUT),
        EVENTPOSE_RGB_CACHE=str(OUT/'work'),EVENTPOSE_LOSSLESS_RGB_CACHE='1',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')


def conditions():
    goal=read(GOAL)
    sha=hashlib.sha256((OUT/'split_snapshot.json').read_bytes()).hexdigest()
    common=dict(experiment='rgb_event_new_split_24h',seed=261026,hours=3,
        deadline_unix=goal['deadline_unix'],last_blocks=8,head_lr=1e-5,backbone_lr=1e-6,
        batch_size=16,workers=2,pa_weight=100,root_weight=25,validation_minutes=45,
        backbone_warmup_seconds=1800,warmup_steps=100,augment=True,minimum_coverage=.95,
        output_root=str(OUT),rgb_cache_root=str(OUT/'work'),split_sha256=sha)
    values=[dict(common,name='rgb_original',mode='rgb',original=True),
            dict(common,name='rgb_baseline',mode='rgb')]
    for encoding in ('count33','bins100','surface20','visualize_gray'):
        for mode in ('event','fusion'):
            values.append(dict(common,name=mode+'_'+encoding,mode=mode,encoding=encoding))
    for value in values:
        path=OUT/'configs'/(value['name']+'.json')
        if not path.exists():write(path,value)
        else:
            saved=read(path)
            assert all(saved[key]==value[key] for key in value)
    write(OUT/'experiment_manifest.json',dict(conditions=values,
        split_sha256=sha,trial_hours=3,old_finetuned_weights_used=False,
        test_selection_forbidden=True,created_unix=time.time()))
    return values


def live(job):
    now=identity(job['pid'])
    return now is not None and all(now[key]==job[key] for key in ('start_ticks','command','pgid'))


def main():
    lock_root=ROOT/'.experiment_storage/rgb_event_new_split_24h/locks'
    lock_root.mkdir(parents=True,exist_ok=True)
    lock=(lock_root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    goal=read(GOAL);configs=conditions()
    state_path=OUT/'pipeline_state.json'
    state=read(state_path) if state_path.exists() else dict(jobs=[],history=[],failed=[])
    children={};handles={};stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    def launch(role,command,gpu=None,uuid=None):
        env=environment()
        if gpu is not None:env.update(CUDA_VISIBLE_DEVICES=uuid,EVENTPOSE_PHYSICAL_GPU=str(gpu))
        else:env['CUDA_VISIBLE_DEVICES']=''
        log=(OUT/'logs'/(role.replace(':','_')+'.log')).open('a',buffering=1)
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=register_job(goal,process.pid,role)
        record.update(role=role,gpu=gpu,gpu_uuid=uuid,launched_unix=time.time())
        state['jobs'].append(record);children[process.pid]=process;handles[process.pid]=log
        print(json.dumps(dict(event='launch',**record)),flush=True)
    def train_done(name):
        path=OUT/'runs'/name/'complete.json'
        return path.exists() and read(path)['status']=='complete' and read(path)['successful_steps']>0
    def test_done(name):
        path=OUT/'test/tasks.json'
        return path.exists() and all((OUT/'results'/name/t['session']/'complete.json').exists() for t in read(path))
    python=str(ROOT/'.venv-humanpose/bin/python')
    while time.time()<goal['deadline_unix']-300 and not stopping[0]:
        for job in list(state['jobs']):
            process=children.get(job['pid'])
            code=process.poll() if process is not None else None
            if (process is not None and code is None) or (process is None and live(job)):continue
            state['history'].append(dict(job,exit_code=code,ended_unix=time.time()))
            if code not in (None,0):state['failed'].append(job['role'])
            state['jobs'].remove(job)
            if job['pid'] in handles:handles.pop(job['pid']).close()
        running={j['role'] for j in state['jobs']}
        attempted=running|{j['role'] for j in state['history']}
        # Only explicit, diagnosed failures can be retried by changing this list.
        retry=OUT/'retry_requests.json'
        if retry.exists():
            requested=set(read(retry)['roles'])
            state['failed']=[r for r in state['failed'] if r not in requested]
            state['history']=[j for j in state['history'] if j['role'] not in requested]
            attempted-=requested
            retry.rename(OUT/('retry_consumed_'+str(time.time_ns())+'.json'))
        test_ready=(OUT/'test/preparation_complete.json').exists()
        rgb_ready=all((OUT/('controlled_'+r+'_index.json')).exists() for r in ('training','validation'))
        prep_ready=(OUT/'train_cache_available_complete.json').exists()
        if prep_ready and test_ready and 'events' not in attempted:
            launch('events',[python,'-u',str(ROOT/'tools/prepare_new_split_experiment.py'),'events'])
        plan=read(OUT/'event_cache_plan.json') if (OUT/'event_cache_plan.json').exists() else None
        train_cache_ready=bool(plan) and all((OUT/'event_cache'/(s['session']+'.h5')).exists()
            for s in plan['sessions'] if s['role']!='test')
        test_cache_ready=bool(plan) and all((OUT/'event_cache'/(s['session']+'.h5')).exists()
            for s in plan['sessions'] if s['role']=='test')
        for gpu in (2,4,6,5):
            if any(j['gpu']==gpu for j in state['jobs']):continue
            probe=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if probe.returncode or probe.stdout.strip():continue
            probe=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            if probe.returncode:continue
            uuid=probe.stdout.strip()
            assert uuid.startswith('GPU-')
            chosen=None
            for config in configs[1:]:
                role='train:'+config['name']
                ready=rgb_ready and (config['mode']=='rgb' or train_cache_ready)
                if ready and role not in attempted and time.time()+3*3600+4*3600<goal['deadline_unix']:
                    chosen=('train',config);break
            if chosen is None and test_ready:
                for config in configs:
                    role='test:'+config['name']
                    ready=config.get('original') or train_done(config['name'])
                    if ready and (config['mode']=='rgb' or test_cache_ready) and role not in attempted and not test_done(config['name']):
                        chosen=('test',config);break
            if chosen is None:continue
            kind,config=chosen;path=OUT/'configs'/(config['name']+'.json')
            if kind=='train':
                cfg=read(path);cfg['trial_started_unix']=time.time();write(path,cfg)
                command=[python,'-u',str(ROOT/'scripts/train_multihmr.py'),'--config',str(path)]
            else:
                command=[python,'-u',str(ROOT/'scripts/evaluate_multihmr.py'),'worker','--event-config',str(path),'--rank','0','--workers','1']
            role=kind+':'+config['name'];launch(role,command,gpu,uuid);attempted.add(role)
        state.update(updated_unix=time.time(),rgb_ready=rgb_ready,train_event_ready=train_cache_ready,
            test_event_ready=test_cache_ready,training_complete=[c['name'] for c in configs[1:] if train_done(c['name'])],
            test_complete=[c['name'] for c in configs if test_done(c['name'])],deadline_unix=goal['deadline_unix'])
        write(state_path,state)
        if len(state['training_complete'])==9 and len(state['test_complete'])==10 and not state['jobs']:
            write(OUT/'training_test_pipeline_complete.json',dict(completed_unix=time.time(),
                final_statistics_speed_visualization_audit_required=True));return
        time.sleep(15)
    write(state_path,dict(state,status='deadline_or_stop',stopped_unix=time.time()))


if __name__=='__main__':main()

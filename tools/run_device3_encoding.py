"""Identity-registered, deadline-bounded encoding trial and full-Test scheduler."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from device3_encoding_scope import ROOT,GOAL,TRAIN_NAMES,EVAL_NAMES
from goal_runtime_control import write,register_job,identity
def read(p):return json.loads(Path(p).read_text())

def main():
    g=read(GOAL);out=Path(g['output_root']);local=ROOT/'.experiment_storage/device3_encoding_12h';local.mkdir(parents=True,exist_ok=True)
    lock=(local/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    register_job(g,os.getpid(),'encoding_scheduler');statepath=out/'pipeline_state.json'
    state=read(statepath) if statepath.exists() else dict(jobs=[],history=[],failures=[])
    state.pop('status',None)
    children={};stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    python=str(ROOT/'.venv-humanpose/bin/python')
    env=dict(os.environ,EVENTPOSE_MULTISENSOR_OUT=str(out),EVENTPOSE_EVENT_OUT=str(out),EVENTPOSE_GOAL_CONFIG=str(GOAL),
             CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    def launch(role,cmd,gpu=None):
        e=dict(env)
        if gpu:e.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],EVENTPOSE_PHYSICAL_GPU=str(gpu['physical_gpu']))
        with (out/'logs'/(role.replace(':','_')+'.log')).open('a') as log:
            p=subprocess.Popen([python,'-u']+cmd,env=e,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        r=dict(register_job(g,p.pid,role),gpu=gpu,launched_unix=time.time());children[p.pid]=p;state['jobs'].append(r)
        print(json.dumps(dict(event='launch',**r)),flush=True)
    def complete(n):
        p=out/'runs'/n/'complete.json'
        return p.exists() and read(p)['status'] in ('complete','pilot_complete') and read(p)['successful_steps']>0
    def tested(n):return all((out/'results'/n/t['session']/'complete.json').exists() for t in read(out/'test/tasks.json'))
    test_workers=int(g.get('test_workers',4))
    def shard_done(n,rank):
        tasks=read(out/'test/tasks.json');parents=sorted({t['parent_session'] for t in tasks})[rank::test_workers]
        return all((out/'results'/n/t['session']/'complete.json').exists() for t in tasks if t['parent_session'] in parents)
    def cache_ready(cfg):
        manifest=read(out/'manifest.json');e=cfg['encoding']
        return all((out/'encoding_cache'/e/(n+'.h5')).exists() for n in manifest['training_sessions']+manifest['validation_sessions'])
    import pynvml as nv
    nv.nvmlInit()
    while time.time()<g['deadline_unix']-120 and not stop[0]:
        for j in list(state['jobs']):
            p=children.get(j['pid']);code=p.poll() if p else None
            if (p and code is None) or (not p and identity(j['pid'])=={k:j[k] for k in ('pgid','start_ticks','command')}):continue
            state['history'].append(dict(j,exit_code=code,ended_unix=time.time()));state['jobs'].remove(j)
            if code not in (0,None):state['failures'].append(dict(role=j['role'],exit_code=code))
            elif p is None:
                parts=j['role'].split(':');kind,name=parts[:2]
                finished=(complete(('pilot_' if kind=='pilot' else '')+name) if kind in ('pilot','train') else
                          shard_done(name,int(parts[2][4:])) if len(parts)>2 else tested(name))
                if not finished:state['failures'].append(dict(role=j['role'],exit_code='adopted_worker_ended_without_completion'))
        attempted={j['role'] for j in state['jobs']+state['history']}
        # Any recovery is an explicit reviewed request; never repeat trials silently.
        retry=out/'retry_requests.json'
        if retry.exists():
            roles=set(read(retry)['roles']);assert not roles&{j['role'] for j in state['jobs']}
            state['history']=[j for j in state['history'] if j['role'] not in roles]
            state['failures']=[j for j in state['failures'] if j['role'] not in roles];attempted-=roles
            retry.rename(out/('retry_consumed_'+str(time.time_ns())+'.json'))
        ready=(out/'source_reuse_audit.json').exists() and (out/'encoding_cpu_preflight.json').exists()
        if ready:
            configs={n:read(out/'configs'/(n+'.json')) for n in EVAL_NAMES}
            probe=read(out/'gpu_preflight.json')
            for gpu in probe['records']:
                if gpu['status']!='passed' or gpu['physical_gpu'] not in g['allowed_gpus']:continue
                if any(j.get('gpu') and j['gpu']['uuid']==gpu['uuid'] for j in state['jobs']):continue
                try:
                    h=nv.nvmlDeviceGetHandleByUUID(gpu['uuid']);assert nv.nvmlDeviceGetMinorNumber(h)==gpu['physical_gpu']
                    if nv.nvmlDeviceGetComputeRunningProcesses(h):continue
                except nv.NVMLError:continue
                choice=None;train_choice=None
                for n in TRAIN_NAMES:
                    if not cache_ready(configs[n]):continue
                    pilot='pilot_'+n
                    if not complete(pilot):
                        if 'pilot:'+n not in attempted:train_choice=('pilot',n);break
                    elif 'train:'+n not in attempted and not complete(n) and time.time()+(configs[n]['hours']+.75)*3600<g['deadline_unix']:
                        r=read(out/'runs'/pilot/'complete.json');assert r['step']==8 and r['successful_steps']>=4
                        diag=[json.loads(s) for s in (out/'runs'/pilot/'pilot_gradient_diagnostics.jsonl').read_text().splitlines()]
                        if configs[n]['auxiliary_sensor']:assert any((d.get('fusion_encoder_gradient_l2') or 0)>0 for d in diag)
                        if configs[n]['mode']=='event':assert any((d.get('adapter_gradient_l2') or 0)>0 for d in diag)
                        train_choice=('train',n);break
                # Keep up to three training lanes moving while one GPU can
                # continuously evaluate. This avoids adding a full Test delay
                # before every three-hour trial on the critical deadline path.
                active_training=sum(j['role'].startswith(('train:','pilot:')) for j in state['jobs'])
                if train_choice and active_training<3:choice=train_choice
                if choice is None:
                    priority_path=out/'evaluation_priority.json'
                    priority=read(priority_path).get('conditions',[]) if priority_path.exists() else []
                    assert len(priority)==len(set(priority)) and set(priority)<=set(TRAIN_NAMES)
                    evaluation_order=priority+[n for n in TRAIN_NAMES if n not in priority]
                    for n in evaluation_order:
                        if not complete(n) or 'test:'+n in attempted or tested(n):continue
                        for rank in range(test_workers):
                            role='test:'+n+':rank'+str(rank)
                            if role not in attempted and not shard_done(n,rank):choice=('test',n,rank);break
                        if choice:break
                    if choice is None and 'test:pre_rgb3' not in attempted:choice=('test','pre_rgb3')
                # Do not waste an idle fourth GPU if no Test work is ready.
                if choice is None:choice=train_choice
                if choice is None:continue
                kind,n=choice[:2];cfg=configs[n];path=out/'configs'/(n+'.json');role=kind+':'+n
                if kind=='test':
                    if not all((out/'encoding_cache'/cfg['encoding']/(s+'.h5')).exists() for s in read(out/'manifest.json')['test_sessions']):continue
                    rank=choice[2] if len(choice)>2 else 0;workers=test_workers if len(choice)>2 else 1
                    if len(choice)>2:role+=':rank'+str(rank)
                    cmd=['scripts/evaluate_multihmr.py','worker','--event-config',str(path),'--rank',str(rank),'--workers',str(workers)]
                else:
                    if kind=='pilot':
                        cfg=dict(cfg,name='pilot_'+n,pilot_steps=8,pilot_train_limit=64,pilot_validation_limit=32,workers=0,trial_started_unix=time.time())
                        path=out/'configs'/('pilot_'+n+'.json');write(path,cfg)
                    elif 'trial_started_unix' not in cfg:cfg['trial_started_unix']=time.time();write(path,cfg)
                    cmd=['scripts/train_multihmr.py','--config',str(path)]
                launch(role,cmd,gpu);attempted.add(role)
            state.update(training_complete=[n for n in TRAIN_NAMES if complete(n)],test_complete=[n for n in EVAL_NAMES if tested(n)])
            if len(state['test_complete'])==len(EVAL_NAMES) and not state['jobs']:
                assert len(state['training_complete'])==len(TRAIN_NAMES)
                write(out/'all_training_test_complete.json',dict(completed_unix=time.time(),final_audit_required=True))
                write(statepath,state);return
        state.update(updated_unix=time.time(),ready=ready,deadline_unix=g['deadline_unix']);write(statepath,state)
        time.sleep(20)
    write(statepath,dict(state,status='deadline_or_stop',stopped_unix=time.time()))

if __name__=='__main__':main()

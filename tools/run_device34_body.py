"""Own-job-only scheduler: six primary trials/table, then six seed replications."""
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
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.goal_runtime_control import identity,register_job,write
from multihmr.multisensor_model import CONDITIONS
OUT=ROOT/'exp/261026/device34_body_28h'
GOAL=ROOT/'_agents/research/261026/device34_body_goal_28h.json'


def read(path):return json.loads(path.read_text())


def main():
    local=ROOT/'.experiment_storage/device34_body_28h';local.mkdir(parents=True,exist_ok=True)
    lock=(local/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    goal=read(GOAL);python=str(ROOT/'.venv-humanpose/bin/python')
    state_path=OUT/'pipeline_state.json'
    state=read(state_path) if state_path.exists() else dict(jobs=[],history=[],failed=[])
    children={};logs={};stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    env=dict(os.environ,EVENTPOSE_MULTISENSOR_OUT=str(OUT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    def live(job):
        now=identity(job['pid'])
        return now is not None and all(now[k]==job[k] for k in ('pgid','start_ticks','command'))
    def launch(role,command,gpu=None,gpu_info=None):
        e=dict(env,CUDA_VISIBLE_DEVICES='')
        if gpu is not None:e.update(CUDA_VISIBLE_DEVICES=gpu_info['uuid'],EVENTPOSE_PHYSICAL_GPU=str(gpu))
        log=(OUT/'logs'/(role.replace(':','_')+'.log')).open('a',buffering=1)
        proc=subprocess.Popen(command,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job=register_job(goal,proc.pid,role)
        job.update(role=role,gpu=gpu,gpu_info=gpu_info,launched_unix=time.time())
        state['jobs'].append(job);children[proc.pid]=proc;logs[proc.pid]=log
        print(json.dumps(dict(event='launch',**job)),flush=True)
    def train_done(name):
        p=OUT/'runs'/name/'complete.json'
        return p.exists() and read(p)['status']=='complete' and read(p)['successful_steps']>0 and Path(read(p)['checkpoint']).is_file()
    def test_done(name):
        p=OUT/'test/tasks.json'
        return p.exists() and bool(read(p)) and all((OUT/'results'/name/t['session']/'complete.json').exists() for t in read(p))
    while time.time()<goal['deadline_unix']-300 and not stopping[0]:
        for job in list(state['jobs']):
            proc=children.get(job['pid']);code=proc.poll() if proc else None
            if (proc is not None and code is None) or (proc is None and live(job)):continue
            state['history'].append(dict(job,exit_code=code,ended_unix=time.time()))
            if code not in (None,0):state['failed'].append(job['role'])
            state['jobs'].remove(job)
            if job['pid'] in logs:logs.pop(job['pid']).close()
        attempted={j['role'] for j in state['jobs']+state['history']}
        retry=OUT/'retry_requests.json'
        if retry.exists():
            requested=set(read(retry)['roles'])
            state['failed']=[r for r in state['failed'] if r not in requested]
            state['history']=[j for j in state['history'] if j['role'] not in requested]
            attempted-=requested;retry.rename(OUT/('retry_consumed_'+str(time.time_ns())+'.json'))
        audited=(OUT/'source_audit_complete.json').exists()
        configs=[]
        if audited:
            manifest=read(OUT/'manifest.json')
            for seed in (261026,261027):
                for condition,(primary,auxiliary) in CONDITIONS.items():
                    name=condition if seed==261026 else condition+'_seed'+str(seed)
                    cfg=dict(experiment='device34_body_28h',condition=condition,name=name,seed=seed,hours=3,
                        deadline_unix=goal['deadline_unix'],output_root=str(OUT),split_sha256=manifest['split_sha256'],
                        mode='event' if primary.startswith('event') else 'rgb',encoding='count33',last_blocks=8,
                        head_lr=1e-5,backbone_lr=1e-6,batch_size=16,workers=2,pa_weight=100,root_weight=25,
                        validation_minutes=45,backbone_warmup_seconds=1800,warmup_steps=100,augment=True,
                        augmentation='same seeded multiplicative intensity gain U[.9,1.1] for active modalities; no spatial warp',
                        minimum_coverage=.95,primary_sensor=primary,auxiliary_sensor=auxiliary,
                        initialization='original released multiHMR_896_L.pt; independent seed trial')
                    path=OUT/'configs'/(name+'.json')
                    if not path.exists():write(path,cfg)
                    configs.append(cfg)
            write(OUT/'experiment_manifest.json',dict(primary=configs[:6],replications=configs[6:],
                split_sha256=manifest['split_sha256'],primary_table_first=True))
        primary_complete=bool(configs) and all(test_done(c['name']) for c in configs[:6])
        if primary_complete and 'primary_report' not in attempted:
            launch('primary_report',[python,'-u',str(ROOT/'tools/finalize_device34_body.py'),'primary'])
        first_report=(OUT/'primary_table_verified.json').exists()
        for gpu in range(2,7):
            if any(j['gpu']==gpu for j in state['jobs']):continue
            probe=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if probe.returncode or probe.stdout.strip():continue
            probe=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            if probe.returncode:continue
            parts=[s.strip() for s in probe.stdout.strip().split(',')]
            info=dict(index=int(parts[0]),uuid=parts[1],pci_bus_id=parts[2],name=parts[3])
            assert info['index']==gpu and info['uuid'].startswith('GPU-')
            if audited and 'cache' not in attempted:
                launch('cache',[python,'-u',str(ROOT/'tools/prepare_device34_body.py'),'cache'],gpu,info)
                attempted.add('cache');continue
            chosen=None
            eligible=configs if first_report else configs[:6]
            if (OUT/'training_cache_complete.json').exists():
                for cfg in eligible:
                    role='train:'+cfg['name']
                    if role not in attempted and time.time()+7*3600<goal['deadline_unix']:
                        chosen=('train',cfg);break
            if chosen is None and (OUT/'cache_complete.json').exists():
                for cfg in eligible:
                    role='test:'+cfg['name']
                    if train_done(cfg['name']) and role not in attempted and not test_done(cfg['name']):
                        chosen=('test',cfg);break
            if chosen is None:continue
            kind,cfg=chosen;path=OUT/'configs'/(cfg['name']+'.json')
            if kind=='train':
                saved=read(path);saved['trial_started_unix']=time.time();write(path,saved)
                cmd=[python,'-u',str(ROOT/'scripts/train_multihmr.py'),'--config',str(path)]
            else:cmd=[python,'-u',str(ROOT/'scripts/evaluate_multihmr.py'),'worker','--event-config',str(path),'--rank','0','--workers','1']
            role=kind+':'+cfg['name'];launch(role,cmd,gpu,info);attempted.add(role)
        state.update(updated_unix=time.time(),source_audit_complete=audited,primary_table_verified=first_report,
            training_complete=[c['name'] for c in configs if train_done(c['name'])],
            test_complete=[c['name'] for c in configs if test_done(c['name'])],deadline_unix=goal['deadline_unix'])
        write(state_path,state)
        if configs and all(test_done(c['name']) for c in configs) and not state['jobs']:
            write(OUT/'all_training_test_complete.json',dict(completed_unix=time.time(),final_audit_required=True));return
        time.sleep(15)
    write(state_path,dict(state,status='deadline_or_stop',stopped_unix=time.time()))


if __name__=='__main__':main()

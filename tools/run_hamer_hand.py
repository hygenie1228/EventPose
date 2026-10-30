"""Single-GPU0 scheduler; ownership records, no body-job intervention."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261028/hamer_hand'
NAMES=['rgb3','rgb3_event4_count33','rgb3_event4_bins100','rgb3_event4_surface20']
PYTHON=str(ROOT/'.venv-humanpose/bin/python')


def read(path):return json.loads(Path(path).read_text())


def register(pid,role):
    from prepare_hamer_hand import write
    p=Path('/proc')/str(pid)
    row=dict(pid=pid,pgid=os.getpgid(pid),start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19],
        command=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),role=role,started_unix=time.time())
    path=OUT/'managed_jobs.json'
    with (ROOT/'_agents/research/261028/hamer_jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=read(path) if path.exists() else []
        rows=[r for r in rows if not (r['pid']==pid and r['start_ticks']==row['start_ticks'])]
        rows.append(row);write(path,rows)
    return row


def main():
    from prepare_hamer_hand import write
    import pynvml as nv
    (OUT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'_agents/research/261028/hamer_scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);register(os.getpid(),'hand_scheduler')
    nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByIndex(0)
    assert nv.nvmlDeviceGetMinorNumber(handle)==0
    uuid=nv.nvmlDeviceGetUUID(handle);bus=nv.nvmlDeviceGetPciInfo(handle).busId
    if isinstance(uuid,bytes):uuid=uuid.decode()
    if isinstance(bus,bytes):bus=bus.decode()
    write(OUT/'gpu_allocation.json',dict(physical_gpu=0,uuid=uuid,bus_id=bus,allowed_gpus=[0],checked_unix=time.time()))
    state=dict(status='waiting_for_pretrained_and_validated_crops',gpu=0,gpu_uuid=uuid,history=[],started_unix=time.time())
    def launch(role,args,gpu=False):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
            CUDA_VISIBLE_DEVICES=uuid if gpu else '',EVENTPOSE_PHYSICAL_GPU='0',
            HF_HOME=str(OUT/'tmp/huggingface'),TORCH_HOME=str(OUT/'tmp/torch'))
        with (OUT/'logs'/(role+'.log')).open('a') as f:
            p=subprocess.Popen([PYTHON,'-u',*args],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,
                start_new_session=True,env=env)
        row=register(p.pid,role);state.update(status=role,active=row)
        write(OUT/'pipeline_state.json',state);return p,row
    setup=None
    if not (OUT/'pretrained_manifest.json').exists():setup=launch('pretrained_setup',['tools/setup_hamer_hand.py'])
    active=None;last_report=0
    def pilot_index_ready():
        import h5py
        m=read(OUT/'manifest.json')
        for role in ('training','validation'):
            path=OUT/('pilot_'+role+'_index.json')
            if path.exists():continue
            available=[n for n in m[role+'_sessions'] if (OUT/'cache'/(n+'.h5')).exists()]
            if not available:return False
            name=available[0]
            with h5py.File(OUT/'cache'/(name+'.h5'),'r',locking=False) as f:
                assert f.attrs['complete'] and f.attrs['role']==role and f.attrs['cache_version']==2
                records=[dict(session=name,row=i,frame=int(frame)) for i,frame in enumerate(f['frame'][:])]
            write(path,records)
        return True
    while True:
        now=time.time()
        if (OUT/'preparation_failure.json').exists():
            state.update(status='failed',failure='CPU preparation: '+read(OUT/'preparation_failure.json')['error'])
            write(OUT/'pipeline_state.json',state)
            subprocess.run([PYTHON,'tools/report_hamer_hand.py'],cwd=ROOT,check=True);return
        if setup:
            code=setup[0].poll()
            if code is not None:
                state['history'].append(dict(setup[1],exit_code=code));setup=None
                if code:state.update(status='failed',failure='pretrained_setup');write(OUT/'pipeline_state.json',state);return
        if active:
            code=active[0].poll()
            if code is not None:
                row=active[1];state['history'].append(dict(row,exit_code=code,ended_unix=now));active=None;state.pop('active',None)
                if code:
                    state.update(status='failed',failure=row['role']);write(OUT/'pipeline_state.json',state)
                    subprocess.run([PYTHON,'tools/report_hamer_hand.py'],cwd=ROOT,check=True);return
                role=row['role']
                if role.startswith('train_'):
                    result=read(OUT/'runs'/role[len('train_'):] /'complete.json')
                    if result['status']!='complete':
                        state.update(status='failed',failure='Training interrupted: '+role)
                        write(OUT/'pipeline_state.json',state)
                        subprocess.run([PYTHON,'tools/report_hamer_hand.py'],cwd=ROOT,check=True);return
                if role.startswith('pilot_'):
                    r=read(OUT/'runs'/role/'complete.json');assert r['successful_steps']>=4 and r['step']==8
                    cfg=read(OUT/'configs'/(role+'.json'))
                    if cfg['condition']!='rgb3':
                        diag=[json.loads(s) for s in (OUT/'runs'/role/'training.jsonl').read_text().splitlines()]
                        assert any(d['fusion_encoder_gradient_l2']>0 for d in diag),'Event encoder received no gradient'
                    write(OUT/'pilot_checks'/(role+'.json'),dict(status='passed',step=r['step'],successful_steps=r['successful_steps']))
                subprocess.run([PYTHON,'tools/report_hamer_hand.py'],cwd=ROOT,check=True)
        if now-last_report>=1800:
            # Long training/evaluation checks and report refreshes are bounded to 30min.
            subprocess.run([PYTHON,'tools/report_hamer_hand.py'],cwd=ROOT,check=True);last_report=now
        if active is None:
            if not (OUT/'pretrained_manifest.json').exists():state['status']='waiting_for_pretrained'
            elif not (OUT/'cpu_preflight.json').exists():state['status']='waiting_for_crop_verification'
            elif nv.nvmlDeviceGetComputeRunningProcesses(handle):state['status']='waiting_for_idle_GPU0'
            else:
                args=None;role=None
                if not (OUT/'gpu_preflight.json').exists():
                    role='gpu_preflight';args=['tools/check_hamer_hand.py','gpu']
                elif not pilot_index_ready():
                    state['status']='waiting_for_first_Train_cache_for_pilot'
                else:
                    training_ready=all((OUT/(r+'_index.json')).exists() for r in ('training','validation'))
                    for name in NAMES:
                        pilot='pilot_'+name
                        if not (OUT/'pilot_checks'/(pilot+'.json')).exists():
                            cfg=dict(read(OUT/'configs'/(name+'.json')),name=pilot,pilot_steps=8,workers=0,
                                pilot_training_index=str(OUT/'pilot_training_index.json'),
                                pilot_validation_index=str(OUT/'pilot_validation_index.json'))
                            path=OUT/'configs'/(pilot+'.json');write(path,cfg)
                            role=pilot;args=['scripts/train_hamer.py','--config',str(path)];break
                    cache_complete=(OUT/'cache_complete.json').exists()
                    if args is None and cache_complete:
                        for name in ['pre_rgb3']+NAMES:
                            trained=(OUT/'runs'/name/'complete.json').exists()
                            if (name=='pre_rgb3' or trained) and not (OUT/'results'/name/'complete.json').exists():
                                cfg=OUT/'configs'/('rgb3.json' if name=='pre_rgb3' else name+'.json')
                                role='test_'+name;args=['scripts/evaluate_hamer.py','--config',str(cfg)]
                                if name=='pre_rgb3':args+=['--pretrained']
                                break
                    if args is None and training_ready:
                        for name in NAMES:
                            if not (OUT/'runs'/name/'complete.json').exists():
                                role='train_'+name;args=['scripts/train_hamer.py','--config',str(OUT/'configs'/(name+'.json'))];break
                    if args is None:
                        if not training_ready:state['status']='waiting_for_complete_Train_caches'
                        elif not cache_complete:state['status']='waiting_for_Test_caches'
                        elif all((OUT/'results'/n/'complete.json').exists() for n in ['pre_rgb3']+NAMES):
                            p,row=launch('final_verification',['tools/verify_hamer_hand.py'])
                            code=p.wait();state['history'].append(dict(row,exit_code=code,ended_unix=time.time()))
                            state.update(status='complete' if code==0 else 'failed',failure=None if code==0 else 'final_verification')
                            state.pop('active',None);write(OUT/'pipeline_state.json',state)
                            subprocess.run([PYTHON,'tools/report_hamer_hand.py'],cwd=ROOT,check=True);return
                if args:active=launch(role,args,gpu=True)
        state['updated_unix']=time.time();write(OUT/'pipeline_state.json',state)
        time.sleep(30)


if __name__=='__main__':main()

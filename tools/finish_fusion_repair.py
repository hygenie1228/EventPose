"""Finish actual holdout ablations, same-GPU timings and full artifact audit."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from goal_runtime_control import register_job, write
from fusion_repair_scope import FUSION_NAMES,TRAIN_NAMES
ROOT=Path(__file__).resolve().parents[1]
GOAL=ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json'


def read(path):return json.loads(path.read_text())


def main():
    goal=read(GOAL);out=Path(goal['output_root'])
    local=ROOT/'.experiment_storage/device34_fusion_repair_15h';local.mkdir(parents=True,exist_ok=True)
    lock=(local/'postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    register_job(goal,os.getpid(),'fusion_repair_postprocess_waiter')
    state_path=out/'postprocess_state.json';python=str(ROOT/'.venv-humanpose/bin/python')
    env=dict(os.environ,EVENTPOSE_MULTISENSOR_OUT=str(out),EVENTPOSE_EVENT_OUT=str(out),
             OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    def launch(role,args,gpu=None):
        e=dict(env)
        if gpu is not None:e.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],EVENTPOSE_PHYSICAL_GPU=str(gpu['physical_gpu']))
        with (out/'logs'/(role.replace(':','_')+'.log')).open('a') as log:
            p=subprocess.Popen([python,'-u']+args,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            register_job(goal,p.pid,role)
            write(state_path,dict(status='running',role=role,pid=p.pid,gpu=gpu,updated_unix=time.time()))
            code=p.wait()
            if code:
                write(state_path,dict(status='failed',role=role,pid=p.pid,exit_code=code,updated_unix=time.time()))
                raise RuntimeError(role+' failed; diagnose before retrying')
    import pynvml as n
    n.nvmlInit()
    def idle_gpu():
        probe=read(out/'gpu_preflight.json');state=read(out/'pipeline_state.json')
        proven=set(state.get('proven_cuda_uuids',[]))
        for gpu in probe['records']:
            if gpu['physical_gpu'] not in goal['allowed_gpus'] or not (gpu['status']=='passed' or gpu['uuid'] in proven):continue
            try:
                h=n.nvmlDeviceGetHandleByUUID(gpu['uuid'])
                assert n.nvmlDeviceGetMinorNumber(h)==gpu['physical_gpu']
                if not n.nvmlDeviceGetComputeRunningProcesses(h):return gpu
            except n.NVMLError:continue
        return None
    while time.time()<goal['deadline_unix']-600:
        if not (out/'all_training_test_complete.json').exists():
            training_ready=all((out/'runs'/name/'complete.json').exists() and
                               read(out/'runs'/name/'complete.json')['status']=='complete'
                               for name in TRAIN_NAMES)
            ablations_ready=all((out/'ablations'/(name+'.json')).exists() and
                               read(out/'ablations'/(name+'.json'))['status']=='passed'
                               for name in FUSION_NAMES)
            if not training_ready or ablations_ready:
                write(state_path,dict(status='waiting_full_Test' if training_ready else 'waiting_training',
                                      updated_unix=time.time()))
                time.sleep(30);continue
        gpu=idle_gpu()
        if gpu is None:
            write(state_path,dict(status='waiting_idle_authorized_GPU',updated_unix=time.time()))
            time.sleep(30);continue
        for name in FUSION_NAMES:
            path=out/'ablations'/(name+'.json')
            if path.exists():assert read(path)['status']=='passed';continue
            assert time.time()<goal['deadline_unix']-300
            launch('fusion_repair_holdout_ablation:'+name,[str(ROOT/'tools/ablate_fusion_repair.py'),name],gpu)
        if not (out/'all_training_test_complete.json').exists():
            write(state_path,dict(status='ablations_complete_waiting_full_Test',updated_unix=time.time()))
            time.sleep(30);continue
        if not (out/'runtime/summary.json').exists():
            assert time.time()<goal['deadline_unix']-600
            launch('fusion_repair_runtime_supervise',[str(ROOT/'tools/benchmark_fusion_repair.py'),'supervise','--gpu',str(gpu['physical_gpu'])])
        launch('fusion_repair_final_raw_report_audit',[str(ROOT/'tools/finalize_fusion_repair.py')])
        write(state_path,dict(status='artifact_audit_finished_agent_review_required',updated_unix=time.time(),goal_complete=False))
        return
    write(state_path,dict(status='deadline_insufficient_time_remaining',updated_unix=time.time(),goal_complete=False))


if __name__=='__main__':main()

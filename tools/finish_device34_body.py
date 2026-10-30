"""Wait for registered trials, run same-GPU timing, then final raw audit/report."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_device34_body import OUT,GOAL,save
from tools.goal_runtime_control import register_job


def main():
    goal=json.loads(GOAL.read_text());status=OUT/'postprocess_state.json'
    while time.time()<goal['deadline_unix']-600:
        if (OUT/'all_training_test_complete.json').exists():break
        time.sleep(30)
    else:return
    chosen=None
    while time.time()<goal['deadline_unix']-600:
        for gpu in range(2,7):
            p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if not p.returncode and not p.stdout.strip():chosen=gpu;break
        if chosen is not None:break
        time.sleep(30)
    if chosen is None:return
    env=dict(os.environ,EVENTPOSE_MULTISENSOR_OUT=str(OUT),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    stages=[('runtime',[sys.executable,'-u',str(ROOT/'tools/benchmark_device34_body.py'),'supervise','--gpu',str(chosen)]),
        ('final_audit_report',[sys.executable,'-u',str(ROOT/'tools/finalize_device34_body.py'),'final'])]
    for stage,command in stages:
        assert time.time()<goal['deadline_unix']-300
        save(status,dict(stage=stage,status='running',updated_unix=time.time()))
        with (OUT/'logs'/('postprocess_'+stage+'.log')).open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            register_job(goal,process.pid,'device34_postprocess:'+stage)
            code=process.wait()
        if code:
            save(status,dict(stage=stage,status='failed',exit_code=code,updated_unix=time.time()));return
    save(status,dict(status='artifact_audit_finished_agent_completion_review_required',updated_unix=time.time(),goal_complete=False))


if __name__=='__main__':main()

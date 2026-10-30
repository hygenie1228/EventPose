"""Prove that deadline shutdown preserves a trainer before its loader workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from goal_runtime_control import register_job,stop_registered,write


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_shutdown_guard_test')
    out=Path(goal['output_root']);folder=out/'shutdown_guard_test';folder.mkdir(exist_ok=True)
    worker="""import signal,time,sys
from pathlib import Path
p=Path(sys.argv[1]);p.with_suffix('.worker_ready').write_text('ready')
def stop(*_):
 p.with_suffix('.worker_signal').write_text('terminated');sys.exit(0)
signal.signal(signal.SIGTERM,stop)
while True:time.sleep(.05)
"""
    parent="""import signal,time,sys,subprocess,json
from pathlib import Path
p=Path(sys.argv[1]);child=subprocess.Popen([sys.executable,'-c',sys.argv[2],str(p)])
while not p.with_suffix('.worker_ready').exists():time.sleep(.01)
def stop(*_):
 time.sleep(.2)
 alive=child.poll() is None and not p.with_suffix('.worker_signal').exists()
 p.write_text(json.dumps({'worker_alive_during_checkpoint':alive,'checkpoint_saved':True}))
 child.terminate();child.wait(timeout=3);sys.exit(0)
signal.signal(signal.SIGTERM,stop);p.with_suffix('.ready').write_text('ready')
while True:time.sleep(.05)
"""
    marker=folder/(str(time.time_ns())+'.json')
    process=subprocess.Popen([sys.executable,'-u','-c',parent,str(marker),worker],start_new_session=True)
    register_job(goal,process.pid,'train:shutdown_guard_synthetic')
    fixture=dict(goal,managed_jobs_file=str(folder/'fixture_jobs.json'))
    write(fixture['managed_jobs_file'],{'jobs':[]});register_job(fixture,process.pid,'train:shutdown_guard_synthetic')
    until=time.monotonic()+5
    while not marker.with_suffix('.ready').exists() and time.monotonic()<until:time.sleep(.02)
    assert marker.with_suffix('.ready').exists()
    stopped=stop_registered(fixture);assert process.wait(timeout=3)==0
    record=json.loads(marker.read_text());assert record['worker_alive_during_checkpoint'] and record['checkpoint_saved']
    result=dict(status='passed',parent_only_SIGTERM_before_worker_shutdown=True,
                synthetic_checkpoint=str(marker),stopped=stopped,actual_training_touched=False)
    write(out/'trainer_shutdown_guard_verified.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

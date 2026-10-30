"""Resume checkpointed training only after the user's immediate Test finishes."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json
import finalize_track2_experiments as finish
import run_track2_experiments as runtime
runtime.REGISTRY_NAME='priority_resume_jobs.json'


def read(path):return json.loads(Path(path).read_text())


def main(pid):
    with (OUTPUT/'priority_resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        identity=finish.process_identity(pid)
        save_json(OUTPUT/'priority_resume_status.json',dict(status='waiting_quick_test',pid=os.getpid(),quick_pid=pid,
            started_unix=time.time(),quick_start_ticks=identity))
        while identity is not None and finish.process_identity(pid)==identity:
            if time.time()>=deadline:raise TimeoutError('Global deadline while quick Test is live')
            time.sleep(min(30,max(1,deadline-time.time())))
        result=read(OUTPUT/'quick_comparison.json')
        assert all(v['sequences']==10 for v in result['models'].values())
        finish.report()
        save_json(OUTPUT/'priority_resume_status.json',dict(status='resuming_checkpointed_refits',pid=os.getpid(),updated_unix=time.time()))
        runtime.run([str(runtime.PYTHON),'-u',str(ROOT/'tools/finalize_track2_experiments.py'),
            '--wait-training-pid','56652'],[],'resume_first_pipeline_after_quick',deadline)
        previous=read(OUTPUT/'finalizer.json')
        assert previous['status']=='artifacts_ready_for_final_audit'
        save_json(OUTPUT/'priority_resume_status.json',dict(status='running_cd_h_followup',pid=os.getpid(),updated_unix=time.time()))
        runtime.run([str(runtime.PYTHON),'-u',str(ROOT/'tools/run_track2_cd_wave2.py'),
            '--wait-pipeline-pid',str(previous['pid'])],[],'cd_h_followup_after_quick',deadline)
        save_json(OUTPUT/'priority_resume_status.json',dict(status='artifacts_ready_for_final_audit',pid=os.getpid(),completed_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-quick-pid',type=int,required=True)
    args=p.parse_args()
    try:main(args.wait_quick_pid)
    except BaseException as error:
        save_json(OUTPUT/'priority_resume_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();finish.report();raise

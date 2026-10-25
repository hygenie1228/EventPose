"""Deadline-bounded postprocessing, without claiming scientific completion."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.goal_runtime_control import register_job, write

OUT = ROOT / 'exp/261024/rgb_event_36h'


def main():
    local = ROOT / '.experiment_storage/rgb_event_36h'
    local.mkdir(parents=True, exist_ok=True)
    lock = (local / 'postprocess.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    goal = json.loads((ROOT / '_agents/research/261024/rgb_event_goal_36h.json').read_text())
    status = OUT / 'postprocess_status.json'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')

    def stage(name, command):
        assert time.time() < goal['deadline_unix'] - 300
        write(status, dict(status='running', stage=name, updated_unix=time.time(), pid=os.getpid()))
        with (OUT / 'logs' / ('postprocess_' + name + '.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            register_job(goal, process.pid, 'event_postprocess:' + name)
            code = process.wait()
        if code:
            write(status, dict(status='failed', stage=name, exit_code=code, updated_unix=time.time()))
            raise RuntimeError(name + ' failed; inspect preserved log, do not blindly restart')

    write(status, dict(status='waiting_for_primary_training_and_full_test',
                       pid=os.getpid(), updated_unix=time.time(), deadline_unix=goal['deadline_unix']))
    while time.time() < goal['deadline_unix'] - 300:
        if (OUT / 'training_test_pipeline_complete.json').exists():
            break
        time.sleep(30)
    else:
        write(status, dict(status='deadline_before_postprocess', updated_unix=time.time()))
        return
    # The producer creates the marker only after its jobs have terminated.
    # Independent external work may still occupy GPUs; wait without touching it.
    gpu = None
    while time.time() < goal['deadline_unix'] - 300:
        for candidate in range(2, 7):
            active = subprocess.check_output(['nvidia-smi', '-i', str(candidate),
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if not active:
                gpu = candidate
                break
        if gpu is not None:
            break
        time.sleep(30)
    if gpu is None:
        write(status, dict(status='deadline_no_free_gpu', updated_unix=time.time()))
        return
    stage('runtime', [sys.executable, '-u', str(ROOT / 'tools/benchmark_event_pipeline.py'),
                      'supervise', '--gpu', str(gpu)])
    stage('comparison', [sys.executable, '-u', '-m', 'utils.compare_event_methods'])
    stage('report', [sys.executable, '-u', '-m', 'utils.event_report'])
    write(status, dict(status='artifacts_generated_final_audit_required', updated_unix=time.time(),
                       gpu=gpu, goal_complete=False,
                       remaining=['raw recomputation and provenance audit', 'cache cost and capacity',
                                  'seed replication feasibility and evidence',
                                  'one verified recommendation and report number audit']))


if __name__ == '__main__':
    main()

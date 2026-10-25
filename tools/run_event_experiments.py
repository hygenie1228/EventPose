"""One bounded scheduler for controlled RGB/event training and full Test jobs."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.goal_runtime_control import identity, register_job, write

OUT = ROOT / 'exp/261024/rgb_event_36h'


def read(path):
    return json.loads(Path(path).read_text())


def role_complete(kind, name):
    if kind == 'train':
        path = OUT / 'runs' / name / 'complete.json'
        if not path.exists():
            return False
        value = read(path)
        return value['status'] == 'complete' and value['successful_steps'] > 0 and Path(value['checkpoint']).is_file()
    tasks = read(ROOT / '.experiment_storage/humanpose_gt_segments/tasks.json')
    folder = OUT / 'results' / name
    return len(tasks) == 320 and all((folder / task['session'] / 'complete.json').is_file()
                                     and (folder / task['session'] / 'result.npz').is_file() for task in tasks)


def live(job):
    current = identity(job['pid'])
    return current is not None and all(current[key] == job[key] for key in ('start_ticks', 'command', 'pgid'))


def main():
    lock_path = ROOT / '.experiment_storage/rgb_event_36h/pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    goal = read(ROOT / '_agents/research/261024/rgb_event_goal_36h.json')
    manifest = read(OUT / 'experiment_manifest.json')
    configs = {value['name']: OUT / 'configs' / (value['name'] + '.json') for value in manifest['conditions']}
    order = [name for wave in manifest['suggested_waves'] for name in wave]
    assert len(order) == len(set(order)) == 9
    assert read(OUT / 'algorithm_selection.json')['algorithm'] == 'Multi-HMR'
    config = {name: read(path) for name, path in configs.items()}
    state_path = OUT / 'pipeline_state.json'
    state = read(state_path) if state_path.exists() else dict(jobs=[], history=[], failed=[])
    children = {}; logs = {}; stopping = [False]
    def stop_signal(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop_signal); signal.signal(signal.SIGINT, stop_signal)
    # Adopt the already-running RGB primary; never restart it or its clock.
    rgb_path = OUT / 'primary_rgb_job.json'
    if rgb_path.exists() and not any(job['kind'] == 'train' and job['name'] == 'rgb_baseline' for job in state['jobs']):
        current = read(rgb_path)
        if live(current):
            state['jobs'].append(dict(current, kind='train', name='rgb_baseline'))
    pilot_jobs = []
    for path in [OUT / 'pilot_wave1_jobs.json', OUT / 'pilot_wave2_jobs.json']:
        if path.exists():
            pilot_jobs += read(path)
    plan = read(OUT / 'event_cache_plan.json')
    assert plan['config']['version'] == 2
    train_sessions = {value['session'] for value in plan['sessions'] if value['role'] in ('training', 'validation')}
    test_sessions = {value['session'] for value in plan['sessions'] if value['role'] == 'test'}
    admitted_training = set()
    while time.time() < goal['deadline_unix'] and not stopping[0]:
        for job in list(state['jobs']):
            handle = children.get(job['pid'])
            code = handle.poll() if handle is not None else None
            if (handle is not None and code is None) or (handle is None and live(job)):
                continue
            good = role_complete(job['kind'], job['name']) and (code is None or code == 0)
            state['history'].append(dict(job, ended_unix=time.time(), exit_code=code,
                                         status='complete' if good else 'failed'))
            if not good:
                state['failed'].append(job['kind'] + ':' + job['name'])
            state['jobs'].remove(job)
            if job['pid'] in logs:
                logs.pop(job['pid']).close()
        # A retry is accepted only when an agent has diagnosed a concrete failure
        # and explicitly written the task name; no observation-timeout restarts.
        retry_path = OUT / 'retry_requests.json'
        if retry_path.exists():
            retries = read(retry_path)
            for name in retries.get('tasks', []):
                if name in state['failed']:
                    state['failed'].remove(name)
            retry_path.rename(OUT / ('retry_requests_consumed_' + str(time.time_ns()) + '.json'))
        present = {path.stem for path in (OUT / 'event_cache').glob('*.h5') if not path.name.endswith('.partial.h5')}
        train_ready = train_sessions <= present
        test_ready = test_sessions <= present
        pilot_gate = OUT / 'pilot_validation.json'
        pilots_ready = pilot_gate.exists() and read(pilot_gate).get('status') == 'passed'
        occupied = {job['gpu'] for job in state['jobs']}
        occupied |= {job['gpu'] for job in pilot_jobs if live(job)}
        running = {job['kind'] + ':' + job['name'] for job in state['jobs']}
        for gpu in (2, 3, 4, 5, 6):
            if gpu in occupied:
                continue
            chosen = None
            if train_ready and pilots_ready:
                for name in order:
                    tag = 'train:' + name
                    if tag not in running and tag not in state['failed'] and not role_complete('train', name):
                        duration = config[name]['hours'] * 3600
                        if 'trial_started_unix' in config[name]:
                            duration = max(0, config[name]['trial_started_unix'] + duration - time.time())
                        # Keep at least three hours for full Test, controlled
                        # online/cache runtime, statistics and final verification.
                        if time.time() + duration + 3 * 3600 < goal['deadline_unix']:
                            chosen = ('train', name); break
            if chosen is None:
                for name in ['rgb_original'] + order:
                    tag = 'test:' + name
                    trained = name == 'rgb_original' or role_complete('train', name)
                    cache_ready = config[name]['mode'] == 'rgb' or test_ready
                    if trained and cache_ready and tag not in running and tag not in state['failed'] and not role_complete('test', name):
                        chosen = ('test', name); break
            if chosen is None:
                continue
            kind, name = chosen
            if kind == 'train':
                cfg = read(configs[name])
                if 'trial_started_unix' not in cfg:
                    cfg['trial_started_unix'] = time.time(); write(configs[name], cfg)
                config[name] = cfg
                command = [sys.executable, '-u', str(ROOT / 'scripts/train_multihmr.py'), '--config', str(configs[name])]
                admitted_training.add(name)
            else:
                command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_multihmr.py'), 'worker',
                           '--event-config', str(configs[name]), '--rank', '0', '--workers', '1']
            log_path = OUT / 'logs' / ('pipeline_' + kind + '_' + name + '.log')
            log = log_path.open('a', buffering=1)
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = register_job(goal, process.pid, kind + ':' + name)
            record.update(kind=kind, name=name, gpu=gpu, launched_unix=time.time(),
                          config=str(configs[name]), log=str(log_path))
            state['jobs'].append(record); children[process.pid] = process; logs[process.pid] = log
            running.add(kind + ':' + name); occupied.add(gpu)
            print(json.dumps(dict(event='launch', **record)), flush=True)
        done_training = [name for name in order if role_complete('train', name)]
        done_test = [name for name in configs if role_complete('test', name)]
        state.update(updated_unix=time.time(), training_complete=done_training, test_complete=done_test,
                     train_cache_ready=train_ready, test_cache_ready=test_ready, pilots_verified=pilots_ready,
                     deadline_unix=goal['deadline_unix'])
        write(state_path, state)
        if len(done_training) == 9 and len(done_test) == 10 and not state['jobs']:
            write(OUT / 'training_test_pipeline_complete.json', dict(completed_unix=time.time(),
                training_conditions=9, tested_conditions=10, segments_per_condition=320,
                runtime_statistics_final_audit_still_required=True))
            return
        time.sleep(15)
    # The independent wall watchdog handles all registered groups. This scheduler
    # stops admitting work immediately and preserves state for the next session.
    write(state_path, dict(state, status='deadline_or_user_stop', stopped_unix=time.time()))


if __name__ == '__main__':
    main()

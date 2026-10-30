"""Own-job scheduler: real GPU pilots, five controlled trials, then full Test."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from goal_runtime_control import identity, register_job, write

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json'
from fusion_repair_scope import TRAIN_NAMES,PRETRAIN_NAMES,EVAL_NAMES
CONDITIONS = ['rgb3_event4', 'event3_rgb4', 'event3_event4', 'rgb3_short90', 'rgb4', 'rgb3_event3', 'rgb4_event4']


def read(path):
    return json.loads(Path(path).read_text())


def main():
    goal = read(GOAL); out = Path(goal['output_root'])
    local = ROOT / '.experiment_storage/device34_fusion_repair_15h'
    local.mkdir(parents=True, exist_ok=True)
    lock = (local / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    register_job(goal, os.getpid(), 'fusion_repair_pipeline')
    python = str(ROOT / '.venv-humanpose/bin/python')
    state_path = out / 'pipeline_state.json'
    state = read(state_path) if state_path.exists() else dict(jobs=[], history=[], failed=[])
    children = {}; logs = {}; stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    env = dict(os.environ, EVENTPOSE_MULTISENSOR_OUT=str(out), EVENTPOSE_EVENT_OUT=str(out),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')

    def live(job):
        now = identity(job['pid'])
        return now is not None and now == {k: job[k] for k in ('pgid', 'start_ticks', 'command')}

    def launch(role, cmd, gpu=None):
        e = dict(env)
        if gpu is not None:
            e.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], EVENTPOSE_PHYSICAL_GPU=str(gpu['physical_gpu']))
        log = (out / 'logs' / (role.replace(':', '_') + '.log')).open('a', buffering=1)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(register_job(goal, proc.pid, role), gpu=gpu, launched_unix=time.time())
        children[proc.pid] = proc; logs[proc.pid] = log; state['jobs'].append(record)
        print(json.dumps(dict(event='launched', **record)), flush=True)

    def complete(name, pilot=False):
        path = out / 'runs' / name / 'complete.json'
        if not path.exists():
            return False
        row = read(path)
        if not pilot and name=='rgb3' and row['status']=='complete_user_shortened_with_checkpoint_recovery':
            recovery=read(out/'short_rgb_recovery.json')
            return recovery['status']=='passed' and recovery['no_new_optimizer_updates'] and row['successful_steps']>0
        if not pilot and name=='rgb3' and row['status']=='interrupted':
            amendment=out/'user_budget_amendment.json'
            if amendment.exists():
                stop_record=read(amendment)['rgb3']
                assert row['completed_unix']>=stop_record['stop_signal_unix']
                assert row['elapsed_seconds']>=stop_record['elapsed_at_stop_signal_seconds']-2
                return row['successful_steps']>0 and (out/'checkpoints'/name/'best.pt').exists()
        return row['status'] == ('pilot_complete' if pilot else 'complete') and row['successful_steps'] > 0

    def test_complete(name):
        return all((out / 'results' / name / t['session'] / 'complete.json').exists() for t in read(out / 'test/tasks.json'))

    def pilot_audit():
        if not all(complete('pilot_' + n, True) for n in CONDITIONS):
            return False
        for n in CONDITIONS:
            run = out / 'runs' / ('pilot_' + n)
            row = read(run / 'complete.json')
            assert row['step'] == 8 and row['successful_steps'] >= 4
            records = [json.loads(s) for s in (run / 'pilot_gradient_diagnostics.jsonl').read_text().splitlines()]
            if n not in ('rgb3', 'rgb3_short90','rgb4'):
                grads = [r.get('fusion_encoder_gradient_l2') for r in records]
                assert any(v is not None and math.isfinite(v) and v > 0 for v in grads), n
            if n == 'event3_event4':
                assert any((r.get('adapter_gradient_l2') or 0) > 0 for r in records)
        # Exact RGB starting path: compare actual native detections on same holdout.
        for control, fusion in [('rgb3_short90', 'rgb3_event4'), ('rgb4', 'event3_rgb4'),('rgb3_short90','rgb3_event3'),('rgb4','rgb4_event4')]:
            a = read(out / 'runs' / ('pilot_' + control) / 'baseline.json')
            b = read(out / 'runs' / ('pilot_' + fusion) / 'baseline.json')
            assert a['matched_instances'] == b['matched_instances'] and a['valid_instances'] == b['valid_instances']
            for k in ('root_mpjpe_mm', 'pa_mpjpe_mm', 'mpjpe_mm'):
                assert a[k] is not None and abs(a[k] - b[k]) <= .01, (control, fusion, k)
        write(out / 'gpu_model_pilots_verified.json', dict(status='passed', checked_unix=time.time(),
              trials=CONDITIONS, updates=8, minimum_successful_updates=4,
              rgb_native_initialization_equal=True, event_encoder_gradient_observed=True,
              scope='Train/holdout only; pilot weights are discarded before main trials'))
        return True

    import pynvml as n
    n.nvmlInit()
    while time.time() < goal['deadline_unix'] - 120 and not stop[0]:
        for job in list(state['jobs']):
            proc = children.get(job['pid']); code = proc.poll() if proc else None
            if (proc is not None and code is None) or (proc is None and live(job)):
                continue
            state['history'].append(dict(job, exit_code=code, ended_unix=time.time()))
            if code not in (None, 0):
                state['failed'].append(job['role'])
            state['jobs'].remove(job)
            if job['pid'] in logs:
                logs.pop(job['pid']).close()
        attempted = {j['role'] for j in state['jobs'] + state['history']}
        # No blind retries. A human/agent must diagnose failure and explicitly request one.
        retry = out / 'retry_requests.json'
        if retry.exists():
            roles = set(read(retry)['roles'])
            assert not roles & {j['role'] for j in state['jobs']}
            state['history'] = [j for j in state['history'] if j['role'] not in roles]
            state['failed'] = [r for r in state['failed'] if r not in roles]
            attempted -= roles
            retry.rename(out / ('retry_consumed_' + str(time.time_ns()) + '.json'))
        probe = read(out / 'gpu_preflight.json')
        proven=set(state.get('proven_cuda_uuids',[]))
        for record in probe['records']:
            if record['status']=='passed':proven.add(record['uuid'])
            elif record['status']!='busy_preserved':proven.discard(record['uuid'])
        state['proven_cuda_uuids']=sorted(proven)
        probe_role = 'gpu_recovery_probe:' + str(int(probe['checked_unix']))
        if time.time() >= probe['checked_unix'] + 1800 and not any(j['role'].startswith('gpu_recovery_probe:') for j in state['jobs']):
            if probe_role not in attempted:
                launch(probe_role, [python, '-u', str(ROOT / 'tools/check_fusion_repair_gpus.py')])
        ready = all((out / name).exists() and read(out / name)['status'] == 'passed'
                    for name in ('source_reuse_audit.json', 'repair_cpu_preflight.json', 'real_data_cpu_preflight.json',
                                 'actual_input_audit.json', 'current_dataset_audit.json',
                                 'body_holdout_subject_audit.json', 'event_normalization_Train.json',
                                 'native_cpu_preflight.json', 'ablation_cpu_preflight.json'))
        configs = {name: read(out / 'configs' / (name + '.json')) for name in CONDITIONS} if ready else {}
        pilot_path=out/'gpu_model_pilots_verified.json'
        verified_pilots=set(read(pilot_path)['trials']) if pilot_path.exists() else set()
        pilots_passed = set(CONDITIONS)<=verified_pilots
        if ready and not pilots_passed and all(complete('pilot_' + name, True) for name in CONDITIONS):
            pilots_passed = pilot_audit()
            verified_pilots=set(CONDITIONS)
        for gpu in probe['records']:
            if not ready or gpu['uuid'] not in proven or gpu['physical_gpu'] not in goal['allowed_gpus']:
                continue
            if any(j.get('gpu') and j['gpu']['uuid'] == gpu['uuid'] for j in state['jobs']):
                continue
            try:
                handle = n.nvmlDeviceGetHandleByUUID(gpu['uuid'])
                assert n.nvmlDeviceGetMinorNumber(handle) == gpu['physical_gpu']
                launch_apps=n.nvmlDeviceGetComputeRunningProcesses(handle)
                if launch_apps:
                    continue
                launch_bus=n.nvmlDeviceGetPciInfo(handle).busId
                if isinstance(launch_bus,bytes):launch_bus=launch_bus.decode()
                gpu=dict(gpu,launch_nvml_compute_pids=[],launch_nvml_bus=launch_bus,launch_checked_unix=time.time())
            except n.NVMLError:
                continue
            chosen = None
            extension_ready=(out/'same_device_extension_cpu.json').exists() and read(out/'same_device_extension_cpu.json')['status']=='passed'
            if not pilots_passed:
                for name in CONDITIONS:
                    if name in verified_pilots or (name in ('rgb3_event3','rgb4_event4') and not extension_ready):continue
                    role = 'pilot:' + name
                    if role not in attempted and not complete('pilot_' + name, True):
                        chosen = ('pilot', name); break
            if chosen is None:
                for name in CONDITIONS:
                    if name not in verified_pilots:continue
                    if 'train:' + name not in attempted and not complete(name) and time.time() + 4 * 3600 < goal['deadline_unix']:
                        chosen = ('train', name); break
                if chosen is None:
                    # Present trained RGB/fusion results promptly; retain original-weight controls.
                    for name in ['rgb3_short90', 'rgb4', 'rgb3_event4', 'event3_rgb4',
                                 'event3_event4', 'rgb3_event3', 'rgb4_event4'] + PRETRAIN_NAMES:
                        if (name in PRETRAIN_NAMES or complete(name)) and 'test:' + name not in attempted and not test_complete(name):
                            chosen = ('test', name); break
            if chosen is None:
                continue
            kind, name = chosen
            path = out / 'configs' / (name + '.json')
            if kind == 'pilot':
                cfg = dict(configs[name], name='pilot_' + name, pilot_steps=8, pilot_train_limit=64,
                           pilot_validation_limit=32, workers=0, trial_started_unix=time.time())
                path = out / 'configs' / ('pilot_' + name + '.json'); write(path, cfg)
            elif kind == 'train':
                cfg = read(path)
                # A retry preserves the original wall-clock budget and config for safe resume.
                if 'trial_started_unix' not in cfg:
                    cfg['trial_started_unix'] = time.time()
                    if name=='rgb3_short90':cfg['deadline_unix']=min(goal['deadline_unix'],cfg['trial_started_unix']+90*60+300)
                    write(path, cfg)
            cmd = ([python, '-u', str(ROOT / 'scripts/evaluate_multihmr.py'), 'worker', '--event-config', str(path), '--rank', '0', '--workers', '1']
                   if kind == 'test' else [python, '-u', str(ROOT / 'scripts/train_multihmr.py'), '--config', str(path)])
            role = kind + ':' + name; launch(role, cmd, gpu); attempted.add(role)
        state.update(updated_unix=time.time(), ready=ready, pilots_verified=pilots_passed,
                     training_complete=[name for name in CONDITIONS if complete(name)],
                     test_complete=[name for name in EVAL_NAMES if configs and test_complete(name)],
                     usable_gpus=probe['usable_gpus'], deadline_unix=goal['deadline_unix'])
        write(state_path, state)
        if configs and all(test_complete(name) for name in EVAL_NAMES) and not state['jobs']:
            assert all(complete(name) for name in CONDITIONS)
            write(out / 'all_training_test_complete.json', dict(completed_unix=time.time(),training_conditions=CONDITIONS,evaluation_conditions=EVAL_NAMES,final_audit_required=True))
            return
        time.sleep(15)
    write(state_path, dict(state, status='deadline_or_stop', stopped_unix=time.time()))


if __name__ == '__main__':
    main()

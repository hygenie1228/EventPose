"""Bounded Track 2 preparation and three parallel two-GPU adaptation trials.

Scientific training stays in scripts/train_multihmr.py. This supervisor owns
only these new jobs; it never resumes or modifies an older experiment.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from multihmr.track2_data import OUTPUT, plan, save_json

PYTHON = ROOT / '.venv-humanpose/bin/python'
CHILDREN = {}
MUTEX = threading.Lock()
REGISTRY_NAME = 'managed_jobs.json'
CANCELLED = threading.Event()


def register():
    jobs = []
    for pid, value in CHILDREN.items():
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
            jobs.append(dict(pid=pid, pgid=os.getpgid(pid), start_ticks=stat[19], **value))
        except FileNotFoundError:
            pass
    save_json(OUTPUT / REGISTRY_NAME, dict(supervisor_pid=os.getpid(), jobs=jobs,
        updated_unix=time.time()))


def stop_children(signum=None, frame=None):
    CANCELLED.set()
    with MUTEX:
        for pid in list(CHILDREN):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if signum is not None:
        raise SystemExit(128 + signum)


def run(command, gpus, label, deadline):
    assert all(gpu in range(1,7) for gpu in gpus),'GPU 0 must remain untouched'
    resource_file=OUTPUT/'gpu_resources.json'
    resources=json.loads(resource_file.read_text()) if resource_file.exists() else None
    if resources:
        assert all(gpu in resources['healthy_gpus'] for gpu in gpus),('Unavailable GPU requested',gpus)
        mask=','.join(resources['uuids'][str(gpu)] for gpu in gpus)
    else:mask=','.join(map(str,gpus))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=mask,
        PYTHONUTF8='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    log = OUTPUT / 'logs' / (label + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        with MUTEX:
            CHILDREN[process.pid] = dict(label=label, command=list(map(str, command)),
                gpus=gpus, started_unix=time.time())
            register()
            with (OUTPUT/'job_history.jsonl').open('a') as history:
                history.write(json.dumps(dict(event='started',pid=process.pid,**CHILDREN[process.pid]))+'\n')
        try:
            code = process.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'{label}: global 12-hour deadline reached')
        finally:
            with MUTEX:
                with (OUTPUT/'job_history.jsonl').open('a') as history:
                    history.write(json.dumps(dict(event='exited',pid=process.pid,label=label,
                        gpus=gpus,returncode=process.poll(),time=time.time()))+'\n')
                CHILDREN.pop(process.pid, None)
                register()
        if code:
            raise RuntimeError(f'{label} failed ({code}); inspect {log}')
    return label


def healthy_gpus():
    resource_file=OUTPUT/'gpu_resources.json'
    return json.loads(resource_file.read_text())['healthy_gpus'] if resource_file.exists() else list(range(1,7))


def run_training(command,index,label,deadline):
    """Lease an available healthy pair; a third candidate waits without overlap."""
    resource_file=OUTPUT/'gpu_resources.json'
    pairs=json.loads(resource_file.read_text())['training_pairs'] if resource_file.exists() else [[1,2],[3,4],[5,6]]
    locks=OUTPUT/'locks';locks.mkdir(exist_ok=True)
    while time.time()<deadline and not CANCELLED.is_set():
        for offset in range(len(pairs)):
            pair=pairs[(index+offset)%len(pairs)]
            with (locks/('training_pair_'+'_'.join(map(str,pair))+'.lock')).open('a') as lease:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if CANCELLED.is_set():raise RuntimeError('Training dispatch cancelled')
                # Another controller can finish a queued run while we wait for
                # its GPU lease. Recheck under the lease to prevent duplicate DDP.
                completed=OUTPUT/'runs'/label/'complete.json'
                if completed.exists():
                    assert json.loads(completed.read_text())['successful_steps']>0
                    return label
                return run(command,pair,label,deadline)
        CANCELLED.wait(2)
    raise RuntimeError('Training dispatch cancelled or original deadline reached')


def configurations(deadline):
    common = dict(experiment='track2_multihmr_12h', output_root=str(OUTPUT.resolve()),
        seed=261025, hours=2, deadline_unix=deadline, batch_size=8, workers=2,
        pa_weight=100, root_weight=25, validation_minutes=10,
        warmup_steps=50, augment=True, weight_decay=.1,
        max_steps=2500, max_epochs=20, early_stop_patience=4,
        horizontal_flip_probability=0, no_test_selection=True)
    trials = [dict(name='head_strong', last_blocks=0, head_lr=3e-6,
                   backbone_lr=0, anchor_l2_weight=2e5),
              dict(name='last2_strong', last_blocks=2, head_lr=3e-6,
                   backbone_lr=2e-7, anchor_l2_weight=1e5),
              dict(name='last4_strong', last_blocks=4, head_lr=3e-6,
                   backbone_lr=2e-7, anchor_l2_weight=5e4)]
    for value in trials:
        config = dict(common, **value)
        save_json(OUTPUT / 'configs' / (value['name'] + '.json'), config)
    pilot = dict(common, **dict(name='pilot', last_blocks=0, head_lr=3e-6,
        backbone_lr=0, anchor_l2_weight=2e5, pilot_steps=10, batch_size=2,
        hours=.25, early_stop_patience=0))
    save_json(OUTPUT / 'configs/pilot.json', pilot)
    return trials


def train_command(name):
    return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(ROOT / 'scripts/train_multihmr.py'), '--config',
        str(OUTPUT / 'configs' / (name + '.json'))]


def main(phase):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop_children)
        signal.signal(signal.SIGINT, stop_children)
        budget = json.loads((OUTPUT / 'budget.json').read_text())
        deadline = budget['deadline_unix']
        assert time.time() < deadline
        manifest = plan()
        trials = configurations(deadline)
        save_json(OUTPUT / 'supervisor.json', dict(pid=os.getpid(), status='preparing',
            started_unix=time.time(), deadline_unix=deadline))
        if phase == 'all' or phase == 'prepare':
            def shard(index):
                for sample in manifest['train'][index::6]:
                    run([str(PYTHON), '-u', str(ROOT / 'tools/prepare_track2_multihmr.py'),
                         '--sample', sample], [index + 1], 'prepare_' + sample, deadline)
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(shard, i) for i in range(6)]
                for future in futures:
                    future.result()
            run([str(PYTHON), str(ROOT / 'tools/prepare_track2_multihmr.py'), '--finalize'],
                [], 'finalize_preparation', deadline)
        if phase == 'prepare':
            save_json(OUTPUT / 'supervisor.json', dict(pid=os.getpid(), status='prepared',
                updated_unix=time.time(), deadline_unix=deadline))
            return
        if not (OUTPUT / 'runs/pilot/complete.json').exists():
            run(train_command('pilot'), [1, 2], 'pilot', deadline)
        save_json(OUTPUT / 'supervisor.json', dict(pid=os.getpid(), status='training',
            updated_unix=time.time(), deadline_unix=deadline))
        def trial(index):
            name = trials[index]['name']
            if not (OUTPUT / 'runs' / name / 'complete.json').is_file():
                run(train_command(name), budget['physical_gpu_pairs'][index], name, deadline)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(trial, i) for i in range(3)]
            for future in futures:
                future.result()
        save_json(OUTPUT / 'supervisor.json', dict(pid=os.getpid(), status='trained_pending_evaluation',
            updated_unix=time.time(), deadline_unix=deadline))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'prepare', 'train'], default='all')
    args = parser.parse_args()
    try:
        main(args.phase)
    except BaseException as error:
        save_json(OUTPUT / 'supervisor_error.json', dict(error=repr(error), time=time.time()))
        stop_children()
        raise

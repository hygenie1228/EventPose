"""Use an idle leased GPU pair for completed fixed searches' pending proxies."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import signal
import time

import pynvml
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT, save_json, sha256


def read(path):
    return json.loads(path.read_text())


def main():
    with (OUTPUT / 'locks/fixed_proxy_backfill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME = 'fixed_proxy_backfill_jobs.json'
        runtime.register()
        signal.signal(signal.SIGTERM, runtime.stop_children)
        signal.signal(signal.SIGINT, runtime.stop_children)
        plan = read(OUTPUT / 'fixed_view_adaptation_plan.json')
        names = [c['name'] for c in plan['trials']]
        assert all((OUTPUT / 'runs' / name / 'complete.json').exists() for name in names)
        pending = []
        for name in names:
            if (OUTPUT / 'fixed_checkpoint_choices' / (name + '.json')).exists():
                continue
            for kind in ['best', 'last']:
                tag = name + '_' + kind + '_fixed'
                if not (OUTPUT / 'validation_smplx' / tag / 'metrics.json').exists():
                    pending.append((name, kind, tag))
        resources = read(OUTPUT / 'gpu_resources.json')
        deadline = min(read(OUTPUT / 'budget.json')['deadline_unix'], time.time() + 900)
        pynvml.nvmlInit()

        def state(status, **extra):
            save_json(OUTPUT / 'fixed_proxy_backfill_status.json', dict(
                status=status, pid=os.getpid(), Test_used=False,
                model_selection_performed=False, source_exports_changed=False,
                updated_unix=time.time(), **extra))

        if not pending:
            state('nothing_pending')
            return
        for pair in resources['training_pairs']:
            with (OUTPUT / 'locks' / ('training_pair_' + '_'.join(map(str, pair)) + '.lock')).open('a') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if any(pynvml.nvmlDeviceGetComputeRunningProcesses(
                        pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in pair):
                    continue
                tasks = pending[:len(pair)]
                state('validating_completed_search_checkpoints', gpus=pair, tasks=tasks)

                def evaluate(task, gpu):
                    name, kind, tag = task
                    checkpoint = OUTPUT / 'checkpoints' / name / (kind + '.pt')
                    command = [str(runtime.PYTHON), '-u', str(runtime.ROOT / 'scripts/evaluate_multihmr.py'),
                               'track2-validate', '--checkpoint', str(checkpoint),
                               '--validation-tag', tag, '--input-view', 'native_rgb_static_crop']
                    # The evaluation launcher's per-tag lock also protects a
                    # concurrent main-controller request for this exact result.
                    runtime.run(command, [gpu], 'fixed_proxy_backfill_' + tag, deadline)
                    value = read(OUTPUT / 'validation_smplx' / tag / 'metrics.json')
                    assert value['checkpoint_sha256'] == sha256(checkpoint)
                    assert value['evaluation_protocol'] == 'native_rgb_static_crop_target_head_v1'
                    assert not value['test_used'] and not value['GT_used_for_detection_selection']
                    return dict(tag=tag, gpu=gpu, checkpoint_sha256=value['checkpoint_sha256'])

                with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                    futures = [pool.submit(evaluate, task, gpu) for task, gpu in zip(tasks, pair)]
                    completed = [future.result() for future in futures]
                state('completed_pending_proxy_evaluations', completed=completed)
                return
        state('not_run_no_idle_pair')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save_json(OUTPUT / 'fixed_proxy_backfill_error.json', dict(error=repr(error), time=time.time()))
        runtime.stop_children()
        raise

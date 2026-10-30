"""Prefetch two frozen Train-MHR validation clips on an idle GPU pair."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import pynvml
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command
from multihmr.track2_data import OUTPUT, save_json, sha256


def read(path):
    return json.loads(path.read_text())


def main(name, prefix):
    with (OUTPUT / 'locks' / (prefix + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME = prefix + '_jobs.json'
        runtime.register()
        signal.signal(signal.SIGTERM, runtime.stop_children)
        signal.signal(signal.SIGINT, runtime.stop_children)

        def state(status, **extra):
            save_json(OUTPUT / (prefix + '_status.json'), dict(
                status=status, pid=os.getpid(), Test_used=False,
                model_selection_performed=False, source_exports_changed=False,
                updated_unix=time.time(), **extra))

        trial = 'search_' + name
        if (OUTPUT / 'final_selected.json').exists():
            state('not_run_Train_comparison_already_complete')
            return
        # No Test scores or fullTrain checkpoint enters this Train16/4 search evaluation.
        choice = read(OUTPUT / 'fixed_checkpoint_choices' / (name + '.json'))
        checkpoint = Path(choice['checkpoint'])
        assert not choice['test_used'] and choice['metrics']['coverage'] >= .95
        assert sha256(checkpoint) == choice['metrics']['checkpoint_sha256']
        manifest = read(OUTPUT / 'manifest.json')
        tasks = [s for s in reversed(manifest['validation'])
                 if not (OUTPUT / 'predictions' / trial / s / 'metrics.json').exists()][:2]
        assert not set(tasks) & set(manifest['test'])
        if not tasks:
            state('nothing_pending')
            return
        resources = read(OUTPUT / 'gpu_resources.json')
        deadline = min(read(OUTPUT / 'budget.json')['deadline_unix'], time.time() + 1200)
        pynvml.nvmlInit()
        for pair in resources['training_pairs']:
            with (OUTPUT / 'locks' / ('training_pair_' + '_'.join(map(str, pair)) + '.lock')).open('a') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if any(pynvml.nvmlDeviceGetComputeRunningProcesses(
                        pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in pair):
                    continue
                state('evaluating_two_Train_validation_clips', trial=trial, samples=tasks, gpus=pair)

                def evaluate(sample, gpu):
                    for stage in ['track2-infer', 'track2-convert', 'track2-score']:
                        command = evaluation_command(stage, trial, sample)
                        if stage == 'track2-infer':
                            command.extend(['--input-view', 'native_rgb_static_crop', '--checkpoint', str(checkpoint),
                                            '--fixed-view-plan', str(OUTPUT / 'fixed_view_all30_plan.json')])
                        runtime.run(command, [gpu], prefix + '_' + stage + '_' + sample, deadline)
                    folder = OUTPUT / 'predictions' / trial / sample
                    value = read(folder / 'metrics.json')
                    info = read(folder / 'inference.json')
                    assert info['checkpoint_sha256'] == sha256(checkpoint)
                    assert info['evaluation_protocol'] == value['evaluation_protocol'] == 'native_rgb_static_crop_target_head_v1'
                    assert not info['gt_used_for_detection_or_tracking'] and not info['target_foreground_only_input']
                    assert info['all_accepted_heads_in_target_mask']
                    assert value['full_frame_roster'] and value['frames'] == manifest['samples'][sample]['frames']
                    assert value['gt_sha256'] == manifest['samples'][sample]['gt_sha256']
                    assert value['conversion_revision'] == 2 and value['prediction_sha256'] == sha256(folder / 'human_params_mhr_v2.npz')
                    return dict(sample=sample, gpu=gpu, frames=value['frames'])

                with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                    futures = [pool.submit(evaluate, sample, gpu) for sample, gpu in zip(tasks, pair)]
                    completed = [future.result() for future in futures]
                state('completed_two_Train_validation_clips', trial=trial, completed=completed)
                return
        state('not_run_no_idle_pair')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['fixed_head_cd100_rel100', 'fixed_last2_cd100_rel100',
                                              'fixed_last4_cd100_rel100'], default='fixed_last4_cd100_rel100')
    args = parser.parse_args()
    prefix = 'fixed_validation_backfill' if args.candidate == 'fixed_last4_cd100_rel100' else 'fixed_validation_backfill_' + args.candidate.split('_')[1]
    try:
        main(args.candidate, prefix)
    except BaseException as error:
        save_json(OUTPUT / (prefix + '_error.json'), dict(error=repr(error), time=time.time()))
        runtime.stop_children()
        raise

"""Compare the prior nine pipelines and three native-surface adaptations.

Train-only selection is frozen before the new Test scores. Source outputs and
global summaries are promoted only after whole comparisons and exports finish.
"""
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack
import argparse
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import time

import numpy as np
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
from multihmr.track2_data import OUTPUT, SOURCE, save_json, sha256
from multihmr.target_crop import BODY_PROTOCOLS
from audit_track2_strict_results import audit_heads

KEYS = ['CD_H_cm', 'ACC_H_cm_per_frame2']


def read(path):
    return json.loads(path.read_text())


def main(completion_timeout_seconds=None):
    with (OUTPUT / 'locks/native_surface_finalization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME = 'native_surface_finalization_jobs.json'
        runtime.register()
        signal.signal(signal.SIGTERM, runtime.stop_children)
        signal.signal(signal.SIGINT, runtime.stop_children)
        manifest = read(OUTPUT / 'manifest.json')
        plan = read(OUTPUT / 'native_surface_adaptation_plan.json')
        assert plan.get('ready_to_launch'), 'Complete twelve-pipeline reporting/audit integration before launching'
        names = [c['name'] for c in plan['trials']]
        input_view=plan['input_view'];protocol=BODY_PROTOCOLS[input_view]
        assert plan['evaluation_protocol']==protocol
        assert all(c['strict_target_mask'] and not c['target_foreground'] for c in plan['trials'])
        assert all(c['target_body_overlap'] for c in plan['trials'])
        deadline = read(OUTPUT / 'budget.json')['deadline_unix']
        if completion_timeout_seconds is not None:
            # Explicit goal reactivation permits completing interrupted
            # inference/export work; it does not reset the training budget.
            assert 0 < completion_timeout_seconds <= 10800
            assert time.time() >= deadline
            assert read(OUTPUT / 'native_surface_training_status.json')['status'] == 'all_native_surface_searches_and_Train20_refits_complete'
            mapping = read(OUTPUT / 'native_surface_refit_mappings.json')
            assert set(mapping) == set(names)
            assert all(v['complete_training_cost_below3h'] for v in mapping.values())
            frozen = read(OUTPUT / 'body_target_final_selected.json')
            assert len(frozen['models']) == 12 and not frozen['test_used_for_selection']
            for sample, value in manifest['samples'].items():
                assert sha256(SOURCE / sample / 'human/human_params_gt.npz') == value['gt_sha256']
            completion_started = time.time()
            recovery_path = OUTPUT / 'native_surface_completion_recovery.json'
            previous_recovery = read(recovery_path) if recovery_path.exists() else None
            if previous_recovery is not None:
                assert previous_recovery['original_budget_sha256'] == sha256(OUTPUT / 'budget.json')
                assert previous_recovery['frozen_selection_sha256'] == sha256(OUTPUT / 'body_target_final_selected.json')
                assert previous_recovery['completion_timeout_seconds'] == completion_timeout_seconds
                completion_started = previous_recovery['started_unix']
                assert time.time() < completion_started + completion_timeout_seconds - 60, 'Completion recovery timeout already elapsed'
                archive = OUTPUT / 'failure_history' / ('completion_recovery_receipt_' + str(time.time_ns()) + '.json')
                archive.write_bytes(recovery_path.read_bytes())
            save_json(OUTPUT / 'native_surface_completion_recovery.json', dict(
                authorization='User explicitly reactivated the paused goal; complete already-trained evaluations and exports only',
                original_budget_deadline_unix=deadline,
                original_budget_sha256=sha256(OUTPUT / 'budget.json'),
                completion_timeout_seconds=completion_timeout_seconds,
                frozen_selection_sha256=sha256(OUTPUT / 'body_target_final_selected.json'),
                training_restarted=False, started_unix=completion_started,
                resumed_unix=time.time(),
                supervisor_pid=os.getpid(),
                supervisor_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                completion_deadline_unix=completion_started + completion_timeout_seconds))
            deadline = completion_started + completion_timeout_seconds
            os.environ['TRACK2_COMPLETION_DEADLINE_UNIX'] = str(deadline)
            error = OUTPUT / 'native_surface_finalization_error.json'
            if error.exists():
                archive = OUTPUT / 'failure_history' / ('post_reboot_evaluation_' + str(time.time_ns()))
                archive.mkdir(parents=True)
                error.rename(archive / error.name)
        gpus = runtime.healthy_gpus()

        def state(status, **extra):
            save_json(OUTPUT / 'native_surface_finalization_status.json', dict(
                status=status, pid=os.getpid(), models=names, test_used_for_selection=False,
                updated_unix=time.time(), **extra))

        def check_training():
            assert not (OUTPUT / 'native_surface_training_error.json').exists()
            path = OUTPUT / 'native_surface_training_status.json'
            if path.exists():
                status = read(path)['status']
                if status.startswith('not_run_'):
                    state(status)
                    return False
            assert time.time() < deadline - 60
            return True

        state('waiting_for_native_surface_training_start')
        while not all((OUTPUT / ('native_surface_prior_' + label + '.json')).exists()
                      for label in ['selected', 'test', 'exports']):
            if not check_training():
                return
            time.sleep(10)
        prior = read(OUTPUT / 'native_surface_prior_selected.json')
        prior_test = read(OUTPUT / 'native_surface_prior_test.json')
        prior_exports = read(OUTPUT / 'native_surface_prior_exports.json')
        assert len(prior['models']) == len(prior_test['models']) == 9
        reference = next(iter(prior['models'].values()))['per_sequence'][manifest['validation'][0]]['protocol']
        view = read(OUTPUT / 'fixed_view_all30_plan.json')

        def checkpoint(name, full):
            prefix = 'native_surface_refit_choices' if full else 'native_surface_checkpoint_choices'
            value = read(OUTPUT / prefix / (name + '.json'))
            return value['refit']['checkpoint'] if full else value['checkpoint']

        def verify(name, sample, full):
            trial = name if full else 'search_' + name
            folder = OUTPUT / 'predictions' / trial / sample
            if not (folder / 'metrics.json').exists():
                return None
            value = read(folder / 'metrics.json')
            info = read(folder / 'inference.json')
            assert value['evaluation_protocol'] == info['evaluation_protocol'] == protocol
            assert info['checkpoint_sha256'] == sha256(Path(checkpoint(name, full)))
            assert info['input_view'] == input_view and not info['target_foreground_only_input']
            assert info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
            assert info['target_body_overlap'] and info['body_box_rule_sha256']==sha256(OUTPUT/'body_target_gate_Train_rule.json')
            assert info['all_accepted_heads_in_target_mask'] and not info['gt_used_for_detection_or_tracking']
            assert info.get('depth_decoder', 'published') == 'published'
            if input_view=='native_rgb_static_crop':assert info['static_crop_box'] == view['samples'][sample]['box']
            assert info['target_mask_sha256'] == sha256(SOURCE / sample / 'processed/human_mask.mp4')
            assert value['conversion_revision'] == 2 and value['protocol'] == reference and value['full_frame_roster']
            assert value['frames'] == manifest['samples'][sample]['frames']
            assert value['gt_sha256'] == manifest['samples'][sample]['gt_sha256']
            assert value['prediction_sha256'] == sha256(folder / 'human_params_mhr_v2.npz')
            audit_heads(folder,sample,manifest,info)
            return value

        def stages(name, sample, full, gpu, score=True):
            if score and verify(name, sample, full) is not None:
                return
            trial = name if full else 'search_' + name
            for stage in ['track2-infer', 'track2-convert'] + (['track2-score'] if score else []):
                assert not runtime.CANCELLED.is_set()
                command = finish.evaluation_command(stage, trial, sample)
                if stage == 'track2-infer':
                    command.extend(['--checkpoint', checkpoint(name, full), '--input-view', input_view,'--strict-target-mask','--target-body-overlap'])
                    if input_view=='native_rgb_static_crop':
                        command.extend(['--fixed-view-plan', str(OUTPUT / 'fixed_view_all30_plan.json')])
                runtime.run(command, [gpu], 'native_surface_final_' + stage + '_' + trial + '_' + sample, deadline)
                if stage=='track2-infer':
                    folder=OUTPUT/'predictions'/trial/sample
                    audit=audit_heads(folder,sample,manifest)
                    save_json(folder/'strict_head_mask_audit.json',audit)

        def aggregate(name, samples, full):
            values = {s: verify(name, s, full) for s in samples}
            assert all(values.values())
            return dict(**{k: float(np.mean([v[k] for v in values.values()])) for k in KEYS},
                        sequences=len(samples), frames=sum(v['frames'] for v in values.values()), per_sequence=values)

        def validation_snapshot():
            models = {}
            coverage = {}
            for name in names:
                if not (OUTPUT / 'native_surface_checkpoint_choices' / (name + '.json')).exists():
                    continue
                if not all((OUTPUT / 'predictions' / ('search_' + name) / s / 'metrics.json').exists() for s in manifest['validation']):
                    continue
                models[name] = aggregate(name, manifest['validation'], False)
                coverage[name] = float(np.mean([read(OUTPUT / 'predictions' / ('search_' + name) / s / 'inference.json')['detection_frame_rate']
                                                for s in manifest['validation']]))
            save_json(OUTPUT / 'native_surface_validation_mhr_progress.json', dict(
                models=models, target_coverage=coverage, test_used_for_selection=False, updated_unix=time.time()))
            return models, coverage

        # While DDP uses the two pairs, GPU6 evaluates completed search weights.
        # Yield between clips to let the workers' short best/last proxy jobs run.
        state('validating_native_surface_Train4_MHR_during_training')
        while not (OUTPUT / 'native_surface_refit_mappings.json').exists():
            if not check_training():
                return
            pending_proxy = any((OUTPUT / 'runs' / name / 'complete.json').exists()
                                and not (OUTPUT / 'native_surface_checkpoint_choices' / (name + '.json')).exists() for name in names)
            if pending_proxy:
                time.sleep(3)
                continue
            task = None
            for name in names:
                if not (OUTPUT / 'native_surface_checkpoint_choices' / (name + '.json')).exists():
                    continue
                for sample in manifest['validation']:
                    if verify(name, sample, False) is None:
                        task = name, sample
                        break
                if task:
                    break
            if task:
                name, sample = task
                state('validating_native_surface_Train4_MHR_during_training', current_model=name, current_sample=sample)
                with (OUTPUT / 'locks/fixed_gpu6_evaluation.lock').open('a') as lease:
                    fcntl.flock(lease, fcntl.LOCK_EX)
                    stages(name, sample, False, 6)
                validation_snapshot()
            else:
                time.sleep(5)
        mapping = read(OUTPUT / 'native_surface_refit_mappings.json')
        assert set(mapping) == set(names) and all(v['complete_training_cost_below3h'] for v in mapping.values())
        # Freeze the complete corrected scope before scoring any new Test clip.
        base=read(OUTPUT/'body_target_evaluation_plan.json')['models'];assert len(base)==9
        extra=[]
        for name in names:
            search=checkpoint(name,False);full=checkpoint(name,True)
            extra.append(dict(name=name,source_trial=name,input_view=input_view,evaluation_protocol=protocol,
                search_checkpoint=search,full_checkpoint=full,search_checkpoint_sha256=sha256(Path(search)),
                full_checkpoint_sha256=sha256(Path(full))))
        scope_path=OUTPUT/'body_target_final_evaluation_plan.json'
        frozen_scope=dict(models=base+extra,test_used_for_selection=False,created_unix=time.time())
        if scope_path.exists():assert read(scope_path)['models']==frozen_scope['models']
        else:save_json(scope_path,frozen_scope)

        def parallel(tasks, full, score=True,model_priority=None):
            queue = Queue()
            ranks={name:i for i,name in enumerate(model_priority or [])}
            if model_priority is not None:assert {n for n,_ in tasks}<=set(ranks)
            ordered=sorted(tasks,key=lambda x:(ranks.get(x[0],0),-manifest['samples'][x[1]]['frames']))
            for task in ordered:
                queue.put(task)

            def worker(gpu):
                while not runtime.CANCELLED.is_set():
                    try:
                        name, sample = queue.get_nowait()
                    except Empty:
                        return
                    try:
                        stages(name, sample, full, gpu, score)
                    finally:
                        queue.task_done()

            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [pool.submit(worker, gpu) for gpu in gpus]
                try:
                    finished, _ = wait(futures, return_when=FIRST_EXCEPTION)
                    for future in finished:
                        future.result()
                    for future in futures:
                        future.result()
                except BaseException:
                    runtime.stop_children()
                    raise

        # All refits are finished. Own both pair leases before dispatching all
        # five GPUs, so any CPU backfill controller must finish its current work.
        state('waiting_for_GPU_pairs_for_remaining_validation')
        with ExitStack() as leases:
            resources = read(OUTPUT / 'gpu_resources.json')
            for pair in resources['training_pairs']:
                lease = leases.enter_context((OUTPUT / 'locks' / ('training_pair_' + '_'.join(map(str, pair)) + '.lock')).open('a'))
                fcntl.flock(lease, fcntl.LOCK_EX)
            parallel([(name, s) for name in names for s in manifest['validation']], False)
            validation, coverage = validation_snapshot()
            assert set(validation) == set(names)
            models = dict(prior['models'], **validation)
            all_coverage = dict(prior['target_coverage'], **coverage)
            order = prior['model_order'] + names
            eligible = [n for n in order if all_coverage[n] >= .95]
            assert len(order) == len(models) == 12 and eligible
            winner = min(eligible, key=lambda n: (models[n]['CD_H_cm'], models[n]['ACC_H_cm_per_frame2'], order.index(n)))
            protocols = dict(prior['input_protocols'], **{n: protocol for n in names})
            selection = dict(trial=winner, models=models, eligible_trials=eligible, target_coverage=all_coverage,
                             model_order=order, input_protocols=protocols, evaluation_protocol=protocols[winner],
                             selection='Full Train4 MHR CD-H across all twelve; coverage>=.95, ACC tie-break; never Test',
                             test_used_for_selection=False, selected_unix=time.time())
            previous_selection = OUTPUT / 'native_surface_selected.json'
            if previous_selection.exists():
                frozen = read(previous_selection)
                assert {k: v for k, v in frozen.items() if k != 'selected_unix'} == {k: v for k, v in selection.items() if k != 'selected_unix'}
                selection = frozen
            save_json(OUTPUT / 'native_surface_selected.json', selection)
            save_json(OUTPUT / 'body_target_final_selected.json', selection)
            state('evaluating_frozen_native_surface_models_Test10', selected_trial=winner)
            priority=sorted(names,key=lambda n:(models[n]['CD_H_cm'],models[n]['ACC_H_cm_per_frame2'],order.index(n)))
            save_json(OUTPUT/'native_surface_Test_dispatch_order.json',dict(model_order=priority,
                ordering_source='Complete Train4 MHR CD-H then ACC-H, frozen before new Test scoring',
                Test_used_for_ordering=False,all_three_models_required=True,all_ten_sequences_per_model_required=True,
                selected_unix=selection['selected_unix'],created_unix=time.time()))
            parallel([(name, s) for name in names for s in manifest['test']], True,model_priority=priority)
            tested = {name: aggregate(name, manifest['test'], True) for name in names}
            save_json(OUTPUT / 'native_surface_test_comparison.json', dict(
                models=tested, test_used_for_selection=False, evaluation_protocol=protocol, completed_unix=time.time()))
            combined = dict(prior_test['models'], **tested)
            comparison = dict(models=combined, input_protocols=protocols, test_used_for_selection=False,
                              selected_trial=winner, aggregation='equal sequence weights; every original frame', completed_unix=time.time())
            if winner in names:
                state('generating_native_surface_selected_all30', selected_trial=winner)
                parallel([(winner, s) for s in manifest['train']], True, False)
                receipts = {}
                for sample in sorted(manifest['samples']):
                    receipts[sample] = finish.export_sample(winner, sample, manifest)
                    save_json(OUTPUT / 'native_surface_export_receipts.json', dict(
                        trial=winner, samples=receipts, evaluation_protocol=protocols[winner],
                        test_used_for_selection=False, updated_unix=time.time()))
            else:
                assert winner == prior['trial']
                receipts = prior_exports['samples']
            assert set(receipts) == set(manifest['samples'])
            for sample, item in receipts.items():
                assert sha256(Path(item['path'])) == item['sha256']
                assert sha256(SOURCE / sample / 'human/human_params_gt.npz') == manifest['samples'][sample]['gt_sha256']
            exported = dict(trial=winner, samples=receipts, evaluation_protocol=protocols[winner],
                            test_used_for_selection=False,strict_target_mask=True,
                            retained_existing_prior_exports=winner not in names, updated_unix=time.time())
            save_json(OUTPUT / 'native_surface_export_receipts.json', exported)
            save_json(OUTPUT / 'body_target_final_test_comparison.json', comparison)
            save_json(OUTPUT / 'body_target_final_export_receipts.json', exported)
        state('all30_ready_for_independent_strict_final_audit', selected_trial=winner, exports=30)
        subprocess.run([str(runtime.PYTHON), str(runtime.ROOT / 'tools/report_track2_experiments.py')], check=True)
        visual = OUTPUT / 'target_foreground_visuals' / (winner + '_exported') / 'all/receipt.json'
        reuse_visual = False
        if visual.exists():
            receipt = read(visual)
            reuse_visual = set(receipt['samples']) == set(manifest['samples']) and all(
                v['native_prediction_sha256'] == receipts[s]['sha256'] for s, v in receipt['samples'].items())
        if not reuse_visual:
            runtime.run([str(runtime.PYTHON), '-u', str(runtime.ROOT / 'tools/visualize_track2_foreground.py'),
                         '--trial', winner, '--exported', '--split', 'all'], [], 'native_surface_final_source_visuals_CPU', deadline)
        save_json(OUTPUT / 'native_surface_visual_preparation_status.json', dict(
            status='rendered_pending_agent_review' if not reuse_visual else 'existing_matching_source_visuals_retained',
            trial=winner, GPU_used=False, updated_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-timeout-seconds', type=int,
                        help='After explicit goal reactivation, finish completed-training evaluation/export without changing budget.json (max 10800).')
    args = parser.parse_args()
    try:
        main(args.completion_timeout_seconds)
    except BaseException as error:
        save_json(OUTPUT / 'native_surface_finalization_error.json', dict(error=repr(error), time=time.time()))
        runtime.stop_children()
        raise

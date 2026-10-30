"""Orchestrate the frozen native-surface Train16/4 searches and Train20 refits.

No GPU work starts before the prior nine pipelines and actual correctness pilot
finish. Training, validation, and checkpoint serialization remain in scripts/.
"""
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT, save_json, sha256
from multihmr.target_crop import BODY_PROTOCOLS


def read(path):
    return json.loads(path.read_text())


def main():
    with (OUTPUT / 'locks/native_surface_training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME = 'native_surface_training_jobs.json'
        runtime.register()
        signal.signal(signal.SIGTERM, runtime.stop_children)
        signal.signal(signal.SIGINT, runtime.stop_children)
        plan = read(OUTPUT / 'native_surface_adaptation_plan.json')
        assert plan.get('ready_to_launch'), 'Complete twelve-pipeline reporting/audit integration before launching'
        trials = plan['trials']
        input_view=plan['input_view'];protocol=BODY_PROTOCOLS[input_view]
        assert plan['evaluation_protocol']==protocol
        deadline = read(OUTPUT / 'budget.json')['deadline_unix']
        recovery_path=OUTPUT/'native_surface_mask_decode_recovery.json'
        recovery=read(recovery_path) if recovery_path.exists() else {}
        if recovery:
            assert recovery['existing_wave_started_before_admission_cutoff']
            assert read(OUTPUT/'target_mask_decoder_recovery_preflight.json')['status']=='passed_actual_CPU_decode'
            for name,item in recovery['previous_attempts'].items():
                assert sha256(OUTPUT/'configs'/(name+'.json'))==item['config_sha256']
        assert len(trials) == 3
        assert all(c['native_surface_target'] and not c['target_foreground'] and c['strict_target_mask'] for c in trials)
        assert all(c['target_body_overlap'] for c in trials)
        assert all(bool(c.get('fixed_view_plan'))==(input_view=='native_rgb_static_crop') for c in trials)
        assert all(not c.get('refit_init_checkpoint') and not c.get('bootstrap_checkpoint') for c in trials)

        def state(status, **extra):
            save_json(OUTPUT / 'native_surface_training_status.json', dict(
                status=status, pid=os.getpid(), trials=[c['name'] for c in trials],
                test_used_for_selection=False, source_exports_changed=False, updated_unix=time.time(), **extra))

        if (OUTPUT / 'native_surface_refit_mappings.json').exists():
            state('all_native_surface_searches_and_Train20_refits_complete')
            return
        state('waiting_for_prior_nine_and_actual_native_surface_pilot')
        while True:
            assert time.time()<deadline-60
            if time.time() > deadline - 4 * 3600 and not recovery:
                state('not_run_insufficient_remaining_budget')
                return
            prior = OUTPUT / 'body_target_evaluation_status.json'
            pilot = OUTPUT / 'native_surface_pilot_status.json'
            if prior.exists() and pilot.exists():
                if (read(prior)['status'] == 'all30_ready_for_independent_strict_audit'
                        and read(pilot)['status'] == 'complete_actual_native_surface_two_GPU_pilot'):
                    break
            assert not (OUTPUT / 'body_target_evaluation_error.json').exists()
            assert not (OUTPUT / 'native_surface_pilot_error.json').exists()
            time.sleep(20)
        prior_selected = read(OUTPUT / 'body_target_selected.json')
        prior_test = read(OUTPUT / 'body_target_test_comparison.json')
        prior_exports = read(OUTPUT / 'body_target_export_receipts.json')
        assert len(prior_selected['models']) == len(prior_test['models']) == 9
        assert not prior_selected['test_used_for_selection']
        initial = read(OUTPUT / 'init_comparison.json')['models']['init_v6']['CD_H_cm']
        if prior_test['models'][prior_selected['trial']]['CD_H_cm'] < initial:
            original_plan=read(OUTPUT/'body_target_evaluation_plan.json')['models']
            previous=next(m for m in original_plan if m['name']==prior_selected['trial'])
            if previous['full_checkpoint'] is not None:
                import torch
                saved=torch.load(previous['full_checkpoint'],map_location='cpu',mmap=True)
                if saved['step']>0:
                    state('not_run_prior_selection_already_beats_init_v6')
                    return
        assert len(prior_exports['samples']) == 30
        assert all(sha256(Path(v['path'])) == v['sha256'] for v in prior_exports['samples'].values())
        checks = read(OUTPUT / 'native_surface_dataset_checks.json')
        assert checks['default_dataset_unchanged'] and checks['checked']['full_training']['frames'] == 1107
        assert sha256(OUTPUT / 'native_surface_dataset_checks.json') == plan['dataset_checks_sha256']
        assert sha256(OUTPUT / 'native_surface_target_preparation.json') == plan['target_preparation_sha256']
        pilot_plan=read(OUTPUT/'body_target_surface_pilot_plan.json')
        assert pilot_plan['input_view']==input_view and pilot_plan['evaluation_protocol']==protocol
        pilot_name=pilot_plan['config']['name']
        pilot_done = read(OUTPUT / 'runs'/pilot_name/'complete.json')
        assert pilot_done['successful_steps'] > 0
        assert pilot_done['last_metrics']['cd_proxy_target_space'] == 'native_MHR_GT'
        assert pilot_done['last_metrics']['evaluation_protocol']==protocol
        assert read(OUTPUT / 'runs'/pilot_name/'started.json')['world_size'] == 2
        # Exercise the full best/last validation entry point with discarded
        # pilot weights before starting any of the three actual searches.
        tag='strict_native_surface_pilot_posthoc'
        pilot_checkpoint=OUTPUT/'checkpoints'/pilot_name/'last.pt'
        command=[str(runtime.PYTHON),'-u',str(runtime.ROOT/'scripts/evaluate_multihmr.py'),
            'track2-validate','--checkpoint',str(pilot_checkpoint),'--validation-tag',tag,
            '--input-view',input_view,'--native-surface-target','--strict-target-mask','--target-body-overlap','--batch-size','2']
        state('checking_actual_strict_native_surface_posthoc_validation')
        runtime.run(command,[6],tag,deadline)
        posthoc=read(OUTPUT/'validation_smplx'/tag/'metrics.json')
        assert posthoc['checkpoint_sha256']==sha256(pilot_checkpoint)
        assert posthoc['step']==pilot_done['step'] and posthoc['step']>0
        assert posthoc['evaluation_protocol']==protocol and posthoc['strict_target_mask']
        assert posthoc['target_mask_dilation_kernel']==0 and posthoc['cd_proxy_target_space']=='native_MHR_GT'
        assert posthoc['target_body_overlap'] and posthoc['body_box_rule_sha256']==sha256(OUTPUT/'body_target_gate_Train_rule.json')
        assert not posthoc['test_used'] and not posthoc['GT_used_for_detection_selection']
        assert set(posthoc['per_sequence'])==set(read(OUTPUT/'manifest.json')['validation'])
        assert all(sha256(Path(v['path']))==v['sha256'] for v in prior_exports['samples'].values())
        save_json(OUTPUT/'strict_surface_posthoc_preflight.json',dict(status='passed_actual_GPU_validation',
            checkpoint_sha256=posthoc['checkpoint_sha256'],metrics_sha256=sha256(OUTPUT/'validation_smplx'/tag/'metrics.json'),
            Test_used=False,pilot_weights_discarded=True,source_exports_unchanged=True,checked_unix=time.time()))
        for label, value in [('selected', prior_selected), ('test', prior_test), ('exports', prior_exports)]:
            destination = OUTPUT / ('native_surface_prior_' + label + '.json')
            if destination.exists():
                assert read(destination) == value
            else:
                save_json(destination, value)
        for config in trials:
            save_json(OUTPUT / 'configs' / (config['name'] + '.json'), config)
        full_count = len(read(OUTPUT / 'work/full_training_index.json'))
        fit_count = len(read(OUTPUT / 'work/training_index.json'))
        assert (fit_count, full_count) == (851, 1107)
        state('training_native_surface_trials')

        def worker(index):
            cfg = trials[index]
            name = cfg['name']
            destination = OUTPUT / 'native_surface_refit_choices' / (name + '.json')
            if destination.exists():
                value = read(destination)
                assert sha256(Path(value['refit']['checkpoint'])) == value['refit']['checkpoint_sha256']
                return name, value
            runtime.run_training(runtime.train_command(name), index, name, deadline)
            done = read(OUTPUT / 'runs' / name / 'complete.json')
            assert done['successful_steps'] > 0 and done['last_metrics']['evaluation_protocol'] == protocol
            assert done['last_metrics']['cd_proxy_target_space'] == 'native_MHR_GT'
            assert read(OUTPUT / 'runs' / name / 'started.json')['world_size'] == 2
            candidates = {}
            with (OUTPUT / 'locks/fixed_gpu6_evaluation.lock').open('a') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX)
                for kind in ['best', 'last']:
                    checkpoint = OUTPUT / 'checkpoints' / name / (kind + '.pt')
                    tag = name + '_' + kind + '_native_surface'
                    command = [str(runtime.PYTHON), '-u', str(runtime.ROOT / 'scripts/evaluate_multihmr.py'),
                               'track2-validate', '--checkpoint', str(checkpoint), '--validation-tag', tag,
                               '--input-view', input_view, '--native-surface-target','--strict-target-mask','--target-body-overlap']
                    runtime.run(command, [6], 'native_surface_proxy_' + tag, deadline)
                    value = read(OUTPUT / 'validation_smplx' / tag / 'metrics.json')
                    assert value['checkpoint_sha256'] == sha256(checkpoint)
                    assert value['evaluation_protocol'] == protocol and value['cd_proxy_target_space'] == 'native_MHR_GT'
                    assert value['strict_target_mask'] and value['target_mask_dilation_kernel']==0
                    assert value['target_body_overlap'] and value['body_box_rule_sha256']==sha256(OUTPUT/'body_target_gate_Train_rule.json')
                    assert not value['test_used'] and not value['GT_used_for_detection_selection']
                    candidates[kind] = value
            eligible = [k for k, v in candidates.items() if v['coverage'] >= .95]
            assert eligible
            kind = min(eligible, key=lambda k: (candidates[k]['cd_h_smplx_proxy_cm'], ['best', 'last'].index(k)))
            choice = dict(selected_candidate=kind, checkpoint=candidates[kind]['checkpoint'], metrics=candidates[kind],
                          candidates=candidates, test_used=False)
            save_json(OUTPUT / 'native_surface_checkpoint_choices' / (name + '.json'), choice)
            updates = choice['metrics']['step']
            refit_seconds = 0
            if updates == 0:
                path = Path(choice['checkpoint'])
                refit = dict(checkpoint=str(path), checkpoint_sha256=sha256(path), full_train_refit=True,
                             refit_performed_this_wave=False, reason='Selected unchanged original initialization')
            else:
                refit_name = 'refit_' + name
                hours = min(cfg['refit_hours_max'], (10800 - done['elapsed_seconds'] - 120) / 3600)
                assert hours > 0
                refit_cfg = dict(cfg, name=refit_name, training_role='full_training', fixed_full_train=True,
                                 hours=hours, max_steps=min(5000, max(1, round(updates * full_count / fit_count))),
                                 max_epochs=0, early_stop_patience=0, selection_metric='pa_mpjpe_mm',
                                 recipe_selection=choice['metrics'])
                save_json(OUTPUT / 'configs' / (refit_name + '.json'), refit_cfg)
                runtime.run_training(runtime.train_command(refit_name), index, refit_name, deadline)
                completion = read(OUTPUT / 'runs' / refit_name / 'complete.json')
                assert completion['successful_steps'] > 0 and not completion['validation_used_for_selection']
                assert read(OUTPUT / 'runs' / refit_name / 'started.json')['world_size'] == 2
                path = OUTPUT / 'checkpoints' / refit_name / 'best.pt'
                refit_seconds = completion['elapsed_seconds']
                refit = dict(checkpoint=str(path), checkpoint_sha256=sha256(path), full_train_refit=True,
                             refit_performed_this_wave=True, complete=completion)
            recovery_seconds=recovery.get('previous_attempts',{}).get(name,{}).get('additional_nonresumed_attempt_seconds',0)
            assert done['elapsed_seconds'] + refit_seconds + recovery_seconds < 10800
            result = dict(search=choice, refit=refit, search_seconds=done['elapsed_seconds'], refit_seconds=refit_seconds,
                          training_lineage_seconds=0,recovery_attempt_seconds=recovery_seconds, complete_training_cost_below3h=True,
                          test_used_for_selection=False, input_view=input_view, native_surface_target=True,
                          strict_target_mask=True,evaluation_protocol=protocol)
            save_json(destination, result)
            return name, result

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, i) for i in range(3)]
            try:
                finished, _ = wait(futures, return_when=FIRST_EXCEPTION)
                for future in finished:
                    future.result()
                results = dict(f.result() for f in futures)
            except BaseException:
                runtime.stop_children()
                raise
        save_json(OUTPUT / 'native_surface_refit_mappings.json', results)
        state('all_native_surface_searches_and_Train20_refits_complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save_json(OUTPUT / 'native_surface_training_error.json', dict(error=repr(error), time=time.time()))
        runtime.stop_children()
        raise

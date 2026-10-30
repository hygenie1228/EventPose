"""Predeclared CD-H-selected continuation search within the original 12h window.

Starts only after the first pipeline is genuinely terminal and verified. Uses
the already frozen Train/Test split and never tunes on fine-tuned Test scores.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
import finalize_track2_experiments as finish
import run_track2_experiments as runtime

NAMES=['cd_head_abs25','cd_head_abs100','cd_last2_abs100']
STATE_PREFIX='cd_wave2'
ARCHIVE_NAME='wave1_summary'
WAVE_NUMBER=2
MIN_REMAINING_HOURS=3


def read(path):return json.loads(Path(path).read_text())


def phase(state,**extra):
    save_json(OUTPUT/(STATE_PREFIX+'_status.json'),dict(status=state,pid=os.getpid(),updated_unix=time.time(),**extra))


def merge_cd_selection(old,current):
    combined=dict(current)
    for key in ['trials','candidates']:
        combined[key]=dict(old.get(key,{}));combined[key].update(current.get(key,{}))
    combined['includes_two_search_waves']=True
    save_json(OUTPUT/'internal_cd_h_selection.json',combined)


def main(previous_pid,plan_name='cd_wave2_plan.json'):
    global NAMES,STATE_PREFIX,ARCHIVE_NAME,WAVE_NUMBER,MIN_REMAINING_HOURS
    settings=read(OUTPUT/plan_name)
    NAMES=[cfg['name'] for cfg in settings['trials']]
    STATE_PREFIX=settings.get('state_prefix','cd_wave2')
    runtime.REGISTRY_NAME=STATE_PREFIX+'_jobs.json'
    ARCHIVE_NAME=settings.get('previous_archive','wave1_summary')
    WAVE_NUMBER=settings.get('search_wave',2)
    MIN_REMAINING_HOURS=settings.get('minimum_remaining_hours',3)
    with (OUTPUT/(STATE_PREFIX+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        signal.signal(signal.SIGINT,runtime.stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        plan=settings
        assert (plan.get('configs_fixed_before_any_finetuned_test_inference')
            or plan.get('exploratory_plan_frozen_before_candidate_training')
            or plan.get('all_candidate_configs_fixed_before_their_training'))
        assert plan.get('test_used_for_selection',False) is False
        phase('queued_after_first_pipeline',previous_pipeline_pid=previous_pid)
        background=None
        def worker(index):
            name=NAMES[index]
            if not (OUTPUT/'runs'/name/'complete.json').exists():
                runtime.run_training(runtime.train_command(name),index,name,deadline)
            complete=read(OUTPUT/'runs'/name/'complete.json')
            assert complete['successful_steps']>0 and complete['selection_metric']=='cd_h_smplx_proxy_cm'
        def overlap_if_ready():
            nonlocal background
            if background is not None or time.time()>=deadline-MIN_REMAINING_HOURS*3600:return
            # Wave2 can overlap initial evaluation now. Wave3 waits until
            # wave2 finishes initial training, then overlaps validation/refit work.
            ready=WAVE_NUMBER==2
            if WAVE_NUMBER==3:
                state_file=OUTPUT/'cd_wave2_status.json'
                ready=state_file.exists() and read(state_file)['status'] in ['validating_cd_h_candidates','evaluating_cd_h_candidates_on_frozen_test']
            pilot_name=plan.get('pilot_name','cd_metric_pilot')
            pilot_receipt=OUTPUT/'runs'/pilot_name/'complete.json'
            if ready and pilot_receipt.exists():
                checked=read(pilot_receipt);assert checked['successful_steps']>0
                pool=ThreadPoolExecutor(max_workers=len(NAMES))
                futures=[pool.submit(worker,index) for index in range(len(NAMES))]
                background=(pool,futures)
                phase('training_while_previous_evaluation',previous_pipeline_pid=previous_pid,concurrent_training_and_evaluation=True)
        overlap_if_ready()
        identity=finish.process_identity(previous_pid)
        while identity is not None and finish.process_identity(previous_pid)==identity:
            overlap_if_ready()
            if time.time()>=deadline:raise TimeoutError('Deadline while first pipeline is live')
            time.sleep(min(30,max(1,deadline-time.time())))
        prior=read(OUTPUT/'supervisor.json')
        assert prior['status']=='artifacts_ready_for_final_audit',prior
        assert len(read(OUTPUT/'export_receipts.json')['samples'])==30
        # Completed background search does not make refit/Test/export free.
        # Preserve the previous fully verified30 outputs when the remaining
        # original window cannot fund another complete comparison stage.
        if time.time()>=deadline-MIN_REMAINING_HOURS*3600:
            if background is not None:
                pool,futures=background
                try:
                    for future in futures:future.result()
                finally:pool.shutdown(wait=True)
            phase('search_complete_additional_refit_skipped_insufficient_budget',
                minimum_remaining_hours=MIN_REMAINING_HOURS,
                previous_selected_trial=read(OUTPUT/'selected.json')['trial'],
                completed_search_trials=[n for n in NAMES if (OUTPUT/'runs'/n/'complete.json').exists()])
            return
        archive=OUTPUT/ARCHIVE_NAME;archive.mkdir(exist_ok=True)
        for name in ['selected.json','comparison.json','evaluation_checkpoints.json','internal_cd_h_selection.json',
                     'validation_mhr_comparison.json','export_receipts.json','final_audit.json']:
            destination=archive/name
            if not destination.exists():shutil.copyfile(OUTPUT/name,destination)
        previous_selection=read(archive/'selected.json')
        previous_comparison=read(archive/'comparison.json')
        previous_validation=read(archive/'validation_mhr_comparison.json')
        previous_cd=read(archive/'internal_cd_h_selection.json')
        previous_checkpoints=read(archive/'evaluation_checkpoints.json')
        manifest=read(OUTPUT/'manifest.json')
        if background is None and time.time()>=deadline-MIN_REMAINING_HOURS*3600:
            phase('skipped_insufficient_original_budget',minimum_remaining_hours=MIN_REMAINING_HOURS)
            return
        pilot_name=plan.get('pilot_name','cd_metric_pilot')
        pilot=dict(plan['trials'][0],name=pilot_name,pilot_steps=20,batch_size=2,hours=.2,
            early_stop_patience=0,max_epochs=0,deadline_unix=deadline)
        if not (OUTPUT/'runs'/pilot_name/'complete.json').exists():
            save_json(OUTPUT/'configs'/(pilot_name+'.json'),pilot)
        phase('cd_h_training_pilot')
        finish.status('cd_h_training_pilot')
        if not (OUTPUT/'runs'/pilot_name/'complete.json').exists():
            runtime.run_training(runtime.train_command(pilot_name),0,pilot_name,deadline)
        tested=read(OUTPUT/'runs'/pilot_name/'complete.json')
        assert tested['successful_steps']>0 and tested['selection_metric']=='cd_h_smplx_proxy_cm'
        phase('training_cd_h_candidates',candidate_count=len(NAMES))
        finish.status('training_cd_h_candidates',candidate_count=len(NAMES))
        if background is not None:
            pool,futures=background
            try:
                for future in futures:future.result()
            finally:pool.shutdown(wait=True)
        else:finish.parallel(worker,len(NAMES))
        finish.TRIALS=NAMES
        phase('validating_cd_h_candidates')
        finish.select_smplx_checkpoints(manifest,deadline)
        current_cd=read(OUTPUT/'internal_cd_h_selection.json')
        merge_cd_selection(previous_cd,current_cd)
        finish.validation_mhr(manifest,deadline)
        current_validation=read(OUTPUT/'validation_mhr_comparison.json')
        finish.full_train_refits(manifest,deadline)
        current_mapping=read(OUTPUT/'evaluation_checkpoints.json')
        merged_mapping=dict(previous_checkpoints);merged_mapping.update(current_mapping)
        save_json(OUTPUT/'evaluation_checkpoints.json',merged_mapping)
        all_validation=dict(previous_validation['models']);all_validation.update(current_validation['models'])
        # Each model is chosen from its own held-out validation before refitting.
        # All scores here use the identical four full Train-validation videos.
        eligible=['original']
        all_cd=read(OUTPUT/'internal_cd_h_selection.json')
        minimum_coverage=read(OUTPUT/'target_foreground_protocol.json')['minimum_target_detection_coverage']
        for name in list(previous_cd['trials'])+NAMES:
            if name in NAMES and all_cd['trials'][name]['metrics']['step']==0:continue
            if all_cd['trials'][name]['metrics']['coverage']>=minimum_coverage:eligible.append(name)
        winner=min(eligible,key=lambda name:(all_validation[name]['CD_H_cm'],all_validation[name]['ACC_H_cm_per_frame2'],eligible.index(name)))
        merged_validation=dict(current_validation,models=all_validation,selected_trial=winner,
            test_used_for_selection=False,eligible_trials=eligible,minimum_target_detection_coverage=minimum_coverage,
            selection='Train-only full-frame MHR CD-H primary; ACC-H tie-break; multiple frozen search waves')
        save_json(OUTPUT/'validation_mhr_comparison.json',merged_validation)
        save_json(OUTPUT/'selected.json',dict(trial=winner,selection=merged_validation['selection'],
            test_used_for_selection=False,mhr_validation=merged_validation,
            selected_unix=time.time(),wave2_plan_created_unix=plan['created_unix'],
            checkpoint=merged_mapping[winner]['checkpoint'] if winner!='original' else None))
        phase('evaluating_cd_h_candidates_on_frozen_test',selected_trial=winner)
        finish.run_test(manifest,deadline)
        current_test=read(OUTPUT/'comparison.json')
        models=dict(previous_comparison['models']);models.update(current_test['models'])
        save_json(OUTPUT/'comparison.json',dict(current_test,models=models,search_waves=WAVE_NUMBER,
            test_used_for_selection=False,compared_finetuned_recipes=len(models)-1))
        if winner==previous_selection['trial']:
            receipts=read(archive/'export_receipts.json')
            for sample,item in receipts['samples'].items():assert sha256(Path(item['path']))==item['sha256']
            save_json(OUTPUT/'export_receipts.json',receipts)
        else:
            finish.exports(winner,manifest,deadline)
        receipts=read(OUTPUT/'export_receipts.json')['samples']
        assert len(receipts)==30
        for name in NAMES:
            cfg=read(OUTPUT/'configs'/(name+'.json'))
            initial=read(OUTPUT/'runs'/name/'complete.json')
            refit=merged_mapping[name].get('complete',{})
            assert initial['elapsed_seconds']+refit.get('elapsed_seconds',0)+cfg['bootstrap_elapsed_seconds']<3*3600
        for sample,item in receipts.items():
            assert sha256(Path(item['path']))==item['sha256']
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==manifest['samples'][sample]['gt_sha256']
        assert all(set(v['per_sequence'])==set(manifest['test']) for v in models.values())
        save_json(OUTPUT/(STATE_PREFIX+'_audit.json'),dict(all_candidate_updates_positive=True,
            online_cd_h_pilot_passed=True,trial_training_with_bootstrap_and_refit_below3h=True,
            all_models_same_test10=True,init_v0_v6_compared=True,
            all30_export_hashes_and_gt_preserved=True,test_used_for_selection=False,
            requires_agent_review=True,completed_unix=time.time()))
        phase('artifacts_ready_for_final_audit',selected_trial=winner)
        finish.status('artifacts_ready_for_final_audit',selected_recipe=winner,exports=30,search_waves=WAVE_NUMBER,
            completed_unix=time.time(),deadline_unix=deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-pipeline-pid',type=int,required=True);p.add_argument('--plan',default='cd_wave2_plan.json')
    a=p.parse_args()
    try:main(a.wait_pipeline_pid,a.plan)
    except BaseException as error:
        save_json(OUTPUT/(STATE_PREFIX+'_error.json'),dict(error=repr(error),time=time.time()))
        runtime.stop_children();finish.report();raise

"""Resolve the predeclared Train-only input rule and queue corrected controls.

The discarded correctness pilot must pass before training can allocate a pair.
This controller does not read Test scores or choose a model from Test results.
"""
from concurrent.futures import FIRST_EXCEPTION,ThreadPoolExecutor,wait
import ast
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json,sha256
from multihmr.target_crop import BODY_PROTOCOLS


def read(path):return json.loads(path.read_text())


def main():
    with (OUTPUT/'locks/strict_surface_preparation.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='strict_surface_preparation_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        def state(status,**extra):
            save_json(OUTPUT/'strict_surface_preparation_status.json',dict(status=status,pid=os.getpid(),
                Test_scores_read=False,updated_unix=time.time(),**extra))
        state('waiting_for_strict_pilot_Train_only_input_resolution')
        pilot_path=OUTPUT/'body_target_surface_pilot_plan.json'
        while not pilot_path.exists():
            if time.time()>deadline-4*3600:
                state('not_run_insufficient_remaining_budget');return
            assert not (OUTPUT/'body_target_evaluation_error.json').exists()
            assert not (OUTPUT/'native_surface_pilot_error.json').exists()
            time.sleep(30)
        pilot=read(pilot_path);rule_path=OUTPUT/'body_target_surface_input_rule.json';rule=read(rule_path)
        assert not pilot['Test_used_for_input_choice'] and pilot['input_rule_sha256']==sha256(rule_path)
        view=pilot['input_view'];protocol=BODY_PROTOCOLS[view]
        assert pilot['evaluation_protocol']==protocol and pilot['config']['strict_target_mask'] and pilot['config']['target_body_overlap']
        destination=OUTPUT/'native_surface_adaptation_plan.json'
        archived=OUTPUT/'invalid_dilated_head_gate/native_surface_adaptation_plan_before_body_resolution.json'
        if not archived.exists():
            assert sha256(destination)==rule['training_plan_template_sha256']
            archived.write_bytes(destination.read_bytes())
        assert sha256(archived)==rule['training_plan_template_sha256']
        template=read(archived);plan=dict(template);trials=[]
        for old in template['trials']:
            config=dict(old,strict_target_mask=True,target_body_overlap=True,target_foreground=False)
            if view=='native_rgb_static_crop':config['fixed_view_plan']=str(OUTPUT/'fixed_view_training_plan.json')
            else:config.pop('fixed_view_plan',None)
            assert config['augment'] and config['weight_decay']==.1 and config['anchor_l2_weight']>0
            assert config['hours']+config['refit_hours_max']<3
            assert not config.get('refit_init_checkpoint') and not config.get('init_checkpoint')
            trials.append(config)
        sources=['tools/run_track2_native_surface_pilot.py','tools/run_track2_native_surface_training.py',
            'tools/finalize_track2_native_surface_wave.py','tools/track2_strict_scope.py',
            'tools/audit_track2_strict_results.py','tools/audit_track2_final_pipeline.py',
            'tools/report_track2_experiments.py','tools/visualize_track2_strict_identity.py']
        for source in sources:ast.parse((runtime.ROOT/source).read_text())
        assert sha256(OUTPUT/'native_surface_dataset_checks.json')==template['dataset_checks_sha256']
        assert sha256(OUTPUT/'native_surface_target_preparation.json')==template['target_preparation_sha256']
        plan.update(trials=trials,input_view=view,evaluation_protocol=protocol,ready_to_launch=True,
            status='strict_input_resolved_waiting_for_actual_pilot',
            strict_target_mask=True,target_body_overlap=True,correction_required=None,
            strict_input_resolution=dict(pilot_plan_sha256=sha256(pilot_path),input_rule_sha256=sha256(rule_path),
                selected_Train_control=pilot['resolved_from_Train_control'],Test_scores_read=False),
            orchestration_preparation=dict(strict_twelve_scope=True,actual_pilot_required=True,
                source_scripts={s:sha256(runtime.ROOT/s) for s in sources},checked_unix=time.time()))
        if read(destination).get('strict_input_resolution'):
            existing=read(destination)
            assert existing['trials']==trials and existing['strict_input_resolution']==plan['strict_input_resolution']
        else:save_json(destination,plan)
        # Preserve expected SIGTERM receipts from the invalid expanded-mask
        # queue. An unexpected error remains a real failure and is not hidden.
        for name in ['native_surface_training_error.json','native_surface_finalization_error.json']:
            path=OUTPUT/name
            if path.exists():
                assert read(path)['error']=='SystemExit(143)',read(path)
                path.rename(OUTPUT/'invalid_dilated_head_gate'/(path.stem+'_before_strict_'+str(time.time_ns())+'.json'))
        state('queued_strict_surface_training_and_evaluation_controllers',input_view=view)
        commands=[('tools/run_track2_native_surface_training.py','strict_surface_training_controller'),
            ('tools/finalize_track2_native_surface_wave.py','strict_surface_evaluation_controller')]
        def run(item):
            source,label=item
            runtime.run([str(runtime.PYTHON),'-u',str(runtime.ROOT/source)],[],label,deadline)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,c) for c in commands]
            try:
                completed,_=wait(futures,return_when=FIRST_EXCEPTION)
                for future in completed:future.result()
                for future in futures:future.result()
            except BaseException:
                runtime.stop_children();raise
        state('strict_surface_controllers_finished_inspect_actual_artifacts')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'strict_surface_preparation_error.json',dict(error=repr(error),updated_unix=time.time()))
        runtime.stop_children();raise

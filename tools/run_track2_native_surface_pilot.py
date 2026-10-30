"""Validate optional native-MHR Train targets after the frozen nine pipelines.

This discarded two-GPU pilot neither selects a model nor changes source outputs.
"""
from pathlib import Path
import fcntl
import json
import os
import signal
import time
import pynvml
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json,sha256
from multihmr.target_crop import BODY_PROTOCOLS


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/native_surface_pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_surface_pilot_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        template=read(OUTPUT/'native_surface_pilot_plan.json')
        name='body_native_surface_two_GPU_pilot'
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        def state(status,**extra):
            save_json(OUTPUT/'native_surface_pilot_status.json',dict(status=status,pid=os.getpid(),
                trial=name,Test_used=False,pilot_weights_discarded=True,source_exports_changed=False,
                updated_unix=time.time(),**extra))
        state('waiting_for_corrected_strict_nine_pipeline_all30_exports')
        while True:
            if time.time()>deadline-4*3600:
                state('not_run_insufficient_remaining_budget');return
            path=OUTPUT/'body_target_evaluation_status.json'
            if path.exists() and read(path)['status']=='all30_ready_for_independent_strict_audit':break
            assert not (OUTPUT/'body_target_evaluation_error.json').exists()
            time.sleep(30)
        # The input rule was frozen before corrected Test scoring. Resolve it
        # from the two unchanged controls' complete Train4 values only.
        rule_path=OUTPUT/'body_target_surface_input_rule.json'
        rule=read(rule_path);prior=read(OUTPUT/'body_target_selected.json')
        assert not prior['test_used_for_selection'] and len(prior['models'])==9
        order=rule['candidate_order'];controls={n:prior['models'][n] for n in order}
        assert all(v['sequences']==4 and v['frames']==2714 for v in controls.values())
        independent=read(OUTPUT/'body_target_independent_Train4_models.json')['models']
        for control in order:
            assert independent[control]['frames']==2714 and independent[control]['accepted_heads_outside_original_mask']==0
            for key in ['CD_H_cm','ACC_H_cm_per_frame2']:
                assert abs(independent[control][key]-controls[control][key])<1e-9
        chosen=min(order,key=lambda n:(controls[n]['CD_H_cm'],controls[n]['ACC_H_cm_per_frame2'],order.index(n)))
        input_view='native_rgb_static_crop' if chosen=='body_original_fixed_crop' else 'native_rgb'
        config=dict(template['config'],name=name,strict_target_mask=True,target_body_overlap=True,target_foreground=False)
        if input_view=='native_rgb_static_crop':config['fixed_view_plan']=str(OUTPUT/'fixed_view_training_plan.json')
        else:config.pop('fixed_view_plan',None)
        plan=dict(config=config,input_view=input_view,evaluation_protocol=BODY_PROTOCOLS[input_view],
            resolved_from_Train_control=chosen,Train_control_metrics=controls,
            input_rule_sha256=sha256(rule_path),Test_used_for_input_choice=False,
            pilot_weights_discarded=True,registered_unix=time.time())
        plan_path=OUTPUT/'body_target_surface_pilot_plan.json'
        if plan_path.exists():
            saved=read(plan_path)
            assert {k:v for k,v in saved.items() if k!='registered_unix'}=={k:v for k,v in plan.items() if k!='registered_unix'}
            plan=saved
        else:save_json(plan_path,plan)
        checks=read(OUTPUT/'native_surface_dataset_checks.json')
        assert checks['checked']['full_training']['frames']==1107 and checks['default_dataset_unchanged']
        receipts=read(OUTPUT/'body_target_export_receipts.json')
        assert len(receipts['samples'])==30
        assert receipts['strict_target_mask'] and receipts['trial']==prior['trial']
        before={s:sha256(Path(v['path'])) for s,v in receipts['samples'].items()}
        assert before=={s:v['sha256'] for s,v in receipts['samples'].items()}
        config_path=OUTPUT/'configs'/(name+'.json')
        if config_path.exists():assert read(config_path)==plan['config']
        else:save_json(config_path,plan['config'])
        resources=read(OUTPUT/'gpu_resources.json');pynvml.nvmlInit()
        completed=OUTPUT/'runs'/name/'complete.json'
        while not completed.exists():
            assert time.time()<deadline-600
            launched=False
            for pair in resources['training_pairs']:
                with (OUTPUT/'locks'/('training_pair_'+'_'.join(map(str,pair))+'.lock')).open('a') as lease:
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if any(pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in pair):continue
                    state('running_actual_two_GPU_native_surface_pilot',gpus=pair)
                    runtime.run(runtime.train_command(name),pair,name,deadline)
                    launched=True;break
            if not launched:time.sleep(30)
        done=read(completed);metrics=done['last_metrics']
        assert done['successful_steps']>0
        assert read(OUTPUT/'runs'/name/'started.json')['world_size']==2
        assert read(OUTPUT/'runs'/name/'configuration.json')['native_surface_target']
        assert read(OUTPUT/'runs'/name/'configuration.json')['strict_target_mask']
        assert metrics['cd_proxy_target_space']=='native_MHR_GT'
        assert metrics['evaluation_protocol']==plan['evaluation_protocol']
        assert metrics['strict_target_mask'] and metrics['target_mask_dilation_kernel']==0
        assert metrics['target_body_overlap'] and metrics['body_box_rule_sha256']==sha256(OUTPUT/'body_target_gate_Train_rule.json')
        assert not metrics['GT_used_for_detection_selection']
        assert {s:sha256(Path(v['path'])) for s,v in receipts['samples'].items()}==before
        state('complete_actual_native_surface_two_GPU_pilot',step=done['step'],
            successful_steps=done['successful_steps'],metrics=metrics,source_sha_unchanged=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_surface_pilot_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

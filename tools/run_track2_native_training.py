"""Run the frozen native-RGB motion trials after controls and exports finish."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/native_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_training_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'native_trajectory_plan.json');trials=plan['trials']
        assert len(trials)==3 and all(not cfg['target_foreground'] and cfg['validation_target_head_gate'] for cfg in trials)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        def state(value,**extra):
            save_json(OUTPUT/'native_training_status.json',dict(status=value,pid=os.getpid(),
                trial_names=[c['name'] for c in trials],test_used=False,updated_unix=time.time(),**extra))
        state('waiting_for_all_Train_controls_and_old30_exports')
        while time.time()<deadline-4*3600:
            native=read(OUTPUT/'native_rgb_control_status.json')
            recovery=read(OUTPUT/'selected_export_recovery_status.json')
            helpers=[read(OUTPUT/f'native_rgb_backfill_gpu{g}_status.json')['status'] for g in [2,4,5]]
            if native['status']=='complete' and recovery['status']=='all30_exports_complete_pending_current_input_control' and all(s=='complete' for s in helpers):break
            for name in ['native_rgb_control_error.json','selected_export_recovery_error.json']:
                assert not (OUTPUT/name).exists(),read(OUTPUT/name)
            time.sleep(20)
        else:raise TimeoutError('Insufficient extended budget to finish the full new wave')
        receipts=read(OUTPUT/'export_receipts.json')
        assert receipts['trial']=='pair_head_cd100' and len(receipts['samples'])==30
        assert all(sha256(Path(v['path']))==v['sha256'] for v in receipts['samples'].values())
        assert read(OUTPUT/'native_relative_motion_loss_checks.json')['double_precision_gradcheck']
        assert read(OUTPUT/'native_target_selection_checks.json')['high_confidence_background_rejected']
        for cfg in trials:
            path=OUTPUT/'configs'/(cfg['name']+'.json')
            if path.exists():assert read(path)==cfg
            else:save_json(path,cfg)
        pilot_name='native_motion_head_gate_pilot'
        pilot=dict(trials[-1],name=pilot_name,pilot_steps=20,batch_size=2,hours=.2,
            early_stop_patience=0,max_epochs=0)
        if not (OUTPUT/'runs'/pilot_name/'complete.json').exists():
            save_json(OUTPUT/'configs'/(pilot_name+'.json'),pilot)
            state('testing_actual_two_GPU_native_motion_and_head_gate')
            runtime.run_training(runtime.train_command(pilot_name),0,pilot_name,deadline)
        tested=read(OUTPUT/'runs'/pilot_name/'complete.json')
        assert tested['successful_steps']>0
        assert tested['last_metrics']['evaluation_protocol']=='native_rgb_target_head_v1'
        assert not tested['last_metrics']['GT_used_for_detection_selection']
        state('training_three_native_RGB_trials',pilot=tested)
        def worker(index):
            cfg=trials[index]
            runtime.run_training(runtime.train_command(cfg['name']),index,cfg['name'],deadline)
            done=read(OUTPUT/'runs'/cfg['name']/'complete.json')
            assert done['successful_steps']>0 and done['selection_metric']=='cd_h_smplx_proxy_cm'
            assert done['elapsed_seconds']+cfg['bootstrap_elapsed_seconds']+cfg['refit_hours_max']*3600<10800
            assert done['last_metrics']['evaluation_protocol']=='native_rgb_target_head_v1'
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,i) for i in range(3)]
            try:
                for future in futures:future.result()
            except BaseException:runtime.stop_children();raise
        save_json(OUTPUT/'native_training_complete.json',dict(trials=[c['name'] for c in trials],
            source_plan_sha256=sha256(OUTPUT/'native_trajectory_plan.json'),test_used=False,completed_unix=time.time()))
        state('all_three_native_searches_complete_pending_heldout_MHR_selection_and_Train20_refit')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_training_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

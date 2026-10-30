"""Three bounded fixed-viewport adaptations plus concurrent frozen baseline.

Starts only after the native-RGB wave has finished all GPU work and exports.
Core training/inference/conversion remain in the user-facing scripts.
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_EXCEPTION
from pathlib import Path
import fcntl,json,os,signal,time
import numpy as np
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command,process_identity
from multihmr.track2_data import OUTPUT,save_json,sha256
PROTOCOL='native_rgb_static_crop_target_head_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/fixed_view_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='fixed_training_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'fixed_view_adaptation_plan.json');manifest=read(OUTPUT/'manifest.json')
        deadline=read(OUTPUT/'budget.json')['deadline_unix'];trials=plan['trials']
        assert len(trials)==3 and all(not c['target_foreground'] and c['fixed_view_plan'] for c in trials)
        def state(value,**extra):save_json(OUTPUT/'fixed_training_status.json',dict(status=value,pid=os.getpid(),
            trials=[c['name'] for c in trials],test_used_for_selection=False,updated_unix=time.time(),**extra))
        state('waiting_for_native_wave_all30_and_frozen_input_preparation')
        while True:
            assert time.time()<deadline-6*3600,'Insufficient remaining budget for full fixed-viewport comparison'
            native=read(OUTPUT/'native_finalization_status.json')
            if native['status']=='native_all30_ready_for_independent_audit' and (OUTPUT/'fixed_view_all30_plan.json').exists():break
            assert not (OUTPUT/'native_finalization_error.json').exists()
            assert process_identity(native['pid']) is not None,'Native finalizer ended before all30 exports'
            assert not (OUTPUT/'fixed_view_preparation_error.json').exists()
            time.sleep(30)
        receipts=read(OUTPUT/'native_export_receipts.json')
        assert len(receipts['samples'])==30
        assert all(sha256(Path(v['path']))==v['sha256'] for v in receipts['samples'].values())
        assert read(OUTPUT/'fixed_view_dataset_checks.json')['validation_records']==256
        assert set(read(OUTPUT/'fixed_view_all30_plan.json')['samples'])==set(manifest['samples'])
        pilot_name='fixed_view_two_GPU_pilot'
        pilot=dict(trials[-1],name=pilot_name,pilot_steps=20,batch_size=2,hours=.2,max_epochs=0,early_stop_patience=0)
        if not (OUTPUT/'runs'/pilot_name/'complete.json').exists():
            save_json(OUTPUT/'configs'/(pilot_name+'.json'),pilot);state('running_actual_fixed_view_two_GPU_pilot')
            runtime.run_training(runtime.train_command(pilot_name),0,pilot_name,deadline)
        pilot_done=read(OUTPUT/'runs'/pilot_name/'complete.json')
        assert pilot_done['successful_steps']>0 and pilot_done['last_metrics']['evaluation_protocol']==PROTOCOL
        assert not pilot_done['last_metrics']['GT_used_for_detection_selection']
        state('training_fixed_view_trials_with_concurrent_original_Test',pilot_successful_steps=pilot_done['successful_steps'])
        def baseline():
            destination=OUTPUT/'fixed_baseline_Test_comparison.json'
            if destination.exists():return read(destination)
            results={};trial='diagnostic_original_static_crop'
            with (OUTPUT/'locks/fixed_gpu6_evaluation.lock').open('a') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                for sample in manifest['test']:
                    for stage in ['track2-infer','track2-convert','track2-score']:
                        assert not runtime.CANCELLED.is_set(),'Fixed-viewport wave cancelled'
                        command=evaluation_command(stage,trial,sample)
                        if stage=='track2-infer':command.extend(['--input-view','native_rgb_static_crop','--original-weights',
                            '--fixed-view-plan',str(OUTPUT/'fixed_view_all30_plan.json')])
                        runtime.run(command,[6],'fixed_baseline_'+stage+'_'+sample,deadline)
                    folder=OUTPUT/'predictions'/trial/sample;value=read(folder/'metrics.json');info=read(folder/'inference.json')
                    assert value['evaluation_protocol']==info['evaluation_protocol']==PROTOCOL
                    assert value['frames']==manifest['samples'][sample]['frames'] and value['full_frame_roster']
                    assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                    assert info['delta_checkpoint'] is None and info['all_accepted_heads_in_target_mask']
                    assert not info['target_foreground_only_input'] and not info['gt_used_for_detection_or_tracking']
                    results[sample]=value
                    save_json(OUTPUT/'fixed_baseline_Test_status.json',dict(status='running',pid=os.getpid(),
                        completed=list(results),test_used_for_selection=False,updated_unix=time.time()))
            result=dict(trial=trial,per_sequence=results,summary={k:float(np.mean([v[k] for v in results.values()]))
                for k in ['CD_H_cm','ACC_H_cm_per_frame2']},evaluation_protocol=PROTOCOL,test_used_for_selection=False,completed_unix=time.time())
            save_json(destination,result);return result
        full=read(OUTPUT/'work/full_training_index.json');fitting=read(OUTPUT/'work/training_index.json')
        def worker(index):
            cfg=trials[index];name=cfg['name'];result_file=OUTPUT/'fixed_refit_choices'/(name+'.json')
            if result_file.exists():
                result=read(result_file);assert sha256(Path(result['refit']['checkpoint']))==result['refit']['checkpoint_sha256'];return name,result
            runtime.run_training(runtime.train_command(name),index,name,deadline)
            done=read(OUTPUT/'runs'/name/'complete.json')
            assert done['successful_steps']>0 and done['last_metrics']['evaluation_protocol']==PROTOCOL
            while not (OUTPUT/'fixed_baseline_Test_comparison.json').exists():
                assert not runtime.CANCELLED.is_set(),'Fixed-viewport wave cancelled'
                assert time.time()<deadline-4*3600;time.sleep(20)
            candidates={}
            with (OUTPUT/'locks/fixed_gpu6_evaluation.lock').open('a') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                for kind in ['best','last']:
                    assert not runtime.CANCELLED.is_set(),'Fixed-viewport wave cancelled'
                    path=OUTPUT/'checkpoints'/name/(kind+'.pt');tag=name+'_'+kind+'_fixed'
                    command=[str(runtime.PYTHON),'-u',str(runtime.ROOT/'scripts/evaluate_multihmr.py'),
                        'track2-validate','--checkpoint',str(path),'--validation-tag',tag,'--input-view','native_rgb_static_crop']
                    runtime.run(command,[6],'fixed_proxy_'+tag,deadline)
                    value=read(OUTPUT/'validation_smplx'/tag/'metrics.json')
                    assert value['evaluation_protocol']==PROTOCOL and not value['GT_used_for_detection_selection'] and not value['test_used']
                    assert value['checkpoint_sha256']==sha256(path);candidates[kind]=value
            eligible=[kind for kind,value in candidates.items() if value['coverage']>=.95];assert eligible
            kind=min(eligible,key=lambda k:(candidates[k]['cd_h_smplx_proxy_cm'],['best','last'].index(k)))
            choice=dict(selected_candidate=kind,checkpoint=candidates[kind]['checkpoint'],metrics=candidates[kind],
                candidates=candidates,test_used=False);save_json(OUTPUT/'fixed_checkpoint_choices'/(name+'.json'),choice)
            updates=choice['metrics']['step'];refit_seconds=0
            if updates==0:
                path=Path(choice['checkpoint']);refit=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),full_train_refit=True,
                    refit_performed_this_wave=False,reason='Train validation selected original initialization; no Train16-specific updates')
            else:
                refit_name='refit_'+name;hours=min(cfg['refit_hours_max'],(10800-done['elapsed_seconds']-120)/3600);assert hours>0
                refit_cfg=dict(cfg,name=refit_name,training_role='full_training',fixed_full_train=True,hours=hours,
                    max_steps=min(5000,max(1,round(updates*len(full)/len(fitting)))),max_epochs=0,early_stop_patience=0,
                    selection_metric='pa_mpjpe_mm',recipe_selection=choice['metrics'])
                save_json(OUTPUT/'configs'/(refit_name+'.json'),refit_cfg)
                runtime.run_training(runtime.train_command(refit_name),index,refit_name,deadline)
                completion=read(OUTPUT/'runs'/refit_name/'complete.json')
                assert completion['successful_steps']>0 and not completion['validation_used_for_selection']
                path=OUTPUT/'checkpoints'/refit_name/'best.pt';refit_seconds=completion['elapsed_seconds']
                refit=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),full_train_refit=True,refit_performed_this_wave=True,complete=completion)
            assert done['elapsed_seconds']+refit_seconds<10800
            result=dict(search=choice,refit=refit,search_seconds=done['elapsed_seconds'],refit_seconds=refit_seconds,
                complete_training_cost_below3h=True,test_used_for_selection=False,input_view='native_rgb_static_crop')
            save_json(result_file,result);return name,result
        with ThreadPoolExecutor(max_workers=4) as pool:
            baseline_future=pool.submit(baseline);futures=[pool.submit(worker,i) for i in range(3)]
            try:
                finished,_=wait([baseline_future,*futures],return_when=FIRST_EXCEPTION)
                for future in finished:future.result()
                results=dict(f.result() for f in futures);baseline_future.result()
            except BaseException:runtime.stop_children();raise
        save_json(OUTPUT/'fixed_refit_mappings.json',results)
        state('all_fixed_view_searches_and_Train20_refits_complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'fixed_training_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

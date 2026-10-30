"""Retained native-RGB checkpoint validation and bounded Train20 refits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/native_refits.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_refit_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'native_trajectory_plan.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        manifest=read(OUTPUT/'manifest.json');work=OUTPUT/'work'
        fitting=read(work/'training_index.json');full=fitting+read(work/'validation_index.json')
        assert {x['session'] for x in full}==set(manifest['train'])
        assert not {x['session'] for x in full}&set(manifest['test'])
        if (work/'full_training_index.json').exists():assert read(work/'full_training_index.json')==full
        else:save_json(work/'full_training_index.json',full)
        def state(value):save_json(OUTPUT/'native_refits_status.json',dict(status=value,pid=os.getpid(),
            trials=[cfg['name'] for cfg in plan['trials']],test_used_for_selection=False,updated_unix=time.time()))
        state('waiting_for_native_search_checkpoints_and_GPU6_baseline')
        def worker(index):
            cfg=plan['trials'][index];name=cfg['name'];complete=OUTPUT/'runs'/name/'complete.json'
            result_file=OUTPUT/'native_refit_choices'/(name+'.json')
            if result_file.exists():
                result=read(result_file);assert sha256(Path(result['refit']['checkpoint']))==result['refit']['checkpoint_sha256'];return name,result
            while time.time()<deadline-3*3600:
                baseline=OUTPUT/'native_baseline_Test_comparison.json'
                if complete.exists() and baseline.exists():break
                for filename in ['native_training_error.json','native_baseline_Test_error.json']:
                    assert not (OUTPUT/filename).exists(),read(OUTPUT/filename)
                time.sleep(30)
            else:raise TimeoutError('Native search/baseline did not finish within authorized budget')
            done=read(complete);assert done['successful_steps']>0
            candidates={}
            with (OUTPUT/'locks/native_gpu6_evaluation.lock').open('a') as gpu_lease:
                fcntl.flock(gpu_lease,fcntl.LOCK_EX)
                for kind in ['best','last']:
                    checkpoint=OUTPUT/'checkpoints'/name/(kind+'.pt');tag=name+'_'+kind+'_native'
                    command=[str(runtime.PYTHON),'-u',str(runtime.ROOT/'scripts/evaluate_multihmr.py'),
                        'track2-validate','--validation-tag',tag,'--checkpoint',str(checkpoint),'--input-view','native_rgb']
                    runtime.run(command,[6],'native_proxy_'+tag,deadline)
                    value=read(OUTPUT/'validation_smplx'/tag/'metrics.json')
                    assert value['evaluation_protocol']=='native_rgb_target_head_v1'
                    assert value['detection_selection']=='target_mask_head_highest_confidence'
                    assert not value['GT_used_for_detection_selection'] and not value['test_used']
                    assert value['checkpoint_sha256']==sha256(checkpoint)
                    candidates[kind]=value
            eligible=[kind for kind,value in candidates.items() if value['coverage']>=.95]
            assert eligible,('No native checkpoint passes target coverage',name)
            kind=min(eligible,key=lambda k:(candidates[k]['cd_h_smplx_proxy_cm'],['best','last'].index(k)))
            choice=dict(selected_candidate=kind,checkpoint=candidates[kind]['checkpoint'],metrics=candidates[kind],
                candidates=candidates,test_used=False)
            save_json(OUTPUT/'native_checkpoint_choices'/(name+'.json'),choice)
            updates=choice['metrics']['step'];parent_init=Path(cfg['refit_init_checkpoint'])
            inherited=read(OUTPUT/'runs'/parent_init.parent.name/'complete.json')
            lineage_seconds=cfg['bootstrap_elapsed_seconds']+inherited['elapsed_seconds']
            if updates==0:
                refit=dict(checkpoint=str(parent_init),checkpoint_sha256=sha256(parent_init),
                    full_train_refit=True,refit_performed_this_wave=False,complete=inherited,
                    reason='Native validation selected initialization; use matching all20 Train counterpart')
                refit_seconds=0
            else:
                refit_name='refit_'+name
                hours=min(cfg['refit_hours_max'],(10800-done['elapsed_seconds']-lineage_seconds-120)/3600)
                assert hours>0
                configuration=dict(cfg,name=refit_name,training_role='full_training',fixed_full_train=True,
                    target_foreground=False,hours=hours,max_steps=min(5000,max(1,round(updates*len(full)/len(fitting)))),
                    max_epochs=0,early_stop_patience=0,recipe_selection=choice['metrics'],
                    selection_source_trial=name,selection_metric='pa_mpjpe_mm',
                    init_checkpoint=str(parent_init),deadline_unix=deadline)
                config_path=OUTPUT/'configs'/(refit_name+'.json')
                if config_path.exists():assert read(config_path)==configuration
                else:save_json(config_path,configuration)
                runtime.run_training(runtime.train_command(refit_name),index,refit_name,deadline)
                refit_done=read(OUTPUT/'runs'/refit_name/'complete.json')
                assert refit_done['successful_steps']>0 and not refit_done['validation_used_for_selection']
                assert refit_done['training_role']=='full_training' and refit_done['last_metrics']=={}
                checkpoint=OUTPUT/'checkpoints'/refit_name/'best.pt'
                refit=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                    full_train_refit=True,refit_performed_this_wave=True,complete=refit_done)
                refit_seconds=refit_done['elapsed_seconds']
            assert done['elapsed_seconds']+lineage_seconds+refit_seconds<10800
            result=dict(search=choice,refit=refit,training_lineage_seconds=lineage_seconds,
                search_seconds=done['elapsed_seconds'],refit_seconds=refit_seconds,
                complete_training_cost_below3h=True,input_view='native_rgb',test_used_for_selection=False)
            save_json(result_file,result);return name,result
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,i) for i in range(3)]
            try:results=dict(f.result() for f in futures)
            except BaseException:runtime.stop_children();raise
        save_json(OUTPUT/'native_refit_mappings.json',results)
        state('all_native_Train20_refits_complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_refits_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

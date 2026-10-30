"""Train-only comparison of fixed RGB viewports, frozen Test and final all30."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_EXCEPTION
from pathlib import Path
from queue import Queue,Empty
import fcntl,json,os,signal,subprocess,time
import numpy as np
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
PROTOCOL='native_rgb_static_crop_target_head_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/fixed_finalization.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='fixed_finalization_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'fixed_view_adaptation_plan.json');manifest=read(OUTPUT/'manifest.json')
        native_plan=read(OUTPUT/'native_evaluation_plan.json')
        native_order=[native_plan['initialization_model'],native_plan['legacy_model'],*native_plan['trials']]
        metric_reference=read(OUTPUT/'init_comparison.json')['models']['init_v6']['per_sequence'][manifest['test'][0]]['protocol']
        original='diagnostic_original_static_crop';trials=[c['name'] for c in plan['trials']];names=[original,*trials]
        deadline=read(OUTPUT/'budget.json')['deadline_unix'];gpus=runtime.healthy_gpus()
        def state(value,**extra):save_json(OUTPUT/'fixed_finalization_status.json',dict(status=value,pid=os.getpid(),
            models=names,test_used_for_selection=False,updated_unix=time.time(),**extra))
        def wait_for(path,reserve=600):
            while not path.exists():
                assert time.time()<deadline-reserve
                assert not (OUTPUT/'fixed_training_error.json').exists()
                training=read(OUTPUT/'fixed_training_status.json')
                assert finish.process_identity(training['pid']) is not None,'Fixed training supervisor ended before required artifact'
                time.sleep(20)
            return read(path)
        def checkpoint(name,full=False):
            if name==original:return None
            value=read(OUTPUT/('fixed_refit_choices' if full else 'fixed_checkpoint_choices')/(name+'.json'))
            return value['refit']['checkpoint'] if full else value['checkpoint']
        def verify(trial,sample,ckpt):
            folder=OUTPUT/'predictions'/trial/sample
            if not (folder/'metrics.json').exists():return None
            value=read(folder/'metrics.json');info=read(folder/'inference.json')
            assert value['evaluation_protocol']==info['evaluation_protocol']==PROTOCOL
            assert info['input_view']=='native_rgb_static_crop' and not info['target_foreground_only_input']
            assert info.get('depth_decoder','published')=='published' and info['all_accepted_heads_in_target_mask']
            assert not info['gt_used_for_detection_or_tracking'] and info['checkpoint_sha256']==(sha256(Path(ckpt)) if ckpt else None)
            assert info['static_crop_box']==read(OUTPUT/'fixed_view_all30_plan.json')['samples'][sample]['box']
            assert value['conversion_revision']==2 and value['full_frame_roster']
            assert value['protocol']==metric_reference
            assert value['frames']==manifest['samples'][sample]['frames'] and value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
            return value
        def stages(trial,sample,ckpt,gpu,score=True):
            if score and verify(trial,sample,ckpt) is not None:return
            for stage in ['track2-infer','track2-convert']+(['track2-score'] if score else []):
                assert not runtime.CANCELLED.is_set()
                command=finish.evaluation_command(stage,trial,sample)
                if stage=='track2-infer':
                    command.extend(['--input-view','native_rgb_static_crop','--fixed-view-plan',str(OUTPUT/'fixed_view_all30_plan.json')])
                    command.extend(['--checkpoint',ckpt] if ckpt else ['--original-weights'])
                runtime.run(command,[gpu],'fixed_final_'+stage+'_'+trial+'_'+sample,deadline)
        def aggregate(values):
            assert set(values)==set(manifest['test']) or set(values)==set(manifest['validation'])
            return dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in values.values()])),
                ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in values.values()])),
                frames=sum(v['frames'] for v in values.values()),sequences=len(values),per_sequence=values)
        state('waiting_for_fixed_original_Test_and_search_choices')
        wait_for(OUTPUT/'fixed_baseline_Test_comparison.json',4*3600)
        validation={};coverage={};pending=list(names)
        while pending:
            progressed=False
            for name in list(pending):
                if name!=original and not (OUTPUT/'fixed_checkpoint_choices'/(name+'.json')).exists():continue
                ckpt=checkpoint(name);trial=name if name==original else 'search_'+name;values={}
                for sample in manifest['validation']:
                    state('validating_fixed_Train4_MHR',current_model=name,current_sample=sample)
                    if verify(trial,sample,ckpt) is None:
                        with (OUTPUT/'locks/fixed_gpu6_evaluation.lock').open('a') as lease:
                            fcntl.flock(lease,fcntl.LOCK_EX);stages(trial,sample,ckpt,6)
                    values[sample]=verify(trial,sample,ckpt)
                validation[name]=aggregate(values)
                coverage[name]=float(np.mean([read(OUTPUT/'predictions'/trial/s/'inference.json')['detection_frame_rate'] for s in values]))
                save_json(OUTPUT/'fixed_validation_mhr_progress.json',dict(models=validation,target_coverage=coverage,
                    test_used_for_selection=False,evaluation_protocol=PROTOCOL,updated_unix=time.time()))
                pending.remove(name);progressed=True
            if not progressed:
                assert time.time()<deadline-3*3600 and not (OUTPUT/'fixed_training_error.json').exists();time.sleep(20)
        native=read(OUTPUT/'native_selected.json');all_models=dict(native['models'],**validation)
        assert set(native['models'])==set(native_order)
        all_coverage=dict(native['target_coverage'],**coverage);order=native_order+names
        eligible=[name for name in order if all_coverage[name]>=.95];assert eligible
        winner=min(eligible,key=lambda name:(all_models[name]['CD_H_cm'],all_models[name]['ACC_H_cm_per_frame2'],order.index(name)))
        input_protocols={name:'native_rgb_target_head_v1' for name in native['models']};input_protocols.update({name:PROTOCOL for name in names})
        selection=dict(trial=winner,models=all_models,eligible_trials=eligible,target_coverage=all_coverage,
            model_order=order,input_protocols=input_protocols,evaluation_protocol=input_protocols[winner],
            selection='Full Train4 MHR CD-H; coverage>=.95; ACC tie-break; search16-Train weights; never Test',
            test_used_for_selection=False,selected_unix=time.time())
        save_json(OUTPUT/'fixed_validation_mhr_comparison.json',dict(models=validation,target_coverage=coverage,
            test_used_for_selection=False,evaluation_protocol=PROTOCOL,completed_unix=time.time()))
        save_json(OUTPUT/'final_selected.json',selection)
        state('waiting_for_all_fixed_Train20_refits',selected_trial=winner)
        mapping=wait_for(OUTPUT/'fixed_refit_mappings.json',2*3600)
        assert set(mapping)==set(trials) and all(v['complete_training_cost_below3h'] for v in mapping.values())
        def parallel(tasks,score):
            queue=Queue()
            for task in sorted(tasks,key=lambda v:manifest['samples'][v[1]]['frames'],reverse=True):queue.put(task)
            def worker(gpu):
                while not runtime.CANCELLED.is_set():
                    try:name,sample=queue.get_nowait()
                    except Empty:return
                    try:stages(name,sample,checkpoint(name,True),gpu,score)
                    finally:queue.task_done()
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures=[pool.submit(worker,gpu) for gpu in gpus]
                try:
                    finished,_=wait(futures,return_when=FIRST_EXCEPTION)
                    for future in finished:future.result()
                    for future in futures:future.result()
                except BaseException:runtime.stop_children();raise
        state('evaluating_frozen_fixed_models_Test10',selected_trial=winner)
        parallel([(name,s) for name in names for s in manifest['test']],True)
        models={name:aggregate({s:verify(name,s,checkpoint(name,True)) for s in manifest['test']}) for name in names}
        save_json(OUTPUT/'fixed_test_comparison.json',dict(models=models,test_used_for_selection=False,
            evaluation_protocol=PROTOCOL,conversion_revision=2,completed_unix=time.time()))
        combined=dict(read(OUTPUT/'native_test_comparison.json')['models'],**models)
        save_json(OUTPUT/'final_test_comparison.json',dict(models=combined,input_protocols=input_protocols,
            test_used_for_selection=False,selected_trial=winner,aggregation='equal sequence weights; every original frame',completed_unix=time.time()))
        if winner in names:
            state('generating_final_fixed_selected_all30',selected_trial=winner)
            parallel([(winner,s) for s in manifest['train']],False)
            receipts={}
            for sample in sorted(manifest['samples']):
                receipts[sample]=finish.export_sample(winner,sample,manifest)
                save_json(OUTPUT/'final_export_receipts.json',dict(trial=winner,samples=receipts,
                    evaluation_protocol=PROTOCOL,test_used_for_selection=False,updated_unix=time.time()))
        else:
            assert winner==native['trial']
            receipts=read(OUTPUT/'native_export_receipts.json')['samples']
            save_json(OUTPUT/'final_export_receipts.json',dict(trial=winner,samples=receipts,
                evaluation_protocol=input_protocols[winner],test_used_for_selection=False,retained_existing_native_exports=True,updated_unix=time.time()))
        assert set(receipts)==set(manifest['samples'])
        for sample,item in receipts.items():
            assert sha256(Path(item['path']))==item['sha256']
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==manifest['samples'][sample]['gt_sha256']
        state('all30_ready_for_independent_final_audit',selected_trial=winner,exports=30)
        subprocess.run([str(runtime.PYTHON),str(runtime.ROOT/'tools/report_track2_experiments.py')],check=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'fixed_finalization_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

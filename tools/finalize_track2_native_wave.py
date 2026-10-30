"""Select on native-RGB Train MHR, compare Test10, and export all30."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue,Empty
import fcntl,json,os,signal,subprocess,time
import numpy as np
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
PROTOCOL='native_rgb_target_head_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/native_finalization.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_finalization_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'native_evaluation_plan.json');manifest=read(OUTPUT/'manifest.json')
        deadline=read(OUTPUT/'budget.json')['deadline_unix'];original=plan['initialization_model'];legacy=plan['legacy_model']
        names=[original,legacy]+plan['trials'];gpus=runtime.healthy_gpus()
        def state(value,**extra):save_json(OUTPUT/'native_finalization_status.json',dict(status=value,pid=os.getpid(),
            models=names,test_used_for_selection=False,updated_unix=time.time(),**extra))
        def wait_for(path):
            while not path.exists():
                assert time.time()<deadline-600,'Extended experiment deadline reached'
                for filename in ['native_training_error.json','native_refits_error.json','native_baseline_Test_error.json']:
                    assert not (OUTPUT/filename).exists(),read(OUTPUT/filename)
                time.sleep(30)
            return read(path)
        def checkpoint_for(name,full=False):
            if name==original:return None
            if name==legacy:return plan['legacy_full_train' if full else 'legacy_search']['checkpoint']
            path=OUTPUT/('native_refit_choices' if full else 'native_checkpoint_choices')/(name+'.json')
            value=read(path);return value['refit']['checkpoint'] if full else value['checkpoint']
        def verified(trial,sample,checkpoint):
            folder=OUTPUT/'predictions'/trial/sample
            if not (folder/'metrics.json').exists():return None
            metric=read(folder/'metrics.json');info=read(folder/'inference.json')
            assert info['evaluation_protocol']==metric['evaluation_protocol']==PROTOCOL
            assert not info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert not info['gt_used_for_detection_or_tracking']
            assert info['checkpoint_sha256']==(sha256(Path(checkpoint)) if checkpoint else None)
            assert metric['conversion_revision']==2 and metric['full_frame_roster']
            assert metric['frames']==manifest['samples'][sample]['frames']
            assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
            return metric
        def stages(trial,sample,checkpoint,gpu,score=True):
            if score and verified(trial,sample,checkpoint) is not None:return
            for stage in ['track2-infer','track2-convert']+(['track2-score'] if score else []):
                command=finish.evaluation_command(stage,trial,sample)
                if stage=='track2-infer':
                    command.extend(['--input-view','native_rgb'])
                    command.extend(['--checkpoint',checkpoint] if checkpoint else ['--original-weights'])
                runtime.run(command,[gpu],'native_final_'+stage+'_'+trial+'_'+sample,deadline)
        def aggregate(values):
            assert len(values)==10 or set(values)==set(manifest['validation'])
            return dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in values.values()])),
                ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in values.values()])),
                frames=sum(v['frames'] for v in values.values()),sequences=len(values),per_sequence=values)
        state('waiting_for_native_baseline_and_search_choices')
        wait_for(OUTPUT/'native_baseline_Test_comparison.json')
        validation={};coverage={};pending=list(names)
        while pending:
            progressed=False
            for name in list(pending):
                if name in plan['trials'] and not (OUTPUT/'native_checkpoint_choices'/(name+'.json')).exists():continue
                checkpoint=checkpoint_for(name);trial=original if name==original else 'search_'+name
                values={}
                for sample in manifest['validation']:
                    state('validating_native_Train_MHR',current_model=name,current_sample=sample)
                    if verified(trial,sample,checkpoint) is None:
                        with (OUTPUT/'locks/native_gpu6_evaluation.lock').open('a') as gpu_lease:
                            fcntl.flock(gpu_lease,fcntl.LOCK_EX)
                            stages(trial,sample,checkpoint,6)
                        time.sleep(.1)
                    values[sample]=verified(trial,sample,checkpoint)
                validation[name]=aggregate(values)
                coverage[name]=float(np.mean([read(OUTPUT/'predictions'/trial/s/'inference.json')['detection_frame_rate'] for s in manifest['validation']]))
                save_json(OUTPUT/'native_validation_mhr_progress.json',dict(models=validation,
                    target_coverage=coverage,test_used_for_selection=False,evaluation_protocol=PROTOCOL,updated_unix=time.time()))
                pending.remove(name);progressed=True
            if not progressed:
                assert time.time()<deadline-4*3600
                assert not (OUTPUT/'native_refits_error.json').exists()
                time.sleep(30)
        eligible=[n for n in names if coverage[n]>=.95]
        assert eligible
        winner=min(eligible,key=lambda n:(validation[n]['CD_H_cm'],validation[n]['ACC_H_cm_per_frame2'],names.index(n)))
        selection=dict(trial=winner,models=validation,eligible_trials=eligible,target_coverage=coverage,
            selection='Native RGB Train4 MHR CD-H primary; ACC-H tie-break; coverage>=0.95',
            test_used_for_selection=False,evaluation_protocol=PROTOCOL,selected_unix=time.time())
        save_json(OUTPUT/'native_validation_mhr_comparison.json',selection)
        save_json(OUTPUT/'native_selected.json',selection)
        state('waiting_for_all_native_Train20_refits',selected_trial=winner)
        mappings=wait_for(OUTPUT/'native_refit_mappings.json')
        assert set(mappings)==set(plan['trials'])
        assert all(v['complete_training_cost_below3h'] for v in mappings.values())
        for n in names:
            path=checkpoint_for(n,full=True)
            if path is not None:assert Path(path).exists()
        # These Train-only controls use the same GPU-pair leases as refits.
        # Finish them before launching the five-GPU comparison, which uses all
        # healthy devices at once and must not overlap a queued control.
        state('waiting_for_Train_cause_controls',selected_trial=winner)
        for control in ['static_crop_Train_control','exact_depth_order_Train_control']:
            while not (OUTPUT/(control+'_comparison.json')).exists():
                assert time.time()<deadline-4*3600,'Train cause control exceeded final-evaluation allowance'
                assert not (OUTPUT/(control+'_error.json')).exists(),read(OUTPUT/(control+'_error.json'))
                time.sleep(20)
        def parallel_cases(tasks,score):
            queue=Queue()
            for task in sorted(tasks,key=lambda v:manifest['samples'][v[1]]['frames'],reverse=True):queue.put(task)
            def worker(gpu):
                while True:
                    try:name,sample=queue.get_nowait()
                    except Empty:return
                    try:stages(name,sample,checkpoint_for(name,full=True),gpu,score)
                    finally:queue.task_done()
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures=[pool.submit(worker,gpu) for gpu in gpus]
                try:
                    for future in futures:future.result()
                except BaseException:runtime.stop_children();raise
        state('evaluating_frozen_native_models_Test10',selected_trial=winner)
        parallel_cases([(name,sample) for name in names for sample in manifest['test']],True)
        models={name:aggregate({s:verified(name,s,checkpoint_for(name,full=True)) for s in manifest['test']}) for name in names}
        save_json(OUTPUT/'native_test_comparison.json',dict(models=models,test_used_for_selection=False,
            evaluation_protocol=PROTOCOL,conversion_revision=2,aggregation='equal sequence weights',completed_unix=time.time()))
        state('generating_native_selected_all30',selected_trial=winner)
        parallel_cases([(winner,s) for s in manifest['train']],False)
        archive=OUTPUT/'foreground_final_summary';archive.mkdir(exist_ok=True)
        for filename in ['selected.json','comparison.json','validation_mhr_comparison.json','export_receipts.json']:
            target=archive/filename
            if not target.exists():
                import shutil
                shutil.copyfile(OUTPUT/filename,target)
        receipts={}
        for sample in sorted(manifest['samples']):
            receipts[sample]=finish.export_sample(winner,sample,manifest)
            save_json(OUTPUT/'native_export_receipts.json',dict(trial=winner,samples=receipts,
                evaluation_protocol=PROTOCOL,test_used_for_selection=False,updated_unix=time.time()))
        assert len(receipts)==30
        for sample,value in receipts.items():
            assert sha256(Path(value['path']))==value['sha256']
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==manifest['samples'][sample]['gt_sha256']
        state('native_all30_ready_for_independent_audit',selected_trial=winner,exports=30)
        subprocess.run([str(runtime.PYTHON),str(runtime.ROOT/'tools/report_track2_experiments.py')],check=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_finalization_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

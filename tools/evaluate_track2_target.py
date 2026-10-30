"""Rebuild the initial four-model benchmark using only supplied target masks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,json,os,queue,signal,sys,threading,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json,sha256
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
runtime.REGISTRY_NAME='target_eval_jobs.json'
NAMES=['original','head_strong','last2_strong','last4_strong']


def read(path):return json.loads(Path(path).read_text())


def aggregate(trial,samples,manifest,mappings):
    values={}
    for sample in samples:
        folder=OUTPUT/'predictions'/trial/sample;metric=read(folder/'metrics.json');info=read(folder/'inference.json')
        assert metric['evaluation_protocol']==info['evaluation_protocol']=='target_foreground_head_v1'
        assert metric['conversion_revision']==2 and metric['frames']==manifest['samples'][sample]['frames']
        assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
        assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
        assert info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
        if trial!='original':assert info['checkpoint_sha256']==mappings[trial]['checkpoint_sha256']
        values[sample]=metric
    return dict(CD_H_cm=float(np.mean([x['CD_H_cm'] for x in values.values()])),
        ACC_H_cm_per_frame2=float(np.mean([x['ACC_H_cm_per_frame2'] for x in values.values()])),
        per_sequence=values,sequences=len(samples),frames=sum(x['frames'] for x in values.values()))


def main():
    with (OUTPUT/'target_eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGTERM,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');mappings=read(OUTPUT/'evaluation_checkpoints.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        tasks=queue.Queue();publish_lock=threading.Lock()
        # A corrected original Test result is urgent; the fixed algorithm
        # selects the recipe only from the separate four Train-validation clips.
        roster=[('original',s,True) for s in manifest['test']]
        roster.extend((name,s,False) for name in ['original']+['search_'+n for n in NAMES[1:]] for s in manifest['validation'])
        roster.extend((name,s,True) for name in NAMES[1:] for s in manifest['test'])
        for task in roster:tasks.put(task)
        def status(state,**extra):
            save_json(OUTPUT/'target_eval_status.json',dict(status=state,pid=os.getpid(),updated_unix=time.time(),**extra))
        def publish():
            with publish_lock:
                models={}
                for name in NAMES:
                    if all((OUTPUT/'predictions'/name/s/'metrics.json').exists() for s in manifest['test']):models[name]=aggregate(name,manifest['test'],manifest,mappings)
                save_json(OUTPUT/'target_completed_test_models.json',dict(models=models,conversion_revision=2,evaluation_protocol='target_foreground_head_v1',test_used_for_selection=False,updated_unix=time.time()))
        def command(stage,trial,sample):
            value=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,'--trial',trial,'--sample',sample]
            if stage=='track2-infer' and trial!='original':value.extend(['--checkpoint',mappings[trial]['checkpoint']])
            return value
        def worker(gpu):
            while True:
                try:trial,sample,is_test=tasks.get_nowait()
                except queue.Empty:return
                for stage in ['track2-infer','track2-convert','track2-score']:
                    runtime.run(command(stage,trial,sample),[gpu],'target_'+stage+'_'+trial+'_'+sample,deadline)
                if is_test:publish()
                tasks.task_done()
        status('target_crop_validation_and_test',tasks=tasks.qsize(),gpus=runtime.healthy_gpus())
        save_json(OUTPUT/'supervisor.json',dict(status='target_mask_crop_re_evaluation',pid=os.getpid(),training_continues=True,updated_unix=time.time()))
        with ThreadPoolExecutor(max_workers=len(runtime.healthy_gpus())) as pool:
            futures=[pool.submit(worker,gpu) for gpu in runtime.healthy_gpus()]
            try:
                for future in futures:future.result()
            except BaseException:runtime.stop_children();raise
        validation={name:aggregate('original' if name=='original' else 'search_'+name,manifest['validation'],manifest,mappings) for name in NAMES}
        winner=min(NAMES,key=lambda n:(validation[n]['CD_H_cm'],validation[n]['ACC_H_cm_per_frame2'],NAMES.index(n)))
        validation_report=dict(models=validation,selected_trial=winner,test_used_for_selection=False,conversion_revision=2,evaluation_protocol='target_foreground_head_v1',selection='Target-only Train validation4 full-frame MHR CD-H primary, ACC-H tie-break')
        save_json(OUTPUT/'validation_mhr_comparison.json',validation_report)
        save_json(OUTPUT/'selected.json',dict(trial=winner,selection=validation_report['selection'],test_used_for_selection=False,conversion_revision=2,evaluation_protocol='target_foreground_head_v1',selected_unix=time.time()))
        models=read(OUTPUT/'target_completed_test_models.json')['models'];assert len(models)==4
        save_json(OUTPUT/'comparison.json',dict(models=models,conversion_revision=2,evaluation_protocol='target_foreground_head_v1',aggregation='equal sequence weights',test_used_for_selection=False,official_leaderboard_parity=False,completed_unix=time.time()))
        save_json(OUTPUT/'evaluation_validity.json',dict(protocol='target_foreground_head_v1',prior_unmasked_scores_valid_for_requested_target=False,re_evaluation_required=False,all_initial_models_same_target_protocol=True,updated_unix=time.time()))
        status('target_winner_all30_export',selected_trial=winner)
        receipts=finish.exports(winner,manifest,deadline)
        finish.audit(manifest,winner,receipts)
        status('artifacts_ready_for_final_audit',selected_trial=winner,exports=30)
        finish.status('artifacts_ready_for_final_audit',selected_recipe=winner,exports=30,evaluation_protocol='target_foreground_head_v1',conversion_revision=2)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'target_eval_error.json',dict(error=repr(error),time=time.time()));runtime.stop_children();raise

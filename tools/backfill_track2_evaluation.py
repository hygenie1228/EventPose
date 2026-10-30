"""Fill verified idle GPUs with fixed pending validation/Test cases.

Uses the normal stage locks and caches; never writes global model selection,
checkpoint mappings, comparison summaries, or source exports.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command,process_identity
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    import pynvml
    with (OUTPUT/'locks/evaluation_backfill.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='evaluation_backfill_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children)
        signal.signal(signal.SIGINT,runtime.stop_children)
        supervisor=read(OUTPUT/'supervisor.json')
        assert supervisor['status']=='validating_mhr_train_holdout'
        assert process_identity(supervisor['pid']) is not None
        manifest=read(OUTPUT/'manifest.json');resources=read(OUTPUT/'gpu_resources.json')
        mappings=read(OUTPUT/'evaluation_checkpoints.json')
        validation_trial='search_cd_last2_abs100'
        validation_sample='wet_floor_sign-pick-place-ground-08'
        test_trial='pair_head_cd50';test_sample='baseball_bat-rotate-05'
        assert validation_sample in manifest['validation'] and test_sample in manifest['test']
        assert not (OUTPUT/'predictions'/validation_trial/validation_sample/'inference.json').exists()
        completed=read(OUTPUT/'runs'/('refit_'+test_trial)/'complete.json')
        assert completed['training_role']=='full_training' and not completed['validation_used_for_selection']
        assert completed['successful_steps']>0
        tasks=[dict(gpu=6,trial=validation_trial,sample=validation_sample,
                    checkpoint=mappings[validation_trial]['checkpoint'],purpose='Train-validation backfill'),
               dict(gpu=5,trial=test_trial,sample=test_sample,
                    checkpoint=str(OUTPUT/'checkpoints'/('refit_'+test_trial)/'best.pt'),
                    purpose='Fixed Test case of already completed Train20 refit; no recipe selection')]
        pynvml.nvmlInit()
        for task in tasks:
            assert task['gpu'] in resources['healthy_gpus']
            handle=pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(task['gpu'])])
            assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            task['checkpoint_sha256']=sha256(Path(task['checkpoint']))
        assert tasks[0]['checkpoint_sha256']==mappings[validation_trial]['checkpoint_sha256']
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        def status(value):
            save_json(OUTPUT/'evaluation_backfill_status.json',dict(status=value,pid=os.getpid(),
                tasks=tasks,main_pid=supervisor['pid'],changes_to_selection_or_metric=False,
                test_used_for_selection=False,updated_unix=time.time()))
        status('running_fixed_cases_on_verified_idle_GPUs')
        def worker(task):
            for stage in ['track2-infer','track2-convert','track2-score']:
                command=evaluation_command(stage,task['trial'],task['sample'])
                if stage=='track2-infer':command.extend(['--checkpoint',task['checkpoint']])
                runtime.run(command,[task['gpu']],
                    'evaluation_backfill_'+stage+'_'+task['trial']+'_'+task['sample'],deadline)
            folder=OUTPUT/'predictions'/task['trial']/task['sample']
            info=read(folder/'inference.json');metric=read(folder/'metrics.json')
            assert info['checkpoint_sha256']==task['checkpoint_sha256']
            assert info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert metric['evaluation_protocol']=='target_foreground_head_v1' and metric['conversion_revision']==2
            assert metric['full_frame_roster'] and metric['frames']==manifest['samples'][task['sample']]['frames']
            assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,task) for task in tasks]
            try:
                for future in futures:future.result()
            except BaseException:runtime.stop_children();raise
        status('complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'evaluation_backfill_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

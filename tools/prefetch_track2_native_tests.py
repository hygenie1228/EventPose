"""Use idle healthy GPU pairs for frozen Test cases while Train MHR finishes.

Never changes training, checkpoint choice, hyperparameters or source exports.
Stops dispatch when the main five-GPU Test phase starts; stage locks permit
completed work to be reused and prevent duplicate conversion of the same case.
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_EXCEPTION
from pathlib import Path
from queue import Queue,Empty
import fcntl,json,os,signal,threading,time
import pynvml
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command,process_identity
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/native_test_prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_test_prefetch_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'native_evaluation_plan.json');manifest=read(OUTPUT/'manifest.json')
        resources=read(OUTPUT/'gpu_resources.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        completed=[];mutex=threading.Lock();pynvml.nvmlInit()
        def state(value,**extra):
            with mutex:save_json(OUTPUT/'native_test_prefetch_status.json',dict(status=value,pid=os.getpid(),
                completed=list(completed),test_used_for_selection=False,source_exports_changed=False,updated_unix=time.time(),**extra))
        def main_owns_all_devices():
            value=read(OUTPUT/'native_finalization_status.json')
            return value['status'] in ['evaluating_frozen_native_models_Test10','generating_native_selected_all30','native_all30_ready_for_independent_audit']
        state('waiting_for_two_completed_refits_and_an_idle_pair')
        while not main_owns_all_devices():
            choices={name:read(OUTPUT/'native_refit_choices'/(name+'.json')) for name in plan['trials'] if (OUTPUT/'native_refit_choices'/(name+'.json')).exists()}
            active=set()
            if (OUTPUT/'native_refit_jobs.json').exists():
                for job in read(OUTPUT/'native_refit_jobs.json')['jobs']:
                    if process_identity(job['pid'])==job['start_ticks']:active.add(job['label'])
            if len(choices)>=2 and all(name in choices or 'refit_'+name in active for name in plan['trials']):break
            assert time.time()<deadline-1800 and not (OUTPUT/'native_finalization_error.json').exists()
            assert not (OUTPUT/'native_refits_error.json').exists()
            current=read(OUTPUT/'native_finalization_status.json');assert process_identity(current['pid']) is not None
            time.sleep(30)
        if main_owns_all_devices():state('main_Test_phase_already_started_no_prefetch');return
        checkpoints={plan['legacy_model']:plan['legacy_full_train']['checkpoint']}
        checkpoints.update({name:v['refit']['checkpoint'] for name,v in choices.items()})
        tasks=Queue()
        for sample in sorted(manifest['test'],key=lambda s:manifest['samples'][s]['frames'],reverse=True):
            for name in checkpoints:
                if not (OUTPUT/'predictions'/name/sample/'metrics.json').exists():tasks.put((name,sample))
        save_json(OUTPUT/'native_test_prefetch_plan.json',dict(checkpoints={n:dict(path=p,sha256=sha256(p)) for n,p in checkpoints.items()},
            Test_roster=manifest['test'],input_view='native_rgb',selection_unchanged=True,test_used_for_selection=False,
            registered_unix=time.time(),basis='Fixed native evaluation plan, immutable completed Train20 refits, idle pair allocation only'))
        def worker(gpu):
            while not main_owns_all_devices() and not runtime.CANCELLED.is_set():
                try:name,sample=tasks.get_nowait()
                except Empty:return
                try:
                    for stage in ['track2-infer','track2-convert','track2-score']:
                        assert not runtime.CANCELLED.is_set()
                        command=evaluation_command(stage,name,sample)
                        if stage=='track2-infer':command.extend(['--input-view','native_rgb','--checkpoint',checkpoints[name]])
                        runtime.run(command,[gpu],'native_test_prefetch_'+stage+'_'+name+'_'+sample,deadline)
                    folder=OUTPUT/'predictions'/name/sample;value=read(folder/'metrics.json');info=read(folder/'inference.json')
                    assert value['evaluation_protocol']==info['evaluation_protocol']=='native_rgb_target_head_v1'
                    assert info['checkpoint_sha256']==sha256(checkpoints[name]) and info['all_accepted_heads_in_target_mask']
                    assert not info['target_foreground_only_input'] and not info['gt_used_for_detection_or_tracking']
                    assert value['frames']==manifest['samples'][sample]['frames'] and value['full_frame_roster']
                    assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                    with mutex:completed.append(dict(trial=name,sample=sample,gpu=gpu))
                    state('prefetching_frozen_Test_on_idle_pairs')
                finally:tasks.task_done()
        def pair_worker(pair):
            while not main_owns_all_devices() and not runtime.CANCELLED.is_set() and not tasks.empty():
                assert time.time()<deadline-1800
                assert not (OUTPUT/'native_finalization_error.json').exists() and not (OUTPUT/'native_refits_error.json').exists()
                path=OUTPUT/'locks'/('training_pair_'+'_'.join(map(str,pair))+'.lock')
                with path.open('a') as lease:
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:time.sleep(20);continue
                    busy=any(pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in pair)
                    if busy:time.sleep(20);continue
                    assert time.time()<deadline-1800
                    state('prefetching_frozen_Test_on_idle_pairs',active_pair=pair)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures=[pool.submit(worker,gpu) for gpu in pair]
                        finished,_=wait(futures,return_when=FIRST_EXCEPTION)
                        try:
                            for future in finished:future.result()
                            for future in futures:future.result()
                        except BaseException:runtime.stop_children();raise
                return
        with ThreadPoolExecutor(max_workers=len(resources['training_pairs'])) as pool:
            futures=[pool.submit(pair_worker,pair) for pair in resources['training_pairs']]
            finished,_=wait(futures,return_when=FIRST_EXCEPTION)
            try:
                for future in finished:future.result()
                for future in futures:future.result()
            except BaseException:runtime.stop_children();raise
        state('prefetch_complete_or_handed_to_main_Test_phase')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_test_prefetch_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

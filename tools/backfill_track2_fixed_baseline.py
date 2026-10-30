"""Backfill up to four frozen original-crop Test cases on an idle GPU pair.

No training recipe, checkpoint choice, Test roster, metric or source export changes.
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
    with (OUTPUT/'locks/fixed_baseline_backfill.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='fixed_baseline_backfill_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');plan=read(OUTPUT/'fixed_view_adaptation_plan.json')
        resources=read(OUTPUT/'gpu_resources.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        pair=[4,5];assert pair in resources['training_pairs'];pynvml.nvmlInit()
        completed=[];mutex=threading.Lock();trial='diagnostic_original_static_crop'
        def state(status,**extra):
            with mutex:save_json(OUTPUT/'fixed_baseline_backfill_status.json',dict(status=status,pid=os.getpid(),
                completed=list(completed),source_exports_changed=False,test_used_for_selection=False,
                updated_unix=time.time(),**extra))
        def baseline_ready():return (OUTPUT/'fixed_baseline_Test_comparison.json').exists()
        def refit_requested():return any((OUTPUT/'configs').glob('refit_fixed_*.json'))
        state('waiting_for_dispatched_searches_and_actual_idle_GPU4_5')
        while time.time()<deadline-1800:
            if baseline_ready() or refit_requested():
                state('not_needed_baseline_ready_or_refit_priority');return
            assert not (OUTPUT/'fixed_training_error.json').exists()
            jobs=read(OUTPUT/'fixed_training_jobs.json')
            assert process_identity(jobs['supervisor_pid']) is not None
            events=[json.loads(line) for line in (OUTPUT/'job_history.jsonl').read_text().splitlines()]
            started={e.get('label') for e in events if e.get('event')=='started'}
            if not all(c['name'] in started for c in plan['trials']):time.sleep(30);continue
            with (OUTPUT/'locks/training_pair_4_5.lock').open('a') as lease:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:time.sleep(30);continue
                if any(pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in pair):
                    time.sleep(30);continue
                if baseline_ready() or refit_requested():continue
                current=set()
                for job in read(OUTPUT/'fixed_training_jobs.json')['jobs']:
                    if job['label'].startswith('fixed_baseline_') and '--sample' in job['command']:
                        current.add(job['command'][job['command'].index('--sample')+1])
                samples=[s for s in reversed(manifest['test']) if s not in current
                    and not (OUTPUT/'predictions'/trial/s/'metrics.json').exists()][:4]
                if not samples:state('no_remaining_nonactive_cases');return
                save_json(OUTPUT/'fixed_baseline_backfill_plan.json',dict(trial=trial,samples=samples,gpus=pair,
                    checkpoint=None,input_view='native_rgb_static_crop',fixed_view_plan_sha256=sha256(OUTPUT/'fixed_view_all30_plan.json'),
                    inference_protocol='native_rgb_static_crop_target_head_v1',conversion_revision=2,
                    maximum_cases=4,registered_unix=time.time(),test_used_for_selection=False))
                queue=Queue()
                for sample in samples:queue.put(sample)
                state('backfilling_frozen_original_crop_Test',gpus=pair)
                def worker(gpu):
                    while not runtime.CANCELLED.is_set() and not baseline_ready() and not refit_requested():
                        try:sample=queue.get_nowait()
                        except Empty:return
                        try:
                            for stage in ['track2-infer','track2-convert','track2-score']:
                                command=evaluation_command(stage,trial,sample)
                                if stage=='track2-infer':command.extend(['--input-view','native_rgb_static_crop','--original-weights',
                                    '--fixed-view-plan',str(OUTPUT/'fixed_view_all30_plan.json')])
                                runtime.run(command,[gpu],'fixed_backfill_'+stage+'_'+sample,deadline)
                            folder=OUTPUT/'predictions'/trial/sample;value=read(folder/'metrics.json');info=read(folder/'inference.json')
                            assert value['evaluation_protocol']==info['evaluation_protocol']=='native_rgb_static_crop_target_head_v1'
                            assert info['checkpoint_sha256'] is None and not info['target_foreground_only_input']
                            assert info['all_accepted_heads_in_target_mask'] and not info['gt_used_for_detection_or_tracking']
                            assert value['conversion_revision']==2 and value['full_frame_roster']
                            assert value['frames']==manifest['samples'][sample]['frames']
                            assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                            with mutex:completed.append(dict(sample=sample,gpu=gpu))
                            state('backfilling_frozen_original_crop_Test',gpus=pair)
                        finally:queue.task_done()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(worker,gpu) for gpu in pair]
                    done,_=wait(futures,return_when=FIRST_EXCEPTION)
                    try:
                        for future in done:future.result()
                        for future in futures:future.result()
                    except BaseException:runtime.stop_children();raise
                state('backfill_complete_or_yielded_to_refits');return
        state('not_run_insufficient_remaining_budget')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'fixed_baseline_backfill_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

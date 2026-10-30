"""Train-only full inference/MHR control for the depth-decoder ordering."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from pathlib import Path
import fcntl,json,os,signal,threading,time
import numpy as np
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    import pynvml
    prefix='exact_depth_order_Train_control';trial='diagnostic_original_depth_order'
    with (OUTPUT/'locks'/f'{prefix}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME=prefix+'_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');resources=read(OUTPUT/'gpu_resources.json')
        deadline=read(OUTPUT/'budget.json')['deadline_unix'];results={};mutex=threading.Lock()
        plan_path=OUTPUT/(prefix+'_plan.json')
        plan=dict(trial=trial,samples=manifest['validation'],input_view='native_rgb',original_weights=True,
            depth_decoder='exp_then_focal',published_decoder_unchanged=True,GT_used_for_inference=False,
            test_used=False,registered_unix=time.time(),
            purpose='Exact inference plus frozen MHR-v2 conversion; isolate decoder order without crop or pose/weight changes')
        if not plan_path.exists():save_json(plan_path,plan)
        else:assert read(plan_path)['samples']==manifest['validation']
        def status(value,**extra):
            with mutex:save_json(OUTPUT/(prefix+'_status.json'),dict(status=value,pid=os.getpid(),
                trial=trial,completed=list(results),test_used=False,updated_unix=time.time(),**extra))
        status('waiting_for_static_crop_control_and_free_pair')
        pynvml.nvmlInit();pair=None;lease=None
        while time.time()<deadline-2*3600 and pair is None:
            if not (OUTPUT/'static_crop_Train_control_comparison.json').exists():time.sleep(20);continue
            for candidate in resources['training_pairs']:
                candidate_lease=(OUTPUT/'locks'/('training_pair_'+'_'.join(map(str,candidate))+'.lock')).open('a')
                try:fcntl.flock(candidate_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate_lease.close();continue
                busy=any(pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in candidate)
                if busy:candidate_lease.close();continue
                pair=candidate;lease=candidate_lease;break
            if pair is None:time.sleep(20)
        assert pair is not None,'No free pair before control deadline'
        status('running_exact_decoder_control',gpus=pair)
        tasks=Queue()
        for sample in sorted(manifest['validation'],key=lambda s:manifest['samples'][s]['frames'],reverse=True):tasks.put(sample)
        def worker(gpu):
            while True:
                try:sample=tasks.get_nowait()
                except Empty:return
                assert sample not in manifest['test']
                for stage in ['track2-infer','track2-convert','track2-score']:
                    command=evaluation_command(stage,trial,sample)
                    if stage=='track2-infer':command.extend(['--input-view','native_rgb','--original-weights','--depth-decoder','exp_then_focal'])
                    runtime.run(command,[gpu],prefix+'_'+stage+'_'+sample,deadline)
                folder=OUTPUT/'predictions'/trial/sample;info=read(folder/'inference.json');metric=read(folder/'metrics.json')
                assert info['depth_decoder']=='exp_then_focal' and info['delta_checkpoint'] is None
                assert info['evaluation_protocol']==metric['evaluation_protocol']=='native_rgb_exp_then_focal_target_head_v1'
                assert not info['target_foreground_only_input'] and not info['gt_used_for_detection_or_tracking']
                assert info['all_accepted_heads_in_target_mask'] and metric['full_frame_roster']
                assert metric['frames']==manifest['samples'][sample]['frames']
                assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
                assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                with mutex:results[sample]=metric
                status('running_exact_decoder_control',gpus=pair)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,gpu) for gpu in pair]
                try:
                    for future in futures:future.result()
                except BaseException:runtime.stop_children();raise
        finally:lease.close()
        assert set(results)==set(manifest['validation'])
        summary={key:float(np.mean([v[key] for v in results.values()])) for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
        save_json(OUTPUT/(prefix+'_comparison.json'),dict(summary=summary,per_sequence=results,test_used=False,
            experimental_decoder_variant=True,registered_plan_sha256=sha256(plan_path),completed_unix=time.time()))
        status('complete',summary=summary)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'exact_depth_order_Train_control_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

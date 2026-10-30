"""Train-only native-RGB control, retaining exactly the foreground head gate."""
from pathlib import Path
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    import pynvml
    with (OUTPUT/'locks/native_rgb_control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_rgb_control_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children)
        signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');resources=read(OUTPUT/'gpu_resources.json')
        gpu=6;trial='diagnostic_original_native_rgb'
        assert gpu in resources['healthy_gpus'] and gpu!=0
        own=manifest['train'][resources['healthy_gpus'].index(gpu)::len(resources['healthy_gpus'])]
        assert all((OUTPUT/'predictions/pair_head_cd100'/s/'conversion_v2.json').exists() for s in own)
        pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(gpu)])
        assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle),'GPU must actually be idle'
        samples=manifest['validation'];assert len(samples)==4 and set(samples).isdisjoint(manifest['test'])
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        plan=dict(trial=trial,samples=samples,gpu=gpu,original_weights=True,
            native_rgb_unchanged=True,mask_used_only_for_output_head_gate=True,
            shared_K_resize_padding_head_gate_and_conversion=True,
            test_used=False,changes_to_selection_or_source_exports=False,
            registered_unix=time.time())
        save_json(OUTPUT/'native_rgb_control_plan.json',plan)
        def status(value,**extra):
            save_json(OUTPUT/'native_rgb_control_status.json',dict(status=value,pid=os.getpid(),
                trial=trial,gpu=gpu,samples=samples,updated_unix=time.time(),**extra))
        status('running_Train_validation_native_RGB_control')
        results={}
        for sample in samples:
            for stage in ['track2-infer','track2-convert','track2-score']:
                cmd=evaluation_command(stage,trial,sample)
                if stage=='track2-infer':cmd.extend(['--input-view','native_rgb','--original-weights'])
                runtime.run(cmd,[gpu],'native_rgb_control_'+stage+'_'+sample,deadline)
            folder=OUTPUT/'predictions'/trial/sample
            info=read(folder/'inference.json');metric=read(folder/'metrics.json')
            assert info['evaluation_protocol']==metric['evaluation_protocol']=='native_rgb_target_head_v1'
            assert not info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert info['delta_checkpoint'] is None and not info['gt_used_for_detection_or_tracking']
            assert metric['full_frame_roster'] and metric['frames']==manifest['samples'][sample]['frames']
            assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
            results[sample]=metric
            status('running_Train_validation_native_RGB_control',completed=list(results))
        baseline={s:read(OUTPUT/'predictions/original'/s/'metrics.json') for s in samples}
        summary={condition:{key:sum(v[key] for v in values.values())/4
                    for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
                 for condition,values in [('foreground_input',baseline),('native_RGB_input',results)]}
        save_json(OUTPUT/'native_rgb_control_comparison.json',dict(summary=summary,
            per_sequence=dict(foreground_input=baseline,native_RGB_input=results),
            test_used=False,condition_changed='Only RGB background retained; same target-output gate',
            completed_unix=time.time()))
        status('complete',summary=summary)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_rgb_control_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

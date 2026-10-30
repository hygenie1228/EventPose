"""Frozen native RGB original baseline on Test10, separate from training choice."""
from pathlib import Path
import fcntl,json,os,signal,time
import numpy as np
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    import pynvml
    with (OUTPUT/'locks/native_baseline_Test.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_baseline_Test_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');resources=read(OUTPUT/'gpu_resources.json');gpu=6
        assert read(OUTPUT/'native_rgb_control_status.json')['status']=='complete'
        assert read(OUTPUT/'native_trajectory_plan.json')['native_RGB_Test_scores_not_observed']
        visual=read(OUTPUT/'target_foreground_visuals/diagnostic_original_native_rgb/validation/receipt.json')
        assert not visual['manual_review_pending'] and visual['visual_review']['reviewed_frames']==12
        pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(gpu)])
        assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        trial='diagnostic_original_native_rgb';deadline=read(OUTPUT/'budget.json')['deadline_unix']
        save_json(OUTPUT/'native_baseline_Test_plan.json',dict(samples=manifest['test'],trial=trial,gpu=gpu,
            input_view='native_rgb',original_weights=True,head_gate_fixed_on_Train=True,
            selection_uses_GT_pose=False,test_used_for_selection=False,registered_unix=time.time()))
        rows={}
        def state(value):save_json(OUTPUT/'native_baseline_Test_status.json',dict(status=value,pid=os.getpid(),
            gpu=gpu,trial=trial,completed=list(rows),total=10,test_used_for_selection=False,updated_unix=time.time()))
        state('running_frozen_native_RGB_Test10_baseline')
        for sample in manifest['test']:
            for stage in ['track2-infer','track2-convert','track2-score']:
                command=evaluation_command(stage,trial,sample)
                if stage=='track2-infer':command.extend(['--input-view','native_rgb','--original-weights'])
                runtime.run(command,[gpu],'native_baseline_'+stage+'_'+sample,deadline)
            folder=OUTPUT/'predictions'/trial/sample;value=read(folder/'metrics.json');info=read(folder/'inference.json')
            assert info['evaluation_protocol']==value['evaluation_protocol']=='native_rgb_target_head_v1'
            assert not info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert info['delta_checkpoint'] is None and not info['gt_used_for_detection_or_tracking']
            assert value['full_frame_roster'] and value['frames']==manifest['samples'][sample]['frames']
            assert value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
            rows[sample]=value;state('running_frozen_native_RGB_Test10_baseline')
        summary={k:float(np.mean([v[k] for v in rows.values()])) for k in ['CD_H_cm','ACC_H_cm_per_frame2']}
        save_json(OUTPUT/'native_baseline_Test_comparison.json',dict(trial=trial,summary=summary,per_sequence=rows,
            frames=sum(v['frames'] for v in rows.values()),sequences=10,test_used_for_selection=False,
            evaluation_protocol='native_rgb_target_head_v1',conversion_revision=2,completed_unix=time.time()))
        state('complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_baseline_Test_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

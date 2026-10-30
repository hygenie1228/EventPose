"""Verify the occlusion counterfactual, then reevaluate the same nine weights."""
import fcntl,json,os,signal,time
from pathlib import Path
import numpy as np
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(path.read_text())


def main():
    with (OUTPUT/'locks/body_target_repair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='body_target_repair_jobs.json';runtime.register()
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        def state(status,**extra):save_json(OUTPUT/'body_target_pilot_status.json',dict(status=status,pid=os.getpid(),updated_unix=time.time(),**extra))
        state('waiting_for_existing_gpu_evaluation_to_finish')
        while read(OUTPUT/'strict_evaluation_status.json')['status']!='all30_ready_for_independent_strict_audit':
            assert time.time()<deadline-3600 and not (OUTPUT/'strict_evaluation_error.json').exists()
            time.sleep(15)
        sample='spray_bottle-left-desk-ground-02';trial='body_gate_detector_replay';limit=340
        state('running_actual_body_detector_counterfactual',sample=sample)
        command=[str(runtime.PYTHON),'-u',str(runtime.ROOT/'scripts/evaluate_multihmr.py'),'track2-infer',
            '--trial',trial,'--sample',sample,'--limit',str(limit),'--input-view','native_rgb',
            '--original-weights','--strict-target-mask','--target-body-overlap','--trace-frames','305','306','319']
        runtime.run(command,[6],'body_target_detector_replay',deadline)
        folder=OUTPUT/'smoke_predictions'/trial/sample;trace=read(folder/'detection_trace.json')
        rule=read(OUTPUT/'body_target_gate_Train_rule.json');receipts={}
        with np.load(OUTPUT/'predictions/strict_original_RGB'/sample/'smplx_predictions.npz') as old:
            for frame in [305,306,319]:
                value=trace['frames'][str(frame)];candidates=value['candidates']
                allowed=[c for c in candidates if c['inside_selected_mask']];assert allowed
                previous=max(allowed,key=lambda c:c['score'])
                assert abs(previous['score']-float(old['confidence'][frame]))<1e-6
                assert abs(previous['pelvis_Z_m']-float(old['joints_cam_m'][frame,0,2]))<1e-6
                selected=next(c for c in candidates if c['candidate']==value['selected_candidate'])
                assert selected['inside_selected_mask'] and selected['body_box_iou']>=rule['minimum_body_box_iou']
                if frame in [306,319]:
                    assert previous['body_box_iou']<rule['minimum_body_box_iou']
                    assert selected['candidate']!=previous['candidate']
                receipts[str(frame)]=dict(old_head_only=previous,new_head_and_body=selected)
        save_json(OUTPUT/'body_target_detector_replay_check.json',dict(status='passed',frames=receipts,
            same_forward_candidates=True,weights_changed=False,GT_pose_read=False,Test_scores_read=False,
            Train_rule_sha256=sha256(OUTPUT/'body_target_gate_Train_rule.json'),verified_unix=time.time()))
        state('passed_actual_body_detector_replay',sample=sample)
        runtime.run([str(runtime.PYTHON),'-u',str(runtime.ROOT/'tools/run_track2_strict_evaluation.py'),'--body-target'],
            [],'body_target_nine_evaluation_controller',deadline)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'body_target_repair_error.json',dict(error=repr(error),updated_unix=time.time()))
        runtime.stop_children();raise

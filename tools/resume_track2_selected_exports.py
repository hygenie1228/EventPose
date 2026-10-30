"""Resume only terminal selected-model exports after the budget encoding fix."""
from pathlib import Path
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'locks/selected_export_recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert finish.process_identity(86744) is None
        assert not read(OUTPUT/'target_contenders_jobs.json')['jobs']
        runtime.REGISTRY_NAME='selected_export_recovery_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');selected=read(OUTPUT/'selected.json');trial=selected['trial']
        assert not selected['test_used_for_selection']
        comparison=read(OUTPUT/'comparison.json')
        assert len(comparison['models'])==10
        assert all(set(v['per_sequence'])==set(manifest['test']) for v in comparison['models'].values())
        def state(value,**extra):
            save_json(OUTPUT/'selected_export_recovery_status.json',dict(status=value,pid=os.getpid(),
                selected_trial=trial,original_controller_pid=86744,training_restarted=False,
                test_restarted=False,updated_unix=time.time(),**extra))
        state('resuming_cached_selected_exports')
        finish.exports(trial,manifest,read(OUTPUT/'budget.json')['deadline_unix'])
        receipts=read(OUTPUT/'export_receipts.json');assert receipts['trial']==trial and len(receipts['samples'])==30
        for sample,item in receipts['samples'].items():
            assert sha256(Path(item['path']))==item['sha256']
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==manifest['samples'][sample]['gt_sha256']
        state('all30_exports_complete_pending_current_input_control',exports=30)
        finish.status('artifacts_ready_for_final_audit',selected_recipe=trial,exports=30,
            note='Foreground condition complete; native RGB cause investigation is running separately')
        finish.report()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'selected_export_recovery_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

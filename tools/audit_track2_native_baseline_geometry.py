"""Wait for the live original RGB Test run, then independently audit on CPU."""
from pathlib import Path
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json


def main():
    with (OUTPUT/'locks/native_baseline_geometry_audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='native_baseline_geometry_audit_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        deadline=json.loads((OUTPUT/'budget.json').read_text())['deadline_unix']
        baseline=json.loads((OUTPUT/'native_baseline_Test_status.json').read_text());pid=baseline['pid']
        def identity():
            path=Path(f'/proc/{pid}/stat')
            if not path.exists():return None
            parts=path.read_text().split(') ',1)[1].split()
            return parts[19] if parts[0]!='Z' else None
        tick=identity()
        def state(value):save_json(OUTPUT/'native_baseline_geometry_audit_status.json',dict(status=value,
            pid=os.getpid(),baseline_pid=pid,baseline_start_ticks=tick,gpu_used=False,test_used_for_selection=False,updated_unix=time.time()))
        state('waiting_for_current_original_RGB_Test10')
        while not (OUTPUT/'native_baseline_Test_comparison.json').exists():
            assert tick is not None and identity()==tick,'Baseline supervisor ended without complete Test comparison'
            assert time.time()<deadline-1800
            assert not (OUTPUT/'native_baseline_Test_error.json').exists()
            time.sleep(30)
        state('independently_recomputing_native_Test10_geometry_on_CPU')
        runtime.run([str(runtime.PYTHON),'-u',str(runtime.ROOT/'tools/diagnose_track2_target_geometry.py'),
            '--trial','diagnostic_original_native_rgb','--input-view','native_rgb','--motion-diagnostics'],[],
            'native_baseline_geometry_CPU',deadline)
        result=json.loads((OUTPUT/'selected_geometry_diagnostics/diagnostic_original_native_rgb.json').read_text())
        manifest=json.loads((OUTPUT/'manifest.json').read_text())
        assert set(result['samples'])==set(manifest['test']) and result['summary'] and result['motion_summary']
        assert result['evaluation_protocol']=='native_rgb_target_head_v1' and not result['test_used_for_selection']
        state('complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_baseline_geometry_audit_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

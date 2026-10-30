"""Prioritize the foreground-trained candidates within the original12h budget."""
from pathlib import Path
import fcntl,json,os,subprocess,time
import run_track2_experiments as runtime
from finalize_track2_experiments import process_identity
from multihmr.track2_data import OUTPUT,save_json


def read(path):return json.loads(path.read_text())


def main():
    with (OUTPUT/'target_followup_guardian.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        revision=read(OUTPUT/'target_followup_order_revision.json')
        initial=revision['initial_evaluation_pid'];identity=revision['initial_evaluation_start_ticks']
        save_json(OUTPUT/'target_training_guardian_status.json',dict(status='search_complete_waiting_initial_target_outputs',
            pid=os.getpid(),execution_order=['paired_foreground','absolute_loss'],updated_unix=time.time()))
        while process_identity(initial)==identity:
            if time.time()>=deadline:raise TimeoutError('Original12h deadline while initial evaluation is live')
            time.sleep(20)
        assert read(OUTPUT/'target_eval_status.json')['status']=='artifacts_ready_for_final_audit'
        assert len(read(OUTPUT/'export_receipts.json')['samples'])==30
        assert read(OUTPUT/'comparison.json')['evaluation_protocol']=='target_foreground_head_v1'
        # The separate foreground training controller owns its live DDP jobs.
        # This guardian never takes their GPU leases or restarts those jobs.
        paired=read(OUTPUT/'paired_wave3_plan.json')
        while not all((OUTPUT/'runs'/cfg['name']/'complete.json').exists() for cfg in paired['trials']):
            if time.time()>=deadline:raise TimeoutError('Original12h deadline before paired search completed')
            time.sleep(20)
        assert not (OUTPUT/'target_followup_start.json').exists(),'Followups already launched'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        with (OUTPUT/'paired_wave3.log').open('a') as log:
            foreground=subprocess.Popen([str(runtime.PYTHON),'-u',str(runtime.ROOT/'tools/run_track2_cd_wave2.py'),
                '--wait-pipeline-pid',str(initial),'--plan','paired_wave3_plan.json'],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        with (OUTPUT/'target_followup.log').open('a') as log:
            absolute=subprocess.Popen([str(runtime.PYTHON),'-u',str(runtime.ROOT/'tools/run_track2_cd_wave2.py'),
                '--wait-pipeline-pid',str(foreground.pid),'--plan','cd_wave2_plan.json'],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        save_json(OUTPUT/'target_followup_start.json',dict(wave2_pid=absolute.pid,wave3_pid=foreground.pid,
            execution_order=[foreground.pid,absolute.pid],foreground_target_candidates_first=True,started_unix=time.time()))
        save_json(OUTPUT/'target_training_guardian_status.json',dict(status='target_followups_started',
            wave2_pid=absolute.pid,wave3_pid=foreground.pid,execution_order=['paired_foreground','absolute_loss'],updated_unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'target_followup_guardian_error.json',dict(error=repr(error),time=time.time()))
        raise

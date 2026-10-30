"""Adopt live DDP jobs without restarting; queue third and target-only followups."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json
import run_track2_experiments as runtime
from finalize_track2_experiments import process_identity
runtime.REGISTRY_NAME='target_training_jobs.json'


def read(path):return json.loads(Path(path).read_text())


def main():
    with (OUTPUT/'target_training_guardian.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        adopted=read(OUTPUT/'target_training_adoption.json')['jobs'];deadline=read(OUTPUT/'budget.json')['deadline_unix']
        ready=threading.Barrier(len(adopted)+1)
        def adopt(job):
            pair=job['gpus'];lease=OUTPUT/'locks'/('training_pair_'+'_'.join(map(str,pair))+'.lock')
            with lease.open('a') as held:
                fcntl.flock(held,fcntl.LOCK_EX)
                ready.wait(timeout=120)
                pid=job['pid'];identity=job['start_ticks']
                while process_identity(pid)==identity:
                    if time.time()>=deadline:raise TimeoutError('Original deadline while adopted job is live')
                    time.sleep(10)
                completed=read(OUTPUT/'runs'/job['label']/'complete.json');assert completed['successful_steps']>0
        def third():
            ready.wait(timeout=120)
            save_json(OUTPUT/'target_training_guardian_status.json',dict(status='adopted_two_live_jobs_training_third_queued',pid=os.getpid(),adopted_jobs=adopted,updated_unix=time.time()))
            name='cd_head_abs25'
            if not (OUTPUT/'runs'/name/'complete.json').exists():runtime.run_training(runtime.train_command(name),0,name,deadline)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(adopt,job) for job in adopted]+[pool.submit(third)]
            for future in futures:future.result()
        save_json(OUTPUT/'target_training_guardian_status.json',dict(status='three_search_trials_complete',pid=os.getpid(),updated_unix=time.time()))
        initial=read(OUTPUT/'target_eval_start.json')['pid'];identity=process_identity(initial)
        while identity is not None and process_identity(initial)==identity:
            if time.time()>=deadline:raise TimeoutError('Original deadline while target evaluation is live')
            time.sleep(20)
        assert read(OUTPUT/'target_eval_status.json')['status']=='artifacts_ready_for_final_audit'
        # All three search jobs are already complete; this controller now
        # performs target-only checkpoint validation/refits/Test/export.
        with (OUTPUT/'target_followup.log').open('a') as log:
            wave2=subprocess.Popen([str(runtime.PYTHON),'-u',str(ROOT/'tools/run_track2_cd_wave2.py'),'--wait-pipeline-pid',str(initial)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        with (OUTPUT/'paired_wave3.log').open('a') as log:
            wave3=subprocess.Popen([str(runtime.PYTHON),'-u',str(ROOT/'tools/run_track2_cd_wave2.py'),'--wait-pipeline-pid',str(wave2.pid),'--plan','paired_wave3_plan.json'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        save_json(OUTPUT/'target_followup_start.json',dict(wave2_pid=wave2.pid,wave3_pid=wave3.pid,started_unix=time.time()))
        save_json(OUTPUT/'target_training_guardian_status.json',dict(status='target_followups_started',wave2_pid=wave2.pid,wave3_pid=wave3.pid,updated_unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'target_training_guardian_error.json',dict(error=repr(error),time=time.time()));runtime.stop_children();raise

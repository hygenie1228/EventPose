"""Use spare GPU6 during training, then healthy GPUs, for uniform v2 conversion."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,fcntl,json,os,queue,signal,sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,save_json
import run_track2_experiments as runtime
from finalize_track2_experiments import process_identity
runtime.REGISTRY_NAME='conversion_v2_jobs.json'


def read(path):return json.loads(Path(path).read_text())


def main(initial_pid,last_pid):
    with (OUTPUT/'conversion_v2_prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        names=['head_strong','last2_strong','last4_strong']
        for plan in ['cd_wave2_plan.json','paired_wave3_plan.json']:
            path=OUTPUT/plan
            if path.exists():names.extend(cfg['name'] for cfg in read(path)['trials'])
        roster=[(name,sample) for name in ['original']+['search_'+n for n in names] for sample in manifest['validation']]
        roster.extend((name,sample) for name in ['original']+names for sample in manifest['test'])
        def status(state,**extra):
            save_json(OUTPUT/'conversion_v2_prefetch_status.json',dict(status=state,pid=os.getpid(),updated_unix=time.time(),**extra))
        def pending():
            tasks=[]
            for name,sample in roster:
                folder=OUTPUT/'predictions'/name/sample
                if all((folder/file).exists() for file in ['conversion.json','human_params_mhr.npz','inference.json','smplx_predictions.npz']) and not (folder/'metrics_v2.json').exists():tasks.append((name,sample))
            return tasks
        def evaluate(task,gpu):
            name,sample=task
            for stage in ['track2-refine','track2-score']:
                command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,'--trial',name,'--sample',sample,'--conversion-revision','2']
                runtime.run(command,[gpu],'v2_'+stage+'_'+name+'_'+sample,deadline)
        initial_identity=process_identity(initial_pid);last_identity=process_identity(last_pid)
        status('waiting_initial_evaluation',initial_pid=initial_pid,last_pipeline_pid=last_pid)
        while initial_identity is not None and process_identity(initial_pid)==initial_identity:
            if time.time()>=deadline:raise TimeoutError('Original deadline while waiting initial pipeline')
            time.sleep(20)
        pilot=read(OUTPUT/'conversion_v2_pilot_status.json');assert pilot['status']=='passed',pilot
        import pynvml
        pynvml.nvmlInit();resources=read(OUTPUT/'gpu_resources.json');assert 6 in resources['healthy_gpus']
        handles={gpu:pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(gpu)]) for gpu in resources['healthy_gpus']}
        with ThreadPoolExecutor(max_workers=len(handles)) as pool:
            active={};claimed=set()
            while last_identity is not None and process_identity(last_pid)==last_identity:
                if time.time()>=deadline:raise TimeoutError('Original deadline in v2 prefetch')
                for gpu,(future,task) in list(active.items()):
                    if future.done():future.result();claimed.remove(task);del active[gpu]
                tasks=[task for task in pending() if task not in claimed]
                for gpu in [6]+[g for g in handles if g!=6]:
                    if not tasks:break
                    if gpu in active or pynvml.nvmlDeviceGetComputeRunningProcesses(handles[gpu]):continue
                    task=tasks.pop(0);claimed.add(task);active[gpu]=(pool.submit(evaluate,task,gpu),task)
                    status('evaluating_alongside_training',active_gpus=list(active),pending_ready=len(tasks),current=list(task))
                time.sleep(10)
            for future,task in active.values():future.result()
        tasks=queue.Queue()
        for task in pending():tasks.put(task)
        gpus=runtime.healthy_gpus();status('finishing_ready_roster',remaining=tasks.qsize(),gpus=gpus)
        def worker(gpu):
            while True:
                try:task=tasks.get_nowait()
                except queue.Empty:return
                evaluate(task,gpu);tasks.task_done()
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,gpu) for gpu in gpus]
            for future in futures:future.result()
        completed=[(name,sample) for name,sample in roster if (OUTPUT/'predictions'/name/sample/'metrics_v2.json').exists()]
        status('ready_for_uniform_comparison_audit',completed_pairs=len(completed),last_pipeline_state=read(OUTPUT/'supervisor.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--after-initial-pid',type=int,required=True);parser.add_argument('--last-pipeline-pid',type=int,required=True);args=parser.parse_args()
    try:main(args.after_initial_pid,args.last_pipeline_pid)
    except BaseException as error:
        save_json(OUTPUT/'conversion_v2_prefetch_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

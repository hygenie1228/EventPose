"""Pretrain the three declared target-only paired candidates during evaluation.

This controller writes only its own training runs. Model comparison, Train20
refits and exports remain serialized by the existing evaluation controllers.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json
runtime.REGISTRY_NAME='foreground_pair_training_jobs.json'


def read(path):return json.loads(path.read_text())


def main():
    with (OUTPUT/'foreground_pair_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'paired_wave3_plan.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        pilot=read(OUTPUT/'runs'/plan['pilot_name']/'complete.json')
        assert pilot['successful_steps']==pilot['step']==20
        assert read(OUTPUT/'paired_cd_full_gradient_checks.json')['passed']
        assert read(OUTPUT/'target_training_input_checks.json')['passed']
        assert plan['test_used_for_selection'] is False
        def state(status,**extra):
            save_json(OUTPUT/'paired_foreground_search_status.json',dict(status=status,pid=os.getpid(),
                updated_unix=time.time(),deadline_unix=deadline,target_foreground=True,
                full_anchor_sim3_gradient=True,test_used_for_selection=False,**extra))
        state('target_foreground_search_while_evaluation',trials=[c['name'] for c in plan['trials']])
        def train(index):
            cfg=plan['trials'][index];name=cfg['name']
            assert cfg['target_foreground'] and cfg['paired_alignment_full_gradient']
            if not (OUTPUT/'runs'/name/'complete.json').exists():
                runtime.run_training(runtime.train_command(name),index,name,deadline)
            result=read(OUTPUT/'runs'/name/'complete.json')
            assert result['successful_steps']>0
            assert result['elapsed_seconds']+cfg['bootstrap_elapsed_seconds']+cfg['refit_hours_max']*3600<3*3600
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(train,index) for index in range(3)]
            for future in futures:future.result()
        state('target_foreground_search_complete_refit_and_evaluation_pending')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        runtime.stop_children()
        save_json(OUTPUT/'paired_foreground_search_error.json',dict(error=repr(error),time=time.time()))
        raise

"""Refit declared candidates on Train20 while the initial Test evaluation runs.

Only immutable Train-only SMPL-X proxy choices determine update budgets.
Global recipe selection, checkpoint mapping and exports stay with the finalizer.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,signal,time
import run_track2_experiments as runtime
from multihmr.track2_data import OUTPUT,save_json,sha256
runtime.REGISTRY_NAME='target_refit_prefetch_jobs.json'


def read(path):return json.loads(path.read_text())


def main():
    with (OUTPUT/'target_refit_prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        signal.signal(signal.SIGINT,runtime.stop_children)
        plan=read(OUTPUT/'target_pooled_followup_plan.json');deadline=read(OUTPUT/'budget.json')['deadline_unix']
        manifest=read(OUTPUT/'manifest.json');work=OUTPUT/'work'
        training=read(work/'training_index.json');records=training+read(work/'validation_index.json')
        assert read(work/'full_training_index.json')==records
        assert {r['session'] for r in records}==set(manifest['train'])
        assert {r['session'] for r in records}.isdisjoint(manifest['test'])
        minimum=read(OUTPUT/'target_foreground_protocol.json')['minimum_target_detection_coverage']
        def state(status):
            save_json(OUTPUT/'target_refit_prefetch_status.json',dict(status=status,pid=os.getpid(),
                deadline_unix=deadline,updated_unix=time.time(),test_used_for_selection=False,
                candidate_names=[c['name'] for c in plan['trials']]))
        state('refitting_declared_Train20_candidates_while_evaluation')
        def worker(index):
            cfg=plan['trials'][index];name=cfg['name']
            while True:
                values={k:read(OUTPUT/'validation_smplx'/(name+'_'+k)/'metrics.json')
                    if (OUTPUT/'validation_smplx'/(name+'_'+k)/'metrics.json').exists() else {} for k in ['best','last']}
                if all(v.get('evaluation_protocol')=='target_foreground_head_v1'
                    and v.get('detection_selection')=='target_mask_head_highest_confidence' for v in values.values()):break
                if time.time()>=deadline:raise TimeoutError('Original deadline awaiting target-only checkpoint proxy')
                time.sleep(10)
            eligible=[k for k,v in values.items() if v['coverage']>=minimum];assert eligible,name
            kind=min(eligible,key=lambda k:(values[k]['cd_h_smplx_proxy_cm'],['best','last'].index(k)))
            selected=values[kind];checkpoint=OUTPUT/'checkpoints'/name/(kind+'.pt')
            assert selected['checkpoint_sha256']==sha256(checkpoint)
            assert selected['test_used'] is False
            assert set(selected['per_sequence'])==set(manifest['validation'])
            updates=selected['step'];assert updates>0,'Initialization-only candidate needs no additional refit'
            complete=read(OUTPUT/'runs'/name/'complete.json');bootstrap=cfg.get('bootstrap_elapsed_seconds',0)
            refit_name='refit_'+name
            hours=min(cfg.get('refit_hours_max',1.),max(.05,2.9-(complete['elapsed_seconds']+bootstrap)/3600))
            search_configuration=read(OUTPUT/'runs'/name/'configuration.json')
            configuration=dict(cfg,name=refit_name,training_role='full_training',fixed_full_train=True,
                target_foreground=search_configuration.get('target_foreground',False),
                hours=hours,max_steps=min(3500,max(1,round(updates*len(records)/len(training)))),
                max_epochs=0,early_stop_patience=0,recipe_selection=selected,
                selection_source_trial=name,deadline_unix=deadline,selection_metric='pa_mpjpe_mm',
                init_checkpoint=cfg.get('refit_init_checkpoint',cfg.get('init_checkpoint')))
            save_json(OUTPUT/'configs'/(refit_name+'.json'),configuration)
            save_json(OUTPUT/'refit_prefetch_choices'/(name+'.json'),dict(selected_candidate=kind,
                metrics=selected,configuration=configuration,Train20_sequences=manifest['train'],
                test_used_for_selection=False,selected_unix=time.time()))
            if not (OUTPUT/'runs'/refit_name/'complete.json').exists():
                runtime.run_training(runtime.train_command(refit_name),index,refit_name,deadline)
            result=read(OUTPUT/'runs'/refit_name/'complete.json')
            assert result['successful_steps']>0 and result['training_role']=='full_training'
            assert not result['validation_used_for_selection']
            assert complete['elapsed_seconds']+result['elapsed_seconds']+bootstrap<10800
        with ThreadPoolExecutor(max_workers=len(plan['trials'])) as pool:
            futures=[pool.submit(worker,i) for i in range(len(plan['trials']))]
            try:
                for future in futures:future.result()
            except BaseException:runtime.stop_children();raise
        state('all_Train20_refits_complete_pending_MHR_recipe_selection')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'target_refit_prefetch_error.json',dict(error=repr(error),time=time.time()))
        raise

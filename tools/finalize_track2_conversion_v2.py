"""Compare all completed recipes under uniform v2 conversion and export winner."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,fcntl,json,os,queue,shutil,signal,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
runtime.REGISTRY_NAME='conversion_v2_final_jobs.json'


def read(path):return json.loads(Path(path).read_text())


def aggregate(trial,samples,manifest,mappings):
    values={}
    for sample in samples:
        folder=OUTPUT/'predictions'/trial/sample
        metric=read(folder/'metrics_v2.json');info=read(folder/'inference.json');record=manifest['samples'][sample]
        assert metric['conversion_revision']==2 and metric['gt_sha256']==record['gt_sha256']
        assert metric['frames']==record['frames'] and metric['full_frame_roster']
        assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
        if trial!='original':assert info['checkpoint_sha256']==mappings[trial]['checkpoint_sha256']
        assert not info['gt_used_for_detection_or_tracking']
        conversion=read(folder/'conversion_v2.json')
        assert not conversion['gt_parameters_used'] and not conversion['temporal_smoothing'] and not conversion['identity_refitted']
        assert conversion['source_sha256']==info['result_sha256']==sha256(folder/'smplx_predictions.npz')
        assert conversion['initial_sha256']==sha256(folder/'human_params_mhr.npz')
        assert conversion['output_sha256']==metric['prediction_sha256']
        with np.load(folder/'human_params_mhr.npz') as a,np.load(folder/'human_params_mhr_v2.npz') as b:
            assert np.array_equal(a['shape'],b['shape']) and np.array_equal(a['scales'],b['scales'])
        values[sample]=metric
    return dict(CD_H_cm=float(np.mean([x['CD_H_cm'] for x in values.values()])),
        ACC_H_cm_per_frame2=float(np.mean([x['ACC_H_cm_per_frame2'] for x in values.values()])),
        sequences=len(samples),frames=sum(x['frames'] for x in values.values()),per_sequence=values)


def main(prefetch_pid):
    with (OUTPUT/'conversion_v2_final.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGTERM,runtime.stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix'];identity=finish.process_identity(prefetch_pid)
        save_json(OUTPUT/'conversion_v2_final_status.json',dict(status='waiting_uniform_v2_roster',pid=os.getpid(),prefetch_pid=prefetch_pid))
        while identity is not None and finish.process_identity(prefetch_pid)==identity:
            if time.time()>=deadline:raise TimeoutError('Original deadline while waiting uniform v2 roster')
            time.sleep(20)
        state=read(OUTPUT/'conversion_v2_prefetch_status.json');assert state['status']=='ready_for_uniform_comparison_audit',state
        assert read(OUTPUT/'supervisor.json')['status']=='artifacts_ready_for_final_audit'
        manifest=read(OUTPUT/'manifest.json');mappings=read(OUTPUT/'evaluation_checkpoints.json')
        previous=read(OUTPUT/'comparison.json');names=list(previous['models']);assert 'original' in names and len(names)>=4
        cd=read(OUTPUT/'internal_cd_h_selection.json')
        validation={name:aggregate('original' if name=='original' else 'search_'+name,manifest['validation'],manifest,mappings) for name in names}
        eligible=['original']+[name for name in names if name!='original' and cd['trials'][name]['metrics']['coverage']>=cd['baseline']['coverage']-.005]
        winner=min(eligible,key=lambda name:(validation[name]['CD_H_cm'],validation[name]['ACC_H_cm_per_frame2'],names.index(name)))
        selection=dict(trial=winner,conversion_revision=2,test_used_for_selection=False,
            selection='Same four Train-only full-frame MHR CD-H primary, ACC-H tie-break, proxy coverage gate; common conversion revision2',
            checkpoint=mappings[winner]['checkpoint'] if winner!='original' else None,selected_unix=time.time())
        validation_result=dict(models=validation,selected_trial=winner,conversion_revision=2,test_used_for_selection=False,selection=selection['selection'],completed_unix=time.time())
        # Complete recipe selection before reading any v2 Test aggregate.
        save_json(OUTPUT/'selected_v2.json',selection);save_json(OUTPUT/'validation_mhr_comparison_v2.json',validation_result)
        models={name:aggregate(name,manifest['test'],manifest,mappings) for name in names}
        comparison=dict(models=models,conversion_revision=2,aggregation='equal sequence weights',test_used_for_selection=False,
            units=dict(CD_H='cm',ACC_H='cm/frame^2'),official_leaderboard_parity=False,source_manifest_sha256=sha256(OUTPUT/'manifest.json'),completed_unix=time.time())
        save_json(OUTPUT/'comparison_v2.json',comparison)
        archive=OUTPUT/'conversion_v1_final';archive.mkdir(exist_ok=True)
        for name in ['selected.json','comparison.json','validation_mhr_comparison.json','export_receipts.json','final_audit.json']:
            if not (archive/name).exists():shutil.copyfile(OUTPUT/name,archive/name)
        tasks=queue.Queue()
        for sample in manifest['train']:tasks.put(sample)
        save_json(OUTPUT/'conversion_v2_final_status.json',dict(status='preparing_winner_all30',pid=os.getpid(),winner=winner,updated_unix=time.time()))
        def worker(gpu):
            while True:
                try:sample=tasks.get_nowait()
                except queue.Empty:return
                for stage in ['track2-infer','track2-convert','track2-refine']:
                    command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,'--trial',winner,'--sample',sample]
                    if stage=='track2-refine':command.extend(['--conversion-revision','2'])
                    if stage=='track2-infer' and winner!='original':command.extend(['--checkpoint',mappings[winner]['checkpoint']])
                    runtime.run(command,[gpu],'v2_export_'+stage+'_'+winner+'_'+sample,deadline)
                tasks.task_done()
        with ThreadPoolExecutor(max_workers=len(runtime.healthy_gpus())) as pool:
            futures=[pool.submit(worker,gpu) for gpu in runtime.healthy_gpus()]
            try:
                for future in futures:future.result()
            except BaseException:runtime.stop_children();raise
        receipts={}
        for sample in sorted(manifest['samples']):
            receipts[sample]=finish.export_sample(winner,sample,manifest,conversion_revision=2)
            save_json(OUTPUT/'export_receipts_v2.json',dict(trial=winner,conversion_revision=2,samples=receipts,updated_unix=time.time()))
        assert len(receipts)==30
        for sample,item in receipts.items():
            assert sha256(Path(item['path']))==item['sha256']
            with np.load(item['path']) as z:
                assert z['pose'].shape==(manifest['samples'][sample]['frames'],136)
                assert z['shape'].shape==(45,) and z['scales'].shape==(68,)
                assert all(np.isfinite(z[key]).all() for key in ['pose','shape','scales'])
                provenance=json.loads(z['report'].item());assert provenance['conversion_revision']==2 and provenance['trial']==winner
                assert not provenance['gt_pose_identity_used_for_inference']
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==manifest['samples'][sample]['gt_sha256']
        # Publish a coherent revision only after every model and all exports pass.
        save_json(OUTPUT/'comparison.json',comparison);save_json(OUTPUT/'selected.json',selection)
        save_json(OUTPUT/'validation_mhr_comparison.json',validation_result)
        save_json(OUTPUT/'export_receipts.json',dict(trial=winner,conversion_revision=2,samples=receipts,updated_unix=time.time()))
        save_json(OUTPUT/'conversion_v2_audit.json',dict(all_models_same_conversion_revision=True,
            model_count=len(models),frozen_test10_full_frames=True,Train_validation4_only_selection=True,
            GT_free_conversion_and_inference=True,all30_exports_reloaded_and_hashed=True,all30_GT_hashes_preserved=True,
            version1_archived=True,requires_agent_review=True,completed_unix=time.time()))
        save_json(OUTPUT/'conversion_v2_final_status.json',dict(status='artifacts_ready_for_agent_audit',winner=winner,pid=os.getpid(),completed_unix=time.time()))
        finish.status('artifacts_ready_for_final_audit',selected_recipe=winner,exports=30,conversion_revision=2,completed_unix=time.time())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-prefetch-pid',type=int,required=True);args=parser.parse_args()
    try:main(args.wait_prefetch_pid)
    except BaseException as error:
        save_json(OUTPUT/'conversion_v2_final_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

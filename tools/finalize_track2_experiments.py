"""Finish Train-only recipe selection, full-Train refits, MHR Test and exports.

Wait for the existing training supervisor by process identity. Never relaunch it
because a status observation expires. All GPU work is restricted to GPUs 1–6.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
from run_track2_experiments import run,stop_children,train_command,PYTHON,healthy_gpus,run_training

TRIALS=['head_strong','last2_strong','last4_strong']


def parallel(function,count):
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures=[pool.submit(function,i) for i in range(count)]
        try:
            return [future.result() for future in as_completed(futures)]
        except BaseException:
            stop_children()
            raise


def read(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if fields[0]=='Z':return None
        return fields[19]
    except FileNotFoundError:return None


def report():
    subprocess.run([str(PYTHON),str(ROOT/'tools/report_track2_experiments.py')],cwd=ROOT,check=True)


def status(state,**extra):
    save_json(OUTPUT/'supervisor.json',dict(pid=os.getpid(),status=state,updated_unix=time.time(),**extra))
    report()


def wait_training(pid,deadline):
    identity=process_identity(pid)
    if identity is not None:
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'run_track2_experiments.py' not in command:identity=None
        except FileNotFoundError:identity=None
    save_json(OUTPUT/'finalizer.json',dict(pid=os.getpid(),status='waiting_existing_training',
        training_supervisor_pid=pid,training_start_ticks=identity,started_unix=time.time()))
    while identity is not None and process_identity(pid)==identity:
        if time.time()>=deadline:raise TimeoutError('Global deadline while original training is still live')
        # Lifecycle-only observation, not GPU/metric polling. The original
        # supervisor and its deadlines remain authoritative.
        time.sleep(min(30,max(1,deadline-time.time())))
    for name in TRIALS:
        complete=read(OUTPUT/'runs'/name/'complete.json')
        assert complete['status']=='complete' and complete['successful_steps']>0
    save_json(OUTPUT/'finalizer.json',dict(pid=os.getpid(),status='initial_training_verified',
        training_supervisor_pid=pid,updated_unix=time.time()))


def full_train_refits(manifest,deadline):
    work=OUTPUT/'work'
    training=read(work/'training_index.json')
    validation=read(work/'validation_index.json')
    records=training+validation
    assert {x['session'] for x in records}==set(manifest['train'])
    assert not {x['session'] for x in records}&set(manifest['test'])
    save_json(work/'full_training_index.json',records)
    recipes={}
    cd_choices=read(OUTPUT/'internal_cd_h_selection.json')['trials']
    for name in TRIALS:
        best=cd_choices[name]['metrics']
        baseline=read(OUTPUT/'runs'/name/'baseline.json')
        complete=read(OUTPUT/'runs'/name/'complete.json')
        recipes[name]=dict(validation=best,baseline=baseline,initial_training=complete)
    mhr_selection=read(OUTPUT/'validation_mhr_comparison.json')
    selected=mhr_selection['selected_trial']
    save_json(OUTPUT/'selected.json',dict(trial=selected,selection=mhr_selection['selection'],
        mhr_validation=mhr_selection,
        test_used_for_selection=False,recipes=recipes,selected_unix=time.time(),
        final_fit='Refit each fixed recipe on all 20 Train sequences, with update budgets fixed before Test inference'))
    status('refitting_all20_train',selected_recipe=selected)
    def worker(index):
        name=TRIALS[index];parent=read(OUTPUT/'configs'/(name+'.json'))
        parent_complete=recipes[name]['initial_training']
        updates=recipes[name]['validation']['step']
        if updates==0:
            if parent.get('refit_init_checkpoint'):
                counterpart=Path(parent['refit_init_checkpoint'])
                complete=read(OUTPUT/'runs'/counterpart.parent.name/'complete.json')
                assert complete['training_role']=='full_training'
                return name,dict(checkpoint=str(counterpart),parent_trial=name,
                    full_train_refit=True,refit_performed_this_wave=False,
                    reason='CD-H selected the continuation initialization; use its all20-Train counterpart',
                    recipe_validation=recipes[name]['validation'],complete=complete,
                    checkpoint_sha256=sha256(counterpart))
            # The validation-selected model is exactly its initialization.
            # Do not force unselected updates into a model labelled "best".
            return name,dict(checkpoint=cd_choices[name]['checkpoint'],
                parent_trial=name,full_train_refit=False,reason='Validation selected original initialization',
                recipe_validation=recipes[name]['validation'])
        refit_name='refit_'+name
        bootstrap=parent.get('bootstrap_elapsed_seconds',0)
        hours=min(parent.get('refit_hours_max',1.),max(.05,2.9-(parent_complete['elapsed_seconds']+bootstrap)/3600))
        search_configuration=read(OUTPUT/'runs'/name/'configuration.json')
        config=dict(parent,name=refit_name,training_role='full_training',fixed_full_train=True,
            target_foreground=search_configuration.get('target_foreground',False),
            hours=hours,max_steps=min(3500,max(1,round(updates*len(records)/len(training)))),
            max_epochs=0,early_stop_patience=0,recipe_selection=recipes[name]['validation'],
            selection_source_trial=name,deadline_unix=deadline,selection_metric='pa_mpjpe_mm',
            init_checkpoint=parent.get('refit_init_checkpoint',parent.get('init_checkpoint')))
        save_json(OUTPUT/'configs'/(refit_name+'.json'),config)
        if not (OUTPUT/'runs'/refit_name/'complete.json').exists():
            run_training(train_command(refit_name),index,refit_name,deadline)
        complete=read(OUTPUT/'runs'/refit_name/'complete.json')
        assert complete['successful_steps']>0 and complete['training_role']=='full_training'
        assert not complete['validation_used_for_selection']
        assert parent_complete['elapsed_seconds']+complete['elapsed_seconds']+bootstrap<3*3600
        return name,dict(checkpoint=str(OUTPUT/'checkpoints'/refit_name/'best.pt'),
            parent_trial=name,full_train_refit=True,recipe_validation=recipes[name]['validation'],
            complete=complete,checkpoint_sha256=sha256(OUTPUT/'checkpoints'/refit_name/'best.pt'))
    mappings=dict(parallel(worker,len(TRIALS)))
    aliases=read(OUTPUT/'evaluation_checkpoints.json')
    aliases.update(mappings)
    mapping_path=OUTPUT/'evaluation_checkpoints.json'
    retained=read(mapping_path) if mapping_path.exists() else {}
    retained.update(aliases)
    save_json(mapping_path,retained)
    return selected


def evaluation_command(stage,trial,sample,limit=0):
    command=[str(PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),stage,
        '--trial',trial,'--sample',sample]
    if limit:command.extend(['--limit',str(limit)])
    return command


def evaluation_smoke(manifest,deadline):
    # Conversion tuning/debugging is done on Train, never on Test geometry.
    sample=manifest['train'][0]
    for stage in ['track2-infer','track2-convert']:
        run(evaluation_command(stage,'original',sample,limit=8),[1],
            'smoke_'+stage,deadline)
    folder=OUTPUT/'smoke_predictions/original'/sample
    with np.load(folder/'human_params_mhr.npz') as z:
        assert z['pose'].shape==(8,136) and z['shape'].shape==(45,) and z['scales'].shape==(68,)
        assert all(np.isfinite(z[k]).all() for k in ['pose','shape','scales'])
    save_json(OUTPUT/'evaluation_smoke.json',dict(status='passed',sample=sample,
        test_used=False,conversion=read(folder/'conversion.json'),updated_unix=time.time()))


def select_smplx_checkpoints(manifest,deadline):
    """Re-rank retained best/last candidates by SMPL-X CD-H, before any Test use."""
    gpus=healthy_gpus();workers=len(gpus)
    tasks=[('original',None)]
    for name in TRIALS:
        for kind in ['best','last']:
            tasks.append((name+'_'+kind,OUTPUT/'checkpoints'/name/(kind+'.pt')))
    status('validating_smplx_cd_h',retained_checkpoint_candidates=2*len(TRIALS))
    def worker(index):
        for tag,path in tasks[index::workers]:
            command=[str(PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),
                'track2-validate','--validation-tag',tag]
            if path:command.extend(['--checkpoint',str(path)])
            run(command,[gpus[index]],'smplx_cd_'+tag,deadline)
    parallel(worker,workers)
    baseline=read(OUTPUT/'validation_smplx/original/metrics.json')
    choices={};candidates={}
    for name in TRIALS:
        values={kind:read(OUTPUT/'validation_smplx'/(name+'_'+kind)/'metrics.json') for kind in ['best','last']}
        minimum_coverage=read(OUTPUT/'target_foreground_protocol.json')['minimum_target_detection_coverage']
        eligible=[kind for kind,value in values.items() if value['coverage']>=minimum_coverage]
        assert eligible,('No CD-H candidate meets coverage gate',name)
        kind=min(eligible,key=lambda kind:(values[kind]['cd_h_smplx_proxy_cm'],['best','last'].index(kind)))
        choices[name]=dict(checkpoint=values[kind]['checkpoint'],metrics=values[kind],selected_candidate=kind)
        candidates[name]=values
    save_json(OUTPUT/'internal_cd_h_selection.json',dict(trials=choices,candidates=candidates,
        baseline=baseline,test_used=False,selected_unix=time.time(),
        limitation='The initial live runs retained PA-selected best and final last checkpoints; only these saved candidates can be re-ranked by CD-H.',
        criterion='SMPL-X fitted-GT-space CD-H proxy, frame0 fixed Sim(3), FPS64 body, equal four-sequence weights'))
    report()


def validation_mhr(manifest,deadline):
    """Select on the requested metrics using only the four held-out Train clips."""
    aliases={}
    choices=read(OUTPUT/'internal_cd_h_selection.json')['trials']
    for name in TRIALS:
        checkpoint=Path(choices[name]['checkpoint'])
        aliases['search_'+name]=dict(checkpoint=str(checkpoint),parent_trial=name,
            full_train_refit=False,checkpoint_sha256=sha256(checkpoint),purpose='Train-only MHR recipe validation')
    mapping_path=OUTPUT/'evaluation_checkpoints.json'
    retained=read(mapping_path) if mapping_path.exists() else {}
    retained.update(aliases)
    save_json(mapping_path,retained)
    cases=['original']+['search_'+name for name in TRIALS]
    gpus=healthy_gpus();workers=len(gpus)
    tasks=[(trial,sample) for trial in cases for sample in manifest['validation']]
    status('validating_mhr_train_holdout',total_sequence_models=len(tasks))
    def shard(index):
        for trial,sample in tasks[index::workers]:
            for stage in ['track2-infer','track2-convert','track2-score']:
                run(evaluation_command(stage,trial,sample),[gpus[index]],
                    'validation_'+stage+'_'+trial+'_'+sample,deadline)
    parallel(shard,workers)
    models={}
    for trial in cases:
        samples={name:read(OUTPUT/'predictions'/trial/name/'metrics.json') for name in manifest['validation']}
        assert len(samples)==4 and not set(samples)&set(manifest['test'])
        for sample,metrics in samples.items():
            assert metrics['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            assert metrics['frames']==manifest['samples'][sample]['frames']
        name=trial.removeprefix('search_')
        models[name]=dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in samples.values()])),
            ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in samples.values()])),
            per_sequence=samples,sequences=4)
    baseline=models['original']
    eligible=['original']
    proxy_baseline=read(OUTPUT/'internal_cd_h_selection.json')['baseline']
    for name in TRIALS:
        best=choices[name]['metrics']
        if best['coverage']>=read(OUTPUT/'target_foreground_protocol.json')['minimum_target_detection_coverage']:eligible.append(name)
    for name,value in models.items():
        value['normalized_selection_score']=.5*value['CD_H_cm']/max(baseline['CD_H_cm'],1e-12)+.5*value['ACC_H_cm_per_frame2']/max(baseline['ACC_H_cm_per_frame2'],1e-12)
    order=['original']+TRIALS
    selected=min(eligible,key=lambda name:(models[name]['CD_H_cm'],models[name]['ACC_H_cm_per_frame2'],order.index(name)))
    save_json(OUTPUT/'validation_mhr_comparison.json',dict(models=models,selected_trial=selected,
        selection='Train-only four-sequence full-frame MHR CD-H primary, ACC-H tie-break; target-head detection coverage >= .95; retained checkpoints selected by SMPL-X CD-H proxy',
        selection_primary='CD_H_cm',selection_tiebreak='ACC_H_cm_per_frame2',
        test_used_for_selection=False,completed_unix=time.time()))
    report()


def run_test(manifest,deadline):
    gpus=healthy_gpus();workers=len(gpus)
    tasks=[(trial,sample) for trial in ['original']+TRIALS for sample in manifest['test']]
    status('evaluating_test',total_sequence_models=len(tasks))
    def shard(index):
        for trial,sample in tasks[index::workers]:
            for stage in ['track2-infer','track2-convert','track2-score']:
                run(evaluation_command(stage,trial,sample),[gpus[index]],
                    stage+'_'+trial+'_'+sample,deadline)
    parallel(shard,workers)
    values={}
    for trial in ['original']+TRIALS:
        samples={name:read(OUTPUT/'predictions'/trial/name/'metrics.json') for name in manifest['test']}
        assert len(samples)==10
        for name,metrics in samples.items():
            assert metrics['frames']==manifest['samples'][name]['frames']
            assert metrics['gt_sha256']==manifest['samples'][name]['gt_sha256']
            assert metrics['full_frame_roster']
        values[trial]=dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in samples.values()])),
            ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in samples.values()])),
            sequences=10,frames=sum(x['frames'] for x in samples.values()),
            per_sequence=samples)
    comparison=dict(models=values,aggregation='equal sequence weights',
        source_manifest_sha256=sha256(OUTPUT/'manifest.json'),test_used_for_selection=False,
        units=dict(CD_H='cm',ACC_H='cm/frame^2'),official_leaderboard_parity=False,
        conversion_revision=2,evaluation_protocol='target_foreground_head_v1',
        completed_unix=time.time())
    save_json(OUTPUT/'comparison.json',comparison)
    status('test_complete_pending_exports')


def export_sample(trial,sample,manifest,conversion_revision=2,destination_override=None):
    folder=OUTPUT/'predictions'/trial/sample
    prediction=folder/('human_params_mhr_v2.npz' if conversion_revision==2 else 'human_params_mhr.npz')
    conversion_report=folder/('conversion_v2.json' if conversion_revision==2 else 'conversion.json')
    inference=read(folder/'inference.json')
    record=manifest['samples'][sample]
    assert inference['frames']==record['frames'] and inference['full_original_frame_rate']
    assert not inference['gt_used_for_detection_or_tracking']
    input_protocol=inference.get('evaluation_protocol')
    assert input_protocol in ['target_foreground_head_v1','native_rgb_target_head_v1','native_rgb_static_crop_target_head_v1',
        'native_rgb_strict_mask_target_head_v2','native_rgb_static_crop_strict_mask_target_head_v2',
        'native_rgb_strict_mask_body_box_v3','native_rgb_static_crop_strict_mask_body_box_v3'],'Target-specific inference required'
    assert inference.get('all_accepted_heads_in_target_mask')
    assert inference.get('target_foreground_only_input')==(input_protocol=='target_foreground_head_v1')
    assert read(conversion_report)['evaluation_protocol']==input_protocol
    assert inference['target_mask_sha256']==sha256(SOURCE/sample/'processed/human_mask.mp4')
    assert sha256(SOURCE/sample/'human/human_params_gt.npz')==record['gt_sha256']
    with np.load(prediction) as z:
        arrays={key:z[key] for key in z.files}
    with np.load(folder/'smplx_predictions.npz') as z:
        arrays['valid_input']=z['detection_valid']
        arrays['detection_confidence']=z['confidence']
    count=record['frames']
    for key,shape in [('pose',(count,136)),('shape',(45,)),('scales',(68,))]:
        assert arrays[key].shape==shape and np.isfinite(arrays[key]).all()
    provenance=dict(trial=trial,checkpoint=inference['delta_checkpoint'],
        checkpoint_sha256=inference['checkpoint_sha256'],prediction_source_sha256=sha256(prediction),
        rgb_inference=read(folder/'inference.json'),conversion=read(conversion_report),conversion_revision=conversion_revision,
        gt_pose_identity_used_for_inference=False,test_used_for_selection=False,
        evaluation_protocol=input_protocol,target_mask_used_for_acceptance=True,target_foreground_only_input=inference['target_foreground_only_input'],
        native_view_preserved=inference['input_view']!='native_rgb_static_crop',native_camera_coordinates_preserved=True,
        fixed_view_input=inference['input_view']=='native_rgb_static_crop',
        coordinate_frame='opencv_camera',human_model='official MHR v1.0.1 LOD1',
        fps=record['fps'],frame_count=count)
    arrays.update(report=np.asarray(json.dumps(provenance)),frame_names=np.asarray([f'{i:08d}' for i in range(count)]),
        fps=np.asarray(record['fps']),coordinate_frame=np.asarray('opencv_camera'),
        world_to_camera=np.asarray(record['world_to_camera']),K=np.asarray(record['K']),
        intrinsics=np.broadcast_to(np.array([record['K'][0][0],record['K'][1][1],
            record['K'][0][2],record['K'][1][2]],np.float32),(count,4)).copy(),
        camera_extrinsics=np.broadcast_to(np.eye(4,dtype=np.float32),(count,4,4)).copy(),
        GT_used=np.asarray(False),body_model=np.asarray('MHR'))
    stage=OUTPUT/'export_staging'/sample/'human_params_uranus.npz'
    stage.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(stage,**arrays)
    destination=Path(destination_override) if destination_override is not None else SOURCE/sample/'human/human_params_uranus.npz'
    destination.parent.mkdir(parents=True,exist_ok=True)
    previous=None
    if destination.exists() and sha256(destination)!=sha256(stage):
        previous=sha256(destination)
        archive=OUTPUT/'previous_exports'/sample/(previous+'.npz')
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():shutil.copyfile(destination,archive)
    temporary=destination.with_name('.human_params_uranus.partial.npz')
    shutil.copyfile(stage,temporary)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    temporary.replace(destination)
    assert sha256(destination)==sha256(stage)
    assert sha256(SOURCE/sample/'human/human_params_gt.npz')==record['gt_sha256']
    return dict(path=str(destination),sha256=sha256(destination),frames=count,
        checkpoint_sha256=inference['checkpoint_sha256'],conversion_revision=conversion_revision,previous_sha256=previous,
        gt_sha256_unchanged=record['gt_sha256'],exported_unix=time.time())


def exports(trial,manifest,deadline):
    gpus=healthy_gpus();workers=len(gpus)
    tasks=manifest['train']
    status('export_inference_all30',selected_recipe=trial)
    def shard(index):
        for sample in tasks[index::workers]:
            for stage in ['track2-infer','track2-convert']:
                run(evaluation_command(stage,trial,sample),[gpus[index]],
                    'export_'+stage+'_'+sample,deadline)
    parallel(shard,workers)
    receipts={}
    for sample in sorted(manifest['samples']):
        receipts[sample]=export_sample(trial,sample,manifest)
        save_json(OUTPUT/'export_receipts.json',dict(trial=trial,samples=receipts,
            updated_unix=time.time()))
    assert len(receipts)==30
    return receipts


def audit(manifest,selection,receipts):
    checks={}
    checks['sequence_split_20_10']=len(manifest['train'])==20 and len(manifest['test'])==10 and not set(manifest['train'])&set(manifest['test'])
    launch=read(OUTPUT/'initial_training_jobs.json')
    checks['three_two_gpu_trials']=len(launch['jobs'])==3 and all(
        job['cuda_visible_devices']==','.join(map(str,job['gpus'])) and len(job['gpus'])==2
        and '--nproc_per_node=2' in job['command'] for job in launch['jobs'])
    checks['gpu0_preserved']=all(0 not in job['gpus'] for job in launch['jobs'])
    checks['all_trial_updates_positive']=all(read(OUTPUT/'runs'/name/'complete.json')['successful_steps']>0 for name in TRIALS)
    checks['metric_checks']=all(value is True or key=='known_acceleration_cm_per_frame2' for key,value in read(OUTPUT/'metric_checks.json').items())
    checkpoints=read(OUTPUT/'evaluation_checkpoints.json')
    checks['full20_refit_or_selected_initialization']=all(checkpoints[name]['full_train_refit'] or checkpoints[name]['reason']=='Validation selected original initialization' for name in TRIALS)
    checks['no_test_selection']=not read(OUTPUT/'selected.json')['test_used_for_selection']
    checks['three_hour_per_trial_total']=all(read(OUTPUT/'runs'/name/'complete.json')['elapsed_seconds']+
        checkpoints[name].get('complete',{}).get('elapsed_seconds',0)<3*3600 for name in TRIALS)
    comparison=read(OUTPUT/'comparison.json')
    checks['four_models_same_test10']=all(set(v['per_sequence'])==set(manifest['test']) for v in comparison['models'].values()) and len(comparison['models'])==4
    checks['all_test_predictions_target_only_revision2']=True
    for name,model in comparison['models'].items():
        for sample,metric in model['per_sequence'].items():
            folder=OUTPUT/'predictions'/name/sample
            info=read(folder/'inference.json')
            assert info['evaluation_protocol']==metric['evaluation_protocol']=='target_foreground_head_v1'
            assert info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert info['target_mask_sha256']==sha256(SOURCE/sample/'processed/human_mask.mp4')
            assert metric['conversion_revision']==2 and metric['full_frame_roster']
            assert metric['frames']==manifest['samples'][sample]['frames']
            assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
            assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            if name!='original':assert info['checkpoint_sha256']==checkpoints[name]['checkpoint_sha256']
    checks['all30_exports_reloaded']=True
    for sample,receipt in receipts.items():
        path=Path(receipt['path'])
        assert sha256(path)==receipt['sha256']
        with np.load(path) as z:
            for key,shape in [('pose',(manifest['samples'][sample]['frames'],136)),('shape',(45,)),('scales',(68,))]:
                assert z[key].shape==shape and np.isfinite(z[key]).all()
            metadata=json.loads(z['report'].item())
            assert not metadata['gt_pose_identity_used_for_inference']
            assert metadata['trial']==selection
            assert metadata['evaluation_protocol']=='target_foreground_head_v1' and metadata['conversion_revision']==2
            assert metadata['target_foreground_only_input'] and metadata['rgb_inference']['all_accepted_heads_in_target_mask']
            assert metadata['prediction_source_sha256']==sha256(OUTPUT/'predictions'/selection/sample/'human_params_mhr_v2.npz')
            assert str(z['coordinate_frame'])=='opencv_camera'
            assert z['intrinsics'].shape==(manifest['samples'][sample]['frames'],4)
        assert sha256(SOURCE/sample/'human/human_params_gt.npz')==manifest['samples'][sample]['gt_sha256']
    checks['all30_gt_hashes_preserved']=True
    assert all(checks.values()),checks
    save_json(OUTPUT/'final_audit.json',dict(checks=checks,completed_unix=time.time(),
        automated_artifact_audit_only=True,requires_agent_review=True))


def main(pid):
    with (OUTPUT/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stop_children);signal.signal(signal.SIGINT,stop_children)
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        wait_training(pid,deadline)
        manifest=read(OUTPUT/'manifest.json')
        # The user requested both existing init baselines. The independent
        # baseline worker may use the already-released head-trial GPU pair.
        init_start=read(OUTPUT/'init_comparison_start.json')
        init_pid=init_start['pid'];identity=process_identity(init_pid)
        while identity is not None and process_identity(init_pid)==identity:
            if time.time()>=deadline:raise TimeoutError('Deadline while init comparisons are live')
            time.sleep(min(30,max(1,deadline-time.time())))
        assert len(read(OUTPUT/'init_comparison.json')['models'])==2
        select_smplx_checkpoints(manifest,deadline)
        status('validating_evaluation_pipeline')
        evaluation_smoke(manifest,deadline)
        validation_mhr(manifest,deadline)
        selection=full_train_refits(manifest,deadline)
        run_test(manifest,deadline)
        receipts=exports(selection,manifest,deadline)
        audit(manifest,selection,receipts)
        status('artifacts_ready_for_final_audit',selected_recipe=selection,exports=30,
            completed_unix=time.time(),deadline_unix=deadline)
        save_json(OUTPUT/'finalizer.json',dict(pid=os.getpid(),status='artifacts_ready_for_final_audit',
            completed_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-training-pid',type=int,required=True)
    args=parser.parse_args()
    try:main(args.wait_training_pid)
    except BaseException as error:
        save_json(OUTPUT/'finalizer_error.json',dict(error=repr(error),time=time.time()))
        stop_children();report();raise

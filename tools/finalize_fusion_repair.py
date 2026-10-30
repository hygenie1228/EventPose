"""Actual final artifact audit; human agent review remains required afterward."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write, identity
from analyze_fusion_repair import analyze, NAMES
from render_fusion_repair import render, verify_tables
from fusion_repair_scope import PRETRAIN_NAMES,EVAL_NAMES,FUSION_NAMES
GOAL = ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def final_audit(goal):
    import torch
    import h5py
    torch.set_num_threads(2)
    out=Path(goal['output_root'])
    assert (out/'all_training_test_complete.json').exists()
    gates=['source_reuse_audit.json','actual_input_audit.json','current_dataset_audit.json','body_holdout_subject_audit.json','event_normalization_Train.json',
           'real_data_cpu_preflight.json','native_cpu_preflight.json','repair_cpu_preflight.json',
           'gpu_model_pilots_verified.json','ablation_cpu_preflight.json',
           'same_device_extension_cpu.json','source_revision_scope_extension.json',
           'released_RGB_native_cpu.json','trainer_shutdown_guard_verified.json']
    for gate in gates:
        assert read(out/gate)['status']=='passed',gate
    assert set(read(out/'gpu_model_pilots_verified.json')['trials'])==set(NAMES)
    assert read(out/'actual_input_audit.json')['independent_cpu_GT_samples']==54
    assert len(read(out/'current_dataset_audit.json')['records'])==78
    for row in read(out/'reused_groundtruth_hashes.json')['records']:
        assert sha(Path(row['path']))==row['sha256']
    reuse=read(out/'source_reuse_audit.json')
    cache_started=[];cache_ended=[];session_seconds=[]
    for row in reuse['cache_records']:
        p=out/'cache'/(row['session']+'.h5')
        assert p.stat().st_size==row['bytes'] and sha(p)==row['cache_sha256']
        with h5py.File(p,'r',locking=False) as f:
            cache_started.append(float(f.attrs['started_unix']));cache_ended.append(float(f.attrs['completed_unix']))
            session_seconds.append(float(f.attrs['total_seconds']))
    cache_cost=dict(input_cache_bytes=reuse['cache_bytes'],new_cache_files=0,new_cache_bytes=0,
                    prior_wall_span_seconds=max(cache_ended)-min(cache_started),
                    prior_session_wall_seconds_sum=sum(session_seconds),reused_verified=True)
    write(out/'cache_cost.json',cache_cost)
    import numpy as np
    manifest=read(out/'manifest.json');counts={};available={}
    for role in ('training','validation'):
        counts[role]=available[role]=0
        for session in manifest[role+'_sessions']:
            with np.load(out/'source_maps'/(session+'.npz')) as z:
                valid=z['gt_valid'].any(1)
                counts[role]+=int(valid.sum())
                available[role]+=int((valid&z['all_sensors_available'][z['gt_source_positions']]).sum())
    coverage=dict(full_train_valid_frames=counts['training']+counts['validation'],
                  training_valid_frames=counts['training'],training_common_available_frames=available['training'],
                  validation_valid_frames=counts['validation'],fixed_subset_frames=6501)
    write(out/'training_coverage.json',coverage)
    original=read(out/'original_pretrained_manifest.json')
    assert sha(Path(original['path']))==original['sha256']
    state=read(out/'pipeline_state.json'); jobs=state['jobs']+state['history']
    def gpu_proof(pid,role,uuid,physical):
        job=next(j for j in jobs if j['pid']==pid and j['role']==role)
        assert physical in goal['allowed_gpus'] and job['gpu']['physical_gpu']==physical
        assert job['gpu']['uuid']==uuid and job['gpu']['nvml_minor']==physical
        assert job['gpu']['launch_nvml_compute_pids']==[]
        assert job['gpu']['launch_nvml_bus']
        return job['gpu']
    training={};sources={};source_loaded={};checkpoint_records=[];executions=[];implementations=[]
    for name in NAMES:
        config=read(out/'configs'/(name+'.json'));complete=read(out/'runs'/name/'complete.json')
        assert config['architecture']=='epipolar_repair_norm_v2'
        assert config['event_normalization_sha256']==sha(out/'event_normalization_Train.json')
        assert config['event_encoder_gains']==read(out/'event_normalization_Train.json')['event_encoder_gains']
        started=read(out/'runs'/name/'started.json');best=read(out/'runs'/name/'best.json')
        assert complete['status']=='complete' and config['hours']==3 and 'pilot_steps' not in config
        if name in ('rgb3_short90','rgb4'):
            assert started['finish_unix']<=started['started_unix']+90*60+.1
            assert complete['elapsed_seconds']<90*60+300
        assert 0<complete['elapsed_seconds']<4*3600 and 0<complete['successful_steps']<=complete['step']
        assert complete['samples_seen']==16*complete['step'] and complete['subset_frames']==6501
        assert started['training_frames']==6501 and started['validation_frames']==288 and started['batch_size']==16
        assert started['initialization']['sha256']==config['initialization_sha256']
        assert started['initialization']['original_pretrained_sha256']==original['sha256']
        assert started['initialization']['old_fusion_loaded'] is False
        gpu_proof(started['pid'],'train:'+name,started['gpu_uuid'],started['physical_gpu'])
        implementations.append(started['implementation_sha256']);training[name]=complete
        for kind in ('best','last'):
            path=out/'checkpoints'/name/(kind+'.pt');value=torch.load(path,map_location='cpu')
            assert value['config']==config and value['test_used_for_selection'] is False
            assert value['model_delta'] and all(torch.isfinite(v).all() for v in value['model_delta'].values())
            assert sum(v.numel() for v in value['model_delta'].values())==started['trainable_parameters']
            if kind=='best':
                training[name]['best_checkpoint_training_elapsed_seconds']=value['training_elapsed_seconds']
                assert value['metrics_apply_to_checkpoint'] and value['metrics_step']==value['step']==best['step']
                for key,v in value['metrics'].items():assert best[key]==v
            else:
                for key in ('step','successful_steps','samples_seen','unique_frames_seen'):
                    assert value[key]==complete[key]
                assert len(set(value['seen_indices']))==value['unique_frames_seen']<=6501
                assert all(0<=i<6501 for i in value['seen_indices'])
            checkpoint_records.append(dict(name=name,kind=kind,path=str(path),sha256=sha(path),step=value['step'],
                                           configuration_equal=True,tensors_finite=True,test_used_for_selection=False))
            del value
        source=Path(config['initialization_checkpoint'])
        if str(source) not in source_loaded:
            assert sha(source)==config['initialization_sha256']
            value=torch.load(source,map_location='cpu')
            assert value['test_used_for_selection'] is False and value['config']['split_sha256']==config['split_sha256']
            source_loaded[str(source)]={k:value[k] for k in ('samples_seen','unique_frames_seen','training_elapsed_seconds','step','successful_steps')}
            del value
        sources[name]=source_loaded[str(source)]
        execution=read(out/'results'/name/'execution_rank0.json')
        assert execution['GT_used_as_model_input'] is False and execution['condition']==config['condition']
        assert set(execution['pixel_datasets_read'])==set(s for s in (config['primary_sensor'],config['auxiliary_sensor']) if s)
        gpu_proof(execution['pid'],'test:'+name,execution['gpu_uuid'],int(execution['physical_gpu']))
        implementations.append(execution['implementation_sha256']);executions.append(execution)
    for name in PRETRAIN_NAMES:
        config=read(out/'configs'/(name+'.json'));path=out/'checkpoints'/name/'best.pt'
        value=torch.load(path,map_location='cpu')
        assert value['config']==config and config['weight_stage']=='released_pretrained'
        assert value['step']==value['successful_steps']==0 and value['model_delta']=={}
        assert value['released_checkpoint']==original['path'] and value['released_checkpoint_sha256']==original['sha256']
        assert value['test_used_for_selection'] is False
        checkpoint_records.append(dict(name=name,kind='released_source_manifest',path=str(path),sha256=sha(path),step=0,
                                      released_checkpoint=original['path'],released_checkpoint_sha256=original['sha256']))
        execution=read(out/'results'/name/'execution_rank0.json')
        assert execution['GT_used_as_model_input'] is False and execution['condition']==config['condition']
        assert set(execution['pixel_datasets_read'])=={config['primary_sensor']}
        gpu_proof(execution['pid'],'test:'+name,execution['gpu_uuid'],int(execution['physical_gpu']))
        implementations.append(execution['implementation_sha256']);executions.append(execution)
    from multihmr.multisensor_repair import implementation_provenance
    approved=read(out/'source_revision_scope_extension.json')['approved_implementations']
    assert implementation_provenance()==approved[-1]
    assert implementations and all(value in approved for value in implementations)
    for implementation in implementations:
        for digest in implementation.values():assert sha(out/'source_snapshots'/(digest+'.txt'))==digest
    runtime=read(out/'runtime/summary.json')
    assert set(runtime)=={n+'_'+s for n in EVAL_NAMES for s in ('cached','online')}
    uuids={r['gpu_uuid'] for r in runtime.values()};assert len(uuids)==1
    for r in runtime.values():
        assert r['frames']==300 and r['seconds']>0 and abs(r['fps']-300/r['seconds'])<1e-8
        assert r['input_parity_passed'] and r['physical_gpu'] in goal['allowed_gpus']
        assert r['compute_pids_before']==r['compute_pids_after']==[r['pid']]
    ordered=sorted(runtime.values(),key=lambda r:r['process_started_unix'])
    assert all(a['process_finished_unix']<=b['process_started_unix'] for a,b in zip(ordered,ordered[1:]))
    ablations={n:read(out/'ablations'/(n+'.json')) for n in FUSION_NAMES}
    for n,a in ablations.items():
        assert a['status']=='passed' and a['test_used'] is False and a['physical_gpu'] in goal['allowed_gpus']
        assert a['checkpoint_sha256']==next(r['sha256'] for r in checkpoint_records if r['name']==n and r['kind']=='best')
        assert set(a['rows'])=={'intact','missing_aux','zero_aux','swap_aux_polarity'}
        assert all(r['total_gt_instances']==288 and r['GT_used_as_inference_input'] is False for r in a['rows'].values())
        for row in a['rows'].values():
            guarded=row.get('resource_guarded_frames',[])
            assert all(g['native_candidates']>row['maximum_people_per_image'] for g in guarded)
            guarded_frames={g['frame_index'] for g in guarded}
            assert all(int(k.split(':')[0]) not in guarded_frames for k in row['errors_mm'])
            assert row.get('resource_guarded_gt_instances',0)<=288
        for relative,digest in a.get('diagnostic_implementation_sha256',{}).items():
            assert sha(ROOT/relative)==digest and sha(out/'source_snapshots'/(digest+'.txt'))==digest
    result=analyze(goal)
    assert result['raw_recalculation']['passed'] and result['frames']==24080 and result['segments']==80
    # Stop only this Goal's registered observer before hashing the final report.
    stopped=[]
    for job in read(Path(goal['managed_jobs_file']))['jobs']:
        if job['role']!='fusion_repair_report_30min':continue
        current=identity(job['pid'])
        if current is not None and current=={k:job[k] for k in ('pgid','start_ticks','command')}:
            os.killpg(job['pid'],signal.SIGTERM);stopped.append(job['pid'])
    until=time.monotonic()+5
    while any(identity(pid) is not None for pid in stopped) and time.monotonic()<until:time.sleep(.1)
    assert all(identity(pid) is None for pid in stopped)
    report,recommendation=render(goal,result,training,sources,runtime,cache_cost,ablations,coverage)
    proof=read(out/'rendered_table_proof.json');verify_tables(report.read_text(),proof['tables'])
    figures=read(out/'visualization_provenance.json')
    for key in ('chart','image'):
        assert Path(figures[key]).stat().st_size>1000
    learning=read(out/'learning_curve_provenance.json');assert Path(learning['path']).stat().st_size>1000
    assert len(learning['curves'])==6
    audit=dict(status='passed_artifact_audit_agent_review_required',goal_complete=False,checked_unix=time.time(),
               checkpoints=checkpoint_records,source_training=sources,execution_proofs=executions,
               implementation_sha256_by_execution=implementations,approved_scope_revision=approved,original_pretrained=original,
               raw_five_metric_recalculation=result['raw_recalculation'],same_gpu_uuid=next(iter(uuids)),
               runtime_runs=18,all_nine_full_test=True,all_seven_training_complete=True,frame_mapping_verified=True,
               report=str(report),report_sha256=sha(report),numeric_tables_verified=proof['verification'],
               visualization=figures,learning_curve=learning,observer_stopped_pids=stopped,recommendation=recommendation,
               diagnostic_resource_failure_GT={n:{m:r.get('resource_guarded_gt_instances',0) for m,r in a['rows'].items()} for n,a in ablations.items()},
               remaining_agent_review='Inspect actual images, source proofs, report and improvement/limitations before deciding Goal completion')
    write(out/'final_artifact_audit.json',audit)
    print(json.dumps(dict(status=audit['status'],report=str(report),goal_complete=False)),flush=True)


if __name__=='__main__':
    goal=read(GOAL);register_job(goal,os.getpid(),'fusion_repair_final_raw_report_audit');final_audit(goal)

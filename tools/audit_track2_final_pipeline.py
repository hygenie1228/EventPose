"""Audit the complete native/fixed RGB scope against actual final artifacts.

Every missing item remains incomplete. This tool never changes selection,
source NPZs, model scores, or the goal's status.
"""
from pathlib import Path
import argparse,json,random,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
NATIVE='native_rgb_target_head_v1';FIXED='native_rgb_static_crop_target_head_v1'
KEYS=['CD_H_cm','ACC_H_cm_per_frame2']


def read(path):return json.loads(Path(path).read_text())


def main(strict=False):
    manifest=read(OUTPUT/'manifest.json');native_plan=read(OUTPUT/'native_evaluation_plan.json')
    fixed_plan=read(OUTPUT/'fixed_view_adaptation_plan.json')
    surface_plan_path=OUTPUT/'native_surface_adaptation_plan.json'
    surface_plan=read(surface_plan_path) if surface_plan_path.exists() else {}
    surface_status_path=OUTPUT/'native_surface_training_status.json'
    surface_status=read(surface_status_path).get('status') if surface_status_path.exists() else None
    surface_active=(surface_status in ['training_native_surface_trials','all_native_surface_searches_and_Train20_refits_complete']
        or any((OUTPUT/'runs'/c['name']/'started.json').exists() for c in surface_plan.get('trials',[])))
    surface_names=[c['name'] for c in surface_plan.get('trials',[])] if surface_active else []
    native_names=[native_plan['initialization_model'],native_plan['legacy_model'],*native_plan['trials']]
    fixed_names=['diagnostic_original_static_crop',*[c['name'] for c in fixed_plan['trials']]]
    names=native_names+fixed_names+surface_names;protocols={n:NATIVE for n in native_names};protocols.update({n:FIXED for n in fixed_names+surface_names})
    prefix='strict' if strict else 'final'
    if strict:
        from track2_strict_scope import scope
        strict_plan,prefix=scope();strict_models=strict_plan['models']
        strict_by_name={m['name']:m for m in strict_models}
        names=list(strict_by_name);protocols={n:m['evaluation_protocol'] for n,m in strict_by_name.items()}
    baselines=read(OUTPUT/'init_comparison.json')['models']
    reference=baselines['init_v6']['per_sequence'][manifest['test'][0]]['protocol'];checks={}
    def check(name,function):
        try:checks[name]=dict(passed=True,detail=function())
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as error:
            checks[name]=dict(passed=False,detail=str(error) or 'Required evidence not established')
    def checkpoint(name,full):
        if strict:return strict_by_name[name]['full_checkpoint' if full else 'search_checkpoint']
        if name in [native_names[0],fixed_names[0]]:return None
        if name==native_plan['legacy_model']:return native_plan['legacy_full_train' if full else 'legacy_search']['checkpoint']
        prefix='native_surface' if name in surface_names else 'fixed' if name in fixed_names else 'native'
        value=read(OUTPUT/(prefix+('_refit_choices' if full else '_checkpoint_choices'))/(name+'.json'))
        return value['refit']['checkpoint'] if full else value['checkpoint']
    def metric(name,sample,search=False):
        trial='search_'+name if search and name not in [native_names[0],fixed_names[0]] else name
        if strict:trial='search_'+name if search and checkpoint(name,False) is not None else name
        folder=OUTPUT/'predictions'/trial/sample;value=read(folder/'metrics.json');info=read(folder/'inference.json')
        assert info['evaluation_protocol']==value['evaluation_protocol']==protocols[name]
        assert not info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
        assert not info['gt_used_for_detection_or_tracking'] and info.get('depth_decoder','published')=='published'
        ckpt=checkpoint(name,not search);assert info['checkpoint_sha256']==(sha256(Path(ckpt)) if ckpt else None)
        assert info['target_mask_sha256']==sha256(SOURCE/sample/'processed/human_mask.mp4')
        assert value['conversion_revision']==2 and value['full_frame_roster'] and value['protocol']==reference
        assert value['frames']==manifest['samples'][sample]['frames'] and value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
        assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
        if strict:
            from audit_track2_strict_results import audit_case
            _,_,audit=audit_case(strict_by_name[name],sample,not search,manifest,reference)
            assert info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
            review_flags(trial,sample,audit)
        if protocols[name] in [FIXED,'native_rgb_static_crop_strict_mask_target_head_v2','native_rgb_static_crop_strict_mask_body_box_v3']:assert info['static_crop_box']==read(OUTPUT/'fixed_view_all30_plan.json')['samples'][sample]['box']
        return value,info
    def average(summary,values):
        assert set(summary['per_sequence'])==set(values)
        for key in KEYS:
            assert abs(summary[key]-np.mean([v[key] for v in values.values()]))<1e-9
            for sample,value in values.items():assert summary['per_sequence'][sample][key]==value[key]
    def review_flags(trial,sample,audit):
        if not audit['temporal_flags']:return
        reviews=read(OUTPUT/'strict_temporal_manual_reviews.json')['cases']
        reviewed=reviews[trial+'/'+sample]
        assert reviewed['raw_sha256']==audit['raw_sha256'] and reviewed['target_actor_confirmed']
        assert reviewed['reviewed_by_agent'] and reviewed['review_notes']
        assert reviewed['frames']==[f['frames'] for f in audit['temporal_flags']]
        assert reviewed['images'],'Flagged identity changes require actual visual review'
        for image in reviewed['images']:assert sha256(Path(image['path']))==image['sha256']
    def data():
        assert len(manifest['samples'])==30 and len(manifest['train'])==20 and len(manifest['test'])==10
        samples=sorted(manifest['samples']);random.Random(261025).shuffle(samples)
        assert manifest['train']==sorted(samples[:20]) and manifest['test']==sorted(samples[20:])
        assert set(manifest['training']).isdisjoint(manifest['validation'])
        assert len(manifest['training'])==16 and len(manifest['validation'])==4
        assert set(manifest['training'])|set(manifest['validation'])==set(manifest['train'])
        provenance=read(OUTPUT/'init_v6_generation_provenance_audit.json')['samples']
        assert {p.stem for p in (OUTPUT/'work/targets').glob('*.npz')}==set(manifest['train'])
        for sample,record in manifest['samples'].items():
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==record['gt_sha256']
            assert sha256(SOURCE/sample/'human/human_params_init.npz')==provenance[sample]['init_sha256']
            assert sha256(SOURCE/sample/'meta_info.json')==record['metadata_sha256']
            assert sha256(SOURCE/sample/'processed/intrinsic.json')==record['intrinsic_sha256']
            if sample in manifest['train']:
                with np.load(OUTPUT/'work/targets'/(sample+'.npz')) as z:
                    assert z['source_sha256'].item()==record['gt_sha256']
                    for key in ['joints','rotations','betas','transl']:assert np.isfinite(z[key]).all()
        for inputs in read(OUTPUT/'init_input_inventory.json')['inputs'].values():
            for item in inputs.values():assert sha256(Path(item['path']))==item['sha256']
        if surface_names:
            prepared=read(OUTPUT/'native_surface_target_preparation.json')
            assert not prepared['Test_labels_read'] and set(prepared['samples'])==set(manifest['train'])
            assert prepared['total_Train_cached_frames']==1107
            for item in prepared['samples'].values():assert sha256(Path(item['target_cache_path']))==item['target_cache_sha256']
            assert sha256(OUTPUT/'native_surface_target_preparation.json')==surface_plan['target_preparation_sha256']
            assert sha256(OUTPUT/'native_surface_dataset_checks.json')==surface_plan['dataset_checks_sha256']
        return 'Frozen20/10 split, Train16/4, all20 actual SMPL-X labels; all30 GT/init_v6 and Test init_v0 unchanged'
    def training():
        import torch
        torch.set_num_threads(2)
        native=read(OUTPUT/'native_refit_mappings.json');fixed=read(OUTPUT/'fixed_refit_mappings.json')
        surface=read(OUTPUT/'native_surface_refit_mappings.json') if surface_names else {}
        original=torch.load(ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt',map_location='cpu',mmap=True)['model_state_dict']
        events=[json.loads(line) for line in (OUTPUT/'job_history.jsonl').read_text().splitlines()]
        required=native_plan['trials']+[c['name'] for c in fixed_plan['trials']]+surface_names
        assert len(required)==6+len(surface_names)
        for name in required:
            cfg=read(OUTPUT/'runs'/name/'configuration.json');done=read(OUTPUT/'runs'/name/'complete.json')
            started=read(OUTPUT/'runs'/name/'started.json');result=(surface if name in surface else fixed if name in fixed else native)[name]
            assert started['world_size']==2 and done['successful_steps']>0
            assert cfg['augment'] and cfg['weight_decay']>0 and cfg['anchor_l2_weight']>0
            assert not cfg['target_foreground'] and cfg['validation_target_head_gate'] and cfg['no_test_selection']
            if name in surface_names:
                assert cfg['native_surface_target'] and not cfg.get('refit_init_checkpoint')
                if strict:assert cfg['strict_target_mask']
                if prefix.startswith('body_target'):assert cfg['target_body_overlap']
                assert result['training_lineage_seconds']==0
                assert read(OUTPUT/'native_surface_checkpoint_choices'/(name+'.json'))['metrics']['cd_proxy_target_space']=='native_MHR_GT'
            assert result['search_seconds']+result['refit_seconds']+result.get('training_lineage_seconds',0)+result.get('recovery_attempt_seconds',0)<10800
            assert result['search_seconds']==done['elapsed_seconds']
            if name in native:
                ancestor=Path(cfg['refit_init_checkpoint']).parent.name
                lineage=cfg['bootstrap_elapsed_seconds']+read(OUTPUT/'runs'/ancestor/'complete.json')['elapsed_seconds']
                assert result['training_lineage_seconds']==lineage
            assert result['complete_training_cost_below3h'] and not result['test_used_for_selection']
            jobs=[e for e in events if e.get('event')=='started' and e.get('label')==name]
            recovery=read(OUTPUT/'native_surface_mask_decode_recovery.json') if (OUTPUT/'native_surface_mask_decode_recovery.json').exists() else {}
            previous=recovery.get('previous_attempts',{}).get(name,{})
            expected_jobs=previous.get('recorded_search_launches_before_recovery',0)+1 if previous.get('resume_or_retry_required') else 1
            assert len(jobs)==expected_jobs and all(len(j['gpus'])==2 and set(j['gpus'])<=set([1,2,4,5]) for j in jobs)
            if expected_jobs>1:
                exits={e['pid']:e for e in events if e.get('event')=='exited' and e.get('label')==name}
                for old,new in zip(jobs,jobs[1:]):assert exits[old['pid']]['time']<=new['started_unix'],'Concurrent duplicate training'
                assert result.get('recovery_attempt_seconds',0)==previous['additional_nonresumed_attempt_seconds']
            assert '--nproc_per_node=2' in jobs[0]['command']
            refit=result['refit'];assert refit['full_train_refit'] and sha256(Path(refit['checkpoint']))==refit['checkpoint_sha256']
            saved=torch.load(refit['checkpoint'],map_location='cpu',mmap=True)
            assert saved['model_delta']
            changed=0
            for key,tensor in saved['model_delta'].items():
                assert key in original and tensor.shape==original[key].shape and torch.isfinite(tensor).all()
                changed+=not torch.equal(tensor,original[key])
            if refit['refit_performed_this_wave']:
                assert saved['step']>0 and changed>0
                assert refit['complete']['successful_steps']>0 and not refit['complete']['validation_used_for_selection']
                assert result['refit_seconds']==refit['complete']['elapsed_seconds']
                refit_name=Path(refit['checkpoint']).parent.name
                refit_cfg=read(OUTPUT/'runs'/refit_name/'configuration.json')
                assert refit_cfg['augment'] and refit_cfg['weight_decay']>0 and refit_cfg['anchor_l2_weight']>0
                assert refit_cfg['no_test_selection'] and refit_cfg['fixed_full_train'] and refit_cfg['training_role']=='full_training'
                assert not refit_cfg['target_foreground'] and refit_cfg['validation_target_head_gate']
                if name in surface_names:
                    assert refit_cfg['native_surface_target']
                    if strict:assert refit_cfg['strict_target_mask']
                    if prefix.startswith('body_target'):assert refit_cfg['target_body_overlap']
                assert read(OUTPUT/'runs'/refit_name/'started.json')['world_size']==2
                refit_jobs=[e for e in events if e.get('event')=='started' and e.get('label')==refit_name]
                assert len(refit_jobs)==1 and len(refit_jobs[0]['gpus'])==2 and '--nproc_per_node=2' in refit_jobs[0]['command']
            else:
                assert saved['step']==0 and changed==0,'A non-refitted checkpoint must be the unchanged original initialization'
        assert all(0 not in e.get('gpus',[]) for e in events if e.get('event')=='started')
        return f'{len(required)} current trials with actual two-GPU updates, finite changed checkpoints, three regularizers and complete training lineage below3h; GPU0 preserved'
    def selection():
        selected=read(OUTPUT/(prefix+'_selected.json'));assert set(selected['models'])==set(names)
        if strict:assert len(names)==9+len(surface_names),'Include every launched native-surface candidate in the corrected final scope'
        assert not selected['test_used_for_selection'] and selected['model_order']==names
        eligible=[]
        for name,summary in selected['models'].items():
            results={s:metric(name,s,True) for s in manifest['validation']}
            average(summary,{s:v[0] for s,v in results.items()})
            coverage=float(np.mean([v[1]['detection_frame_rate'] for v in results.values()]))
            assert abs(coverage-selected['target_coverage'][name])<1e-12
        for name in names:
            if selected['target_coverage'][name]>=.95:eligible.append(name)
        assert selected['eligible_trials']==eligible
        winner=min(eligible,key=lambda n:(selected['models'][n]['CD_H_cm'],selected['models'][n]['ACC_H_cm_per_frame2'],names.index(n)))
        assert winner==selected['trial'] and selected['evaluation_protocol']==protocols[winner]
        if strict:
            events=[json.loads(line) for line in (OUTPUT/'job_history.jsonl').read_text().splitlines()]
            base_selection=read(OUTPUT/('body_target_selected.json' if prefix.startswith('body_target') else 'strict_selected.json'))
            for name in names:
                for sample in manifest['test']:
                    label=('native_surface_final_track2-score_' if name in surface_names else 'strict_score_')+name+'_'+sample
                    chosen=selected['selected_unix'] if name in surface_names else base_selection['selected_unix']
                    jobs=[e for e in events if e.get('event')=='started' and e.get('label')==label]
                    assert jobs and min(e['started_unix'] for e in jobs)>=chosen,(name,sample,'Test scoring preceded Train-only selection')
        if surface_names:
            frozen=read(OUTPUT/'native_surface_selected.json')
            assert frozen==selected and selected['selected_unix']<read(OUTPUT/'native_surface_test_comparison.json')['completed_unix']
            events=[json.loads(line) for line in (OUTPUT/'job_history.jsonl').read_text().splitlines()]
            for trial in surface_names:
                for sample in manifest['test']:
                    jobs=[e for e in events if e.get('event')=='started' and e.get('label')=='native_surface_final_track2-infer_'+trial+'_'+sample]
                    assert jobs and min(e['started_unix'] for e in jobs)>=selected['selected_unix']
        return f'Actual minimum of {len(names)} same-roster Train4 pipelines; coverage gate and ACC tie-break independently recomputed'
    def test():
        assert reference['scorer_sha256']==sha256(ROOT/'src/utils/v2d_official_metric.py'),'Current metric implementation differs from the frozen scoring protocol'
        assert reference['roles']==read(OUTPUT/'body_roles.json')['roles'],'Current MHR vertex roles differ from frozen baselines'
        comparison=read(OUTPUT/(prefix+'_test_comparison.json'));assert set(comparison['models'])==set(names)
        assert not comparison['test_used_for_selection']
        if not strict:assert comparison['input_protocols']==protocols
        for name,summary in comparison['models'].items():
            values={s:metric(name,s)[0] for s in manifest['test']};average(summary,values)
            assert sum(v['frames'] for v in values.values())==7610
        for name,summary in baselines.items():
            assert set(summary['per_sequence'])==set(manifest['test']);average(summary,summary['per_sequence'])
            for sample,value in summary['per_sequence'].items():
                assert value['frames']==manifest['samples'][sample]['frames'] and value['protocol']==reference
                assert value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
                assert value['prediction_sha256']==read(OUTPUT/'init_input_inventory.json')['inputs'][sample][name]['sha256']
        return f'{len(names)} models and both init baselines; identical Test10, original7610frames, MHR-v2 and equal sequence weights'
    def exports():
        winner=read(OUTPUT/(prefix+'_selected.json'))['trial'];receipts=read(OUTPUT/(prefix+'_export_receipts.json'))
        ckpt=checkpoint(winner,True);expected_checkpoint_sha=sha256(Path(ckpt)) if ckpt else None
        assert receipts['trial']==winner and set(receipts['samples'])==set(manifest['samples'])
        for sample,item in receipts['samples'].items():
            path=SOURCE/sample/'human/human_params_uranus.npz';pred=OUTPUT/'predictions'/winner/sample/'human_params_mhr_v2.npz';r=manifest['samples'][sample]
            assert sha256(path)==item['sha256']
            if strict:
                from audit_track2_strict_results import audit_heads
                independent=audit_heads(pred.parent,sample,manifest)
                review_flags(winner,sample,independent)
                audit=read(pred.parent/'strict_head_mask_audit.json')
                info=read(pred.parent/'inference.json')
                assert info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
                assert audit['outside_original_mask']==0 and audit['frames']==r['frames']
                assert audit['raw_sha256']==sha256(pred.parent/'smplx_predictions.npz')
                assert audit['mask_sha256']==sha256(SOURCE/sample/'processed/human_mask.mp4')
            with np.load(path) as z,np.load(pred) as source:
                for key,shape in [('pose',(r['frames'],136)),('shape',(45,)),('scales',(68,))]:
                    assert z[key].shape==shape and np.isfinite(z[key]).all() and np.array_equal(z[key],source[key])
                report=json.loads(z['report'].item());info=report['rgb_inference']
                assert report['trial']==winner and report['evaluation_protocol']==protocols[winner] and report['conversion_revision']==2
                assert info['checkpoint_sha256']==expected_checkpoint_sha and report['checkpoint_sha256']==expected_checkpoint_sha
                assert not report['gt_pose_identity_used_for_inference'] and not report['target_foreground_only_input']
                assert info['all_accepted_heads_in_target_mask'] and info.get('depth_decoder','published')=='published'
                assert report['prediction_source_sha256']==sha256(pred)
                assert str(z['coordinate_frame'])=='opencv_camera' and str(z['body_model'])=='MHR' and not bool(z['GT_used'])
                assert np.array_equal(z['K'],np.asarray(r['K'])) and float(z['fps'])==r['fps']
                assert np.array_equal(z['world_to_camera'],np.asarray(r['world_to_camera']))
                assert z['intrinsics'].shape==(r['frames'],4)
                assert np.array_equal(z['camera_extrinsics'],np.broadcast_to(np.eye(4),(r['frames'],4,4)))
                if protocols[winner] in [FIXED,'native_rgb_static_crop_strict_mask_target_head_v2','native_rgb_static_crop_strict_mask_body_box_v3']:assert info['static_crop_box']==read(OUTPUT/'fixed_view_all30_plan.json')['samples'][sample]['box']
                with np.load(pred.parent/'smplx_predictions.npz') as raw:
                    assert np.array_equal(z['valid_input'],raw['detection_valid'])
                    assert np.array_equal(z['detection_confidence'],raw['confidence'])
        return 'All30 actual requested source NPZs reloaded; exact selected MHR arrays, camera/fps/shape metadata and source hashes'
    def performance():
        selected=read(OUTPUT/(prefix+'_selected.json'))['trial'];models=read(OUTPUT/(prefix+'_test_comparison.json'))['models']
        if strict:
            import torch
            ckpt=checkpoint(selected,True)
            assert ckpt is not None,'Train winner was an unadapted model'
            saved=torch.load(ckpt,map_location='cpu',mmap=True)
            assert saved['step']>0,'Selected checkpoint contains no fine-tuning updates'
        else:assert selected not in [native_names[0],fixed_names[0]],'Train winner was an unadapted model'
        if selected in fixed_names+surface_names:
            recipe_prefix='native_surface' if selected in surface_names else 'fixed'
            assert read(OUTPUT/(recipe_prefix+'_checkpoint_choices')/(selected+'.json'))['metrics']['step']>0
        actual=models[selected]['CD_H_cm'];initial=baselines['init_v6']['CD_H_cm']
        assert actual<initial,f'Selected Test CD-H {actual:.6f}cm has not beaten init_v6 {initial:.6f}cm'
        return dict(selected=selected,CD_H_cm=actual,init_v6_CD_H_cm=initial,improvement_percent=100*(initial-actual)/initial)
    def visual():
        winner=read(OUTPUT/(prefix+'_selected.json'))['trial'];receipt=read(OUTPUT/'target_foreground_visuals'/(winner+'_exported')/'all/receipt.json')
        assert not receipt['manual_review_pending'] and receipt['exported_files_rendered'] and receipt['frames_rendered']==90
        assert set(receipt['samples'])==set(manifest['samples']) and set(receipt['manual_reviewed_samples'])==set(manifest['samples'])
        for sample,item in receipt['samples'].items():
            assert sha256(Path(item['path']))==item['sha256'] and sha256(SOURCE/sample/'human/human_params_uranus.npz')==item['native_prediction_sha256']
            assert item['frames']==[0,manifest['samples'][sample]['frames']//2,manifest['samples'][sample]['frames']-1]
        return 'All30 actual exports manually reviewed at first/middle/last90 samples; full-frame identity correctness is not inferred from samples alone'
    def reports():
        values=dict(baselines,**read(OUTPUT/(prefix+'_test_comparison.json'))['models'])
        aliases={native_names[0]:'original_RGB',fixed_names[0]:'original_fixed_crop'}
        for path in [ROOT/'_agents/report/261025_트랙2_MultiHMR_미세조정_진행.md',ROOT/'_agents/research/261025/track2_multihmr_progress.md']:
            text=path.read_text().replace('**','');assert text.index('## init_v6 / init_v0')<1000
            for name,value in values.items():assert f'| {aliases.get(name,name)} | 10/10 | {value["CD_H_cm"]:.4f} | {value["ACC_H_cm_per_frame2"]:.4f} |' in text
            assert read(OUTPUT/(prefix+'_selected.json'))['trial'] in text
        return 'Both persistent Markdown files show init_v6/init_v0 front comparisons and complete current native/fixed values'
    def assets():
        assert (OUTPUT/(prefix+'_export_receipts.json')).exists(),'Defer expensive final asset hashing until final exports exist'
        for name,value in read(OUTPUT/'model_assets.json')['assets'].items():assert sha256(Path(value['path']))==value['sha256'],name
        return 'Original Multi-HMR, official MHR and neutral SMPL-X assets match pinned starting hashes'
    def window():
        budget=read(OUTPUT/'budget.json');extension=read(OUTPUT/'user_time_extension.json')
        assert budget['total_hours']==28 and extension['added_hours']==16
        assert budget['deadline_unix']==extension['extended_deadline_unix']
        assert abs(budget['deadline_unix']-budget['started_unix']-28*3600)<1
        assert time.time()>=budget['deadline_unix'],'User-authorized28h work window remains active'
        return 'Original start preserved and authorized12h+16h window elapsed'
    for name,function in [('immutable_data_split_and_labels',data),('bounded_regularized_two_GPU_training',training),
        (('twelve_pipeline_Train_only_selection' if surface_names else 'nine_pipeline_Train_only_selection'),selection),('full_Test_comparisons_and_init_baselines',test),
        ('actual_final_all30_MHR_exports',exports),('fine_tuned_result_beats_init_v6',performance),
        ('actual_export_visual_review',visual),('two_persistent_report_tables',reports),('pinned_model_assets',assets),
        ('authorized_extended_work_window',window)]:check(name,function)
    result=dict(checks=checks,all_required_checks_passed=all(v['passed'] for v in checks.values()),
        checked_unix=time.time(),requires_agent_review=True,goal_status_not_changed=True)
    result['strict_target_mask']=strict
    save_json(OUTPUT/(prefix+'_completion_scope_audit.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict',action='store_true',help='Audit corrected, unexpanded-mask evaluation and exports')
    main(parser.parse_args().strict)

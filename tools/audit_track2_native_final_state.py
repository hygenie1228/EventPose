"""Independent CPU audit of the full native-RGB goal scope and final outputs.

Missing evidence is recorded as incomplete. This never marks the goal complete,
changes model selection, or writes model/dataset files.
"""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
PROTOCOL='native_rgb_target_head_v1'
METRICS=['CD_H_cm','ACC_H_cm_per_frame2']


def read(path):return json.loads(Path(path).read_text())


def main():
    manifest=read(OUTPUT/'manifest.json');plan=read(OUTPUT/'native_evaluation_plan.json')
    names=[plan['initialization_model'],plan['legacy_model']]+plan['trials'];checks={}
    reference=read(OUTPUT/'init_comparison.json')['models']['init_v6']['per_sequence'][manifest['test'][0]]['protocol']
    def check(name,function):
        try:checks[name]=dict(passed=True,detail=function())
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as error:
            checks[name]=dict(passed=False,detail=str(error) or 'Required assertion not satisfied')
    def checkpoint(name,full):
        if name==plan['initialization_model']:return None
        if name==plan['legacy_model']:return plan['legacy_full_train' if full else 'legacy_search']['checkpoint']
        value=read(OUTPUT/('native_refit_choices' if full else 'native_checkpoint_choices')/(name+'.json'))
        return value['refit']['checkpoint'] if full else value['checkpoint']
    def metric(trial,sample,ckpt):
        folder=OUTPUT/'predictions'/trial/sample;value=read(folder/'metrics.json');info=read(folder/'inference.json')
        assert value['evaluation_protocol']==info['evaluation_protocol']==PROTOCOL
        assert not info['target_foreground_only_input'] and info['input_view']=='native_rgb'
        assert info.get('depth_decoder','published')=='published'
        assert info['all_accepted_heads_in_target_mask'] and not info['gt_used_for_detection_or_tracking']
        assert info['target_mask_sha256']==sha256(SOURCE/sample/'processed/human_mask.mp4')
        assert info['checkpoint_sha256']==(sha256(Path(ckpt)) if ckpt else None)
        assert value['conversion_revision']==2 and value['protocol']==reference and value['full_frame_roster']
        assert value['frames']==manifest['samples'][sample]['frames']
        assert value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
        assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
        return value
    def verify_average(summary,values):
        assert set(summary['per_sequence'])==set(values)
        for key in METRICS:
            assert abs(summary[key]-np.mean([v[key] for v in values.values()]))<1e-9
            for sample,value in values.items():assert summary['per_sequence'][sample][key]==value[key]
    def data():
        assert len(manifest['train'])==20 and len(manifest['test'])==10 and len(manifest['samples'])==30
        assert not set(manifest['train'])&set(manifest['test'])
        assert set(manifest['training'])|set(manifest['validation'])==set(manifest['train'])
        assert len(manifest['training'])==16 and len(manifest['validation'])==4
        provenance=read(OUTPUT/'init_v6_generation_provenance_audit.json')['samples']
        for sample,record in manifest['samples'].items():
            assert sha256(SOURCE/sample/'human/human_params_gt.npz')==record['gt_sha256']
            assert sha256(SOURCE/sample/'human/human_params_init.npz')==provenance[sample]['init_sha256']
        for inputs in read(OUTPUT/'init_input_inventory.json')['inputs'].values():
            for value in inputs.values():assert sha256(Path(value['path']))==value['sha256']
        return 'Frozen20/10 and16/4 splits; all30 GT/init_v6 and both Test init baselines immutable'
    def training():
        mapping=read(OUTPUT/'native_refit_mappings.json');cfgs=read(OUTPUT/'native_trajectory_plan.json')['trials']
        assert len(cfgs)==3 and set(mapping)==set(plan['trials'])
        events=[json.loads(line) for line in (OUTPUT/'job_history.jsonl').read_text().splitlines()]
        for cfg in cfgs:
            name=cfg['name'];done=read(OUTPUT/'runs'/name/'complete.json');result=mapping[name]
            actual=read(OUTPUT/'runs'/name/'configuration.json')
            assert read(OUTPUT/'runs'/name/'started.json')['world_size']==2
            for key in ['augment','weight_decay','anchor_l2_weight','target_foreground','validation_target_head_gate','no_test_selection']:
                assert actual[key]==cfg[key],('Actual configuration differs from registered plan',name,key)
            assert cfg['augment'] and cfg['weight_decay']>0 and cfg['anchor_l2_weight']>0
            assert not cfg['target_foreground'] and cfg['validation_target_head_gate'] and cfg['no_test_selection']
            assert done['successful_steps']>0 and result['complete_training_cost_below3h']
            assert result['training_lineage_seconds']+result['search_seconds']+result['refit_seconds']<10800
            jobs=[e for e in events if e.get('event')=='started' and e.get('label')==name]
            assert len(jobs)==1 and len(jobs[0]['gpus'])==2 and set(jobs[0]['gpus'])<=set([1,2,4,5])
            assert '--nproc_per_node=2' in jobs[0]['command']
            refit=result['refit'];assert sha256(Path(refit['checkpoint']))==refit['checkpoint_sha256']
            assert refit['full_train_refit'] and not result['test_used_for_selection']
            if refit['refit_performed_this_wave']:
                assert refit['complete']['successful_steps']>0 and not refit['complete']['validation_used_for_selection']
                refit_name=Path(refit['checkpoint']).parent.name
                refit_actual=read(OUTPUT/'runs'/refit_name/'configuration.json')
                assert refit_actual['training_role']=='full_training'
                assert read(OUTPUT/'runs'/refit_name/'started.json')['world_size']==2
                for key in ['augment','weight_decay','anchor_l2_weight','target_foreground']:
                    assert refit_actual[key]==actual[key]
                refit_jobs=[e for e in events if e.get('event')=='started' and e.get('label')==refit_name]
                assert len(refit_jobs)==1 and len(refit_jobs[0]['gpus'])==2
                assert set(refit_jobs[0]['gpus'])<=set([1,2,4,5])
                assert '--nproc_per_node=2' in refit_jobs[0]['command']
        return 'All three native trials: actual two-GPU updates, three regularizers, fullTrain refits, complete lineage below3h'
    def validation():
        selected=read(OUTPUT/'native_selected.json');assert set(selected['models'])==set(names)
        assert not selected['test_used_for_selection'] and selected['evaluation_protocol']==PROTOCOL
        eligible=[]
        for name,summary in selected['models'].items():
            trial=name if name==plan['initialization_model'] else 'search_'+name
            values={s:metric(trial,s,checkpoint(name,False)) for s in manifest['validation']}
            verify_average(summary,values)
            coverage=np.mean([read(OUTPUT/'predictions'/trial/s/'inference.json')['detection_frame_rate'] for s in values])
            assert abs(coverage-selected['target_coverage'][name])<1e-12
            if coverage>=.95:eligible.append(name)
        assert eligible==selected['eligible_trials']
        winner=min(eligible,key=lambda n:(selected['models'][n]['CD_H_cm'],selected['models'][n]['ACC_H_cm_per_frame2'],names.index(n)))
        assert winner==selected['trial']
        return 'All five same-input Train4 MHR comparisons; eligible Train minimum independently recomputed'
    def test():
        result=read(OUTPUT/'native_test_comparison.json')
        assert set(result['models'])==set(names) and not result['test_used_for_selection']
        for name,summary in result['models'].items():
            values={s:metric(name,s,checkpoint(name,True)) for s in manifest['test']}
            assert sum(v['frames'] for v in values.values())==7610
            verify_average(summary,values)
        return 'Five models × Test10 × every original frame; identical metric roles/math, revision2 and equal-sequence averages'
    def exports():
        receipts=read(OUTPUT/'native_export_receipts.json');winner=read(OUTPUT/'native_selected.json')['trial']
        assert receipts['trial']==winner and set(receipts['samples'])==set(manifest['samples'])
        for sample,item in receipts['samples'].items():
            path=SOURCE/sample/'human/human_params_uranus.npz';record=manifest['samples'][sample]
            assert sha256(path)==item['sha256']
            source=OUTPUT/'predictions'/winner/sample/'human_params_mhr_v2.npz'
            with np.load(path) as z,np.load(source) as pred:
                for key,shape in [('pose',(record['frames'],136)),('shape',(45,)),('scales',(68,))]:
                    assert z[key].shape==shape and np.isfinite(z[key]).all() and np.array_equal(z[key],pred[key])
                report=json.loads(z['report'].item());info=report['rgb_inference']
                assert report['trial']==winner and report['evaluation_protocol']==PROTOCOL and report['conversion_revision']==2
                assert not report['target_foreground_only_input'] and not report['gt_pose_identity_used_for_inference']
                assert info['all_accepted_heads_in_target_mask'] and info.get('depth_decoder','published')=='published'
                assert report['prediction_source_sha256']==sha256(source)
                assert str(z['coordinate_frame'])=='opencv_camera' and str(z['body_model'])=='MHR' and not bool(z['GT_used'])
                assert np.array_equal(z['K'],np.asarray(record['K'])) and float(z['fps'])==record['fps']
                assert z['intrinsics'].shape==(record['frames'],4)
                assert np.array_equal(z['camera_extrinsics'],np.broadcast_to(np.eye(4),(record['frames'],4,4)))
        return 'All30 actual source Uranus NPZs reloaded; exact selected arrays, native MHR/camera/fps metadata and source hashes'
    def performance():
        winner=read(OUTPUT/'native_selected.json')['trial'];models=read(OUTPUT/'native_test_comparison.json')['models']
        baseline=read(OUTPUT/'init_comparison.json')['models']['init_v6']['CD_H_cm']
        assert winner!=plan['initialization_model'],'Train winner was the unadapted model'
        value=models[winner]['CD_H_cm'];assert value<baseline,f'Selected CD-H {value:.6f}cm has not beaten init_v6 {baseline:.6f}cm'
        return dict(selected=winner,CD_H_cm=value,init_v6_CD_H_cm=baseline,improvement_percent=100*(baseline-value)/baseline)
    for name,function in [('immutable_data_and_splits',data),('actual_regularized_two_GPU_training',training),
        ('Train_only_selection',validation),('complete_native_Test_comparison',test),('actual_all30_exports',exports),
        ('user_init_v6_performance_requirement',performance)]:check(name,function)
    save_json(OUTPUT/'native_completion_scope_audit.json',dict(checks=checks,all_checks_passed=all(v['passed'] for v in checks.values()),
        checked_unix=time.time(),goal_not_automatically_marked_complete=True,
        remaining_manual_scope=['Current selected export visual review','Both persistent report tables match current native results','Extended work-window requirement','Original initial-trial scope and model assets checks']))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

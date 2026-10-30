"""Audit the requested Track2 scope from current artifacts, without GPU work."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
PROTOCOL='target_foreground_head_v1'
INITIAL=['head_strong','last2_strong','last4_strong']


def read(path):return json.loads(path.read_text())


def main():
    manifest=read(OUTPUT/'manifest.json');checks={}
    def check(name,run):
        try:
            detail=run();checks[name]=dict(passed=True,detail=detail)
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as error:
            checks[name]=dict(passed=False,detail=str(error) or 'Required assertion not satisfied')
    def split():
        assert len(manifest['train'])==20 and len(manifest['test'])==10
        assert set(manifest['train']).isdisjoint(manifest['test'])
        assert len(manifest['training'])==16 and len(manifest['validation'])==4
        assert set(manifest['training']).isdisjoint(manifest['validation'])
        assert set(manifest['train'])==set(manifest['training'])|set(manifest['validation'])
        assert set(manifest['samples'])==set(manifest['train'])|set(manifest['test'])
        for name,record in manifest['samples'].items():
            assert sha256(SOURCE/name/'human/human_params_gt.npz')==record['gt_sha256']
        return '30 immutable GT inputs; 20/10 split and Train-only 16/4 validation'
    def training():
        jobs=read(OUTPUT/'initial_training_jobs.json')['jobs']
        assert {j['label'] for j in jobs}==set(INITIAL)
        assert all(len(j['gpus'])==2 and 0 not in j['gpus'] and '--nproc_per_node=2' in j['command'] for j in jobs)
        starts=[];ends=[]
        mapping=read(OUTPUT/'evaluation_checkpoints.json')
        for name in INITIAL:
            cfg=read(OUTPUT/'configs'/(name+'.json'));start=read(OUTPUT/'runs'/name/'started.json');done=read(OUTPUT/'runs'/name/'complete.json')
            assert cfg['augment'] and cfg['weight_decay']>0 and cfg['anchor_l2_weight']>0
            assert done['successful_steps']>0
            assert done['elapsed_seconds']+mapping[name].get('complete',{}).get('elapsed_seconds',0)<10800
            assert mapping[name]['full_train_refit'] or mapping[name].get('reason')=='Validation selected original initialization'
            starts.append(start['started_unix']);ends.append(start['started_unix']+done['elapsed_seconds'])
        assert max(starts)<min(ends),'The three initial trials did not overlap'
        return 'Three concurrent two-GPU regularized trials; each search + Train20 refit below3h'
    def labels():
        quality=read(OUTPUT/'label_quality_checks.json')
        assert set(quality['samples'])==set(manifest['train'])
        for name in manifest['train']:
            with np.load(OUTPUT/'work/targets'/(name+'.npz')) as z:
                for key in ['rotations','betas','joints','transl']:assert np.isfinite(z[key]).all(),(name,key)
        return 'Actual fitted SMPL-X supervision reloaded for all20 Train sequences'
    def baselines():
        inventory=read(OUTPUT/'init_input_inventory.json')['inputs'];comparison=read(OUTPUT/'init_comparison.json')['models']
        assert set(inventory)==set(manifest['test'])
        for name,inputs in inventory.items():
            for baseline in ['init_v6','init_v0']:
                assert sha256(Path(inputs[baseline]['path']))==inputs[baseline]['sha256']
                assert inputs[baseline]['frames']==manifest['samples'][name]['frames']
        for baseline in ['init_v6','init_v0']:
            model=comparison[baseline];assert set(model['per_sequence'])==set(manifest['test'])
            for metric in ['CD_H_cm','ACC_H_cm_per_frame2']:
                assert abs(model[metric]-np.mean([v[metric] for v in model['per_sequence'].values()]))<1e-9
        return 'Both supplied init baselines: same complete Test10 and unchanged input hashes'
    def comparison():
        result=read(OUTPUT/'comparison.json');assert result['evaluation_protocol']==PROTOCOL and result['conversion_revision']==2
        required={'original',*INITIAL}
        followup=OUTPUT/'target_pooled_followup_plan.json'
        if followup.exists():required.update(cfg['name'] for cfg in read(followup)['trials'])
        assert required<=set(result['models']),'Accepted initial/follow-up comparison scope is incomplete'
        assert result['test_used_for_selection'] is False
        mapping=read(OUTPUT/'evaluation_checkpoints.json')
        reference=read(OUTPUT/'init_comparison.json')['models']['init_v6']['per_sequence'][manifest['test'][0]]['protocol']
        for model,summary in result['models'].items():
            assert set(summary['per_sequence'])==set(manifest['test'])
            for name,metric in summary['per_sequence'].items():
                folder=OUTPUT/'predictions'/model/name;info=read(folder/'inference.json')
                assert info['evaluation_protocol']==metric['evaluation_protocol']==PROTOCOL
                assert info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
                assert info['target_mask_sha256']==sha256(SOURCE/name/'processed/human_mask.mp4')
                assert metric['conversion_revision']==2 and metric['frames']==manifest['samples'][name]['frames'] and metric['full_frame_roster']
                assert metric['gt_sha256']==manifest['samples'][name]['gt_sha256']
                assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                assert metric['protocol']==reference,'Metric formulas, vertex roles or alignment differ across models'
                if model!='original':assert info['checkpoint_sha256']==mapping[model]['checkpoint_sha256']
            for key in ['CD_H_cm','ACC_H_cm_per_frame2']:
                assert abs(summary[key]-np.mean([v[key] for v in summary['per_sequence'].values()]))<1e-9
        return f"{len(result['models'])} models on all Test frames; target head/foreground policy and revision2 verified"
    def selection():
        validation=read(OUTPUT/'validation_mhr_comparison.json');selected=read(OUTPUT/'selected.json')
        assert selected['test_used_for_selection'] is False and validation['test_used_for_selection'] is False
        for model,value in validation['models'].items():
            assert set(value['per_sequence'])==set(manifest['validation'])
            for name,metric in value['per_sequence'].items():
                assert metric['evaluation_protocol']==PROTOCOL and metric['conversion_revision']==2
                assert metric['frames']==manifest['samples'][name]['frames'] and metric['full_frame_roster']
                assert metric['gt_sha256']==manifest['samples'][name]['gt_sha256']
                prediction=Path(metric['prediction_path']);assert sha256(prediction)==metric['prediction_sha256']
                info=read(prediction.parent/'inference.json')
                assert info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
                assert info['target_mask_sha256']==sha256(SOURCE/name/'processed/human_mask.mp4')
            for key in ['CD_H_cm','ACC_H_cm_per_frame2']:
                assert abs(value[key]-np.mean([v[key] for v in value['per_sequence'].values()]))<1e-9
        assert selected['trial']==validation['selected_trial']
        eligible=validation.get('eligible_trials',['original',*INITIAL])
        assert 'original' in eligible and set(eligible)<=set(validation['models'])
        cd_selection=read(OUTPUT/'internal_cd_h_selection.json')
        minimum=read(OUTPUT/'target_foreground_protocol.json')['minimum_target_detection_coverage']
        for name in eligible:
            if name!='original':assert cd_selection['trials'][name]['metrics']['coverage']>=minimum
        winner=min(eligible,key=lambda n:(validation['models'][n]['CD_H_cm'],validation['models'][n]['ACC_H_cm_per_frame2'],eligible.index(n)))
        assert selected['trial']==winner,'Saved recipe is not the actual eligible Train-validation minimum'
        return 'Four full Train-validation videos; actual CD-H minimum and ACC-H tie-break independently recomputed'
    def assets():
        for name,record in read(OUTPUT/'model_assets.json')['assets'].items():
            assert sha256(Path(record['path']))==record['sha256'],name
        return 'Original Multi-HMR, official MHR and neutral SMPL-X files match their pinned initial hashes'
    def exports():
        receipts=read(OUTPUT/'export_receipts.json');selected=read(OUTPUT/'selected.json')['trial']
        assert receipts['trial']==selected and set(receipts['samples'])==set(manifest['samples'])
        for name,item in receipts['samples'].items():
            path=SOURCE/name/'human/human_params_uranus.npz';assert sha256(path)==item['sha256']
            with np.load(path) as z,np.load(OUTPUT/'predictions'/selected/name/'human_params_mhr_v2.npz') as source:
                for key,shape in [('pose',(manifest['samples'][name]['frames'],136)),('shape',(45,)),('scales',(68,))]:
                    assert z[key].shape==shape and np.isfinite(z[key]).all() and np.array_equal(z[key],source[key])
                report=json.loads(z['report'].item())
                assert report['trial']==selected and report['evaluation_protocol']==PROTOCOL and report['conversion_revision']==2
                assert report['target_foreground_only_input'] and report['rgb_inference']['all_accepted_heads_in_target_mask']
                assert not report['gt_pose_identity_used_for_inference']
                assert report['prediction_source_sha256']==sha256(OUTPUT/'predictions'/selected/name/'human_params_mhr_v2.npz')
                assert str(z['coordinate_frame'])=='opencv_camera' and z['intrinsics'].shape==(manifest['samples'][name]['frames'],4)
        return 'All30 requested uranus outputs reloaded and exactly match the selected native MHR predictions'
    def reports():
        report=(ROOT/'_agents/report/261025_트랙2_MultiHMR_미세조정_진행.md').read_text()
        research=(ROOT/'_agents/research/261025/track2_multihmr_progress.md').read_text()
        assert report.index('init_v6 / init_v0')<report.index('## 현재 상태') and PROTOCOL in report and PROTOCOL in research
        values=dict(read(OUTPUT/'init_comparison.json')['models']);values.update(read(OUTPUT/'comparison.json')['models'])
        for name,value in values.items():
            row=f'| {name} | 10/10 | {value["CD_H_cm"]:.4f} | {value["ACC_H_cm_per_frame2"]:.4f} |'
            assert row in report.replace('**','')
            assert row in research.replace('**','')
        return 'Both persistent Markdown files; front table matches complete current comparison'
    def window():
        budget=read(OUTPUT/'budget.json');hours=budget.get('total_hours',12)
        assert abs(budget['deadline_unix']-budget['started_unix']-hours*3600)<1
        if hours!=12:
            extension=read(OUTPUT/'user_time_extension.json')
            assert hours==28 and extension['added_hours']==16
            assert budget['deadline_unix']==extension['extended_deadline_unix']
            assert abs(extension['original_deadline_unix']-budget['started_unix']-43200)<1
        assert time.time()>=budget['deadline_unix'],'User-authorized experiment window still in progress'
        return f'Original start preserved; user-authorized {hours}h window elapsed'
    def requested_performance():
        selected=read(OUTPUT/'selected.json')['trial']
        models=read(OUTPUT/'comparison.json')['models']
        baseline=read(OUTPUT/'init_comparison.json')['models']['init_v6']['CD_H_cm']
        actual=models[selected]['CD_H_cm']
        assert actual<baseline,f'User-requested init_v6 target unmet: selected CD-H {actual:.6f}cm >= init_v6 {baseline:.6f}cm'
        return 'Train-selected model beats init_v6 on the same complete Test10; no Test-based checkpoint selection'
    for name,run in [('fixed_split_and_GT',split),('initial_training_scope',training),('all_Train_SMPLX_labels',labels),
        ('both_init_baselines',baselines),('complete_target_Test_comparison',comparison),('Train_only_recipe_selection',selection),
        ('all30_native_MHR_exports',exports),('persistent_report_tables',reports),('persistent_assets_unchanged',assets),
        ('original12h_window',window),('user_requested_init_v6_CD_H_improvement',requested_performance)]:check(name,run)
    result=dict(checks=checks,all_required_checks_passed=all(c['passed'] for c in checks.values()),
        automated_evidence_only=True,requires_agent_review=True,checked_unix=time.time())
    save_json(OUTPUT/'completion_scope_audit.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()

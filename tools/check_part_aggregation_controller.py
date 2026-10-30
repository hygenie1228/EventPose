"""CPU checks for holdout selection readiness and report retry timing."""
import json,hashlib
import tempfile
from pathlib import Path
from run_part_aggregation import pending_selection_diagnostics,report_due,population_note,measured_report_table,pending_tests,pending_backfill,frozen_family_ready,training_budget_forecast,early_selection_status,SELECTION_CANDIDATES,selected_control_names,pending_selected_control

def main():
    root=Path(__file__).resolve().parents[1]
    temporary=root/'exp/261029/part_aware_dynamic_aggregation/tmp'
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as folder:
        out=Path(folder);trained_names={'v3_part_dynamic','v2_part_dynamic'}
        for name in trained_names|{'v3_part_no_motion_aux'}:
            path=out/'runs'/name/'best.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'pa_mpjpe_mm':24.5}))
        trained=lambda name:name in trained_names
        assert set(pending_selection_diagnostics(out,trained))==trained_names
        diagnostics=out/'diagnostics';diagnostics.mkdir()
        (diagnostics/'v2_part_dynamic.json').write_text('{}')
        assert pending_selection_diagnostics(out,trained)==['v3_part_dynamic']
        (diagnostics/'v3_part_dynamic.json').write_text('{}')
        assert pending_selection_diagnostics(out,trained)==[]
        (diagnostics/'v3_part_dynamic.json').unlink()
        assert pending_selection_diagnostics(out,trained,{'name':'v2_part_dynamic'})==[]
        assert not report_due(100,0,400,True)
        assert not report_due(399,0,400,True)
        assert report_due(400,0,400,True)
        assert not report_due(400,0,400,False)
        assert report_due(1800,0,400,False)
        note=population_note(dict(completed=['rgb_only','part_dynamic'],segments=80,GT_instances=24080,
            metrics={'rgb_only':{'samples':24056},'part_dynamic':{'samples':24056}},
            coverage={'rgb_only':{'fraction':.999},'part_dynamic':{'fraction':.998}}))
        assert '24056' in note and '24080' in note and '99.900%' in note and '99.800%' in note
        assert '전체 Test 완료 조건 없음' in population_note({})
        import sys
        sys.path.insert(0,str(root/'src'))
        from multihmr.part_aggregation_evaluation import report_table
        result=dict(conditions=['rgb_only','part_dynamic'],all_complete=True,
            metrics={'rgb_only':{'frame_aligned_pa_mpjpe_mm':24.95257},
                     'part_dynamic':{'frame_aligned_pa_mpjpe_mm':24.95635}})
        table=measured_report_table(result,report_table)
        assert '**24.95**' in table and '**24.96**' not in table
        assert '**' not in measured_report_table(dict(result,all_complete=False),report_table)
        stages=[('v4',['old']),('v5',['new']),('v3_ablations',['control']),('seed2',['unfrozen_choice'])]
        trained=lambda n:n=='old';attempted=lambda role:role=='test:old'
        approved=lambda stage:stage!='v3_ablations'
        assert pending_tests(stages,lambda n:n in ('old','new'),lambda n:False,attempted,approved)==['new']
        assert pending_backfill('v4',stages,trained,attempted,approved,lambda s:False,{})==[]
        assert pending_backfill('v4',stages,trained,attempted,approved,lambda s:True,{})==['new']
        assert pending_backfill('v4',stages,trained,lambda role:role=='train:new',approved,lambda s:True,{})==[]
        assert pending_backfill('v4',stages,trained,attempted,approved,lambda s:True,{'new':{}})==[]
        # An approved longer budget cannot dispatch after config/source drift,
        # or when its trigger refers to a different frozen plan or Test use.
        configs=out/'configs';configs.mkdir()
        source=out/'frozen_model.txt';source.write_text('fixed source')
        config=configs/'long_part.json';config.write_text('{"max_successful_steps":6000}')
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        planpath=out/'long_design_frozen.json'
        planpath.write_text(json.dumps(dict(deadline_unix=1793321712,
            source_sha256={str(source):digest(source)},config_sha256={'long_part':digest(config)})))
        triggerpath=out/'long_validation_trigger.json'
        trigger=dict(approved=True,Test_used=False,Test_files_read=False,frozen_plan_sha256=digest(planpath))
        triggerpath.write_text(json.dumps(trigger))
        assert frozen_family_ready(root,out,'long')
        config.write_text('{"max_successful_steps":9000}')
        assert not frozen_family_ready(root,out,'long')
        config.write_text('{"max_successful_steps":6000}')
        source.write_text('changed forward')
        assert not frozen_family_ready(root,out,'long')
        source.write_text('fixed source')
        triggerpath.write_text(json.dumps(dict(trigger,Test_used=True)))
        assert not frozen_family_ready(root,out,'long')
        triggerpath.write_text(json.dumps(dict(trigger,frozen_plan_sha256='wrong')))
        assert not frozen_family_ready(root,out,'long')
        stages=[('v6',['old']),('v6_long',['long_part']),('seed2',['unfrozen'])]
        assert pending_backfill('v6',stages,lambda n:n=='old',lambda role:False,
                                lambda stage:True,lambda stage:True,{})==['long_part']
        config=dict(name='v3_part_no_motion_aux',hours=3,architecture_version='v3',
                    max_successful_steps=3000,batch_size=8,hidden=128,split_sha256='split',init_sha256='init')
        references=[dict(run=dict(status='complete',successful_steps=3000,elapsed_seconds=elapsed),
                         config=dict(config,name='parent')) for elapsed in [4200,4900,5000,4800]]
        decision=training_budget_forecast(0,12000,config,references)
        assert decision['can_launch'] and decision['required_seconds']==10750
        assert not training_budget_forecast(0,10750,config,references)['can_launch']
        assert not training_budget_forecast(0,12000,config,references[:3])['can_launch']
        assert not training_budget_forecast(0,12000,dict(config,name='seed2_part_dynamic'),references)['can_launch']
        assert not training_budget_forecast(0,12000,dict(config,max_successful_steps=6000),references)['can_launch']
        bad=[dict(r) for r in references];bad[0]=dict(bad[0],run=dict(status='complete',successful_steps=2000,elapsed_seconds=3000))
        assert not training_budget_forecast(0,12000,config,bad)['can_launch']
        assert training_budget_forecast(0,15000,config)['can_launch']
        candidate=SELECTION_CANDIDATES[-1]
        trained=lambda n:n!=candidate
        status=early_selection_status(trained,lambda role:False,{},lambda n:True)
        assert not status['ready'] and status['waiting']==[candidate] and not status['new_budget_skips']
        status=early_selection_status(trained,lambda role:True,{},lambda n:False)
        assert not status['ready'] and status['waiting']==[candidate] and not status['new_budget_skips']
        status=early_selection_status(trained,lambda role:False,{},lambda n:False)
        assert status['ready'] and status['new_budget_skips']==[candidate]
        assert early_selection_status(trained,lambda role:False,{candidate:{}},lambda n:True)['ready']
        assert early_selection_status(lambda n:True,lambda role:False,{},lambda n:True)['ready']
        control_names=selected_control_names({'name':'v6_long_part_dynamic'})
        assert control_names==['v6_long_part_dynamic','v6_long_part_dynamic_missing_event','v6_long_part_dynamic_shuffle_event']
        incomplete=dict(conditions=control_names,all_complete=False,
            metrics={control_names[0]:{'frame_aligned_pa_mpjpe_mm':24.7}},completed=control_names[:1])
        text=measured_report_table(incomplete,report_table)
        assert control_names[1] in text and control_names[2] in text and '**' not in text
        selection={'name':'v6_long_part_dynamic'}
        assert pending_selected_control(None,lambda n:False,lambda role:False) is None
        assert pending_selected_control(selection,lambda n:False,lambda role:False)['control']=='intact'
        assert pending_selected_control(selection,lambda n:n==control_names[0],lambda role:False)['control']=='missing_event'
        assert pending_selected_control(selection,lambda n:n==control_names[0],lambda role:role=='test:'+control_names[1])['control']=='shuffle_event'
        assert pending_selected_control(selection,lambda n:False,lambda role:True) is None
    print('PASS: trained-candidate diagnosis barrier, frozen selection, completion-change retry backoff, periodic report')

if __name__=='__main__':main()

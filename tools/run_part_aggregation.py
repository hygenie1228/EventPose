"""Persistent 20-hour controlled experiment scheduler; no model-service calls.

Only this experiment's registered processes are managed. Other GPU allocations
are preserved. Training workers enforce the same deadline independently.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time,threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from goal_runtime_control import register_job,identity,stop_registered

GOAL=ROOT/'_agents/research/261029/part_aware_dynamic_aggregation_goal_20h.json'
PRIMARY=['rgb_only','fixed','global_dynamic','part_dynamic']
V2_PRIMARY=['v2_'+n for n in PRIMARY]
V2_ABLATIONS=['v2_global_dynamic_no_support','v2_part_dynamic_no_support','v2_global_dynamic_no_warmup','v2_part_dynamic_no_warmup']
V3_PRIMARY=['v3_'+n for n in PRIMARY]
V3_ABLATIONS=['v3_part_static','v3_global_no_motion_aux','v3_part_no_motion_aux','v3_fixed_recent']
V4_PRIMARY=['v4_fixed','v4_global_dynamic','v4_part_dynamic','v4_part_static']
V5_PRIMARY=['v5_fixed','v5_global_dynamic','v5_part_dynamic','v5_part_static']
V6_PRIMARY=['v6_fixed','v6_global_dynamic','v6_part_dynamic','v6_part_static']
V6_LONG=['v6_long_'+n for n in ['fixed','global_dynamic','part_dynamic','part_static']]
SELECTION_CANDIDATES=['v6_long_part_dynamic','v6_part_dynamic','v5_part_dynamic','v4_part_dynamic','v3_part_dynamic','v3_part_no_motion_aux','v2_part_dynamic','v2_part_dynamic_no_support','v2_part_dynamic_no_warmup']
GPUS={1:'GPU-d98bf2b1-073b-0738-bc33-0fc86356d368',2:'GPU-ba69cb68-67b5-09a9-24cd-205c450821d5',
      4:'GPU-fee21915-f6ec-b600-ed15-787fd4c98a03',5:'GPU-88cf24af-0bc7-ad30-f05a-9c02ccdb936b'}

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');temp.replace(p)

def pending_selection_diagnostics(out,trained,selection=None):
    if selection is not None:return []
    return [n for n in SELECTION_CANDIDATES if trained(n) and
            (out/'runs'/n/'best.json').exists() and
            read(out/'runs'/n/'best.json').get('pa_mpjpe_mm') is not None and
            not (out/'diagnostics'/(n+'.json')).exists()]

def report_due(now,last_report,retry_after,completion_changed):
    return now>=retry_after and (now-last_report>=1800 or completion_changed)

def population_note(result):
    completed=result.get('completed',[])
    samples={row.get('samples') for row in result.get('metrics',{}).values()}
    sample_text=str(next(iter(samples))) if len(samples)==1 else '미집계' if not samples else '불일치: '+str(sorted(samples))
    coverage=[]
    for name in completed:
        fraction=result.get('coverage',{}).get(name,{}).get('fraction')
        if fraction is not None:coverage.append(name+' '+f'{100*fraction:.3f}%')
    return ('비교 집합: '+(', '.join(completed) or '전체 Test 완료 조건 없음')+
            '. 전체 구간 '+str(result.get('segments','—'))+'개, GT '+str(result.get('GT_instances','—'))+
            '개, 다섯 지표 공통 평가 표본 '+sample_text+'개. 조건별 교집합 이전 검출 coverage: '+
            (', '.join(coverage) or '미집계')+'. 부분 Test 평균은 포함하지 않는다.')

def measured_report_table(result,table_builder):
    # Preserve frozen metric code; apply exact-minimum emphasis only in display.
    lines=table_builder(dict(result,all_complete=False)).splitlines()
    if not result.get('all_complete'):return '\n'.join(lines)
    keys=['first_frame_aligned_mpjpe_mm','first_frame_aligned_pa_mpjpe_mm',
          'frame_aligned_mpjpe_mm','frame_aligned_pa_mpjpe_mm','root_depth_error_mm']
    measured=result.get('metrics',{})
    for i,name in enumerate(result['conditions']):
        cells=lines[i+2].split('|')
        for j,key in enumerate(keys):
            value=measured.get(name,{}).get(key)
            available=[row[key] for row in measured.values() if row.get(key) is not None]
            if value is not None and available and value==min(available):
                cells[j+2]=' **'+cells[j+2].strip()+'** '
        lines[i+2]='|'.join(cells)
    return '\n'.join(lines)

def pending_tests(stages,trained,tested,attempted,approved):
    return [n for stage,names in stages if stage!='candidates' and approved(stage)
            for n in names if trained(n) and not attempted('test:'+n) and not tested(n)]

def pending_backfill(current_stage,stages,trained,attempted,approved,pilots_ready,budget_skips):
    eligible={'v2','v3','v4','v5','v6','v6_long','v3_ablations','v2_ablations'}
    future=False;names=[]
    for stage,conditions in stages:
        if stage==current_stage:future=True;continue
        if not future or stage not in eligible or not approved(stage) or not pilots_ready(stage):continue
        names.extend(n for n in conditions if n not in budget_skips and not trained(n) and not attempted('train:'+n))
    return names

def frozen_family_ready(root,out,label):
    import hashlib
    planpath=out/(label+'_design_frozen.json');triggerpath=out/(label+'_validation_trigger.json')
    if not planpath.exists() or not triggerpath.exists():return False
    plan=read(planpath);trigger=read(triggerpath)
    if not trigger.get('approved') or trigger.get('Test_used') is not False or trigger.get('Test_files_read') is not False:return False
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(planpath)!=trigger['frozen_plan_sha256'] or plan['deadline_unix']!=1793321712:return False
    return all((root/p).exists() and digest(root/p)==h for p,h in plan['source_sha256'].items()) and all(
        (out/'configs'/(n+'.json')).exists() and digest(out/'configs'/(n+'.json'))==h for n,h in plan['config_sha256'].items())

def training_budget_forecast(now,deadline,config,references=None):
    full=config['hours']*3600+3600
    forecast=dict(required_seconds=full,policy='Full configured training cap plus one-hour Test reserve')
    if now+full<deadline:
        return dict(forecast,can_launch=True)
    # Only the eight already-frozen V2/V3 controls may use a measured update
    # forecast. Accuracy values are never read. Preserve their model, 3000
    # update target, configured 3h cap, and trainer's independent hard deadline.
    if config['name'] in V2_ABLATIONS+V3_ABLATIONS and references and len(references)==4:
        durations=[]
        for reference in references:
            run=reference['run'];parent=reference['config']
            if run.get('status')!='complete' or run.get('successful_steps')!=3000:return dict(forecast,can_launch=False)
            for key in ['architecture_version','max_successful_steps','batch_size','hidden','split_sha256','init_sha256']:
                if config.get(key)!=parent.get(key):return dict(forecast,can_launch=False)
            duration=run.get('elapsed_seconds',0)
            if not 0<duration<=parent['hours']*3600:return dict(forecast,can_launch=False)
            durations.append(duration)
        required=max(durations)*1.25+300+3600+600
        if required<full:
            forecast=dict(required_seconds=required,reference_elapsed_seconds=durations,
                policy='Same-family completed 3000-update maximum x1.25 +5min training overhead +1h Test +10min audit',
                estimate_not_runtime_guarantee=True,model_config_and_update_target_unchanged=True)
    return dict(forecast,can_launch=now+forecast['required_seconds']<deadline)

def early_selection_status(trained,attempted,budget_skips,can_fit):
    waiting=[];new_skips=[]
    for name in SELECTION_CANDIDATES:
        if trained(name) or name in budget_skips:continue
        # An attempted live/failed/incomplete training is never silently
        # treated as an absent candidate, even when the budget is now short.
        if attempted('train:'+name) or can_fit(name):waiting.append(name)
        else:new_skips.append(name)
    return dict(ready=not waiting,waiting=waiting,new_budget_skips=new_skips)

def selected_control_names(selection):
    name=selection['name']
    return [name,name+'_missing_event',name+'_shuffle_event']

def pending_selected_control(selection,tested,attempted):
    if not selection:return None
    for name,control in zip(selected_control_names(selection),['intact','missing_event','shuffle_event']):
        if tested(name) or attempted('test:'+name):continue
        return dict(name=name,control=control,model=selection['name'])
    return None

def main():
    import pynvml as nv
    from multihmr.part_aggregation_evaluation import analyze,report_table
    g=read(GOAL);out=ROOT/g['output_dir'];deadline=g['deadline_unix']
    local=ROOT/'.experiment_storage/part_aggregation_261029';local.mkdir(parents=True,exist_ok=True)
    lock=(local/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    register_job(g,os.getpid(),'aggregation_scheduler')
    statefile=out/'pipeline_state.json'
    state=read(statefile) if statefile.exists() else dict(jobs=read(out/'pilot_launches.json'),history=[],failures=[],stage='primary')
    state.setdefault('attempts',{})
    children={};stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    state.setdefault('parent_goal_observed_status',None)
    state.setdefault('user_control_stop',None)
    if g.get('parent_goal_control'):
        def user_control_watch():
            from part_aggregation_control import read_parent_control
            missing=0
            while not stop[0] and time.time()<deadline:
                try:
                    result=read_parent_control(g['parent_goal_control'])
                    state['parent_goal_observed_status']=result['status']
                    missing=missing+1 if result['status']=='session_missing' else 0
                    if result['stop'] or missing>=3:
                        notice=dict(result,observed_unix=time.time(),reason=result['reason'] or 'Parent session absent in three successful control reads')
                        state['user_control_stop']=notice;stop[0]=True
                        write(local/'user_control_stop.json',notice)
                        return
                except Exception:
                    # A UI/network outage is not authorization to stop the 20h task.
                    pass
                time.sleep(30)
        threading.Thread(target=user_control_watch,daemon=True).start()
    def deadline_guard():
        # Independent of cache/report I/O in the scheduler's main thread.
        while time.time()<deadline-120:
            time.sleep(min(30,max(.1,deadline-120-time.time())))
        targets=list(state['jobs']);signalled=[]
        for job in targets:
            current=identity(job['pid'])
            if current!={k:job[k] for k in ('pgid','start_ticks','command')}:continue
            try:
                if job['role'].startswith('train:'):os.kill(job['pid'],signal.SIGTERM)
                else:os.killpg(job['pid'],signal.SIGTERM)
                signalled.append(job)
            except ProcessLookupError:pass
        time.sleep(60)
        for job in signalled:
            if identity(job['pid'])=={k:job[k] for k in ('pgid','start_ticks','command')}:
                try:os.killpg(job['pid'],signal.SIGKILL)
                except ProcessLookupError:pass
        write(local/'deadline_guard.json',dict(deadline_unix=deadline,checked_unix=time.time(),signalled=signalled))
    threading.Thread(target=deadline_guard,daemon=True).start()
    nv.nvmlInit();base=read(out/'configs/part_dynamic.json')
    manifest=read(Path(base['source_root'])/'manifest.json');tasks=read(Path(base['source_root'])/'test/tasks.json')
    stages=[('primary',PRIMARY),('v2',V2_PRIMARY),('v3',V3_PRIMARY),('v4',V4_PRIMARY),('v5',V5_PRIMARY),('v6',V6_PRIMARY),('v6_long',V6_LONG),('v3_ablations',V3_ABLATIONS),('v2_ablations',V2_ABLATIONS),('candidates',['part_low_lr','part_no_map_loss','part_wide','fixed_recent']),
            ('seed2',['seed2_'+n for n in PRIMARY]),('extended',['extended_'+n for n in PRIMARY]),
            ('seed3',['seed3_'+n for n in PRIMARY])]
    state.setdefault('budget_skips',{})
    def config(name):return read(out/'configs'/(name+'.json'))
    def budget_fits(name):
        cfg=config(name);parents=V3_PRIMARY if name in V3_ABLATIONS else V2_PRIMARY if name in V2_ABLATIONS else []
        references=[]
        for parent in parents:
            path=out/'runs'/parent/'complete.json'
            if not path.exists():references=[];break
            references.append(dict(run=read(path),config=config(parent)))
        forecast=training_budget_forecast(time.time(),deadline,cfg,references)
        if 'reference_elapsed_seconds' in forecast:
            forecast.update(recorded_unix=time.time(),reference_names=parents,deadline_unix=deadline)
            state.setdefault('measured_training_budget_forecasts',{})[name]=forecast
            write(out/'measured_training_budget_forecasts.json',state['measured_training_budget_forecasts'])
        return forecast['can_launch']
    # Freeze repeated seeds and validation-only candidate changes before Test.
    for seedlabel,seed in [('seed2',261030),('seed3',261031)]:
        for n in PRIMARY:
            p=out/'configs'/(seedlabel+'_'+n+'.json')
            if not p.exists():write(p,dict(config(n),name=seedlabel+'_'+n,seed=seed,comparison_names=[seedlabel+'_'+v for v in PRIMARY],reference_name=seedlabel+'_rgb_only'))
    for n,overrides in [('part_low_lr',dict(fusion_lr=1e-4)),('part_no_map_loss',dict(part_supervision_weight=0.)),
                        ('part_wide',dict(hidden=192,fusion_lr=1e-4)),('fixed_recent',dict(variant='fixed_recent'))]:
        p=out/'configs'/(n+'.json')
        if not p.exists():write(p,dict(base,name=n,**overrides))
    def trained(n):
        p=out/'runs'/n/'complete.json'
        return p.exists() and read(p)['status'] in ('complete','pilot_complete') and read(p)['successful_steps']>0
    def tested(n):return all((out/'results'/n/t['session']/'complete.json').exists() for t in tasks)
    def memory_ready(role):return all((out/'memory_cache'/(n+'.h5')).exists() for n in manifest[role+'_sessions'])
    def running(role):return any(j['role']==role for j in state['jobs'])
    def attempted(role):return bool(state['attempts'].get(role)) or any(j['role']==role for j in state['jobs']+state['history'])
    def pilot_ok(n):
        if not trained('pilot_'+n):return False
        r=read(out/'runs'/('pilot_'+n)/'complete.json')
        if r['successful_steps']!=8:return False
        diag=[json.loads(v) for v in (out/'runs'/('pilot_'+n)/'pilot_gradient_diagnostics.jsonl').read_text().splitlines()]
        variant=config(n)['variant']
        if variant!='rgb_only' and not any(d['event_encoder_gradient_l2']>0 for d in diag):return False
        if variant in ('part_aware','part_agnostic') and not any(d['time_score_gradient_l2']>0 for d in diag):return False
        if config(n).get('architecture_version') in ('calibrated_part_motion_horizon_v3','calibrated_part_frozen_rgb_v4','calibrated_part_dense_frozen_rgb_v5','calibrated_part_semantic_frozen_rgb_v6'):
            if not any(d.get('motion_head_gradient_l2',0)>0 for d in diag):return False
            if variant in ('part_aware','part_agnostic') and not any(d.get('event_use_head_gradient_l2',0)>0 for d in diag):return False
        if config(n).get('freeze_RGB_base'):
            receipt=out/'runs'/('pilot_'+n)/'frozen_RGB_checkpoint_verified.json'
            if not receipt.exists():
                import torch
                saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
                assert saved['model_delta'] and all(k.startswith('fusion.') for k in saved['model_delta'])
                assert not any(k.startswith('fusion.encoder.raw_encoder.') for k in saved['model_delta'])
                assert saved['config']['freeze_RGB_base'] and saved['test_used_for_selection'] is False
                write(receipt,dict(passed=True,checkpoint=r['checkpoint'],successful_steps=saved['successful_steps'],
                    adapter_only_parameter_tensors=len(saved['model_delta']),verified_unix=time.time()))
        return True
    def v2_approved():
        receipt=out/'v2_validation_trigger.json'
        if receipt.exists():return read(receipt)['approved']
        diag=out/'diagnostics/part_dynamic.json';audit=ROOT/'_agents/research/261029/부위별_집계_v2_독립_CPU감사.json'
        if not diag.exists() or not audit.exists():return False
        cpu=read(audit);plan=read(out/'v2_design_frozen.json')
        if cpu['status']!='passed' or cpu['sources']['src/multihmr/part_aggregation_v2.py']!=plan['v2_source_sha256']:return False
        d=read(diag);c=d['controls'];intact=c['intact']
        mass=sum(sum(row) for row in intact['mean_time_weights'])/len(intact['mean_time_weights'])
        delta={k:None if c[k]['pa_mpjpe_mm'] is None or intact['pa_mpjpe_mm'] is None else c[k]['pa_mpjpe_mm']-intact['pa_mpjpe_mm'] for k in ('zero_event','missing_event','shuffle_event')}
        ignored=mass<.01 and all(delta[k] is not None and abs(delta[k])<.05 for k in ('zero_event','shuffle_event'))
        write(receipt,dict(approved=True,validated_unix=time.time(),evidence='Completed Train-holdout event interventions plus independently proven v1 null-projection-bias invariant violation',Test_used=False,mean_event_probability=mass,control_PA_deltas_mm=delta,content_ignored_by_predeclared_threshold=ignored,cpu_audit_passed=True,source_diagnostic_sha256=__import__('hashlib').sha256(diag.read_bytes()).hexdigest(),design_frozen_unix=plan['design_frozen_unix']))
        return True
    def v3_approved():
        receipt=out/'v3_validation_trigger.json'
        if receipt.exists():return read(receipt)['approved']
        diag=out/'diagnostics/v2_part_dynamic.json';audit=ROOT/'_agents/research/261029/부위별_집계_v3_독립_CPU감사.json'
        planpath=out/'v3_design_frozen.json'
        if not diag.exists() or not audit.exists() or not planpath.exists():return False
        cpu=read(audit);plan=read(planpath)
        if cpu['status']!='passed' or cpu['sources']['src/multihmr/part_aggregation_v3.py']!=plan['source_sha256']['src/multihmr/part_aggregation_v3.py']:return False
        import numpy as np
        with np.load(out/'diagnostics/v2_part_dynamic_attention.npz') as z:
            w=z['time_weights'].astype(np.float64);mass=w.sum(-1);valid=mass>=.05
            q=np.divide(w,mass[...,None],out=np.zeros_like(w),where=mass[...,None]>0)
            spread=(q.max(-1)-q.min(-1))[valid]
        d=read(diag);strong=read(out/'runs/v2_rgb_only/best.json')['pa_mpjpe_mm']
        write(receipt,dict(approved=True,validated_unix=time.time(),Test_used=False,
            evidence='Frozen Train-holdout V2 restores event use but has no native RGB pose gain; explicit motion supervision probes temporal selection',
            V2_intact_PA_mm=d['controls']['intact']['pa_mpjpe_mm'],strong_RGB_holdout_PA_mm=strong,
            V2_mean_event_probability=float(mass.mean()),conditional_bin_spread_median=float(np.median(spread)) if len(spread) else None,
            conditional_bin_spread_90pct=float(np.quantile(spread,.9)) if len(spread) else None,
            design_frozen_unix=plan['design_frozen_unix'],cpu_audit_passed=True))
        return True
    def validation_choice():
        if state.get('validation_selection') is not None:return state['validation_selection']['name']
        assert not pending_selection_diagnostics(out,trained), 'Candidate holdout interventions must finish before selection'
        candidates=SELECTION_CANDIDATES
        choices=[(read(out/'runs'/n/'best.json'),n) for n in candidates if trained(n) and (out/'runs'/n/'best.json').exists()]
        choices=[(r,n) for r,n in choices if r.get('pa_mpjpe_mm') is not None]
        if not choices:choices=[(read(out/'runs/part_dynamic/best.json'),'part_dynamic')]
        rgb=[read(out/'runs'/n/'best.json')['pa_mpjpe_mm'] for n in ['v2_rgb_only','v3_rgb_only'] if (out/'runs'/n/'best.json').exists()]
        reference=min(rgb) if rgb else float('inf');qualifications={}
        for metrics,n in choices:
            p=out/'diagnostics'/(n+'.json');evidence=read(p)['controls'] if p.exists() else None
            usable=evidence and all(evidence[k].get('pa_mpjpe_mm') is not None for k in ['intact','zero_event','missing_event','shuffle_event'])
            deltas={k:evidence[k]['pa_mpjpe_mm']-evidence['intact']['pa_mpjpe_mm'] for k in ['zero_event','missing_event','shuffle_event']} if usable else {}
            eligible=metrics.get('coverage',0)>=.95 and metrics['pa_mpjpe_mm']<reference-.05 and len(deltas)==3 and all(v>.05 for v in deltas.values())
            qualifications[n]=dict(event_content_gain_confirmed_on_holdout=eligible,RGB_gain_mm=reference-metrics['pa_mpjpe_mm'],control_PA_deltas_mm=deltas)
        choices.sort(key=lambda x:(not qualifications[x[1]]['event_content_gain_confirmed_on_holdout'],x[0].get('coverage',0)<.95,x[0].get('pa_mpjpe_mm',float('inf'))))
        state.setdefault('validation_selection',dict(name=choices[0][1],metric='Train holdout PA; prefer >=0.05mm gains vs strongest RGB and zero/missing/shuffle controls, else explicitly unconfirmed fallback',candidate_metrics={n:r['pa_mpjpe_mm'] for r,n in choices},qualifications=qualifications,strongest_RGB_PA_mm=reference,Test_used=False,selected_unix=time.time()))
        return state['validation_selection']['name']

    def v4_approved():
        receipt=out/'v4_validation_trigger.json'
        if not receipt.exists():return False
        approval=read(receipt)
        if not approval.get('approved') or approval.get('Test_used') is not False:return False
        audit=read(ROOT/'_agents/research/261029/부위별_집계_v4_단일담당_CPU감사.json')
        plan=read(out/'v4_design_frozen.json')
        source='src/multihmr/part_aggregation_v4.py'
        return audit['status']=='passed' and audit['sources'][source]==plan['source_sha256'][source]

    def v5_approved():
        receipt=out/'v5_validation_trigger.json'
        if not receipt.exists():return False
        approval=read(receipt)
        if not approval.get('approved') or approval.get('Test_used') is not False:return False
        audit=read(ROOT/'_agents/research/261029/부위별_집계_v5_단일담당_CPU감사.json')
        plan=read(out/'v5_design_frozen.json');source='src/multihmr/part_aggregation_v5.py'
        return audit['status']=='passed' and audit['sources'][source]==plan['source_sha256'][source]

    def v6_approved():
        receipt=out/'v6_validation_trigger.json'
        if not receipt.exists():return False
        approval=read(receipt)
        if not approval.get('approved') or approval.get('Test_used') is not False:return False
        audit=read(ROOT/'_agents/research/261029/부위별_집계_v6_단일담당_CPU감사.json')
        plan=read(out/'v6_design_frozen.json');source='src/multihmr/part_aggregation_v6.py'
        return audit['status']=='passed' and audit['sources'][source]==plan['source_sha256'][source]

    def v6_long_approved():return v6_approved() and frozen_family_ready(ROOT,out,'v6_long')

    def stage_approved(stage):
        if stage.startswith('v2'):return v2_approved()
        if stage.startswith('v3'):return v3_approved()
        if stage=='v4':return v4_approved()
        if stage=='v5':return v5_approved()
        if stage=='v6':return v6_approved()
        if stage=='v6_long':return v6_long_approved()
        return True

    def stage_pilots_ready(stage):
        family=V6_LONG if stage=='v6_long' else V6_PRIMARY if stage=='v6' else V5_PRIMARY if stage=='v5' else V4_PRIMARY if stage=='v4' else V3_PRIMARY if stage.startswith('v3') else V2_PRIMARY
        return all(pilot_ok(n) for n in family)
    def launch(role,n,gpu,evalconfig=None):
        cfg=evalconfig or config(n);path=out/'configs'/(n+'.json')
        if role.startswith('diagnose:'):
            path=out/'configs'/(n+'_diagnosis.json');write(path,dict(cfg,task='diagnose'))
            cmd=['scripts/train_multihmr.py','--config',str(path)]
        elif role.startswith(('train:','pilot:')):
            # Experiment-specific SSD backing; never relocate live checkpoints.
            backing=local/'checkpoints'/n;backing.mkdir(parents=True,exist_ok=True)
            link=out/'checkpoints'/n
            if not link.exists():link.symlink_to(backing,target_is_directory=True)
            cmd=['scripts/train_multihmr.py','--config',str(path)]
        else:
            if evalconfig is not None:
                path=out/'configs'/(cfg['evaluation_name']+'_evaluation.json');write(path,cfg)
            cmd=['scripts/evaluate_multihmr.py','worker','--event-config',str(path),'--rank','0','--workers','1']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[gpu],EVENTPOSE_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (out/'logs'/(role.replace(':','_')+'.log')).open('a') as log:
            p=subprocess.Popen([str(ROOT/'.venv-humanpose/bin/python'),'-u']+cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rec=dict(register_job(g,p.pid,role),physical_gpu=gpu,gpu_uuid=GPUS[gpu],launched_unix=time.time())
        children[p.pid]=p;state['jobs'].append(rec);state['attempts'][role]=state['attempts'].get(role,0)+1
        print(json.dumps(dict(event='launch',**rec)),flush=True)
    report=ROOT/g['report'];report_body=report.read_text().split('## 실행 범위',1)[1]
    last_report=0;report_retry_after=0
    def refresh_report(final=False):
        nonlocal last_report,report_body
        report_body=report.read_text().split('## 실행 범위',1)[1].split('## 자동 실행 현황',1)[0]
        primary=analyze(base,PRIMARY)
        sections=['# 부위별 동적 이벤트 집계: 네트워크 설계와 실험','',measured_report_table(primary,report_table),'',population_note(primary),'']
        from datetime import datetime,timezone,timedelta
        kst=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
        sections+=['갱신: '+kst+'。 단계: `'+state['stage']+'`. V1 전체 Test 완료 '+str(len(primary['completed']))+'/4조건. V2·V3·V4·V5·V6 완료 범위는 아래 추가 비교에 표시한다.','',
                   '## 실행 범위'+report_body]
        sections+=['','## 자동 실행 현황','',f"실행 중 {len(state['jobs'])}개, 종료 기록 {len(state['history'])}개, 실패 기록 {len(state['failures'])}개. 고정 마감은 2026-10-30 09:55:12 KST다.",'',
                   '| 실험 | 성공 업데이트 | 출발 검증 PA | 최선 검증 PA | 검증 ΔPA | 검증 coverage | 전체 Test |','|---|---:|---:|---:|---:|---:|---|']
        for stage,names in stages:
            for n in names:
                complete=out/'runs'/n/'complete.json';bestpath=out/'runs'/n/'best.json';prog=out/'runs'/n/'progress.json'
                cr=read(complete) if complete.exists() else read(prog) if prog.exists() else {}
                bm=read(bestpath) if bestpath.exists() else {}
                pa=bm.get('pa_mpjpe_mm');cov=bm.get('coverage')
                baselinepath=out/'runs'/n/'baseline.json';baselinepa=read(baselinepath).get('pa_mpjpe_mm') if baselinepath.exists() else None
                delta=pa-baselinepa if pa is not None and baselinepa is not None else None
                sections.append('| '+n+' | '+str(cr.get('successful_steps','—'))+' | '+('—' if baselinepa is None else f'{baselinepa:.3f}')+' | '+('—' if pa is None else f'{pa:.3f}')+' | '+('—' if delta is None else f'{delta:+.3f}')+' | '+('—' if cov is None else f'{100*cov:.2f}%')+' | '+('완료' if tested(n) else '미완료')+' |')
        for stage,names in stages:
            if stage not in ('seed2','seed3','extended','v2','v2_ablations','v3','v3_ablations','v4','v5','v6','v6_long'):continue
            if any(tested(n) for n in names):
                summary_names=V2_PRIMARY+names if stage=='v2_ablations' else ['v2_rgb_only']+V3_PRIMARY+names if stage=='v3_ablations' else ['v2_rgb_only']+names if stage in ('v3','v4','v5','v6','v6_long') else names
                result=analyze(dict(base,reference_name='v2_rgb_only' if stage in ('v2_ablations','v3','v3_ablations','v4','v5','v6','v6_long') else stage+'_rgb_only'),summary_names)
                write(out/(stage+'_comparison.json'),result)
                sections+=['','### 추가 비교: '+stage,'',measured_report_table(result,report_table),'',population_note(result)]
        if state.get('validation_selection'):
            control_names=selected_control_names(state['validation_selection'])
            if any(tested(n) for n in control_names):
                result=analyze(dict(base,reference_name=control_names[0]),control_names)
                write(out/'selected_model_event_controls_comparison.json',result)
                sections+=['','## 선택 모델의 전체 Test 이벤트 개입','',
                    '같은 Train-selected checkpoint의 원본·이벤트 제거·셔플 조건이다. 선택은 Train 기준으로 이미 고정됐으며 아래 Test 점수로 다시 고르지 않는다. Shuffle은 각 추론 batch 안에서 이벤트를 한 칸 순환 이동한다. 개입 조건의 양수 오차 차이는 원본보다 오차가 커졌음을 뜻한다.',
                    '',measured_report_table(result,report_table),'',population_note(result)]
        # Keep the primary comparison at its documented canonical path.
        primary=analyze(base,PRIMARY)
        if state.get('validation_selection'):
            sections+=['','## 검증 기준 추가 설계 선택','',json.dumps(state['validation_selection'],ensure_ascii=False),
                       '', '추가 설계는 Train holdout만으로 선택했다. 주 비교의 네 조건 결과를 이 선택으로 대체하지 않는다.']
        for diag in sorted((out/'diagnostics').glob('*.json')) if (out/'diagnostics').exists() else []:
            d=read(diag)
            if not isinstance(d,dict) or not isinstance(d.get('controls'),dict) or not d.get('name'):
                continue
            sections+=['','## 이벤트 사용 개입 진단: '+d['name'],'',
                       '같은 frozen checkpoint의 Train 내부 검증 288프레임을 사용했다. Test 주표를 대체하지 않는다.','',
                       '| 입력 개입 | PA-MPJPE (mm) | root MPJPE (mm) | coverage |','|---|---:|---:|---:|']
            for control,v in d['controls'].items():
                sections.append('| '+control+' | '+('—' if v['pa_mpjpe_mm'] is None else f"{v['pa_mpjpe_mm']:.3f}")+' | '+('—' if v['root_mpjpe_mm'] is None else f"{v['root_mpjpe_mm']:.3f}")+' | '+f"{100*v['coverage']:.2f}%"+' |')
        if state['budget_skips']:sections+=['','## 고정 예산과 검증 우선순위에 따른 미실행 조건','',json.dumps(state['budget_skips'],ensure_ascii=False)]
        if final and state.get('user_control_stop'):
            sections+=['','## 사용자 제어로 중단','', '상위 Goal의 명시적인 pause·clear·교체를 확인하여 본인 등록 작업만 체크포인트를 보존하고 중단했다. 고정20시간 마감으로 완료한 것으로 표시하지 않는다.']
        elif final:sections+=['','## 고정 시간 종료','', '20시간 고정 마감에 본인 작업의 체크포인트를 보존하고 종료했다. 미완료 조건과 실패는 위 현황 및 실행 기록에 남긴다. 성능 향상 여부는 실측 Test 결과로만 판단한다.']
        report.write_text('\n'.join(sections)+'\n')
        write(out/'last_report_update.json',dict(updated_unix=time.time(),stage=state['stage'],completed=primary['completed'],final=final));last_report=time.time()
    while time.time()<deadline-90 and not stop[0]:
        # Lightweight process supervision is automatic; model-based reviews stay 30min apart.
        for j in list(state['jobs']):
            p=children.get(j['pid']);code=p.poll() if p else None
            current=identity(j['pid'])
            if current=={k:j[k] for k in ('pgid','start_ticks','command')}:continue
            state['jobs'].remove(j);state['history'].append(dict(j,exit_code=code,ended_unix=time.time()))
            if code not in (0,None):state['failures'].append(dict(role=j['role'],exit_code=code,log=str(out/'logs'/(j['role'].replace(':','_')+'.log'))))
        if all(pilot_ok(n) for n in PRIMARY):
            if not (out/'pilot_preflight_passed.json').exists():write(out/'pilot_preflight_passed.json',dict(all_four_passed=True,successful_steps_each=8,actual_event_gradient=True,actual_dynamic_weight_gradient=True,checked_unix=time.time()))
            if state.get('validation_selection') is None:
                early=early_selection_status(trained,attempted,state['budget_skips'],
                    lambda name:not (out/'configs'/(name+'.json')).exists() or budget_fits(name))
                for name in early['new_budget_skips']:
                    state['budget_skips'][name]=dict(reason='Cannot fit the frozen update/runtime forecast plus Test/audit reserve before original deadline; resolved before Train-only selection',skipped_unix=time.time())
                if early['new_budget_skips']:write(out/'budget_skips.json',state['budget_skips'])
                if early['ready'] and not pending_selection_diagnostics(out,trained):
                    validation_choice()
                    state['validation_selection']['frozen_before_waiting_for_other_Test_completion']=True
            current_stage=None
            for stage,names in stages:
                if stage.startswith('v2') and not v2_approved():continue
                if stage.startswith('v3') and not v3_approved():continue
                if stage=='v4' and not v4_approved():continue
                if stage=='v5' and not v5_approved():continue
                if stage=='v6' and not v6_approved():continue
                if stage=='v6_long' and not v6_long_approved():continue
                require_eval=stage!='candidates'
                if not all(n in state['budget_skips'] or (trained(n) and (tested(n) or not require_eval)) for n in names):current_stage=(stage,names);break
            pending=[]
            if current_stage is None or current_stage[0] in ('seed2','seed3','extended'):
                pending=pending_selection_diagnostics(out,trained,state.get('validation_selection'))
            if pending:
                state['stage']='awaiting_train_holdout_diagnostics'
                state['pending_selection_diagnostics']=pending
                for gpu in GPUS:
                    if any(j.get('physical_gpu')==gpu for j in state['jobs']):continue
                    h=nv.nvmlDeviceGetHandleByUUID(GPUS[gpu])
                    if nv.nvmlDeviceGetComputeRunningProcesses(h):continue
                    for n in pending:
                        if not attempted('diagnose:'+n):
                            launch('diagnose:'+n,n,gpu);break
            elif current_stage is None:
                state.pop('pending_selection_diagnostics',None)
                chosen=validation_choice()
                controls=['missing_event','shuffle_event']
                names=[chosen]+[chosen+'_'+v for v in controls]
                state['stage']='selected_model_event_controls'
                for gpu in GPUS:
                    if any(j.get('physical_gpu')==gpu for j in state['jobs']):continue
                    h=nv.nvmlDeviceGetHandleByUUID(GPUS[gpu])
                    if nv.nvmlDeviceGetComputeRunningProcesses(h):continue
                    diagnostic_role='diagnose:'+chosen
                    if trained(chosen) and not (out/'diagnostics'/(chosen+'.json')).exists() and not attempted(diagnostic_role):
                        launch(diagnostic_role,chosen,gpu);continue
                    for n in names:
                        role='test:'+n
                        if tested(n) or attempted(role):continue
                        ec=dict(config(chosen),evaluation_name=n,evaluation_control='intact' if n==chosen else n.removeprefix(chosen+'_'))
                        launch(role,chosen,gpu,ec);break
                if all(tested(n) for n in names):
                    state['stage']='analysis_and_final_verification'
                    write(out/'all_scheduled_training_test_complete.json',dict(completed_unix=time.time(),validation_selection=state['validation_selection'],final_review_required=True))
            else:
                state.pop('pending_selection_diagnostics',None)
                stage,names=current_stage;state['stage']=stage
                if v3_approved():
                    for n in ['part_low_lr','part_no_map_loss','part_wide','fixed_recent']:
                        if not trained(n) and not attempted('train:'+n):
                            state['budget_skips'].setdefault(n,dict(reason='V1 event use failed on Train holdout; prioritize motion-supervised V3 and matched controls',replaced_by='V3 validation-driven experiments',skipped_unix=time.time()))
                if stage in ('seed2','seed3'):
                    chosen=validation_choice();seed=261030 if stage=='seed2' else 261031
                    for n,original in zip(names,PRIMARY):
                        if attempted('train:'+n) or trained(n):continue
                        previous=config(n)
                        if previous.get('selected_origin')!=chosen:
                            archive=out/'configs_archive';archive.mkdir(exist_ok=True);write(archive/(n+'_original_plan.json'),previous)
                            write(out/'configs'/(n+'.json'),dict(config(chosen),name=n,variant=config(original)['variant'],seed=seed,selected_origin=chosen,comparison_names=names,reference_name=stage+'_rgb_only'))
                if stage=='extended':
                    chosen=validation_choice()
                    for n,original in zip(names,PRIMARY):
                        p=out/'configs'/(n+'.json')
                        if not p.exists():
                            c=dict(config(chosen),name=n,variant=config(original)['variant'],max_successful_steps=6000,
                                   comparison_names=names,reference_name='extended_rgb_only')
                            write(p,c)
                train_ready=memory_ready('training') and memory_ready('validation')
                test_ready=memory_ready('test')
                for gpu in GPUS:
                    if any(j.get('physical_gpu')==gpu for j in state['jobs']):continue
                    h=nv.nvmlDeviceGetHandleByUUID(GPUS[gpu])
                    if nv.nvmlDeviceGetComputeRunningProcesses(h):continue
                    choice=None
                    diagnostic_names=['part_dynamic']
                    if v2_approved():diagnostic_names+=['global_dynamic','v2_global_dynamic','v2_part_dynamic','v2_part_dynamic_no_support','v2_part_dynamic_no_warmup']
                    if v3_approved():diagnostic_names+=['v3_global_dynamic','v3_part_dynamic','v3_part_static','v3_part_no_motion_aux']
                    if v4_approved():diagnostic_names+=['v4_global_dynamic','v4_part_dynamic','v4_part_static']
                    if v5_approved():diagnostic_names+=['v5_global_dynamic','v5_part_dynamic','v5_part_static']
                    if v6_approved():diagnostic_names+=['v6_global_dynamic','v6_part_dynamic','v6_part_static']
                    if v6_long_approved():diagnostic_names+=['v6_long_global_dynamic','v6_long_part_dynamic','v6_long_part_static']
                    if state.get('validation_selection'):diagnostic_names.append(state['validation_selection']['name'])
                    for n in dict.fromkeys(diagnostic_names):
                        if trained(n) and not (out/'diagnostics'/(n+'.json')).exists() and not attempted('diagnose:'+n):
                            choice=('diagnose:'+n,n);break
                    if choice is None and test_ready:
                        selected=pending_selected_control(state.get('validation_selection'),tested,attempted)
                        if selected:
                            name=selected['name'];chosen=selected['model'];control=selected['control'];role='test:'+name
                            if control=='intact':launch(role,chosen,gpu)
                            else:launch(role,chosen,gpu,dict(config(chosen),evaluation_name=name,evaluation_control=control))
                            state.setdefault('early_selected_control_assignments',[]).append(dict(name=name,selected_unix=time.time(),while_stage=stage,Train_only_selection_frozen=True,priority='Selected frozen-model event verification before unassigned comparison Test'))
                            continue
                    if choice is None and test_ready:
                        queued=pending_tests(stages,trained,tested,attempted,stage_approved)
                        if queued:choice=('test:'+queued[0],queued[0])
                    if choice is None:
                        for family,approved in [(V6_LONG,v6_long_approved()),(V6_PRIMARY,v6_approved()),(V5_PRIMARY,v5_approved()),(V4_PRIMARY,v4_approved()),(V3_PRIMARY,v3_approved()),(V2_PRIMARY,v2_approved())]:
                            if not approved:continue
                            for n in family:
                                if not trained('pilot_'+n) and not attempted('pilot:'+n):choice=('pilot:'+n,'pilot_'+n);break
                            if choice:break
                    if choice is None and train_ready:
                        for n in names:
                            if n in state['budget_skips'] or trained(n) or attempted('train:'+n):continue
                            if stage.startswith('v2') and not all(pilot_ok(v) for v in V2_PRIMARY):continue
                            if stage.startswith('v3') and not all(pilot_ok(v) for v in V3_PRIMARY):continue
                            if stage=='v4' and not all(pilot_ok(v) for v in V4_PRIMARY):continue
                            if stage=='v5' and not all(pilot_ok(v) for v in V5_PRIMARY):continue
                            if stage=='v6' and not all(pilot_ok(v) for v in V6_PRIMARY):continue
                            if stage=='v6_long' and not all(pilot_ok(v) for v in V6_LONG):continue
                            # Reserve time for train-selected Test and a final audit.
                            if not budget_fits(n):
                                state['budget_skips'][n]=dict(reason='Cannot fit full authorized training budget plus Test reserve before fixed deadline',skipped_unix=time.time())
                                write(out/'budget_skips.json',state['budget_skips']);continue
                            choice=('train:'+n,n);break
                    if choice is None and train_ready:
                        for n in pending_backfill(stage,stages,trained,attempted,stage_approved,stage_pilots_ready,state['budget_skips']):
                            if not budget_fits(n):
                                state['budget_skips'][n]=dict(reason='Cannot fit training plus Test reserve before fixed deadline',skipped_unix=time.time())
                                write(out/'budget_skips.json',state['budget_skips']);continue
                            choice=('train:'+n,n)
                            state.setdefault('early_training_assignments',[]).append(dict(name=n,while_stage=stage,selected_unix=time.time(),reason='Idle GPU; no unassigned completed-model Test; approved frozen plan and pilots ready'))
                            break
                    if choice:launch(*choice,gpu)
        else:state['stage']='pilot_checks_and_memory_preparation'
        completed_test_conditions=[n for _,names in stages for n in names if tested(n)]
        if state.get('validation_selection'):
            completed_test_conditions += [n for n in selected_control_names(state['validation_selection'])[1:] if tested(n)]
        completion_changed=completed_test_conditions!=state.get('reported_completed_test_conditions',[])
        if report_due(time.time(),last_report,report_retry_after,completion_changed):
            try:
                refresh_report()
                state['reported_completed_test_conditions']=completed_test_conditions
                report_retry_after=0
            except Exception as e:
                state['failures'].append(dict(role='report',error=repr(e),time=time.time()));print('REPORT ERROR',repr(e),flush=True)
                report_retry_after=time.time()+300  # Completion changes also respect retry backoff.
        state.update(updated_unix=time.time(),deadline_unix=deadline)
        write(statefile,state)
        time.sleep(20)
    # Trainers stop saving well before the fixed cutoff; signal only verified own jobs.
    write(out/'deadline_shutdown_started.json',dict(started_unix=time.time(),deadline_unix=deadline,interrupted=stop[0]))
    shutdown=stop_registered(dict(g,managed_jobs_file=g['managed_jobs_file']))
    write(out/'deadline_shutdown_jobs.json',shutdown)
    try:refresh_report(final=True)
    except Exception as e:write(out/'final_report_error.json',dict(error=repr(e)))
    while time.time()<deadline and not stop[0]:time.sleep(min(20,max(.1,deadline-time.time())))
    write(out/'deadline_status.json',dict(status='user_control_stop' if state.get('user_control_stop') else 'fixed_20h_finished' if time.time()>=deadline else 'interrupted',finished_unix=time.time(),deadline_unix=deadline,final_agent_verification_required=True,user_control=state.get('user_control_stop')))

if __name__=='__main__':main()

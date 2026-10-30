"""One manual review per30min; this helper never launches or changes a worker."""
import datetime,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from goal_runtime_control import identity

def read(p):return json.loads(Path(p).read_text())

def write_current_note(result,folder):
    tz=datetime.timezone(datetime.timedelta(hours=9))
    fmt=lambda timestamp:datetime.datetime.fromtimestamp(timestamp,tz).strftime('%Y-%m-%d %H:%M:%S KST')
    primary=['rgb_only','fixed','global_dynamic','part_dynamic']
    groups={'V1':primary,'V2':['v2_'+n for n in primary],'V3':['v3_'+n for n in primary]}
    for version in ('v4','v5','v6','v6_long'):
        groups[version.upper()]=[version+'_'+n for n in ('fixed','global_dynamic','part_dynamic','part_static')]
    complete=[]
    for label,names in groups.items():
        count=sum(result['conditions'].get(n,{}).get('completed_Test_segments',0)==80 for n in names)
        complete.append(label+' '+str(count)+'/'+str(len(names)))
    manager='신원 일치' if result['manager_identity_valid'] else '신원 재확인 필요'
    lines=['# 부위별 동적 집계 현재 상태','',
        '기록: '+fmt(result['checked_unix'])+' · 단일 담당 `/root` · 단계 `'+result['stage']+'`.',
        '관리기 '+str(result['manager_pid'])+' ('+manager+'), 등록 활성 작업 '+str(len(result['active_jobs']))+'개.',
        '주 비교군의 전체 Test 완료: '+', '.join(complete)+' (후속 대조군은 별도).',
        '다음 직접 점검: '+fmt(result['next_direct_review_unix'])+' · 고정 마감: '+fmt(result['deadline_unix'])+'.',
        '성능 개선은 [실험보고서]('+str(ROOT/'_agents/report/261029_부위별_동적_이벤트_집계_설계와_실험.md')+')의 실측·통계로 판정한다.','',
        '| 현재 작업 | GPU | 성공 업데이트 | 전체 Test 구간 | 신원 |','|---|---:|---:|---:|---|']
    for job in result['active_jobs']:
        row=result['conditions'].get(job['role'].split(':',1)[-1],{})
        lines.append('| '+job['role']+' | '+str(job.get('gpu','—'))+' | '+str(row.get('successful_steps','—'))+
            ' | '+str(row.get('completed_Test_segments','—'))+'/80 | '+('일치' if job['identity_valid'] else '재확인 필요')+' |')
    lines+=['','미해결 실패 기록: '+str(len(result['unresolved_failures']))+'개. 이 문서는 위 기록 시각의 상태이며 최종 완료 감사가 아니다.']
    path=folder/'부위별_동적집계_현재상태.md';temporary=path.with_suffix('.tmp.md')
    temporary.write_text('\n'.join(lines)+'\n');temporary.replace(path)

def main():
    out=ROOT/'exp/261029/part_aware_dynamic_aggregation'
    settings=read(ROOT/'_agents/research/261029/part_aware_dynamic_aggregation_goal_20h.json')
    notice=ROOT/'.experiment_storage/part_aggregation_261029/user_control_stop.json'
    if notice.exists():
        print(json.dumps(dict(user_control_stop=read(notice)),ensure_ascii=False));return
    state=read(out/'pipeline_state.json');manager=read(out/'scheduler_launch.json')
    now=time.time();kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    result=dict(checked_unix=now,checked_kst=kst.isoformat(),reviewer='/root',stage=state['stage'],
        manager_pid=manager['pid'],manager_identity_valid=identity(manager['pid'])=={k:manager[k] for k in ('pgid','start_ticks','command')},
        parent_goal_observed_status=state.get('parent_goal_observed_status'),active_jobs=[],conditions={},
        validation_selection=state.get('validation_selection'),
        unresolved_failures=[f for f in state['failures'] if not f.get('resolved')],next_direct_review_unix=now+1800,
        deadline_unix=settings['deadline_unix'])
    for job in state['jobs']:
        result['active_jobs'].append(dict(role=job['role'],pid=job['pid'],gpu=job.get('physical_gpu'),
            identity_valid=identity(job['pid'])=={k:job[k] for k in ('pgid','start_ticks','command')}))
    source=Path(read(out/'configs/part_dynamic.json')['source_root']);tasks=read(source/'test/tasks.json')
    primary=['rgb_only','fixed','global_dynamic','part_dynamic']
    names=primary+['v2_'+n for n in primary]+['v3_'+n for n in primary]+[
        'v3_part_static','v3_global_no_motion_aux','v3_part_no_motion_aux','v3_fixed_recent',
        'v4_fixed','v4_global_dynamic','v4_part_dynamic','v4_part_static',
        'v5_fixed','v5_global_dynamic','v5_part_dynamic','v5_part_static',
        'v6_fixed','v6_global_dynamic','v6_part_dynamic','v6_part_static',
        'v2_global_dynamic_no_support','v2_part_dynamic_no_support',
        'v2_global_dynamic_no_warmup','v2_part_dynamic_no_warmup']
    names+=['v6_long_'+n for n in ['fixed','global_dynamic','part_dynamic','part_static']]
    names+=['pilot_v6_long_'+n for n in ['fixed','global_dynamic','part_dynamic','part_static']]
    names+=['pilot_v3_'+n for n in primary]
    names+=['pilot_v4_'+n for n in ['fixed','global_dynamic','part_dynamic','part_static']]
    names+=['pilot_v5_'+n for n in ['fixed','global_dynamic','part_dynamic','part_static']]
    names+=['pilot_v6_'+n for n in ['fixed','global_dynamic','part_dynamic','part_static']]
    if state.get('validation_selection'):
        chosen=state['validation_selection']['name']
        names += [chosen+'_missing_event',chosen+'_shuffle_event']
    for name in names:
        run=out/'runs'/name;row={}
        if (run/'progress.json').exists():
            p=read(run/'progress.json');row.update(successful_steps=p['successful_steps'],step=p['step'],last_progress_unix=p['updated_unix'],
                mean_event_mass=sum(map(sum,p['time_weights_mean']))/6,part_time_weight_std=p['part_time_weight_std'],
                uniform_warmup_active=p.get('uniform_warmup_active'),high_density_residual_energy=p.get('high_density_residual_energy'),
                low_density_residual_energy=p.get('low_density_residual_energy'))
        if (run/'complete.json').exists():
            p=read(run/'complete.json');row.update(terminal_status=p['status'],successful_steps=p['successful_steps'],step=p['step'],
                terminal_completed_unix=p['completed_unix'],target_steps_reached=p.get('matched_update_target_reached'))
        if (run/'baseline.json').exists():
            baseline=read(run/'baseline.json');row['baseline_PA_mm']=baseline['pa_mpjpe_mm']
            row['baseline_motion_path_MAE_px']=baseline.get('motion_path_mae_px')
        if (run/'best.json').exists():
            p=read(run/'best.json');row.update(best_holdout_PA_mm=p['pa_mpjpe_mm'],best_step=p['step'],holdout_coverage=p['coverage'],
                best_mean_event_mass=sum(map(sum,p['mean_time_weights']))/6,
                best_mean_part_time_weights=p['mean_time_weights'])
            row['best_motion_path_MAE_px']=p.get('motion_path_mae_px')
            row['best_motion_UV_endpoint_error_px']=p.get('motion_uv_endpoint_error_px')
            if row.get('baseline_PA_mm') is not None:row['best_PA_change_from_start_mm']=p['pa_mpjpe_mm']-row['baseline_PA_mm']
        row['completed_Test_segments']=sum((out/'results'/name/t['session']/'complete.json').exists() for t in tasks)
        if (out/'diagnostics'/(name+'.json')).exists():
            c=read(out/'diagnostics'/(name+'.json'))['controls']
            row['finished_holdout_controls_PA_mm']={k:v['pa_mpjpe_mm'] for k,v in c.items()}
        result['conditions'][name]=row
    result['main_comparison_completed']=read(out/'comparison.json')['completed']
    if state.get('validation_selection'):
        chosen=state['validation_selection']['name']
        result['selected_model_event_controls_completed_segments']={
            name:result['conditions'][name]['completed_Test_segments']
            for name in [chosen,chosen+'_missing_event',chosen+'_shuffle_event']}
    result['last_report_update']=read(out/'last_report_update.json')
    result['frozen_source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ['src/multihmr/part_aggregation_model.py','src/multihmr/part_aggregation_v2.py','src/multihmr/part_aggregation_v3.py','src/multihmr/part_aggregation_v4.py','src/multihmr/part_aggregation_v5.py','src/multihmr/part_aggregation_v6.py']}
    folder=ROOT/'_agents/research/261029';path=folder/('부위별_동적집계_정기검토_'+kst.strftime('%H%M')+'.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    owner=dict(owner=result['reviewer'],last_direct_review_unix=now,next_direct_review_unix=now+1800,deadline_unix=settings['deadline_unix'],
        last_snapshot=str(path),reason='User transferred full execution ownership to the single agent in the visible tab')
    (folder/'part_aware_monitor_owner.json').write_text(json.dumps(owner,ensure_ascii=False,indent=2)+'\n')
    write_current_note(result,folder)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

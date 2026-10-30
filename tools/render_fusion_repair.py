"""Korean report and figures, with numeric cells checked against source values."""
import json
from pathlib import Path
import numpy as np
from analyze_fusion_repair import NAMES, HEADLINE, PAIRS
from fusion_repair_scope import LABELS,EVAL_NAMES,PRETRAIN_NAMES,HIST_NAMES,FUSION_NAMES,PRIMARY_POPULATION,HISTORICAL_POPULATION


def label(name):
    return ('기존 ' + LABELS[name[4:]]) if name.startswith('old_') else LABELS[name]


def number(value, places=2, bold=False):
    return dict(value=value, places=places, bold=bold)


def cell(value):
    if not isinstance(value, dict):
        return str(value)
    v = value['value']; p = value['places']
    if v is None:
        text = '공통표본없음'
    elif isinstance(v, (list, tuple)):
        text = '[' + ', '.join(f'{a:.{p}f}' for a in v) + ']'
    else:
        text = f'{v:.{p}f}'
    return '**' + text + '**' if value['bold'] else text


def verify_tables(text, proofs):
    lines = text.splitlines()
    for key, proof in proofs.items():
        start = lines.index('<!-- FUSION_TABLE ' + key + ' -->')
        assert [v.strip() for v in lines[start+1].strip('|').split('|')] == proof['headers']
        for i, expected in enumerate(proof['rows']):
            actual = [v.strip() for v in lines[start+3+i].strip('|').split('|')]
            assert len(actual) == len(expected)
            for raw, source in zip(actual, expected):
                if not isinstance(source, dict):
                    assert raw == str(source)
                    continue
                assert raw.startswith('**') == source['bold']
                raw = raw.replace('**', '')
                value = source['value']
                if value is None:
                    assert raw == '공통표본없음'
                elif isinstance(value, (list, tuple)):
                    parsed = [float(x) for x in raw.strip('[]').split(',')]
                    assert len(parsed) == len(value)
                    assert np.allclose(parsed, value, rtol=0, atol=.5*10**(-source['places'])+1e-8)
                else:
                    assert np.isfinite(float(raw))
                    assert abs(float(raw)-value) <= .5*10**(-source['places'])+1e-8
    return dict(status='passed', tables=len(proofs), rows=sum(len(p['rows']) for p in proofs.values()))


def snapshot_stress_table(out):
    diagnostic=json.loads((out/'first_holdout_snapshot_Train_ablation_components.json').read_text())
    assert diagnostic['status']=='passed' and diagnostic['Test_used'] is False
    rows=[]
    for r in diagnostic['records']:
        errors=r['single_Train_frame_matched_errors_mm']
        assert set(errors['intact'])==set(errors['missing_aux'])=={'0'}
        rows.append([label(r['condition']),number(r['step'],0),
                     number(r['auxiliary_residual_relative_L2']*100),
                     number(r['linear_components']['zero_aux_residual_relative_L2']*100),
                     number(errors['intact']['0'][2]),number(errors['missing_aux']['0'][2]),
                     number(r['native_detections']['zero_aux'],0)])
    return ['조건','중간 update','정상 잔차 (%)','0입력 잔차 (%)','정상 PA (mm)','분기 off PA (mm)','0입력 검출 수'],rows


def figures(out, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import h5py
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    dest = out / 'visualizations'; dest.mkdir(exist_ok=True)
    names=EVAL_NAMES
    pop = result['populations'][PRIMARY_POPULATION]; values = np.array([[pop[n][k] for k in HEADLINE] for n in names])
    assert np.isfinite(values.astype(float)).all(), 'Common Test population required for final comparison figure'
    titles = ['first frame-aligned\nMPJPE', 'first frame-aligned\nPA-MPJPE',
              'frame-aligned\nMPJPE', 'frame-aligned\nPA-MPJPE', 'Root position\nerror (3D)']
    fig, axes = plt.subplots(1, 5, figsize=(21, 8), sharey=True)
    for j, ax in enumerate(axes):
        ax.barh(range(len(names)), values[:, j], color=['#a7b0bc','#c0c7d0','#2563a6','#3e86c6','#7560a7','#9279bc','#cc7637','#d49b45','#b36a81'])
        ax.set_yticks(range(len(names)), [label(n) for n in names], fontsize=10,fontproperties=font)
        ax.set_xlim(0, float(values[:, j].max())*1.35); ax.set_title(titles[j], fontsize=11)
        ax.set_xlabel('mm'); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        for i, v in enumerate(values[:, j]):
            ax.text(v+float(values[:, j].max())*.025, i, f'{v:.2f}', va='center',
                    weight='bold' if v == values[:, j].min() else 'normal')
    axes[0].invert_yaxis()
    fig.suptitle('원본 · 짧은 RGB 학습 · 긴 센서융합 학습 · 전체 Test', fontproperties=font, fontsize=20)
    fig.text(.5, .02, f"동일 공통 GT {pop[names[0]]['samples']:,}개 · 학습량 차이 포함 · 낮을수록 좋음 · CI/Holm은 본문 참조",
             fontproperties=font, ha='center', fontsize=10)
    fig.subplots_adjust(left=.17, right=.99, top=.76, bottom=.14, wspace=.3)
    chart = dest / '수정융합_전체Test_다섯지표.png'; fig.savefig(chart, dpi=180); fig.savefig(chart.with_suffix('.svg')); plt.close(fig)
    tasks = json.loads((out / 'test/tasks.json').read_text()); chosen = None
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt = z['joints_cam_m'][0]; valid = z['valid'][0]
            assert z['source_rgb_positions'][0] == 0
        predictions = {}; common = valid.copy()
        for n in names:
            with np.load(out / 'results' / n / task['session'] / 'result.npz') as z:
                predictions[n] = z['matched_joints_cam_m'][0]
                common &= z['matched_pred_indices'][0] >= 0
        if common.any():
            chosen = (task, gt, predictions, int(np.flatnonzero(common)[0])); break
    assert chosen is not None
    task, gt, predictions, person = chosen; absolute = task['start_frame']
    with h5py.File(out / 'cache' / (task['parent_session']+'.h5'), 'r', locking=False) as f:
        rows = f['frames'][:]; i = int(np.searchsorted(rows, absolute)); assert rows[i] == absolute
        background = f['rgb3'][i].transpose(1,2,0); K = f['rgb3_K_primary'][:]
        native_height,native_width=map(int,f['rgb3_native_shape'][:])
    scale=min(background.shape[0]/native_height,background.shape[1]/native_width)
    display_height,display_width=round(native_height*scale),round(native_width*scale)
    left=(background.shape[1]-display_width)//2;top=(background.shape[0]-display_height)//2
    bones = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),
             (9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21)]
    def draw(ax, joints, color):
        points = joints[:22] @ K.T; uv = points[:,:2]/points[:,2:]
        for a,b in bones:
            if points[[a,b],2].min() > 0:
                ax.plot(uv[[a,b],0], uv[[a,b],1], color=color, linewidth=1.5)
    fig, axes = plt.subplots(3,3,figsize=(18,11))
    for ax,n in zip(axes.flat,names):
        ax.imshow(background); draw(ax,gt[person],'#28d464'); draw(ax,predictions[n][person],'#ff4b4b')
        ax.set_xlim(left,left+display_width);ax.set_ylim(top+display_height,top)
        ax.set_title(label(n), fontsize=10,fontproperties=font); ax.axis('off')
    for ax in list(axes.flat)[len(names):]:ax.axis('off')
    fig.text(.5,.015,'초록: GT · 빨강: 실제 예측 · 동일 Test 시각·동일 인물 · RGB3 배경은 보고용 · 표시에서 공통 letterbox 여백만 제거',
             ha='center',fontproperties=font,fontsize=12)
    fig.tight_layout(rect=(0,.04,1,1)); image = dest/'수정융합_실제Test_예측.png';fig.savefig(image,dpi=160);plt.close(fig)
    proof = dict(task=task['session'], absolute_frame=absolute, relative_frame=0, person=person,
                 selection='First frozen task with exact frame0 common detection; never selected by error',
                 RGB3_background_display_only=True, event_only_RGB_inference=False,
                 display_only_letterbox_bounds=[left,top,left+display_width,top+display_height],
                 model_input_and_projection_coordinates_unchanged=True,
                 chart_values_mm=values.tolist(), chart_metrics=HEADLINE, chart_conditions=names,
                 chart=str(chart), image=str(image))
    (out/'visualization_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return chart,image


def learning_curve_figure(out,result,training,source_training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    families=[('rgb3',['rgb3_short90','rgb3_event4','rgb3_event3']),('rgb4',['rgb4','event3_rgb4','rgb4_event4'])]
    fig,axes=plt.subplots(2,5,figsize=(22,9));records=[]
    titles=['first frame-aligned MPJPE','first frame-aligned PA-MPJPE','frame-aligned MPJPE','frame-aligned PA-MPJPE','3D Root position error']
    for i,(sensor,methods) in enumerate(families):
        pop=result['populations']['learning_family_'+sensor]
        for method,color in zip(methods,['#2166ac','#d45f35','#39935d']):
            base_hours=source_training[method]['training_elapsed_seconds']/3600
            hours=[0.,base_hours,base_hours+training[method]['best_checkpoint_training_elapsed_seconds']/3600]
            keys=['pre_'+sensor,'old_'+sensor,method]
            values=[[pop[n][k] for k in HEADLINE] for n in keys]
            assert np.isfinite(values).all()
            records.append(dict(method=method,population='learning_family_'+sensor,samples=pop[method]['samples'],
                                hours=hours,stages=keys,metrics=HEADLINE,values_mm=values))
            for j in range(5):
                ax=axes[i,j];ax.plot(hours,[v[j] for v in values],'-o',color=color,label=label(method),lw=1.8,markersize=5)
                ax.set_title(titles[j],fontsize=10);ax.set_xlabel('checkpoint cumulative ETRI trial time (h)',fontsize=8)
                ax.set_ylabel('mm');ax.grid(alpha=.2)
        axes[i,0].text(.02,.02,f"{sensor} 기준 · 공통 GT {pop[methods[0]]['samples']:,}",transform=axes[i,0].transAxes,
                       va='bottom',fontproperties=font,fontsize=9)
        axes[i,0].legend(prop=font,fontsize=8,loc='best')
    fig.suptitle('학습량별 실측 경향: 원본 → 출처 checkpoint → 이번 선택 checkpoint',fontproperties=font,fontsize=19)
    fig.text(.5,.01,'장치별 공통 GT · 점은 실제 평가값 · 선은 점 연결 · 최종 융합점은 센서와 학습량 효과를 함께 포함 · 단조 개선을 가정하지 않음',fontproperties=font,ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.94));path=out/'visualizations/학습량별_실측_경향.png';fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.svg'));plt.close(fig)
    proof=dict(path=str(path),curves=records,interpretation='Actual selected-checkpoint cumulative training time; not the allocated search budget; no monotonic improvement assumed')
    (out/'learning_curve_provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    return path


def render(goal, result, training, source_training, runtime, cache_cost, ablations,coverage):
    from datetime import datetime,timezone,timedelta
    import time
    out=Path(goal['output_root']); proofs={}
    lines=['# 센서융합 열화 원인 분석과 재실험','',
           f"고정15시간 범위: 2026-10-27 13:15:31~2026-10-28 04:15:31 KST. 집계 시각 {datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')}, "
           f"현재 총 경과 {(time.time()-goal['started_unix'])/3600:.3f}시간. 허용 물리GPU는1·2·4·5이며 GPU3은 사용하지 않았다.",'',
           '기존 event 주센서와 저해상도 RGB 보조 CNN의 비대칭을 수정했다. RGB가 있는 네 융합 조건은 해당 RGB의 원래 검출·복원 경로를 유지한다. '
           '최신 사용자 지시로 RGB-only는 더 짧게, 융합은3시간 학습했다. 결과에는 학습량 차이가 포함되므로 센서만의 인과 효과를 주장하지 않는다. 원본 RGB 두 조건은 ETRI 미세조정0 update다.','',
           '모델 구조·입출력 도식: [전체 구조와 학습·평가 파이프라인](261027_센서융합_모델구조와_입출력_파이프라인.md).','',
           f"Test {result['segments']}구간·{result['frames']:,}프레임·유효 GT {result['gt_instances']:,}개. 주표9조건 공통 GT {result['populations'][PRIMARY_POPULATION]['rgb3_short90']['samples']:,}개.",'']
    def table(key, headers, rows):
        proofs[key]=dict(headers=headers,rows=rows)
        lines.extend(['<!-- FUSION_TABLE '+key+' -->','| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|'])
        lines.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in rows);lines.append('')
    metric_headers=['조건','first frame-aligned MPJPE (mm)','first frame-aligned PA-MPJPE (mm)',
                    'frame-aligned MPJPE (mm)','frame-aligned PA-MPJPE (mm)','Root position error (mm)']
    def headline(group, names):
        pop=result['populations'][group]
        return [[label(n)]+[number(pop[n][k],bold=pop[n][k] is not None and pop[n][k]==min(pop[m][k] for m in names if pop[m][k] is not None)) for k in HEADLINE] for n in names]
    introduction=lines[2:]
    lines=lines[:2]
    lines+=['## 미세조정 전·짧은 RGB·긴 융합의 전체 Test 비교',''];table('primary',metric_headers,headline(PRIMARY_POPULATION,EVAL_NAMES))
    lines+=introduction
    lines+=['첫 정렬은 정확 frame0의 pelvis 평행이동 또는 proper Sim3를 구간 전체에 고정하고, frame-aligned는 매 프레임 다시 계산한다. '
            'Root position error는 정렬 전 공통 device3 RGB 카메라의 full3D pelvis 오차다. RPE-Z와 구분한다. '
            '첫 프레임의 PA 변환은 이후 프레임의 최적 변환이 아니므로 첫 PA 오차가 첫 pelvis 정렬 오차보다 클 수 있다.','',
            '## 과거 방법과 수정 방법의 동일 표본 비교','',
            '기존 결과는 과거 실험의 실제 예측이며 새 방법 결과가 아니다. 이 표는 기존/새14실행 공통 GT이므로 위9조건 표와 모집단이 다르다. '
            '원본·출처 기존학습·이번 추가학습의 누적 예산 차이를 함께 해석해야 한다.','']
    table('historical',metric_headers,headline(HISTORICAL_POPULATION,['old_'+n for n in HIST_NAMES]+EVAL_NAMES))
    lines+=['## 검출과 비정렬 보조 지표','']
    table('auxiliary',['조건','전체 GT 검출률 (%)','절대 MPJPE (mm)','RPE-Z (mm)','pelvis 속도 오차 (mm/s)','pelvis 상대 body22 가속도 오차 (mm/s²)'],
          [[label(n),number(result['coverage'][n]['fraction']*100)]+[number(result['populations'][PRIMARY_POPULATION][n][k]) for k in
          ['absolute_mpjpe_mm','root_depth_error_mm','pelvis_velocity_error_mmps','root_acceleration_error_mmps2']] for n in EVAL_NAMES])
    lines+=['속도는 비정렬 pelvis의3D 속도 오차다. 가속도는 매 프레임 pelvis를 뺀 body22 관절의 상대 가속도 오차이며 절대 root 궤적의 가속도가 아니다. '
            '보정 timestamp의 실제 시각 차이를 사용하고 누락 프레임·사람 대응 실패·구간 경계를 연결하지 않는다.','',
            '시간적 안정성은 일관되게 좋아지지 않았다. RGB+event 네 조건 모두 해당 RGB 단독보다 가속도 오차가 증가했고, '
            '속도 오차도 일부 조건에서 증가했다. 주요5지표의 pose 오차 개선과 시간적 안정성을 구분해야 한다.','']
    table('own',['조건','자체 frame0 적격 표본','자체 PA-MPJPE (mm)','자체 Root error (mm)'],
          [[label(n),number(result['populations']['own_'+n][n]['samples'],0),number(result['populations']['own_'+n][n][HEADLINE[3]]),
            number(result['populations']['own_'+n][n]['pelvis_error_mm'])] for n in EVAL_NAMES])
    lines+=['검출 실패는0오차로 채우지 않는다. 공통 검출 오차와 전체GT 검출률을 함께 보고한다. 자체 모집단 표는 공통표본 감소에 대한 보조 설명이며 조건 순위용이 아니다.','',
            '## 학습 예산과 checkpoint 출처','']
    original=json.loads((out/'original_pretrained_manifest.json').read_text())
    rgb4_started=json.loads((out/'runs/rgb4/started.json').read_text())
    pipeline=json.loads((out/'pipeline_state.json').read_text())
    rgb4_launch=max(j['launched_unix'] for j in pipeline['history']+pipeline['jobs'] if j['role']=='train:rgb4')
    rgb4_remaining_minutes=(rgb4_started['finish_unix']-rgb4_launch)/60
    lines += [f"원본 RGB3/RGB4는 공개 Multi-HMR `{original['path']}` (SHA256 `{original['sha256']}`)에서 ETRI 추가학습0 update로 평가했다. "
              '원본 두 조건의 best.pt는 공개 가중치 출처를 참조하는 작은 manifest이며 미세조정 가중치가 아니다. '
              f"선택 checkpoint의 step·실제 SHA와 실행 증거는 [최종 감사 원자료]({out/'final_artifact_audit.json'})에 보존했다.",'',
              'RGB-only는 동일3시간 optimizer 스케줄을90분 wall budget에서 조기 종료했다. 표의 새 경과 시간에는 종료 후 최종 검증·저장이 포함된다. '
              'RGB4는 첫 launch가 최소3시간 설정 검사에서 CUDA 학습 전에 실패했고, 그 대기·수정 시간도 원래90분 budget에 포함했다. '
              f'실패 후 정상 재실행부터 예정 종료까지 남은 시간은 {rgb4_remaining_minutes:.2f}분이었다. '
              '이 시간에도 모델 로딩·정기 검증이 포함되므로 성공 update·본 표본 수로 실제 학습량을 함께 확인해야 한다.','']
    table('training',['조건','새 경과 (h)','성공 update','본 표본','고유 프레임','완료 subset epoch','subset 고유coverage (%)','전체Train 고유coverage (%)','출처 본 표본','출처 학습 (h)'],
          [[label(n),number(training[n]['elapsed_seconds']/3600,3),number(training[n]['successful_steps'],0),number(training[n]['samples_seen'],0),
            number(training[n]['unique_frames_seen'],0),number(training[n]['completed_subset_epochs'],0),
            number(100*training[n]['unique_frames_seen']/6501),number(100*training[n]['unique_frames_seen']/coverage['full_train_valid_frames']),number(source_training[n]['samples_seen'],0),
            number(source_training[n]['training_elapsed_seconds']/3600,3)] for n in NAMES])
    for n in NAMES:
        config=json.loads((out/'configs'/(n+'.json')).read_text())
        lines += [f"- {label(n)}: 최종 `{out/'checkpoints'/n/'best.pt'}`; 초기화 `{config['initialization_checkpoint']}` (SHA256 `{config['initialization_sha256']}`)."]
    recovery=json.loads((out/'short_rgb_recovery.json').read_text())
    lines+=['','### 본 비교에서 제외한 사용자 중단 실행','',
            '기존 RGB3 실행은 사용자 단축 지시 때 이미90분을 넘었고 그룹 종료 신호로 DataLoader도 종료돼 최종 추가 저장 전 예외가 발생했다. '
            '주기 checkpoint를 복구해 추가 optimizer update 없이 holdout을 검증했다. 이 실행을90분 결과로 바꾸거나 본 비교에 혼합하지 않았고 '
            '본 비교 RGB3는 별도 rgb3_short90 trial이다. 이 실패·복구 시간도 전체15시간에 포함된다.','']
    table('excluded_rgb3_reference',['실행','실제 종료 관측 (h)','보존 checkpoint 경과 (h)','보존 update','미보존 update 하한','복구 holdout PA (mm)'],
          [['RGB3 사용자 중단 참고',number(recovery['actual_elapsed_seconds']/3600,3),
            number(recovery['checkpoint_training_elapsed_seconds']/3600,3),number(recovery['stored_step'],0),
            number(recovery['unrecovered_update_lower_bound'],0),number(recovery['metrics']['pa_mpjpe_mm'])]])
    lines+=['',f"전체 body Train 유효 프레임은 {coverage['full_train_valid_frames']:,}개, 내부 holdout 피험자를 제외한 학습 프레임은 {coverage['training_valid_frames']:,}개, "
            f"모든 센서가 가용한 학습 후보는 {coverage['training_common_available_frames']:,}개다. 고정6501프레임 subset은 전체Train의 {6501/coverage['full_train_valid_frames']*100:.3f}%, "
            f"holdout 제외 학습 프레임의 {6501/coverage['training_valid_frames']*100:.3f}%다.",'',
            '학습한7조건은 같은6501프레임 균형 subset, batch16, 같은 증강을 사용했다. 사용자 지시로 RGB-only는 단축하고 융합은3시간을 사용했다. 각 subset epoch의 drop_last는5프레임이다. '
            '전체Train1epoch 학습으로 표시하지 않는다. 기존 출처 학습과 새 추가학습의 본 표본 수를 합쳐도 고유 프레임 수가 늘었다는 뜻은 아니다. '
            'checkpoint는 Train 내부 피험자분리 holdout288프레임으로 선택했고 Test는 선택에 사용하지 않았다. '
            'RGB 대조군과 해당 fusion은 같은 RGB 출처를 사용했다. event-only는 출처 event 모델의 base/projection만 사용하고 과거 fusion은 제외했다.','',
            '## 통계와 동작속도 분석','',
            '효과는 수정/비교 조건 minus 기준이며 음수는 개선이다. 같은 쌍의 공통 GT에서 피험자/세션 cluster bootstrap10000회·95%CI와 sign-flip 검정을 계산했다. '
            '사전에 고정한13쌍의 PA와13쌍의3D root를 각각 Holm 보정했다. 군집 내 프레임을 독립 표본으로 취급하지 않았다. 학습량이 다른 쌍의 차이는 학습 시간 효과도 포함한다.','']
    table('statistics',['비교·지표','공통 표본','군집','효과 (mm)','상대효과 (%)','95%CI (mm)','Holm p'],
          [[k,number(t['samples'],0),number(t['clusters'],0),number(t['effect']),number(t.get('relative_percent')),number(t['ci95']),number(t['holm_p_value'],5)] for k,t in result['tests'].items()])
    motion=[]
    for pair,bins in result['motion_strata'].items():
        a,b=pair.split('__')
        for category,values in bins.items():
            x,y=values[a],values[b];assert x['count']==y['count']
            motion.append([pair,category,number(x['count'],0),number(x['sum_mm']/x['count'] if x['count'] else None),
                           number(y['sum_mm']/y['count'] if y['count'] else None)])
    table('motion',['비교','GT 동작 구간','공통 표본','기준 PA (mm)','비교 PA (mm)'],motion)
    lines+=['GT body22 속도 .05m/s 이하=still, .5m/s 이상=fast, 중간=slow이며 시각 간격과 연속 프레임을 확인했다. GT jitter가 구간 분류에 영향을 줄 수 있다.','',
            '## event 기여도: 내부 holdout ablation','',
            '아래는 Test가 아닌 Train holdout 진단이다. 입력 제거·0 입력·극성 교환만 바꾸고 동일 checkpoint를 사용했다. 각 쌍의 공통 검출 GT만 오차를 비교하고 전체 검출률을 함께 표시한다.','']
    ablation_rows=[]
    for n,a in ablations.items():
        for mode,p in a['pairs'].items():
            delta=p['ablated_minus_intact_mm']
            ablation_rows.append([label(n),mode,number(a['rows']['intact']['coverage']*100),number(a['rows'][mode]['coverage']*100),
                                  number(a['rows'][mode].get('resource_guarded_gt_instances',0),0),
                                  number(p['common_instances'],0),number(delta[2] if delta else None),number(delta[3] if delta else None)])
    table('ablation',['조건','제거 방식','정상 성공률 (%)','변경 성공률 (%)','자원실패 GT','공통 표본','변경-정상 PA (mm)','변경-정상 Root (mm)'],ablation_rows)
    lines+=['event-only의 인위적0 보조입력 진단에서 native 검출 후보가 폭증해684.89GiB 할당 시도로 종료했다. '
            '실패 로그를 보존했고 진단에만 프레임당 후보16개 초과 시 해당 프레임 전체를 자원실패로 기록하는 보호를 추가했다. '
            '상위 후보 일부로 대체하지 않았고 정상 Test의 모델·checkpoint·검출 기준은 변경하지 않았다. '
            '성공률은 전체 GT 대비 매칭 수로 자원실패도 포함하며, 보호된 프레임의 native 검출률이나 pose 오차를 평가했다고 주장하지 않는다. '
            '공통 표본이 없는 변경의 오차는 미평가다.','']
    lines+=['event 고유 정보의 기여는 강하게 입증되지 않았다. 학습된 RGB+event 모델에서 보조 분기를 끈 missing_aux의 '
            'PA 변화는 표 반올림 기준0.00~0.02mm이고 극성을 바꿔도 변화가 작거나 방향이 일정하지 않았다. '
            '따라서 RGB 단독 대비 개선을 event 정보 자체의 효과로 단정할 수 없다. RGB 경로 보존과 더 긴 학습의 효과가 함께 포함된다. '
            '동일3시간 RGB 대조군과 반복 seed를 추가해야 센서 효과를 분리할 수 있으며, 이번 사용자 지정 단축 RGB 비교는 이를 대체하지 않는다.','']
    lines+=['## 중간 checkpoint의0입력 스트레스 검사','',
            '아래는 첫 정기 검증 checkpoint를 불변 snapshot으로 보존해 고정 Train 한 프레임에서 CPU 추론한 진단이다. '
            '위 최종 전체 holdout ablation 및 Test와 구분한다. 분기off는 같은 모델의 잔차를 끄는 것이며 별도로 학습한 RGB 단독 대조군이 아니다. '
            '잔차는 주 특징 대비 L2 비이며,0입력은 tensor만0으로 바꾸고 가용성 플래그는1로 유지한 경우다.','']
    stress_headers,stress_rows=snapshot_stress_table(out)
    table('early_zero_input_stress',stress_headers,stress_rows)
    lines+=['고정 Train subset에는 global 완전0 event가 없었으므로 인위적 분포 밖 스트레스 검사이기도 하다. '
            '정상 입력 대비0입력에서 잔차가 커지고 검출이 사라지는 민감성을 확인했다. 가용성 플래그0의 정확한 fallback과 구분한다. '
            '이 한 프레임 결과를 전체 성능이나 정상 입력 열화의 원인으로 일반화하지 않는다. '
            '중간 진단 때문에 진행 중 trial의 입력 정책·가중치·예산을 변경하지 않았다.','']
    lines+=['## 동일 GPU 전체 파이프라인 속도','']
    table('runtime',['조건','캐시 fps','온라인 fps','캐시 실행 로딩 (s)','온라인 실행 로딩 (s)','peak GPU (GiB)','peak host (GiB)','parameter 수'],
          [[label(n),number(runtime[n+'_cached']['fps']),number(runtime[n+'_online']['fps']),number(runtime[n+'_cached']['loading_seconds']),
            number(runtime[n+'_online']['loading_seconds']),
            number(max(x['peak_allocated_bytes'] for mode in ('cached','online') for x in runtime[n+'_'+mode]['segments'])/1024**3),
            number(max(runtime[n+'_cached']['host_peak_rss_bytes'],runtime[n+'_online']['host_peak_rss_bytes'])/1024**3),number(runtime[n+'_cached']['total_parameters'],0)] for n in EVAL_NAMES])
    uuid=runtime[NAMES[0]+'_cached']['gpu_uuid']
    lines += [f'모든18실행은 동일 UUID `{uuid}`에서 순차 수행했고 전후 compute PID가 실행 본인뿐임을 검사했다. '
              '고정3세션×100프레임에서 입력 열기·원본 디코드/인코딩 또는 캐시I/O·검출·복원·좌표변환·매칭·저장·닫기를 포함한다. '
              '모델 로딩과 parity 검사는 fps에서 분리했다. OS 캐시를 강제 비우지 않았고 온라인 입력은 보관 archive를 여는 구현으로 실제 live stream 지연과 다르다.','',
              f"캐시 {cache_cost['input_cache_bytes']/1024**3:.3f} GiB를 검증해 재사용했다. 이번 실험의 새 입력 캐시 생성은0회·추가 저장0바이트다. "
              f"기존 파일 생성 시각으로 확인한 전체 생성 span은 {cache_cost['prior_wall_span_seconds']/3600:.3f}시간이며 병렬 세션 처리시간 합과 다르다.",'']
    controls=[n for n in ('rgb3_short90','rgb4') if training[n]['best_metrics']['coverage']>=.95]
    recommendation=min(controls,key=lambda n:training[n]['best_metrics']['pa_mpjpe_mm'])
    eligible=[]
    for ref,var in [('rgb3_short90','rgb3_event4'),('rgb4','event3_rgb4'),('rgb3_short90','rgb3_event3'),('rgb4','rgb4_event4')]:
        pa=result['tests'][ref+'__'+var+'_pa'];root=result['tests'][ref+'__'+var+'_root_position']
        if (pa['ci95'] and pa['ci95'][1]<0 and pa['holm_p_value']<.05 and root['effect']<=0
            and result['coverage'][var]['fraction']>=result['coverage'][ref]['fraction']-.01
            and training[var]['best_metrics']['pa_mpjpe_mm']<training[ref]['best_metrics']['pa_mpjpe_mm']):
            eligible.append(var)
    if eligible:
        recommendation=min(eligible,key=lambda n:training[n]['best_metrics']['pa_mpjpe_mm'])
    lines+=['## 단일 추천과 해석','',f'추천: **{label(recommendation)}**. '
            +('Train holdout에서 개선되고 Test의 보정 PA 개선·root 비열화·검출률 조건까지 통과한 후보 중 Train PA로 골랐다.' if eligible else
              'event 추가의 보정된 개선 기준을 통과한 후보가 없어, Train holdout으로 선택한 단순 RGB 대조군을 추천한다. 개선을 입증하지 못한 결과도 그대로 보고한다.'),'',
            '기존 RGB3+event4는 RGB3 자체보다 미세하게 좋았으나 더 강한 RGB4보다 못했다. 큰 열화는 event를 주 검출/복원 입력으로 사용하면서 RGB를 작은 보조 CNN으로 제한한 조건에서 관측됐다. '
            '새 구조는 그 RGB 경로를 보존하며, auxiliary LR과 공간 adapter·기하 샘플링을 함께 변경했다. 고정 Train 6,501프레임의 비영 encoded pixel 중앙값32/255로부터 두 event encoder 입력 gain4를 고정하고 새 학습 분기에 GroupNorm을 적용했다. 기존 projection 경로의 스케일은 보존한다. 여러 변경을 묶었으므로 각각의 독립 인과 효과를 단정하지 않는다.','',
            '한계: 기존 Test 결과를 본 뒤 문제를 제기한 동일 benchmark 재실험이므로 완전한 맹검 신규 Test가 아니다. 새 checkpoint/설정 선택은 holdout으로 한정했다. '
            '새 추가학습은 한 seed이며 원본부터 독립 seed 반복이 아니다. 기존 warmstart가 학습한 Train 경험도 함께 보고했다. '
            'event는 저장 희소 관측의 count33이며 원시 이벤트 개별 시각·횟수 복원을 주장하지 않는다. 정확한 실시간 센서 배포·새 피험자 일반화를 보장하지 않는다.','']
    if eligible:
        reference={'rgb3_event4':'rgb3_short90','rgb3_event3':'rgb3_short90',
                   'event3_rgb4':'rgb4','rgb4_event4':'rgb4'}[recommendation]
        evidence=result['tests'][reference+'__'+recommendation+'_pa']
        lines+=['추천의 근거 수치: Train holdout PA-MPJPE는 '
                f"{training[reference]['best_metrics']['pa_mpjpe_mm']:.4f}→{training[recommendation]['best_metrics']['pa_mpjpe_mm']:.4f}mm다. "
                f"쌍별 공통 Test {evidence['samples']:,}표본의 PA 효과는 {evidence['effect']:.4f}mm, "
                f"95%CI [{evidence['ci95'][0]:.4f}, {evidence['ci95'][1]:.4f}]mm, Holm p={evidence['holm_p_value']:.6f}다. "
                '주표는9조건 공통 GT이며 이 통계는 두 조건 공통 GT이므로 평균과 차이가 다를 수 있다. '
                '추천 후보의 순서는 사전 고정한 Train PA로 정했으며 Test 최저값으로 바꾸지 않았다. '
                '후보 간 작은 차이의 재현성은 seed 반복으로 확인하지 못했다.','']
        lines += [f"추천 조건의 캐시 처리량은 {runtime[recommendation+'_cached']['fps']:.2f}fps, "
                  f"온라인 처리량은 {runtime[recommendation+'_online']['fps']:.2f}fps다. "
                  f"대조 RGB의 온라인 {runtime[reference+'_online']['fps']:.2f}fps보다 느리므로 입력 인코딩·archive I/O 비용을 함께 고려해야 한다. "
                  '실시간 처리나 event 고유 기여를 입증한 추천은 아니다.','']
    chart,image=figures(out,result)
    trend=learning_curve_figure(out,result,training,source_training)
    lines+=['## 전체 Test 비교 그림','',f'![수정융합 다섯 지표]({chart.resolve()})','',
            '## 학습량별 실측 경향','',f'![학습량별 실측 경향]({trend.resolve()})','',
            '각 장치의 원본·출처·수정 방법들이 함께 검출한 동일 GT에서 계산했다. 가로축은 선택 checkpoint가 실제 거친 누적 학습 시간이며 trial 전체 탐색 예산과 다르다. '
            '원본과 출처 점은 같은 초기 경로이고 마지막 융합점은 센서와 학습량 변경이 함께 포함된다. 결과가 단조롭게 좋아진다고 가정하거나 값을 조정하지 않았다.','',
            '## 실제 예측','',f'![수정융합 실제 Test 예측]({image.resolve()})','',
            '모든 패널은 같은 시각·인물의 실제 예측이며 RGB3 배경은 보고용이다. event-only의 추론에 RGB를 넣지 않았다.','']
    text='\n'.join(lines)+'\n';proof=verify_tables(text,proofs)
    path=Path(goal['report']);path.write_text(text)
    # Re-read the persisted artifact; formatting or write errors must not pass.
    verify_tables(path.read_text(),proofs)
    (out/'rendered_table_proof.json').write_text(json.dumps(dict(status='passed',verification=proof,tables=proofs),ensure_ascii=False,indent=2)+'\n')
    return path,recommendation

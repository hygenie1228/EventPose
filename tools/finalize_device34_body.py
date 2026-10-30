"""Six-condition raw aggregation, independent metric audit and Korean report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from scipy.spatial.transform import Rotation
from multihmr.multisensor_model import CONDITIONS
from utils.chunk_alignment import METRICS
from utils.compare_event_methods import segment_metrics,summarize
from utils.paired_statistics import paired_cluster,holm_adjust,motion_strata
from tools.prepare_device34_body import OUT,save

HEADLINE=METRICS+['pelvis_error_mm']
LABELS=dict(rgb3='device3 RGB',rgb4='device4 RGB',rgb3_rgb4='device3 RGB + device4 RGB',
    rgb3_event4='device3 RGB + device4 event',event3_rgb4='device3 event + device4 RGB',
    event3_event4='device3 event + device4 event')


def read(path):return json.loads(path.read_text())


def verify_main_table(text,result):
    """Parse rendered Markdown values; a report hash alone cannot prove numbers."""
    lines=text.splitlines()
    header=next(i for i,line in enumerate(lines) if line.startswith('| 조건 | first frame-aligned MPJPE'))
    columns=[v.strip() for v in lines[header].strip('|').split('|')]
    assert len(columns)==6 and columns[-1]=='Root position error (mm)'
    assert 'RPE-Z' not in lines[header]
    rendered=lines[header+2:header+8];assert len(rendered)==6
    population=result['populations']['common6'];names=result['conditions']
    for line,name in zip(rendered,names):
        cells=[v.strip() for v in line.strip('|').split('|')]
        assert len(cells)==6 and cells[0]==LABELS[name]
        for cell,key in zip(cells[1:],HEADLINE):
            expected=population[name][key]
            if expected is None:
                assert cell=='미평가';continue
            actual=float(cell.replace('**',''))
            assert abs(actual-expected)<=.00500001,(name,key,actual,expected)
            others=[population[n][key] for n in names]
            best=all(v is not None for v in others) and expected==min(others)
            assert cell.startswith('**')==best,(name,key,'best value formatting')
    return dict(rows=6,numeric_columns=5,header_correct=True,all_rendered_values_match=True,best_values_correct=True)


def verify_supporting_tables(text,result,training=None,runtime=None):
    import re
    lines=text.splitlines();names=result['conditions'];population=result['populations']['common6']
    def table(prefix):
        index=next(i for i,line in enumerate(lines) if line.startswith(prefix))+2
        rows=[]
        while index<len(lines) and lines[index].startswith('|'):
            rows.append([c.strip().replace('**','').strip('`') for c in lines[index].strip('|').split('|')]);index+=1
        return rows
    def number(cell,expected,places=2):
        if expected is None:assert cell=='미평가';return
        actual=float(cell.replace(',',''));assert np.isfinite(actual)
        assert abs(actual-expected)<=.5*10**(-places)+1e-8,(cell,expected,places)
    match=re.search(r'Test (\d+)구간·([\d,]+)프레임, 유효 GT ([\d,]+)명-프레임',text)
    assert match and [int(v.replace(',','')) for v in match.groups()]==[result['segments'],result['frames'],result['gt_instances']]
    rows=table('| 조건 | 전체 GT 검출률');assert len(rows)==6
    for cells,name in zip(rows,names):
        assert cells[0]==LABELS[name] and len(cells)==6
        row=population[name]
        values=[100*result['coverage'][name]['fraction'],row['absolute_mpjpe_mm'],row['root_depth_error_mm'],
            row['pelvis_velocity_error_mmps'],row['root_acceleration_error_mmps2']]
        for cell,value in zip(cells[1:],values):number(cell,value)
    rows=table('| 조건 | 자체 frame0');assert len(rows)==6
    for cells,name in zip(rows,names):
        own=result['populations']['own_'+name][name]
        assert cells[0]==LABELS[name] and int(cells[1])==own['samples'] and len(cells)==5
        for cell,value in zip(cells[2:],[own[METRICS[3]],own['pelvis_error_mm'],own['root_depth_error_mm']]):number(cell,value)
    rows=table('| RGB3 대비 조건·지표');assert len(rows)==len(result['tests'])
    for cells,(key,test) in zip(rows,result['tests'].items()):
        assert cells[0]==key and int(cells[1])==test['samples'] and int(cells[2])==test['clusters']
        number(cells[3],test['effect']);number(cells[4],test.get('relative_percent'));number(cells[6],test['holm_p_value'],5)
        if test['ci95'] is None:assert cells[5]=='미평가'
        else:
            ends=cells[5].strip('[]').split(',');assert len(ends)==2
            for cell,expected in zip(ends,test['ci95']):number(cell,expected)
    if training is not None:
        rows=table('| 조건 | 실제 경과');assert len(rows)==6
        for cells,name in zip(rows,names):
            row=training[name];assert cells[0]==LABELS[name]
            number(cells[1],row['elapsed_seconds']/3600,3)
            assert [int(c) for c in cells[2:6]]==[row['successful_steps'],row['samples_seen'],row['unique_frames_seen'],row['completed_subset_epochs']]
            assert cells[6]==row['checkpoint']
    if runtime is not None:
        rows=table('| 조건 | 캐시 fps');assert len(rows)==6
        for cells,name in zip(rows,names):
            a=runtime[name+'_cached'];b=runtime[name+'_online'];assert cells[0]==LABELS[name]
            expected=[a['fps'],b['fps'],a['loading_seconds'],max(r['peak_allocated_bytes'] for r in a['segments'])/1024**3,
                max(a['host_peak_rss_bytes'],b['host_peak_rss_bytes'])/1024**3]
            for cell,value in zip(cells[1:6],expected):number(cell,value)
            assert int(cells[6].replace(',',''))==a['total_parameters']
    return dict(test_counts=True,coverage_and_errors=True,own_population=True,
        effect_sizes_CI_Holm=True,training_budget=training is not None,runtime=runtime is not None)


def verify_followup_tables(text,primary,replication):
    """Check rendered repeat-seed and motion tables against their raw aggregates."""
    lines=text.splitlines()
    def rows(prefix):
        index=next(i for i,line in enumerate(lines) if line.startswith(prefix))+2
        result=[]
        while index<len(lines) and lines[index].startswith('|'):
            result.append([c.strip() for c in lines[index].strip('|').split('|')]);index+=1
        return result
    def number(cell,value):
        if value is None:assert cell=='미평가';return
        assert np.isfinite(float(cell)) and abs(float(cell)-value)<=.00500001,(cell,value)
    repeat_rows=rows('| 조건 | 주 seed PA');assert len(repeat_rows)==6
    for cells,name in zip(repeat_rows,CONDITIONS):
        assert len(cells)==5 and cells[0]==LABELS[name]
        a=primary['populations']['common6'][name]
        b=replication['populations']['common6'][name+'_seed261027']
        for cell,value in zip(cells[1:],[a[METRICS[3]],b[METRICS[3]],a['pelvis_error_mm'],b['pelvis_error_mm']]):
            number(cell,value)
    motion_rows=rows('| RGB3 대비 조건 | 동작 |')
    expected=[(name,label,members) for name,bins in primary['motion_strata'].items() for label,members in bins.items()]
    assert len(motion_rows)==len(expected)
    for cells,(name,label,members) in zip(motion_rows,expected):
        a=members['rgb3'];b=members[name]
        assert len(cells)==5 and cells[:2]==[LABELS[name],label]
        assert int(cells[2])==a['count']==b['count']
        for cell,row in zip(cells[3:],[a,b]):
            number(cell,row['sum_mm']/row['count'] if row['count'] else None)
    return dict(repeat_seed_rendered_values=True,motion_strata_rendered_values=True)


def independent_errors(pred,target,mask,positions):
    """Separate SciPy proper-rotation implementation, starting from raw joints."""
    first=np.flatnonzero(positions==0)
    eligible=mask[first[0]] if len(first) else np.zeros(mask.shape[1],bool)
    keep=mask&eligible[None]
    sums=np.zeros(5,np.float64);count=int(keep.sum())
    def fit(p,t):
        pm=p.mean(0);tm=t.mean(0);x=p-pm;y=t-tm
        r=Rotation.align_vectors(y,x)[0].as_matrix()
        assert np.linalg.det(r)>0
        scale=float((y*(x@r.T)).sum()/np.square(x).sum())
        assert scale>0
        return r,scale,tm-scale*(pm@r.T)
    for person in np.flatnonzero(eligible):
        p=pred[keep[:,person],person,:22].astype(np.float64)
        t=target[keep[:,person],person,:22].astype(np.float64)
        ap=pred[first[0],person,:22].astype(np.float64)
        at=target[first[0],person,:22].astype(np.float64)
        r,s,b=fit(ap,at)
        sums[0]+=np.linalg.norm(p+(at[0]-ap[0])-t,axis=-1).mean(-1).sum()
        sums[1]+=np.linalg.norm(s*(p@r.T)+b-t,axis=-1).mean(-1).sum()
        sums[2]+=np.linalg.norm((p-p[:,:1])-(t-t[:,:1]),axis=-1).mean(-1).sum()
        for pp,tt in zip(p,t):
            rr,ss,bb=fit(pp,tt)
            sums[3]+=np.linalg.norm(ss*(pp@rr.T)+bb-tt,axis=-1).mean()
        sums[4]+=np.linalg.norm(p[:,0]-t[:,0],axis=-1).sum()
    return sums,count


def aggregate(replications=False):
    manifest=read(OUT/'manifest.json')
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['split_sha256']
    tasks=read(OUT/'test/tasks.json');names=list(CONDITIONS)
    if replications:names=[n+'_seed261027' for n in names]
    reference=names[0]
    groups={'common6':names,**{'own_'+n:[n] for n in names},**{'pair_'+n:[reference,n] for n in names[1:]}}
    rows={group:{n:[] for n in members} for group,members in groups.items()}
    coverage=dict.fromkeys(names,0);total=0;raw_sums={n:np.zeros(5) for n in names};raw_counts=dict.fromkeys(names,0)
    motion={n:{label:{member:dict(count=0,sum_mm=0.) for member in (reference,n)}
        for label in ('still','slow','fast','unknown')} for n in names[1:]}
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions'];stamps=z['timestamps_ns']
            assert np.array_equal(z['original_source_rgb_positions'],positions+task['start_frame'])
            assert positions[0]==0 and len(positions)==task['video_frames']
        total+=int(valid.sum());predictions={};masks={}
        for name in names:
            folder=OUT/'results'/name/task['session'];complete=read(folder/'complete.json')
            assert complete['task']['split_sha256']==manifest['split_sha256']
            assert complete['task']['matching_protocol']=='primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1'
            assert complete['task']['start_frame']==task['start_frame'] and complete['task']['end_frame']==task['end_frame']
            with np.load(folder/'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                p=z['matched_joints_cam_m'].astype(np.float64)
                mask=valid&(z['matched_pred_indices']>=0)
                assert p.shape==gt.shape and np.isfinite(p[mask]).all()
                assert np.isnan(p[valid&~mask]).all()
                predictions[name]=p;masks[name]=mask;coverage[name]+=int(mask.sum())
        strata=motion_strata(gt,valid,positions,task['fps'],timestamps_ns=stamps)
        for group,members in groups.items():
            common=np.logical_and.reduce([masks[n] for n in members])
            for name in members:
                row,errors,eligible=segment_metrics(predictions[name],gt,common,positions,task['fps'],timestamps_ns=stamps)
                row.update(session=task['session'],parent_session=task['parent_session'],
                    cluster=int(task['parent_session'].split('_')[3]))
                rows[group][name].append(row)
                if group=='common6':
                    independent,n=independent_errors(predictions[name],gt,common,positions)
                    expected=np.array(row['alignment_sums_m']+[row['pelvis_sum_m']])
                    assert n==row['count'] and np.allclose(independent,expected,rtol=1e-6,atol=1e-7),(name,task['session'])
                    raw_sums[name]+=independent;raw_counts[name]+=n
                if group.startswith('pair_'):
                    variant=group[5:]
                    for label,code in [('still',0),('slow',1),('fast',2),('unknown',-1)]:
                        selected=eligible&(strata==code)
                        motion[variant][label][name]['count']+=int(selected.sum())
                        motion[variant][label][name]['sum_mm']+=float(errors[selected,3].sum()*1000)
    populations={group:{n:summarize(values) for n,values in members.items()} for group,members in rows.items()}
    tests={};clusters=sorted({r['cluster'] for r in rows['common6'][reference]})
    for variant in names[1:]:
        group=rows['pair_'+variant]
        for metric,index in [('pa',3),('root_position',4)]:
            sums={};counts=None
            for name in (reference,variant):
                selected=[[r for r in group[name] if r['cluster']==c] for c in clusters]
                sums[name]=[1000*sum(r['alignment_sums_m'][index] if index<4 else r['pelvis_sum_m'] for r in rr) for rr in selected]
                current=[sum(r['count'] for r in rr) for rr in selected]
                if counts is not None:assert counts==current
                counts=current
            tests[variant+'_'+metric]=paired_cluster(sums[reference],sums[variant],counts,seed=261026)
    for suffix in ('_pa','_root_position'):
        adjusted=holm_adjust({k:v['p_value'] for k,v in tests.items() if k.endswith(suffix)})
        for k,p in adjusted.items():tests[k]['holm_p_value']=p
    result=dict(split_sha256=manifest['split_sha256'],conditions=names,gt_instances=total,
        frames=sum(t['video_frames'] for t in tasks),segments=len(tasks),
        coverage={n:dict(matched=coverage[n],total=total,fraction=coverage[n]/total) for n in names},
        populations=populations,tests=tests,motion_strata=motion,
        raw_recalculation=dict(implementation='independent scipy Rotation.align_vectors, proper Sim3',
            sums_m={n:v.tolist() for n,v in raw_sums.items()},counts=raw_counts,passed=True),updated_unix=time.time())
    prefix='replication_' if replications else ''
    save(OUT/(prefix+'segment_sums.json'),rows);save(OUT/(prefix+'comparison.json'),result)
    return result


def visualize(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    tasks=read(OUT/'test/tasks.json');names=result['conditions']
    bones=[(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),
        (9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21)]
    for task in tasks:
        with np.load(task['gt_file']) as z:gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
        predictions=[];common=valid.copy()
        for name in names:
            with np.load(OUT/'results'/name/task['session']/'result.npz') as z:
                predictions.append(z['matched_joints_cam_m']);common&=z['matched_pred_indices']>=0
        hits=np.argwhere(common)
        selection='first common matched sample'
        if not len(hits):
            hits=np.argwhere(valid);selection='first valid GT sample; missing predictions explicitly shown'
        if not len(hits):continue
        frame,person=map(int,hits[0]);absolute=task['start_frame']+int(positions[frame])
        with h5py.File(OUT/'cache'/(task['parent_session']+'.h5'),'r',locking=False) as f:
            index=int(np.searchsorted(f['frames'][:],absolute));image=f['rgb3'][index].transpose(1,2,0);K=f['rgb3_K_primary'][:]
        fig,axes=plt.subplots(2,3,figsize=(16,10))
        for ax,name,p in zip(axes.flat,names,predictions):
            ax.imshow(image)
            for joints,color in [(gt[frame,person,:22],'#35ef91'),(p[frame,person,:22],'#ff5364')]:
                if not np.isfinite(joints).all():continue
                hom=joints@K.T;uv=hom[:,:2]/hom[:,2:]
                for a,b in bones:ax.plot(uv[[a,b],0],uv[[a,b],1],color=color,lw=1.7)
            if not np.isfinite(p[frame,person]).all():ax.text(.03,.94,'NO MATCHED DETECTION',transform=ax.transAxes,color='red',fontsize=12)
            ax.set_title(LABELS[name]);ax.axis('off')
        fig.suptitle('Green: GT   Red: prediction | identical GT time | RGB3 display only')
        path=OUT/'visualizations/공통표본_여섯조건_예측.png';path.parent.mkdir(exist_ok=True)
        fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
        save(OUT/'visualization_provenance.json',dict(task=task['session'],frame=absolute,person=person,
            selection=selection,image=str(path),
            event_only_RGB_inference_input=False))
        return path
    return None


def metric_figure(result):
    """Presentation figure from the verified six-condition common Test population."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    names=result['conditions'];population=result['populations']['common6']
    values=np.array([[population[n][k] for k in HEADLINE] for n in names],dtype=float)
    assert values.shape==(6,5) and np.isfinite(values).all()
    samples={population[n]['samples'] for n in names};assert len(samples)==1
    titles=['first frame-aligned\nMPJPE','first frame-aligned\nPA-MPJPE',
            'frame-aligned\nMPJPE','frame-aligned\nPA-MPJPE','Root position\nerror (3D)']
    labels=['D3 RGB','D4 RGB','D3 RGB + D4 RGB','D3 RGB + D4 event',
            'D3 event + D4 RGB','D3 event + D4 event']
    colors=['#2563a6','#3e86c6','#7560a7','#9279bc','#cc7637','#e4a264']
    fig,axes=plt.subplots(1,5,figsize=(19,5.8),sharey=True)
    for j,ax in enumerate(axes):
        ax.barh(np.arange(6),values[:,j],height=.62,color=colors)
        ax.set_title(titles[j],fontsize=12,pad=16)
        ax.set_xlim(0,float(values[:,j].max())*1.33)
        ax.set_xlabel('mm',fontsize=11)
        ax.set_yticks(np.arange(6),labels,fontsize=11)
        ax.tick_params(axis='y',length=0,pad=10)
        ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        best=int(np.argmin(values[:,j]))
        for i,value in enumerate(values[:,j]):
            ax.text(value+values[:,j].max()*.025,i,f'{value:.2f}',va='center',
                    fontsize=10,fontweight='bold' if i==best else 'normal')
    axes[0].invert_yaxis()
    fig.suptitle('Body 동작 · 두 장치 여섯 조건 · 전체 Test',fontproperties=font,fontsize=22,y=.98)
    fig.text(.5,.88,f"동일 공통 GT {next(iter(samples)):,}개 · 각 조건 3시간 학습 · 오차는 낮을수록 좋음",
             ha='center',fontproperties=font,fontsize=12,color='#475569')
    fig.text(.5,.025,'첫 실험 seed261026의 점추정치. 반복 실험·내부 검증값과 별개이며, 통계적 유의성은 보고서의 CI/Holm 결과를 참조.',
             ha='center',fontproperties=font,fontsize=10,color='#475569')
    fig.subplots_adjust(left=.145,right=.99,top=.72,bottom=.15,wspace=.28)
    path=OUT/'visualizations/전체Test_여섯조건_다섯지표.png';path.parent.mkdir(exist_ok=True)
    fig.savefig(path,dpi=180,facecolor='white')
    fig.savefig(path.with_suffix('.svg'),facecolor='white');plt.close(fig)
    save(OUT/'metric_figure_provenance.json',dict(conditions=names,metrics=HEADLINE,
         values_mm=values.tolist(),common_samples=next(iter(samples)),seed=261026,
         split_sha256=result['split_sha256'],image=str(path),svg=str(path.with_suffix('.svg')),
         scope='Primary full-Test common population point estimates only; not repeat or holdout'))
    return path


def report(result):
    names=result['conditions'];population=result['populations']['common6']
    def fmt(value):return '미평가' if value is None else f'{value:.2f}'
    lines=['# 신체 동작의 두 장치·두 센서 미세조정 비교','',
        '동일 body Train/Test와 원본 Multi-HMR 초기화로 여섯 조건을 비교했다. '
        '아래 수치는 새 실험의 실제 전체 Test 예측에서 계산했으며 과거 실험 결과를 재사용하지 않았다.','',
        f"Test {result['segments']}구간·{result['frames']:,}프레임, 유효 GT {result['gt_instances']:,}명-프레임. "
        f"주표의 여섯 조건 공통·정확 frame0 적격 표본은 {population[names[0]]['samples']:,}개다.", '',
        '| 조건 | first frame-aligned MPJPE (mm) | first frame-aligned PA-MPJPE (mm) | frame-aligned MPJPE (mm) | frame-aligned PA-MPJPE (mm) | Root position error (mm) |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        values=[]
        for key in HEADLINE:
            value=population[name][key];all_values=[population[n][key] for n in names]
            text=fmt(value)
            if all(v is not None for v in all_values) and value==min(all_values):text='**'+text+'**'
            values.append(text)
        lines.append('| '+LABELS[name]+' | '+' | '.join(values)+' |')
    lines+=['','첫 정렬은 정확한 구간 frame0에서 구한 변환을 고정한다. MPJPE는 pelvis 평행이동, '
        'PA는 반사를 금지한 proper Sim(3)이다. 첫 프레임에서 맞춘 회전·크기를 이후 프레임에 고정하므로 '
        'first frame-aligned PA-MPJPE는 이후 동작에서 first frame-aligned MPJPE보다 커질 수 있다. '
        'Root position error는 정렬 전 공통 device3 RGB 카메라 좌표에서 '
        '`mean(norm(root_pred-root_GT))*1000`이다. RPE-Z는 다음 보조표에서 구분한다. '
        '미검출을 0오차로 계산하지 않았으며 공통 표본이 없으면 미평가로 표시한다.','',
        '| 조건 | 전체 GT 검출률 (%) | 절대 MPJPE (mm) | RPE-Z (mm) | 속도 오차 (mm/s) | 가속도 오차 (mm/s²) |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        row=population[name]
        values=[result['coverage'][name]['fraction']*100,row['absolute_mpjpe_mm'],row['root_depth_error_mm'],
            row['pelvis_velocity_error_mmps'],row['root_acceleration_error_mmps2']]
        texts=[]
        for i,value in enumerate(values):
            ref=([result['coverage'][n]['fraction']*100 for n in names] if i==0 else
                [population[n][['absolute_mpjpe_mm','root_depth_error_mm','pelvis_velocity_error_mmps','root_acceleration_error_mmps2'][i-1]] for n in names])
            text=fmt(value)
            if all(v is not None for v in ref) and value==(max(ref) if i==0 else min(ref)):text='**'+text+'**'
            texts.append(text)
        lines.append('| '+LABELS[name]+' | '+' | '.join(texts)+' |')
    lines+=['','## 각 조건 자체의 검출 표본','',
        '다음 값은 조건마다 서로 다른 검출 표본을 사용하므로 이 표로 직접 순위를 정하지 않는다. '
        '주표는 여섯 조건의 동일 공통 표본을 유지하며, 자체 표본 표는 검출 누락에 따른 선택 편향을 확인하기 위한 보조 자료다.','',
        '| 조건 | 자체 frame0 적격 표본 | frame-aligned PA-MPJPE (mm) | Root position error (mm) | RPE-Z (mm) |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        own=result['populations']['own_'+name][name]
        lines.append(f"| {LABELS[name]} | {own['samples']} | {fmt(own[METRICS[3]])} | {fmt(own['pelvis_error_mm'])} | {fmt(own['root_depth_error_mm'])} |")
    lines+=['','## 학습 예산과 checkpoint 출처','',
        '| 조건 | 실제 경과 (시간) | 성공 update | 본 샘플 | 고유 프레임 | 완료 subset epoch | checkpoint |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name in names:
        row=read(OUT/'runs'/name/'complete.json')
        assert row['successful_steps']>0 and row['elapsed_seconds']<4*3600
        lines.append(f"| {LABELS[name]} | {row['elapsed_seconds']/3600:.3f} | {row['successful_steps']} | "
            f"{row['samples_seen']} | {row['unique_frames_seen']} | {row['completed_subset_epochs']} | `{row['checkpoint']}` |")
    manifest=read(OUT/'manifest.json');cache=read(OUT/'training_cache_complete.json')
    source_rows={name:read(OUT/'source_summaries'/(name+'.json')) for name in manifest['training_sessions']+manifest['validation_sessions']}
    full_train_valid=sum(row['valid_gt_frames'] for row in source_rows.values())
    fit_valid=sum(source_rows[name]['valid_gt_frames'] for name in manifest['training_sessions'])
    lines+=['',f"body 전체 Train의 유효 원본 프레임은 {full_train_valid:,}개이고, holdout을 제외한 학습 세션의 유효 원본 프레임은 "
        f"{fit_valid:,}개다. 고정 학습 subset {cache['training_frames']:,}개는 각각 "
        f"{100*cache['training_frames']/full_train_valid:.3f}% 및 {100*cache['training_frames']/fit_valid:.3f}%다. "
        '고유 관측 프레임은 위 조건별 값이며 반복 관측 횟수를 원본 coverage로 세지 않았다.']
    lines+=['',f"원본 사전학습 checkpoint에서 각 조건을 독립 초기화했다. seed261026, batch16, AdamW·AMP, "
        '공통 multiplicative intensity 증강, head/adapter 우선 및 30분 후 후반8개 backbone의 작은 학습률 업데이트를 사용했다. '
        '학습 창은 조건마다 3시간이며 로딩·검증·저장 경과를 포함하고 실제 성공 update 수는 위와 같다. '
        'checkpoint 선택은 Train 내부 holdout만 사용했다.', '',
        f"학습 {len(manifest['training_sessions'])}세션·{cache['training_frames']:,} 고정 프레임, "
        f"holdout {len(manifest['validation_sessions'])}세션·{cache['validation_frames']:,}프레임. "
        '피험자별 body 세션에서 시간대를 균등하게 뽑고 모든 센서가 사용 가능한 동일 subset을 사용했다. '
        '완료 epoch는 이 subset에 대한 값이며 전체 원본 Train의 epoch를 의미하지 않는다.','',
        '## 통계 비교','',
        '| RGB3 대비 조건·지표 | 공통 표본 | 피험자 cluster | 차이 (mm) | 상대 차이 (%) | 95% CI (mm) | Holm p |',
        '|---|---:|---:|---:|---:|---|---:|']
    for key,row in result['tests'].items():
        ci='미평가' if row['ci95'] is None else '['+', '.join(fmt(v) for v in row['ci95'])+']'
        p='미평가' if row['holm_p_value'] is None else f"{row['holm_p_value']:.5f}"
        lines.append(f"| {key} | {row['samples']} | {row['clusters']} | {fmt(row['effect'])} | {fmt(row.get('relative_percent'))} | {ci} | {p} |")
    lines+=['','차이는 비교 조건−RGB3이며 음수는 오차 감소다. 10,000회 피험자 cluster bootstrap과 '
        'cluster sign-flip 검정을 적용했고 PA 및 root 위치의 각 5개 비교에 별도로 Holm 보정을 적용했다. '
        '프레임 수를 독립 표본 수로 취급하지 않았다. 효과는 공통 프레임 수로 가중한 평균 오차 차이다. '
        'sign-flip p값은 귀무가설 아래 피험자별 차이의 부호 교환 가능성·대칭성을 가정하며, '
        '유의하지 않은 결과가 두 방법의 동등성을 증명하지는 않는다.','',
        '## 좌표와 입력 검증','',
        'device3·4의 RGB와 이벤트는 센서별 내외부 calibration으로 분리 처리했다. 주 센서는 원본 Multi-HMR '
        'backbone/검출기를 사용하고 보조 센서는 별도 encoder와 보정된 ray·camera origin의 attention으로 결합했다. '
        '이 구현은 주·보조 센서가 비대칭이다. 보조 RGB도 사전학습된 두 번째 Multi-HMR 대신 작은 CNN을 사용하므로 '
        '두 개의 대형 RGB backbone을 사용하는 방식 전체의 성능을 대표하지 않는다. '
        '보조 encoder가 추가되는 조건의 parameter·비용은 최종 속도 검증에서 구분한다. '
        '보정 timestamp 기준 과거 RGB 프레임 및 count33 극성분리 이벤트를 대응시켰으며 미래 이벤트, '
        'event MP4, GT box/pose/depth를 추론 입력으로 사용하지 않았다. '
        '이벤트 두 센서 조건은 실제 픽셀 접근 기록에서도 RGB 입력이 없어야 통과한다. '
        '검출 대응은 각 주센서의 원래 픽셀 평면에서 동일 GT 인물에 대해 계산하고, '
        '대응된 3차원 예측은 공통 device3 RGB 카메라 좌표에서 원형 그대로 평가했다. '
        '다른 카메라로의 재투영 때문에 깊이 오차가 검출 실패로 중복 집계되는 것을 방지했다.','',
        '## 검증과 후속 실험','',
        '원시 관절로부터 독립 SciPy 회전 적합 구현으로 주표 5지표를 다시 계산해 집계와 대조했다. '
        '이는 여섯 조건 표를 먼저 완성한 단계이며, 이후 seed261027의 여섯 추가 trial, '
        '동일 GPU 온라인/캐시 속도·메모리·캐시 비용 및 최종 보고서 감사를 계속한다. '
        '단일 seed의 cluster CI는 학습 seed 변동을 측정하지 않는다. 아직 최종 Goal 완료를 뜻하지 않는다.','']
    chart=metric_figure(result)
    lines+=['## 전체 Test 다섯 지표 비교 그림','',f'![전체 Test 여섯 조건 다섯 지표]({chart.resolve()})','',
        '첫 실험의 공통 GT 점추정치이며 반복 실험·내부 검증값과 구분한다. 각 패널의 축 범위가 다르므로 패널 사이 막대 길이보다 표시 수치로 비교한다.','']
    image=visualize(result)
    if image:lines+=['## 실제 예측 시각화','',f'![공통 표본의 여섯 조건 예측]({image.resolve()})','',
        '녹색은 GT, 빨간색은 실제 예측이다. 공통 매칭 표본이 없으면 첫 유효 GT 시각에서 미검출을 표시한다. '
        'RGB3 배경은 보고용이며 이벤트 단독 추론 입력이 아니다.','']
    path=ROOT/'_agents/report/261026_신체동작_두장치_여섯조건_비교.md'
    path.write_text('\n'.join(lines))
    verify_main_table(path.read_text(),result)
    verify_supporting_tables(path.read_text(),result,training={n:read(OUT/'runs'/n/'complete.json') for n in names})
    return path


def final_report():
    import torch
    import os
    import signal
    from tools.goal_runtime_control import identity
    # The user requested a single continuously updated report. Before its final
    # write/hash audit, stop only that report's registered observer, preserving
    # all training/evaluation processes and checking against PID reuse.
    goal=read(ROOT/'_agents/research/261026/device34_body_goal_28h.json')
    stopped_writers=[]
    for job in read(Path(goal['managed_jobs_file']))['jobs']:
        if job['role']!='device34_progress_table_30min':continue
        expected={key:job[key] for key in ('start_ticks','command','pgid')}
        if identity(job['pid'])!=expected:continue
        assert job['pgid']==job['pid'] and job['pgid']!=os.getpgrp()
        try:os.killpg(job['pgid'],signal.SIGTERM)
        except ProcessLookupError:continue
        until=time.monotonic()+5
        while identity(job['pid'])==expected and time.monotonic()<until:time.sleep(.1)
        assert identity(job['pid'])!=expected,'Report writer still live; refuse racing final audit'
        stopped_writers.append(job['pid'])
    primary=read(OUT/'comparison.json');replication=read(OUT/'replication_comparison.json')
    runtime=read(OUT/'runtime/summary.json');manifest=read(OUT/'manifest.json')
    input_audit=read(OUT/'actual_input_audit.json')
    assert input_audit['status']=='passed' and input_audit['split_sha256']==manifest['split_sha256']
    assert input_audit['all_frame_mappings_verified'] and input_audit['all_event_causality_verified'] and input_audit['all_relative_calibrations_verified']
    assert input_audit['independent_cpu_GT_samples']>0
    identities={v['gpu_uuid'] for v in runtime.values()};assert len(identities)==1 and len(runtime)==12
    assert all(v['input_parity_passed'] for v in runtime.values())
    checkpoint_records=[]
    configuration_parity=[]
    for condition in CONDITIONS:
        primary_config=read(OUT/'configs'/f'{condition}.json')
        repeat_config=read(OUT/'configs'/f'{condition}_seed261027.json')
        differences={key for key in set(primary_config)|set(repeat_config)
            if primary_config.get(key)!=repeat_config.get(key)}
        assert differences<={'name','seed','trial_started_unix'},(condition,differences)
        assert primary_config['seed']==261026 and repeat_config['seed']==261027
        assert primary_config['hours']==repeat_config['hours']==3
        configuration_parity.append(dict(condition=condition,passed=True,
            differing_fields=sorted(differences)))
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    base=ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'
    original=dict(path=str(base),sha256=sha(base))
    all_configs=read(OUT/'experiment_manifest.json')
    for config in all_configs['primary']+all_configs['replications']:
        name=config['name'];complete=read(OUT/'runs'/name/'complete.json')
        assert complete['status']=='complete' and complete['successful_steps']>0 and complete['elapsed_seconds']<4*3600
        best=OUT/'checkpoints'/name/'best.pt';last=OUT/'checkpoints'/name/'last.pt'
        assert best.is_file() and last.is_file()
        checkpoint=torch.load(best,map_location='cpu')
        live_config=read(OUT/'configs'/f'{name}.json')
        assert checkpoint['config']==live_config,(name,'best checkpoint configuration differs')
        assert checkpoint['config']['condition']==config['condition']
        assert checkpoint['config']['split_sha256']==manifest['split_sha256']
        assert checkpoint['config']['seed']==config['seed'] and checkpoint['test_used_for_selection'] is False
        assert checkpoint['metrics_apply_to_checkpoint'] and checkpoint['metrics_step']==checkpoint['step']
        assert all(torch.isfinite(v).all() for v in checkpoint['model_delta'].values())
        final_checkpoint=torch.load(last,map_location='cpu')
        assert final_checkpoint['config']==live_config,(name,'last checkpoint configuration differs')
        for key in ('condition','split_sha256','seed'):
            assert final_checkpoint['config'][key]==config[key]
        assert final_checkpoint['test_used_for_selection'] is False
        assert final_checkpoint['step']==complete['step']
        assert final_checkpoint['successful_steps']==complete['successful_steps']
        assert final_checkpoint['samples_seen']==complete['samples_seen']
        assert final_checkpoint['unique_frames_seen']==complete['unique_frames_seen']
        assert all(torch.isfinite(v).all() for v in final_checkpoint['model_delta'].values())
        checkpoint_records.append(dict(name=name,best=str(best),sha256=sha(best),last=str(last),
            last_sha256=sha(last),last_tensors_finite=True,last_training_counts_verified=True,
            selected_step=checkpoint['step'],final_step=complete['step'],successful_updates=complete['successful_steps']))
    event_sources=[OUT/'results'/n/'execution_rank0.json' for n in ['event3_event4','event3_event4_seed261027']]
    for path in event_sources:
        execution=read(path);assert set(execution['pixel_datasets_read'])=={'event3','event4'}
    path=report(primary);lines=path.read_text().splitlines()
    lines=[('주실험 여섯 조건과 seed261027의 여섯 추가 trial에서 전체 Test를 완료했다. '
        '원시 관절로부터 독립 SciPy proper rotation 구현으로 주표 다섯 지표를 재계산하고 집계와 대조했다. '
        '단일 seed의 cluster CI는 학습 seed 변동을 측정하지 않는다.'
        if line.startswith('원시 관절로부터 독립 SciPy') else line) for line in lines]
    lines+=['','## 원본 가중치와 선택 checkpoint 검증','',
        f"모든 trial의 원본 초기화는 `{original['path']}`이며 SHA256은 `{original['sha256']}`이다. "
        '조건별 선택 checkpoint 경로는 학습 예산 표에 분리했고, 각 best.pt의 SHA256·선택 update·최종 update는 '
        f"[최종 artifact 감사 기록]({(OUT/'final_artifact_audit.json').resolve()})에 보존한다. "
        '선택된 가중치와 holdout 평가의 update 일치, seed·split·조건 일치와 저장 tensor의 유한값을 확인했다.']
    lines+=['','## 동일 GPU의 전체 처리 비용','',
        '| 조건 | 캐시 fps | 온라인 fps | 모델 로딩 (초) | GPU 최대 할당 (GiB) | CPU 최대 RSS (GiB) | 모델 parameter |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in CONDITIONS:
        cached=runtime[name+'_cached'];online=runtime[name+'_online']
        peak=max(r['peak_allocated_bytes'] for r in cached['segments'])/1024**3
        rss=max(cached['host_peak_rss_bytes'],online['host_peak_rss_bytes'])/1024**3
        c=f"{cached['fps']:.2f}";o=f"{online['fps']:.2f}"
        if cached['fps']==max(runtime[n+'_cached']['fps'] for n in CONDITIONS):c='**'+c+'**'
        if online['fps']==max(runtime[n+'_online']['fps'] for n in CONDITIONS):o='**'+o+'**'
        lines.append(f"| {LABELS[name]} | {c} | {o} | {cached['loading_seconds']:.2f} | {peak:.2f} | {rss:.2f} | {cached['total_parameters']:,} |")
    cache=read(OUT/'cache_complete.json')
    cache_started=read(OUT/'preparation_gpu_identity.json')['started_unix']
    cache_workers=read(OUT/'preparation_gpu_identity.json').get('cpu_workers',4)
    cache_wall=cache['completed_unix']-cache_started
    lines+=['',f"동일 GPU UUID `{next(iter(identities))}`에서 각 조건을 독립 프로세스로 순차 실행했다. "
        '같은 Test 세션 세 곳의 첫 100프레임씩을 batch16으로 처리했다. 원본 RGB 디코드·이벤트 인코딩 또는 캐시 I/O·'
        '검출·복원·공통좌표 변환·GT 매칭·저장을 포함하며 모델 로딩은 분리했다. '
        '온라인/캐시 입력의 수치 일치를 별도로 검사했다. 원본 입력은 NFS dataset에서 읽고 공통 입력 캐시는 로컬 SSD에서 읽으므로 '
        '두 경로의 fps 차이는 인코딩 비용과 저장장치·네트워크 I/O 차이를 함께 포함한다. '
        '온라인은 저장된 원본 archive를 읽어 대응 프레임을 재계산하는 경로이며 실시간 센서 스트리밍 실측이 아니다. '
        'OS 캐시를 강제로 비우지 않았고 짧은 구간 및 archive 초기 로딩의 영향이 남는다.', '',
        f"공통 입력 캐시 용량은 {cache['bytes']/1024**3:.2f}GiB이고 GT 복원을 포함한 전체 준비 wall time은 "
        f"{cache_wall/3600:.2f}시간이다. 세션별 생성 경과 합은 "
        f"{cache['summed_session_seconds']/3600:.2f}시간이다. 최종 CPU 병렬 worker {cache_workers}개이며 경과 합을 전체 생성 wall time과 혼동하지 않는다. "
        '캐시 fps는 일회성 캐시 생성 비용을 제외하므로 온라인 fps와 함께 해석한다.','',
        '## 추가 seed trial의 확인','',
        '| 조건 | 주 seed PA (mm) | 추가 seed PA (mm) | 주 seed RPE (mm) | 추가 seed RPE (mm) |',
        '|---|---:|---:|---:|---:|']
    for name in CONDITIONS:
        a=primary['populations']['common6'][name];b=replication['populations']['common6'][name+'_seed261027']
        fmt=lambda x:'미평가' if x is None else f'{x:.2f}'
        lines.append('| '+LABELS[name]+' | '+' | '.join(fmt(x) for x in [a[METRICS[3]],b[METRICS[3]],a['pelvis_error_mm'],b['pelvis_error_mm']])+' |')
    lines+=['','각 seed 안에서는 여섯 조건의 동일 공통 GT를 사용했다. seed 사이의 검출 교집합은 달라질 수 있으므로 '
        '이 표의 평균 차이를 순수 seed 효과로 단정하지 않는다. 두 seed는 광범위한 seed 불확실성을 추정하기에 부족하다. '
        '추가 seed도 원본에서 독립 초기화했고 같은 데이터·증강·예산·holdout 선택 규칙을 사용했다.','',
        '## 동작 속도별 확인','',
        '| RGB3 대비 조건 | 동작 | 공통 표본 | RGB3 PA (mm) | 비교 조건 PA (mm) |',
        '|---|---|---:|---:|---:|']
    for name,bins in primary['motion_strata'].items():
        for label,members in bins.items():
            a=members['rgb3'];b=members[name];assert a['count']==b['count']
            means=['미평가','미평가'] if not a['count'] else [f"{a['sum_mm']/a['count']:.2f}",f"{b['sum_mm']/b['count']:.2f}"]
            lines.append(f"| {LABELS[name]} | {label} | {a['count']} | {means[0]} | {means[1]} |")
    lines+=['','동작 구분은 보정 timestamp의 GT 중앙차분 속도로 사전 고정했다. 정지는 ≤0.05m/s, 빠른 동작은 ≥0.5m/s이며 '
        'unknown은 경계·누락이다. GT 잡음의 영향이 남는 보조 분석이다.','']
    candidates=[]
    for name in list(CONDITIONS)[1:]:
        test=primary['tests'][name+'_pa'];root=primary['tests'][name+'_root_position']
        repeated=replication['tests'][name+'_seed261027_pa']
        if (test['holm_p_value'] is not None and test['holm_p_value']<.05 and test['effect']<0
            and root['effect'] is not None and root['effect']<=0
            and repeated['effect'] is not None and repeated['effect']<0
            and primary['coverage'][name]['fraction']>=primary['coverage']['rgb3']['fraction']-.01):
            candidates.append((test['effect'],name))
    recommended=min(candidates)[1] if candidates else 'rgb3'
    reason=('PA 개선이 Holm 보정 후 유의하고, root 위치 점 추정이 악화되지 않았으며 추가 seed에서 개선 방향이 유지되고 검출률도 유지되었다.'
        if candidates else '다중비교와 추가 seed·위치·검출률을 함께 고려할 때 복잡한 추가 센서의 일관된 이득을 확정할 근거가 부족해 단순한 RGB3 기준선을 우선한다.')
    lines+=['## 발표 결론','',f"우선 추천은 **{LABELS[recommended]}**다. {reason}", '',
        '최소 오차의 굵은 값은 점 추정의 순위이며 모든 쌍에서 통계적으로 우수하다는 뜻은 아니다. '
        '학습 subset, 두 seed, GT 및 센서 보정 오차, 검출 조건부 공통 표본과 실행 환경에 대한 한계가 있다. '
        '이벤트가 정지 부위의 모든 외형 정보를 복원한다고 가정하지 않는다.','']
    path.write_text('\n'.join(lines))
    rendered_audit=verify_main_table(path.read_text(),primary)
    supporting_audit=verify_supporting_tables(path.read_text(),primary,
        training={n:read(OUT/'runs'/n/'complete.json') for n in CONDITIONS},runtime=runtime)
    followup_audit=verify_followup_tables(path.read_text(),primary,replication)
    report_digest=sha(path)
    save(OUT/'final_artifact_audit.json',dict(status='passed_artifact_audit_agent_review_required',
        split_sha256=manifest['split_sha256'],original_checkpoint=original,checkpoints=checkpoint_records,
        primary_repeat_configuration_parity=configuration_parity,
        full_test_conditions=12,primary_conditions=6,raw_primary_metrics_recomputed=primary['raw_recalculation']['passed'],
        raw_replication_metrics_recomputed=replication['raw_recalculation']['passed'],same_runtime_gpu=next(iter(identities)),
        rendered_table_audit=rendered_audit,
        rendered_supporting_tables_audit=supporting_audit,
        rendered_followup_tables_audit=followup_audit,
        periodic_report_writers_stopped_before_hash=stopped_writers,
        actual_input_audit=str(OUT/'actual_input_audit.json'),
        event_only_no_RGB_reads=True,recommendation=recommended,report=str(path),report_sha256=report_digest,
        completed_unix=time.time(),goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['primary','replication','final']);a=p.parse_args()
    if a.stage=='final':
        aggregate(False);aggregate(True);final_report();sys.exit(0)
    result=aggregate(a.stage=='replication')
    if a.stage=='primary':
        path=report(result)
        save(OUT/'primary_table_verified.json',dict(verified_unix=time.time(),report=str(path),
            split_sha256=result['split_sha256'],raw_five_metrics_recomputed=True,all_six_full_test=True,
            additional_trials_speed_final_audit_still_required=True))

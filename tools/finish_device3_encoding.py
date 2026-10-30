"""Run final analysis/same-GPU timing and prepare review evidence; never auto-complete."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from device3_encoding_scope import GOAL,EVAL_NAMES,TRAIN_NAMES,LABELS,METRICS,PAIRS
from goal_runtime_control import write,register_job
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def figures(g):
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(g['output_root']);tasks=read(out/'test/tasks.json')
    gallery=read(ROOT/'exp/261028/device34_test_video_gallery/manifest.json')
    clips=gallery.get('clips',gallery.get('selected_clips',[]))
    # Freeze five published diverse Test examples, independent of new errors.
    selected=[]
    for c in clips:
        key=c.get('session',c.get('task',{}).get('session'))
        if key:
            match=next((t for t in tasks if t['session']==key),None)
            if match:selected.append(match)
    if len(selected)!=5:
        # Five disjoint participants by deterministic Test order, no error selection.
        selected=[];subjects=set()
        for t in tasks:
            pid=t['parent_session'].split('_')[3]
            if pid not in subjects:selected.append(t);subjects.add(pid)
            if len(selected)==5:break
    write(out/'visualization_selection.json',dict(tasks=selected,selection='Fixed five diverse published clips when key compatible; otherwise first five distinct Test participants by frozen task order; never selected by model errors'))
    edges=[(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21)]
    paths=[]
    for ix,task in enumerate(selected,1):
        pos=task['video_frames']//2
        with np.load(task['gt_file']) as z:gt=z['joints_cam_m'][pos,:,:22];valid=z['valid'][pos]
        predictions={}
        for n in EVAL_NAMES:
            with np.load(out/'results'/n/task['session']/'result.npz') as z:
                a=z['matched_joints_cam_m'][pos,:,:22];keep=z['matched_pred_indices'][pos]>=0
                predictions[n]=(a,keep)
        fig=plt.figure(figsize=(18,17));absolute=[]
        for a,k in predictions.values():
            if k.any():absolute.append(a[k].reshape(-1,3))
        if valid.any():absolute.append(gt[valid].reshape(-1,3))
        points=np.concatenate(absolute) if absolute else np.array([[0,0,2],[1,1,3]])
        low=points.min(0)-.15;high=points.max(0)+.15
        with h5py.File(out/'cache'/(task['parent_session']+'.h5'),'r',locking=False) as f:
            row=int(np.searchsorted(f['frames'][:],task['start_frame']+pos));rgb=f['rgb3'][row].transpose(1,2,0)
            assert int(f['frames'][row])==task['start_frame']+pos
            frame_timestamp_ns=int(f['timestamp_ns'][row])
        ax=fig.add_subplot(4,4,1);ax.imshow(rgb);ax.set_title('RGB3 reference');ax.axis('off')
        for j,e in enumerate(('count33','bins100','surface20'),2):
            sensor_rows=[]
            with h5py.File(out/'encoding_cache'/e/(task['parent_session']+'.h5'),'r',locking=False) as f:
                assert int(f['frames'][row])==task['start_frame']+pos and int(f['timestamp_ns'][row])==frame_timestamp_ns
                for sensor in ('event3','event4'):
                    v=f[sensor][row].astype(float)/255;images=[]
                    for pair in range(v.shape[0]//2):
                        im=np.zeros((224,224,3));im[:,:,0]=np.clip(v[2*pair]*4,0,1);im[:,:,2]=np.clip(v[2*pair+1]*4,0,1);images.append(im)
                    sensor_rows.append(np.concatenate(images,axis=1))
            ax=fig.add_subplot(4,4,j);ax.imshow(np.concatenate(sensor_rows,axis=0));ax.set_title(e+' / event3 top, event4 bottom / gain4');ax.axis('off')
        for j,n in enumerate(EVAL_NAMES,5):
            ax=fig.add_subplot(4,4,j,projection='3d');a,keep=predictions[n]
            for person in np.flatnonzero(valid):
                for u,v in edges:ax.plot(gt[person,[u,v],0],gt[person,[u,v],2],-gt[person,[u,v],1],color='gray',alpha=.5,lw=1)
                if keep[person]:
                    for u,v in edges:ax.plot(a[person,[u,v],0],a[person,[u,v],2],-a[person,[u,v],1],color='tab:blue',lw=1.2)
            ax.set(xlim=(low[0],high[0]),ylim=(low[2],high[2]),zlim=(-high[1],-low[1]),title=n+'\nmatched '+str(int((keep&valid).sum()))+'/'+str(int(valid.sum())),xlabel='X(m)',ylabel='Z(m)',zlabel='-Y(m)')
            ax.set_box_aspect([high[0]-low[0],high[2]-low[2],high[1]-low[1]])
        fig.suptitle(task['session']+' | frame '+str(task['start_frame']+pos)+' | timestamp_ns '+str(frame_timestamp_ns)+'\ncommon RGB3 camera | gray GT, blue actual prediction | no alignment | references and GT only for visualization')
        fig.tight_layout(rect=(0,0,1,.95));p=out/'figures'/f'실제예측_구간{ix}.png';p.parent.mkdir(exist_ok=True);fig.savefig(p,dpi=130);plt.close(fig);paths.append(str(p))
    write(out/'visualization_manifest.json',dict(paths=paths,actual_Test_predictions=True,GT_for_display_only=True))

def audit_checkpoints(g):
    import torch
    out=Path(g['output_root']);records=[];torch.set_num_threads(2)
    for n in TRAIN_NAMES:
        cfg=read(out/'configs'/(n+'.json'));done=read(out/'runs'/n/'complete.json');assert done['status']=='complete'
        for stage in ('best','last'):
            p=out/'checkpoints'/n/(stage+'.pt');v=torch.load(p,map_location='cpu')
            assert v['config']==cfg and v['successful_steps']>0 and v['test_used_for_selection'] is False
            assert v['metrics_apply_to_checkpoint'] and v['metrics_step']==v['step']
            assert v['unique_frames_seen']==len(set(v['seen_indices']))<=6501
            assert all(torch.isfinite(t).all() for t in v['model_delta'].values())
            records.append(dict(name=n,stage=stage,path=str(p),sha256=sha(p),successful_updates=v['successful_steps'],
                samples_seen=v['samples_seen'],unique_frames=v['unique_frames_seen'],elapsed_seconds=v['training_elapsed_seconds']))
            del v
    write(out/'checkpoint_audit.json',dict(status='passed',records=records,original_initialization=True))

def final_addendum(g):
    out=Path(g['output_root']);c=read(out/'comparison.json');runtime=read(out/'runtime/summary.json')
    lines=['## 최종 검증과 해석','',
        f'전체11조건 {c["segments"]}구간·{c["frames"]:,}프레임을 실제 평가했다. '
        f'원시 예측에서5지표를 독립 재계산했으며 오차합 최대차는 {c["independent_maximum_sum_difference_m"]:.3g}m다. '
        '이는 두 계산 구현의 차이이며 모델의 예측오차가0이라는 의미가 아니다.', '',
        '| 조건 | 전체 GT 검출수 / GT | 검출률 |','|---|---:|---:|']
    for n in EVAL_NAMES:
        r=c['coverage'][n];lines.append(f'| {LABELS[n]} | {r["matched"]:,} / {r["total"]:,} | {100*r["fraction"]:.2f}% |')
    lines+=['','## 같은 GPU 순차 속도','',
        '같은 물리GPU·UUID에서 조건별 새 프로세스로 캐시와 온라인을 순차 실행했다. '
        '실제 캐시I/O·원본 디코드/인과적 인코딩·검출·복원·좌표변환·매칭·저장을 포함한다. '
        '고정300프레임 기준이며 OS 캐시는 비우지 않았다. 온라인은 원시 event 스트림이 아니라 저장 archive 접근이다.', '',
        '| 조건 | 캐시 fps | 온라인 fps | 로딩(초) | 캐시 GPU peak(GiB) | 온라인 host RSS(GiB) |','|---|---:|---:|---:|---:|---:|']
    uuids={r['gpu_uuid'] for r in runtime.values()};assert len(uuids)==1
    lines+=[]
    for n in EVAL_NAMES:
        a=runtime[n+'_cached'];b=runtime[n+'_online'];peak=max(r['peak_allocated_bytes'] for r in a['segments'])/1024**3
        lines.append(f'| {LABELS[n]} | {a["fps"]:.2f} | {b["fps"]:.2f} | {a["loading_seconds"]:.2f} | {peak:.2f} | {b["host_peak_rss_bytes"]/1024**3:.2f} |')
    cache=read(out/'encoding_cache_complete.json')
    lines+=['',f'새 bins100·surface20 캐시 생성량 {cache["bytes"]/1024**3:.3f}GiB. '
        'count33/RGB는 검증된 기존 캐시의 외부 읽기전용 참조이며 복제 용량과 구분한다. '
        '각 세션 생성시간·SHA·원본archive SHA는 encoding_receipts/에 기록했다. '+
        f'동일 GPU UUID: `{next(iter(uuids))}`.', '',
        '## 사전 지정 통계 비교','',
        '전체11조건 공통 GT, 피험자별 cluster bootstrap10,000회·95%CI, 양측 cluster sign-flip 검정과15비교 Holm 보정이다. '
        'PA와 Root에 각각 독립적인15비교 family를 적용한다. 음수 효과는 비교모델의 오차 감소다. '
        '원본/짧은 RGB와 긴 event조건의 비교에는 학습시간 차이가 포함된다.', '',
        '| 비교 | frame-aligned PA 변화(mm) | 95%CI(mm) | Holm p | 상대변화 |','|---|---:|---|---:|---:|']
    for suffix in ('_pa','_root_position'):
        if suffix=='_root_position':lines+=['','| 비교 | 3D Root 변화(mm) | 95%CI(mm) | Holm p | 상대변화 |','|---|---:|---|---:|---:|']
        for a,b in PAIRS:
            r=c['tests'][a+'__'+b+suffix];ci=r['ci95'];effect=r['effect'];p=r.get('holm_p_value');relative=r.get('relative_percent')
            fmt=lambda v:'미평가' if v is None else f'{v:.4f}'
            lines.append(f'| {a} → {b} | {fmt(effect)} | '+('미평가' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]')+f' | {fmt(p)} | {fmt(relative)}% |')
    lines+=['','첫 프레임 PA는 frame0에서 얻은 변환을 고정하므로 이후 구간 평균에서 첫 프레임 MPJPE보다 클 수도 있다. '
        '매프레임 PA는 같은 표본의 매프레임 MPJPE보다 작거나 같은지 수치 검증한다. 두 값이 반드시2배 차이가 나야 하는 것은 아니다.', '',
        '## 각 조건 자체의 유효 Test 표본','',
        '주표의 공통GT 축소 영향을 확인하기 위한 보조값이다. 각 행의 표본이 달라 이 표로 공정한 순위를 주장하지 않는다.', '',
        '| 조건 | 자체 frame0 유효 표본 | 자체 frame-aligned PA(mm) | 자체 3D Root(mm) |','|---|---:|---:|---:|']
    for n in EVAL_NAMES:
        r=c['own_metrics'][n];fmt=lambda v:'미평가' if v is None else f'{v:.2f}'
        lines.append(f'| {LABELS[n]} | {r["samples"]:,} | {fmt(r["frame_aligned_pa_mpjpe_mm"])} | {fmt(r["pelvis_error_mm"])} |')
    lines+=['','## 동작속도별 frame-aligned PA-MPJPE','',
        'GT body22 중심차분 속도로 고정한 정지≤0.05m/s, 느림0.05~0.5m/s, 빠름≥0.5m/s다. GT jitter의 영향을 받을 수 있다.', '',
        '| 조건 | 정지(mm) | 느림(mm) | 빠름(mm) |','|---|---:|---:|---:|']
    for n in EVAL_NAMES:
        cells=[]
        for label in ('still','slow','fast'):
            r=c['motion_strata'][n][label];cells.append('미평가' if not r['count'] else f'{r["sum_mm"]/r["count"]:.2f}')
        lines.append('| '+LABELS[n]+' | '+' | '.join(cells)+' |')
    # Recommend on Train holdout only, never choose encodings from Test minima.
    eligible=[]
    for n in TRAIN_NAMES:
        v=read(out/'runs'/n/'best.json')
        if v['coverage']>=.95 and v['pa_mpjpe_mm'] is not None:eligible.append((v['pa_mpjpe_mm'],n))
    selected=min(eligible)[1] if eligible else 'rgb3'
    lines+=['','## 단일 추천과 한계','',
        f'이번 예산에서의 단일 추천은 **{LABELS[selected]}**이다. Test 최저값으로 고른 결과가 아니라 '
        '사전에 고정한 Train holdout 검출률95% 이상 조건의 PA 최솟값으로 선택했다. '
        'Test 효과·CI·Root 오차·검출률을 위 표에서 함께 판단해야 하며 개선이 없으면 개선으로 주장하지 않는다.',
        '한 seed의 비교다. 여러 독립 seed 반복을 완료한 실험으로 주장하지 않는다. '
        'Train6501프레임 고정 subset 결과이며 전체 Train을 한 epoch 학습한 것으로 해석하지 않는다. '
        '장치 대칭은 범위 축소 가정으로만 사용했으며 다른 장치 방향의 동일 성능을 입증하지 않는다. '
        '이전 공개 Test를 반복 평가한 후속 실험으로 완전히 맹검인 새 Test가 아니다.', '',
        '## 실제 예측 시각화','',
        'RGB·세 인코딩의 event3(위)/event4(아래) 표시와11조건 실제 예측 body22를 공통 카메라의 같은 축에서 비교한다. '
        '회색은 GT, 파랑은 예측이며 정렬하거나 mesh로 대체하지 않았다. GT는 평가·표시에만 사용한다.']
    for p in read(out/'visualization_manifest.json')['paths']:lines+=['',f'![실제 예측 비교]({p})']
    (out/'final_report_addendum.md').write_text('\n'.join(lines)+'\n')
    from report_device3_encoding import snapshot
    snapshot(g)
    report=Path(g['report']);text=report.read_text();numeric=[]
    for n in EVAL_NAMES:
        expected=[]
        for k in METRICS:
            value=c['metrics'][n][k];expected.append('미평가' if value is None else f'{value:.2f}')
        line=next(s for s in text.splitlines() if s.startswith('| '+LABELS[n]+' |'))
        assert [s.strip().replace('**','') for s in line.split('|')[2:-1]]==expected
        numeric.append(dict(name=n,values=expected))
    write(out/'report_numeric_verification.json',dict(status='passed',report_sha256=sha(report),rows=numeric))

def main():
    g=read(GOAL);out=Path(g['output_root']);register_job(g,os.getpid(),'encoding_finish_supervisor')
    while not (out/'all_training_test_complete.json').exists():
        if time.time()>g['deadline_unix']-600:return
        time.sleep(30)
    env=dict(os.environ,EVENTPOSE_GOAL_CONFIG=str(GOAL),EVENTPOSE_MULTISENSOR_OUT=str(out),EVENTPOSE_EVENT_OUT=str(out),
        OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    def run(role,cmd):
        with (out/'logs'/(role+'.log')).open('a') as log:
            p=subprocess.Popen([str(ROOT/'.venv-humanpose/bin/python'),'-u']+cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        register_job(g,p.pid,role);assert p.wait()==0,role
    run('encoding_final_raw_analysis',['tools/analyze_device3_encoding.py'])
    # All model jobs have finished; wait for an actually idle approved GPU.
    import pynvml as nv
    nv.nvmlInit();gpu=None
    while time.time()<g['deadline_unix']-900 and gpu is None:
        for r in read(out/'gpu_preflight.json')['records']:
            if r['status']!='passed':continue
            try:
                h=nv.nvmlDeviceGetHandleByUUID(r['uuid'])
                if nv.nvmlDeviceGetMinorNumber(h)==r['physical_gpu'] and not nv.nvmlDeviceGetComputeRunningProcesses(h):gpu=r;break
            except nv.NVMLError:continue
        if gpu is None:time.sleep(30)
    assert gpu is not None,'No idle same-GPU timing device before deadline'
    run('encoding_sequential_speed',['tools/benchmark_fusion_repair.py','supervise','--gpu',str(gpu['physical_gpu'])])
    os.environ.update({k:v for k,v in env.items() if k.startswith('EVENTPOSE_')})
    audit_checkpoints(g);figures(g);final_addendum(g)
    write(out/'ready_for_agent_review.json',dict(status='automated_checks_passed',goal_complete=False,
        actual_visual_review_required=True,all_Test_complete=True,checkpoint_audit='passed',report_numeric_audit='passed',
        report_sha256=sha(g['report']),prepared_unix=time.time()))
if __name__=='__main__':main()

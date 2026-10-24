"""Compare completed SMPL-X methods on the same frozen Test population."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .humanpose_metrics import joint_errors
from .temporal_metrics import temporal_errors
from .report_format import highlight_table_bests
from .chunk_alignment import aligned_errors,METRICS

ROOT=Path(__file__).resolve().parents[2]
NLF=ROOT/'exp/261024/nlf_finetune_12h'
MULTI=ROOT/'exp/261023/multihmr_finetune_14h'
SOURCE=ROOT/'.experiment_storage/humanpose_gt_segments'

def write_report(result):
    """Write the user-facing comparison in Korean without changing measurements."""
    methods=result['methods']
    labels={'Multi-HMR pretrained':'Multi-HMR 기본 모델','NLF pretrained':'NLF 기본 모델',
            'Multi-HMR fine-tuned':'Multi-HMR 학습 모델','NLF fine-tuned':'NLF 학습 모델'}
    lines=['# NLF / Multi-HMR Test 비교','',
        f"Test 64세션·device0, 10초 × 5개 고정 구간으로 총 320구간·{result['rgb_frames']:,} RGB 프레임을 평가했다. "
        f"아래 신체 22관절 지표는 모든 모델에 공통 매칭된 {result['common_matched_instances']:,} 사람-프레임을 사용한다. "
        '오차 단위는 mm이며 낮을수록 좋다. 미검출은 0 오차로 처리하지 않고 검출률로 따로 보고한다.','',
        '| 모델 | 첫 프레임 정렬 MPJPE | 첫 프레임 정렬 PA-MPJPE | 프레임별 정렬 MPJPE | 프레임별 정렬 PA-MPJPE |',
        '|---|---:|---:|---:|---:|']
    for name,m in methods.items():lines.append('| '+labels.get(name,name)+' | '+' | '.join(f'{m[k]:.2f}' for k in METRICS)+' |')
    lines+=['','각 구간의 정확한 첫 프레임에서 모든 모델에 공통 검출된 사람만 포함했다. 첫 프레임 정렬은 '
        '그 변환을 10초 전체에 고정하고, 프레임별 정렬은 매 프레임 다시 계산한다. MPJPE 정렬은 pelvis 평행이동만 사용하며, '
        'PA는 반사를 허용하지 않는 회전·양의 등방 스케일·평행이동을 사용한다. PA는 스케일 오차를 제거하므로 '
        'metric scale의 절대 위치 정확도는 아래 정렬 전 오차로 판단해야 한다.','',
        '## 절대 위치·검출률·관측 처리 속도','',
        '| 모델 | 정렬 전 MPJPE (mm) | Pelvis 위치 오차 (mm) | GT 매칭률 (coverage) | 관측 fps |','|---|---:|---:|---:|---:|']
    for name,m in methods.items():lines.append(f"| {labels.get(name,name)} | {m['mpjpe_mm']:.2f} | {m['pelvis_error_mm']:.2f} | {100*m['coverage']:.3f}% | {m['observed_segment_fps']:.2f} |")
    lines+=['','관측 fps는 영상 디코딩·모델 실행·관절 복원·GT 매칭을 포함하지만 최초 모델 로딩과 영상의 SSD 복사는 제외한다. '
        '다른 GPU 작업과 동시에 측정됐으므로 이 값만으로 공정한 속도 배수를 단정하지 않는다.','',
        '## 움직임 일관성','',
        '| 모델 | Pelvis 속도 오차 (mm/s) | Root 상대 가속도 오차 (mm/s²) |','|---|---:|---:|']
    for name,m in methods.items():lines.append(f"| {labels.get(name,name)} | {m['pelvis_velocity_error_mmps']:.2f} | {m['root_acceleration_error_mmps2']:.2f} |")
    lines+=['','모든 모델이 공통 매칭한 동일 인물의 연속 원본 프레임만 사용한다. 누락된 프레임이나 서로 다른 구간 사이를 연결하지 않는다. '
        '시간 간격은 원본 프레임 번호와 영상 fps로 계산하며 smoothing은 적용하지 않았다. GT 자체의 시간적 잡음도 영향을 줄 수 있다.','',
        '## 전신 55관절 보조 지표','',
        '| 모델 | 정렬 전 MPJPE (mm) | Root 정렬 MPJPE (mm) | PA-MPJPE (mm) |','|---|---:|---:|---:|']
    for name,m in methods.items():lines.append('| '+labels.get(name,name)+' | '+' | '.join(f"{m['body55'][k]:.2f}" for k in ['mpjpe_mm','root_mpjpe_mm','pa_mpjpe_mm'])+' |')
    if 'controlled_runtime' in result:
        lines+=['','## 동일 GPU 순차 처리 속도','',
            '| 모델 | 전체 처리 fps | 모델 로딩 (초) | 프로세스 전체 시간 (초) | 최대 VRAM (GiB) |','|---|---:|---:|---:|---:|']
        for name,b in result['controlled_runtime'].items():
            memory=max(v['peak_allocated_bytes'] for v in b['segments'])/1024**3
            lines.append(f"| {name} | {b['fps']:.2f} | {b['load_seconds']:.2f} | {b['cold_process_wall_seconds']:.2f} | {memory:.2f} |")
        lines+=['','GPU 0에서 동일한 100프레임 구간 3개를 모델별로 순차 처리했다. 전체 처리 시간에는 영상 복사와 결과 저장을 포함하며, '
            '프로세스 시간에는 import와 모델 로딩도 포함한다. 짧은 측정이므로 캐시 상태와 초기 실행 비용의 영향을 받을 수 있다.']
    lines+=['','## 종합 판단','']
    if 'NLF fine-tuned' in methods and 'controlled_runtime' in result:
        keys={'primary':'pa_mpjpe_mm','absolute_placement':'mpjpe_mm','root_acceleration':'root_acceleration_error_mmps2'}
        recommendation={k:min(methods,key=lambda n:methods[n][metric]) for k,metric in keys.items()}
        recommendation['detection_coverage']=max(methods,key=lambda n:methods[n]['coverage'])
        recommendation['throughput']=max(result['controlled_runtime'],key=lambda n:result['controlled_runtime'][n]['fps'])
        recommendation['policy']='프레임별 PA-MPJPE를 우선하고 절대 위치·검출률·가속도·처리 속도의 차이를 별도로 비교한다.'
        result['recommendation']=recommendation
        lines += [f"**우선 추천: {labels.get(recommendation['primary'],recommendation['primary'])}.** 공통 Test의 프레임별 PA-MPJPE가 가장 낮다. "
            '다른 항목까지 모두 우수하다는 뜻은 아니며 아래 장단점을 함께 확인해야 한다.','',
            '| 비교 항목 | 가장 좋은 모델 |','|---|---|']
        for title,k in [('프레임별 PA-MPJPE','primary'),('절대 위치','absolute_placement'),('검출률','detection_coverage'),('Root 상대 가속도','root_acceleration'),('전체 처리 속도','throughput')]:
            lines.append(f"| {title} | {labels.get(recommendation[k],recommendation[k])} |")
    else:
        lines+=['현재 완료된 결과에서는 학습한 Multi-HMR을 우선 추천한다. NLF 학습 후 전체 Test와 동일 GPU 속도 측정이 아직 남아 있으므로 '
            '최종 종합 추천은 해당 결과까지 확인한 뒤 확정한다. NLF 기본 모델은 Multi-HMR 기본 모델보다 자세·절대 위치 오차가 낮다.']
    lines+=['','## 결과 출처와 평가 범위','',
        'Train과 Test는 피험자가 겹치지 않으며 Test는 가중치 업데이트나 체크포인트 선택에 사용하지 않았다. '
        '학습 모델의 사용 체크포인트는 아래 결과 경로로 구분한다. 표의 모델명에는 체크포인트 단계 표시를 붙이지 않는다.']
    for name,source in result.get('method_sources',{}).items():lines.append(f"- {labels.get(name,name)}: `{source}`")
    lines+=['',f"원시 집계: `{NLF.relative_to(ROOT)}/comparison.json`. 모델별 공통 평가 인구가 바뀌면 별도 보고서의 수치와 직접 비교하지 않는다."]
    (NLF/'comparison.json').write_text(json.dumps(result,indent=2))
    (ROOT/'_agents/report/261024_인체자세추정_모델_성능_비교.md').write_text('\n'.join(highlight_table_bests(lines))+'\n')

def main():
    tasks=json.loads((SOURCE/'tasks.json').read_text())
    manifest=json.loads((MULTI/'manifest.json').read_text())
    assert len(tasks)==320 and len({t['parent_session'] for t in tasks})==64
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['test_split_sha256']
    candidates={'Multi-HMR pretrained':SOURCE/'multi_hmr',
                'NLF pretrained':ROOT/'.experiment_storage/nlf_finetune_12h/nlf_pretrained',
                'NLF fine-tuned':ROOT/'.experiment_storage/nlf_finetune_12h/nlf_finetuned'}
    final=ROOT/'.experiment_storage/multihmr_finetune_14h/final_test/multihmr_finetuned'
    if all((final/t['session']/'complete.json').exists() for t in tasks):
        candidates['Multi-HMR fine-tuned']=final
    else:
        candidates['Multi-HMR fine-tuned']=ROOT/'.experiment_storage/multihmr_finetune_14h/test_snapshots/20261024_040315/multihmr_finetuned'
    methods={name:path for name,path in candidates.items()
             if all((path/t['session']/'complete.json').exists() for t in tasks)}
    assert 'NLF pretrained' in methods,'Wait until every NLF baseline segment finishes'
    sums={name:np.zeros(4,np.float64) for name in methods}
    alignment_sums={name:np.zeros(4,np.float64) for name in methods}
    full_body_sums={name:np.zeros(3,np.float64) for name in methods}
    coverage={name:0 for name in methods};elapsed={name:0. for name in methods}
    temporal={name:np.zeros(4,np.float64) for name in methods}
    valid_total=paired_total=anchor_people=0
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt=z['joints_cam_m'];valid=z['valid'];positions=z['source_rgb_positions']
            assert np.array_equal(z['original_source_rgb_positions'],positions+task['start_frame'])
        predictions={};masks={}
        for name,path in methods.items():
            folder=path/task['session']
            complete=json.loads((folder/'complete.json').read_text())
            assert complete['task']['video_frames']==task['end_frame']-task['start_frame']
            assert not complete['task'].get('smoke')
            elapsed[name]+=complete['elapsed_seconds']
            with np.load(folder/'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'],positions)
                prediction=z['matched_joints_cam_m'];assert prediction.shape==gt.shape
                mask=valid&np.isfinite(prediction).all((2,3))
                assert np.array_equal(mask,valid&(z['matched_pred_indices']>=0))
                predictions[name]=prediction;masks[name]=mask;coverage[name]+=int(mask.sum())
        common=np.logical_and.reduce(list(masks.values()))
        first=np.flatnonzero(positions==0)
        eligible=common[first[0]].copy() if len(first) else np.zeros(gt.shape[1],bool)
        common&=eligible[None,:];paired_total+=int(common.sum());anchor_people+=int(eligible.sum())
        valid_total+=int(valid.sum())
        for name,prediction in predictions.items():
            for person in np.flatnonzero(eligible):
                keep=common[:,person]
                alignment_sums[name]+=aligned_errors(prediction[keep,person,:22],gt[keep,person,:22],
                    prediction[first[0],person,:22],gt[first[0],person,:22]).sum(0)
            temporal[name]+=temporal_errors(prediction,gt,common,positions,task['fps'])
            full_p=prediction[common].astype(np.float64);full_t=gt[common].astype(np.float64)
            full_errors=np.array([joint_errors(a,b) for a,b in zip(full_p,full_t)]).reshape(-1,3)
            assert np.isfinite(full_errors).all()
            full_body_sums[name]+=full_errors.sum(0)
            p=prediction[common][:,:22].astype(np.float64);t=gt[common][:,:22].astype(np.float64)
            errors=np.array([joint_errors(a,b) for a,b in zip(p,t)]).reshape(-1,3)
            assert np.isfinite(errors).all()
            sums[name][:3]+=errors.sum(0)
            sums[name][3]+=np.linalg.norm(p[:,0]-t[:,0],axis=-1).sum()
    assert paired_total>0
    frames=sum(t['video_frames'] for t in tasks)
    result=dict(updated_unix=time.time(),test_sessions=64,segments=320,rgb_frames=frames,
                valid_instances=valid_total,common_matched_instances=paired_total,eligible_person_chunks=anchor_people,unit='mm',methods={})
    result['method_sources']={name:str(path) for name,path in methods.items()}
    benchmark=NLF/'runtime_benchmark/summary.json'
    if benchmark.exists():result['controlled_runtime']=json.loads(benchmark.read_text())
    for name in methods:
        result['methods'][name]=dict(zip(['mpjpe_mm','root_mpjpe_mm','pa_mpjpe_mm','pelvis_error_mm'],
            (1000*sums[name]/paired_total).tolist()))
        assert np.allclose(alignment_sums[name][2:],sums[name][1:3],rtol=1e-8,atol=1e-8)
        result['methods'][name].update(dict(zip(METRICS,(1000*alignment_sums[name]/paired_total).tolist())))
        result['methods'][name].update(matched_instances=coverage[name],coverage=coverage[name]/valid_total,
            sum_segment_seconds=elapsed[name],observed_segment_fps=frames/elapsed[name])
        vs,vc,acs,ac=temporal[name]
        result['methods'][name].update(pelvis_velocity_error_mmps=1000*vs/vc if vc else None,
            root_acceleration_error_mmps2=1000*acs/ac if ac else None,
            velocity_pairs=int(vc),acceleration_triples=int(ac))
        result['methods'][name]['body55']=dict(zip(['mpjpe_mm','root_mpjpe_mm','pa_mpjpe_mm'],
            (1000*full_body_sums[name]/paired_total).tolist()))
    (NLF/'comparison.json').write_text(json.dumps(result,indent=2))
    write_report(result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

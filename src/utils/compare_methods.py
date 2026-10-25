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

def write_speed_report(result):
    """Keep the user's original speed-report path current with controlled results."""
    if len(result['methods'])!=4 or len(result.get('controlled_runtime',{}))!=4:return
    labels={'Multi-HMR pretrained':'Multi-HMR 기본 모델','NLF pretrained':'NLF 기본 모델',
            'Multi-HMR fine-tuned':'Multi-HMR 학습 모델','NLF fine-tuned':'NLF 학습 모델'}
    runtime_keys={'Multi-HMR pretrained':'multihmr_pretrained','NLF pretrained':'nlf_pretrained',
                  'Multi-HMR fine-tuned':'multihmr_finetuned','NLF fine-tuned':'nlf_finetuned'}
    order=['Multi-HMR pretrained','NLF pretrained','Multi-HMR fine-tuned','NLF fine-tuned']
    lines=['# 학습과 추론 속도 최종 비교','',
        '**기존 RGB 4개 모델의 전체 Test와 동일 GPU 순차 속도 측정이 완료된 결과로 갱신했다.** '
        '이 파일명에 남아 있는 중간 표기는 기존 링크 유지를 위한 것이다. RGB/event 확장 실험의 결과는 아직 포함하지 않는다.','',
        f"실제 Test 64세션·320구간·{result['rgb_frames']:,} RGB 프레임이다. 자세 오차는 네 모델 공통 {result['common_matched_instances']:,} 사람-프레임, "
        '검출률은 전체 유효 GT를 분모로 사용한다. 오차 단위는 mm이고 낮을수록 좋으며 검출률과 fps는 높을수록 좋다.','',
        '## 정확도와 동일 GPU 전체 처리 속도','',
        '| 모델 | 프레임별 PA-MPJPE (mm) | 절대 MPJPE (mm) | GT 매칭률 (coverage) | 전체 처리 fps |',
        '|---|---:|---:|---:|---:|']
    for name in order:
        m=result['methods'][name];b=result['controlled_runtime'][runtime_keys[name]]
        lines.append(f"| {labels[name]} | {m['pa_mpjpe_mm']:.2f} | {m['mpjpe_mm']:.2f} | {100*m['coverage']:.3f}% | {b['fps']:.3f} |")
    gpu=next(iter(result['controlled_runtime'].values()))['gpu_identity']
    lines+=['',f"측정 GPU: `{gpu}`. 동일한 100프레임 구간 3개를 각 모델이 순차 처리했다. 영상 복사·디코드·전처리·검출·인체 복원·GT 매칭·결과 저장을 포함하며, 모델 로딩은 아래 별도 시간으로 분리했다. "
        '이전 동시 작업에서 관측한 7.21/5.12/4.97 fps는 이번 공정한 순차 측정값으로 교체했다. 짧은 측정이므로 초기 실행 및 캐시 영향은 남는다.','',
        '## 로딩과 메모리','',
        '| 모델 | 모델 로딩 (초) | 프로세스 전체 시간 (초) | 최대 PyTorch 할당 VRAM (GiB) |','|---|---:|---:|---:|']
    for name in order:
        b=result['controlled_runtime'][runtime_keys[name]]
        memory=max(segment['peak_allocated_bytes'] for segment in b['segments'])/1024**3
        lines.append(f"| {labels[name]} | {b['load_seconds']:.2f} | {b['cold_process_wall_seconds']:.2f} | {memory:.2f} |")
    lines+=['','프로세스 전체 시간에는 import와 모델 로딩도 포함한다. VRAM은 PyTorch 할당량으로 드라이버 전체 점유량과 다르다. Multi-HMR 배치는32, NLF 배치는16으로 기존 파이프라인 설정을 유지했다.','',
        '## 학습 속도','',
        '설정당 GPU2개·GPU당 배치16의 완료 기록을 사용했다. 검증·저장 등이 포함된 누적 성공 업데이트당 평균이다. NLF는 인물 crop, Multi-HMR은 여러 인물이 포함된 전체 프레임이므로 두 처리량을 동등 작업의 배속으로 해석하지 않는다.','']
    for family,root,names in [('Multi-HMR',MULTI,['w1_head','w1_last4','w1_last8','w2_refine']),
                              ('NLF',NLF,['head','last_stage','last_two_stages'])]:
        unit='전체 프레임/초' if family=='Multi-HMR' else '인물 crop/초'
        lines += [f"### {family}",'',f'| 설정 | 성공 업데이트당 초 | {unit} |','|---|---:|---:|']
        for name in names:
            value=json.loads((root/'runs'/name/'complete.json').read_text())
            seconds=value['elapsed_seconds'];steps=value['successful_steps']
            lines.append(f"| {name} | {seconds/steps:.3f} | {32*steps/seconds:.2f} |")
        lines.append('')
    lines+=['## 추천과 한계','',
        '**우선 추천은 Multi-HMR 학습 모델이다.** 자세·절대 위치·검출률이 가장 좋고, 기본 Multi-HMR과 거의 같은 전체 처리 속도를 유지했다. '
        'NLF 학습 모델은 이번 Test에서 기본 NLF보다도 자세 오차가 커졌으므로 학습 내부 검증 개선을 Test 개선으로 해석하지 않는다. '
        '시간적 가속도 오차는 기본 Multi-HMR이 더 낮은 항목이 있어 모든 지표에서 우세한 것은 아니다.','',
        '첫 frame0 고정 정렬/프레임별 정렬의 네 지표와 움직임 일관성은 [전체 모델 비교 보고서]'
        '(/home/namhj/EventPose/_agents/report/261024_인체자세추정_모델_성능_비교.md)에 함께 정리했다.','',
        '원시 결과: `exp/261024/nlf_finetune_12h/comparison.json`, `runtime_benchmark/summary.json`. '
        '전체1280개 결과 재집계 및 보고서 숫자 검증 기록: `final_audit.json`. '
        'Train 내부 검증을 체크포인트 선택에 사용했으며 Test를 사용하지 않았다. 학습 범위와 체크포인트 출처는 전체 비교 보고서와 각 실험 selected.json을 따른다.']
    (ROOT/'_agents/report/261024_학습과_추론_속도_중간_비교.md').write_text('\n'.join(highlight_table_bests(lines))+'\n')

def write_report(result):
    """Write the user-facing comparison in Korean without changing measurements."""
    methods=result['methods']
    labels={'Multi-HMR pretrained':'Multi-HMR 기본 모델','NLF pretrained':'NLF 기본 모델',
            'Multi-HMR fine-tuned':'Multi-HMR 학습 모델','NLF fine-tuned':'NLF 학습 모델'}
    runtime_labels={'multihmr_pretrained':labels['Multi-HMR pretrained'],
                    'nlf_pretrained':labels['NLF pretrained'],
                    'multihmr_finetuned':labels['Multi-HMR fine-tuned'],
                    'nlf_finetuned':labels['NLF fine-tuned']}
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
            '| 모델 | 전체 처리 fps | 모델 로딩 (초) | 프로세스 전체 시간 (초) | 최대 PyTorch 할당 VRAM (GiB) |','|---|---:|---:|---:|---:|']
        for name,b in result['controlled_runtime'].items():
            memory=max(v['peak_allocated_bytes'] for v in b['segments'])/1024**3
            lines.append(f"| {runtime_labels.get(name,name)} | {b['fps']:.2f} | {b['load_seconds']:.2f} | {b['cold_process_wall_seconds']:.2f} | {memory:.2f} |")
        gpu_ids={b['cuda_visible_devices'] for b in result['controlled_runtime'].values()}
        assert len(gpu_ids)==1, 'Controlled runtime must use the same physical GPU'
        lines+=['',f"GPU {next(iter(gpu_ids))}에서 동일한 100프레임 구간 3개를 모델별로 순차 처리했다. 전체 처리 시간에는 영상 복사와 결과 저장을 포함하며, "
            '프로세스 시간에는 import와 모델 로딩도 포함한다. Multi-HMR 배치는 32프레임, NLF 배치는 16프레임으로 각 평가 파이프라인 설정을 유지했다. '
            'VRAM은 PyTorch 할당량이며 드라이버의 전체 점유량과 다르다. 짧은 측정이므로 캐시 상태와 초기 실행 비용의 영향을 받을 수 있다.']
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
            lines.append(f"| {title} | {labels.get(recommendation[k],runtime_labels.get(recommendation[k],recommendation[k]))} |")
        runtime_keys={'Multi-HMR pretrained':'multihmr_pretrained','NLF pretrained':'nlf_pretrained',
                      'Multi-HMR fine-tuned':'multihmr_finetuned','NLF fine-tuned':'nlf_finetuned'}
        chosen=recommendation['primary'];winner=methods[chosen]
        winner_fps=result['controlled_runtime'][runtime_keys[chosen]]['fps']
        lines+=['','추천 모델과 다른 모델의 실제 차이는 다음과 같다. 오차는 감소, 검출률과 fps는 증가할수록 유리하다.','']
        def change(value,reference,unit,percent=False):
            difference=100*(value/reference-1) if percent else value-reference
            return f"{abs(difference):.2f}{unit} {'증가' if difference>0 else '감소' if difference<0 else '동일'}"
        for name,m in methods.items():
            if name==chosen:continue
            other_fps=result['controlled_runtime'][runtime_keys[name]]['fps']
            lines.append(f"- {labels[name]} 대비: PA 오차 {change(winner['pa_mpjpe_mm'],m['pa_mpjpe_mm'],'%',True)}, "
                f"절대 MPJPE {change(winner['mpjpe_mm'],m['mpjpe_mm'],' mm')}, "
                f"검출률 {change(100*winner['coverage'],100*m['coverage'],'%p')}, "
                f"Root 상대 가속도 오차 {change(winner['root_acceleration_error_mmps2'],m['root_acceleration_error_mmps2'],'%',True)}, "
                f"전체 처리 fps {change(winner_fps,other_fps,'%',True)}.")
    else:
        lines+=['현재 완료된 결과에서는 학습한 Multi-HMR을 우선 추천한다. NLF 학습 후 전체 Test와 동일 GPU 속도 측정이 아직 남아 있으므로 '
            '최종 종합 추천은 해당 결과까지 확인한 뒤 확정한다. NLF 기본 모델은 Multi-HMR 기본 모델보다 자세·절대 위치 오차가 낮다.']
    lines+=['','## 결과 출처와 평가 범위','',
        'Train과 Test는 피험자가 겹치지 않으며 Test는 가중치 업데이트나 체크포인트 선택에 사용하지 않았다. '
        'Train 219세션·68명 중 187세션은 가중치 학습, 32세션은 내부 holdout 모델 선택에 사용했다. '
        'Test는 64세션·20명이다. '
        '학습 모델의 사용 체크포인트는 아래 결과 경로로 구분한다. 표의 모델명에는 체크포인트 단계 표시를 붙이지 않는다.']
    lines+=['','원래 작업 범위는 2026-10-24 05:19:25–17:19:25 UTC이며 새 12시간으로 재계산하지 않았다. '
        'NLF 학습 종료 기준은 13:47:05 UTC로 유지했다. 최종 Test는 import 오류 수정 후 GPU 접근 환경이 복구된 17:07 UTC에 재개했다.']
    if result['updated_unix']>1792862365:
        lines+=['GPU 실행 환경 제약과 평가 오류 복구로 최종 평가·속도 비교 완료는 원래 종료 목표 시각을 넘겼다. '
            '이 시간 초과를 추가 학습 시간으로 사용하지 않았다.']
    for name,source in result.get('method_sources',{}).items():lines.append(f"- {labels.get(name,name)}: `{source}`")
    lines+=['',f"원시 집계: `{NLF.relative_to(ROOT)}/comparison.json`. 모델별 공통 평가 인구가 바뀌면 별도 보고서의 수치와 직접 비교하지 않는다."]
    (NLF/'comparison.json').write_text(json.dumps(result,indent=2))
    (ROOT/'_agents/report/261024_인체자세추정_모델_성능_비교.md').write_text('\n'.join(highlight_table_bests(lines))+'\n')
    write_speed_report(result)

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

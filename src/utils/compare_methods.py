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
    lines=['# NLF / Multi-HMR Test comparison','',
        f'All 64 device0 sessions, 320 fixed segments, and {frames:,} RGB frames. '
        f'Body22 errors use the same {paired_total:,} person-frames matched by every listed method. '
        'All pose errors are mm; lower is better. Missing detections are reported through coverage, not zero errors.','',
        'Only person-chunks matched at exact frame 0 by every listed model enter this common population. '
        'First-frame transforms remain fixed for all 10 seconds; frame alignment is recomputed independently.','',
        '| Model | First frame aligned MPJPE | First frame aligned PA-MPJPE | Frame aligned MPJPE | Frame aligned PA-MPJPE |',
        '|---|---:|---:|---:|---:|']
    for name,m in result['methods'].items():
        lines.append('| '+name+' | '+' | '.join(f'{m[k]:.2f}' for k in METRICS)+' |')
    multi_source=methods.get('Multi-HMR fine-tuned')
    if multi_source is not None:
        stage='final selected checkpoint' if multi_source==final else 'training-intermediate checkpoint'
        lines+=['',f'Multi-HMR fine-tuned results use a {stage}, evaluated on all 320 Test segments. '
            f'Prediction source: `{multi_source.relative_to(ROOT)}`.']
    lines+=['','MPJPE alignment uses pelvis translation. PA alignment uses proper Sim(3), without reflection.','',
        '## Absolute placement, coverage, and observed throughput','',
        '| Model | Absolute MPJPE (mm) | Pelvis position (mm) | Matched GT | Observed fps |','|---|---:|---:|---:|---:|']
    for name,m in result['methods'].items():
        lines.append(f'| {name} | {m["mpjpe_mm"]:.2f} | {m["pelvis_error_mm"]:.2f} | '
                     f'{100*m["coverage"]:.3f}% | {m["observed_segment_fps"]:.2f} |')
    lines+=['','Observed fps includes clip decoding, model execution, joint recovery, and matching, '
        'but excludes initial checkpoint loading and session video staging. Runs can overlap with other GPU jobs; '
        'these figures do not establish a controlled hardware speed ratio. A separate isolated end-to-end benchmark is required.',
        '', 'PA aligns each person-frame with proper Sim(3), removing scale error. Absolute MPJPE and pelvis position '
        'therefore determine metric placement quality. Test is not used for checkpoint selection.',
        '', 'Only complete methods are listed. NLF fine-tuning remains pending until its final Test is complete. '
        'Exact source directories are recorded in comparison.json under method_sources.']
    lines+=['','## Motion consistency','',
        '| Model | Pelvis velocity error (mm/s) | Root-relative body22 acceleration error (mm/s²) |',
        '|---|---:|---:|']
    for name,m in result['methods'].items():
        lines.append(f'| {name} | {m["pelvis_velocity_error_mmps"]:.2f} | {m["root_acceleration_error_mmps2"]:.2f} |')
    lines+=['','Temporal metrics require the same GT person to be matched by every listed model in '
        'consecutive source frames. They never bridge missing frames or segment boundaries. '
        'Frame intervals use source frame numbers and nominal video fps. These measure agreement '
        'with the dataset motion labels, whose own temporal noise may affect the values. No smoothing is applied.']
    report=ROOT/'_agents/report/nlf_multihmr_comparison.md';report.parent.mkdir(parents=True,exist_ok=True)
    lines+=['','## Whole-body 55-joint diagnostic','',
        '| Model | MPJPE (mm) | Root MPJPE (mm) | PA-MPJPE (mm) |','|---|---:|---:|---:|']
    for name,m in result['methods'].items():
        b=m['body55'];lines.append(f'| {name} | {b["mpjpe_mm"]:.2f} | {b["root_mpjpe_mm"]:.2f} | {b["pa_mpjpe_mm"]:.2f} |')
    if 'controlled_runtime' in result:
        lines+=['','## Sequential single-GPU runtime study','',
            '| Model | Pipeline fps | Model load (s) | Cold process wall (s) | Peak allocated VRAM (GiB) |',
            '|---|---:|---:|---:|---:|']
        for name,b in result['controlled_runtime'].items():
            memory=max(s['peak_allocated_bytes'] for s in b['segments'])/1024**3
            lines.append(f'| {name} | {b["fps"]:.2f} | {b["load_seconds"]:.2f} | '
                         f'{b["cold_process_wall_seconds"]:.2f} | {memory:.2f} |')
        lines+=['','Each model runs sequentially on GPU 0 over the same three 100-frame Test excerpts, '
            'selected to cover different person counts. Pipeline time includes video staging and result storage. '
            'Cold process time additionally includes imports and model loading. OS/NFS cache warmth and '
            'first-call compilation can affect these short-run measurements; full-Test accuracy uses all 320 segments.']
    if 'NLF fine-tuned' in methods and 'controlled_runtime' in result:
        pose_best=min(result['methods'],key=lambda n:result['methods'][n]['pa_mpjpe_mm'])
        placement_best=min(result['methods'],key=lambda n:result['methods'][n]['mpjpe_mm'])
        coverage_best=max(result['methods'],key=lambda n:result['methods'][n]['coverage'])
        motion_best=min(result['methods'],key=lambda n:result['methods'][n]['root_acceleration_error_mmps2'])
        runtime_best=max(result['controlled_runtime'],key=lambda n:result['controlled_runtime'][n]['fps'])
        result['recommendation']=dict(primary=pose_best,absolute_placement=placement_best,
            detection_coverage=coverage_best,root_acceleration=motion_best,throughput=runtime_best,
            policy='Prioritize the user-specified Root-aligned PA-MPJPE; report other dimensions as explicit tradeoffs, without an arbitrary weighted score.')
        lines+=['','## Recommendation','',f'**Primary recommendation: {pose_best}.** '
            'It has the lowest Root-aligned PA-MPJPE on the common Test population, the predefined primary metric. '
            'This does not imply superiority on every criterion.','',
            '| Criterion | Best observed method |','|---|---|',
            f'| Root-aligned PA-MPJPE | {pose_best} |',f'| Absolute camera-space MPJPE | {placement_best} |',
            f'| Detection/matching coverage | {coverage_best} |',f'| Root-relative acceleration error | {motion_best} |',
            f'| Controlled pipeline throughput | {runtime_best} |','',
            'Use the absolute-position and coverage columns when metric placement or missed people are decisive. '
            'Use the sequential runtime study when processing budget is decisive. The primary recommendation follows '
            'the requested pose metric rather than an invented weighted composite score.']
    else:lines+=['','## Recommendation status','',
        'This comparison is interim. A final overall recommendation requires both the fine-tuned NLF Test '
        'and the controlled runtime study. No missing method is assumed to outperform a completed one.']
    (NLF/'comparison.json').write_text(json.dumps(result,indent=2))
    report.write_text('\n'.join(highlight_table_bests(lines))+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

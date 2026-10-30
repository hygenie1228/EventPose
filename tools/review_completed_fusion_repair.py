"""Final requirement audit after a human agent opens the actual figures."""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
from goal_runtime_control import read, write, identity, register_job
from fusion_repair_scope import TRAIN_NAMES, EVAL_NAMES, FUSION_NAMES, PRIMARY_POPULATION, PAIRS
from render_fusion_repair import verify_tables
from multihmr.multisensor_repair import implementation_provenance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-reviewed', action='store_true', required=True)
    parser.parse_args()
    goal = read(ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json')
    register_job(goal, os.getpid(), 'fusion_repair_completion_agent_review')
    out = Path(goal['output_root']); audit = read(out / 'final_artifact_audit.json')
    report = Path(goal['report']); text = report.read_text()
    assert time.time() < goal['deadline_unix']
    assert audit['all_nine_full_test'] and audit['all_seven_training_complete']
    assert audit['status'] == 'passed_artifact_audit_agent_review_required'
    assert sha(report) == audit['report_sha256']
    manifest = read(out / 'manifest.json')
    assert sha(ROOT / 'dataset/split.json') == manifest['split_sha256']
    sets = {k: set(v) for k, v in read(out / 'body_holdout_subject_audit.json')['subjects'].items()}
    assert [len(sets[k]) for k in ('training', 'validation', 'test')] == [51, 9, 18]
    assert all(not sets[x] & sets[y] for x, y in [('training', 'validation'), ('training', 'test'), ('validation', 'test')])
    for role in ('training', 'validation', 'test'):
        assert all(s.endswith('_b') for s in manifest[role + '_sessions'])
    index = read(out / 'training_index.json'); counts = collections.Counter(r['session'] for r in index)
    assert len(index) == 6501 and set(counts) == set(manifest['training_sessions'])
    assert max(counts.values()) == 128 and min(counts.values()) > 0
    inputs = read(out / 'actual_input_audit.json')
    assert inputs['sessions'] == 78 and inputs['independent_cpu_GT_samples'] == 54
    assert all(inputs[k] for k in ['all_frame_mappings_verified', 'all_event_causality_verified', 'all_relative_calibrations_verified'])
    reuse = read(out / 'source_reuse_audit.json')
    assert not reuse['overlap'] and reuse['old_results_relabelled_as_new'] is False
    assert len(reuse['cache_records']) == 78 and reuse['split_sha256'] == manifest['split_sha256']
    for row in reuse['cache_records']:
        stat = (out / 'cache' / (row['session'] + '.h5')).stat()
        assert stat.st_size == row['bytes'] and stat.st_mtime <= audit['checked_unix']
    archive_count = 0
    for row in read(out / 'current_dataset_audit.json')['records']:
        assert row['all_cached_GT_parameters_equal_current_source']
        assert row['corrected_clocks_and_calibrations_equal_current_source']
        for archive in row['source_archives']:
            stat = Path(archive['path']).stat()
            assert stat.st_size == archive['bytes'] and stat.st_mtime_ns == archive['mtime_ns']
            archive_count += 1
    for row in read(out / 'reused_groundtruth_hashes.json')['records']:
        assert Path(row['path']).stat().st_mtime <= audit['checked_unix']
    print('data_scope_and_post_audit_immutability: passed', flush=True)
    assert len(audit['checkpoints']) == 16
    for name in TRAIN_NAMES:
        assert {r['kind'] for r in audit['checkpoints'] if r['name'] == name} == {'best', 'last'}
        complete = read(out / 'runs' / name / 'complete.json'); started = read(out / 'runs' / name / 'started.json')
        assert complete['status'] == 'complete' and 0 < complete['elapsed_seconds'] < 14400
        assert complete['samples_seen'] == complete['step'] * 16 and complete['unique_frames_seen'] == 6501
        assert started['test_used_for_selection'] is False
        seconds = 5400 if name in ('rgb3_short90', 'rgb4') else 10800
        assert abs(started['finish_unix'] - started['started_unix'] - seconds) < .01
    for row in audit['checkpoints']:
        stat = Path(row['path']).stat(); assert stat.st_size > 0 and stat.st_mtime <= audit['checked_unix']
        if row['kind'] in ('best', 'last'):
            assert row['tensors_finite'] and row['configuration_equal'] and not row['test_used_for_selection']
    approved = read(out / 'source_revision_scope_extension.json')['approved_implementations']
    assert implementation_provenance() == approved[-1]
    assert len(audit['execution_proofs']) == 9
    for execution in audit['execution_proofs']:
        name = Path(execution['checkpoint']).parent.name
        config = read(out / 'configs' / (name + '.json'))
        assert name in EVAL_NAMES and execution['GT_used_as_model_input'] is False
        assert int(execution['physical_gpu']) in goal['allowed_gpus']
        assert set(execution['pixel_datasets_read']) == {s for s in (config['primary_sensor'], config['auxiliary_sensor']) if s}
        assert execution['implementation_sha256'] in approved
        if name == 'event3_event4': assert set(execution['pixel_datasets_read']) == {'event3', 'event4'}
    tasks = read(out / 'test/tasks.json')
    assert len(tasks) == 80 and sum(t['video_frames'] for t in tasks) == 24080
    assert all((out / 'results' / n / t['session'] / 'complete.json').exists() for n in EVAL_NAMES for t in tasks)
    comparison = read(out / 'comparison.json'); raw = comparison['raw_recalculation']
    keys = ['first_frame_aligned_mpjpe_mm', 'first_frame_aligned_pa_mpjpe_mm', 'frame_aligned_mpjpe_mm', 'frame_aligned_pa_mpjpe_mm', 'pelvis_error_mm']
    assert raw['passed'] and raw['maximum_sum_difference_m'] < 1e-7
    for name in EVAL_NAMES:
        values = comparison['populations'][PRIMARY_POPULATION][name]
        count = raw['counts'][PRIMARY_POPULATION][name]
        assert count == values['samples'] == 22337
        assert np.allclose(np.array(raw['sums_m'][PRIMARY_POPULATION][name]) * 1000 / count, [values[k] for k in keys], rtol=1e-10, atol=1e-8)
        assert comparison['coverage'][name]['total'] == 24080
    assert len(comparison['tests']) == 26 and len(PAIRS) == 13
    assert read(out / 'analysis_protocol.json')['bootstrap_replicates'] == 10000
    for row in comparison['tests'].values():
        assert row['clusters'] == 18 and abs(row['effect'] - (row['variant_mean'] - row['reference_mean'])) < 1e-10
        assert row['holm_p_value'] >= row['p_value'] - 1e-12
    print('checkpoint_execution_full_Test_and_statistics: passed', flush=True)
    runtime = read(out / 'runtime/summary.json'); assert len(runtime) == 18
    ordered = sorted(runtime.values(), key=lambda r: r['process_started_unix'])
    assert {r['gpu_uuid'] for r in ordered} == {audit['same_gpu_uuid']}
    assert all(r['physical_gpu'] == 1 and r['frames'] == 300 and r['input_parity_passed'] and r['compute_pids_before'] == r['compute_pids_after'] == [r['pid']] for r in ordered)
    assert all(a['process_finished_unix'] <= b['process_started_unix'] for a, b in zip(ordered, ordered[1:]))
    for name in FUSION_NAMES:
        ablation = read(out / 'ablations' / (name + '.json'))
        assert ablation['status'] == 'passed' and not ablation['test_used'] and len(ablation['rows']) == 4
        for row in ablation['rows'].values():
            assert row['total_gt_instances'] == 288 and not row['GT_used_as_inference_input']
            guarded = {v['frame_index'] for v in row.get('resource_guarded_frames', [])}
            assert all(int(k.split(':')[0]) not in guarded for k in row['errors_mm'])
    assert audit['diagnostic_resource_failure_GT']['event3_event4']['zero_aux'] == 288
    assert audit['diagnostic_resource_failure_GT']['rgb4_event4']['zero_aux'] == 285
    proof = read(out / 'rendered_table_proof.json'); verify_tables(text, proof['tables'])
    assert proof['verification'] == {'status': 'passed', 'tables': 11, 'rows': 154}
    assert text.index('FUSION_TABLE primary') < text.index('고정15시간 범위:')
    assert len(proof['tables']['primary']['headers']) == 6 and len(proof['tables']['primary']['rows']) == 9
    for term in ['0.00~0.02mm', '동일3시간 RGB 대조군', '가속도 오차가 증가', '74.02분', '원본 RGB3/RGB4', '후보16개 초과', '한 seed', 'RPE-Z', 'pelvis 상대 body22', '전체Train1epoch 학습으로 표시하지 않는다']:
        assert term in text, term
    assert audit['recommendation'] == min(TRAIN_NAMES, key=lambda n: read(out / 'runs' / n / 'complete.json')['best_metrics']['pa_mpjpe_mm']) == 'rgb3_event4'
    visual = audit['visualization']; curves = audit['learning_curve']
    assert visual['relative_frame'] == 0 and visual['RGB3_background_display_only'] and not visual['event_only_RGB_inference']
    assert visual['display_only_letterbox_bounds'] == [0, 196, 896, 700]
    assert len(curves['curves']) == 6 and all(c['hours'][0] == 0 and c['hours'][2] > c['hours'][1] for c in curves['curves'])
    assert np.allclose(visual['chart_values_mm'], [[comparison['populations'][PRIMARY_POPULATION][n][k] for k in keys] for n in EVAL_NAMES])
    figures = [visual['chart'], visual['image'], curves['path'], str(out / 'figures/신체센서융합_논문형_전체구조.png')]
    figure_hashes = {p: sha(p) for p in figures}
    architecture = ROOT / '_agents/report/261027_센서융합_모델구조와_입출력_파이프라인.md'
    for document in (report, architecture):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            if target.startswith(('https://', 'http://')): continue
            path = Path(target); path = path if path.is_absolute() else document.parent / path
            assert path.exists(), str(path)
    paper = read(out / 'figures/논문형_전체구조_출처.json')
    assert sha(paper['actual_example']) == paper['example_sha256']
    for path, digest in paper['outputs'].items(): assert sha(path) == digest
    assert sha(ROOT / 'tools/render_fusion_repair.py') == audit['report_render_implementation_sha256']
    active = []
    for job in read(Path(goal['managed_jobs_file']))['jobs']:
        current = identity(job['pid'])
        if job['pid'] != os.getpid() and current is not None and current == {k: job[k] for k in ('pgid', 'start_ticks', 'command')}:
            active.append({'pid': job['pid'], 'role': job['role']})
    assert all(j['role'] in ('fusion_repair_deadline_watchdog', 'fusion_repair_15h_deadline_watchdog') for j in active), active
    sources = ['tools/analyze_fusion_repair.py', 'tools/render_fusion_repair.py', 'tools/finalize_fusion_repair.py', 'tools/benchmark_fusion_repair.py', 'tools/ablate_fusion_repair.py', 'tools/review_completed_fusion_repair.py', 'src/utils/compare_event_methods.py', 'src/utils/chunk_alignment.py', 'src/utils/humanpose_metrics.py', 'src/utils/temporal_metrics.py', 'src/utils/paired_statistics.py', 'src/multihmr/multisensor_diagnostics.py']
    source_hashes = {s: sha(ROOT / s) for s in sources}
    for source, digest in source_hashes.items():
        target = out / 'source_snapshots' / (digest + '.txt')
        if not target.exists(): target.write_bytes((ROOT / source).read_bytes())
        assert sha(target) == digest
    review = dict(status='passed_agent_review_native_completion_pending', checked_unix=time.time(), goal_complete=False,
                  report_sha256=sha(report), architecture_report_sha256=sha(architecture),
                  requirement_matrix='_agents/research/261027/센서융합_Goal_최종완료_요건별_대조.md',
                  all_seven_training_terminal=True, all_nine_full_Test_verified=True, new_Test_raw_segments=720,
                  Test_frames_per_condition=24080, common_nine_GT_samples=22337, checkpoint_actual_loads=14,
                  released_source_manifests=2, source_archives_metadata_rechecked=archive_count,
                  numeric_tables_verified=proof['verification'],
                  independent_raw_five_metric_maximum_sum_difference_m=raw['maximum_sum_difference_m'],
                  same_GPU_sequential_runs=18, same_GPU_uuid=audit['same_gpu_uuid'], all_five_holdout_ablation_conditions=True,
                  resource_failures_reported_separately=audit['diagnostic_resource_failure_GT'],
                  images_opened_and_visually_reviewed=figure_hashes,
                  visualization_review='Nine same-frame actual predictions and common camera projection; background letterbox display crop only. Paper initialization example distinguished from released0 and new Test. Six learning curves use real selected checkpoint times and common GT.',
                  analysis_source_sha256=source_hashes, recommendation_matches_Train_only_argmin=audit['recommendation'],
                  no_training_evaluation_or_benchmark_workers_remaining=True, remaining_owned_managers=active,
                  deadline_unix=goal['deadline_unix'], finished_before_deadline=True,
                  limitations_verified=['unequal added learning budgets and RGB4 failed-launch interval', 'warmstart cumulative experience separated from released zero', 'single seed and repeated benchmark', 'tiny auxiliary-removal effect, no isolated event contribution claim', 'pelvis-relative body acceleration worsened', 'zero-event resource failures and unassessed poses explicit', 'event-only worse than RGB', 'online archive timing and OS cache limitations'])
    write(out / 'agent_completion_review.json', review)
    audit['agent_review_status'] = 'passed'; audit['agent_review_file'] = str(out / 'agent_completion_review.json')
    write(out / 'final_artifact_audit.json', audit)
    print('all_requirements_and_manual_figure_review: passed; native completion pending', flush=True)


if __name__ == '__main__':
    main()

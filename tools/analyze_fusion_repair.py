"""Full raw Test comparison with both RGB controls and labelled historical runs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write
import numpy as np
from utils.chunk_alignment import METRICS
from utils.compare_event_methods import segment_metrics, summarize
from utils.paired_statistics import paired_cluster, holm_adjust, motion_strata
from finalize_device34_body import independent_errors
from fusion_repair_scope import TRAIN_NAMES,PRETRAIN_NAMES,EVAL_NAMES,HIST_NAMES,PAIRS,PRIMARY_POPULATION,HISTORICAL_POPULATION
NAMES = TRAIN_NAMES
HEADLINE = METRICS + ['pelvis_error_mm']
GOAL = ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json'


def read(path):
    return json.loads(path.read_text())


def protocol():
    return dict(new_conditions=NAMES, pretrained_conditions=PRETRAIN_NAMES,historical_conditions=['old_' + n for n in HIST_NAMES],
                pairs=[list(p) for p in PAIRS], metrics=HEADLINE,
                common_populations=[PRIMARY_POPULATION,HISTORICAL_POPULATION,'learning_rgb3','learning_rgb4','learning_family_rgb3','learning_family_rgb4'],
                clusters='whole body participant/session; 18 unique body subjects',
                bootstrap_replicates=10000, seed=261027,
                Holm_families=f'all {len(PAIRS)} prespecified PA pairs; separately all {len(PAIRS)} root-position pairs',
                budget_interpretation='Latest user requests shorter RGB-only and longer fusion; duration and modality effects are confounded',
                motion_thresholds_mps=[.05, .5], selection='Train holdout only',
                historical_results_label='Historical existing predictions; never relabelled as new method',
                ablation_modes=['intact','missing_aux','zero_aux','swap_aux_polarity'],
                recommendation='Default RGB chosen by Train holdout PA at coverage>=.95. Fusion eligible only if Train PA improves its paired RGB, Test PA CI upper<0 and Holm<.05, root effect<=0, coverage drop<=1 percentage point; choose eligible fusion by Train PA.')


def analyze(goal):
    out = Path(goal['output_root']); prior = Path(goal['prior_experiment'])
    assert read(out / 'analysis_protocol.json') == protocol()
    manifest = read(out / 'manifest.json')
    assert hashlib.sha256((ROOT / 'dataset/split.json').read_bytes()).hexdigest() == manifest['split_sha256']
    tasks = read(out / 'test/tasks.json')
    assert len(tasks) == 80 and sum(t['video_frames'] for t in tasks) == 24080
    roots = {n: out for n in EVAL_NAMES} | {'old_' + n: prior for n in HIST_NAMES}
    canonical = {n: n for n in EVAL_NAMES} | {'old_' + n: n for n in HIST_NAMES}
    groups = {PRIMARY_POPULATION:EVAL_NAMES,HISTORICAL_POPULATION:list(roots),
              'learning_rgb3':['pre_rgb3','old_rgb3','rgb3_short90'],
              'learning_rgb4':['pre_rgb4','old_rgb4','rgb4'],
              'learning_family_rgb3':['pre_rgb3','old_rgb3','rgb3_short90','rgb3_event4','rgb3_event3'],
              'learning_family_rgb4':['pre_rgb4','old_rgb4','rgb4','event3_rgb4','rgb4_event4']}
    groups.update({'own_' + n: [n] for n in roots})
    groups.update({'pair_' + a + '__' + b: [a, b] for a, b in PAIRS})
    rows = {g: {n: [] for n in members} for g, members in groups.items()}
    coverage = dict.fromkeys(roots, 0); total = 0; hashes = []
    raw_sums = {g: {n: np.zeros(5) for n in groups[g]} for g in (PRIMARY_POPULATION,HISTORICAL_POPULATION,'learning_rgb3','learning_rgb4','learning_family_rgb3','learning_family_rgb4')}
    raw_counts = {g: dict.fromkeys(groups[g], 0) for g in raw_sums}
    motion = {a + '__' + b: {label: {n: dict(count=0, sum_mm=0.) for n in (a, b)}
                            for label in ('still', 'slow', 'fast', 'unknown')} for a, b in PAIRS}
    maximum_difference = 0.
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt = z['joints_cam_m']; valid = z['valid']; positions = z['source_rgb_positions']; stamps = z['timestamps_ns']
            assert positions[0] == 0 and len(positions) == task['video_frames']
            assert np.array_equal(z['original_source_rgb_positions'], positions + task['start_frame'])
        total += int(valid.sum()); predictions = {}; masks = {}
        for name, source in roots.items():
            folder = source / 'results' / canonical[name] / task['session']
            complete = read(folder / 'complete.json')['task']
            assert complete['split_sha256'] == manifest['split_sha256']
            expected_config=read(source/'configs'/(canonical[name]+'.json'))
            assert complete['condition'] == expected_config['condition']
            assert complete['checkpoint'] == str(source / 'checkpoints' / canonical[name] / 'best.pt')
            assert complete['matching_protocol'] == 'primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1'
            for key in ('parent_session', 'start_frame', 'end_frame', 'video_frames'):
                assert complete[key] == task[key]
            path = folder / 'result.npz'
            with np.load(path) as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'], positions)
                pred = z['matched_joints_cam_m'].astype(np.float64)
                mask = valid & (z['matched_pred_indices'] >= 0)
                assert pred.shape == gt.shape and np.isfinite(pred[mask]).all()
                assert np.isnan(pred[valid & ~mask]).all()
            predictions[name] = pred; masks[name] = mask; coverage[name] += int(mask.sum())
            hashes.append(dict(name=name, task=task['session'], path=str(path),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        strata = motion_strata(gt, valid, positions, task['fps'], timestamps_ns=stamps)
        for group, members in groups.items():
            common = np.logical_and.reduce([masks[n] for n in members])
            for name in members:
                row, errors, eligible = segment_metrics(predictions[name], gt, common, positions, task['fps'], timestamps_ns=stamps)
                row.update(session=task['session'], parent_session=task['parent_session'], cluster=int(task['parent_session'].split('_')[3]))
                rows[group][name].append(row)
                if group in raw_sums:
                    sums, n = independent_errors(predictions[name], gt, common, positions)
                    expected = np.array(row['alignment_sums_m'] + [row['pelvis_sum_m']])
                    assert n == row['count'] and np.allclose(sums, expected, rtol=1e-6, atol=1e-7)
                    maximum_difference = max(maximum_difference, float(np.abs(sums-expected).max()))
                    raw_sums[group][name] += sums; raw_counts[group][name] += n
                if group.startswith('pair_'):
                    pair = group[5:]
                    for label, code in [('still', 0), ('slow', 1), ('fast', 2), ('unknown', -1)]:
                        keep = eligible & (strata == code)
                        motion[pair][label][name]['count'] += int(keep.sum())
                        motion[pair][label][name]['sum_mm'] += float(errors[keep, 3].sum() * 1000)
        print(json.dumps(dict(raw_segment_verified=task['session'])), flush=True)
    populations = {g: {n: summarize(values) for n, values in members.items()} for g, members in rows.items()}
    tests = {}
    for reference, variant in PAIRS:
        pair = reference + '__' + variant; group = rows['pair_' + pair]
        clusters = sorted({r['cluster'] for r in group[reference]})
        for label, index in [('pa', 3), ('root_position', 4)]:
            sums = {}; counts = None
            for name in (reference, variant):
                selected = [[r for r in group[name] if r['cluster'] == c] for c in clusters]
                sums[name] = [1000 * sum(r['alignment_sums_m'][index] if index < 4 else r['pelvis_sum_m'] for r in rr) for rr in selected]
                current = [sum(r['count'] for r in rr) for rr in selected]
                if counts is not None:
                    assert current == counts
                counts = current
            tests[pair + '_' + label] = paired_cluster(sums[reference], sums[variant], counts, seed=261027)
    for suffix in ('_pa', '_root_position'):
        adjusted = holm_adjust({k: v['p_value'] for k, v in tests.items() if k.endswith(suffix)})
        for key, value in adjusted.items():
            tests[key]['holm_p_value'] = value
    result = dict(split_sha256=manifest['split_sha256'], conditions=NAMES,pretrained_conditions=PRETRAIN_NAMES,historical=['old_'+n for n in HIST_NAMES],
                  segments=len(tasks), frames=sum(t['video_frames'] for t in tasks), gt_instances=total,
                  populations=populations, tests=tests, motion_strata=motion,
                  coverage={n: dict(matched=coverage[n], total=total, fraction=coverage[n]/total) for n in roots},
                  raw_recalculation=dict(passed=True, independent='SciPy proper Rotation.align_vectors vs NumPy SVD',
                     sums_m={g: {n: v.tolist() for n, v in members.items()} for g, members in raw_sums.items()},
                     counts=raw_counts, maximum_sum_difference_m=maximum_difference), updated_unix=time.time())
    write(out / 'segment_sums.json', rows); write(out / 'raw_prediction_hashes.json', hashes)
    write(out / 'comparison.json', result)
    return result


if __name__ == '__main__':
    goal = read(GOAL); register_job(goal, os.getpid(), 'fusion_repair_full_raw_analysis')
    analyze(goal)

"""Frozen Test aggregation with explicit populations and paired cluster tests."""
import json
from pathlib import Path
import time
import numpy as np

from .chunk_alignment import METRICS, aligned_errors
from .humanpose_metrics import joint_errors
from .temporal_metrics import temporal_errors
from .paired_statistics import paired_cluster, holm_adjust, motion_strata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'exp/261024/rgb_event_36h'


def segment_metrics(prediction, gt, common, positions, fps):
    """Sufficient sums for one population; all errors internally metres.

    Exact frame0 eligibility applies to every reported pose/temporal quantity.
    Motion strata subsequently subdivide these same eligible matches.
    """
    first = np.flatnonzero(positions == 0)
    eligible = common[first[0]].copy() if len(first) else np.zeros(gt.shape[1], bool)
    common = common & eligible[None]
    errors = np.full((*common.shape, 4), np.nan, np.float64)
    for person in np.flatnonzero(eligible):
        keep = common[:, person]
        errors[keep, person] = aligned_errors(prediction[keep, person, :22], gt[keep, person, :22],
            prediction[first[0], person, :22], gt[first[0], person, :22])
    count = int(common.sum())
    p = prediction[common].astype(np.float64)
    target = gt[common].astype(np.float64)
    full = np.array([joint_errors(a, b) for a, b in zip(p, target)]).reshape(-1, 3)
    absolute = np.linalg.norm(p[:, :22] - target[:, :22], axis=-1).mean(-1)
    pelvis = np.linalg.norm(p[:, 0] - target[:, 0], axis=-1)
    temporal = temporal_errors(prediction, gt, common, positions, fps)
    assert np.isfinite(errors[common]).all() and np.isfinite(full).all()
    return dict(count=count, anchors=int(eligible.sum()),
                alignment_sums_m=np.nansum(errors, axis=(0, 1)).tolist(),
                absolute_sum_m=float(absolute.sum()), pelvis_sum_m=float(pelvis.sum()),
                body55_sums_m=full.sum(0).tolist(), temporal_sums= list(temporal)), errors, common


def summarize(rows):
    count = sum(row['count'] for row in rows)
    temporal = np.array([row['temporal_sums'] for row in rows], np.float64).reshape(-1, 4).sum(0)
    def mean(key):
        return float(1000 * sum(row[key] for row in rows) / count) if count else None
    alignment = np.array([row['alignment_sums_m'] for row in rows], np.float64).reshape(-1, 4).sum(0)
    body55 = np.array([row['body55_sums_m'] for row in rows], np.float64).reshape(-1, 3).sum(0)
    return dict(samples=count, anchor_people=sum(row['anchors'] for row in rows),
                **{key: float(1000 * alignment[i] / count) if count else None for i, key in enumerate(METRICS)},
                absolute_mpjpe_mm=mean('absolute_sum_m'), pelvis_error_mm=mean('pelvis_sum_m'),
                body55={key: float(1000 * body55[i] / count) if count else None
                        for i, key in enumerate(('absolute_mpjpe_mm', 'root_mpjpe_mm', 'pa_mpjpe_mm'))},
                pelvis_velocity_error_mmps=float(1000 * temporal[0] / temporal[1]) if temporal[1] else None,
                root_acceleration_error_mmps2=float(1000 * temporal[2] / temporal[3]) if temporal[3] else None,
                velocity_samples=int(temporal[1]), acceleration_samples=int(temporal[3]))


def main():
    from multihmr.event_data import save
    read = lambda path: json.loads(path.read_text())
    tasks = read(ROOT / '.experiment_storage/humanpose_gt_segments/tasks.json')
    conditions = read(OUT / 'experiment_manifest.json')['conditions']
    plan = read(OUT / 'statistical_analysis_plan.json')
    names = [config['name'] for config in conditions]
    assert len(names) == 10 and len(tasks) == 320 and sum(t['video_frames'] for t in tasks) == 96124
    groups = {'all10': names, **{'own_' + name: [name] for name in names},
              **{name: ['rgb_baseline', name] for name in plan['variants']}}
    rows = {group: {name: [] for name in members} for group, members in groups.items()}
    strata = {variant: {label: {name: [] for name in ['rgb_baseline', variant]}
               for label in ['still', 'slow', 'fast', 'unknown']} for variant in plan['variants']}
    coverage = dict.fromkeys(names, 0)
    gt_total = 0
    for task in tasks:
        with np.load(task['gt_file']) as data:
            gt = data['joints_cam_m']; valid = data['valid']; positions = data['source_rgb_positions']
            assert np.array_equal(data['original_source_rgb_positions'], positions + task['start_frame'])
        gt_total += int(valid.sum())
        predictions = {}; masks = {}
        for name in names:
            folder = OUT / 'results' / name / task['session']
            complete = read(folder / 'complete.json')
            assert not complete.get('partial') and not complete['task'].get('smoke')
            assert complete['task']['video_frames'] == task['video_frames']
            with np.load(folder / 'result.npz') as data:
                assert data['seen_gt_rows'].all() and np.array_equal(data['source_rgb_positions'], positions)
                prediction = data['matched_joints_cam_m']
                assert prediction.shape == gt.shape
                mask = valid & np.isfinite(prediction).all((2, 3))
                assert np.array_equal(mask, valid & (data['matched_pred_indices'] >= 0))
                predictions[name] = prediction; masks[name] = mask
                coverage[name] += int(mask.sum())
        motion = motion_strata(gt, valid, positions, task['fps'])
        for group, members in groups.items():
            common = np.logical_and.reduce([masks[name] for name in members])
            for name in members:
                row, errors, eligible_common = segment_metrics(predictions[name], gt, common, positions, task['fps'])
                row.update(session=task['session'], parent_session=task['parent_session'],
                           cluster=plan['session_to_cluster'][task['parent_session']])
                rows[group][name].append(row)
                if group in plan['variants']:
                    for label, code in [('still', 0), ('slow', 1), ('fast', 2), ('unknown', -1)]:
                        selected = eligible_common & (motion == code)
                        # Stratum pose means retain the original frame0 transform.
                        strata[group][label][name].append(dict(samples=int(selected.sum()),
                            pa_sum_mm=float(errors[selected, 3].sum() * 1000)))
    primary = {}; secondary = {}
    clusters = sorted(plan['clusters'])
    for variant in plan['variants']:
        for metric, destination in [('pose', primary), ('velocity', secondary), ('acceleration', secondary)]:
            sums = {}; counts = None
            for name in ('rgb_baseline', variant):
                values = []; current_counts = []
                for cluster in clusters:
                    selected = [row for row in rows[variant][name] if row['cluster'] == cluster]
                    if metric == 'pose':
                        values.append(1000 * sum(row['alignment_sums_m'][3] for row in selected))
                        current_counts.append(sum(row['count'] for row in selected))
                    else:
                        index = 0 if metric == 'velocity' else 2
                        values.append(1000 * sum(row['temporal_sums'][index] for row in selected))
                        current_counts.append(sum(row['temporal_sums'][index + 1] for row in selected))
                if counts is not None:
                    assert counts == current_counts
                counts = current_counts; sums[name] = values
            destination[variant if metric == 'pose' else variant + '_' + metric] = paired_cluster(
                sums['rgb_baseline'], sums[variant], counts)
    for family in (primary, secondary):
        adjusted = holm_adjust({key: value['p_value'] for key, value in family.items()})
        for key, value in family.items():
            value['holm_p_value'] = adjusted[key]
    result = dict(updated_unix=time.time(), conditions=names, gt_instances=gt_total,
        rgb_frames=96124, segments=320, coverage={name: dict(matched=coverage[name],
            total=gt_total, fraction=coverage[name] / gt_total) for name in names},
        populations={group: {name: summarize(values) for name, values in members.items()}
                     for group, members in rows.items()}, primary_tests=primary, secondary_tests=secondary,
        motion_strata=strata, analysis_plan=str(OUT / 'statistical_analysis_plan.json'))
    save(OUT / 'comparison_segment_sums.json', rows)
    save(OUT / 'comparison.json', result)


if __name__ == '__main__':
    main()

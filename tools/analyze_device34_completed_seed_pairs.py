"""Partial repeat diagnostic on one common GT population across eight runs."""
import json
import os
from pathlib import Path
import time

from goal_runtime_control import register_job

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/261026/device34_body_28h'


def main():
    goal = json.loads((ROOT / '_agents/research/261026/device34_body_goal_28h.json').read_text())
    register_job(goal, os.getpid(), 'device34_completed_seed_pairs_cpu')
    from finalize_device34_body import read, save, np, independent_errors, segment_metrics, HEADLINE
    conditions = ['rgb3', 'rgb4', 'rgb3_rgb4', 'rgb3_event4']
    names = conditions + [n + '_seed261027' for n in conditions]
    for name in names[4:]:
        assert read(OUT / (name + '_raw_test_cpu_audit.json'))['status'] == 'passed'
    tasks = read(OUT / 'test/tasks.json')
    totals = {n: np.zeros(5) for n in names}
    count = frames = 0
    difference = 0.
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt = z['joints_cam_m']; valid = z['valid']; positions = z['source_rgb_positions']
            stamps = z['timestamps_ns']
            assert positions[0] == 0
        predictions = {}; masks = []
        for name in names:
            folder = OUT / 'results' / name / task['session']
            complete = read(folder / 'complete.json')['task']
            assert complete['condition'] == name.removesuffix('_seed261027')
            assert complete['split_sha256'] == read(OUT / 'manifest.json')['split_sha256']
            for key in ('parent_session', 'start_frame', 'end_frame', 'video_frames'):
                assert complete[key] == task[key]
            with np.load(folder / 'result.npz') as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'], positions)
                pred = z['matched_joints_cam_m'].astype(np.float64)
                mask = valid & (z['matched_pred_indices'] >= 0)
                assert pred.shape == gt.shape and np.isfinite(pred[mask]).all()
                assert np.isnan(pred[valid & ~mask]).all()
                predictions[name] = pred
                masks.append(mask)
        common = np.logical_and.reduce(masks)
        counts = []
        for name, pred in predictions.items():
            sums, n = independent_errors(pred, gt, common, positions)
            reference, _, _ = segment_metrics(pred, gt, common, positions, task['fps'], timestamps_ns=stamps)
            expected = np.array(reference['alignment_sums_m'] + [reference['pelvis_sum_m']])
            assert n == reference['count'] and np.allclose(sums, expected, rtol=1e-6, atol=1e-7)
            difference = max(difference, float(np.abs(sums - expected).max()))
            totals[name] += sums; counts.append(n)
        assert len(set(counts)) == 1
        count += counts[0]; frames += len(positions)
    result = dict(status='passed_partial_seed_pair_diagnostic', conditions=conditions, runs=names,
                  segments=len(tasks), frames=frames, common_samples=count,
                  metrics_mm={n: dict(zip(HEADLINE, (v * 1000 / count).tolist())) for n, v in totals.items()},
                  maximum_independent_sum_difference_m=difference, checked_unix=time.time(),
                  split_sha256=read(OUT / 'manifest.json')['split_sha256'],
                  scope='First four conditions, both seeds, common eight-run GT and exact frame0. Not the six-condition primary or final repeat comparison.')
    save(OUT / 'completed_four_seed_pairs_common_gt.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

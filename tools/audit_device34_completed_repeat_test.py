"""Audit one completed repeat Test from all raw predictions, without ranking models."""
import argparse
import json
import os
from pathlib import Path
import time

from goal_runtime_control import register_job

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/261026/device34_body_28h'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    config = json.loads((ROOT / '_agents/research/261026/device34_body_goal_28h.json').read_text())
    register_job(config, os.getpid(), 'raw_repeat_test_audit:' + args.name)
    from finalize_device34_body import read, independent_errors, segment_metrics, np, CONDITIONS
    assert args.name in [n + '_seed261027' for n in CONDITIONS]
    manifest = read(OUT / 'manifest.json')
    tasks = read(OUT / 'test/tasks.json')
    sums = np.zeros(5)
    count = frames = matches = total = 0
    maximum_difference = 0.0
    for task in tasks:
        folder = OUT / 'results' / args.name / task['session']
        complete = read(folder / 'complete.json')['task']
        assert complete['split_sha256'] == manifest['split_sha256']
        assert complete['matching_protocol'] == 'primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1'
        assert complete['checkpoint'] == str(OUT / 'checkpoints' / args.name / 'best.pt')
        assert complete['condition'] == args.name.removesuffix('_seed261027')
        for key in ('start_frame', 'end_frame', 'video_frames', 'parent_session'):
            assert complete[key] == task[key]
        with np.load(task['gt_file']) as z:
            gt = z['joints_cam_m']; valid = z['valid']; positions = z['source_rgb_positions']; stamps = z['timestamps_ns']
            assert positions[0] == 0
            assert np.array_equal(z['original_source_rgb_positions'], positions + task['start_frame'])
        with np.load(folder / 'result.npz') as z:
            assert z['seen_gt_rows'].all()
            assert np.array_equal(z['source_rgb_positions'], positions)
            pred = z['matched_joints_cam_m'].astype(np.float64)
            mask = valid & (z['matched_pred_indices'] >= 0)
            assert pred.shape == gt.shape and np.isfinite(pred[mask]).all()
            assert np.isnan(pred[valid & ~mask]).all()
        row, _, _ = segment_metrics(pred, gt, mask, positions, task['fps'], timestamps_ns=stamps)
        independent, n = independent_errors(pred, gt, mask, positions)
        expected = np.array(row['alignment_sums_m'] + [row['pelvis_sum_m']])
        assert n == row['count'] and np.allclose(independent, expected, rtol=1e-6, atol=1e-7)
        maximum_difference = max(maximum_difference, float(np.abs(independent - expected).max()))
        sums += independent; count += n
        frames += len(positions); matches += int(mask.sum()); total += int(valid.sum())
    result = dict(status='passed', name=args.name, segments=len(tasks), frames=frames,
                  matched_instances=matches, total_gt_instances=total,
                  own_frame0_eligible_count=count, independent_five_metric_sums_m=sums.tolist(),
                  maximum_sum_difference_m=maximum_difference, checked_unix=time.time(),
                  scope='Single-condition own population; not the six-condition common-GT table')
    destination = OUT / (args.name + '_raw_test_cpu_audit.json')
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(destination)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

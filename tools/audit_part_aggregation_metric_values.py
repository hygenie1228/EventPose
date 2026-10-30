"""Independently check five reported metrics from completed stored predictions.

Uses SciPy's proper rotation fit, without importing the production metric code.
This performs no inference and never updates a comparison or model selection.
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/261029/part_aware_dynamic_aggregation'
KEYS = ['first_frame_aligned_mpjpe_mm', 'first_frame_aligned_pa_mpjpe_mm',
        'frame_aligned_mpjpe_mm', 'frame_aligned_pa_mpjpe_mm', 'root_depth_error_mm']


def read(path):
    return json.loads(path.read_text())


def fit(pred, target):
    center_p, center_t = pred.mean(0), target.mean(0)
    x, y = pred-center_p, target-center_t
    rotation, _ = Rotation.align_vectors(y, x)
    assert np.linalg.det(rotation.as_matrix()) > .999999
    scale = np.sum(rotation.apply(x)*y)/np.sum(x*x)
    assert scale > 0
    return rotation, scale, center_p, center_t


def apply_fit(pred, fitted):
    rotation, scale, center_p, center_t = fitted
    return scale*rotation.apply((pred-center_p).reshape(-1, 3)).reshape(pred.shape)+center_t


def errors(pred, target):
    # Each input contains one person's frames, with the exact original frame0
    # first. The anchor transform stays fixed for both first-frame metrics.
    root = pred+(target[0, 0]-pred[0, 0])
    first_pa = apply_fit(pred, fit(pred[0], target[0]))
    frame_root = pred-pred[:, :1]+target[:, :1]
    frame_pa = np.stack([apply_fit(p, fit(p, t)) for p, t in zip(pred, target)])
    return np.column_stack([
        np.linalg.norm(root-target, axis=-1).mean(-1),
        np.linalg.norm(first_pa-target, axis=-1).mean(-1),
        np.linalg.norm(frame_root-target, axis=-1).mean(-1),
        np.linalg.norm(frame_pa-target, axis=-1).mean(-1),
        np.abs(pred[:, 0, 2]-target[:, 0, 2])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    summary = read(OUT/args.comparison)
    assert summary['all_complete'] and not summary['pending']
    names = summary['conditions']
    config = read(OUT/'configs'/(names[0]+'.json'))
    tasks = read(Path(config['source_root'])/'test/tasks.json')
    assert len(tasks) == 80
    # Analytic translation and time-varying translation distinguish unaligned
    # depth from aligned pose, and fixed anchors from per-frame alignment.
    rng = np.random.default_rng(261029)
    target = rng.normal(size=(3, 22, 3))*.1+[0, 0, 3]
    value = errors(target+[.1, .2, .3], target)
    assert np.max(np.abs(value[:, :4])) < 1e-12
    assert np.max(np.abs(value[:, 4]-.3)) < 1e-12
    pred = target.copy(); pred[1:] += [.1, 0, 0]
    value = errors(pred, target)
    assert np.allclose(value[1:, 0], .1) and np.max(value[:, 2:]) < 1e-12
    totals = {name: np.zeros(5) for name in names}
    count = anchors = 0
    for task in tasks:
        with np.load(task['gt_file'], allow_pickle=False) as z:
            gt = z['joints_cam_m'][..., :22, :].astype(np.float64)
            valid = z['valid']; positions = z['source_rgb_positions']
        assert positions[0] == 0 and np.all(np.diff(positions) > 0)
        predictions = {}; common = valid & np.isfinite(gt).all((2, 3))
        for name in names:
            folder = OUT/'results'/name/task['session']
            receipt = read(folder/'complete.json')
            assert not receipt.get('partial') and not receipt['task'].get('smoke')
            with np.load(folder/'result.npz', allow_pickle=False) as z:
                assert z['seen_gt_rows'].all()
                assert np.array_equal(z['source_rgb_positions'], positions)
                pred = z['matched_joints_cam_m'][..., :22, :].astype(np.float64)
                matched = z['matched_pred_indices'] >= 0
            assert pred.shape == gt.shape
            predictions[name] = pred
            common &= matched & np.isfinite(pred).all((2, 3))
        eligible = common[0]
        anchors += int(eligible.sum())
        for person in np.flatnonzero(eligible):
            keep = common[:, person]
            assert keep[0]
            count += int(keep.sum())
            for name, pred in predictions.items():
                totals[name] += errors(pred[keep, person], gt[keep, person]).sum(0)
    values = {}; differences = {}
    for name in names:
        assert summary['metrics'][name]['samples'] == count
        measured = 1000*totals[name]/count
        expected = np.array([summary['metrics'][name][key] for key in KEYS])
        delta = np.abs(measured-expected)
        assert np.max(delta) < 1e-7, (name, measured, expected, delta)
        values[name] = dict(zip(KEYS, measured.tolist()))
        differences[name] = float(delta.max())
    result = dict(status='passed', checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        comparison=args.comparison, conditions=names, segments=80, common_samples=count,
        exact_frame0_anchor_people=anchors, independent_metrics_mm=values,
        maximum_absolute_report_difference_mm=differences,
        method='Stored common matched Body22 coordinates; independent SciPy proper rotation and positive scale; exact fixed frame0; unaligned absolute pelvis Z; no production metric imports',
        limitation='Verifies stored prediction metric arithmetic and common population; does not independently rerun detector matching or inference')
    folder = OUT/'artifact_audits'; folder.mkdir(exist_ok=True)
    (folder/(args.label+'_independent_metrics.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

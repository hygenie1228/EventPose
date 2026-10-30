"""Display full-Test results as conditions finish; never select on Test."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
from utils.compare_event_methods import segment_metrics, summarize
from finalize_device34_body import independent_errors
from fusion_repair_scope import EVAL_NAMES
from goal_runtime_control import write


def read(path):
    return json.loads(path.read_text())


def completed_test_snapshot(out, names_scope=None):
    out = Path(out)
    tasks = read(out / 'test/tasks.json')
    assert len(tasks) == 80 and sum(t['video_frames'] for t in tasks) == 24080
    names = [n for n in (EVAL_NAMES if names_scope is None else names_scope) if all(
        (out / 'results' / n / t['session'] / 'complete.json').exists() for t in tasks)]
    if not names:
        return None
    path = out / 'interim_full_test_comparison.json'
    files = [out / 'results' / n / t['session'] / f
             for n in names for t in tasks for f in ('result.npz', 'complete.json')]
    signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    signature_hash = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    if path.exists():
        cached = read(path)
        if cached['conditions'] == names and cached['input_signature_sha256'] == signature_hash:
            return cached
    configs = {n: read(out / 'configs' / (n + '.json')) for n in names}
    rows = {n: [] for n in names}
    hashes = []
    max_difference = 0.
    for task in tasks:
        with np.load(task['gt_file']) as z:
            gt, valid, positions, stamps = (z[k] for k in
                ('joints_cam_m', 'valid', 'source_rgb_positions', 'timestamps_ns'))
            assert positions[0] == 0 and len(positions) == task['video_frames']
            assert np.array_equal(z['original_source_rgb_positions'], positions + task['start_frame'])
        predictions, masks = {}, {}
        for name in names:
            folder = out / 'results' / name / task['session']
            complete = read(folder / 'complete.json')['task']
            assert complete['split_sha256'] == configs[name]['split_sha256']
            assert complete['condition'] == configs[name]['condition']
            assert complete['checkpoint'] == str(out / 'checkpoints' / name / 'best.pt')
            for key in ('parent_session', 'start_frame', 'end_frame', 'video_frames'):
                assert complete[key] == task[key]
            raw = folder / 'result.npz'
            with np.load(raw) as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'], positions)
                pred = z['matched_joints_cam_m'].astype(np.float64)
                mask = valid & (z['matched_pred_indices'] >= 0)
                assert pred.shape == gt.shape and np.isfinite(pred[mask]).all()
                assert np.isnan(pred[valid & ~mask]).all()
            predictions[name], masks[name] = pred, mask
            hashes.append(dict(condition=name, segment=task['session'], sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        common = np.logical_and.reduce([masks[n] for n in names])
        for name in names:
            row, _, _ = segment_metrics(predictions[name], gt, common, positions, task['fps'], timestamps_ns=stamps)
            sums, count = independent_errors(predictions[name], gt, common, positions)
            expected = np.array(row['alignment_sums_m'] + [row['pelvis_sum_m']])
            assert count == row['count'] and np.allclose(sums, expected, rtol=1e-6, atol=1e-7)
            max_difference = max(max_difference, float(np.abs(sums - expected).max()))
            rows[name].append(row)
    result = dict(status='verified_completed_full_Test_only', conditions=names, segments=80,
                  frames=24080, metrics={n: summarize(rows[n]) for n in names},
                  input_signature_sha256=signature_hash, raw_prediction_sha256=hashes,
                  independent_five_metric_maximum_sum_difference_m=max_difference,
                  updated_unix=time.time(), used_for_training_or_checkpoint_selection=False,
                  population='Common detected GT with exact frame0 across the currently completed conditions. Recomputed when this set changes; not the final nine-condition population.')
    write(path, result)
    return result

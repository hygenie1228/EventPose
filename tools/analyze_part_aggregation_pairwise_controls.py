"""Describe completed part-vs-control pairs without selecting models or inference."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from utils.part_aggregation_metrics import common_detection_mask, score_chunk, summarize_chunks
from utils.paired_statistics import paired_cluster, holm_adjust

OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['v4', 'v5', 'v6', 'v6_long'], required=True)
    family = parser.parse_args().family
    part = family+'_part_dynamic'
    controls = [family+'_'+suffix for suffix in ['fixed', 'global_dynamic', 'part_static']]
    config = read(OUT/'configs'/(part+'.json'))
    tasks = read(Path(config['source_root'])/'test/tasks.json')
    assert len(tasks) == 80
    rows = {control:{name:[] for name in [control, part]} for control in controls}
    for name in controls+[part]:
        assert read(OUT/'runs'/name/'complete.json')['status'] == 'complete'
        assert all((OUT/'results'/name/task['session']/'complete.json').exists() for task in tasks)
    for task in tasks:
        with np.load(task['gt_file'], allow_pickle=False) as z:
            gt = z['joints_cam_m']; valid = z['valid']; positions = z['source_rgb_positions']
        assert positions[0] == 0
        predictions = {}
        for name in controls+[part]:
            folder = OUT/'results'/name/task['session']
            receipt = read(folder/'complete.json')
            assert not receipt.get('partial') and not receipt['task'].get('smoke')
            assert receipt['task']['split_sha256'] == config['split_sha256']
            with np.load(folder/'result.npz', allow_pickle=False) as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'], positions)
                prediction = z['matched_joints_cam_m']
                matched = valid & (z['matched_pred_indices'] >= 0)
                assert np.isfinite(prediction[matched]).all()
                assert not np.isfinite(prediction[valid & ~matched]).any()
            predictions[name] = prediction
        for control in controls:
            common = common_detection_mask({name:predictions[name] for name in [control, part]}, gt, valid)
            for name in [control, part]:
                row, _, _ = score_chunk(predictions[name], gt, valid, positions, common=common)
                row['cluster'] = int(task['parent_session'].split('_')[3])
                rows[control][name].append(row)
    tests = {}; metrics = {}
    for control, members in rows.items():
        metrics[control] = {name:summarize_chunks(values) for name, values in members.items()}
        clusters = sorted({row['cluster'] for row in members[part]}); assert len(clusters) == 18
        for metric, index in [('frame_pa', 3), ('RPE_Z', 4)]:
            values = {}; counts = None
            for name in [control, part]:
                groups = [[row for row in members[name] if row['cluster'] == cluster] for cluster in clusters]
                current = [sum(row['count'] for row in group) for group in groups]
                assert counts is None or counts == current
                counts = current
                values[name] = [1000*sum(row['sums_m'][index] for row in group) for group in groups]
            tests[control+'_'+metric] = paired_cluster(values[control], values[part], counts, seed=261029)
    for suffix in ['_frame_pa', '_RPE_Z']:
        adjusted = holm_adjust({key:row['p_value'] for key,row in tests.items() if key.endswith(suffix)})
        for key, value in adjusted.items():
            tests[key]['holm_p_value_within_three_controls'] = value
    result = dict(family=family, checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                  status='complete', part_model=part, controls=controls, segments=80,
                  pair_metrics=metrics, tests=tests, direction='Part dynamic minus each control; negative means lower error',
                  interpretation='Direct full-Test control analysis, with each pair using its own exact-frame0/common population. Holm correction within these three controls per metric, not across all adaptive research versions. No Test-based method/checkpoint selection.',
                  inference_performed=False, selection_performed=False,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [ROOT/'src/utils/part_aggregation_metrics.py', ROOT/'src/utils/paired_statistics.py']})
    folder = OUT/'artifact_audits'; folder.mkdir(exist_ok=True)
    (folder/(family+'_part_vs_controls.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(tests, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

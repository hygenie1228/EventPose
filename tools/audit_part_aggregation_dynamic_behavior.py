"""Check measured Train temporal behavior from completed attention arrays only."""
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['v4', 'v5', 'v6', 'v6_long'], required=True)
    family = parser.parse_args().family
    rows = {}; sources = {}; keys = part_names = None
    for suffix in ['global_dynamic', 'part_dynamic', 'part_static']:
        name = family+'_'+suffix
        path = OUT/'diagnostics'/(name+'_attention.npz')
        diagpath = path.with_name(name+'.json')
        diag = read(diagpath)
        assert diag['Test_used'] is False and diag['frames'] == 288
        assert read(OUT/'runs'/name/'complete.json')['status'] == 'complete'
        with np.load(path, allow_pickle=False) as z:
            current = list(zip(z['sessions'].tolist(), z['frames'].tolist()))
            assert len(current) == len(set(current)) == 288
            if keys is None:
                keys = current; part_names = z['part_names'].tolist()
            assert current == keys and z['part_names'].tolist() == part_names
            w = z['time_weights'].astype(np.float64)
            null = z['null_weights'].astype(np.float64)
            mass = z['event_probability'].astype(np.float64)
            horizon = z['predicted_horizon_ms'].astype(np.float64)
            midpoints = z['bin_midpoints_ms']
            age = z['conditional_bin_midpoint_age_ms'].astype(np.float64)
            threshold = float(z['minimum_event_probability_for_age'])
        assert w.shape == (288, 6, 3) and null.shape == mass.shape == horizon.shape == (288, 6)
        assert np.isfinite(w).all() and np.isfinite(null).all() and np.isfinite(horizon).all()
        assert (w >= 0).all() and (null >= 0).all()
        assert np.allclose(w.sum(-1), mass, atol=1e-6, rtol=0)
        assert np.allclose(mass+null, 1, atol=1e-6, rtol=0)
        assert (horizon >= 33-1e-3).all() and (horizon <= 200+1e-3).all()
        assert threshold == .05
        defined = mass >= threshold
        conditional = np.divide(w, mass[..., None], out=np.zeros_like(w), where=mass[..., None] > 0)
        expected_age = np.sum(conditional*midpoints, axis=-1)
        assert np.isfinite(age[defined]).all()
        assert np.allclose(age[defined], expected_age[defined], atol=1e-4, rtol=0)
        assert np.isnan(age[~defined]).all(), 'Null/weak event mass must not receive an invented zero age'
        between = [float(conditional[i, mask].std(0).mean()) for i, mask in enumerate(defined) if mask.sum() >= 2]
        across = [float(conditional[defined[:, j], j].std(0).mean()) if defined[:, j].sum() > 1 else None for j in range(6)]
        rows[name] = dict(mean_event_probability=float(mass.mean()), age_defined_fraction=float(defined.mean()),
                          between_part_conditional_weight_std_mean=float(np.mean(between)) if between else None,
                          within_part_across_query_conditional_weight_std=across,
                          horizon_across_query_std_ms=horizon.std(0).tolist(),
                          maximum_same_query_between_part_weight_difference=float(np.max(np.ptp(w, axis=1))),
                          final_weight_scope='Shared across all parts' if suffix == 'global_dynamic' else 'Part specific')
        if suffix == 'global_dynamic':
            assert np.max(np.ptp(w, axis=1)) < 1e-6
            assert np.max(np.ptp(null, axis=1)) < 1e-6
        if suffix == 'part_static':
            assert np.max(np.ptp(horizon, axis=0)) < 1e-6
        sources[str(path.relative_to(ROOT))] = sha(path)
        sources[str(diagpath.relative_to(ROOT))] = sha(diagpath)
    dynamic = rows[family+'_part_dynamic']
    demonstrated = (dynamic['between_part_conditional_weight_std_mean'] is not None and
                    dynamic['between_part_conditional_weight_std_mean'] > 1e-6 and
                    min(dynamic['horizon_across_query_std_ms']) > 1e-6 and
                    all(v is not None and v > 1e-6 for v in dynamic['within_part_across_query_conditional_weight_std']))
    result = dict(status='verified_measured_control_invariants', family=family,
                  checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                  split='Train holdout only', frames=288, part_names=part_names,
                  part_and_query_variation_demonstrated=demonstrated, conditions=rows, source_sha256=sources,
                  interpretation='Descriptive measured variation and matched-control invariants; no pose-gain or independent causal proof. Global internal horizons precede averaging; static means static horizon, not constant quality/event-use features.',
                  inference_performed=False, Test_files_read=False)
    folder = OUT/'artifact_audits'; folder.mkdir(exist_ok=True)
    (folder/(family+'_dynamic_behavior.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

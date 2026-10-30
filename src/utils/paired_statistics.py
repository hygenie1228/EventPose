"""Paired cluster uncertainty and familywise correction for frozen Test results."""
import numpy as np


def holm_adjust(pvalues):
    """Holm familywise adjustment; missing tests remain missing, not zero."""
    ordered = sorted(pvalues, key=lambda key: 1. if pvalues[key] is None else pvalues[key])
    adjusted = {}; previous = 0.; total = len(ordered)
    for rank, key in enumerate(ordered):
        value = pvalues[key]
        if value is None:
            adjusted[key] = None
            continue
        assert np.isfinite(value) and 0 <= value <= 1
        previous = max(previous, min(1., (total - rank) * value))
        adjusted[key] = previous
    return adjusted


def paired_cluster(reference_sums, variant_sums, counts, seed=261024, replicates=10000):
    """Bootstrap whole independent clusters, preserving within-cluster frames.

    Sums are mm-valued error sums on exactly the same matched samples. Counts
    are their common sample counts. The effect is variant minus reference.
    A two-sided cluster sign-flip test complements the percentile bootstrap CI;
    it assumes exchangeability/symmetry of paired cluster errors under the null.
    Tiny numbers of clusters cannot acquire tiny p-values from many frames.
    """
    reference = np.asarray(reference_sums, np.float64)
    variant = np.asarray(variant_sums, np.float64)
    counts = np.asarray(counts, np.float64)
    assert reference.shape == variant.shape == counts.shape and counts.ndim == 1
    assert np.isfinite(counts).all() and (counts >= 0).all()
    keep = counts > 0
    reference, variant, counts = reference[keep], variant[keep], counts[keep]
    assert np.isfinite(reference).all() and np.isfinite(variant).all()
    number = len(counts)
    if not number:
        return dict(status='no_common_samples', clusters=0, samples=0, effect=None, ci95=None, p_value=None)
    reference_mean = float(reference.sum() / counts.sum())
    variant_mean = float(variant.sum() / counts.sum())
    difference = variant - reference
    effect = float(difference.sum() / counts.sum())
    result = dict(status='insufficient_clusters' if number < 2 else 'tested', clusters=number,
                  samples=int(counts.sum()), reference_mean=reference_mean, variant_mean=variant_mean,
                  effect=effect, relative_percent=100 * effect / reference_mean if reference_mean else None,
                  ci95=None, p_value=None, test='two-sided whole-cluster sign flip',
                  ci_method='paired whole-cluster percentile bootstrap, weighted by common sample count')
    if number < 2:
        return result
    rng = np.random.default_rng(seed)
    indices = rng.integers(number, size=(replicates, number))
    bootstrap = difference[indices].sum(1) / counts[indices].sum(1)
    result['ci95'] = np.quantile(bootstrap, [.025, .975]).tolist()
    if number <= 16:
        identifiers = np.arange(1 << number, dtype=np.uint64)
        bits = ((identifiers[:, None] >> np.arange(number, dtype=np.uint64)) & 1).astype(np.int8)
        signs = 2 * bits - 1
        null = signs @ difference / counts.sum()
        result['p_value'] = float(np.mean(np.abs(null) >= abs(effect) - 1e-12))
        result['permutations'] = len(null)
        result['exact_sign_flip'] = True
    else:
        permutations = max(50000, replicates)
        signs = 2 * rng.integers(2, size=(permutations, number), dtype=np.int8) - 1
        null = signs @ difference / counts.sum()
        result['p_value'] = float((1 + np.count_nonzero(np.abs(null) >= abs(effect) - 1e-12)) / (1 + permutations))
        result['permutations'] = permutations
        result['exact_sign_flip'] = False
    return result


def participant_components(metadata, sessions):
    """Keep repeated participants and all their co-occurring sessions together."""
    wanted = set(sessions)
    rows = [row for row in metadata if row['session'] in wanted]
    assert {row['session'] for row in rows} == wanted
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for row in rows:
        ids = [str(value) for value in row['participant_ids']]
        assert ids, row['session']
        for value in ids:
            parent[find(value)] = find(ids[0])
    members = {}
    for value in parent:
        members.setdefault(find(value), []).append(value)
    canonical = {root: 'participants_' + '_'.join(sorted(values)) for root, values in members.items()}
    mapping = {row['session']: canonical[find(str(row['participant_ids'][0]))] for row in rows}
    return mapping, {canonical[root]: sorted(values) for root, values in members.items()}


def motion_strata(gt, valid, positions, fps, still_mps=.05, fast_mps=.5, timestamps_ns=None):
    """GT body22 central-difference speed bins, fixed before seeing predictions.

    -1 unknown, 0 still, 1 slow, 2 fast. No segment/gap is bridged. These are
    operational GT-derived bins, not human annotations; GT jitter can affect them.
    """
    gt = np.asarray(gt, np.float64)[:, :, :22]
    valid = np.asarray(valid, bool)
    positions = np.asarray(positions)
    result = np.full(valid.shape, -1, np.int8)
    if len(gt) < 3:
        return result
    consecutive = (np.diff(positions)[:-1] == 1) & (np.diff(positions)[1:] == 1)
    keep = valid[:-2] & valid[1:-1] & valid[2:] & consecutive[:, None]
    dt=(np.asarray(timestamps_ns,dtype=np.int64)[2:]-np.asarray(timestamps_ns,dtype=np.int64)[:-2])/1e9 if timestamps_ns is not None else np.full(len(gt)-2,2/fps)
    assert np.all(dt>0)
    speed = np.linalg.norm((gt[2:] - gt[:-2]) / dt[:,None,None,None], axis=-1).mean(-1)
    labels = np.where(speed <= still_mps, 0, np.where(speed >= fast_mps, 2, 1)).astype(np.int8)
    result[1:-1][keep] = labels[keep]
    return result


def self_check():
    zero = paired_cluster([10] * 10, [10] * 10, [1] * 10, replicates=1000)
    assert zero['effect'] == 0 and zero['p_value'] == 1
    gain = paired_cluster([20] * 10, [10] * 10, [1] * 10, replicates=1000)
    assert gain['ci95'] == [-10., -10.] and gain['p_value'] < .01
    few = paired_cluster([20] * 2, [10] * 2, [1] * 2, replicates=1000)
    assert few['p_value'] == .5, 'Many frames must not create extra independent clusters'
    weighted = paired_cluster([20, 20000], [30, 19900], [1, 1000], replicates=1000)
    assert weighted['ci95'][1] > 1
    missing = paired_cluster([0], [0], [0])
    assert missing['effect'] is None and missing['p_value'] is None
    corrected = holm_adjust({'a': .001, 'b': .02, 'c': .04, 'd': None})
    assert corrected == {'a': .004, 'b': .06, 'c': .08, 'd': None}
    metadata = [dict(session='a', participant_ids=[1]), dict(session='b', participant_ids=[1, 2]),
                dict(session='c', participant_ids=[2]), dict(session='d', participant_ids=[3])]
    mapping, groups = participant_components(metadata, ['a', 'b', 'c', 'd'])
    assert mapping['a'] == mapping['b'] == mapping['c'] != mapping['d'] and len(groups) == 2
    gt = np.zeros((5, 1, 22, 3)); gt[:, :, :, 2] = 3
    labels = motion_strata(gt, np.ones((5, 1), bool), np.arange(5), 30)
    assert (labels[1:-1] == 0).all() and labels[0, 0] == -1
    gap = motion_strata(gt, np.ones((5, 1), bool), np.array([0, 1, 4, 5, 6]), 30)
    assert gap[1, 0] == gap[2, 0] == -1
    return dict(paired_clusters=True, small_cluster_limit=True, missing_not_zero=True,
                weighted_uncertainty=True, holm=True, participant_dependence=True, no_gap_bridge=True)


if __name__ == '__main__':
    print(self_check())

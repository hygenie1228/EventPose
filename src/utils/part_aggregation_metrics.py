"""Body22 metrics for part-aware aggregation, preserving exact chunk frame0.

All inputs and sufficient sums are metres. Only summaries convert to mm.
Input predictions must already be matched to GT identities by an unchanged
native-camera 2D detector matching protocol; this module never rematches people.
"""
from __future__ import annotations

import numpy as np

from .chunk_alignment import METRICS as FOUR_METRICS, aligned_errors

METRICS = list(FOUR_METRICS) + ["root_depth_error_mm"]
PARTS = {
    "torso_head": [0, 3, 6, 9, 12, 15],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
}


def score_chunk(prediction, target, valid, positions, *, common=None):
    """Score one original 10-second chunk, never substituting a later anchor.

    Shapes: prediction/target [frames, GT slots, >=22, 3]; valid/common
    [frames, GT slots]; positions are zero-based source positions in this chunk.
    `common` is the intersection of finite valid detections of compared methods.
    RPE-Z uses unaligned camera Z, and shares the four pose metrics' population.
    """
    p = np.asarray(prediction, np.float64)[..., :22, :]
    t = np.asarray(target, np.float64)[..., :22, :]
    v = np.asarray(valid, bool)
    pos = np.asarray(positions)
    if p.shape != t.shape or p.ndim != 4 or p.shape[-2:] != (22, 3):
        raise ValueError("Predictions and targets must match [T,P,22+,3]")
    if v.shape != p.shape[:2] or pos.shape != (len(p),):
        raise ValueError("GT validity or chunk positions do not match predictions")
    if len(pos) and (np.diff(pos) <= 0).any():
        raise ValueError("Chunk positions must be strictly increasing")
    matched = v & np.isfinite(p).all((2, 3)) & np.isfinite(t).all((2, 3))
    if common is not None:
        shared = np.asarray(common, bool)
        if shared.shape != v.shape or np.any(shared & ~matched):
            raise ValueError("Common population includes an invalid or missing prediction")
        matched &= shared
    first = np.flatnonzero(pos == 0)
    eligible = matched[first[0]].copy() if len(first) == 1 else np.zeros(v.shape[1], bool)
    selected = matched & eligible[None]
    sums = np.zeros(5, np.float64)
    series = np.full((*v.shape, 5), np.nan, np.float64)
    root_relative = (p-p[:, :, :1])-(t-t[:, :, :1])
    part_sums = dict.fromkeys(PARTS, 0.)
    for person in np.flatnonzero(eligible):
        keep = selected[:, person]
        errors = aligned_errors(p[keep, person], t[keep, person], p[first[0], person], t[first[0], person])
        depth = np.abs(p[keep, person, 0, 2]-t[keep, person, 0, 2])
        series[keep, person] = np.column_stack([errors, depth])
        sums += series[keep, person].sum(0)
    for name, ids in PARTS.items():
        part_sums[name] = float(np.linalg.norm(root_relative[selected][:, ids], axis=-1).mean(-1).sum())
    if not np.isfinite(sums).all():
        raise ValueError("Non-finite metric sum")
    return dict(count=int(selected.sum()), anchors=int(eligible.sum()),
                sums_m=sums.tolist(), matched_instances=int(matched.sum()),
                valid_instances=int(v.sum()),
                excluded_person_chunks=int((v.any(0) & ~eligible).sum()),
                part_root_aligned_sums_m=part_sums), series, selected


def summarize_chunks(rows):
    """Person-frame weighted means; missing/unmatched samples never get zero error."""
    rows = list(rows)
    count = sum(r["count"] for r in rows)
    sums = np.asarray([r["sums_m"] for r in rows], np.float64).reshape(-1, 5).sum(0)
    valid = sum(r["valid_instances"] for r in rows)
    matched = sum(r["matched_instances"] for r in rows)
    return dict(samples=count, anchor_people=sum(r["anchors"] for r in rows),
                excluded_person_chunks=sum(r["excluded_person_chunks"] for r in rows),
                **{k: float(1000*sums[i]/count) if count else None for i, k in enumerate(METRICS)},
                coverage=float(matched/valid) if valid else None,
                matched_instances=matched, valid_instances=valid,
                part_root_aligned_mpjpe_mm={k: float(1000*sum(r["part_root_aligned_sums_m"][k] for r in rows)/count)
                                           if count else None for k in PARTS})


def common_detection_mask(predictions, target, valid):
    """One population for all methods; infer outputs must be matched beforehand."""
    mask = np.asarray(valid, bool) & np.isfinite(np.asarray(target)[..., :22, :]).all((2, 3))
    for prediction in predictions.values():
        mask &= np.isfinite(np.asarray(prediction)[..., :22, :]).all((2, 3))
    return mask


def self_check():
    rng = np.random.default_rng(261029)
    t = rng.normal(size=(4, 1, 22, 3))*.2 + [0, 0, 3]
    valid = np.ones((4, 1), bool); positions = np.arange(4)
    row, _, _ = score_chunk(t, t, valid, positions)
    assert np.max(np.abs(row["sums_m"])) < 1e-12 and row["count"] == 4
    p = t+[.1, .2, .3]
    row, _, _ = score_chunk(p, t, valid, positions)
    assert np.max(np.abs(row["sums_m"][:4])) < 1e-12
    assert abs(row["sums_m"][4]-.3*4) < 1e-12
    assert abs(summarize_chunks([row])["root_depth_error_mm"]-300) < 1e-8
    p = t.copy(); p[1:] += [.1, 0, 0]
    row, _, _ = score_chunk(p, t, valid, positions)
    assert abs(row["sums_m"][0]-.3) < 1e-12 and row["sums_m"][2] < 1e-12
    p[0] = np.nan
    row, _, selected = score_chunk(p, t, valid, positions)
    assert row["count"] == 0 and row["excluded_person_chunks"] == 1 and not selected.any()
    row, _, _ = score_chunk(t[1:], t[1:], valid[1:], positions[1:])
    assert row["count"] == 0, "Do not substitute the first available frame for exact frame0"
    common = common_detection_mask(dict(a=t, b=p), t, valid)
    assert not common[0].any() and common[1:].all()
    assert summarize_chunks([])["root_depth_error_mm"] is None
    return dict(identity=True, aligned_translation=True, RPE_Z_not_3D_norm=True,
                exact_frame0=True, fixed_anchor=True, common_missing_detections=True,
                empty_population_no_fake_zero=True, body22=True)


if __name__ == "__main__":
    print(self_check())

"""Diagnose raw prediction jitter on complete Test clips; never select models."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from multihmr.track2_data import OUTPUT, save_json, sha256


def main():
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    models = {}
    for trial in ['diagnostic_original_native_rgb', 'diagnostic_original_static_crop',
                  'native_last2_cd100_rel100']:
        clips = {}
        for sample in manifest['test']:
            folder = OUTPUT / 'predictions' / trial / sample
            info = json.loads((folder / 'inference.json').read_text())
            assert not info['target_foreground_only_input']
            assert not info['gt_used_for_detection_or_tracking']
            assert info['all_accepted_heads_in_target_mask']
            path = folder / 'smplx_predictions.npz'
            assert sha256(path) == info['result_sha256']
            with np.load(path) as data:
                joints = data['joints_cam_m'][:, :22].astype(np.float64)
                valid = data['detection_valid']
                confidence = data['confidence']
                assert np.array_equal(data['frames'], np.arange(len(joints)))
            assert len(joints) == manifest['samples'][sample]['frames']
            assert np.isfinite(joints).all()
            step = np.linalg.norm(np.diff(joints[:, 0], axis=0), axis=-1) * 100
            acceleration = np.linalg.norm(np.diff(joints, n=2, axis=0), axis=-1).mean(axis=-1) * 100
            largest = np.argsort(step)[-5:][::-1]
            clips[sample] = dict(
                frames=len(joints), missed_detections=int((~valid).sum()),
                raw_root_step_mean_cm=float(step.mean()),
                raw_root_step_p99_cm=float(np.quantile(step, .99)),
                raw_root_step_max_cm=float(step.max()),
                raw_body22_acceleration_magnitude_mean_cm_per_frame2=float(acceleration.mean()),
                largest_root_steps=[dict(
                    frames=[int(i), int(i + 1)], distance_cm=float(step[i]),
                    both_detected=bool(valid[i] and valid[i + 1]),
                    confidence=confidence[i:i + 2].tolist(),
                    raw_root_camera_m=joints[i:i + 2, 0].tolist()) for i in largest],
                raw_prediction_sha256=info['result_sha256'],
                evaluation_protocol=info['evaluation_protocol'])
        assert len(clips) == 10 and sum(v['frames'] for v in clips.values()) == 7610
        key = 'raw_body22_acceleration_magnitude_mean_cm_per_frame2'
        models[trial] = dict(sequences=10, frames=7610, per_sequence=clips,
                            equal_sequence_mean_raw_acceleration_magnitude=sum(v[key] for v in clips.values()) / 10)
    result = dict(models=models, GT_read=False, test_used_for_selection=False,
                  diagnosis_only=True, source_exports_changed=False,
                  caveat='Unaligned raw prediction acceleration magnitude, not ACC-H error. '
                         'Largest steps alone do not prove identity switches. '
                         'All frames include the frozen inference missing-detection completion.',
                  completed_unix=time.time())
    save_json(OUTPUT / 'raw_temporal_stability_diagnostic.json', result)
    print(json.dumps({trial: {k: v for k, v in value.items() if k != 'per_sequence'}
                      for trial, value in models.items()}))


if __name__ == '__main__':
    main()

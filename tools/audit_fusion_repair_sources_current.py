"""Compare frozen assets with the actual currently mounted read-only dataset."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write


def main():
    goal = json.loads((ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal, os.getpid(), 'fusion_repair_current_dataset_audit')
    import h5py
    import numpy as np
    import cv2
    from utils.event_encoding import calibration
    cv2.setNumThreads(1)
    out = Path(goal['output_root']); manifest = json.loads((out / 'manifest.json').read_text())
    assert hashlib.sha256((ROOT / 'dataset/split.json').read_bytes()).hexdigest() == manifest['split_sha256']
    records = []
    for session in manifest['training_sessions'] + manifest['validation_sessions'] + manifest['test_sessions']:
        folder = ROOT / 'dataset' / session
        with np.load(out / 'source_maps' / (session + '.npz')) as z:
            maps = {k: z[k] for k in z.files}
        with np.load(out / 'raw' / (session + '.npz')) as z:
            raw = {k: z[k] for k in z.files}
        with h5py.File(out / 'cache' / (session + '.h5'), 'r', locking=False) as f:
            cache_started = float(f.attrs['started_unix'])
        with np.load(folder / 'device0/event.npz') as z:
            assert np.array_equal(z['rgb__timestamp_ns'], maps['canonical_rgb_timestamp_ns'])
        with h5py.File(folder / 'device0/label.h5', 'r', locking=False) as f:
            assert f.attrs['complete'] and f.attrs['coordinate_unit'] == 'metre'
            positions = f['frames/source_rgb_position'][:]
            assert np.array_equal(positions, maps['gt_source_positions'])
            assert np.array_equal(f['frames/source_rgb_timestamp_ns'][:], maps['gt_timestamp_ns'])
            valid = f['params/person_valid'][:].astype(bool)
            assert np.array_equal(valid, maps['gt_valid'])
            indices = np.searchsorted(positions, raw['positions'])
            safe = np.minimum(indices, len(positions)-1)
            matched = (indices < len(positions)) & (positions[safe] == raw['positions'])
            assert np.array_equal(raw['valid'], valid[safe] & matched[:, None])
            for stored, source in [('poses', 'smplx_pose_axis_angle'), ('betas', 'smplx_betas'), ('transl', 'translation_world_m')]:
                current = f['params/' + source][:][safe].astype(np.float32)
                assert np.array_equal(current, raw[stored], equal_nan=True), (session, source)
        sources = []
        for device in (3, 4):
            prefix = 'device' + str(device) + '_'; source = folder / ('device' + str(device))
            with np.load(source / 'event.npz') as z:
                assert np.array_equal(z['timestamp_ns'], maps[prefix+'event_timestamp_ns'])
                assert np.array_equal(z['rgb__timestamp_ns'], maps[prefix+'rgb_timestamp_ns'])
                cameras = calibration(z)
                for kind in ('rgb', 'event'):
                    R = z['rgb_calibration__'+kind+'__rotation_matrix'].astype(np.float64)
                    t = z['rgb_calibration__'+kind+'__translation'].astype(np.float64).reshape(3)
                    assert np.array_equal(R, maps[prefix+kind+'_R']) and np.array_equal(t, maps[prefix+kind+'_t'])
                    assert np.array_equal(cameras['K_'+kind], maps[prefix+kind+'_K'])
            cap = cv2.VideoCapture(str(source / 'rgb.mp4')); assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(maps[prefix+'rgb_timestamp_ns'])
            assert np.array_equal([int(cap.get(4)), int(cap.get(3))], maps[prefix+'rgb_shape'])
            cap.release()
            for file in (source / 'event.npz', source / 'rgb.mp4'):
                stat = file.stat()
                assert stat.st_mtime_ns <= int(cache_started*1e9), (str(file), 'Source changed after cache construction')
                sources.append(dict(path=str(file.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        records.append(dict(session=session, all_cached_GT_parameters_equal_current_source=True,
                            corrected_clocks_and_calibrations_equal_current_source=True,
                            source_archives=sources, input_archive_modified_after_cache=False))
        print(json.dumps(dict(current_dataset_verified=session, count=len(records))), flush=True)
    write(out / 'current_dataset_audit.json', dict(status='passed', checked_unix=time.time(),
          source=str((ROOT/'dataset').resolve()), split_sha256=manifest['split_sha256'], records=records,
          scope='All frozen cached GT parameters and current dataset clocks/calibration/video metadata compared. Cache full-byte hashes verified separately; original RGB/event archives not fully rehashed.'))
    print(json.dumps(dict(status='passed', sessions=len(records))), flush=True)


if __name__ == '__main__':
    main()

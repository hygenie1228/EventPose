"""Read-only device0 input audit and resumable four-encoding experiment cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from utils.event_encoding import ENCODINGS, EventFrames, calibration, self_check

OUT = ROOT / 'exp/261024/rgb_event_36h'
RGB_CACHE = ROOT / '.experiment_storage/multihmr_finetune_14h'
TEST = ROOT / '.experiment_storage/humanpose_gt_segments/tasks.json'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); temp.replace(path)


def fingerprint(path):
    with zipfile.ZipFile(path) as z:
        entries = [(item.filename, item.CRC, item.file_size) for item in z.infolist()]
    stat = path.stat()
    fields = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, zip_entry_crcs=entries)
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def plan(size):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT / 'exp/261023/multihmr_finetune_14h/manifest.json')
    groups = {}
    for role in ('training', 'validation'):
        for row in read(RGB_CACHE / (role + '_index.json')):
            key = row['session']
            groups.setdefault(key, {'session': key, 'role': role, 'frames': set()})['frames'].add(row['frame'])
    for task in read(TEST):
        key = task['parent_session']
        assert key not in groups or groups[key]['role'] == 'test'
        groups.setdefault(key, {'session': key, 'role': 'test', 'frames': set()})['frames'].update(
            range(task['start_frame'], task['end_frame']))
    assert len(groups) == 283
    assert {key for key, value in groups.items() if value['role'] == 'training'} == set(manifest['training_sessions'])
    assert {key for key, value in groups.items() if value['role'] == 'validation'} == set(manifest['validation_sessions'])
    assert {key for key, value in groups.items() if value['role'] == 'test'} == set(manifest['test_sessions'])
    config = dict(version=2, size=size, count_cap=256, encodings=ENCODINGS,
                  quantization='round(255*x), uint8; reader divides by 255',
                  raster='split counts summed into target pixels before log; surface keeps most recent observation',
                  gray='native gray parity with visualize_data.py, then INTER_AREA and neutral padding128',
                  windows='causal (t-window,t], exact int64-ns bin endpoints; latest stored event <= RGB corrected timestamp',
                  coordinate_frame='rectified event camera, never treated as RGB-aligned',
                  split_sha256=manifest['test_split_sha256'])
    config['sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    local = ROOT / '.experiment_storage/rgb_event_36h' / ('event_cache_' + config['sha256'][:12])
    local.mkdir(parents=True, exist_ok=True)
    link = OUT / 'event_cache'
    if not link.exists():
        link.symlink_to(local, target_is_directory=True)
    assert link.resolve() == local.resolve(), 'Existing cache uses a different frozen configuration'
    sessions = [dict(value, frames=sorted(value['frames'])) for value in groups.values()]
    # Smaller holdout first permits an early model pilot without waiting for Train.
    order = {'validation': 0, 'training': 1, 'test': 2}
    sessions.sort(key=lambda value: (order[value['role']], value['session']))
    result = dict(config=config, sessions=sessions, cache_root=str(link),
                  frames_by_role={role: sum(len(value['frames']) for value in sessions if value['role'] == role)
                                  for role in order}, dataset_read_only=True)
    save(OUT / 'event_cache_plan.json', result)
    analysis = ROOT / '_agents/report/261025_이벤트데이터분석.md'
    snapshot = ROOT / '_agents/research/261024/event_analysis_adopted_snapshot.md'
    if not snapshot.exists():
        snapshot.write_text(analysis.read_text())
    save(OUT / 'event_analysis_source.json', dict(path=str(analysis), snapshot=str(snapshot),
         sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), adopted_unix=time.time()))
    return result


def inspect_session(item):
    import h5py
    source = ROOT / 'dataset' / item['session'] / 'device0' / 'event.npz'
    with np.load(source, allow_pickle=False) as z:
        cameras = calibration(z)
        times = z['timestamp_ns'].astype(np.int64)
        rgb = z['rgb__timestamp_ns'].astype(np.int64)
        assert np.all(np.diff(times) >= 0) and np.all(np.diff(rgb) > 0)
        frames = np.array(item['frames'], np.int64)
        assert frames.min() >= 0 and frames.max() < len(rgb)
        targets = rgb[frames]
        rows = np.searchsorted(times, targets, side='right') - 1
        safe = np.maximum(rows, 0)
        available = (rows >= 0) & ((targets - times[safe]) <= 20_000_000)
        assert np.all(times[safe[rows >= 0]] <= targets[rows >= 0])
        result = dict(session=item['session'], role=item['role'], source=str(source),
                      source_fingerprint=fingerprint(source),
                      native_event_shape=z['image_shape'].tolist(), event_frames=len(times), rgb_frames=len(rgb),
                      requested_frames=len(frames), covered_frames=int(available.sum()),
                      noncovered_frames=int((~available).sum()),
                      event_start_ns=int(times[0]), event_end_ns=int(times[-1]),
                      rgb_start_ns=int(rgb[0]), rgb_end_ns=int(rgb[-1]),
                      camera_baseline_m=float(np.linalg.norm(cameras['t_rgb_to_event'])),
                      calibration={key: value.tolist() for key, value in cameras.items()})
        with h5py.File(source.parent / 'label.h5', 'r') as labels:
            positions = labels['frames/source_rgb_position'][:]
            assert np.array_equal(labels['frames/source_rgb_timestamp_ns'][:], rgb[positions]), item['session']
            result['label_rgb_timestamp_mapping_verified'] = True
    return result


def cache_session(arguments):
    import cv2
    import h5py
    cv2.setNumThreads(1)
    item, config, cache_root, deadline = arguments
    destination = Path(cache_root) / (item['session'] + '.h5')
    source = ROOT / 'dataset' / item['session'] / 'device0' / 'event.npz'
    source_fingerprint = fingerprint(source)
    if destination.exists():
        with h5py.File(destination, 'r') as f:
            assert f.attrs['config_sha256'] == config['sha256']
            assert f.attrs['source_fingerprint'] == source_fingerprint
            if f.attrs.get('complete', False):
                return dict(session=item['session'], role=item['role'], frames=len(item['frames']),
                            bytes=destination.stat().st_size, reused=True)
    started = time.perf_counter()
    observations = EventFrames.load(source, size=config['size'], count_cap=config['count_cap'])
    load_seconds = time.perf_counter() - started
    frames = np.array(item['frames'], np.int64)
    assert frames.max() < len(observations.rgb_times)
    partial = destination.with_suffix('.partial.h5')
    with h5py.File(partial, 'a') as f:
        if 'frames' not in f:
            f.attrs.update(config_sha256=config['sha256'], source_fingerprint=source_fingerprint,
                           complete=False, role=item['role'], source=str(source))
            f.create_dataset('frames', data=frames)
            f.create_dataset('timestamp_ns', data=observations.rgb_times[frames])
            f.create_dataset('written', shape=(len(frames),), dtype=np.bool_)
            f.create_dataset('event_available', shape=(len(frames),), dtype=np.bool_)
            f.create_dataset('latest_event_timestamp_ns', shape=(len(frames),), dtype=np.int64)
            for name, spec in ENCODINGS.items():
                shape = (len(frames), spec['channels'], config['size'], config['size'])
                f.create_dataset(name, shape=shape, dtype=np.uint8, chunks=(1, *shape[1:]), compression='lzf')
            for name, value in observations.cameras.items():
                f.create_dataset(name, data=value)
            f.create_dataset('event_pixel_transform', data=observations.pixel_transform)
            f.create_dataset('native_event_shape', data=observations.native_shape)
        assert f.attrs['config_sha256'] == config['sha256'] and np.array_equal(f['frames'][:], frames)
        assert f.attrs['source_fingerprint'] == source_fingerprint
        for row in np.flatnonzero(~f['written'][:]):
            if time.time() >= deadline - 30:
                raise RuntimeError('User wall deadline reached; resumable partial cache preserved')
            encoded, metadata = observations.encode(int(observations.rgb_times[frames[row]]))
            for name, value in encoded.items():
                assert value.shape == (ENCODINGS[name]['channels'], config['size'], config['size'])
                assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
                f[name][row] = np.rint(255 * value).astype(np.uint8)
            f['event_available'][row] = metadata['event_available']
            f['latest_event_timestamp_ns'][row] = metadata['latest_event_timestamp_ns']
            f['written'][row] = True
            if row % 64 == 0:
                f.flush()
        assert f['written'][:].all()
        valid = f['event_available'][:]
        assert np.all(f['latest_event_timestamp_ns'][:][valid] <= f['timestamp_ns'][:][valid])
        f.attrs.update(complete=True, completed_unix=time.time(), load_seconds=load_seconds,
                       total_seconds=time.perf_counter() - started)
        f.flush()
    partial.replace(destination)
    return dict(session=item['session'], role=item['role'], frames=len(frames),
                bytes=destination.stat().st_size, load_seconds=load_seconds,
                total_seconds=time.perf_counter() - started, reused=False)


def main(args):
    current = plan(args.size)
    check = self_check()
    save(OUT / 'event_encoding_self_checks.json', check)
    sessions = current['sessions'][:args.limit] if args.limit else current['sessions']
    goal = read(ROOT / '_agents/research/261024/rgb_event_goal_36h.json')
    context = mp.get_context('spawn')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        if args.stage == 'audit':
            futures = [pool.submit(inspect_session, item) for item in sessions]
        else:
            futures = [pool.submit(cache_session, (item, current['config'], current['cache_root'],
                                                   goal['deadline_unix'])) for item in sessions]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            save(OUT / ('event_' + args.stage + '_status.json'),
                 dict(status='running', updated_unix=time.time(), complete_sessions=len(results),
                      total_sessions=len(sessions), last_session=result['session']))
            print(json.dumps(result), flush=True)
    summary = dict(status='complete', completed_unix=time.time(), sessions=len(results),
                   config=current['config'], results=results, limited=bool(args.limit),
                   frames_by_role=current['frames_by_role'])
    save(OUT / ('event_' + args.stage + '_summary.json'), summary)
    print('Completed', args.stage, len(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'build'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--limit', type=int, default=0)
    main(parser.parse_args())

"""Full artifact audit; scientific recommendation/seed review remain explicit."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
OUT = ROOT / 'exp/261024/rgb_event_36h'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def equal(before, after, location='root'):
    if isinstance(before, dict):
        assert before.keys() == after.keys(), location
        for key in before:
            if key != 'updated_unix':
                equal(before[key], after[key], location + '.' + key)
    elif isinstance(before, list):
        assert len(before) == len(after), location
        for index, (a, b) in enumerate(zip(before, after)):
            equal(a, b, location + '[' + str(index) + ']')
    elif isinstance(before, (float, int)) and not isinstance(before, bool):
        assert math.isfinite(before) and math.isclose(before, after, rel_tol=1e-10, abs_tol=1e-10), location
    else:
        assert before == after, location


def audit_cache(expected_sha256):
    import h5py
    import numpy as np
    from utils.event_encoding import calibration, ENCODINGS
    from tools.prepare_event_cache import fingerprint
    cache_plan = read(OUT / 'event_cache_plan.json')
    assert cache_plan['config']['version'] == 2
    assert cache_plan['config']['sha256'] == expected_sha256
    # Every completed session cache must retain the frozen source/config identity.
    cache_rows = []
    planned_sessions = {item['session']: item for item in cache_plan['sessions']}
    for path in sorted((OUT / 'event_cache').glob('*.h5')):
        with h5py.File(path, 'r') as handle:
            assert handle.attrs['complete'] and handle.attrs['config_sha256'] == expected_sha256
            item = planned_sessions[path.stem]
            source = ROOT / 'dataset' / path.stem / 'device0/event.npz'
            assert handle.attrs['source_fingerprint'] == fingerprint(source)
            frame_ids = handle['frames'][:]
            assert np.array_equal(frame_ids, item['frames']) and handle['written'][:].all()
            with np.load(source, allow_pickle=False) as archive:
                timestamps = archive['rgb__timestamp_ns'][frame_ids]
                event_times = archive['timestamp_ns']
                assert np.array_equal(handle['timestamp_ns'][:], timestamps)
                latest = np.searchsorted(event_times, timestamps, side='right') - 1
                latest_times = np.where(latest >= 0, event_times[np.maximum(latest, 0)], -1)
                available = (latest >= 0) & (timestamps - latest_times <= 20_000_000)
                assert np.array_equal(handle['event_available'][:], available)
                assert np.array_equal(handle['latest_event_timestamp_ns'][:], latest_times)
                for key, value in calibration(archive).items():
                    assert np.allclose(handle[key][:], value, atol=1e-7)
            for encoding, spec in ENCODINGS.items():
                assert handle[encoding].dtype == np.uint8
                assert handle[encoding].shape == (len(frame_ids), spec['channels'], 224, 224)
            cache_rows.append(dict(session=path.stem, bytes=path.stat().st_size,
                                   frames=len(handle['frames']),
                                   load_seconds=float(handle.attrs['load_seconds']),
                                   total_seconds=float(handle.attrs['total_seconds']),
                                   completed_unix=float(handle.attrs['completed_unix'])))
    assert len(cache_rows) == 283
    return cache_rows


def main():
    import h5py
    import numpy as np
    import torch
    from multihmr.event_data import save
    from utils.compare_event_methods import main as recompute
    from utils.event_report import render, REPORT, supporting_evidence
    from utils.event_encoding import calibration, ENCODINGS
    from tools.prepare_event_cache import fingerprint
    conditions = read(OUT / 'experiment_manifest.json')['conditions']
    tasks = read(ROOT / '.experiment_storage/humanpose_gt_segments/tasks.json')
    assert len(conditions) == 10 and len(tasks) == 320
    assert sum(t['video_frames'] for t in tasks) == 96124
    plan = read(OUT / 'statistical_analysis_plan.json')
    assert len(plan['clusters']) == 11 and len(plan['session_to_cluster']) == 64
    training = {}; provenance = {}
    for config in conditions:
        name = config['name']
        if name == 'rgb_original':
            continue
        complete = read(OUT / 'runs' / name / 'complete.json')
        assert complete['status'] == 'complete' and complete['successful_steps'] > 0
        assert complete['test_used_for_selection'] is False
        checkpoint = Path(complete['checkpoint'])
        assert checkpoint.resolve() == (OUT / 'checkpoints' / name / 'best.pt').resolve()
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['successful_steps'] > 0 and saved['successful_steps'] <= complete['successful_steps']
        assert saved['test_used_for_selection'] is False and saved['metrics_apply_to_checkpoint'] is True
        assert saved['metrics_step'] == saved['step']
        for key in ('name', 'mode', 'encoding', 'seed', 'cache_sha256'):
            assert saved['config'].get(key) == config.get(key), (name, key)
        delta = saved['model_delta']
        assert delta and all(torch.isfinite(tensor).all() for tensor in delta.values())
        provenance[name] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
                                selected_step=saved['step'], successful_steps=complete['successful_steps'],
                                elapsed_seconds=complete['elapsed_seconds'])
        training[name] = complete
        del saved, delta
    checked = 0
    for task in tasks:
        for config in conditions:
            folder = OUT / 'results' / config['name'] / task['session']
            metadata = read(folder / 'complete.json')['task']
            assert not metadata.get('smoke')
            for key in ('start_frame', 'end_frame', 'video_frames', 'parent_session', 'gt_file'):
                assert metadata[key] == task[key], (config['name'], task['session'], key)
            assert metadata['input_mode'] == config['mode']
            assert metadata['event_encoding'] == config.get('encoding')
            assert metadata['rgb_pixels_used'] is (config['mode'] != 'event')
            expected = 'original' if config.get('original') else str(OUT / 'checkpoints' / config['name'] / 'best.pt')
            assert metadata['checkpoint'] == expected
            checked += 1
    cache_rows = audit_cache(conditions[0]['cache_sha256'])
    runtime = read(OUT / 'runtime/summary.json')
    expected = {config['name'] + '_' + source for config in conditions
                for source in (['cached'] if config['mode'] == 'rgb' else ['cached', 'online'])}
    assert runtime.keys() == expected and len(runtime) == 18
    assert len({row['gpu_identity'] for row in runtime.values()}) == 1
    previous = 0
    for row in sorted(runtime.values(), key=lambda item: item['process_started_unix']):
        assert row.get('physical_gpu',row['cuda_visible_devices']) in ('2', '3', '4', '5', '6')
        assert row['process_started_unix'] >= previous
        previous = row['process_finished_unix']
        assert previous > row['process_started_unix']
        assert row['frames'] == 300 and len(row['segments']) == 3
        assert math.isclose(row['fps'], 300 / sum(segment['seconds'] for segment in row['segments']))
        assert all(segment['frames'] == 100 for segment in row['segments'])
    before = read(OUT / 'comparison.json')
    recompute()  # Loads and validates every one of the 3200 raw Test files.
    after = read(OUT / 'comparison.json')
    equal(before, after)
    prior, cache = supporting_evidence()
    assert REPORT.read_text() == render(after, runtime, training, prior, cache), 'Generated report differs from actual artifacts'
    save(OUT / 'artifact_audit.json', dict(status='artifacts_passed_scientific_review_required',
        checked_unix=time.time(), raw_test_results=checked, full_raw_recomputation=True,
        report_exact_render_match=True, checkpoint_provenance=provenance,
        cache=dict(sessions=283, bytes=sum(row['bytes'] for row in cache_rows), sessions_metadata=cache_rows),
        runtime_processes=18, goal_complete=False,
        remaining=['cache wall/CPU cost interpretation', 'repeat seed feasibility and limitations',
                   'single recommendation and practical tradeoff review', 'final requirement audit']))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    if args.cache_only:
        from multihmr.event_data import save
        plan = read(OUT / 'event_cache_plan.json')
        rows = audit_cache(plan['config']['sha256'])
        save(OUT / 'cache_final_audit.json', dict(status='passed', checked_unix=time.time(),
             config_sha256=plan['config']['sha256'], sessions=len(rows),
             bytes=sum(row['bytes'] for row in rows), source_mapping_causality_calibration_verified=True,
             note='Numerical encoding parity is covered by event_input_parity.json; this audit checks all session metadata',
             sessions_metadata=rows))
    else:
        main()

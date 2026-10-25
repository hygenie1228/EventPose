"""Audit final experiment evidence; run only after training and evaluation finish."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
NLF = ROOT / 'exp/261024/nlf_finetune_12h'
REPORT = ROOT / '_agents/report/261024_인체자세추정_모델_성능_비교.md'
METHODS = {
    'Multi-HMR pretrained': 'multihmr_pretrained',
    'NLF pretrained': 'nlf_pretrained',
    'Multi-HMR fine-tuned': 'multihmr_finetuned',
    'NLF fine-tuned': 'nlf_finetuned',
}


def read(path):
    return json.loads(path.read_text())


def same_values(before, after, location='result'):
    """Check all saved numeric measurements against full raw-data recomputation."""
    if isinstance(before, dict):
        assert set(before) == set(after), location
        for key in before:
            if key != 'updated_unix':
                same_values(before[key], after[key], location + '.' + key)
    elif isinstance(before, list):
        assert len(before) == len(after), location
        for index, (a, b) in enumerate(zip(before, after)):
            same_values(a, b, location + '[' + str(index) + ']')
    elif isinstance(before, (float, int)) and not isinstance(before, bool):
        assert math.isfinite(before) and math.isclose(before, after, rel_tol=1e-10, abs_tol=1e-10), location
    else:
        assert before == after, location


def main():
    plan = read(NLF / 'training_plan.json')
    assert plan['end_unix'] == 1792849625.8986695
    assert plan['test_used_for_selection'] is False
    assert time.time() >= plan['end_unix'], 'Training deadline has not arrived'
    scores = {}
    for config in plan['configs']:
        run = NLF / 'runs' / config['name']
        complete = read(run / 'complete.json')
        best = read(run / 'best.json')
        assert complete['successful_steps'] > 0 and best['successful_steps'] > 0
        assert math.isfinite(best['pa_mm'])
        assert math.isclose(complete['best_pa_mm'], best['pa_mm'], abs_tol=1e-8)
        scores[config['name']] = best['pa_mm']
    selected = read(NLF / 'selected.json')
    assert selected['test_used_for_selection'] is False
    assert scores[selected['config']['name']] == min(scores.values())
    checkpoint = Path(selected['checkpoint'])
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    metadata = read(checkpoint.with_suffix('.json'))
    verification = read(checkpoint.with_suffix('.verification.json'))
    assert metadata['test_used_for_selection'] is False
    assert Path(metadata['source_delta']).resolve() == (NLF / 'checkpoints' / selected['config']['name'] / 'best.pt').resolve()
    assert verification['status'] == 'passed' and verification['changed_tensors'] > 0
    assert verification['step'] == metadata['step']
    before = read(NLF / 'comparison.json')
    assert set(before['methods']) == set(METHODS), 'All four final models are required'
    assert before['test_sessions'] == 64 and before['segments'] == 320
    assert before['rgb_frames'] == 96124 and before['unit'] == 'mm'
    assert 'final_test/multihmr_finetuned' in before['method_sources']['Multi-HMR fine-tuned']
    tasks = read(ROOT / '.experiment_storage/humanpose_gt_segments/tasks.json')
    for source in before['method_sources'].values():
        for task in tasks:
            folder = Path(source) / task['session']
            assert (folder / 'result.npz').is_file(), str(folder)
            complete = read(folder / 'complete.json')
            assert not complete['task'].get('smoke')
            assert complete['task']['video_frames'] == task['video_frames']
    benchmark = read(NLF / 'runtime_benchmark/summary.json')
    assert set(benchmark) == set(METHODS.values())
    identities = {value['gpu_identity'] for value in benchmark.values()}
    assert len(identities) == 1
    runtime_tasks = read(NLF / 'runtime_benchmark/tasks.json')
    expected_sessions = [task['parent_session'] for task in runtime_tasks]
    assert len(expected_sessions) == len(set(expected_sessions)) == 3
    previous_end = 0
    for method in METHODS.values():
        value = benchmark[method]
        assert value['cuda_visible_devices'] in ('2','3','4','5','6') and value['frames'] == 300
        assert value['process_started_unix'] >= previous_end
        previous_end = value['process_finished_unix']
        assert previous_end > value['process_started_unix']
        assert value['seconds'] > 0 and value['load_seconds'] > 0
        assert value['cold_process_wall_seconds'] >= value['seconds']
        assert math.isclose(value['fps'], 300 / value['seconds'], rel_tol=1e-12)
        assert [segment['session'] for segment in value['segments']] == expected_sessions
        assert all(segment['frames'] == 100 and segment['peak_allocated_bytes'] > 0
                   for segment in value['segments'])
    original_report = REPORT.read_text()
    from utils.compare_methods import main as recompute
    # The existing aggregation checks every frame mapping and matched index and
    # recomputes all pose/absolute/temporal errors from the 1,280 raw results.
    with contextlib.redirect_stdout(io.StringIO()):
        recompute()
    after = read(NLF / 'comparison.json')
    same_values(before, after)
    assert original_report == REPORT.read_text(), 'Report differed from regenerated raw-data report'
    assert original_report.count('**우선 추천:') == 1
    for value in after['methods'].values():
        assert 0 <= value['coverage'] <= 1
        assert value['velocity_pairs'] > 0 and value['acceleration_triples'] > 0
    evidence = dict(status='passed',verified_unix=time.time(),training_scores=scores,
                    selected=selected['config']['name'],models=list(METHODS),
                    segments_per_model=320,total_raw_results=1280,rgb_frames=96124,
                    common_matched_instances=after['common_matched_instances'],
                    gpu_identity=next(iter(identities)),runtime_frames_per_model=300,
                    report=str(REPORT),recommendation=after['recommendation'],
                    scope='Complete raw-data recomputation and report equality; dataset read-only')
    (NLF / 'final_audit.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()

"""Independently verify complete fixed-view Train4 or Test10; no selection."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from multihmr.track2_data import OUTPUT, SOURCE, save_json, sha256


def read(path):
    return json.loads(path.read_text())


def main(new_only=False, split='validation', family='fixed'):
    manifest = read(OUTPUT / 'manifest.json')
    prefix = 'native_surface' if family == 'native-surface' else 'fixed'
    plan = read(OUTPUT / ('native_surface_adaptation_plan.json' if prefix == 'native_surface' else 'fixed_view_adaptation_plan.json'))
    view = read(OUTPUT / 'fixed_view_all30_plan.json')
    reference = read(OUTPUT / 'native_selected.json')['models']['diagnostic_original_native_rgb']['per_sequence'][manifest['validation'][0]]['protocol']
    is_test = split == 'test'
    roster = manifest[split]
    count, frames = (10, 7610) if is_test else (4, 2714)
    assert len(roster) == count and sum(manifest['samples'][s]['frames'] for s in roster) == frames
    destination = OUTPUT / (prefix + ('_completed_Test_models.json' if is_test else '_completed_Train4_models.json'))
    result = read(destination) if destination.exists() else dict(models={}, model_selection_performed=False)
    added = []
    names = (['diagnostic_original_static_crop'] if prefix == 'fixed' else []) + [c['name'] for c in plan['trials']]
    for name in names:
        if new_only and name in result['models']:
            continue
        original = name == 'diagnostic_original_static_crop'
        trial = name if original or is_test else 'search_' + name
        if not all((OUTPUT / 'predictions' / trial / s / 'metrics.json').exists() for s in roster):
            continue
        if original:
            checkpoint = None
        elif is_test:
            refit = read(OUTPUT / (prefix + '_refit_choices') / (name + '.json'))['refit']
            checkpoint = Path(refit['checkpoint'])
        else:
            checkpoint = Path(read(OUTPUT / (prefix + '_checkpoint_choices') / (name + '.json'))['checkpoint'])
        if prefix == 'native_surface':
            choice = read(OUTPUT / 'native_surface_checkpoint_choices' / (name + '.json'))
            assert choice['metrics']['cd_proxy_target_space'] == 'native_MHR_GT' and not choice['test_used']
            assert read(OUTPUT / 'runs' / name / 'configuration.json')['native_surface_target']
        checkpoint_sha = sha256(checkpoint) if checkpoint else None
        values = {}
        coverage = []
        for sample in roster:
            folder = OUTPUT / 'predictions' / trial / sample
            value = read(folder / 'metrics.json')
            info = read(folder / 'inference.json')
            assert value['full_frame_roster'] and value['frames'] == manifest['samples'][sample]['frames']
            assert value['protocol'] == reference and value['conversion_revision'] == 2
            assert value['gt_sha256'] == manifest['samples'][sample]['gt_sha256'] == sha256(SOURCE / sample / 'human/human_params_gt.npz')
            assert value['prediction_sha256'] == sha256(folder / 'human_params_mhr_v2.npz')
            assert info['checkpoint_sha256'] == checkpoint_sha
            assert info['input_view'] == 'native_rgb_static_crop' and not info['target_foreground_only_input']
            assert info['evaluation_protocol'] == value['evaluation_protocol'] == 'native_rgb_static_crop_target_head_v1'
            assert info['all_accepted_heads_in_target_mask'] and not info['gt_used_for_detection_or_tracking']
            assert info['static_crop_box'] == view['samples'][sample]['box']
            assert info['target_mask_sha256'] == sha256(SOURCE / sample / 'processed/human_mask.mp4')
            assert info.get('depth_decoder', 'published') == 'published'
            assert all(math.isfinite(value[k]) for k in ['CD_H_cm', 'ACC_H_cm_per_frame2'])
            values[sample] = value
            coverage.append(info['detection_frame_rate'])
        assert len(values) == count and sum(v['frames'] for v in values.values()) == frames
        summary = {k: sum(v[k] for v in values.values()) / count for k in ['CD_H_cm', 'ACC_H_cm_per_frame2']}
        role = 'original weights' if original else ('Train20 full-training checkpoint' if refit['refit_performed_this_wave'] else 'unchanged original initialization, no refit') if is_test else 'Train16/4 selected search checkpoint'
        summary.update(sequences=count, frames=frames, per_sequence=values, target_coverage=sum(coverage) / count,
                       checkpoint=str(checkpoint) if checkpoint else None, checkpoint_sha256=checkpoint_sha,
                       checkpoint_role=role, Test_used=is_test, test_used_for_selection=False,
                       independently_checked_unix=time.time())
        result['models'][name] = summary
        added.append(name)
        print(name, json.dumps({k: v for k, v in summary.items() if k != 'per_sequence'}))
    result.update(updated_unix=time.time(), evaluation_split=split, Test_used=is_test,
                  family=family, test_used_for_selection=False, whole_Train4_only=not is_test, whole_Test10_only=is_test)
    save_json(destination, result)
    print(json.dumps(dict(newly_checked=added, completed_models=list(result['models']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-only', action='store_true')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--family', choices=['fixed', 'native-surface'], default='fixed')
    args = parser.parse_args()
    main(args.new_only, args.split, args.family)

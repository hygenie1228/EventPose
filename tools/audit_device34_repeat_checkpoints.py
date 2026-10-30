"""CPU audit of completed independent body repeat trials, without GPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from goal_runtime_control import register_job

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/261026/device34_body_28h'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=('rgb3', 'rgb4', 'rgb3_rgb4', 'rgb3_event4',
                                                'event3_rgb4', 'event3_event4'))
    args = parser.parse_args()
    config = json.loads((ROOT / '_agents/research/261026/device34_body_goal_28h.json').read_text())
    register_job(config, os.getpid(), 'device34_repeat_checkpoint_cpu_audit:' + (args.condition or 'first_four'))
    import torch
    torch.set_num_threads(2)
    records = []
    conditions = (args.condition,) if args.condition else ('rgb3', 'rgb4', 'rgb3_rgb4', 'rgb3_event4')
    for condition in conditions:
        name = condition + '_seed261027'
        live = json.loads((OUT / 'configs' / (name + '.json')).read_text())
        complete = json.loads((OUT / 'runs' / name / 'complete.json').read_text())
        assert complete['status'] == 'complete'
        assert 0 < complete['elapsed_seconds'] < 4 * 3600
        assert live['hours'] == 3 and live['seed'] == 261027
        for kind in ('best', 'last'):
            path = OUT / 'checkpoints' / name / (kind + '.pt')
            value = torch.load(path, map_location='cpu')
            assert value['config'] == live, (name, kind, 'configuration')
            assert value['test_used_for_selection'] is False
            assert value['model_delta']
            assert all(torch.isfinite(t).all().item() for t in value['model_delta'].values())
            if kind == 'best':
                assert value['metrics_apply_to_checkpoint']
                assert value['metrics_step'] == value['step']
            else:
                for key in ('step', 'successful_steps', 'samples_seen', 'unique_frames_seen'):
                    assert value[key] == complete[key], (name, key)
            digest = hashlib.sha256()
            with path.open('rb') as checkpoint_file:
                for block in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            records.append(dict(name=name, kind=kind, step=value['step'], sha256=digest.hexdigest(),
                                configuration_equal=True, tensors_finite=True,
                                test_used_for_selection=False))
            del value
    result = dict(status='passed', checked_unix=time.time(), records=records, conditions=list(conditions),
                  scope='Specified completed repeat trials only; full Test and final audit remain required')
    filename = args.condition + '_seed261027_checkpoint_cpu_audit.json' if args.condition else 'first_four_repeat_checkpoint_cpu_audit.json'
    path = OUT / filename
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Validate repaired sensor roles and actual loader/target/inference mappings."""
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
    register_job(goal, os.getpid(), 'fusion_repair_actual_loader_cpu')
    os.environ['EVENTPOSE_MULTISENSOR_OUT'] = goal['output_root']
    import numpy as np
    import torch
    from multihmr.multisensor_data import Frames, collate, input_keys
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    from multihmr.multisensor_inference import CachedSensors
    from scripts.train_multihmr import targets
    torch.set_num_threads(2)
    out = Path(goal['output_root'])
    assert json.loads((out / 'source_reuse_audit.json').read_text())['status'] == 'passed'
    import h5py
    verified_index_rows = {}
    for role in ('training', 'validation'):
        rows = json.loads((out / (role + '_index.json')).read_text())
        for session in sorted({r['session'] for r in rows}):
            chosen = [r for r in rows if r['session'] == session]
            indices = np.array([r['cache_row'] for r in chosen])
            with h5py.File(out / 'cache' / (session + '.h5'), 'r', locking=False) as f:
                assert np.array_equal(f['frames'][:][indices], [r['frame'] for r in chosen])
                for sensor in ('rgb3', 'rgb4', 'event3', 'event4'):
                    assert f[sensor + '_available'][:][indices].all()
            with np.load(out / 'targets' / (session + '.npz')) as z:
                assert z['valid'][indices].any(1).all()
        verified_index_rows[role] = len(rows)
    records = []
    for condition, layout in REPAIR_LAYOUTS.items():
        for role in ('training', 'validation'):
            data = Frames(role, condition=condition, sensor_layout=layout)
            items = [data[i] for i in range(2)]
            batch = collate(items)
            b, gt = targets(batch, torch.device('cpu'))
            assert gt['joints'].shape[-2:] == (55, 3)
            assert gt['idx'][0].max() < 2
            keys = set(input_keys(b))
            assert not keys & {'joints', 'centers', 'valid', 'betas', 'rotations', 'index'}
            key = 'image' if layout[0].startswith('rgb') else 'event'
            assert key in keys
            for i in range(2):
                row = data.records[i]
                source = CachedSensors(row['session'], condition, sensor_layout=layout)
                sample = source.sample(row['frame'])
                assert sample is not None and source.primary == layout[0] and source.aux == layout[1]
                assert set(sample) == keys
                for name in keys:
                    assert torch.equal(sample[name], batch[name][i]), (condition, role, name)
                with np.load(out / 'targets' / (row['session'] + '.npz')) as z:
                    expected = z['joints'][row['cache_row']] @ source.R.T + source.t
                assert np.allclose(items[i]['joints'], expected)
                pixels = set(source.pixel_datasets_read)
                assert pixels == set(s for s in layout if s)
                if condition == 'event3_event4':
                    assert pixels == {'event3', 'event4'} and 'image' not in keys
                records.append(dict(condition=condition, role=role, row=i, primary=source.primary,
                                    auxiliary=source.aux, pixels=sorted(pixels),
                                    loader_inference_tensors_equal=True, target_camera_correct=True))
                source.close()
            for handle in data.handles.values():
                handle.close()
    write(out / 'real_data_cpu_preflight.json', dict(status='passed', checked_unix=time.time(),
          records=records, all_index_rows_verified=verified_index_rows,
          scope='Actual Train/holdout tensors and targets only; no GPU model or Test performance'))
    print(json.dumps(dict(status='passed', sensor_role_sample_checks=len(records), event_only_no_rgb=True)), flush=True)


if __name__ == '__main__':
    main()

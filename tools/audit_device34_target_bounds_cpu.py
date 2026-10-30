"""Check affected trial's real target geometry with the training CPU functions."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.goal_runtime_control import register_job, write


def main():
    out = ROOT / 'exp/261026/device34_body_28h'
    goal = json.loads((ROOT / '_agents/research/261026/device34_body_goal_28h.json').read_text())
    register_job(goal, os.getpid(), 'device34_target_bounds_cpu_audit')
    import numpy as np
    import torch
    from multihmr.multisensor_data import Frames, collate
    from scripts.train_multihmr import targets
    torch.set_num_threads(2)
    counts = {}
    for role in ('training', 'validation'):
        dataset = Frames(role, condition='event3_event4')
        items = []
        for index, record in enumerate(dataset.records):
            f = dataset.handle(record['session'])
            i = record['cache_row']
            assert int(f['frames'][i]) == record['frame']
            target = dataset.target(record['session'])
            R = f['event3_R_common_to_sensor'][:].astype(np.float32)
            t = f['event3_t_common_to_sensor'][:].astype(np.float32)
            item = {k: target[k][i].copy() for k in ('joints', 'centers', 'rotations', 'betas', 'valid')}
            item['joints'] = item['joints'] @ R.T + t
            item['centers'] = item['centers'] @ R.T + t
            item['rotations'][:, 0] = R @ item['rotations'][:, 0]
            item.update(index=index, K=f['event3_K_primary'][:].astype(np.float32),
                        original_K=f['event3_K_native'][:].astype(np.float32),
                        image_shape=f['event3_native_shape'][:].astype(np.int64))
            items.append(item)
        for handle in dataset.handles.values():
            handle.close()
        batches = 0
        for seed in range(5):
            indices = np.random.default_rng(261027 + seed).permutation(len(items))
            for start in range(0, len(items), 16):
                batch = collate([items[int(i)] for i in indices[start:start + 16]])
                n = len(batch['index'])
                assert all(len(batch[k]) == n for k in ('valid', 'centers', 'K', 'original_K', 'image_shape'))
                _, values = targets(batch, torch.device('cpu'))
                bi, y, x, _ = values['idx']
                assert (bi >= 0).all() and (bi < n).all()
                assert (x >= 0).all() and (x < 64).all() and (y >= 0).all() and (y < 64).all()
                batches += 1
        counts[role] = dict(frames=len(items), permutations=5, batches=batches)
    result = dict(status='passed', condition='event3_event4', roles=counts,
                  checked_unix=time.time(), actual_training_functions=['multisensor_data.collate', 'train_multihmr.targets'],
                  scope='CPU geometry and indexing only; no sensor pixels or GPU operations; does not reproduce or establish cause of GPU failure')
    write(out / 'target_bounds_cpu_audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

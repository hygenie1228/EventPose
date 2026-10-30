"""Post-training Train-holdout event ablations, without Test-based selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write
GOAL = ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json'
from fusion_repair_scope import FUSION_NAMES
NAMES = FUSION_NAMES


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('condition', choices=NAMES); args = parser.parse_args()
    goal = json.loads(GOAL.read_text()); out = Path(goal['output_root'])
    register_job(goal, os.getpid(), 'fusion_repair_holdout_ablation:' + args.condition)
    os.environ['EVENTPOSE_MULTISENSOR_OUT'] = str(out)
    assert int(os.environ['EVENTPOSE_PHYSICAL_GPU']) in goal['allowed_gpus']
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from multihmr.multisensor_data import Frames, collate
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    from multihmr.multisensor_inference import load_model
    from multihmr.multisensor_diagnostics import holdout_modality_errors
    torch.set_num_threads(2)
    config = json.loads((out / 'configs' / (args.condition + '.json')).read_text())
    checkpoint = out / 'checkpoints' / args.condition / 'best.pt'
    model = load_model(config, checkpoint)
    data = Frames('validation', condition=args.condition, sensor_layout=REPAIR_LAYOUTS[args.condition])
    assert len(data) == 288
    loader = DataLoader(data, batch_size=16, collate_fn=collate, num_workers=0, shuffle=False)
    reg = torch.from_numpy(np.load(out / 'joint_regressor.npy')).cuda()
    rows = {}
    for mode in ('intact', 'missing_aux', 'zero_aux', 'swap_aux_polarity'):
        assert time.time() < goal['deadline_unix'] - 120
        rows[mode] = holdout_modality_errors(model, loader, reg, mode, maximum_people_per_image=16)
        if args.condition == 'event3_event4':
            assert 'image' not in rows[mode]['input_keys']
        print(json.dumps(dict(condition=args.condition, mode=mode,
              matched=rows[mode]['matched_instances'], total=rows[mode]['total_gt_instances'])), flush=True)
    reference = rows['intact']; pairs = {}
    for mode in list(rows)[1:]:
        row = rows[mode]; assert row['gt_keys'] == reference['gt_keys']
        common = sorted(set(row['errors_mm']) & set(reference['errors_mm']))
        a = np.array([reference['errors_mm'][k] for k in common])
        b = np.array([row['errors_mm'][k] for k in common])
        pairs[mode] = dict(common_instances=len(common), intact_mean_mm=a.mean(0).tolist() if common else None,
                           ablated_mean_mm=b.mean(0).tolist() if common else None,
                           ablated_minus_intact_mm=(b-a).mean(0).tolist() if common else None,
                           metric_order=['absolute_MPJPE', 'frame_aligned_MPJPE', 'frame_aligned_PA_MPJPE', 'full3D_root'])
    result = dict(status='passed', condition=args.condition, role='Train subject holdout only', rows=rows,
                  pairs=pairs, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], physical_gpu=int(os.environ['EVENTPOSE_PHYSICAL_GPU']),
                  pid=os.getpid(), checked_unix=time.time(), test_used=False,
                  interpretation='Diagnostic altered inputs only; pairwise common GT and full coverage reported; never replace Test metrics. Frames with >16 native candidates are resource failures before HPH; no top-k substitution; list and GT counts explicit. Normal Test inference unchanged.')
    result['diagnostic_implementation_sha256']={}
    for relative in ('src/multihmr/multisensor_diagnostics.py','tools/ablate_fusion_repair.py'):
        content=(ROOT/relative).read_bytes();digest=hashlib.sha256(content).hexdigest()
        result['diagnostic_implementation_sha256'][relative]=digest
        destination=out/'source_snapshots'/(digest+'.txt')
        if not destination.exists():destination.write_bytes(content)
    write(out / 'ablations' / (args.condition + '.json'), result)
    for handle in data.handles.values():
        handle.close()


if __name__ == '__main__':
    main()

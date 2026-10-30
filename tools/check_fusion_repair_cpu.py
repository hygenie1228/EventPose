"""Synthetic camera and gradient checks for the repair before GPU training."""
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write


def main():
    config = json.loads((ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(config, os.getpid(), 'fusion_repair_cpu_geometry_gradient')
    import torch
    from multihmr.multisensor_repair import EpipolarEventFusion, REPAIR_LAYOUTS
    from multihmr.multisensor_model import CONDITIONS, self_check
    torch.set_num_threads(2); torch.manual_seed(261027)
    assert CONDITIONS['event3_rgb4'] == ('event3', 'rgb4')
    assert REPAIR_LAYOUTS['event3_rgb4'] == ('rgb4', 'event3')
    fusion = EpipolarEventFusion(2, 16, hidden=16, grid=2,normalize=True,input_gain=4.)
    K = torch.tensor([[[20., 0., 7.5], [0., 20., 7.5], [0., 0., 1.]]])
    R = torch.eye(3)[None]; t = torch.tensor([[.2, 0., 0.]])
    grid, _ = fusion.projection(K, K, R, t, 16, 16)
    uv = (grid + 1) * 15 / 2
    original = torch.tensor([[3.5, 3.5], [11.5, 3.5], [3.5, 11.5], [11.5, 11.5]])
    expected = original[:, None].expand(-1, len(fusion.depths_m), -1).clone()
    expected[..., 0] += 4 / fusion.depths_m
    assert torch.allclose(uv[0], expected, atol=2e-6)
    features = torch.randn(2, 16, 16)
    events = torch.rand(2, 2, 224, 224)
    context = (events, K.expand(2, -1, -1), K.expand(2, -1, -1), R.expand(2, -1, -1),
               t.expand(2, -1), torch.ones(2), 16)
    assert torch.equal(fusion(features, context), features)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-3)
    encoder_grad = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = (fusion(features, context) - (features + .1)).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in fusion.parameters() if p.grad is not None)
        encoder_grad.append(sum(float(p.grad.abs().sum()) for p in fusion.encoder.parameters() if p.grad is not None))
        optimizer.step()
    assert encoder_grad[1] > 0 and encoder_grad[2] > 0
    missing = (*context[:5], torch.zeros(2), context[-1])
    assert torch.equal(fusion(features, missing), features)
    backwards_compatible = self_check()
    result = dict(status='passed', checked_unix=time.time(), camera_translation_expected_pixels='fx*tx/Z',
                  geometry_max_error_pixels=float((uv[0]-expected).abs().max()), encoder_gradients=encoder_grad,
                  exact_initial_RGB_identity=True, exact_missing_event_fallback_after_updates=True,
                  old_model_self_check=backwards_compatible, new_layouts=REPAIR_LAYOUTS,
                  scope='Synthetic CPU geometry/identity/gradient checks only. No real training or Test improvement proven.')
    write(Path(config['output_root']) / 'repair_cpu_preflight.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""CPU-only invariants for frozen-v1 / support-gated-v2 dispatch.

No model checkpoint, dataset file, CUDA device, or running job is inspected.
Synthetic inputs exercise the real v2 spatial/temporal forward and factory.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

import torch
from torch import nn

from multihmr.part_aggregation_model import PartTemporalAggregation
from multihmr.part_aggregation_v2 import SpatialSupportAggregation, VERSION
from multihmr.part_aggregation_factory import build_part_model as factory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(events=None, *, no_bins=False, available=True, off_camera=False):
    b = 2
    if events is None:
        events = torch.rand(b, 3, 2, 224, 224)
    K = torch.tensor([[600., 0, 447.5], [0, 600., 447.5], [0, 0, 1.]])[None].repeat(b, 1, 1)
    aux = torch.tensor([[150., 0, 111.5], [0, 150., 111.5], [0, 0, 1.]])[None].repeat(b, 1, 1)
    R = torch.eye(3)[None].repeat(b, 1, 1)
    t = torch.zeros(b, 3)
    if off_camera:
        t[:, 0] = 1000.
    return (events, K, aux, R, t, torch.full((b,), float(available)), 896,
            torch.full((b, 3), float(not no_bins)))


def module(variant="part_aware", *, support=True, nonzero_output=True):
    model = SpatialSupportAggregation(feature_dim=32, hidden=16, grid=16,
                                      variant=variant, support=support, warmup_steps=500)
    with torch.no_grad():
        if nonzero_output:
            model.output.weight.normal_(std=.15)
        # Make the learned selection unmistakably different from warmup.
        model.time_score[-1].bias.fill_(-2.)
        model.null_score.weight.zero_(); model.null_score.bias.fill_(2.)
        model.part_queries.normal_(std=.5)
    return model


def main():
    torch.set_num_threads(2); torch.manual_seed(261029)
    features = torch.randn(2, 256, 32)
    observations = context()
    checks = {}; details = {}
    v1_sha_before = sha(ROOT/"src/multihmr/part_aggregation_model.py")
    model = module()
    assert model.output.bias is None
    checks["output_projection_has_no_bias"] = True

    # Same parameters: only train/eval mode and step determine warmup policy.
    model.train(); model.set_training_step(0)
    model(features, observations)
    assert model.last_warmup_active and model.variant == "part_aware"
    assert torch.allclose(model.last_time_weights, torch.full((2, 6, 3), 1/3))
    assert torch.equal(model.last_null_weights, torch.zeros(2, 6))
    model.set_training_step(499); model(features, observations)
    assert model.last_warmup_active
    model.set_training_step(500); model(features, observations)
    assert not model.last_warmup_active and model.last_null_weights.mean() > .5
    checks["uniform_only_while_training_and_step_below_500"] = True
    model.set_training_step(0); model.eval()
    reference = model(features, observations).detach()
    weights = model.last_time_weights.detach().clone()
    assert not model.last_warmup_active and model.last_null_weights.mean() > .5
    checks["step0_eval_uses_learned_gate"] = True

    # The successful_training_step intentionally is not needed for inference.
    model.set_training_step(1234)
    payload = io.BytesIO(); torch.save(model.state_dict(), payload); payload.seek(0)
    restored = module()
    restored.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True))
    assert restored.successful_training_step == 0
    restored.eval(); reloaded = restored(features, observations).detach()
    assert not restored.last_warmup_active
    assert torch.equal(weights, restored.last_time_weights)
    assert torch.allclose(reference, reloaded, rtol=0, atol=0)
    checks["state_dict_reload_at_step0_eval_reproduces_learned_gate"] = True

    # Projection has actual learned nonzero weights; identity is not a zero-init shortcut.
    assert model.output.weight.abs().sum() > 0
    model.eval(); model.set_training_step(0)
    with torch.no_grad():
        model.time_score[-1].weight.zero_(); model.time_score[-1].bias.fill_(-1000.)
        model.null_score.weight.zero_(); model.null_score.bias.fill_(1000.)
    all_null = model(features, observations)
    assert model.last_time_weights.count_nonzero() == 0
    assert torch.equal(model.last_null_weights, torch.ones(2, 6))
    assert torch.equal(all_null, features)
    checks["all_null_exact_RGB_identity_with_nonzero_projection"] = True
    checks["null_has_no_projection_bias_leak"] = True

    # Empty/missing/out-of-camera evidence must also be an exact RGB identity.
    restored.eval()
    for label, obs in [("empty_bins", context(no_bins=True)),
                       ("missing_sensor", context(available=False)),
                       ("off_camera_rays", context(off_camera=True))]:
        result = restored(features, obs)
        assert torch.equal(result, features), label
        if label in ("missing_sensor", "off_camera_rays"):
            assert restored.last_support_diagnostics["calibration_supported_patch_count"] == 0
    checks["missing_empty_and_off_camera_evidence_identity"] = True
    checks["geometry_diagnostics_exclude_unobservable_patches"] = True

    # Spatial density gate reads predicted maps only, rather than a GT mask.
    supported = module(support=True); unmasked = module(support=False)
    maps = torch.zeros(2, 6, 256); maps[:, :, 40] = 1.
    residual = torch.ones(2, 256, 32)
    supported.last_part_maps = maps; unmasked.last_part_maps = maps
    gated = supported._apply_support(supported.output, (torch.ones(2, 256, 16),), residual)
    passthrough = unmasked._apply_support(unmasked.output, (), residual)
    assert torch.equal(gated[:, 40], residual[:, 40])
    assert not gated[:, :40].any() and not gated[:, 41:].any()
    assert torch.equal(passthrough, residual)
    assert supported.last_support_diagnostics["predicted_low_density_patch_count"] == 510
    assert supported.last_support_diagnostics["predicted_high_density_patch_count"] == 2
    checks["predicted_density_suppresses_low_mass_patches"] = True
    checks["support_gate_has_no_GT_forward_argument"] = "joints" not in inspect.signature(supported.forward).parameters
    assert checks["support_gate_has_no_GT_forward_argument"]

    # Calibrated-valid support counts are bounded and energy diagnostics finite.
    restored.eval(); restored(features, observations)
    diagnostic = restored.last_support_diagnostics
    assert 0 < diagnostic["calibration_supported_patch_count"] <= 512
    assert diagnostic["predicted_high_density_patch_count"] + diagnostic["predicted_low_density_patch_count"] == diagnostic["calibration_supported_patch_count"]
    for key in ("high_density_residual_energy", "low_density_residual_energy"):
        if diagnostic[key] is not None:
            assert diagnostic[key] >= 0 and torch.isfinite(torch.tensor(diagnostic[key]))
    details["valid_geometry_diagnostics"] = diagnostic
    checks["geometry_density_diagnostic_partition_consistent"] = True

    # Both dynamic controls have the same warmup, observation candidates and parameters.
    part = module("part_aware"); global_model = module("part_agnostic")
    global_model.load_state_dict(part.state_dict())
    assert sum(p.numel() for p in part.parameters()) == sum(p.numel() for p in global_model.parameters())
    part.train(); global_model.train(); part.set_training_step(0); global_model.set_training_step(0)
    yp = part(features, observations); yg = global_model(features, observations)
    assert torch.equal(yp, yg)
    assert part.last_warmup_active and global_model.last_warmup_active
    part.eval(); global_model.eval()
    part(features, observations); global_model(features, observations)
    assert not part.last_warmup_active and not global_model.last_warmup_active
    assert torch.equal(global_model.last_time_weights[:, :1].expand(-1, 6, -1), global_model.last_time_weights)
    assert torch.allclose(global_model.last_time_weights[:, 0], part.last_time_weights.mean(1))
    checks["global_and_part_share_parameters_support_candidates_and_warmup"] = True
    checks["global_control_differs_only_by_shared_temporal_distribution"] = True

    # Optimizing genuine event feature/gate parameters remains possible after warmup.
    part.train(); part.set_training_step(500); part.zero_grad(set_to_none=True)
    y = part(features, observations)
    (y-features).square().mean().backward()
    for prefix in ("encoder.", "spatial_transformer.", "time_score.", "output."):
        grad = sum(float(p.grad.square().sum()) for name, p in part.named_parameters()
                   if name.startswith(prefix) and p.grad is not None)
        assert grad > 0, prefix
        details[prefix+"gradient_l2"] = grad**.5
    checks["post_warmup_event_encoder_transformer_gate_projection_receive_gradients"] = True

    # Factory dispatch is tested without loading any RGB pretrained asset.
    import multihmr.part_aggregation_model as v1_module
    import multihmr.part_aggregation_v2 as v2_module
    for version, expected in [(VERSION, "v2"), ("calibrated_part_temporal_transformer_v1", "v1")]:
        with patch.object(v1_module, "build_part_model", return_value="v1") as old, patch.object(v2_module, "build_part_model", return_value="v2") as new:
            assert factory(dict(architecture_version=version), "cpu", checkpoint=None) == expected
            assert (new.call_count, old.call_count) == ((1, 0) if expected == "v2" else (0, 1))
    checks["explicit_factory_preserves_frozen_v1_and_routes_v2"] = True
    train_source = (ROOT/"src/multihmr/part_aggregation_training.py").read_text()
    eval_source = (ROOT/"src/multihmr/part_aggregation_evaluation.py").read_text()
    assert "from .part_aggregation_factory import build_part_model" in train_source
    assert "from .part_aggregation_factory import build_part_model" in eval_source
    assert "set_training_step(successful)" in train_source
    assert "model.eval()" in train_source and "checkpoint=checkpoint).eval()" in eval_source
    checks["training_and_evaluation_use_versioned_factory_and_eval_mode"] = True
    assert sha(ROOT/"src/multihmr/part_aggregation_model.py") == v1_sha_before
    checks["v1_source_preserved"] = True
    result = dict(status="passed", CPU_only=True, CUDA_or_running_jobs_inspected=False,
                  synthetic_only=True, checkpoint_or_dataset_loaded=False,
                  checks=checks, details=details,
                  sources={name: sha(ROOT/name) for name in ["src/multihmr/part_aggregation_v2.py",
                    "src/multihmr/part_aggregation_factory.py", "src/multihmr/part_aggregation_model.py",
                    "src/multihmr/part_aggregation_training.py", "src/multihmr/part_aggregation_evaluation.py"]})
    destination = ROOT/"_agents/research/261029/부위별_집계_v2_독립_CPU감사.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

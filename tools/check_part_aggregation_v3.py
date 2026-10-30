"""Independent CPU checks for train-only motion supervision and v3 inference.

Does not query CUDA or running jobs. Existing assets are only read; generated
checks use synthetic features and one read-only RGB-only validation sample.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multihmr.part_aggregation_model import model_inputs
from multihmr.part_aggregation_v3 import MotionHorizonAggregation,MotionHorizonScore,VERSION
from multihmr.part_aggregation_factory import build_part_model as factory
from check_part_aggregation_v2 import context


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(variant='part_aware',static=False):
    return MotionHorizonAggregation(feature_dim=32,hidden=16,variant=variant,
        warmup_steps=500,support=True,static_horizon=static,path_prior=[14.]*6)


def gradients(model,prefix):
    return sum(float(p.grad.detach().square().sum()) for n,p in model.named_parameters()
               if n.startswith(prefix) and p.grad is not None)**.5


def main():
    torch.set_num_threads(2);torch.manual_seed(261029)
    frozen_names=['src/multihmr/part_aggregation_model.py','src/multihmr/part_aggregation_v2.py']
    before={n:sha(ROOT/n) for n in frozen_names}
    checks={};details={}
    features=torch.randn(2,256,32);obs=context()
    model=make().eval()
    assert model.time_score.motion_head[0].in_features==6*16+3
    assert model.time_score.quality_head[0].in_features==3*16+3
    with torch.no_grad():model.output.weight.normal_(std=.1)
    result=model(features,obs)
    assert result.shape==features.shape and torch.isfinite(result).all()
    assert model.time_score.last_predicted_path.shape==(2,6)
    assert model.time_score.last_predicted_uv.shape==(2,6,2)
    assert model.time_score.last_horizon_ms.shape==(2,6)
    checks['real_forward_shapes_MLP_concatenation_dimensions']=True
    assert model.output.bias is None
    w=model.last_time_weights.detach().clone();null=model.last_null_weights.detach().clone()
    assert torch.allclose(w.sum(-1)+null,torch.ones_like(null))
    assert model.time_score.last_horizon_ms.min()>=33 and model.time_score.last_horizon_ms.max()<=200
    checks['bounded_soft_causal_horizon_and_separate_null_mass']=True

    # Warmup is train-only, including after reloading at step0.
    model.train();model.set_training_step(0);model(features,obs)
    assert model.last_warmup_active and torch.allclose(model.last_time_weights,torch.full((2,6,3),1/3))
    assert model.time_score.last_predicted_path is not None
    model.eval();model(features,obs)
    assert not model.last_warmup_active and torch.allclose(model.last_time_weights,w)
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    restored=make().eval();restored.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=True))
    assert restored.successful_training_step==0
    assert torch.equal(restored(features,obs),result)
    assert not restored.last_warmup_active and torch.equal(restored.last_time_weights,w)
    checks['eval_step0_reload_uses_motion_gate_not_uniform_warmup']=True

    # A learned negative quality logit must give exact RGB identity despite
    # nonzero projection and nonempty, calibrated event candidates.
    with torch.no_grad():
        restored.time_score.quality_head[-1].weight.zero_()
        restored.time_score.quality_head[-1].bias.fill_(-1000.)
    assert torch.equal(restored(features,obs),features)
    assert restored.last_time_weights.count_nonzero()==0
    assert torch.equal(restored.last_null_weights,torch.ones(2,6))
    for label,o in [('empty_bins',context(no_bins=True)),('missing_sensor',context(available=False)),('off_camera',context(off_camera=True))]:
        assert torch.equal(model(features,o),features),label
    checks['all_null_missing_empty_off_camera_exact_RGB_identity']=True
    checks['output_projection_bias_absent']=True

    # Local occupancy: only an event block that projects onto the selected RGB
    # patch contributes. Global activity outside a part does not become local.
    sparse=torch.zeros(2,3,2,224,224);sparse[:,0,:,:14,:14]=1
    occ_model=make().eval();occ_model._motion_context=context(events=sparse)
    maps=torch.zeros(2,6,256);maps[:,:,0]=1;maps[:,1,0]=0;maps[:,1,255]=1
    occ_model.last_part_maps=maps
    occupancy=occ_model._part_occupancy()
    assert occupancy.shape==(2,6,3)
    assert torch.allclose(occupancy[:,0,0],torch.ones(2),atol=2e-5)
    assert occupancy[:,1].abs().max()<1e-6
    assert occupancy[:,:,1:].abs().max()<1e-6
    occ_model._motion_context=context(events=sparse,off_camera=True)
    assert not occ_model._part_occupancy().any()
    occ_model._motion_context=None
    checks['part_local_occupancy_uses_calibrated_support_not_global_activity']=True

    # Same dynamic/static head parameters. Only the static horizon ignores the
    # query-dependent predicted motion, while retaining its motion auxiliary.
    dynamic=MotionHorizonScore(16,False,[14.]*6)
    static=MotionHorizonScore(16,True,[14.]*6)
    with torch.no_grad():
        for p in dynamic.motion_head.parameters():p.zero_()
        dynamic.motion_head[0].weight[0,0]=1
        dynamic.motion_head[-1].weight[0,0]=1
        dynamic.motion_head[-1].bias[1:]=torch.tensor([.5,-.25])
    static.load_state_dict(dynamic.state_dict())
    x=torch.zeros(2,6,3,49);x[0,:,0,16]=.2;x[1,:,0,16]=2.
    def conditional(logits):
        full=torch.cat([logits.squeeze(-1),torch.zeros(2,6,1)],-1).softmax(-1)
        return full[...,:3]/full[...,:3].sum(-1,keepdim=True),full
    wd,fd=conditional(dynamic(x));ws,fs=conditional(static(x))
    assert not torch.allclose(wd[0],wd[1])
    assert torch.allclose(ws[0],ws[1])
    assert not torch.allclose(static.last_predicted_path[0],static.last_predicted_path[1])
    assert torch.equal(static.last_horizon_ms[0],static.last_horizon_ms[1])
    assert torch.equal(dynamic.last_predicted_uv[0,0],torch.tensor([10.,-5.]))
    assert torch.allclose(fd[...,:3].sum(-1),torch.full((2,6),torch.sigmoid(torch.tensor(2.))),atol=1e-6)
    checks['dynamic_horizon_changes_with_query_motion_static_prior_does_not']=True
    checks['static_control_keeps_same_motion_head_auxiliary_capacity']=True
    checks['signed_uv_head_preserves_direction']=True
    checks['event_use_quality_and_conditional_time_distribution_separate']=True
    details['controlled_dynamic_horizons_ms']=dynamic.last_horizon_ms[:,0].tolist()
    details['controlled_static_horizons_ms']=static.last_horizon_ms[:,0].tolist()

    # Invalid labels are masked before log/path/direction losses. Altering an
    # invalid target must not alter loss or add nonfinite gradients.
    auxiliary=make().train();auxiliary.set_training_step(0)
    auxiliary(features,obs)
    target=torch.full((2,6),30.);uv=torch.full((2,6,2),5.);valid=torch.ones(2,6,dtype=torch.bool)
    batch=dict(motion_gt_path_px=target,motion_gt_uv_px=uv,motion_gt_valid=valid)
    baseline,_=auxiliary.motion_supervision(batch)
    valid[0,0]=False
    bad=dict(motion_gt_path_px=target.clone(),motion_gt_uv_px=uv.clone(),motion_gt_valid=valid.clone())
    bad['motion_gt_path_px'][0,0]=float('nan');bad['motion_gt_uv_px'][0,0]=float('inf')
    clean=dict(motion_gt_path_px=target.clone(),motion_gt_uv_px=uv.clone(),motion_gt_valid=valid.clone())
    lbad,_=auxiliary.motion_supervision(bad);lclean,_=auxiliary.motion_supervision(clean)
    assert torch.isfinite(lbad) and torch.equal(lbad,lclean)
    invalid=dict(motion_gt_path_px=torch.full((2,6),float('nan')),
                 motion_gt_uv_px=torch.full((2,6,2),float('inf')),motion_gt_valid=torch.zeros(2,6,dtype=torch.bool))
    empty,components=auxiliary.motion_supervision(invalid)
    assert torch.isfinite(empty) and components['motion_path_loss']==0 and components['motion_direction_loss']==0
    checks['invalid_GT_path_UV_masked_before_auxiliary_loss']=True

    # Last-layer zero initialization blocks upstream aux gradient on step0.
    # After one actual head update the encoder must learn during uniform warmup,
    # even though the pose fusion projection is still exactly zero initialized.
    optimizer=torch.optim.SGD(auxiliary.parameters(),lr=.05)
    optimizer.zero_grad(set_to_none=True);baseline.backward()
    first_encoder=gradients(auxiliary,'encoder.');first_head=gradients(auxiliary,'time_score.motion_head.')
    assert first_head>0 and first_encoder==0
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    auxiliary.set_training_step(1);auxiliary(features,obs)
    loss,_=auxiliary.motion_supervision(batch);loss.backward()
    second_encoder=gradients(auxiliary,'encoder.');second_transformer=gradients(auxiliary,'spatial_transformer.')
    assert auxiliary.last_warmup_active and second_encoder>0 and second_transformer>0
    assert auxiliary.output.weight.abs().sum()==0
    checks['motion_aux_executes_during_uniform_warmup']=True
    checks['motion_aux_reaches_event_encoder_after_first_head_update_during_warmup']=True
    details.update(first_step_event_encoder_gradient_l2=first_encoder,first_step_motion_head_gradient_l2=first_head,
        second_step_event_encoder_gradient_l2=second_encoder,second_step_spatial_transformer_gradient_l2=second_transformer)

    # RGB-only has the same auxiliary head but never receives event candidates.
    rgb=make('rgb_only').train()
    no_events=(None,obs[1],None,None,None,None,896,None)
    assert torch.equal(rgb(features,no_events),features)
    assert rgb.time_score.last_predicted_path.shape==(2,6)
    assert rgb.time_score.last_predicted_uv.shape==(2,6,2)
    checks['RGB_only_motion_auxiliary_fallback_without_event_tensor']=True

    # Actual read-only validation sample: the GT fields travel only to losses.
    from multihmr.part_aggregation_data_v3 import PartMotionFrames
    from multihmr.multisensor_data import collate
    val=PartMotionFrames('validation',limit=1,use_events=False)
    try:
        item=val[0];batch=collate([item]);inputs=model_inputs(batch)
        assert set(inputs)=={'image','K'}
        assert all(k not in inputs for k in ['motion_gt_path_px','motion_gt_uv_px','motion_gt_valid','joints','valid'])
        reader=val.reader(val.records[0]['session'])
        assert reader.pixel_datasets_read=={'rgb3'} and reader.memory is None
        assert item['motion_gt_path_px'].shape==(6,) and item['motion_gt_uv_px'].shape==(6,2)
        assert item['motion_gt_valid'].dtype==np.bool_
    finally:
        for reader in val.handles.values():reader.close()
    checks['actual_RGB_only_dataset_reads_no_event_pixels']=True
    checks['motion_GT_labels_outside_model_inputs_whitelist']=True
    summary=json.loads((ROOT/'exp/261029/part_aware_dynamic_aggregation/diagnostics/motion_supervision_final_verification.json').read_text())
    assert summary['status']=='passed' and summary['inference_GT_forbidden'] and not summary['future_GT_used']
    details['verified_motion_target_counts']=summary['roles']

    import multihmr.part_aggregation_v3 as v3_module
    import multihmr.part_aggregation_v2 as v2_module
    import multihmr.part_aggregation_model as v1_module
    for version,expected in [(VERSION,'v3'),(v2_module.VERSION,'v2'),('v1','v1')]:
        with patch.object(v3_module,'build_part_model',return_value='v3'),patch.object(v2_module,'build_part_model',return_value='v2'),patch.object(v1_module,'build_part_model',return_value='v1'):
            assert factory(dict(architecture_version=version),'cpu')==expected
    train_source=(ROOT/'src/multihmr/part_aggregation_training.py').read_text()
    assert "PartMotionFrames as PartAggregationFrames" in train_source
    assert "motion_aux_weight" in train_source and "model.fusion.motion_supervision(b)" in train_source
    assert "from .part_aggregation_factory import build_part_model" in train_source
    checks['versioned_factory_and_conditional_training_auxiliary_dispatch']=True
    assert before=={n:sha(ROOT/n) for n in frozen_names}
    checks['frozen_V1_V2_sources_preserved']=True
    result=dict(status='passed',CPU_only=True,CUDA_or_running_jobs_inspected=False,
        synthetic_network_inputs=True,actual_dataset_check='One read-only cached RGB-only holdout sample, no raw source writes',
        checks=checks,details=details,sources={n:sha(ROOT/n) for n in frozen_names+['src/multihmr/part_aggregation_v3.py','src/multihmr/part_aggregation_data_v3.py','src/multihmr/part_aggregation_factory.py','src/multihmr/part_aggregation_training.py']})
    dest=ROOT/'_agents/research/261029/부위별_집계_v3_독립_CPU감사.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

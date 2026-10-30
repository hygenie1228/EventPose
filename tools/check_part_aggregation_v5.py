"""CPU-only prototype checks; this is not permission to launch V5 training."""
import argparse,hashlib,io,json,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.part_aggregation_v3 import MotionHorizonAggregation
from multihmr.part_aggregation_v5 import DenseSpatialAggregation
from check_part_aggregation_v2 import context

def make(variant='part_aware',static=False):
    return DenseSpatialAggregation(32,16,variant=variant,static_horizon=static,warmup_steps=2)

def main(actual_config=None):
    torch.set_num_threads(2);torch.manual_seed(261029)
    checks={};obs=context();features=torch.randn(2,4096,32)
    coarse=MotionHorizonAggregation(32,16);model=make();model.load_coarse_parameters(coarse.state_dict())
    assert model.encoder[0].weight.shape==(16,2,7,7) and model.position.shape==(1,1024,16)
    assert torch.allclose(model.encoder[0].weight.sum((-1,-2)),coarse.encoder[0].weight.sum((-1,-2)),atol=1e-6)
    checks['coarse_initialization_kernel_sum_and_position_interpolation']=True
    grid,valid=model.patch_projection(obs[1],obs[2],obs[3],obs[4],896,224)
    expected=-1+2*10/31
    assert torch.allclose(grid[:,10*32+10,:,0],torch.full_like(grid[:,10*32+10,:,0],expected),atol=2e-6)
    assert torch.allclose(grid[:,10*32+10,:,1],torch.full_like(grid[:,10*32+10,:,1],expected),atol=2e-6)
    checks['calibrated_stride7_centres_match_RGB_query_grid']=True
    model.eval();assert torch.equal(model(features,obs),features)
    with torch.no_grad():
        model.output.weight.normal_(std=.05);model.time_score.motion_head[-1].weight.normal_(std=.01)
    output=model(features,obs);weights=model.last_time_weights.clone()
    assert output.shape==features.shape and torch.isfinite(output).all()
    assert model.last_part_maps.shape==(2,6,1024)
    assert torch.allclose(weights.sum(-1)+model.last_null_weights,torch.ones(2,6))
    assert not torch.equal(model.time_score.last_horizon_ms[0],model.time_score.last_horizon_ms[1])
    assert (model.time_score.last_horizon_ms>=33).all() and (model.time_score.last_horizon_ms<=200).all()
    checks['dense_actual_module_shapes_and_dynamic_horizons']=True
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    restored=make().eval();restored.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=True))
    assert torch.equal(restored(features,obs),output) and torch.equal(restored.last_time_weights,weights)
    assert not restored.last_warmup_active and restored.successful_training_step==0
    checks['eval_step0_reload_exact_learned_gate']=True
    for missing in [context(available=False),context(no_bins=True),context(off_camera=True)]:
        assert torch.equal(model(features,missing),features)
    with torch.no_grad():
        restored.time_score.quality_head[-1].weight.zero_();restored.time_score.quality_head[-1].bias.fill_(-1000.)
    assert torch.equal(restored(features,obs),features)
    checks['missing_empty_off_camera_all_null_identity']=True
    sparse=torch.zeros(2,3,2,224,224);sparse[:,0,:,70:77,70:77]=1
    model._motion_context=context(events=sparse);model.last_part_maps=torch.zeros(2,6,1024)
    model.last_part_maps[:,:,10*32+10]=1
    occupancy=model._part_occupancy();assert torch.allclose(occupancy[:,:,0],torch.ones(2,6),atol=3e-5)
    assert occupancy[:,:,1:].abs().max()<1e-6;model._motion_context=None
    checks['local_occupancy_uses_exact_stride7_patch']=True
    for variant,static in [('fixed_all',False),('part_agnostic',False),('part_aware',True),('rgb_only',False)]:
        control=make(variant,static).eval();control(features,obs)
        if variant=='fixed_all':assert torch.allclose(control.last_time_weights,torch.full((2,6,3),1/3))
        if variant=='part_agnostic':assert torch.equal(control.last_time_weights[:,0],control.last_time_weights[:,5])
        if static:assert torch.equal(control.time_score.last_horizon_ms[0],control.time_score.last_horizon_ms[1])
        if variant=='rgb_only':assert torch.equal(control(features,(None,*obs[1:])),features)
    checks['matched_fixed_global_static_and_RGB_controls']=True
    model=make().train();model.set_training_step(0);model(features,obs)
    assert model.last_warmup_active and torch.allclose(model.last_time_weights,torch.full((2,6,3),1/3))
    model.set_training_step(2)
    target=dict(motion_gt_path_px=torch.full((2,6),25.),motion_gt_uv_px=torch.ones(2,6,2),motion_gt_valid=torch.ones(2,6,dtype=torch.bool))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);result=model(features,obs);aux,_=model.motion_supervision(target)
        (result.square().mean()+.5*aux).backward();optimizer.step()
    for prefix in ['encoder.','time_score.motion_head.','time_score.quality_head.']:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.named_parameters() if name.startswith(prefix)),prefix
    checks['post_warmup_pose_and_motion_gradient_paths']=True
    sources={};actual_tensors=0
    if actual_config:
        from multihmr.part_aggregation_v5 import build_part_model
        config=json.loads(Path(actual_config).read_text());real=build_part_model(config,'cpu')
        assert real.fusion.grid==32 and real.fusion.encoder[0].kernel_size==(7,7)
        assert real.fusion.output.weight.count_nonzero()==0
        real.train();assert not real.base.training and all(not p.requires_grad for p in real.base.parameters())
        reference=torch.load(config['frozen_RGB_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        actual=real.base.state_dict()
        for name,value in reference['model_delta'].items():
            if name.startswith('base.'):
                assert torch.equal(actual[name.removeprefix('base.')],value);actual_tensors+=1
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in config['source_sha256']}
        assert sources==config['source_sha256']
        checks['actual_dense_builder_frozen_RGB_and_zero_projection']=True
        checks['actual_Train_selected_RGB_parameter_equality']=True
    result=dict(status='passed' if actual_config else 'prototype_passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),CPU_only=True,
        synthetic_inputs_only=True,Test_used=False,checks=checks,
        model_source_sha256=hashlib.sha256((ROOT/'src/multihmr/part_aggregation_v5.py').read_bytes()).hexdigest(),
        actual_RGB_parameter_tensors=actual_tensors,sources=sources,
        limitation='CPU synthetic module tests and optional actual builder/parameter verification; no actual-image prediction, CUDA pilot or accuracy verification')
    filename='부위별_집계_v5_단일담당_CPU감사.json' if actual_config else '부위별_집계_v5_구조_CPU검토.json'
    p=ROOT/'_agents/research/261029'/filename;p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--actual-config')
    main(parser.parse_args().actual_config)

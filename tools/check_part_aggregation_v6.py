"""CPU prototype checks of shared frozen semantic event representations."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.part_aggregation_v4 import FrozenRGBModel
from multihmr.part_aggregation_v5 import DenseSpatialAggregation
from multihmr.part_aggregation_v6 import FrozenEventSemanticEncoder,replace_with_semantic_encoder
from check_part_aggregation_v2 import context

class RawEncoder(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(3,32);self.calls=[]
    def get_intermediate_layers(self,image):
        self.calls.append((tuple(image.shape),self.training))
        side=image.shape[-1]//14
        patches=F.adaptive_avg_pool2d(image,side).flatten(2).transpose(1,2)
        return (self.linear(patches),)
class Backbone(nn.Module):
    def __init__(self):super().__init__();self.encoder=RawEncoder();self.calls=0
    def forward(self,image):self.calls+=1;return self.encoder.get_intermediate_layers(image)[0]
class Base(nn.Module):
    img_size=896
    def __init__(self):super().__init__();self.backbone=Backbone()
    def forward(self,image,K=None,**kw):return self.backbone(image)
def call(model,image,obs):
    return model(image=image,K=obs[1],auxiliary_K=obs[2],primary_to_aux_rotation=obs[3],
        primary_to_aux_translation=obs[4],auxiliary_available=obs[5],event_bins=obs[0],event_bin_available=obs[7])
def make(base=None,variant='part_aware',static=False):
    model=FrozenRGBModel(base or Base(),DenseSpatialAggregation(32,16,variant=variant,static_horizon=static,warmup_steps=2))
    return replace_with_semantic_encoder(model)

def main(actual_config=None):
    torch.set_num_threads(2);torch.manual_seed(261029);checks={}
    image=torch.randn(2,3,896,896);obs=context();base=Base().eval();initial=copy.deepcopy(base.state_dict())
    reference=base(image).detach();model=make(base).train()
    assert model.fusion.encoder.raw_encoder is model.base.backbone.encoder
    assert not model.base.training and not model.fusion.encoder.raw_encoder.training
    before=base.backbone.calls;actual=call(model,image,obs)
    assert base.backbone.calls==before+1 and torch.equal(actual,reference)
    assert base.backbone.encoder.calls[-1]==((6,3,448,448),False)
    checks['inner_ViT_bypass_avoids_RGB_hook_recursion_and_is_frozen']=True
    encoder=model.fusion.encoder;zero=torch.zeros(1,2,224,224)
    assert encoder.pseudo_RGB(zero).count_nonzero()==0
    isolated=zero.clone();isolated[0,0,20,20]=1
    assert encoder.pseudo_RGB(isolated).count_nonzero()==0
    pair=isolated.clone();pair[0,0,20,21]=1
    color=encoder.pseudo_RGB(pair);assert color[:,0].max()>0 and color[:,2].max()==0
    assert color.min()>=0 and color.max()<=1
    checks['fixed_polarity_color_log_brightness_and_weak_support']=True
    x=torch.randn(2,1024,16);model.fusion.spatial_transformer.eval()
    assert torch.equal(model.fusion.spatial_transformer(x),x)
    checks['spatial_Transformer_initial_identity_preserves_semantic_basis']=True
    model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);model.fusion.set_training_step(step)
        out=call(model,image,obs);target={'motion_gt_path_px':torch.full((2,6),25.),'motion_gt_uv_px':torch.ones(2,6,2),'motion_gt_valid':torch.ones(2,6,dtype=torch.bool)}
        aux,_=model.fusion.motion_supervision(target);(out.square().mean()+.5*aux).backward();optimizer.step()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(torch.equal(value,base.state_dict()[key]) for key,value in initial.items())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.fusion.encoder.projection.parameters())
    checks['projection_learns_while_shared_RGB_parameters_remain_exact']=True
    selected={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith('fusion.') and not name.startswith('fusion.encoder.raw_encoder.') for name in selected)
    model.eval();expected=call(model,image,obs).detach();restored_base=Base();restored_base.load_state_dict(initial)
    restored=make(restored_base).eval();loaded=restored.load_state_dict(selected,strict=False)
    assert not loaded.unexpected_keys and torch.equal(call(restored,image,obs),expected)
    for bad in [context(available=False),context(no_bins=True),context(off_camera=True)]:
        assert torch.equal(call(restored,image,bad),reference)
    checks['adapter_only_reload_and_missing_event_identity']=True
    for variant,static in [('fixed_all',False),('part_agnostic',False),('part_aware',True),('rgb_only',False)]:
        control=make(variant=variant,static=static).eval();output=call(control,image,obs)
        assert output.shape==(2,4096,32) and torch.isfinite(output).all()
        if variant=='fixed_all':assert torch.allclose(control.fusion.last_time_weights,torch.full((2,6,3),1/3))
        if variant=='part_agnostic':assert torch.equal(control.fusion.last_time_weights[:,0],control.fusion.last_time_weights[:,5])
        if static:assert torch.equal(control.fusion.time_score.last_horizon_ms[0],control.fusion.time_score.last_horizon_ms[1])
    checks['matched_fixed_global_static_RGB_semantic_controls']=True
    sources={};count=0;feature_shape=None
    if actual_config:
        from multihmr.part_aggregation_v6 import build_part_model
        from multihmr.part_aggregation_data import MemorySensors
        cfg=json.loads(Path(actual_config).read_text());real=build_part_model(cfg,'cpu').eval()
        assert real.fusion.encoder.raw_encoder is real.base.backbone.encoder
        real.train();assert not real.base.training and not real.fusion.encoder.raw_encoder.training
        assert all(not p.requires_grad for p in real.base.parameters())
        selected={name for name,p in real.named_parameters() if p.requires_grad}
        assert all(name.startswith('fusion.') and not name.startswith('fusion.encoder.raw_encoder.') for name in selected)
        reference=torch.load(cfg['frozen_RGB_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        state=real.base.state_dict()
        for name,value in reference['model_delta'].items():
            if name.startswith('base.'):
                assert torch.equal(state[name.removeprefix('base.')],value);count+=1
        checks['actual_builder_shared_frozen_RGB_and_adapter_only_parameters']=True
        for layer in real.base.backbone.encoder.modules():
            if layer.__class__.__name__=='MemEffAttention':sys.modules[layer.__class__.__module__].XFORMERS_AVAILABLE=False
        row=json.loads((Path(cfg['source_root'])/'training_index.json').read_text())[0]
        reader=MemorySensors(row['session'],source_root=cfg['source_root'],cache_root=cfg['output_root'])
        events=reader.sample(row['frame'])['event_bins'];real.eval();hook_calls=[]
        hook=real.base.backbone.register_forward_hook(lambda *args:hook_calls.append(True))
        with torch.no_grad():features=real.fusion.encoder(events)
        hook.remove();reader.close();feature_shape=list(features.shape)
        assert feature_shape==[3,128,32,32] and torch.isfinite(features).all() and not hook_calls
        checks['actual_causal_Train_event_ViT_shape_and_hook_bypass_CPU_SDPA']=True
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in cfg['source_sha256']}
        assert sources==cfg['source_sha256']
    result=dict(status='passed' if actual_config else 'prototype_passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),CPU_only=True,
        synthetic_fake_backbone_only=not bool(actual_config),Test_used=False,checks=checks,sources=sources,
        actual_RGB_parameter_tensors=count,actual_Train_event_feature_shape=feature_shape,
        model_source_sha256=hashlib.sha256((ROOT/'src/multihmr/part_aggregation_v6.py').read_bytes()).hexdigest(),
        limitation='CPU structural/shared-weight checks; actual event probe uses CPU SDPA fallback; no actual pose accuracy or CUDA pilot verification')
    filename='부위별_집계_v6_단일담당_CPU감사.json' if actual_config else '부위별_집계_v6_의미특징_구조_CPU검토.json'
    p=ROOT/'_agents/research/261029'/filename;p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--actual-config')
    main(parser.parse_args().actual_config)

"""Single-agent CPU audit of frozen-RGB policy and actual checkpoint loading."""
import copy,hashlib,io,json,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.part_aggregation_v3 import MotionHorizonAggregation
from multihmr.part_aggregation_v4 import FrozenRGBModel,build_part_model
from check_part_aggregation_v2 import context

class ToyBase(nn.Module):
    img_size=896
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(nn.Linear(32,32),nn.Dropout(.9))
    def forward(self,image,K=None,**kwargs):return self.backbone(image)

def call(model,image,obs):
    return model(image=image,K=obs[1],auxiliary_K=obs[2],primary_to_aux_rotation=obs[3],
        primary_to_aux_translation=obs[4],auxiliary_available=obs[5],event_bins=obs[0],event_bin_available=obs[7])

def main():
    torch.set_num_threads(2);torch.manual_seed(261029)
    checks={};details=[];obs=context();image=torch.randn(2,256,32)
    target=dict(motion_gt_path_px=torch.full((2,6),25.),motion_gt_uv_px=torch.ones(2,6,2),motion_gt_valid=torch.ones(2,6,dtype=torch.bool))
    for variant,static in [('fixed_all',False),('part_agnostic',False),('part_aware',False),('part_aware',True),('rgb_only',False)]:
        base=ToyBase().eval();reference=base(image).detach();initial=copy.deepcopy(base.state_dict())
        fusion=MotionHorizonAggregation(32,16,variant=variant,warmup_steps=2,static_horizon=static)
        model=FrozenRGBModel(base,fusion).train()
        assert not model.base.training and not model.base.backbone.training and model.fusion.training
        assert all(not p.requires_grad for p in model.base.parameters())
        assert torch.equal(call(model,image,obs),reference)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        for step in range(4):
            model.fusion.set_training_step(step);optimizer.zero_grad(set_to_none=True)
            output=call(model,image,obs);aux,_=fusion.motion_supervision(target)
            loss=output.square().mean()+.5*aux;loss.backward();optimizer.step()
            assert all(p.grad is None for p in base.parameters())
            assert all(torch.equal(value,base.state_dict()[name]) for name,value in initial.items())
        if static:assert torch.equal(fusion.time_score.last_horizon_ms[0],fusion.time_score.last_horizon_ms[1])
        model.eval();expected=call(model,image,obs).detach()
        delta={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        assert delta and all(n.startswith('fusion.') for n in delta)
        restored_base=ToyBase();restored_base.load_state_dict(initial)
        restored=FrozenRGBModel(restored_base,MotionHorizonAggregation(32,16,variant=variant,warmup_steps=2,static_horizon=static)).eval()
        loaded=restored.load_state_dict(delta,strict=False)
        assert not loaded.unexpected_keys
        assert torch.equal(call(restored,image,obs),expected) and not restored.fusion.last_warmup_active
        assert torch.equal(call(restored,image,context(available=False)),reference)
        details.append(dict(variant=variant,static=static,frozen_base_exact=True,adapter_only_checkpoint_reload_exact=True))
    checks['synthetic_freeze_mode_identity_gradient_static_and_reload_controls']=True
    out=ROOT/'exp/261029/part_aware_dynamic_aggregation';config=json.loads((out/'configs/v4_part_dynamic.json').read_text())
    real=build_part_model(config,'cpu')
    assert all(not p.requires_grad for p in real.base.parameters()) and real.fusion.output.weight.count_nonzero()==0
    real.train();assert not real.base.training and real.fusion.training
    reference=torch.load(config['frozen_RGB_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    state=real.base.state_dict();count=0
    for name,value in reference['model_delta'].items():
        if name.startswith('base.'):
            assert torch.equal(state[name.removeprefix('base.')],value);count+=1
    checks['actual_strong_RGB_checkpoint_selected_parameter_bytes_exact']=True
    checks['actual_V3_fusion_checkpoint_loaded_and_zero_output_identity']=True
    checks['actual_base_frozen_and_eval_after_train_call']=True
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in config['source_sha256']}
    assert sources==config['source_sha256']
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),CPU_only=True,
        reviewer='single owning agent; not an independent-agent review',checks=checks,synthetic_controls=details,
        actual_RGB_selected_parameter_tensors=count,sources=sources,Test_used=False,
        limitation='Synthetic wrapper forwards plus actual full model construction/checkpoint parameter verification; no actual-image CPU prediction or CUDA pilot yet')
    p=ROOT/'_agents/research/261029/부위별_집계_v4_단일담당_CPU감사.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

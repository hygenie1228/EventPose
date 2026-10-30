"""Motion-supervised causal horizons with a separate event-use gate.

GT motion appears only in the training loss. The inference head estimates
past200ms motion from observed event features and predicts a soft history window.
"""
import hashlib,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .part_aggregation_v2 import SpatialSupportAggregation,build_part_model as build_v2

VERSION='calibrated_part_motion_horizon_v3'

class ZeroNullScore(nn.Module):
    def forward(self,x):return x.new_zeros((*x.shape[:-1],1))

class MotionHorizonScore(nn.Module):
    def __init__(self,hidden=128,static_horizon=False,path_prior=None):
        super().__init__();self.hidden=hidden;self.static_horizon=bool(static_horizon)
        self.part_identity=nn.Parameter(torch.randn(6,hidden)*.02)
        self.motion_head=nn.Sequential(nn.Linear(6*hidden+3,2*hidden),nn.GELU(),nn.Linear(2*hidden,3))
        self.quality_head=nn.Sequential(nn.Linear(3*hidden+3,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.motion_head[-1].weight);nn.init.zeros_(self.motion_head[-1].bias)
        nn.init.zeros_(self.quality_head[-1].weight);nn.init.constant_(self.quality_head[-1].bias,2.)
        prior=torch.tensor(path_prior if path_prior is not None else [14.]*6).float().clamp_min(.1)/20
        self.raw_static_path=nn.Parameter(torch.log(torch.expm1(prior)))
        self.raw_radius=nn.Parameter(torch.full((6,),math.log(math.expm1(5.9))))
        self.register_buffer('bin_centres_ms',torch.tensor([16.5,66.5,150.]))
        self.availability=None;self.local_occupancy_provider=None
        self.last_predicted_path=None;self.last_predicted_uv=None;self.last_horizon_ms=None
    def forward(self,x):
        H=self.hidden;B,P,T,_=x.shape;assert P==6 and T==3
        rgb=x[:,:,0,:H];observations=x[:,:,:,H:2*H]
        occupancy=x[:,:,:,-1]
        if self.local_occupancy_provider is not None:
            local=self.local_occupancy_provider()
            if local is not None:occupancy=local.to(occupancy.dtype)
        identity=self.part_identity[None].expand(B,-1,-1)
        differences=torch.cat([observations[:,:,0]-observations[:,:,1],observations[:,:,1]-observations[:,:,2]],-1)
        raw=self.motion_head(torch.cat([observations.flatten(2),differences,identity,occupancy],-1))
        self.last_predicted_path=20*F.softplus(raw[...,0]+self.raw_static_path[None])
        self.last_predicted_uv=20*raw[...,1:3]
        path=20*F.softplus(self.raw_static_path)[None].expand(B,-1) if self.static_horizon else self.last_predicted_path
        radius=F.softplus(self.raw_radius)[None]+.1
        horizon=(200*radius/(path+.1)).clamp(33,200)
        self.last_horizon_ms=horizon
        window=torch.sigmoid((horizon[...,None]-self.bin_centres_ms)/10.)
        if self.availability is not None:window=window*self.availability[:,None,:].to(window.dtype)
        distribution=window/window.sum(-1,keepdim=True).clamp_min(1e-8)
        quality=self.quality_head(torch.cat([rgb,observations.mean(2),identity,occupancy],-1))
        # softmax([log q_b + s, 0]) == [q_b*sigmoid(s), 1-sigmoid(s)].
        # Event-use mass and conditional temporal shape have separate parameters.
        return distribution.clamp_min(1e-12).log()[...,None]+quality[:,:,None,:]

class MotionHorizonAggregation(SpatialSupportAggregation):
    def __init__(self,feature_dim=1024,hidden=128,variant='part_aware',support=True,warmup_steps=500,static_horizon=False,path_prior=None):
        super().__init__(feature_dim,hidden=hidden,variant=variant,support=support,warmup_steps=warmup_steps)
        self.time_score=MotionHorizonScore(hidden,static_horizon,path_prior)
        self.null_score=ZeroNullScore()
        self._rgb_part=None;self._motion_context=None
        self.part_norm.register_forward_hook(self._capture_rgb_part)
        self.time_score.local_occupancy_provider=self._part_occupancy
    def _capture_rgb_part(self,module,inputs,output):self._rgb_part=output
    def _part_occupancy(self):
        context=self._motion_context
        if context is None or context[0] is None:return None
        events,K,auxK,R,t,available,size,bin_available=context
        B,T,_,S,_=events.shape;G=self.grid
        with torch.autocast(events.device.type,enabled=False):
            grid,valid=self.projection(K,auxK,R,t,size,S)
            uv=(grid+1)*(S-1)/2;fg=2*((uv-6.5)/14)/(G-1)-1
            valid=valid&(fg.abs()<=1).all(-1)
            raw=F.avg_pool2d(events.float().abs().mean(2).reshape(B*T,1,S,S),14,14)
            sample=F.grid_sample(raw,fg[:,None].expand(-1,T,-1,-1,-1).flatten(0,1),align_corners=True)
            sample=sample.reshape(B,T,G*G,-1)
            v=valid[:,None].float();sample=(sample*v).sum(-1)/v.sum(-1).clamp_min(1)
            return torch.einsum('bpn,btn->bpt',self.last_part_maps.float(),sample)
    def forward(self,features,context):
        self._motion_context=context;self.time_score.availability=context[-1]
        try:
            result=super().forward(features,context)
            if self.variant=='rgb_only':
                # A matched RGB-only motion auxiliary head, without event pixels.
                rgb=self._rgb_part;B,P,H=rgb.shape;pr=rgb[:,:,None].expand(-1,-1,3,-1)
                fake=torch.cat([pr,pr,self.time_embedding[None,None].expand(B,P,-1,-1),pr.new_zeros(B,P,3,1)],-1)
                self.time_score(fake)
            return result
        finally:
            self._motion_context=None
    def motion_supervision(self,batch):
        path=self.time_score.last_predicted_path.float();uv=self.time_score.last_predicted_uv.float()
        target=batch['motion_gt_path_px'].float();target_uv=batch['motion_gt_uv_px'].float()
        mask=batch['motion_gt_valid'].bool()&torch.isfinite(target)&torch.isfinite(target_uv).all(-1)
        zero=path.sum()*0
        if mask.any():
            path_loss=F.smooth_l1_loss(torch.log1p(path[mask]),torch.log1p(target[mask]),beta=.25)
            direction_loss=F.smooth_l1_loss(uv[mask]/20,target_uv[mask]/20,beta=.25)
        else:path_loss=direction_loss=zero
        radius=F.softplus(self.time_score.raw_radius)+.1
        radius_loss=torch.log(radius/6).square().mean()
        return path_loss+direction_loss+.01*radius_loss,dict(motion_path_loss=path_loss,motion_direction_loss=direction_loss,motion_radius_regularization=radius_loss)

def build_part_model(config,device='cuda',checkpoint=None):
    if config.get('v3_model_source_sha256'):
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['v3_model_source_sha256']
    model=build_v2(config,device=device,checkpoint=None)
    old=model.fusion
    fusion=MotionHorizonAggregation(model.base.embed_dim,config.get('hidden',128),variant=config['variant'],
        support=config.get('spatial_support',True),warmup_steps=config.get('uniform_aggregation_warmup_steps',500),
        static_horizon=config.get('static_part_horizon',False),path_prior=config.get('motion_path_prior_px')).to(device)
    state={k:v for k,v in old.state_dict().items() if not k.startswith(('time_score.','null_score.'))}
    loaded=fusion.load_state_dict(state,strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('time_score.') for k in loaded.missing_keys)
    fusion.collect_support_diagnostics_in_eval=config.get('collect_support_diagnostics_in_eval',False)
    model.fusion=fusion
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['successful_steps']>0 and saved['test_used_for_selection'] is False
        assert saved['config']['architecture_version']==VERSION and saved['config']['variant']==config['variant']
        assert saved['config']['split_sha256']==config['split_sha256']
        assert not model.load_state_dict(saved['model_delta'],strict=False).unexpected_keys
    return model

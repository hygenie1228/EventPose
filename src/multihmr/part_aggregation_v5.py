"""Dense event feature grid with the frozen motion-conditioned policy.

Experimental until its own CPU audit, frozen design and CUDA pilots pass.
Seven-pixel patches keep more spatial samples than the original stride14 grid.
"""
import math
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .part_aggregation_v3 import MotionHorizonAggregation

VERSION='calibrated_part_dense_frozen_rgb_v5'

class DenseSpatialAggregation(MotionHorizonAggregation):
    def __init__(self,feature_dim=1024,hidden=128,variant='part_aware',support=True,
                 warmup_steps=100,static_horizon=False,path_prior=None):
        super().__init__(feature_dim,hidden,variant,support,warmup_steps,static_horizon,path_prior)
        self.grid=32;self.patch_stride=7;self.patch_centre=3.
        self.encoder=nn.Sequential(nn.Conv2d(2,hidden,7,7),nn.GroupNorm(8,hidden))
        self.position=nn.Parameter(torch.randn(1,32*32,hidden)*.01)

    def patch_projection(self,K,auxK,R,t,size,event_size):
        with torch.autocast(K.device.type,enabled=False):
            grid,valid=self.projection(K,auxK,R,t,size,event_size)
            pixels=(grid+1)*(event_size-1)/2
            features=2*((pixels-self.patch_centre)/self.patch_stride)/(self.grid-1)-1
            return features,valid&(features.abs()<=1).all(-1)

    def _part_occupancy(self):
        context=self._motion_context
        if context is None or context[0] is None:return None
        events,K,auxK,R,t,available,size,bins=context
        B,T,_,S,_=events.shape
        with torch.autocast(events.device.type,enabled=False):
            grid,valid=self.patch_projection(K,auxK,R,t,size,S)
            raw=F.avg_pool2d(events.float().abs().mean(2).reshape(B*T,1,S,S),7,7)
            sampled=F.grid_sample(raw,grid[:,None].expand(-1,T,-1,-1,-1).flatten(0,1),align_corners=True)
            sampled=sampled.reshape(B,T,self.grid*self.grid,-1);mask=valid[:,None].float()
            sampled=(sampled*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            return torch.einsum('bpn,btn->bpt',self.last_part_maps.float(),sampled)

    def _apply_support(self,module,inputs,residual):
        maps=self.last_part_maps;density=maps.float().sum(1)*(maps.shape[-1]/maps.shape[1])
        gate=density.clamp(0,1);self.last_spatial_support=gate
        if self.use_spatial_support:residual=residual*gate[...,None].to(residual.dtype)
        collect=self.successful_training_step%25 in (0,24) if self.training else self.collect_support_diagnostics_in_eval
        if collect:
            with torch.no_grad():
                events,K,auxK,R,t,available,size,bins=self._diagnostic_context
                _,valid=self.patch_projection(K,auxK,R,t,size,events.shape[-1])
                supported=valid.any(-1)&bins.any(-1)[:,None].bool()&available[:,None].bool()
                high=(gate>=.5)&supported;low=(gate<.5)&supported
                energy=residual.float().square().mean(-1)
                self.last_support_diagnostics=dict(predicted_high_density_patch_count=int(high.sum()),
                    predicted_low_density_patch_count=int(low.sum()),high_density_residual_energy=float(energy[high].mean()) if high.any() else None,
                    low_density_residual_energy=float(energy[low].mean()) if low.any() else None,
                    mean_predicted_support=float(gate.mean()),calibration_supported_patch_count=int(supported.sum()),
                    diagnostic_training_step=self.successful_training_step,
                    interpretation='Predicted part density, not GT segmentation; calibrated stride7 patch centres')
        return residual

    def _aggregate(self,features,context):
        events,K,auxK,R,t,available,size,bins=context
        B,N,C=features.shape;side=math.isqrt(N);G=self.grid;T=3;H=self.hidden
        assert side*side==N
        pooled=F.adaptive_avg_pool2d(features.transpose(1,2).reshape(B,C,side,side),G).flatten(2).transpose(1,2)
        rgb=self.rgb_norm(self.query(pooled))
        maps=torch.einsum('ph,bnh->bpn',self.part_queries,rgb+self.position)/math.sqrt(H)
        maps=maps.float().softmax(-1).to(rgb.dtype);self.last_part_maps=maps
        part_rgb=self.part_norm(torch.einsum('bpn,bnh->bph',maps,rgb))
        if self.variant=='rgb_only':
            self.residual_energy=features.sum()*0;self.last_time_weights=features.new_zeros(B,6,3)
            self.last_null_weights=features.new_ones(B,6)
            return features
        encoded=self.encoder(events.flatten(0,1)*self.input_gain)
        assert encoded.shape[-2:]==(G,G)
        tokens=self.spatial_transformer(encoded.flatten(2).transpose(1,2)+self.position)
        encoded=tokens.transpose(1,2).reshape(B*T,H,G,G)
        grid,valid=self.patch_projection(K,auxK,R,t,size,events.shape[-1])
        sample_grid=grid[:,None].expand(-1,T,-1,-1,-1).flatten(0,1)
        samples=F.grid_sample(encoded,sample_grid.to(encoded.dtype),align_corners=True,padding_mode='zeros')
        samples=samples.permute(0,2,3,1).reshape(B,T,G*G,-1,H)
        depth_logits=torch.einsum('bnh,btnzh->btnz',rgb,samples)/math.sqrt(H)
        mask=valid[:,None]&bins[:,:,None,None].bool()
        dw=depth_logits.float().masked_fill(~mask,-1e4).softmax(-1)*mask
        dw=dw/dw.sum(-1,keepdim=True).clamp_min(1e-8)
        per_time=(dw.to(samples.dtype)[...,None]*samples).sum(-2)
        part_event=torch.einsum('bpn,btnh->bpth',maps,per_time)
        pr=part_rgb[:,:,None].expand(-1,-1,T,-1);te=self.time_embedding[None,None].expand(B,6,-1,-1)
        occupancy=events.float().abs().mean((2,3,4))[:,None,:,None].expand(-1,6,-1,-1)
        logits=self.time_score(torch.cat([pr,part_event,te,occupancy.to(pr.dtype)],-1)).squeeze(-1).float()
        logits=logits.masked_fill(~bins[:,None].bool(),-1e4)
        weights=torch.cat([logits,self.null_score(part_rgb).float()],-1).softmax(-1)
        if self.variant=='part_agnostic':weights=weights.mean(1,keepdim=True).expand(-1,6,-1)
        elif self.variant in ('fixed_all','fixed_recent'):
            w=bins.float()
            if self.variant=='fixed_recent':w=w*F.one_hot(torch.zeros(B,dtype=torch.long,device=w.device),3)
            w=w/w.sum(-1,keepdim=True).clamp_min(1)
            weights=torch.cat([w,1-w.sum(-1,keepdim=True)],-1)[:,None].expand(-1,6,-1)
        self.last_time_weights=weights[...,:3];self.last_null_weights=weights[...,-1]
        assignment=maps/maps.sum(1,keepdim=True).clamp_min(1e-8)
        local=torch.einsum('bpn,bpt->bnt',assignment,weights[...,:3].to(maps.dtype))
        message=torch.einsum('bnt,btnh->bnh',local,per_time)
        support=valid.any(-1)&bins.any(-1)[:,None].bool()
        residual=self.output(message)*support[...,None]
        self.residual_energy=residual.float().square().mean()
        residual=F.interpolate(residual.transpose(1,2).reshape(B,C,G,G),(side,side),mode='bilinear',align_corners=False)
        return features+available[:,None,None].to(features.dtype)*residual.flatten(2).transpose(1,2)

    def forward(self,features,context):
        original=self.variant;self._motion_context=context;self._diagnostic_context=context
        self.time_score.availability=context[-1]
        self.last_warmup_active=bool(self.training and self.successful_training_step<self.uniform_warmup_steps and original in ('part_aware','part_agnostic'))
        if self.last_warmup_active:self.variant='fixed_all'
        try:
            result=self._aggregate(features,context)
            if original=='rgb_only':
                rgb=self._rgb_part;B,P,H=rgb.shape;pr=rgb[:,:,None].expand(-1,-1,3,-1)
                self.time_score(torch.cat([pr,pr,self.time_embedding[None,None].expand(B,P,-1,-1),pr.new_zeros(B,P,3,1)],-1))
            return result
        finally:
            self.variant=original;self._motion_context=None;self._diagnostic_context=None

    def part_supervision(self,joints,K,valid,primary_size=896):
        # Preserve the original physical RGB heatmap width while changing grid.
        from .part_aggregation_model import PART_JOINTS
        maps=self.last_part_maps.float();G=self.grid
        centres=(torch.arange(G,device=maps.device).float()+.5)*primary_size/G-.5
        yy,xx=torch.meshgrid(centres,centres,indexing='ij');xy=torch.stack([xx,yy],-1).flatten(0,1)
        hom=joints.float()@K.float().transpose(-1,-2)[:,None];uv=hom[...,:2]/hom[...,2:].clamp_min(.01)
        targets=[];sigma=primary_size/16*1.3
        for group in PART_JOINTS:
            points=uv[:,:,list(group)];dist=(xy[None,None,None]-points[:,:,:,None]).square().sum(-1)
            vis=valid[:,:,None,None]&(hom[:,:,list(group),2:]>0)
            heat=(torch.exp(-dist/(2*sigma*sigma))*vis).amax((1,2))
            targets.append(heat/heat.sum(-1,keepdim=True).clamp_min(1e-8))
        target=torch.stack(targets,1).detach()
        return -(target*maps.clamp_min(1e-8).log()).sum(-1).mean()

    def load_coarse_parameters(self,state):
        state=dict(state);weight=state['encoder.0.weight']
        state['encoder.0.weight']=F.avg_pool2d(weight,2,2)*4
        position=state['position'].transpose(1,2).reshape(1,self.hidden,16,16)
        state['position']=F.interpolate(position,(32,32),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        loaded=self.load_state_dict(state,strict=False)
        assert not loaded.unexpected_keys and not set(loaded.missing_keys)&set(dict(self.named_parameters()))
        with torch.no_grad():self.output.weight.zero_()

def build_part_model(config,device='cuda',checkpoint=None):
    from .part_aggregation_v4 import build_part_model as build_coarse
    if config.get('v5_model_source_sha256'):
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['v5_model_source_sha256']
    model=build_coarse(config,device=device,checkpoint=None)
    fusion=DenseSpatialAggregation(model.base.embed_dim,config.get('hidden',128),variant=config['variant'],
        support=config.get('spatial_support',True),warmup_steps=config.get('uniform_aggregation_warmup_steps',100),
        static_horizon=config.get('static_part_horizon',False),path_prior=config.get('motion_path_prior_px')).to(device)
    fusion.load_coarse_parameters(model.fusion.state_dict())
    fusion.collect_support_diagnostics_in_eval=config.get('collect_support_diagnostics_in_eval',False)
    model.fusion=fusion
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert saved['test_used_for_selection'] is False and saved['successful_steps']>0
        assert saved['config']['architecture_version']==VERSION and saved['config']['variant']==config['variant']
        assert saved['config']['split_sha256']==config['split_sha256']
        assert saved['config']['frozen_RGB_checkpoint_sha256']==config['frozen_RGB_checkpoint_sha256']
        assert all(key.startswith('fusion.') for key in saved['model_delta'])
        loaded=model.load_state_dict(saved['model_delta'],strict=False)
        assert not loaded.unexpected_keys
        assert not set(loaded.missing_keys)&{name for name,p in model.named_parameters() if p.requires_grad}
    return model

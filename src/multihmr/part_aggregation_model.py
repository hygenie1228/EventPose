"""RGB-preserving, calibrated part-aware temporal event aggregation.

Anatomical part maps are predicted from RGB tokens. Ground-truth joints are
accepted only by the separate training loss, never by ``forward``.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .multisensor_repair import EpipolarEventFusion

PART_NAMES = ('torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'head')
PART_JOINTS = ((0,3,6,9,12), (13,16,18,20), (14,17,19,21),
               (1,4,7,10), (2,5,8,11), (12,15))


class PartTemporalAggregation(EpipolarEventFusion):
    """Factor calibrated depth matching, anatomical routing and temporal choice.

The three controls retain identical candidate observations. Part-agnostic
retains the same six-query parameterization but shares its time distribution.
"""
    def __init__(self, feature_dim=1024, hidden=128, grid=16, variant='part_aware', bins=3):
        super().__init__(2, feature_dim, hidden=hidden, grid=grid, normalize=True, input_gain=4.)
        self.variant=variant; self.bins=bins; self.hidden=hidden
        # A patch embedding and spatial Transformer, shared across time.
        self.encoder=nn.Sequential(nn.Conv2d(2,hidden,14,14),nn.GroupNorm(8,hidden))
        spatial=nn.TransformerEncoderLayer(hidden,4,hidden*3,dropout=0.,batch_first=True,norm_first=True)
        self.spatial_transformer=nn.TransformerEncoder(spatial,1,enable_nested_tensor=False)
        self.part_queries=nn.Parameter(torch.randn(6,hidden)*.02)
        self.rgb_norm=nn.LayerNorm(hidden)
        self.part_norm=nn.LayerNorm(hidden)
        self.time_embedding=nn.Parameter(torch.randn(bins,hidden)*.02)
        self.position=nn.Parameter(torch.randn(1,grid*grid,hidden)*.01)
        self.time_score=nn.Sequential(nn.Linear(hidden*3+1,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.null_score=nn.Linear(hidden,1)
        self.last_part_maps=None; self.last_time_weights=None; self.last_null_weights=None

    def forward(self,features,context):
        events,K,auxK,R,t,available,primary_size,bin_available=context
        B,N,C=features.shape; side=math.isqrt(N); G=self.grid; T=self.bins; H=self.hidden
        assert side*side==N
        pooled=F.adaptive_avg_pool2d(features.transpose(1,2).reshape(B,C,side,side),G).flatten(2).transpose(1,2)
        rgb=self.rgb_norm(self.query(pooled))
        # Anatomical queries read RGB; maps are learned without GT at inference.
        part_logits=torch.einsum('ph,bnh->bpn',self.part_queries,rgb+self.position)/math.sqrt(H)
        maps=part_logits.float().softmax(-1).to(rgb.dtype)
        part_rgb=self.part_norm(torch.einsum('bpn,bnh->bph',maps,rgb))
        self.last_part_maps=maps
        if self.variant=='rgb_only':
            self.residual_energy=features.sum()*0
            self.last_time_weights=features.new_zeros((B,6,T));self.last_null_weights=features.new_ones((B,6))
            return features
        assert tuple(events.shape[1:3])==(T,2)
        encoded=self.encoder(events.flatten(0,1)*self.input_gain)
        assert encoded.shape[-2:]==(G,G)
        tokens=encoded.flatten(2).transpose(1,2)
        tokens=self.spatial_transformer(tokens+self.position)
        encoded=tokens.transpose(1,2).reshape(B*T,H,G,G)
        with torch.autocast(features.device.type,enabled=False):
            grid,valid=self.projection(K,auxK,R,t,primary_size,events.shape[-1])
            # Conv14/stride14 patch centres are 6.5+14*i, unlike the old conv8 encoder.
            uv=(grid+1)*(events.shape[-1]-1)/2
            fg=2*((uv-6.5)/14)/(G-1)-1
            valid=valid & (fg.abs()<=1).all(-1)
        sample_grid=fg[:,None].expand(-1,T,-1,-1,-1).flatten(0,1)
        samples=F.grid_sample(encoded,sample_grid.to(encoded.dtype),align_corners=True,
                              padding_mode='zeros').permute(0,2,3,1).reshape(B,T,G*G,-1,H)
        depth_logits=torch.einsum('bnh,btnzh->btnz',rgb,samples)/math.sqrt(H)
        mask=valid[:,None]&bin_available[:,:,None,None].bool()
        dw=depth_logits.float().masked_fill(~mask,-1e4).softmax(-1)*mask
        dw=dw/dw.sum(-1,keepdim=True).clamp_min(1e-8)
        per_time=(dw.to(samples.dtype)[...,None]*samples).sum(-2)  # B,T,N,H
        part_event=torch.einsum('bpn,btnh->bpth',maps,per_time)
        pr=part_rgb[:,:,None,:].expand(-1,-1,T,-1)
        te=self.time_embedding[None,None].expand(B,6,-1,-1)
        occupancy=events.float().abs().mean((2,3,4))[:,None,:,None].expand(-1,6,-1,-1)
        logits=self.time_score(torch.cat([pr,part_event,te,occupancy.to(pr.dtype)],-1)).squeeze(-1)
        # Softmax includes an explicit option to use no event evidence.
        logits=logits.float().masked_fill(~bin_available[:,None,:].bool(),-1e4)
        null=self.null_score(part_rgb).float()
        weights=torch.cat([logits,null],-1).softmax(-1)
        if self.variant=='part_agnostic':weights=weights.mean(1,keepdim=True).expand(-1,6,-1)
        elif self.variant in ('fixed_all','fixed_recent'):
            w=bin_available.float()
            if self.variant=='fixed_recent':w=w*F.one_hot(torch.zeros(B,dtype=torch.long,device=w.device),T)
            w=w/w.sum(-1,keepdim=True).clamp_min(1)
            weights=torch.cat([w,1-w.sum(-1,keepdim=True)],-1)[:,None].expand(-1,6,-1)
        self.last_time_weights=weights[...,:T];self.last_null_weights=weights[...,-1]
        # Route per-part temporal preferences back to spatial RGB patches.
        assignment=maps/(maps.sum(1,keepdim=True).clamp_min(1e-8))
        local_weights=torch.einsum('bpn,bpt->bnt',assignment,weights[...,:T].to(maps.dtype))
        message=torch.einsum('bnt,btnh->bnh',local_weights,per_time)
        support=valid.any(-1)&bin_available.any(-1)[:,None]
        residual=self.output(message)*support[...,None]
        self.residual_energy=residual.float().square().mean()
        residual=residual.transpose(1,2).reshape(B,C,G,G)
        residual=F.interpolate(residual,(side,side),mode='bilinear',align_corners=False)
        return features+available[:,None,None].to(features.dtype)*residual.flatten(2).transpose(1,2)

    def part_supervision(self,joints,K,valid,primary_size=896):
        """Train-only weak joint-neighbourhood supervision; no segmentation GT."""
        maps=self.last_part_maps.float();B,P,N=maps.shape;G=self.grid
        pixels=(torch.arange(G,device=maps.device).float()+.5)*primary_size/G-.5
        yy,xx=torch.meshgrid(pixels,pixels,indexing='ij');xy=torch.stack([xx,yy],-1).flatten(0,1)
        hom=joints.float()@K.float().transpose(-1,-2)[:,None]
        uv=hom[...,:2]/hom[...,2:].clamp_min(.01)
        targets=[]
        sigma=primary_size/G*1.3
        for group in PART_JOINTS:
            points=uv[:,:,list(group)]
            dist=(xy[None,None,None]-points[:,:,:,None]).square().sum(-1)
            vis=valid[:,:,None,None]& (hom[:,:,list(group),2:]>0)
            heat=(torch.exp(-dist/(2*sigma*sigma))*vis).amax((1,2))
            targets.append(heat/heat.sum(-1,keepdim=True).clamp_min(1e-8))
        target=torch.stack(targets,1).detach()
        return -(target*maps.clamp_min(1e-8).log()).sum(-1).mean()


class PartAggregationMultiHMR(nn.Module):
    input_keys=('image','K','auxiliary_K','primary_to_aux_rotation','primary_to_aux_translation',
                'auxiliary_available','event_bins','event_bin_available')
    mode='rgb'
    def __init__(self,base,variant='part_aware',hidden=128):
        super().__init__();self.base=base;self.variant=variant
        self.fusion=PartTemporalAggregation(base.embed_dim,hidden=hidden,variant=variant)
        self._context=None
        self._hook_handle=base.backbone.register_forward_hook(self._fuse)
    def _fuse(self,module,inputs,output):
        return self.fusion(output,self._context) if self._context is not None else output
    def forward(self,image,K,auxiliary_K=None,primary_to_aux_rotation=None,primary_to_aux_translation=None,
                auxiliary_available=None,event_bins=None,event_bin_available=None,**kwargs):
        self._context=(event_bins,K,auxiliary_K,primary_to_aux_rotation,
                       primary_to_aux_translation,auxiliary_available,self.base.img_size,event_bin_available)
        try:return self.base(image,K=K,**kwargs)
        finally:self._context=None


def model_inputs(batch):
    return {k:batch[k] for k in PartAggregationMultiHMR.input_keys if k in batch}


def build_part_model(config,device='cuda',checkpoint=None):
    if config.get('model_source_sha256'):
        import hashlib
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['model_source_sha256'], 'Frozen aggregation forward source changed'
    from utils.pose_evaluation import load_multihmr
    base=load_multihmr(device=device)
    init=Path(config['rgb_initialization_checkpoint'])
    if str(init)!='released':
        saved=torch.load(init,map_location='cpu',weights_only=False)
        assert saved['test_used_for_selection'] is False and saved['successful_steps']>0
        assert saved['config']['condition']=='rgb3' and saved['config']['split_sha256']==config['split_sha256']
        delta={k.removeprefix('base.'):v for k,v in saved['model_delta'].items() if k.startswith('base.')}
        assert delta and not base.load_state_dict(delta,strict=False).unexpected_keys
    model=PartAggregationMultiHMR(base,config['variant'],config.get('hidden',128)).to(device)
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['successful_steps']>0 and saved['test_used_for_selection'] is False
        assert saved['config']['variant']==config['variant']
        assert saved['config']['split_sha256']==config['split_sha256']
        assert not model.load_state_dict(saved['model_delta'],strict=False).unexpected_keys
    return model

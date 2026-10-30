"""Prototype: reuse the fixed RGB ViT for causal event-bin representations.

The inner ViT method bypasses the RGB wrapper's fusion hook. It is shared,
frozen and always in eval mode; only the event projection and fusion learn.
"""
import copy,hashlib,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

VERSION='calibrated_part_semantic_frozen_rgb_v6'

class FrozenEventSemanticEncoder(nn.Module):
    def __init__(self,raw_encoder,projection,norm,size=448):
        super().__init__();self.raw_encoder=raw_encoder
        self.projection=copy.deepcopy(projection);self.norm=copy.deepcopy(norm);self.size=int(size)
        self.raw_encoder.requires_grad_(False);self.raw_encoder.eval()
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('image_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode);self.raw_encoder.eval();return self

    def pseudo_RGB(self,events):
        assert events.ndim==4 and events.shape[1:]==(2,224,224)
        values=events.float().clamp(0,1)
        support=(F.avg_pool2d((values.sum(1,keepdim=True)>0).float(),3,1,1)*9>=2).float()
        brightness=(values*(math.log1p(256)/math.log(9))).clamp(0,1).sqrt()*support
        positive,negative=brightness[:,0],brightness[:,1]
        rgb=torch.stack([positive,.25*positive+.65*negative,negative],1)
        return F.interpolate(rgb,(self.size,self.size),mode='bilinear',align_corners=False)

    def forward(self,events):
        image=self.pseudo_RGB(events)
        normalized=(image-self.image_mean)/self.image_std
        # No outer Dinov2Backbone.forward call: its RGB fusion hook must not
        # recursively process the event batch as another RGB query.
        with torch.no_grad():tokens=self.raw_encoder.get_intermediate_layers(normalized)[0]
        encoded=self.norm(self.projection(tokens))
        side=math.isqrt(encoded.shape[1]);assert side==32 and side*side==encoded.shape[1]
        return encoded.transpose(1,2).reshape(len(events),encoded.shape[-1],side,side)

def replace_with_semantic_encoder(model,size=448):
    fusion=model.fusion
    fusion.encoder=FrozenEventSemanticEncoder(model.base.backbone.encoder,fusion.query,fusion.rgb_norm,size).to(next(model.parameters()).device)
    fusion.input_gain=1.
    # Preserve an initially aligned RGB/event feature basis through the residual
    # spatial Transformer, then allow it to learn under the common pose loss.
    with torch.no_grad():
        for layer in fusion.spatial_transformer.layers:
            layer.self_attn.out_proj.weight.zero_();layer.self_attn.out_proj.bias.zero_()
            layer.linear2.weight.zero_();layer.linear2.bias.zero_()
        fusion.time_score.quality_head[-1].weight.zero_();fusion.time_score.quality_head[-1].bias.fill_(2.)
        fusion.output.weight.zero_()
    return model

def build_part_model(config,device='cuda',checkpoint=None):
    from .part_aggregation_v5 import build_part_model as build_dense
    if config.get('v6_model_source_sha256'):
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['v6_model_source_sha256']
    model=replace_with_semantic_encoder(build_dense(config,device=device,checkpoint=None),config.get('event_ViT_size',448))
    assert model.fusion.encoder.raw_encoder is model.base.backbone.encoder
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert saved['test_used_for_selection'] is False and saved['successful_steps']>0
        assert saved['config']['architecture_version']==VERSION and saved['config']['variant']==config['variant']
        assert saved['config']['split_sha256']==config['split_sha256']
        assert saved['config']['frozen_RGB_checkpoint_sha256']==config['frozen_RGB_checkpoint_sha256']
        selected={name for name,p in model.named_parameters() if p.requires_grad}
        assert set(saved['model_delta'])==selected
        assert all(name.startswith('fusion.') and not name.startswith('fusion.encoder.raw_encoder.') for name in selected)
        loaded=model.load_state_dict(saved['model_delta'],strict=False)
        assert not loaded.unexpected_keys and not set(loaded.missing_keys)&selected
    return model

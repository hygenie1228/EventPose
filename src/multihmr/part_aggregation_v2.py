"""Version 2: predicted spatial support and event-representation warmup.

The frozen v1 forward is unchanged. A hook gates its pre-upsampling residual
using absolute predicted part-map density, not an anatomical segmentation.
"""
import hashlib
from pathlib import Path
import torch
from .part_aggregation_model import PartTemporalAggregation,build_part_model as build_v1

VERSION='calibrated_part_temporal_transformer_v2'


class SpatialSupportAggregation(PartTemporalAggregation):
    def __init__(self,feature_dim=1024,hidden=128,grid=16,variant='part_aware',
                 support=True,warmup_steps=500):
        super().__init__(feature_dim,hidden,grid,variant)
        # Null must not leak a learned RGB-only projection bias.
        projection=self.output
        self.output=torch.nn.Linear(hidden,feature_dim,bias=False)
        with torch.no_grad():self.output.weight.copy_(projection.weight)
        self.use_spatial_support=bool(support)
        self.uniform_warmup_steps=int(warmup_steps)
        self.successful_training_step=0
        self.last_spatial_support=None
        self.last_support_diagnostics={}
        self.last_warmup_active=False
        self.collect_support_diagnostics_in_eval=True
        self._diagnostic_context=None
        self._spatial_support_hook=self.output.register_forward_hook(self._apply_support)

    def set_training_step(self,step):self.successful_training_step=int(step)

    def _apply_support(self,module,inputs,residual):
        maps=self.last_part_maps
        density=maps.float().sum(1)*(maps.shape[-1]/maps.shape[1])
        gate=density.clamp(0,1)
        self.last_spatial_support=gate
        if self.use_spatial_support:residual=residual*gate[...,None].to(residual.dtype)
        collect=(self.successful_training_step%25 in (0,24)) if self.training else self.collect_support_diagnostics_in_eval
        if not collect:return residual
        with torch.no_grad():
            supported=torch.ones_like(gate,dtype=torch.bool)
            if self._diagnostic_context is not None:
                events,K,auxK,R,t,available,size,bin_available=self._diagnostic_context
                with torch.autocast(residual.device.type,enabled=False):
                    grid,valid=self.projection(K,auxK,R,t,size,events.shape[-1])
                    uv=(grid+1)*(events.shape[-1]-1)/2
                    fg=2*((uv-6.5)/14)/(self.grid-1)-1
                    valid=valid&(fg.abs()<=1).all(-1)
                supported=valid.any(-1)&bin_available.any(-1)[:,None].bool()&available[:,None].bool()
            high=(gate>=.5)&supported;low=(gate<.5)&supported
            energy=residual.float().square().mean(-1)
            self.last_support_diagnostics=dict(
                predicted_high_density_patch_count=int(high.sum()),predicted_low_density_patch_count=int(low.sum()),
                high_density_residual_energy=float(energy[high].mean()) if high.any() else None,
                low_density_residual_energy=float(energy[low].mean()) if low.any() else None,
                mean_predicted_support=float(gate.mean()),
                calibration_supported_patch_count=int(supported.sum()),
                diagnostic_training_step=self.successful_training_step,
                interpretation='Predicted part-map density, not GT foreground/background segmentation')
        return residual

    def forward(self,features,context):
        original=self.variant
        self._diagnostic_context=context
        self.last_warmup_active=bool(self.training and self.successful_training_step<self.uniform_warmup_steps and original in ('part_aware','part_agnostic'))
        if self.last_warmup_active:
            self.variant='fixed_all'
        try:return super().forward(features,context)
        finally:
            self.variant=original
            self._diagnostic_context=None


def build_part_model(config,device='cuda',checkpoint=None):
    if config.get('v2_model_source_sha256'):
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['v2_model_source_sha256']
    model=build_v1(config,device=device,checkpoint=None)
    fusion=SpatialSupportAggregation(model.base.embed_dim,config.get('hidden',128),variant=config['variant'],
        support=config.get('spatial_support',True),warmup_steps=config.get('uniform_aggregation_warmup_steps',500)).to(device)
    # Identical initial parameters to the v1 module; only the stated policy changes.
    state=model.fusion.state_dict();state.pop('output.bias')
    fusion.load_state_dict(state,strict=True)
    if config.get('null_initial_bias') is not None:
        with torch.no_grad():fusion.null_score.bias.fill_(config['null_initial_bias'])
    model.fusion=fusion
    fusion.collect_support_diagnostics_in_eval=bool(config.get('collect_support_diagnostics_in_eval',False))
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['successful_steps']>0 and saved['test_used_for_selection'] is False
        assert saved['config']['architecture_version']==VERSION
        assert saved['config']['variant']==config['variant']
        assert saved['config']['split_sha256']==config['split_sha256']
        assert not model.load_state_dict(saved['model_delta'],strict=False).unexpected_keys
    return model

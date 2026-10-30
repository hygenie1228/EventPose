"""Calibrated device3/device4 adapters around the released Multi-HMR model.

The primary sensor supplies native Multi-HMR detection and reconstruction.
An optional separate auxiliary encoder attends through calibrated camera rays.
No GT depth, pose, box, or image-space correspondence enters this module.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .event_model import EventImageAdapter, CalibratedEventFusion


CONDITIONS = {
    'rgb3': ('rgb3', None),
    'rgb4': ('rgb4', None),
    'rgb3_rgb4': ('rgb3', 'rgb4'),
    'rgb3_event4': ('rgb3', 'event4'),
    'event3_rgb4': ('event3', 'rgb4'),
    'event3_event4': ('event3', 'event4'),
    'rgb3_event3': ('rgb3', 'event3'),
    'rgb4_event4': ('rgb4', 'event4'),
}


def checkpoint_frozen_blocks(base,last_blocks):
    """Preserve gradients to event input adapters without retaining every ViT activation."""
    from torch.utils.checkpoint import checkpoint
    blocks=list(base.backbone.encoder.blocks)
    frozen=blocks[:-last_blocks] if last_blocks else blocks
    for block in frozen:
        if getattr(block,'_multisensor_input_checkpoint',False):continue
        original=block.forward
        def forward(*args,_original=original,**kwargs):
            return checkpoint(_original,*args,use_reentrant=False,**kwargs)
        block.forward=forward
        block._multisensor_input_checkpoint=True


class MultiSensorMultiHMR(nn.Module):
    input_keys = ('image', 'event', 'K', 'auxiliary', 'auxiliary_K',
                  'primary_to_aux_rotation', 'primary_to_aux_translation', 'auxiliary_available')

    def __init__(self, base, condition, sensor_layout=None, fusion_factory=CalibratedEventFusion, event_channels=2):
        super().__init__()
        self.base = base
        self.condition = condition
        self.primary_sensor, self.auxiliary_sensor = sensor_layout or CONDITIONS[condition]
        self.mode = 'event' if self.primary_sensor.startswith('event') else 'rgb'
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
        if self.mode == 'event':
            self.adapter = EventImageAdapter(event_channels)
        if self.auxiliary_sensor is not None:
            channels = event_channels if self.auxiliary_sensor.startswith('event') else 3
            self.fusion = fusion_factory(channels, base.embed_dim)
            self._hook_handle = base.backbone.register_forward_hook(self._fuse)
        self._context = None

    def _fuse(self, module, inputs, output):
        return self.fusion(output, self._context) if self._context is not None else output

    def forward(self, image=None, event=None, K=None, auxiliary=None, auxiliary_K=None,
                primary_to_aux_rotation=None, primary_to_aux_translation=None,
                auxiliary_available=None, **kwargs):
        if self.mode == 'event':
            assert image is None, 'An event primary never receives RGB primary pixels'
            image = (self.adapter(event, self.base.img_size)-self.mean)/self.std
        else:
            assert event is None and image is not None
        if self.auxiliary_sensor is not None:
            assert auxiliary is not None
            self._context = (auxiliary, K, auxiliary_K, primary_to_aux_rotation,
                             primary_to_aux_translation, auxiliary_available, self.base.img_size)
        try:
            return self.base(image, K=K, **kwargs)
        finally:
            self._context = None


def model_inputs(batch, model):
    return {key: batch[key] for key in model.input_keys if key in batch}


def self_check():
    class Backbone(nn.Module):
        def forward(self, x):
            return F.adaptive_avg_pool2d(x, (4,4)).flatten(2).transpose(1,2).repeat(1,1,4)
    class Base(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=Backbone();self.embed_dim=12;self.img_size=16
        def forward(self, image, K=None, **kwargs):
            return self.backbone(image)
    K=torch.tensor([[[20.,0,7.5],[0,20.,7.5],[0,0,1.]]])
    passed=[]
    for condition,(primary,aux) in CONDITIONS.items():
        torch.manual_seed(9)
        model=MultiSensorMultiHMR(Base(),condition)
        batch=dict(K=K)
        if primary.startswith('event'):batch['event']=torch.rand(1,2,16,16)
        else:batch['image']=torch.rand(1,3,16,16)
        if aux:
            batch.update(auxiliary=torch.rand(1,2 if aux.startswith('event') else 3,16,16),
                auxiliary_K=K,primary_to_aux_rotation=torch.eye(3)[None],
                primary_to_aux_translation=torch.tensor([[.3,0.,0.]]),auxiliary_available=torch.ones(1))
        out=model(**model_inputs(batch,model));assert out.shape==(1,16,12)
        assert model._context is None
        if condition=='event3_event4':assert 'image' not in model_inputs(batch,model)
        if aux:
            out.square().mean().backward()
            assert model.fusion.output.weight.grad.abs().sum()>0
        passed.append(condition)
    return dict(conditions=passed,event_only_no_rgb=True,calibrated_fusion_gradient=True,context_cleared=True)

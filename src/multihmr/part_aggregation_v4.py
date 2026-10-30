"""Frozen strong RGB with a matched, warm-started event aggregation adapter.

V3 supplies Train-selected motion features to every event control. RGB weights
and module mode stay fixed; zero output projection starts at the strong RGB.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from .part_aggregation_model import PartAggregationMultiHMR
from .part_aggregation_v3 import build_part_model as build_v3

VERSION='calibrated_part_frozen_rgb_v4'

class FrozenRGBModel(PartAggregationMultiHMR):
    def __init__(self,base,fusion):
        nn.Module.__init__(self)
        self.base=base;self.fusion=fusion;self.variant=fusion.variant;self._context=None
        self._hook_handle=base.backbone.register_forward_hook(self._fuse)
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        return self

def _load_verified(path,expected_sha,split):
    path=Path(path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    assert digest.hexdigest()==expected_sha
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['test_used_for_selection'] is False and saved['successful_steps']>0
    assert saved['config']['split_sha256']==split
    return saved

def build_part_model(config,device='cuda',checkpoint=None):
    assert config['freeze_RGB_base']
    if config.get('v4_model_source_sha256'):
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['v4_model_source_sha256']
    original=build_v3(config,device=device,checkpoint=None)
    reference=_load_verified(config['frozen_RGB_checkpoint'],config['frozen_RGB_checkpoint_sha256'],config['split_sha256'])
    assert reference['config']['name']=='v2_rgb_only' and reference['config']['variant']=='rgb_only'
    delta={k.removeprefix('base.'):v for k,v in reference['model_delta'].items() if k.startswith('base.')}
    assert delta and not original.base.load_state_dict(delta,strict=False).unexpected_keys
    del reference,delta
    warm=_load_verified(config['fusion_initialization_checkpoint'],config['fusion_initialization_sha256'],config['split_sha256'])
    assert warm['config']['name']=='v3_part_dynamic'
    delta={k.removeprefix('fusion.'):v for k,v in warm['model_delta'].items() if k.startswith('fusion.')}
    loaded=original.fusion.load_state_dict(delta,strict=False)
    assert not loaded.unexpected_keys
    assert not set(loaded.missing_keys)&set(dict(original.fusion.named_parameters()))
    with torch.no_grad():original.fusion.output.weight.zero_()
    original._hook_handle.remove()
    model=FrozenRGBModel(original.base,original.fusion)
    del warm,delta,original
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert saved['test_used_for_selection'] is False and saved['successful_steps']>0
        assert saved['config']['architecture_version']==VERSION and saved['config']['variant']==config['variant']
        assert saved['config']['split_sha256']==config['split_sha256']
        assert saved['config']['frozen_RGB_checkpoint_sha256']==config['frozen_RGB_checkpoint_sha256']
        assert all(k.startswith('fusion.') for k in saved['model_delta'])
        loaded=model.load_state_dict(saved['model_delta'],strict=False)
        assert not loaded.unexpected_keys
        assert not set(loaded.missing_keys)&{n for n,p in model.named_parameters() if p.requires_grad}
    return model

"""RGB-preserving fusion with calibration-defined auxiliary sampling.

This version has independent checkpoint provenance. Old model layouts remain
unchanged. The event-only layout consumes no RGB observations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .multisensor_model import MultiSensorMultiHMR
from .event_model import EventImageAdapter

REPAIR_LAYOUTS = {
    'rgb3': ('rgb3', None),
    'rgb4': ('rgb4', None),
    'rgb3_event4': ('rgb3', 'event4'),
    'event3_rgb4': ('rgb4', 'event3'),
    'event3_event4': ('event3', 'event4'),
    'rgb3_event3': ('rgb3', 'event3'),
    'rgb4_event4': ('rgb4', 'event4'),
}


class EpipolarEventFusion(nn.Module):
    """Sample event features along calibrated rays at fixed positive Z depths."""
    def __init__(self, channels, feature_dim, hidden=128, grid=16, normalize=False,input_gain=1.):
        super().__init__()
        self.grid = grid
        self.input_gain=float(input_gain)
        layers=[]
        for a,b,k,p in [(channels,32,5,2),(32,64,3,1),(64,hidden,3,1)]:
            layers.append(nn.Conv2d(a,b,k,2,p))
            if normalize:layers.append(nn.GroupNorm(8,b))
            layers.append(nn.GELU())
        self.encoder=nn.Sequential(*layers)
        self.query = nn.Linear(feature_dim, hidden)
        self.output = nn.Linear(hidden, feature_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('depths_m', torch.tensor([.5, .75, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8.]))

    def projection(self, K, auxiliary_K, rotation, translation, primary_size, auxiliary_size):
        """Camera-Z samples, not GT depth or assumed pixel correspondence."""
        axis = (torch.arange(self.grid, device=K.device).float() + .5) * primary_size / self.grid - .5
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(1, -1, 3)
        rays = pixels @ torch.linalg.inv(K.float()).transpose(-1, -2)
        rays = rays / rays[..., 2:].clamp_min(1e-8)
        points = rays[:, :, None] * self.depths_m[None, None, :, None]
        points = points @ rotation.float().transpose(-1, -2)[:, None] + translation.float()[:, None, None]
        projected = points @ auxiliary_K.float().transpose(-1, -2)[:, None]
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
        grid = 2 * uv / (auxiliary_size - 1) - 1
        valid = (points[..., 2] > 0) & (grid.abs() <= 1).all(-1)
        return grid, valid

    def forward(self, features, context):
        events, K, auxiliary_K, rotation, translation, available, primary_size = context
        batch, count, dim = features.shape
        side = round(count ** .5)
        assert side ** 2 == count and events.shape[-1] == events.shape[-2]
        with torch.autocast(features.device.type, enabled=False):
            grid, valid = self.projection(K, auxiliary_K, rotation, translation, primary_size, events.shape[-1])
        encoded = self.encoder(events*self.input_gain)
        # Three odd-kernel, symmetric-padding stride-2 convolutions have centres
        # at input pixels 0,8,16,... . Do not stretch their 28 centres to 223.
        pixel_uv = (grid + 1) * (events.shape[-1] - 1) / 2
        feature_grid = 2 * (pixel_uv / 8) / (encoded.shape[-1] - 1) - 1
        valid = valid & (feature_grid.abs() <= 1).all(-1)
        sampled = F.grid_sample(encoded, feature_grid.to(encoded.dtype), align_corners=True,
                                mode='bilinear', padding_mode='zeros').permute(0, 2, 3, 1)
        pooled = F.adaptive_avg_pool2d(features.transpose(1, 2).reshape(batch, dim, side, side), self.grid)
        query = self.query(pooled.flatten(2).transpose(1, 2))
        logits = (query[:, :, None] * sampled).sum(-1) / query.shape[-1] ** .5
        # Finite masked softmax also handles rays with no auxiliary support.
        weights = logits.float().masked_fill(~valid, -1e4).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        message = (weights.to(sampled.dtype)[..., None] * sampled).sum(-2)
        residual = self.output(message) * valid.any(-1)[..., None]
        self.residual_energy=residual.float().square().mean()
        residual = residual.transpose(1, 2).reshape(batch, dim, self.grid, self.grid)
        residual = F.interpolate(residual, (side, side), mode='bilinear', align_corners=False)
        return features + available[:, None, None].to(features.dtype) * residual.flatten(2).transpose(1, 2)


class RepairedMultiSensorMultiHMR(MultiSensorMultiHMR):
    def __init__(self, base, condition,normalize=False,event_gains=None,event_channels=2):
        from functools import partial
        gains=event_gains or {};primary,auxiliary=REPAIR_LAYOUTS[condition]
        factory=partial(EpipolarEventFusion,normalize=normalize,input_gain=gains.get(auxiliary,1.))
        super().__init__(base, condition, sensor_layout=(primary,auxiliary), fusion_factory=factory,event_channels=event_channels)
        if self.mode == 'event':
            self.adapter = SpatialEventAdapter(normalize=normalize,input_gain=gains.get(primary,1.),channels=event_channels)


class SpatialEventAdapter(EventImageAdapter):
    """Retain the verified polarity projection and learn local event structure."""
    def __init__(self,normalize=False,input_gain=1.,channels=2):
        super().__init__(channels)
        self.input_gain=float(input_gain)
        layers=[nn.Conv2d(channels,32,3,padding=1)]
        if normalize:layers.append(nn.GroupNorm(8,32))
        self.refiner=nn.Sequential(*layers,nn.GELU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, events, size):
        projected = self.projection(events) + self.refiner(events*self.input_gain)
        return F.interpolate(projected, (size, size), mode='bilinear', align_corners=False)


def build_repair_model(base,config):
    assert config['architecture'] in ('epipolar_repair_v1','epipolar_repair_norm_v2')
    normalized=config['architecture']=='epipolar_repair_norm_v2'
    gains=config.get('event_encoder_gains',{})
    if normalized:
        import hashlib
        import json
        from pathlib import Path
        assert set(gains)=={'event3','event4'} and all(1<=v<=8 for v in gains.values())
        path=Path(config['output_root'])/'event_normalization_Train.json'
        raw=path.read_bytes();normalization=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==config['event_normalization_sha256']
        assert normalization['event_encoder_gains']==gains
        assert normalization['split_sha256']==config['split_sha256']
    from utils.event_encoding import ENCODINGS
    channels=ENCODINGS[config.get('encoding','count33')]['channels']
    return RepairedMultiSensorMultiHMR(base,config['condition'],normalize=normalized,event_gains=gains,event_channels=channels)


def load_verified_initialization(model, config):
    """Load only audited Train-selected base/primary adapter; discard old fusion."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(config['initialization_checkpoint'])
    base_manifest=json.loads((Path(config['output_root'])/'original_pretrained_manifest.json').read_text())
    expected_base=Path(__file__).resolve().parents[2]/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'
    assert Path(base_manifest['path'])==expected_base
    base_digest=hashlib.sha256()
    with expected_base.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):base_digest.update(block)
    assert base_digest.hexdigest()==base_manifest['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == config['initialization_sha256']
    if config.get('initialization_kind') == 'released_original':
        assert path.resolve()==expected_base.resolve()
        assert digest.hexdigest()==base_manifest['sha256']
        return dict(path=str(path),sha256=digest.hexdigest(),source_successful_updates=0,
                    copied_keys=0,old_fusion_loaded=False,test_used_for_selection=False,
                    original_pretrained_sha256=base_digest.hexdigest())
    saved = torch.load(path, map_location='cpu')
    assert saved['config']['condition'] == config['initialization_condition']
    assert saved['config']['split_sha256'] == config['split_sha256']
    assert saved['test_used_for_selection'] is False and saved['metrics_apply_to_checkpoint']
    assert saved['metrics_step'] == saved['step'] and saved['successful_steps'] > 0
    assert config['initialization_condition'] == (model.primary_sensor if model.mode == 'rgb' else 'event3_event4')
    state = {k: v for k, v in saved['model_delta'].items()
             if k.startswith('base.') or (model.mode == 'event' and k.startswith('adapter.projection.'))}
    assert state and all(torch.isfinite(v).all() for v in state.values())
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys
    return dict(path=str(path), sha256=digest.hexdigest(), source_step=saved['step'],
                source_successful_updates=saved['successful_steps'], copied_keys=len(state),
                old_fusion_loaded=False, test_used_for_selection=False,
                original_pretrained_sha256=base_digest.hexdigest())


def implementation_provenance(snapshot_root=None):
    import hashlib
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    names = ['scripts/train_multihmr.py', 'scripts/evaluate_multihmr.py',
             'src/multihmr/multisensor_repair.py', 'src/multihmr/multisensor_model.py',
             'src/multihmr/multisensor_data.py', 'src/multihmr/multisensor_inference.py',
             'src/utils/pose_evaluation.py', 'src/multihmr/pretrained_inference.py',
             'src/utils/event_encoding.py','src/multihmr/event_model.py']
    result={}
    for name in names:
        data=(root/name).read_bytes();digest=hashlib.sha256(data).hexdigest();result[name]=digest
        if snapshot_root is not None:
            folder=Path(snapshot_root);folder.mkdir(parents=True,exist_ok=True)
            destination=folder/(digest+'.txt')
            if not destination.exists():
                temporary=folder/(digest+'.'+str(os.getpid())+'.tmp')
                temporary.write_bytes(data);temporary.replace(destination)
            assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest
    return result

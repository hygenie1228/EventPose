"""Event-input and calibrated late-fusion adapters around the unchanged Multi-HMR.

Event-only receives no RGB pixels. Fusion uses sensor-specific rays and camera
origin, not a pixelwise RGB/event overlay or GT-dependent depth reprojection.
"""
import torch
from torch import nn
from torch.nn import functional as F


def rays(K, grid, image_size):
    axis = (torch.arange(grid, device=K.device, dtype=torch.float32) + .5) * image_size / grid - .5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(1, -1, 3)
    direction = pixels @ torch.linalg.inv(K.float()).transpose(-1, -2)
    return F.normalize(direction, dim=-1)


class EventImageAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(channels, 3, 1)
        with torch.no_grad():
            self.projection.weight.zero_(); self.projection.bias.zero_()
            if channels == 1:
                self.projection.weight[:, 0, 0, 0] = 1
            else:
                assert channels % 2 == 0
                self.projection.bias.fill_(.5)
                pairs = channels // 2
                self.projection.weight[0, 0::2, 0, 0] = .5 / pairs
                self.projection.weight[2, 1::2, 0, 0] = -.5 / pairs
                self.projection.weight[1, 0::2, 0, 0] = .25 / pairs
                self.projection.weight[1, 1::2, 0, 0] = -.25 / pairs

    def forward(self, events, size):
        return F.interpolate(self.projection(events), (size, size), mode='bilinear', align_corners=False)


class CalibratedEventFusion(nn.Module):
    def __init__(self, channels, feature_dim, hidden=128, grid=16):
        super().__init__()
        self.grid = grid
        self.encoder = nn.Sequential(nn.Conv2d(channels, 16, 5, 2, 2), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, hidden, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((grid, grid)))
        self.query = nn.Linear(feature_dim, hidden)
        self.geometry = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0, batch_first=True)
        self.output = nn.Linear(hidden, feature_dim)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, rgb_features, context):
        batch, count, dim = rgb_features.shape
        side = round(count ** .5)
        assert side ** 2 == count
        events, K_rgb, K_event, rotation, translation, available, rgb_size = context
        grid = self.grid
        # Geometry is solved in float32, outside the AMP matrix inverse path.
        with torch.autocast(rgb_features.device.type, enabled=False):
            rgb_rays = rays(K_rgb, grid, rgb_size)
            event_rays = rays(K_event, grid, events.shape[-1]) @ rotation.float()
            event_origin = -(translation.float().unsqueeze(1) @ rotation.float())
            event_geometry = torch.cat([event_rays, event_origin.expand(-1, grid ** 2, -1)], -1)
            rgb_geometry = torch.cat([rgb_rays, torch.zeros_like(rgb_rays)], -1)
        image_map = rgb_features.transpose(1, 2).reshape(batch, dim, side, side)
        query = F.adaptive_avg_pool2d(image_map, (grid, grid)).flatten(2).transpose(1, 2)
        query = self.query(query) + self.geometry(rgb_geometry)
        values = self.encoder(events).flatten(2).transpose(1, 2) + self.geometry(event_geometry)
        residual, _ = self.attention(query, values, values, need_weights=False)
        residual = self.output(residual).transpose(1, 2).reshape(batch, dim, grid, grid)
        residual = F.interpolate(residual, (side, side), mode='bilinear', align_corners=False)
        residual = residual.flatten(2).transpose(1, 2)
        return rgb_features + self.gate.sigmoid() * available[:, None, None].to(residual.dtype) * residual


class EventMultiHMR(nn.Module):
    def __init__(self, base, mode, channels=0):
        super().__init__()
        assert mode in ('rgb', 'event', 'fusion')
        self.base = base; self.mode = mode; self._context = None
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        if mode == 'event':
            self.adapter = EventImageAdapter(channels)
        elif mode == 'fusion':
            self.fusion = CalibratedEventFusion(channels, base.embed_dim)
            self._hook_handle = base.backbone.register_forward_hook(self._fuse)

    def _fuse(self, module, inputs, output):
        return self.fusion(output, self._context) if self._context is not None else output

    def forward(self, image=None, event=None, K=None, event_K=None,
                event_rotation=None, event_translation=None, event_available=None, **kwargs):
        if self.mode == 'event':
            assert image is None, 'RGB pixels are forbidden for event-only'
            image = (self.adapter(event, self.base.img_size) - self.mean) / self.std
        elif self.mode == 'fusion':
            assert event is not None and image is not None
            self._context = (event, K, event_K, event_rotation, event_translation,
                             event_available, self.base.img_size)
        try:
            return self.base(image, K=K, **kwargs)
        finally:
            self._context = None


def self_check():
    """Check geometry, exact pretrained initialization, and missing-event fallback."""
    torch.manual_seed(261024)
    K = torch.tensor([[[20., 0, 7.5], [0, 20., 7.5], [0, 0, 1.]]])
    r = rays(K, 2, 16)
    assert r.shape == (1, 4, 3) and torch.allclose(r.norm(dim=-1), torch.ones(1, 4))
    adapter = EventImageAdapter(1)
    gray = torch.full((1, 1, 8, 8), 128 / 255)
    assert torch.allclose(adapter(gray, 16), torch.full((1, 3, 16, 16), 128 / 255))
    fusion = CalibratedEventFusion(2, 16, hidden=16, grid=2)
    features = torch.randn(1, 16, 16)
    event = torch.rand(1, 2, 16, 16)
    context = (event, K, K, torch.eye(3)[None], torch.zeros(1, 3), torch.ones(1), 16)
    assert torch.equal(fusion(features, context), features)
    fusion(features, context).square().mean().backward()
    assert fusion.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        fusion.output.weight.normal_()
    missing = (*context[:5], torch.zeros(1), context[-1])
    assert torch.equal(fusion(features, missing), features)
    return dict(calibrated_rays=True, grayscale_initialization=True, pretrained_identity=True,
                fusion_gradient=True, missing_event_fallback=True)


if __name__ == '__main__':
    print(self_check())

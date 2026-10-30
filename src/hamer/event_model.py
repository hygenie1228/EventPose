"""Official HaMeR ViT-H/MANO head with the body study's calibrated fusion."""
from pathlib import Path
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F

from multihmr.multisensor_repair import EpipolarEventFusion, SpatialEventAdapter

LAYOUTS={'rgb3':('rgb3',None),'rgb3_event4':('rgb3','event4'),
         'event3_event4':('event3','event4'),'rgb3_event3':('rgb3','event3')}


class HandEpipolarFusion(EpipolarEventFusion):
    """Same fixed-depth ray attention on HaMeR's rectangular 16x12 feature map."""
    def forward(self,features,auxiliary,K,auxiliary_K,rotation,translation,available):
        b,dim,h,w=features.shape
        with torch.autocast(features.device.type,enabled=False):
            # ViT takes img[:,:,:,32:-32] from the 256x256 hand crop.
            yy,xx=torch.meshgrid((torch.arange(h,device=K.device)+.5)*256/h-.5,
                (torch.arange(w,device=K.device)+.5)*192/w+32-.5,indexing='ij')
            pixels=torch.stack([xx,yy,torch.ones_like(xx)],-1).reshape(1,-1,3)
            rays=pixels@torch.linalg.inv(K.float()).transpose(-1,-2)
            points=rays[:,:,None]*self.depths_m[None,None,:,None]
            points=points@rotation.float().transpose(-1,-2)[:,None]+translation.float()[:,None,None]
            hom=points@auxiliary_K.float().transpose(-1,-2)[:,None]
            uv=hom[...,:2]/hom[...,2:].clamp_min(1e-8)
        encoded=self.encoder(auxiliary*self.input_gain)
        grid=2*(uv/8)/(encoded.shape[-1]-1)-1
        valid=(points[...,2]>.05)&(grid.abs()<=1).all(-1)
        sampled=F.grid_sample(encoded,grid.to(encoded.dtype),align_corners=True).permute(0,2,3,1)
        query=self.query(features.flatten(2).transpose(1,2))
        logits=(query[:,:,None]*sampled).sum(-1)/query.shape[-1]**.5
        weights=logits.float().masked_fill(~valid,-1e4).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        message=(weights.to(sampled.dtype)[...,None]*sampled).sum(-2)
        residual=self.output(message)*valid.any(-1)[...,None]
        self.residual_energy=residual.float().square().mean()
        return features+available[:,None,None,None]*residual.transpose(1,2).reshape(b,dim,h,w)


class EventHaMeR(nn.Module):
    def __init__(self,cfg,condition,channels=2):
        super().__init__()
        from .models.backbones import create_backbone
        from .models.heads import build_mano_head
        from .models.mano_wrapper import MANO
        self.backbone=create_backbone(cfg);self.mano_head=build_mano_head(cfg)
        self.mano=MANO(**{k.lower():v for k,v in dict(cfg.MANO).items()})
        self.last_blocks=8
        self.primary,self.auxiliary=LAYOUTS[condition]
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        if self.primary.startswith('event'):
            self.adapter=SpatialEventAdapter(normalize=True,input_gain=4.,channels=channels)
        if self.auxiliary:
            self.fusion=HandEpipolarFusion(channels,1280,normalize=True,input_gain=4.)

    def forward(self,batch):
        assert 'joints' not in batch,'GT joints must not enter inference'
        if self.primary.startswith('event'):
            assert 'image' not in batch,'Event-only path received RGB pixels'
            x=(self.adapter(batch['event'],256)-self.mean)/self.std
        else:x=batch['image']
        features=self.backbone(x[:,:,:,32:-32])
        if self.auxiliary:
            features=self.fusion(features,batch['auxiliary'],batch['K'],batch['auxiliary_K'],
                batch['rotation'],batch['translation'],batch['auxiliary_available'])
        params,cam,_=self.mano_head(features)
        # MANO matrix rotations and skinning run in FP32, as in the official model.
        with torch.autocast(x.device.type,enabled=False):
            output=self.mano(**{k:v.float() for k,v in params.items()},pose2rot=False)
            joints=output.joints.float();vertices=output.vertices.float()
            sign=(batch['right'].float()*2-1)
            mirror=torch.stack([sign,torch.ones_like(sign),torch.ones_like(sign)],-1)
            joints=joints*mirror[:,None];vertices=vertices*mirror[:,None]
            # Decode weak-perspective camera using calibrated intrinsics, not demo's
            # guessed focal length. K_native is physical, unflipped sensor geometry.
            K=batch['native_K'].float();center=batch['box_center'].float()
            size=batch['box_size'].float();s=cam[:,0].float().clamp_min(1e-5)
            depth=2*torch.sqrt(K[:,0,0]*K[:,1,1])/(size*s)
            tx=sign*cam[:,1].float()+(center[:,0]-K[:,0,2])*depth/K[:,0,0]
            ty=cam[:,2].float()+(center[:,1]-K[:,1,2])*depth/K[:,1,1]
            translation=torch.stack([tx,ty,depth],-1)
            sensor_joints=joints+translation[:,None]
            common=(sensor_joints-batch['sensor_t'][:,None])@batch['sensor_R']
        return dict(joints=common,vertices=vertices,cam=cam,mano=params)

    def configure_trainable(self,last_blocks=8):
        self.last_blocks=last_blocks
        for p in self.parameters():p.requires_grad_(False)
        for p in self.mano_head.parameters():p.requires_grad_(True)
        for block in list(self.backbone.blocks)[-last_blocks:] if last_blocks else []:
            for p in block.parameters():p.requires_grad_(True)
        for key in ('fusion','adapter'):
            if hasattr(self,key):
                for p in getattr(self,key).parameters():p.requires_grad_(True)
        # Checkpointing preserves event adapter gradients through frozen ViT blocks.
        if self.primary.startswith('event'):self.backbone.use_checkpoint=True

    def train(self,mode=True):
        super().train(mode)
        # Frozen pretrained blocks should not inject stochastic-depth noise.
        blocks=list(self.backbone.blocks)
        for block in blocks[:-self.last_blocks] if self.last_blocks else blocks:block.eval()
        self.mano.eval()
        return self


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_model(config,device='cpu',finetuned=None):
    from .configs import get_config
    from utils.event_encoding import ENCODINGS
    root=Path(config['output_root']);manifest=json.loads((root/'pretrained_manifest.json').read_text())
    path=Path(manifest['checkpoint']);assert file_sha256(path)==manifest['checkpoint_sha256']
    assert file_sha256(manifest['config'])==manifest['config_sha256']
    assert file_sha256(manifest['mean_params'])==manifest['mean_params_sha256']
    if 'licensed_MANO_right_sha256' in manifest:
        assert file_sha256(Path(__file__).resolve().parents[2]/'data/human_model_files/mano/MANO_RIGHT.pkl')==manifest['licensed_MANO_right_sha256']
    cfg=get_config(manifest['config'],update_cachedir=False);cfg.defrost()
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    cfg.MANO.MODEL_PATH=str(Path(__file__).resolve().parents[2]/'data/human_model_files/mano')
    cfg.MANO.MEAN_PARAMS=manifest['mean_params'];cfg.freeze()
    torch.manual_seed(config['seed'])
    model=EventHaMeR(cfg,config['condition'],ENCODINGS[config['encoding']]['channels'])
    saved=torch.load(path,map_location='cpu',weights_only=False)
    state=saved['state_dict'];expected=model.state_dict()
    base={k:v for k,v in state.items() if k.startswith(('backbone.','mano_head.','mano.'))}
    result=model.load_state_dict(base,strict=False)
    assert not result.unexpected_keys,result.unexpected_keys
    assert all(k.startswith(('adapter.','fusion.','mean','std')) for k in result.missing_keys),result.missing_keys
    assert all(torch.equal(expected[k],v) for k,v in base.items()),'Released tensor load mismatch'
    if finetuned:
        tuned=torch.load(finetuned,map_location='cpu',weights_only=False)
        for key in ('condition','encoding','split_sha256','seed'):assert tuned['config'][key]==config[key]
        assert tuned['test_used_for_selection'] is False and tuned['successful_steps']>0
        assert tuned['metrics_step']==tuned['step']
        assert not model.load_state_dict(tuned['model_delta'],strict=False).unexpected_keys
    model.configure_trainable(config.get('last_blocks',8))
    return model.to(device)

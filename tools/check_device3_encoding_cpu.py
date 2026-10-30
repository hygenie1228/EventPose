"""Channel/geometry/gradient and original-initialization preflight, no Test."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from device3_encoding_scope import GOAL,TRAIN_NAMES
from goal_runtime_control import write,register_job

def main():
    g=json.loads(GOAL.read_text());register_job(g,os.getpid(),'encoding_cpu_preflight');out=Path(g['output_root'])
    os.environ['EVENTPOSE_MULTISENSOR_OUT']=str(out)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from utils.event_encoding import ENCODINGS,self_check
    from multihmr.multisensor_repair import build_repair_model
    torch.set_num_threads(2)
    class Backbone(nn.Module):
        def __init__(self):super().__init__();self.scale=nn.Parameter(torch.ones(()))
        def forward(self,x):return self.scale*F.adaptive_avg_pool2d(x,(4,4)).flatten(2).transpose(1,2).repeat(1,1,4)
    class Base(nn.Module):
        def __init__(self):super().__init__();self.backbone=Backbone();self.embed_dim=12;self.img_size=32
        def forward(self,image,**kwargs):return self.backbone(image)
    while not (out/'source_reuse_audit.json').exists():
        assert time.time()<g['deadline_unix']-300;time.sleep(5)
    records=[]
    for n in TRAIN_NAMES:
        c=json.loads((out/'configs'/(n+'.json')).read_text());torch.manual_seed(c['seed'])
        model=build_repair_model(Base(),c);channels=ENCODINGS[c['encoding']]['channels']
        K=torch.tensor([[[20.,0,15.5],[0,20.,15.5],[0,0,1.]]])
        batch=dict(K=K)
        if model.mode=='rgb':batch['image']=torch.rand(1,3,32,32)
        else:batch['event']=torch.rand(1,channels,32,32)
        if c['auxiliary_sensor']:
            batch.update(auxiliary=torch.rand(1,channels,32,32),auxiliary_K=K,
                primary_to_aux_rotation=torch.eye(3)[None],primary_to_aux_translation=torch.tensor([[.3,0,0]]),
                auxiliary_available=torch.ones(1))
        first=model(**batch);assert torch.isfinite(first).all()
        if c['auxiliary_sensor']:
            missing=model(**dict(batch,auxiliary_available=torch.zeros(1)))
            assert torch.equal(first,missing),'Zero residual must preserve primary pretrained path exactly'
        opt=torch.optim.AdamW(model.parameters(),lr=.001);grads=[]
        for _ in range(3):
            opt.zero_grad();y=model(**batch);loss=(y-2.).square().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if c['auxiliary_sensor']:grads.append(sum(float(p.grad.abs().sum()) for p in model.fusion.encoder.parameters() if p.grad is not None))
            opt.step()
        if grads:assert max(grads)>0
        if model.mode=='event':assert 'image' not in batch and model.adapter.projection.in_channels==channels
        records.append(dict(name=n,channels=channels,finite=True,encoder_gradient=max(grads) if grads else None,
                            rgb_in_event_only=False,initial_residual_zero=True))
    write(out/'encoding_cpu_preflight.json',dict(status='passed',checked_unix=time.time(),records=records,event_encoding=self_check(),
        scope='CPU channel/gradient/geometry contract; actual native GPU eight-step pilots required before each main trial'))
    print(json.dumps(dict(status='passed',conditions=len(records))),flush=True)
if __name__=='__main__':main()

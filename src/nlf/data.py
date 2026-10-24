from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Calibrated Train-only SMPL-X person crops backed by existing SSD images."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'.experiment_storage/multihmr_finetune_14h'

class PersonCrops(Dataset):
    def __init__(self,role='training',resolution=384,augment=True):
        assert role in ['training','validation']
        self.frames=json.loads((CACHE/f'{role}_index.json').read_text())
        self.resolution=resolution;self.augment=augment;self.cache=OrderedDict()
        self.people=[];self.excluded_behind_camera=0
        for i,record in enumerate(self.frames):
            z=self.targets(record['session'])
            for person in np.flatnonzero(z['valid'][record['row']]):
                joints=z['joints'][record['row'],person]
                if not np.isfinite(joints).all() or (joints[:,2]>.05).sum()<3:
                    self.excluded_behind_camera+=1
                    continue
                self.people.append((i,int(person)))
        self.cache.clear()

    def targets(self,session):
        if session not in self.cache:
            with np.load(CACHE/'targets'/f'{session}.npz') as z:
                self.cache[session]={k:z[k] for k in ['joints','valid','K']}
            if len(self.cache)>12:self.cache.popitem(last=False)
        return self.cache[session]

    def __len__(self):return len(self.people)

    def __getitem__(self,index):
        frame_id,person=self.people[index];row=self.frames[frame_id]
        z=self.targets(row['session']);j=z['joints'][row['row'],person].copy()
        K=z['K'].copy();assert np.isfinite(j).all()
        projection=j@K.T
        uv=projection[:,:2]/projection[:,2:]
        visible=(j[:,2]>.05)&np.isfinite(uv).all(-1)
        assert visible.sum()>=3
        low=uv[visible].min(0);high=uv[visible].max(0)
        center=(low+high)/2;side=max(float((high-low).max())*1.2,32.)
        if self.augment:
            side*=np.random.uniform(.9,1.15)
            center+=np.random.uniform(-.04,.04,2)*side
        left,top=np.floor(center-side/2).astype(int)
        side=int(np.ceil(side));size=self.resolution
        with Image.open(CACHE/'images'/row['session']/f'{row["frame"]:06d}.jpg') as im:
            image=im.convert('RGB').crop((left,top,left+side,top+side))
            image=image.resize((size,size),Image.Resampling.BILINEAR)
            if self.augment:image=ImageEnhance.Brightness(image).enhance(np.random.uniform(.85,1.15))
            image=np.array(image,dtype=np.float32)/255.
        K[0,2]-=left;K[1,2]-=top;K[:2]*=size/side
        return dict(image=torch.from_numpy(image).permute(2,0,1),
                    K=torch.from_numpy(K).float(),joints=torch.from_numpy(j).float())

def pa_loss(pred,target):
    """Proper similarity alignment, per-person body22, in metres."""
    x=pred[:,:22]-pred[:,:1];y=target[:,:22]-target[:,:1]
    x=x-x.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True)
    covariance=x.transpose(1,2)@y
    u,s,vh=torch.linalg.svd(covariance)
    sign=torch.linalg.det(u@vh).detach()
    correction=torch.ones_like(s);correction[:,-1]=sign
    rotation=(u*correction[:,None,:])@vh
    scale=(s*correction).sum(-1)/x.square().sum((1,2)).clamp_min(1e-8)
    aligned=(x@rotation)*scale[:,None,None]
    return torch.linalg.vector_norm(aligned-y,dim=-1).mean()

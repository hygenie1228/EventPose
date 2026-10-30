"""Oracle GT hand crops; RGB-only never reads event pixels."""
from collections import OrderedDict
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HandFrames(Dataset):
    def __init__(self,config,role,augment=False,records=None,limit=0):
        self.root=Path(config['output_root']);self.config=config;self.augment=augment
        self.records=records if records is not None else json.loads((self.root/(role+'_index.json')).read_text())
        if limit:self.records=[self.records[int(i)] for i in np.linspace(0,len(self.records)-1,min(limit,len(self.records)),dtype=int)]
        assert self.records
        self.handles=OrderedDict()

    def __len__(self):return len(self.records)

    def handle(self,session):
        if session not in self.handles:
            self.handles[session]=h5py.File(self.root/'cache'/(session+'.h5'),'r',locking=False)
            f=self.handles[session];assert f.attrs['complete'] and f.attrs['split_sha256']==self.config['split_sha256']
            assert f.attrs['cache_version']==2
            if len(self.handles)>4:_,old=self.handles.popitem(last=False);old.close()
        self.handles.move_to_end(session)
        return self.handles[session]

    def __getitem__(self,index):
        record=self.records[index];f=self.handle(record['session']);i=record['row'];fi=int(f['frame_row'][i])
        value=f['rgb3'][i].astype(np.float32)/255
        gain=float(np.random.uniform(.9,1.1)) if self.augment else 1.
        if self.augment:value=np.clip(value*gain,0,1)
        value=(value-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
        result=dict(image=value,joints=f['joints'][i],right=f['right'][i].astype(np.float32),K=f['K'][i],
            native_K=f['native_K'][:],box_center=f['box_center'][i],box_size=f['box_size'][i],
            sensor_R=np.eye(3,dtype=np.float32),sensor_t=np.zeros(3,np.float32),index=np.int64(index))
        if self.config['condition']!='rgb3':
            auxiliary=f['event4_'+self.config['encoding']][fi].astype(np.float32)/255
            if self.augment:auxiliary=np.clip(auxiliary*gain,0,1)
            result.update(auxiliary=auxiliary,auxiliary_K=f['auxiliary_K'][:],rotation=f['rotation'][:],
                translation=f['translation'][:],auxiliary_available=np.float32(f['auxiliary_available'][fi]))
        return {k:torch.as_tensor(v) for k,v in result.items()}

    def __getstate__(self):
        state=dict(self.__dict__);state['handles']=OrderedDict();return state


def to_device(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def model_inputs(batch):return {k:v for k,v in batch.items() if k not in ('joints','index')}

"""Common-clock body samples with explicitly separated sensor coordinates."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .multisensor_model import CONDITIONS

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('EVENTPOSE_MULTISENSOR_OUT',ROOT/'exp/261026/device34_body_28h'))


def input_keys(batch):
    from .multisensor_model import MultiSensorMultiHMR
    return {key:batch[key] for key in MultiSensorMultiHMR.input_keys if key in batch}


class Frames(Dataset):
    def __init__(self,role,mode=None,encoding=None,augment=False,limit=0,sessions=None,condition=None,index_path=None,sensor_layout=None):
        assert role in ('training','validation') and condition in CONDITIONS
        self.records=json.loads((Path(index_path) if index_path else OUT/(role+'_index.json')).read_text())
        if sessions is not None:self.records=[r for r in self.records if r['session'] in set(sessions)]
        if limit:
            self.records=[self.records[int(i)] for i in np.linspace(0,len(self.records)-1,min(limit,len(self.records)),dtype=int)]
        assert self.records
        self.primary,self.auxiliary=sensor_layout or CONDITIONS[condition]
        self.encoding=encoding or 'count33'
        self.cache_root=OUT/'encoding_cache'/self.encoding if (OUT/'encoding_cache'/self.encoding).exists() else OUT/'cache'
        self.augment=augment
        self.handles=OrderedDict();self.targets=OrderedDict()

    def __len__(self):return len(self.records)

    def handle(self,session):
        if session not in self.handles:
            self.handles[session]=h5py.File(self.cache_root/(session+'.h5'),'r',locking=False)
            assert self.handles[session].attrs['complete']
            assert self.handles[session].attrs['event_encoding']==self.encoding
            if len(self.handles)>8:
                _,old=self.handles.popitem(last=False);old.close()
        self.handles.move_to_end(session)
        return self.handles[session]

    def target(self,session):
        if session not in self.targets:
            with np.load(OUT/'targets'/(session+'.npz'),allow_pickle=False) as z:
                self.targets[session]={k:z[k] for k in z.files}
            if len(self.targets)>8:self.targets.popitem(last=False)
        self.targets.move_to_end(session)
        return self.targets[session]

    def __getitem__(self,index):
        row=self.records[index];name=row['session'];i=row['cache_row']
        f=self.handle(name);target=self.target(name)
        assert int(f['frames'][i])==row['frame']
        primary=self.primary
        R=f[primary+'_R_common_to_sensor'][:].astype(np.float32)
        t=f[primary+'_t_common_to_sensor'][:].astype(np.float32)
        fields={k:target[k][i].copy() for k in ('joints','centers','rotations','betas','valid')}
        fields['joints']=fields['joints']@R.T+t
        fields['centers']=fields['centers']@R.T+t
        fields['rotations'][:,0]=R@fields['rotations'][:,0]
        result=dict(fields,index=index,K=f[primary+'_K_primary'][:].astype(np.float32),
            original_K=f[primary+'_K_native'][:].astype(np.float32),
            image_shape=f[primary+'_native_shape'][:].astype(np.int64))
        assert f[primary+'_available'][i]
        gain=float(np.random.uniform(.9,1.1)) if self.augment else 1.
        value=torch.from_numpy(f[primary][i].astype(np.float32)/255)
        if primary.startswith('rgb'):
            value=torch.clamp(value*gain,0,1)
            result['image']=(value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        else:result['event']=torch.clamp(value*gain,0,1)
        if self.auxiliary is not None:
            aux=self.auxiliary
            value=torch.from_numpy(f[aux][i].astype(np.float32)/255)
            if aux.startswith('rgb'):
                value=F.interpolate(value[None],size=(224,224),mode='area')[0]
                value=(torch.clamp(value*gain,0,1)-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            else:value=torch.clamp(value*gain,0,1)
            Ra=f[aux+'_R_common_to_sensor'][:].astype(np.float32)
            ta=f[aux+'_t_common_to_sensor'][:].astype(np.float32)
            relative=Ra@R.T
            result.update(auxiliary=value,auxiliary_K=f[aux+'_K_auxiliary'][:].astype(np.float32),
                primary_to_aux_rotation=relative,primary_to_aux_translation=ta-relative@t,
                auxiliary_available=np.float32(f[aux+'_available'][i]))
        return result

    def __getstate__(self):
        value=dict(self.__dict__);value['handles']=OrderedDict();return value


def collate(items):
    variable={'joints','centers','rotations','betas','valid'}
    batch={key:torch.stack([torch.as_tensor(item[key]) for item in items])
           for key in items[0] if key not in variable|{'index'}}
    people=max(len(item['valid']) for item in items)
    for key in variable:
        array=np.zeros((len(items),people,*items[0][key].shape[1:]),dtype=items[0][key].dtype)
        for i,item in enumerate(items):array[i,:len(item[key])]=np.nan_to_num(item[key])
        batch[key]=torch.from_numpy(array)
    batch['index']=[item['index'] for item in items]
    return batch

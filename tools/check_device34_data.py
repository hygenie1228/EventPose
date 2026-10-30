"""Check nonidentity camera transforms and forbid RGB reads in event-only."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import h5py
import numpy as np
import torch
import multihmr.multisensor_data as data
import multihmr.multisensor_inference as inference
from multihmr.multisensor_model import CONDITIONS
from tools.prepare_device34_body import OUT,save


def main():
    torch.set_num_threads(2);(OUT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT/'tmp') as tmp:
        data.OUT=inference.OUT=Path(tmp)
        for folder in ('cache','targets'):(data.OUT/folder).mkdir()
        records=[dict(session='synthetic_b',cache_row=i,frame=10+i) for i in range(2)]
        for role in ('training','validation'):(data.OUT/(role+'_index.json')).write_text(json.dumps(records))
        joints=np.random.default_rng(8).normal(size=(2,1,55,3)).astype('f');joints[...,2]+=5
        rotations=np.broadcast_to(np.eye(3,dtype='f'),(2,1,53,3,3)).copy()
        np.savez(data.OUT/'targets/synthetic_b.npz',joints=joints,centers=joints[:,:,15],rotations=rotations,
            betas=np.zeros((2,1,16),'f'),valid=np.ones((2,1),bool))
        K=np.array([[800,0,447.5],[0,800,447.5],[0,0,1]],dtype='f');cameras={}
        with h5py.File(data.OUT/'cache/synthetic_b.h5','w',locking=False) as f:
            f.attrs['complete']=True;f['frames']=np.array([10,11])
            for sensor,angle,shift in [('rgb3',0,0),('rgb4',.3,.3),('event3',.1,.05),('event4',-.2,-.3)]:
                R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],dtype='f')
                t=np.array([shift,.01,-.05],dtype='f');cameras[sensor]=(R,t)
                event=sensor.startswith('event');c=2 if event else 3;size=224 if event else 896
                f.create_dataset(sensor,shape=(2,c,size,size),dtype='u1',fillvalue=128)
                f[sensor+'_available']=np.ones(2,bool)
                for key,value in dict(K_primary=K,K_auxiliary=K,K_native=K,native_shape=np.array([896,896]),
                    R_common_to_sensor=R,t_common_to_sensor=t).items():f[sensor+'_'+key]=value
        checked=[];original=h5py.Dataset.__getitem__;reads=[]
        def read(dataset,key):
            if dataset.name in ('/rgb3','/rgb4'):reads.append(dataset.name)
            return original(dataset,key)
        h5py.Dataset.__getitem__=read
        try:
            for condition,(primary,aux) in CONDITIONS.items():
                reads.clear();dataset=data.Frames('training',condition=condition);item=dataset[0]
                inputs=data.input_keys(data.collate([item,dataset[1]]))
                assert not set(inputs)&{'joints','centers','rotations','betas','valid'}
                R,t=cameras[primary];assert np.allclose((item['joints']-t)@R,joints[0],atol=2e-6)
                if primary.startswith('event'):assert 'image' not in inputs
                if condition=='event3_event4':assert not reads
                assert (aux is not None)==('auxiliary' in inputs)
                for handle in dataset.handles.values():handle.close()
                reads.clear();source=inference.CachedSensors('synthetic_b',condition);sample=source.sample(10)
                if condition=='event3_event4':assert not reads and set(source.pixel_datasets_read)=={'event3','event4'}
                if aux:
                    Ra,ta=cameras[aux];relative=sample['primary_to_aux_rotation'].numpy()
                    shift=sample['primary_to_aux_translation'].numpy()
                    assert np.allclose(relative,Ra@R.T)
                    assert np.allclose((joints[0]@R.T+t)@relative.T+shift,joints[0]@Ra.T+ta,atol=2e-6)
                source.close();checked.append(condition)
        finally:h5py.Dataset.__getitem__=original
    result=dict(conditions=checked,nonidentity_camera_roundtrip=True,
        cross_sensor_geometry=True,GT_fields_excluded_from_inputs=True,
        event_only_no_RGB_pixel_reads=True,shared_sample_count=True)
    save(OUT/'data_cpu_checks.json',result);print(json.dumps(result))


if __name__=='__main__':main()

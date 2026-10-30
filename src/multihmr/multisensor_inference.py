"""Inference on the common clock; inputs contain sensor observations only."""
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from .multisensor_model import MultiSensorMultiHMR,CONDITIONS
from .multisensor_data import OUT


def load_model(config,checkpoint):
    from utils.pose_evaluation import load_multihmr
    if config.get('weight_stage')=='released_pretrained':
        from .pretrained_inference import released_rgb_model
        return released_rgb_model(config,checkpoint,device='cuda')
    if config.get('architecture') in ('epipolar_repair_v1','epipolar_repair_norm_v2'):
        from .multisensor_repair import build_repair_model,load_verified_initialization
        model=build_repair_model(load_multihmr(),config).cuda().eval()
        load_verified_initialization(model,config)
    else:model=MultiSensorMultiHMR(load_multihmr(),config['condition']).cuda().eval()
    saved=torch.load(checkpoint,map_location='cpu')
    assert saved['successful_steps']>0 and saved['test_used_for_selection'] is False
    assert saved['config']['condition']==config['condition']
    assert saved['config']['split_sha256']==config['split_sha256']
    assert saved['config']['seed']==config['seed']
    assert saved['config'].get('architecture')==config.get('architecture')
    assert saved['metrics_apply_to_checkpoint'] and saved['metrics_step']==saved['step']
    assert not model.load_state_dict(saved['model_delta'],strict=False).unexpected_keys
    return model


class CachedSensors:
    def __init__(self,session,condition,sensor_layout=None,encoding='count33'):
        self.encoding=encoding
        cache=OUT/'encoding_cache'/encoding if (OUT/'encoding_cache'/encoding).exists() else OUT/'cache'
        self.f=h5py.File(cache/(session+'.h5'),'r',locking=False)
        assert self.f.attrs['complete']
        assert self.f.attrs['event_encoding']==encoding
        self.primary,self.aux=sensor_layout or CONDITIONS[condition]
        self.frames=self.f['frames'][:]
        self.R=self.f[self.primary+'_R_common_to_sensor'][:].astype(np.float32)
        self.t=self.f[self.primary+'_t_common_to_sensor'][:].astype(np.float32)
        self.metadata={k:self.f[k][:].astype(np.float32) for sensor in (self.primary,self.aux) if sensor
                       for k in (sensor+'_K_primary',sensor+'_K_auxiliary',sensor+'_R_common_to_sensor',sensor+'_t_common_to_sensor')}
        self.pixel_datasets_read=set()

    def sensor(self,sensor,index,auxiliary=False):
        self.pixel_datasets_read.add(sensor)
        value=torch.from_numpy(self.f[sensor][index].astype(np.float32)/255)
        if sensor.startswith('rgb'):
            if auxiliary:value=F.interpolate(value[None],size=(224,224),mode='area')[0]
            value=(value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        return value

    def sample(self,frame):
        i=int(np.searchsorted(self.frames,frame));assert self.frames[i]==frame
        if not self.f[self.primary+'_available'][i]:return None
        result=dict(K=torch.from_numpy(self.metadata[self.primary+'_K_primary']))
        key='event' if self.primary.startswith('event') else 'image'
        result[key]=self.sensor(self.primary,i)
        if self.aux:
            Ra=self.metadata[self.aux+'_R_common_to_sensor'];ta=self.metadata[self.aux+'_t_common_to_sensor']
            rotation=Ra@self.R.T
            result.update(auxiliary=self.sensor(self.aux,i,True),
                auxiliary_K=torch.from_numpy(self.metadata[self.aux+'_K_auxiliary']),
                primary_to_aux_rotation=torch.from_numpy(rotation),
                primary_to_aux_translation=torch.from_numpy(ta-rotation@self.t),
                auxiliary_available=torch.tensor(float(self.f[self.aux+'_available'][i])))
        if self.primary.startswith('event') and self.aux and self.aux.startswith('event'):
            assert not any(name.startswith('rgb') for name in self.pixel_datasets_read)
        return result

    def close(self):self.f.close()


class OnlineSensors(CachedSensors):
    """Decode original RGB and causally encode sparse events on demand."""
    def __init__(self,session,condition,sensor_layout=None,encoding='count33'):
        super().__init__(session,condition,sensor_layout=sensor_layout,encoding=encoding)
        from .multisensor_data import ROOT
        from utils.event_encoding import EventFrames
        self.source=ROOT/'dataset'/session
        with np.load(OUT/'source_maps'/(session+'.npz')) as z:
            self.maps={k:z[k] for k in z.files if k.endswith('rgb_positions')}
        self.events={};self.captures={};self.rgb_previous={};self.encoding_seconds=0.;self.decode_seconds=0.
        for sensor in (self.primary,self.aux):
            if sensor and sensor.startswith('event'):
                self.events[sensor]=EventFrames.load(self.source/('device'+sensor[-1])/'event.npz',size=224)

    def sensor(self,sensor,index,auxiliary=False):
        self.pixel_datasets_read.add(sensor)
        if sensor.startswith('event'):
            start=time.perf_counter()
            values,metadata=self.events[sensor].encode(int(self.f['timestamp_ns'][index]),encodings=[self.encoding])
            assert metadata['latest_event_timestamp_ns']<=self.f['timestamp_ns'][index] or not metadata['event_available']
            value=np.rint(values[self.encoding]*255).astype(np.uint8).astype(np.float32)/255
            self.encoding_seconds+=time.perf_counter()-start
            return torch.from_numpy(value)
        import cv2
        from PIL import Image
        from utils.event_encoding import fit_geometry
        if not self.f[sensor+'_available'][index]:
            size=224 if auxiliary else 896
            value=torch.zeros(3,size,size)
            return (value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        started=time.perf_counter();frame=int(self.f['frames'][index])
        position=int(self.maps['device'+sensor[-1]+'_rgb_positions'][frame])
        if sensor in self.rgb_previous and self.rgb_previous[sensor][0]==position:
            value=self.rgb_previous[sensor][1]
        else:
            if sensor not in self.captures:
                cap=cv2.VideoCapture(str(self.source/('device'+sensor[-1])/'rgb.mp4'));assert cap.isOpened()
                self.captures[sensor]=cap
            cap=self.captures[sensor]
            cursor=round(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if position<cursor or position-cursor>100:
                assert cap.set(cv2.CAP_PROP_POS_FRAMES,position)
            else:
                for _ in range(position-cursor):assert cap.grab()
            ok,bgr=cap.read();assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES))==position+1
            native=self.f[sensor+'_native_shape'][:]
            (nh,nw,top,left),_=fit_geometry(native,896)
            image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((nw,nh),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(896,896));canvas.paste(image,(left,top))
            value=torch.from_numpy(np.asarray(canvas).transpose(2,0,1).copy()).float()/255
            self.rgb_previous[sensor]=(position,value)
        if auxiliary:value=F.interpolate(value[None],size=(224,224),mode='area')[0]
        result=(value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        self.decode_seconds+=time.perf_counter()-started
        return result

    def close(self):
        for cap in self.captures.values():cap.release()
        self.events.clear();self.rgb_previous.clear()
        super().close()


def infer(model,collector,source,batch_size=16):
    collector.matching_sensor_geometry=(source.R,source.t,
        source.f[source.primary+'_K_native'][:].astype(np.float32),
        tuple(map(int,source.f[source.primary+'_native_shape'][:])))
    collector.task['matching_protocol']='primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1'
    pending=[];ids=[];original=model.base.detection
    def detection(*args,**kwargs):
        output=original(*args,**kwargs);ids[:]=output[2][0].detach().cpu().tolist();return output
    model.base.detection=detection
    def flush():
        selected=[(i,item) for i,item in enumerate(pending) if item[1] is not None]
        grouped=[[] for _ in pending]
        if selected:
            batch={key:torch.stack([item[key] for _,(_,item) in selected]).cuda() for key in selected[0][1][1]}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                people=model(**batch,is_training=False,det_thresh=.3,nms_kernel_size=3)
            assert len(people)==len(ids)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                for batch_id,person in zip(ids,people):
                    joints=(collector.reg@person['v3d'].float()).cpu().numpy()
                    joints=(joints-source.t)@source.R
                    assert np.isfinite(joints).all()
                    grouped[selected[batch_id][0]].append(joints)
        for (pos,_),predictions in zip(pending,grouped):
            collector.current_pos=pos
            collector.collect(np.stack(predictions) if predictions else np.empty((0,55,3),np.float32),collector.K)
        pending.clear()
    try:
        for pos in range(collector.task['video_frames']):
            pending.append((pos,source.sample(collector.task['start_frame']+pos)))
            if len(pending)==batch_size:flush()
        if pending:flush()
    finally:model.base.detection=original

"""Actual RGB/event detections on frozen RGB-clock Test segments, in RGB space."""
from pathlib import Path
import json
import time

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T

from .event_data import OUT, ROOT, event_to_rgb
from .event_model import EventMultiHMR
from utils.event_encoding import ENCODINGS, EventFrames
from utils.video_frames import frames


def load_model(config, checkpoint=None):
    from utils.pose_evaluation import load_multihmr
    channels = ENCODINGS[config['encoding']]['channels'] if config['mode'] != 'rgb' else 0
    model = EventMultiHMR(load_multihmr(), config['mode'], channels).cuda().eval()
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['test_used_for_selection'] is False and saved['successful_steps'] > 0
        assert saved['config']['mode'] == config['mode']
        assert saved['config'].get('encoding') == config.get('encoding')
        assert not model.load_state_dict(saved['model_delta'], strict=False).unexpected_keys
    return model


class CachedEvents:
    def __init__(self, session, encoding):
        self.handle = h5py.File(OUT / 'event_cache' / (session + '.h5'), 'r')
        assert self.handle.attrs['complete']
        self.encoding = encoding
        self.frames = self.handle['frames'][:]
        self.cameras = {name: self.handle[name][:].astype(np.float32) for name in
                        ('K_event', 'R_rgb_to_event', 't_rgb_to_event', 'event_pixel_transform')}
        self.K = self.cameras['event_pixel_transform'] @ self.cameras['K_event']

    def frame(self, frame):
        index = int(np.searchsorted(self.frames, frame))
        assert index < len(self.frames) and self.frames[index] == frame
        return self.handle[self.encoding][index].astype(np.float32) / 255, bool(self.handle['event_available'][index])

    def close(self):
        self.handle.close()


class OnlineEvents:
    def __init__(self, session, encoding, size=224):
        self.source = EventFrames.load(ROOT / 'dataset' / session / 'device0/event.npz', size=size)
        self.encoding = encoding
        self.cameras = {name: value.astype(np.float32) for name, value in self.source.cameras.items()}
        self.K = (self.source.pixel_transform @ self.cameras['K_event']).astype(np.float32)
        self.encoding_seconds = 0.; self.frames_encoded = 0

    def frame(self, frame):
        started = time.perf_counter()
        encoded, metadata = self.source.encode(int(self.source.rgb_times[frame]), encodings=[self.encoding])
        # Match the frozen cache's quantization; online/cached models receive the same values.
        value = np.rint(255 * encoded[self.encoding]).astype(np.uint8).astype(np.float32) / 255
        self.encoding_seconds += time.perf_counter() - started; self.frames_encoded += 1
        return value, metadata['event_available']

    def close(self):
        pass


def infer(model, collector, encoding=None, batch_size=16, event_source=None):
    task = collector.task
    if model.mode != 'rgb':
        if event_source is None:
            event_source = CachedEvents(task['parent_session'], encoding)
            owned_source = True
        else:
            owned_source = False
    else:
        owned_source = False
    norm = T.Compose([T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    pending = []; ids = []; original = model.base.detection

    def detection(*args, **kwargs):
        output = original(*args, **kwargs); ids[:] = output[2][0].detach().cpu().tolist()
        return output

    model.base.detection = detection

    def flush():
        selected = [index for index, item in enumerate(pending)
                    if model.mode != 'event' or item['available']]
        grouped = [[] for _ in pending]
        if selected:
            inputs = {'K': torch.from_numpy(np.stack([pending[index]['K'] for index in selected])).cuda()}
            if model.mode != 'event':
                inputs['image'] = torch.stack([pending[index]['image'] for index in selected]).cuda()
            if model.mode != 'rgb':
                inputs['event'] = torch.from_numpy(np.stack([pending[index]['event'] for index in selected])).cuda()
                cameras = event_source.cameras
                inputs['event_K'] = torch.from_numpy(event_source.K.copy()).cuda()[None].expand(len(selected), -1, -1)
                inputs['event_rotation'] = torch.from_numpy(cameras['R_rgb_to_event'].copy()).cuda()[None].expand(len(selected), -1, -1)
                inputs['event_translation'] = torch.from_numpy(cameras['t_rgb_to_event'].copy()).cuda()[None].expand(len(selected), -1)
                inputs['event_available'] = torch.tensor([pending[index]['available'] for index in selected], device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                people = model(**inputs, is_training=False, det_thresh=.3, nms_kernel_size=3)
            assert len(people) == len(ids)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                for batch_id, person in zip(ids, people):
                    joints = (collector.reg @ person['v3d'].float()).cpu().numpy()
                    if model.mode == 'event':
                        joints = event_to_rgb(joints, event_source.cameras['R_rgb_to_event'],
                                              event_source.cameras['t_rgb_to_event'])
                    assert np.isfinite(joints).all()
                    grouped[selected[batch_id]].append(joints)
        for item, predictions in zip(pending, grouped):
            collector.current_pos = item['pos']
            collector.collect(np.stack(predictions) if predictions else np.empty((0, 55, 3), np.float32), collector.K)
        pending.clear()

    # This branch never opens/decodes/stages RGB, including missing-event frames.
    iterator = ((pos, None) for pos in range(task['video_frames'])) if model.mode == 'event' else frames(task)
    try:
        for pos, rgb in iterator:
            item = {'pos': pos}
            if model.mode != 'rgb':
                item['event'], item['available'] = event_source.frame(task['start_frame'] + pos)
                size = item['event'].shape[-1]; scale = model.base.img_size / size
                upsample = np.array([[scale, 0, (scale - 1) / 2],
                                     [0, scale, (scale - 1) / 2], [0, 0, 1]], np.float32)
                item['K'] = upsample @ event_source.K
            if model.mode != 'event':
                height, width = rgb.shape[:2]; size = model.base.img_size
                image = ImageOps.contain(Image.fromarray(rgb), (size, size), method=Image.Resampling.LANCZOS)
                nw, nh = image.size; left, top = (size - nw) // 2, (size - nh) // 2
                image = ImageOps.expand(image, border=(left, top, size - nw - left, size - nh - top))
                item['image'] = norm(image)
                K = collector.K.copy(); K[0] *= nw / width; K[1] *= nh / height
                K[0, 2] += left; K[1, 2] += top; item['K'] = K
            pending.append(item)
            if len(pending) == batch_size:
                flush()
        if pending:
            flush()
    finally:
        model.base.detection = original
        if owned_source:
            event_source.close()

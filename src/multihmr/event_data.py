"""Controlled device0 RGB/event frames; event-only never reads an RGB image."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'exp/261024/rgb_event_36h'
RGB_CACHE = ROOT / '.experiment_storage/multihmr_finetune_14h'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.{}.tmp'.format(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)); temporary.replace(path)


def prepare_controlled_indices():
    """Identical Train/holdout frames for every condition, without Test filtering."""
    for role in ('training', 'validation'):
        target = OUT / ('controlled_' + role + '_index.json')
        original = (RGB_CACHE / (role + '_index.json')).read_bytes()
        source_hash = hashlib.sha256(original).hexdigest()
        if target.exists():
            saved = json.loads(target.read_text())
            assert saved['source_sha256'] == source_hash
            continue
        records = json.loads(original)
        sessions = sorted({record['session'] for record in records})
        selected = []; excluded = 0
        for session in sessions:
            rows = [record for record in records if record['session'] == session]
            with np.load(ROOT / 'dataset' / session / 'device0/event.npz', allow_pickle=False) as z:
                times = z['timestamp_ns']; rgb_times = z['rgb__timestamp_ns']
                frames = np.array([record['frame'] for record in rows])
                timestamps = rgb_times[frames]
                positions = np.searchsorted(times, timestamps, side='right') - 1
                available = (positions >= 0) & (timestamps - times[np.maximum(positions, 0)] <= 20_000_000)
            selected.extend(record for record, keep in zip(rows, available) if keep)
            excluded += int((~available).sum())
        selected.sort(key=lambda row: (row['session'], row['frame']))
        assert {row['session'] for row in selected} == set(sessions)
        save(target, dict(records=selected, source_sha256=source_hash, excluded_unavailable_frames=excluded,
                          original_frames=len(records), retained_frames=len(selected),
                          policy='Same causal event-available Train/holdout frames for every trained condition; Test is never filtered'))


def event_to_rgb(points, rotation, translation):
    return (points - translation) @ rotation


class Frames(Dataset):
    def __init__(self, role, mode, encoding=None, augment=False, limit=0, sessions=None):
        assert role in ('training', 'validation') and mode in ('rgb', 'event', 'fusion')
        self.role = role; self.mode = mode; self.encoding = encoding; self.augment = augment
        self.records = json.loads((OUT / ('controlled_' + role + '_index.json')).read_text())['records']
        if sessions is not None:
            allowed = set(sessions)
            self.records = [row for row in self.records if row['session'] in allowed]
        if limit:
            ids = np.linspace(0, len(self.records) - 1, min(limit, len(self.records)), dtype=int)
            self.records = [self.records[int(index)] for index in ids]
        assert self.records
        self.targets = OrderedDict(); self.handles = OrderedDict(); self.metadata = {}
        self.norm = T.Compose([T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.jitter = T.ColorJitter(.2, .2, .15, .03)

    def __len__(self):
        return len(self.records)

    def target(self, session):
        if session not in self.targets:
            with np.load(RGB_CACHE / 'targets' / (session + '.npz')) as z:
                self.targets[session] = {key: z[key] for key in z.files}
            if len(self.targets) > 16:
                self.targets.popitem(last=False)
        self.targets.move_to_end(session)
        return self.targets[session]

    def events(self, session, frame):
        if session not in self.handles:
            handle = h5py.File(OUT / 'event_cache' / (session + '.h5'), 'r')
            assert handle.attrs['complete']
            self.handles[session] = handle
            if session not in self.metadata:
                self.metadata[session] = {key: handle[key][:] for key in
                    ('frames', 'K_event', 'R_rgb_to_event', 't_rgb_to_event', 'event_pixel_transform', 'native_event_shape')}
            if len(self.handles) > 32:
                _, previous = self.handles.popitem(last=False); previous.close()
        self.handles.move_to_end(session)
        handle = self.handles[session]; metadata = self.metadata[session]
        row = int(np.searchsorted(metadata['frames'], frame))
        assert row < len(metadata['frames']) and metadata['frames'][row] == frame
        event = handle[self.encoding][row].astype(np.float32) / 255
        available = bool(handle['event_available'][row])
        return event, metadata, available

    def __getitem__(self, index):
        record = self.records[index]; name = record['session']; row = record['row']
        target = self.target(name)
        fields = {key: target[key][row].copy() for key in ('joints', 'centers', 'rotations', 'betas', 'valid')}
        original_K = target['K'].copy(); image_shape = target['image_shape'].copy()
        result = dict(fields, index=index)
        if self.mode in ('event', 'fusion'):
            event, metadata, available = self.events(name, record['frame'])
            assert available, 'Controlled Train/holdout index must be event-available'
            result.update(event=event, event_available=np.float32(available),
                          event_K=(metadata['event_pixel_transform'] @ metadata['K_event']).astype(np.float32),
                          event_rotation=metadata['R_rgb_to_event'].astype(np.float32),
                          event_translation=metadata['t_rgb_to_event'].astype(np.float32))
        if self.mode == 'event':
            # No Image.open, video decode, RGB detector/crop or RGB feature in this path.
            rotation = result['event_rotation']; translation = result['event_translation']
            result['joints'] = fields['joints'] @ rotation.T + translation
            result['centers'] = fields['centers'] @ rotation.T + translation
            result['rotations'][:, 0] = rotation @ fields['rotations'][:, 0]
            original_K = metadata['K_event'].astype(np.float32)
            image_shape = metadata['native_event_shape'].astype(np.int64)
            size = event.shape[-1]; scale = 896 / size
            upsample = np.array([[scale, 0, (scale - 1) / 2],
                                 [0, scale, (scale - 1) / 2], [0, 0, 1]], np.float32)
            K = upsample @ result['event_K']
        else:
            with Image.open(RGB_CACHE / 'images' / name / ('{:06d}.jpg'.format(record['frame']))) as source:
                image = source.convert('RGB')
            width, height = image.size
            assert (height, width) == tuple(image_shape)
            image = ImageOps.contain(image, (896, 896), method=Image.Resampling.LANCZOS)
            nw, nh = image.size; left, top = (896 - nw) // 2, (896 - nh) // 2
            image = ImageOps.expand(image, border=(left, top, 896 - nw - left, 896 - nh - top))
            if self.augment:
                image = self.jitter(image)
            result['image'] = self.norm(image)
            K = original_K.copy(); K[0] *= nw / width; K[1] *= nh / height
            K[0, 2] += left; K[1, 2] += top
        result.update(K=K.astype(np.float32), original_K=original_K.astype(np.float32),
                      image_shape=image_shape.astype(np.int64))
        return result

    def __getstate__(self):
        value = dict(self.__dict__); value['handles'] = OrderedDict()
        return value


def collate(items):
    result = {}
    for key in ('image', 'event', 'K', 'original_K', 'image_shape', 'event_K', 'event_rotation',
                'event_translation', 'event_available'):
        if key in items[0]:
            result[key] = torch.stack([torch.as_tensor(item[key]) for item in items])
    people = max(len(item['valid']) for item in items)
    for key in ('joints', 'centers', 'rotations', 'betas', 'valid'):
        shape = (len(items), people, *items[0][key].shape[1:])
        array = np.zeros(shape, dtype=items[0][key].dtype)
        for index, item in enumerate(items):
            array[index, :len(item[key])] = np.nan_to_num(item[key])
        result[key] = torch.from_numpy(array)
    result['index'] = [item['index'] for item in items]
    return result


def model_inputs(batch, mode):
    keys = ['K']
    if mode != 'event':
        keys.append('image')
    if mode != 'rgb':
        keys += ['event', 'event_K', 'event_rotation', 'event_translation', 'event_available']
    return {key: batch[key] for key in keys}


def self_check():
    # A known nonidentity camera transform must round-trip all joints in metres.
    angle = .2
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([-.046, -.001, .010])
    points = np.random.default_rng(5).normal(size=(2, 55, 3)); points[..., 2] += 4
    event = points @ rotation.T + translation
    assert np.allclose(event_to_rgb(event, rotation, translation), points)
    batch = {'K': 1, 'event': 2, 'event_K': 3, 'event_rotation': 4, 'event_translation': 5, 'event_available': 6}
    assert 'image' not in model_inputs(batch, 'event')
    return dict(camera_roundtrip=True, event_only_has_no_rgb_input=True)


if __name__ == '__main__':
    print(self_check())

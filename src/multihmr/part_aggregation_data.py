"""Causal same-device RGB/event memory for part-aware temporal aggregation.

Original datasets and prior experiment caches are read-only.  The three bins
contain stored CSR active-pixel observations, not reconstructed raw events.
GT stays on the audited common clock. Event memory ends at the actual latest
RGB3 frame time, so no event newer than the RGB observation is admitted.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'exp/261028/device3_encoding_12h'
DEFAULT_OUTPUT = ROOT / 'exp/261029/part_aware_dynamic_aggregation'
BIN_AGE_BOUNDS_MS = np.array([[0, 33], [33, 100], [100, 200]], np.int64)
MEMORY_VERSION = 'causal_actual_rgb3_nonoverlap_33_100_200_v1'


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def _stat(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def _check_deadline(deadline_unix):
    if deadline_unix is not None and time.time() >= deadline_unix - 120:
        raise TimeoutError('Part-aggregation cache reached its fixed wall deadline')


def encode_memory(events, query_timestamp_ns):
    """Newest-to-oldest disjoint (left-open, right-closed) log-count bins."""
    query = int(query_timestamp_ns)
    values = []; latest = []; earliest = []; observed = []; nframes = []
    for near, far in BIN_AGE_BOUNDS_MS:
        start, end = query - int(far)*1_000_000, query - int(near)*1_000_000
        lo, hi = np.searchsorted(events.times, [start, end], side='right')
        values.append(events._counts(start, end))
        latest.append(int(events.times[hi-1]) if hi > lo else -1)
        earliest.append(int(events.times[lo]) if hi > lo else -1)
        observed.append(bool(int(events.indptr[hi]) > int(events.indptr[lo])))
        nframes.append(int(hi-lo))
        if hi > lo:
            assert start < int(events.times[lo]) <= int(events.times[hi-1]) <= end <= query
    values = np.stack(values)
    assert values.shape == (3, 2, events.size, events.size)
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    return values, dict(bin_available=np.array(observed, np.bool_),
        bin_latest_timestamp_ns=np.array(latest, np.int64),
        bin_earliest_timestamp_ns=np.array(earliest, np.int64),
        bin_stored_frame_count=np.array(nframes, np.int32))


def _prepare_one(args):
    """A session is committed by atomic rename only after source preservation checks."""
    import fcntl
    from utils.event_encoding import EventFrames
    name, source_root, output_root, deadline_unix = args
    source_root, output_root = Path(source_root), Path(output_root)
    cache = output_root / 'memory_cache'; cache.mkdir(parents=True, exist_ok=True)
    destination = cache/(name+'.h5')
    # Local lock avoids an NFS lock-service dependency while retaining atomic H5 commit.
    lock_root = ROOT/'.experiment_storage/part_aggregation_cache_locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(destination.resolve()).encode()).hexdigest()
    with (lock_root/(lock_id+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _check_deadline(deadline_unix)
        original = source_root/'cache'/(name+'.h5')
        with h5py.File(original, 'r', locking=False) as f:
            assert f.attrs['complete'] and f.attrs['event_encoding'] == 'count33'
            frames = f['frames'][:]; clock = f['timestamp_ns'][:]
            splitsha = str(f.attrs['split_sha256'])
            native_event_K = f['event3_K_native'][:]
            if destination.exists():
                with h5py.File(destination, 'r', locking=False) as d:
                    assert d.attrs['complete'] and d.attrs['memory_version'] == MEMORY_VERSION
                    assert d.attrs['split_sha256'] == splitsha
                    assert np.array_equal(d['frames'][:], frames)
                    assert np.array_equal(d['timestamp_ns'][:], clock)
                return json.loads((output_root/'memory_receipts'/(name+'.json')).read_text())
        eventpath = ROOT/'dataset'/name/'device3/event.npz'
        before = _stat(eventpath); started = time.time()
        events = EventFrames.load(eventpath, size=224)
        _check_deadline(deadline_unix)
        with np.load(source_root/'source_maps'/(name+'.npz'), allow_pickle=False) as maps:
            assert np.array_equal(clock, maps['canonical_rgb_timestamp_ns'][frames])
            rgb_positions = maps['device3_rgb_positions'][frames]
            assert (rgb_positions >= 0).all()
            rgb_stamps = maps['device3_rgb_timestamp_ns'][rgb_positions]
            rgb_available = maps['device3_rgb_available'][frames]
            assert (clock[rgb_available] >= rgb_stamps[rgb_available]).all()
            assert (clock[rgb_available]-rgb_stamps[rgb_available] <= 50_000_000).all()
            assert np.allclose(events.cameras['K_event'], maps['device3_event_K'])
            assert np.allclose(events.cameras['K_event'], native_event_K)
            expected_R = maps['device3_event_R'] @ maps['device3_rgb_R'].T
            expected_t = maps['device3_event_t'] - expected_R @ maps['device3_rgb_t']
            assert np.allclose(events.cameras['R_rgb_to_event'], expected_R, atol=1e-7)
            assert np.allclose(events.cameras['t_rgb_to_event'], expected_t, atol=1e-7)
        partial = destination.with_name(destination.name+'.%d.partial' % os.getpid())
        parity = []; selected = set(np.linspace(0, len(clock)-1, min(7, len(clock)), dtype=int))
        try:
            with h5py.File(partial, 'w', locking=False) as d, h5py.File(original, 'r', locking=False) as f:
                d.attrs.update(complete=False, memory_version=MEMORY_VERSION, session=name,
                    source_cache=str(original.resolve()), split_sha256=splitsha,
                    semantics='Stored CSR active-pixel observations; not original raw event multiplicity',
                    query_clock='actual latest device3 RGB corrected timestamp <= common GT clock',
                    time_window='(query-far,query-near], newest-to-oldest', count_cap=256,
                    source_event_path=before['path'], source_event_bytes=before['bytes'],
                    source_event_mtime_ns=before['mtime_ns'], started_unix=started)
                d.create_dataset('frames', data=frames)
                d.create_dataset('timestamp_ns', data=clock)
                d.create_dataset('rgb3_timestamp_ns', data=rgb_stamps)
                d.create_dataset('event_query_timestamp_ns', data=rgb_stamps)
                d.create_dataset('rgb3_query_age_ns', data=clock-rgb_stamps)
                d.create_dataset('rgb3_source_positions', data=rgb_positions)
                d.create_dataset('bin_age_bounds_ms', data=BIN_AGE_BOUNDS_MS)
                ds = d.create_dataset('event_bins', (len(clock),3,2,224,224), dtype=np.uint8,
                    chunks=(1,3,2,224,224), compression='lzf')
                names = {'bin_available':np.bool_, 'bin_latest_timestamp_ns':np.int64,
                    'bin_earliest_timestamp_ns':np.int64, 'bin_stored_frame_count':np.int32}
                for key, dtype in names.items():d.create_dataset(key, (len(clock),3), dtype=dtype)
                available=d.create_dataset('event_available',(len(clock),),dtype=np.bool_)
                for i, query in enumerate(rgb_stamps):
                    _check_deadline(deadline_unix)
                    values, metadata = encode_memory(events, int(query))
                    quant = np.rint(values*255).astype(np.uint8)
                    ds[i] = quant
                    for key in names:d[key][i] = metadata[key]
                    pos=int(np.searchsorted(events.times,int(query),side='right'))-1
                    available[i]=bool(pos>=0 and int(query)-int(events.times[pos])<=20_000_000)
                    assert int(query)<=int(clock[i])
                    # Exact online/cache parity at actual RGB3 time. Old count33
                    # cache ends at the common clock and intentionally differs.
                    if i in selected:
                        online,_=events.encode(int(query),encodings=['count33'])
                        assert np.array_equal(quant[0],np.rint(online['count33']*255).astype(np.uint8)),(name,i,'online count33 mismatch')
                        parity.append(dict(row=i,frame=int(frames[i]),query_ns=int(query),
                            common_gt_ns=int(clock[i]),
                            latest_event_ns=int(metadata['bin_latest_timestamp_ns'][0]),
                            rgb_ns=int(rgb_stamps[i]),rgb_age_ms=float((query-rgb_stamps[i])/1e6)))
                after = _stat(eventpath); assert before == after
                d.attrs.update(complete=True, completed_unix=time.time(), total_seconds=time.time()-started)
                d.flush()
            partial.replace(destination)
        finally:
            del events
            if partial.exists():partial.unlink()
        record = dict(session=name, path=str(destination), bytes=destination.stat().st_size,
            frames=len(clock), seconds=time.time()-started, causal=True,
            source_archive=before, source_stat_preserved=True, count33_parity=parity,
            relative_translation_m=expected_t.tolist(), relative_rotation=expected_R.tolist(),
            rgb_query_age_ms=dict(min=float((clock-rgb_stamps).min()/1e6),
                median=float(np.median(clock-rgb_stamps)/1e6), max=float((clock-rgb_stamps).max()/1e6)),
            pixel_alignment_assumed=False, future_event_used=False)
        _json(output_root/'memory_receipts'/(name+'.json'), record)
        return record


def prepare_event_memory(source_root=DEFAULT_SOURCE, output_root=DEFAULT_OUTPUT, workers=4,
                         deadline_unix=None, sessions=None):
    """Build all frozen Train/holdout/Test frame plans, without resampling Test."""
    source_root, output_root = Path(source_root), Path(output_root)
    manifest = json.loads((source_root/'manifest.json').read_text())
    split = json.loads((ROOT/'dataset/split.json').read_text())
    splitsha = hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()
    assert manifest['split_sha256'] == splitsha
    train, val, test = [manifest[key+'_sessions'] for key in ('training','validation','test')]
    assert set(train+val) == {n for n in split['train'] if n.endswith('_b')}
    assert set(test) == {n for n in split['test'] if n.endswith('_b')}
    assert not set(manifest['train_subjects']) & set(manifest['validation_subjects'])
    assert not (set(manifest['train_subjects']) | set(manifest['validation_subjects'])) & set(manifest['test_subjects'])
    # Two early Train/holdout sessions make the first real-data pilot available quickly.
    order = list(dict.fromkeys(train[:2]+val[:2]+train+val+test))
    if sessions is not None:
        assert set(sessions) <= set(order); order = [n for n in order if n in set(sessions)]
    _json(output_root/'memory_protocol.json', dict(version=MEMORY_VERSION,
        source_root=str(source_root.resolve()), split_sha256=splitsha,
        bin_age_bounds_ms=BIN_AGE_BOUNDS_MS.tolist(), size=224, channels_per_bin=2,
        stored_observations_not_raw_events=True, train_sessions=len(train),
        validation_sessions=len(val), test_sessions=len(test),
        event_query_is_actual_rgb3_timestamp=True,rgb_can_precede_GT_clock=True,
        event_newer_than_RGB_forbidden=True,pixel_alignment_assumed=False,
        fixed_train_holdout_reused=True, test_frame_plan_reused=True,
        deadline_unix=deadline_unix, requested_sessions=order))
    items = [(n,str(source_root),str(output_root),deadline_unix) for n in order]
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(_prepare_one, args):args[0] for args in items}
        for future in as_completed(futures):
            results.append(future.result())
            _json(output_root/'memory_cache_progress.json', dict(completed=len(results),
                total=len(items), last_session=futures[future], updated_unix=time.time(),
                completed_sessions=[r['session'] for r in results]))
            print(json.dumps(dict(event='memory_cache_ready',session=futures[future],completed=len(results))),flush=True)
    result = dict(status='passed', version=MEMORY_VERSION, results=results,
        frames=sum(r['frames'] for r in results), bytes=sum(r['bytes'] for r in results),
        completed_unix=time.time(), full_protocol=len(order)==len(train+val+test))
    _json(output_root/('memory_cache_complete.json' if result['full_protocol'] else 'memory_cache_subset_complete.json'),result)
    return result


class MemorySensors:
    """Test input reader with the same geometry contract as CachedSensors."""
    primary = 'rgb3'
    aux = 'event3'
    def __init__(self, session, source_root=DEFAULT_SOURCE, cache_root=DEFAULT_OUTPUT,
                 use_events=True, condition=None, **kwargs):
        self.session=session; self.source_root=Path(source_root); self.cache_root=Path(cache_root)
        self.use_events=use_events; self.pixel_datasets_read=set()
        self.f=h5py.File(self.source_root/'cache'/(session+'.h5'),'r',locking=False)
        assert self.f.attrs['complete']
        self.frames=self.f['frames'][:]
        self.R=self.f['rgb3_R_common_to_sensor'][:].astype(np.float32)
        self.t=self.f['rgb3_t_common_to_sensor'][:].astype(np.float32)
        self.metadata={k:self.f[k][:].astype(np.float32) for sensor in ('rgb3','event3')
            for k in (sensor+'_K_primary',sensor+'_K_auxiliary',sensor+'_R_common_to_sensor',sensor+'_t_common_to_sensor')}
        self.memory=None
        if use_events:
            self.memory=h5py.File(self.cache_root/'memory_cache'/(session+'.h5'),'r',locking=False)
            assert self.memory.attrs['complete'] and self.memory.attrs['memory_version']==MEMORY_VERSION
            assert np.array_equal(self.memory['frames'][:],self.frames)

    def sample(self, frame):
        i=int(np.searchsorted(self.frames,frame))
        assert i<len(self.frames) and int(self.frames[i])==frame
        if not self.f['rgb3_available'][i]:return None
        self.pixel_datasets_read.add('rgb3')
        rgb=torch.from_numpy(self.f['rgb3'][i].astype(np.float32)/255)
        result=dict(image=(rgb-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None],
            K=torch.from_numpy(self.metadata['rgb3_K_primary']))
        if self.use_events:
            self.pixel_datasets_read.add('event3')
            Ra=self.metadata['event3_R_common_to_sensor'];ta=self.metadata['event3_t_common_to_sensor']
            rotation=Ra@self.R.T
            result.update(event_bins=torch.from_numpy(self.memory['event_bins'][i].astype(np.float32)/255),
                event_bin_available=torch.from_numpy(self.memory['bin_available'][i].astype(np.float32)),
                bin_age_bounds_ms=torch.from_numpy(BIN_AGE_BOUNDS_MS.astype(np.float32)),
                auxiliary_K=torch.from_numpy(self.metadata['event3_K_auxiliary']),
                primary_to_aux_rotation=torch.from_numpy(rotation),
                primary_to_aux_translation=torch.from_numpy(ta-rotation@self.t),
                auxiliary_available=torch.tensor(float(self.memory['event_available'][i])))
        return result

    def close(self):
        self.f.close()
        if self.memory is not None:self.memory.close()


class PartAggregationFrames(Dataset):
    """Frozen subject-disjoint Train/holdout records and calibrated sensor inputs."""
    def __init__(self,role,source_root=DEFAULT_SOURCE,cache_root=DEFAULT_OUTPUT,
                 augment=False,limit=0,sessions=None,index_path=None,use_events=True,**kwargs):
        assert role in ('training','validation')
        self.source_root=Path(source_root);self.cache_root=Path(cache_root)
        self.records=json.loads((Path(index_path) if index_path else self.source_root/(role+'_index.json')).read_text())
        if sessions is not None:self.records=[r for r in self.records if r['session'] in set(sessions)]
        if limit:self.records=[self.records[int(i)] for i in np.linspace(0,len(self.records)-1,min(limit,len(self.records)),dtype=int)]
        assert self.records
        self.use_events=use_events;self.augment=augment
        self.handles=OrderedDict();self.targets=OrderedDict()
        self.primary='rgb3';self.auxiliary='event3' if use_events else None

    def __len__(self):return len(self.records)

    def reader(self,session):
        if session not in self.handles:
            self.handles[session]=MemorySensors(session,self.source_root,self.cache_root,use_events=self.use_events)
            if len(self.handles)>8:
                _,old=self.handles.popitem(last=False);old.close()
        self.handles.move_to_end(session)
        return self.handles[session]

    def handle(self,session):return self.reader(session).f

    def target(self,session):
        if session not in self.targets:
            with np.load(self.source_root/'targets'/(session+'.npz'),allow_pickle=False) as z:
                self.targets[session]={k:z[k] for k in z.files}
            if len(self.targets)>8:self.targets.popitem(last=False)
        self.targets.move_to_end(session);return self.targets[session]

    def __getitem__(self,index):
        row=self.records[index];name=row['session'];i=row['cache_row']
        reader=self.reader(name);target=self.target(name)
        assert int(reader.frames[i])==row['frame']
        result=reader.sample(row['frame']);assert result is not None
        R,t=reader.R,reader.t
        fields={k:target[k][i].copy() for k in ('joints','centers','rotations','betas','valid')}
        fields['joints']=fields['joints']@R.T+t
        fields['centers']=fields['centers']@R.T+t
        fields['rotations'][:,0]=R@fields['rotations'][:,0]
        if self.augment:
            # Shared modest gain preserves the previous controlled augmentation.
            gain=float(np.random.uniform(.9,1.1))
            mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
            result['image']=torch.clamp((result['image']*std+mean)*gain,0,1).sub(mean).div(std)
            if self.use_events:result['event_bins']=torch.clamp(result['event_bins']*gain,0,1)
        result.update(fields,index=index,
            original_K=reader.f['rgb3_K_native'][:].astype(np.float32),
            image_shape=reader.f['rgb3_native_shape'][:].astype(np.int64))
        return result

    def __getstate__(self):
        value=dict(self.__dict__);value['handles']=OrderedDict();return value


def collate(items):
    from .multisensor_data import collate as base_collate
    return base_collate(items)


def self_check():
    """Boundary, polarity, and future-leakage test with integer-nanosecond clocks."""
    from utils.event_encoding import EventFrames
    q=1_700_000_000_000_000_000
    ages=np.array([201_000_000,200_000_000,100_000_000,33_000_000,1,0,-1],np.int64)
    stamps=q-ages;order=np.argsort(stamps);stamps=stamps[order]
    indices=np.arange(len(stamps));sign=np.array([127,-128,127,-128,127,-128,127],np.int8)
    events=EventFrames(stamps,np.arange(len(stamps)+1),indices,sign,(3,3),size=3)
    values,meta=encode_memory(events,q)
    assert meta['bin_stored_frame_count'].tolist()==[2,1,1]
    assert values[0].sum()>0 and values[1].sum()>0 and values[2].sum()>0
    assert not values[:,:,2,0].any(),'Future point admitted'
    assert not values[:,:,0,0].any() and not values[:,:,0,1].any(),'Old/open-boundary point admitted'
    assert np.array_equal(values[0],events._counts(q-33_000_000,q))
    empty,_=encode_memory(events,int(stamps[0])-1)
    assert not empty.any()
    return dict(integer_ns_boundaries=True,left_open_right_closed=True,
        no_future_event=True,disjoint_bins=True,polarity_separated=True,empty_memory=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','check'])
    p.add_argument('--source-root',default=str(DEFAULT_SOURCE))
    p.add_argument('--output-root',default=str(DEFAULT_OUTPUT))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--deadline-unix',type=float)
    p.add_argument('--sessions',nargs='*')
    a=p.parse_args()
    print(json.dumps(self_check()),flush=True)
    if a.action=='prepare':
        prepare_event_memory(a.source_root,a.output_root,a.workers,a.deadline_unix,a.sessions)

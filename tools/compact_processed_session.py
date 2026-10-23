"""Compact the already-copied session; never write to either source dataset."""
import concurrent.futures
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
SESSION = ROOT / 'etri_processed/20260720_125338_040324_200_f'
SOURCE = ROOT / 'dataset' / SESSION.name
WORK = ROOT / 'exp/261023/compact_processed'
GROUPS = ('params', 'bboxes', 'frames')


def video(device, output):
    with h5py.File(SOURCE/'RGB'/device.name/'images.h5', 'r') as f:
        g = f['streams/images']
        offsets = g['offsets'][:]
        timestamps = g['timestamp_ns'][:]
        fps = 1e9 / float(np.median(np.diff(timestamps.astype(np.int64))))
        # Preserve sequence order; original nanosecond timestamps are retained in event.npz.
        process = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-threads','2',
            '-f','image2pipe','-framerate',str(fps),'-vcodec','mjpeg','-i','-',
            '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(output)], stdin=subprocess.PIPE)
        try:
            for start in range(0, len(offsets)-1, 128):
                stop = min(start+128, len(offsets)-1)
                process.stdin.write(g['encoded_data'][int(offsets[start]):int(offsets[stop])].tobytes())
        finally:
            process.stdin.close()
            code = process.wait()
        assert code == 0, f'ffmpeg failed: {device}'
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,nb_frames,width,height,avg_frame_rate',
        '-of','json',str(output)]))['streams'][0]
    assert int(probe['nb_frames']) == len(offsets)-1
    assert probe['codec_name'] == 'h264' and probe['pix_fmt'] == 'yuv420p'
    print(device.name,'RGB validated',probe['nb_frames'],'frames',flush=True)
    return probe


def events(device, output):
    source = SOURCE/'Event'/device.name
    manifest = json.loads((source/'manifest.json').read_text())
    n = int(manifest['frames'])
    nnz = sum(int(s['nnz']) for s in manifest['shards'])
    indptr = np.zeros(n+1, np.int64)
    indices = np.empty(nnz, np.int32)
    data = np.empty(nnz, np.int8)
    metadata = {}
    pixel_shape = None
    offset = 0
    for shard_number,shard in enumerate(manifest['shards']):
        start, stop = int(shard['start']), int(shard['stop'])
        with np.load(source/shard['matrix'], allow_pickle=False) as z:
            ptr, idx, values = z['indptr'], z['indices'], z['data']
            assert ptr.shape == (stop-start+1,)
            assert int(ptr[-1]) == len(idx) == len(values)
            indptr[start:stop+1] = ptr.astype(np.int64)+offset
            indices[offset:offset+len(idx)] = idx
            data[offset:offset+len(values)] = values
            offset += len(idx)
        with np.load(source/shard['frames'], allow_pickle=False) as z:
            for key in z.files:
                value = z[key]
                if key in ('image_shape','background_value'):
                    if key in metadata: assert np.array_equal(metadata[key],value)
                    metadata[key] = value
                else:
                    if key not in metadata:
                        metadata[key] = np.empty((n,*value.shape[1:]), dtype=value.dtype)
                    elif value.dtype.kind in 'US' and value.dtype.itemsize>metadata[key].dtype.itemsize:
                        metadata[key]=metadata[key].astype(np.promote_types(metadata[key].dtype,value.dtype))
                    metadata[key][start:stop] = value
        if shard_number % 20 == 0:
            print(device.name,'merged Event shard',shard_number+1,'/',len(manifest['shards']),flush=True)
    assert offset == nnz and indptr[-1] == nnz
    metadata['event_metadata_json'] = np.str_((source/'metadata.json').read_text())
    # Retain original RGB timing/calibration despite conversion to a constant-rate MP4.
    with h5py.File(SOURCE/'RGB'/device.name/'images.h5','r') as f:
        for key in ('timestamp_ns','raw_timestamp_ns','frame_index','source_position','applied_delta_ns'):
            if key in f['streams/images']:
                metadata['rgb__'+key] = f['streams/images/'+key][:]
        def collect(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.dtype.kind != 'O':
                metadata['rgb_calibration__'+name.replace('/','__')] = obj[()]
        f['calibration'].visititems(collect)
    h,w = map(int,metadata['image_shape'])
    print(device.name,'compressing merged Event',flush=True)
    arrays=dict(format=np.array(b'csr'),shape=np.array([n,h*w]),indptr=indptr,
        indices=indices,data=data,frame_height=np.array(h),frame_width=np.array(w),**metadata)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for key,value in arrays.items():
            with archive.open(key+'.npy','w',force_zip64=True) as member:
                np.lib.format.write_array(member,np.asarray(value),allow_pickle=False)
    print(device.name,'validating merged Event',flush=True)
    with np.load(output,allow_pickle=False) as z:
        assert np.array_equal(z['indptr'],indptr)
        assert np.array_equal(z['indices'],indices)
        assert np.array_equal(z['data'],data)
        for key in metadata: assert np.array_equal(z[key],metadata[key]),key
    print(device.name,'Event validated',n,'frames',flush=True)
    return {'frames':n,'nnz':nnz,'shape':[n,h*w],
        'encoding':'CSR int8 deltas: pixel = background_value + data; -128 -> 0, +127 -> 255',
        'rgb_timestamps':'rgb__timestamp_ns'}


def label(device,output):
    source = SOURCE/'RGBD'/f'{device.name}_mamma.h5'
    with h5py.File(source,'r') as fs, h5py.File(output,'w') as fd:
        for group in GROUPS: fs.copy(group,fd)
        # Keep coordinate units and body-model metadata as attributes, without extra groups.
        for key,value in fs.attrs.items():
            if not any(term in key for term in ('vertex','verts','faces','checkpoint','chunk','mask')):
                fd.attrs[key] = value
        fd.attrs['person_masks_stored']=False
        if 'results/person_valid' in fs and 'person_valid' not in fd['params']:
            fd.create_dataset('params/person_valid',data=fs['results/person_valid'][:],compression='gzip')
    with h5py.File(source,'r') as fs, h5py.File(output,'r') as fd:
        assert set(fd.keys()) == set(GROUPS)
        def verify(name,obj):
            if not isinstance(obj,h5py.Dataset): return
            other = fd[name]
            assert obj.shape == other.shape and obj.dtype == other.dtype
            if obj.shape and obj.shape[0]:
                for i in sorted(set([0,obj.shape[0]//2,obj.shape[0]-1])):
                    left,right=obj[i],other[i]
                    if obj.dtype.kind == 'O':
                        assert all(np.array_equal(a,b) for a,b in zip(np.asarray(left).flat,np.asarray(right).flat)),name
                    else:
                        assert np.array_equal(left,right),name
        for group in GROUPS: fs[group].visititems(lambda n,o: verify(group+'/'+n,o))


def run_device(device,completed):
        stage=WORK/device.name
        stage.mkdir(exist_ok=True)
        if device.name in completed:
            print('Already completed',device.name,flush=True)
            return {'device':device.name,'previously_validated':True}
        print('Starting',device.name,flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            v=pool.submit(video,device,device/'rgb.mp4')
            e=pool.submit(events,device,device/'event.npz')
            label(device,device/'label.h5')
            info={'device':device.name,'video':v.result(),'event':e.result()}
        print('Validated',device.name,flush=True)
        (stage/'complete.json').write_text(json.dumps(info,indent=2)+'\n')
        return info


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    assert SESSION.resolve().is_relative_to((ROOT/'etri_processed').resolve())
    devices = [SESSION/f'device{i}' for i in range(5)]
    results=[]
    ready_path=WORK/'validation.json'
    completed={d['device'] for d in json.loads(ready_path.read_text())} if ready_path.exists() else set()
    completed.update(p.parent.name for p in WORK.glob('device*/complete.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_device,device,completed) for device in devices]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (WORK/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
    # Only remove copied inputs after all five replacements have been validated.
    for device in devices:
        for directory in ('RGB','Event'):
            path=device/directory
            if path.exists():
                assert path.is_dir() and not path.is_symlink()
                shutil.rmtree(path)
        expected={'event.npz','rgb.mp4','label.h5'}
        if device.name=='device0':
            expected.update(p.name for p in device.iterdir() if p.name in ('preview.mp4','event.mp4'))
        assert {p.name for p in device.iterdir()}==expected
    if (SESSION/'labels').exists(): shutil.rmtree(SESSION/'labels')
    assert {p.name for p in SESSION.iterdir()}=={f'device{i}' for i in range(5)}
    print('Completed all devices; removed copied RGB/Event/labels inputs. Source dataset unchanged.',flush=True)


if __name__ == '__main__':
    main()

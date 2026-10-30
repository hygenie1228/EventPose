"""Read-only, CPU-only pilot diagnostics for event human-motion research ideas.

GT-centred support and oracle anchors are diagnostics, never deployed inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import cv2
import h5py
import numpy as np
import torch
import smplx
from smplx.lbs import batch_rodrigues, batch_rigid_transform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/261026/idea_motion_probe'


def prefix(path, key, count):
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, order, dtype = np.lib.format._read_array_header(stream, version)
        assert not order and len(shape) == 1 and count <= shape[0]
        payload = stream.read(int(count) * dtype.itemsize)
        assert len(payload) == count * dtype.itemsize
        return np.frombuffer(payload, dtype=dtype).copy()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def scalar(v):
    return float(v) if np.isfinite(v) else None


def joint_positions(model, pose, betas, transl):
    with torch.no_grad():
        b = torch.as_tensor(betas)
        shaped = model.v_template[None] + torch.einsum('bl,vkl->bvk', b, model.shapedirs)
        rest = torch.einsum('jv,bvk->bjk', model.J_regressor, shaped)
        rotations = batch_rodrigues((torch.as_tensor(pose) + model.pose_mean).reshape(-1, 3)).reshape(-1, 55, 3, 3)
        joints, _ = batch_rigid_transform(rotations, rest, model.parents)
        return joints[:, :22].numpy() + transl[:, None]


def load_gt(model, name, seconds, hz):
    path = ROOT/'dataset'/name/'device0/label.h5'
    with h5py.File(path, 'r', locking=False) as f:
        timestamps = f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
        valid = f['params/person_valid'][:, 0].astype(bool)
        queries = timestamps[0] + np.arange(1., seconds, 1/hz) * 1e9
        selected = np.unique(np.searchsorted(timestamps, queries.astype(np.int64)))
        selected = selected[(selected < len(timestamps)) & valid[np.minimum(selected, len(valid)-1)]]
        pose = f['params/smplx_pose_axis_angle'][selected, 0]
        betas = f['params/smplx_betas'][selected, 0]
        transl = f['params/translation_world_m'][selected, 0]
        times = timestamps[selected]
    joints = joint_positions(model, pose, betas, transl)
    # Independently validate fast FK against the public SMPL-X forward function.
    with torch.no_grad():
        p = torch.from_numpy(pose[:2]); b = torch.from_numpy(betas[:2])
        reference = model(betas=b, global_orient=p[:, :3], body_pose=p[:, 3:66],
            jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
            left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
            transl=torch.from_numpy(transl[:2]), expression=torch.zeros((len(p), 10))).joints[:, :22].numpy()
    discrepancy = float(np.max(np.abs(reference-joints[:2])))
    assert discrepancy < 2e-5, discrepancy
    return times, joints, discrepancy


def event_probe(name, device, times, joints, max_seconds):
    path = ROOT/'dataset'/name/f'device{device}'/'event.npz'
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    with np.load(path, allow_pickle=False) as z:
        et = z['timestamp_ns'].astype(np.int64); ptr = z['indptr']
        h, w = map(int, z['image_shape'])
        K = z['rgb_calibration__event__intrinsic_matrix'].astype(np.float64)
        R = z['rgb_calibration__event__rotation_matrix'].astype(np.float64)
        trans = z['rgb_calibration__event__translation'].astype(np.float64)
        rawdelta = np.unique(z['applied_delta_ns'])
    last = int(np.searchsorted(et, times[-1], side='right'))
    idx = prefix(path, 'indices', int(ptr[last])); data = prefix(path, 'data', int(ptr[last]))
    assert np.all(np.isin(data, [-128, 127]))
    points = joints @ R.T + trans
    uvw = points @ K.T
    uv = uvw[..., :2]/uvw[..., 2:]
    visible = (points[..., 2]>.1) & (uv[..., 0]>=20) & (uv[..., 0]<w-20) & (uv[..., 1]>=20) & (uv[..., 1]<h-20)
    gh, gw = h//10, w//10
    grid = ((idx//w)//10)*gw + ((idx%w)//10)
    widths = [10,33,100,200]
    density = np.full((len(times), len(widths), 22), np.nan)
    displacement = np.full_like(density, np.nan)
    union_pixels = np.zeros((len(times), len(widths)))
    counts = np.zeros((len(times), len(widths)))
    age = np.zeros(len(times))
    local_maps = []
    features=[]
    speeds = np.linalg.norm(np.diff(joints, axis=0),axis=-1)/ (np.diff(times)/1e9)[:,None]
    for i,t in enumerate(times):
        latest = int(np.searchsorted(et,t,side='right'))-1
        age[i] = (int(t)-int(et[latest]))/1e6 if latest>=0 else 1e9
        assert latest < 0 or int(et[latest])<=int(t)
        cx = np.clip((uv[i,:,0]/10).astype(int),2,gw-3)
        cy = np.clip((uv[i,:,1]/10).astype(int),2,gh-3)
        for wi,ms in enumerate(widths):
            lo,hi = np.searchsorted(et,[int(t)-ms*1_000_000,int(t)],side='right')
            a,b=map(int,ptr[[lo,hi]])
            pixels=np.unique(grid[a:b])
            active=np.zeros(gh*gw,np.float32); active[pixels]=1
            active=active.reshape(gh,gw)
            integral=cv2.integral(active)
            support=(integral[cy+3,cx+3]-integral[cy-2,cx+3]-integral[cy+3,cx-2]+integral[cy-2,cx-2])/25
            density[i,wi]=np.where(visible[i],support,np.nan)
            union_pixels[i,wi]=len(np.unique(idx[a:b]));counts[i,wi]=b-a
            prev=np.searchsorted(times,int(t)-ms*1_000_000,side='left')
            if prev<i:
                displacement[i,wi]=np.linalg.norm(uv[i]-uv[prev],axis=-1)
        # Direction-sensitive coarse features from two causal event subwindows.
        row=[]
        for left,right in [(100,50),(50,0)]:
            lo,hi=np.searchsorted(et,[int(t)-left*1_000_000,int(t)-right*1_000_000],side='right')
            a,b=map(int,ptr[[lo,hi]])
            coarse=((idx[a:b]//w)*6//h)*8+((idx[a:b]%w)*8//w)
            for positive in [True,False]:
                mask=(data[a:b]>0)==positive
                hist=np.bincount(coarse[mask],minlength=48)
                row.extend(np.log1p(hist))
        features.append(row)
    stepstats=np.diff(et[:last]); after=(path.stat().st_size,path.stat().st_mtime_ns)
    assert before==after
    arrays=dict(times=times,joints=joints,uv=uv,visible=visible,density=density,displacement=displacement,
        union_pixels=union_pixels,counts=counts,event_age_ms=age,features=np.array(features))
    np.savez_compressed(OUT/f'{name}_device{device}.npz',**arrays)
    return dict(session=name,device=device,samples=len(times),source=str(path),
        source_size=before[0],source_mtime_ns=before[1],prefix_observations=len(idx),
        event_interval_ms_quantiles=[float(v) for v in np.quantile(stepstats/1e6,[.1,.5,.9,.99])],
        applied_delta_ns=rawdelta.tolist(),available_fraction=float(np.mean(age<=20)),
        density_median=[float(np.nanmedian(density[:,i])) for i in range(4)],
        zero_support_fraction=[float(np.nanmean(np.where(np.isfinite(density[:,i]),density[:,i]==0,np.nan))) for i in range(4)],
        observation_count_median=np.median(counts,axis=0).tolist(),
        unique_pixel_median=np.median(union_pixels,axis=0).tolist(),readonly_stat_unchanged=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=21);p.add_argument('--sessions',type=int,default=6)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    started=time.time()
    raw=(ROOT/'dataset/split.json').read_bytes();split=json.loads(raw)
    alltrain=[n for n in split['train'] if n.endswith('_b')]
    chosen=[];people=set()
    for name in alltrain:
        pid=int(name.split('_')[3])
        if pid not in people and (ROOT/'dataset'/name/'device0/label.h5').is_file():
            chosen.append(name);people.add(pid)
        if len(chosen)==args.sessions:break
    assert not set(chosen)&set(split['test'])
    save(OUT/'manifest.json',dict(start_unix=started,split_sha256=hashlib.sha256(raw).hexdigest(),
        split_counts={k:len(v) for k,v in split.items()},sessions=chosen,devices=[0],
        extra_views_first_sessions=[1,2],seconds=args.seconds,hz=10,windows_ms=[10,33,100,200],
        test_used=False,gt_projection='diagnostic only',thread_count=2,
        note='Counts are stored active-pixel observations. GT anchors and GT-centred patches are oracle diagnostics.'))
    (OUT/'split_snapshot.json').write_bytes(raw)
    model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',num_betas=16,use_pca=False)
    results=[]
    for number,name in enumerate(chosen):
        times,joints,fkerr=load_gt(model,name,args.seconds,10)
        for device in ([0,1,2] if number<2 else [0]):
            print('BEGIN',name,device,flush=True)
            r=event_probe(name,device,times,joints,args.seconds);r['fk_max_difference_m']=fkerr;results.append(r)
            save(OUT/'source_audit.json',results)
            print('DONE',name,device,'elapsed',round(time.time()-started,1),flush=True)
    save(OUT/'completion.json',dict(start_unix=started,end_unix=time.time(),elapsed_seconds=time.time()-started,session_views=len(results),success=True))


if __name__=='__main__':main()

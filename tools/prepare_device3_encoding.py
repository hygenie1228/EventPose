"""Freeze verified same-split sensor inputs and produce causal encoding caches."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import write,register_job
GOAL=ROOT/'_agents/research/261028/device3_encoding_goal_12h.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def encode_session(name):
    import h5py
    import numpy as np
    from utils.event_encoding import EventFrames,ENCODINGS
    g=read(GOAL);out=Path(g['output_root']);base=(out/'cache'/(name+'.h5')).resolve()
    begin=time.time();paths={e:out/'encoding_cache'/e/(name+'.h5') for e in ('bins100','surface20')}
    if all(p.exists() for p in paths.values()):
        return read(out/'encoding_receipts'/(name+'.json'))
    assert not any(p.exists() for p in paths.values()),'Partial completed pair requires explicit inspection'
    with h5py.File(base,'r',locking=False) as f:
        clock=f['timestamp_ns'][:];frames=f['frames'][:];files={}
        try:
            for e,p in paths.items():
                p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial.h5')
                assert not q.exists(),'Interrupted encoding requires explicit inspection'
                d=h5py.File(q,'w',locking=False);files[e]=d
                d.attrs.update(dict(f.attrs));d.attrs.update(complete=False,event_encoding=e,source_cache=str(base))
                for k in f:
                    if k not in ('event3','event4'):d[k]=h5py.ExternalLink(str(base),'/'+k)
                for sensor in ('event3','event4'):
                    c=ENCODINGS[e]['channels'];d.create_dataset(sensor,shape=(len(clock),c,224,224),
                        dtype=np.uint8,chunks=(1,c,224,224),compression='lzf')
            parity=[];source_provenance=[]
            for sensor in ('event3','event4'):
                source=ROOT/'dataset'/name/('device'+sensor[-1])/'event.npz'
                events=EventFrames.load(source,size=224)
                selected=set(np.linspace(0,len(clock)-1,min(7,len(clock)),dtype=int).tolist())
                for i,t in enumerate(clock):
                    assert time.time()<g['deadline_unix']-300
                    enc,meta=events.encode(int(t),encodings=['bins100','surface20']+(['count33'] if i in selected else []))
                    assert meta['latest_event_timestamp_ns']<=int(t) or not meta['event_available']
                    assert meta['event_available']==bool(f[sensor+'_available'][i])
                    assert meta['latest_event_timestamp_ns']==int(f[sensor+'_latest_timestamp_ns'][i])
                    if i in selected:
                        assert np.array_equal(np.rint(enc['count33']*255).astype(np.uint8),f[sensor][i])
                        parity.append(dict(sensor=sensor,row=i,frame=int(frames[i]),timestamp_ns=int(t)))
                    for e in paths:
                        v=enc[e];assert v.shape==(ENCODINGS[e]['channels'],224,224) and np.isfinite(v).all()
                        assert v.min()>=0 and v.max()<=1
                        files[e][sensor][i]=np.rint(v*255).astype(np.uint8)
                source_provenance.append(dict(sensor=sensor,path=str(source.resolve()),sha256=sha(source),
                    observations=len(events.indices),stored_frames=len(events.times)))
                del events
            for e,d in files.items():d.attrs.update(complete=True,total_seconds=time.time()-begin,completed_unix=time.time());d.flush()
        finally:
            for d in files.values():d.close()
    for p in paths.values():p.with_suffix('.partial.h5').replace(p)
    receipt=dict(session=name,frames=len(clock),seconds=time.time()-begin,causal=True,count33_parity=parity,
        source_archives=source_provenance,encodings={e:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for e,p in paths.items()})
    write(out/'encoding_receipts'/(name+'.json'),receipt);return receipt

def main():
    import h5py
    import numpy as np
    import torch
    from utils.event_encoding import self_check,ENCODINGS
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    g=read(GOAL);register_job(g,os.getpid(),'encoding_source_audit_and_cache');out=Path(g['output_root']);prior=Path(g['prior_experiment'])
    raw=(ROOT/'dataset/split.json').read_bytes();split=json.loads(raw);splitsha=hashlib.sha256(raw).hexdigest()
    manifest=read(prior/'manifest.json');assert splitsha==manifest['split_sha256']
    ids={r:{int(p) for n in split[r] for p in n.split('_')[3:] if p.isdigit()} for r in split}
    assert not ids['train']&ids['test']
    for role in ('training','validation','test'):assert all(n.endswith('_b') for n in manifest[role+'_sessions'])
    assert set(manifest['training_sessions']+manifest['validation_sessions'])=={n for n in split['train'] if n.endswith('_b')}
    assert set(manifest['test_sessions'])=={n for n in split['test'] if n.endswith('_b')}
    assert not set(manifest['train_subjects'])&set(manifest['validation_subjects'])
    assert not (set(manifest['train_subjects'])|set(manifest['validation_subjects']))&set(manifest['test_subjects'])
    old=read(prior/'source_reuse_audit.json');expected={r['session']:r for r in old['cache_records']}
    sessions=manifest['training_sessions']+manifest['validation_sessions']+manifest['test_sessions'];records=[]
    for n in sessions:
        p=prior/'cache'/(n+'.h5');digest=sha(p);assert digest==expected[n]['cache_sha256']
        with h5py.File(p,'r',locking=False) as f, np.load(prior/'source_maps'/(n+'.npz')) as z:
            assert f.attrs['complete'] and f.attrs['split_sha256']==splitsha
            frame=f['frames'][:];clock=f['timestamp_ns'][:]
            assert np.array_equal(clock,z['canonical_rgb_timestamp_ns'][frame])
            mapping=hashlib.sha256(frame.tobytes()+clock.tobytes()).hexdigest();assert mapping==expected[n]['mapping_sha256']
            for sensor in ('event3','event4'):
                available=f[sensor+'_available'][:];assert np.all(f[sensor+'_latest_timestamp_ns'][:][available]<=clock[available])
        records.append(dict(session=n,sha256=digest,frame_mapping_sha256=mapping,bytes=p.stat().st_size))
    for folder in ('cache','source_maps','targets','raw'):
        p=out/folder
        if not p.exists():p.symlink_to((prior/folder).resolve(),target_is_directory=True)
    for filename in ('training_index.json','validation_index.json','joint_regressor.npy','split_snapshot.json'):
        (out/filename).write_bytes((prior/filename).read_bytes())
    (out/'test').mkdir(exist_ok=True);write(out/'test/tasks.json',read(prior/'test/tasks.json'))
    for e in g['encodings']:(out/'encoding_cache'/e).mkdir(parents=True,exist_ok=True)
    for n in sessions:(out/'encoding_cache/count33'/(n+'.h5')).symlink_to((prior/'cache'/(n+'.h5')).resolve())
    base=ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt';base_sha=sha(base)
    assert base_sha==read(prior/'original_pretrained_manifest.json')['sha256']
    write(out/'original_pretrained_manifest.json',dict(path=str(base),sha256=base_sha,bytes=base.stat().st_size))
    norm=dict(status='passed',event_encoder_gains={'event3':4.,'event4':4.},split_sha256=splitsha,
        rule='Fixed gain4 from previous Train-only protocol; identical for every encoding; no Test-fitted statistics')
    write(out/'event_normalization_Train.json',norm)
    configs=[]
    for condition,encoding in [('rgb3','count33')]+[(c,e) for e in g['encodings'] for c in ('rgb3_event4','event3_event4','rgb3_event3')]:
        oldcfg=read(prior/'configs'/('rgb3_short90.json' if condition=='rgb3' else condition+'.json'))
        name='rgb3' if condition=='rgb3' else condition+'_'+encoding
        cfg=dict(oldcfg,experiment='device3_encoding_12h',name=name,condition=condition,encoding=encoding,seed=261028,
            output_root=str(out),deadline_unix=g['deadline_unix'],hours=1.5 if condition=='rgb3' else 3,
            initialization_kind='released_original',initialization_checkpoint=str(base),initialization_sha256=base_sha,
            initialization='Original released Multi-HMR; zero prior ETRI updates',initialization_condition=condition,
            event_encoder_gains=norm['event_encoder_gains'],event_normalization_sha256=sha(out/'event_normalization_Train.json'))
        cfg.pop('trial_started_unix',None);cfg.pop('initialization_manifest',None)
        write(out/'configs'/(name+'.json'),cfg);configs.append(cfg)
    pre=dict(configs[0],name='pre_rgb3',hours=0,weight_stage='released_pretrained')
    write(out/'configs/pre_rgb3.json',pre)
    dest=out/'checkpoints/pre_rgb3/best.pt';dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=pre,step=0,successful_steps=0,model_delta={},test_used_for_selection=False,
        released_checkpoint=str(base),released_checkpoint_sha256=base_sha),dest)
    manifest.update(experiment='device3_encoding_12h',created_unix=time.time(),event_encodings={e:ENCODINGS[e] for e in g['encodings']},
        previous_result_relabelled=False,old_weights_reused=False,source_experiment=str(prior),event_encoding='per-config three encodings')
    write(out/'manifest.json',manifest);write(out/'experiment_manifest.json',dict(configs=[pre]+configs,selection='Train holdout only'))
    write(out/'source_reuse_audit.json',dict(status='passed',split_sha256=splitsha,split_realpath=str((ROOT/'dataset/split.json').resolve()),
        train_ids=sorted(ids['train']),test_ids=sorted(ids['test']),overlap=[],cache_records=records,encoding_self_check=self_check(),
        old_finetuned_weights_reused=False,old_Test_predictions_reused=False,training_frames=len(read(out/'training_index.json')),
        validation_frames=len(read(out/'validation_index.json')),checked_unix=time.time()))
    print('SOURCE_AUDIT_PASSED',flush=True)
    # Independent archives, bounded CPU/memory; process-group ownership inherited.
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(encode_session,n):n for n in manifest['training_sessions']+manifest['validation_sessions']+manifest['test_sessions']}
        for future in as_completed(futures):
            results.append(future.result());write(out/'encoding_cache_progress.json',dict(completed=len(results),total=len(sessions),
                last_session=futures[future],updated_unix=time.time()));print('ENCODED '+futures[future],flush=True)
    write(out/'encoding_cache_complete.json',dict(status='passed',results=results,completed_unix=time.time(),
        bytes=sum(v['bytes'] for r in results for v in r['encodings'].values())))

if __name__=='__main__':main()

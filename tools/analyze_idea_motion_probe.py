"""Train-subject-held-out linear motion probes and native-support summaries."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
from idea_event_motion_probe import ROOT, OUT, save, scalar


def ridge(x,y,tx):
    mu=x.mean(0);std=np.maximum(x.std(0),.01)
    a=(x-mu)/std;b=(tx-mu)/std;ym=y.mean(0)
    weight=np.linalg.solve(a.T@a+100*np.eye(a.shape[1]),a.T@(y-ym))
    return b@weight+ym


def errors(prediction,target):
    d=prediction-target
    absolute=np.linalg.norm(d,axis=-1).mean()*1000
    body=np.linalg.norm(d-d[:,:1],axis=-1).mean()*1000
    depth=np.abs(d[:,0,2]).mean()*1000
    return dict(absolute_mpjpe_mm=float(absolute),root_relative_mpjpe_mm=float(body),rpe_z_mm=float(depth))


def motion():
    manifest=json.loads((OUT/'manifest.json').read_text());sessions=manifest['sessions']
    folds=[];data={}
    rng=np.random.default_rng(261026)
    for session in sessions:
        with np.load(OUT/f'{session}_device0.npz') as z:
            j=z['joints'];t=z['times'];ef=z['features'];counts=z['counts'];uv=z['uv'];density=z['density']
        velocity=(j[1:]-j[:-1])/(np.diff(t)/1e9)[:,None,None]
        relative=j-j[:,:1]
        hist=np.concatenate([relative[1:-1].reshape(len(j)-2,-1),velocity[:-1].reshape(len(j)-2,-1)],axis=1)
        e=ef[2:];y=(j[2:]-j[1:-1]).reshape(len(j)-2,-1)
        data[session]=dict(hist=hist,event=e,shuffled=np.roll(e,len(e)//2,axis=0),y=y,
            anchor=j[1:-1],target=j[2:],constant=velocity[:-1]*(np.diff(t)[1:]/1e9)[:,None,None],
            correlation=scalar(spearmanr(np.log1p(counts[1:,1]),np.linalg.norm(velocity,axis=-1).mean(1)).statistic),
            displacement200_p90=scalar(np.nanquantile(np.linalg.norm(uv[2:]-uv[:-2],axis=-1),.9)))
    aggregate={key:[] for key in ['hold','constant_velocity','history_ridge','history_event_ridge','history_shuffled_event_ridge']}
    for held in sessions:
        train=[data[s] for s in sessions if s!=held];v=data[held]
        y=np.concatenate([d['y'] for d in train])
        predictions={'hold':v['anchor'],'constant_velocity':v['anchor']+v['constant']}
        for method,key in [('history_ridge',None),('history_event_ridge','event'),('history_shuffled_event_ridge','shuffled')]:
            def inputs(d):return d['hist'] if key is None else np.concatenate([d['hist'],d[key]],axis=1)
            x=np.concatenate([inputs(d) for d in train]);delta=ridge(x,y,inputs(v)).reshape(-1,22,3)
            predictions[method]=v['anchor']+delta
        result={m:errors(p,v['target']) for m,p in predictions.items()}
        for m,p in predictions.items():aggregate[m].append((p,v['target']))
        folds.append(dict(held_out_session=held,subject_id=int(held.split('_')[3]),frames=len(v['target']),metrics=result,
            event_count_speed_spearman=v['correlation'],projected_joint_displacement200_p90_px=v['displacement200_p90']))
    summary={m:errors(np.concatenate([v[0] for v in pairs]),np.concatenate([v[1] for v in pairs])) for m,pairs in aggregate.items()}
    obj=dict(protocol='Train-only leave-one-person-out; oracle GT at previous two 10Hz steps; ridge alpha=100 fixed; no model/alpha selection on held-out or Test',
        limitations=['One-step teacher-forced diagnostic, not deployable tracking accuracy.','Six short beginning segments, no independent action/environment generalization.','Shuffled event control uses a half-clip circular shift; features remain causal only before this diagnostic permutation.'],
        sessions=len(sessions),samples=sum(f['frames'] for f in folds),folds=folds,pooled=summary)
    save(OUT/'motion_linear_probe.json',obj)


def native():
    out=ROOT/'exp/261026/idea_motion_native_probe'
    audit=json.loads((out/'source_audit.json').read_text())
    results=[]
    for a in audit:
        name=a['session'];device=a['device']
        with np.load(out/f'{name}_device{device}.npz') as z:
            j=z['joints'];t=z['times'];support=z['support'];background=z['background_support'];uv=z['uv'];old=z['old_fraction']
        speed=np.linalg.norm(np.diff(j,axis=0),axis=-1)/(np.diff(t)/1e9)[:,None]
        visible=np.isfinite(support[1:,1]);q=np.nanquantile(speed[visible],[.25,.75])
        low=visible&(speed<=q[0]);high=visible&(speed>=q[1])
        row=dict(session=name,device=device,frames=len(t),subject_id=int(name.split('_')[3]),
            support_median=np.nanmedian(support,axis=(0,2)).tolist(),
            low_support_fraction=[float(np.nanmean(np.where(np.isfinite(support[:,wi]),support[:,wi]<.01,np.nan))) for wi in range(4)],
            speed_quartiles_m_s=q.tolist(),support33_slow_median=float(np.nanmedian(support[1:,1][low])),
            support33_fast_median=float(np.nanmedian(support[1:,1][high])),
            speed_support_spearman=scalar(spearmanr(speed[visible],support[1:,1][visible]).statistic),
            background_median=np.nanmedian(background,axis=0).tolist(),
            old_only_pixel_fraction_median=np.median(old,axis=0).tolist(),
            displacement200_p90_px=float(np.nanquantile(np.linalg.norm(uv[2:]-uv[:-2],axis=-1),.9)))
        results.append(row)
    complement=[]
    for name in sorted({a['session'] for a in audit}):
        relevant=[a for a in audit if a['session']==name]
        if len(relevant)<2:continue
        supports=[]
        for a in sorted(relevant,key=lambda a:a['device']):
            with np.load(out/f"{name}_device{a['device']}.npz") as z:supports.append(z['support'][:,1])
        stack=np.stack(supports);common=np.isfinite(stack).all(0)
        low=stack<.01
        complement.append(dict(session=name,common_joint_frames=int(common.sum()),
            per_view_low_fraction=[float(np.mean(l[common])) for l in low],
            all_views_low_fraction=float(np.mean(np.all(low,axis=0)[common])),
            device0_low_other_has_support_fraction=float(np.mean((low[0]&np.any(~low[1:],axis=0))[common])),
            interpretation='GT-centred support union only; does not establish visibility, correct correspondences or 3D accuracy.'))
    save(out/'support_summary.json',dict(patch='41x41 native pixels around projected GT body22 joint; activity=unique nonzero pixels/1681',
        low_threshold=.01,window_ms=[10,33,100,200],rows=results,cross_view=complement,
        limitations=['GT-centred patches only for diagnosis.','Joint centres need not coincide with event-generating edges.','Activity includes noise, background and occluding people.','Windows may mix current and stale body locations.']))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['motion','native']);a=p.parse_args()
    globals()[a.mode]()

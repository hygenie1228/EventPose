"""Describe measured part/time choices against independent causal GT motion.

No GT motion enters model inference. Null has no observation age. Temporal-bin
centres provide a conditional proxy, not the mean timestamp of raw events.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261029/part_aware_dynamic_aggregation'

def motion_horizon_summary(prediction,horizon,target,valid,weights):
    prediction=np.asarray(prediction,np.float64);horizon=np.asarray(horizon,np.float64)
    target=np.asarray(target,np.float64);valid=np.asarray(valid,bool)
    assert prediction.shape==horizon.shape==target.shape==valid.shape
    assert np.isfinite(horizon).all() and (horizon>=33-1e-3).all() and (horizon<=200+1e-3).all()
    mask=valid&np.isfinite(prediction)&np.isfinite(target)
    mass=weights.sum(-1);q=np.divide(weights,mass[...,None],out=np.zeros_like(weights),where=mass[...,None]>0)
    between=[]
    for query in range(len(q)):
        active=mass[query]>=.05
        if active.sum()>=2:between.append(float(q[query,active].std(0).mean()))
    parts=[]
    for j in range(target.shape[1]):
        v=mask[:,j];h=horizon[v,j]
        parts.append(dict(valid_frames=int(v.sum()),
            motion_path_MAE_px=float(np.abs(prediction[v,j]-target[v,j]).mean()) if v.any() else None,
            horizon_query_std_ms=float(h.std()) if len(h) else None,
            horizon_quantiles_ms=np.quantile(h,[.05,.5,.95]).tolist() if len(h) else None,
            horizon_at_33ms_fraction=float(np.isclose(h,33,atol=1e-3).mean()) if len(h) else None,
            horizon_at_200ms_fraction=float(np.isclose(h,200,atol=1e-3).mean()) if len(h) else None))
    return dict(parts=parts,mean_between_part_conditional_weight_std=float(np.mean(between)) if between else None,
        interpretation='Query variation is descriptive, not proof of pose gain. Horizon is a deterministic function of predicted path; their correlation is not independent evidence. Static/global controls and event interventions remain required.')

def analyze(name='part_dynamic'):
    from scipy.stats import spearmanr
    p=OUT/'diagnostics';p.mkdir(exist_ok=True)
    weights=p/(name+'_attention.npz');motion=p/'holdout_part_motion_targets.npz'
    if not weights.exists() or not motion.exists() or not (p/(name+'.json')).exists():
        return dict(status='pending',attention_ready=weights.exists(),motion_ready=motion.exists())
    with np.load(weights,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    with np.load(motion,allow_pickle=False) as f:g={k:f[k] for k in f.files}
    lookup={(str(s),int(t)):i for i,(s,t) in enumerate(zip(g['sessions'],g['frames']))}
    ids=np.array([lookup[(str(s),int(t))] for s,t in zip(a['sessions'],a['frames'])])
    speed=g['event3_part_path_px'][ids].astype(np.float64)
    age=a['conditional_bin_midpoint_age_ms'].astype(np.float64)
    mass=a['event_probability'].astype(np.float64)
    assert age.shape==mass.shape==speed.shape
    valid=np.isfinite(age)&np.isfinite(speed)&(mass>=.05)
    subjects=np.array([str(s).split('_')[3] for s in a['sessions']])
    def rho(x,y):
        mask=np.isfinite(x)&np.isfinite(y)
        if mask.sum()<3 or np.ptp(x[mask])<1e-8 or np.ptp(y[mask])<1e-8:return None
        r=float(spearmanr(x[mask],y[mask]).statistic)
        return r if np.isfinite(r) else None
    clusters={}
    for subject in sorted(set(subjects)):
        sel=subjects==subject;vv=valid[sel]
        # Descriptive within-subject relation across paired frame/part cells.
        clusters[subject]=dict(frames=int(sel.sum()),valid_part_cells=int(vv.sum()),
            descriptive_spearman_motion_age=rho(speed[sel][vv],age[sel][vv]))
    parts=[]
    for j,part in enumerate(a['part_names']):
        v=valid[:,j]
        parts.append(dict(part=str(part),frames=len(age),age_defined_frames=int(v.sum()),
            mean_event_probability=float(mass[:,j].mean()),mean_event3_path_px=float(speed[:,j].mean()),
            mean_conditional_bin_midpoint_age_ms=float(age[v,j].mean()) if v.any() else None,
            descriptive_spearman_motion_age=rho(speed[v,j],age[v,j])))
    rr=[v['descriptive_spearman_motion_age'] for v in clusters.values() if v['descriptive_spearman_motion_age'] is not None]
    result=dict(status='complete',name=name,split='Train holdout only; not Test',frames=len(age),
        subjects=len(set(subjects)),sessions=len(set(a['sessions'])),part_names=a['part_names'].tolist(),
        independent_units='subjects/sessions; frame-part cells are repeated measurements, not independent people',
        motion='Mean per-joint native event3 image-plane path length over causal past 200 ms; 30Hz GT approximation',
        age='Conditional temporal-bin-centre proxy in ms, not actual mean event timestamp; null has no age',
        age_threshold='event selection probability >= 0.05, declared before main training',
        valid_age_fraction=float(valid.mean()),parts=parts,subject_descriptions=clusters,
        median_within_subject_descriptive_spearman=float(np.median(rr)) if rr else None,
        inferential_p_value=None,Test_used=False)
    horizon=a.get('predicted_horizon_ms',np.empty(0));prediction=a.get('predicted_motion_path_px',np.empty(0))
    has_horizon=horizon.shape==prediction.shape==speed.shape
    if has_horizon:
        # Original holdout diagnostic is immutable and predates training masks.
        with np.load(p/'validation_part_motion_training_targets.npz',allow_pickle=False) as f:
            lookup_mask={(str(s),int(t)):i for i,(s,t) in enumerate(zip(f['sessions'],f['frames']))}
            mask_ids=np.array([lookup_mask[(str(s),int(t))] for s,t in zip(a['sessions'],a['frames'])])
            motion_valid=f['valid'][mask_ids,None]&f['part_valid'][mask_ids]
        result['motion_horizon_diagnostics']=motion_horizon_summary(prediction,horizon,speed,motion_valid,a['time_weights'])
    (p/(name+'_motion_relation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Torso','L. arm','R. arm','L. leg','R. leg','Head']
    fig,axes=plt.subplots(4 if has_horizon else 3,1,figsize=(9,9.2 if has_horizon else 7.4),sharex=True,layout='constrained')
    xx=np.arange(6)
    axes[0].bar(xx,[x['mean_event3_path_px'] for x in parts],color='#e9a65c',width=.62)
    axes[0].set_ylabel('GT path (px / 200 ms)')
    display_name='V1 part-aware' if name=='part_dynamic' else name.replace('_',' ')
    axes[0].set_title(display_name+': motion and event-use diagnostic',fontweight='bold')
    axes[1].bar(xx,[x['mean_event_probability'] for x in parts],color='#4b9bc5',width=.62)
    axes[1].set_ylim(0,1);axes[1].set_ylabel('P(event evidence)')
    for i,item in enumerate(parts):
        probability=item['mean_event_probability']
        axes[1].text(i,min(probability+.04,.96),f'{probability:.2g}',
                     ha='center',va='bottom',fontsize=8,color='#42677d')
    means=[np.nan if x['mean_conditional_bin_midpoint_age_ms'] is None else x['mean_conditional_bin_midpoint_age_ms'] for x in parts]
    axes[2].bar(xx,means,color='#8c7fbd',width=.62)
    for i,value in enumerate(means):
        if not np.isfinite(value):
            axes[2].axvspan(i-.34,i+.34,color='#eeeeee',zorder=0)
            axes[2].text(i,90,'Undefined:\nP(event) < 0.05',ha='center',va='center',fontsize=8,color='#666666')
    axes[2].set_ylim(0,200);axes[2].set_ylabel('Conditional bin age (ms)')
    if has_horizon:
        hparts=result['motion_horizon_diagnostics']['parts']
        med=np.array([item['horizon_quantiles_ms'][1] for item in hparts])
        lo=np.array([item['horizon_quantiles_ms'][0] for item in hparts])
        hi=np.array([item['horizon_quantiles_ms'][2] for item in hparts])
        axes[3].bar(xx,med,color='#6aab91',width=.62)
        axes[3].errorbar(xx,med,yerr=np.stack([med-lo,hi-med]),fmt='none',ecolor='#285d48',capsize=4)
        axes[3].set_ylim(0,210);axes[3].set_ylabel('Predicted horizon (ms)')
        axes[3].set_title('Median and 5th–95th query percentiles; descriptive spread, not a confidence interval',fontsize=9)
    axes[-1].set_xticks(xx,labels)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel(f"Means over {len(age)} held-out frames, {len(set(subjects))} subjects. Null has no age; bin centres are proxies.",fontsize=9)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams['svg.fonttype']='none'
    for ext in ('png','svg','pdf'):fig.savefig(folder/(name+'_motion_relation.'+ext),dpi=180)
    plt.close(fig)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='part_dynamic');a=p.parse_args()
    print(json.dumps(analyze(a.name),ensure_ascii=False,indent=2))

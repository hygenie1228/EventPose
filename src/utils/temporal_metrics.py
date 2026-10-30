"""Temporal errors on consecutive, jointly matched GT person identities."""
import numpy as np

def temporal_errors(prediction,target,common,positions,fps,timestamps_ns=None):
    """Return sums/counts: pelvis velocity (m/s), body22 root acceleration (m/s²).

    Use corrected integer timestamps when supplied, otherwise nominal video fps.
    Never bridge missing frames, unmatched people, or segment boundaries.
    No temporal alignment/filter is applied.
    """
    assert fps>0
    p=np.asarray(prediction,dtype=np.float64)[:,:,:22]
    t=np.asarray(target,dtype=np.float64)[:,:,:22]
    consecutive=np.diff(positions)==1
    pair=common[:-1]&common[1:]&consecutive[:,None]
    triple=pair[:-1]&pair[1:]
    dt=(np.diff(np.asarray(timestamps_ns,dtype=np.int64))/1e9 if timestamps_ns is not None
        else np.full(max(0,len(p)-1),1/fps))
    assert np.all(dt>0)
    velocity=(np.diff(p[:,:,0],axis=0)-np.diff(t[:,:,0],axis=0))/dt[:,None,None]
    pr=p-p[:,:,:1];tr=t-t[:,:,:1]
    relative_velocity=(np.diff(pr,axis=0)-np.diff(tr,axis=0))/dt[:,None,None,None]
    acceleration=np.diff(relative_velocity,axis=0)/((dt[:-1]+dt[1:])/2)[:,None,None,None]
    v=np.linalg.norm(velocity[pair],axis=-1)
    a=np.linalg.norm(acceleration[triple],axis=-1).mean(-1)
    assert np.isfinite(v).all() and np.isfinite(a).all()
    return float(v.sum()),len(v),float(a.sum()),len(a)

"""Temporal errors on consecutive, jointly matched GT person identities."""
import numpy as np

def temporal_errors(prediction,target,common,positions,fps):
    """Return sums/counts: pelvis velocity (m/s), body22 root acceleration (m/s²).

    Frame time is source frame number / nominal video fps. Never bridge missing
    frames, unmatched people, or segment boundaries. No temporal alignment/filter.
    """
    assert fps>0
    p=np.asarray(prediction,dtype=np.float64)[:,:,:22]
    t=np.asarray(target,dtype=np.float64)[:,:,:22]
    consecutive=np.diff(positions)==1
    pair=common[:-1]&common[1:]&consecutive[:,None]
    triple=pair[:-1]&pair[1:]
    velocity=(np.diff(p[:,:,0],axis=0)-np.diff(t[:,:,0],axis=0))*fps
    pr=p-p[:,:,:1];tr=t-t[:,:,:1]
    acceleration=(np.diff(pr,n=2,axis=0)-np.diff(tr,n=2,axis=0))*fps**2
    v=np.linalg.norm(velocity[pair],axis=-1)
    a=np.linalg.norm(acceleration[triple],axis=-1).mean(-1)
    assert np.isfinite(v).all() and np.isfinite(a).all()
    return float(v.sum()),len(v),float(a.sum()),len(a)

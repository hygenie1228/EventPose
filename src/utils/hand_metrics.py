"""The body's four 10-second alignment metrics, with hand wrist as root."""
import numpy as np


def similarity(pred,target):
    x=np.asarray(pred,np.float64);y=np.asarray(target,np.float64)
    a=x.mean(0);b=y.mean(0);xc=x-a;yc=y-b
    u,s,vh=np.linalg.svd(xc.T@yc);d=np.ones(3);d[-1]=np.linalg.det(u@vh)
    rotation=(u*d)@vh;scale=(s*d).sum()/max((xc*xc).sum(),1e-12)
    assert scale>0 and np.linalg.det(rotation)>.99999
    return scale,rotation,b-scale*a@rotation


def four_errors(pred,target,first_pred,first_target):
    """Errors in metres. Frame0 is supplied explicitly; never choose a later frame."""
    shift=first_target[0]-first_pred[0]
    first=np.linalg.norm(pred+shift-target,axis=-1).mean()
    scale,rotation,translation=similarity(first_pred,first_target)
    first_pa=np.linalg.norm(scale*pred@rotation+translation-target,axis=-1).mean()
    relative=(pred-pred[:1])-(target-target[:1])
    frame=np.linalg.norm(relative,axis=-1).mean()
    scale,rotation,translation=similarity(pred,target)
    frame_pa=np.linalg.norm(scale*pred@rotation+translation-target,axis=-1).mean()
    return np.array([first,first_pa,frame,frame_pa],np.float64)


def score_session(pred,target,frames,people,right,intervals):
    assert pred.shape==target.shape and pred.shape[1:]==(21,3)
    valid=np.isfinite(pred).all((1,2))&np.isfinite(target).all((1,2));values=[];keys=[];excluded=[]
    for segment,(start,end) in enumerate(intervals):
        ids=np.flatnonzero((frames>=start)&(frames<end))
        for person,side in sorted(set(zip(people[ids].tolist(),right[ids].tolist()))):
            selected=ids[(people[ids]==person)&(right[ids]==side)]
            anchor=selected[frames[selected]==start]
            if len(anchor)!=1 or not valid[anchor[0]]:
                excluded.append(dict(segment=segment,person=int(person),right=int(side),reason='original_frame0_missing',samples=len(selected)))
                continue
            a=anchor[0]
            for i in selected[valid[selected]]:
                values.append(four_errors(pred[i],target[i],pred[a],target[a]));keys.append([segment,int(frames[i]),int(person),int(side)])
    return np.asarray(values,np.float64).reshape(-1,4),np.asarray(keys,np.int64).reshape(-1,4),excluded


def self_check():
    rng=np.random.default_rng(261028);x=rng.normal(size=(21,3))*.05
    identity=four_errors(x,x,x,x);assert abs(identity).max()<1e-12
    q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,-1]*=np.linalg.det(q)
    y=1.4*x@q+[.2,.3,2]
    errors=four_errors(x,y,x,y);assert errors[1]<1e-12 and errors[3]<1e-12
    drift=four_errors(x+[.1,0,0],x,x,x);assert np.isclose(drift[0],.1) and drift[2]<1e-12
    reflect=x.copy();reflect[:,0]*=-1;assert four_errors(reflect,x,reflect,x)[3]>.001
    v,_,excluded=score_session(x[None],x[None],np.array([1]),np.array([0]),np.array([0]),[(0,2)])
    assert len(v)==0 and len(excluded)==1
    return dict(identity=True,proper_sim3=True,reflection_rejected=True,fixed_frame0_translation=True,
        missing_original_frame0_rejected=True,wrist_root=True,root_position_error_computed=False)

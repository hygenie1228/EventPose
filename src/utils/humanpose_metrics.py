from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Explicit camera-space MPJPE metrics and image-space person matching."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def joint_errors(pred, target, root=0):
    """Return m-valued errors. PA uses proper Sim(3), never a reflection.

    Root-aligned PA: center each skeleton at its pelvis, then conventional
    least-squares Procrustes with rotation, positive isotropic scale and translation.
    """
    pred=np.asarray(pred,np.float64);target=np.asarray(target,np.float64)
    assert pred.shape==target.shape and pred.ndim==2 and pred.shape[1]==3
    absolute=np.linalg.norm(pred-target,axis=-1).mean()
    x=pred-pred[root];y=target-target[root]
    relative=np.linalg.norm(x-y,axis=-1).mean()
    xc=x-x.mean(0);yc=y-y.mean(0)
    u,s,vt=np.linalg.svd(xc.T@yc)
    d=np.ones(3);d[-1]=np.linalg.det(u@vt)
    rotation=(u*d)@vt
    denominator=np.square(xc).sum()
    assert denominator>1e-12,'Degenerate predicted skeleton'
    scale=np.sum(s*d)/denominator
    assert scale>0
    aligned=scale*xc@rotation+y.mean(0)
    pa=np.linalg.norm(aligned-y,axis=-1).mean()
    return np.array([absolute,relative,pa],np.float64)


def project(joints,K):
    homogeneous=joints@np.asarray(K).T
    return homogeneous[...,:2]/np.maximum(homogeneous[...,2:],1e-8)


def match_people(pred,target,valid,pred_K,gt_K,image_shape):
    """Hungarian assignment on in-frame GT body-joint reprojection, not 3D error.

    Return GT-slot -> prediction index. Gate mean image-space error at 25% of
    image diagonal; count unmatched/missing instances separately during reporting.
    pred_K and gt_K must project to the same original sensor pixel plane.
    """
    h,w=image_shape
    gt_ids=np.flatnonzero(valid)
    if not len(pred) or not len(gt_ids):return {}
    good=np.isfinite(pred).all((1,2))&(pred[:,0,2]>.05)
    pred_ids=np.flatnonzero(good)
    if not len(pred_ids):return {}
    uv_pred=project(pred[pred_ids,:22],pred_K)
    uv_gt=project(target[gt_ids,:22],gt_K)
    cost=np.full((len(gt_ids),len(pred_ids)),1e6)
    for i,g in enumerate(gt_ids):
        uv=uv_gt[i];mask=(target[g,:22,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        if mask.any():cost[i]=np.linalg.norm(uv_pred[:,mask]-uv[None,mask],axis=-1).mean(1)/np.hypot(w,h)
    rows,cols=linear_sum_assignment(cost)
    return {int(gt_ids[i]):int(pred_ids[j]) for i,j in zip(rows,cols) if cost[i,j]<=.25}


def match_people_in_sensor(pred_common,target_common,valid,rotation,translation,K,image_shape):
    """Match in the observing sensor; keep returned GT identities in common space.

    Reprojection into an unrelated reference camera can turn a native-plane
    detection with depth error into a miss. Root/pose errors are scored later
    from the unchanged common-camera predictions, never corrected by matching.
    """
    pred=pred_common@rotation.T+translation
    target=target_common@rotation.T+translation
    return match_people(pred,target,valid,K,K,image_shape)


def self_check():
    rng=np.random.default_rng(261023);x=rng.normal(size=(22,3));x[0]=0
    assert np.max(joint_errors(x,x))<1e-12
    t=np.array([1,2,3.]);e=joint_errors(x+t,x)
    assert np.isclose(e[0],np.linalg.norm(t)) and e[1]<1e-12 and e[2]<1e-12
    q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,-1]*=np.linalg.det(q)
    e=joint_errors(1.7*x@q+t,x)
    assert e[0]>.1 and e[1]>.1 and e[2]<1e-10
    reflected=x.copy();reflected[:,0]*=-1
    assert joint_errors(reflected,x)[2]>.1,'Reflection must not be accepted as a proper rotation'
    body=.1*x;body[:,2]+=4
    people=np.stack([body+[-1,0,0],body+[1,0,0]])
    K=np.array([[500,0,640],[0,500,360],[0,0,1.]])
    assert match_people(people[::-1],people,np.array([True,True]),K,K,(720,1280))=={0:1,1:0}
    assert match_people(np.empty((0,22,3)),people,np.array([True,True]),K,K,(720,1280))=={}
    return {'identity':True,'translation':True,'rotation_scale_translation':True,
            'reflection_rejected':True,'swapped_person_matching':True,'empty_detection':True}


if __name__=='__main__':print(self_check())

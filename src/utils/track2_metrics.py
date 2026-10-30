"""Identical public V2D human formulas and ChallengeHOR's local MHR sampling."""
import hashlib
import numpy as np
from utils import v2d_official_metric as official


class SmplxCDHProxy:
    """Reuse the exact reference sampling on reconstructed SMPL-X GT/predictions."""

    def __init__(self,body,device,roles):
        import torch
        from multihmr.track2_data import mapping
        self.body=body;self.device=device
        selected=torch.tensor(roles['alignment']+roles['body'],device=device,dtype=torch.long)
        ids,weights=mapping('smplx2mhr',body.faces,device)
        self.ids=ids[selected];self.weights=weights[selected]

    def points(self,vertices):
        from multihmr.track2_data import transfer
        return transfer(vertices,self.ids,self.weights)

    def targets(self,batch,return_torch=False):
        import torch
        import roma
        n=len(batch['rotations']);device=self.device
        rotation=roma.rotmat_to_rotvec(batch['rotations'][:,0].to(device))
        zeros=lambda width:torch.zeros(n,width,device=device)
        vertex=self.body(global_orient=rotation[:,0],body_pose=rotation[:,1:22].flatten(1),
            left_hand_pose=rotation[:,22:37].flatten(1),right_hand_pose=rotation[:,37:52].flatten(1),
            jaw_pose=rotation[:,52],betas=batch['betas'][:,0].to(device),
            transl=batch['smplx_translation'][:,0].to(device),expression=zeros(10),
            leye_pose=zeros(3),reye_pose=zeros(3)).vertices
        error=((self.body.J_regressor@vertex).cpu()-batch['joints'][:,0].cpu()).abs().max().item()
        assert error<2e-4,('SMPL-X supervision reconstruction mismatch',error)
        points=self.points(vertex)
        return (points if return_torch else points.cpu().numpy()),error

    @staticmethod
    def summarize(records):
        sequences={}
        for record in records:sequences.setdefault(record['session'],[]).append(record)
        values={}
        for name,items in sequences.items():
            items.sort(key=lambda item:item['frame'])
            assert items[0]['frame']==0,('Original frame0 required for CD-H',name)
            pred=np.asarray([x['pred'] for x in items],np.float64)
            target=np.asarray([x['target'] for x in items],np.float64)
            transform=official.fit_similarity_transform(pred[0,:27],target[0,:27])
            aligned=official.apply_similarity_transform(pred[:,27:],transform)
            cd=float(official.symmetric_chamfer_distance(aligned,target[:,27:]).mean()*100)
            values[name]=dict(cd_h_smplx_proxy_cm=cd,matched_frames=len(items),
                first_frame_sim3_scale=float(transform[0]),
                first_frame_sim3_rotation=transform[1].tolist(),
                first_frame_sim3_translation_m=transform[2].tolist())
        return dict(cd_h_smplx_proxy_cm=float(np.mean([v['cd_h_smplx_proxy_cm'] for v in values.values()])),
            cd_proxy_per_sequence=values,not_final_mhr_cd_h=True)


def paired_frame0_chamfer(predicted,target,pairs):
    """Differentiable official-style CD-H: [time frames, frame0 anchors].

    Sampling is FPS27 alignment + FPS64 body. Fit one proper Sim(3) to each
    predicted/GT anchor. Differentiate its SVD, scale, rotation and translation:
    the alignment anchor and scored time frame have different objectives, so
    treating the fitted alignment as constant drops valid anchor gradients.
    Coordinates and loss remain metres. The reference symmetric CD sums the
    two directed means; it does not multiply their sum by one half.
    """
    import torch
    assert predicted.shape==target.shape and predicted.shape[0]==2*pairs
    time_points=predicted[:pairs,27:]
    anchors=predicted[pairs:,:27]
    gt_time=target[:pairs,27:]
    gt_anchors=target[pairs:,:27]
    center=anchors.mean(1,keepdim=True)
    gt_center=gt_anchors.mean(1,keepdim=True)
    x=anchors-center;y=gt_anchors-gt_center
    u,s,vh=torch.linalg.svd(x.transpose(1,2)@y)
    d=torch.ones_like(s);d[:,-1]=torch.det(u@vh).sign()
    rotation=(u*d[:,None])@vh
    scale=(s*d).sum(1)/x.square().sum((1,2)).clamp_min(1e-10)
    aligned=scale[:,None,None]*((time_points-center)@rotation)+gt_center
    distances=torch.cdist(aligned,gt_time,p=2,compute_mode='donot_use_mm_for_euclid_dist')
    return (distances.min(2).values.mean(1)+distances.min(1).values.mean(1)).mean()


def metric_arrays(body, vertices, joints):
    n = len(vertices)
    return dict(mhr_vertices=vertices[:, body.roles['alignment']],
        mhr_joints=joints[:, list(official.MHR_TABLE3_BODY_JOINT_INDICES)],
        human_surface_points=vertices[:, body.roles['body']],
        # Human metrics do not read object values. Dummy object arrays satisfy
        # the unchanged scorer contract; object metrics are never reported.
        object_surface_points=np.zeros((n, len(body.roles['body']), 3), np.float64),
        object_translation=np.zeros((n, 3), np.float64))


def score_human(body, prediction, target):
    pv, pj = body(np.concatenate([prediction['pose'],
        np.broadcast_to(prediction['scales'], (len(prediction['pose']), 68))], 1), prediction['shape'])
    tv, tj = body(np.concatenate([target['pose'],
        np.broadcast_to(target['scales'], (len(target['pose']), 68))], 1), target['shape'])
    assert len(pv) == len(tv)
    a, b = metric_arrays(body, pv, pj), metric_arrays(body, tv, tj)
    values = official._episode_metrics(a, b, np.arange(len(pv)), None)
    zero = official._episode_metrics(b, b, np.arange(len(tv)), None)
    assert abs(zero['cd_h_cm']) < 1e-8 and abs(zero['acc_h_cm']) < 1e-8
    return dict(CD_H_cm=values['cd_h_cm'], ACC_H_cm_per_frame2=values['acc_h_cm'],
        frames=len(pv), full_frame_roster=True, gt_self_check_passed=True)


def protocol(body):
    from pathlib import Path
    return dict(official_leaderboard_parity=False,
        source='ChallengeHOR/scripts/run_evaluation.py local protocol and unchanged public official human metric functions',
        alignment='one proper Sim(3) from original frame 0 FPS27 non-hand vertices, fixed for the sequence',
        CD_H='symmetric unsquared nearest-neighbor Euclidean distance on FPS64 non-hand vertices, cm',
        ACC_H='body22 second difference error on consecutive original frames, cm/frame^2, without fps^2 multiplication',
        aggregation='equal sequence weights, identical frozen Test roster',
        roles={k:v.tolist() for k,v in body.roles.items()},
        scorer_sha256=hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest(),
        caveat='Organizer host vertex roles/quantized model asset are undisclosed; same documented local substitute as the referenced evaluation script.')

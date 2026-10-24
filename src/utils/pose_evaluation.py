from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Full available-Test evaluation; all writes stay in the experiment/report roots.

prepare: freeze existing test membership, reconstruct GT meshes/joints read-only.
worker: run one pretrained model, without temporal subsampling, on a session shard.
report: evaluate both methods on identical GT instances and write the final report.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'exp/261023/humanpose_gt_eval'


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def atomic_npz(path,**value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**value);tmp.replace(path)


def configure_torch():
    import torch
    torch.set_num_threads(2)
    torch.hub.set_dir(str(ROOT/'data/cache/torch/hub'))
    os.environ['HF_HOME']=str(ROOT/'data/cache/huggingface')
    return torch


def prepare():
    import cv2
    import h5py
    import smplx
    from humanpose_metrics import self_check
    torch=configure_torch()
    assert h5py.version.hdf5_version_tuple>=(2,0,0),h5py.version.info
    atomic_json(OUT/'metric_self_checks.json',self_check())
    inv=json.loads((OUT/'session_inventory.json').read_text())
    model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',
        gender='neutral',num_betas=16,use_pca=False,flat_hand_mean=False).cuda().eval()
    reg=model.J_regressor
    assert tuple(reg.shape)==(55,10475)
    np.save(OUT/'joint_regressor.npy',reg.cpu().numpy())
    tasks=[];excluded=[]
    for session in inv['available_sessions']:
        source=ROOT/'dataset'/session/'device0';dest=OUT/'gt'/f'{session}.npz'
        try:
            cap=cv2.VideoCapture(str(source/'rgb.mp4'))
            assert cap.isOpened(),'RGB not readable'
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
            width,height=int(cap.get(3)),int(cap.get(4));cap.release()
            with h5py.File(source/'label.h5','r') as f:
                assert bool(f.attrs['complete']),'GT is incomplete'
                assert f.attrs['coordinate_unit']=='metre'
                positions=f['frames/source_rgb_position'][:].astype(np.int64)
                stamps=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
                valid=f['params/person_valid'][:].astype(bool)
                poses=f['params/smplx_pose_axis_angle'][:].astype(np.float32)
                betas=f['params/smplx_betas'][:].astype(np.float32)
                transl=f['params/translation_world_m'][:].astype(np.float32)
            assert len(np.unique(positions))==len(positions),'Duplicate RGB->GT mapping'
            assert np.all(np.diff(positions)>0) and positions.min()>=0 and positions.max()<count
            valid &= np.isfinite(poses).all(-1)&np.isfinite(betas).all(-1)&np.isfinite(transl).all(-1)
            assert poses.shape[-1]==165 and betas.shape[-1]==16
            with np.load(source/'event.npz',allow_pickle=False) as z:
                K=z['rgb_calibration__rgb__intrinsic_matrix'].astype(np.float32)
                R=z['rgb_calibration__rgb__rotation_matrix'].astype(np.float32)
                t=z['rgb_calibration__rgb__translation'].astype(np.float32).reshape(3)
            if not dest.exists():
                target=np.full((*valid.shape,55,3),np.nan,np.float32)
                ids=np.argwhere(valid)
                for start in range(0,len(ids),64):
                    selected=ids[start:start+64];fi,pi=selected.T
                    p=torch.from_numpy(poses[fi,pi]).cuda();b=torch.from_numpy(betas[fi,pi]).cuda()
                    tr=torch.from_numpy(transl[fi,pi]).cuda()
                    with torch.no_grad():
                        v=model(betas=b,global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],
                            leye_pose=p[:,69:72],reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],
                            right_hand_pose=p[:,120:165],transl=tr,
                            expression=torch.zeros((len(p),model.num_expression_coeffs),device=p.device,dtype=p.dtype)).vertices
                        joints=(reg@v).cpu().numpy()
                    target[fi,pi]=joints@R.T+t
                atomic_npz(dest,joints_cam_m=target,valid=valid,source_rgb_positions=positions,
                           timestamps_ns=stamps,K=K,R_w2c=R,t_w2c=t,image_shape=[height,width])
            tasks.append(dict(session=session,device='device0',video=str(source/'rgb.mp4'),
                gt_file=str(dest),gt_frames=len(positions),valid_gt_instances=int(valid.sum()),
                video_frames=count,fps=fps,width=width,height=height))
            atomic_json(OUT/'preparation_progress.json',dict(prepared=len(tasks),total=len(inv['available_sessions']),session=session))
            print('Prepared',len(tasks),session,'GT frames',len(positions),'RGB frames',count,flush=True)
        except Exception as e:
            excluded.append(dict(session=session,reason=repr(e)))
            print('PREPARATION ERROR',session,repr(e),flush=True)
    # A present but unreadable session is a failure to resolve, never silently omitted.
    atomic_json(OUT/'preparation_errors.json',excluded)
    assert not excluded,excluded
    atomic_json(OUT/'tasks.json',tasks)
    atomic_json(OUT/'protocol.json',dict(
        temporal_sampling='none: every RGB frame is passed to the model, every mapped GT frame is scored',
        camera='device0 (canonical camera, one view per session)',
        unit='metres',coordinate_frame='each input RGB camera; no global alignment for MPJPE',
        joint_definition='same neutral SMPL-X 55x10475 mesh-to-joint regressor for GT and predictions',
        headline_joints='SMPL-X body joints 0..21 (22 joints), pelvis index 0',
        supplementary_joints='SMPL-X joints 0..54 (55 joints)',
        gt_model='neutral SMPL-X, 16 betas, non-PCA hands, flat_hand_mean=False, zero expression',
        matching='Hungarian on visible GT body-joint 2D reprojection; mean distance <= 0.25 image diagonal',
        primary_population='same GT person-frames matched by BOTH methods; misses reported separately',
        mpjpe='mean Euclidean joint distance, original camera coordinates, no alignment',
        root_aligned_mpjpe='subtract each skeleton pelvis; no rotation or scale alignment',
        root_aligned_pa_mpjpe='pelvis centering, then proper per-person per-frame Sim(3) Procrustes (rotation, scale, translation; reflection forbidden)',
        human3r='native FP32 recurrent lighter + TTT3R, 512 scene / 896 human features, sequence state maintained for full video',
        multi_hmr='native FP32 SMPL-X 896 model; calibrated intrinsics with resize/pad transform',
        no_finetuning=True,test_sessions=len(tasks),gt_frames=sum(x['gt_frames'] for x in tasks),
        video_frames=sum(x['video_frames'] for x in tasks),valid_gt_instances=sum(x['valid_gt_instances'] for x in tasks)))
    print('Preparation complete',len(tasks),flush=True)


class Collector:
    def __init__(self,task,method):
        import torch
        self.task=task;self.method=method
        with np.load(task['gt_file']) as z:
            self.gt=z['joints_cam_m'];self.valid=z['valid'];self.positions=z['source_rgb_positions']
            self.K=z['K'];self.image_shape=tuple(z['image_shape'])
        if task.get('smoke'):
            keep=self.positions<task['video_frames']
            self.gt=self.gt[keep];self.valid=self.valid[keep];self.positions=self.positions[keep]
        self.row_for_pos={int(pos):i for i,pos in enumerate(self.positions)}
        self.matched=np.full(self.gt.shape,np.nan,np.float32)
        self.indices=np.full(self.valid.shape,-1,np.int16)
        self.n_pred=np.zeros(len(self.gt),np.int16)
        self.seen=np.zeros(len(self.gt),bool)
        self.current_pos=-1;self.started=time.time();self.last_progress=0
        self.reg=torch.from_numpy(np.load(OUT/'joint_regressor.npy')).cuda()
        self.folder=(OUT/'smoke' if task.get('smoke') else OUT)/method/task['session']
        self.folder.mkdir(parents=True,exist_ok=True)

    def collect(self,joints,predK):
        from humanpose_metrics import match_people
        row=self.row_for_pos.get(self.current_pos)
        if row is not None:
            self.seen[row]=True;self.n_pred[row]=len(joints)
            mapping=match_people(joints,self.gt[row],self.valid[row],predK,self.K,self.image_shape)
            for gt_id,pred_id in mapping.items():
                self.matched[row,gt_id]=joints[pred_id];self.indices[row,gt_id]=pred_id
        if self.current_pos-self.last_progress>=250:
            self.progress();self.last_progress=self.current_pos
        if self.current_pos and self.current_pos%2000==0:self.save(partial=True)

    def progress(self,status='running'):
        elapsed=time.time()-self.started
        value=dict(method=self.method,session=self.task['session'],status=status,pid=os.getpid(),
            rgb_position=self.current_pos,video_frames=self.task['video_frames'],gt_rows_seen=int(self.seen.sum()),
            matched_instances=int((self.indices>=0).sum()),elapsed_seconds=elapsed,
            fps=(self.current_pos+1)/max(elapsed,1e-9),updated_unix=time.time())
        atomic_json(self.folder/'progress.json',value)
        print(json.dumps(value),flush=True)

    def save(self,partial=False):
        atomic_npz(self.folder/('partial.npz' if partial else 'result.npz'),
            matched_joints_cam_m=self.matched,matched_pred_indices=self.indices,n_predictions=self.n_pred,
            seen_gt_rows=self.seen,source_rgb_positions=self.positions)
        if not partial:
            assert self.seen.all(),np.flatnonzero(~self.seen)
            assert self.current_pos+1==self.task['video_frames']
            self.progress('complete')
            atomic_json(self.folder/'complete.json',dict(task=self.task,elapsed_seconds=time.time()-self.started,
                                                       metric_input='result.npz',updated_unix=time.time()))
            (self.folder/'partial.npz').unlink(missing_ok=True)


def rgb_frames(task):
    import cv2
    cap=cv2.VideoCapture(task['video']);assert cap.isOpened()
    for i in range(task['video_frames']):
        ok,image=cap.read();assert ok,(task['session'],i,'decode failed')
        yield i,cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if not task.get('smoke'):
        ok,_=cap.read();assert not ok,'Video frame count changed since preparation'
    cap.release()


def load_multihmr(checkpoint_path=None):
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from multihmr.model import Model
    ckpt=torch.load(checkpoint_path or ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt',map_location='cpu')
    args=dict(vars(ckpt['args']));args['type']=args['train_return_type'];args['img_size']=args['img_size'][0]
    model=Model(**args).cuda().eval();load=model.load_state_dict(ckpt['model_state_dict'],strict=False)
    assert all(k.startswith('smpl_layer.') for k in load.missing_keys)
    assert all(k.startswith(('mlp_j2d.','mlp_fov.','mlp_princpt.','mlp_offset_up.')) for k in load.unexpected_keys)
    return model


def infer_multihmr(model,collector):
    import torch
    from PIL import Image,ImageOps
    import torchvision.transforms as T
    norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    size=model.img_size
    for pos,rgb in rgb_frames(collector.task):
        collector.current_pos=pos
        image=Image.fromarray(rgb);w,h=image.size
        small=ImageOps.contain(image,(size,size),method=Image.Resampling.LANCZOS)
        nw,nh=small.size;l=(size-nw)//2;t=(size-nh)//2
        image=ImageOps.expand(small,border=(l,t,size-nw-l,size-nh-t))
        K=collector.K.copy();K[0]*=nw/w;K[1]*=nh/h;K[0,2]+=l;K[1,2]+=t
        with torch.no_grad():
            people=model(norm(image)[None].cuda(),is_training=False,nms_kernel_size=3,det_thresh=.3,K=torch.from_numpy(K)[None].cuda())
            if people:
                verts=torch.stack([p['v3d'] for p in people]);joints=(collector.reg@verts).float().cpu().numpy()
            else:joints=np.empty((0,55,3),np.float32)
        collector.collect(joints,collector.K)


def load_human3r():
    repo=ROOT/'thirdparties/Human3R';os.chdir(repo)
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'))
    from dust3r.model import ARCroco3DStereo
    return ARCroco3DStereo.from_pretrained(str(ROOT/'data/pretrained/human3r/human3r_896L.pth')).cuda().eval()


def infer_human3r(model,collector,amp=False):
    import torch
    import roma
    from PIL import Image
    import dust3r.model as model_module
    from dust3r.utils.image import ImgNorm,_resize_pil_image,pad_image
    from dust3r.utils.geometry import get_camera_parameters
    from dust3r.utils.smpl_layer import SMPL_Layer
    from dust3r.post_process import estimate_focal_knowing_depth
    layers={}
    affine=None
    original_to_cpu=model_module.to_cpu

    def views():
        nonlocal affine
        for pos,rgb in rgb_frames(collector.task):
            collector.current_pos=pos
            original=Image.fromarray(rgb);ow,oh=original.size
            image=_resize_pil_image(original,512);rw,rh=image.size
            cx,cy=rw//2,rh//2;halfw=((2*cx)//16)*8;halfh=((2*cy)//16)*8
            if rw==rh:halfh=int(3*halfw/4)
            left,top=cx-halfw,cy-halfh
            image=image.crop((left,top,cx+halfw,cy+halfh))
            w,h=image.size;img=ImgNorm(image)[None]
            affine=np.array([[rw/ow,0,-left],[0,rh/oh,-top],[0,0,1]],np.float32)
            yield dict(img=img,ray_map=torch.full((1,6,h,w),torch.nan),true_shape=torch.tensor([[h,w]],dtype=torch.int32),
                idx=pos,instance=str(pos),camera_pose=torch.eye(4)[None],img_mask=torch.tensor([True]),
                ray_mask=torch.tensor([False]),update=torch.tensor([True]),reset=torch.tensor([False]),
                img_mhmr=pad_image(img,model.mhmr_img_res),
                K_mhmr=get_camera_parameters(model.mhmr_img_res,device='cpu'))

    @torch.autocast('cuda',enabled=False)
    def consume(value):
        # Upstream accumulates dense CPU outputs for visualization. Consume its
        # actual per-frame SMPL-X outputs online, preserving all recurrent state,
        # and retain only metric joints needed for full-video evaluation.
        if not isinstance(value,dict) or 'pts3d_in_self_view' not in value:
            return original_to_cpu(value)
        shape=value.get('smpl_shape',torch.empty(1,0,10,device='cuda'))[0].float()
        n=len(shape);joints=np.empty((0,55,3),np.float32)
        if n:
            dim=shape.shape[-1]
            if dim not in layers:
                layers[dim]=SMPL_Layer(type='smplx',gender='neutral',num_betas=dim,kid=False,person_center='head').cuda().eval()
            rot=roma.rotmat_to_rotvec(value['smpl_rotmat'][0].float())
            trans=value['smpl_transl'][0].float();expr=value['smpl_expression'][0].float()
            Kdummy=torch.eye(3,device='cuda').expand(n,-1,-1)
            body=layers[dim](rot,shape,trans,None,None,K=Kdummy,expression=expr)
            joints=(collector.reg@body['smpl_v3d']).float().cpu().numpy()
        points=value['pts3d_in_self_view'].float();_,h,w,_=points.shape
        pp=torch.tensor([[w//2,h//2]],device=points.device,dtype=torch.float32)
        focal=float(estimate_focal_knowing_depth(points,pp,focal_mode='weiszfeld')[0])
        Kself=np.array([[focal,0,w//2],[0,focal,h//2],[0,0,1]],np.float32)
        collector.collect(joints,np.linalg.inv(affine)@Kself)
        return {}  # never used in model state updates; only its visualization list

    model_module.to_cpu=consume
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            model.forward_recurrent_lighter(views(),'cuda',ret_state=False,use_ttt3r=True)
    finally:model_module.to_cpu=original_to_cpu


def worker(method,rank,world_size,limit):
    configure_torch()
    task_file=OUT/'tasks.json'
    expansion_file=OUT/'scope_expansion.json'
    if expansion_file.exists():
        expansion=json.loads(expansion_file.read_text())
        if expansion['status'] in ['preparing','ready']:
            # Original supervisor retries must preserve the original shard map.
            task_file=OUT/'tasks_initial42.json'
    tasks=json.loads(task_file.read_text())
    tasks=sorted(tasks,key=lambda x:(-x['video_frames'],x['session']))[rank::world_size]
    if limit:tasks=tasks[:limit]
    pending=[t for t in tasks if not (OUT/method/t['session']/'complete.json').exists()]
    if not pending:return
    model=load_human3r() if method=='human3r' else load_multihmr()
    for task in pending:
        collector=Collector(task,method);collector.progress()
        try:
            if method=='human3r':infer_human3r(model,collector)
            else:infer_multihmr(model,collector)
            collector.save()
        except Exception as e:
            collector.save(partial=True);collector.progress('failed')
            atomic_json(collector.folder/'error.json',dict(error=repr(e),pid=os.getpid()))
            raise


def smoke(method):
    configure_torch()
    tasks=json.loads((OUT/'tasks.json').read_text())
    task=next(t for t in tasks if t['session']=='20260720_161542_272630_205_b').copy()
    task.update(video_frames=20,smoke=True)
    model=load_human3r() if method=='human3r' else load_multihmr()
    collector=Collector(task,method)
    if method=='human3r':infer_human3r(model,collector)
    else:infer_multihmr(model,collector)
    collector.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','smoke'])
    p.add_argument('--method',choices=['human3r','multi_hmr']);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='smoke':smoke(a.method)
    else:worker(a.method,a.rank,a.world_size,a.limit)


if __name__=='__main__':main()

from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Run RGB-only Human3R/Multi-HMR demos without writing to input datasets.

Each model runs in its own process to avoid conflicting upstream module names.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/261023/human3r_multihmr_test'


def prepare():
    import cv2
    import numpy as np
    split = json.loads((ROOT/'dataset/split.json').read_text())
    session = '20260720_161542_272630_205_b'
    assert session in split['test']
    source = ROOT/'dataset'/session/'device0'
    cap = cv2.VideoCapture(str(source/'rgb.mp4'))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids = np.rint(np.arange(0, min(60, total/fps), .2)*fps).astype(int)
    image_dir = OUT/'inputs/frames'
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = {int(v): i for i, v in enumerate(ids)}
    for pos in range(int(ids[-1])+1):
        ok, image = cap.read()
        assert ok, pos
        if pos in selected:
            assert cv2.imwrite(str(image_dir/f'{selected[pos]:06d}.jpg'), image,
                               [cv2.IMWRITE_JPEG_QUALITY, 97])
    cap.release()
    with np.load(source/'event.npz', allow_pickle=False) as z:
        calibration = {k: z['rgb_calibration__rgb__'+v] for k,v in
                       [('K','intrinsic_matrix'),('R_w2c','rotation_matrix'),('t_w2c','translation')]}
    np.savez(OUT/'inputs/calibration.npz', **calibration)
    manifest = dict(session=session, split='test', device='device0', source=str(source/'rgb.mp4'),
                    source_fps=fps, source_frames=total, source_frame_ids=ids.tolist(),
                    source_times_s=(ids/fps).tolist(), output_fps=5, frames=len(ids),
                    duration_s=len(ids)/5, selection='first 60 seconds, nearest frame at 5 Hz',
                    models_use_rgb_only=True, gt_pose_used_for_inference=False)
    (OUT/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


def setup(repo):
    import torch
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'data/cache/torch/hub'))
    os.environ['HF_HOME'] = str(ROOT/'data/cache/huggingface')
    os.chdir(repo)
    sys.path.insert(0,str(repo))
    return torch


def multihmr(limit):
    repo = ROOT/'src'
    torch = setup(repo)
    import numpy as np
    from PIL import Image, ImageOps
    import torchvision.transforms as T
    from multihmr.model import Model
    checkpoint = ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'
    ckpt = torch.load(checkpoint,map_location='cpu')
    print('Checkpoint arguments:',ckpt['args'],flush=True)
    args = vars(ckpt['args']) if not isinstance(ckpt['args'],dict) else ckpt['args']
    args=dict(args)
    args['type']=args['train_return_type']
    if isinstance(args['img_size'],(tuple,list)):args['img_size']=args['img_size'][0]
    model = Model(**args).cuda().eval()
    loaded=model.load_state_dict(ckpt['model_state_dict'],strict=False)
    assert all(k.startswith('smpl_layer.') for k in loaded.missing_keys),loaded
    assert all(k.startswith(('mlp_j2d.','mlp_fov.','mlp_princpt.','mlp_offset_up.'))
               for k in loaded.unexpected_keys),loaded
    (OUT/'multi_hmr/checkpoint_load.json').write_text(json.dumps(dict(
        missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys,
        explanation='Official demo uses strict=False. Body-model buffers come from local SMPL-X; unused auxiliary heads are ignored.'),indent=2))
    print('Loaded all inference network parameters; external SMPL-X buffers and unused auxiliary heads accounted for.',flush=True)
    size = model.img_size
    print('Input size',size,flush=True)
    calibration = np.load(OUT/'inputs/calibration.npz')
    K_orig = calibration['K'].astype(np.float32)
    files = sorted((OUT/'inputs/frames').glob('*.jpg'))
    if limit: files=files[:limit]
    dest=OUT/'multi_hmr/predictions';dest.mkdir(parents=True,exist_ok=True)
    transform=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    np.save(OUT/'multi_hmr/faces.npy',model.smpl_layer[f'neutral_{model.num_betas}'].bm_x.faces)
    started=time.time()
    for i,p in enumerate(files):
        if (dest/f'{i:06d}.npz').exists():continue
        image=Image.open(p).convert('RGB'); w,h=image.size
        # Follow official ImageOps.contain + symmetric pad preprocessing.
        resized=ImageOps.contain(image,(size,size),method=Image.Resampling.LANCZOS)
        nw,nh=resized.size; left=(size-nw)//2; top=(size-nh)//2
        padded=ImageOps.expand(resized,border=(left,top,size-nw-left,size-nh-top))
        K=K_orig.copy();K[0]*=nw/w;K[1]*=nh/h;K[0,2]+=left;K[1,2]+=top
        x=transform(padded)[None].cuda();kt=torch.tensor(K)[None].cuda()
        with torch.no_grad():
            humans=model(x,is_training=False,nms_kernel_size=3,det_thresh=.3,K=kt)
        def array(key,empty):
            return np.stack([a[key].detach().float().cpu().numpy() for a in humans]) if humans else np.empty(empty,np.float32)
        verts=array('v3d',(0,10475,3))
        np.savez_compressed(dest/f'{i:06d}.npz',vertices_cam_m=verts,K=K,
                            source_K=K_orig,source_size=[w,h],padded_size=size,
                            resize=[nw,nh],padding=[left,top],
                            scores=array('scores',(0,)),
                            camera_to_world=np.eye(4,dtype=np.float32))
        if i%10==0:print(f'Multi-HMR {i+1}/{len(files)} people={len(humans)} elapsed={time.time()-started:.1f}s',flush=True)
    print('Multi-HMR finished',flush=True)


def human3r(limit):
    repo=ROOT/'thirdparties/Human3R'
    torch=setup(repo)
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    import roma
    from inference_only import prepare_input
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.smpl_layer import SMPL_Layer
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.inference import inference_recurrent_lighter
    checkpoint=ROOT/'data/pretrained/human3r/human3r_896L.pth'
    model=ARCroco3DStereo.from_pretrained(str(checkpoint)).cuda().eval()
    files=sorted((OUT/'inputs/frames').glob('*.jpg'))
    if limit:files=files[:limit]
    views=prepare_input(list(map(str,files)),[True]*len(files),size=512,
                        img_res=model.mhmr_img_res,reset_interval=10000000)
    # Official demo supplies pseudo intrinsics here; retain that native behavior.
    started=time.time()
    result,_=inference_recurrent_lighter(views,model,'cuda',use_ttt3r=True)
    print('Human3R inference seconds',time.time()-started,flush=True)
    dest=OUT/'human3r/predictions';dest.mkdir(parents=True,exist_ok=True)
    layer=None
    for i,(pred,view) in enumerate(zip(result['pred'],result['views'])):
        points=pred['pts3d_in_self_view'].float().cpu()
        _,h,w,_=points.shape
        pp=torch.tensor([[w//2,h//2]],dtype=torch.float32)
        focal=estimate_focal_knowing_depth(points,pp,focal_mode='weiszfeld').cpu()
        K=torch.eye(3);K[0,0]=focal[0];K[1,1]=focal[0];K[:2,2]=pp[0]
        c2w=pose_encoding_to_camera(pred['camera_pose'].float()).cpu()[0]
        shape=pred.get('smpl_shape',torch.empty(1,0,10))[0].float().cpu()
        if layer is None:
            layer=SMPL_Layer(type='smplx',gender='neutral',num_betas=shape.shape[-1],kid=False,person_center='head')
            np.save(OUT/'human3r/faces.npy',layer.bm_x.faces)
        n=len(shape)
        verts=np.empty((0,10475,3),np.float32)
        if n:
            rot=roma.rotmat_to_rotvec(pred['smpl_rotmat'][0].float().cpu())
            trans=pred['smpl_transl'][0].float().cpu()
            expr=pred['smpl_expression'][0].float().cpu()
            with torch.no_grad():
                body=layer(rot,shape,trans,None,None,K=K.expand(n,-1,-1),expression=expr)
            verts=body['smpl_v3d'].detach().numpy()
        color=((view['img'][0].permute(1,2,0).cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        np.savez_compressed(dest/f'{i:06d}.npz',vertices_cam_m=verts,K=K.numpy(),
                            camera_to_world=c2w.numpy(),points_cam_m=points[0].numpy(),
                            confidence=pred['conf_self'][0].float().cpu().numpy(),color_rgb=color,
                            human_ids=pred.get('smpl_id',torch.empty(1,0))[0].cpu().numpy())
        if i%20==0:print(f'Human3R exported {i+1}/{len(files)} people={n}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','multi_hmr','human3r'])
    parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='multi_hmr':multihmr(args.limit)
    else:human3r(args.limit)


if __name__=='__main__':main()

"""Read-only CPU diagnostic: GT head support in provided masks on every frame.

GT is used only to audit mask/coordinate consistency. This never changes
inference, detection selection, scores, thresholds, checkpoints, or exports.
Head support alone is not proof of perfect segmentation or person identity.
"""
from pathlib import Path
import json,sys,time
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,decode_mhr,save_json,sha256
from utils.v2d_official_metric import MHR_TABLE3_BODY_JOINT_INDICES


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    deadline=json.loads((OUTPUT/'budget.json').read_text())['deadline_unix']
    model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    folder=OUTPUT/'full_frame_target_mask_audit';folder.mkdir(exist_ok=True)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)).astype(bool)
    results={};started=time.time()
    for name in manifest['test']+manifest['train']:
        if time.time()>=deadline:raise TimeoutError('Original experiment deadline during mask diagnostic')
        record=manifest['samples'][name];gtpath=SOURCE/name/'human/human_params_gt.npz'
        maskpath=SOURCE/name/'processed/human_mask.mp4'
        assert sha256(gtpath)==record['gt_sha256']
        with np.load(gtpath) as z:
            pose=torch.as_tensor(z['pose'],dtype=torch.float32)
            shape=torch.as_tensor(z['shape'],dtype=torch.float32)
            scales=torch.as_tensor(z['scales'],dtype=torch.float32)
        count=record['frames'];assert len(pose)==count
        head=[]
        with torch.no_grad():
            for start in range(0,count,16):
                vertices,joints=decode_mhr(model,pose[start:start+16],shape,scales)
                head.append(joints[:,MHR_TABLE3_BODY_JOINT_INDICES[15]].numpy())
                del vertices,joints
        head=np.concatenate(head)
        E=np.asarray(record['world_to_camera'],np.float64)
        camera=head@E[:3,:3].T+E[:3,3]
        pixels=camera@np.asarray(record['K'],np.float64).T
        finite=np.isfinite(pixels).all(1)&(camera[:,2]>.05)
        uv=pixels[:,:2]/np.maximum(pixels[:,2:],1e-8)
        valid=finite&(uv[:,0]>=0)&(uv[:,0]<record['width'])&(uv[:,1]>=0)&(uv[:,1]<record['height'])
        raw=np.zeros(count,bool);support=np.zeros(count,bool);nonempty=np.zeros(count,bool)
        capture=cv2.VideoCapture(str(maskpath));assert capture.isOpened()
        for frame in range(count):
            ok,image=capture.read();assert ok,(name,frame)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if gray.shape!=(record['height'],record['width']):
                gray=cv2.resize(gray,(record['width'],record['height']),interpolation=cv2.INTER_NEAREST)
            nonempty[frame]=np.any(gray>127)
            if not valid[frame]:continue
            x,y=np.rint(uv[frame]).astype(int)
            if not (0<=x<record['width'] and 0<=y<record['height']):
                valid[frame]=False;continue
            raw[frame]=gray[y,x]>127
            x0=max(0,x-7);x1=min(record['width'],x+8)
            y0=max(0,y-7);y1=min(record['height'],y+8)
            # Exact single-pixel equivalent of the frozen ellipse15 dilation.
            support[frame]=np.any((gray[y0:y1,x0:x1]>127)&kernel[y0-y+7:y1-y+7,x0-x+7:x1-x+7])
        assert not capture.read()[0],('Mask contains extra frames',name)
        capture.release()
        outside=np.flatnonzero(valid&~support)
        np.savez_compressed(folder/(name+'.npz'),GT_head_pixels=uv,GT_head_camera_m=camera,
            GT_head_in_image=valid,mask_nonempty=nonempty,GT_head_in_raw_mask=raw,GT_head_in_dilated_support=support)
        results[name]=dict(frames=count,GT_head_in_image_frames=int(valid.sum()),
            GT_head_in_raw_mask_frames=int((valid&raw).sum()),GT_head_supported_frames=int((valid&support).sum()),
            GT_head_outside_support_frames=int(len(outside)),outside_support_frame_ids=outside.tolist(),
            mask_empty_frames=int((~nonempty).sum()),GT_sha256=record['gt_sha256'],mask_sha256=sha256(maskpath))
        assert sha256(gtpath)==record['gt_sha256']
        save_json(folder/'audit.json',dict(status='complete' if len(results)==30 else 'running',samples=results,
            frames_checked=sum(x['frames'] for x in results.values()),clips_checked=len(results),
            evaluation_protocol='target_foreground_head_v1',GT_used_for_mask_coordinate_audit_only=True,
            GT_used_for_inference_or_recipe_selection=False,changes_to_existing_results=False,
            all_frames_manually_reviewed=False,head_support_does_not_prove_segmentation_or_identity=True,
            support_kernel='same native ellipse15 as frozen inference',gpu_used=False,
            elapsed_seconds=time.time()-started,updated_unix=time.time()))
        print(name,'frames',count,'GT head outside support',len(outside),'invalid/out-of-image',int((~valid).sum()),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'full_frame_target_mask_audit_error.json',dict(error=repr(error),time=time.time()))
        raise

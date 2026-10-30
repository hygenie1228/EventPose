"""Inspect foreground masks and current target-only predictions on CPU."""
from pathlib import Path
import argparse,fcntl,json,os,sys,time
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,save_json,sha256
from visualize_track2_validation import joints,draw


def main(trial='original',exported=False,split='test',samples=None,frame_triplet=None):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    manifest=json.loads((OUTPUT/'manifest.json').read_text());out=OUTPUT/'target_foreground_visuals'
    if trial!='original' or exported:out=out/(trial+'_exported' if exported else trial)
    if split!='test':out=out/split
    if frame_triplet is not None:
        assert len(frame_triplet)==3 and list(frame_triplet)==sorted(set(frame_triplet))
        out=out/('frames_'+'_'.join(map(str,frame_triplet)))
    out.mkdir(parents=True,exist_ok=True)
    receipt={}
    names=sorted(manifest['samples']) if split=='all' else manifest[split]
    if samples:
        assert set(samples)<=set(names)
        names=samples
    for name in names:
        folder=OUTPUT/'predictions'/trial/name;inference=json.loads((folder/'inference.json').read_text())
        assert inference['evaluation_protocol'] in ['target_foreground_head_v1','native_rgb_target_head_v1',
            'native_rgb_static_crop_target_head_v1','native_rgb_exp_then_focal_target_head_v1',
            'native_rgb_strict_mask_target_head_v2','native_rgb_static_crop_strict_mask_target_head_v2',
            'native_rgb_strict_mask_body_box_v3','native_rgb_static_crop_strict_mask_body_box_v3']
        assert inference['all_accepted_heads_in_target_mask']
        native_input=not inference['target_foreground_only_input']
        static_box=inference.get('static_crop_box')
        record=manifest['samples'][name];frames=list(frame_triplet) if frame_triplet is not None else [0,record['frames']//2,record['frames']-1]
        assert min(frames)>=0 and max(frames)<record['frames']
        gt=joints(model,SOURCE/name/'human/human_params_gt.npz',frames)
        E=np.asarray(record['world_to_camera'],np.float32);gt=gt@E[:3,:3].T+E[:3,3]
        init=joints(model,SOURCE/name/'human/human_params_init.npz',frames)
        with np.load(folder/'smplx_predictions.npz') as z:raw=z['joints_cam_m'][frames,:22];valid=z['detection_valid'][frames]
        path=SOURCE/name/'human/human_params_uranus.npz' if exported else folder/'human_params_mhr_v2.npz'
        if exported:
            with np.load(path) as z:
                meta=json.loads(z['report'].item())
                assert meta['trial']==trial and meta['rgb_inference']['checkpoint_sha256']==inference['checkpoint_sha256']
                assert meta['evaluation_protocol']==inference['evaluation_protocol'] and str(z['coordinate_frame'])=='opencv_camera'
        native=joints(model,path,frames) if path.exists() else None
        cap=cv2.VideoCapture(str(SOURCE/name/'video.mp4'));maskcap=cv2.VideoCapture(str(SOURCE/name/'processed/human_mask.mp4'))
        scale=480/record['width'];height=round(record['height']*scale);K=np.asarray(record['K'],np.float32);K[:2]*=scale;rows=[]
        for i,frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);maskcap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,bgr=cap.read();ok2,mask=maskcap.read();assert ok and ok2
            gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            if gray.shape!=bgr.shape[:2]:gray=cv2.resize(gray,(bgr.shape[1],bgr.shape[0]),interpolation=cv2.INTER_NEAREST)
            foreground=bgr.copy();foreground[gray<=127]=[104,116,124]
            panels=[]
            for label,points in [('target input',None),('init_v6',init),(trial+' raw SMPLX',raw),(trial+(' exported MHR' if exported else ' MHR v2'),native)]:
                panel=cv2.resize(foreground if label=='target input' and not native_input else bgr,(480,height))
                if label=='target input' and static_box:
                    from multihmr.target_crop import prepare_native_rgb_box
                    actual,_,_=prepare_native_rgb_box(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),np.asarray(record['K'],np.float32),static_box)
                    square=cv2.resize(cv2.cvtColor(np.asarray(actual),cv2.COLOR_RGB2BGR),(height,height))
                    left=(480-height)//2;panel=cv2.copyMakeBorder(square,0,0,left,480-height-left,cv2.BORDER_CONSTANT,value=(0,0,0))
                if points is not None:draw(panel,gt[i],K,(40,230,40));draw(panel,points[i],K,(30,30,245))
                panel=cv2.copyMakeBorder(panel,44,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                cv2.putText(panel,f'{label} / frame {frame}',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
                text=f'GT Z {gt[i,0,2]:.2f}m / pred {points[i,0,2]:.2f}m' if points is not None else (('fixed crop RGB / corrected K' if static_box else 'original RGB retained' if native_input else 'target RGB only') if label=='target input' else 'conversion pending')
                if points is raw or points is native:text+=f' / detected {bool(valid[i])}'
                cv2.putText(panel,text,(8,35),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA);panels.append(panel)
            rows.append(np.concatenate(panels,axis=1))
        cap.release();maskcap.release();destination=out/(name+'.jpg');assert cv2.imwrite(str(destination),np.concatenate(rows,axis=0))
        receipt[name]=dict(path=str(destination),frames=frames,sha256=sha256(destination),
            inference_protocol=inference['evaluation_protocol'],sampled_detection_valid=valid.tolist(),
            actual_input_static_crop_box=static_box,input_panel_preserves_aspect_ratio=True,
            raw_root_camera_m=raw[:,0].tolist(),GT_used_for_visual_evaluation_only=True,
            prediction_color='red',GT_color='green',MHR_v2_available=native is not None,
            native_prediction_path=str(path),native_prediction_sha256=sha256(path) if path.exists() else None)
        print(name,'visual prepared',flush=True)
    save_json(out/'receipt.json',dict(samples=receipt,frames_checked=3*len(receipt),
        split=split,frames_rendered=3*len(receipt),test_used_for_selection=False,
        manual_review_pending=True,all_frames_manually_reviewed=False,
        trial=trial,exported_files_rendered=exported,evaluation_protocol=inference['evaluation_protocol'],gpu_used=False,created_unix=time.time()))


def watch_final(strict=False,body_target=False):
    """Prepare actual exported-source panels once the frozen comparison finishes."""
    prefix='body_target' if body_target else 'strict' if strict else 'final'
    with (OUTPUT/'locks'/(prefix+'_visual_preparation.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def state(status,**extra):
            save_json(OUTPUT/(prefix+'_visual_preparation_status.json'),dict(status=status,pid=os.getpid(),
                gpu_used=False,manual_review_pending=True,updated_unix=time.time(),**extra))
        state('waiting_for_final_all30_exports')
        deadline=json.loads((OUTPUT/'budget.json').read_text())['deadline_unix']
        while True:
            path=OUTPUT/('body_target_evaluation_status.json' if body_target else 'strict_evaluation_status.json' if strict else 'fixed_finalization_status.json')
            ready='all30_ready_for_independent_strict_audit' if strict else 'all30_ready_for_independent_final_audit'
            if path.exists() and json.loads(path.read_text())['status']==ready:break
            assert not (OUTPUT/('body_target_evaluation_error.json' if body_target else 'strict_evaluation_error.json' if strict else 'fixed_finalization_error.json')).exists()
            if time.time()>=deadline:
                state('not_rendered_final_exports_incomplete_at_deadline');return
            time.sleep(30)
        selection=json.loads((OUTPUT/(prefix+'_selected.json')).read_text())
        receipts=json.loads((OUTPUT/(prefix+'_export_receipts.json')).read_text())
        trial=selection['trial'];assert receipts['trial']==trial and len(receipts['samples'])==30
        if strict:
            assert not selection['test_used_for_selection'] and receipts['strict_target_mask']
            for name in receipts['samples']:
                folder=OUTPUT/'predictions'/trial/name
                info=json.loads((folder/'inference.json').read_text())
                audit=json.loads((folder/'strict_head_mask_audit.json').read_text())
                assert info['strict_target_mask'] and info['target_mask_dilation_kernel']==0
                if body_target:assert info['target_body_overlap'] and audit.get('target_body_overlap',True)
                assert not info['target_foreground_only_input'] and not info['gt_used_for_detection_or_tracking']
                assert audit['outside_original_mask']==0 and audit['raw_sha256']==sha256(folder/'smplx_predictions.npz')
                assert audit['mask_sha256']==sha256(SOURCE/name/'processed/human_mask.mp4')
        for name,item in receipts['samples'].items():
            assert Path(item['path']).resolve()==(SOURCE/name/'human/human_params_uranus.npz').resolve()
            assert sha256(item['path'])==item['sha256']
        state('rendering_actual_final_source_all30',trial=trial)
        main(trial,exported=True,split='all')
        path=OUTPUT/'target_foreground_visuals'/(trial+'_exported')/'all/receipt.json'
        visual=json.loads(path.read_text())
        assert set(visual['samples'])==set(receipts['samples']) and visual['frames_rendered']==90
        assert visual['exported_files_rendered'] and visual['manual_review_pending']
        for name,item in receipts['samples'].items():
            assert sha256(item['path'])==item['sha256']
            assert visual['samples'][name]['native_prediction_sha256']==item['sha256']
        state('final_source_90_samples_rendered_pending_manual_review',trial=trial,
            receipt=str(path),receipt_sha256=sha256(path),frames_rendered=90)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='original');parser.add_argument('--exported',action='store_true')
    parser.add_argument('--split',choices=['train','test','validation','all'],default='test')
    parser.add_argument('--samples',nargs='+')
    parser.add_argument('--frame-triplet',nargs=3,type=int,help='Render three diagnostic frames into a separate directory')
    parser.add_argument('--watch-final',action='store_true')
    parser.add_argument('--watch-strict-final',action='store_true')
    parser.add_argument('--watch-body-final',action='store_true')
    args=parser.parse_args()
    if args.watch_body_final:watch_final(strict=True,body_target=True)
    elif args.watch_strict_final:watch_final(strict=True)
    elif args.watch_final:watch_final()
    else:main(args.trial,args.exported,args.split,args.samples,args.frame_triplet)

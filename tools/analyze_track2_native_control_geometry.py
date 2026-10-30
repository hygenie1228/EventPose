"""CPU Train-only controls separating input effects from global motion/pose."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
import smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,mapping,transfer,decode_mhr,save_json,sha256
from utils import v2d_official_metric as official
from finalize_track2_experiments import process_identity


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--alternative-trial',choices=['diagnostic_original_static_crop','diagnostic_original_depth_order'])
    parser.add_argument('--learned-trial',choices=['native_head_cd100','native_head_cd100_rel100','native_last2_cd100_rel100'])
    parser.add_argument('--head-pelvis-counterfactual',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    manifest=read(OUTPUT/'manifest.json');samples=manifest['validation']
    assert len(samples)==4 and set(samples).isdisjoint(manifest['test'])
    body=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    smpl=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
    all_ids,all_weights=mapping('smplx2mhr',smpl.faces,'cpu')
    conditions=[('foreground','original','target_foreground_head_v1'),('native_RGB','diagnostic_original_native_rgb','native_rgb_target_head_v1')]
    control='native_rgb_control';destination=OUTPUT/'native_rgb_control_geometry.json'
    if args.alternative_trial:
        is_crop=args.alternative_trial=='diagnostic_original_static_crop'
        label='static_crop' if is_crop else 'experimental_depth_order'
        protocol='native_rgb_static_crop_target_head_v1' if is_crop else 'native_rgb_exp_then_focal_target_head_v1'
        conditions=[conditions[1],(label,args.alternative_trial,protocol)]
        control='static_crop_Train_control' if is_crop else 'exact_depth_order_Train_control'
        destination=OUTPUT/(control+'_geometry.json')
    if args.head_pelvis_counterfactual:
        assert args.alternative_trial=='diagnostic_original_static_crop'
        destination=OUTPUT/'head_pelvis_Train_counterfactual_geometry.json'
    learned_checkpoint_sha=None
    if args.learned_trial:
        assert not args.alternative_trial and not args.head_pelvis_counterfactual
        trial='search_'+args.learned_trial
        checkpoint=Path(read(OUTPUT/'native_checkpoint_choices'/(args.learned_trial+'.json'))['checkpoint'])
        learned_checkpoint_sha=sha256(checkpoint)
        conditions=[conditions[1],(args.learned_trial,trial,'native_rgb_target_head_v1')]
        destination=OUTPUT/(args.learned_trial+'_Train_geometry.json')
        for sample in samples:
            assert (OUTPUT/'predictions'/trial/sample/'inference.json').exists()
        supervisor=None;tick=None
    else:
        supervisor=read(OUTPUT/(control+'_status.json'))['pid'];tick=process_identity(supervisor)
    deadline=read(OUTPUT/'budget.json')['deadline_unix'];results={}
    for sample in samples:
        record=manifest['samples'][sample]
        for _,trial,_ in conditions:
            native=OUTPUT/'predictions'/trial/sample
            while not (native/'inference.json').exists():
                assert tick is not None and time.time()<deadline and process_identity(supervisor)==tick,'Control supervisor ended before required inference'
                time.sleep(15)
        reference=read(OUTPUT/'predictions/original'/sample/'metrics.json')['protocol']['roles']
        selected=reference['alignment']+reference['body'];ids=all_ids[selected];weights=all_weights[selected]
        with np.load(SOURCE/sample/'human/human_params_gt.npz') as z:
            pose=torch.tensor(z['pose'],dtype=torch.float32);shape=torch.tensor(z['shape'],dtype=torch.float32);scales=torch.tensor(z['scales'],dtype=torch.float32)
        target=[];gt_roots=[]
        with torch.no_grad():
            for start in range(0,len(pose),8):
                verts,joints=decode_mhr(body,pose[start:start+8],shape,scales)
                target.append(verts[:,selected].numpy());gt_roots.append(joints[:,official.MHR_TABLE3_BODY_JOINT_INDICES[0]].numpy())
        target=np.concatenate(target).astype(np.float64);gt_roots=np.concatenate(gt_roots)
        E=np.asarray(record['world_to_camera']);target=target@E[:3,:3].T+E[:3,3];gt_roots=gt_roots@E[:3,:3].T+E[:3,3]
        values={}
        for label,trial,protocol in conditions:
            folder=OUTPUT/'predictions'/trial/sample;info=read(folder/'inference.json')
            assert info['evaluation_protocol']==protocol and info['all_accepted_heads_in_target_mask']
            assert info['frames']==record['frames']
            if args.learned_trial and label==args.learned_trial:
                assert info['delta_checkpoint'] is not None and info['checkpoint_sha256']==learned_checkpoint_sha
            else:assert info['delta_checkpoint'] is None
            with np.load(folder/'smplx_predictions.npz') as z:
                vertices=z['vertices_cam_m'];all_joints=z['joints_cam_m'];roots=all_joints[:,0];valid=z['detection_valid']
                assert np.array_equal(z['K'],np.asarray(record['K'],np.float32))
            mapped=[]
            with torch.no_grad():
                for start in range(0,len(vertices),16):mapped.append(transfer(torch.tensor(vertices[start:start+16]),ids,weights).numpy())
            pred=np.concatenate(mapped).astype(np.float64)
            fixed=official.fit_similarity_transform(pred[0,:27],target[0,:27])
            aligned=official.apply_similarity_transform(pred,fixed)
            fixed_cd=float(official.symmetric_chamfer_distance(aligned[:,27:],target[:,27:]).mean()*100)
            shift=target[:,:27].mean(1)-aligned[:,:27].mean(1)
            centered=aligned+shift[:,None]
            centered_cd=float(official.symmetric_chamfer_distance(centered[:,27:],target[:,27:]).mean()*100)
            per_frame=np.stack([official.apply_similarity_transform(pred[i],official.fit_similarity_transform(pred[i,:27],target[i,:27])) for i in range(len(pred))])
            per_frame_cd=float(official.symmetric_chamfer_distance(per_frame[:,27:],target[:,27:]).mean()*100)
            chosen=[0,len(pred)//2,len(pred)-1]
            values[label]=dict(raw_SMPLX_fixed_frame0_CD_proxy_cm=fixed_cd,
                diagnostic_per_frame_center_corrected_CD_cm=centered_cd,
                diagnostic_per_frame_Sim3_CD_cm=per_frame_cd,
                fixed_alignment_centroid_trajectory_RMSE_cm=float(np.sqrt(np.mean(np.sum(shift**2,axis=1)))*100),
                frame0_Sim3_scale=float(fixed[0]),detection_coverage=float(valid.mean()),
                sampled_frames=chosen,predicted_camera_root_Z_m=roots[chosen,2].tolist(),GT_camera_root_Z_m=gt_roots[chosen,2].tolist(),
                inference_sha256=sha256(folder/'inference.json'))
            if args.head_pelvis_counterfactual:
                # GT-free hypothetical reinterpretation, not an upstream fix:
                # move the whole body along its head ray until pelvis Z equals
                # the original head Z. Pose and head projection stay fixed.
                head=all_joints[:,15].astype(np.float64)
                delta=head/head[:,2:]*(head[:,2]-roots[:,2])[:,None]
                hypothetical=pred+delta[:,None]
                fitted=official.fit_similarity_transform(hypothetical[0,:27],target[0,:27])
                hypothetical=official.apply_similarity_transform(hypothetical,fitted)
                values[label]['pelvis_depth_reinterpretation_CD_proxy_cm']=float(official.symmetric_chamfer_distance(hypothetical[:,27:],target[:,27:]).mean()*100)
                values[label]['counterfactual_is_not_a_proven_pretrained_decoder_fix']=True
        results[sample]=dict(frames=record['frames'],conditions=values)
        save_json(destination,dict(samples=results,test_used=False,GT_for_diagnostics_only=True,
            all_four_complete=False,diagnostic_scores_not_official_MHR_CD_H=True,
            caveat='Alternative alignments are counterfactual diagnostics, not additive error components or final metrics.',updated_unix=time.time()))
        print(sample,json.dumps(values),flush=True)
    summary_keys=['raw_SMPLX_fixed_frame0_CD_proxy_cm','diagnostic_per_frame_center_corrected_CD_cm','diagnostic_per_frame_Sim3_CD_cm','fixed_alignment_centroid_trajectory_RMSE_cm']
    if args.head_pelvis_counterfactual:summary_keys.append('pelvis_depth_reinterpretation_CD_proxy_cm')
    summary={label:{key:float(np.mean([v['conditions'][label][key] for v in results.values()]))
                for key in summary_keys}
             for label,_,_ in conditions}
    save_json(destination,dict(samples=results,summary=summary,test_used=False,GT_for_diagnostics_only=True,
        all_four_complete=True,diagnostic_scores_not_official_MHR_CD_H=True,
        caveat='No additive decomposition claim; same Train roster and output head gate, with explicit original/frozen learned checkpoint provenance.',updated_unix=time.time()))


if __name__=='__main__':main()

"""Train-only sampled counterfactual for focal scaling versus log decoding.

This is an experimental decoder variant, not a claim about upstream correctness.
Original pose/identity/network outputs and all published scores stay unchanged.
"""
from pathlib import Path
import json,sys,time,math
import numpy as np,torch,smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,decode_mhr,mapping,transfer,save_json,sha256
from utils import v2d_official_metric as official


def main():
    torch.set_num_threads(2);manifest=json.loads((OUTPUT/'manifest.json').read_text())
    body=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    smpl=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
    all_ids,all_weights=mapping('smplx2mhr',smpl.faces,'cpu');rows={}
    out=OUTPUT/'checkpoint_depth_compatibility';out.mkdir(exist_ok=True)
    for sample in manifest['validation']:
        record=manifest['samples'][sample];frames=np.unique(np.linspace(0,record['frames']-1,64,dtype=int))
        roles=json.loads((OUTPUT/'predictions/diagnostic_original_native_rgb'/sample/'metrics.json').read_text())['protocol']['roles']
        selected=roles['alignment']+roles['body'];ids,weights=all_ids[selected],all_weights[selected]
        source=OUTPUT/'predictions/diagnostic_original_native_rgb'/sample/'smplx_predictions.npz'
        with np.load(source) as z:verts=torch.tensor(z['vertices_cam_m'][frames]);rv=torch.tensor(z['rotvec'][frames]);betas=torch.tensor(z['shape'][frames])
        with torch.no_grad():
            generated=smpl(global_orient=rv[:,0],body_pose=rv[:,1:22].flatten(1),
                left_hand_pose=rv[:,22:37].flatten(1),right_hand_pose=rv[:,37:52].flatten(1),
                jaw_pose=rv[:,52],betas=betas,transl=torch.zeros(len(frames),3),expression=torch.zeros(len(frames),10),
                leye_pose=torch.zeros(len(frames),3),reye_pose=torch.zeros(len(frames),3))
            translation=(verts-generated.vertices).median(1).values
            residual=verts-generated.vertices-translation[:,None]
            head=generated.joints[:,15]+translation
            assert torch.isfinite(head).all() and (head[:,2]>.05).all()
            # Reconstruct the original decoder's scalar without re-running its
            # network. r is fixed by supplied K and original 896 contain resize.
            r=(record['K'][0][0]*896/max(record['width'],record['height']))/(896/(2*math.tan(math.radians(30))))
            z_old=head[:,2];z_variant=torch.exp(torch.log(z_old)/r)*r
            delta=head/z_old[:,None]*(z_variant-z_old)[:,None]
            original=transfer(verts,ids,weights).numpy().astype(np.float64)
            alternative=original+delta[:,None].numpy()
            head_uv_before=head[:,:2]/head[:,2:]
            head_new=head+delta
            assert torch.allclose(head_uv_before,head_new[:,:2]/head_new[:,2:],atol=1e-6)
        with np.load(SOURCE/sample/'human/human_params_gt.npz') as z:
            pose=torch.tensor(z['pose'][frames],dtype=torch.float32);shape=torch.tensor(z['shape'],dtype=torch.float32);scales=torch.tensor(z['scales'],dtype=torch.float32)
        target=[]
        with torch.no_grad():
            for start in range(0,len(frames),8):
                v,_=decode_mhr(body,pose[start:start+8],shape,scales);target.append(v[:,selected].numpy())
        target=np.concatenate(target).astype(np.float64);E=np.asarray(record['world_to_camera']);target=target@E[:3,:3].T+E[:3,3]
        scores={}
        for label,pred in [('published_order',original),('experimental_exp_then_focal',alternative)]:
            transform=official.fit_similarity_transform(pred[0,:27],target[0,:27]);aligned=official.apply_similarity_transform(pred,transform)
            scores[label]=float(official.symmetric_chamfer_distance(aligned[:,27:],target[:,27:]).mean()*100)
        rows[sample]=dict(sampled_frames=frames.tolist(),fixed_focal_ratio=r,CD_surface_proxy_cm=scores,
            head_depth_old_m=z_old.tolist(),head_depth_variant_m=z_variant.tolist(),
            reconstruction_body_median_residual_mm=float(residual.norm(dim=-1).median()*1000),
            original_source_sha256=sha256(source),same_head_projection=True,weights_and_local_pose_unchanged=True)
        save_json(out/'depth_order_sampled_control.json',dict(samples=rows,test_used=False,all_four_complete=False,
            experimental_decoder_variant=True,not_official_MHR_CD_H=True,updated_unix=time.time()))
        print(sample,r,scores,flush=True)
    summary={key:float(np.mean([v['CD_surface_proxy_cm'][key] for v in rows.values()])) for key in ['published_order','experimental_exp_then_focal']}
    save_json(out/'depth_order_sampled_control.json',dict(samples=rows,summary=summary,test_used=False,
        all_four_complete=True,experimental_decoder_variant=True,not_official_MHR_CD_H=True,
        approximation='Primary head translation reconstructed from zero-expression SMPL-X and median mesh translation; original face expression was not retained.',
        no_changes_to_original_outputs_training_or_selection=True,updated_unix=time.time()))


if __name__=='__main__':main()

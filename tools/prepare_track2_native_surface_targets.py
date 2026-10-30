"""Cache exact native MHR Train surfaces and measure fitted-target discrepancy.

This prepares a possible later refinement; existing labels, training configs,
selection, frozen comparisons and source exports remain unchanged.
"""
from pathlib import Path
import fcntl,json,os,sys,time
import numpy as np,torch,smplx,roma
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,decode_mhr,mapping,transfer,sha256,save_json
from utils import v2d_official_metric as official


def read(path):return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(2)
    with (OUTPUT/'locks/native_surface_target_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read(OUTPUT/'manifest.json');roles=read(OUTPUT/'body_roles.json')['roles']
        selected=roles['alignment']+roles['body'];assert len(selected)==91
        role_sha=sha256(OUTPUT/'body_roles.json');out=OUTPUT/'work/native_surface_targets';out.mkdir(exist_ok=True)
        mhr=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
        body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
        ids,weights=mapping('smplx2mhr',body.faces,'cpu');ids,weights=ids[selected],weights[selected]
        results={}
        for sample in manifest['train']:
            assert sample not in manifest['test'];r=manifest['samples'][sample];label=OUTPUT/'work/targets'/(sample+'.npz')
            with np.load(label) as z:cache={k:z[k] for k in z.files}
            frames=cache['frames'];assert frames[0]==0 and cache['source_sha256'].item()==r['gt_sha256']
            with np.load(SOURCE/sample/'human/human_params_gt.npz') as z:
                pose=torch.tensor(z['pose'][frames],dtype=torch.float32);shape=torch.tensor(z['shape'],dtype=torch.float32);scales=torch.tensor(z['scales'],dtype=torch.float32)
            E=np.asarray(r['world_to_camera'],np.float64);native=[];fitted=[];max_error=0.
            with torch.no_grad():
                for start in range(0,len(frames),16):
                    stop=min(len(frames),start+16);n=stop-start
                    verts,_=decode_mhr(mhr,pose[start:stop],shape,scales)
                    native.append((verts[:,selected].numpy().astype(np.float64)@E[:3,:3].T+E[:3,3]).astype(np.float32))
                    rv=roma.rotmat_to_rotvec(torch.tensor(cache['rotations'][start:stop,0]));zero=lambda width:torch.zeros(n,width)
                    output=body(global_orient=rv[:,0],body_pose=rv[:,1:22].flatten(1),
                        left_hand_pose=rv[:,22:37].flatten(1),right_hand_pose=rv[:,37:52].flatten(1),jaw_pose=rv[:,52],
                        betas=torch.tensor(cache['betas'][start:stop,0]),transl=torch.tensor(cache['transl'][start:stop]),
                        expression=zero(10),leye_pose=zero(3),reye_pose=zero(3))
                    error=float(((body.J_regressor@output.vertices)-torch.tensor(cache['joints'][start:stop,0])).abs().max())
                    max_error=max(max_error,error);assert error<2e-4
                    fitted.append(transfer(output.vertices,ids,weights).numpy())
            native=np.concatenate(native);fitted=np.concatenate(fitted);assert native.shape==(len(frames),91,3) and np.isfinite(native).all()
            path=out/(sample+'.npz');temporary=path.with_suffix('.partial.npz')
            np.savez_compressed(temporary,surface_points_cam_m=native,frames=frames,source_gt_sha256=np.asarray(r['gt_sha256']),
                fitted_label_sha256=np.asarray(sha256(label)),metric_roles_sha256=np.asarray(role_sha),coordinate_frame=np.asarray('opencv_camera'))
            temporary.replace(path)
            transform=official.fit_similarity_transform(fitted[0,:27].astype(np.float64),native[0,:27].astype(np.float64))
            aligned=official.apply_similarity_transform(fitted[:,27:].astype(np.float64),transform)
            discrepancy=float(official.symmetric_chamfer_distance(aligned,native[:,27:].astype(np.float64)).mean()*100)
            results[sample]=dict(frames=len(frames),target_cache_path=str(path),target_cache_sha256=sha256(path),
                fitted_target_to_native_GT_CD_proxy_cm=discrepancy,max_fitted_label_reconstruction_error_m=max_error,
                direct_native_point_correspondence_mean_mm=float(np.linalg.norm(fitted-native,axis=-1).mean()*1000))
            save_json(OUTPUT/'native_surface_target_preparation_status.json',dict(status='preparing_Train20_only',pid=os.getpid(),
                completed=list(results),Test_labels_read=False,existing_training_unchanged=True,updated_unix=time.time()))
            print(sample,len(frames),'fitted_target_discrepancy_cm',discrepancy,flush=True)
        assert sum(v['frames'] for v in results.values())==1107 and set(results)==set(manifest['train'])
        val={s:results[s] for s in manifest['validation']}
        save_json(OUTPUT/'native_surface_target_preparation.json',dict(samples=results,validation=val,
            validation_fitted_target_CD_discrepancy_cm=float(np.mean([v['fitted_target_to_native_GT_CD_proxy_cm'] for v in val.values()])),
            total_Train_cached_frames=1107,GT_for_training_or_diagnostics_only=True,Test_labels_read=False,
            existing_labels_and_model_scores_unchanged=True,new_training_not_launched=True,completed_unix=time.time(),
            caveat='This measures target-space mismatch, not an additive error term or a proven irreducible accuracy floor.'))
        save_json(OUTPUT/'native_surface_target_preparation_status.json',dict(status='complete',pid=os.getpid(),
            completed=list(results),Test_labels_read=False,existing_training_unchanged=True,updated_unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'native_surface_target_preparation_error.json',dict(error=repr(error),time=time.time()));raise

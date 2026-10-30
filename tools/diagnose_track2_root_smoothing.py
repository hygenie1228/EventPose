"""Train-only CPU control: fixed five-frame Gaussian on MHR root translation.

The original predictions, frozen comparisons, selection and source exports are
untouched. This is an explicitly separate postprocessing experiment.
"""
from pathlib import Path
import fcntl,json,os,sys,time
import numpy as np,torch
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,save_json,sha256
from utils.mhr_evaluation_body import LocalMHRBody
from utils.track2_metrics import score_human
from utils.v2d_official_metric import MHR_TABLE3_BODY_JOINT_INDICES


def read(path):return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in ['pose','shape','scales']}


def main():
    torch.set_num_threads(2)
    with (OUTPUT/'locks/root_smoothing_Train_control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read(OUTPUT/'manifest.json');trial='diagnostic_original_static_crop'
        folder=OUTPUT/'diagnostic_root_smoothing';folder.mkdir(exist_ok=True)
        plan_path=folder/'plan.json';plan=dict(model=trial,samples=manifest['validation'],sigma_frames=1.,
            truncate=2.,mode='nearest',parameters='root translation only: first3 MHR pose entries',
            GT_used_for_filter=False,test_used=False,original_comparisons_unchanged=True,registered_unix=time.time())
        if not plan_path.exists():save_json(plan_path,plan)
        body=LocalMHRBody(ASSETS/'mhr/mhr_model.pt',MHR_TABLE3_BODY_JOINT_INDICES,device='cpu')
        body.roles={k:np.asarray(v) for k,v in read(OUTPUT/'body_roles.json')['roles'].items()}
        first=manifest['validation'][0];base=arrays(OUTPUT/'predictions'/trial/first/'human_params_mhr_v2.npz')
        # Verify root parameter semantics with the actual model before filtering.
        params=np.concatenate([base['pose'][[0]],np.broadcast_to(base['scales'],(1,68))],1)
        v0,j0=body(params,base['shape']);jacobian=[]
        for axis in range(3):
            moved=params.copy();moved[0,axis]+=.01;v1,j1=body(moved,base['shape'])
            delta=v1-v0;mean=delta.mean((0,1))
            assert np.max(np.abs(delta-mean))<5e-6,'First3 pose parameters are not verified global translation'
            assert np.max(np.abs((j1[:,list(MHR_TABLE3_BODY_JOINT_INDICES)]-j0[:,list(MHR_TABLE3_BODY_JOINT_INDICES)])-mean))<5e-6
            jacobian.append((mean/.01).tolist())
        save_json(folder/'root_parameter_semantics_check.json',dict(camera_displacement_metres_per_parameter=jacobian,
            exact_model_translation_verified=True,scored_body22_joint_translation_verified=True,
            joint0_unscored_fixed_origin_excluded_from_translation_assertion=True,gpu_used=False,checked_unix=time.time()))
        results={};init_results={}
        for sample in manifest['validation']:
            assert sample not in manifest['test']
            path=OUTPUT/'predictions'/trial/sample/'human_params_mhr_v2.npz';before=sha256(path)
            original=arrays(path);target=arrays(SOURCE/sample/'human/human_params_gt.npz')
            previous=read(path.parent/'metrics.json');assert previous['evaluation_protocol']=='native_rgb_static_crop_target_head_v1'
            baseline=score_human(body,original,target)
            assert abs(baseline['CD_H_cm']-previous['CD_H_cm'])<.02
            assert abs(baseline['ACC_H_cm_per_frame2']-previous['ACC_H_cm_per_frame2'])<.02
            variant={k:v.copy() for k,v in original.items()}
            variant['pose'][:,:3]=gaussian_filter1d(original['pose'][:,:3].astype(np.float64),sigma=1.,axis=0,mode='nearest',truncate=2.)
            assert np.array_equal(variant['pose'][:,3:],original['pose'][:,3:])
            smoothed=score_human(body,variant,target)
            stage=folder/(sample+'.npz');np.savez_compressed(stage,**variant)
            assert sha256(path)==before
            results[sample]=dict(baseline=baseline,root_sigma1=smoothed,base_prediction_sha256=before,
                GT_sha256=manifest['samples'][sample]['gt_sha256'],variant_path=str(stage),variant_sha256=sha256(stage))
            init_path=SOURCE/sample/'human/human_params_init.npz';init_results[sample]=score_human(body,arrays(init_path),target)
            save_json(folder/'progress.json',dict(samples=results,all_four_complete=False,GT_for_evaluation_only=True,
                source_exports_changed=False,test_used=False,updated_unix=time.time()))
            print(sample,dict(baseline=baseline,root_sigma1=smoothed),flush=True)
        assert set(results)==set(manifest['validation'])
        summary={condition:{key:float(np.mean([v[condition][key] for v in results.values()])) for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
            for condition in ['baseline','root_sigma1']}
        save_json(folder/'comparison.json',dict(samples=results,summary=summary,all_four_complete=True,
            GT_used_for_filter=False,GT_used_for_evaluation_only=True,test_used=False,source_exports_changed=False,
            plan_sha256=sha256(plan_path),postprocessing_experiment_not_main_comparison=True,completed_unix=time.time()))
        save_json(OUTPUT/'init_v6_Train4_MHR_baseline.json',dict(per_sequence=init_results,
            summary={key:float(np.mean([v[key] for v in init_results.values()])) for key in ['CD_H_cm','ACC_H_cm_per_frame2']},
            sequences=4,frames=sum(v['frames'] for v in init_results.values()),input_params_unchanged=True,test_used=False,completed_unix=time.time()))
        print('all_four_Train_root_filter_control',summary,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'root_smoothing_Train_error.json',dict(error=repr(error),time=time.time()));raise

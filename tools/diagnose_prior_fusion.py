"""Real legacy-weight modality probe and Train-only event activity audit."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from goal_runtime_control import register_job,write


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_legacy_weights_and_Train_activity_diagnostic')
    os.environ['EVENTPOSE_MULTISENSOR_OUT']=goal['output_root'];os.environ['XFORMERS_DISABLED']='1'
    import gc
    import hashlib
    import h5py
    import numpy as np
    import torch
    from utils.pose_evaluation import load_multihmr
    from utils.humanpose_metrics import match_people,joint_errors
    from multihmr.multisensor_model import MultiSensorMultiHMR,CONDITIONS
    from multihmr.multisensor_data import Frames,collate,input_keys
    torch.set_num_threads(2);out=Path(goal['output_root']);prior=Path(goal['prior_experiment'])
    expected={}
    for file in ('first_four_checkpoint_cpu_audit.json','remaining_two_checkpoint_cpu_audit.json'):
        for row in json.loads((prior/file).read_text())['records']:
            expected[row['condition']]=row['checkpoints']['best']['sha256']
    records=[]
    reg=torch.from_numpy(np.load(out/'joint_regressor.npy'))
    for condition in ('rgb3_event4','event3_rgb4','event3_event4'):
        print(json.dumps(dict(phase='legacy_model_cpu',condition=condition)),flush=True)
        config=json.loads((prior/'configs'/(condition+'.json')).read_text())
        path=prior/'checkpoints'/condition/'best.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==expected[condition]
        saved=torch.load(path,map_location='cpu')
        assert saved['config']==config and saved['test_used_for_selection'] is False
        torch.manual_seed(config['seed']);base=load_multihmr(device='cpu');model=MultiSensorMultiHMR(base,condition).eval()
        assert not model.load_state_dict(saved['model_delta'],strict=False).unexpected_keys
        del saved
        data=Frames('training',condition=condition);batch=collate([data[0]]);inputs=input_keys(batch)
        R=data.handle(data.records[0]['session'])[model.primary_sensor+'_R_common_to_sensor'][:]
        t=data.handle(data.records[0]['session'])[model.primary_sensor+'_t_common_to_sensor'][:]
        with torch.inference_mode():
            image=inputs['image'] if model.mode=='rgb' else (model.adapter(inputs['event'],base.img_size)-model.mean)/model.std
            raw=base.backbone(image)
            context=(inputs['auxiliary'],inputs['K'],inputs['auxiliary_K'],inputs['primary_to_aux_rotation'],
                     inputs['primary_to_aux_translation'],inputs['auxiliary_available'],base.img_size)
            fused=model.fusion(raw,context);residual=fused-raw
            ratio=float(residual.norm()/raw.norm())
            original=base.backbone.forward;base.backbone.forward=lambda *_args,**_kwargs:raw
            predictions={};errors={}
            try:
                for mode in ('intact','missing_aux','zero_aux'):
                    b=dict(inputs)
                    if mode=='missing_aux':b['auxiliary_available']=torch.zeros_like(b['auxiliary_available'])
                    if mode=='zero_aux':b['auxiliary']=torch.zeros_like(b['auxiliary'])
                    people=model(**b,is_training=False,det_thresh=.3,nms_kernel_size=3)
                    joints=np.stack([(reg@p['v3d']).numpy() for p in people]) if people else np.empty((0,55,3))
                    target=batch['joints'][0].numpy();valid=batch['valid'][0].numpy().astype(bool)
                    K=batch['original_K'][0].numpy();shape=batch['image_shape'][0].tolist()
                    mapping=match_people(joints,target,valid,K,K,shape)
                    errors[mode]={str(g):[float(v)*1000 for v in joint_errors(joints[p,:22],target[g,:22])]
                                  +[float(np.linalg.norm(joints[p,0]-target[g,0]))*1000] for g,p in mapping.items()}
                    predictions[mode]=(joints-t)@R
            finally:base.backbone.forward=original
        dest=out/'diagnostics'/(condition+'_legacy_Train_ablation.npz')
        np.savez_compressed(dest,**{k+'_joints_common_camera_m':v for k,v in predictions.items()})
        records.append(dict(condition=condition,primary=model.primary_sensor,auxiliary=model.auxiliary_sensor,
                            checkpoint=str(path),sha256=digest,row=data.records[0],
                            trained_gate_logit=float(model.fusion.gate),trained_gate_sigmoid=float(model.fusion.gate.sigmoid()),
                            initial_gate_sigmoid=float(torch.tensor(-3.).sigmoid()),
                            auxiliary_residual_relative_L2=ratio,output_weight_L2=float(model.fusion.output.weight.norm()),
                            single_Train_frame_matched_errors_mm=errors,
                            error_order=['absolute_MPJPE','pelvis_aligned_MPJPE','PA_MPJPE','full3D_root'],
                            raw_prediction=str(dest)))
        for h in data.handles.values():h.close()
        del model,base,data,raw,fused,residual,image,batch,inputs,predictions
        gc.collect()
    # Full fixed Train subset only; these are active encoded pixels, not raw-event counts.
    index=json.loads((out/'training_index.json').read_text());activity={s:[] for s in ('event3','event4')}
    for session in sorted({r['session'] for r in index}):
        rows=[r for r in index if r['session']==session]
        with np.load(out/'targets'/(session+'.npz')) as z:
            gt=z['joints'];valid=z['valid']
        with h5py.File(out/'cache'/(session+'.h5'),'r',locking=False) as f:
            for sensor in activity:
                R=f[sensor+'_R_common_to_sensor'][:];t=f[sensor+'_t_common_to_sensor'][:];K=f[sensor+'_K_auxiliary'][:]
                for row in rows:
                    i=row['cache_row'];value=f[sensor][i];mask=np.any(value>0,axis=0)
                    points=gt[i,valid[i],:22].reshape(-1,3)@R.T+t
                    projected=points@K.T;uv=projected[points[:,2]>.05,:2]/projected[points[:,2]>.05,2:]
                    roi_fraction=None;roi_height=None
                    if len(uv):
                        lo=np.floor(uv.min(0)-5).astype(int);hi=np.ceil(uv.max(0)+6).astype(int)
                        x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,224)
                        if x1>x0 and y1>y0:
                            roi_fraction=float(mask[y0:y1,x0:x1].mean());roi_height=int(y1-y0)
                    activity[sensor].append(dict(session=session,frame=row['frame'],global_active_fraction=float(mask.mean()),
                                                body_ROI_active_fraction=roi_fraction,body_ROI_height_pixels=roi_height))
    summary={}
    for sensor,rows in activity.items():
        g=np.array([r['global_active_fraction'] for r in rows]);r=np.array([x['body_ROI_active_fraction'] for x in rows if x['body_ROI_active_fraction'] is not None])
        h=np.array([x['body_ROI_height_pixels'] for x in rows if x['body_ROI_height_pixels'] is not None])
        summary[sensor]=dict(frames=len(rows),global_empty_fraction=float(np.mean(g==0)),
              global_active_fraction_quantiles=np.quantile(g,[0,.25,.5,.75,1]).tolist(),ROI_frames=len(r),
              ROI_empty_fraction=float(np.mean(r==0)),ROI_active_fraction_quantiles=np.quantile(r,[0,.25,.5,.75,1]).tolist(),
              ROI_height_pixels_quantiles=np.quantile(h,[0,.25,.5,.75,1]).tolist())
    write(out/'legacy_fusion_Train_diagnostic.json',dict(status='passed',checked_unix=time.time(),legacy_weight_probes=records,
          Train_event_activity=summary,activity_scope='6501 fixed Train frames per event sensor; GT-projected ROI for analysis only, never inference/crop input',
          pixel_semantics='Nonzero encoded count33 pixels of stored sparse observations; not recovered individual events',
          limitation='Legacy model effects use one fixed Train frame, not a population performance estimate; coupled base adaptation remains'))
    write(out/'Train_event_activity_rows.json',activity)
    print(json.dumps(dict(status='passed',legacy_probes=3,Train_activity=summary)),flush=True)


if __name__=='__main__':main()

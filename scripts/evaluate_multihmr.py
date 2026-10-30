from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""One final frozen Test evaluation after validation-only checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import utils.pose_evaluation as base
from utils.segment_inference import multihmr
from multihmr.data import ROOT,OUT,WORK,save

SNAPSHOT=os.environ.get('MULTIHMR_TEST_SNAPSHOT','')
TEST=WORK/'test_snapshots'/SNAPSHOT if SNAPSHOT else WORK/'final_test'
RESULT_OUT=OUT/'test_snapshots'/SNAPSHOT if SNAPSHOT else OUT
METHOD='multihmr_finetuned'


def track2_evaluation_deadline():
    """Keep the original budget unless its verified completion supervisor resumes."""
    from multihmr.track2_data import OUTPUT, sha256
    budget=OUTPUT/'budget.json'
    deadline=json.loads(budget.read_text())['deadline_unix']
    override=os.environ.get('TRACK2_COMPLETION_DEADLINE_UNIX')
    if override is not None:
        recovery=json.loads((OUTPUT/'native_surface_completion_recovery.json').read_text())
        supervisor=Path('/proc')/str(recovery['supervisor_pid'])
        assert os.getppid()==recovery['supervisor_pid']
        assert supervisor.joinpath('stat').read_text().split()[21]==str(recovery['supervisor_start_ticks'])
        assert not recovery['training_restarted']
        assert recovery['original_budget_sha256']==sha256(budget)
        assert recovery['frozen_selection_sha256']==sha256(OUTPUT/'body_target_final_selected.json')
        assert float(override)==recovery['completion_deadline_unix']
        assert 0<recovery['completion_timeout_seconds']<=10800
        deadline=float(override)
    return deadline


def prepare():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['test_split_sha256']
    source=ROOT/'.experiment_storage/humanpose_gt_segments'
    tasks=json.loads((source/'tasks.json').read_text())
    assert len(tasks)==320 and {t['parent_session'] for t in tasks}==set(manifest['test_sessions'])
    TEST.mkdir(parents=True,exist_ok=True)
    for folder in ['rgb','logs']:(TEST/folder).mkdir(exist_ok=True)
    shutil.copyfile(source/'joint_regressor.npy',TEST/'joint_regressor.npy')
    save(TEST/'tasks.json',tasks)
    return tasks


def worker(rank,workers=5):
    base.OUT=TEST;base.configure_torch()
    tasks=json.loads((TEST/'tasks.json').read_text())
    chosen=json.loads((RESULT_OUT/'selected.json').read_text())
    model=base.load_multihmr(chosen['checkpoint'])
    names=sorted({t['parent_session'] for t in tasks})[rank::workers]
    for name in names:
        clips=[t for t in tasks if t['parent_session']==name]
        if all((TEST/METHOD/t['session']/'complete.json').exists() for t in clips):
            for t in clips:
                shutil.copytree(TEST/METHOD/t['session'],RESULT_OUT/'final_test_results'/t['session'],dirs_exist_ok=True)
            continue
        cached=TEST/'rgb'/f'{name}.mp4';shutil.copyfile(clips[0]['source_video'],cached)
        try:
            for raw in clips:
                if (TEST/METHOD/raw['session']/'complete.json').exists():continue
                task=dict(raw,video=str(cached))
                collector=base.Collector(task,METHOD)
                try:multihmr(model,collector);collector.save()
                except Exception:
                    collector.save(partial=True);raise
            for t in clips:
                shutil.copytree(TEST/METHOD/t['session'],RESULT_OUT/'final_test_results'/t['session'],dirs_exist_ok=True)
        finally:cached.unlink(missing_ok=True)


def supervise(gpus=(2,3,4,5,6)):
    tasks=prepare();running=[]
    for rank,gpu in enumerate(gpus):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TMPDIR=str(WORK/'tmp'))
        log=(TEST/'logs'/f'worker_{rank}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--rank',str(rank),'--workers',str(len(gpus))],
                           env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        running.append((p,log))
    while any(p.poll() is None for p,_ in running):
        done=sum((TEST/METHOD/t['session']/'complete.json').exists() for t in tasks)
        save(RESULT_OUT/'final_test_status.json',dict(status='running',completed_segments=done,total_segments=320,
             updated_unix=time.time(),workers=[dict(pid=p.pid,exit_code=p.poll()) for p,_ in running]))
        time.sleep(15)
    for p,log in running:log.close();assert p.returncode==0,(p.pid,p.returncode)
    assert all((TEST/METHOD/t['session']/'complete.json').exists() for t in tasks)
    subprocess.run([sys.executable,str(ROOT/'src/utils/report_multihmr.py')],check=True,cwd=ROOT)
    save(RESULT_OUT/'final_test_status.json',dict(status='complete',completed_segments=320,updated_unix=time.time()))


def prepare_event_test(config_path):
    from multihmr.event_data import OUT as event_out
    config=json.loads(Path(config_path).read_text())
    if config.get('experiment') == 'rgb_event_new_split_24h':
        assert event_out.resolve()==Path(config['output_root']).resolve()
        assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==config['split_sha256']
        source=event_out/'test'
        tasks=json.loads((source/'tasks.json').read_text())
        frozen=json.loads((event_out/'split_snapshot.json').read_text())
        expected={(session,i,start,end) for session,clips in frozen['test'].items()
                  for i,(start,end) in enumerate(clips)}
        assert {(t['parent_session'],t['segment'],t['start_frame'],t['end_frame']) for t in tasks}==expected
        result_root=event_out/'results'
        result_root.mkdir(parents=True,exist_ok=True)
        for name in ['rgb','logs']:(result_root/name).mkdir(exist_ok=True)
        regressor=result_root/'joint_regressor.npy'
        if not regressor.exists():
            temporary=regressor.with_name(regressor.name+'.'+str(os.getpid())+'.tmp')
            shutil.copyfile(source/'joint_regressor.npy',temporary);temporary.replace(regressor)
        return config,tasks,result_root,event_out
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==manifest['test_split_sha256']
    source=ROOT/'.experiment_storage/humanpose_gt_segments'
    tasks=json.loads((source/'tasks.json').read_text())
    assert len(tasks)==320 and sum(task['video_frames'] for task in tasks)==96124
    result_root=ROOT/'.experiment_storage/rgb_event_36h/test_results'
    result_root.mkdir(parents=True,exist_ok=True)
    for name in ['rgb','logs']:(result_root/name).mkdir(exist_ok=True)
    link=event_out/'results'
    try:link.symlink_to(result_root,target_is_directory=True)
    except FileExistsError:assert link.resolve()==result_root.resolve()
    regressor=result_root/'joint_regressor.npy'
    if not regressor.exists():
        temporary=regressor.with_name(regressor.name+'.'+str(os.getpid())+'.tmp')
        shutil.copyfile(source/'joint_regressor.npy',temporary);temporary.replace(regressor)
    return config,tasks,result_root,event_out


def event_worker(config_path,rank,workers):
    part_config=json.loads(Path(config_path).read_text())
    if part_config.get('experiment')=='part_aware_dynamic_aggregation_20h':
        from multihmr.part_aggregation_evaluation import evaluate
        return evaluate(part_config,rank,workers)
    if json.loads(Path(config_path).read_text()).get('experiment') in ('device34_body_28h','device34_fusion_repair_15h','device3_encoding_12h'):
        return multisensor_worker(config_path,rank,workers)
    from multihmr.event_inference import load_model,infer,CachedEvents
    config,tasks,result_root,event_out=prepare_event_test(config_path)
    base.OUT=result_root;base.configure_torch()
    checkpoint=None if config.get('original') else event_out/'checkpoints'/config['name']/'best.pt'
    loading_started=time.time()
    model=load_model(config,checkpoint)
    if config.get('experiment')=='rgb_event_new_split_24h':
        import torch
        save(result_root/config['name']/('execution_rank'+str(rank)+'.json'),dict(
            pid=os.getpid(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            physical_gpu=os.environ.get('EVENTPOSE_PHYSICAL_GPU'),cuda_name=torch.cuda.get_device_name(),
            checkpoint=str(checkpoint) if checkpoint else 'original multiHMR_896_L.pt',
            split_sha256=config['split_sha256'],loading_seconds=time.time()-loading_started,
            started_unix=loading_started,mode=config['mode'],encoding=config.get('encoding'),
            rgb_pixels_used=config['mode']!='event',GT_used_as_model_input=False))
    names=sorted({task['parent_session'] for task in tasks})[rank::workers]
    for name in names:
        clips=[task for task in tasks if task['parent_session']==name]
        if all((result_root/config['name']/task['session']/'complete.json').exists() for task in clips):continue
        cached=result_root/'rgb'/(config['name']+'_'+name+'.mp4')
        source=None
        try:
            if config['mode']!='event':shutil.copyfile(clips[0]['source_video'],cached)
            if config['mode']!='rgb':source=CachedEvents(name,config['encoding'])
            for raw in clips:
                if (result_root/config['name']/raw['session']/'complete.json').exists():continue
                # An event-only task deliberately has no usable RGB video input.
                task=dict(raw,video=str(cached) if config['mode']!='event' else '',
                          checkpoint=str(checkpoint) if checkpoint else 'original',input_mode=config['mode'],
                          event_encoding=config.get('encoding'),rgb_pixels_used=config['mode']!='event')
                collector=base.Collector(task,config['name'])
                try:
                    infer(model,collector,config.get('encoding'),config.get('batch_size',16),event_source=source)
                    collector.save()
                except Exception:
                    collector.save(partial=True);raise
        finally:
            if source is not None:source.close()
            cached.unlink(missing_ok=True)


def multisensor_worker(config_path,rank,workers):
    import torch
    from multihmr.multisensor_data import OUT as sensor_out
    from multihmr.multisensor_inference import load_model,CachedSensors,infer
    config=json.loads(Path(config_path).read_text())
    sensor_layout=None
    if config.get('architecture') in ('epipolar_repair_v1','epipolar_repair_norm_v2'):
        from multihmr.multisensor_repair import REPAIR_LAYOUTS,implementation_provenance
        sensor_layout=REPAIR_LAYOUTS[config['condition']]
        assert sensor_layout==(config['primary_sensor'],config['auxiliary_sensor'])
        assert int(os.environ.get('EVENTPOSE_PHYSICAL_GPU','-1')) in config['allowed_gpus']
        assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
    assert sensor_out.resolve()==Path(config['output_root']).resolve()
    assert hashlib.sha256((ROOT/'dataset/split.json').read_bytes()).hexdigest()==config['split_sha256']
    tasks=json.loads((sensor_out/'test/tasks.json').read_text())
    result_root=sensor_out/'results';result_root.mkdir(exist_ok=True)
    if not (result_root/'joint_regressor.npy').exists():
        tmp=result_root/('regressor_'+str(os.getpid())+'.tmp')
        shutil.copyfile(sensor_out/'joint_regressor.npy',tmp);tmp.replace(result_root/'joint_regressor.npy')
    base.OUT=result_root;base.configure_torch()
    checkpoint=sensor_out/'checkpoints'/config['name']/'best.pt'
    started=time.time();model=load_model(config,checkpoint)
    provenance=dict(pid=os.getpid(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        physical_gpu=os.environ.get('EVENTPOSE_PHYSICAL_GPU'),cuda_name=torch.cuda.get_device_name(),
        checkpoint=str(checkpoint),split_sha256=config['split_sha256'],
        condition=config['condition'],loading_seconds=time.time()-started,started_unix=started,
        implementation_sha256=implementation_provenance(sensor_out/'source_snapshots') if sensor_layout is not None else None,
        GT_used_as_model_input=False,pixel_datasets_read=[])
    for name in sorted({t['parent_session'] for t in tasks})[rank::workers]:
        source=CachedSensors(name,config['condition'],sensor_layout=sensor_layout,encoding=config.get('encoding','count33'))
        try:
            for raw in [t for t in tasks if t['parent_session']==name]:
                if (result_root/config['name']/raw['session']/'complete.json').exists():continue
                task=dict(raw,checkpoint=str(checkpoint),condition=config['condition'],split_sha256=config['split_sha256'])
                collector=base.Collector(task,config['name'])
                try:infer(model,collector,source,config['batch_size']);collector.save()
                except Exception:collector.save(partial=True);raise
            provenance['pixel_datasets_read']=sorted(set(provenance['pixel_datasets_read'])|source.pixel_datasets_read)
            save(result_root/config['name']/('execution_rank'+str(rank)+'.json'),provenance)
        finally:source.close()
    if config['condition']=='event3_event4':assert not any(s.startswith('rgb') for s in provenance['pixel_datasets_read'])


def event_supervise(config_path,gpus):
    config,tasks,result_root,event_out=prepare_event_test(config_path)
    jobs=[]
    for rank,gpu in enumerate(gpus):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(result_root/'logs'/(config['name']+'_'+str(rank)+'.log')).open('a',buffering=1)
        process=subprocess.Popen([sys.executable,'-u',__file__,'worker','--event-config',str(config_path),
            '--rank',str(rank),'--workers',str(len(gpus))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((process,log))
    status_path=event_out/'test_status'/(config['name']+'.json')
    def status(state):
        return dict(status=state,completed_segments=sum((result_root/config['name']/task['session']/'complete.json').exists() for task in tasks),
                    total_segments=320,updated_unix=time.time(),gpus=list(gpus),
                    workers=[dict(pid=process.pid,exit_code=process.poll()) for process,_ in jobs])
    while any(process.poll() is None for process,_ in jobs):
        save(status_path,status('running'));time.sleep(15)
    for _,log in jobs:log.close()
    if any(process.returncode!=0 for process,_ in jobs):
        save(status_path,status('failed'));raise RuntimeError('Event comparison Test worker failed; inspect logs')
    assert all((result_root/config['name']/task['session']/'complete.json').exists() for task in tasks)
    save(status_path,status('complete'))


def track2_infer(trial,sample,limit=0,batch_size=8,checkpoint_path=None,input_view='target_foreground',original_weights=False,depth_decoder='published',fixed_view_plan_path=None,strict_target_mask=False,trace_frames=None,target_body_overlap=False):
    """Target-mask crop RGB inference; reject detections outside the target."""
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image,ImageOps
    from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
    from multihmr.target_crop import PROTOCOL,prepare_target_crop,target_acceptance,target_head_acceptance
    if input_view=='native_rgb':PROTOCOL='native_rgb_target_head_v1'
    if input_view=='native_rgb_static_crop':PROTOCOL='native_rgb_static_crop_target_head_v1'
    if strict_target_mask:
        from multihmr.target_crop import STRICT_PROTOCOLS
        assert input_view in STRICT_PROTOCOLS and depth_decoder=='published'
        PROTOCOL=STRICT_PROTOCOLS[input_view]
    body_rule=None
    if target_body_overlap:
        from multihmr.target_crop import BODY_PROTOCOLS,target_body_box_iou
        assert strict_target_mask
        body_rule=json.loads((OUTPUT/'body_target_gate_Train_rule.json').read_text())
        PROTOCOL=BODY_PROTOCOLS[input_view]
    assert input_view in ['target_foreground','native_rgb','native_rgb_static_crop']
    assert not (original_weights and checkpoint_path is not None)
    base.configure_torch()
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    assert depth_decoder in ['published','exp_then_focal']
    if depth_decoder!='published':
        assert original_weights and checkpoint_path is None and input_view=='native_rgb'
        assert sample in manifest['validation'] and sample not in manifest['test']
        assert trial=='diagnostic_original_depth_order'
        PROTOCOL='native_rgb_exp_then_focal_target_head_v1'
    record=manifest['samples'][sample]
    static_box=None;static_plan_path=None
    if input_view=='native_rgb_static_crop':
        static_plan_path=Path(fixed_view_plan_path) if fixed_view_plan_path else OUTPUT/'static_crop_Train_control_plan.json'
        static=json.loads(static_plan_path.read_text())
        if fixed_view_plan_path:assert static['policy']=='fixed_mask_union_view_v1'
        else:assert sample in manifest['validation']
        assert sample in static['samples'];static_box=static['samples'][sample]['box']
    else:assert fixed_view_plan_path is None
    folder=OUTPUT/('smoke_predictions' if limit else 'predictions')/trial/sample
    folder.mkdir(parents=True,exist_ok=True)
    result_file=folder/'smplx_predictions.npz'
    if result_file.exists() and (folder/'inference.json').exists():
        cached=json.loads((folder/'inference.json').read_text())
        if cached.get('evaluation_protocol')!=PROTOCOL:raise RuntimeError('Old unmasked prediction must be archived before target inference')
        if static_box:assert cached['static_crop_box']==static_box,'Cached fixed viewport differs'
        if checkpoint_path is not None:assert cached['checkpoint_sha256']==sha256(checkpoint_path),'Cached checkpoint differs'
        if original_weights:assert cached['delta_checkpoint'] is None,'Cached inference used adapted weights'
        print('Verified completed target inference',sample,trial,flush=True);return
    model=base.load_multihmr()
    if depth_decoder=='exp_then_focal':
        # Train-only counterfactual; the published model implementation and all
        # normal inference/training keep their original decoding order.
        import types
        from multihmr.utils.camera import undo_focal_length_normalization,undo_log_depth
        assert model.nearness
        def experimental_depth(self,x,dist,K):
            dist=undo_focal_length_normalization(undo_log_depth(dist),K[:,[0],[0]],fovn=self.fovn,img_size=x.shape[-1])
            return torch.clamp(dist,0,50) if self.clip_dist else dist
        model.to_euclidean_dist=types.MethodType(experimental_depth,model)
    checkpoint=None
    if checkpoint_path is not None or (trial!='original' and not original_weights):
        evaluation_checkpoints=OUTPUT/'evaluation_checkpoints.json'
        mapping=json.loads(evaluation_checkpoints.read_text()) if evaluation_checkpoints.exists() else {}
        checkpoint=Path(checkpoint_path) if checkpoint_path is not None else Path(mapping[trial]['checkpoint']) if trial in mapping else OUTPUT/'checkpoints'/trial/'best.pt'
        delta=torch.load(checkpoint,map_location='cpu')
        loaded=model.load_state_dict(delta['model_delta'],strict=False)
        assert not loaded.unexpected_keys
    model.eval()
    image_ids=[];original_detection=model.detection
    def detection(*args,**kwargs):
        value=original_detection(*args,**kwargs)
        image_ids[:]=value[2][0].cpu().tolist()
        return value
    model.detection=detection
    norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    K_original=np.array(record['K'],np.float32)
    count=min(record['frames'],limit) if limit else record['frames']
    vertices=np.full((count,10475,3),np.nan,np.float32)
    joints=np.full((count,55,3),np.nan,np.float32)
    rotvec=np.full((count,53,3),np.nan,np.float32)
    shape=np.full((count,model.num_betas),np.nan,np.float32)
    confidence=np.zeros(count,np.float32);detections=np.zeros(count,np.int16)
    valid=np.zeros(count,bool);previous=None
    head_queries=np.full((count,2),np.nan,np.float64);head_pixels=np.full((count,2),-1,np.int32);traces={};trace_frames=set(trace_frames or [])
    overlap=np.zeros(count,np.float32);torso_overlap=np.zeros(count,np.float32)
    body_box_overlap=np.zeros(count,np.float64)
    crop_boxes=np.zeros((count,4),np.int32);mask_valid=np.zeros(count,bool)
    reg=model.smpl_layer[f'neutral_{model.num_betas}'].bm_x.J_regressor
    cap=cv2.VideoCapture(str(SOURCE/sample/'video.mp4'));assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==record['frames'],'RGB/GT timeline mismatch'
    assert abs(cap.get(cv2.CAP_PROP_FPS)-record['fps'])<.01,'RGB/GT FPS mismatch'
    mask_path=SOURCE/sample/'processed/human_mask.mp4'
    from multihmr.track2_data import open_track2_target_mask
    mask_cap=open_track2_target_mask(sample)
    assert int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))==record['frames'],'Target-mask timeline mismatch'
    assert abs(mask_cap.get(cv2.CAP_PROP_FPS)-record['fps'])<.01,'Target-mask FPS mismatch'
    started=time.time()
    deadline=track2_evaluation_deadline()
    try:
        for start in range(0,count,batch_size):
            if time.time()>=deadline-30:raise TimeoutError('12-hour deadline during RGB inference')
            images=[];cameras=[];supports=[];crop_infos=[]
            stop=min(count,start+batch_size)
            for frame in range(start,stop):
                ok,bgr=cap.read();assert ok,(sample,frame)
                ok_mask,mask=mask_cap.read();assert ok_mask,(sample,frame,'missing target mask')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
                assert (w,h)==(record['width'],record['height'])
                image,K,support,info=prepare_target_crop(rgb,mask,K_original,strict_target_mask=strict_target_mask)
                if input_view=='native_rgb':
                    # Preserve the original RGB context. The mask only gates
                    # output detections; resize, padding, K and head gate match.
                    native=ImageOps.contain(Image.fromarray(rgb),(896,896),method=Image.Resampling.LANCZOS)
                    nw,nh=native.size;left,top=info['padding']
                    image=ImageOps.expand(native,border=(left,top,896-nw-left,896-nh-top))
                elif input_view=='native_rgb_static_crop':
                    from multihmr.target_crop import prepare_native_rgb_box
                    valid_mask=info['mask_valid']
                    image,K,info=prepare_native_rgb_box(rgb,K_original,static_box)
                    info['mask_valid']=valid_mask
                crop_boxes[frame]=info['box'];mask_valid[frame]=info['mask_valid']
                cameras.append(K);images.append(norm(image));supports.append(support);crop_infos.append(info)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                people=model(torch.stack(images).cuda(),K=torch.as_tensor(np.stack(cameras)).cuda(),
                    is_training=False,det_thresh=.3,nms_kernel_size=3)
            grouped=[[] for _ in images]
            for i,person in zip(image_ids,people):grouped[i].append(person)
            for offset,persons in enumerate(grouped):
                frame=start+offset;detections[frame]=len(persons)
                candidates=[]
                trace=[]
                for person_index,person in enumerate(persons):
                    v=person['v3d'].float();j=(reg@v).cpu().numpy()
                    projected=j[:22]@K_original.T
                    uv=projected[:,:2]/projected[:,2:]
                    area=float(np.maximum(uv.max(0)-uv.min(0),1).prod())
                    if not np.isfinite(j).all() or j[0,2]<=0:continue
                    _,mask_overlap,torso=target_acceptance(j,K_original,supports[offset])
                    head_inside,head_xy=target_head_acceptance(person['loc'].detach().float().cpu().numpy(),crop_infos[offset],supports[offset])
                    box_iou=target_body_box_iou(j,K_original,supports[offset]) if target_body_overlap else 0.
                    if frame in trace_frames:trace.append(dict(candidate=person_index,head_query_native_xy=head_xy.tolist(),score=float(person['scores']),inside_selected_mask=bool(head_inside),pelvis_Z_m=float(j[0,2]),body_box_iou=box_iou))
                    if not mask_valid[frame] or not head_inside:continue
                    if target_body_overlap and box_iou<body_rule['minimum_body_box_iou']:continue
                    rank=(box_iou,float(person['scores'])) if target_body_overlap else float(person['scores'])
                    candidates.append((rank,person,j,mask_overlap,torso,head_xy,person_index))
                if not candidates:
                    if frame in trace_frames:traces[str(frame)]=dict(candidates=trace,selected_candidate=None)
                    continue
                _,person,j,mask_overlap,torso,head_xy,person_index=max(candidates,key=lambda x:x[0])
                head_queries[frame]=head_xy
                head_pixels[frame]=np.rint(head_xy).astype(np.int32)
                if frame in trace_frames:traces[str(frame)]=dict(candidates=trace,selected_candidate=person_index)
                overlap[frame]=mask_overlap;torso_overlap[frame]=torso
                if target_body_overlap:body_box_overlap[frame]=target_body_box_iou(j,K_original,supports[offset])
                vertices[frame]=person['v3d'].float().cpu().numpy();joints[frame]=j
                rotvec[frame]=person['rotvec'].float().cpu().numpy()
                shape[frame]=person['shape'].float().cpu().numpy()
                confidence[frame]=float(person['scores']);valid[frame]=True;previous=j[0]
            if stop%128==0 or stop==count:
                print(json.dumps(dict(event='track2_inference',trial=trial,sample=sample,
                    frames=stop,total=count,elapsed_seconds=time.time()-started)),flush=True)
    finally:
        cap.release();mask_cap.release();model.detection=original_detection
    observed=np.flatnonzero(valid);assert len(observed),'No RGB person detected in sequence'
    # Nearest observed RGB prediction fills misses. Never use GT to fill a frame.
    for frame in np.flatnonzero(~valid):
        nearest=observed[np.abs(observed-frame).argmin()]
        for values in [vertices,joints,rotvec,shape]:values[frame]=values[nearest]
    temporary=result_file.with_suffix('.partial.npz')
    np.savez_compressed(temporary,vertices_cam_m=vertices,joints_cam_m=joints,
        rotvec=rotvec,shape=shape,detection_valid=valid,detection_counts=detections,
        confidence=confidence,frames=np.arange(count),K=K_original,crop_boxes=crop_boxes,target_mask_valid=mask_valid,
        target_joint_mask_overlap=overlap,target_torso_mask_overlap=torso_overlap,head_query_native_xy=head_queries,head_query_native_pixel=head_pixels,
        target_body_box_iou=body_box_overlap)
    temporary.replace(result_file)
    save_json(folder/'inference.json',dict(trial=trial,sample=sample,frames=count,
        started_unix=started,completed_unix=time.time(),
        elapsed_seconds=time.time()-started,detection_frame_rate=float(valid.mean()),
        imputed_frames=np.flatnonzero(~valid).tolist(),gt_used_for_detection_or_tracking=False,
        tracking=('Non-target RGB replaced with neutral background; ' if input_view=='target_foreground' else 'Original RGB retained; mask used only to select output target; ')+
            ('head query inside original mask and Train-frozen projected body-box IoU required; highest body-box IoU then confidence; no fallback' if target_body_overlap else 'head-query inside target mask required; highest detection confidence; no background-person fallback'),
        target_foreground_only_input=input_view=='target_foreground',input_view=input_view,depth_decoder=depth_decoder,all_accepted_heads_in_target_mask=True,background_rgb=[124,116,104] if input_view=='target_foreground' else None,
        evaluation_protocol=PROTOCOL,target_mask_path=str(mask_path),target_mask_sha256=sha256(mask_path),
        strict_target_mask=bool(strict_target_mask),target_mask_dilation_kernel=0 if strict_target_mask else 15,
        target_body_overlap=bool(target_body_overlap),body_box_min_iou=body_rule['minimum_body_box_iou'] if body_rule else None,
        body_box_rule_sha256=sha256(OUTPUT/'body_target_gate_Train_rule.json') if body_rule else None,
        crop_margin=None if static_box else .2,static_crop_box=static_box,
        static_crop_plan_sha256=sha256(static_plan_path) if static_box else None,
        view_protocol=('static crop with corrected K; original RGB retained within crop' if static_box else 'native image geometry/K; '+('only target RGB pixels retained' if input_view=='target_foreground' else 'full original RGB retained')),target_mask_frame_rate=float(mask_valid.mean()),
        minimum_accepted_joint_overlap=float(overlap[valid].min()),minimum_accepted_torso_overlap=float(torso_overlap[valid].min()),
        source_of_target_identity='provided processed/human_mask.mp4; no GT3D pose or identity parameters read',
        original_checkpoint=str(ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'),
        delta_checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        result_sha256=sha256(result_file),full_original_frame_rate=not bool(limit)))
    if trace_frames:save_json(folder/'detection_trace.json',dict(frames=traces,evaluation_protocol=PROTOCOL,GT_read=False))


def track2_convert(trial,sample,limit=0):
    import numpy as np
    import smplx
    from multihmr.track2_data import OUTPUT,ASSETS,save_json
    from utils.mhr_conversion import fit_prediction_sequence
    base.configure_torch()
    folder=OUTPUT/('smoke_predictions' if limit else 'predictions')/trial/sample
    target=folder/'human_params_mhr.npz'
    if target.exists() and (folder/'conversion.json').exists():
        if not limit:track2_refine(trial,sample)
        return
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    with np.load(folder/'smplx_predictions.npz') as z:
        vertices=z['vertices_cam_m']
    body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,
        flat_hand_mean=True,num_betas=10)
    report=fit_prediction_sequence(vertices,body.faces,manifest['samples'][sample]['world_to_camera'],
        target,track2_evaluation_deadline())
    report['evaluation_protocol']=json.loads((folder/'inference.json').read_text())['evaluation_protocol']
    save_json(folder/'conversion.json',report)
    if not limit:track2_refine(trial,sample)


def track2_refine(trial,sample):
    """Prediction-only revision2; keep version1 predictions and scores intact."""
    import numpy as np
    import smplx
    from multihmr.track2_data import OUTPUT,ASSETS,save_json,sha256
    from utils.mhr_conversion import refine_prediction_sequence
    base.configure_torch()
    folder=OUTPUT/'predictions'/trial/sample
    initial=folder/'human_params_mhr.npz';source=folder/'smplx_predictions.npz'
    target=folder/'human_params_mhr_v2.npz';receipt=folder/'conversion_v2.json'
    if target.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if old['initial_sha256']==sha256(initial) and old['source_sha256']==sha256(source) and old.get('output_sha256')==sha256(target):return
        raise RuntimeError('Revision2 source changed; archive its outputs before recomputing')
    with np.load(source) as z:vertices=z['vertices_cam_m']
    body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
    report=refine_prediction_sequence(vertices,body.faces,initial,target,
        track2_evaluation_deadline())
    report['output_sha256']=sha256(target)
    report['evaluation_protocol']=json.loads((folder/'inference.json').read_text())['evaluation_protocol']
    save_json(receipt,report)


def track2_score(trial,sample,conversion_revision=2):
    import numpy as np
    from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,save_json,sha256
    from utils.mhr_evaluation_body import LocalMHRBody
    from utils.track2_metrics import score_human,protocol
    from utils.v2d_official_metric import MHR_TABLE3_BODY_JOINT_INDICES
    base.configure_torch()
    folder=OUTPUT/'predictions'/trial/sample
    baseline_roots={'init_v6':SOURCE,'init_v0':ROOT/'exp2/hj/challenge_track2_v0'}
    metrics_file=folder/('metrics_v2.json' if conversion_revision==2 else 'metrics.json')
    prediction_name='human_params_mhr_v2.npz' if conversion_revision==2 else 'human_params_mhr.npz'
    pred_path=baseline_roots[trial]/sample/'human/human_params_init.npz' if trial in baseline_roots else folder/prediction_name
    gt_path=SOURCE/sample/'human/human_params_gt.npz'
    if metrics_file.exists():
        previous=json.loads(metrics_file.read_text())
        if previous['prediction_sha256']==sha256(pred_path) and previous['gt_sha256']==sha256(gt_path):return
    with np.load(pred_path) as z:pred={k:z[k] for k in ['pose','scales','shape']}
    with np.load(gt_path) as z:gt={k:z[k] for k in ['pose','scales','shape']}
    body=LocalMHRBody(ASSETS/'mhr/mhr_model.pt',MHR_TABLE3_BODY_JOINT_INDICES)
    values=score_human(body,pred,gt)
    values.update(trial=trial,sample=sample,prediction_sha256=sha256(pred_path),
        prediction_path=str(pred_path),gt_sha256=sha256(gt_path),protocol=protocol(body))
    values['conversion_revision']=conversion_revision
    values['evaluation_protocol']=json.loads((folder/'inference.json').read_text())['evaluation_protocol'] if trial not in baseline_roots else 'target_foreground_head_v1'
    save_json(metrics_file,values)
    if conversion_revision==2:save_json(folder/'metrics.json',values)
    print(json.dumps({k:v for k,v in values.items() if k!='protocol'}),flush=True)


def track2_validate_smplx(checkpoint_path,tag,batch_size=8,input_view='target_foreground',native_surface_target=False,strict_target_mask=False,target_body_overlap=False):
    """CD-H proxy on held-out fitted SMPL-X meshes; no MHR parameter fitting.

    Map SMPL-X surfaces to the reference body's fixed FPS27/FPS64 points using
    official barycentric correspondence. Keep one frame-0 Sim(3) per sequence.
    This is explicitly a SMPL-X supervision-space proxy, not final MHR CD-H.
    """
    import numpy as np
    import torch
    import roma
    import cv2
    from torch.utils.data import DataLoader
    import scripts.train_multihmr as training
    from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,mapping,transfer,save_json,sha256
    from multihmr.target_crop import target_head_acceptance
    from utils.mhr_evaluation_body import LocalMHRBody
    from utils import v2d_official_metric as official
    from utils.humanpose_metrics import match_people
    assert input_view in ['target_foreground','native_rgb','native_rgb_static_crop']
    validation_protocol='native_rgb_static_crop_target_head_v1' if input_view=='native_rgb_static_crop' else 'target_foreground_head_v1' if input_view=='target_foreground' else 'native_rgb_target_head_v1'
    if strict_target_mask:
        from multihmr.target_crop import STRICT_PROTOCOLS
        validation_protocol=STRICT_PROTOCOLS[input_view]
    body_rule=None
    if target_body_overlap:
        from multihmr.target_crop import BODY_PROTOCOLS,target_body_box_iou
        assert strict_target_mask
        body_rule=json.loads((OUTPUT/'body_target_gate_Train_rule.json').read_text())
        validation_protocol=BODY_PROTOCOLS[input_view]
    proxy_target_space='native_MHR_GT' if native_surface_target else 'fitted_SMPLX_GT'
    destination=OUTPUT/'validation_smplx'/tag/'metrics.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        expected_sha=sha256(checkpoint_path) if checkpoint_path else None
        if (previous['checkpoint_sha256']==expected_sha
            and previous.get('evaluation_protocol')==validation_protocol
            and previous.get('cd_proxy_target_space','fitted_SMPLX_GT')==proxy_target_space
            and (not strict_target_mask or (previous.get('strict_target_mask') is True
                and previous.get('target_mask_dilation_kernel')==0))
            and (not target_body_overlap or (previous.get('target_body_overlap') is True
                and previous.get('body_box_rule_sha256')==sha256(OUTPUT/'body_target_gate_Train_rule.json')))
            and previous.get('detection_selection')=='target_mask_head_highest_confidence'):return
        archive=destination.with_name('metrics_previous_'+str(previous['evaluated_unix']).replace('.','_')+'.json')
        destination.replace(archive)
    base.configure_torch();training.WORK=OUTPUT/'work'
    dataset=training.Frames('validation',target_foreground=input_view=='target_foreground',
        fixed_view_plan=OUTPUT/'fixed_view_training_plan.json' if input_view=='native_rgb_static_crop' else None,native_surface_target=native_surface_target)
    loader=DataLoader(dataset,batch_size=batch_size,collate_fn=training.collate,num_workers=2,pin_memory=True)
    model=base.load_multihmr()
    step=0
    if checkpoint_path:
        checkpoint_path=Path(checkpoint_path)
        value=torch.load(checkpoint_path,map_location='cpu');step=value['step']
        result=model.load_state_dict(value['model_delta'],strict=False);assert not result.unexpected_keys
        del value
    model.eval();body=model.smpl_layer[f'neutral_{model.num_betas}'].bm_x
    reference=LocalMHRBody(ASSETS/'mhr/mhr_model.pt',official.MHR_TABLE3_BODY_JOINT_INDICES)
    selected=np.concatenate([reference.roles['alignment'],reference.roles['body']])
    ids,weights=mapping('smplx2mhr',body.faces,'cuda');ids=ids[selected];weights=weights[selected]
    del reference
    image_ids=[];original=model.detection
    def detection(*args,**kwargs):
        value=original(*args,**kwargs);image_ids[:]=value[2][0].cpu().tolist();return value
    model.detection=detection
    sequences={};mask_readers={};matched=0;total=0;max_reconstruction_error=0.;started=time.time()
    deadline=json.loads((OUTPUT/'budget.json').read_text())['deadline_unix']
    try:
        for batch in loader:
            if time.time()>=deadline-30:raise TimeoutError('12-hour deadline during SMPL-X validation')
            n=len(batch['index']);rotation=roma.rotmat_to_rotvec(batch['rotations'][:,0].cuda())
            zeros=lambda width:torch.zeros(n,width,device='cuda')
            with torch.no_grad():
                gt=body(global_orient=rotation[:,0],body_pose=rotation[:,1:22].flatten(1),
                    left_hand_pose=rotation[:,22:37].flatten(1),right_hand_pose=rotation[:,37:52].flatten(1),
                    jaw_pose=rotation[:,52],betas=batch['betas'][:,0].cuda(),
                    transl=batch['smplx_translation'][:,0].cuda(),expression=zeros(10),
                    leye_pose=zeros(3),reye_pose=zeros(3)).vertices
                discrepancy=((body.J_regressor@gt).cpu()-batch['joints'][:,0]).abs().max().item()
                max_reconstruction_error=max(max_reconstruction_error,discrepancy)
                assert discrepancy<2e-4,('GT SMPL-X reconstruction mismatch',discrepancy)
                gt_points=transfer(gt,ids,weights).cpu().numpy()
                if native_surface_target:gt_points=batch['native_surface_points'][:,0].numpy()
                with torch.autocast('cuda',dtype=torch.float16):
                    people=model(batch['image'].cuda(),K=batch['K'].cuda(),is_training=False,
                        det_thresh=.3,nms_kernel_size=3)
                grouped=[[] for _ in range(n)]
                for image_id,person in zip(image_ids,people):grouped[image_id].append(person)
                for i,persons in enumerate(grouped):
                    record=dataset.records[batch['index'][i]];name=record['session']
                    block=sequences.setdefault(name,dict(frames=[],pred=[],target=[],expected=0))
                    block['expected']+=1;total+=1
                    if name not in mask_readers:
                        from multihmr.track2_data import open_track2_target_mask
                        mask_readers[name]=open_track2_target_mask(name)
                    cap=mask_readers[name];cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame'])
                    ok,mask=cap.read();assert ok,(name,record['frame'])
                    height,width=batch['image_shape'][i].tolist()
                    gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                    if gray.shape!=(height,width):gray=cv2.resize(gray,(width,height),interpolation=cv2.INTER_NEAREST)
                    support=gray>127 if strict_target_mask else cv2.dilate((gray>127).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))>0
                    native=batch['original_K'][i].numpy();camera=batch['K'][i].numpy()
                    sx,sy=camera[0,0]/native[0,0],camera[1,1]/native[1,1]
                    info=dict(box=[0,0,width,height],scale=[sx,sy],
                        padding=[camera[0,2]-sx*native[0,2],camera[1,2]-sy*native[1,2]])
                    accepted=[p for p in persons if target_head_acceptance(p['loc'].detach().float().cpu().numpy(),info,support)[0]]
                    if not accepted:continue
                    if target_body_overlap:
                        ranked=[]
                        for person in accepted:
                            joints=(body.J_regressor@person['v3d'].float()).cpu().numpy()
                            iou=target_body_box_iou(joints,native,support)
                            if iou>=body_rule['minimum_body_box_iou']:ranked.append((iou,float(person['scores']),person))
                        if not ranked:continue
                        chosen=max(ranked,key=lambda row:row[:2])[2]
                    else:chosen=max(accepted,key=lambda p:float(p['scores']))
                    points=transfer(chosen['v3d'].float()[None],ids,weights)[0].cpu().numpy()
                    block['frames'].append(record['frame']);block['pred'].append(points);block['target'].append(gt_points[i]);matched+=1
    finally:
        model.detection=original
        for cap in mask_readers.values():cap.release()
    per_sequence={}
    for name,block in sequences.items():
        assert block['frames'] and block['frames'][0]==0,('Original frame 0 must be matched',name)
        pred=np.asarray(block['pred'],np.float64);target=np.asarray(block['target'],np.float64)
        transform=official.fit_similarity_transform(pred[0,:27],target[0,:27])
        aligned=official.apply_similarity_transform(pred[:,27:],transform)
        cd=float(official.symmetric_chamfer_distance(aligned,target[:,27:]).mean()*100)
        per_sequence[name]=dict(cd_h_smplx_proxy_cm=cd,matched_frames=len(pred),expected_frames=block['expected'],
            coverage=len(pred)/block['expected'],frames=block['frames'])
    value=dict(cd_h_smplx_proxy_cm=float(np.mean([v['cd_h_smplx_proxy_cm'] for v in per_sequence.values()])),
        coverage=matched/total,matched_instances=matched,valid_instances=total,step=step,
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_sha256=sha256(checkpoint_path) if checkpoint_path else None,
        per_sequence=per_sequence,max_gt_reconstruction_error_m=max_reconstruction_error,
        elapsed_seconds=time.time()-started,evaluated_unix=time.time(),
        metric='Mapped SMPL-X prediction surface CD-H proxy, target space '+proxy_target_space+', official Meta topology mapping, FPS27 frame0 fixed Sim(3), FPS64 body, equal sequence weights, cm',
        cd_proxy_target_space=proxy_target_space,
        not_final_mhr_cd_h=True,test_used=False,evaluation_protocol=validation_protocol,target_foreground_only_input=input_view=='target_foreground',input_view=input_view,
        detection_selection='target_mask_head_highest_confidence',GT_used_for_detection_selection=False,
        unmatched_frames_excluded_from_proxy=True)
    if strict_target_mask:value.update(strict_target_mask=True,target_mask_dilation_kernel=0)
    if target_body_overlap:value.update(target_body_overlap=True,body_box_min_iou=body_rule['minimum_body_box_iou'],body_box_rule_sha256=sha256(OUTPUT/'body_target_gate_Train_rule.json'))
    save_json(destination,value);print(json.dumps({k:v for k,v in value.items() if k!='per_sequence'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['supervise','worker','track2-infer','track2-convert','track2-score','track2-validate','track2-refine']);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--event-config',type=Path);p.add_argument('--workers',type=int,default=5)
    p.add_argument('--gpus',default='2,3,4,5,6')
    p.add_argument('--trial',default='original');p.add_argument('--sample');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--conversion-revision',type=int,choices=[1,2],default=2)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--validation-tag',default='original')
    p.add_argument('--input-view',choices=['target_foreground','native_rgb','native_rgb_static_crop'],default='target_foreground')
    p.add_argument('--original-weights',action='store_true')
    p.add_argument('--depth-decoder',choices=['published','exp_then_focal'],default='published')
    p.add_argument('--fixed-view-plan',type=Path)
    p.add_argument('--native-surface-target',action='store_true')
    p.add_argument('--strict-target-mask',action='store_true')
    p.add_argument('--target-body-overlap',action='store_true')
    p.add_argument('--trace-frames',nargs='*',type=int)
    a=p.parse_args()
    if a.stage.startswith('track2-'):
        if a.native_surface_target:assert a.stage=='track2-validate','Native GT surfaces are Train validation targets only'
        assert a.sample or a.stage=='track2-validate','--sample is required for Track 2'
        import fcntl
        from multihmr.track2_data import OUTPUT
        locks=OUTPUT/'locks';locks.mkdir(parents=True,exist_ok=True)
        lock_values=[a.stage,a.trial,a.sample,a.limit,a.validation_tag,'target_foreground_head_v1']
        if a.stage=='track2-infer' and (a.input_view!='target_foreground' or a.original_weights):
            lock_values.extend([a.input_view,a.original_weights])
        if a.depth_decoder!='published':lock_values.append(a.depth_decoder)
        if a.fixed_view_plan:lock_values.append(str(a.fixed_view_plan.resolve()))
        if a.native_surface_target:lock_values.append('native_MHR_GT')
        if a.strict_target_mask:lock_values.append('strict_original_target_mask_v2')
        if a.target_body_overlap:lock_values.append('strict_target_body_box_v3')
        if a.conversion_revision==2:lock_values.append(2)
        lock_key=hashlib.sha256(json.dumps(lock_values).encode()).hexdigest()
        with (locks/(lock_key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if a.stage=='track2-validate':track2_validate_smplx(a.checkpoint,a.validation_tag,a.batch_size,a.input_view,a.native_surface_target,a.strict_target_mask,a.target_body_overlap)
            elif a.stage=='track2-infer':track2_infer(a.trial,a.sample,a.limit,a.batch_size,a.checkpoint,a.input_view,a.original_weights,a.depth_decoder,a.fixed_view_plan,a.strict_target_mask,a.trace_frames,a.target_body_overlap)
            elif a.stage=='track2-convert':track2_convert(a.trial,a.sample,a.limit)
            elif a.stage=='track2-refine':
                assert a.conversion_revision==2 and not a.limit
                track2_refine(a.trial,a.sample)
            else:track2_score(a.trial,a.sample,a.conversion_revision)
        raise SystemExit(0)
    gpus=tuple(int(value) for value in a.gpus.split(','))
    assert gpus and len(set(gpus))==len(gpus) and all(gpu in range(2,7) for gpu in gpus)
    if a.event_config:
        if a.stage=='supervise':event_supervise(a.event_config,gpus)
        else:event_worker(a.event_config,a.rank,a.workers)
    elif a.stage=='supervise':supervise(gpus)
    else:worker(a.rank,a.workers)

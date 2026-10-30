"""Fixed, mask-union Train-only viewport control on a genuinely free GPU pair."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue,Empty
import fcntl,json,math,os,signal,threading,time
import cv2,numpy as np
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    import pynvml
    with (OUTPUT/'locks/static_crop_Train_control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME='static_crop_Train_control_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');resources=read(OUTPUT/'gpu_resources.json');cv2.setNumThreads(1)
        plan_path=OUTPUT/'static_crop_Train_control_plan.json'
        if plan_path.exists():plan=read(plan_path)
        else:
            specs={}
            for sample in manifest['validation']:
                r=manifest['samples'][sample];cap=cv2.VideoCapture(str(SOURCE/sample/'processed/human_mask.mp4'))
                assert cap.isOpened();bounds=np.array([r['width'],r['height'],0,0]);frames=0
                while True:
                    ok,mask=cap.read()
                    if not ok:break
                    yy,xx=np.where(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)>127)
                    if len(xx):bounds=np.array([min(bounds[0],xx.min()),min(bounds[1],yy.min()),max(bounds[2],xx.max()+1),max(bounds[3],yy.max()+1)])
                    frames+=1
                cap.release();assert frames==r['frames']
                desired=2*r['K'][0][0]*math.tan(math.radians(30))
                w=min(r['width'],math.ceil(max(desired,(bounds[2]-bounds[0])*1.1)))
                h=min(r['height'],math.ceil(max(desired,(bounds[3]-bounds[1])*1.1)))
                x=max(0,min(r['width']-w,math.floor((bounds[0]+bounds[2]-w)/2)))
                y=max(0,min(r['height']-h,math.floor((bounds[1]+bounds[3]-h)/2)))
                box=[int(x),int(y),int(x+w),int(y+h)]
                assert x<=bounds[0] and y<=bounds[1] and x+w>=bounds[2] and y+h>=bounds[3]
                specs[sample]=dict(box=box,mask_union=bounds.tolist(),frames=frames,
                    mask_sha256=sha256(SOURCE/sample/'processed/human_mask.mp4'),GT_used=False)
            plan=dict(samples=specs,registered_unix=time.time(),test_used=False,
                policy='One fixed RGB viewport per Train clip; mask union with margin and 60deg reference size; no pixel erasure, no dynamic zoom; corrected K',
                original_RGB_baseline_and_training_unchanged=True)
            save_json(plan_path,plan)
        results={};mutex=threading.Lock();trial='diagnostic_original_static_crop'
        deadline=read(OUTPUT/'budget.json')['deadline_unix']
        def status(value,**extra):
            with mutex:save_json(OUTPUT/'static_crop_Train_control_status.json',dict(status=value,pid=os.getpid(),
                trial=trial,completed=list(results),test_used=False,updated_unix=time.time(),**extra))
        status('waiting_for_free_training_pair')
        pynvml.nvmlInit();leases=[];pair=None
        while time.time()<deadline-2*3600 and pair is None:
            for candidate in resources['training_pairs']:
                path=OUTPUT/'locks'/('training_pair_'+'_'.join(map(str,candidate))+'.lock');lease=path.open('a')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lease.close();continue
                busy=any(pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(g)])) for g in candidate)
                if busy:lease.close();continue
                pair=candidate;leases.append(lease);break
            if pair is None:time.sleep(20)
        assert pair is not None,'No free GPU pair before control deadline'
        status('running_fixed_Train_viewport_control',gpus=pair)
        tasks=Queue()
        for sample in sorted(plan['samples'],key=lambda s:manifest['samples'][s]['frames'],reverse=True):tasks.put(sample)
        def worker(gpu):
            while True:
                try:sample=tasks.get_nowait()
                except Empty:return
                for stage in ['track2-infer','track2-convert','track2-score']:
                    command=evaluation_command(stage,trial,sample)
                    if stage=='track2-infer':command.extend(['--input-view','native_rgb_static_crop','--original-weights'])
                    runtime.run(command,[gpu],'static_crop_'+stage+'_'+sample,deadline)
                folder=OUTPUT/'predictions'/trial/sample;info=read(folder/'inference.json');metric=read(folder/'metrics.json')
                assert info['evaluation_protocol']==metric['evaluation_protocol']=='native_rgb_static_crop_target_head_v1'
                assert not info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
                assert info['delta_checkpoint'] is None and not info['gt_used_for_detection_or_tracking']
                assert metric['full_frame_roster'] and metric['frames']==manifest['samples'][sample]['frames']
                assert metric['gt_sha256']==manifest['samples'][sample]['gt_sha256']
                assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
                with mutex:results[sample]=metric
                status('running_fixed_Train_viewport_control',gpus=pair)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,gpu) for gpu in pair]
                try:
                    for future in futures:future.result()
                except BaseException:runtime.stop_children();raise
        finally:
            for lease in leases:lease.close()
        assert len(results)==4
        summary={key:float(np.mean([v[key] for v in results.values()])) for key in ['CD_H_cm','ACC_H_cm_per_frame2']}
        save_json(OUTPUT/'static_crop_Train_control_comparison.json',dict(summary=summary,per_sequence=results,
            test_used=False,registered_plan_sha256=sha256(plan_path),completed_unix=time.time()))
        status('complete',summary=summary)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'static_crop_Train_control_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

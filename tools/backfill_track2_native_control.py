"""Run one pending Train control after this GPU's export shard is complete."""
from pathlib import Path
import argparse,fcntl,json,os,signal,time
import run_track2_experiments as runtime
from finalize_track2_experiments import evaluation_command
from multihmr.track2_data import OUTPUT,save_json,sha256


def read(path):return json.loads(Path(path).read_text())


def main(gpu,sample):
    import pynvml
    prefix=f'native_rgb_backfill_gpu{gpu}'
    with (OUTPUT/'locks'/(prefix+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.REGISTRY_NAME=prefix+'_jobs.json'
        signal.signal(signal.SIGTERM,runtime.stop_children);signal.signal(signal.SIGINT,runtime.stop_children)
        manifest=read(OUTPUT/'manifest.json');resources=read(OUTPUT/'gpu_resources.json')
        assert gpu in resources['healthy_gpus'] and gpu!=0
        assert sample in manifest['validation'] and sample not in manifest['test']
        own=manifest['train'][resources['healthy_gpus'].index(gpu)::len(resources['healthy_gpus'])]
        deadline=read(OUTPUT/'budget.json')['deadline_unix'];trial='diagnostic_original_native_rgb'
        pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(resources['uuids'][str(gpu)])
        def status(value):save_json(OUTPUT/(prefix+'_status.json'),dict(status=value,pid=os.getpid(),gpu=gpu,
            sample=sample,trial=trial,test_used=False,source_exports_written=False,updated_unix=time.time()))
        status('waiting_for_own_exports_and_idle_GPU')
        while time.time()<deadline-60:
            if all((OUTPUT/'predictions/pair_head_cd100'/s/'conversion_v2.json').exists() for s in own) and not pynvml.nvmlDeviceGetComputeRunningProcesses(handle):break
            time.sleep(20)
        else:raise TimeoutError('No idle GPU before authorized deadline')
        status('running_same_Train_native_control')
        for stage in ['track2-infer','track2-convert','track2-score']:
            command=evaluation_command(stage,trial,sample)
            if stage=='track2-infer':command.extend(['--input-view','native_rgb','--original-weights'])
            runtime.run(command,[gpu],prefix+'_'+stage+'_'+sample,deadline)
        folder=OUTPUT/'predictions'/trial/sample;info=read(folder/'inference.json');metric=read(folder/'metrics.json')
        assert info['evaluation_protocol']==metric['evaluation_protocol']=='native_rgb_target_head_v1'
        assert info['delta_checkpoint'] is None and not info['target_foreground_only_input']
        assert info['all_accepted_heads_in_target_mask'] and not info['gt_used_for_detection_or_tracking']
        assert metric['frames']==manifest['samples'][sample]['frames'] and metric['full_frame_roster']
        assert metric['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
        status('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--sample',required=True);args=p.parse_args()
    try:main(args.gpu,args.sample)
    except BaseException as error:
        save_json(OUTPUT/f'native_rgb_backfill_gpu{args.gpu}_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

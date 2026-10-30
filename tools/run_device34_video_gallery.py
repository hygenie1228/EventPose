"""Own registered inference/render jobs; never touch unrelated processes."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from goal_runtime_control import read, write, register_job


def main():
    path = ROOT / '_agents/research/261028/device34_test_video_goal.json'
    goal = read(path); out = Path(goal['output_root']); work = Path(goal['work_root'])
    lock = (work / 'manager.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    register_job(goal, os.getpid(), 'video_gallery_manager')
    python = str(ROOT / '.venv-humanpose/bin/python'); (out / 'logs').mkdir(exist_ok=True)
    import pynvml as nv
    nv.nvmlInit(); GPUs = []
    inference_needed=not all((out/'meshes'/name/'complete.json').exists() for name in goal['models'])
    for index in range(nv.nvmlDeviceGetCount()) if inference_needed else []:
        handle = nv.nvmlDeviceGetHandleByIndex(index); physical = nv.nvmlDeviceGetMinorNumber(handle)
        if physical not in goal['allowed_gpus']: continue
        uuid = nv.nvmlDeviceGetUUID(handle); uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        if nv.nvmlDeviceGetComputeRunningProcesses(handle): continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2')
        probe = subprocess.run([python, '-c', 'import torch; x=torch.arange(32,device="cuda").sum().item(); torch.cuda.synchronize(); assert x==496; print(x)'], env=env, capture_output=True, text=True)
        bus = nv.nvmlDeviceGetPciInfo(handle).busId; bus = bus.decode() if isinstance(bus, bytes) else bus
        GPUs.append(dict(physical_gpu=physical, uuid=uuid, bus=bus, passed=probe.returncode == 0, stdout=probe.stdout, stderr=probe.stderr))
    if inference_needed:
        write(out / 'gpu_preflight.json', dict(checked_unix=time.time(), records=GPUs))
        assert any(g['passed'] for g in GPUs), 'No healthy idle authorized GPU; inspect probe evidence'
    jobs = []; attempted = set(); history = []
    def launch(role, args, gpu=None):
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                   EVENTPOSE_MULTISENSOR_OUT=goal['source_root'], CUDA_VISIBLE_DEVICES='')
        if gpu: env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], EVENTPOSE_PHYSICAL_GPU=str(gpu['physical_gpu']))
        log = (out / 'logs' / (role.replace(':', '_') + '.log')).open('a')
        proc = subprocess.Popen([python, '-u', str(ROOT / 'scripts/visualize_multihmr.py'), *args, '--goal', str(path)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(register_job(goal, proc.pid, role), gpu=gpu, launched_unix=time.time())
        jobs.append((proc, log, record)); attempted.add(role)
        print(json.dumps(dict(launched=record)), flush=True)
    order = ['rgb3_short90', 'rgb4', 'rgb3_event4', 'event3_rgb4', 'event3_event4', 'rgb3_event3', 'rgb4_event4', 'pre_rgb3', 'pre_rgb4']
    while True:
        for item in list(jobs):
            proc, log, record = item; code = proc.poll()
            if code is None: continue
            log.close(); jobs.remove(item); history.append(dict(record, exit_code=code, finished_unix=time.time()))
            if code:
                write(out / 'manager_state.json', dict(status='failed', failure=history[-1], history=history,
                                                       live_jobs=[j[2] for j in jobs]))
                raise RuntimeError('Concrete worker failure: ' + record['role'] + '; other workers preserved')
        for gpu in GPUs:
            if not gpu['passed'] or any(j[2].get('gpu') and j[2]['gpu']['uuid'] == gpu['uuid'] for j in jobs): continue
            handle = nv.nvmlDeviceGetHandleByUUID(gpu['uuid'])
            if nv.nvmlDeviceGetComputeRunningProcesses(handle): continue
            for name in order:
                role = 'video_mesh:' + name
                if role not in attempted and not (out / 'meshes' / name / 'complete.json').exists():
                    launch(role, ['infer', '--name', name], gpu); break
        render_module = ROOT / 'src/multihmr/video_gallery_render.py'
        if render_module.exists():
            if 'video_references' not in attempted and not (out / 'references_complete.json').exists():
                launch('video_references', ['references'])
            baseline_ready = (out / 'meshes/rgb3_short90/complete.json').exists()
            if (out / 'references_complete.json').exists() and baseline_ready:
                for name in order:
                    if not (out / 'meshes' / name / 'complete.json').exists(): continue
                    for clip in goal['clips']:
                        if not (out/'render_pilot_approval.json').exists() and (name,clip['clip'])!=('rgb3_short90','clip01'):
                            continue
                        role = 'video_render:' + name + ':' + clip['clip']
                        if role in attempted or (out / 'mp4' / name / (clip['clip'] + '.json')).exists(): continue
                        if sum(j[2]['role'].startswith('video_render:') for j in jobs) >= 4: break
                        launch(role, ['render', '--name', name, '--clip', clip['clip']])
        write(out / 'manager_state.json', dict(status='running', updated_unix=time.time(), history=history,
                                               live_jobs=[j[2] for j in jobs], gpu_records=GPUs))
        if all((out / 'mp4' / name / (clip['clip'] + '.json')).exists() for name in order for clip in goal['clips']) and not jobs:
            write(out / 'all_videos_rendered.json', dict(status='rendered_verification_required', expected=45, updated_unix=time.time()))
            return
        time.sleep(15)


if __name__ == '__main__': main()

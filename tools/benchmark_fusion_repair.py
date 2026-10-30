"""Same physical GPU, sequential full-pipeline cached/online timing."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write
GOAL = Path(os.environ.get('EVENTPOSE_GOAL_CONFIG', ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json'))
from fusion_repair_scope import EVAL_NAMES
NAMES = EVAL_NAMES
if 'device3_encoding' in GOAL.name:
    from device3_encoding_scope import EVAL_NAMES as NAMES


def read(path):
    return json.loads(path.read_text())


def active(uuid):
    import pynvml as n
    n.nvmlInit()
    return {p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByUUID(uuid))}


def worker(name, kind, goal):
    import shutil
    import torch
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    from multihmr.multisensor_inference import load_model, CachedSensors, OnlineSensors, infer
    import utils.pose_evaluation as base
    out = Path(goal['output_root']); config = read(out / 'configs' / (name + '.json'))
    assert int(os.environ['EVENTPOSE_PHYSICAL_GPU']) in goal['allowed_gpus']
    base.configure_torch(); work = out / 'runtime/work' / (name + '_' + kind)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / 'joint_regressor.npy', work / 'joint_regressor.npy'); base.OUT = work
    begin = time.perf_counter(); model = load_model(config, out / 'checkpoints' / name / 'best.pt')
    torch.cuda.synchronize(); loading = time.perf_counter() - begin
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    active_before=active(uuid)
    assert active_before == {os.getpid()}, 'Concurrent compute invalidates speed measurement'
    tasks = read(out / 'runtime/tasks.json'); records = []
    cls = CachedSensors if kind == 'cached' else OnlineSensors
    for raw in tasks:
        task = dict(raw, video_frames=100, smoke=True, condition=config['condition'])
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin = time.perf_counter()
        source = cls(raw['parent_session'], config['condition'], sensor_layout=REPAIR_LAYOUTS[config['condition']],encoding=config.get('encoding','count33'))
        opened = time.perf_counter() - begin
        collector = base.Collector(task, name)
        infer(model, collector, source, 16); collector.save()
        encoding = getattr(source, 'encoding_seconds', None); decode = getattr(source, 'decode_seconds', None)
        pixels = sorted(source.pixel_datasets_read); source.close(); torch.cuda.synchronize()
        records.append(dict(task=raw['session'], frames=100, seconds=time.perf_counter()-begin,
                            source_open_seconds=opened, encoding_seconds=encoding, decode_seconds=decode,
                            pixel_datasets_read=pixels, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    begin = time.perf_counter()
    cached = CachedSensors(tasks[0]['parent_session'], config['condition'], sensor_layout=REPAIR_LAYOUTS[config['condition']],encoding=config.get('encoding','count33'))
    online = OnlineSensors(tasks[0]['parent_session'], config['condition'], sensor_layout=REPAIR_LAYOUTS[config['condition']],encoding=config.get('encoding','count33'))
    a = cached.sample(tasks[0]['start_frame']); b = online.sample(tasks[0]['start_frame'])
    assert (a is None) == (b is None)
    if a is not None:
        assert set(a) == set(b)
        assert all(torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-6) for k in a)
    cached.close(); online.close(); parity_seconds = time.perf_counter() - begin
    active_after=active(uuid)
    assert active_after == {os.getpid()}
    seconds = sum(r['seconds'] for r in records)
    write(out / 'runtime' / (name + '_' + kind + '.json'),
          dict(name=name, source=kind, frames=300, seconds=seconds, fps=300/seconds,
               loading_seconds=loading, host_peak_rss_bytes=rss, segments=records,
               total_parameters=sum(p.numel() for p in model.parameters()), gpu_uuid=uuid,
               physical_gpu=int(os.environ['EVENTPOSE_PHYSICAL_GPU']), pid=os.getpid(),
               cuda_name=torch.cuda.get_device_name(), input_parity_passed=True,
               compute_pids_before=sorted(active_before),compute_pids_after=sorted(active_after),
               separate_input_parity_seconds=parity_seconds,
               included='source open; cache I/O or original decode/resize and causal event encoding; detection; reconstruction; camera transform; matching; save; source close',
               limitations='300 frozen Test frames, new process per condition; OS caches not evicted; original archive opening included online; not a live event stream'))


def supervise(physical, goal):
    import pynvml as n
    n.nvmlInit(); out = Path(goal['output_root'])
    assert physical in goal['allowed_gpus'] and time.time() < goal['deadline_unix'] - 600
    preflight = read(out / 'gpu_preflight.json')
    proven=set(read(out/'pipeline_state.json').get('proven_cuda_uuids',[]))
    gpu = next(r for r in preflight['records'] if r['physical_gpu'] == physical and (r['status'] == 'passed' or r['uuid'] in proven))
    h = n.nvmlDeviceGetHandleByUUID(gpu['uuid'])
    assert n.nvmlDeviceGetMinorNumber(h) == physical
    assert not active(gpu['uuid'])
    tasks = read(out / 'test/tasks.json'); parents = sorted({t['parent_session'] for t in tasks})
    chosen = [next(t for t in tasks if t['parent_session'] == parents[i]) for i in (0, len(parents)//2, len(parents)-1)]
    write(out / 'runtime/tasks.json', chosen)
    for name in NAMES:
        for kind in ('cached', 'online'):
            path = out / 'runtime' / (name + '_' + kind + '.json')
            if path.exists():
                assert read(path)['gpu_uuid'] == gpu['uuid']
                continue
            assert time.time() < goal['deadline_unix'] - 300 and not active(gpu['uuid'])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], EVENTPOSE_PHYSICAL_GPU=str(physical),
                       EVENTPOSE_MULTISENSOR_OUT=str(out), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
            started = time.time()
            with (out / 'runtime' / (name + '_' + kind + '.log')).open('a') as log:
                proc = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--name', name, '--source', kind],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                register_job(goal, proc.pid, 'fusion_repair_runtime:' + name + ':' + kind)
                assert proc.wait() == 0, (name, kind)
            result = read(path); result.update(gpu_identity=gpu, process_started_unix=started,
                  process_finished_unix=time.time(), cold_process_wall_seconds=time.time()-started)
            write(path, result)
    write(out / 'runtime/summary.json', {name+'_'+kind: read(out / 'runtime' / (name+'_'+kind+'.json'))
                                        for name in NAMES for kind in ('cached', 'online')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['worker', 'supervise'])
    parser.add_argument('--gpu', type=int); parser.add_argument('--name', choices=NAMES)
    parser.add_argument('--source', choices=['cached', 'online']); args = parser.parse_args()
    goal = read(GOAL)
    register_job(goal, os.getpid(), 'fusion_repair_runtime_' + args.stage)
    if args.stage == 'worker':
        worker(args.name, args.source, goal)
    else:
        supervise(args.gpu, goal)

"""Sequential, isolated full-pipeline timing for the frozen ten conditions.

Launch only after the scheduler releases the chosen GPU; each worker is also
registered with the hard-deadline guard. Online archive initialization is timed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
OUT = Path(os.environ.get('EVENTPOSE_EVENT_OUT', ROOT / 'exp/261024/rgb_event_36h'))
WORK = OUT/'runtime/work' if os.environ.get('EVENTPOSE_EVENT_OUT') else ROOT / '.experiment_storage/rgb_event_36h/runtime'


def read(path):
    return json.loads(path.read_text())


def worker(name, source_kind):
    import torch
    import utils.pose_evaluation as base
    from multihmr.event_inference import load_model, infer, CachedEvents, OnlineEvents
    from multihmr.event_data import save
    config = read(OUT / 'configs' / (name + '.json'))
    base.configure_torch()
    base.OUT = WORK / (name + '_' + source_kind)
    base.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT/'test/joint_regressor.npy' if os.environ.get('EVENTPOSE_EVENT_OUT') else ROOT / '.experiment_storage/humanpose_gt_segments/joint_regressor.npy',
                    base.OUT / 'joint_regressor.npy')
    checkpoint = None if config.get('original') else OUT / 'checkpoints' / name / 'best.pt'
    started = time.perf_counter()
    model = load_model(config, checkpoint)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    records = []
    for raw in read(OUT / 'runtime/tasks.json'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        video = base.OUT / 'input.mp4'
        source = None
        try:
            if config['mode'] != 'event':
                shutil.copyfile(raw['source_video'], video)
            staging_seconds = time.perf_counter() - started
            opening = time.perf_counter()
            if config['mode'] != 'rgb':
                cls = OnlineEvents if source_kind == 'online' else CachedEvents
                source = cls(raw['parent_session'], config['encoding'])
            source_open_seconds = time.perf_counter() - opening
            task = dict(raw, video=str(video) if config['mode'] != 'event' else '',
                        video_frames=100, smoke=True, input_mode=config['mode'],
                        rgb_pixels_used=config['mode'] != 'event', event_encoding=config.get('encoding'))
            collector = base.Collector(task, name)
            infer(model, collector, config.get('encoding'), 16, source)
            collector.save()
            if source is not None:
                source.close()
            torch.cuda.synchronize()
            records.append(dict(session=raw['session'], frames=100,
                seconds=time.perf_counter() - started, staging_seconds=staging_seconds,
                event_source_open_seconds=source_open_seconds,
                online_encoding_seconds=getattr(source, 'encoding_seconds', None),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        finally:
            video.unlink(missing_ok=True)
    seconds = sum(row['seconds'] for row in records)
    save(OUT / 'runtime' / (name + '_' + source_kind + '.json'),
         dict(name=name, mode=config['mode'], encoding=config.get('encoding'), source=source_kind,
              checkpoint=str(checkpoint) if checkpoint else 'original', load_seconds=load_seconds,
              frames=300, seconds=seconds, fps=300 / seconds, segments=records,
              includes='video staging when used; event archive/cache open; online encoding when used; decode; detection; model; SMPL-X; GT matching; result save; source close',
              cache_policy='OS cache not forcibly evicted; cold process initialization reported separately',
              gpu=torch.cuda.get_device_name()))


def supervise(gpu):
    from tools.goal_runtime_control import register_job
    from multihmr.event_data import save
    assert gpu in range(2, 7)
    goal = read(ROOT / '_agents/research/261026/new_split_agent_goal.json' if os.environ.get('EVENTPOSE_EVENT_OUT') else ROOT / '_agents/research/261024/rgb_event_goal_36h.json')
    assert time.time() < goal['deadline_unix']
    # Completion marker is necessary but not sufficient: also demand an empty GPU.
    assert (OUT / 'training_test_pipeline_complete.json').is_file()
    active = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not active, 'Chosen GPU has a compute process; do not interfere'
    if os.environ.get('EVENTPOSE_EVENT_OUT'):
        source=read(OUT/'test/tasks.json')
        parents=sorted({t['parent_session'] for t in source})
        tasks=[next(t for t in source if t['parent_session']==parents[i]) for i in (0,len(parents)//2,len(parents)-1)]
    else:tasks = read(ROOT / 'exp/261024/nlf_finetune_12h/runtime_benchmark/tasks.json')
    assert len(tasks) == 3 and len({task['parent_session'] for task in tasks}) == 3
    save(OUT / 'runtime/tasks.json', tasks)
    identity = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    gpu_uuid = identity.split(',')[1].strip()
    conditions = read(OUT / 'experiment_manifest.json')['conditions']
    for config in conditions:
        for source in (['cached'] if config['mode'] == 'rgb' else ['cached', 'online']):
            assert time.time() < goal['deadline_unix'] - 300
            path = OUT / 'runtime' / (config['name'] + '_' + source + '.json')
            # A result without supervisor provenance is incomplete and must be reviewed.
            if path.exists():
                assert read(path).get('gpu_identity') == identity
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, EVENTPOSE_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1')
            started = time.time()
            with (OUT / 'runtime' / (config['name'] + '_' + source + '.log')).open('a') as log:
                process = subprocess.Popen([sys.executable, '-u', __file__, 'worker',
                    '--name', config['name'], '--source', source], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                register_job(goal, process.pid, 'event_runtime_' + config['name'] + '_' + source)
                assert process.wait() == 0, str(path)
            result = read(path)
            result.update(gpu_identity=identity, cuda_visible_devices=gpu_uuid,physical_gpu=str(gpu),
                          process_started_unix=started, process_finished_unix=time.time(),
                          cold_process_wall_seconds=time.time() - started)
            save(path, result)
    save(OUT / 'runtime/summary.json', {path.stem: read(path)
         for path in sorted((OUT / 'runtime').glob('*.json')) if path.name not in ('tasks.json', 'summary.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['worker', 'supervise'])
    parser.add_argument('--gpu', type=int, choices=range(2, 7), default=3)
    parser.add_argument('--name')
    parser.add_argument('--source', choices=['cached', 'online'], default='cached')
    args = parser.parse_args()
    if args.stage == 'worker':
        worker(args.name, args.source)
    else:
        supervise(args.gpu)

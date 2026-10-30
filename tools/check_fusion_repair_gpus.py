"""Verify user-authorized physical GPUs via UUID, idle NVML state and real CUDA."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from goal_runtime_control import register_job, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads(Path(os.environ.get('EVENTPOSE_GOAL_CONFIG', ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json')).read_text())
    register_job(config, os.getpid(), 'fusion_repair_gpu_preflight')
    import pynvml as n
    n.nvmlInit()
    records = []
    for path in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        text = path.read_text()
        physical = int(re.search(r'Device Minor:\s*(\d+)', text)[1])
        if physical not in config['allowed_gpus']:
            continue
        uuid = re.search(r'GPU UUID:\s*(\S+)', text)[1]
        row = dict(physical_gpu=physical, uuid=uuid, proc_bus=path.parent.name)
        try:
            h = n.nvmlDeviceGetHandleByUUID(uuid)
            pci = n.nvmlDeviceGetPciInfo(h)
            bus = pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId
            assert int(bus.split(':')[-2], 16) == int(path.parent.name.split(':')[-2], 16)
            apps = n.nvmlDeviceGetComputeRunningProcesses(h)
            device_name=n.nvmlDeviceGetName(h)
            if isinstance(device_name,bytes):device_name=device_name.decode()
            row.update(nvml_bus=bus, compute_pids=[a.pid for a in apps], nvml_minor=n.nvmlDeviceGetMinorNumber(h),
                       nvml_name=device_name,total_memory_bytes=n.nvmlDeviceGetMemoryInfo(h).total)
            assert row['nvml_minor'] == physical
            if apps:
                row['status'] = 'busy_preserved'
            else:
                cmd = [sys.executable, '-c', 'import torch;torch.cuda.set_device(0);x=torch.arange(32,device="cuda").sum().item();torch.cuda.synchronize();assert x==496;print(x)']
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid))
                register_job(config, proc.pid, 'fusion_repair_cuda_probe:gpu' + str(physical))
                try:
                    stdout, stderr = proc.communicate(timeout=25)
                    row.update(status='passed' if proc.returncode == 0 else 'cuda_initialization_failed',
                               exit_code=proc.returncode, stdout=stdout[-500:], stderr=stderr[-1500:])
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    row['status'] = 'probe_timeout_no_training_started'
        except Exception as error:
            row.update(status='nvml_or_identity_failed', error=repr(error))
        records.append(row)
    result = dict(checked_unix=time.time(), allowed_gpus=config['allowed_gpus'], records=records,
                  usable_gpus=[r['physical_gpu'] for r in records if r['status'] == 'passed'])
    write(Path(config['output_root']) / 'gpu_preflight.json', result)
    write(Path(config['output_root']) / 'gpu_probe_history' / (str(int(result['checked_unix']))+'.json'),result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

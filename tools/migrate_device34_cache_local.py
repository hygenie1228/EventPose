"""Move only owned, completed input caches to local SSD; preserve NFS originals."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.goal_runtime_control import identity,write
from tools.prepare_device34_body import OUT
import h5py


def main():
    assert not (OUT/'training_cache_complete.json').exists()
    assert not list((OUT/'runs').glob('*/started.json'))
    state=json.loads((OUT/'pipeline_state.json').read_text());job=next(j for j in state['jobs'] if j['role']=='cache')
    assert identity(job['pid'])=={k:job[k] for k in ('pgid','start_ticks','command')}
    local=ROOT/'.experiment_storage/device34_body_28h/storage';local.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(local).free>140*1024**3
    os.killpg(job['pid'],signal.SIGTERM)
    until=time.monotonic()+15
    while identity(job['pid']) is not None and time.monotonic()<until:time.sleep(.2)
    assert identity(job['pid']) is None
    stamp=str(int(time.time()));counts={};total=0
    for folder in ('cache','source_maps','raw','targets'):
        source=OUT/folder;destination=local/folder;destination.mkdir(exist_ok=True)
        assert not source.is_symlink()
        files=[p for p in source.iterdir() if p.is_file() and '.partial.' not in p.name and '.tmp.' not in p.name]
        for i,p in enumerate(files):
            if folder=='cache':
                with h5py.File(p,'r',locking=False) as f:assert f.attrs['complete']
            target=destination/p.name;temporary=target.with_name(target.name+'.copying')
            shutil.copyfile(p,temporary);assert temporary.stat().st_size==p.stat().st_size;temporary.replace(target)
            total+=target.stat().st_size
            if i%10==0:print(json.dumps(dict(folder=folder,copied=i+1,total_files=len(files),bytes=total)),flush=True)
        backup=OUT/(folder+'_nfs_preserved_'+stamp);source.rename(backup);source.symlink_to(destination,target_is_directory=True)
        counts[folder]=len(files)
    staging=local/'input_stage';staging.mkdir(exist_ok=True);(OUT/'input_stage').symlink_to(staging,target_is_directory=True)
    write(OUT/'local_cache_migration.json',dict(completed_unix=time.time(),previous_job=job,
        preserved_NFS_backups=True,local_storage=str(local),copied_files=counts,bytes=total,
        dataset_modified=False,training_started=False,input_copy_concurrency=2))
    write(OUT/'retry_requests.json',dict(roles=['cache'],reason='Verified completed inputs on local SSD; sequential staging of original source streams'))
    print(json.dumps(dict(status='local_storage_ready_cache_resume_requested',counts=counts,bytes=total)),flush=True)


if __name__=='__main__':main()

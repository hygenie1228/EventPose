"""Replace only a verified aggregation controller, preserving its workers."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import shlex
import subprocess
import time
from goal_runtime_control import identity,register_job
from run_part_aggregation import ROOT,GOAL,read,write

def upgrade(expected_pid,reason):
    goal=read(GOAL);out=ROOT/goal['output_dir']
    if (ROOT/'.experiment_storage/part_aggregation_261029/user_control_stop.json').exists():
        raise RuntimeError('User stop notice exists; preserving the stopped experiment')
    assert time.time()<goal['deadline_unix']-300
    old=read(out/'scheduler_launch.json')
    assert old['pid']==expected_pid
    old_identity={k:old[k] for k in ('pgid','start_ticks','command')}
    assert identity(expected_pid)==old_identity
    archive=out/'controller_upgrades';archive.mkdir(exist_ok=True)
    previous=out/'single_owner_controller_upgrade.json'
    if previous.exists():
        receipt=read(previous)
        write(archive/(str(receipt['manager']['pid'])+'.json'),receipt)
    source=(ROOT/'tools/run_part_aggregation.py').read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    snapshots=out/'source_snapshots';snapshots.mkdir(exist_ok=True)
    (snapshots/(digest+'.txt')).write_bytes(source)
    # Freeze dispatch while reconciling the registry with the last saved state.
    # A worker may have been registered since the controller's last 20s save.
    os.kill(expected_pid,signal.SIGSTOP);killed=False
    try:
        time.sleep(.1)
        assert identity(expected_pid)==old_identity
        state=read(out/'pipeline_state.json')
        registered=read(ROOT/goal['managed_jobs_file'])['jobs']
        live=[j for j in registered if j['role'].startswith(('pilot:','train:','test:','diagnose:'))
              and identity(j['pid'])=={k:j[k] for k in ('pgid','start_ticks','command')}]
        child_pids=[int(p) for p in Path(f'/proc/{expected_pid}/task/{expected_pid}/children').read_text().split()]
        live_ids={j['pid'] for j in live}
        assert all(identity(pid) is None or pid in live_ids for pid in child_pids), 'Unregistered live child; resume controller and retry at an idle boundary'
        for job in live:
            if not any(j['pid']==job['pid'] and j['start_ticks']==job['start_ticks'] for j in state['jobs']):
                job=dict(job)
                args=shlex.split(job['command'])
                option='--config' if '--config' in args else '--event-config'
                cfg=read(args[args.index(option)+1])
                assert Path(cfg['output_root']).resolve()==out.resolve()
                physical=None
                for item in Path(f"/proc/{job['pid']}/environ").read_bytes().split(b'\0'):
                    if item.startswith(b'EVENTPOSE_PHYSICAL_GPU='):physical=int(item.split(b'=',1)[1])
                assert physical in goal['gpu_candidates_physical']
                job['physical_gpu']=physical
                state['jobs'].append(job)
                state['attempts'].setdefault(job['role'],1)
        workers={str(j['pid']):identity(j['pid']) for j in state['jobs']}
        assert expected_pid not in [int(pid) for pid in workers]
        write(out/'pipeline_state.json',state)
        # SIGTERM invokes the experiment-wide stop handler. Kill only the
        # verified controller PID, never its process group or its workers.
        os.kill(expected_pid,signal.SIGKILL);killed=True
    finally:
        if not killed and identity(expected_pid)==old_identity:os.kill(expected_pid,signal.SIGCONT)
    until=time.monotonic()+5
    while identity(expected_pid) is not None and time.monotonic()<until:time.sleep(.1)
    assert identity(expected_pid) is None
    with (out/'logs/scheduler_single_owner.log').open('a') as log:
        process=subprocess.Popen([str(ROOT/'.venv-humanpose/bin/python'),'-u','tools/run_part_aggregation.py'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    record=dict(register_job(goal,process.pid,'aggregation_scheduler'),started_unix=time.time(),
        deadline_unix=goal['deadline_unix'],replaces_scheduler_pid=expected_pid,
        existing_workers_preserved=True,source_sha256=digest)
    write(out/'scheduler_launch.json',record)
    time.sleep(2)
    assert process.poll() is None, 'Replacement controller exited; inspect scheduler_single_owner.log'
    after={pid:identity(int(pid)) for pid in workers}
    changed={pid:dict(before=before,after=after[pid]) for pid,before in workers.items()
        if before is not None and after[pid] is not None and after[pid]!=before}
    assert not changed,changed
    receipt=dict(manager=record,worker_before=workers,worker_after=after,
        naturally_finished_workers=[pid for pid,before in workers.items() if before is not None and after[pid] is None],
        reason=reason,original_deadline_unix=goal['deadline_unix'],checked_unix=time.time())
    write(out/'single_owner_controller_upgrade.json',receipt)
    write(archive/(str(record['pid'])+'.json'),receipt)
    print(receipt)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-pid',type=int,required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    upgrade(args.expected_pid,args.reason)

"""Read-only audit of the original deadline, registered processes and saved runs."""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from goal_runtime_control import identity


def read(path):
    return json.loads(path.read_text())


def shutdown_requirements(now, deadline, end_record, live, unknown, missing_checkpoints):
    return dict(original_deadline_reached=now >= deadline,
                manager_fixed_deadline_receipt=bool(end_record and end_record.get('status') == 'fixed_20h_finished'
                    and end_record.get('deadline_unix') == deadline and end_record.get('finished_unix', 0) >= deadline),
                no_registered_owned_processes_alive=not live,
                no_unknown_registered_process_identity=not unknown,
                saved_completed_run_checkpoints_present=not missing_checkpoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    goal = read(ROOT/'_agents/research/261029/part_aware_dynamic_aggregation_goal_20h.json')
    deadline = goal['deadline_unix']; assert deadline == 1793321712
    now = time.time()
    if now < deadline:
        parser.error('Original fixed deadline has not been reached; refusing an early shutdown-completion audit')
    out = ROOT/goal['output_dir']; registry = read(ROOT/goal['managed_jobs_file'])
    live = []; mismatches = []; unknown = []
    for record in registry['jobs']:
        try:
            current = identity(record['pid'])
        except (PermissionError, OSError) as error:
            unknown.append(dict(pid=record['pid'], reason=type(error).__name__)); continue
        if current is None:
            continue
        expected = {key:record[key] for key in ['start_ticks','command','pgid']}
        if current == expected:
            live.append(dict(pid=record['pid'], role=record['role'], identity=current))
        else:
            mismatches.append(dict(recorded_pid=record['pid'], recorded_role=record['role'],
                                   status='Different identity preserved; no signal sent'))
    checkpoints = []; missing = []
    for path in sorted((out/'runs').glob('*/complete.json')):
        run = read(path)
        if run.get('status') not in ['complete','pilot_complete','interrupted'] or run.get('successful_steps',0) <= 0:
            continue
        name = path.parent.name
        best = Path(run['checkpoint']); last = out/'checkpoints'/name/'last.pt'
        present = dict(best=best.exists() and best.stat().st_size > 0,
                       last=last.exists() and last.stat().st_size > 0)
        checkpoints.append(dict(name=name, status=run['status'], successful_steps=run['successful_steps'],
                                checkpoint_files_present=present))
        if not present['last'] or (run['status'] in ['complete','pilot_complete'] and not present['best']):
            missing.append(name)
    endpath = out/'deadline_status.json'; end = read(endpath) if endpath.exists() else None
    checks = shutdown_requirements(now, deadline, end, live, unknown, missing)
    result = dict(status='passed' if all(checks.values()) else 'incomplete',
                  checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                  deadline_unix=deadline, requirements=checks, deadline_status=end,
                  registered_live_processes=live, different_identities_preserved=mismatches,
                  unknown_identities=unknown, completed_run_checkpoints=checkpoints,
                  missing_checkpoints=missing,
                  scope='Read-only original deadline/manager terminal receipt, all registered exact process identities, nonempty best/last checkpoint presence; no signal, training, evaluation or Goal state changes',
                  limitation='Checkpoint presence does not replace byte/content audits; this is not a performance-gain or overall-goal completion declaration')
    folder = out/'artifact_audits'; folder.mkdir(exist_ok=True)
    (folder/'owned_execution_shutdown.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='completed_run_checkpoints'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

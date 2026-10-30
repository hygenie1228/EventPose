"""Apply a user-authorized goal at an idle boundary and enforce its wall deadline.

No model calls, training restarts, or GPU polling. Only registered process groups
with matching /proc identities can be stopped by the deadline watchdog.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def request(config, suffix, body=None):
    key = Path(config.get('key_file', ROOT / '.experiment_storage/research_workbench/access-key')).read_text().strip()
    url = config.get('api_base', 'http://127.0.0.1:18080')
    url += '/api/chats/' + config['session_id'] + '/' + suffix
    req = Request(url, data=None if body is None else json.dumps(body).encode(),
                  headers={'Cookie': 'workbench=' + key, 'Content-Type': 'application/json'})
    with urlopen(req, timeout=15) as response:
        return json.load(response)


def identity(pid):
    try:
        stat = Path('/proc/{}/stat'.format(pid)).read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        command = Path('/proc/{}/cmdline'.format(pid)).read_bytes().replace(b'\0', b' ').decode().strip()
        return {'start_ticks': stat[19], 'command': command, 'pgid': os.getpgid(pid)}
    except (FileNotFoundError, ProcessLookupError):
        return None


def register_job(config, pid, role):
    """Register only a new session leader; preserve concurrent job registrations."""
    import fcntl
    registry = Path(config['managed_jobs_file'])
    registry.parent.mkdir(parents=True, exist_ok=True)
    current = identity(pid)
    assert current is not None and current['pgid'] == pid
    # Registry lives on NFS; keep advisory locking on local SSD so an unavailable
    # NFS lock service cannot stall job registration after a process is launched.
    lock_path = ROOT / '.experiment_storage/rgb_event_36h' / (digest(str(registry)) + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = read(registry) if registry.exists() else {'jobs': []}
        record = dict(pid=pid, role=role, **current)
        if not any(job['pid'] == pid and job['start_ticks'] == current['start_ticks'] for job in value['jobs']):
            value['jobs'].append(record)
        write(registry, value)
    return record


def stop_registered(config):
    registry = Path(config['managed_jobs_file'])
    jobs = read(registry).get('jobs', []) if registry.exists() else []
    stopped = []
    for job in jobs:
        pid = int(job['pid'])
        current = identity(pid)
        if not current:
            continue
        if current != {key: job[key] for key in ('start_ticks', 'command', 'pgid')}:
            stopped.append({'pid': pid, 'status': 'identity_mismatch_preserved'})
            continue
        if current['pgid'] != pid or pid == os.getpgrp():
            stopped.append({'pid': pid, 'status': 'unsafe_group_preserved'})
            continue
        try:
            # Let the trainer save first. Signalling its DataLoader workers at
            # the same instant raises a worker-failure exception in the parent
            # before its graceful checkpoint handler reaches the loop boundary.
            if job.get('role','').startswith('train:'):
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        # Keep checkpoints/logs; training implementations must also receive this
        # same deadline and save before expiry. Never delete experiment files.
        until = time.monotonic() + 30
        while identity(pid) == current and time.monotonic() < until:
            time.sleep(.2)
        if identity(pid) == current:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stopped.append({'pid': pid, 'status': 'stopped_at_user_deadline'})
    return stopped


def matching_goal(config):
    goal = request(config, 'goal')['goal']
    if not goal or goal.get('threadId') != config['thread_id']:
        return None
    if digest(goal['objective']) != config['objective_sha256']:
        return None
    return goal


def apply_when_idle(config, status_path):
    while time.time() < config['deadline_unix']:
        current = request(config, 'goal')['goal']
        if current and digest(current['objective']) == config['objective_sha256']:
            write(status_path, {'status': 'goal_verified', 'goal': current, 'verified_unix': time.time()})
            return
        if current and digest(current['objective']) != config['previous_objective_sha256']:
            raise RuntimeError('Goal was changed by another user action; preserving it')
        try:
            request(config, 'message', {'goal': True, 'prompt': config['objective']})
        except HTTPError as error:
            if error.code != 409:
                raise
            # The owning app-server will apply the goal when this turn ends.
            time.sleep(.05)
            continue
        current = matching_goal(config)
        if not current or current['status'] != 'active':
            raise RuntimeError('Goal POST succeeded but native verification failed')
        write(status_path, {'status': 'goal_verified', 'goal': current, 'verified_unix': time.time()})
        return
    raise RuntimeError('Goal application deadline expired')


def watchdog(config, status_path):
    write(status_path, {'status': 'armed', 'deadline_unix': config['deadline_unix'], 'armed_unix': time.time()})
    while time.time() < config['deadline_unix']:
        time.sleep(min(30, config['deadline_unix'] - time.time()))
    goal = None
    try:
        goal = matching_goal(config)
        if goal is None:
            write(status_path, {'status': 'superseded_goal_preserved', 'checked_unix': time.time()})
            return
        if goal['status'] == 'complete':
            write(status_path, {'status': 'already_complete', 'checked_unix': time.time()})
            return
    except Exception as error:
        write(status_path, {'status': 'deadline_api_temporarily_unavailable', 'error': str(error)})
    # At the hard wall limit stop only this goal's registered jobs, even if the
    # UI endpoint is temporarily down. Never enumerate/kill other GPU users.
    paused_unix = None
    if goal is not None:
        try:
            response = request(config, 'goal', {'action': 'pause'})
            assert response['goal']['status'] == 'paused'
            paused_unix = time.time()
        except Exception:
            pass
    stopped = stop_registered(config)
    if paused_unix is not None:
        write(status_path, {'status': 'paused_at_deadline', 'paused_unix': paused_unix,
                            'stopped_jobs': stopped, 'goal': response['goal']})
        return
    while True:
        try:
            goal = matching_goal(config)
            if goal is None or goal['status'] == 'complete':
                write(status_path, {'status': 'goal_changed_or_completed', 'stopped_jobs': stopped})
                return
            response = request(config, 'goal', {'action': 'pause'})
            assert response['goal']['status'] == 'paused'
            write(status_path, {'status': 'paused_at_deadline', 'paused_unix': time.time(),
                                'stopped_jobs': stopped, 'goal': response['goal']})
            return
        except Exception as error:
            write(status_path, {'status': 'jobs_stopped_pause_retry', 'error': str(error), 'stopped_jobs': stopped})
            time.sleep(5)


def self_test():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import tempfile
    with tempfile.TemporaryDirectory(prefix='eventpose_goal_guard_') as directory:
        folder = Path(directory)
        state = {'goal': {'threadId': 'test', 'objective': 'test goal', 'status': 'active'}}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(state).encode())
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['action'] == 'pause'
                state['goal']['status'] = 'paused'
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(state).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        managed = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            write(folder / 'jobs.json', {'jobs': [dict(pid=managed.pid, **identity(managed.pid))]})
            (folder / 'key').write_text('test-only-key')
            config = {'session_id': 'test', 'thread_id': 'test', 'objective_sha256': digest('test goal'),
                      'deadline_unix': time.time() + .1, 'managed_jobs_file': str(folder / 'jobs.json'),
                      'key_file': str(folder / 'key'), 'api_base': 'http://127.0.0.1:' + str(server.server_port)}
            watchdog(config, folder / 'status.json')
            assert state['goal']['status'] == 'paused'
            assert read(folder / 'status.json')['status'] == 'paused_at_deadline'
            assert managed.wait(timeout=3) < 0 and unrelated.poll() is None
            print('Deadline pause + registered job stop + unrelated process preservation: passed')
        finally:
            for process in (managed, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait()
            server.shutdown(); server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply', 'watchdog', 'self-test'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--status', type=Path)
    args = parser.parse_args()
    if args.action == 'self-test':
        self_test()
    else:
        config = read(args.config)
        (apply_when_idle if args.action == 'apply' else watchdog)(config, args.status)

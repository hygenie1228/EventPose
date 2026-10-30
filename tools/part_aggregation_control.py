"""Read-only workbench control watcher for this experiment's parent Goal.

Automatic model-capacity `blocked` is not a user pause. Explicit pause/clear or
replacement of this Goal stops only the experiment's registered workers.
"""
import hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen

def classify_session(session,settings):
    if session is None:return dict(stop=False,status='session_missing',reason=None)
    if session.get('codexThread')!=settings['thread_id']:
        return dict(stop=True,status='replaced',reason='Parent session moved to another thread')
    goal=session.get('goal')
    if goal is None:return dict(stop=True,status='cleared',reason='Parent Goal was explicitly cleared')
    if goal.get('status')=='paused':return dict(stop=True,status='paused',reason='Parent Goal was explicitly paused')
    digest=hashlib.sha256(goal.get('objective','').encode()).hexdigest()
    if digest!=settings['objective_sha256']:
        return dict(stop=True,status='replaced',reason='Parent Goal objective was replaced')
    return dict(stop=False,status=goal.get('status'),reason=None)

def read_parent_control(settings):
    key=Path(settings['key_file']).read_text().strip()
    request=Request(settings['api_base']+'/api/chats?pane='+settings.get('pane','agents'),
                    headers={'Cookie':'workbench='+key})
    with urlopen(request,timeout=10) as stream:sessions=json.load(stream)
    session=next((s for s in sessions if s['id']==settings['session_id']),None)
    return classify_session(session,settings)

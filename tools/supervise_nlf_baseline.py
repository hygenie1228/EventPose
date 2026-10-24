"""Adopt live baseline workers and recover only terminal/missing ranks."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261024/nlf_finetune_12h'
WORK=ROOT/'.experiment_storage/nlf_finetune_12h'

def workers():
    found={}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            args=directory.joinpath('cmdline').read_bytes().decode().split('\0')
            if not args or 'python' not in Path(args[0]).name:continue
            if not any('evaluate_nlf' in arg for arg in args) or 'worker' not in args:continue
            if '--method' in args and args[args.index('--method')+1]!='nlf_pretrained':continue
            state=directory.joinpath('stat').read_text().split(') ',1)[1][0]
            if state=='Z':continue
            rank=int(args[args.index('--rank')+1]);assert rank not in found,(rank,found)
            found[rank]=int(directory.name)
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):continue
    return found

def main():
    tasks=json.loads((WORK/'tasks.json').read_text());assert len(tasks)==320
    children=[];attempts={rank:0 for rank in range(6)}
    while True:
        for p,_ in children:p.poll()
        alive=workers();done=sum((WORK/'nlf_pretrained'/t['session']/'complete.json').exists() for t in tasks)
        record=dict(status='inference_complete' if done==320 else 'running',completed_segments=done,
                    workers=alive,updated_unix=time.time(),restarts=attempts)
        target=OUT/'nlf_pretrained_recovery_status.json';temp=target.with_suffix('.tmp.json')
        temp.write_text(json.dumps(record,indent=2));temp.replace(target)
        if done==320:break
        names=sorted({t['parent_session'] for t in tasks})
        for rank in range(6):
            assigned=set(names[rank::6])
            if all((WORK/'nlf_pretrained'/t['session']/'complete.json').exists() for t in tasks if t['parent_session'] in assigned):continue
            if rank in alive:continue
            attempts[rank]+=1;assert attempts[rank]<=3,(rank,'repeated worker failures')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
            log=(OUT/'logs'/f'baseline_recovery_{rank}.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate_nlf.py'),'worker',
                                '--rank',str(rank),'--batch','16'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((p,log))
        time.sleep(15)
    for p,log in children:
        p.wait();log.close();assert p.returncode==0,p.pid
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-m','utils.compare_methods'],env=env,cwd=ROOT,check=True)
    (OUT/'nlf_pretrained_recovery_status.json').write_text(json.dumps(dict(status='complete',completed_segments=320,
        updated_unix=time.time(),comparison='comparison.json'),indent=2))

if __name__=='__main__':main()

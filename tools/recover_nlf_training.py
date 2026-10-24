"""Adopt healthy NLF jobs and restart only failed runs, preserving their deadline."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261024/nlf_finetune_12h'

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:return False

def save(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    plan=json.loads((OUT/'training_plan.json').read_text());configs=plan['configs']
    status=json.loads((OUT/'training_status.json').read_text())
    pids={r['name']:r['pid'] for r in status['jobs']};attempts={c['name']:0 for c in configs}
    children={}
    while True:
        for p,log in children.values():p.poll()
        done=[]
        for i,c in enumerate(configs):
            name=c['name'];complete=OUT/'runs'/name/'complete.json'
            if complete.exists():done.append(name);continue
            if alive(pids[name]):continue
            attempts[name]+=1;assert attempts[name]<=3,(name,'repeated training failure')
            assert time.time()<plan['end_unix']-600,'No training time left'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpus'],OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                LD_LIBRARY_PATH='/home/namhj/miniconda3/envs/videohor/lib/python3.10/site-packages/nvidia/cudnn/lib')
            command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',f'--master_port={29840+i}',
                str(ROOT/'scripts/train_nlf.py'),'--name',name,'--batch','16','--lr',str(c['lr']),
                '--pa-weight',str(c['pa_weight']),'--end-unix',str(plan['end_unix']),'--resume']
            if c['backbone']:command+=['--backbone','--backbone-start',str(c['backbone_start'])]
            log=(OUT/'logs'/f'{name}_recovered.log').open('a',buffering=1)
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children[name]=(p,log);pids[name]=p.pid
        save(OUT/'training_recovery_status.json',dict(status='training',updated_unix=time.time(),
            end_unix=plan['end_unix'],pids=pids,restarts=attempts,completed=done))
        if len(done)==3:break
        time.sleep(15)
    for p,log in children.values():p.wait();log.close();assert p.returncode==0,p.pid
    # The original supervisor owns healthy original children; wait for it to reap
    # them and record its failed original run before replacing final orchestration.
    while alive(80164):time.sleep(15)
    candidates=[]
    for c in configs:
        run=OUT/'runs'/c['name'];finished=json.loads((run/'complete.json').read_text())
        assert finished['successful_steps']>0
        best=json.loads((run/'best.json').read_text());candidates.append((best['pa_mm'],c))
    _,winner=min(candidates,key=lambda item:item[0]);checkpoint=OUT/'checkpoints/best_full.torchscript'
    subprocess.run([sys.executable,str(ROOT/'tools/export_nlf_finetuned.py'),'--delta',
        str(OUT/'checkpoints'/winner['name']/'best.pt'),'--output',str(checkpoint)],cwd=ROOT,check=True)
    save(OUT/'selected.json',dict(config=winner,checkpoint=str(checkpoint),test_used_for_selection=False))
    save(OUT/'training_recovery_status.json',dict(status='final_test',updated_unix=time.time()))
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_nlf.py'),'supervise','--method','nlf_finetuned',
        '--checkpoint',str(checkpoint),'--batch','16'],cwd=ROOT,check=True)
    save(OUT/'training_recovery_status.json',dict(status='complete',updated_unix=time.time()))

if __name__=='__main__':main()

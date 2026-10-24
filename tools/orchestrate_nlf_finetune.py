"""Run three two-GPU NLF experiments, export a frozen winner, then run Test."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261024/nlf_finetune_12h'

def save(path,value):
    temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def main():
    start=time.time();end=start+8*3600
    configs=[dict(name='head',gpus='0,1',lr=1e-5,pa_weight=100,backbone=False),
             dict(name='last_stage',gpus='2,3',lr=3e-6,pa_weight=150,backbone=True,backbone_start=7),
             dict(name='last_two_stages',gpus='4,5',lr=1e-6,pa_weight=200,backbone=True,backbone_start=6)]
    save(OUT/'training_plan.json',dict(started_unix=start,end_unix=end,configs=configs,test_used_for_selection=False))
    logs=OUT/'logs';logs.mkdir(exist_ok=True);jobs=[]
    for i,c in enumerate(configs):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpus'],OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
            LD_LIBRARY_PATH='/home/namhj/miniconda3/envs/videohor/lib/python3.10/site-packages/nvidia/cudnn/lib')
        command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',f'--master_port={29840+i}',
            str(ROOT/'scripts/train_nlf.py'),'--name',c['name'],'--batch','16','--lr',str(c['lr']),
            '--pa-weight',str(c['pa_weight']),'--end-unix',str(end)]
        if c['backbone']:command+=['--backbone','--backbone-start',str(c['backbone_start'])]
        log=(logs/f'{c["name"]}.log').open('a',buffering=1)
        proc=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((c,proc,log))
    while any(p.poll() is None for _,p,_ in jobs):
        save(OUT/'training_status.json',dict(status='training',updated_unix=time.time(),end_unix=end,
             jobs=[dict(name=c['name'],pid=p.pid,exit_code=p.poll()) for c,p,_ in jobs]))
        time.sleep(15)
    for c,p,log in jobs:
        log.close();assert p.returncode==0,(c,p.returncode)
        completed=json.loads((OUT/'runs'/c['name']/'complete.json').read_text())
        assert completed['successful_steps']>0
    candidates=[(json.loads((OUT/'runs'/c['name']/'best.json').read_text())['pa_mm'],c) for c in configs]
    _,winner=min(candidates,key=lambda item:item[0])
    checkpoint=OUT/'checkpoints/best_full.torchscript'
    subprocess.run([sys.executable,str(ROOT/'tools/export_nlf_finetuned.py'),'--delta',
        str(OUT/'checkpoints'/winner['name']/'best.pt'),'--output',str(checkpoint)],cwd=ROOT,check=True)
    save(OUT/'selected.json',dict(config=winner,checkpoint=str(checkpoint),test_used_for_selection=False))
    save(OUT/'training_status.json',dict(status='final_test',updated_unix=time.time()))
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_nlf.py'),'supervise','--method','nlf_finetuned',
        '--checkpoint',str(checkpoint),'--batch','16'],cwd=ROOT,check=True)
    save(OUT/'training_status.json',dict(status='training_and_inference_complete',updated_unix=time.time()))

if __name__=='__main__':main()

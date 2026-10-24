from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[1]
_sys.path.insert(0,str(_PROJECT_ROOT/"tools"))
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""14-hour two-wave search; 3 concurrent 2-GPU runs, validation-only selection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multihmr.data import ROOT,OUT,WORK,ASSETS,save


def run_group(configs,wave):
    processes={};completed=[];attempts={c['name']:0 for c in configs}
    while len(completed)<len(configs):
        for i,config in enumerate(configs):
            name=config['name']
            if name in completed:continue
            done=OUT/'runs'/name/'complete.json'
            if done.exists():
                result=json.loads(done.read_text())
                assert result['elapsed_seconds']>=4*3600 and result['successful_steps']>0
                if name in processes:
                    proc,log=processes[name]
                    if proc.poll() is None:continue
                    log.close();assert proc.returncode==0
                    del processes[name]
                completed.append(name);continue
            if name in processes:
                proc,log=processes[name]
                if proc.poll() is None:continue
                log.close();del processes[name]
                assert attempts[name]<3,dict(failed=name,exit_code=proc.returncode)
            pair=[2*i,2*i+1]
            config['gpu_pair']=pair;path=OUT/'configs'/f'{name}.json';save(path,config)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),OMP_NUM_THREADS='2',
                     TMPDIR=str(WORK/'tmp'),NCCL_ASYNC_ERROR_HANDLING='1')
            log=(WORK/'logs'/f'{name}.log').open('a',buffering=1)
            command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port',str(29720+i),
                     str(ROOT/'tools/train_multihmr_etr.py'),'--config',str(path)]
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes[name]=(proc,log);attempts[name]+=1
            print('Started',name,'GPUs',pair,'PID',proc.pid,flush=True)
        save(OUT/'search_status.json',dict(wave=wave,status='training',updated_unix=time.time(),completed=completed,
             workers=[dict(name=n,pid=p.pid,exit_code=p.poll(),attempt=attempts[n]) for n,(p,_) in processes.items()]))
        if len(completed)<len(configs):time.sleep(15)
    return configs


def best_of(configs):
    candidates=[]
    for config in configs:
        metrics=json.loads((OUT/'runs'/config['name']/'best.json').read_text())
        candidates.append((metrics['pa_mpjpe_mm'],config,metrics))
    return min(candidates,key=lambda x:x[0])


def export_checkpoint(config,metrics):
    import torch
    delta_path=ASSETS/config['name']/'best.pt'
    delta=torch.load(delta_path,map_location='cpu')
    base=torch.load(delta['base_checkpoint'],map_location='cpu')
    assert set(delta['model_delta'])<=set(base['model_state_dict'])
    base['model_state_dict'].update(delta['model_delta'])
    base['fine_tuning']=dict(config=config,validation=metrics,selection='held-out Train subjects only',source_delta=str(delta_path))
    path=ASSETS/'best_full.pt';temporary=path.with_suffix('.tmp.pt');torch.save(base,temporary);temporary.replace(path)
    return path


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert (OUT/'preparation_complete.json').exists() and (OUT/'pilots_complete.json').exists()
    assert all((OUT/'runs'/f'pilot16_blocks{n}'/'complete.json').exists() for n in [0,4,8])
    first_file=OUT/'wave1.json'
    if first_file.exists():first=json.loads(first_file.read_text())
    else:
        first=[];end=time.time()+6*3600
        for name,blocks,lr,bb in [('w1_head',0,5e-6,0),('w1_last4',4,5e-6,5e-7),('w1_last8',8,1e-5,1e-6)]:
            first.append(dict(name=name,last_blocks=blocks,head_lr=lr,backbone_lr=bb,hours=6,deadline_unix=end,
                              batch_size=16,workers=2,pa_weight=100,root_weight=25,validation_minutes=45,augment=True))
        save(first_file,first)
    run_group(first,1)
    score,winner,metrics=best_of(first)
    save(OUT/'wave1_selection.json',dict(winner=winner,metrics=metrics,criterion='minimum validation root-aligned PA-MPJPE, mm'))
    baseline=json.loads((OUT/'runs'/first[0]['name']/'baseline.json').read_text())
    second_file=OUT/'wave2.json'
    if second_file.exists():second=json.loads(second_file.read_text())
    else:
        final_reserve=3600
        end=manifest['deadline_unix']-final_reserve
        hours=min(8.,(end-time.time())/3600)
        assert hours>=4,dict(reason='Cannot fit a new four-hour experiment before final Test reserve',hours=hours)
        start_checkpoint=str(ASSETS/winner['name']/'best.pt')
        lr=winner['head_lr']*.35;bb=max(winner['backbone_lr']*.35,1e-7)
        second=[]
        for name,blocks,pa,root,head_lr,backbone_lr in [
            ('w2_refine',winner['last_blocks'],100,25,lr,bb),
            ('w2_pa_focus',max(4,winner['last_blocks']),200,10,lr,bb),
            ('w2_deeper',min(12,max(4,winner['last_blocks']+4)),150,20,lr,bb*.5)]:
            second.append(dict(name=name,last_blocks=blocks,head_lr=head_lr,backbone_lr=backbone_lr,
                hours=hours,deadline_unix=end,batch_size=16,workers=2,pa_weight=pa,root_weight=root,
                validation_minutes=45,augment=True,init_checkpoint=start_checkpoint,minimum_coverage=baseline['coverage']-.005,
                horizontal_flip_probability=.5 if name=='w2_deeper' else 0))
        save(second_file,second)
    run_group(second,2)
    score,winner,metrics=best_of(first+second)
    checkpoint=export_checkpoint(winner,metrics)
    save(OUT/'selected.json',dict(config=winner,validation=metrics,baseline_validation=baseline,
         checkpoint=str(checkpoint),selected_unix=time.time(),test_used_for_selection=False))
    save(OUT/'search_status.json',dict(status='final_test',updated_unix=time.time(),selected=winner['name']))
    subprocess.run([sys.executable,str(ROOT/'tools/evaluate_multihmr_finetuned.py'),'supervise'],cwd=ROOT,check=True)
    save(OUT/'search_status.json',dict(status='complete',updated_unix=time.time(),selected=winner['name'],
                                    report='_agents/report/multihmr_finetuning_14h.md'))


if __name__=='__main__':main()

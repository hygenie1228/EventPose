"""Synthetic GPU execution preflight, never a dataset score or optimizer trial."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from utils.pose_evaluation import load_multihmr,configure_torch
from multihmr.multisensor_model import MultiSensorMultiHMR,CONDITIONS
from tools.prepare_device34_body import OUT,save


def main():
    configure_torch();torch.manual_seed(261026)
    started=time.time();base=load_multihmr().eval();results=[]
    K=torch.tensor([[[800.,0,447.5],[0,800.,447.5],[0,0,1.]]],device='cuda')
    auxK=K.clone();auxK[:,:2]*=.25;auxK[:,0,2]-=.375;auxK[:,1,2]-=.375
    for condition,(primary,aux) in CONDITIONS.items():
        model=MultiSensorMultiHMR(base,condition).cuda().eval()
        inputs=dict(K=K)
        if primary.startswith('event'):inputs['event']=torch.zeros(1,2,224,224,device='cuda')
        else:inputs['image']=torch.zeros(1,3,896,896,device='cuda')
        if aux:inputs.update(auxiliary=torch.zeros(1,2 if aux.startswith('event') else 3,224,224,device='cuda'),
            auxiliary_K=auxK,primary_to_aux_rotation=torch.eye(3,device='cuda')[None],
            primary_to_aux_translation=torch.tensor([[.3,0.,0.]],device='cuda'),auxiliary_available=torch.ones(1,device='cuda'))
        torch.cuda.reset_peak_memory_stats();begin=time.time()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            people=model(**inputs,is_training=False,det_thresh=.3,nms_kernel_size=3)
        torch.cuda.synchronize()
        assert all(torch.isfinite(person['v3d']).all() for person in people)
        results.append(dict(condition=condition,seconds=time.time()-begin,predictions=len(people),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),passed=True))
        print(json.dumps(results[-1]),flush=True)
        if hasattr(model,'_hook_handle'):model._hook_handle.remove()
        del model,inputs,people
    save(OUT/'model_gpu_preflight.json',dict(status='passed',
        description='합성 입력의 GPU 실행 사전검사이며 실제 데이터 학습·평가 성능이 아님',
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],physical_gpu=os.environ['EVENTPOSE_PHYSICAL_GPU'],
        cuda_name=torch.cuda.get_device_name(),results=results,started_unix=started,completed_unix=time.time()))


def training_graph():
    from scripts.train_multihmr import configure_model
    from multihmr.multisensor_model import checkpoint_frozen_blocks
    configure_torch();torch.manual_seed(261026);base=load_multihmr()
    configure_model(base,8);checkpoint_frozen_blocks(base,8)
    batch=16;results=[]
    K=torch.tensor([[[800.,0,447.5],[0,800.,447.5],[0,0,1.]]],device='cuda').expand(batch,-1,-1)
    auxK=K.clone();auxK[:,:2]*=.25;auxK[:,0,2]-=.375;auxK[:,1,2]-=.375
    idx=(torch.arange(batch,device='cuda'),torch.full((batch,),32,device='cuda',dtype=torch.long),
         torch.full((batch,),32,device='cuda',dtype=torch.long),torch.zeros(batch,device='cuda',dtype=torch.long))
    for condition,(primary,aux) in CONDITIONS.items():
        base.zero_grad(set_to_none=True);model=MultiSensorMultiHMR(base,condition).cuda().train()
        inputs=dict(K=K)
        if primary.startswith('event'):inputs['event']=torch.rand(batch,2,224,224,device='cuda')*.2
        else:inputs['image']=torch.rand(batch,3,896,896,device='cuda')-.5
        if aux:inputs.update(auxiliary=torch.rand(batch,2 if aux.startswith('event') else 3,224,224,device='cuda')*.2,
            auxiliary_K=auxK,primary_to_aux_rotation=torch.eye(3,device='cuda')[None].expand(batch,-1,-1),
            primary_to_aux_translation=torch.tensor([[.3,0.,0.]],device='cuda').expand(batch,-1),auxiliary_available=torch.ones(batch,device='cuda'))
        torch.cuda.reset_peak_memory_stats();begin=time.time()
        with torch.autocast('cuda',dtype=torch.float16):output=model(**inputs,idx=idx,is_training=True)
        loss=output['v3d'].float().square().mean()*.001+output['scores'].float().mean()
        assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        if primary.startswith('event'):assert model.adapter.projection.weight.grad is not None
        results.append(dict(condition=condition,batch_size=batch,seconds=time.time()-begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite_gradients=True,passed=True))
        print(json.dumps(results[-1]),flush=True)
        if hasattr(model,'_hook_handle'):model._hook_handle.remove()
        del model,inputs,output,loss,gradients
    save(OUT/'training_graph_gpu_preflight.json',dict(status='passed',
        description='합성 입력의 batch16 AMP forward/backward 검사. optimizer update 및 실제 데이터 학습이 아님.',
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],physical_gpu=os.environ['EVENTPOSE_PHYSICAL_GPU'],results=results,completed_unix=time.time()))


if __name__=='__main__':
    import argparse
    import subprocess
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-gpu',action='store_true');parser.add_argument('--training',action='store_true');args=parser.parse_args()
    if args.wait_for_gpu:
        deadline=json.loads((ROOT/'_agents/research/261026/device34_body_goal_28h.json').read_text())['deadline_unix']
        chosen=False
        while time.time()<deadline-300 and not (OUT/'source_audit_complete.json').exists():
            for gpu in range(2,7):
                p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                if p.returncode or p.stdout.strip():continue
                p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
                if p.returncode:continue
                os.environ.update(CUDA_VISIBLE_DEVICES=p.stdout.strip(),EVENTPOSE_PHYSICAL_GPU=str(gpu))
                chosen=True;break
            if chosen:break
            time.sleep(30)
        if not chosen:sys.exit(0)
    if args.training:training_graph()
    else:main()

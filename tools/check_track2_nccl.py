"""Small post-fault collective check on the authorized training pairs."""
import os
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group(os.environ.get('TRACK2_DISTRIBUTED_BACKEND','nccl'),timeout=timedelta(seconds=45))
x=torch.tensor([rank+1.],device='cuda')
dist.all_reduce(x)
assert x.item()==3.
model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4,2).cuda(rank),device_ids=[rank])
optimizer=torch.optim.SGD(model.parameters(),lr=.01)
for step in range(3):
 optimizer.zero_grad();loss=model(torch.ones(3,4,device='cuda')*(rank+1)).square().mean();loss.backward();optimizer.step()
parameters=torch.cat([p.detach().flatten() for p in model.parameters()]);other=parameters.clone();dist.broadcast(other,0)
assert torch.equal(parameters,other)
records=[None,None];dist.all_gather_object(records,dict(rank=rank,step=3));assert len(records)==2

print(dict(rank=rank,all_reduce=x.item(),gpu=torch.cuda.get_device_name(rank),checked_unix=time.time()),flush=True)
dist.destroy_process_group()

"""Check every exported tensor against its original or adapted source."""
from pathlib import Path
import argparse
import json
import torch
import torchvision

ROOT=Path(__file__).resolve().parents[2]

def main(a):
    original=torch.jit.load(str(ROOT/'data/pretrained/nlf/nlf_l_multi.torchscript'),map_location='cpu')
    exported=torch.jit.load(str(a.checkpoint),map_location='cpu')
    delta=torch.load(a.delta,map_location='cpu')
    metadata=json.loads(a.checkpoint.with_suffix('.json').read_text())
    assert metadata['test_used_for_selection'] is False
    assert metadata['step']==delta['step'] and delta['successful_steps']>0
    old=original.state_dict();new=exported.state_dict();assert set(old)==set(new)
    adapted={f'crop_model.{k}':v for k,v in delta['model_state_dict'].items()}
    assert set(adapted)<=set(new)
    changed=0
    for name,value in new.items():
        target=adapted.get(name,old[name]).to(value.dtype)
        assert torch.equal(value,target),name
        if not torch.equal(value,old[name]):changed+=1
    assert changed>0,'Export rounded every adapted parameter back to its original value'
    assert set(exported._c._method_names())==set(original._c._method_names())
    result=dict(status='passed',step=delta['step'],successful_steps=delta['successful_steps'],
                tensors_checked=len(new),adapted_tensors=len(adapted),changed_tensors=changed,
                exported_checkpoint=str(a.checkpoint),test_used_for_selection=False)
    a.checkpoint.with_suffix('.verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--delta',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);main(p.parse_args())

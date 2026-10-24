from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Export a Train-holdout-selected NLF delta to the original inference API."""
import argparse
import json
from pathlib import Path
import torch
import torchvision
from nlf.data import ROOT

def main(a):
    delta=torch.load(a.delta,map_location='cpu')
    assert delta['successful_steps']>0
    full=torch.jit.load(str(ROOT/'data/pretrained/nlf/nlf_l_multi.torchscript'),map_location='cpu')
    original=full.crop_model.state_dict()
    assert all(k in original and original[k].shape==v.shape for k,v in delta['model_state_dict'].items())
    full.crop_model.load_state_dict(delta['model_state_dict'],strict=False)
    # No inferred canonical weights have been cached in this fresh model.
    # The pretrained field stays frozen in adaptation.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.tmp.torchscript')
    torch.jit.save(full,str(temporary));temporary.replace(a.output)
    metadata=dict(source_delta=str(a.delta),checkpoint=str(a.output),step=delta['step'],
                  config=delta['config'],selection_pa_mm=delta['selection_pa_mm'],test_used_for_selection=False)
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--delta',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())

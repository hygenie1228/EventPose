"""Released zero-update RGB initialization, separately labelled from fine-tuning."""
import hashlib
import json
from pathlib import Path
import torch


def released_rgb_model(config,checkpoint,device='cuda'):
    from utils.pose_evaluation import load_multihmr
    from .multisensor_repair import build_repair_model
    assert config['weight_stage']=='released_pretrained'
    assert config['condition'] in ('rgb3','rgb4') and config['auxiliary_sensor'] is None
    saved=torch.load(checkpoint,map_location='cpu')
    assert saved['config']==config and saved['step']==saved['successful_steps']==0
    assert saved['model_delta']=={} and saved['test_used_for_selection'] is False
    manifest=json.loads((Path(config['output_root'])/'original_pretrained_manifest.json').read_text())
    assert saved['released_checkpoint']==manifest['path']
    assert saved['released_checkpoint_sha256']==manifest['sha256']
    digest=hashlib.sha256()
    with Path(manifest['path']).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    assert digest.hexdigest()==manifest['sha256']
    torch.manual_seed(config['seed'])
    return build_repair_model(load_multihmr(device=device),config).to(device).eval()

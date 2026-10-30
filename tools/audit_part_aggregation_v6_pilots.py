"""Audit completed V6 CUDA pilots and reconstruct their real models on CPU."""
import argparse,gc,hashlib,json,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.part_aggregation_v6 import build_part_model
def read(path):return json.loads(path.read_text())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=['v6','v6_long'],default='v6')
    family=parser.parse_args().family
    torch.set_num_threads(2);out=ROOT/'exp/261029/part_aware_dynamic_aggregation';rows=[]
    for suffix in ['fixed','global_dynamic','part_dynamic','part_static']:
        name='pilot_'+family+'_'+suffix;run=out/'runs'/name;complete=read(run/'complete.json');cfg=read(out/'configs'/(name+'.json'))
        assert complete['status']=='pilot_complete' and complete['successful_steps']==8
        gradients=[json.loads(line) for line in (run/'pilot_gradient_diagnostics.jsonl').read_text().splitlines()]
        labels=['event_encoder','motion_head']
        if suffix!='fixed':labels+=['time_score','event_use_head']
        maxima={label:max(row.get(label+'_gradient_l2',0) for row in gradients) for label in labels}
        assert all(value>0 for value in maxima.values()),(name,maxima)
        path=Path(complete['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert saved['successful_steps']==8 and saved['test_used_for_selection'] is False
        assert all(key.startswith('fusion.') and not key.startswith('fusion.encoder.raw_encoder.') for key in saved['model_delta'])
        model=build_part_model(cfg,'cpu',checkpoint=path).eval()
        assert model.fusion.encoder.raw_encoder is model.base.backbone.encoder
        assert not model.fusion.training and not model.fusion.encoder.raw_encoder.training
        parameters=dict(model.named_parameters())
        for key,value in saved['model_delta'].items():assert torch.equal(parameters[key],value)
        reference=torch.load(cfg['frozen_RGB_checkpoint'],map_location='cpu',weights_only=False,mmap=True);state=model.base.state_dict();count=0
        for key,value in reference['model_delta'].items():
            if key.startswith('base.'):
                assert torch.equal(state[key.removeprefix('base.')],value);count+=1
        rows.append(dict(name=name,actual_CUDA_successful_updates=8,actual_CUDA_gradient_maxima=maxima,
            checkpoint_sha256=sha(path),actual_full_CPU_model_reconstruction=True,selected_parameter_tensors_exact=len(saved['model_delta']),
            frozen_RGB_parameter_tensors_exact=count,shared_inner_ViT_identity=True,inference_modules_eval=True))
        del model,saved,reference,state,parameters;gc.collect()
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),pilots=rows,
        scope='Existing actual CUDA pilot records plus real full-model CPU checkpoint reconstruction and parameter equality; no new GPU work',
        limitation='No CPU full pose forward or CPU/CUDA prediction equivalence check; pilot accuracy is not main/full Test accuracy')
    (out/(family+'_verified_cuda_pilots_and_full_CPU_reload.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

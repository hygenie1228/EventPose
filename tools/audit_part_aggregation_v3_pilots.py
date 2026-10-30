"""Audit completed CUDA pilots and reload their learned fusion weights on CPU."""
import argparse
import gc
import hashlib
import io
import json
import sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.part_aggregation_v3 import MotionHorizonAggregation
from check_part_aggregation_v2 import context

def read(path):return json.loads(path.read_text())

def main(version='v3'):
    torch.set_num_threads(2);torch.manual_seed(261029)
    out=ROOT/'exp/261029/part_aware_dynamic_aggregation';rows=[]
    conditions=['rgb_only','fixed','global_dynamic','part_dynamic'] if version=='v3' else ['fixed','global_dynamic','part_dynamic','part_static']
    for condition in conditions:
        name='pilot_'+version+'_'+condition;run=out/'runs'/name;complete=read(run/'complete.json')
        assert complete['status']=='pilot_complete' and complete['successful_steps']==8
        gradients=[json.loads(line) for line in (run/'pilot_gradient_diagnostics.jsonl').read_text().splitlines()]
        required=['motion_head']
        if condition!='rgb_only':required+=['event_encoder']
        if condition in ['global_dynamic','part_dynamic','part_static']:required+=['time_score','event_use_head']
        maxima={label:max(row.get(label+'_gradient_l2',0) for row in gradients) for label in required}
        assert all(value>0 for value in maxima.values()),(name,maxima)
        path=Path(complete['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        cfg=saved['config'];assert saved['successful_steps']==8 and saved['metrics_step']==8
        assert saved['test_used_for_selection'] is False and saved['inference_uniform_warmup_disabled']
        versions=dict(v3='calibrated_part_motion_horizon_v3',v4='calibrated_part_frozen_rgb_v4',v5='calibrated_part_dense_frozen_rgb_v5')
        assert cfg['architecture_version']==versions[version]
        if version in ('v4','v5'):assert cfg['freeze_RGB_base'] and all(k.startswith('fusion.') for k in saved['model_delta'])
        delta={key.removeprefix('fusion.'):value for key,value in saved['model_delta'].items() if key.startswith('fusion.')}
        feature_dim=delta['query.weight'].shape[1]
        def construct():
            cls=MotionHorizonAggregation
            if version=='v5':
                from multihmr.part_aggregation_v5 import DenseSpatialAggregation
                cls=DenseSpatialAggregation
            return cls(feature_dim,cfg['hidden'],variant=cfg['variant'],
                support=cfg['spatial_support'],warmup_steps=cfg['uniform_aggregation_warmup_steps'],
                static_horizon=cfg['static_part_horizon'],path_prior=cfg['motion_path_prior_px']).eval()
        model=construct();load=model.load_state_dict(delta,strict=False)
        assert not load.unexpected_keys
        assert not set(load.missing_keys)&set(dict(model.named_parameters()))
        observations=context();features=torch.randn(2,1024 if version=='v5' else 256,feature_dim)
        with torch.no_grad():reference=model(features,observations);weights=model.last_time_weights.clone()
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        restored=construct();restored.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=True))
        with torch.no_grad():actual=restored(features,observations)
        assert torch.isfinite(actual).all() and torch.equal(actual,reference)
        assert torch.equal(restored.last_time_weights,weights) and not restored.last_warmup_active
        assert restored.successful_training_step==0
        rows.append(dict(name=name,actual_CUDA_successful_updates=8,checkpoint_successful_updates=saved['successful_steps'],
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual_CUDA_gradient_maxima=maxima,
            learned_fusion_CPU_reload_exact=True,eval_step0_uniform_warmup_disabled=True,
            runtime_training_source_sha256=cfg.get('runtime_training_source_sha256')))
        del saved,delta,model,restored,actual,reference,buffer;gc.collect()
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        pilots=rows,scope='Existing real-data CUDA pilot artifacts plus CPU fusion roundtrip on synthetic inputs; no new GPU inference, training or Test evaluation',
        limitation='Full Multi-HMR backbone reconstruction and CPU-vs-CUDA output equivalence were not tested')
    (out/(version+'_verified_cuda_pilots_and_fusion_reload.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',choices=['v3','v4','v5'],default='v3')
    main(parser.parse_args().version)

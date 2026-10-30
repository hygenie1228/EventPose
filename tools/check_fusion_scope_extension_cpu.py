"""Verify exact old-path code preservation and real new-layout native inputs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job,write


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_scope_extension_CPU')
    os.environ.update(EVENTPOSE_MULTISENSOR_OUT=goal['output_root'],XFORMERS_DISABLED='1',CUDA_VISIBLE_DEVICES='')
    import gc
    import numpy as np
    import torch
    from multihmr.multisensor_repair import build_repair_model,load_verified_initialization,implementation_provenance,REPAIR_LAYOUTS
    from multihmr.multisensor_data import Frames,collate,input_keys
    from multihmr.multisensor_inference import CachedSensors
    from multihmr.multisensor_model import self_check
    from multihmr.pretrained_inference import released_rgb_model
    from utils.pose_evaluation import load_multihmr
    torch.set_num_threads(2);out=Path(goal['output_root'])
    old=json.loads((out/'source_revision_pre_extension.json').read_text())['implementation_sha256']
    now=implementation_provenance(out/'source_snapshots');proofs=[]
    for name,digest in old.items():
        source=(out/'source_snapshots'/(digest+'.txt')).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==digest
        a=ast.parse(source);b=ast.parse((ROOT/name).read_text())
        if name in ['src/multihmr/multisensor_model.py','src/multihmr/multisensor_repair.py']:
            target='CONDITIONS' if name.endswith('multisensor_model.py') else 'REPAIR_LAYOUTS'
            node=next(n for n in b.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
            keep=[i for i,k in enumerate(node.value.keys) if k.value not in ['rgb3_event3','rgb4_event4']]
            node.value.keys=[node.value.keys[i] for i in keep];node.value.values=[node.value.values[i] for i in keep]
            if target=='REPAIR_LAYOUTS':
                function=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='implementation_provenance')
                for n in ast.walk(function):
                    if isinstance(n,ast.List):n.elts=[x for x in n.elts if not (isinstance(x,ast.Constant) and x.value=='src/multihmr/pretrained_inference.py')]
        elif name=='src/multihmr/multisensor_inference.py':
            function=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='load_model')
            assert isinstance(function.body[1],ast.If) and "weight_stage" in ast.unparse(function.body[1].test)
            function.body.pop(1)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
        proofs.append(dict(path=name,old_sha256=digest,current_sha256=now[name],old_forward_code_unchanged=True))
    checks=[]
    for name in ['rgb3_event3','rgb4_event4']:
        config=json.loads((out/'configs'/(name+'.json')).read_text());torch.manual_seed(config['seed'])
        data=Frames('training',condition=name,sensor_layout=REPAIR_LAYOUTS[name]);sample=data[0];batch=collate([sample])
        source=CachedSensors(data.records[0]['session'],name,sensor_layout=REPAIR_LAYOUTS[name])
        inference=source.sample(data.records[0]['frame'])
        for key,value in inference.items():assert torch.equal(value,batch[key][0]),key
        source.close()
        base=load_multihmr(device='cpu');model=build_repair_model(base,config).eval();init=load_verified_initialization(model,config)
        inputs=input_keys(batch)
        with torch.inference_mode():
            raw=base.backbone(inputs['image']);original=base.backbone.forward;base.backbone.forward=lambda *_args,**_kwargs:raw
            try:
                a=model(**inputs,is_training=False,det_thresh=.3,nms_kernel_size=3)
                b=model(**dict(inputs,auxiliary_available=torch.zeros_like(inputs['auxiliary_available'])),is_training=False,det_thresh=.3,nms_kernel_size=3)
            finally:base.backbone.forward=original
        assert len(a)==len(b)>0
        assert all(torch.equal(x['v3d'],y['v3d']) for x,y in zip(a,b))
        checks.append(dict(condition=name,layout=REPAIR_LAYOUTS[name],row=data.records[0],
                           native_initial_prediction_equal=True,pixel_input_parity=True,initialization=init))
        for handle in data.handles.values():handle.close()
        model._hook_handle.remove();del base,model,data,raw,a,b,batch,inputs;gc.collect()
    released=[]
    for name in ['pre_rgb3','pre_rgb4']:
        config=json.loads((out/'configs'/(name+'.json')).read_text())
        model=released_rgb_model(config,out/'checkpoints'/name/'best.pt',device='cpu')
        assert model.auxiliary_sensor is None and model.primary_sensor==config['condition']
        released.append(dict(name=name,zero_ETRI_updates=True,verified_original_source=True))
        del model;gc.collect()
    write(out/'same_device_extension_cpu.json',dict(status='passed',checked_unix=time.time(),records=checks,
          old_path_AST_equivalence=proofs,released_baselines=released,original_self_check=self_check(),
          Test_used=False,scope='Real Train initialization and old-path source equivalence; GPU pilots still required'))
    write(out/'source_revision_scope_extension.json',dict(status='passed',reason='User-authorized same-device layouts and zero-update source branch',
          approved_implementations=[old,now],AST_equivalence=proofs,checked_unix=time.time(),
          interpretation='Existing model forward paths unchanged; dictionary extensions, provenance list extension and a distinct released-pretrained loader branch only'))
    print(json.dumps(dict(status='passed',new_conditions=2,released_baselines=2,old_path_equivalence=True)),flush=True)


if __name__=='__main__':main()

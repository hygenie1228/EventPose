"""Native original-weight RGB inference on a frozen Train frame; no Test scoring."""
import json
import os
from pathlib import Path
import sys
import gc
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job,write


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_released_RGB_native_CPU')
    os.environ.update(EVENTPOSE_MULTISENSOR_OUT=goal['output_root'],CUDA_VISIBLE_DEVICES='',XFORMERS_DISABLED='1')
    import torch
    from multihmr.pretrained_inference import released_rgb_model
    from multihmr.multisensor_data import Frames,collate,input_keys
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    torch.set_num_threads(2);out=Path(goal['output_root']);records=[]
    for name in ['pre_rgb3','pre_rgb4']:
        config=json.loads((out/'configs'/(name+'.json')).read_text())
        model=released_rgb_model(config,out/'checkpoints'/name/'best.pt',device='cpu')
        data=Frames('training',condition=config['condition'],sensor_layout=REPAIR_LAYOUTS[config['condition']])
        batch=collate([data[0]]);inputs=input_keys(batch)
        assert set(inputs)=={'K','image'}
        with torch.inference_mode():
            features=model.base.backbone(inputs['image']);assert features.shape==(1,4096,1024)
            original=model.base.backbone.forward;model.base.backbone.forward=lambda *_args,**_kwargs:features
            try:
                people=model(**inputs,is_training=False,det_thresh=.3,nms_kernel_size=3)
                direct=model.base(inputs['image'],K=inputs['K'],is_training=False,det_thresh=.3,nms_kernel_size=3)
            finally:model.base.backbone.forward=original
        assert len(people)==len(direct)
        assert all(torch.isfinite(p['v3d']).all() and torch.equal(p['v3d'],q['v3d']) for p,q in zip(people,direct))
        records.append(dict(name=name,row=data.records[0],native_detections=len(people),
                            direct_released_base_prediction_equal=True,ETRI_finetune_updates=0,
                            GT_used_as_input=False,Test_used=False))
        for h in data.handles.values():h.close()
        del model,data,features,people,direct,batch,inputs;gc.collect()
    write(out/'released_RGB_native_cpu.json',dict(status='passed',checked_unix=time.time(),records=records,
        scope='Actual full native original-weight Train-frame inference, not Test metrics'))
    print(json.dumps(dict(status='passed',records=records)),flush=True)


if __name__=='__main__':main()

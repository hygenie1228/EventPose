"""Check the optional native MHR Train-target path without using a GPU."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from scripts import train_multihmr as training
from multihmr.track2_data import OUTPUT,save_json
from utils.track2_metrics import paired_frame0_chamfer


def main():
    torch.set_num_threads(2)
    training.WORK=OUTPUT/'work'
    plan=OUTPUT/'fixed_view_training_plan.json'
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    checked={}
    for role,expected in [('training',851),('validation',256),('full_training',1107)]:
        dataset=training.Frames(role,fixed_view_plan=plan,native_surface_target=True)
        assert len(dataset)==expected
        assert {r['session'] for r in dataset.records}<=set(manifest['train'])
        assert not {r['session'] for r in dataset.records}&set(manifest['test'])
        for i,record in enumerate(dataset.records):
            item=dataset[i];points=item['native_surface_points']
            assert points.shape==(1,91,3) and np.isfinite(points).all()
            batch=training.collate([item]);device_batch,gt=training.targets(batch,'cpu')
            assert device_batch['native_surface_points'].shape==(1,1,91,3)
            assert len(gt['idx'][0])==1
            with np.load(training.WORK/'native_surface_targets'/f'{record["session"]}.npz') as exact:
                assert exact['coordinate_frame'].item()=='opencv_camera'
                assert exact['frames'][record['row']]==record['frame']
                assert np.array_equal(points[0],exact['surface_points_cam_m'][record['row']])
        checked[role]=dict(frames=len(dataset),all_points_exact=True,all_centers_visible=True)
    native=training.Frames('validation',fixed_view_plan=plan,native_surface_target=True)
    default=training.Frames('validation',fixed_view_plan=plan)
    for i in [0,64,128,255]:
        a,b=native[i],default[i]
        assert 'native_surface_points' not in b
        for key in b:
            assert torch.equal(a[key],b[key]) if torch.is_tensor(b[key]) else np.array_equal(a[key],b[key])
    pairs=training.AnchorFrames('training',fixed_view_plan=plan,native_surface_target=True)
    selected=[10,100,200,400]
    items=[pairs[i] for i in selected]
    for i,(_,anchor) in zip(selected,items):
        record=pairs.records[pairs.time_indices[i]]
        assert pairs.records[anchor['index']]['session']==record['session']
        assert pairs.records[anchor['index']]['frame']==0
    batch=training.collate_anchors(items)
    points=batch['native_surface_points'][:,0]
    generator=torch.Generator().manual_seed(261025)
    predicted=(points+torch.randn(points.shape,generator=generator)*.005).requires_grad_()
    loss=paired_frame0_chamfer(predicted,points,len(items));loss.backward()
    assert torch.isfinite(loss) and loss>0 and torch.isfinite(predicted.grad).all()
    assert predicted.grad[:len(items)].abs().sum()>0 and predicted.grad[len(items):].abs().sum()>0
    result=dict(checked=checked,default_dataset_unchanged=True,paired_frame0_same_clip=True,
        loss_m=float(loss.detach()),time_gradient_l1=float(predicted.grad[:len(items)].abs().sum()),
        anchor_gradient_l1=float(predicted.grad[len(items):].abs().sum()),
        GPU_used=False,Test_labels_read=False,actual_two_GPU_pilot_still_required=True,
        completed_unix=time.time())
    save_json(OUTPUT/'native_surface_dataset_checks.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

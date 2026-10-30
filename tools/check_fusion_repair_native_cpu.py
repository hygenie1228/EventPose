"""Real Train-frame native prediction/initial RGB preservation check on CPU."""
import gc
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write


def main():
    goal = json.loads((ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal, os.getpid(), 'fusion_repair_native_cpu_initialization')
    os.environ['EVENTPOSE_MULTISENSOR_OUT'] = goal['output_root']
    os.environ['XFORMERS_DISABLED'] = '1'
    import numpy as np
    import torch
    from utils.pose_evaluation import load_multihmr, atomic_npz
    from multihmr.multisensor_data import Frames, collate, input_keys
    from multihmr.multisensor_repair import build_repair_model, REPAIR_LAYOUTS, load_verified_initialization
    torch.set_num_threads(2)
    out = Path(goal['output_root']); records = []
    for name in ('rgb3_event4', 'event3_rgb4', 'event3_event4'):
        print(json.dumps(dict(phase='loading_real_model_cpu', condition=name)), flush=True)
        config = json.loads((out / 'configs' / (name + '.json')).read_text())
        torch.manual_seed(config['seed'])
        base = load_multihmr(device='cpu')
        model = build_repair_model(base, config).eval()
        provenance = load_verified_initialization(model, config)
        data = Frames('training', condition=name, sensor_layout=REPAIR_LAYOUTS[name])
        batch = collate([data[0]]); inputs = input_keys(batch)
        if name == 'event3_event4':
            assert 'image' not in inputs and model.primary_sensor == 'event3'
        with torch.inference_mode():
            image = inputs['image'] if model.mode == 'rgb' else (model.adapter(inputs['event'], base.img_size)-model.mean)/model.std
            started = time.time(); raw_features = base.backbone(image)
            seconds = time.time()-started
            print(json.dumps(dict(phase='native_backbone_cpu_complete', condition=name, seconds=seconds)), flush=True)
            assert torch.isfinite(raw_features).all()
            original_forward = base.backbone.forward
            base.backbone.forward = lambda *_args, **_kwargs: raw_features
            try:
                predictions = model(**inputs, is_training=False, det_thresh=.3, nms_kernel_size=3)
                missing = dict(inputs, auxiliary_available=torch.zeros_like(inputs['auxiliary_available']))
                baseline = model(**missing, is_training=False, det_thresh=.3, nms_kernel_size=3)
            finally:
                base.backbone.forward = original_forward
            assert len(predictions) == len(baseline)
            for a, b in zip(predictions, baseline):
                assert torch.isfinite(a['v3d']).all() and torch.equal(a['v3d'], b['v3d'])
            if model.mode == 'rgb':
                assert predictions, 'No native RGB detection on the frozen Train diagnostic frame'
            reg = torch.from_numpy(np.load(out / 'joint_regressor.npy'))
            joints = np.stack([(reg @ person['v3d']).numpy() for person in predictions]) if predictions else np.empty((0,55,3))
        dest = out / 'diagnostics' / (name + '_initial_train_prediction.npz'); dest.parent.mkdir(exist_ok=True)
        atomic_npz(dest, predicted_joints_primary_camera_m=joints,
                   gt_joints_primary_camera_m=batch['joints'].numpy(), gt_valid=batch['valid'].numpy())
        records.append(dict(condition=name, primary=model.primary_sensor, auxiliary=model.auxiliary_sensor,
                            row=data.records[0], native_CPU_backbone_shape=list(raw_features.shape),
                            backbone_seconds=seconds, native_detections=len(predictions),
                            initial_auxiliary_present_vs_absent_vertices_equal=True,
                            initialization=provenance, inference_GT_input=False,
                            source_role='training', saved_prediction=str(dest)))
        for handle in data.handles.values():
            handle.close()
        model._hook_handle.remove()
        del raw_features, predictions, baseline, model, base, data, image, batch, inputs
        gc.collect()
    write(out / 'native_cpu_preflight.json', dict(status='passed', checked_unix=time.time(), records=records,
          scope='Real Train-frame full native model on CPU; backbone cached for on/off comparison; not GPU speed or Test performance'))
    print(json.dumps(dict(status='passed', conditions=len(records))), flush=True)


if __name__ == '__main__':
    main()

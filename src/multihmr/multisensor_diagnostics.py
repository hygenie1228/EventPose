"""Train-holdout modality diagnostics; GT is used only after native inference."""
import numpy as np
import torch
from utils.humanpose_metrics import joint_errors, match_people
from .multisensor_data import input_keys


def guard_diagnostic_detections(value, batch_indices, maximum_people):
    """Reject overloaded diagnostic frames, never choose a top-k subset."""
    scores, detected, indices = value
    overloaded = []
    for image, frame in enumerate(batch_indices):
        count = int((indices[0] == image).sum().item())
        if count > maximum_people:
            overloaded.append(dict(frame_index=int(frame), image_in_batch=image, native_candidates=count))
    if overloaded:
        keep = torch.ones_like(indices[0], dtype=torch.bool)
        for row in overloaded:
            keep &= indices[0] != row['image_in_batch']
        value = scores, detected[keep], tuple(index[keep] for index in indices)
    return value, overloaded


@torch.no_grad()
def holdout_modality_errors(model, loader, regressor, mode='intact', maximum_people_per_image=None):
    assert mode in ('intact', 'missing_aux', 'zero_aux', 'swap_aux_polarity')
    device = next(model.parameters()).device
    original = model.base.detection; image_ids = []; batch_indices = []
    resource_guarded_frames = []; guarded_gt_instances = 0
    def detection(*args, **kwargs):
        value = original(*args, **kwargs)
        if maximum_people_per_image is not None:
            value, overloaded = guard_diagnostic_detections(value, batch_indices, maximum_people_per_image)
            resource_guarded_frames.extend(overloaded)
        image_ids[:] = value[2][0].detach().cpu().tolist()
        return value
    model.eval(); model.base.detection = detection
    errors = {}; gt_keys = []; observed_input_keys = set()
    try:
        for batch in loader:
            batch_indices[:] = list(batch['index'])
            previous_guarded_count = len(resource_guarded_frames)
            inputs = {k: v.to(device) for k, v in input_keys(batch).items()}
            observed_input_keys.update(inputs)
            assert not set(inputs) & {'joints', 'centers', 'valid', 'betas', 'rotations'}
            if model.mode == 'event':
                assert 'image' not in inputs
            if mode == 'missing_aux':
                inputs['auxiliary_available'] = torch.zeros_like(inputs['auxiliary_available'])
            elif mode == 'zero_aux':
                inputs['auxiliary'] = torch.zeros_like(inputs['auxiliary'])
            elif mode == 'swap_aux_polarity':
                assert inputs['auxiliary'].shape[1] == 2
                inputs['auxiliary'] = inputs['auxiliary'].flip(1)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                people = model(**inputs, is_training=False, det_thresh=.3, nms_kernel_size=3)
            guarded_gt_instances += sum(int(batch['valid'][row['image_in_batch']].sum())
                                        for row in resource_guarded_frames[previous_guarded_count:])
            assert len(people) == len(image_ids)
            grouped = [[] for _ in batch['index']]
            for image, person in zip(image_ids, people):
                joints = (regressor @ person['v3d'].float()).cpu().numpy()
                assert np.isfinite(joints).all()
                grouped[image].append(joints)
            for image, predictions in enumerate(grouped):
                target = batch['joints'][image].numpy(); valid = batch['valid'][image].numpy().astype(bool)
                K = batch['original_K'][image].numpy(); shape = batch['image_shape'][image].tolist()
                pred = np.stack(predictions) if predictions else np.empty((0, 55, 3), np.float32)
                mapping = match_people(pred, target, valid, K, K, shape)
                gt_keys.extend(str(batch['index'][image]) + ':' + str(i) for i in np.flatnonzero(valid))
                for person, detected in mapping.items():
                    key = str(batch['index'][image]) + ':' + str(person)
                    values = list(joint_errors(pred[detected, :22], target[person, :22]))
                    values.append(float(np.linalg.norm(pred[detected, 0] - target[person, 0])))
                    assert key not in errors
                    errors[key] = [float(v) * 1000 for v in values]
    finally:
        model.base.detection = original
    assert len(gt_keys) == len(set(gt_keys))
    return dict(mode=mode, total_gt_instances=len(gt_keys), matched_instances=len(errors),
                coverage=len(errors)/len(gt_keys), gt_keys=gt_keys, errors_mm=errors,
                maximum_people_per_image=maximum_people_per_image,
                resource_guarded_frames=resource_guarded_frames,
                resource_guarded_gt_instances=guarded_gt_instances,
                coverage_interpretation='Matched/total GT; includes explicitly recorded diagnostic resource failures, not native recall for guarded frames',
                input_keys=sorted(observed_input_keys), GT_used_as_inference_input=False)

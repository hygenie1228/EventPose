"""Resolve the frozen corrected comparison scope without reading Test scores."""
import json
from multihmr.track2_data import OUTPUT


def scope():
    body=OUTPUT/'body_target_evaluation_plan.json'
    if body.exists():
        extended_body=OUTPUT/'body_target_final_evaluation_plan.json'
        path=extended_body if extended_body.exists() else body
        plan=json.loads(path.read_text());models=plan['models'];names=[m['name'] for m in models]
        assert len(names)==len(set(names)) and len(names) in [9,12]
        assert all(m['evaluation_protocol'] in ['native_rgb_strict_mask_body_box_v3','native_rgb_static_crop_strict_mask_body_box_v3'] for m in models)
        return plan,'body_target_final' if extended_body.exists() else 'body_target'
    extended=OUTPUT/'strict_final_evaluation_plan.json'
    path=extended if extended.exists() else OUTPUT/'strict_evaluation_plan.json'
    plan=json.loads(path.read_text())
    models=plan['models'];names=[m['name'] for m in models]
    assert len(names)==len(set(names)) and len(names) in [9,12]
    assert all(m['evaluation_protocol'] in ['native_rgb_strict_mask_target_head_v2',
        'native_rgb_static_crop_strict_mask_target_head_v2'] for m in models)
    return plan,'strict_final' if extended.exists() else 'strict'

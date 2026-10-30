"""Train-only motion scale relative to event feature sampling stride."""
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'exp/261029/part_aware_dynamic_aggregation'
    config=json.loads((out/'configs/v4_part_dynamic.json').read_text())
    with np.load(out/'diagnostics/training_part_motion_targets.npz',allow_pickle=False) as f:
        assert bool(f['training_supervision_allowed']) and not bool(f['inference_input_allowed'])
        path=f['event3_part_path_px'];valid=f['valid'][:,None]&f['part_valid'];parts=f['part_names'].tolist()
        sessions=sorted(set(f['sessions'].tolist()))
    ratios=[]
    for name in sessions:
        with h5py.File(Path(config['source_root'])/'cache'/(name+'.h5'),'r',locking=False) as f:
            native=f['event3_K_native'][:];resized=f['event3_K_auxiliary'][:]
            ratio=resized[0,0]/native[0,0]
            assert np.isclose(ratio,resized[1,1]/native[1,1],atol=1e-7)
            ratios.append(float(ratio))
    assert np.allclose(ratios,ratios[0],atol=1e-7)
    scale=ratios[0];rows=[]
    for j,name in enumerate(parts):
        values=path[valid[:,j],j];quantiles=np.quantile(values,[.25,.5,.75])
        rows.append(dict(part=name,valid_queries=len(values),native_path_quantiles_px=quantiles.tolist(),
            path_quantiles_in_14px_feature_strides=(quantiles/(14/scale)).tolist(),
            path_quantiles_in_7px_feature_strides=(quantiles/(7/scale)).tolist()))
    result=dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        Train_only=True,Test_used=False,training_sessions=len(sessions),native_to_224_scale=scale,
        current_event_feature_grid=16,current_patch_stride_pixels=14,native_pixels_per_current_stride=14/scale,
        candidate_feature_grid=32,candidate_patch_stride_pixels=7,native_pixels_per_candidate_stride=7/scale,
        parts=rows,interpretation='200ms GT path divided by feature stride is a spatial-scale diagnostic, not net displacement, a proof of information loss, or evidence that a higher-resolution model improves pose')
    (out/'diagnostics/training_event_spatial_scale.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

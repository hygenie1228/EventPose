"""Attach training-only causal motion labels; model input whitelist is unchanged."""
from pathlib import Path
import numpy as np
from .part_aggregation_data import PartAggregationFrames

class PartMotionFrames(PartAggregationFrames):
    def __init__(self,role,*args,**kwargs):
        super().__init__(role,*args,**kwargs)
        filename='training_part_motion_targets.npz' if role=='training' else 'validation_part_motion_training_targets.npz'
        with np.load(Path(self.cache_root)/'diagnostics'/filename,allow_pickle=False) as f:
            self.motion={k:f[k] for k in f.files}
        assert not bool(self.motion['inference_input_allowed'])
        assert bool(self.motion['training_supervision_allowed'])==(role=='training')
        self.motion_rows={(str(s),int(t)):i for i,(s,t) in enumerate(zip(self.motion['sessions'],self.motion['frames']))}
        assert all((r['session'],int(r['frame'])) in self.motion_rows for r in self.records)
    def __getitem__(self,index):
        result=super().__getitem__(index);record=self.records[index];i=self.motion_rows[(record['session'],int(record['frame']))]
        path=self.motion['event3_part_path_px'][i].astype(np.float32)
        uv=self.motion['event3_part_displacement_uv_px'][i].astype(np.float32)
        valid=self.motion['part_valid'][i].astype(bool)&bool(self.motion['valid'][i])
        valid&=np.isfinite(path)&np.isfinite(uv).all(-1)
        result.update(motion_gt_path_px=np.nan_to_num(path),motion_gt_uv_px=np.nan_to_num(uv),motion_gt_valid=valid)
        return result

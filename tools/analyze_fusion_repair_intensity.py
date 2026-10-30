"""Train-only contrast statistics and a frozen, label-free encoder input gain."""
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from goal_runtime_control import register_job,write


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_Train_intensity_statistics')
    import h5py
    import numpy as np
    out=Path(goal['output_root']);index=json.loads((out/'training_index.json').read_text())
    assert not list((out/'runs').glob('*/started.json')),'Input normalization is frozen before all GPU trials'
    assert len(index)==6501
    hist={s:np.zeros(256,np.int64) for s in ('event3','event4')}
    for session in sorted({r['session'] for r in index}):
        rows=sorted(r['cache_row'] for r in index if r['session']==session)
        with h5py.File(out/'cache'/(session+'.h5'),'r',locking=False) as f:
            for sensor in hist:
                pixels=f[sensor][rows]
                hist[sensor]+=np.bincount(pixels.reshape(-1),minlength=256)
    records={};gains={}
    for sensor,h in hist.items():
        cdf=np.cumsum(h[1:]);assert cdf[-1]>0
        values=[int(np.searchsorted(cdf,q*(int(cdf[-1])-1)+1))+1 for q in (0,.25,.5,.75,.95,.99,1)]
        target=.5/(values[2]/255)
        gain=float(2**min(3,max(0,round(math.log2(target)))))
        gains[sensor]=gain
        records[sensor]=dict(histogram_u8=h.tolist(),nonzero_quantiles_u8=values,
                             quantile_probabilities=[0,.25,.5,.75,.95,.99,1],
                             active_fraction=float(h[1:].sum()/h.sum()),fixed_gain=gain)
    write(out/'event_normalization_Train.json',dict(status='passed',checked_unix=time.time(),frames_per_sensor=6501,
          training_index_sha256=hashlib.sha256((out/'training_index.json').read_bytes()).hexdigest(),
          split_sha256=hashlib.sha256((out/'split_snapshot.json').read_bytes()).hexdigest(),
          records=records,event_encoder_gains=gains,
          rule='Nearest power-of-two gain in [1,8] bringing the Train median nonzero encoded pixel toward .5; no clipping',
          scope='Fixed Train subset only; no labels, holdout or Test used. Gain applies in learned encoders/refiner, not to the warmstarted projection path.'))
    print(json.dumps(dict(status='passed',gains=gains,quantiles={s:r['nonzero_quantiles_u8'] for s,r in records.items()})),flush=True)


if __name__=='__main__':main()

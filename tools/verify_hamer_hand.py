"""Require real checkpoints, full Test, paired metrics and decodable figures."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261028/hamer_hand'


def main():
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from hamer.event_model import file_sha256
    from report_hamer_hand import main as report,NAMES
    from prepare_hamer_hand import write
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['split_sha256']==file_sha256(ROOT/'dataset/split.json')
    groups=[set(manifest[k]) for k in ('train_subjects','validation_subjects','test_subjects')]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    for n in NAMES[1:]:
        r=json.loads((OUT/'runs'/n/'complete.json').read_text());assert r['status']=='complete' and r['successful_steps']>0
        cfg=json.loads((OUT/'configs'/(n+'.json')).read_text());assert r['elapsed_seconds']>=cfg['hours']*3600-5
        ckpt=torch.load(OUT/'checkpoints'/n/'best.pt',map_location='cpu',weights_only=False)
        assert ckpt['test_used_for_selection'] is False and ckpt['metrics_step']==ckpt['step'] and ckpt['successful_steps']>0
        assert all(torch.isfinite(v).all() for v in ckpt['model_delta'].values())
    for n in NAMES:
        for session in manifest['test_sessions']:
            c=json.loads((OUT/'results'/n/session/'complete.json').read_text())
            assert c['full_test_intervals'] and c['root_position_error_computed'] is False
            with h5py.File(OUT/'cache'/(session+'.h5'),'r',locking=False) as f:
                assert c['processed_samples']==len(f['frame']) and f.attrs['cache_version']==2
                with np.load(OUT/'results'/n/session/'predictions.npz') as z:
                    assert np.array_equal(z['target_joints_cam_m'],f['joints'][:])
                    assert z['pred_joints_cam_m'].shape==z['target_joints_cam_m'].shape
                    assert np.isfinite(z['pred_joints_cam_m']).all() and np.isfinite(z['errors_m']).all()
        with Image.open(OUT/'visualizations'/(n+'.png')) as image:
            image.load();assert image.width==1024 and image.height==560
    result=report();assert result['complete'] and result['common_samples']>0
    write(OUT/'final_verification.json',dict(status='passed',conditions=NAMES,full_Test=True,
        matched_GT_samples=result['common_samples'],checkpoints_and_figures_verified=True,
        root_metrics=False,completed_unix=time.time()))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

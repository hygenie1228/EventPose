"""Verify only whole Test10 results; never choose a model from Test scores."""
from pathlib import Path
import argparse
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,sha256,save_json


def read(path):return json.loads(Path(path).read_text())


def main(new_only=False):
    manifest=read(OUTPUT/'manifest.json');plan=read(OUTPUT/'native_evaluation_plan.json')
    initial=read(OUTPUT/'init_comparison.json')
    reference=initial['models']['init_v6']['per_sequence'][manifest['test'][0]]['protocol']
    path=OUTPUT/'native_completed_Test_models.json'
    result=read(path) if path.exists() else dict(models={},whole_Test10_only=True,test_used_for_selection=False)
    added=[]
    for name in [plan['legacy_model']]+plan['trials']:
        if new_only and name in result['models']:continue
        if not all((OUTPUT/'predictions'/name/s/'metrics.json').exists() for s in manifest['test']):continue
        if name==plan['legacy_model']:checkpoint=Path(plan['legacy_full_train']['checkpoint'])
        else:checkpoint=Path(read(OUTPUT/'native_refit_choices'/(name+'.json'))['refit']['checkpoint'])
        checkpoint_sha=sha256(checkpoint);values={};coverage=[]
        for sample in manifest['test']:
            folder=OUTPUT/'predictions'/name/sample
            value=read(folder/'metrics.json');info=read(folder/'inference.json')
            assert value['frames']==manifest['samples'][sample]['frames'] and value['full_frame_roster']
            assert value['conversion_revision']==2 and value['protocol']==reference
            assert value['evaluation_protocol']==info['evaluation_protocol']=='native_rgb_target_head_v1'
            assert value['gt_sha256']==manifest['samples'][sample]['gt_sha256']
            assert value['prediction_sha256']==sha256(folder/'human_params_mhr_v2.npz')
            assert info['checkpoint_sha256']==checkpoint_sha and info['input_view']=='native_rgb'
            assert not info['target_foreground_only_input'] and info['all_accepted_heads_in_target_mask']
            assert not info['gt_used_for_detection_or_tracking']
            assert info.get('depth_decoder','published')=='published'
            assert info['target_mask_sha256']==sha256(SOURCE/sample/'processed/human_mask.mp4')
            assert all(math.isfinite(value[k]) for k in ['CD_H_cm','ACC_H_cm_per_frame2'])
            values[sample]=value;coverage.append(info['detection_frame_rate'])
        assert len(values)==10 and sum(v['frames'] for v in values.values())==7610
        summary={k:sum(v[k] for v in values.values())/10 for k in ['CD_H_cm','ACC_H_cm_per_frame2']}
        summary.update(sequences=10,frames=7610,per_sequence=values,checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,evaluation_protocol='native_rgb_target_head_v1',
            target_coverage_equal_sequence_mean=sum(coverage)/10,test_used_for_selection=False,
            independently_checked_unix=time.time())
        result['models'][name]=summary;added.append(name)
        print(name,json.dumps({k:v for k,v in summary.items() if k!='per_sequence'}),flush=True)
    result['updated_unix']=time.time();save_json(path,result)
    print(json.dumps(dict(newly_checked=added,whole_Test10_models_checked=list(result['models']))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--new-only',action='store_true')
    main(parser.parse_args().new_only)

"""Audit historical starting-RGB Test predictions; never relabel as new work."""
import hashlib,json,sys,time,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from utils.part_aggregation_metrics import score_chunk,summarize_chunks

OUT=ROOT/'exp/261029/part_aware_dynamic_aggregation'
SOURCE=ROOT/'exp/261028/device3_encoding_12h'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    tasks=json.loads((SOURCE/'test/tasks.json').read_text())
    source_manifest=json.loads((SOURCE/'manifest.json').read_text())
    sessions=source_manifest['training_sessions']+source_manifest['validation_sessions']+source_manifest['test_sessions']
    assert all(os.path.samefile(SOURCE/'cache'/(n+'.h5'),SOURCE/'encoding_cache/count33'/(n+'.h5')) for n in sessions)
    checkpoint_audit=json.loads((SOURCE/'checkpoint_audit.json').read_text())
    checkpoint=next(r for r in checkpoint_audit['records'] if r['name']=='rgb3' and r['stage']=='best')
    assert checkpoint['sha256']==manifest['init_sha256']
    assert checkpoint['successful_updates']==1651
    recorded={(v['name'],v['segment']):v['sha256'] for v in json.loads((SOURCE/'raw_prediction_hashes.json').read_text())}
    output=[];receipts=[];matched=total=0
    for task in tasks:
        folder=SOURCE/'results/rgb3'/task['session']
        receipt=json.loads((folder/'complete.json').read_text());oldtask=receipt['task']
        for k in ('session','parent_session','segment','start_frame','end_frame','video_frames','gt_file'):
            assert oldtask[k]==task[k]
        assert oldtask['split_sha256']==manifest['split_sha256'] and oldtask['condition']=='rgb3'
        assert Path(oldtask['checkpoint']).resolve()==Path(checkpoint['path']).resolve()
        assert oldtask['matching_protocol']=='primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1'
        digest=sha(folder/'result.npz');assert digest==recorded[('rgb3',task['session'])]
        with np.load(task['gt_file']) as f:
            gt=f['joints_cam_m'];valid=f['valid'];positions=f['source_rgb_positions']
        with np.load(folder/'result.npz') as f:
            assert np.array_equal(f['source_rgb_positions'],positions) and f['seen_gt_rows'].all()
            pred=f['matched_joints_cam_m'];mask=valid&(f['matched_pred_indices']>=0)
            assert np.isfinite(pred[mask]).all()
        row,_,_=score_chunk(pred,gt,valid,positions)
        row.update(session=task['session'],parent_session=task['parent_session']);output.append(row)
        matched+=int(mask.sum());total+=int(valid.sum())
        receipts.append(dict(segment=task['session'],result_sha256=digest,gt_file=task['gt_file']))
    result=dict(status='passed',label='Historical starting RGB3; prior experiment, not newly trained/evaluated',
        source_experiment=str(SOURCE),source_condition='rgb3',source_successful_updates=1651,
        source_checkpoint=checkpoint,checkpoint_matches_shared_initialization=True,
        split_sha256=manifest['split_sha256'],segments=len(tasks),frames=sum(t['video_frames'] for t in tasks),
        GT_instances=total,matched=matched,coverage=matched/total,
        coordinate_frame='device3 rectified RGB camera',clock='original device0 common RGB clock',
        task_geometry_and_interval_identity_verified=True,raw_prediction_hashes_verified=True,
        RGB_pixel_cache_same_inode_verified=True,cache_sessions_verified=len(sessions),
        own_population_metrics=summarize_chunks(output),
        comparison_note='Own historical matched population only; recompute common population before comparing with new models',
        primary_four_condition_table_replaced=False,Test_used_for_model_selection=False,
        provenance=receipts,verified_unix=time.time())
    target=OUT/'historical_starting_rgb.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    link=OUT/'results/starting_rgb_history';link.parent.mkdir(exist_ok=True)
    if not link.exists():link.symlink_to((SOURCE/'results/rgb3').resolve(),target_is_directory=True)
    else:assert link.resolve()==(SOURCE/'results/rgb3').resolve()
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

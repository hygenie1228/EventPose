"""Audit completed selected-model Test interventions without further inference."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from audit_part_aggregation_results import read, sha, training_completion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation'


def validate_control_config(base, config, control):
    assert control in ['missing_event', 'shuffle_event']
    assert config['evaluation_name'] == base['name']+'_'+control
    assert config['evaluation_control'] == control
    allowed = {'evaluation_name', 'evaluation_control'}
    assert all(base.get(key) == config.get(key) for key in (set(base)|set(config))-allowed)


def main():
    state = read(OUT/'pipeline_state.json')
    selection = state['validation_selection']; chosen = selection['name']
    assert selection['Test_used'] is False
    frozen_selection=read(OUT/'artifact_audits/Train_selection_snapshot.json')
    assert frozen_selection['selection']==selection,'Train selection changed after observed freeze'
    assert sha(ROOT/frozen_selection['source_review'])==frozen_selection['source_review_sha256']
    summary = read(OUT/'selected_model_event_controls_comparison.json')
    names = [chosen, chosen+'_missing_event', chosen+'_shuffle_event']
    assert summary['conditions'] == names and summary['all_complete'] and not summary['pending']
    assert summary['Test_used_for_selection'] is False
    basepath = OUT/'configs'/(chosen+'.json'); base = read(basepath)
    assert base['deadline_unix'] == 1793321712
    training = training_completion(base, read(OUT/'runs'/chosen/'complete.json'))
    checkpoint = OUT/'checkpoints'/chosen/'best.pt'; digest = sha(checkpoint)
    assert digest==frozen_selection['actual_chosen_checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['test_used_for_selection'] is False and saved['config']['name'] == chosen
    assert saved['metrics_apply_to_checkpoint'] and saved['metrics_step'] == saved['step']
    assert saved['step'] == read(OUT/'runs'/chosen/'best.json')['step']
    selected_step = saved['step']; del saved
    tasks = read(Path(base['source_root'])/'test/tasks.json'); assert len(tasks) == 80
    config_hashes = {chosen:sha(basepath)}; executions = {}
    controls = {chosen:'intact', chosen+'_missing_event':'missing_event', chosen+'_shuffle_event':'shuffle_event'}
    for name, control in controls.items():
        if control != 'intact':
            path = OUT/'configs'/(name+'_evaluation.json')
            validate_control_config(base, read(path), control); config_hashes[name] = sha(path)
        execution = read(OUT/'results'/name/'execution_rank0.json')
        assert execution['completed'] and execution['checkpoint_sha256'] == digest
        assert execution['evaluation_control'] == control
        assert execution['test_used_for_selection'] is False and execution['GT_used_as_model_input'] is False
        if control != 'intact':
            assert execution['started_unix'] >= selection['selected_unix']
        executions[name] = execution
    common_count = gt_count = 0; counts = dict.fromkeys(names, 0); raw_hashes = []
    for task in tasks:
        with np.load(task['gt_file'], allow_pickle=False) as z:
            gt = z['joints_cam_m']; valid = z['valid']; positions = z['source_rgb_positions']
        assert positions[0] == 0 and np.all(np.diff(positions) > 0) and len(positions) == task['video_frames']
        gt_count += int(valid.sum()); common = valid & np.isfinite(gt).all((2, 3))
        for name in names:
            folder = OUT/'results'/name/task['session']; receipt = read(folder/'complete.json')
            record = receipt['task']
            assert not receipt.get('partial') and not record.get('smoke')
            assert record['checkpoint_sha256'] == digest and record['split_sha256'] == base['split_sha256']
            assert record['evaluation_control'] == controls[name]
            for key in ['parent_session', 'start_frame', 'end_frame', 'video_frames']:
                assert record[key] == task[key]
            path = folder/'result.npz'
            with np.load(path, allow_pickle=False) as z:
                assert z['seen_gt_rows'].all() and np.array_equal(z['source_rgb_positions'], positions)
                pred = z['matched_joints_cam_m']; matched = valid & (z['matched_pred_indices'] >= 0)
                assert pred.shape == gt.shape and np.isfinite(pred[matched]).all()
                assert not np.isfinite(pred[valid & ~matched]).any()
            common &= matched; counts[name] += int(matched.sum())
            raw_hashes.append(dict(condition=name, segment=task['session'], sha256=sha(path)))
        common_count += int((common & common[0][None]).sum())
    assert gt_count == summary['GT_instances']
    for name in names:
        assert common_count == summary['metrics'][name]['samples']
        assert counts[name] == summary['coverage'][name]['matched']
    result = dict(status='passed', checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                  chosen_model=chosen, checkpoint_sha256=digest, selected_checkpoint_step=selected_step,
                  Train_selection=selection, training_completion=training, conditions=names,
                  segments_each=80, prediction_files=len(raw_hashes), GT_instances=gt_count,
                  common_five_metric_samples=common_count, own_matched_instances=counts,
                  evaluation_config_sha256=config_hashes, executions=executions,
                  raw_prediction_sha256=raw_hashes,
                  scope='Same frozen checkpoint and unchanged model/data config, exact Test intervention labels, selection-before-control-execution, full stored frames and exact-frame0/common population; no inference or metric arithmetic re-estimation',
                  limitation='Recorded flags/timeline do not independently prove every external decision; independent metric arithmetic is verified separately')
    path = OUT/'artifact_audits/selected_model_event_controls.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['raw_prediction_sha256','executions','Train_selection']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

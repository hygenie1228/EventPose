"""Freeze diverse Test clips and the already validated model sources."""
from pathlib import Path
import hashlib
import json
import time
import sys
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fusion_repair_scope import EVAL_NAMES, LABELS


def main():
    source = ROOT / 'exp/261027/device34_fusion_repair_15h'
    out = ROOT / 'exp/261028/device34_test_video_gallery'
    config_path = ROOT / '_agents/research/261028/device34_test_video_goal.json'
    if config_path.exists():
        print('Existing frozen gallery retained:', config_path); return
    out.mkdir(parents=True, exist_ok=False)
    work = Path('/mnt/disk5/EventPose/261028/device34_test_video_gallery')
    work.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(p.read_text())
    audit = read(source / 'final_artifact_audit.json')
    assert audit['goal_complete'] and audit['agent_review_status'] == 'passed'
    split_hash = hashlib.sha256((ROOT / 'dataset/split.json').read_bytes()).hexdigest()
    assert split_hash == read(source / 'manifest.json')['split_sha256']
    tasks = read(source / 'test/tasks.json'); candidates = []
    for task in tasks:
        assert task['parent_session'].endswith('_b')
        with np.load(task['gt_file']) as z:
            joints = z['joints_cam_m'][:, :, :22].astype(np.float64)
            stamps = z['timestamps_ns']; valid = z['valid']
            dt = np.diff(stamps) / 1e9
            good = valid[:-1] & valid[1:]
            speeds = np.linalg.norm(np.diff(joints, axis=0) / dt[:, None, None, None], axis=-1).mean(-1)
            speed = float(np.median(speeds[good]))
        candidates.append(dict(task=task, motion_median_mps=speed,
                               participant=int(task['parent_session'].split('_')[3])))
    selected = []; used = set()
    for number, quantile in enumerate((.1, .3, .5, .7, .9), 1):
        target = float(np.quantile([c['motion_median_mps'] for c in candidates], quantile))
        pick = min((c for c in candidates if c['participant'] not in used),
                   key=lambda c: (abs(c['motion_median_mps'] - target), c['task']['session']))
        selected.append(dict(pick, clip=f'clip{number:02d}', motion_quantile_target=quantile))
        used.add(pick['participant'])
    models = {}
    for name in EVAL_NAMES:
        checkpoint = source / 'checkpoints' / name / 'best.pt'
        row = next(r for r in audit['checkpoints'] if r['name'] == name and r['kind'] in ('best', 'released_source_manifest'))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['sha256']
        models[name] = dict(label=LABELS[name], config=str(source / 'configs' / (name + '.json')),
                            checkpoint=str(checkpoint), sha256=row['sha256'])
    old_report = ROOT / '_agents/report/261027_센서융합_열화원인_분석과_재실험.md'
    snapshot = out / '정량보고서_영상목록추가전_검증본.md'
    shutil.copyfile(old_report, snapshot)
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == audit['report_sha256']
    config = dict(experiment='device34_test_video_gallery', created_unix=time.time(), source_root=str(source),
                  output_root=str(out), work_root=str(work), managed_jobs_file=str(out / 'managed_jobs.json'),
                  allowed_gpus=[1, 2, 4, 5], models=models, clips=selected, split_sha256=split_hash,
                  selection='GT body22 motion-speed quantiles .1/.3/.5/.7/.9, five distinct participants; never prediction error',
                  inference_read_only_source=True, original_report=str(old_report), source_report_snapshot=str(snapshot),
                  source_report_snapshot_sha256=audit['report_sha256'],
                  gallery_report=str(ROOT / '_agents/report/261028_대표_테스트_구간별_센서융합_영상목록.md'),
                  expected_videos=45, original_pretrained_manifest=str(source / 'original_pretrained_manifest.json'),
                  pointcloud_depth_scale_m_per_unit=.001,
                  depth_scale_provenance='Delivered example exp/_code_backup/warp_rgbd_h5_to_event.py default .001; archive tag says source_raw_unit. This is an explicitly disclosed scale assumption, not independent sensor-unit certification.')
    (out / 'selection_candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    (out / 'managed_jobs.json').write_text('{"jobs": []}\n')
    print(json.dumps(dict(config=str(config_path), clips=[(c['clip'], c['task']['session'], c['motion_median_mps']) for c in selected]), ensure_ascii=False))


if __name__ == '__main__':
    main()

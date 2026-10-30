"""Freeze verified prior Train assets into a separate repair experiment."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    goal = json.loads((ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal, os.getpid(), 'fusion_repair_source_and_warmstart_audit')
    import h5py
    import numpy as np
    import torch
    from multihmr.multisensor_repair import REPAIR_LAYOUTS
    torch.set_num_threads(2)
    out = Path(goal['output_root']); prior = Path(goal['prior_experiment'])
    raw = (ROOT / 'dataset/split.json').read_bytes(); split = json.loads(raw)
    split_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((prior / 'manifest.json').read_text())
    assert split_sha == manifest['split_sha256']
    ids = {r: {int(p) for n in split[r] for p in n.split('_')[3:] if p.isdigit()} for r in split}
    assert not ids['train'] & ids['test']
    old_subject = json.loads((prior / 'full_split_subject_audit.json').read_text())
    assert old_subject['split_sha256'] == split_sha
    assert sorted(ids['train']) == old_subject['train_ids'] and sorted(ids['test']) == old_subject['test_ids']
    for role in ('training', 'validation', 'test'):
        assert all(n.endswith('_b') for n in manifest[role + '_sessions'])
    assert set(manifest['training_sessions'] + manifest['validation_sessions']) == {n for n in split['train'] if n.endswith('_b')}
    assert set(manifest['test_sessions']) == {n for n in split['test'] if n.endswith('_b')}
    assert not set(manifest['train_subjects']) & (set(manifest['validation_subjects']) | set(manifest['test_subjects']))
    assert not set(manifest['validation_subjects']) & set(manifest['test_subjects'])
    audit = json.loads((prior / 'actual_input_audit.json').read_text())
    assert audit['status'] == 'passed' and audit['split_sha256'] == split_sha
    mapping_hashes = {r['session']: r['frame_mapping_sha256'] for r in audit['records']}
    cache_records = []
    for name in manifest['training_sessions'] + manifest['validation_sessions'] + manifest['test_sessions']:
        with np.load(prior / 'source_maps' / (name + '.npz')) as z:
            maps = {k: z[k] for k in z.files}
        with np.load(prior / 'targets' / (name + '.npz')) as z:
            positions = z['positions']; valid = z['valid']; joints = z['joints']
            assert np.isfinite(joints[valid]).all()
        cache = prior / 'cache' / (name + '.h5')
        with h5py.File(cache, 'r', locking=False) as f:
            assert f.attrs['complete'] and f.attrs['split_sha256'] == split_sha
            frames = f['frames'][:]; stamps = f['timestamp_ns'][:]
            assert np.array_equal(positions, frames) and np.all(np.diff(frames) > 0)
            assert np.array_equal(stamps, maps['canonical_rgb_timestamp_ns'][frames])
            mapping_sha = hashlib.sha256(frames.tobytes() + stamps.tobytes()).hexdigest()
            assert mapping_sha == mapping_hashes[name]
            for sensor in ('rgb3', 'rgb4', 'event3', 'event4'):
                kind = 'event' if sensor.startswith('event') else 'rgb'; prefix = 'device' + sensor[-1] + '_'
                assert f[sensor].shape == (len(frames), 2, 224, 224) if kind == 'event' else f[sensor].shape == (len(frames), 3, 896, 896)
                available = f[sensor + '_available'][:]
                assert np.array_equal(available, maps[prefix + kind + '_available'][frames])
                if kind == 'event':
                    assert np.all(f[sensor + '_latest_timestamp_ns'][:][available] <= stamps[available])
                R = maps[prefix + kind + '_R'] @ maps['device3_rgb_R'].T
                t = maps[prefix + kind + '_t'] - R @ maps['device3_rgb_t']
                assert np.allclose(f[sensor + '_R_common_to_sensor'][:], R, atol=1e-8)
                assert np.allclose(f[sensor + '_t_common_to_sensor'][:], t, atol=1e-8)
        cache_records.append(dict(session=name, frames=len(frames), mapping_sha256=mapping_sha,
                                  cache_sha256=sha(cache), bytes=cache.stat().st_size))
        print(json.dumps(dict(cache_verified=name, count=len(cache_records))), flush=True)
    warmstarts = {}
    for name in ('rgb3', 'rgb4', 'event3_event4'):
        path = prior / 'checkpoints' / name / 'best.pt'
        value = torch.load(path, map_location='cpu')
        original = json.loads((prior / 'configs' / (name + '.json')).read_text())
        assert value['config'] == original and original['split_sha256'] == split_sha
        assert value['successful_steps'] > 0 and value['test_used_for_selection'] is False
        assert value['metrics_apply_to_checkpoint'] and value['metrics_step'] == value['step']
        assert all(torch.isfinite(t).all() for t in value['model_delta'].values())
        warmstarts[name] = dict(path=str(path), sha256=sha(path), condition=name, step=value['step'],
                                source_config=original, selection='Train-subject holdout only',
                                copied_keys='base.*; also adapter.projection.* for event-only; old fusion excluded')
        del value
    # Links are to experiment assets, never to writable dataset destinations.
    for name in ('cache', 'source_maps', 'targets', 'raw', 'test'):
        dest = out / name
        if dest.exists():
            assert dest.is_symlink() and dest.resolve() == (prior / name).resolve()
        else:
            dest.symlink_to((prior / name).resolve(), target_is_directory=True)
    digests = {}
    for name in ('training_index.json', 'validation_index.json', 'joint_regressor.npy', 'split_snapshot.json'):
        source = prior / name; dest = out / name
        if dest.exists():
            assert dest.read_bytes() == source.read_bytes()
        else:
            dest.write_bytes(source.read_bytes())
        digests[name] = sha(dest)
    index = json.loads((out / 'training_index.json').read_text())
    holdout = json.loads((out / 'validation_index.json').read_text())
    assert len(index) == 6501 and len(holdout) == 288
    for rows, role in ((index, 'training'), (holdout, 'validation')):
        assert {r['session'] for r in rows} == set(manifest[role + '_sessions'])
        assert len({(r['session'], r['frame']) for r in rows}) == len(rows)
    configs = []
    if not (out/'event_normalization_Train.json').exists():
        from analyze_fusion_repair_intensity import main as intensity_audit
        intensity_audit()
    normalization=json.loads((out/'event_normalization_Train.json').read_text())
    assert normalization['status']=='passed' and normalization['frames_per_sensor']==6501
    assert normalization['training_index_sha256']==digests['training_index.json']
    assert normalization['split_sha256']==split_sha
    original_base=ROOT/'data/human_model_files/pose_estimate/multiHMR_896_L.pt'
    original_sha=sha(original_base);base_proof=out/'original_pretrained_manifest.json'
    if base_proof.exists():assert json.loads(base_proof.read_text())['sha256']==original_sha
    else:write(base_proof,dict(path=str(original_base),sha256=original_sha,bytes=original_base.stat().st_size,frozen_unix=time.time()))
    for condition, (primary, auxiliary) in REPAIR_LAYOUTS.items():
        source = primary if primary.startswith('rgb') else 'event3_event4'
        cfg = dict(experiment='device34_fusion_repair_15h', architecture='epipolar_repair_norm_v2',
                   condition=condition, name=condition, seed=261027, hours=3,
                   deadline_unix=goal['deadline_unix'], output_root=str(out), allowed_gpus=goal['allowed_gpus'],
                   split_sha256=split_sha, mode='event' if primary.startswith('event') else 'rgb',
                   encoding='count33',event_encoder_gains=normalization['event_encoder_gains'],
                   event_normalization_sha256=sha(out/'event_normalization_Train.json'),
                   primary_sensor=primary, auxiliary_sensor=auxiliary,
                   last_blocks=8, head_lr=1e-5, backbone_lr=1e-6, adapter_lr=1e-4, fusion_lr=1e-3,
                   batch_size=16, workers=2, pa_weight=100, root_weight=25,
                   validation_minutes=30, backbone_warmup_seconds=1800, warmup_steps=100,
                   augment=True, minimum_coverage=.95, residual_regularization=.01,
                   initialization_checkpoint=warmstarts[source]['path'], initialization_sha256=warmstarts[source]['sha256'],
                   initialization_condition=source, initialization='verified prior same-split Train-only selected checkpoint; old fusion excluded',
                   initialization_manifest=str(out / 'warmstart_manifest.json'))
        config_path=out/'configs'/(condition+'.json')
        if config_path.exists():
            saved=json.loads(config_path.read_text())
            assert all(saved.get(k)==v for k,v in cfg.items()),'Never overwrite an existing trial definition'
            assert set(saved)-set(cfg)<={'trial_started_unix'}
        else:write(config_path,cfg)
        configs.append(cfg)
    write(out / 'warmstart_manifest.json', warmstarts)
    manifest.update(created_unix=time.time(), experiment='device34_fusion_repair_15h',
                    old_weights_caches_results_reused=True, reuse_scope='Verified Train weights and input caches; no old fusion results as new results',
                    original_manifest_sha256=sha(prior / 'manifest.json'), copied_file_sha256=digests,
                    layouts=REPAIR_LAYOUTS, source_experiment=str(prior))
    write(out / 'manifest.json', manifest)
    write(out / 'experiment_manifest.json', dict(configs=configs, selection='Train holdout only', total_wall_hours=15))
    write(out / 'source_reuse_audit.json', dict(status='passed', checked_unix=time.time(), split_sha256=split_sha,
          full_split_train_ids=sorted(ids['train']), full_split_test_ids=sorted(ids['test']), overlap=[],
          cache_records=cache_records, cache_bytes=sum(r['bytes'] for r in cache_records),
          warmstart_conditions=list(warmstarts), training_frames=len(index), validation_frames=len(holdout),
          test_segments=80, inference_GT_input=False, old_results_relabelled_as_new=False))
    print(json.dumps(dict(status='passed', caches=len(cache_records), warmstarts=list(warmstarts))), flush=True)


if __name__ == '__main__':
    main()

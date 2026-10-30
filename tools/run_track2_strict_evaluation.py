"""Repair target association for all nine frozen models, then compare/export.

No training or Test-dependent model choice. Every accepted detector head is
independently checked against the original, unexpanded target-mask pixels.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import cv2
import fcntl
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import signal
import time

import numpy as np
import run_track2_experiments as runtime
import finalize_track2_experiments as finish
from multihmr.track2_data import OUTPUT, SOURCE, save_json, sha256


def read(path):
    return json.loads(path.read_text())


class ConversionBusy(Exception):
    pass


def main(body_target=False):
    prefix='body_target' if body_target else 'strict'
    with (OUTPUT / 'locks'/(prefix+'_evaluation.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.REGISTRY_NAME = prefix+'_evaluation_jobs.json'
        runtime.register()
        signal.signal(signal.SIGTERM, runtime.stop_children)
        signal.signal(signal.SIGINT, runtime.stop_children)
        manifest = read(OUTPUT / 'manifest.json')
        plan = read(OUTPUT / (prefix+'_evaluation_plan.json'))
        models = {m['name']: m for m in plan['models']}
        names = list(models)
        deadline = read(OUTPUT / 'budget.json')['deadline_unix']
        reference = read(OUTPUT / 'init_comparison.json')['models']['init_v6']['per_sequence'][manifest['test'][0]]['protocol']
        signature = dict(conversion_revision=2, assets=read(OUTPUT / 'model_assets.json')['assets'],
                         converter_sha=sha256(runtime.ROOT / 'src/utils/mhr_conversion.py'),
                         fitter_sha=sha256(runtime.ROOT / 'src/utils/mhr_reference_fitting.py'))

        def state(status, **extra):
            save_json(OUTPUT / (prefix+'_evaluation_status.json'), dict(status=status, pid=os.getpid(),
                model_count=len(names), test_used_for_selection=False, updated_unix=time.time(), **extra))

        state('waiting_for_actual_strict_detector_pilot')
        pilot_file='body_target_pilot_status.json' if body_target else 'strict_pilot_status.json'
        pilot_ready='passed_actual_body_detector_replay' if body_target else 'complete_actual_strict_head_inference'
        while not (OUTPUT/pilot_file).exists() or read(OUTPUT/pilot_file)['status']!=pilot_ready:
            assert time.time() < deadline - 3600
            time.sleep(5)

        def folder(name, sample, full):
            m = models[name]
            return OUTPUT / 'predictions' / (name if full or m['search_checkpoint'] is None else 'search_' + name) / sample

        def audit_heads(path, sample):
            info = read(path / 'inference.json')
            assert info['strict_target_mask'] and info['target_mask_dilation_kernel'] == 0
            if body_target:
                assert info['target_body_overlap'] and info['body_box_rule_sha256']==sha256(OUTPUT/'body_target_gate_Train_rule.json')
            assert not info['target_foreground_only_input'] and not info['gt_used_for_detection_or_tracking']
            assert info['target_mask_sha256'] == sha256(SOURCE / sample / 'processed/human_mask.mp4')
            raw_sha = sha256(path / 'smplx_predictions.npz')
            receipt = path / 'strict_head_mask_audit.json'
            if receipt.exists():
                old = read(receipt)
                if old['raw_sha256'] == raw_sha and old['mask_sha256'] == info['target_mask_sha256']:
                    assert old['outside_original_mask'] == 0
                    return
            with np.load(path / 'smplx_predictions.npz') as z:
                valid = z['detection_valid']
                xy = z['head_query_native_pixel'] if 'head_query_native_pixel' in z else np.rint(z['head_query_native_xy']).astype(np.int32)
            cap = cv2.VideoCapture(str(SOURCE / sample / 'processed/human_mask.mp4'))
            assert cap.isOpened()
            outside = []
            for i in range(len(valid)):
                ok, mask = cap.read()
                assert ok
                if valid[i]:
                    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    height, width = manifest['samples'][sample]['height'], manifest['samples'][sample]['width']
                    if gray.shape != (height, width):
                        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
                    x, y = xy[i]
                    if not (0 <= x < width and 0 <= y < height and gray[y, x] > 127):
                        outside.append(i)
            cap.release()
            assert not outside, (sample, outside[:10], 'Strict detector head outside ORIGINAL mask')
            save_json(receipt, dict(raw_sha256=raw_sha, mask_sha256=info['target_mask_sha256'], frames=len(valid),
                accepted_heads=int(valid.sum()), imputed_frames=int((~valid).sum()), outside_original_mask=0,
                GT_read=False, verified_unix=time.time()))

        def geometry(path):
            with np.load(path / 'smplx_predictions.npz') as z:
                v = np.ascontiguousarray(z['vertices_cam_m'])
            digest = hashlib.sha256()
            digest.update(str((v.shape, str(v.dtype))).encode())
            digest.update(memoryview(v).cast('B'))
            return digest.hexdigest()

        def copy_conversion(source, target):
            for filename in ['human_params_mhr.npz', 'human_params_mhr_v2.npz']:
                if source != target:
                    shutil.copyfile(source / filename, target / filename)
            protocol = read(target / 'inference.json')['evaluation_protocol']
            for filename in ['conversion.json', 'conversion_v2.json']:
                value = read(source / filename)
                value.update(evaluation_protocol=protocol, exact_input_geometry_cache_reuse=True,
                             reused_conversion_folder=str(source), cache_reuse_uses_GT=False)
                if filename == 'conversion_v2.json':
                    value.update(source_sha256=sha256(target / 'smplx_predictions.npz'),
                                 initial_sha256=sha256(target / 'human_params_mhr.npz'),
                                 output_sha256=sha256(target / 'human_params_mhr_v2.npz'))
                save_json(target / filename, value)

        def convert(name, sample, full, gpu):
            path = folder(name, sample, full)
            digest = geometry(path)
            key = hashlib.sha256(json.dumps(dict(sample=sample, geometry=digest, signature=signature), sort_keys=True).encode()).hexdigest()
            cache = OUTPUT / 'work/strict_conversion_cache' / (key + '.json')
            with (OUTPUT / 'locks' / ('strict_conversion_' + key + '.lock')).open('a') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ConversionBusy
                source = None
                if cache.exists():
                    cached = read(cache)
                    candidate = Path(cached['folder'])
                    assert cached['geometry_sha256'] == digest and sha256(candidate / 'human_params_mhr_v2.npz') == cached['prediction_sha256']
                    source = candidate
                else:
                    m = models[name]
                    legacy = m['source_trial'] if full or m['search_checkpoint'] is None else 'search_' + m['source_trial']
                    candidate = OUTPUT / 'predictions' / legacy / sample
                    if all((candidate / f).exists() for f in ['smplx_predictions.npz','human_params_mhr.npz','human_params_mhr_v2.npz','conversion.json','conversion_v2.json']):
                        if geometry(candidate) == digest:
                            source = candidate
                if source is not None:
                    copy_conversion(source, path)
                else:
                    runtime.run(finish.evaluation_command('track2-convert', path.parent.name, sample), [gpu],
                                'strict_convert_' + path.parent.name + '_' + sample, deadline)
                save_json(cache, dict(folder=str(path), geometry_sha256=digest,
                    prediction_sha256=sha256(path / 'human_params_mhr_v2.npz'), signature=signature, GT_used=False))
                save_json(path / 'conversion_geometry_cache_proof.json', dict(geometry_sha256=digest,
                    exact_geometry_reused=source is not None, reused_from=str(source) if source else None,
                    no_metric_reused=True, GT_used=False, verified_unix=time.time()))

        def verify(name, sample, full):
            path = folder(name, sample, full)
            if not (path / 'metrics.json').exists():
                return None
            m = models[name]
            value = read(path / 'metrics.json');info = read(path / 'inference.json')
            assert value['evaluation_protocol'] == info['evaluation_protocol'] == m['evaluation_protocol']
            assert info['checkpoint_sha256'] == m['full_checkpoint_sha256' if full else 'search_checkpoint_sha256']
            assert info['strict_target_mask'] and info['target_mask_dilation_kernel'] == 0
            assert value['protocol'] == reference and value['conversion_revision'] == 2 and value['full_frame_roster']
            assert value['frames'] == manifest['samples'][sample]['frames']
            assert value['gt_sha256'] == manifest['samples'][sample]['gt_sha256']
            assert value['prediction_sha256'] == sha256(path / 'human_params_mhr_v2.npz')
            audit_heads(path, sample)
            return value

        def stages(name, sample, full, gpu, score=True):
            path = folder(name, sample, full)
            if score and verify(name, sample, full) is not None:
                return
            m = models[name];ckpt = m['full_checkpoint' if full else 'search_checkpoint']
            command = finish.evaluation_command('track2-infer', path.parent.name, sample)
            command.extend(['--input-view', m['input_view'], '--strict-target-mask'])
            if body_target:command.append('--target-body-overlap')
            command.extend(['--checkpoint', ckpt] if ckpt else ['--original-weights'])
            if m['input_view'] == 'native_rgb_static_crop':
                command.extend(['--fixed-view-plan', str(OUTPUT / 'fixed_view_all30_plan.json')])
            runtime.run(command, [gpu], 'strict_infer_' + path.parent.name + '_' + sample, deadline)
            audit_heads(path, sample)
            convert(name, sample, full, gpu)
            if score:
                runtime.run(finish.evaluation_command('track2-score', path.parent.name, sample), [gpu],
                            'strict_score_' + path.parent.name + '_' + sample, deadline)

        def parallel(samples, full, only=None, score=True):
            queue = Queue()
            for task in sorted([(n, s) for n in (only or names) for s in samples],
                               key=lambda t: manifest['samples'][t[1]]['frames'], reverse=True):
                queue.put(task)
            def worker(gpu):
                while not runtime.CANCELLED.is_set():
                    try: name, sample = queue.get_nowait()
                    except Empty: return
                    try:
                        stages(name, sample, full, gpu, score)
                    except ConversionBusy:
                        queue.put((name, sample));time.sleep(.2)
                    finally:
                        queue.task_done()
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(worker, g) for g in runtime.healthy_gpus()]
                try:
                    for future in futures:future.result()
                except BaseException:
                    runtime.stop_children();raise

        def aggregate(name, samples, full):
            values = {s: verify(name, s, full) for s in samples}
            assert all(values.values())
            coverage = np.mean([read(folder(name,s,full)/'inference.json')['detection_frame_rate'] for s in samples])
            return dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in values.values()])),
                ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in values.values()])),
                sequences=len(samples), frames=sum(v['frames'] for v in values.values()), per_sequence=values,
                target_coverage=float(coverage), strict_target_mask=True)

        state('evaluating_nine_strict_Train4_models')
        parallel(manifest['validation'], False)
        validation={n:aggregate(n,manifest['validation'],False) for n in names}
        save_json(OUTPUT/(prefix+'_validation_mhr_progress.json'),dict(models=validation,test_used_for_selection=False))
        eligible=[n for n in names if validation[n]['target_coverage']>=.95];assert eligible
        winner=min(eligible,key=lambda n:(validation[n]['CD_H_cm'],validation[n]['ACC_H_cm_per_frame2'],names.index(n)))
        selection=dict(trial=winner,models=validation,eligible_trials=eligible,model_order=names,
            target_coverage={n:validation[n]['target_coverage'] for n in names},
            input_protocols={n:models[n]['evaluation_protocol'] for n in names},
            evaluation_protocol=models[winner]['evaluation_protocol'],test_used_for_selection=False,selected_unix=time.time())
        save_json(OUTPUT/(prefix+'_selected.json'),selection)
        state('evaluating_nine_strict_Test10_models',selected_trial=winner)
        parallel(manifest['test'],True)
        tested={n:aggregate(n,manifest['test'],True) for n in names}
        save_json(OUTPUT/(prefix+'_completed_Test_models.json'),dict(models=tested,test_used_for_selection=False,strict_target_mask=True))
        save_json(OUTPUT/(prefix+'_test_comparison.json'),dict(models=tested,selected_trial=winner,test_used_for_selection=False))
        state('generating_strict_selected_all30',selected_trial=winner)
        parallel(manifest['train'],True,only=[winner],score=False)
        receipts={}
        for sample in sorted(manifest['samples']):
            receipts[sample]=finish.export_sample(winner,sample,manifest)
            save_json(OUTPUT/(prefix+'_export_receipts.json'),dict(trial=winner,samples=receipts,strict_target_mask=True,updated_unix=time.time()))
        state('all30_ready_for_independent_strict_audit',selected_trial=winner,exports=30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--body-target',action='store_true');args=parser.parse_args()
    try:main(args.body_target)
    except BaseException as error:
        save_json(OUTPUT/(('body_target' if args.body_target else 'strict')+'_evaluation_error.json'),dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

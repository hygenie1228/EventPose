"""Score untouched v6/v0 MHR init files against the same frozen v6 Test GT."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
import run_track2_experiments as runtime
runtime.REGISTRY_NAME='init_comparison_jobs.json'


def main(gpus):
    import fcntl
    with (OUTPUT/'init_comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,runtime.stop_children)
        manifest=json.loads((OUTPUT/'manifest.json').read_text())
        deadline=json.loads((OUTPUT/'budget.json').read_text())['deadline_unix']
        roots={'init_v6':SOURCE,'init_v0':ROOT/'exp2/hj/challenge_track2_v0'}
        inputs={};same_parameters={}
        for sample in manifest['test']:
            inputs[sample]={};parameters={}
            reference=json.loads((SOURCE/sample/'meta_info.json').read_text())
            for version,root in roots.items():
                path=root/sample/'human/human_params_init.npz'
                meta=json.loads((root/sample/'meta_info.json').read_text())
                assert meta['num_frames']==reference['num_frames'] and meta['fps']==reference['fps']
                assert meta['original_sequence_name']==reference['original_sequence_name']
                with np.load(path) as z:
                    arrays={k:z[k] for k in ['pose','shape','scales']}
                    assert arrays['pose'].shape==(reference['num_frames'],136)
                    assert arrays['shape'].shape==(45,) and arrays['scales'].shape==(68,)
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    if 'frame_names' in z.files:assert np.array_equal(z['frame_names'],np.asarray([f'{i:08d}' for i in range(reference['num_frames'])]))
                parameters[version]=arrays
                inputs[sample][version]=dict(path=str(path),sha256=sha256(path),frames=reference['num_frames'])
            same_parameters[sample]=all(np.array_equal(parameters['init_v6'][k],parameters['init_v0'][k]) for k in ['pose','shape','scales'])
        save_json(OUTPUT/'init_input_inventory.json',dict(inputs=inputs,v0_v6_identical_parameters=same_parameters,
            same_test_gt=True,created_unix=time.time()))
        tasks=[(version,sample) for version in roots for sample in manifest['test']]
        save_json(OUTPUT/'init_comparison_status.json',dict(status='running',pid=os.getpid(),gpus=gpus,time=time.time()))
        def worker(index):
            for version,sample in tasks[index::len(gpus)]:
                command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/evaluate_multihmr.py'),
                    'track2-score','--trial',version,'--sample',sample]
                runtime.run(command,[gpus[index]],'baseline_'+version+'_'+sample,deadline)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,i) for i in range(len(gpus))]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                runtime.stop_children();raise
        models={}
        for version in roots:
            values={name:json.loads((OUTPUT/'predictions'/version/name/'metrics.json').read_text()) for name in manifest['test']}
            for name,value in values.items():
                assert value['gt_sha256']==manifest['samples'][name]['gt_sha256']
                assert value['prediction_sha256']==inputs[name][version]['sha256']
                assert value['frames']==manifest['samples'][name]['frames']
            models[version]=dict(CD_H_cm=float(np.mean([v['CD_H_cm'] for v in values.values()])),
                ACC_H_cm_per_frame2=float(np.mean([v['ACC_H_cm_per_frame2'] for v in values.values()])),
                sequences=10,frames=sum(v['frames'] for v in values.values()),per_sequence=values)
        save_json(OUTPUT/'init_comparison.json',dict(models=models,aggregation='same frozen Test10, equal sequence weights',
            v0_v6_identical_parameters=same_parameters,completed_unix=time.time(),test_used_for_model_selection=False))
        save_json(OUTPUT/'init_comparison_status.json',dict(status='complete',pid=os.getpid(),gpus=gpus,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',default='1,2');a=p.parse_args()
    gpus=[int(v) for v in a.gpus.split(',')];assert gpus and all(g in range(1,7) for g in gpus)
    try:main(gpus)
    except BaseException as error:
        save_json(OUTPUT/'init_comparison_error.json',dict(error=repr(error),time=time.time()))
        runtime.stop_children();raise

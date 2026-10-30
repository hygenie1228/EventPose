"""Fresh CPU GT audit under the new experiment, preserving old audit files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from goal_runtime_control import register_job, write


def main():
    config = json.loads((ROOT / '_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(config, os.getpid(), 'fusion_repair_fresh_groundtruth_cpu_audit')
    import audit_device34_inputs as audit
    out = Path(config['output_root'])
    # The audit reads all mappings and reconstructs 54 GT samples from SMPL-X.
    # Its save helper writes only the explicitly supplied destination under OUT.
    audit.OUT = out
    audit.main()
    proof = json.loads((out / 'actual_input_audit.json').read_text())
    assert proof['status'] == 'passed' and proof['independent_cpu_GT_samples'] == 54
    paths = sorted((out / 'targets').glob('*.npz'))
    paths += [Path(t['gt_file']) for t in json.loads((out / 'test/tasks.json').read_text())]
    records = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    write(out / 'reused_groundtruth_hashes.json', dict(status='passed', checked_unix=time.time(), records=records))
    print(json.dumps(dict(status='passed', groundtruth_files=len(records))), flush=True)


if __name__ == '__main__':
    main()

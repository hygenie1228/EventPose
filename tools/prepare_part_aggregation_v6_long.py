"""Freeze matched longer V6 training from completed Train evidence only."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation'
SUFFIXES = ['fixed', 'global_dynamic', 'part_dynamic', 'part_static']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    planpath = OUT/'v6_long_design_frozen.json'
    assert not planpath.exists()
    evidence = {}; source_hashes = {}; names = ['v6_long_'+s for s in SUFFIXES]
    plan = read(OUT/'v6_design_frozen.json')
    for path, expected in plan['source_sha256'].items():
        if path == 'tools/run_part_aggregation.py':
            continue  # Scheduling integration has its own archived receipt.
        assert sha(ROOT/path) == expected, path
        source_hashes[path] = expected
    assert read(OUT/'v6_verified_cuda_pilots_and_full_CPU_reload.json')['status'] == 'passed'
    for suffix in SUFFIXES:
        oldname = 'v6_'+suffix
        run = OUT/'runs'/oldname
        complete = read(run/'complete.json'); best = read(run/'best.json')
        assert complete['status'] == 'complete' and complete['successful_steps'] == 3000
        assert best['step'] == complete['step'] and best['successful_steps'] == 3000
        assert best['coverage'] == 1 and complete['test_used_for_selection'] is False
        evidence[oldname] = dict(best_PA_mm=best['pa_mpjpe_mm'], best_at_last_step=True,
                                complete_sha256=sha(run/'complete.json'), best_sha256=sha(run/'best.json'))
    for suffix in ['global_dynamic', 'part_dynamic', 'part_static']:
        path = OUT/'diagnostics'/('v6_'+suffix+'.json')
        diag = read(path); assert diag['Test_used'] is False
        evidence['v6_'+suffix]['Train_controls_PA_mm'] = {key:v['pa_mpjpe_mm'] for key,v in diag['controls'].items()}
        evidence['v6_'+suffix]['Train_diagnostic_sha256'] = sha(path)
    config_hashes = {}; checked = []
    for suffix in SUFFIXES:
        original = read(OUT/'configs'/('v6_'+suffix+'.json'))
        name = 'v6_long_'+suffix
        config = dict(original, name=name, max_successful_steps=6000,
                      comparison_names=['v2_rgb_only']+names,
                      training_budget_parent='v6_'+suffix,
                      training_budget_restart_from_common_initialization=True)
        assert config['hours'] == 3 and config['deadline_unix'] == 1793321712
        changed = {key for key in set(config)|set(original) if config.get(key)!=original.get(key)}
        assert changed == {'name', 'max_successful_steps', 'comparison_names',
                           'training_budget_parent', 'training_budget_restart_from_common_initialization'}
        path = OUT/'configs'/(name+'.json'); write(path, config); config_hashes[name] = sha(path)
        pilot = dict(config, name='pilot_'+name, pilot_steps=8, pilot_train_limit=32,
                     pilot_validation_limit=16, workers=0, warmup_steps=2,
                     uniform_aggregation_warmup_steps=2,
                     pilot_training_sessions=['20260720_131207_863008_200_b'],
                     pilot_validation_sessions=['20260721_092034_900028_101_b'])
        path = OUT/'configs'/('pilot_'+name+'.json'); write(path, pilot); config_hashes['pilot_'+name] = sha(path)
        checked.append(dict(name=name, exact_changed_keys=sorted(changed),
                            same_frozen_RGB_and_fusion_initialization=True,
                            same_model_loss_optimizer_data_and_3h_cap=True))
    stamp = datetime.now(timezone(timedelta(hours=9))).isoformat()
    frozen = dict(design_frozen_kst=stamp, primary_names=names, config_sha256=config_hashes,
                  source_sha256=source_hashes, Train_evidence=evidence,
                  Test_used=False, Test_files_read=False, deadline_unix=1793321712,
                  rationale='All four completed V6 conditions selected their last 3000-update checkpoint on Train holdout; examine matched longer optimization without changing frozen model or modality protocol',
                  maximum_successful_updates=6000, hours=3,
                  initialization='Restart each condition from the same strong RGB and V3 fusion initialization; not a V6 checkpoint resume',
                  interpretation='The cosine learning-rate horizon also stretches with the 6000-update target; this is a matched longer-budget policy, not an isolated sample-count causal claim',
                  execution_requires='Passed existing V6 CPU architecture and real reload audits, exact config/source check, four new registered actual CUDA8-update pilots and fusion-only checkpoint gates, unchanged fixed deadline')
    write(planpath, frozen)
    audit = dict(status='passed', checked_kst=stamp, scope='CPU configuration/source consistency only; reuses prior V6 architecture and real checkpoint reconstruction evidence; no new forward or CUDA accuracy claim',
                 conditions=checked, model_source_sha256=source_hashes['src/multihmr/part_aggregation_v6.py'],
                 config_sha256=config_hashes, Test_files_read=False, new_CUDA_pilots_still_required=True)
    write(ROOT/'_agents/research/261029/부위별_집계_v6_장기학습_CPU설정감사.json', audit)
    write(OUT/'v6_long_validation_trigger.json', dict(approved=True, approved_kst=stamp,
          Test_used=False, Test_files_read=False, evidence='Completed V6 Train best-at-update-cap and frozen Train event interventions',
          frozen_plan_sha256=sha(planpath), prior_V6_actual_checkpoint_CPU_reload_passed=True,
          new_four_CUDA_pilots_required=True, deadline_unix=1793321712))
    print(json.dumps(dict(status='frozen_approved_pending_controller_and_CUDA_pilots',names=names,checked_kst=stamp), ensure_ascii=False))


if __name__ == '__main__':
    main()

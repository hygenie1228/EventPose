"""Validate an explicit resume authorization without resetting training budgets."""


def resume_clock(config, checkpoint, now):
    authorization = config.get('resume_authorization')
    old = checkpoint['config']
    if authorization is None:
        assert old == config, 'Resume configuration changed'
        started = checkpoint['started_unix']
    else:
        allowed = {'deadline_unix', 'trial_started_unix', 'resume_authorization'}
        assert {k: v for k, v in old.items() if k not in allowed} == {
            k: v for k, v in config.items() if k not in allowed}, 'Learning protocol changed'
        assert authorization['name'] == config['name']
        if checkpoint.get('resume_generation') == authorization['generation']:
            started = checkpoint['started_unix']
        else:
            assert checkpoint['step'] == authorization['source_step'], 'Unexpected source checkpoint'
            elapsed = authorization['prior_elapsed_seconds']
            assert 0 <= elapsed < config['hours'] * 3600
            assert authorization['resumed_unix'] <= now
            started = authorization['resumed_unix'] - elapsed
    finish = min(started + config['hours'] * 3600, config['deadline_unix'] - 300)
    return started, finish


def self_check():
    original = dict(name='rgb', hours=12, deadline_unix=50000, head_lr=1e-5)
    checkpoint = dict(config=original, step=40, started_unix=100)
    auth = dict(name='rgb', generation='new_window', source_step=40,
                prior_elapsed_seconds=36000, resumed_unix=100000)
    config = dict(original, deadline_unix=180000, resume_authorization=auth)
    started, finish = resume_clock(config, checkpoint, 100100)
    assert started == 64000 and finish == 107200  # 2h remain, not a fresh 12h.
    saved = dict(checkpoint, config=config, started_unix=started, resume_generation='new_window')
    assert resume_clock(config, saved, 101000) == (started, finish)
    try:
        resume_clock(dict(config, head_lr=2e-5), checkpoint, 100100)
    except AssertionError:
        pass
    else:
        raise AssertionError('Protocol mutation must be rejected')
    assert resume_clock(dict(config, deadline_unix=105000), checkpoint, 100100)[1] == 104700
    return dict(prior_work_preserved=True, no_repeat_pause_credit=True,
                protocol_mutation_rejected=True, global_deadline_enforced=True)


if __name__ == '__main__':
    print(self_check())

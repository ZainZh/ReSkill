"""CPU tests of the loop infrastructure: splits, exact layout preflight, branch create/list, bundle freeze,
stub-graph static audit + execution on fakes, task score J. No simulator, no GPU (layout preflight needs ROBODOJO_ROOT).

    AR_EVIDENCE is pointed at a temporary root (code/base symlinked to the real one), so nothing is written
    into the real frozen/ or branches/.
"""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1]
REAL_CODE = TOOLS.parent
PROGRAMS = REAL_CODE.parent / 'programs'


@pytest.fixture()
def root(tmp_path, monkeypatch):
    code = tmp_path / 'code'
    (code / 'state').mkdir(parents=True)
    (code / 'base').symlink_to(REAL_CODE / 'base')
    (tmp_path / 'queue').mkdir()
    monkeypatch.setenv('AR_EVIDENCE', str(tmp_path))
    monkeypatch.setenv('AR_PIDS', str(tmp_path / 'code' / 'state' / 'pids.test.json'))   # keep the real registry clean
    return tmp_path


def tool(root, *args):
    env = dict(os.environ, AR_EVIDENCE=str(root), PYTHONDONTWRITEBYTECODE='1', AR_LAUNCH_GAP_S='0',
               AR_LAUNCH_LOCK=str(root / 'gate.lock'), AR_PIDS=str(root / 'code' / 'state' / 'pids.test.json'))
    return subprocess.run([sys.executable, '-B', str(TOOLS / args[0]), *args[1:]], env=env, capture_output=True,
                          text=True, cwd=str(root))


def _arconf():
    sys.path.append(str(TOOLS))
    import arconf
    return importlib.reload(arconf)


def test_splits_and_test_lists():
    arconf = _arconf()
    assert len(arconf.TASKS) == 10
    assert arconf.SPLITS['dev'] == (0, tuple(range(8)))
    assert arconf.SPLITS['val'] == (1, tuple(range(40, 48)))
    assert arconf.SPLITS['val2'] == (1, tuple(range(48, 55)))
    assert not set(arconf.SPLITS['dev3'][1]) & (set(arconf.SPLITS['val'][1]) | set(arconf.SPLITS['val2'][1]))
    seed, lists = arconf.TEST
    assert seed == 2 and set(lists) == set(arconf.TASKS)
    for task, ids in lists.items():
        assert list(ids) == sorted(set(ids)) and all(0 <= i <= 54 for i in ids) and len(ids) <= 50, task
    assert len(lists['stack_blocks_by_language']) == 30 and not {4, 17, 21} & set(lists['stack_blocks'])


def test_preflight_exact_family_every_task(root):
    sys.path.append(str(TOOLS))
    import run_episode
    for task in run_episode.arconf.TASKS:
        for split in ('dev', 'val'):
            seed, layouts = run_episode.arconf.SPLITS[split]
            for layout in layouts:
                entry = run_episode.scene_entry(task, split, layout, seed)
                pre = run_episode.episode.preflight_layout(task, split, layout, entry, run_episode.arconf.LAYOUT_ROOT)
                assert pre['path'].endswith(f'/{seed}/{task}_{layout}.json')
    # prefix families must not cross: a by_language file is refused for the base task and vice versa
    for task, other in (('stack_blocks', 'stack_blocks_by_language'), ('classify_objects', 'classify_objects_by_language')):
        entry = run_episode.scene_entry(other, 'dev', 3, 0)
        bad = dict(entry, task=task)
        assert run_episode.episode.check_entry(task, 'dev', 3, bad), (task, other)
        assert not run_episode.episode.layout_pattern(task).fullmatch(f'{other}_7.json')
        assert run_episode.episode.layout_pattern(other).fullmatch(f'{other}_7.json')
        assert not run_episode.episode.layout_pattern(other).fullmatch(f'{task}_7.json')


def test_split_refusals():
    sys.path.append(str(TOOLS))
    import run_episode
    with pytest.raises(SystemExit):
        run_episode.resolve_split('dev', 8)
    with pytest.raises(SystemExit):
        run_episode.resolve_split('val', 39)
    with pytest.raises(SystemExit, match='sealed'):
        run_episode.resolve_split('test', 0, False, 'stack_bowls')
    seed, lists = run_episode.arconf.TEST
    assert run_episode.resolve_split('test', lists['stack_bowls'][0], True, 'stack_bowls') == 2


def test_branch_create_list_freeze_audit(root):
    r = tool(root, 'branch.py', 'create', '--task', 'stack_bowls', '--id', 'b0', '--from', 'base', '--note', 't')
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info['skeleton'] is True
    d = root / 'code' / 'branches' / 'stack_bowls' / 'b0'
    assert (d / 'tasks/stack_bowls/graph.py').is_file() and (d / 'NOTES.md').is_file()
    assert tool(root, 'branch.py', 'create', '--task', 'stack_bowls', '--id', 'b0', '--from', 'base').returncode != 0
    r = tool(root, 'branch.py', 'create', '--task', 'classify_objects', '--id', 'c0', '--from', str(PROGRAMS / 'classify_objects'))
    assert r.returncode == 0 and json.loads(r.stdout)['skeleton'] is False, r.stderr
    r = tool(root, 'branch.py', 'create', '--task', 'stack_bowls', '--id', 'b1', '--from', 'branches/stack_bowls/b0')
    assert r.returncode == 0 and (root / 'code/branches/stack_bowls/b1/PARENT_NOTES.md').is_file(), r.stderr
    r = tool(root, 'branch.py', 'list', '--json')
    rows = json.loads(r.stdout)
    assert {(x['task'], x['id']) for x in rows} == {('stack_bowls', 'b0'), ('stack_bowls', 'b1'),
                                                     ('classify_objects', 'c0')}
    # freeze each bundle through run_episode (content-addressed frozen copy in the tmp root)
    for task in ('stack_bowls', 'classify_objects'):
        bid = 'b0' if task == 'stack_bowls' else 'c0'
        r = tool(root, 'run_episode.py', '--bundle', f'code/branches/{task}/{bid}', '--task', task, '--freeze-only')
        assert r.returncode == 0, r.stderr
        cand = Path(json.loads(r.stdout)['candidate'])
        assert (cand / 'tasks' / task / 'graph.py').is_file() and not (cand / 'NOTES.md').exists()
        sys.path.insert(0, str(cand))
        try:
            from runner.audit import review
            assert review(cand, task)['passed'], review(cand, task)['findings']
        finally:
            sys.path.remove(str(cand))
            for m in [m for m in sys.modules if m == 'runner' or m.startswith('runner.')]:
                del sys.modules[m]


@pytest.mark.parametrize('task', ['general_pickup', 'build_tower', 'stack_blocks_by_language'])
def test_stub_graph_runs_on_fakes(root, task):
    assert tool(root, 'branch.py', 'create', '--task', task, '--id', 's', '--from', 'base').returncode == 0
    d = root / 'code' / 'branches' / task / 's'
    code = f'''
import sys, importlib, json
sys.path.insert(0, {str(d)!r})
from agraph.fakes import FakeWorld, make_context
from agraph.contract import GraphExecutor
ctx = make_context(FakeWorld(objects=[dict(name='a', category='x', xyz=[.3, 0, .8])]), instruction='do it',
                   resolver=lambda query, world: [])
ctx.perception.observe = lambda stage='observe': ctx.perception.last_frame()
ctx.params.update(json.load(open({str(d / 'tasks' / task / 'params.json')!r})))
g = importlib.import_module('tasks.{task}.graph')
assert g.TASK_ID == {task!r}
out = GraphExecutor(g.build_graph(ctx)).run(ctx)
print(out['terminal'])
'''
    r = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, cwd=str(d))
    assert r.returncode == 0 and r.stdout.strip() == '__done__', r.stderr + r.stdout


def test_task_score_j(root):
    res = root / 'queue' / 'results.jsonl'
    rows = []
    for layout in range(8):
        rows.append(dict(task='stack_bowls', branch='b0', bundle_sha256='a' * 64, split='dev', layout=layout,
                         status='verified', success=layout < 2, native_score=100 if layout < 2 else 20,
                         job_id=f'j{layout}', attempt=1, run_dir=f'r{layout}'))
    rows.append(dict(task='stack_bowls', branch='b0', bundle_sha256='a' * 64, split='dev', layout=3, status='infra',
                     job_id='x', attempt=1, run_dir='rx'))
    rows.append(dict(task='stack_bowls', branch='b1', bundle_sha256='b' * 64, split='dev', layout=0,
                     status='unresolved', failure_type='process_watchdog_timeout', job_id='y', attempt=1, run_dir='ry'))
    res.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    r = tool(root, 'task_score.py')
    assert r.returncode == 0, r.stderr
    data = json.loads((root / 'code/state/task_scores.json').read_text())
    full = [x for x in data['dev'] if x['complete']]
    assert len(full) == 1 and full[0]['J'] == pytest.approx(100 * 2 / 8 + (200 + 6 * 20) / 8)
    part = [x for x in data['dev'] if not x['complete']]
    assert part[0]['branch'] == 'b1' and part[0]['J'] is None and part[0]['mean_score'] == 0
    md = (root / 'code/state/TASK_SCORES.md').read_text()
    assert 'Complete DEV' in md and 'Screening only' in md


def test_enqueue_branch_bundle(root):
    assert tool(root, 'branch.py', 'create', '--task', 'general_pickup', '--id', 'g0', '--from', 'base').returncode == 0
    r = tool(root, 'queue.py', 'enqueue', '--bundle', 'code/branches/general_pickup/g0', '--task', 'general_pickup',
             '--split', 'dev', '--layouts', '0-7', '--tag', 't')
    assert r.returncode == 0, r.stderr
    jobs = [json.loads(p.read_text()) for p in (root / 'queue' / 'jobs').glob('*.json')]
    assert len(jobs) == 8 and {j['branch'] for j in jobs} == {'g0'} and {j['layout'] for j in jobs} == set(range(8))
    assert all(j['seed_directory'] == 0 for j in jobs)
    bad = tool(root, 'queue.py', 'enqueue', '--bundle', 'code/branches/general_pickup/g0', '--task', 'stack_bowls',
               '--split', 'dev', '--layouts', '0')
    assert bad.returncode != 0 and 'belongs to task' in bad.stderr
    assert tool(root, 'queue.py', 'enqueue', '--bundle', 'code/branches/general_pickup/g0', '--task', 'general_pickup',
                '--split', 'dev', '--layouts', '8').returncode != 0
    assert tool(root, 'queue.py', 'enqueue', '--bundle', 'code/branches/general_pickup/g0', '--task', 'general_pickup',
                '--split', 'dev', '--layouts', '0', '--branch', 'other').returncode != 0
    for task in ('classify_objects_by_language', 'pack_objects_into_box', 'organize_table', 'stack_blocks'):
        assert tool(root, 'branch.py', 'create', '--task', task, '--id', 'x', '--from', 'base').returncode == 0
        r = tool(root, 'queue.py', 'enqueue', '--bundle', f'code/branches/{task}/x', '--task', task,
                 '--split', 'val', '--layouts', '40')
        assert r.returncode == 0, r.stderr


def test_run_episode_every_task_fake_sim(root, tmp_path):
    """Full run_episode path (freeze, exact preflight, binding path, launch, classification, RESULT.json) for every
    task with a branch bundle, with a fake simulator that exits without a native result -> 'unresolved'."""
    fake = tmp_path / 'fake_python.sh'
    fake.write_text('#!/bin/bash\necho "fake sim $@"\nexit 1\n')
    fake.chmod(0o755)
    sys.path.append(str(TOOLS))
    import arconf
    for task in arconf.TASKS:
        assert tool(root, 'branch.py', 'create', '--task', task, '--id', 'f', '--from', 'base').returncode == 0
        out = root / 'runs' / task / 'dev07'
        r = tool(root, 'run_episode.py', '--bundle', f'code/branches/{task}/f', '--task', task, '--split', 'dev',
                 '--layout', '7', '--gpu', '0', '--out', str(out), '--skip-service-check', '--runtime-env', '/dev/null',
                 '--python', str(fake), '--timeout', '60')
        res = json.loads((out / 'RESULT.json').read_text())
        run = json.loads((out / 'RUN.json').read_text())
        assert res['status'] == 'unresolved' and res['task'] == task, (task, r.stderr[-2000:])
        assert run['layout_file'] == f'{task}_7.json' and run['seed_directory'] == 0
        assert f'--layout-file {task}_7.json' in (out / 'process.log').read_text()


def _fake(tmp_path, name, body):
    fake = tmp_path / name
    fake.write_text('#!/bin/bash\n' + body)
    fake.chmod(0o755)
    return fake


def _run(root, task, out, fake, extra_env=None, timeout='120'):
    env = dict(os.environ, AR_EVIDENCE=str(root), PYTHONDONTWRITEBYTECODE='1',
               AR_PIDS=str(root / 'code' / 'state' / 'pids.test.json'))
    env.update(extra_env or {})
    return subprocess.Popen([sys.executable, '-B', str(TOOLS / 'run_episode.py'), '--bundle', f'code/branches/{task}/f',
                             '--task', task, '--split', 'dev', '--layout', '1', '--gpu', '0', '--out', str(out),
                             '--skip-service-check', '--runtime-env', '/dev/null', '--python', str(fake),
                             '--timeout', timeout], env=env, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def test_native_crash_before_control_is_infra_and_fast(root, tmp_path):
    import time
    assert tool(root, 'branch.py', 'create', '--task', 'stack_bowls', '--id', 'f', '--from', 'base').returncode == 0
    fake = _fake(tmp_path, 'crash.sh', 'echo "PYTHONPATH=$PYTHONPATH"\necho "[60.3s] app ready"\n'
                 'echo "0 [0ms] [Warning] [carb.crashreporter-breakpad.plugin] [crash] A crash has occurred.  x"\n'
                 'sleep 600\n')
    out = root / 'runs' / 'crash'
    t0 = time.time()
    p = _run(root, 'stack_bowls', out, fake, dict(AR_LAUNCH_LOCK=str(tmp_path / 'gate.lock'), AR_LAUNCH_GAP_S='0'))
    p.communicate(timeout=60)
    res = json.loads((out / 'RESULT.json').read_text())
    assert time.time() - t0 < 30
    assert res['status'] == 'infra', res
    assert res['infrastructure_evidence'][0]['kind'] == 'native_crash_before_control'
    log = (out / 'process.log').read_text()
    assert str(TOOLS / 'simsite') in log.splitlines()[0]


def test_launch_gate_serialises_starts(root, tmp_path):
    assert tool(root, 'branch.py', 'create', '--task', 'stack_bowls', '--id', 'f', '--from', 'base').returncode == 0
    fake = _fake(tmp_path, 'slow.sh', 'date +%s.%N\nsleep 2\necho "[2s] app ready"\nsleep 3\nexit 1\n')
    env = dict(AR_LAUNCH_LOCK=str(tmp_path / 'gate.lock'), AR_LAUNCH_GAP_S='4', AR_LAUNCH_SETTLE_S='1')
    procs = [_run(root, 'stack_bowls', root / 'runs' / f'g{i}', fake, env) for i in range(3)]
    for p in procs:
        p.communicate(timeout=120)
    starts = sorted(float((root / 'runs' / f'g{i}' / 'process.log').read_text().split()[0]) for i in range(3))
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 3.9 for g in gaps), gaps          # >= gap and >= app ready + settle (3 s)
    for i in range(3):
        run = json.loads((root / 'runs' / f'g{i}' / 'RUN.json').read_text())
        assert run['launch_gate']['released'] in ('app_ready', 'process_exit')

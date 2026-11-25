#!/usr/bin/env python3
"""File-based episode queue + GPU-slot supervisor of the development loop (state on the shared file system).

    queue.py enqueue  --bundle DIR --task T --split dev|dev2|dev3|val|val2 --layouts 0-7|0,3,5 [--priority N] [--tag X] [--branch B]
                      (DIR = branches/<task>/<id> or a frozen program; for a branch bundle --branch defaults to <id>)
    queue.py status   [--all] [--json]
    queue.py cancel   JOB_ID ... | --tag X | --branch B          (queued -> cancelled; running -> SIGTERM by recorded PID)
    queue.py results  [--tag X] [--branch B] [--task T]          (sync finished RESULTs into code/state/results.jsonl + summary)
    queue.py config   [--gpus 0,1,...] [--slots-per-gpu N] [--gpu-slots G=N ...] [--pause|--resume]
    queue.py supervise [--gpus 0-7] [--slots-per-gpu N]          (daemon on the GPU node; tmux ar-queue via tools/queue_ctl.sh)

Layout (EVIDENCE = arconf.RESKILL_ROOT):
  queue/jobs/<id>.json      job record; created by enqueue, afterwards written only by the supervisor
  queue/cancel/<id>         cancel request (any host); the supervisor acts on it
  queue/CONFIG.json         gpus / slots_per_gpu / paused; re-read by the supervisor every loop
  queue/results.jsonl       one line per finished job attempt (RESULT.json + job fields), supervisor-written
  queue/NODE_STATUS.<host>.json, queue/gpu_samples.<host>.jsonl   monitoring (nvidia-smi every 30 s)
  runs/<task>/<split><II>/<job_id>_a<attempt>/   RUN_DIR of tools/run_episode.py

Rules: one infra retry only for evidenced infrastructure failures (native exit 99 + PhysX fatal line);
a service-preflight infra result (no simulator started) is re-queued without consuming the retry
(at most 5 times). Episode wall timeout 2700 s (run_episode watchdog) -> unresolved; the supervisor
backstop SIGTERMs a job at timeout + 600 s. Processes are tracked in code/state/pids*.json and killed
only by recorded PID. TEST is never accepted by the queue.
"""
import os
import sys

# tools/queue.py shadows the stdlib ``queue`` module: keep this directory LAST on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE] + [_HERE]

import argparse
import json
from pathlib import Path
import random
import signal
import socket
import subprocess
import time

import arconf  # noqa: E402
import pidreg  # noqa: E402

Q = arconf.QUEUE
JOBS, CANCEL = Q / 'jobs', Q / 'cancel'
CONFIG = Q / 'CONFIG.json'
RESULTS = Q / 'results.jsonl'
LOCAL_RESULTS = arconf.STATE / 'results.jsonl'
HOST = socket.gethostname()
BACKSTOP_S = 600
MIN_FREE_MIB = 12000
LAUNCH_GAP_S = 20
HOST_LAUNCH_GAP_S = 75                 # min seconds between two launches on one host (CONFIG host_launch_gap_s)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{HOST}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str) + '\n')
    os.replace(tmp, path)


def parse_ints(text):
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def jobs(include_done=False):
    """Active jobs (jobs/*.json); finished/cancelled ones are archived in jobs/done/."""
    paths = sorted(JOBS.glob('*.json')) + (sorted((JOBS / 'done').glob('*.json')) if include_done else [])
    return [j for j in (read(p) for p in paths) if j]


def job_path(jid):
    active = JOBS / f'{jid}.json'
    return active if active.exists() else JOBS / 'done' / f'{jid}.json'


TASK_QUEUE_CAP = 70    # queued-job cap per task: about 2 versions x 30 episodes + a few smoke runs
BOOST_TASKS = ()       # optional: tasks whose jobs are raised to at least priority 6 (empty = no boost)
TOP_TASKS = ()         # optional: tasks whose jobs are raised to at least priority 7 (empty = no boost)


# ---------------------------------------------------------------- client commands
def cmd_enqueue(a):
    if a.split not in arconf.SPLITS:
        raise SystemExit('queue accepts dev|dev2|dev3|val|val2 only (TEST is sealed)')
    # Hidden model-selection test sets: val/val2 can only be enqueued by the orchestrator (AR_COORD=1);
    # development agents can only use dev/dev2/dev3.
    if a.split in ('val', 'val2') and os.environ.get('AR_COORD') != '1':
        raise SystemExit(f'split {a.split} is RESERVED for the orchestrator (AR_COORD=1); development agents use dev|dev2|dev3')
    seed, allowed = arconf.split_layouts(a.split, a.task)
    layouts = parse_ints(a.layouts)
    bad = [i for i in layouts if i not in allowed]
    if bad:
        raise SystemExit(f'layouts {bad} not in {a.split}' + (f' ({allowed[0]}-{allowed[-1]})' if allowed else ' (none for this task)'))
    # Per-task queue cap TASK_QUEUE_CAP (priority >= 8 orchestrator checks exempt); enqueues beyond it are refused
    if a.priority < 8:
        base = a.task
        n = sum(1 for j in jobs() if j['state'] == 'queued' and j['task'] == base)
        if n + len(layouts) > TASK_QUEUE_CAP:
            raise SystemExit(f'queue cap: {base} already has {n} queued jobs, cap {TASK_QUEUE_CAP}. '
                             'Wait until the queued versions finish their first pass')
    bundle = Path(a.bundle).resolve()
    branches = (arconf.CODE / 'branches').resolve()
    if branches in bundle.parents:                       # branch bundle: branches/<task>/<id>
        rel = bundle.relative_to(branches).parts
        if len(rel) != 2:
            raise SystemExit(f'branch bundle must be branches/<task>/<id>, got {bundle}')
        if rel[0] != a.task:
            raise SystemExit(f'bundle {bundle} belongs to task {rel[0]}, not {a.task}')
        if a.branch is None:
            a.branch = rel[1]
        elif a.branch != rel[1]:
            raise SystemExit(f'--branch {a.branch} does not match bundle branch id {rel[1]}')
    import run_episode                                   # freeze on the enqueuing host (bundle may be local)
    candidate, record = run_episode.ensure_frozen(a.bundle, a.task)
    created = []
    # optional priority boost (BOOST_TASKS / TOP_TASKS; empty by default)
    if a.task in BOOST_TASKS:
        a.priority = max(a.priority, 6)
    if a.task in TOP_TASKS:
        a.priority = max(a.priority, 7)
    for layout in layouts:
        jid = f'{time.strftime("%m%d%H%M%S")}-{a.task[:5]}-{a.split}{layout:02d}-{random.randrange(16**4):04x}'
        job = dict(id=jid, state='queued', task=a.task, split=a.split, layout=layout, seed_directory=seed,
                   bundle=str(Path(a.bundle).resolve()), candidate=str(candidate),
                   bundle_sha256=record['source_sha256'], priority=a.priority, tag=a.tag, branch=a.branch,
                   pin_gpu=a.pin_gpu, created_utc=utc(), created_epoch=time.time(), enqueued_by=HOST, attempt=1,
                   preflight_requeues=0,
                   history=[])
        write(JOBS / f'{jid}.json', job)
        created.append(jid)
    print(json.dumps(dict(enqueued=created, candidate=str(candidate), bundle_sha256=record['source_sha256'])))


def cmd_status(a):
    rows = jobs(include_done=a.all)
    counts = {}
    for j in rows:
        counts[j['state']] = counts.get(j['state'], 0) + 1
    cfg = read(CONFIG, {})
    nodes = {p.name[len('NODE_STATUS.'):-5]: read(p) for p in Q.glob('NODE_STATUS.*.json')}
    if a.json:
        print(json.dumps(dict(counts=counts, config=cfg, nodes=nodes,
                              jobs=[j for j in rows if a.all or j['state'] in ('queued', 'running')]), indent=1))
        return
    print('config:', cfg)
    print('counts:', counts)
    for host, n in nodes.items():
        if not n:
            continue
        print(f"node {host}: supervisor pid {n.get('pid')} age {time.time() - n.get('epoch', 0):.0f}s "
              f"services {n.get('services')} load {n.get('loadavg')}")
        for g in n.get('gpus', []):
            print(f"  gpu{g['index']}: {g['mem_used']}/{g['mem_total']} MiB util {g['util']}% running {g['running']}")
    for j in sorted(rows, key=lambda j: (j['state'] != 'running', -j['priority'], j['created_epoch'])):
        if not a.all and j['state'] not in ('queued', 'running'):
            continue
        age = time.time() - (j.get('started_epoch') or j['created_epoch'])
        print(f"{j['id']:40s} {j['state']:9s} {j['task']:25s} {j['split']}{j['layout']:02d} a{j['attempt']} "
              f"gpu {j.get('gpu')} p{j['priority']} {j.get('branch') or ''}/{j.get('tag') or ''} {age / 60:.0f}m "
              f"{(j.get('result') or {}).get('status', '')}")


def cmd_cancel(a):
    targets = set(a.ids)
    if a.tag or a.branch:
        # Branch ids and tags repeat across tasks: cancelling by filter needs both --task and --branch,
        # and every given condition must match.
        if not (a.task and a.branch):
            raise SystemExit('cancel by --tag/--branch requires both --task and --branch (ids are shared across tasks)')
        for j in jobs():
            if j.get('task') == a.task and j.get('branch') == a.branch and (not a.tag or j.get('tag') == a.tag):
                targets.add(j['id'])
    CANCEL.mkdir(parents=True, exist_ok=True)
    for jid in sorted(targets):
        if not (JOBS / f'{jid}.json').exists():
            print('unknown or already finished job', jid)
            continue
        (CANCEL / jid).write_text(json.dumps(dict(utc=utc(), host=HOST)))
        print('cancel requested', jid)


def _jsonl_rows(path):
    # Several hosts append to the same file over the network FS; a rare torn line is skipped with a warning
    rows, bad = [], 0
    for line in path.read_text().splitlines() if path.exists() else []:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f'warning: skipped {bad} corrupt line(s) in {path}', file=sys.stderr)
    return rows


def cmd_results(a):
    rows = _jsonl_rows(RESULTS)
    LOCAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if LOCAL_RESULTS.exists():
        for r in _jsonl_rows(LOCAL_RESULTS):
            have.add((r.get('job_id'), r.get('attempt')))
    new = [r for r in rows if (r.get('job_id'), r.get('attempt')) not in have]
    if new:
        with LOCAL_RESULTS.open('a') as f:
            for r in new:
                f.write(json.dumps(r, default=str) + '\n')
    sel = [r for r in rows if (not a.tag or r.get('tag') == a.tag) and (not a.branch or r.get('branch') == a.branch)
           and (not a.task or r.get('task') == a.task)]
    groups = {}
    for r in sel:
        if r.get('superseded'):
            continue
        key = (r.get('branch'), r.get('tag'), r.get('task'), r.get('split'), r.get('bundle_sha256', '')[:12])
        g = groups.setdefault(key, dict(n=0, verified=0, success=0, score=0., infra=0, unresolved=0, interrupted=0,
                                        cost=0., wall=0.))
        g['n'] += 1
        st = arconf.effective_status(r)
        if st == 'verified':
            g['verified'] += 1
            g['success'] += bool(r.get('success'))
            g['score'] += float(r.get('native_score') or 0)
        elif st in ('infra', 'unresolved', 'interrupted'):
            g[st] += 1
        g['cost'] += float(r.get('semantic_cost_usd') or 0)
        g['wall'] += float(r.get('wall_seconds') or 0)
    print(f'synced {len(new)} new rows into {LOCAL_RESULTS}')
    for key, g in sorted(groups.items(), key=str):
        mean = g['score'] / g['verified'] if g['verified'] else None
        print(f"branch={key[0]} tag={key[1]} {key[2]} {key[3]} sha={key[4]}: verified {g['verified']}/{g['n']} "
              f"success {g['success']} mean_score {mean if mean is None else round(mean, 1)} "
              f"infra {g['infra']} unresolved {g['unresolved']} interrupted {g['interrupted']} "
              f"cost ${g['cost']:.2f} wall {g['wall'] / 3600:.1f}h")
    if a.json:
        print(json.dumps(sel, indent=1, default=str))


def cmd_config(a):
    cfg = read(CONFIG, {}) or {}
    if a.gpus is not None:
        cfg['gpus'] = parse_ints(a.gpus)
    if a.slots_per_gpu is not None:
        cfg['slots_per_gpu'] = a.slots_per_gpu
    for item in a.gpu_slots or []:
        g, n = item.split('=')
        cfg.setdefault('gpu_slots', {})[str(int(g))] = int(n)
    if a.clear_gpu_slots:
        cfg.pop('gpu_slots', None)
    if a.host_launch_gap is not None:
        cfg['host_launch_gap_s'] = a.host_launch_gap
    if a.pause:
        cfg['paused'] = True
    if a.resume:
        cfg['paused'] = False
    cfg['updated_utc'] = utc()
    write(CONFIG, cfg)
    print(json.dumps(cfg))


# ---------------------------------------------------------------- supervisor
def gpu_sample():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30).stdout
        rows = {}
        for line in out.strip().splitlines():
            i, used, total, util = [x.strip() for x in line.split(',')]
            rows[int(i)] = dict(index=int(i), mem_used=int(used), mem_total=int(total), util=int(util))
        return rows
    except Exception:
        return {}


def services_ok():
    import urllib.request
    problems = []
    if os.environ.get('AR_SKIP_SERVICE_CHECK'):
        return problems
    # Checks the service the task programs use: SAM 3 (/health).
    try:
        with urllib.request.urlopen(arconf.SAM_ENDPOINT.rstrip('/') + '/health', timeout=5) as r:
            json.load(r)
    except Exception as exc:
        problems.append(f'sam3: {exc}')
    return problems


class Supervisor:
    def __init__(self, gpus, slots):
        self.default = dict(gpus=gpus, slots_per_gpu=slots, paused=False)
        self.procs = {}                     # job id -> Popen (launched by this supervisor instance)
        self.last_launch = {}
        self.last_status = 0.
        self.last_raw = None                # last CONFIG.json read successfully

    def config(self):
        cfg = dict(self.default)
        raw = read(CONFIG)
        # Reading CONFIG can fail while another host os.replace()s it on the shared FS: keep the last config read
        # instead of falling back to the command-line defaults, and pause if none was ever read.
        if isinstance(raw, dict) and raw:
            self.last_raw = raw
        elif self.last_raw:
            raw = self.last_raw
        else:
            return dict(cfg, paused=True)
        keys = ('gpus', 'slots_per_gpu', 'paused', 'gpu_slots', 'host_launch_gap_s')
        cfg.update({k: v for k, v in raw.items() if k in keys})
        # per-host overrides (CONFIG hosts.<hostname>)
        cfg.update({k: v for k, v in ((raw.get('hosts') or {}).get(HOST) or {}).items() if k in keys})
        return cfg

    def job(self, jid):
        return read(job_path(jid))

    def save(self, job):
        if job['state'] in ('done', 'cancelled'):
            write(JOBS / 'done' / f"{job['id']}.json", job)
            (JOBS / f"{job['id']}.json").unlink(missing_ok=True)
        else:
            write(JOBS / f"{job['id']}.json", job)

    def running(self):
        return [j for j in jobs() if j['state'] == 'running' and j.get('host') == HOST]

    def sim_started(self, job):
        """Epoch the simulator was started (RUN.json process_started_epoch; run_episode may first wait in the
        host launch gate), else the supervisor launch time."""
        run = read(Path(job['run_dir']) / 'RUN.json', {}) if job.get('run_dir') else {}
        return float((run or {}).get('process_started_epoch') or job['started_epoch'])

    def alive(self, job):
        p = self.procs.get(job['id'])
        if p is not None:
            return p.poll() is None
        start = pidreg.proc_start(job['pid']) if job.get('pid') else None
        return start is not None and start == job.get('pid_start')

    def launch(self, job, gpu):
        run_dir = arconf.RUNS / job['task'] / f"{job['split']}{job['layout']:02d}" / f"{job['id']}_a{job['attempt']}"
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        # Cross-host claim: several supervisors share the queue directory; mkdir is atomic on the shared FS,
        # so only the host that creates the run directory launches the job.
        fresh = self.job(job['id'])
        if not fresh or fresh.get('state') != 'queued':
            return False
        try:
            run_dir.mkdir()
        except FileExistsError:
            return False
        log = run_dir.parent / f"{run_dir.name}.queue.log"
        cmd = [arconf.native_python(), '-B', '-u', str(arconf.TOOLS / 'run_episode.py'), '--bundle', job['candidate'],
               '--task', job['task'], '--split', job['split'], '--layout', str(job['layout']), '--gpu', str(gpu),
               '--out', str(run_dir), '--tag', job.get('tag') or job.get('branch') or 'q',
               '--timeout', str(arconf.EPISODE_TIMEOUT_S)]
        with open(log, 'w') as stream:
            p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                 cwd=str(arconf.CODE))
        self.procs[job['id']] = p
        pidreg.add(p.pid, cmd, role='queue_job', gpu=gpu, job_id=job['id'], run_dir=str(run_dir))
        job.update(state='running', gpu=gpu, host=HOST, pid=p.pid, pid_start=pidreg.proc_start(p.pid),
                   run_dir=str(run_dir), started_utc=utc(), started_epoch=time.time())
        self.save(job)
        self.last_launch[gpu] = time.time()
        print(f'{utc()} launch {job["id"]} gpu {gpu} pid {p.pid}', flush=True)
        return True

    def finalize(self, job):
        run_dir = Path(job['run_dir'])
        result = read(run_dir / 'RESULT.json')
        if result is None:
            result = dict(task=job['task'], split=job['split'], layout=job['layout'], status='interrupted',
                          success=None, native_score=None, bundle=job['bundle'], candidate=job['candidate'],
                          bundle_sha256=job['bundle_sha256'], run_dir=str(run_dir),
                          error='run_episode exited without RESULT.json')
        p = self.procs.pop(job['id'], None)
        pidreg.end(job['pid'], exit_code=None if p is None else p.returncode)
        if job.get('cancel_requested'):
            result['status'] = 'interrupted' if result.get('status') != 'verified' else result['status']
        row = dict(result, job_id=job['id'], attempt=job['attempt'], tag=job.get('tag'), branch=job.get('branch'),
                   priority=job['priority'], queue_host=HOST, finished_utc=result.get('finished_utc') or utc())
        evidence = result.get('infrastructure_evidence') or []
        preflight = any(e.get('kind') == 'service_preflight' for e in evidence)
        retry = result.get('status') == 'infra' and not job.get('cancel_requested') and (
            (preflight and job['preflight_requeues'] < 5) or (not preflight and evidence and job['attempt'] == 1))
        row['superseded'] = bool(retry)
        with RESULTS.open('a') as f:
            f.write(json.dumps(row, default=str) + '\n')
        job['history'].append(dict(attempt=job['attempt'], run_dir=str(run_dir), status=result.get('status'),
                                   gpu=job.get('gpu'), finished_utc=utc()))
        if retry:
            if preflight:
                job['preflight_requeues'] += 1
                # A preflight requeue keeps the attempt number, hence the run directory name; launch claims by mkdir, so
                # the old directory must be moved away (kept as .preflightN) or the job could never start again.
                try:
                    run_dir.rename(run_dir.with_name(f'{run_dir.name}.preflight{job["preflight_requeues"]}'))
                except OSError as exc:
                    print(f'{utc()} preflight dir rename failed {run_dir}: {exc!r}', flush=True)
            else:
                job['attempt'] += 1
            job.update(state='queued', pid=None, pid_start=None, gpu=None, run_dir=None)
            print(f'{utc()} requeue {job["id"]} (infra: {"preflight" if preflight else "evidenced"})', flush=True)
        else:
            job.update(state='done', result={k: result.get(k) for k in ('status', 'success', 'native_score',
                                                                         'native_actions', 'wall_seconds')},
                       finished_utc=utc())
            print(f'{utc()} done {job["id"]} {job["result"]}', flush=True)
        self.save(job)

    def handle_cancels(self):
        for marker in sorted(CANCEL.glob('*')) if CANCEL.exists() else []:
            # skip already-archived markers
            if marker.is_dir() or marker.name.endswith('.unknown'):
                continue
            job = self.job(marker.name)
            if job is None:
                (CANCEL / 'unknown').mkdir(exist_ok=True)
                try:
                    marker.rename(CANCEL / 'unknown' / marker.name)
                except OSError:            # moved by another host already, or a stale NFS handle
                    pass
                continue
            if job['state'] == 'queued':
                job.update(state='cancelled', finished_utc=utc())
                self.save(job)
                marker.unlink(missing_ok=True)
            elif job['state'] == 'running' and job.get('host') == HOST:
                if not job.get('cancel_requested'):
                    job.update(cancel_requested=utc())
                    self.save(job)
                    self.terminate(job, 'cancel')
                elif time.time() - _epoch(job['cancel_requested']) > 90:
                    self.terminate(job, 'cancel-kill', sig=signal.SIGKILL)
                if not self.alive(job):
                    marker.unlink(missing_ok=True)
            elif job['state'] in ('done', 'cancelled'):
                marker.unlink(missing_ok=True)

    def terminate(self, job, why, sig=signal.SIGTERM):
        """Signal the recorded run_episode PID (its own session); on SIGKILL also its recorded sim PID."""
        if not self.alive(job):
            return
        try:
            os.killpg(job['pid'], sig)
        except ProcessLookupError:
            pass
        # Policy-server episodes start the policy server and the env client in another session; killing run_episode
        # alone leaves them on the GPU. Also signal the pids recorded in PS_LAUNCH.json.
        launch = read(Path(job.get('run_dir') or '.') / 'PS_LAUNCH.json', {}) or {}
        for key in ('client_pid', 'server_pid'):
            pid = launch.get(key)
            if pid:
                for kill in (lambda: os.killpg(pid, sig), lambda: os.kill(pid, sig)):
                    try:
                        kill()
                        break
                    except (ProcessLookupError, PermissionError):
                        pass
        if sig == signal.SIGKILL:
            run = read(Path(job['run_dir']) / 'RUN.json', {}) or {}
            sim = run.get('pid')
            rows = [r for r in pidreg.find(pid=sim) if r.get('host') == HOST] if sim else []
            if rows and pidreg.alive(rows[-1]):
                try:
                    os.killpg(sim, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        print(f'{utc()} {why} {job["id"]} signal {sig}', flush=True)

    def status(self, cfg, problems):
        snap = gpu_sample()
        running = self.running()
        for g in snap.values():
            g['running'] = sum(1 for j in running if j.get('gpu') == g['index'])
        row = dict(host=HOST, pid=os.getpid(), epoch=time.time(), utc=utc(), config=cfg,
                   services='ok' if not problems else problems, loadavg=os.getloadavg(),
                   mem_available_gib=_mem_available(), gpus=list(snap.values()),
                   running=[dict(id=j['id'], gpu=j.get('gpu'), task=j['task'], layout=j['layout']) for j in running])
        write(Q / f'NODE_STATUS.{HOST}.json', row)
        with (Q / f'gpu_samples.{HOST}.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=row['epoch'], gpus=row['gpus'], loadavg=row['loadavg'],
                                    mem_available_gib=row['mem_available_gib'],
                                    running=[(j['id'], j.get('gpu')) for j in running])) + '\n')
        return snap

    def loop(self):
        JOBS.mkdir(parents=True, exist_ok=True)
        CANCEL.mkdir(parents=True, exist_ok=True)
        print(f'{utc()} supervisor start host {HOST} pid {os.getpid()} default {self.default}', flush=True)
        snap = {}
        while True:
            try:
                snap = self.step(snap)
            except Exception as exc:                 # transient shared-FS/IO errors must not kill the daemon
                import traceback
                print(f'{utc()} supervisor loop error: {exc!r}\n{traceback.format_exc()}', flush=True)
                time.sleep(10)

    def step(self, snap):
        if True:
            cfg = self.config()
            self.handle_cancels()
            for job in self.running():
                if not self.alive(job):
                    self.finalize(job)
                elif time.time() - self.sim_started(job) > arconf.EPISODE_TIMEOUT_S + BACKSTOP_S \
                        and not job.get('backstop'):
                    job['backstop'] = utc()
                    self.save(job)
                    self.terminate(job, 'backstop')
            problems = services_ok()
            if time.time() - self.last_status > 30:
                snap = self.status(cfg, problems)
                self.last_status = time.time()
            if cfg.get('paused') or problems:
                time.sleep(5)
                return snap
            running = self.running()
            per_gpu = {g: sum(1 for j in running if j.get('gpu') == g) for g in cfg['gpus']}
            # order by current running count per task so that one large enqueue cannot take every slot
            all_jobs = jobs()
            share = {}
            for j in all_jobs:
                if j['state'] == 'running':
                    k = j['task']    # share per task
                    share[k] = share.get(k, 0) + 1
            # Fair share per task: cap = ceil(total slots over all hosts / number of tasks with demand). A task at its cap
            # gets no new job; the share of tasks without demand goes to the others. Priority orders jobs within a task.
            demand = {j['task'] for j in all_jobs if j['state'] in ('queued', 'running')}
            fair_cap = max(1, -(-total_slots(self.last_raw or {}) // max(1, len(demand))))
            queued = sorted((j for j in all_jobs if j['state'] == 'queued'),
                            key=lambda j: (share.get(j['task'], 0), -j['priority'], j['created_epoch']))
            limits = {g: int((cfg.get('gpu_slots') or {}).get(str(g), cfg['slots_per_gpu'])) for g in cfg['gpus']}
            host_gap = float(cfg.get('host_launch_gap_s', HOST_LAUNCH_GAP_S))
            # a task above its cap may use a free slot only if no other task below its cap is waiting (no idle slots)
            for job in queued:
                k = job['task']
                if share.get(k, 0) >= fair_cap and any(
                        share.get(o['task'], 0) < fair_cap for o in queued
                        if o['state'] == 'queued' and o['task'] != k):
                    continue
                if time.time() - max(self.last_launch.values(), default=0.) < host_gap:
                    break                      # stagger Isaac starts on this host (run_episode also gates)
                pin = job.get('pin_gpu')
                free = [g for g in cfg['gpus'] if per_gpu[g] < limits[g] and (pin is None or g == pin)
                        and time.time() - self.last_launch.get(g, 0) > LAUNCH_GAP_S
                        and (g not in snap or snap[g]['mem_total'] - snap[g]['mem_used'] > MIN_FREE_MIB)]
                if not free:
                    continue
                gpu = min(free, key=lambda g: (per_gpu[g], g))
                if not self.launch(job, gpu):
                    continue
                per_gpu[gpu] += 1
                k = job['task']
                share[k] = share.get(k, 0) + 1
                snap = {}                      # memory figure is stale after a launch
            time.sleep(5)
            return snap


def total_slots(raw):
    """Total slots over all hosts (CONFIG hosts: gpus / gpu_slots / slots_per_gpu), for the per-task fair share."""
    tot = 0
    for h in (raw.get('hosts') or {}).values():
        if h.get('paused'):
            continue
        spg = int(h.get('slots_per_gpu', raw.get('slots_per_gpu', 4)))
        tot += sum(int((h.get('gpu_slots') or {}).get(str(g), spg)) for g in h.get('gpus', []))
    return tot or 64


def _epoch(text):
    return time.mktime(time.strptime(text, '%Y-%m-%dT%H:%M:%SZ')) - time.timezone


def _mem_available():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable'):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        return None


def cmd_supervise(a):
    lock = Q / f'SUPERVISOR.{HOST}.json'
    old = read(lock)
    if old and pidreg.proc_start(old.get('pid', -1)) == old.get('pid_start'):
        raise SystemExit(f'supervisor already running on {HOST}: pid {old["pid"]}')
    write(lock, dict(pid=os.getpid(), pid_start=pidreg.proc_start(os.getpid()), started_utc=utc()))
    Supervisor(parse_ints(a.gpus), a.slots_per_gpu).loop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('enqueue')
    e.add_argument('--bundle', required=True)
    e.add_argument('--task', required=True, choices=arconf.TASKS)
    e.add_argument('--split', required=True, choices=('dev', 'dev2', 'dev3', 'val', 'val2'))
    e.add_argument('--layouts', required=True, help='e.g. 0-7 or 0,3,7 (dev: seed 0 layouts 0-7; val: seed 1 layouts 40-47)')
    e.add_argument('--priority', type=int, default=3)   # dev jobs default to p3 (higher runs first; the orchestrator uses p5 for test-set runs)
    e.add_argument('--tag', default=None)
    e.add_argument('--branch', default=None)
    e.add_argument('--pin-gpu', type=int, default=None, help='run only on this GPU (measurements)')
    s = sub.add_parser('status')
    s.add_argument('--all', action='store_true')
    s.add_argument('--json', action='store_true')
    c = sub.add_parser('cancel')
    c.add_argument('ids', nargs='*')
    c.add_argument('--tag')
    c.add_argument('--branch')
    c.add_argument('--task')
    r = sub.add_parser('results')
    r.add_argument('--tag')
    r.add_argument('--branch')
    r.add_argument('--task')
    r.add_argument('--json', action='store_true')
    g = sub.add_parser('config')
    g.add_argument('--gpus')
    g.add_argument('--slots-per-gpu', type=int)
    g.add_argument('--gpu-slots', action='append', help='per-GPU override, e.g. --gpu-slots 2=3')
    g.add_argument('--clear-gpu-slots', action='store_true')
    g.add_argument('--host-launch-gap', type=float, help='seconds between launches on one host (new supervisors)')
    g.add_argument('--pause', action='store_true')
    g.add_argument('--resume', action='store_true')
    v = sub.add_parser('supervise')
    v.add_argument('--gpus', default='0-7')
    v.add_argument('--slots-per-gpu', type=int, default=2)
    a = ap.parse_args()
    dict(enqueue=cmd_enqueue, status=cmd_status, cancel=cmd_cancel, results=cmd_results, config=cmd_config,
         supervise=cmd_supervise)[a.cmd](a)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One native RoboDojo episode for a bundle directory -> RUN_DIR/RESULT.json.

    python tools/run_episode.py --bundle DIR --task T --split dev|dev2|dev3|val|val2 --layout I --gpu G --out RUN_DIR [--tag X]
                                [--timeout 2700] [--mailbox M]

* ``--bundle`` is any directory laid out like base/ (agraph/, runner/, runtime_params.json, tasks/<T>/).
  It is frozen first into the content-addressed store (``frozen/<T>/<sha16>/<T>/v001``,
  reused when identical), and the episode runs from that frozen copy (hash re-verified before and after).
  Only agraph/, runner/, runtime_params.json, tasks/__init__.py and tasks/<T>/ are part of a bundle.
* Splits: arconf.SPLITS (dev = native seed 0, layouts 0-7; robustness layouts dev2/dev3; test sets val/val2).
  TEST (seed 2, per-task lists arconf.TEST_LAYOUTS) is sealed: ``--split test`` additionally requires
  ``--unseal-test`` (orchestrator only).
* Harness reuse (base/runner/episode.py): layout preflight (native ``<task>_<int>.json`` + file sha256),
  first-observation instruction binding (shared ``instruction_bindings/``), launch of the candidate's
  runner/bootstrap.py (native src/eval_client/main.py; CUDA_VISIBLE_DEVICES=<gpu> unless AR_GPU_ISOLATION=0, physical --gpu),
  the input audit (CONTROL_SOURCE_AUDIT / RUNTIME_AUDIT), native result verification and the semantic
  request ledger. Runtime controller inputs are unchanged.
* status: verified (native scored, audit pass) | infra (objective evidence: native exit 99 + PhysX fatal
  line, or a service preflight failure before launch) | unresolved (no valid native result, watchdog
  timeout, verification problems) | interrupted (SIGTERM/SIGINT, e.g. queue cancel).
"""
import os
import sys

# tools/queue.py shadows the stdlib ``queue`` module: keep this directory LAST on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE] + [_HERE]

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

import arconf  # noqa: E402
import pidreg  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Fixed harness library: base/runner/episode.py (+ base/agraph via its sys.path insert).
episode = _load('ar_harness_episode', arconf.BASE / 'runner' / 'episode.py')
from agraph import candidates  # noqa: E402  (base/agraph, inserted by episode.py)

STATUS = {'verified': 'verified', episode.INFRA: 'infra', episode.UNRESOLVED: 'unresolved',
          'verification_failed': 'unresolved'}

# Native crash handling (Kit can segfault shortly after "app ready"; the carb crash reporter then symbolises frames for
# a long time). The process log is watched; on the crash-reporter line the process group is killed within seconds. A crash before the controller started is objective infrastructure evidence.
CRASH_MARKERS = ('[carb.crashreporter-breakpad.plugin] [crash] A crash has occurred',)
APP_READY = 'app ready'
SIMSITE = Path(_HERE) / 'simsite'
# Host-local Isaac launch gate: serialises Kit start-up per GPU (flock in /tmp, not on the shared file system).
LAUNCH_LOCK = Path(os.environ.get('AR_LAUNCH_LOCK', '/tmp/ar_isaac_launch.gpu{gpu}.lock'))
LAUNCH_GAP_S = float(os.environ.get('AR_LAUNCH_GAP_S', 20))           # min seconds between two Kit starts
LAUNCH_SETTLE_S = float(os.environ.get('AR_LAUNCH_SETTLE_S', 15))      # hold the gate this long after app ready
LAUNCH_HOLD_MAX_S = float(os.environ.get('AR_LAUNCH_HOLD_MAX_S', 300))  # never hold the gate longer than this
LAUNCH_WAIT_MAX_S = float(os.environ.get('AR_LAUNCH_WAIT_MAX_S', 480))  # stop waiting for the gate after this
# Periodic visual trace (tools/simsite/ar_video_trace.py, loaded by simsite/sitecustomize.py in the sim process):
# every AR_VIDEO_EVERY native actions (0 = off) -> RUN_DIR/video_trace/f_<actions>.jpg, stitched to trace.mp4 here.
VIDEO_EVERY = int(os.environ.get('AR_VIDEO_EVERY', 10) or 0)
VIDEO_FPS = int(os.environ.get('AR_VIDEO_FPS', 5))


def stitch_video_trace(out):
    """video_trace/f_*.jpg -> video_trace/trace.mp4 (ffmpeg, libx264, VIDEO_FPS). Never raises."""
    trace = out / 'video_trace'
    frames = sorted(trace.glob('f_*.jpg')) if trace.is_dir() else []
    report = dict(dir=str(trace), every_actions=VIDEO_EVERY, frames=len(frames), mp4=None)
    if not frames:
        return report
    report.update(first=frames[0].name, last=frames[-1].name)
    try:
        listing = trace / 'frames.txt'
        listing.write_text(''.join(f"file '{f.name}'\nduration {1 / VIDEO_FPS:.3f}\n" for f in frames)
                           + f"file '{frames[-1].name}'\n")
        mp4 = trace / 'trace.mp4'
        proc = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                               '-vf', 'format=yuv420p', '-r', str(VIDEO_FPS), '-c:v', 'libx264', '-crf', '28', '-preset', 'veryfast',
                               str(mp4)], capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and mp4.is_file():
            report['mp4'] = str(mp4)
        else:
            report['error'] = (proc.stderr or '')[-400:]
    except Exception as exc:
        report['error'] = repr(exc)
    try:
        rows = [json.loads(x) for x in (trace / 'index.jsonl').read_text().splitlines() if x.strip()]
        report['sample_seconds_total'] = round(sum(r.get('seconds', 0) for r in rows), 2)
    except (OSError, ValueError):
        pass
    return report


class LogWatch:
    """Incremental reader of process.log: first crash-marker line and the 'app ready' moment."""

    def __init__(self, path):
        self.path, self.offset, self.crash, self.ready_epoch, self._tail = Path(path), 0, None, None, ''

    def poll(self):
        try:
            with self.path.open('rb') as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return
        self.offset += len(chunk)
        text = self._tail + chunk.decode(errors='replace')
        lines = text.split('\n')
        self._tail = lines.pop()
        for line in lines:
            if self.ready_epoch is None and APP_READY in line:
                self.ready_epoch = time.time()
            if self.crash is None and any(m in line for m in CRASH_MARKERS):
                self.crash = dict(line=line.strip()[:400], epoch=time.time())


class LaunchGate:
    """Host-wide flock held from just before Popen until app ready + settle (or crash/exit/hold max)."""

    def __init__(self, gpu=0):
        self.lock = Path(str(LAUNCH_LOCK).format(gpu=gpu))
        self.fd, self.report = None, dict(lock=str(self.lock), gap_s=LAUNCH_GAP_S)

    def acquire(self):
        import fcntl
        t0 = time.monotonic()
        self.fd = open(self.lock, 'a+')
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 > LAUNCH_WAIT_MAX_S:
                    self.report['wait_timeout'] = True
                    self.fd.close()
                    self.fd = None
                    break
                time.sleep(2)
        last = 0.
        if self.fd is not None:
            try:
                self.fd.seek(0)
                last = float(json.loads(self.fd.read() or '{}').get('last_start_epoch', 0.))
            except (ValueError, OSError):
                last = 0.
            while time.time() - last < LAUNCH_GAP_S and time.monotonic() - t0 < LAUNCH_WAIT_MAX_S:
                time.sleep(1)
        self.report.update(wait_s=round(time.monotonic() - t0, 1), previous_start_epoch=last or None)

    def started(self, pid):
        if self.fd is not None:
            self.fd.seek(0)
            self.fd.truncate()
            self.fd.write(json.dumps(dict(last_start_epoch=time.time(), pid=pid, host=os.uname().nodename)))
            self.fd.flush()
        self.report['start_epoch'] = time.time()

    def release(self, why):
        if self.fd is not None:
            import fcntl
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
            self.report.update(released=why, held_s=round(time.time() - self.report.get('start_epoch', time.time()), 1))


def monitor(process, out, timeout, gate):
    """Wait for the sim -> (exit_code, failure_type, crash). Watchdog = ``timeout`` s from Popen; a crash-reporter
    line kills the group after a short grace; the launch gate is released at app ready + settle."""
    watch, started = LogWatch(out / 'process.log'), time.monotonic()
    while True:
        try:
            code = process.wait(timeout=2)
            gate.release('process_exit')
            watch.poll()
            return code, ('native_crash' if watch.crash else None), watch.crash
        except subprocess.TimeoutExpired:
            pass
        watch.poll()
        now = time.time()
        if gate.fd is not None and ((watch.ready_epoch and now - watch.ready_epoch >= LAUNCH_SETTLE_S)
                                    or watch.crash or now - gate.report.get('start_epoch', now) > LAUNCH_HOLD_MAX_S):
            gate.release('app_ready' if watch.ready_epoch and not watch.crash else ('crash' if watch.crash else 'hold_max'))
        if watch.crash and now - watch.crash['epoch'] >= 5:          # let the minidump be written, skip symbolisation
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            gate.release('crash')
            watch.crash['killed_after_s'] = round(time.time() - watch.crash['epoch'], 1)
            return process.returncode, 'native_crash', watch.crash
        if time.monotonic() - started >= timeout:
            episode._stop_group(process)
            gate.release('watchdog')
            return None, 'process_watchdog_timeout', watch.crash


def _ledger(fn, *args, **kwargs):
    """pidreg bookkeeping: the registry is only bookkeeping, a failure there must never lose an episode result."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f'[run_episode] pidreg {getattr(fn, "__name__", fn)} failed: {exc!r}', file=sys.stderr, flush=True)
        return None


def _owned(pid, out):
    """pid is alive on this host and its command line contains this RUN_DIR (guards against pid reuse)."""
    try:
        return str(out) in Path(f'/proc/{pid}/cmdline').read_text(errors='replace').replace('\0', ' ')
    except OSError:
        return False


def stop_tree(process, out, grace=20):
    """Stop this episode's whole process tree: the launcher's process group (a policy-server bundle's server and env
    client share it), then the server_pid/client_pid of PS_LAUNCH.json (only if their command line contains this RUN_DIR).
    Does nothing after a normal exit. Returns the stopped pids (RUN.json stopped_stragglers)."""
    def members():
        pids = []
        for d in os.listdir('/proc'):
            if d.isdigit():
                try:
                    if os.getpgid(int(d)) == process.pid:
                        pids.append(int(d))
                except OSError:
                    pass
        try:
            launch = json.loads((out / 'PS_LAUNCH.json').read_text())
        except (OSError, ValueError):
            launch = {}
        pids += [int(launch[k]) for k in ('server_pid', 'client_pid') if launch.get(k) and _owned(launch[k], out)]
        return sorted(set(pids) - {os.getpid()})

    stopped = members()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        left = members()
        if not left:
            break
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for pid in left:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        t0 = time.monotonic()
        while members() and time.monotonic() - t0 < grace:
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(1)
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return stopped


def controller_started(out):
    trace = out / 'trace.jsonl'
    try:
        return trace.exists() and '"controller_start"' in trace.read_text(errors='replace')
    except OSError:
        return True


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_split(split, layout, unseal_test=False, task=None):
    if split == 'test':
        if not unseal_test:
            raise SystemExit('TEST (seed 2) is sealed: refused (requires --unseal-test, orchestrator final only)')
        seed, per_task = arconf.TEST
        allowed = tuple(per_task.get(task) or ())
        if not allowed:
            raise SystemExit(f'no TEST layout list registered for task {task!r}')
    elif split in arconf.SPLITS:
        seed, allowed = arconf.split_layouts(split, task)
    else:
        raise SystemExit(f'unknown split {split!r}')
    if layout not in allowed:
        raise SystemExit(f'layout {layout} not in {split} split (seed {seed}, layouts {_ranges(allowed)})')
    return seed


def _ranges(ids):
    ids, out = sorted(ids), []
    for i in ids:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return ','.join(f'{a}-{b}' if a != b else str(a) for a, b in out)


def ensure_frozen(bundle, task):
    """Frozen candidate dir for ``bundle``: the bundle itself if it is already a verified frozen candidate
    of ``task``, else the content-addressed frozen copy (created once, under a lock)."""
    bundle = Path(bundle).resolve()
    if (bundle / 'CANDIDATE.json').exists():
        record = candidates.verify(bundle)
        if record['task'] != task:
            raise SystemExit(f'frozen bundle {bundle} is for task {record["task"]}')
        return bundle, record
    if not (bundle / 'tasks' / task / 'graph.py').is_file():
        raise SystemExit(f'bundle {bundle} has no tasks/{task}/graph.py')
    digest = candidates.manifest(bundle, task)['source_sha256']
    store = arconf.FROZEN / task / digest[:16]
    target = store / task / 'v001'
    import fcntl
    arconf.FROZEN.mkdir(parents=True, exist_ok=True)
    with open(arconf.FROZEN / '.freeze.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            candidates.freeze(store, task, source=bundle)
            (store / 'SOURCE.json').write_text(json.dumps(dict(bundle=str(bundle), frozen_utc=utc(),
                                                               source_sha256=digest), indent=2))
    record = candidates.verify(target)
    if record['source_sha256'] != digest:
        raise SystemExit(f'frozen copy {target} does not match bundle digest')
    return target, record


def scene_entry(task, split, layout, seed):
    name = f'{task}_{layout}.json'
    path = arconf.LAYOUT_ROOT / str(seed) / name
    if not path.is_file():
        raise SystemExit(f'layout file absent: {path}')
    return dict(task=task, seed_directory=seed, filename=name, layout_id=layout, file_sha256=file_sha(path),
                instruction_sha256=None, scene_token=f'{task}:{split}:{layout:03d}')


def services_preflight(mailbox):
    """Evidence that the SAM 3 service is up before a native process is spent."""
    problems = []
    try:
        with urllib.request.urlopen(arconf.SAM_ENDPOINT.rstrip('/') + '/health', timeout=10) as r:
            json.load(r)
    except Exception as exc:
        problems.append(f'SAM3 service unreachable at {arconf.SAM_ENDPOINT}: {exc}')
    return problems


def run(args):
    if args.task not in arconf.TASKS:
        raise SystemExit(f'unknown task {args.task}')
    seed = resolve_split(args.split, args.layout, args.unseal_test, args.task)
    bsplit = 'dev' if args.split in ('dev2', 'dev3') else ('val' if args.split == 'val2' else args.split)  # the harness runner only knows dev/val/test; val2 shares seed 1 with val (disjoint layout ids)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty; every run gets a fresh directory')
    candidate, record = ensure_frozen(args.bundle, args.task)
    entry = scene_entry(args.task, bsplit, args.layout, seed)
    preflight = episode.preflight_layout(args.task, bsplit, args.layout, entry, arconf.LAYOUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r'[^A-Za-z0-9_.-]', '-', args.tag or 'run')
    episode_id = f'{args.task}__{args.split}{args.layout:02d}__{tag}__{out.name}__' \
                 f'{hashlib.sha256(str(out).encode()).hexdigest()[:8]}'
    mailbox = Path(args.mailbox).resolve()
    if args.split in ('dev2', 'dev3'):
        # dev2/dev3 use their own instruction-binding file names (dev3 is seed 1: same layout ids as DEV, other instructions)
        binding_path = (Path(arconf.BINDINGS).resolve() / args.task / 'INSTRUCTION_BINDINGS'
                        / f'{args.split}_{int(args.layout):02d}.json')
    else:
        binding_path = episode.instruction_binding_path(arconf.BINDINGS, args.task, args.layout, bsplit)
    state = dict(task=args.task, split=args.split, layout=args.layout, seed_directory=seed, tag=args.tag,
                 bundle=str(Path(args.bundle).resolve()), candidate=str(candidate),
                 bundle_sha256=record['source_sha256'], gpu=args.gpu, episode_id=episode_id,
                 layout_file=entry['filename'], layout_file_sha256=entry['file_sha256'],
                 scene_token=entry['scene_token'], layout_preflight=preflight,
                 instruction_binding_path=str(binding_path), timeout_seconds=args.timeout,
                 host=os.uname().nodename, status='running', started_utc=utc(), privilege='none',
                 input_profile='sensor_rgbd', budget='native')
    (out / 'RUN.json').write_text(json.dumps(state, indent=2))

    problems = [] if args.skip_service_check or os.environ.get('AR_SKIP_SERVICE_CHECK') else services_preflight(mailbox)
    if problems:
        state.update(status='infra', infrastructure_evidence=[dict(kind='service_preflight', problems=problems)],
                     native_scored=False, finished_utc=utc(), wall_seconds=0.0)
        return finish(out, state, None)

    scene = dict(task=args.task, split=args.split, slot=args.layout, scene_token=entry['scene_token'],
                 file_sha256=entry['file_sha256'])
    env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
    env.update(AG_MAILBOX=str(mailbox), AG_EPISODE_ID=episode_id, ROBODOJO_RUN_ID=out.name,
               AG_INSTRUCTION_BINDING=str(binding_path), AG_SCENE=json.dumps(scene, sort_keys=True),
               AG_REGISTERED_INSTRUCTION_SHA256='', OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS=str(args.omp_threads), ROBODOJO_ROOT=str(arconf.ROBODOJO_ROOT))
    env['AR_SIMSITE'] = str(SIMSITE)
    if VIDEO_EVERY > 0:
        # tags ending in _v<N> get a denser visual trace (one frame every N actions), for diagnosing brief contacts
        m_every = re.search(r'_v(\d+)$', args.tag or '')
        every = max(1, int(m_every.group(1))) if m_every else VIDEO_EVERY
        env.update(AR_VIDEO_TRACE_DIR=str(out / 'video_trace'), AR_VIDEO_EVERY=str(every))
        state['video_trace_hook_sha256'] = file_sha(SIMSITE / 'ar_video_trace.py')
    # GPU isolation: the simulator sees only its physical GPU (CUDA ordinal 0); bootstrap gives the
    # renderer the physical index. Kit otherwise opens a CUDA context on all GPUs and the GPU audit fails.
    # AR_GPU_ISOLATION=0 leaves CUDA_VISIBLE_DEVICES unset. Set after sourcing the runtime env, which unsets it.
    # A bundle whose bootstrap does not support isolation runs without it.
    env.pop('AR_CUDA_VISIBLE', None)
    isolation_capable = 'CUDA_INDEX = 0 if ISOLATED else args.gpu' in (candidate / 'runner' / 'bootstrap.py').read_text()
    if os.environ.get('AR_GPU_ISOLATION', '1') != '0' and isolation_capable:
        env['AR_CUDA_VISIBLE'] = str(args.gpu)
    state['cuda_visible_devices'] = env.get('AR_CUDA_VISIBLE')
    state['gpu_isolation'] = ('on' if env.get('AR_CUDA_VISIBLE') else
                              'off: candidate runner does not support isolation' if not isolation_capable else 'off: AR_GPU_ISOLATION=0')
    shell = ('if [ -f "$1" ]; then source "$1"; fi; export PYTHONPATH="$AR_SIMSITE${PYTHONPATH:+:$PYTHONPATH}"; '
             'py="$2"; shift 2; unset CUDA_VISIBLE_DEVICES; '
             'if [ -n "$AR_CUDA_VISIBLE" ]; then export CUDA_VISIBLE_DEVICES="$AR_CUDA_VISIBLE"; fi; exec "$py" "$@"')
    state['simsite_sitecustomize_sha256'] = file_sha(SIMSITE / 'sitecustomize.py')
    command = ['bash', '-c', shell, 'ag-episode', args.runtime_env, args.python, '-u',
               str(candidate / 'runner' / 'bootstrap.py'), '--task', args.task, '--seed', str(seed),
               '--layout-file', entry['filename'], '--layout-sha256', entry['file_sha256'], '--gpu', str(args.gpu),
               '--out', str(out), '--candidate', str(candidate)]
    interrupted = []

    def on_signal(signum, _frame):
        interrupted.append(signum)
        raise KeyboardInterrupt

    gate = LaunchGate(args.gpu)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        gate.acquire()
    except KeyboardInterrupt:
        state.update(status='interrupted', native_scored=False, finished_utc=utc(), wall_seconds=0.0,
                     launch_gate=gate.report, failure_type='interrupted_in_launch_gate')
        return finish(out, state, interrupted)
    state['launch_gate'] = gate.report
    started = time.monotonic()
    state['process_started_epoch'] = time.time()
    # On normal exit, exception or SIGTERM the whole process tree is stopped in finally (stop_tree); the result is written
    # first (finish), pidreg bookkeeping last and never raising (_ledger).
    with (out / 'process.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        failure = crash = None
        try:
            gate.started(process.pid)
            _ledger(pidreg.add, process.pid, command[6:], role='sim_episode', gpu=args.gpu, run_dir=str(out),
                    parent_pid=os.getpid())
            state['pid'] = process.pid
            (out / 'RUN.json').write_text(json.dumps(state, indent=2))
            state['exit_code'], failure, crash = monitor(process, out, args.timeout, gate)
        except KeyboardInterrupt:
            failure = 'interrupted_by_signal'
        except Exception as exc:
            failure = 'harness_exception'
            state['error'] = repr(exc)
        finally:
            # further SIGTERM/SIGINT during wrap-up are only recorded (job still marked interrupted), result writing continues
            signal.signal(signal.SIGTERM, lambda s, f: interrupted.append(s))
            signal.signal(signal.SIGINT, lambda s, f: interrupted.append(s))
            gate.release('finally')
            state['launch_gate'] = gate.report
            stragglers = stop_tree(process, out)
            if stragglers:
                state['stopped_stragglers'] = stragglers
            if failure in ('interrupted_by_signal', 'harness_exception'):
                state['exit_code'] = process.returncode
    state['wall_seconds'] = time.monotonic() - started
    state['process_finished_epoch'] = time.time()
    try:
        result = settle(out, state, entry, binding_path, candidate, mailbox, failure, crash, interrupted)
    finally:
        _ledger(pidreg.end, process.pid, exit_code=process.returncode)
    return result


def settle(out, state, entry, binding_path, candidate, mailbox, failure, crash, interrupted):
    """Post-process a finished sim process into RUN.json / RESULT.json."""
    if VIDEO_EVERY > 0:
        try:
            state['video_trace'] = stitch_video_trace(out)
        except Exception as exc:
            state['video_trace'] = dict(error=repr(exc))
    if failure:
        state['failure_type'] = failure
    state.update(episode.classify_attempt(out, entry, binding_path, state['exit_code'], state.get('failure_type')))
    if crash is not None:
        state['native_crash'] = crash
        if state['status'] != 'verified' and not controller_started(out):
            # Objective infrastructure evidence: the native process crashed (crash-reporter line) before the
            # controller started, so no policy code had run. Admits the queue's single infra retry.
            state['status'] = episode.INFRA
            state['infrastructure_evidence'] = (state.get('infrastructure_evidence') or []) + [
                dict(kind='native_crash_before_control', log_line=crash['line'], exit_code=state['exit_code'])]
    try:
        candidates.verify(candidate)
        state['candidate_unchanged_after_episode'] = True
    except ValueError as exc:
        state.update(candidate_unchanged_after_episode=False, status='verification_failed', error=str(exc))
    state['harness_status'] = state['status']
    state['status'] = 'interrupted' if interrupted else STATUS.get(state['status'], 'unresolved')
    if failure == 'process_watchdog_timeout':
        state['status'] = 'unresolved'
    try:
        state['semantic'] = episode.semantic_ledger(mailbox, state['episode_id'])
        # Policy-server bundles record requests under the server's own episode id (PS_LAUNCH.json server_episode_id),
        # so look up both ids and keep the one with more calls (only one is non-empty).
        try:
            sid = json.loads((out / 'PS_LAUNCH.json').read_text()).get('server_episode_id')
        except (OSError, ValueError):
            sid = None
        if sid:
            alt = episode.semantic_ledger(mailbox, sid)
            if (alt.get('calls') or 0) > (state['semantic'].get('calls') or 0):
                state['semantic'] = dict(alt, ledger_episode_id=sid)
    except Exception as exc:
        state['semantic'] = dict(error=repr(exc))
    state['finished_utc'] = utc()
    return finish(out, state, interrupted)


def finish(out, state, interrupted):
    (out / 'RUN.json').write_text(json.dumps(state, indent=2, default=str))
    sem = state.get('semantic') or {}
    videos = state.get('videos') or sorted(str(p) for p in (out / 'eval_result').rglob('*.mp4'))
    result = dict(
        task=state['task'], split=state['split'], layout=state['layout'], seed_directory=state['seed_directory'],
        tag=state.get('tag'), bundle=state['bundle'], candidate=state['candidate'],
        bundle_sha256=state['bundle_sha256'], status=state['status'],
        success=state.get('native_success') if state['status'] == 'verified' else None,
        native_score=state.get('native_score') if state['status'] == 'verified' else None,
        native_actions=state.get('native_actions'), native_action_limit=state.get('native_action_limit'),
        control_actions=state.get('control_actions'), wall_seconds=state.get('wall_seconds'),
        semantic_calls=sem.get('calls'), semantic_cost_usd=sem.get('cost_usd'),
        semantic_errors=sem.get('errors'),
        audit_pass=state.get('input_audit_passed'), controller_status=state.get('controller_status'),
        graph_terminal=state.get('graph_terminal'), instruction=state.get('instruction'),
        problems=state.get('problems'), failure_type=state.get('failure_type'), error=state.get('error'),
        infrastructure_evidence=state.get('infrastructure_evidence'), gpu=state['gpu'], host=state.get('host'),
        episode_id=state['episode_id'], started_utc=state.get('started_utc'), finished_utc=state.get('finished_utc'),
        run_dir=str(out), videos=videos, video_trace=(state.get('video_trace') or {}).get('mp4'), trace=str(out / 'trace.jsonl'), graph_trace=str(out / 'graph_trace.json'),
        process_log=str(out / 'process.log'), native_result=state.get('native_result'))
    if result['status'] == 'verified' and result['audit_pass'] is not True:
        result['status'] = 'unresolved'
        result['problems'] = (result['problems'] or []) + ['input audit not passed']
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: result[k] for k in ('task', 'split', 'layout', 'status', 'success', 'native_score',
                                             'native_actions', 'wall_seconds', 'semantic_calls')}))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--task', required=True, choices=arconf.TASKS)
    ap.add_argument('--split', choices=('dev', 'dev2', 'dev3', 'val', 'val2', 'test'))
    ap.add_argument('--layout', type=int)
    ap.add_argument('--gpu', type=int, help='physical GPU index on this host')
    ap.add_argument('--out')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--timeout', type=float, default=arconf.EPISODE_TIMEOUT_S)
    ap.add_argument('--mailbox', default=str(arconf.MAILBOX))
    ap.add_argument('--runtime-env', default=None)
    ap.add_argument('--python', default=None)
    ap.add_argument('--omp-threads', type=int, default=4)
    ap.add_argument('--skip-service-check', action='store_true', help='debug only')
    ap.add_argument('--unseal-test', action='store_true', help='orchestrator final TEST only; never for selection')
    ap.add_argument('--freeze-only', action='store_true', help='freeze/verify the bundle, print the candidate, exit')
    args = ap.parse_args()
    args.runtime_env = args.runtime_env or arconf.runtime_env()
    args.python = args.python or os.environ.get('AR_SIM_PYTHON') or arconf.native_python()
    if args.freeze_only:
        candidate, record = ensure_frozen(args.bundle, args.task)
        print(json.dumps(dict(candidate=str(candidate), bundle_sha256=record['source_sha256'])))
        return
    missing = [k for k in ('split', 'layout', 'gpu', 'out') if getattr(args, k) is None]
    if missing:
        ap.error('required: ' + ', '.join('--' + k for k in missing))
    result = run(args)
    sys.exit(0 if result['status'] in ('verified',) else (3 if result['status'] == 'interrupted' else 2))


if __name__ == '__main__':
    main()

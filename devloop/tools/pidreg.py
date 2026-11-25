"""Process registry: every process started by the loop infrastructure is recorded in state/pids.json.

Kill only by a recorded PID (and only on the host that recorded it). Entries are never deleted;
ended processes get ``ended_utc``. Concurrency: fcntl lock on a sibling lock file (works across
processes on one host; cross-host writers use per-host files, see below).
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import time

CODE = Path(__file__).resolve().parents[1]
RS_HOST = os.environ.get('AR_RS_HOST', '')    # the orchestration host (Claude Code CLI credentials)
# Authoritative per-host file (cross-host fcntl on the shared FS is not relied upon): the RS host uses
# state/pids.json, any other host state/pids.<hostname>.json; rows are mirrored into state/pids.json.
_HOST = socket.gethostname()
REGISTRY = Path(os.environ.get('AR_PIDS') or str(CODE / 'state' / ('pids.json' if _HOST == RS_HOST
                                                                   else f'pids.{_HOST}.json')))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


MAIN = CODE / 'state' / 'pids.json'


def _update(fn):
    """Apply ``fn`` to this host's registry and mirror it into the shared state/pids.json (rows carry host).

    The registry is only bookkeeping and must never crash the caller: concurrent os.replace on the shared FS can raise
    ESTALE, so each write is retried 3 times and then given up."""
    result = None
    for path in ([REGISTRY] + ([MAIN] if REGISTRY != MAIN and not os.environ.get('AR_PIDS') else [])):
        for i in range(3):
            try:
                r = _update_one(path, fn)
                if path == REGISTRY:
                    result = r
                break
            except OSError:
                time.sleep(0.5 * (i + 1))
    return result


def _update_one(path, fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text()) if path.exists() and path.stat().st_size else {}
        except ValueError:
            data = {}
        data.setdefault('note', 'Processes started by autoresearch infra. Kill only by these PIDs, on the recorded host.')
        data.setdefault('processes', [])
        result = fn(data)
        tmp = path.with_name(f'.{path.name}.{_HOST}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(data, indent=1) + '\n')
        os.replace(tmp, path)
        return result


def proc_start(pid):
    """Kernel start time (clock ticks since boot) of a local pid, or None."""
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def add(pid, cmd, role, gpu=None, **extra):
    row = dict(pid=int(pid), host=socket.gethostname(), proc_start=proc_start(pid), started_utc=utc(),
               role=role, gpu=gpu, cmd=cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))[:600], **extra)
    _update(lambda d: d['processes'].append(row))
    return row


def end(pid, **extra):
    host = socket.gethostname()

    def fn(d):
        for row in reversed(d['processes']):
            if row['pid'] == int(pid) and row['host'] == host and 'ended_utc' not in row:
                row.update(ended_utc=utc(), **extra)
                return row
    return _update(fn)


def alive(row):
    """True only if the recorded pid is live on THIS host with the same kernel start time."""
    if row.get('host') != socket.gethostname() or 'ended_utc' in row:
        return False
    return proc_start(row['pid']) is not None and proc_start(row['pid']) == row.get('proc_start')


def find(role=None, **match):
    data = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {'processes': []}
    return [r for r in data['processes'] if (role is None or r.get('role') == role)
            and all(r.get(k) == v for k, v in match.items())]

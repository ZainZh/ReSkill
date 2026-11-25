#!/usr/bin/env python3
"""Zero-API check: report the number of run-time agent (API) requests of each episode and decide whether it made
none. Frozen task programs must make zero run-time agent calls. Read-only; changes no result and no audit semantics.

  python3 -B tools/zero_api_check.py RUN_DIR [RUN_DIR ...]
  python3 -B tools/zero_api_check.py --tag TAG [--branch B] [--task T] [--json] [--all-attempts]

Four independent pieces of evidence per episode (any non-zero means not zero-API):
  1. RESULT.json semantic_calls (receipts counted by episode.semantic_ledger when run_episode finishes);
  2. RUN.json semantic.unanswered_requests (requests sent without a receipt);
  3. current file counts in mailbox/<episode_id>/ requests/, responses/, claims/ (late requests after wrap-up count too);
  4. rows for the episode_id in mailbox/RS_SERVICE_LEDGER.jsonl (service-side count, including failed retries).
zero_api = all four are 0. Only verified episodes with a passed input audit count; the others are listed as usual.
--tag etc. select episodes from queue/results.jsonl (and state/results.jsonl); by default only the latest attempt of
each job is used and superseded rows are skipped.
"""
import argparse
import json
import os
from pathlib import Path
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE] + [_HERE]
import arconf  # noqa: E402

E = arconf.EVIDENCE
MAILBOX = arconf.MAILBOX
RESULT_FILES = (arconf.QUEUE / 'results.jsonl', arconf.STATE / 'results.jsonl')


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:       # noqa: BLE001
        return None


def ledger_counts(episode_ids):
    counts = {e: 0 for e in episode_ids}
    path = MAILBOX / 'RS_SERVICE_LEDGER.jsonl'
    if not episode_ids or not path.exists():
        return counts
    with open(path, errors='replace') as stream:
        for line in stream:
            for e in episode_ids:        # cheap substring prefilter before parsing
                if e in line:
                    try:
                        if json.loads(line).get('episode_id') == e:
                            counts[e] += 1
                    except ValueError:
                        pass
    return counts


def check_dir(run_dir):
    run_dir = Path(run_dir)
    res = _read(run_dir / 'RESULT.json') or {}
    run = _read(run_dir / 'RUN.json') or {}
    eid = res.get('episode_id') or run.get('episode_id')
    box = MAILBOX / eid if eid else None
    files = {k: (len(list((box / k).glob('*'))) if box and (box / k).is_dir() else 0)
             for k in ('requests', 'responses', 'claims')}
    sem = run.get('semantic') or {}
    def pick(k):
        return res[k] if res.get(k) is not None else run.get(k)
    return dict(run_dir=str(run_dir), episode_id=eid, task=pick('task'), split=pick('split'), layout=pick('layout'),
                status=pick('status'), success=res.get('success'),
                native_score=res.get('native_score'), audit_pass=res.get('audit_pass'),
                result_semantic_calls=res.get('semantic_calls'),
                unanswered_requests=sem.get('unanswered_requests'),
                mailbox_files=files, result_missing=not res)


def select(tag=None, branch=None, task=None, all_attempts=False):
    rows = {}
    for path in RESULT_FILES:
        if not path.exists():
            continue
        for line in path.read_text(errors='replace').splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if tag and r.get('tag') != tag or branch and r.get('branch') != branch or task and r.get('task') != task:
                continue
            if r.get('superseded') and not all_attempts:
                continue
            if r.get('run_dir'):
                rows[r['run_dir']] = r
    if not all_attempts:            # latest attempt per job
        best = {}
        for r in rows.values():
            k = r.get('job_id') or r['run_dir']
            if k not in best or (r.get('attempt') or 0) > (best[k].get('attempt') or 0):
                best[k] = r
        rows = {r['run_dir']: r for r in best.values()}
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='*')
    ap.add_argument('--tag')
    ap.add_argument('--branch')
    ap.add_argument('--task')
    ap.add_argument('--all-attempts', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    dirs = list(a.run_dirs)
    if a.tag or a.branch or a.task:
        dirs += select(a.tag, a.branch, a.task, a.all_attempts)
    if not dirs:
        ap.error('no run dirs (give RUN_DIRs or --tag/--branch/--task)')
    rows = [check_dir(d) for d in dirs]
    led = ledger_counts([r['episode_id'] for r in rows if r['episode_id']])
    for r in rows:
        r['service_ledger_rows'] = led.get(r['episode_id'], 0)
        n = [r['result_semantic_calls'], r['unanswered_requests'], *r['mailbox_files'].values(), r['service_ledger_rows']]
        r['semantic_requests'] = max(int(x or 0) for x in n)
        r['zero_api'] = (not r['result_missing']) and r['result_semantic_calls'] is not None and r['semantic_requests'] == 0
        r['counts_as_zero_api'] = r['zero_api'] and r['status'] == 'verified' and r['audit_pass'] is True
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{r['task']} {r['split']}{r['layout']} {r['status']:<11} score={r['native_score']} "
                  f"sem_calls={r['result_semantic_calls']} unanswered={r['unanswered_requests']} "
                  f"mailbox={r['mailbox_files']} ledger={r['service_ledger_rows']} -> "
                  f"{'ZERO_API' if r['zero_api'] else 'API_USED' if r['semantic_requests'] else 'UNKNOWN'}  {r['run_dir']}")
        z = sum(r['counts_as_zero_api'] for r in rows)
        print(f"summary: {len(rows)} episodes, zero_api {sum(r['zero_api'] for r in rows)}, "
              f"verified+audit+zero_api {z}, api_used {sum(r['semantic_requests'] > 0 for r in rows)}")
    return 0 if all(r['zero_api'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())

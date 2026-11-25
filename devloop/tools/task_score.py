#!/usr/bin/env python3
"""Task score J per task x branch x bundle version, from the queue results.

    task_score.py [--task T] [--out-md state/TASK_SCORES.md] [--out-json state/task_scores.json] [--print]

Reads queue/results.jsonl (supervisor-written, authoritative) merged with code/state/results.jsonl
(deduplicated by job_id+attempt; superseded infra attempts excluded). A "version" is one bundle sha256
within one branch (a branch evolves; every edit is a new frozen sha).

Per DEV layout (arconf.SPLITS['dev']: seed 0, layouts 0-7) a version gets
  * verified episodes: native success (0/1) and native score (0-100);
  * watchdog timeouts (status unresolved, failure_type process_watchdog_timeout): counted as success 0, score 0
    (the 45-min cap is part of the protocol, so a timeout is a controller outcome);
  * infra / interrupted / other unresolved: not counted (listed as excluded).
Layout value = mean over its counted episodes; J = 100 * mean_layout(success) + mean_layout(score), computed
ONLY for versions whose counted episodes cover all 8 DEV layouts ("complete DEV"). Versions with partial coverage
are listed separately as screening rows (no J). The test set (val: seed 1, 40-47) is summarised in its own table.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE] + [_HERE]

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

import arconf  # noqa: E402

TIMEOUT = 'process_watchdog_timeout'


def load_rows(paths):
    seen, rows = set(), []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            key = (r.get('job_id'), r.get('attempt'), r.get('run_dir'))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return [r for r in rows if not r.get('superseded')]


def counted(r):
    """(success, score) for a counted episode, else None."""
    st = arconf.effective_status(r)
    if st == 'verified':
        return float(bool(r.get('success'))), float(r.get('native_score') or 0.)
    if st == 'unresolved' and r.get('failure_type') == TIMEOUT:
        return 0., 0.
    return None


def summarise(rows, split, layouts):
    groups = {}
    for r in rows:
        if r.get('split') != split:
            continue
        key = (r.get('task'), r.get('branch') or '(none)', r.get('bundle_sha256') or '?')
        g = groups.setdefault(key, dict(task=key[0], branch=key[1], bundle_sha256=key[2], episodes=0,
                                        per_layout={}, excluded=[], outside=[], tags=set(), wall=[], cost=0.,
                                        first_utc=None, last_utc=None))
        g['episodes'] += 1
        g['tags'].add(r.get('tag'))
        if r.get('wall_seconds'):
            g['wall'].append(float(r['wall_seconds']))
        g['cost'] += float(r.get('semantic_cost_usd') or 0.)
        t = r.get('finished_utc')
        if t:
            g['first_utc'] = min(g['first_utc'] or t, t)
            g['last_utc'] = max(g['last_utc'] or t, t)
        layout = r.get('layout')
        if layout not in layouts:
            g['outside'].append(layout)
            continue
        value = counted(r)
        if value is None:
            g['excluded'].append(dict(layout=layout, status=arconf.effective_status(r), failure_type=r.get('failure_type'),
                                      run_dir=r.get('run_dir')))
            continue
        g['per_layout'].setdefault(layout, []).append(value)
    out = []
    for g in groups.values():
        per = {k: (sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs), len(vs))
               for k, vs in sorted(g['per_layout'].items())}
        n_counted = sum(v[2] for v in per.values())
        succ = sum(v[0] * v[2] for v in per.values())
        complete = set(per) >= set(layouts)
        row = dict(task=g['task'], branch=g['branch'], bundle_sha256=g['bundle_sha256'], split=split,
                   episodes=g['episodes'], counted_episodes=n_counted, layouts_covered=sorted(per),
                   complete=complete, successes=int(round(succ)),
                   mean_score=(sum(v[1] for v in per.values()) / len(per)) if per else None,
                   success_rate=(sum(v[0] for v in per.values()) / len(per)) if per else None,
                   per_layout={str(k): dict(success=v[0], score=round(v[1], 2), n=v[2]) for k, v in per.items()},
                   excluded=g['excluded'], outside_split_layouts=sorted(set(g['outside']), key=str),
                   tags=sorted(t for t in g['tags'] if t), mean_wall_s=(sum(g['wall']) / len(g['wall'])) if g['wall'] else None,
                   semantic_cost_usd=round(g['cost'], 2), first_utc=g['first_utc'], last_utc=g['last_utc'])
        row['J'] = round(100 * row['success_rate'] + row['mean_score'], 2) if complete else None
        out.append(row)
    return out


def fmt(v, nd=1):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def markdown(dev, val, layouts):
    lines = [f'# Task scores (generated {time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())} by tools/task_score.py)', '',
             f'DEV = seed {arconf.SPLITS["dev"][0]} layouts {layouts[0]}-{layouts[-1]} ({len(layouts)} layouts). '
             'J = 100 x success rate + mean native score (both averaged over layouts; repeated runs of a layout are averaged first).',
             'Counted: verified episodes; 45-min timeouts (unresolved/process_watchdog_timeout) count as score 0, no success. '
             'Not counted: infra, interrupted, other unresolved (listed as "excluded"). Version = one bundle sha within a branch.', '']
    for task in arconf.TASKS:
        rows = [r for r in dev if r['task'] == task]
        vrows = [r for r in val if r['task'] == task]
        if not rows and not vrows:
            continue
        lines += [f'## {task} (action limit {arconf.STEP_LIM.get(task, "?")})', '']
        full = sorted((r for r in rows if r['complete']), key=lambda r: (-r['J'], -r['mean_score']))
        part = sorted((r for r in rows if not r['complete']), key=lambda r: (r['branch'], r['last_utc'] or ''))
        if full:
            lines += ['### Complete DEV (ranked)', '',
                      '| rank | branch | sha | J | successes | mean score | counted | per-layout scores | excluded | mean wall s | semantic cost $ |',
                      '|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
            for i, r in enumerate(full, 1):
                per = ' '.join(f"{k}:{v['score']:g}{'✓' if v['success'] >= 1 else ''}" for k, v in r['per_layout'].items())
                lines.append(f"| {i} | {r['branch']} | `{r['bundle_sha256'][:12]}` | **{fmt(r['J'], 1)}** | "
                             f"{r['successes']} | {fmt(r['mean_score'])} | {r['counted_episodes']} | {per} | "
                             f"{len(r['excluded'])} | {fmt(r['mean_wall_s'], 0)} | {r['semantic_cost_usd']:.2f} |")
            lines.append('')
        if part:
            lines += ['### Screening only (incomplete DEV coverage, no J)', '',
                      '| branch | sha | covered layouts | successes | mean score | counted | excluded | mean wall s |',
                      '|---|---|---|---:|---:|---:|---:|---:|']
            for r in part:
                per = ' '.join(f"{k}:{v['score']:g}{'✓' if v['success'] >= 1 else ''}" for k, v in r['per_layout'].items()) or '—'
                lines.append(f"| {r['branch']} | `{r['bundle_sha256'][:12]}` | {per} | {r['successes']} | "
                             f"{fmt(r['mean_score'])} | {r['counted_episodes']} | {len(r['excluded'])} | "
                             f"{fmt(r['mean_wall_s'], 0)} |")
            lines.append('')
        if vrows:
            lines += ['### VAL (seed 1 layouts 40-47)', '',
                      '| branch | sha | coverage | successes | mean score | J (if complete) |', '|---|---|---|---:|---:|---:|']
            for r in vrows:
                lines.append(f"| {r['branch']} | `{r['bundle_sha256'][:12]}` | {len(r['layouts_covered'])}/8 | "
                             f"{r['successes']} | {fmt(r['mean_score'])} | {fmt(r['J'])} |")
            lines.append('')
    if len(lines) <= 5:
        lines.append('(no results yet)')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task')
    ap.add_argument('--results', action='append', help='results jsonl (default: queue + state)')
    ap.add_argument('--out-md', default=str(arconf.STATE / 'TASK_SCORES.md'))
    ap.add_argument('--out-json', default=str(arconf.STATE / 'task_scores.json'))
    ap.add_argument('--print', action='store_true')
    a = ap.parse_args()
    rows = load_rows(a.results or [arconf.QUEUE / 'results.jsonl', arconf.STATE / 'results.jsonl'])
    if a.task:
        rows = [r for r in rows if r.get('task') == a.task]
    dev_seed, dev_layouts = arconf.SPLITS['dev']
    val_seed, val_layouts = arconf.SPLITS['val']
    dev = summarise(rows, 'dev', dev_layouts)
    val = summarise(rows, 'val', val_layouts)
    md = markdown(dev, val, dev_layouts)
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text(md)
    Path(a.out_json).write_text(json.dumps(dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                dev_split=dict(seed=dev_seed, layouts=list(dev_layouts)),
                                                val_split=dict(seed=val_seed, layouts=list(val_layouts)),
                                                dev=dev, val=val), indent=1, ensure_ascii=False) + '\n')
    if a.print:
        print(md)
    else:
        print(f'wrote {a.out_md} and {a.out_json}: {len(dev)} DEV versions, {len(val)} VAL versions')


if __name__ == '__main__':
    main()

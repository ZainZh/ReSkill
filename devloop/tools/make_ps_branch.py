#!/usr/bin/env python3
"""Create a policy-server (ps) branch from an adopted version: branches/<task>/<id> (default ps0).

    python3 tools/make_ps_branch.py --task T --source FROZEN_OR_BRANCH_DIR [--id ps0] [--note TEXT] [--ps-base DIR]

Copies from the source: agraph/ (minus vendor/), runtime_params.json, tasks/__init__.py, tasks/<T>/.
Takes from code/base_ps (--ps-base; the shared policy-server layer, i.e. runner/ and agraph/vendor/ as carried by
the frozen programs under programs/): runner/ (whole), agraph/vendor/ (whole) and agraph/grasp_verify.py. tasks/<T>/graph.py is never
touched. runtime_params.json gets ``ps_fingers: command`` (the only finger source: official obs gripper command; jaw
contact by wrist-camera vision, runner/ps_compat.py) and loses any ``ps_diag`` flag.
Reports (and refuses unless --force) when the source's runner/ or agraph/vendor/ differ from code/base (a task that
customised the shared files needs a manual port). Never overwrites an existing branch.
"""
import argparse
import filecmp
import hashlib
import json
from pathlib import Path
import shutil
import socket
import time

CODE = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '*.bak*')


def differing(a, b):
    out = []
    for p in sorted(Path(a).rglob('*.py')):
        if '__pycache__' in p.parts or '.bak' in p.name:
            continue
        q = Path(b) / p.relative_to(a)
        if not q.exists() or not filecmp.cmp(p, q, shallow=False):
            out.append(str(p.relative_to(a)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--id', default='ps0')
    ap.add_argument('--note', default='')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--ps-base', default=str(CODE / 'base_ps'), help='policy-server shared layer (default code/base_ps)')
    a = ap.parse_args()
    PS = Path(a.ps_base).resolve()
    src, dst = Path(a.source).resolve(), CODE / 'branches' / a.task / a.id
    if dst.exists():
        raise SystemExit(f'{dst} exists')
    if not (src / 'tasks' / a.task / 'graph.py').is_file():
        raise SystemExit(f'{src} has no tasks/{a.task}/graph.py')
    custom = dict(runner=[f for f in differing(src / 'runner', CODE / 'base' / 'runner')
                          if f not in ('episode.py', 'rs_semantic_service.py')],      # harness-only, not used by ps
                  vendor=differing(src / 'agraph' / 'vendor', CODE / 'base' / 'agraph' / 'vendor'))
    if (custom['runner'] or custom['vendor']) and not a.force:
        raise SystemExit(f'source customises shared files {custom}; port by hand or pass --force')
    dst.mkdir(parents=True)
    shutil.copytree(src / 'agraph', dst / 'agraph', ignore=IGNORE)
    shutil.rmtree(dst / 'agraph' / 'vendor')
    shutil.copytree(PS / 'agraph' / 'vendor', dst / 'agraph' / 'vendor', ignore=IGNORE)
    shutil.copytree(PS / 'runner', dst / 'runner', ignore=IGNORE)
    shutil.copy2(PS / 'agraph' / 'grasp_verify.py', dst / 'agraph' / 'grasp_verify.py')
    runtime = json.loads((src / 'runtime_params.json').read_text())
    runtime.pop('ps_diag', None)
    runtime['ps_fingers'] = 'command'
    runtime['_ps_fingers_note'] = ('policy-server: finger joints = official obs gripper COMMAND only (no measured '
                                   'fingers in the obs; no image-based finger estimator); jaw contact by wrist-camera '
                                   'vision (agraph/grasp_verify.py via runner/ps_compat.py)')
    (dst / 'runtime_params.json').write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + '\n')
    (dst / 'tasks').mkdir()
    shutil.copy2(src / 'tasks' / '__init__.py', dst / 'tasks' / '__init__.py')
    shutil.copytree(src / 'tasks' / a.task, dst / 'tasks' / a.task, ignore=IGNORE)
    graph = dst / 'tasks' / a.task / 'graph.py'
    (dst / 'BRANCH.json').write_text(json.dumps(dict(
        task=a.task, id=a.id, source=str(src), architecture='policy_server', shared_from=str(PS),
        source_customised_shared_files=custom, graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), created_host=socket.gethostname(),
        note=a.note), indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(branch=str(dst), customised=custom)))


if __name__ == '__main__':
    main()

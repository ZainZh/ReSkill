#!/usr/bin/env python3
"""Rebuild the seed workspace base/ from the frozen programs.

The seed workspace holds the shared skill library and harness (agraph/, runner/, runtime_params.json). These files
are byte-identical to files published under programs/, so they are not duplicated here. This script copies every file
listed in SHARED_FILES.json from its source program into base/ and verifies its sha256. tools/run_episode.py uses
base/runner/ as the fixed harness; tools/branch.py --from base creates a candidate with a skeleton task graph.

    python3 devloop/base/materialize.py [--programs PATH_TO/programs] [--force]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--programs', default=str(HERE.parents[1] / 'programs'), help='directory holding programs/<task>/')
    ap.add_argument('--force', action='store_true', help='overwrite files that already exist with other content')
    a = ap.parse_args()
    programs = Path(a.programs).resolve()
    spec = json.loads((HERE / 'SHARED_FILES.json').read_text())['files']
    done = 0
    for rel, row in spec.items():
        src = programs / Path(row['source']).relative_to('programs')
        dst = HERE / rel
        if hashlib.sha256(src.read_bytes()).hexdigest() != row['sha256']:
            raise SystemExit(f'sha256 mismatch for {src}')
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == row['sha256']:
            continue
        if dst.exists() and not a.force:
            raise SystemExit(f'{dst} exists with other content (use --force)')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        done += 1
    print(f'{done} files copied, {len(spec) - done} already in place')


if __name__ == '__main__':
    main()

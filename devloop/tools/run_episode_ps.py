#!/usr/bin/env python3
"""One native episode of a POLICY-SERVER bundle -> RUN_DIR/RESULT.json (same receipt as tools/run_episode.py).

    python tools/run_episode_ps.py --bundle DIR --task T --split S --layout I --gpu G --out RUN_DIR [--tag X]
                                   [--diag-fingers] [--diag-ik CASES.json] [other tools/run_episode.py options]

A policy-server bundle (code/base_ps layout: runner/ps_server.py present) carries its own episode launcher as
``runner/bootstrap.py``: it starts the websocket policy server (our controller) and the official env client
(native main.py + demo_policy loop) and assembles result.json afterwards. tools/run_episode.py therefore runs such a
bundle unchanged (freezing, layout preflight, GPU isolation, launch gate, crash watch, verification, RESULT.json);
this wrapper only refuses non-ps bundles and sets the optional diagnosis switches, then hands over to run_episode.py.
The queue needs no new mode: enqueue a ps bundle as usual.

Diagnosis switches (offline analysis only, never read by the policy process):
  --diag-fingers   env client logs measured finger joints after every action (DIAG_FINGERS_GT.jsonl) and the
                   simulator camera matrices every 25 actions (DIAG_CAMERAS_SIM.jsonl); the policy server saves
                   the bottom half of both wrist images per action count (diag_wrist/, write-only)
  --diag-ik FILE   env client solves the IK cases of FILE with the simulator's own planner (DIAG_IK_SIM.json)
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    argv = list(sys.argv[1:])
    if '--diag-fingers' in argv:
        argv.remove('--diag-fingers')
        os.environ['AR_PS_DIAG_FINGERS'] = '1'
        os.environ['AR_PS_DIAG_WRIST'] = '1'
    if '--diag-ik' in argv:
        i = argv.index('--diag-ik')
        os.environ['AR_PS_DIAG_FINGERS'] = '1'
        os.environ['AR_PS_DIAG_IK'] = str(Path(argv[i + 1]).resolve())
        del argv[i:i + 2]
    if '--bundle' in argv:
        bundle = Path(argv[argv.index('--bundle') + 1])
        if not (bundle / 'runner' / 'ps_server.py').is_file():
            raise SystemExit(f'{bundle} is not a policy-server bundle (no runner/ps_server.py); use tools/run_episode.py')
    os.execv(sys.executable, [sys.executable, '-B', '-u', str(HERE / 'run_episode.py')] + argv)


if __name__ == '__main__':
    main()

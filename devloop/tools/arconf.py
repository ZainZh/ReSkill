"""Shared constants for the ReSkill development loop (paths, tasks, data splits, sealed TEST lists, timeouts).

All loop state lives under one directory, ``RESKILL_ROOT``:

    $RESKILL_ROOT/code/          this devloop/ directory (tools/, base/, prompts/, docs/) plus branches/ and state/
    $RESKILL_ROOT/frozen/        content-addressed frozen bundles: <task>/<sha16>/<task>/v001
    $RESKILL_ROOT/runs/          one RUN_DIR per episode attempt
    $RESKILL_ROOT/queue/         file-based job queue shared by all simulation hosts
    $RESKILL_ROOT/mailbox/       per-episode request ledger read by the harness (frozen programs make no requests)
    $RESKILL_ROOT/instruction_bindings/, $RESKILL_ROOT/deployment/

``RESKILL_ROOT`` (or ``AR_EVIDENCE``) selects the directory; by default it is the directory two levels above this
file, i.e. the layout above with devloop/ checked out as ``$RESKILL_ROOT/code``. With several simulation hosts it
must be a shared file system mounted at the same path on every host.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(os.environ.get('RESKILL_ROOT') or os.environ.get('AR_EVIDENCE') or _DEFAULT_ROOT)
RESKILL_ROOT = EVIDENCE
CODE = EVIDENCE / 'code'
BASE = CODE / 'base'                       # seed workspace (shared library + harness, restored by base/materialize.py)
TOOLS = CODE / 'tools'
STATE = CODE / 'state'
FROZEN = EVIDENCE / 'frozen'               # content-addressed frozen bundles: <task>/<sha16>/<task>/v001
RUNS = EVIDENCE / 'runs'
MAILBOX = EVIDENCE / 'mailbox'
BINDINGS = EVIDENCE / 'instruction_bindings'
QUEUE = EVIDENCE / 'queue'
DEPLOY = EVIDENCE / 'deployment'
ROBODOJO_ROOT = Path(os.environ.get('ROBODOJO_ROOT', str(Path.home() / 'github' / 'RoboDojo')))
LAYOUT_ROOT = ROBODOJO_ROOT / 'Assets' / 'Eval_Layout' / 'RoboDojo' / 'arx_x5'
# Loopback SAM 3 segmentation service on every simulation host (tools/services.sh, tools/sam3_service.py).
SAM_ENDPOINT = os.environ.get('AR_SAM_ENDPOINT', 'http://127.0.0.1:18891')
# Host that runs the orchestrator and the development agents (Claude Code CLI credentials live there).
RS_HOST = os.environ.get('AR_RS_HOST', '')
TASKS = ('classify_objects', 'classify_objects_by_language', 'pack_objects_into_box', 'organize_table',
         'build_tower', 'stack_blocks', 'stack_blocks_by_language', 'stack_bowls', 'store_tools_in_toolbox',
         'general_pickup')
# Native per-task action limit (task/RoboDojo/tasks/<task>.py ``self.step_lim``).
# Informational only: the harness reads the live value from the native env (EnvironmentBoundary.step_lim).
STEP_LIM = {'classify_objects': 1100, 'classify_objects_by_language': 1100, 'pack_objects_into_box': 1300,
            'organize_table': 1000, 'build_tower': 1050, 'stack_blocks': 550, 'stack_blocks_by_language': 400,
            'stack_bowls': 800, 'store_tools_in_toolbox': 900, 'general_pickup': 200}

# split -> (native seed directory, allowed layout ids). DEV = seed 0 layouts 0-7, VAL = seed 1 layouts 40-47.
SPLITS = {'dev': (0, tuple(range(0, 8))),
          # model-selection test set ("test layouts" in the paper); hidden from development agents
          'val': (1, tuple(range(40, 48))),
          # robustness layouts, group 1: seed 0 layouts 8-23 (not part of J)
          'dev2': (0, tuple(range(8, 24))),
          # robustness layouts, group 2: seed 1 layouts 0-39, disjoint from the test sets; selection uses L10-39
          'dev3': (1, tuple(range(0, 40))),
          # second model-selection test set, used once a task's failures on the first test set have been inspected
          'val2': (1, tuple(range(48, 55)))}
# Final validation (seed 2) is sealed: per task, the first 50 seed-2 layouts not used in revision or model selection
# (ascending; fewer where fewer such layouts exist: classify_objects_by_language 49, stack_blocks_by_language 30;
# stack_blocks also excludes layouts that nearly coincide with a public demonstration). run_episode refuses TEST
# without --unseal-test; the queue never accepts it.
TEST_SEED = 2
TEST_LAYOUTS = {
    'classify_objects': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54),
    'classify_objects_by_language': (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54),
    'pack_objects_into_box': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    'organize_table': (0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53),
    'build_tower': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52),
    'stack_blocks': (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52),
    'stack_blocks_by_language': (2, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 19, 20, 22, 23, 27, 29, 31, 32, 35, 36, 37, 38, 44, 47, 49, 50, 51, 52),
    'stack_bowls': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49),
    'store_tools_in_toolbox': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52),
    'general_pickup': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50),
}
TEST = (TEST_SEED, TEST_LAYOUTS)


def split_layouts(split, task=None):
    """(seed, allowed layouts) of a non-TEST split."""
    return SPLITS[split]


EPISODE_TIMEOUT_S = 2700


def conda_sh():
    """conda activation script whose presence marks a host with a local RoboDojo conda install."""
    return Path(os.environ.get('AR_CONDA_SH', str(Path.home() / 'miniconda3' / 'etc' / 'profile.d' / 'conda.sh')))


def runtime_env():
    """Shell file sourced before the native simulator. Host with a local conda install: ROBODOJO_ROOT/.rs/runtime_env.sh;
    other worker nodes (conda env unpacked without conda.sh): deployment/runtime_env_node.sh. AR_RUNTIME_ENV overrides."""
    explicit = os.environ.get('AR_RUNTIME_ENV')
    if explicit:
        return explicit
    local = ROBODOJO_ROOT / '.rs' / 'runtime_env.sh'
    if conda_sh().exists() and local.exists():
        return str(local)
    return str(DEPLOY / 'runtime_env_node.sh')


def native_python():
    return os.environ.get('AR_PYTHON', str(Path.home() / 'miniconda3' / 'envs' / 'RoboDojo' / 'bin' / 'python'))


def is_rs_host():
    return bool(RS_HOST) and socket.gethostname() == RS_HOST


def effective_status(r):
    """Scoring status of a result row. An episode whose harness ledger records request errors is an infrastructure
    failure (infra: neither success nor failure); the original row is kept unchanged."""
    st = r.get('status')
    if st in ('verified', 'unresolved') and (r.get('semantic_errors') or 0) > 0:
        return 'infra'
    return st

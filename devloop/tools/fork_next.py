"""Parallel exploration: structurally different fork lines next to the exploitation line (fork-and-flush).

Besides the exploitation line loop.<task>.sN (which develops the incumbent), a task may have up to two structurally
different forks fork.<task>.F1 / F2, each with its own workspace branches/<task>/F<k>_<name>. After every session
(the agent writes REPORT.md and exits) the orchestrator flushes: a fresh agent continues in the same workspace from its
NOTES.md. Pruning: a fork that does not produce a version surpassing the incumbent within its budget (b_explore=2
sessions) is stopped and archived; validated local changes of a fork can be transferred to the main line.

    python3 -B tools/fork_next.py start <task> F1|F2 <direction_file>
    python3 -B tools/fork_next.py next fork.<task>.F<k>.sN [note_file]

Base prompts: state/prompts/loop/<task>.md (AR_LOOP_PROMPTS); fork prompts are written to state/prompts/fork/
(AR_FORK_PROMPTS).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE] + [_HERE]
import arconf  # noqa: E402

C = arconf.CODE; A = C / 'state/agents'
L = Path(os.environ.get('AR_LOOP_PROMPTS') or arconf.STATE / 'prompts' / 'loop')
P = Path(os.environ.get('AR_FORK_PROMPTS') or arconf.STATE / 'prompts' / 'fork'); P.mkdir(parents=True, exist_ok=True)


def latest_loop_report(task):
    rs = sorted((d for d in A.glob(f'loop.{task}.s*') if (d / 'REPORT.md').exists()),
                key=lambda d: (int(re.match(r'\d+', d.name.rsplit('.s', 1)[1]).group()), d.name))
    return (rs[-1].name, (rs[-1] / 'REPORT.md').read_text()[:3500]) if rs else ('(none)', '(the loop line has not reported yet)')


FORK = """

## Your role: fork line {name} (parallel exploration)
- This task has an exploitation line **loop.{task}.sN** (develops the incumbent) and up to two forks
  **fork.{task}.F1 / F2**. You are {fk}.
- **Your solution structure must differ from the loop line and from the other fork.** Do not change the stage the loop
  line is currently changing (see its latest report below); that would duplicate work.
- Workspace: in your first session create your own W with `python3 -B tools/branch.py create --task {task}
  --id {fk}_<short_name> --from branches/{task}/<branch directory of the incumbent>`; later sessions continue in the
  same W, starting from the end of its NOTES.md. Every change sits behind a switch; with the switches off, behaviour is
  identical to the incumbent.
- Evaluation is the same as for the loop line: screening on 3 DEV layouts, then full DEV (task score J), then the
  robustness layouts (dev2, dev3 L10-39), with the incumbent run in the same time window. Smoke-test every switch on a
  few episodes first to confirm the mechanism actually triggers. Adoption is decided by the orchestrator.
- Pruning: a fork that does not surpass the incumbent within its budget is stopped. Every REPORT must therefore state
  this direction's estimated ceiling and whether it is worth continuing.
- Run-time rules are unchanged: control and planning are code; observation-only inputs (simulator RGB-D from the
  official observation, calibration, joint encoders, URDF, instruction text); commanded finger values only; no
  run-time API calls; the final-validation layouts stay sealed.
- REPORT.md must contain a "## Root-cause analysis" section: failures by cause and their shares, and the precision the
  blocking stage needs versus what is achieved.

### Your direction ({fk})
{focus}

### Latest report of the loop line ({lname}; reference only, to avoid duplication)
{lrep}
"""


def launch(name, body):
    out = P / f'{name}.md'; out.write_text(body)
    print(subprocess.run([str(C / 'tools/agent_tmux.sh'), 'start', name, str(out)], capture_output=True, text=True).stdout.strip())


if sys.argv[1] == 'start':
    task, fk, focus = sys.argv[2], sys.argv[3], Path(sys.argv[4]).read_text()
    name = f'fork.{task}.{fk}.s1'
    lname, lrep = latest_loop_report(task)
    base = (L / f'{task}.md').read_text()
    (P / f'{task}.{fk}.focus.md').write_text(focus)
    launch(name, base + FORK.format(name=name, task=task, fk=fk, focus=focus, lname=lname, lrep=lrep))
else:
    old = sys.argv[2]
    m = re.match(r'fork\.(.+)\.([A-Z]+\d)\.s(\d+)[a-z]?$', old); task, fk, n = m.group(1), m.group(2), int(m.group(3))
    name = f'fork.{task}.{fk}.s{n + 1}'
    focus = (P / f'{task}.{fk}.focus.md').read_text()
    lname, lrep = latest_loop_report(task)
    rep = (A / old / 'REPORT.md').read_text() if (A / old / 'REPORT.md').exists() else '(the previous session wrote no REPORT)'
    extra = Path(sys.argv[3]).read_text() if len(sys.argv) > 3 else ''
    base = (L / f'{task}.md').read_text()
    launch(name, base + FORK.format(name=name, task=task, fk=fk, focus=focus, lname=lname, lrep=lrep)
           + (f'\n\n## Report of this fork\'s previous session ({old})\nFirst read state/agents/{old}/REPORT.md and the end of '
              'its stream.jsonl, continue in the same W, and do not enqueue the same versions again.\n\n')
           + rep[:4000] + '\n\n' + extra)

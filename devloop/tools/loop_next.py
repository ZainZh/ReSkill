"""Flush the exploitation line of a task: start loop.<task>.s(N+1) after loop.<task>.sN has reported.

The new prompt = the task's base prompt + the previous session's REPORT (truncated) + an optional orchestrator note.
The new agent works in the same workspace and continues from its NOTES.md. The base prompt of each task lives in
state/prompts/loop/<task>.md (written by the orchestrator from prompts/DEVELOPMENT_AGENT.md; override the directory
with AR_LOOP_PROMPTS).

    python3 -B tools/loop_next.py loop.<task>.sN [note_file]
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
P = Path(os.environ.get('AR_LOOP_PROMPTS') or arconf.STATE / 'prompts' / 'loop')
old = sys.argv[1]
m = re.match(r'loop\.(.+)\.s(\d+)[a-z]?$', old); task, n = m.group(1), int(m.group(2))
new = f'loop.{task}.s{n + 1}'
base = (P / f'{task}.md').read_text()
rep = (A / old / 'REPORT.md').read_text() if (A / old / 'REPORT.md').exists() else '(the previous session wrote no REPORT)'
extra = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else ''
body = base + (f'\n\n## Report of the previous session ({old})\nFirst read state/agents/{old}/REPORT.md and the end of '
               f'its stream.jsonl, then continue; do not enqueue the same versions again.\n\n') + rep[:4000] + '\n\n' + extra
out = P / f'{new}.md'; out.write_text(body)
print(subprocess.run([str(C / 'tools/agent_tmux.sh'), 'start', new, str(out)], capture_output=True, text=True).stdout.strip())

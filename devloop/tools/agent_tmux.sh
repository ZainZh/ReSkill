#!/usr/bin/env bash
# Start an unattended development agent (claude -p) in the tmux session ar-agents; it keeps running after the
# orchestrator session exits.
#   tools/agent_tmux.sh start <name> <prompt_file>   # name e.g. loop.<task>.s4
#   tools/agent_tmux.sh list                          # state of every agent (running / done rc=... / REPORT)
#   tools/agent_tmux.sh stop <name>                   # close its window
#   tools/agent_tmux.sh resume <old_name> <new_name>  # relaunch an agent that exited unexpectedly with its prompt
# Logs: state/agents/<name>/{prompt.md,stream.jsonl,EXIT,REPORT.md}; the agent writes REPORT.md itself before ending.
set -uo pipefail
C=${RESKILL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/code
A=$C/state/agents
SESSION=ar-agents
MODEL=${AR_AGENT_MODEL:-claude-opus-5-5}
mkdir -p "$A"

HEADLESS_NOTE='
## You run unattended (claude -p inside tmux; the orchestrator is not waiting on you interactively)
- Your process exits when you end your reply. Never end your reply while waiting for simulations; poll with blocking
  commands of at most about 9 minutes per Bash call, e.g.
  `timeout 540 bash -c "until python3 -B tools/queue.py results --tag <tag> 2>/dev/null | grep -q \"verified N/N\"; do sleep 60; done"`, then call again until all results are in.
- Do not use sub-agents and do not wait with run_in_background.
- Before ending, write your final report (at most 25 lines, format as in "Report back" above) to `REPORT_PATH`. Without
  this file the orchestrator does not receive your result.
'

case "${1:-}" in
start)
    name=$2; src=$3; d=$A/$name
    [ -e "$d" ] && { echo "exists: $d (pick a new name)"; exit 1; }
    [ -s "$src" ] || { echo "prompt file missing or empty: $src"; exit 1; }
    mkdir -p "$d"
    { cat "$src"; printf '%s\n' "${HEADLESS_NOTE//REPORT_PATH/$d/REPORT.md}"; } > "$d/prompt.md"
    tmux has-session -t $SESSION 2>/dev/null || tmux new-session -d -s $SESSION -n idle
    tmux new-window -d -t $SESSION -n "$name" \
        "bash $C/tools/agent_run.sh $d $MODEL"
    date -u +%FT%TZ > "$d/STARTED"
    echo "started $name"
    ;;
resume)
    # relaunch with the old agent's prompt after an unexpected exit: resume <old_name> <new_name>
    old=$A/$2; new=$3
    [ -f "$old/prompt.md" ] || { echo "no prompt: $old"; exit 1; }
    tmp=$(mktemp)
    { printf '%s\n\n' "## Resume note: the previous agent ($2) exited unexpectedly and did not hand back. First read the end of W/NOTES.md, this branch's queue.py results and status --all, establish which jobs are queued/finished and the current code version in W, then continue; do not enqueue the same jobs again."
      sed '/^## You run unattended/,$d' "$old/prompt.md"; } > "$tmp"
    bash "$0" start "$new" "$tmp"; rm -f "$tmp"
    ;;
list)
    live=$(tmux list-windows -t $SESSION -F '#{window_name}' 2>/dev/null)
    for d in "$A"/*/; do
        n=$(basename "$d")
        if [ -f "$d/EXIT" ]; then s="done $(cat "$d/EXIT")"
        elif printf '%s\n' "$live" | grep -qxF "$n"; then s=running
        else s="stopped (no window)"; fi
        [ -f "$d/REPORT.md" ] && s="$s REPORT"
        printf '%-44s %s\n' "$n" "$s"
    done
    ;;
stop)
    # tmux -t misparses window names containing dots: look the window up by id
    wid=$(tmux list-windows -t $SESSION -F '#{window_id} #{window_name}' | awk -v n="$2" '$2==n{print $1}')
    [ -n "$wid" ] && tmux kill-window -t "$SESSION:$wid" && echo "stopped $2" || echo "no window $2"
    [ -f "$A/$2/EXIT" ] || echo "stopped-by-orchestrator $(date -u +%FT%TZ)" > "$A/$2/EXIT"
    ;;
*) sed -n 2,6p "$0"; exit 1 ;;
esac

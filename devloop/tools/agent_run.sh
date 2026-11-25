#!/bin/bash
# Wrapper around `claude -p` for one headless agent session (started by tools/agent_tmux.sh).
# If a session ends with its tool calls written out as plain text (no REPORT.md), it is resumed with --resume, at most 3 times.
#   tools/agent_run.sh <agent_dir> <model>
d=$1; M=$2
C=${RESKILL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/code
cd "$C"
claude -p "$(cat $d/prompt.md)" --model $M --permission-mode auto --output-format stream-json --verbose > $d/stream.jsonl 2> $d/stderr.log
rc=$?
for i in 1 2 3; do
  [ -f $d/REPORT.md ] && break
  [ -f $d/EXIT ] && break
  tail -n 1 $d/stream.jsonl | grep -q '"result":"[^"]\{0,20\}<tool_calls>' || break
  sid=$(grep -o '"session_id":"[^"]*"' $d/stream.jsonl | head -1 | cut -d'"' -f4)
  [ -n "$sid" ] || break
  echo "auto-resume $i $(date -u +%FT%TZ)" >> $d/AUTORESUME.log
  claude -p "Your previous reply wrote the tool calls as plain text and nothing was executed. Continue the task with real tool calls; do not finish before REPORT.md is written." --resume $sid --model $M --permission-mode auto --output-format stream-json --verbose >> $d/stream.jsonl 2>> $d/stderr.log
  rc=$?
done
echo "rc=$rc $(date -u +%FT%TZ)" > $d/EXIT

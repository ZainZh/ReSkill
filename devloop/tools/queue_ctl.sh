#!/usr/bin/env bash
# Queue supervisor control (run on each GPU host; tmux session ar-queue, setsid nohup without tmux).
#   tools/queue_ctl.sh start [--gpus 0-7] [--slots-per-gpu 2] | stop | status
# Stops only by registered PID. Stopping the supervisor does not kill running episodes (separate sessions);
# a restarted supervisor adopts still-running jobs (checked by pid + start time) and finalizes them as usual.
set -uo pipefail
E=${RESKILL_ROOT:-${AR_EVIDENCE:-$(cd "$(dirname "$0")/../.." && pwd)}}
export RESKILL_ROOT=$E
T=$E/code/tools
HOST=$(hostname)
LOG=$E/queue/supervisor.$HOST.log
PIDF=$E/queue/supervisor.$HOST.pid
mkdir -p "$E/queue"

pyenv() {
    # local conda install (cf. arconf.runtime_env); worker nodes use deployment/runtime_env_node.sh
    local conda_sh=${AR_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
    local rd=${ROBODOJO_ROOT:-$HOME/github/RoboDojo}
    if [ -f "$conda_sh" ] && [ -f "$rd/.rs/runtime_env.sh" ]; then
        source "$rd/.rs/runtime_env.sh" >/dev/null 2>&1
    else
        source "$E/deployment/runtime_env_node.sh"
    fi
}

alive() {
    [ -f "$PIDF" ] || return 1
    local pid; pid=$(cat "$PIDF")
    [ -n "$pid" ] && [ -d "/proc/$pid" ] || return 1
    python3 - "$pid" "$T" <<'PY'
import sys; sys.path.append(sys.argv[2])
import pidreg
rows = [r for r in pidreg.find(pid=int(sys.argv[1])) if r.get('host') == pidreg._HOST and 'ended_utc' not in r]
sys.exit(0 if rows and pidreg.alive(rows[-1]) else 1)
PY
}

case ${1:-status} in
    _run)
        shift
        pyenv
        # The simulator needs no outside network. Started from a shell with proxy variables, Isaac's omniclient fetches
        # assets through the proxy and every episode hangs at reset when the proxy stalls, so proxies are cleared.
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        echo $$ > "$PIDF"
        cd "$E/code"
        exec python -B -u "$T/queue.py" supervise "$@" >> "$LOG" 2>&1 ;;
    start)
        shift
        if alive; then echo "supervisor already running pid $(cat "$PIDF")"; exit 0; fi
        rm -f "$PIDF"
        cmd="bash $T/queue_ctl.sh _run $*"
        if command -v tmux >/dev/null 2>&1; then
            tmux has-session -t ar-queue 2>/dev/null && tmux new-window -d -t ar-queue "$cmd" \
                || tmux new-session -d -s ar-queue "$cmd"
        else
            setsid nohup $cmd > /dev/null 2>&1 < /dev/null &
        fi
        for _ in $(seq 1 20); do [ -s "$PIDF" ] && break; sleep 1; done
        pid=$(cat "$PIDF" 2>/dev/null)
        [ -n "$pid" ] || { echo "supervisor failed to start; see $LOG"; exit 1; }
        python3 - "$pid" "$T" <<'PY'
import sys; sys.path.append(sys.argv[2])
import pidreg
pid = int(sys.argv[1])
pidreg.add(pid, open(f'/proc/{pid}/cmdline').read().replace('\0', ' '), role='queue_supervisor')
PY
        echo "supervisor started pid $pid (log $LOG)" ;;
    stop)
        if ! alive; then echo "supervisor not running on $HOST"; exit 0; fi
        pid=$(cat "$PIDF")
        kill -TERM "$pid"
        for _ in $(seq 1 10); do [ -d "/proc/$pid" ] || break; sleep 1; done
        [ -d "/proc/$pid" ] && kill -KILL "$pid"
        python3 -c "import sys; sys.path.append('$T'); import pidreg; pidreg.end($pid, stopped_by='queue_ctl stop')"
        echo "supervisor stopped (pid $pid)" ;;
    status)
        if alive; then echo "supervisor RUNNING pid $(cat "$PIDF") host $HOST"; else echo "supervisor not running on $HOST"; fi
        pyenv; python -B "$T/queue.py" status ;;
    *) echo "usage: $0 start [--gpus 0-7] [--slots-per-gpu N] | stop | status"; exit 2 ;;
esac

#!/usr/bin/env bash
# Loop services on every simulation host: the SAM 3 segmentation service, bound to 127.0.0.1.
#   tools/services.sh start|stop|status|restart [sam3]
# sam3: SAM 3 /segment (+ /detect, /ground) (tools/sam3_service.py), port 18891; AR_SAM3_GPU selects the GPU
#       (default: the GPU with the least memory in use).
# Runs in tmux session ar-services (setsid nohup without tmux). Processes are only stopped by PIDs registered in
# state/pids*.json, never by name. The Python environment is looked up under $RESKILL_ROOT/envs (override with
# AR_SAM3_PY).
set -uo pipefail
E=${RESKILL_ROOT:-${AR_EVIDENCE:-$(cd "$(dirname "$0")/../.." && pwd)}}
export RESKILL_ROOT=$E
C=$E/code
T=$C/tools
LOGS=$E/services
mkdir -p "$LOGS"
HOST=$(hostname)
SAM3_PORT=18891
SAM3_PY=${AR_SAM3_PY:-$E/envs/localvision/bin/python}   # venv (py3.12, torch 2.7.1+cu126)
SESSION=ar-services

pidfile() { echo "$LOGS/$1.$HOST.pid"; }

alive() {   # alive <name>: pidfile pid is live AND matches the registry row (same proc start)
    local f; f=$(pidfile "$1")
    [ -f "$f" ] || return 1
    local pid; pid=$(cat "$f")
    [ -n "$pid" ] && [ -d "/proc/$pid" ] || return 1
    python3 - "$pid" "$T" <<'PY'
import sys; sys.path.insert(0, sys.argv[2])
import pidreg
pid = int(sys.argv[1])
rows = [r for r in pidreg.find(pid=pid) if r.get('host') == pidreg._HOST and 'ended_utc' not in r]
sys.exit(0 if rows and pidreg.alive(rows[-1]) else 1)
PY
}

run_service() {   # executed inside tmux/nohup: becomes the service process via exec
    local name=$1
    echo $$ > "$(pidfile "$name")"
    case $name in
        sam3)
            export CUDA_VISIBLE_DEVICES=$2 PYTHONDONTWRITEBYTECODE=1
            cd /tmp
            exec "$SAM3_PY" -B -u "$T/sam3_service.py" --port "$SAM3_PORT" --mem-gb "${AR_SAM3_MEM_GB:-10}" >> "$LOGS/sam3.$HOST.log" 2>&1 ;;
    esac
}

pick_gpu() {   # AR_SAM3_GPU, else the GPU with the least memory in use right now
    if [ -n "${AR_SAM3_GPU:-}" ]; then echo "$AR_SAM3_GPU"; return; fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1
}

start_one() {
    local name=$1
    if alive "$name"; then echo "$name already running pid $(cat "$(pidfile "$name")")"; return 0; fi
    rm -f "$(pidfile "$name")"
    local gpu; gpu=$(pick_gpu)
    local cmd="bash $T/services.sh _run $name $gpu"
    if command -v tmux >/dev/null 2>&1; then
        tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n idle "bash"
        tmux new-window -d -t "$SESSION" -n "$name" "$cmd"
    else
        setsid nohup $cmd > /dev/null 2>&1 < /dev/null &
    fi
    for _ in $(seq 1 30); do [ -s "$(pidfile "$name")" ] && break; sleep 1; done
    local pid; pid=$(cat "$(pidfile "$name")" 2>/dev/null)
    [ -n "$pid" ] || { echo "$name failed to start"; return 1; }
    python3 - "$pid" "$name" "$gpu" "$T" <<'PY'
import sys; sys.path.insert(0, sys.argv[4])
import pidreg
pid, name, gpu = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
pidreg.add(pid, open(f'/proc/{pid}/cmdline').read().replace('\0', ' '), role=f'service_{name}', gpu=gpu)
PY
    echo "$name started pid $pid"
    for _ in $(seq 1 300); do
        if python3 -c "import urllib.request,sys; sys.exit(0 if b'sam3_shared' in urllib.request.urlopen('http://127.0.0.1:$SAM3_PORT/health', timeout=2).read() else 1)" 2>/dev/null; then echo "SAM3_READY gpu $gpu port $SAM3_PORT"; return 0; fi
        [ -d "/proc/$pid" ] || { echo "sam3 exited during load (see $LOGS/sam3.$HOST.log)"; return 1; }
        sleep 1
    done
    echo "SAM3 readiness timeout (see $LOGS/sam3.$HOST.log)"; return 1
}

stop_one() {
    local name=$1
    if ! alive "$name"; then echo "$name not running (no live registered pid)"; return 0; fi
    local pid; pid=$(cat "$(pidfile "$name")")
    kill -TERM "$pid"
    for _ in $(seq 1 20); do [ -d "/proc/$pid" ] || break; sleep 1; done
    [ -d "/proc/$pid" ] && kill -KILL "$pid"
    python3 - "$pid" "$T" <<'PY'
import sys; sys.path.insert(0, sys.argv[2])
import pidreg; pidreg.end(int(sys.argv[1]), stopped_by='services.sh stop')
PY
    echo "$name stopped (pid $pid)"
}

status_one() {
    local name=$1
    if alive "$name"; then echo "$name: RUNNING pid $(cat "$(pidfile "$name")") host $HOST"; else echo "$name: not running on $HOST"; fi
    python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:$SAM3_PORT/health', timeout=3)); print('  sam3: healthy gpu', d['gpu'], 'port', d['port'], 'peak GiB', d['mem_peak_gib'], 'stats', d['stats'])" 2>/dev/null || echo "  sam3 endpoint: unreachable"
}

targets() {
    case ${1:-sam3} in
        sam3|all|auto) echo sam3 ;;
        *) echo "unknown service $1" >&2; exit 2 ;;
    esac
}

cmd=${1:-status}
case $cmd in
    _run) run_service "$2" "${3:-}" ;;
    start) for n in $(targets "${2:-sam3}"); do start_one "$n"; done ;;
    stop) for n in $(targets "${2:-sam3}"); do stop_one "$n"; done ;;
    restart) for n in $(targets "${2:-sam3}"); do stop_one "$n"; start_one "$n"; done ;;
    status) for n in $(targets "${2:-sam3}"); do status_one "$n"; done ;;
    *) echo "usage: $0 start|stop|status|restart [sam3]"; exit 2 ;;
esac

#!/bin/bash
# Cluster control, run from any machine with SSH access to both nodes.
#
#   cluster.sh check                       verify both nodes and the direct link, changes nothing
#   cluster.sh setup                       copy this repository to both nodes and build llama.cpp
#   cluster.sh fetch                       download the source GGUF and the draft on the head node
#   cluster.sh trim [MASK]                 cut the source GGUF with a mask, FINAL_MASK by default
#   cluster.sh up                          start the configuration set in cluster.env
#   cluster.sh down                        stop the server and the RPC worker
#   cluster.sh status                      show processes, memory and background jobs
#   cluster.sh bench | smoke               measure speed, or ask questions with checkable answers
#   cluster.sh preflight MODEL_DIR [GIB]   check that a model fits, without loading anything
#   cluster.sh serve MODEL_DIR [args]      start llama-server on any model
#   cluster.sh eval MODEL_DIR NAME [args]  measure quality, results in eval/results on the head node
#   cluster.sh rpc-start [--tcp] [--debug] | rpc-stop | serve-stop
set -euo pipefail
# A French locale prints 114,5 and awk would then read it as 114.
export LC_ALL=C
cd "$(dirname "$0")/.."
[ -f cluster.env ] || { echo "cluster.env is missing: copy cluster.env.example and adapt it" >&2; exit 1; }
source cluster.env
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
BIN=$REMOTE_ROOT/llama.cpp/build/bin
SERVER_PORT=8081
# A private socket keeps our sessions apart from any tmux the same account uses on the nodes.
TMUX_CMD="tmux -L minichonk"

# Weights go to the GPU as pinned memory that the kernel cannot see in a process RSS, so an
# overcommit makes the OOM killer shoot small innocent processes. Two defenses:
#  - preflight refuses a model that does not fit in what each node has available right now.
#  - OOM_FIRST makes our processes the first victims if the kernel has to choose anyway.
OOM_FIRST='echo 1000 > /proc/self/oom_score_adj;'
# Measured at 96 experts, beyond half the weights: 11 GiB on the worker with a 32k context,
# and 11 GiB on the head while computing multiple choice scores with -np 5.
OVERHEAD_SERVE_GIB=12
OVERHEAD_EVAL_GIB=11
SAFETY_MARGIN_GIB=6

# Direct I/O: a mapped file would fill the page cache on top of the unified memory the weights need.
MODEL_ARGS="--rpc $WORKER_LINK_IP:$RPC_PORT -ngl 99 -sm layer --load-mode dio --fit off"

check() {
  local status=0
  echo "== head $HEAD_SSH"
  $SSH "$HEAD_SSH" 'bash -s' -- head "$WORKER_LINK_IP" "$HEAD_LINK_IP" "$CUDA_HOME" < scripts/node_check.sh || status=1
  echo "== worker $WORKER_SSH"
  $SSH "$WORKER_SSH" 'bash -s' -- worker "$HEAD_LINK_IP" "$WORKER_LINK_IP" "$CUDA_HOME" < scripts/node_check.sh || status=1
  echo "== internet from head"
  local proxy_env=""
  [ -n "${PROXY:-}" ] && proxy_env="http_proxy=$PROXY https_proxy=$PROXY"
  if $SSH "$HEAD_SSH" "$proxy_env curl -fsS -m 20 -o /dev/null https://huggingface.co/api/models/$HF_REPO"; then
    echo "  ok    huggingface.co reachable"
  else
    echo "  FAIL  huggingface.co unreachable, check PROXY in cluster.env"
    status=1
  fi
  return "$status"
}

sync_repo() {
  local h
  for h in "$HEAD_SSH" "$WORKER_SSH"; do
    $SSH "$h" "mkdir -p $REMOTE_ROOT/repo $REMOTE_ROOT/logs $REMOTE_ROOT/models"
    rsync -az --delete --exclude .git --exclude __pycache__ ./ "$h:$REMOTE_ROOT/repo/"
  done
}

setup() {
  local h
  sync_repo
  for h in "$HEAD_SSH" "$WORKER_SSH"; do
    echo "== build on $h"
    $SSH "$h" "bash $REMOTE_ROOT/repo/scripts/build_llama.sh" 2>&1 | tail -2 | cut -c1-120
  done
}

# Long jobs run in tmux on the head node so that they survive a disconnection.
background() {
  local session=$1 log=$REMOTE_ROOT/logs/$1.log command=$2
  if $SSH "$HEAD_SSH" "$TMUX_CMD has-session -t $session 2>/dev/null"; then
    echo "$session is already running, log: $log" >&2
    return 1
  fi
  $SSH "$HEAD_SSH" "$TMUX_CMD new-session -d -s $session '$command > $log 2>&1; echo EXIT=\$? >> $log'"
  echo "$session started on the head node, follow it with: cluster.sh status"
}

fetch() {
  sync_repo
  background mc-fetch "bash $REMOTE_ROOT/repo/scripts/fetch_all.sh"
}

trim() {
  local mask=${1:-$FINAL_MASK}
  [ -f "masks/$mask.json" ] || { echo "no mask masks/$mask.json" >&2; return 1; }
  sync_repo
  background mc-trim "bash $REMOTE_ROOT/repo/scripts/trim_variants.sh $SOURCE_QUANT IQ1_S $mask"
}

# Match the exact binary name: tmux keeps the command line of its first session in its own, so
# matching on the full command line can kill the whole tmux server and every session it holds.
rpc_stop() { $SSH "$WORKER_SSH" "$TMUX_CMD kill-session -t mc-rpc 2>/dev/null; pkill -x ggml-rpc-server; true"; }

rpc_start() {
  local env_prefix="" arg
  for arg in "$@"; do
    case "$arg" in
      --tcp) env_prefix="$env_prefix GGML_RPC_NO_RDMA=1" ;;
      --debug) env_prefix="$env_prefix GGML_RPC_DEBUG=1" ;;
      *) echo "unknown option $arg" >&2; return 2 ;;
    esac
  done
  rpc_stop
  sleep 1
  # The worker listens on the direct link only: the RPC protocol has no authentication.
  $SSH "$WORKER_SSH" "$TMUX_CMD new-session -d -s mc-rpc '$OOM_FIRST $env_prefix $BIN/ggml-rpc-server --host $WORKER_LINK_IP --port $RPC_PORT --device CUDA0 > $REMOTE_ROOT/logs/rpc.log 2>&1'"
  local _
  for _ in $(seq 1 30); do
    if $SSH "$WORKER_SSH" "ss -ltn | grep -q $WORKER_LINK_IP:$RPC_PORT"; then
      echo "rpc worker listening on $WORKER_LINK_IP:$RPC_PORT"
      return 0
    fi
    sleep 1
  done
  echo "rpc worker did not start" >&2
  $SSH "$WORKER_SSH" "tail -20 $REMOTE_ROOT/logs/rpc.log" >&2
  return 1
}

preflight() {
  local dir=$1 overhead=${2:-$OVERHEAD_SERVE_GIB} total_bytes need avail h
  total_bytes=$($SSH "$HEAD_SSH" "stat -c %s $dir/*.gguf 2>/dev/null | awk '{s+=\$1} END {printf \"%.0f\", s}'")
  [ "${total_bytes:-0}" -gt 0 ] || { echo "REFUSED: no model in $dir" >&2; return 1; }
  need=$(awk -v b="$total_bytes" -v o="$overhead" 'BEGIN {printf "%.1f", b / 2 / 1073741824 + o}')
  for h in "$HEAD_SSH" "$WORKER_SSH"; do
    avail=$($SSH "$h" "awk '/^MemAvailable:/ {printf \"%.1f\", \$2 / 1048576}' /proc/meminfo")
    echo "preflight $h: needs $need GiB + $SAFETY_MARGIN_GIB margin, available $avail GiB"
    if ! awk -v n="$need" -v m="$SAFETY_MARGIN_GIB" -v a="$avail" 'BEGIN {exit !(n + m <= a)}'; then
      echo "REFUSED: $dir does not fit safely on $h" >&2
      return 1
    fi
  done
}

first_shard() {
  $SSH "$HEAD_SSH" "ls $1/*.gguf | sort | head -1"
}

serve_stop() { $SSH "$HEAD_SSH" "$TMUX_CMD kill-session -t mc-server 2>/dev/null; pkill -x llama-server; true"; }

serve() {
  local dir=$1 model; shift
  serve_stop
  preflight "$dir"
  model=$(first_shard "$dir")
  rpc_start
  $SSH "$HEAD_SSH" "$TMUX_CMD new-session -d -s mc-server '$OOM_FIRST $BIN/llama-server -m $model $MODEL_ARGS --host 127.0.0.1 --port $SERVER_PORT --jinja $* > $REMOTE_ROOT/logs/server.log 2>&1'"
  echo "server starting, log: $REMOTE_ROOT/logs/server.log"
}

wait_ready() {
  local start=$SECONDS health
  while [ $((SECONDS - start)) -lt 600 ]; do
    health=$($SSH "$HEAD_SSH" "curl -s -m 3 --noproxy '*' http://127.0.0.1:$SERVER_PORT/health" 2>/dev/null || true)
    case "$health" in *'"ok"'*) echo "ready after $((SECONDS - start)) s on 127.0.0.1:$SERVER_PORT of the head node"; return 0 ;; esac
    if ! $SSH "$HEAD_SSH" "pgrep -x llama-server >/dev/null"; then
      echo "server died, last log lines:" >&2
      $SSH "$HEAD_SSH" "tail -6 $REMOTE_ROOT/logs/server.log" >&2
      return 1
    fi
    sleep 10
  done
  echo "server not ready after 10 minutes" >&2
  return 1
}

up() {
  serve "$FINAL_MODEL" -c "$FINAL_CTX" -np 1 --no-warmup -md "$FINAL_DRAFT" --spec-type draft-dspark --spec-draft-n-max 3 -ngld 99 -devd CUDA0
  wait_ready
}

on_head_python() {
  $SSH "$HEAD_SSH" "env -u http_proxy -u https_proxy python3 $REMOTE_ROOT/repo/scripts/$*"
}

# Record the lowest available memory seen on each node until killed.
sample_memory() {
  local low_h=999999 low_w=999999 a
  trap 'echo "lowest MemAvailable: head $((low_h / 1024)) GiB, worker $((low_w / 1024)) GiB"; exit 0' TERM
  while :; do
    a=$($SSH "$HEAD_SSH" "awk '/^MemAvailable:/ {print int(\$2 / 1024)}' /proc/meminfo"); [ "$a" -lt "$low_h" ] && low_h=$a
    a=$($SSH "$WORKER_SSH" "awk '/^MemAvailable:/ {print int(\$2 / 1024)}' /proc/meminfo"); [ "$a" -lt "$low_w" ] && low_w=$a
    sleep 5
  done
}

evaluate() {
  local dir=$1 name=$2 model sampler status=0; shift 2
  model=$(first_shard "$dir")
  preflight "$dir" "$OVERHEAD_EVAL_GIB"
  rpc_start >/dev/null
  sample_memory &
  sampler=$!
  $SSH "$HEAD_SSH" "bash $REMOTE_ROOT/repo/scripts/run_eval.sh $model $name $*" || status=$?
  kill -TERM "$sampler"
  wait "$sampler" || true
  rpc_stop
  return "$status"
}

status() {
  local h job
  for h in "$HEAD_SSH" "$WORKER_SSH"; do
    echo "== $h"
    $SSH "$h" "free -h | sed -n 2p; pgrep -a '^(llama-server|llama-perplexi|ggml-rpc-server)' | cut -c1-120 || true"
  done
  for job in mc-fetch mc-trim; do
    $SSH "$HEAD_SSH" "if [ -f $REMOTE_ROOT/logs/$job.log ]; then echo '== $job'; tail -2 $REMOTE_ROOT/logs/$job.log | cut -c1-110; fi"
  done
}

command=${1:-}
[ $# -gt 0 ] && shift
case "$command" in
  check) check ;;
  setup) setup ;;
  fetch) fetch ;;
  trim) trim "$@" ;;
  up) up ;;
  down) serve_stop; rpc_stop ;;
  status) status ;;
  bench) on_head_python bench_speed.py "${1:-300}" ;;
  smoke) on_head_python smoke_test.py ;;
  preflight) preflight "$@" ;;
  serve) serve "$@"; wait_ready ;;
  serve-stop) serve_stop ;;
  eval) evaluate "$@" ;;
  rpc-start) rpc_start "$@" ;;
  rpc-stop) rpc_stop ;;
  *) sed -n 2,15p "$0"; exit 2 ;;
esac

#!/bin/bash
# Read-only health check of one node. Usage: node_check.sh head|worker PEER_LINK_IP LOCAL_LINK_IP CUDA_HOME
role=$1; peer=$2; local_ip=$3; cuda=$4
fail=0
ok() { echo "  ok    $*"; }
ko() { echo "  FAIL  $*"; fail=1; }

"$cuda/bin/nvcc" --version 2>/dev/null | grep -q "release 1[3-9]\." && ok "CUDA 13 or newer" || ko "no CUDA 13 in $cuda"
nvidia-smi -L >/dev/null 2>&1 && ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" || ko "nvidia-smi failed"
[ -f /usr/include/infiniband/verbs.h ] && ok "libibverbs headers" || ko "libibverbs-dev missing, RDMA would be off"
for t in cmake ninja git python3 tmux; do command -v "$t" >/dev/null && ok "$t" || ko "$t missing"; done
[ "$(ulimit -l)" = unlimited ] && ok "memlock unlimited" || echo "  warn  memlock is $(ulimit -l) kB, RDMA uses little but check it"

avail=$(awk '/^MemAvailable:/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)
ok "memory available: $avail GiB"
ip -o -4 addr show | grep -q " $local_ip/" && ok "local link address $local_ip" || ko "$local_ip is not on this node"
# Only the head opens the RPC connection: the worker answers from the address it was reached on.
if [ "$role" = head ]; then
  route=$(ip route get "$peer" 2>/dev/null)
  src=$(sed -nE 's/.* src ([0-9.]+).*/\1/p' <<< "$route")
  dev=$(sed -nE 's/.* dev ([^ ]+).*/\1/p' <<< "$route")
  [ "$src" = "$local_ip" ] && ok "route to $peer leaves from $local_ip via $dev" || ko "route to $peer leaves from $src via $dev, not $local_ip: RDMA would stay off"
fi
ping -c 2 -W 2 -q "$peer" >/dev/null && ok "peer $peer answers" || ko "peer $peer does not answer"
if [ "$role" = head ]; then
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 450 ] && ok "disk: $free GiB free" || ko "disk: $free GiB free, 450 needed for source and trimmed model"
fi
exit "$fail"

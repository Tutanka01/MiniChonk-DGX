#!/bin/bash
# Wait for a download, trim it, then evaluate, all on the head node so nothing depends on the
# machine that launched it. The RPC worker must already be running.
# Usage: chain_on_head.sh FETCH_LOG SOURCE_QUANT OUTPUT_PREFIX MASK RESULT_NAME
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
source cluster.env
fetch_log=$1; quant=$2; prefix=$3; mask=$4; name=$5
model_dir=$REMOTE_ROOT/models/$prefix-$mask
overhead_eval_gib=15
margin_gib=6

until grep -q 'EXIT=' "$fetch_log"; do sleep 30; done
grep -q 'EXIT=0' "$fetch_log" || { echo "FETCH_FAILED"; exit 1; }
echo "fetch ok $(date +%H:%M)"

scripts/trim_variants.sh "$quant" "$prefix" "$mask" | grep -E 'TOTAL_GB|took'
echo "trim ok $(date +%H:%M)"

total=$(stat -c %s "$model_dir"/*.gguf | awk '{s += $1} END {printf "%.0f", s}')
need=$(awk -v b="$total" -v o="$overhead_eval_gib" 'BEGIN {printf "%.1f", b / 2 / 1073741824 + o}')
avail=$(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
echo "preflight head: needs $need GiB + $margin_gib margin, available $avail GiB"
awk -v n="$need" -v m="$margin_gib" -v a="$avail" 'BEGIN {exit !(n + m <= a)}' || { echo "REFUSED"; exit 1; }

low=999999
( while :; do
    a=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
    if [ "$a" -lt "$low" ]; then low=$a; echo "$low" > "$REMOTE_ROOT/logs/$name.head_low_mib"; fi
    sleep 2
  done ) &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

model=$(ls "$model_dir"/*.gguf | sort | head -1)
scripts/run_eval.sh "$model" "$name"
echo "eval ok $(date +%H:%M), lowest head MemAvailable $(cat "$REMOTE_ROOT/logs/$name.head_low_mib") MiB"

#!/bin/bash
# Measure one model: perplexity on the fixed texts, then multiple choice sets. Runs on the head node.
# Usage: run_eval.sh MODEL_FILE RESULT_NAME [llama-perplexity args]
# Stops at the first failure: retrying a load that failed would hammer the nodes again.
set -euo pipefail
source "$(dirname "$0")/../cluster.env"
model=$1; name=$2; shift 2
bin=$REMOTE_ROOT/llama.cpp/build/bin
out=$REMOTE_ROOT/eval/results/$name
args=(-m "$model" --rpc "$WORKER_LINK_IP:$RPC_PORT" -ngl 99 -sm layer --load-mode dio --fit off "$@")
echo 1000 > /proc/self/oom_score_adj
mkdir -p "$out"
cat "$REMOTE_ROOT"/eval/text/*.txt > "$out/all.txt"

"$bin/llama-perplexity" "${args[@]}" -c 2048 -b 1024 --ppl-output-type 1 -f "$out/all.txt" \
  > "$out/perplexity.txt" 2> "$out/perplexity.log"
grep -o 'Final estimate.*' "$out/perplexity.log"

for set in mmlu-validation arc-challenge-validation; do
  "$bin/llama-perplexity" "${args[@]}" -c 2048 -b 1024 -np 5 --multiple-choice --multiple-choice-tasks 200 --seed 42 \
    -bf "$REMOTE_ROOT/eval/$set.bin" > "$out/$set.txt" 2> "$out/$set.log"
  echo "$set: $(grep -o 'Final result.*' "$out/$set.log")"
done

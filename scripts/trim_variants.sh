#!/bin/bash
# Trim the source quant with each mask given by name. Runs on the head node.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster.env
# Usage: trim_variants.sh SOURCE_QUANT OUTPUT_PREFIX MASK...
quant=$1; prefix=$2; shift 2
for v in "$@"; do
  start=$SECONDS
  python3 -m minichonk.trim --src "$REMOTE_ROOT/models/$quant" --dst "$REMOTE_ROOT/models/$prefix-$v" --mask "masks/$v.json" --verify
  echo "$v took $((SECONDS - start)) s"
done

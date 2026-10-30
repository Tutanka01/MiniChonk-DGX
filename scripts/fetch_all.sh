#!/bin/bash
# Download the source GGUF and the DSpark draft on the head node. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster.env
if [ -n "${PROXY:-}" ]; then export http_proxy=$PROXY https_proxy=$PROXY; fi
python3 scripts/fetch_model.py --repo "$HF_REPO" --subdir "$SOURCE_QUANT" --dest "$REMOTE_ROOT/models/$SOURCE_QUANT"
python3 scripts/fetch_model.py --repo "$HF_REPO" --subdir . --match 'dspark-.*Q8_0\.gguf$' --dest "$(dirname "$FINAL_DRAFT")" --rename "$(basename "$FINAL_DRAFT")"

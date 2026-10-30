#!/bin/bash
# Build a pinned llama.cpp with CUDA, RPC and RDMA. Runs on a node and only touches REMOTE_ROOT.
# Does nothing when the pinned commit is already built, so it is safe while a server is running.
set -euo pipefail
source "$(dirname "$0")/../cluster.env"
if [ -n "${PROXY:-}" ]; then export http_proxy=$PROXY https_proxy=$PROXY; fi
SRC=$REMOTE_ROOT/llama.cpp
STAMP=$SRC/build/.minichonk-commit
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$LLAMA_COMMIT" ] && [ -x "$SRC/build/bin/ggml-rpc-server" ]; then
  echo "llama.cpp $LLAMA_COMMIT already built"
  exit 0
fi
# Distribution packages may put an older nvcc first on PATH, which cannot target the GB10.
export PATH=$CUDA_HOME/bin:$PATH
export CUDAToolkit_ROOT=$CUDA_HOME
nvcc --version | grep -q "release 1[3-9]\." || { echo "CUDA 13 or newer not found in $CUDA_HOME"; exit 1; }
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none "$LLAMA_REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet "$LLAMA_COMMIT"
rm -rf "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_RPC_RDMA=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build "$SRC/build" -j "$(nproc)"
"$SRC/build/bin/llama-server" --version
echo "$LLAMA_COMMIT" > "$STAMP"

#!/bin/bash
# Build the fixed evaluation set used to compare trimmed models. Runs on the head node.
set -euo pipefail
source "$(dirname "$0")/../cluster.env"
export http_proxy=$PROXY https_proxy=$PROXY
EVAL=$REMOTE_ROOT/eval
mkdir -p "$EVAL/text"
DS=https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main
for f in mmlu-validation.bin arc-challenge-validation.bin truthful-qa-validation.bin; do
  [ -s "$EVAL/$f" ] || curl -fsSL -o "$EVAL/$f" "$DS/$f"
done

# Keep 48 kB from each source, past the licence headers and front matter.
slice() { python3 -c 'import sys; d = open(sys.argv[1], "rb").read(); open(sys.argv[2], "wb").write(d[100000:148000])' "$1" "$2"; }
fetch_text() {
  local tmp
  tmp=$(mktemp)
  curl -fsSL -o "$tmp" "$1"
  slice "$tmp" "$EVAL/text/$2.txt"
  rm -f "$tmp"
}
fetch_text https://www.gutenberg.org/cache/epub/2701/pg2701.txt en_literature
fetch_text https://www.gutenberg.org/cache/epub/17489/pg17489.txt fr_literature
fetch_text https://www.rfc-editor.org/rfc/rfc9110.txt en_technical
cat /usr/lib/python3.12/argparse.py /usr/lib/python3.12/tarfile.py /usr/lib/python3.12/typing.py > "$EVAL/code.tmp"
slice "$EVAL/code.tmp" "$EVAL/text/code_python.txt"
rm -f "$EVAL/code.tmp"
wc -c "$EVAL"/text/* "$EVAL"/*.bin

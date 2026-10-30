#!/usr/bin/env python3
"""Extract per-expert routing counts from a published imatrix and score prune masks against them."""
import argparse
import json
import pathlib
import re
import struct
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from minichonk import gguf  # noqa: E402


def get_range(url, start, size):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF/resolve/main/imatrix-mistral.gguf")
    p.add_argument("--out", default="masks/expert_usage.json")
    p.add_argument("--masks", nargs="*", default=["masks/letrim50.json", "masks/letrim25.json"])
    args = p.parse_args()

    header = gguf.parse_header(lambda off, size: get_range(args.url, off, min(size, 1 << 20)))
    gguf.check_layout(header)
    counts = {}
    for t in header.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_gate_exps\.weight\.counts$", t.name)
        if m:
            data = get_range(args.url, header.data_start + t.offset, t.nbytes())
            counts[int(m.group(1))] = list(struct.unpack(f"<{len(data) // 4}f", data))
    print(f"layers={len(counts)} chunks={header.kv('imatrix.chunk_count').value} chunk_size={header.kv('imatrix.chunk_size').value}")
    with open(args.out, "w") as f:
        json.dump({"source": args.url, "counts": {str(k): v for k, v in sorted(counts.items())}}, f)

    layers = sorted(counts)
    first = counts[layers[0]]
    print(f"layer {layers[0]}: total={sum(first):.0f} min={min(first):.0f} max={max(first):.0f}")
    for keep in (96, 64, 48, 32, 24, 16):
        cover = [sum(sorted(counts[l], reverse=True)[:keep]) / sum(counts[l]) for l in layers]
        print(f"top-{keep} by usage: routing kept mean={sum(cover) / len(cover):.4f} worst-layer={min(cover):.4f}")
    for path in args.masks:
        with open(path) as f:
            mask = json.load(f)["layers"]
        cover = [sum(counts[l][k] for k in mask[str(l)]["kept"]) / sum(counts[l]) for l in layers]
        print(f"{path}: routing kept mean={sum(cover) / len(cover):.4f} worst-layer={min(cover):.4f}")
    never = sum(1 for l in layers for c in counts[l] if c == 0)
    print(f"experts never routed: {never} of {len(layers) * 128}")
    s = [sorted(counts[l], reverse=True) for l in layers]
    ratios = [x[0] / max(x[63], 1) for x in s]
    print(f"busiest vs 64th expert, median ratio: {sorted(ratios)[len(ratios) // 2]:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prove that expert K means the same thing in the prune mask's checkpoint and in the GGUF.

Mistral's FP8 checkpoint names experts explicitly (layers.N.experts.K.w1.weight). We read
one row per expert over HTTP ranges and compare sign bits between sources: the same weights
agree far above 50%, unrelated weights agree at about 50%.
"""
import argparse
import json
import pathlib
import re
import struct
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from minichonk import gguf  # noqa: E402

HF = "https://huggingface.co"
N_EXPERT = 128
ROW = 6144
RANGE_CAP = 1024 * 1024


def get_range(url, start, size):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


class Safetensors:
    def __init__(self, repo):
        self.repo = repo
        self.index = get_json(f"{HF}/{repo}/resolve/main/consolidated.safetensors.index.json")["weight_map"]
        self.headers = {}

    def locate(self, name):
        fname = self.index[name]
        url = f"{HF}/{self.repo}/resolve/main/{fname}"
        if fname not in self.headers:
            n = struct.unpack("<Q", get_range(url, 0, 8))[0]
            self.headers[fname] = (8 + n, json.loads(get_range(url, 8, n)))
        base, header = self.headers[fname]
        meta = header[name]
        return url, base + meta["data_offsets"][0], meta


def signs_fp8(data):
    return int.from_bytes(bytes(b >> 7 for b in data), "little")


def signs_bf16(data):
    return int.from_bytes(bytes(b >> 7 for b in data[1::2]), "little")


def signs_nvfp4(data, high_first):
    out = bytearray()
    for b in data:
        lo, hi = (b >> 3) & 1, b >> 7
        out += bytes((hi, lo) if high_first else (lo, hi))
    return int.from_bytes(out[:ROW], "little")


def agreement(a, b):
    return 1 - (a ^ b).bit_count() / ROW


def report(title, left, right):
    diagonal = [agreement(left[k], right[k]) for k in range(N_EXPERT)]
    off = [agreement(left[i], right[j]) for i in range(N_EXPERT) for j in range(N_EXPERT) if i != j]
    best = [max(range(N_EXPERT), key=lambda j: agreement(left[i], right[j])) for i in range(N_EXPERT)]
    identity = sum(best[i] == i for i in range(N_EXPERT))
    print(f"{title}: diagonal min={min(diagonal):.3f} mean={sum(diagonal) / N_EXPERT:.3f} | "
          f"off-diagonal max={max(off):.3f} mean={sum(off) / len(off):.3f} | "
          f"best match is K itself for {identity}/{N_EXPERT} experts", flush=True)
    return identity == N_EXPERT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp8", default="mistralai/Mistral-Large-4-1T-A52B")
    p.add_argument("--nvfp4", default="mistralai/Mistral-Large-4-1T-A52B-NVFP4")
    p.add_argument("--gguf", default="unsloth/Mistral-Large-4-1T-A52B-GGUF")
    p.add_argument("--gguf-dir", default="BF16")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 36, 71])
    args = p.parse_args()

    entries = get_json(f"{HF}/api/models/{args.gguf}/tree/main/{args.gguf_dir}?recursive=true")
    shards = sorted(e["path"] for e in entries if e["path"].endswith(".gguf"))
    tensors = {}
    for s in shards:
        url = f"{HF}/{args.gguf}/resolve/main/{s}"
        header = gguf.parse_header(lambda off, size: get_range(url, off, min(size, RANGE_CAP)))
        gguf.check_layout(header)
        for t in header.tensors:
            if re.match(r"blk\.\d+\.ffn_gate_exps\.weight$", t.name):
                tensors[t.name] = (url, header.data_start + t.offset, t)
    print(f"GGUF: parsed {len(shards)} shards, layout checks passed, {len(tensors)} expert gate tensors", flush=True)

    fp8, nvfp4 = Safetensors(args.fp8), Safetensors(args.nvfp4)
    ok = True
    for layer in args.layers:
        url, base, t = tensors[f"blk.{layer}.ffn_gate_exps.weight"]
        if t.dims != [ROW, ROW, N_EXPERT] or t.ggml_type != 30:
            sys.exit(f"unexpected GGUF tensor {t}")
        gguf_signs = [signs_bf16(get_range(url, base + k * t.slab_bytes(), ROW * 2)) for k in range(N_EXPERT)]
        fp8_signs, nv_raw = [], []
        for k in range(N_EXPERT):
            u, off, meta = fp8.locate(f"layers.{layer}.experts.{k}.w1.weight")
            fp8_signs.append(signs_fp8(get_range(u, off, ROW)))
            u, off, meta = nvfp4.locate(f"layers.{layer}.experts.{k}.w1.weight")
            nv_raw.append(get_range(u, off, ROW // 2))
        ok &= report(f"layer {layer} FP8 vs GGUF", fp8_signs, gguf_signs)
        results = {}
        for high_first in (False, True):
            nv_signs = [signs_nvfp4(d, high_first) for d in nv_raw]
            results[high_first] = sum(agreement(nv_signs[k], fp8_signs[k]) for k in range(N_EXPERT)) / N_EXPERT
        high_first = results[True] > results[False]
        nv_signs = [signs_nvfp4(d, high_first) for d in nv_raw]
        ok &= report(f"layer {layer} NVFP4 vs GGUF", nv_signs, gguf_signs)
    print("ORDER_OK" if ok else "ORDER_MISMATCH", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Remove routed experts from a sharded GGUF using a per-layer prune mask.

Experts are slabs along the last tensor dim, stored contiguously, so kept slabs are
copied byte for byte and quantized data is never re-encoded. Only tensors named
below are touched: others such as attn_v_b can also have a 128 dim by coincidence.
"""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

from . import gguf

EXPERT_TENSOR = re.compile(r"^blk\.(\d+)\.ffn_(gate|down|up)_exps\.weight$")
ROUTER_TENSOR = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
COPY_CHUNK = 64 * 1024 * 1024


def load_mask(path, n_expert):
    with open(path) as f:
        mask = json.load(f)
    kept_by_layer = {}
    for layer, entry in mask["layers"].items():
        kept = entry["kept"]
        if kept != sorted(set(kept)):
            raise gguf.GGUFError(f"layer {layer}: kept experts must be sorted and unique")
        if not kept or kept[0] < 0 or kept[-1] >= n_expert:
            raise gguf.GGUFError(f"layer {layer}: kept experts out of range")
        kept_by_layer[int(layer)] = kept
    if len({len(k) for k in kept_by_layer.values()}) != 1:
        raise gguf.GGUFError("every layer must keep the same number of experts")
    return kept_by_layer


def check_source_complete(shards):
    """Refuse a partial download: a split GGUF must hold every shard it announces."""
    first = gguf.parse_file(shards[0])
    count = first.kv("split.count")
    expected = count.value if count else 1
    if len(shards) != expected:
        raise gguf.GGUFError(f"source has {len(shards)} shards, its metadata announces {expected}")


def planned_bytes(shards, kept_by_layer, n_expert):
    total = 0
    for shard in shards:
        out, _ = plan_shard(gguf.parse_file(shard), kept_by_layer, n_expert)
        total += len(gguf.encode_header(out))
        if out.tensors:
            last = out.tensors[-1]
            total += last.offset + last.nbytes()
    return total


def _slice_target(t, kept_by_layer, n_expert):
    """Return the kept slab indices if this tensor must be sliced, else None."""
    m = EXPERT_TENSOR.match(t.name) or ROUTER_TENSOR.match(t.name)
    if not m:
        return None
    layer = int(m.group(1))
    if layer not in kept_by_layer:
        raise gguf.GGUFError(f"{t.name}: no mask entry for layer {layer}")
    wanted_rank = 3 if EXPERT_TENSOR.match(t.name) else 2
    if len(t.dims) != wanted_rank or t.dims[-1] != n_expert:
        raise gguf.GGUFError(f"{t.name}: unexpected shape {t.dims} for {n_expert} experts")
    return kept_by_layer[layer]


def plan_shard(header, kept_by_layer, n_expert):
    """Return the output header and, per tensor, the kept slabs (None means full copy)."""
    out = copy.deepcopy(header)
    n_kept = len(next(iter(kept_by_layer.values())))
    kept_per_tensor = []
    offset = 0
    for t in out.tensors:
        kept = _slice_target(t, kept_by_layer, n_expert)
        kept_per_tensor.append(kept)
        if kept is not None:
            t.dims[-1] = n_kept
        t.offset = offset
        offset = gguf.align_up(offset + t.nbytes(), out.alignment)
    for kv in out.kvs:
        if kv.key.endswith(".expert_count"):
            if kv.value != n_expert:
                raise gguf.GGUFError(f"{kv.key} is {kv.value}, expected {n_expert}")
            kv.value, kv.raw = n_kept, gguf.encode_scalar(kv.kv_type, n_kept)
    return out, kept_per_tensor


def _copy_range(src, dst, offset, size, digest):
    src.seek(offset)
    remaining = size
    while remaining:
        chunk = src.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise gguf.GGUFError("source file is truncated")
        dst.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def trim_shard(src_path, dst_path, kept_by_layer, n_expert):
    src_header = gguf.parse_file(src_path)
    gguf.check_layout(src_header, os.path.getsize(src_path))
    out_header, kept_per_tensor = plan_shard(src_header, kept_by_layer, n_expert)
    part = pathlib.Path(str(dst_path) + ".part")
    digest = hashlib.sha256()
    with open(src_path, "rb") as src, open(part, "wb") as dst:
        encoded = gguf.encode_header(out_header)
        dst.write(encoded)
        digest.update(encoded)
        written = 0
        for src_t, out_t, kept in zip(src_header.tensors, out_header.tensors, kept_per_tensor):
            pad = out_t.offset - written
            if pad:
                dst.write(b"\0" * pad)
                digest.update(b"\0" * pad)
            base = src_header.data_start + src_t.offset
            if kept is None:
                _copy_range(src, dst, base, src_t.nbytes(), digest)
            else:
                slab = src_t.slab_bytes()
                for index in kept:
                    _copy_range(src, dst, base + index * slab, slab, digest)
            written = out_t.offset + out_t.nbytes()
        dst.flush()
        os.fsync(dst.fileno())
    gguf.check_layout(gguf.parse_file(part), os.path.getsize(part))
    part.rename(dst_path)
    return {"name": pathlib.Path(dst_path).name, "bytes": os.path.getsize(dst_path), "sha256": digest.hexdigest()}


def verify_shard(src_path, dst_path, kept_by_layer, n_expert):
    """Re-read the output and compare every sliced slab and copied tensor with the source."""
    src_header = gguf.parse_file(src_path)
    dst_header = gguf.parse_file(dst_path)
    expected, kept_per_tensor = plan_shard(src_header, kept_by_layer, n_expert)
    if [(t.name, t.dims, t.ggml_type, t.offset) for t in expected.tensors] != \
            [(t.name, t.dims, t.ggml_type, t.offset) for t in dst_header.tensors]:
        raise gguf.GGUFError(f"{dst_path}: tensor table differs from plan")
    if [(k.key, k.kv_type, k.raw) for k in expected.kvs] != [(k.key, k.kv_type, k.raw) for k in dst_header.kvs]:
        raise gguf.GGUFError(f"{dst_path}: metadata differs from plan")
    with open(src_path, "rb") as src, open(dst_path, "rb") as dst:
        for src_t, dst_t, kept in zip(src_header.tensors, dst_header.tensors, kept_per_tensor):
            src_base = src_header.data_start + src_t.offset
            dst_base = dst_header.data_start + dst_t.offset
            ranges = [(src_base, dst_base, src_t.nbytes())] if kept is None else \
                [(src_base + k * src_t.slab_bytes(), dst_base + i * src_t.slab_bytes(), src_t.slab_bytes())
                 for i, k in enumerate(kept)]
            for src_off, dst_off, size in ranges:
                src.seek(src_off)
                dst.seek(dst_off)
                done = 0
                while done < size:
                    n = min(COPY_CHUNK, size - done)
                    if src.read(n) != dst.read(n):
                        raise gguf.GGUFError(f"{dst_t.name}: bytes differ from source near {src_off + done}")
                    done += n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=pathlib.Path, help="directory holding the source shards")
    p.add_argument("--dst", required=True, type=pathlib.Path)
    p.add_argument("--mask", required=True)
    p.add_argument("--n-expert", type=int, default=128)
    p.add_argument("--verify", action="store_true", help="byte-compare the output with the source")
    args = p.parse_args(argv)

    shards = sorted(args.src.glob("*.gguf"))
    if not shards:
        sys.exit(f"no GGUF in {args.src}")
    kept_by_layer = load_mask(args.mask, args.n_expert)
    check_source_complete(shards)
    args.dst.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"shards": {}}
    with open(args.mask, "rb") as f:
        mask_sha256 = hashlib.sha256(f.read()).hexdigest()
    # Skipping finished shards is only valid for the mask that produced them.
    if manifest.get("mask_sha256", mask_sha256) != mask_sha256:
        raise gguf.GGUFError(f"{args.dst} was built with another mask, use a new directory")
    manifest["mask_sha256"] = mask_sha256
    manifest["n_kept"] = len(next(iter(kept_by_layer.values())))

    def is_done(shard):
        dst = args.dst / shard.name
        entry = manifest["shards"].get(shard.name)
        return dst.exists() and entry is not None and dst.stat().st_size == entry["bytes"]

    remaining = planned_bytes([s for s in shards if not is_done(s)], kept_by_layer, args.n_expert)
    free = shutil.disk_usage(args.dst).free
    if remaining > free:
        raise gguf.GGUFError(f"needs {remaining / 1e9:.1f} GB, only {free / 1e9:.1f} GB free")
    for shard in shards:
        dst = args.dst / shard.name
        if is_done(shard):
            print(f"skip {shard.name}", flush=True)
        else:
            manifest["shards"][shard.name] = trim_shard(shard, dst, kept_by_layer, args.n_expert)
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"trimmed {shard.name} {manifest['shards'][shard.name]['bytes'] / 1e9:.2f} GB", flush=True)
        if args.verify:
            verify_shard(shard, dst, kept_by_layer, args.n_expert)
            print(f"verified {shard.name}", flush=True)
    print(f"TOTAL_GB {sum(s['bytes'] for s in manifest['shards'].values()) / 1e9:.2f}", flush=True)


if __name__ == "__main__":
    main()

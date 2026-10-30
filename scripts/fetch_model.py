#!/usr/bin/env python3
"""Download GGUF shards from Hugging Face with resume and sha256 verification.

A shard is only renamed to its final name once its hash matches the LFS oid, so
an existing final file is always complete. Reruns skip verified shards.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import urllib.request

HF = "https://huggingface.co"
CHUNK = 16 * 1024 * 1024


def list_shards(repo, subdir, match):
    path = "" if subdir == "." else f"/{subdir}"
    with urllib.request.urlopen(f"{HF}/api/models/{repo}/tree/main{path}", timeout=60) as r:
        entries = json.load(r)
    shards = [e for e in entries if e["type"] == "file" and e["path"].endswith(".gguf") and re.search(match, e["path"])]
    if not shards:
        sys.exit(f"no GGUF matching {match} under {repo}/{subdir}")
    return sorted(shards, key=lambda e: e["path"])


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo, entry, dest_dir, attempts, rename):
    name = rename or pathlib.Path(entry["path"]).name
    final = dest_dir / name
    if final.exists():
        print(f"skip {name}", flush=True)
        return
    part = dest_dir / (name + ".part")
    url = f"{HF}/{repo}/resolve/main/{entry['path']}"
    for attempt in range(1, attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        if done < entry["size"]:
            print(f"get {name} attempt={attempt} from={done}", flush=True)
            subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
                            "--speed-limit", "1048576", "--speed-time", "120",
                            "-C", "-", "-o", str(part), url], check=False)
        if part.exists() and part.stat().st_size == entry["size"]:
            digest = sha256_of(part)
            if digest == entry["lfs"]["oid"]:
                part.rename(final)
                print(f"ok {name}", flush=True)
                return
            print(f"BAD HASH {name}: {digest}, restarting", flush=True)
            part.unlink()
    raise RuntimeError(f"{name}: failed after {attempts} attempts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--subdir", required=True)
    p.add_argument("--dest", required=True, type=pathlib.Path)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--attempts", type=int, default=20)
    p.add_argument("--match", default=".", help="regex on the file path")
    p.add_argument("--rename", help="local name, for a single file")
    args = p.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    shards = list_shards(args.repo, args.subdir, args.match)
    if args.rename and len(shards) != 1:
        sys.exit("--rename needs exactly one matching file")
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(fetch, args.repo, e, args.dest, args.attempts, args.rename) for e in shards]
        for f in futures:
            f.result()
    print(f"ALL_OK {len(shards)} shards", flush=True)


if __name__ == "__main__":
    main()

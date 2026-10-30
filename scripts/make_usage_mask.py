#!/usr/bin/env python3
"""Build a prune mask that keeps the K most routed experts of each layer."""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usage", default="masks/expert_usage.json")
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.usage) as f:
        usage = json.load(f)
    layers = {}
    for layer, counts in usage["counts"].items():
        ranked = sorted(range(len(counts)), key=lambda e: (-counts[e], e))
        layers[layer] = {"kept": sorted(ranked[:args.keep]), "pruned": sorted(ranked[args.keep:])}
    meta = {"method": "top routed experts per layer", "source": usage["source"], "experts_kept_per_layer": args.keep}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "layers": layers}, f)


if __name__ == "__main__":
    main()

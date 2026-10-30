#!/usr/bin/env python3
"""Build a prune mask from AIMER scores: the highest scores are pruned first in each layer."""
import argparse
import json


def build(scores, keep):
    layers = {}
    for layer, values in scores.items():
        ranked = sorted(range(len(values)), key=lambda e: (values[e], e))
        layers[layer] = {"kept": sorted(ranked[:keep]), "pruned": sorted(ranked[keep:])}
    return layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", default="masks/aimer_scores.json")
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.scores) as f:
        data = json.load(f)
    meta = {"method": "AIMER, highest scores pruned", "scores_meta": data["meta"], "experts_kept_per_layer": args.keep}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "layers": build(data["scores"], args.keep)}, f)


if __name__ == "__main__":
    main()

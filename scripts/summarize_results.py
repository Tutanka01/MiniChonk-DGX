#!/usr/bin/env python3
"""Rebuild the results table from the raw files in results/."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results"


def last_pair(path):
    """Multiple choice files hold 'task_count running_score' rows."""
    rows = [line.split() for line in path.read_text().splitlines()]
    rows = [r for r in rows if len(r) == 2 and r[0].isdigit()]
    return int(rows[-1][0]), float(rows[-1][1])


def final_perplexity(path):
    rows = [line.split() for line in path.read_text().splitlines()]
    rows = [r for r in rows if len(r) == 4 and r[0].isdigit()]
    return float(rows[-1][1])


def collect():
    runs = json.loads((ROOT / "runs.json").read_text())
    out = {}
    for name, run in runs.items():
        out[name] = dict(run, perplexity=final_perplexity(ROOT / name / "perplexity.txt"),
                         mmlu=last_pair(ROOT / name / "mmlu-validation.txt"),
                         arc=last_pair(ROOT / name / "arc-challenge-validation.txt"))
    return out


def main():
    print("| Variant | Quant | Experts | Selection | Weights | Perplexity | MMLU | ARC challenge |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in collect().items():
        arc = f"{r['arc'][1]:.1f} %" + ("" if r["arc"][0] == 200 else f" ({r['arc'][0]} questions)")
        print(f"| `{name}` | `{r['quant']}` | {r['experts']} | {r['selection']} | {r['weights_gb']:.0f} GB "
              f"| {r['perplexity']:.2f} | {r['mmlu'][1]:.1f} % | {arc} |")


if __name__ == "__main__":
    main()

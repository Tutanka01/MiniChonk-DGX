#!/usr/bin/env python3
"""Split a cumulative --ppl-output-type 1 result into per-domain perplexity."""
import json
import math
import pathlib
import sys
import urllib.request

results = pathlib.Path(sys.argv[1])
texts = sorted(pathlib.Path(sys.argv[2]).glob("*.txt"))
boundaries, total = [], 0
for t in texts:
    req = urllib.request.Request("http://127.0.0.1:8081/tokenize", json.dumps({"content": t.read_text(errors="replace")}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total += len(json.load(r)["tokens"])
    boundaries.append((t.stem, total))
for variant in sys.argv[3:]:
    rows = [l.split() for l in (results / variant / "perplexity.txt").read_text().splitlines()]
    rows = [(int(r[0]), float(r[1])) for r in rows if len(r) == 4 and r[0].isdigit()]
    # The ppl column is a running geometric mean over chunks of equal scored length.
    chunk_nll = [(i + 1) * math.log(ppl) - (i * math.log(rows[i - 1][1]) if i else 0.0) for i, (_, ppl) in enumerate(rows)]
    out, start = [], 0
    for name, end in boundaries:
        values = [chunk_nll[i] for i, (pos, _) in enumerate(rows) if start <= pos < end]
        out.append(f"{name}={math.exp(sum(values) / len(values)):.2f}" if values else f"{name}=n/a")
        start = end
    overall = math.exp(sum(chunk_nll) / len(chunk_nll))
    print(variant, " ".join(out), f"overall={overall:.2f} final={rows[-1][1]:.2f}")

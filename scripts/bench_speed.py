#!/usr/bin/env python3
"""Measure decode speed on the running server with mixed prompts, one request at a time."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
PROMPTS = [
    "Explique en détail le fonctionnement d'un moteur à quatre temps, étape par étape.",
    "Write a Python class implementing an LRU cache with type hints and unit tests.",
    "Summarize the main causes of the French Revolution as a structured list.",
    "Démontre que la racine carrée de 2 est irrationnelle, puis donne une application concrète.",
]


def ask(prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0.2, "chat_template_kwargs": {"reasoning_effort": "low"}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    gen, secs = 0, 0.0
    for p in PROMPTS:
        out = ask(p, max_tokens)
        t = out["timings"]
        gen += t["predicted_n"]
        secs += t["predicted_ms"] / 1000
        draft = t.get("draft_n") or 0
        accepted = t.get("draft_n_accepted") or 0
        ratio = f" draft {accepted}/{draft} accepted" if draft else ""
        print(f"{t['predicted_n']} tok @ {t['predicted_per_second']:.2f} t/s, prompt {t['prompt_per_second']:.1f} t/s{ratio}", flush=True)
    print(f"OVERALL {gen / secs:.2f} tok/s over {gen} tokens")


if __name__ == "__main__":
    main()

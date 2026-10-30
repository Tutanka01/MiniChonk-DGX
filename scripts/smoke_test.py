#!/usr/bin/env python3
"""Ask questions with checkable answers on the running server, and execute the generated code."""
import json
import re
import subprocess
import sys
import tempfile
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
QUESTIONS = [
    ("Si j'ai 3 boîtes de 12 œufs et que j'en casse 7, combien m'en reste-t-il ?", r"\b29\b"),
    ("Combien font 17 × 23 ?", r"\b391\b"),
    ("Un train part à 14h10 et roule 2h55. À quelle heure arrive-t-il ?", r"17\s*[h:]\s*05"),
    ("Quelle est la capitale de l'Australie ?", r"Canberra"),
    ("Qui a écrit « Les Misérables » ?", r"Hugo"),
    ("Quel est le symbole chimique de l'or ?", r"\bAu\b"),
    ("En quelle année a eu lieu la prise de la Bastille ?", r"1789"),
    ("What is the boiling point of water at sea level in Celsius?", r"\b100\b"),
    ("Who developed the theory of general relativity?", r"Einstein"),
    ("What is the largest planet of the solar system?", r"Jupiter"),
    ("Traduis en anglais : « Le chat dort sur la chaise. »", r"(?i)the cat is sleeping on the chair|the cat sleeps on the chair"),
    ("Quelle est la dérivée de x³ par rapport à x ?", r"3\s*\*?\s*x\s*(\^|²|\*\*)\s*2?|3x²"),
]
CODE_PROMPT = ("Write a Python function `is_prime(n)` and a function `fib(n)` returning the n-th Fibonacci number "
               "with fib(0)=0 and fib(1)=1. Reply with one Python code block only.")
CODE_CHECK = "assert [p for p in range(20) if is_prime(p)] == [2, 3, 5, 7, 11, 13, 17, 19]\nassert fib(10) == 55\nprint('CODE_OK')\n"


def ask(prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0.2, "chat_template_kwargs": {"reasoning_effort": "low"}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["choices"][0]["message"].get("content") or "", out["timings"]


def main():
    passed, gen, secs = 0, 0, 0.0
    for question, expected in QUESTIONS:
        answer, t = ask(question, 400)
        gen += t["predicted_n"]
        secs += t["predicted_ms"] / 1000
        ok = re.search(expected, answer) is not None
        passed += ok
        print(f"{'OK  ' if ok else 'FAIL'} {question} -> {' '.join(answer.split())[:110]}", flush=True)
    answer, t = ask(CODE_PROMPT, 700)
    gen += t["predicted_n"]
    secs += t["predicted_ms"] / 1000
    block = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
    code_ok = False
    if block:
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(block.group(1) + "\n" + CODE_CHECK)
            f.flush()
            run = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=20)
        code_ok = "CODE_OK" in run.stdout
        if not code_ok:
            print(run.stderr[-300:])
    print(f"{'OK  ' if code_ok else 'FAIL'} generated code executed and asserted")
    print(f"SCORE {passed + code_ok}/{len(QUESTIONS) + 1}, {gen / secs:.2f} tok/s over {gen} tokens")
    sys.exit(0 if passed + code_ok == len(QUESTIONS) + 1 else 1)


if __name__ == "__main__":
    main()

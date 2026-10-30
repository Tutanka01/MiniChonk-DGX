# MiniChonk-DGX

Runs [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) (1T parameters, 52B active per
token) on **two NVIDIA GB10 machines** (DGX Spark class, 128 GB each) linked by 200 GbE, at about **10 tokens/s**.

The smallest published GGUF is 233 GB and does not fit. This project removes routed experts from it, byte for byte
and without re-quantizing, down to 206 GB, then splits it across both machines with llama.cpp RPC over RDMA and
speeds up decoding with DSpark speculation.

> **This is a degraded model.** It keeps 112 of 128 experts per layer at about 1.7 bits per routed weight. No score
> exists for the untrimmed model on this benchmark, so the quality loss is not quantified. Validate it on your own
> tasks before relying on it. See [Limits](#limits).

Full explanation, in French: [docs/EXPLICATION.fr.md](docs/EXPLICATION.fr.md).

## Results

| Variant | Quant | Experts | Selection | Weights | Perplexity | MMLU | ARC challenge |
|---|---|---|---|---|---|---|---|
| `usage64` | `UD-IQ1_S` | 64 | most routed | 126 GB | 11.62 | 38.0 % | 43.7 % (142 questions) |
| `letrim50` | `UD-IQ1_S` | 64 | AIMER | 126 GB | 11.12 | 45.5 % | 54.9 % (142 questions) |
| `q2_letrim50` | `UD-Q2_K_XL` | 64 | AIMER | 200 GB | 8.63 | 50.0 % | 56.5 % |
| `letrim25` | `UD-IQ1_S` | 96 | AIMER | 179 GB | 6.45 | 48.0 % | 60.0 % |
| **`aimer112`** | **`UD-IQ1_S`** | **112** | **AIMER** | **206 GB** | **5.31** | **47.0 %** | **63.0 %** |

Lower perplexity is better. Multiple choice margins are about +/- 3.5 points, so MMLU differences among the last
three rows are within noise. Raw files are in [results/](results), and `scripts/summarize_results.py` rebuilds this
table from them.

Final configuration, one request at a time:

| Measure | Result |
|---|---|
| Decode with DSpark | 9.2 to 11.5 tok/s depending on the prompt |
| Decode without DSpark, at 96 experts | 6.33 tok/s |
| Long prompt | 135 tok/s on 6686 tokens |
| Load time | about 3 minutes |
| Memory | 104 GiB on the head, 112 GiB on the worker, context 32768 |
| `scripts/smoke_test.py` | 13/13 easy questions, generated code executed |

## Requirements

- Two machines with an NVIDIA GB10 and about 119 GiB of visible memory each. Anything else is untested.
- A direct link between them. RoCE is used when available, otherwise RPC falls back to TCP.
- CUDA 13 or newer, `cmake`, `ninja`, `git`, `tmux`, `rsync`, `python3`, `libibverbs-dev` on both nodes.
- 450 GB free on the head node. The worker stores no weights.
- A controlling machine with SSH key access to both nodes and `bash`.

## From zero

```bash
cp cluster.env.example cluster.env   # then set hosts, link addresses, proxy, paths
scripts/cluster.sh check             # read-only, fix anything marked FAIL
scripts/cluster.sh setup             # copies this repository to both nodes, builds llama.cpp
scripts/cluster.sh fetch             # 235 GB, about 35 minutes at 110 MB/s, resumable, sha256 verified
scripts/cluster.sh status            # wait for EXIT=0
scripts/cluster.sh trim              # about 10 minutes, verified byte for byte against the source
scripts/cluster.sh status            # wait for EXIT=0
scripts/cluster.sh up                # about 3 minutes, refuses to start if the model does not fit
scripts/cluster.sh smoke             # expects SCORE 13/13
```

Long jobs run on the head node in a private tmux socket (`tmux -L minichonk ls`) and survive a disconnection. Every
step is safe to rerun.

## Using it

The server listens on `127.0.0.1:8081` of the head node, with an OpenAI-compatible API:

```bash
ssh -N -L 8081:127.0.0.1:8081 user@head-node
curl http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Bonjour"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

Without `reasoning_effort`, the chat template asks for `high` and the model can spend many tokens thinking first.
Stop everything with `scripts/cluster.sh down`.

## Limits

- **Quality is only compared between variants.** There is no reference for the full model on this benchmark.
- **One request at a time** in this configuration. Stability over many hours has not been tested.
- **Memory is tight.** Only 7 GiB stay available on the worker. Anything else loaded on these nodes can trigger the
  OOM killer, in which case the inference processes are killed first.
- **The RPC protocol has no authentication.** The worker listens on the direct link only: never expose it.
- **Two nodes add capacity, not speed.** Layer split runs the nodes in turn.

## Repository

| Path | Role |
|---|---|
| `minichonk/gguf.py` | GGUF reader and header writer |
| `minichonk/trim.py` | Trimming tool, with verification against the source |
| `masks/` | Prune masks, AIMER scores, routing counts |
| `results/` | Raw benchmark outputs and run parameters |
| `scripts/cluster.sh` | Single entry point for the cluster |
| `scripts/run_eval.sh`, `fetch_eval.sh` | Quality benchmark |
| `scripts/verify_expert_order.py` | Proves that expert indices match across checkpoints |
| `tests/` | `python3 -m unittest discover -s tests -t .` |

## Credits and licences

See [NOTICE.md](NOTICE.md). Mistral Large 4 and its derivatives are under the Mistral Ecosystem Open-Weight License,
which requires a commercial license from Mistral to operate a Model-as-a-Service business.

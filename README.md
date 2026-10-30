# MiniChonk-DGX

**Run a 1-trillion-parameter model on two desktop machines.**

[![License: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Model: Mistral Large 4](https://img.shields.io/badge/model-Mistral%20Large%204-orange)](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
[![Runtime: llama.cpp](https://img.shields.io/badge/runtime-llama.cpp-green)](https://github.com/ggml-org/llama.cpp)

[Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) has 1T parameters, and its smallest
published GGUF weighs 233 GB. That is too much for two NVIDIA GB10 machines (DGX Spark class, 128 GB each).
MiniChonk-DGX makes it fit: it cuts routed experts out of the GGUF byte for byte, splits the result across both
machines, and serves it behind an OpenAI-compatible API, with one script driving the whole cluster.

| | |
|---|---|
| **Decode speed** | about 10 tok/s (9.2 to 11.5 depending on the prompt) |
| **Model size** | 233 GB → 206 GB, no re-quantization, verified byte for byte against the source |
| **Experts kept** | 112 of 128 per layer, chosen with AIMER |
| **From zero to serving** | 9 commands, about 50 minutes of model download, trimming and loading |
| **Smoke test** | 13/13 easy questions, generated code executed |

> **This is a degraded model.** It keeps 112 of 128 experts per layer at about 1.7 bits per routed weight. No score
> exists for the untrimmed model on this benchmark, so the quality loss is not quantified. Validate it on your own
> tasks before relying on it. See [Limits](#limits).

Want the full story, including what failed? Read the [detailed explanation, in French](docs/EXPLICATION.fr.md).

## How it works

1. **Trim.** Each expert is a contiguous slab of bytes inside its tensor, so experts can be removed without decoding
   or re-quantizing anything. The router is sliced with the same mask, and every output is compared with the source.
2. **Split.** llama.cpp RPC places about half of the layers on each machine, over RDMA when available. The worker
   stores no weights on disk.
3. **Speculate.** A 2.2 GB DSpark draft model proposes 3 tokens at a time and the large model validates them in one
   pass. At 96 experts, this took decoding from 6.33 to 10.46 tok/s (+65 %), without changing what the large model
   would have produced.

## What the benchmarks showed

- **The most used experts are not the ones to keep.** At 64 experts, keeping the most routed ones preserves 73.5 % of
  routing against 56.4 % for AIMER, yet scores far lower: MMLU 38.0 % against 45.5 %, ARC 43.7 % against 54.9 %.
- **At equal memory, experts beat bits on most measures.** 112 experts at 1.7 bits (206 GB) beats 64 experts at 2.7
  bits (200 GB) on perplexity (5.31 against 8.63) and ARC (63.0 % against 56.5 %). MMLU leans the other way (47.0 %
  against 50.0 %), within the margin of error.
- **Trimming frees memory, not time.** Routed experts are 92 % of the weights but only about a quarter of what each
  token reads.
- **A second node adds capacity, not speed.** On a 19 GB test model, one node decodes at 11.06 tok/s and two at
  10.43 tok/s. Memory bandwidth is the limit, not the network.

## Results

| Variant | Quant | Experts | Selection | Weights | Perplexity | MMLU | ARC challenge |
|---|---|---|---|---|---|---|---|
| `usage64` | `UD-IQ1_S` | 64 | most routed | 126 GB | 11.62 | 38.0 % | 43.7 % (142 questions) |
| `letrim50` | `UD-IQ1_S` | 64 | AIMER | 126 GB | 11.12 | 45.5 % | 54.9 % (142 questions) |
| `q2_letrim50` | `UD-Q2_K_XL` | 64 | AIMER | 200 GB | 8.63 | 50.0 % | 56.5 % |
| `letrim25` | `UD-IQ1_S` | 96 | AIMER | 179 GB | 6.45 | 48.0 % | 60.0 % |
| **`aimer112`** | **`UD-IQ1_S`** | **112** | **AIMER** | **206 GB** | **5.31** | **47.0 %** | **63.0 %** |

Lower perplexity is better. Multiple choice margins are about +/- 3.5 points, so MMLU differences among the last
three rows are within noise. Multiple choice parameters also changed between runs to save memory (see
[`results/runs.json`](results/runs.json)). These scores compare variants with each other, not with public leaderboards.
Raw files are in [results/](results), and `scripts/summarize_results.py` rebuilds this table from them.

Final configuration, one request at a time:

| Measure | Result |
|---|---|
| Decode with DSpark | 9.2 to 11.5 tok/s depending on the prompt |
| Decode without DSpark, at 96 experts | 6.33 tok/s |
| Long prompt | 135 tok/s on 6686 tokens |
| Load time | about 3 minutes |
| Memory | 104 GiB on the head, 112 GiB on the worker, context 32768 |
| `scripts/smoke_test.py` | 13/13 easy questions, generated code executed |

## Quick start

### Requirements

- Two machines with an NVIDIA GB10 and about 119 GiB of visible memory each. This was tested on two ThinkStation PGX,
  which use the same GB10 as a DGX Spark. Anything else is untested.
- A direct link between them. RoCE is used when available, otherwise RPC falls back to TCP.
- CUDA 13 or newer, `cmake`, `ninja`, `git`, `tmux`, `rsync`, `python3`, `libibverbs-dev` on both nodes.
- 450 GB free on the head node. The worker stores no weights.
- A controlling machine with SSH key access to both nodes and `bash`.

### From zero

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

### Send a request

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

### Pick another size

Any mask in [`masks/`](masks) can be trimmed and served. Generate your own with
`python3 scripts/make_aimer_mask.py --keep N --out masks/NAME.json`, then:

```bash
scripts/cluster.sh trim NAME         # builds the variant next to the others
# then set FINAL_MASK=NAME in cluster.env
scripts/cluster.sh up                # still refuses a model that does not fit
```

## Built to fail safely

An early load of the full model saturated the worker's memory and got unrelated processes killed. The tooling is
shaped by that incident and by the ways a 200 GB file can silently go wrong:

- **Memory preflight.** `up` refuses any model that would leave less than 6 GiB available, based on what each node has
  free at that moment.
- **Inference dies first.** Inference processes run with `oom_score_adj=1000`, so the kernel should pick them over
  anything else. This has not been tested on a real saturation.
- **Refuses bad inputs.** `trim` rejects a partial download, a mask that differs from the one a resumed run used, and
  a disk that is too small.
- **Targets tensors by name, never by shape.** Some unrelated tensors also have a dimension of 128.
- **Proves expert order.** The mask was computed on another checkpoint. `scripts/verify_expert_order.py` compares
  weight sign bits across files and finds the right expert for 128 of 128 on the layers tested, downloading only a few
  MB.
- **Tested.** 33 unit tests cover slicing, corruption, resuming, masks, published scores and process shutdown:
  `python3 -m unittest discover -s tests -t .`

## Limits

- **Quality is only compared between variants.** There is no reference for the full model on this benchmark, and an
  MMLU of 47 % remains low.
- **No evaluation on real tasks.** The 13/13 smoke test covers easy questions only.
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
| `tests/` | Unit tests |

## Credits and licences

This project stands on the work of [Mistral AI](https://huggingface.co/mistralai) (the model),
[Unsloth](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF) (the source GGUF and DSpark draft),
[LeTrim](https://huggingface.co/akhaliq/Mistral-Large-4-1T-A52B-LeTrim-NVFP4) (published AIMER scores),
[AIMER](https://arxiv.org/abs/2603.18492) (the pruning criterion) and
[llama.cpp](https://github.com/ggml-org/llama.cpp) (the runtime). Details are in [NOTICE.md](NOTICE.md).

The code of this repository is under the [MIT license](LICENSE). Mistral Large 4 and its derivatives are under the
Mistral Ecosystem Open-Weight License, which requires a commercial license from Mistral to operate a Model-as-a-Service
business.

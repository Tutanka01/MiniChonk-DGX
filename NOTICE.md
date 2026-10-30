# Third-party material

No model weights are stored in this repository.

| Material | Source | Terms |
|---|---|---|
| Mistral Large 4 | [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | Mistral Ecosystem Open-Weight License, copied in `masks/LICENSE-Mistral` |
| Source GGUF and DSpark draft, downloaded at setup | [unsloth/Mistral-Large-4-1T-A52B-GGUF](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF) | Same license |
| `masks/aimer_scores.json`, `masks/letrim25.json`, `masks/letrim50.json` | [akhaliq/Mistral-Large-4-1T-A52B-LeTrim-NVFP4](https://huggingface.co/akhaliq/Mistral-Large-4-1T-A52B-LeTrim-NVFP4) | Same license, as stated by that repository |
| `masks/expert_usage.json`, routing counts extracted from the published imatrix | [ggml-org/Mistral-Large-4-1T-A52B-GGUF](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF) | Same license |
| AIMER pruning criterion | [arXiv 2603.18492](https://arxiv.org/abs/2603.18492) | Cited method, not redistributed code |
| llama.cpp, built at setup | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT, not redistributed here |
| Benchmark datasets, downloaded by `scripts/fetch_eval.sh` | [ikawrakow/validation-datasets-for-llama.cpp](https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp), Project Gutenberg, RFC Editor | Not redistributed here |

Files under `results/` hold only scores, no dataset text.

Anything derived from the model, such as trimmed weights, must keep the Mistral notice. Operating a Model-as-a-Service
business on it requires a commercial license from Mistral.

The code of this repository has no license yet: until one is added, default copyright applies.

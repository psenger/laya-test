# laya-test

Standalone local trial of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), a calibrated decision model (ModernBERT-large encoder, plus a multilingual mmBERT checkpoint picked by its `Router`).

Laya is an encoder classifier, not a generative LLM: it returns typed answers (`choice`, `score`, `noul` yes/no probability) in one forward pass and never generates text. oMLX, LM Studio and Ollama only serve generative and embedding models, so it runs here through the `laya` Python package on PyTorch, using the Apple GPU via MPS.

## Run

```bash
uv sync
uv run python run_tests.py
```

The first run downloads the English and multilingual checkpoints (about 1.5 GB) to the Hugging Face cache.

## Results

See [results/RESULTS.md](results/RESULTS.md) for the table and [results/results.json](results/results.json) for the full model output per case.

On an Apple M2 Max (laya 0.3.22, torch 2.14.0) it answered 26 of 28 labelled questions correctly across support triage, guardrails, sentiment and five non-English languages, at 16 to 55 ms per call with warm models. The two misses were both confident:

- French "Je veux résilier mon abonnement immédiatement..." (I want to cancel my subscription immediately): churn risk predicted **no** with 0.938 confidence.
- Japanese "エンタープライズプランの料金を教えてください" (tell me the enterprise plan pricing): department predicted **billing** with 0.999 confidence, expected sales.

Confident errors like these mean a confidence threshold alone won't catch every mistake, so check accuracy on your own data before relying on it.

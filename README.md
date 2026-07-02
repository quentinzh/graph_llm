# Graph LLM Evidence Selector

Token-graph explainable recommendation with:

- leakage-safe per-user personalized token graphs
- neural evidence selector (lightweight GNN + Qwen3-Embedding-0.6B item encoding)
- Qwen3-4B LoRA SFT with graph-guided unlikelihood loss

## Project layout

```text
graph_llm/
  main.py              # entry point
  dataload/            # data loading code
  models/              # model code
  train/               # training and inference
  metrics/             # evaluation metrics
  checkpoints/         # saved weights and embedding caches
  config/              # argument parsing and defaults
  aux/                 # helper scripts and tests
  log/                 # logs and generation outputs
  pretrain_llm/        # pretrained LLM weights
  data/                # input data, graph cache, profiles
```

## Loss

`total_loss = sft_loss + lambda_ul * graph_ul_loss` (default `lambda_ul=0.1`)

- top-M selector evidence tokens are protected
- unselected high-frequency user-graph tokens are suppressed via unlikelihood

## LLM Prompt Layout

For each sample, the model sees:

```text
<User profile>
Current item information:
Title: <item title>
Description: <item description>

Useful token evidence for this explanation: <top-M token evidence>
The explanation of <item title> is "
<generated explanation>
```

## Setup

```bash
# Qwen3-4B causal LM
bash aux/download_qwen3_4b.sh

# Qwen3-Embedding-0.6B (optional; falls back to LM embed_tokens if missing)
bash aux/download_qwen3_embedding_0.6b.sh

# Or install deps and download both models
bash aux/setup_graph_env.sh

# User profiles (reuse gpt1 pipeline)
conda activate fair
python ../gpt1/generate_llama_user_profiles.py \
  --profile-mode structured --folds 1 --scopes train,train_valid \
  --output-dir ../gpt1/user_profiles_structured/Amazon/MoviesAndTV_small_15pct
```

## Train

```bash
bash aux/run.sh \
  --dataset_name Amazon/MoviesAndTV_small_15pct \
  --split_indices 1 \
  --allow_missing_profiles
```

## Smoke tests

```bash
conda run -n fair python aux/tests/test_smoke.py
```

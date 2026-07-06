# Graph LLM Profile Explainer

Profile-conditioned explainable recommendation with token-graph selector, graph UL loss, and Qwen3-4B LoRA SFT.

## Project layout

```text
graph_llm/
  main.py              # entry point
  dataload/            # data loading, graph cache, embeddings
  models/              # model, selector, token graph
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

`total_loss = sft_loss + lambda_ul * graph_ul_loss + lambda_feat * feature_loss`

- GNN selector picks evidence tokens from per-user token graphs; UL loss suppresses unselected high-frequency tokens
- Evidence tokens are **not** inserted as prompt text; they guide training via UL loss and `evidence_bonus` at generation
- `lambda_feat` encourages keyword feature tokens in explanations

## LLM Prompt Layout

```text
<User profile>
Current item information:
Title: <item title>
Description: <item description>

The explanation of <item title> for <user ID> is "
<generated explanation>
```

## Setup

```bash
bash aux/download_qwen3_4b.sh
bash aux/download_qwen3_embedding_0.6b.sh
# or: bash aux/setup_graph_env.sh
```

## Derive profiles for small dataset

If LLM profiles already exist for the full dataset (e.g. `MoviesAndTV_corsa_filtered`),
derive child-dataset caches in seconds without re-running Qwen3-4B:

```bash
conda run -n fair python aux/derive_profiles_from_parent.py \
  --dataset_name Amazon/MoviesAndTV_corsa_filtered_small_15pct/ \
  --source_dataset_name Amazon/MoviesAndTV_corsa_filtered \
  --profile_dir data/profiles \
  --fold 1 \
  --scopes train,train_valid
```

Profile text is inherited from the parent (full interaction history). Training will
prefer the child-specific paths under `data/profiles/Amazon/MoviesAndTV_corsa_filtered_small_15pct/`.

## Train

```bash
bash aux/run.sh \
  --dataset_name Amazon/MoviesAndTV_corsa_filtered_small_15pct \
  --split_indices 1 \
  --allow_missing_profiles
```

## Smoke tests

```bash
conda run -n fair python aux/tests/test_smoke.py
```

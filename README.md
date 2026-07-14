# Graph LLM Profile Explainer

Profile-conditioned explainable recommendation with a tail-aware token-graph selector and Qwen3-4B LoRA SFT.

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

`total_loss = tail_sft_loss + lambda_selector * selector_loss + lambda_feat * feature_loss`

- Tail-weighted SFT increases the contribution of low-frequency content tokens using training-fold statistics only
- GNN selector is directly supervised by graph nodes that overlap with the gold explanation; sampled negatives include hard, frequency-matched, and globally popular nodes
- Each graph keeps 512 stratified nodes by default: 256 tail, 128 item-related, and 128 stable user-preference nodes
- Evidence tokens are **not** inserted as prompt text; the selector receives its own BCE supervision, while hard top-M evidence tokens receive the existing `evidence_bonus` during generation
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
# AutoDL/Linux：默认使用阿里云 PyPI、阿里云 PyTorch CUDA wheel 与 HF 镜像。
bash script/create_env.sh

# 本地已有模型时可直接开始；否则单独通过 HF 镜像下载所需权重。
bash aux/download_qwen3_4b.sh
bash aux/download_qwen3_embedding_0.6b.sh
```

`requirements.txt` 中的 `torch==2.7.1+cu128` 需要 PyTorch 专用 wheel 目录。
该目录由 `script/create_env.sh` 通过 `--find-links` 自动指定为阿里云镜像（不是
`--extra-index-url` 包索引）。若 AutoDL 节点的网络更适合其他镜像，可临时覆盖：

```bash
GRAPH_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
GRAPH_PYTORCH_WHEEL_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128 \
bash script/create_env.sh
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

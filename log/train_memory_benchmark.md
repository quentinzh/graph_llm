# graph_llm 默认训练 smoke

- 生成时间: 2026-09-26T20:58:41.531509Z
- devices: 1
- 配置: 默认 sdpa, batch_size=4, accumulation_steps=8, gradient_checkpointing=False

| 配置 | exit | OOM | train wall (s) | peak GiB |
|------|------|-----|----------------|----------|
| default_microbatch | 0 | False | 1.7939038909971714 | cuda:1=22.54 |

复现: `conda run -n fair python graph_llm/aux/benchmark_train_memory.py --devices 1`

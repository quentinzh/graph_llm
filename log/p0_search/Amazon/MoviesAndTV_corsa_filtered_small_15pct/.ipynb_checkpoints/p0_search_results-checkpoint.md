# graph_llm P0 顺序调参结果

- dataset: `Amazon/MoviesAndTV_corsa_filtered_small_15pct/`
- split_indices: `1`
- updated_at: `2026-07-25T06:15:35`
- selection: test `FMR`（并列时 `rouge_l`）

## 最终选定超参

- `lambda_feat`: `0.01`
- `evidence_bonus`: `0.1`
- `top_m_evidence`: `5`

## Stage 1: lambda_feat

固定 evidence_bonus=0.1, top_m_evidence=5

| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | BLEU-1 | BLEU-4 | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage1_lambda_feat | lfeat1e-3_eb0.1_top5 | 1e-3 | 0.1 | 5 | 11.0507 | 0.7678 | 0.9346 | 0.8494 | 8.7694 | 0.2519 | 0.3692 | 0.1417 | 13.7002 | 1.7684 | 11.1835 |  |
| stage1_lambda_feat | lfeat1e-2_eb0.1_top5 | 1e-2 | 0.1 | 5 | 9.6684 | 0.6639 | 0.9412 | 0.8413 | 8.5889 | 0.1528 | 0.3781 | 0.1368 | 12.9380 | 1.7255 | 10.8755 |  |

## Stage 2: evidence_bonus

固定 lambda_feat=0.01, top_m_evidence=5

| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | BLEU-1 | BLEU-4 | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Stage 3: top_m_evidence

固定 lambda_feat=0.01, evidence_bonus=0.1

| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | BLEU-1 | BLEU-4 | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

# graph_llm P0 顺序调参结果

- dataset: `Amazon/MoviesAndTV_corsa_filtered_small_15pct/`
- split_indices: `1`
- updated_at: `2026-07-27T09:06:08`
- selection: test `FMR`（并列时 `rouge_l`）

## 最终选定超参

- `lambda_feat`: `0.1`
- `evidence_bonus`: `1.0`
- `top_m_evidence`: `5`

## Stage 1: lambda_feat

固定 evidence_bonus=0.1, top_m_evidence=5

| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | BLEU-1 | BLEU-4 | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage1_lambda_feat | lfeat1e-3_eb0.1_top5 | 1e-3 | 0.1 | 5 | 11.0507 | 0.7678 | 0.9346 | 0.8494 | 8.7694 | 0.2519 | 0.3692 | 0.1417 | 13.7002 | 1.7684 | 11.1835 |  |
| stage1_lambda_feat | lfeat1e-2_eb0.1_top5 | 1e-2 | 0.1 | 5 | 9.6684 | 0.6639 | 0.9412 | 0.8413 | 8.5889 | 0.1528 | 0.3781 | 0.1368 | 12.9380 | 1.7255 | 10.8755 |  |
| stage1_lambda_feat | lfeat0.1_eb0.1_top5 | 0.1 | 0.1 | 5 | 9.5517 | 0.5567 | 0.9490 | 0.8363 | 8.0190 | 0.2357 | 0.3728 | 0.1549 | 13.3622 | 1.8348 | 11.0014 | yes |

## Stage 2: evidence_bonus

固定 lambda_feat=0.1, top_m_evidence=5

| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | BLEU-1 | BLEU-4 | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage2_evidence_bonus | lfeat0.1_eb0_top5 | 0.1 | 0 | 5 | 10.2813 | 0.5713 | 0.9432 | 0.8441 | 8.1643 | 0.2443 | 0.3638 | 0.1575 | 13.6856 | 1.8818 | 11.1473 |  |
| stage2_evidence_bonus | lfeat0.1_eb0.1_top5 | 0.1 | 0.1 | 5 | 9.5517 | 0.5567 | 0.9490 | 0.8363 | 8.0190 | 0.2357 | 0.3728 | 0.1549 | 13.3622 | 1.8348 | 11.0014 |  |
| stage2_evidence_bonus | lfeat0.1_eb0.5_top5 | 0.1 | 0.5 | 5 | 10.1257 | 0.6265 | 0.9418 | 0.8390 | 7.9753 | 0.2813 | 0.3584 | 0.1638 | 13.7476 | 1.9360 | 11.2854 |  |
| stage2_evidence_bonus | lfeat0.1_eb1_top5 | 0.1 | 1 | 5 | 10.8355 | 0.7270 | 0.9329 | 0.8436 | 8.2495 | 0.2546 | 0.3763 | 0.1677 | 14.1361 | 1.9305 | 11.4721 | yes |
| stage2_evidence_bonus | lfeat0.1_eb2_top5 | 0.1 | 2 | 5 | 10.2993 | 0.6162 | 0.9386 | 0.8362 | 8.5267 | 0.2823 | 0.3728 | 0.1468 | 13.1800 | 1.6803 | 10.5860 |  |

## Stage 3: top_m_evidence

固定 lambda_feat=0.1, evidence_bonus=1.0

| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | BLEU-1 | BLEU-4 | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage3_top_m_evidence | lfeat0.1_eb1_top5 | 0.1 | 1 | 5 | 10.8355 | 0.7270 | 0.9329 | 0.8436 | 8.2495 | 0.2546 | 0.3763 | 0.1677 | 14.1361 | 1.9305 | 11.4721 | yes |
| stage3_top_m_evidence | lfeat0.1_eb1_top10 | 0.1 | 1 | 10 | 9.3996 | 0.6055 | 0.9529 | 0.8361 | 8.1770 | 0.2280 | 0.3692 | 0.1570 | 13.2621 | 1.7586 | 11.0143 |  |
| stage3_top_m_evidence | lfeat0.1_eb1_top15 | 0.1 | 1 | 15 | 9.2925 | 0.6051 | 0.9514 | 0.8333 | 8.4383 | 0.1866 | 0.3799 | 0.1479 | 13.0327 | 1.7815 | 10.8098 |  |
| stage3_top_m_evidence | lfeat0.1_eb1_top20 | 0.1 | 1 | 20 | 10.0691 | 0.5940 | 0.9445 | 0.8355 | 8.2513 | 0.3044 | 0.3566 | 0.1533 | 13.2150 | 1.6655 | 10.7685 |  |

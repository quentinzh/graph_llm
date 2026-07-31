# graph_llm P1 顺序调参结果

- dataset: `Amazon/MoviesAndTV_corsa_filtered_small_15pct/`
- split_indices: `1`
- updated_at: `2026-07-30T19:27:27`
- selection: test `FMR`（并列时 `rouge_l`）

## 冻结的 P0 最优超参

- `lambda_feat`: `0.1`
- `evidence_bonus`: `1.0`
- `top_m_evidence`: `5`

## 最终选定 P1 超参

- `lambda_selector`: `0.2`
- `selector_feature_positive_weight`: `2.0`
- `lambda_prefix_feature`: `0.1`

## Stage 1: lambda_selector

固定 lambda_feat=0.1, evidence_bonus=1.0, top_m_evidence=5, selector_feature_positive_weight=3.0, lambda_prefix_feature=0.1

| stage | tag | lambda_selector | selector_feature_positive_weight | lambda_prefix_feature | BLEU-1 | BLEU-4 | USR | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage1_lambda_selector | lsel0.1_sfpw3_lpfx0.1 | 0.1 | 3 | 0.1 | 8.6466 | 0.5459 | 0.3857 | 0.9570 | 0.8306 | 8.1377 | 0.2041 | 0.3513 | 0.1505 | 12.8417 | 1.8048 | 10.7446 |  |
| stage1_lambda_selector | lsel0.2_sfpw3_lpfx0.1 | 0.2 | 3 | 0.1 | 10.1137 | 0.6457 | 0.4210 | 0.9450 | 0.8410 | 8.0633 | 0.2818 | 0.3656 | 0.1640 | 13.7830 | 1.9477 | 11.3121 | yes |
| stage1_lambda_selector | lsel0.3_sfpw3_lpfx0.1 | 0.3 | 3 | 0.1 | 10.9037 | 0.6280 | 0.4210 | 0.9374 | 0.8440 | 8.0832 | 0.3406 | 0.3656 | 0.1633 | 13.7726 | 1.7345 | 11.0760 |  |

## Stage 2: selector_feature_positive_weight

固定 lambda_feat=0.1, evidence_bonus=1.0, top_m_evidence=5, lambda_selector=0.2, lambda_prefix_feature=0.1

| stage | tag | lambda_selector | selector_feature_positive_weight | lambda_prefix_feature | BLEU-1 | BLEU-4 | USR | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage2_selector_feature_positive_weight | lsel0.2_sfpw2_lpfx0.1 | 0.2 | 2 | 0.1 | 10.3296 | 0.6830 | 0.4247 | 0.9436 | 0.8402 | 8.1089 | 0.2822 | 0.3674 | 0.1689 | 13.9602 | 1.9486 | 11.3056 | yes |
| stage2_selector_feature_positive_weight | lsel0.2_sfpw3_lpfx0.1 | 0.2 | 3 | 0.1 | 10.1137 | 0.6457 | 0.4210 | 0.9450 | 0.8410 | 8.0633 | 0.2818 | 0.3656 | 0.1640 | 13.7830 | 1.9477 | 11.3121 |  |
| stage2_selector_feature_positive_weight | lsel0.2_sfpw4_lpfx0.1 | 0.2 | 4 | 0.1 | 9.2163 | 0.5529 | 0.3957 | 0.9526 | 0.8341 | 8.0780 | 0.2356 | 0.3620 | 0.1514 | 13.0109 | 1.8128 | 10.8039 |  |

## Stage 3: lambda_prefix_feature

固定 lambda_feat=0.1, evidence_bonus=1.0, top_m_evidence=5, lambda_selector=0.2, selector_feature_positive_weight=2.0

| stage | tag | lambda_selector | selector_feature_positive_weight | lambda_prefix_feature | BLEU-1 | BLEU-4 | USR | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stage3_lambda_prefix_feature | lsel0.2_sfpw2_lpfx0.1 | 0.2 | 2 | 0.1 | 10.3296 | 0.6830 | 0.4247 | 0.9436 | 0.8402 | 8.1089 | 0.2822 | 0.3674 | 0.1689 | 13.9602 | 1.9486 | 11.3056 | yes |
| stage3_lambda_prefix_feature | lsel0.2_sfpw2_lpfx0.2 | 0.2 | 2 | 0.2 | 9.6110 | 0.5831 | 0.4324 | 0.9453 | 0.8368 | 8.3295 | 0.1938 | 0.3763 | 0.1482 | 13.0793 | 1.6950 | 10.7857 |  |
| stage3_lambda_prefix_feature | lsel0.2_sfpw2_lpfx0.3 | 0.2 | 2 | 0.3 | 9.4829 | 0.5443 | 0.4052 | 0.9520 | 0.8344 | 8.0619 | 0.2419 | 0.3548 | 0.1591 | 13.4682 | 1.8496 | 11.0443 |  |

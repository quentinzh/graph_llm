# graph_llm GPT Search 1

- dataset: `Amazon/MoviesAndTV_corsa_filtered_small_15pct/`
- split_indices: `1`
- updated_at: `2026-08-03T18:11:34`
- search data: `validation only`
- selection: maximize `min(BLEU-1, rouge_l)`, then their mean, under validation `FMR >= 0.17`
- test policy: evaluate once after all hyperparameters are fixed

## Stage 1: feature-tail interaction

| stage | tag | lambda_feat | tail_weight_max | lambda_prefix_feature | lora_r | lora_alpha | BLEU-1 | BLEU-4 | USR | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | reused | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| stage1_feature_tail | lfeat0p01_tailmax1p5_lpfx0p05_r16_a32 | 0.0100 | 1.5000 | 0.0500 | 16 | 32 | 9.6010 | 0.6709 | 0.4574 | 0.9414 | 0.8350 | 8.4855 | 0.1520 | 0.3507 | 0.1430 | 13.1568 | 1.7121 | 10.8829 |  |  |
| stage1_feature_tail | lfeat0p01_tailmax2_lpfx0p05_r16_a32 | 0.0100 | 2.0000 | 0.0500 | 16 | 32 | 10.0870 | 0.7220 | 0.4837 | 0.9460 | 0.8418 | 8.6460 | 0.1661 | 0.4011 | 0.1498 | 13.5864 | 1.7761 | 11.1957 |  |  |
| stage1_feature_tail | lfeat0p03_tailmax1p5_lpfx0p05_r16_a32 | 0.0300 | 1.5000 | 0.0500 | 16 | 32 | 9.5627 | 0.6721 | 0.3840 | 0.9411 | 0.8305 | 7.9694 | 0.2017 | 0.3345 | 0.1558 | 13.3761 | 1.8151 | 11.1190 |  |  |
| stage1_feature_tail | lfeat0p03_tailmax2_lpfx0p05_r16_a32 | 0.0300 | 2.0000 | 0.0500 | 16 | 32 | 9.5309 | 0.6779 | 0.4349 | 0.9487 | 0.8357 | 8.3043 | 0.1732 | 0.3543 | 0.1542 | 13.4270 | 1.7693 | 11.1602 |  |  |
| stage1_feature_tail | lfeat0p05_tailmax1p5_lpfx0p05_r16_a32 | 0.0500 | 1.5000 | 0.0500 | 16 | 32 | 10.2284 | 0.6435 | 0.4223 | 0.9383 | 0.8371 | 7.9923 | 0.2294 | 0.3489 | 0.1691 | 13.8484 | 1.8001 | 11.3819 |  | yes |
| stage1_feature_tail | lfeat0p05_tailmax2_lpfx0p05_r16_a32 | 0.0500 | 2.0000 | 0.0500 | 16 | 32 | 9.4877 | 0.6671 | 0.4265 | 0.9472 | 0.8358 | 8.2444 | 0.1829 | 0.3507 | 0.1556 | 13.3471 | 1.8347 | 11.0554 |  |  |

## Stage 2: prefix alignment

| stage | tag | lambda_feat | tail_weight_max | lambda_prefix_feature | lora_r | lora_alpha | BLEU-1 | BLEU-4 | USR | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | reused | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| stage2_prefix_alignment | lfeat0p05_tailmax1p5_lpfx0p05_r16_a32 | 0.0500 | 1.5000 | 0.0500 | 16 | 32 | 10.2284 | 0.6435 | 0.4223 | 0.9383 | 0.8371 | 7.9923 | 0.2294 | 0.3489 | 0.1691 | 13.8484 | 1.8001 | 11.3819 | yes |  |
| stage2_prefix_alignment | lfeat0p05_tailmax1p5_lpfx0_r16_a32 | 0.0500 | 1.5000 | 0.0000 | 16 | 32 | 10.7106 | 0.7186 | 0.4233 | 0.9332 | 0.8363 | 7.9282 | 0.2606 | 0.3345 | 0.1733 | 14.2609 | 2.0037 | 11.6313 |  | yes |
| stage2_prefix_alignment | lfeat0p05_tailmax1p5_lpfx0p1_r16_a32 | 0.0500 | 1.5000 | 0.1000 | 16 | 32 | 10.1630 | 0.6864 | 0.4212 | 0.9323 | 0.8331 | 7.9315 | 0.2221 | 0.3291 | 0.1658 | 13.7952 | 1.8890 | 11.3153 |  |  |

## Stage 3: LoRA capacity

| stage | tag | lambda_feat | tail_weight_max | lambda_prefix_feature | lora_r | lora_alpha | BLEU-1 | BLEU-4 | USR | Distinct-1 | Distinct-2 | ENTR | DIV | FCR | FMR | rouge_1 | rouge_2 | rouge_l | reused | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| stage3_lora_capacity | lfeat0p05_tailmax1p5_lpfx0_r16_a32 | 0.0500 | 1.5000 | 0.0000 | 16 | 32 | 10.7106 | 0.7186 | 0.4233 | 0.9332 | 0.8363 | 7.9282 | 0.2606 | 0.3345 | 0.1733 | 14.2609 | 2.0037 | 11.6313 | yes | yes |
| stage3_lora_capacity | lfeat0p05_tailmax1p5_lpfx0_r32_a64 | 0.0500 | 1.5000 | 0.0000 | 32 | 64 | 10.3323 | 0.7237 | 0.4274 | 0.9384 | 0.8407 | 8.0432 | 0.2175 | 0.3543 | 0.1642 | 13.8312 | 1.7964 | 11.3040 |  |  |

## Final fixed configuration

- tag: `lfeat0p05_tailmax1p5_lpfx0_r16_a32`
- lambda_feat: `0.05`
- tail_weight_max: `1.5`
- lambda_prefix_feature: `0.0`
- lora_r / lora_alpha: `16 / 32`

### Final test metrics

- BLEU-1: `10.4962`
- BLEU-4: `0.7806`
- USR: `0.4292`
- Distinct-1: `0.9333`
- Distinct-2: `0.8394`
- ENTR: `8.0275`
- DIV: `0.2695`
- FCR: `0.3620`
- FMR: `0.1661`
- rouge_1: `13.9216`
- rouge_2: `2.0269`
- rouge_l: `11.3507`

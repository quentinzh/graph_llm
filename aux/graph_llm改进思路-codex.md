# graph_llm 改进思路

## 1. 当前结果诊断

结果文件：`log/Amazon/MoviesAndTV_corsa_filtered_small_15pct/graph_profile.log`

当前 graph_llm 的主要优势是：

- `high` 明显强于 `low`，这个方向符合预期。
- `FCR/FMR` 明显强于 CIER，`FCR` 全组达到 `0.4480`，low 组达到 `0.5820`。
- `USR/Distinct/ENTR` 很强，说明输出不塌缩，句子多样性好。

当前主要短板是：

- 对比 Rober，`B-1/B-4/R-1/R-2/R-L` 仍弱，尤其是 Rober 的 lexical overlap 和 `R-L`。
- low 组虽然 `FCR` 高，但 `FMR` 仍低于 `Rober-low`，说明 feature 覆盖广，但命中参考评论特征的精确性还不够。
- 训练中 `lambda_feat=0.0001`，而日志里的 `Feat` 大约是 `2.6-2.9`，实际贡献只有约 `0.0003`，几乎不影响主损失。
- 当前 evidence 主要通过 `GraphUL` 和 `evidence_bonus` 生效，证据文本没有进入 prompt；这能提高 feature 指标和多样性，但不一定提升 BLEU/ROUGE。
- 旧日志中直接加入 evidence prompt 的尝试出现过严重塌缩，不能简单把长 evidence 文本塞进 prompt。

因此，后续目标不是单纯继续提高多样性，而是提高“生成内容与真实评论的词汇/短语/句法重合”，同时保住现在的 `FCR/FMR/USR/Distinct`。

## 2. 优先级最高：检索-原型-改写

建议实现一个轻量的 prototype retrieval + rewrite 流程：

1. 从 target item 的训练集历史评论里检索 top-k 句子。
2. 从 target user 的历史评论里检索与 target item/aspect 接近的句子。
3. 用当前 graph selector 的 evidence token 对候选句子重排。
4. 只把 1-3 条短 prototype sentence 或其压缩 aspect phrase 放进生成上下文。
5. 让 Qwen 生成时“改写 prototype”，而不是从零生成。

预期收益：

- 直接提升 `B-1/B-4/R-1/R-L`，因为 prototype 来自真实评论分布。
- 降低事实幻觉，避免提到 item 没有的特征。
- 保留当前 graph selector 对 feature 的控制能力。

需要注意：

- 不要塞长 evidence prompt。旧日志显示长 evidence prompt 可能导致塌缩。
- prototype 文本应该限制在 32-64 tokens，并做去重、去高频模板句过滤。
- 可以先不训练新模型，只在 generation 阶段做 rerank 或 prompt augmentation。

文献依据：

- CompExp, WWW 2022：提出 extract-and-refine，用 item 历史评论抽取 prototype，再个性化改写，并用 BLEU 型质量目标抑制泛化句子。https://arxiv.org/abs/2111.00670
- RAG, NeurIPS 2020：检索外部文本能让生成更 factual、specific、diverse。https://arxiv.org/abs/2005.11401
- PRAG, 2022：面向 explainable recommendation，用 personalized retriever + retrieved reviews + keyword-guided QA 提升 factuality 和 informativeness。https://arxiv.org/abs/2209.12613

## 3. 加显式 aspect/content planner

当前模型虽然有 feature loss，但权重太小，实际没有把“该说哪些方面”变成强约束。建议加入一个显式 planner：

- 输入：user profile embedding、target item text、graph evidence tokens。
- 输出：3-5 个 target aspect/opinion keywords。
- 训练目标：预测 ground-truth explanation 中高 TF-IDF 的 feature/opinion words。
- 生成时：把 planner 输出作为短 plan，例如 `aspects: plot, acting, story`。

可以先做最小实现：

- 不新增大模型，只训练一个 MLP 或小 Transformer head。
- 用现有 `feature_position_mask` 和 graph evidence token 做监督。
- 把 `lambda_feat` 从 `1e-4` 扫到 `{1e-3, 5e-3, 1e-2}`。

预期收益：

- 提升 low 组 `FMR`，减少“覆盖了很多 feature 但没覆盖参考 feature”的问题。
- 稳住 `FCR`，避免只为了 BLEU/ROUGE 回退成泛化模板。

文献依据：

- PETER, ACL-IJCNLP 2021：通过 context prediction 把 user/item ID 映射到生成词空间，提升个性化解释生成。https://arxiv.org/abs/2105.11601
- PEPLER, ACM TOIS：连续 prompt + sequential tuning + recommendation regularization，使预训练 LM 更好融合 user/item ID。https://arxiv.org/abs/2202.07371
- HSS, SIGIR Workshop 2019：用 feature-aware attention 和 auto-denoising 强化解释句中的 feature 信息。https://arxiv.org/abs/2101.03392

## 4. 用多候选生成 + 指标感知 rerank

当前使用 greedy generation。建议改成生成多候选，再 rerank：

候选生成：

- `beam_size=4/8`
- 或 `top_p=0.8-0.9, temperature=0.7-0.9` 采样 4-8 条
- 保留当前 repetition control

rerank score：

```text
score =
  alpha * normalized_logprob
+ beta  * prototype_overlap
+ gamma * feature_match
+ delta * graph_evidence_coverage
- eta   * repetition_penalty
- rho   * generic_sentence_penalty
```

其中：

- `prototype_overlap` 可以用 ROUGE-L 或 token F1 近似。
- `feature_match` 用当前 FMR/FCR 的 feature set。
- `generic_sentence_penalty` 惩罚 `good movie`, `great acting`, `very nice` 这类高频泛句。

预期收益：

- 不改变训练即可快速提升 `B-1/R-L`。
- 可以在 dev set 上调权重，避免直接牺牲 `USR/Distinct`。

文献依据：

- Pointer-Generator, ACL 2017：copy 机制和 coverage 能提升 ROUGE 并减少重复。https://arxiv.org/abs/1704.04368
- Minimum Risk Training, ACL 2016：可直接优化 BLEU 等不可微评估指标。https://arxiv.org/abs/1512.02433
- Diverse Beam Search, AAAI 2018：多样化 beam 能改善候选质量。https://arxiv.org/abs/1610.02424
- Unlikelihood Training, ICLR 2020：UL 能减少重复和高频泛化输出。https://arxiv.org/abs/1908.04319

## 5. 强化 graph selector 的监督信号

当前 `GraphUL` 大约 `0.004`，乘 `lambda_ul=0.1` 后贡献约 `0.0004`，也偏弱。建议从“负采样抑制”升级为“正负对比学习”：

正样本：

- ground-truth explanation 中的 feature/opinion tokens。
- 与 ground-truth review embedding 接近的 item-review sentence tokens。

负样本：

- 用户历史高频但 target item 不相关的 token。
- item 全局高频泛化 token。
- stopword 和 generic adjective。

训练目标：

- selected evidence embedding 靠近 target explanation embedding。
- unselected/high-frequency token 远离 target explanation embedding。
- 加 edge dropout 或 node dropout，降低高频节点主导。

预期收益：

- 提升 evidence token 的精确性，从而提升 low 组 `FMR`。
- 降低流行 token 对生成的干扰。

文献依据：

- SGL, SIGIR 2021：图推荐中高阶/高频节点会造成长尾劣化，自监督图学习能改善长尾和噪声鲁棒性。https://arxiv.org/abs/2010.10783
- LightGCN, SIGIR 2020：推荐中的图卷积应简化为核心邻居聚合，避免不必要非线性影响训练。https://arxiv.org/abs/2002.02126
- KGAT, KDD 2019：用注意力聚合高阶关系，可提升准确性、解释性和 side information 利用。https://arxiv.org/abs/1905.07854
- KGCL, SIGIR 2022：知识图谱增强推荐中，对比学习可缓解稀疏和噪声问题。https://arxiv.org/abs/2205.00976

## 6. 加 sentiment/rating alignment

当前模型主要控制 feature，不显式控制情感极性。对 review generation 来说，BLEU/ROUGE 不只依赖 feature，还依赖情感词和句式。

建议：

- 从 rating 或 review sentiment 中构造 polarity label。
- 对生成 token 加 sentiment alignment loss。
- 在 planner 中输出 `positive/negative/neutral` 或 rating bucket。
- 对 high/low 分组分别调 sentiment prior，避免 low 组生成过于正向的泛化句。

预期收益：

- 提升 `R-2/R-L`，因为真实评论中情感词和短语结构更一致。
- 减少 feature 命中但语义方向不一致的情况。

文献依据：

- SAER, WSDM 2021：用 latent sentiment vector 对齐推荐分数和解释文本，提升解释质量。https://arxiv.org/abs/2101.09656
- MTER, SIGIR 2018：联合建模 recommendation 与 opinionated content，在 feature-level 解释上更细。https://arxiv.org/abs/1806.03568

## 7. 推荐实验顺序

第一阶段：不大改模型，快速验证上限。

1. 扫 `lambda_feat`: `{1e-3, 5e-3, 1e-2}`。
2. 扫 `evidence_bonus`: `{0.1, 0.2, 0.3}`。
3. 扫 `top_m_evidence`: `{5, 10, 20}`。
4. 增加 beam/sample 多候选 + rerank。
5. 用 dev set 选择同时提升 `B-1/R-L/FMR` 且不显著降低 `USR/Distinct` 的配置。

第二阶段：加 retrieval prototype。

1. 建 item-review sentence index。
2. 用 Qwen embedding 或现有 embedding backend 检索 top-k。
3. 用 graph evidence token 重排。
4. 只加入短 prototype 或 aspect phrase。
5. 评估 all/low/high 的 BLEU、ROUGE、FMR、FCR、USR。

第三阶段：加 planner 和 selector contrastive loss。

1. planner 预测 top-k aspect/opinion keywords。
2. selector 加 target-review embedding 对比学习。
3. sentiment/rating alignment 作为辅助 loss。
4. 做 ablation：无 planner、无 retrieval、无 contrastive、无 sentiment。

## 8. 最值得优先做的两个改动

如果只做两个改动，建议：

1. **prototype retrieval + rerank**：最直接补 Rober 强项，即 BLEU/ROUGE/R-L。
2. **提高 feature/planner 监督强度**：最直接补 low 组 FMR，同时保住当前 high > low 的结论。

这两个改动的风险最低，因为它们不需要推翻当前 graph_llm 的优势，只是在生成前后增加“更像真实评论”的约束。

## 9. 备注

- CCF/中科院分区会随年份调整，正式写论文或开题时应按投稿年份最新版目录复核。
- 当前建议优先基于 SIGIR、KDD、WWW、ACL、NeurIPS、AAAI、WSDM、ACM TOIS 等高水平来源。
- 如果后续要跑实验，默认使用 `conda` 环境 `fair`，深度学习训练/推理优先放在 `cuda:1`。

## 10. 低成本 ablation 详细方案

这一阶段的目标不是一次性把所有组合跑完，而是用较少训练成本判断三个旋钮的作用方向：

- `lambda_feat`: feature token 在训练损失中的权重。
- `evidence_bonus`: 生成时对 selector 选中 evidence token 的 logit 加成。
- `top_m_evidence`: 每个样本选入多少 evidence token。

当前模型已经有较强的 `USR/Distinct/ENTR` 和 `FCR/FMR`，所以 ablation 的判断重点应放在：

1. `low/high` 关系不能被破坏，最好 high 继续强于 low。
2. `FMR/FCR` 不能明显下降，尤其 low 组 `FMR` 要尽量上升。
3. `B-1/R-1/R-L` 要上升，同时 `USR/Distinct` 不能塌缩。

### 10.1 不建议直接跑满 27 组

完整网格是：

```text
lambda_feat     = {1e-3, 5e-3, 1e-2}
evidence_bonus  = {0.1, 0.2, 0.3}
top_m_evidence  = {5, 10, 20}
```

全组合共 27 次训练，成本较高，而且很多组合会把噪声互相放大。建议先做单因素筛选，再只对少数候选做交互组合。

当前基线可以记为：

```text
lambda_feat=1e-4
evidence_bonus=0.1
top_m_evidence=5
lambda_ul=0.1
```

### 10.2 第一轮：只扫 `lambda_feat`

固定 `evidence_bonus=0.1`、`top_m_evidence=5`，只改变 `lambda_feat`：

```text
A0: lambda_feat=1e-4, evidence_bonus=0.1, top_m_evidence=5
A1: lambda_feat=1e-3, evidence_bonus=0.1, top_m_evidence=5
A2: lambda_feat=5e-3, evidence_bonus=0.1, top_m_evidence=5
A3: lambda_feat=1e-2, evidence_bonus=0.1, top_m_evidence=5
```

这一轮回答的问题是：加强 feature loss 后，`FMR/FCR` 是否上升，BLEU/ROUGE 是否跟着改善。

判断规则：

- 好信号：low 组 `FMR` 上升；all/high `FMR` 不下降；`BLEU-1`、`ROUGE-1`、`ROUGE-L` 上升；`USR` 仍高于 `0.95`。
- 坏信号：`FCR` 上升但 `FMR` 下降，说明模型在乱塞 feature；BLEU/ROUGE 下降，说明 feature loss 干扰了句子自然性；输出出现 feature 堆词。

经验预期：

- `1e-3` 或 `5e-3` 更可能是有效区间。
- `1e-2` 可能过强，容易把生成推向关键词堆叠。

### 10.3 第二轮：固定最佳 `lambda_feat` 后扫 `evidence_bonus`

假设第一轮选出 `lambda_feat=5e-3`，第二轮固定它，只改变 `evidence_bonus`：

```text
B1: lambda_feat=5e-3, evidence_bonus=0.1, top_m_evidence=5
B2: lambda_feat=5e-3, evidence_bonus=0.2, top_m_evidence=5
B3: lambda_feat=5e-3, evidence_bonus=0.3, top_m_evidence=5
```

`evidence_bonus` 主要影响生成阶段的词选择。它不会改变模型训练出的参数，但会改变 greedy decoding 时 evidence token 的被选概率。

判断规则：

- 好信号：`FMR/FCR` 上升；`BLEU-1/ROUGE-1` 上升；句子仍自然。
- 坏信号：输出重复 evidence token；`USR/Distinct` 下降；`BLEU-4/R-L` 不涨，只涨 `FCR`，说明只是插词，不是在生成更像真实评论的短语。

经验预期：

- `0.2` 值得优先试。
- `0.3` 风险偏高，尤其后续和 `top_m_evidence=20` 搭配时容易引入噪声。

### 10.4 第三轮：固定前两项后扫 `top_m_evidence`

假设前两轮选出 `lambda_feat=5e-3`、`evidence_bonus=0.2`，第三轮只改变 `top_m_evidence`：

```text
C1: lambda_feat=5e-3, evidence_bonus=0.2, top_m_evidence=5
C2: lambda_feat=5e-3, evidence_bonus=0.2, top_m_evidence=10
C3: lambda_feat=5e-3, evidence_bonus=0.2, top_m_evidence=20
```

`top_m_evidence` 控制 selector 选多少 evidence token。它同时影响训练期 UL 的正负 token 集合，以及生成期 bonus 的作用范围。

判断规则：

- `top_m_evidence=5`: 精度高、噪声小，但 coverage 可能不足。
- `top_m_evidence=10`: 通常是覆盖和精度的折中。
- `top_m_evidence=20`: 可能提高 `FCR`，但容易引入泛化词和流行词，导致 `FMR/BLEU/ROUGE` 不升反降。

如果 `top_m_evidence=20` 只涨 `FCR` 不涨 `FMR`，不建议保留。

### 10.5 第四轮：只补少量交互组合

前三轮筛完后，最多保留 2-3 个候选配置，再补少量交互实验。例如：

```text
D1: lambda_feat=1e-3, evidence_bonus=0.2, top_m_evidence=10
D2: lambda_feat=5e-3, evidence_bonus=0.2, top_m_evidence=10
D3: lambda_feat=5e-3, evidence_bonus=0.1, top_m_evidence=20
```

不建议优先跑：

```text
lambda_feat=1e-2, evidence_bonus=0.3, top_m_evidence=20
```

这个组合三个旋钮都偏强，容易把生成推向 evidence token 堆叠，损害句子流畅性和 `USR/Distinct`。

### 10.6 beam + rerank 应该放在训练参数筛完之后

beam + rerank 是 generation-only 技术，不应该一开始就加。否则性能提升到底来自训练参数，还是来自解码策略，会变得很难判断。

建议流程：

1. 先用 greedy decoding 做 `lambda_feat/evidence_bonus/top_m_evidence` 筛选。
2. 选出 1-3 个最佳 checkpoint。
3. 对这些 checkpoint 再做 beam/sample 多候选生成。
4. 用 rerank 选择最终输出。

候选生成可以先从以下设置开始：

```text
beam_size = 4 或 8
num_return_sequences = 4 或 8
```

如果使用 sampling：

```text
top_p = 0.8-0.9
temperature = 0.7-0.9
num_return_sequences = 4 或 8
```

rerank 可以先使用如下打分：

```text
score =
  1.0 * normalized_logprob
+ 0.8 * feature_match_score
+ 0.5 * evidence_token_coverage
+ 0.5 * prototype_or_reference_overlap
- 0.7 * repetition_penalty
- 0.5 * generic_sentence_penalty
```

如果暂时还没有 prototype retrieval，可以先移除 `prototype_or_reference_overlap`，只用：

```text
score =
  1.0 * normalized_logprob
+ 0.8 * feature_match_score
+ 0.5 * evidence_token_coverage
- 0.7 * repetition_penalty
- 0.5 * generic_sentence_penalty
```

其中：

- `feature_match_score`: 候选句命中目标 feature set 的比例，可近似对应 `FMR`。
- `evidence_token_coverage`: 候选句覆盖 selector evidence token 的比例。
- `repetition_penalty`: 重复 unigram/bigram 或连续重复 token 的惩罚。
- `generic_sentence_penalty`: 惩罚 `good movie`、`great acting`、`very nice` 等高频泛化短句。

### 10.7 最终配置选择标准

建议先用硬门槛过滤：

```text
必须满足：
- high 组优势趋势不被破坏
- USR >= 0.95
- Distinct-1 不低于当前结果的 95%
- all 组 FMR/FCR 不低于当前结果
```

在满足硬门槛的配置中，再用加权分数排序：

```text
selection_score =
  0.35 * all_R-L
+ 0.25 * all_B-1
+ 0.20 * low_FMR
+ 0.10 * high_FMR
+ 0.10 * all_FCR
```

这样可以避免为了 BLEU/ROUGE 牺牲当前已经较好的 high-vs-low 结论和 feature 指标。

### 10.8 推荐的最小实验清单

如果时间有限，建议先跑 9 组以内：

```text
A1: lambda_feat=1e-3, evidence_bonus=0.1, top_m_evidence=5
A2: lambda_feat=5e-3, evidence_bonus=0.1, top_m_evidence=5
A3: lambda_feat=1e-2, evidence_bonus=0.1, top_m_evidence=5
B1: best_lambda_feat, evidence_bonus=0.2, top_m_evidence=5
B2: best_lambda_feat, evidence_bonus=0.3, top_m_evidence=5
C1: best_lambda_feat, best_evidence_bonus, top_m_evidence=10
C2: best_lambda_feat, best_evidence_bonus, top_m_evidence=20
D1: lambda_feat=1e-3, evidence_bonus=0.2, top_m_evidence=10
D2: lambda_feat=5e-3, evidence_bonus=0.2, top_m_evidence=10
```

跑完后只对最好的 1-2 个 checkpoint 加 beam + rerank。这样成本可控，也能比较清楚地解释每个参数的作用。

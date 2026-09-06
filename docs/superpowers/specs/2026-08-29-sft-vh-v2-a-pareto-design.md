# SFT-VH V2-A Pareto 数据重选设计

日期：2026-08-29（Asia/Shanghai）

状态：已批准设计

范围：在不改变线上 expansion 协议、且第一阶段不引入新 teacher 数据的前提下，构造并评估下一版 QMD `hyde + vec` completion-only SFT release。

## 1. 决策摘要

SFT-VH V2-A 是一次受控的数据重选实验。它复用现有 teacher 候选文本，在干净、冻结的 QMD 环境下重新计算检索指标，只准入在全部五指标（Recall@10、Recall@20、Recall@30、MRR@10、nDCG@10）上不退化且至少一个指标严格提升的 Pareto winner，将语义质量作为 release gate，在相同的 pinned Qwen3-1.7B LoRA 配置上训练。

该实验优化的是通用检索质量，而不是单个 benchmark 或单个 cutoff。目标是在保持 Recall@10 和 Recall@20 的同时，提升五数据集等权宏平均的 Recall@30、MRR@10 和 nDCG@10。

已批准的执行顺序如下：

1. V2-A：复用并重新选择现有候选。
2. V2-B：只有 V2-A 通过晋级门槛后，才增加 FreshStack、新 NFCorpus 候选或其他 teacher。

Webmasters 保留为 stress test。它单独报告，不进入 primary macro，也不能单独否决一个具有通用价值的 expansion 模型。

## 2. 为什么需要 V2-A

当前 `public-main-v2-vh-prompt-v1` adapter 已经证明 SFT 可以改善 QMD 检索：

- Recall@30 提升 1.50 个百分点，paired bootstrap 置信区间显著为正。
- MRR@10 提升 1.48 个百分点，置信区间显著为正。
- nDCG@10 提升 0.64 个百分点，置信区间显著为正。
- Recall@10 和 Recall@20 没有显著提升。

但它的监督数据存在四类可修复问题：

1. 现有 winner rule 优化 `R@20 > raw`，允许 nDCG@10 下降 0.02，且没有保护 Recall@30 和 MRR@10。在 1,017 条 SFT target 中，相对 raw retrieval 有 36 条降低 Recall@30、47 条降低 MRR@10、65 条降低 nDCG@10。
2. 200 条语义抽检中有 20 条 fail、6 条 uncertain。这些结果仅作为诊断，没有阻止对应记录进入 SFT release。
3. 数据来源被 NFCorpus 主导：1,017 条中有 728 条来自 NFCorpus，而 SciFact 只有 2 条。
4. Student 生成结果明显长于训练 target。训练 HyDE 平均约 82 词，held-out SFT 生成的 HyDE 平均约 113 词，最大达到 608 词。

这些证据支持先改善监督质量，再考虑 preference optimization 或 online reinforcement learning。

## 3. 目标与非目标

### 3.1 目标

- 产出 1,000–2,000 条最终训练记录，并额外建立约为训练集 10% 的独立 validation/retrieval-dev split。
- 只准入在全部五指标（Recall@10、Recall@20、Recall@30、MRR@10、nDCG@10）上不退化且至少一个指标严格提升的候选。
- 将实体漂移、约束丢失、否定丢失、无依据事实和歧义过度解释设为阻塞 release 的错误。
- 保留稀缺来源的合规数据，不通过重复采样补足数量。
- 保证数据、prompt、训练、检索和评估 provenance 可复现。
- 分离仅由 prompt 带来的收益与新监督 release 带来的收益。
- 按检索质量而不是 validation loss 选择 checkpoint。

### 3.2 非目标

- V2-A 不改变 runtime `hyde + vec` 输出协议。
- V2-A 不训练 `no expansion` 或 routing action。
- V2-A 不引入 FreshStack 或新生成的 teacher 候选。
- 不将 qualified tie、trade-off 或失败候选作为 SFT 正样本。
- 本实验不实现 DPO、ORPO、GRPO 或 PPO。
- 不宣称多语言质量；V2-A 的训练和 primary evidence 均为英文。
- 不根据 training loss 或协议通过率单独作出生产发布决策。

## 4. 实验标识

现有实验已经使用 `v2-vh` 标识。V2-A 必须使用新的不可变 ID，不能覆盖或含糊地扩展现有 release。

批准的命名模式如下：

```text
source release:  public-distill-v3-vh-pareto
prompt release:  public-distill-v3-vh-pareto-prompt-v1
experiment:      public-main-v3-vh-pareto-prompt-v1
prompt version:  qmd-student-expansion-v3-vh-pareto-v1
```

候选输入 hash、retrieval profile、准入策略、语义决定、split 成员、prompt 字节或物化 JSONL 中的任何一项发生变化，都必须使用新的 release ID 或 experiment ID。

## 5. 候选输入

V2-A 复用现有 V2-VH projection 和 native generation artifact 中的候选文本。旧 winner label 和旧检索指标只作为诊断输入。正式准入必须在新冻结的干净环境中重新评分全部候选。

当前冻结候选池包含 6,223 个 query-level 记录，以 `input_id` 计恰好对应 6,223 个唯一 query；每个 query 包含 3–4 个 source candidate，共计 24,869 个 source candidate。V2-A 必须先对这 24,869 个 source candidate 逐一执行确定性的裁切与规范化，再对 Contract-valid canonical candidate 执行 retrieval rescoring，最后以 6,223 个 query group 为单位执行 Pareto 准入和 winner selection。这里的 6,223 是候选选择阶段的 query 数，不是最终 SFT 训练记录数；最终训练数据仍由通过全部准入流程的 selected candidate 构成，并遵守 1,000–2,000 条训练记录的规模要求。

冻结候选池按来源数据集分布如下：

| 来源数据集 | Query 数 | Query 占比 | Candidate 数 |
| --- | ---: | ---: | ---: |
| `nfcorpus-train-dev` | 2,914 | 46.83% | 11,656 |
| `cqadup-unix` | 884 | 14.21% | 3,524 |
| `cqadup-programmers` | 866 | 13.92% | 3,457 |
| `scifact-train` | 809 | 13.00% | 3,236 |
| `fiqa-train` | 750 | 12.05% | 2,996 |
| **合计** | **6,223** | **100.00%** | **24,869** |

各来源占比独立四舍五入到小数点后两位，分项显示值之和可能与 100.00% 相差 0.01 个百分点。

NFCorpus 占当前 query pool 的 46.83%，是最大的候选来源；其余四个来源各占约 12%–14%。这张表描述的是 V2-A 重新选择之前的输入候选池，不是最终 SFT train/validation 分布。最终来源分布必须在严格 Pareto、语义准入、去重和 split 完成后重新统计；稀缺来源的合规记录全部保留，不为了满足预设比例而丢弃，也不通过重复采样补足数量。

### 5.1 数据集来源与本地位置

五个 source 均来自 BEIR 发布的公开数据归档，原始数据只用于构造 query、corpus 和 qrels；teacher prompt 只接收 query，不接收 gold document 或 qrels。

| `source_id` | 上游数据与使用范围 | 本地标准化数据位置 |
| --- | --- | --- |
| `fiqa-train` | BEIR FiQA，`train` split，金融问答检索 | `finetune/data/public-distill-v0/prepared/fiqa-train/` |
| `cqadup-programmers` | BEIR CQADupStack，`programmers` domain 的 upstream `test` split，编程问答去重检索 | `finetune/data/public-distill-v0/prepared/cqadup-programmers/` |
| `cqadup-unix` | BEIR CQADupStack，`unix` domain 的 upstream `test` split，Unix 问答去重检索 | `finetune/data/public-distill-v0/prepared/cqadup-unix/` |
| `nfcorpus-train-dev` | BEIR NFCorpus，合并 `train` 和 `dev` split，生物医学与健康信息检索 | `finetune/data/public-distill-v0/prepared/qmd-distill-public-v0-nfcorpus-train-dev/` |
| `scifact-train` | BEIR SciFact，`train` split，科学论断与论文证据检索 | `finetune/data/public-distill-v0/prepared/qmd-distill-public-v0-scifact-train/` |

CQADupStack 的 `test` 是上游数据包内的 split 名称。在本实验中，Programmers 和 Unix domain 被明确用作 SFT 候选来源；Android 和 Webmasters 等独立 domain 保留在 held-out/stress evaluation，不与这两个训练来源混用。所有 source 仍必须通过 normalized-query overlap 和 family/group leakage 检查。

每个标准化目录包含 `queries.jsonl`、`documents.jsonl`、`qrels.tsv`、`source-manifest.json`、`leakage-report.json`、`benchmark.yaml` 和 `retrieval-profile.yaml` 等可复现 artifact。上游归档及当前位置如下：

- FiQA：`finetune/data/public-distill-v0/archives/fiqa.zip`，来源 URL 为 `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip`；
- NFCorpus：`finetune/data/public-distill-v0/archives/nfcorpus.zip`，来源 URL 为 `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip`；
- SciFact：`finetune/data/public-distill-v0/archives/scifact.zip`，来源 URL 为 `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`；
- CQADupStack：来源 URL 为 `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip`。本地原压缩包已于 2026-08-31 为缩减仓库体积而删除，但上述两个 prepared 目录仍完整保留；恢复方法和归档 SHA-256 记录在 `finetune/data/public-distill-v0/archives/RESTORE.md`。

候选数据从标准化 source 进入 V2-A 的 lineage 为：

1. FiQA、CQADup Programmers 和 CQADup Unix 的 2,500 个 query 先组成 `finetune/data/public-distill-v0/prepared/pool-main.jsonl`，其旧 teacher 生成结果位于 `finetune/data/public-distill-v0/experiments/public-main-v0/candidates.jsonl`；V2-VH 投影后的 9,977 个 candidate 位于 `finetune/data/public-distill-v0/experiments/v2-vh/projection.jsonl`。
2. NFCorpus 和 SciFact 的 3,723 个 query 使用 V2-VH native generation，共生成 14,892 个 candidate，位于 `finetune/data/public-distill-v0/experiments/v2-vh/native-candidates.jsonl`。
3. 两路 candidate 按 `input_id` 汇总为 6,223 个 query group 和 24,869 个 source candidate；当前统一 query-level ledger 位于 `finetune/data/public-distill-v0/experiments/v2-vh/selected-v2-vh.jsonl`。V2-A 将从这些不可变 source candidate 重新物化 canonical candidate 并构造新的 selection ledger，不覆盖现有 V2-VH artifact。

### 5.2 候选裁切与规范化

V2-A 的标准训练 target 为恰好一条 HyDE，加一条或两条 Vec；正常目标是 `1 HyDE + 2 Vec`，Vec 硬上限为两条。若 source candidate 只有一条合格 Vec，则允许 `1 HyDE + 1 Vec`，不得复制或补造第二条 Vec。

两类 source candidate 的区别及正式输入字段如下：

| 类型 | Source candidate 来源 | 原始特征 | V2-A 正式裁切输入 |
| --- | --- | --- | --- |
| `legacy_v1_projection` | `projection.jsonl`，上游来自 V1 teacher | 原始输出通常为 `3 Lex + 3 Vec + 1 HyDE`；已有 projection 删除 Lex 后多数仍保留 3 Vec | `provenance.original_parsed_output` |
| `native_v2_vh` | `native-candidates.jsonl` | 使用 V2-VH teacher 直接生成；绝大多数有效记录已是 `1 HyDE + 2 Vec` | `output` |

当前 artifact 快照中，9,977 个 legacy projection candidate 有 9,746 个记录了 `dropped_lex_count=3`，其中 8,673 个删除 Lex 后仍为 `1 HyDE + 3 Vec`；14,892 个 native candidate 中有 14,641 个为 `1 HyDE + 2 Vec`、17 个为 `1 HyDE + 1 Vec`，另有 234 个无可用 output。该统计只说明输入形态，V2-A 仍以重新运行 canonicalization 后生成的 manifest 为准。

裁切按以下固定策略对每个 source candidate 执行：

1. 使用严格 parser 读取完整原始输出，保留 raw bytes、解析结果及原始行索引；除了去除每个 payload 首尾的空白字符外，不改写文本。
2. 删除全部 `lex` 行。删除 Lex 是已批准的输出类型投影，不将 Lex 文本合并、改写或转化为 Vec/HyDE。
3. 要求 source candidate 中恰好存在一条非空 `hyde`。缺失 HyDE 或出现多条 HyDE 时标记为 `contract_invalid`，不得选择其中一条继续处理。
4. 按原始出现顺序遍历非空 `vec`。对首尾空白规范化后完全相同的 Vec，只保留第一次出现并记录被删除行的索引与 `exact_duplicate` reason；最多保留前两条不同的 Vec，其余 Vec 以 `vec_cap_exceeded` reason 记录删除。
5. 规范化结果按 `hyde`、`vec`、可选第二条 `vec` 的固定顺序序列化。若没有保留任何 Vec，则标记为 `contract_invalid`；只有一条 Vec 时保留 `1 HyDE + 1 Vec`，不进行内容补全。
6. 对 canonical output 执行第 6 节的完整 Contract 检查。近似重复、语义重复、超长、实体或约束问题均由后续 Contract/语义准入处理，不能通过继续寻找第三条 Vec、改写文本或截断 token 静默修复。

这里的”裁切”只允许删除完整的 Lex 行、完全重复的 Vec 行和超过 Vec 数量上限的完整 Vec 行，绝不允许截断 HyDE 或 Vec 的词、字符或 token。Vec 或 completion 超过硬长度上限时，整个 canonical candidate 直接拒绝。

每个 source candidate 及其 canonical candidate 必须获得新的稳定 ID：

```text
source_candidate_id = {input_id}:{candidate_index}

canonical_candidate_id = sha256(
  source_candidate_id + "\0" +
  canonical_output_bytes
)
```

Canonicalization ledger 至少记录 `source_candidate_id`、`canonical_candidate_id`、source artifact hash、canonical output、deletion_summary（删除 Lex 行数、重复 Vec 数、超限 Vec 数）和最终 Contract 状态。转换不会改变 source artifact；canonical candidate 写入新的 V2-A experiment 目录。

现有 `legacy-subset-candidates.jsonl`、`native-subset-candidates.jsonl` 和 `selected-v2-vh.jsonl` 只作为历史诊断及交叉校验输入，不作为 V2-A 的正式 canonical output 或指标来源。V2-A 不得继承 `1 HyDE + 3 Vec`、旧 `subset_order` 或旧 winner 对应的 retrieval metric；所有 Contract-valid canonical candidate 必须在第 7 节定义的冻结环境中重新评分。

输入 manifest 必须记录：

- 源 artifact 路径、行数、字节数和 SHA-256；
- 唯一 query 数和唯一 candidate 数；
- 候选生成 provider、model revision、prompt version、参数和 provenance；
- source dataset、split、qid、input_id、sample_key，以及可用时的 family/group identifier；
- 完整 raw candidate output 和解析后的 canonical output；
- 与全部 primary held-out query 的重复和重叠报告。

Gold document 和 qrels 绝不能进入 teacher prompt，只能由冻结的 retrieval scorer 使用。

## 6. 输出 Contract

每个候选和 student completion 均使用以下协议：

```text
hyde: <one hypothetical relevant passage>
vec: <one semantic search query>
vec: <optional second complementary semantic search query>
```

Contract 要求：

- 恰好一条 HyDE；
- HyDE 后有一条或两条 Vec；
- 不允许 Lex、think block、说明文字、项目符号或 Markdown wrapper；
- HyDE 目标长度：20–120 词（通过 prompt 引导，不作为硬性拒绝条件）；
- HyDE 安全上限：200 词（超过时拒绝）；
- 每条 Vec 硬上限：48 个 pinned-Qwen token；
- completion 总硬上限：384 个 pinned-Qwen token；
- Vec1 与 Vec2 不能近似重复；
- Vec 不能只是完整复制输入 query。

长度必须被测量并报告，不能通过静默截断强制满足。超过 HyDE/Vec/completion 硬上限时直接拒绝候选。

## 7. 干净环境下重新评分

全部 Contract-valid canonical candidate 必须使用真实 QMD retrieval path，在 `rerank=false` 且关闭自动 expansion 的冻结 profile 下重新评分。`contract_invalid` source/canonical candidate 保留在 ledger 中，但不执行 retrieval，也不能以零分代替 retrieval 结果。

正式 rescoring manifest 必须固定：

- QMD Git commit，且 `qmd_dirty=false`；
- collection name 和 root；
- collection-scoped index fingerprint；
- 当前生效的 document/source mapping；
- embedding model identity 和 SHA-256；
- vector completeness；
- retrieval cutoff 和 limit；
- `result_limit=30`、`per_list_limit=30`、`candidate_limit=40`，或经过显式版本化的后续配置；
- query、document、qrels 和 split hash；
- expansion parser 和 Contract version；
- rerank 关闭；
- retrieval error 为零。

对每条 query 的 raw retrieval 和每个 Contract-valid candidate 计算：

- Recall@10；
- Recall@20；
- Recall@30；
- MRR@10；
- nDCG@10。

## 8. 严格 Pareto 准入与候选选择

### 8.1 硬性准入（Strict Pareto Eligibility）

只有同时满足以下五条 per-query guardrail 的候选才具有 retrieval eligibility：

```text
candidate Recall@10  >= raw Recall@10
candidate Recall@20  >= raw Recall@20
candidate Recall@30  >= raw Recall@30
candidate MRR@10     >= raw MRR@10
candidate nDCG@10    >= raw nDCG@10
```

至少一个指标必须严格高于 raw。任一项退化直接淘汰，不可协商。

### 8.2 字典序 Tie-Breaker（Winner Selection）

同一 query 存在多个合格候选时，按以下字典序选出唯一 winner：

1. 严格提升的指标数量更多（最多 5）— 更偏好检索收益广度；
2. Recall@30 delta 更大；
3. MRR@10 delta 更大；
4. nDCG@10 delta 更大；
5. stable candidate ID 更小 — 纯确定性决胜。

第 1–4 项决定检索收益偏好，第 5 项保证可复现。一旦某一级分出胜负即停止，不继续比较后续级别。

没有合格候选的 query 标记为 `no_pareto_winner`，不得进入 V2-A SFT。

## 9. 语义准入

检索收益不能证明语义忠实性。每个 Pareto-selected candidate 必须通过以下三层语义检查。

### 9.1 确定性检查

确定性检查只覆盖无需 judge 即可判断的项目：

- Vec 通过输出 Contract 中的重复检查，即 Vec1 与 Vec2 不能近似重复。

确定性检查失败的候选在 judge review 前直接拒绝。

### 9.2 全量 LLM Judge

使用独立版本化的 judge 审核所有剩余候选，并返回以下状态之一：

- `pass`；
- `fail`；
- `uncertain`。

Judge 使用规范化 reason code，包括：

- `entity_drift`；
- `lost_constraint`；
- `lost_negation`；
- `unsupported_fact`；
- `unsupported_causality`；
- `ambiguous_overinterpretation`；
- `unsupported_abbreviation_expansion`；
- `non_complementary_vec`；
- `intent_drift`。

必须保留 judge prompt、model revision、decoding parameter、请求与响应 hash、raw response、解析状态和 reason code。

只有最终 `semantic_status=pass` 的记录可以进入 release。

## 10. 修复与失败处理

流水线使用以下明确且互斥的终态：

- `contract_invalid`；
- `no_pareto_winner`；
- `semantic_fail`；
- `eligible`；
- `selected`。

`eligible` 超过 2,000 条容量上限时，在 ledger 中额外标记，不单设终态。

原始输出无法完成结构规范化，以及 canonical output 未通过第 6 节 Contract，终态均为 `contract_invalid`；ledger 必须使用 `failure_stage=canonicalization|contract_validation` 和独立 reason code 区分失败阶段。

错误不得折叠为 `no_pareto_winner`。

候选文本绝不能被静默修复。若重写一个被拒候选，重写结果必须作为具有新 ID 和新 provenance 的新 candidate，并重新执行 Contract validation、retrieval scoring、Pareto admission 和 semantic admission。V2-A 不执行这类再生成；修复属于 fallback/V2-B 路径。

被拒候选继续保留在 ledger 中，用于失败分析和未来可能的 preference pair 构造，但不能作为 SFT 正样本。

## 11. 来源分配与数据规模

最终 train split 包含 1,000–2,000 条记录。Validation/retrieval-dev 独立计算，规模约为 train 的 10%，因此 sealed release 预计共包含约 1,100–2,200 条记录。

分配规则：

1. 稀缺来源的全部合格记录均应使用，其中一部分按 family-disjoint 原则分入 validation。
2. 不为满足目标比例而复制或过采样稀缺来源。
3. 不强制各来源等比例。
4. 将 NFCorpus 占比超过 40% 作为分布预警，而不是硬拒绝规则。
5. 如果合格 train 超过 2,000 条，应减少丰富且重复度高的 stratum，同时保留来源覆盖。
6. 对每个来源记录 `input_count`、`pareto_count`、`semantic_pass_count`、`selected_train_count`、`selected_validation_count` 和拒绝原因。

如果经过语义准入和 family-safe splitting 后的 train 不足 1,000 条，V2-A 不得启动训练。流水线应输出供给报告，并进入需要另行批准的 fallback generation 设计。

## 12. 数据切分

切分要求：

- seed 42；
- 尽可能按来源分层；
- family/group-disjoint；
- normalized-query 与 near-duplicate disjoint；
- validation 规模约为 train 的 10%；
- 与所有 primary held-out 和 stress-test query 零重叠；
- 训练前冻结 split membership 并记录 hash。

Validation split 同时用于 completion-only loss 诊断和 retrieval-dev checkpoint 选择。Validation loss 永远不能用于选择 adapter。

## 13. Student Prompt

V2 prompt 保持现有输出类型，并加入已批准的长度行为。数据物化前必须冻结 prompt 的精确字节。

Prompt 必须包含以下要求。下面保留实际冻结的英文 prompt 原文，避免翻译改变训练配置：

```text
Return exactly one hyde line followed by one or two vec lines.
Target 20-120 words for HyDE; use fewer for a simple query.
Preserve entities, versions, numbers, constraints, negation, comparison, and intent.
Do not invent unsupported facts, causal claims, laws, statistics, or definitions.
Make Vec queries complementary rather than duplicates.
```

正式评估中通过 `SFT-VH + V2 prompt` arm 与 `SFT-VH + 原 prompt` arm 的对比来分离 prompt 效应与数据效应，无需前置独立实验。

## 14. 训练配置

V2-A 保持 SFT-VH 的模型和优化配置不变：

```text
model: Qwen/Qwen3-1.7B
revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
dtype: bfloat16
full_finetuning: false
load_in_4bit: false
max_seq_length: 1024

LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.05
LoRA targets: q/k/v/o/gate/up/down projections

per-device batch: 4
gradient accumulation: 4
effective batch: 16
learning rate: 2e-4
scheduler: cosine
warmup ratio: 0.05
epochs: 3
optimizer: adamw_torch
weight decay: 0.01
seed/data seed: 42 for the candidate run
loss: completion-only
```

加载模型前，对全部 train 和 validation sequence 执行 preflight。超过 1,024 token 的 sequence 直接拒绝；trainer 不得截断 completion-only supervision。

保存 epoch 1、epoch 2 和 epoch 3。设置 `load_best_model_at_end=false`。对每个 checkpoint 执行 retrieval-dev，并根据检索证据选择候选 checkpoint。

Training loss 和 validation loss 只作为诊断。

## 15. Seed 限制

V2-A 训练使用单一 seed 和 data seed 42。单 seed 下的 CI 估计仅反映 query-level variance，不反映 training variance。多 seed 验证留待后续实验。

## 16. 评估分层

### 16.1 Retrieval-Dev

Retrieval-dev 是冻结的 V2 validation split，用于：

- epoch 1/2/3 checkpoint 选择；
- 开发阶段诊断与失败分析。

它不得进入正式 held-out macro 结果。

### 16.2 Primary Held-Out Suite

Primary suite 包含：

- SciFact test；
- FiQA test；
- CQADup Android；
- NFCorpus test；
- FreshStack test。

无论 query 数量多少，每个数据集在 macro 中等权。Primary test 的 query、qrels、document 和结果不得用于修改数据 release、prompt、selection rule、checkpoint rule 或 semantic policy。

进行 V2 test 比较前，必须在相同的干净 profile 下补充 SFT-VH baseline 在 NFCorpus test 和 FreshStack test 上的评估。

### 16.3 Stress Test

以下 stress test 独立报告：

- CQADup Webmasters；
- 短实体和歧义 query；
- 否定、数字、版本和缩写 slice；
- 长代码和错误信息 query；
- 经人工审核的 semantic-drift 集合。

Webmasters 不进入 primary macro，不能单独否决一个具有通用价值的 expansion 模型。实质性退化仍必须公开，并可用于推动后续 routing 设计。

## 17. 正式比较 Arm

正式评估包含：

| Arm | 用途 |
| --- | --- |
| Raw query | 原始检索 baseline |
| SFT-VH + 原 prompt | 当前已部署的 research baseline |
| SFT-VH + V2 prompt | 分离 prompt 效应与数据效应 |
| V2-A selected checkpoint + V2 prompt，seed 42 | 候选数据实验 |

所有非 raw arm 必须使用相同的 parser、retrieval profile、index、embedding model、cutoff、limit、rerank 设置和 latency 测量方法。

## 18. 晋级门槛

V2-A 与 SFT-VH + V2 prompt arm 比较。与 raw 和原始 SFT-VH arm 的比较作为额外系统级证据报告。

在五数据集等权 primary macro 上，V2-A 必须满足：

```text
五指标等权 macro 平均点估计 > 0
至少 3/5 指标的 paired-bootstrap 95% CI 下界 > 0
```

生成 gate：

```text
format errors = 0
fallbacks = 0
generation errors = 0
```

每个数据集必须报告 per-query win/tie/loss 数量及 top gain/loss。等权 macro 结果不得掩盖某个数据集的实质性退化。

Latency 报告包含 expansion generation 和 retrieval。除非后续生产决策明确批准其端到端成本和兼容性，否则 V2-A 仍属于 research candidate。

## 19. 必需 Artifact

V2-A package 必须包含以下核心 artifact：

- 候选输入清单与 canonicalization ledger；
- 重复与泄漏审计；
- 冻结的 retrieval environment 和 index fingerprint；
- raw 与 per-candidate retrieval 结果；
- Pareto selection ledger；
- LLM judge 的 raw 与规范化结果；
- 来源分配报告；
- split manifest；
- sealed SFT train 和 validation JSONL；
- release manifest；
- training config 和 run manifest；
- seed 42 的三个 checkpoint adapter；
- 正式 retrieval run；
- paired bootstrap 和 per-query win/loss 报告。

Raw source 和 candidate artifact 保持不可变。Derived artifact 写入新的 experiment ID，绝不能覆盖 SFT-VH 证据。

## 20. 本设计的批准项

以下决策已在交互过程中确认：

- 优化综合检索质量；
- 使用严格的五指标 per-query Pareto 准入；
- Legacy V1 candidate 删除全部 Lex、按原始顺序最多保留两条不同 Vec，并与 native candidate 一起规范化为 `1 HyDE + 1–2 Vec`；裁切只删除完整行，不截断或改写文本；
- 采用先 V2-A、后 V2-B 的两阶段顺序；
- 稀缺来源的合规记录全部保留，不因配额丢弃；
- 最终训练记录为 1,000–2,000 条；
- 全量确定性检查和 LLM 语义审核；
- HyDE 目标长度 20–120 词（prompt 引导），安全上限 200 词；Vec 硬上限 48 token，completion 硬上限 384 token；
- 使用明确的失败 ledger，不做静默修复；
- seed 42 单 seed 实验，CI 仅反映 query-level variance；
- 使用五数据集 primary held-out suite；
- Webmasters 作为单独报告的 stress test。

# SFT-V2-VH 设计

状态：已批准，待实施计划

日期：2026-08-23

范围：复用已有 DeepSeek 候选，构建一个只生成短 HyDE 和少量互补 Vec 的 completion-only LoRA adapter，并在不使用 reranker 的检索阶段优化 Recall@20。

## 1. 目标与非目标

### 1.1 目标

SFT-V2-VH 将模型任务定义为：

> 输入原始 query，输出一段短 HyDE 和最少量的互补 Vec，使相关文档进入 Top20，同时不降低 R@10、MRR@10 和 nDCG@10。

本轮建立完整闭环：

1. 使用旧候选完成 Vec-count pilot；
2. 复用旧三源候选，并为新增 scientific 来源生成候选；
3. 使用无 reranker 的真实 QMD retrieval 重新选择训练目标；
4. 封板独立 V2 release；
5. 从相同 Qwen3-1.7B base 训练 epoch 1/2；
6. 使用独立 retrieval-dev 选 checkpoint；
7. 在冻结四数据集上完成 no-rerank 正式评估。

### 1.2 非目标

本轮不包含：

- Lex/BM25 expansion 训练；
- HyDE-only 训练或 ablation；
- DPO、GRPO 或其他 preference/RL 训练；
- 动态路由；
- reranker 训练或 reranked retrieval 评估；
- LoRA、LR、batch 等训练参数搜索；
- GGUF、Hub 发布或生产部署；
- 修改或覆盖任何 sealed v0/v1 artifact。

## 2. 已锁定决策

| 项目 | 决策 |
| --- | --- |
| 主指标 | Recall@20 |
| Top10 guardrail | R@10、MRR@10、nDCG@10 均不得逐 query 下降 |
| Retrieval | `rerank: false`，使用原 query BM25/vector、VH vector lists 和 weighted RRF |
| 旧数据 | 复用全部 10,000 组候选，不继承旧 winner 标签 |
| 新来源 | NFCorpus train/dev 与 SciFact train |
| 正式评测 | FiQA test、CQADupStack Android、CQADupStack Webmasters、SciFact test |
| Teacher | `deepseek-v4-flash` |
| 数据规模 | 至少 1,000 条 strong winner |
| Tie | 仅补来源覆盖，最多占最终 SFT 数据 20% |
| 训练 | 从相同 pinned Qwen3-1.7B base 重训，completion-only LoRA，epoch 1/2 |
| 模型选择 | 五源独立 retrieval-dev，五源等权宏平均 |
| 正式 Gate | 四数据集等权宏平均，不设置单数据集否决条件 |

## 3. 版本与不可变边界

以下现有对象保持 sealed，不得原地修改：

```text
public-distill-v0
public-distill-v1
training-target-v1.1
```

SFT-V2-VH 使用独立版本：

```text
release_id: public-distill-v2-vh
contract_version: training-target-v2-vh
student_prompt_version: qmd-student-expansion-v2-vh
```

旧数据的复用必须产生新的派生 artifact。每个候选记录以下 generation provenance 之一：

- `legacy_v1_projection`：旧候选删除 Lex 后的投影；
- `native_v2_vh`：新增来源通过 VH-only Prompt 生成；
- `fallback_v2_vh`：旧 query 无合格 winner 时按需补生成。

不得把三类来源合并成无法区分的 target。

## 4. 数据来源与隔离

### 4.1 五个训练来源

```text
fiqa-train
cqadup-programmers
cqadup-unix
nfcorpus-train-dev
scifact-train
```

FiQA、programmers 和 unix 读取 `public-distill-v0/public-main-v0` 中全部 10,000 组候选。不得只读取旧的 956 个 winner；旧 selection label 和旧 reranked metrics 全部作废。

NFCorpus train/dev 和 SciFact train 使用 VH-only Prompt 新生成候选。SciFact test 保持正式 held-out；报告必须说明它是同数据集 held-out，不再代表完全跨数据集的 zero-shot 泛化。

### 4.2 去重与 retrieval-dev

在候选生成或 SFT target 选择前完成去重和划分：

1. query 使用 Unicode NFKC、casefold、trim 和连续空白折叠；
2. 删除与正式评测 query 的规范化精确碰撞；
3. 来源提供 family/group ID 时，同一 family 不得跨 train/dev/test；
4. 不使用 embedding 或 LLM 猜测不存在的 family；
5. 每个来源按固定 seed 和稳定 query ID 留出 10% 作为 retrieval-dev；
6. retrieval-dev query 永不进入 SFT target。

正式评测四数据集不得参与 Prompt、Vec count、checkpoint 或阈值选择。

## 5. Vec-count pilot

Pilot 从旧三源的 SFT pool 中按来源分层固定抽取 300 个 query。旧候选删除 Lex 后，对每组包含一条 HyDE 和最多三条 Vec 的候选枚举全部非空 Vec 子集：

```text
HyDE + Vec1
HyDE + Vec2
HyDE + Vec3
HyDE + 任意两条 Vec
HyDE + 三条 Vec
```

不运行 HyDE-only。

每个组合先通过 Contract，再使用 `rerank:false` 运行 retrieval。只保留 R@20 严格高于 raw，且 R@10、MRR@10、nDCG@10 均不低于 raw 的组合。

同 query 的组合按以下顺序选择：

1. R@20 更高；
2. R@10 更高；
3. nDCG@10 更高；
4. MRR@10 更高；
5. Vec 更少；
6. 总 token 更短；
7. 稳定 candidate/subset ID。

Vec 上限按旧三源等权宏平均冻结：只有两条 Vec 相对一条继续提高 R@20 且不损害 Top10，才允许最多两条；第三条同理。否则冻结为更小的数量。Pilot 结果和最终 Vec 上限写入 manifest 后，才能生成新增来源候选和正式 student Prompt。

## 6. V2 输出 Contract

逻辑协议为：

```text
hyde: <exactly one hypothetical relevant passage>
vec: <at least one semantic reformulation>
vec: <optional complementary reformulations up to the pilot limit>
```

机械规则：

- 禁止 Lex；
- 顺序固定为 HyDE 后接 Vec；
- HyDE 固定一条；
- Vec 至少一条，最多为 pilot 冻结值；
- 同类型内容规范化后不得重复；
- 每项非空、单行，不含控制字符或 chat control token；
- 禁止静默修复和截断。

### 6.1 HyDE 长度

```text
推荐生成目标：40–80 English words
低于 30 words：warning
超过 100 words：warning
超过 128 个 pinned Qwen tokens：reject
```

不设置 word-count 硬下限或硬上限。word count、token count、warning 和 tokenizer revision 必须写入 candidate provenance。

### 6.2 Vec 与总长度

```text
Vec 推荐生成目标：每条 8–20 English words
单条 Vec 超过 32 个 pinned Qwen tokens：reject
总 completion 超过 224 个 pinned Qwen tokens：reject
```

word count 不作为 Vec 的硬拒绝条件。任何超限 target 只能淘汰或通过 fallback 重新生成，不能中间截断。

## 7. Teacher 与 Prompt

Teacher 请求复用上次已验证配置：

```text
model: deepseek-v4-flash
endpoint: https://api.deepseek.com/chat/completions
thinking: {"type":"disabled"}
response_format: {"type":"json_object"}
temperature: omitted
```

VH-only Prompt 从现有 DeepSeek JSON Prompt 派生，但必须建立新版本并冻结 SHA-256。它要求：

- exactly 1 HyDE；
- pilot 冻结数量范围内的 Vec；
- no Lex；
- HyDE 目标 40–80 English words；
- Vec 简短且互补；
- 保留实体、限制条件和否定；
- 不擅自展开不确定缩写；
- 不引入 query 不支持的实体、事实或因果结论。

每个新 query 独立生成四组完整 VH candidates。API key 不得进入 artifact。

## 8. Retrieval 与 winner 选择

所有 raw、candidate、dev 和正式评估统一使用：

```yaml
result_limit: 30
per_list_limit: 30
candidate_limit: 40
rerank: false
reranker_model: null
auto_generate_expansions: false
```

实际路径为：

```text
original query BM25
+ original query vector
+ HyDE/Vec vector lists
→ weighted RRF
→ Top30
```

旧 artifact 中使用 reranker 产生的 metrics 不能复用。原 query 与每个 VH subset 必须在同一 collection、index、embedding model 和 retrieval profile 下重新执行。

分类规则：

```text
winner:
  candidate R@20 > raw R@20
  candidate R@10 >= raw R@10
  candidate MRR@10 >= raw MRR@10
  candidate nDCG@10 >= raw nDCG@10

qualified_tie:
  四项指标均与 raw 完全相同

tradeoff:
  candidate R@20 > raw R@20
  但至少一项 Top10 指标下降

no_winner:
  其他已成功评分的情况
```

检索异常不是 `no_winner`，必须保留为 error 并恢复评分。

## 9. 语义检查与最小充分 target

机械 Contract 通过且满足 winner/tie 条件后，使用独立 judge 调用检查：

- 实体漂移；
- 否定丢失；
- 限定条件丢失；
- 不确定缩写被擅自展开；
- 无依据事实或因果结论；
- Vec 之间缺少互补性。

Judge 复用 `deepseek-v4-flash` 和 API 配置，但使用独立版本化 judge Prompt，输入原 query 和完整 VH target，输出：

```text
pass | fail | uncertain
reason_codes
```

`pass` 可进入数据集，`fail` 淘汰，`uncertain` 在人工确认前不得进入 SFT。Judge 原始响应、Prompt hash 和模型配置必须保留。

若最高 retrieval candidate 未通过语义检查，按既定 retrieval 排序检查下一候选，不重新调用 teacher。最终 target 始终使用通过所有门禁的最小 Vec 集合。

## 10. SFT release 构成

- strong winner 是主体；
- qualified tie 只从 winner 覆盖不足的来源补充；
- tie 不得超过最终 SFT records 的 20%；
- tradeoff、no_winner、judge fail、未完成人工确认的 uncertain 不进入 SFT；
- 至少获得 1,000 条 strong winner 才允许启动训练。

若 strong winner 不足 1,000，首先只对没有合格 winner 的 query 调用 VH-only teacher 补生成。补生成后重复同一 Contract、retrieval 和 judge 流程，不降低 winner 门槛，也不提高 tie 比例。

Release manifest 至少记录：

- source/release/experiment ID；
- query、candidate 和 selected subset ID；
- generation provenance；
- raw 与 candidate 全部 metrics；
- Contract 和 judge 结果；
- Prompt/tokenizer/model/index/retrieval profile hash；
- word/token 长度；
- train/dev split；
- core artifact SHA-256。

## 11. Student Prompt 与训练

Student Prompt 使用独立 VH-only 版本：

```text
/no_think Generate dense-retrieval expansions for the search query.

Output only:
Exactly 1 hyde followed by the allowed number of vec entries.
Do not output lex entries or explanations.

HyDE target: 40-80 English words.
Each vec should be concise and add a complementary retrieval angle.
Preserve entities, constraints, and negation.

Query: {query}
```

正式 Prompt 将“allowed number”替换为 pilot 冻结后的明确数量要求，不保留动态占位描述。

训练与 ckpt226 保持单变量可比：

- 相同 pinned Qwen3-1.7B revision；
- 相同 LoRA、seed、batch、LR schedule；
- 从 base 重新训练，不从 ckpt226 续训；
- completion-only loss；
- `packing=false`；
- 不允许训练时截断；
- 只训练 epoch 1 和 epoch 2，并保存两个 checkpoint。

validation loss 只记录，不用于 checkpoint 选择。

## 12. Retrieval-dev 与 checkpoint 选择

五个来源各自的 10% retrieval-dev 使用 qrels 和来源独立 collection。模型对 dev query 现场生成 VH，再以 `rerank:false` 检索。

五源等权宏平均按以下顺序选择 epoch：

1. R@20；
2. R@10；
3. nDCG@10；
4. MRR@10。

只有在 retrieval metrics 完全相同时才使用生成 token 更少的 checkpoint。正式四数据集不参与 epoch 选择。

## 13. 正式评估

所有 arm 使用冻结四数据集、同一 index、embedding model 和 no-rerank profile：

| Arm | 作用 |
| --- | --- |
| Raw | 无 expansion 基线 |
| ckpt226 Vec+HyDE | 当前 student 基线 |
| DeepSeek Vec+HyDE | teacher ceiling |
| Base + V2 Prompt | 分离 Prompt 收益 |
| SFT-V2-VH epoch 1 | 候选模型 |
| SFT-V2-VH epoch 2 | 候选模型 |

为避免 Vec 数量混杂，ckpt226 和 DeepSeek 的已有输出先删除 Lex，再按原始顺序最多保留 pilot 冻结数量的 Vec；不得使用 qrels 挑选 Vec。Base 和 SFT arm 直接使用同一个 V2 Contract。这样所有非 raw arm 都是 count-matched Vec+HyDE。

主指标为 R@20；R@10、MRR@10 和 nDCG@10 是 Top10 guardrail；R@30、Contract-valid rate、截断率、生成 token 和延迟作为诊断。

四数据集使用等权宏平均，不设置单数据集否决条件，但必须逐数据集完整报告。paired bootstrap 固定 20,000 次、seed 42。

“Top10 无显著负向退化”定义为：相对对应 baseline 的 paired-bootstrap delta，其 95% CI 上界不得小于 0；若整个区间均低于 0，则判定为显著退化。

### 13.1 三级结果

```text
Level 1 — Pipeline valid
  Contract-valid rate >= 99%
  truncation rate < 1%
  Lex output count = 0
  四数据集评估完整

Level 2 — Research success
  R@20 相对 ckpt226 的宏平均点估计 > 0
  R@20 相对 raw 的宏平均点估计 > 0
  Top10 相对 ckpt226 和 raw 均无显著负向退化

Level 3 — Strong success
  R@20 相对 ckpt226 的 paired-bootstrap 95% CI 下界 > 0
```

达到 Level 2 即可标记为 `research_candidate`；Level 3 表示强统计证据。生成 token 和延迟必须报告，但不阻塞首轮 research success。DeepSeek 只作为 ceiling，不要求首轮追平。

## 14. 未达 Gate 时的优化顺序

所有优化只使用 retrieval-dev，不反复查看正式测试集。

先定位失败层：

- DeepSeek VH 不提升：停止训练优化，检查 VH/no-rerank 假设或 RRF；
- DeepSeek 提升但 Base+Prompt 不提升：检查小模型 Prompt 理解或基础能力；
- Base+Prompt 提升但 SFT 退化：检查训练数据和训练配置；
- R@20 提升但 Top10 下降：减少 Vec 数或清理 candidate displacement 样本；
- 格式或截断失败：修正 Contract 与训练分布。

优化顺序固定为：

1. 比较 epoch 1/2；
2. 增加强 winner，不增加 tie 比例；
3. 对无 winner query 补生成；
4. 清理语义漂移、重复 Vec 和异常长度；
5. 降低最大 Vec 数；
6. 小幅调整 Prompt；
7. 最后才调整 LR 或 LoRA。

本轮失败不直接进入 DPO。

任何改变 Vec 上限、Prompt 或训练参数的优化都必须产生新的实验版本和 retrieval-dev 结果，不能原地修改已封板的 V2 release。

## 15. 实现边界与验证

复用现有 public distill、completion-only training 和 benchmark pipeline，只增加三个能力：

1. 将旧候选投影为 VH subsets；
2. 以 no-rerank R@20/Pareto 规则重新评分和选择；
3. 生成并验证 V2 release、训练和评估 artifacts。

最小测试范围：

- 删除 Lex 后 HyDE/Vec 文本逐字节不变；
- 不生成 HyDE-only subset；
- winner、qualified_tie、tradeoff、no_winner 边界正确；
- `rerank:false` 时 reranker 调用次数为 0；
- HyDE word warning 和 128-token reject 正确；
- Vec 和 completion token hard limit 正确；
- train/retrieval-dev query 零重叠；
- release 行数、hash 和 provenance 可复核；
- 失败恢复读取冻结 candidate，不重复调用 teacher。

中间 artifact 按阶段落盘。API、解析、Contract、retrieval 或 judge 错误必须保留真实状态，不能转换成 `no_winner`。

## 16. 审批后的交付顺序

本设计批准后，下一步只编写实施计划。实施按以下顺序展开：

1. Vec-count pilot；
2. V2 Contract、projection 与 no-rerank selection；
3. 新来源与 fallback generation；
4. semantic judge 与 release；
5. completion-only SFT；
6. retrieval-dev checkpoint 选择；
7. 正式四数据集评估和报告。

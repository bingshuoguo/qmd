# SFT-V2-VH 设计 · rev1.2（带修订标记）

状态：**待重新批准**（rev0 已批准；本稿包含 **8 项生效修订**；R-1、R-4、R-5、R-10 已经评审回退）

日期：2026-08-23（rev1.2）

基线：[2026-08-23-sft-v2-vh-design.md](2026-08-23-sft-v2-vh-design.md)（rev0，不修改）

### rev1.1 → rev1.2 变更

| 变更 | 内容 |
| --- | --- |
| **R-1 回退** | 删除 Gate-0A/Gate-0B 前置阻塞；no-rerank 和长度统计直接进入正式 pipeline。 |
| **R-3 固定** | `EPSILON_NDCG` 固定为 `0.02`，不得按 winner 供给反向调节。 |
| **R-4 回退** | 恢复四数据集等权宏平均为主 Gate；排除 SciFact 的三数据集宏平均作为敏感性诊断。 |
| **R-10 回退** | 恢复至少 1,000 条 strong winner 才允许训练；tie 不得用于补足门槛。 |

范围：复用已有 DeepSeek 候选，构建一个只生成短 HyDE 和少量互补 Vec 的 completion-only LoRA adapter，并在不使用 reranker 的检索阶段优化 Recall@20。

---

## 修订标记约定

| 标记 | 含义 |
| --- | --- |
| ~~删除线~~ | rev0 中被删除的内容 |
| **【R-n】** | 修订编号，对应下方《修订说明》 |
| 🔴 | 阻塞性修订——不改会导致计划失败或结论不可用 |
| 🟡 | 成本修订——不改不会失败，但会浪费显著预算 |
| 🟢 | 一致性修订——修正内部矛盾或表述缺陷 |

---

## 修订说明

| # | 位置 | 级别 | 修订内容 | 依据 |
| --- | --- | --- | --- | --- |
| ~~**R-1**~~ | ~~新增 §0~~ | ⬜ | **【已回退 rev1.2】** 不设置 no-rerank 或长度标定前置 Gate；相关统计在正式 pipeline 中完成 | 用户决定直接进入实验闭环 |
| **R-2** | §6.1 §6.2 | 🔴 | HyDE reject 128→256 tok；单条 Vec 32→48；总 completion 224→384；删除 word warning 频段；新增重复度检查 | 128 使旧候选存活率仅 21.5%；配对检验显示长度对 R@20 无影响（P=0.118，点估计偏向长） |
| **R-3** | §8 | 🔴 | winner 判定的 per-query Top10 硬约束改为容许 epsilon 退化 | rev0 在最耗数据处用最严判据，在最终判定处用最松判据，严格度倒置 |
| ~~**R-4**~~ | ~~§13 §13.1~~ | ⬜ | **【已回退 rev1.2】** 四数据集等权宏平均恢复为主 Gate；三数据集口径作为敏感性诊断 | SciFact train/test 是合法 held-out；该比较明确定位为系统级对比 |
| ~~**R-5**~~ | ~~§5~~ | ⬜ | **【已回退 rev1.1】** ~~删除 Vec 子集全枚举 pilot；Vec 数量由 §9 逐 query 最小化决定，上限事后从分布读取~~ — 保留 rev0 的 Vec-count pilot 原样 | 评审决定保留 pilot；残留功效问题以"报告 CI"方式缓解，见 §5 注记 |
| **R-6** | §9 | 🟡 | semantic judge 从逐条阻塞门禁降级为抽样审计 + 确定性检查；删除人工确认阻塞 | 候选已通过直接检索测试；`uncertain→人工确认` 是全自动流水线中唯一的无界阻塞点 |
| **R-7** | §10 | 🟡 | `qualified_tie` 从 release 配方降级为 ablation 变量 | tie 的定义就是"可证明无检索效果"；v1 的 58% tie 训出的 ckpt226 优势微弱 |
| **R-8** | §11 | 🟢 | 明确 `num_train_epochs: 3`，只保存/评估 epoch 1 与 epoch 2 | "相同 LR schedule" 与 "只训练 epoch 1/2" 在 cosine 下互斥 |
| **R-9** | §7 §11 §6.1 | 🟢 | 拆开「Prompt 生成目标」与「Contract 拒绝阈值」两个旋钮；**Prompt 中显式保留两级长度控制**（目标 40–80 words + 硬上限 never exceed 150 words），Contract 侧保留 256-token reject 作为安全网 | DeepSeek 在 "never exceed 60 words" 指令下实际产出中位 90 词——Prompt 塑造分布、Contract 兜住尾巴，两者都需要 |
| ~~**R-10**~~ | ~~§10~~ | ⬜ | **【已回退 rev1.2】** 至少 1,000 条 strong winner 才允许训练 | 用户选择质量和训练规模硬门槛 |
| **R-11** | §13.1 | 🟢 | 为 Level 3 增加可达性说明 | Level 3 要求吃掉 teacher-student 差距的约 80% |
| **R-12** | §5 §12 | 🟢 | 压缩 tiebreak 阶梯 | per-query 尺度上中间层级几乎不生效 |

### 证据索引

本稿所有数值均从仓库内已封板 artifact 实测，使用 spec 指定的 pinned tokenizer（`Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`）。

| 编号 | 事实 | 来源 |
| --- | --- | --- |
| E-1 | embedding 上下文窗口 = `min(2048, trainContextSize)`，2048 才触发截断 | [src/llm.ts:1217](../../../src/llm.ts#L1217)、`truncateToContextSize()` |
| E-2 | 旧候选 HyDE p50 = 147 tok；>128 tok 占 **73.2%**（n=9,969） | `public-main-v0/candidates.jsonl` |
| E-3 | 旧候选删 Lex 后 completion p50 = 231 tok；>224 tok 占 **55.1%** | 同上 |
| E-4 | rev0 三条限制全过的旧候选仅 **21.5%**；有候选存活的 query 仅 **53.5%**（子集化抢救后 59.8%） | 同上 |
| E-5 | 同 query 配对检验（n=1,510，4 候选跨越 128 边界）：ΔR@20(短−长) = **−0.0069**，95% CI [−0.0182, +0.0047]，P(短更好)=0.118 | `public-main-v0/scored.jsonl` |
| E-6 | 未配对比较为 +0.0049 偏向短——控制 query 后符号翻转，属混杂 | 同上 |
| E-7 | DeepSeek v6 prompt 明写 "20-60 English words; never exceed 60 words"，实际产出 p50 = 118 tok（≈90 词），>128 tok 占 **36.8%**，max 371 tok | `deepseek-json4-repro-system.txt`、`public-eval-v1/*/expansions/` |
| E-8 | ckpt226 HyDE p50=152 / p90=208 / **max=765（三数据集全等于生成上限）**；而 v1 训练数据 HyDE max 仅 415 tok | 同上、`public-main-v1/sft.jsonl` |
| E-9 | 契约放宽至 256/48/384 后，候选存活率 **90.8%**、query 存活率 **99.6%** | `public-main-v0/candidates.jsonl` |
| E-10 | v0 实际产出 956 winner（pre-cap，占 2,500 query 的 38.2%）、2,260 accepted、2,000 materialized = 846 winner + 1,154 tie（**58% 为 tie**） | `public-main-v0/release-manifest.json`、`public-main-v1/final-audit.json` |
| E-11 | v0 选择配置为 `rerank: true` + lexicographic (R@10, nDCG@10, MRR@10) | `public-main-v0/release-manifest.json` |
| E-12 | 仓库内不存在任何 `rerank:false` 的检索运行 | `eval-grid.txt`、`runs/results/` |
| E-13 | teacher(DeepSeek vec+hyde) 相对 ckpt226 在 R@30 的宏平均优势 ≈ **+0.016**；n=2,153 时 recall 的 paired-bootstrap CI 半宽 ≈ **±0.013** | `TYPE-MATCHED-EVAL.md` |

---

## 1. 目标与非目标

### 1.1 目标

SFT-V2-VH 将模型任务定义为：

> 输入原始 query，输出一段短 HyDE 和最少量的互补 Vec，使相关文档进入 Top20；逐 query 保证 R@10 不下降、nDCG@10 下降不超过 0.02，并在聚合评估中保护 R@10、MRR@10 和 nDCG@10。

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

以下两项为本轮**新纳入**的低成本 ablation，只在 retrieval-dev 上进行，各自独立与主版本对比（不交叉），合计增加两次训练：

- **HyDE 长度 ablation**（**【R-2】**）：target ≤128 tok 与 ≤256 tok 两个数据版本各训一次，比较截断率与 R@20。此项用于回答 §6.1 末尾记录的、配对检验**无法**否定的那个假设——"更短的训练目标是否让 student 生成更可靠"；
- **tie ablation**（**【R-7】**）：含 tie 与不含 tie 两个数据版本。

## 2. 已锁定决策

| 项目 | 决策 |
| --- | --- |
| 主指标 | Recall@20 |
| Top10 guardrail | ~~R@10、MRR@10、nDCG@10 均不得逐 query 下降~~ **【R-3】** 逐 query 选择容许 epsilon 退化（§8）；**聚合层面**不得显著负向退化（§13） |
| Retrieval | `rerank: false`，使用原 query BM25/vector、VH vector lists 和 weighted RRF |
| 旧数据 | 复用全部 10,000 组候选，不继承旧 winner 标签 |
| 新来源 | NFCorpus train/dev 与 SciFact train |
| 正式评测 | FiQA test、CQADupStack Android、CQADupStack Webmasters、SciFact test |
| 主 gate 口径 | FiQA test、Android、Webmasters、SciFact test 四数据集等权宏平均；排除 SciFact 的三数据集口径作为敏感性诊断 |
| Teacher | `deepseek-v4-flash` |
| 数据规模 | 至少 1,000 条 strong winner 才允许启动训练（§10） |
| Tie | ~~仅补来源覆盖，最多占最终 SFT 数据 20%~~ **【R-7】** 默认不进入 release；作为 ablation 变量（§10） |
| Vec 数量 | 由 §5 Vec-count pilot 按旧三源等权宏平均冻结全局上限（rev0 策略保留） |
| **【R-2】** 长度契约 | HyDE ≤ 256 tok，单条 Vec ≤ 48 tok，总 completion ≤ 384 tok |
| 训练 | 从相同 pinned Qwen3-1.7B base 重训，completion-only LoRA，**【R-8】** `num_train_epochs: 3`，只保存/评估 epoch 1 与 epoch 2 |
| 模型选择 | 五源独立 retrieval-dev，五源等权宏平均 |
| 正式 Gate | 四数据集等权宏平均，不设置单数据集否决条件；同时报告排除 SciFact 的三数据集敏感性口径 |

## 3. 版本与不可变边界

*（本节 rev0 内容全部保留，无修订。）*

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

NFCorpus train/dev 和 SciFact train 使用 VH-only Prompt 新生成候选。

SciFact train 与 SciFact test 是独立 split；test query、qrels 和 target 不参与训练、winner 选择、Prompt 调整或 checkpoint 选择，因此 SciFact test 是合法 held-out，可参与四数据集等权宏平均。由于 ckpt226 未使用 SciFact train，新旧模型对比应表述为完整系统级对比，而不是受控数据 ablation。

正式报告必须同时给出：

- 四数据集等权宏平均，作为主 Gate；
- 排除 SciFact 的三数据集等权宏平均，作为敏感性诊断，用于观察收益是否主要来自 SciFact 域内训练。

### 4.2 去重与 retrieval-dev

*（本节 rev0 内容全部保留，无修订。）*

在候选生成或 SFT target 选择前完成去重和划分：

1. query 使用 Unicode NFKC、casefold、trim 和连续空白折叠；
2. 删除与正式评测 query 的规范化精确碰撞；
3. 来源提供 family/group ID 时，同一 family 不得跨 train/dev/test；
4. 不使用 embedding 或 LLM 猜测不存在的 family；
5. 每个来源按固定 seed 和稳定 query ID 留出 10% 作为 retrieval-dev；
6. retrieval-dev query 永不进入 SFT target。

正式评测四数据集不得参与 Prompt、Vec count、checkpoint 或阈值选择。

## 5. Vec-count pilot

> **【R-5 已回退】** rev1 曾提议删除本节的全枚举 pilot，改由 §9 逐 query 最小化决定 Vec 数量。评审决定**保留 rev0 原策略**，本节按 rev0 恢复。原修订理由中提出的两个问题，处理方式如下：
>
> - **"样本被契约掏空"**——此问题已被 **【R-2】** 大幅缓解。在 rev0 的 128/32/224 下，300 个抽样 query 中只有约 53.5% 还剩合法候选（有效 n≈160）；放宽到 256/48/384 后 query 存活率升至 **99.6%**（E-9），有效 n≈299。pilot 的功效问题主要来自长度契约，而非 pilot 设计本身。
> - **"没有显著性判据"**——决策规则**不变**（仍按点估计的严格大于），但新增一条非侵入的审计要求，见本节末尾注记。

Pilot 从旧三源的 SFT pool 中按来源分层固定抽取 300 个 query。旧候选删除 Lex 后，对每组包含一条 HyDE 和最多三条 Vec 的候选枚举全部非空 Vec 子集：

```text
HyDE + Vec1
HyDE + Vec2
HyDE + Vec3
HyDE + 任意两条 Vec
HyDE + 三条 Vec
```

不运行 HyDE-only。

每个组合先通过 Contract（**【R-2】** 使用 §6.1/§6.2 的 256/48/384 阈值），再使用 `rerank:false` 运行 retrieval。**【R-3】** 只保留 R@20 严格高于 raw、R@10 不低于 raw、且 nDCG@10 不低于 `raw - 0.02` 的组合（与 §8 的 winner 判据保持一致；rev0 此处的"R@10、MRR@10、nDCG@10 均不低于 raw"随 R-3 一并放宽，否则 pilot 与正式选择会用两套判据）。

**【R-12】** 同 query 的组合按以下顺序选择（rev0 七级压缩为四级——per-query 尺度上 R@10 / nDCG@10 / MRR@10 极度离散，中间层级几乎不生效，绝大多数情况直接落到"Vec 更少"）：

1. R@20 更高；
2. ~~R@10 更高；~~ ~~nDCG@10 更高；~~ ~~MRR@10 更高；~~ Vec 更少；
3. 总 token 更短；
4. 稳定 candidate/subset ID。

Vec 上限按旧三源等权宏平均冻结：只有两条 Vec 相对一条继续提高 R@20 且不损害 Top10，才允许最多两条；第三条同理。否则冻结为更小的数量。Pilot 结果和最终 Vec 上限写入 manifest 后，才能生成新增来源候选和正式 student Prompt。

> **【R-5 回退后的残留风险 · 审计要求】** 上述冻结规则基于宏平均**点估计**的严格大于，未设显著性判据。参照 E-13，四数据集 n≈2,153 时 recall 的 paired-bootstrap CI 半宽已达 ±0.013；pilot 在 n≈300 上比较的 1-vec / 2-vec / 3-vec 差距按 type-matched 报告约为 0.005–0.01 量级，处于噪声尺度内。这意味着**当差异不可分辨时，"严格大于"会系统性地冻结到更小的 Vec 数**。
>
> 决策规则不变，但必须满足：
>
> 1. pilot 报告须对每一级比较（2-vec vs 1-vec、3-vec vs 2-vec）**同时给出 paired-bootstrap 95% CI**（20,000 次，seed 42），与点估计并列；
> 2. 若某级比较的点估计为正但 CI 跨 0，manifest 必须标注 `frozen_under_noise: true`；
> 3. 被标注 `frozen_under_noise` 的 Vec 上限，在 §14 优化顺序第 6 步（"降低最大 Vec 数"）中应优先重新考察其反方向——即也允许**上调**。
>
> 这三条只增加报告内容，不改变冻结结果，因此不影响 pilot 的阻塞位置与交付顺序。

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

### 6.1 HyDE 长度 **【R-2】🔴 + 【R-9】🟢**

> **修订理由（详细）。** rev0 的 128 token reject 有四条可能依据，实测下**没有一条成立**：
>
> **(a) embedding 截断？不是。** `EMBED_CONTEXT_SIZE` 默认 2048，`resolveEmbedTokenLimit()` 取 `min(2048, trainContextSize)`，embeddinggemma-300M 训练窗口即 2048（E-1）。旧候选 HyDE 的 p50 = 147 tok，**占窗口 7%**。128 与任何技术边界无关。（type-matched 报告中的 768 是 `max_new_tokens` 生成预算，不是 embedding 窗口——rev0 混淆了这两个数。）
>
> **(b) 短 HyDE 检索更好？证据方向相反。** 未配对分箱呈**非单调**（128–160 区间 ΔR@20 掉到 +0.0019 谷底，200–260 又回升到 +0.0172），这不是长度效应的形状。取 4 个候选恰好跨越 128 边界的 query 做同 query 配对（同 query、同 teacher、同检索配置，唯一差别是 HyDE 长短，n=1,510）：**ΔR@20(短−长) = −0.0069，95% CI [−0.0182, +0.0047]，P(短更好) = 0.118**（E-5）。控制 query 后符号翻转（E-6）——未配对比较测的是 query 难度，不是 HyDE 长度。**长度对 R@20 没有可检出的影响，点估计甚至偏向长的。**
>
> **(c) 防止生成截断？128 治不了这个病。** ckpt226 三个数据集的 HyDE max **全部精确等于 765 tok**（`max_new_tokens=768` 的天花板），而 p50=152、p90=208（E-8）。这是**退化重复**的签名，不是"写长了"。关键在于 v1 训练数据的 HyDE max 只有 415 tok——**那条 765 的尾巴不是从训练数据学来的**，所以收紧训练数据长度上限对该失败模式无效。
>
> **(d) 对齐 teacher 行为？teacher 自己不遵守。** DeepSeek v6 的 prompt 明写 "20-60 English words; **never exceed 60 words**"，实际产出 p50 = 118 tok（≈90 词），36.8% 超过 128 tok，max 371 tok（E-7）。**而这个不遵守长度指令的 arm，正是 type-matched 评估中赢了 ckpt226 的那一个。** 128 的 reject 会砍掉获胜 teacher 输出的 36.8%，且砍得有偏——留下的是 teacher 恰好写得短的 query，通常意味着 query 更简单。
>
> **代价量化：**
>
> | HyDE cap | 旧候选被拒 (n=9,969) | DeepSeek v6 被拒 (n=1,832) |
> | --- | --- | --- |
> | **128 tok（rev0）** | **73.2%** | **36.8%** |
> | 160 tok | 38.5% | 12.6% |
> | 192 tok | 16.8% | 4.5% |
> | 224 tok | 6.2% | 1.4% |
> | **256 tok（rev1）** | **2.5%** | **0.4%** |
> | 320 tok | 0.5% | 0.2% |

**【R-9】** rev0 把**生成目标**与**拒绝阈值**当成同一件事的两种表述，导致两边都设错。本修订将其拆为两个独立旋钮——**两者都保留，各司其职**：

| 旋钮 | 作用 | 位置 | 取值 |
| --- | --- | --- | --- |
| **Prompt 长度控制** | 塑造分布——让 teacher/student 倾向于写短 HyDE | §7、§11 | 两级：目标 40–80 words；硬上限 `never exceed 150 English words` |
| **Contract hard reject** | 兜住尾巴——拦截退化、跑飞、解析错误 | 本节 | **256 pinned Qwen tokens** |

> **为什么两者都要。** E-7 表明 Prompt 不足以单独承担长度控制（DeepSeek 在 "never exceed 60 words" 下中位产出 90 词）；E-8 表明 Contract 也不足以单独承担（ckpt226 的 765-token 尾巴是退化重复，训练数据侧的长度上限管不到推理期）。Prompt 负责让分布中心左移，Contract 负责让尾巴被拒绝并记录——**取消任何一个都会留下缺口**。
>
> Prompt 侧的硬上限用**词数**表达（150 words），Contract 侧用 **token** 判定（256 tok）。150 words × 1.31 tok/word ≈ 197 tok，相对 256 留约 25% 余量，使得"遵守了 Prompt 却被 Contract 拒绝"成为小概率事件。1.31 的换算比来自 DeepSeek v6 实测（118 tok / ≈90 words，E-7）。正式生成报告继续记录实际换算比，但不把它设为前置 Gate。

```diff
  推荐生成目标：40–80 English words（由 §7 Prompt 承担，见上表）
- 低于 30 words：warning
- 超过 100 words：warning
- 超过 128 个 pinned Qwen tokens：reject
+ 超过 256 个 pinned Qwen tokens：reject
```

**【R-2】** 删除 word-count warning 频段。rev0 中没有任何环节消费这些 warning——它们不影响选择、不影响 gate、不影响报告。同时，teacher 的 word 遵守度已证明有 50% 量级误差（E-7），基于 word 的判定不可靠。

**【R-2】** 新增**退化检查**（这才是真正对应 ckpt226 那条 765-token 尾巴的失败模式）：

```text
HyDE 内部 5-gram 重复率 > 30%：reject（reason_code: degenerate_repetition）
```

不设置 word-count 硬下限或硬上限。word count、token count、~~warning、~~ 重复率和 tokenizer revision 必须写入 candidate provenance。

> **本修订不能否定的一个理由（须记录）。** 上述配对检验证明的是"给定一段 teacher 生成的 HyDE，其长度不影响检索结果"。它**没有**证明"用更短的目标训练 student 不会让 student 生成得更可靠"。后者可能才是 40–80 词背后真正的（rev0 未言明的）假设：ckpt226 训练在 p50=147 的数据上，推理时长出 765 token 的退化尾巴；训练在更短的数据上，尾巴或许消失。这是**合理且可检验**的假设，因此本稿：
>
> - 不把长度限制包装成检索质量措施（该说法已被 E-5 否定）；
> - 将其列为 §1.2 的 **HyDE 长度 ablation**（≤128 vs ≤256 两个数据版本），由 §13 已有的 `截断率` 诊断指标回答；
> - 该 ablation 的成本远低于被删除的 Vec-count pilot。

### 6.2 Vec 与总长度 **【R-2】🔴**

```diff
  Vec 推荐生成目标：每条 8–20 English words（advisory）
- 单条 Vec 超过 32 个 pinned Qwen tokens：reject
+ 单条 Vec 超过 48 个 pinned Qwen tokens：reject
- 总 completion 超过 224 个 pinned Qwen tokens：reject
+ 总 completion 超过 384 个 pinned Qwen tokens：reject
```

> **修订理由。** 旧候选删 Lex 后 completion p50 = 231 tok，**>224 tok 占 55.1%**（E-3）；单条 Vec >32 tok 占 23.8%。若只放宽 HyDE 而不同步放宽这两项，放宽没有意义——总长度上限会独立拒掉过半候选。
>
> 三条限制放在一起的整体效果（未计子集化抢救）：
>
> | 契约配置 | 候选存活率 | query 存活率 |
> | --- | --- | --- |
> | **128 / 32 / 224（rev0）** | **21.5%** | **53.5%** |
> | 192 / 48 / 288 | 75.5% | 96.7% |
> | **256 / 48 / 384（rev1）** | **90.8%** | **99.6%** |
> | 256 / 64 / 448 | 95.3% | 99.9% |
>
> 选择 256/48/384 的理由：DeepSeek 实测 max 371 tok、v1 训练数据 max 415 tok 均在合理范围内，384 的总上限能容纳"一条正常 HyDE + 三条正常 Vec"，同时仍能拦住真正跑飞的输出。

word count 不作为 Vec 的硬拒绝条件。任何超限 target 只能淘汰或通过 fallback 重新生成，不能中间截断。

以上三个阈值在本版中直接冻结为 256/48/384，并写入 manifest；不设置额外长度标定 Gate。

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
- **【R-9】 HyDE 长度控制（两级，必须显式写入 Prompt）**：
  - 目标：`the single hyde query should contain 40-80 English words`；
  - 硬上限：`never exceed 150 English words`（对应 §6.1 的 256-token reject，留约 25% 余量）；
- Vec 简短且互补；
- 保留实体、限制条件和否定；
- 不擅自展开不确定缩写；
- 不引入 query 不支持的实体、事实或因果结论。

> **【R-9】** 关于长度控制的预期校准（须记录，但**不构成删除长度控制的理由**）：DeepSeek 在措辞更强硬的 "never exceed 60 words" 下，实际中位产出为 90 词（E-7）——即 Prompt 的长度指令**有效但不精确**，它把分布往左推，却不能保证上界。因此设计上：
>
> - Prompt 侧保留两级控制，负责塑造分布；
> - Contract 侧保留 256-token reject（§6.1），负责拒绝真正越界的产出；
> - 正式生成必须测出 V2 Prompt 对 40–80 词目标的**实际达成率**与 150 词上限的**实际越界率**，写入 manifest；
> - 若越界率显著（>10%），在 §14 第 7 步调整 Prompt 措辞并产生新版本，**而不是**放宽 Contract。

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

### 分类规则 **【R-3】🔴**

> **修订理由。** rev0 存在**严格度倒置**：
>
> - **§8（逐 query 选数据）**：MRR@10 在单个 query 上掉任意一点，候选即出局；
> - **§13（最终判定）**："95% CI 上界不得小于 0；若整个区间均低于 0，则判定为显著退化"——按 E-13 的 CI 半宽 ±0.013，需要 Top10 真实下降超过 0.013 才算失败。
>
> 也就是说，**在最耗数据、最影响供给的地方用了近乎不可能满足的判据，在真正该保护质量的地方用了近乎不可能触发的判据**。聚合层面的"不退化"本就允许部分 query 退化——用 per-query 硬约束去执行一个 aggregate 约束，代价是 winner 供给崩塌（见 §10），收益为零。
>
> 此外，per-query 尺度上这些指标极度离散（SciFact 平均每 query 仅约 1.1 个相关文档），"下降任意一点"往往意味着单个文档的位次变化，噪声占比极高。

```diff
  winner:
    candidate R@20 > raw R@20
-   candidate R@10 >= raw R@10
-   candidate MRR@10 >= raw MRR@10
-   candidate nDCG@10 >= raw nDCG@10
+   candidate R@10 >= raw R@10                        （保留：recall 不得倒退）
+   candidate nDCG@10 >= raw nDCG@10 - 0.02           （放宽：仅排序指标）
+   （MRR@10 的 per-query 约束移除，仍在 §13 聚合层面作为 guardrail）

  qualified_tie:
    四项指标均与 raw 完全相同

  tradeoff:
    candidate R@20 > raw R@20
-   但至少一项 Top10 指标下降
+   但 nDCG@10 下降超过 0.02，或 R@10 下降

  no_winner:
    其他已成功评分的情况
```

**【R-3】** `EPSILON_NDCG` 固定为 **0.02**，在 pilot、正式 winner 选择和测试中使用同一值，并写入 manifest。不得为了达到 winner 数量而反向调节 epsilon；供给不足只能通过一轮 fallback generation 处理。

**【R-3】** 保留 R@10 的硬约束（recall 是主指标族，不容倒退）；放宽的只有排序指标 nDCG@10，并完全移除 MRR@10 的 per-query 约束——MRR@10 与 nDCG@10 在 per-query 尺度高度相关，同时约束两者不增加保护、只减少供给。MRR@10 仍在 §13 聚合层面作为 guardrail。

检索异常不是 `no_winner`，必须保留为 error 并恢复评分。

## 9. 语义检查与最小充分 target **【R-6】🟡**

> **修订理由。** rev0 要求对每个候选做一次额外 DeepSeek judge 调用（六类失败原因，`pass|fail|uncertain`，最高候选不过就沿排序往下试），且 `uncertain` **在人工确认前不得进入 SFT**。两个问题：
>
> 1. **它在用代理指标复核一个已经直接测过的东西。** 能走到 judge 这一步的候选，已经通过真实 QMD 检索、R@20 严格提升、Top10 未显著退化。judge 要抓的"实体漂移/否定丢失"如果真的损害检索，检索测试已经把它筛掉了；如果没有损害检索，那么在一个**唯一目标是 Recall@20** 的项目里，淘汰它的依据是什么？
> 2. **它在全自动流水线里插了一个无界的人工步骤**，且 rev0 未给出 `uncertain` 的预期比例、处理人或时间预算。这是整条链路上唯一可能让本轮无限期停摆的环节。

~~机械 Contract 通过且满足 winner/tie 条件后，使用独立 judge 调用检查：~~

**【R-6】** 语义检查降级为**两层**结构：

### 9.1 确定性检查（逐条，阻塞）

零 LLM 调用，纯规则，对每个候选 target 执行：

- **否定保留**：原 query 中的否定词（`not/no/without/never/except/non-` 等）必须在 HyDE 中有对应表达；
- **实体覆盖**：原 query 中的大写实体、引号短语、数字/单位在 HyDE 或至少一条 Vec 中出现（规范化后）；
- **Vec 互补性**：任意两条 Vec 之间的 token Jaccard 相似度 < 0.8；
- **退化重复**：见 §6.1。

未通过者淘汰，记录 `reason_code`。

### 9.2 抽样审计（非阻塞）

在 release 封板前，对最终 SFT records **随机抽取 200 条**（固定 seed），用独立版本化 judge Prompt 复用 `deepseek-v4-flash` 检查：

- 实体漂移；
- 否定丢失；
- 限定条件丢失；
- 不确定缩写被擅自展开；
- 无依据事实或因果结论；
- Vec 之间缺少互补性。

输出 `pass | fail | uncertain` 与 `reason_codes`。Judge 原始响应、Prompt hash 和模型配置必须保留。

**审计结果不阻塞 release**，但：

```text
若 fail 率 > 5%：
  必须在 release manifest 中记录，并在 §14 优化顺序中提升"清理语义漂移"的优先级
若 fail 率 > 15%：
  暂停封板，回到 §9.1 补充确定性规则或调整 §7 Prompt，产生新实验版本
```

~~`pass` 可进入数据集，`fail` 淘汰，`uncertain` 在人工确认前不得进入 SFT。~~ **【R-6】** 删除人工确认阻塞。`uncertain` 计入审计统计，不触发逐条人工处理。

### 9.3 最小充分 target

若最高 retrieval candidate 未通过 §9.1 的确定性检查，按既定 retrieval 排序检查下一候选，**不重新调用 teacher**。最终 target 始终使用通过所有门禁的最小 Vec 集合（在 §5 冻结的全局上限之内）。

## 10. SFT release 构成 **【R-7】🟡**

> **【R-7】修订理由。** `qualified_tie` 的定义是"四项指标均与 raw 完全相同"——即**可证明没有产生任何检索效果的扩展**。rev0 允许它占最终数据的 20%，理由是"仅补来源覆盖"。历史数据很有说服力：v1 实际是 **846 winner + 1,154 tie = 58% 为 tie**（E-10），训出的正是 ckpt226——那个相对 raw 优势微弱、在 webmasters 上为负的模型。从 58% 降到 20% 方向正确，但若没有"为什么 tie 有帮助"的假设（格式正则化？长度分布锚定？），它只是在拿次要目标稀释主目标。
>
**【R-7】** release 构成：

- **strong winner 是唯一主体**；
- ~~qualified tie 只从 winner 覆盖不足的来源补充；~~ ~~tie 不得超过最终 SFT records 的 20%；~~
- **qualified_tie 默认不进入 release**。它作为 §1.2 列出的 **tie ablation** 变量：额外产出一个 `+tie` 数据版本（tie 占比 ≤20%，仅补来源覆盖），在 retrieval-dev 上与主版本对比。哪个版本进入正式评估由 retrieval-dev 结果决定，而非预先烧进配方；
- tradeoff、no_winner、语义检查未通过者不进入 SFT。

数据规模使用硬启动 Gate：

```text
strong winner >= 1,000:
  允许训练主版本

strong winner < 1,000:
  只对没有合格 winner 的 query 调用 VH-only teacher 补生成一轮
  补生成后重复同一 Contract、retrieval 和语义检查流程

补生成后仍 < 1,000:
  不启动任何 SFT 训练
  报告实际供给、各来源 winner 率和失败原因
```

qualified tie 不计入 1,000 条 strong winner，也不得用来补足门槛。达到门槛并训练主版本后，才允许额外构造 tie 占比不超过 20% 的 ablation release。

Release manifest 至少记录：

- source/release/experiment ID；
- query、candidate 和 selected subset ID；
- generation provenance；
- raw 与 candidate 全部 metrics；
- Contract 和语义检查结果；
- Prompt/tokenizer/model/index/retrieval profile hash；
- word/token 长度**【R-2】**、重复率；
- **【R-9】** HyDE 的 40–80 词达成率与 150 词越界率；
- **【R-3】** `EPSILON_NDCG` 取值；
- Vec-count pilot 结果、冻结的全局 Vec 上限，及各级比较的 paired-bootstrap CI 与 `frozen_under_noise` 标记（§5）；
- train/dev split；
- core artifact SHA-256。

## 11. Student Prompt 与训练

Student Prompt 使用独立 VH-only 版本：

```text
/no_think Generate dense-retrieval expansions for the search query.

Output only:
Exactly 1 hyde followed by the allowed number of vec entries.
Do not output lex entries or explanations.

HyDE target: 40-80 English words. Never exceed 150 words.
Each vec should be concise and add a complementary retrieval angle.
Preserve entities, constraints, and negation.

Query: {query}
```

**【R-9】** student prompt 的 HyDE 长度控制与 §7 teacher prompt 保持**同一套两级表述**（目标 40–80 words + 硬上限 never exceed 150 words）。理由：ckpt226 在没有任何上限措辞的 prompt 下长出了 765-token 的退化尾巴（E-8），而训练数据侧的长度上限对推理期无约束力——上限措辞是推理期唯一的 prompt 级防线。

正式 Prompt 将"allowed number"替换为 §5 pilot 冻结后的明确数量要求，不保留动态占位描述。

### 训练配置 **【R-8】🟢**

> **修订理由。** rev0 §11 同时要求"相同 LoRA、seed、batch、**LR schedule**"与"只训练 epoch 1 和 epoch 2"。这两条在 cosine schedule 下**互斥**：ckpt226 的来源配置是 `num_train_epochs: 3` + cosine + warmup 0.05。
>
> - 若设 `num_train_epochs=2`：cosine 在第 2 epoch 末衰减到 ~0；而 ckpt226 所在位置（3-epoch cosine 走到 2/3）当时 LR 约为峰值的 25%。整条轨迹都不同——epoch 1 末为 1.0e-4 vs 1.5e-4。
> - 只有保持 `num_train_epochs=3` 而不评估 epoch 3，才是真正的单变量可比。

训练与 ckpt226 保持单变量可比：

- 相同 pinned Qwen3-1.7B revision；
- 相同 LoRA（`r=16, alpha=32, dropout=0.05`，target = q/k/v/o/gate/up/down proj）、seed（42）、batch（4 × accum 4 = 16）、LR schedule（`2.0e-4`，cosine，warmup 0.05）；
- **【R-8】** `num_train_epochs: 3`（**保持不变，以维持与 ckpt226 相同的 LR 轨迹**）；
- **【R-8】** `save_strategy: epoch`，保存 epoch 1、2、3 三个 checkpoint；**只有 epoch 1 与 epoch 2 参与 §12 的选择与 §13 的评估**，epoch 3 仅归档；
- 从 base 重新训练，不从 ckpt226 续训；
- completion-only loss；
- `packing=false`；
- 不允许训练时截断。

validation loss 只记录，不用于 checkpoint 选择。

> **【R-8】** 须在报告中明确记录的一个限制：即便训练超参完全一致，**SFT-V2-VH 与 ckpt226 的对比仍不是受控 ablation**。两者至少在五个维度不同：(1) 训练数据的选择配置（rerank=true + R@10 优先 → rerank=false + R@20 优先，E-11）；(2) 来源（3 源 → 5 源）；(3) 输出契约（含 Lex → 无 Lex）；(4) Prompt 版本；(5) 长度分布。§13 的 `Base + V2 Prompt` arm 能分离 (4)，但 (1) 无法被任何现有 arm 分离。因此 SFT-V2-VH vs ckpt226 应表述为**系统级对比**，不得表述为"训练数据改进带来 X 收益"。

## 12. Retrieval-dev 与 checkpoint 选择

五个来源各自的 10% retrieval-dev 使用 qrels 和来源独立 collection。模型对 dev query 现场生成 VH，再以 `rerank:false` 检索。

**【R-12】** 五源等权宏平均按以下顺序选择 epoch（rev0 四级压缩为两级——见 §5 同类修订理由）：

1. R@20；
2. ~~R@10；~~ ~~nDCG@10；~~ ~~MRR@10；~~ nDCG@10。

只有在 retrieval metrics 完全相同时才使用生成 token 更少的 checkpoint。正式评测数据集不参与 epoch 选择。

**【R-2】【R-7】** §1.2 的两个 ablation（HyDE 长度、tie）同样只在此 retrieval-dev 上评估，选出的组合进入 §13。

## 13. 正式评估

所有 arm 使用冻结数据集、同一 index、embedding model 和 no-rerank profile：

| Arm | 作用 |
| --- | --- |
| Raw | 无 expansion 基线 |
| ckpt226 Vec+HyDE | 当前 student 基线 |
| DeepSeek Vec+HyDE | teacher ceiling |
| Base + V2 Prompt | 分离 Prompt 收益 |
| SFT-V2-VH epoch 1 | 候选模型 |
| SFT-V2-VH epoch 2 | 候选模型 |

*（六 arm 设计保留，无修订——每个 arm 回答一个不同问题，且 §14 的失败定位阶梯直接消费它们。）*

为避免 Vec 数量混杂，ckpt226 和 DeepSeek 的已有输出先删除 Lex，再按原始顺序最多保留 §5 pilot 冻结数量的 Vec；不得使用 qrels 挑选 Vec。Base 和 SFT arm 直接使用同一个 V2 Contract。这样所有非 raw arm 都是 count-matched Vec+HyDE。

主指标为 R@20；R@10、MRR@10 和 nDCG@10 是 Top10 guardrail；R@30、Contract-valid rate、截断率、生成 token 和延迟作为诊断。

宏平均口径：

```text
主 gate 口径：FiQA test + CQADupStack Android + CQADupStack Webmasters + SciFact test
              四数据集等权宏平均（n ≈ 2,153）

敏感性诊断：排除 SciFact 的三数据集等权宏平均。
             该口径用于观察收益是否主要来自 SciFact 域内训练，不取代主 Gate。

必须同时给出四数据集主口径和排除 SciFact 的三数据集敏感性口径，以四数据集为准。
不设置单数据集否决条件，但必须逐数据集完整报告。
```

paired bootstrap 固定 20,000 次、seed 42。

"Top10 无显著负向退化"定义为：相对对应 baseline 的 paired-bootstrap delta，其 95% CI 上界不得小于 0；若整个区间均低于 0，则判定为显著退化。

### 13.1 三级结果 **【R-11】**

```text
Level 1 — Pipeline valid
  Contract-valid rate >= 99%
  truncation rate < 1%
  Lex output count = 0
  四数据集评估完整

Level 2 — Research success
  R@20 相对 ckpt226 的四数据集宏平均点估计 > 0
  R@20 相对 raw 的四数据集宏平均点估计 > 0
  Top10 相对 ckpt226 和 raw 均无显著负向退化

Level 3 — Strong success
  R@20 相对 ckpt226 的 paired-bootstrap 95% CI 下界 > 0
```

达到 Level 2 即可标记为 `research_candidate`；Level 3 表示强统计证据。生成 token 和延迟必须报告，但不阻塞首轮 research success。DeepSeek 只作为 ceiling，不要求首轮追平。

> **【R-11】 Level 3 可达性说明（须写入报告的期望校准）。** 按 E-13：
>
> - teacher（DeepSeek vec+hyde）相对 ckpt226 在 R@30 的宏平均优势 ≈ **+0.016**；
> - n≈2,153 时 recall 的 paired-bootstrap CI 半宽 ≈ **±0.013**。
>
> Level 3 要求 student 相对 ckpt226 的宏平均增益 **> ~0.013**，即**吃掉 teacher-student 差距的约 80%**。这对单轮 LoRA SFT 是很高的门槛。**Level 2 是本轮的现实目标，Level 3 是 stretch**——不应因为未达 Level 3 而触发 §14 的优化循环。§14 的触发条件是未达 **Level 2**。

## 14. 未达 Gate 时的优化顺序

所有优化只使用 retrieval-dev，不反复查看正式测试集。

**【R-11】** 触发条件：未达 **Level 2**（Level 3 未达不触发）。

先定位失败层：

- DeepSeek VH 不提升：检查 VH/no-rerank 假设或 RRF，不继续进行训练参数优化；
- DeepSeek 提升但 Base+Prompt 不提升：检查小模型 Prompt 理解或基础能力；
- Base+Prompt 提升但 SFT 退化：检查训练数据和训练配置；
- R@20 提升但 Top10 下降：清理 candidate displacement 样本；`EPSILON_NDCG=0.02` 不因结果而移动 **【R-3】**；
- 格式或截断失败：**【R-2】** 优先检查退化重复检测是否漏放，再考虑修正 Contract 与训练分布。

优化顺序固定为：

1. 比较 epoch 1/2；
2. **【R-2】【R-7】** 查看 §1.2 两个 ablation（HyDE 长度、tie）的 retrieval-dev 结果，选择更优组合；
3. 增加强 winner；
4. 对无 winner query 补生成（最多一轮，§10）；
5. 清理语义漂移、重复 Vec 和异常长度；
6. 调整最大 Vec 数——若 §5 该级比较被标注 `frozen_under_noise`，**上调与下调同等优先**；
7. 小幅调整 Prompt——**【R-9】** 若实际生成显示 150 词越界率 >10%，优先在此步收紧 Prompt 措辞，**不得改为放宽 §6.1 的 256-token reject**；
8. 最后才调整 LR 或 LoRA。

本轮失败不直接进入 DPO。

任何改变 Vec 上限、长度阈值、Prompt 或训练参数的优化都必须产生新的实验版本和 retrieval-dev 结果，不能原地修改已封板的 V2 release。`EPSILON_NDCG=0.02` 是本轮固定定义，不作为优化变量。

## 15. 实现边界与验证

复用现有 public distill、completion-only training 和 benchmark pipeline，只增加三个能力：

1. 将旧候选投影为 VH subsets；
2. 以 no-rerank R@20/Pareto 规则重新评分和选择；
3. 生成并验证 V2 release、训练和评估 artifacts。

最小测试范围：

- 删除 Lex 后 HyDE/Vec 文本逐字节不变；
- 不生成 HyDE-only subset；
- winner、qualified_tie、tradeoff、no_winner 边界正确 **【R-3】**（含固定 `EPSILON_NDCG=0.02` 边界：恰好等于、略超、略不足三种情况）；
- `rerank:false` 时 reranker 调用次数为 0；
- ~~HyDE word warning 和 128-token reject 正确；~~ **【R-2】** HyDE 256-token reject 与 5-gram 重复度 reject 正确；
- **【R-2】** Vec 48-token 与 completion 384-token hard limit 正确；
- **【R-9】** word count 记录进 provenance 但不参与 Contract 判定（负向测试：一个 160 词但 <256 token 的 HyDE 必须通过 Contract，同时被计入"150 词越界"统计）；
- **【R-12】** Vec 子集选择的 tiebreak 顺序正确，且在指标相同时优先选 Vec 更少的子集；
- **【R-6】** §9.1 确定性检查（否定保留、实体覆盖、Vec Jaccard）正确；
- train/retrieval-dev query 零重叠；
- 主 Gate 使用四数据集宏平均，且报告同时产出排除 SciFact 的三数据集敏感性口径；
- release 行数、hash 和 provenance 可复核；
- 失败恢复读取冻结 candidate，不重复调用 teacher。

中间 artifact 按阶段落盘。API、解析、Contract、retrieval 或语义检查错误必须保留真实状态，不能转换成 `no_winner`。

## 16. 审批后的交付顺序

本设计批准后，下一步只编写实施计划。实施按以下顺序展开：

1. Vec-count pilot（§5，含 CI 审计要求）；
2. V2 Contract、projection 与 no-rerank selection；
3. 新来源与 fallback generation；
4. ~~semantic judge 与 release~~ **【R-6】** 确定性语义检查、抽样审计与 release；
5. strong winner 达到 1,000 条后，执行 completion-only SFT（主版本 + §1.2 两个 ablation 版本）；
6. retrieval-dev checkpoint 与 ablation 选择；
7. 正式评估和报告（四数据集主 Gate + 排除 SciFact 的三数据集敏感性诊断）。

---

## 附录 A：净效果小结

| 维度 | rev0 | rev1.2 |
| --- | --- | --- |
| 旧候选存活率 | 21.5% | 90.8% |
| 有效 query 存活率 | 53.5% | 99.6% |
| Vec-count pilot 有效 n（抽样 300） | ≈160（被长度契约掏空） | ≈299（R-2 放宽后） |
| pilot 决策可审计性 | 仅点估计 | 点估计 + paired-bootstrap CI + `frozen_under_noise` 标记 |
| HyDE 长度控制 | 仅 Contract（128 tok，无依据） | Prompt 两级（40–80 / ≤150 words）+ Contract（256 tok，有依据） |
| 人工阻塞点 | 1 个（judge `uncertain` 逐条确认） | 0 个 |
| 主 Gate 口径 | 四数据集等权宏平均 | 四数据集等权宏平均；三数据集敏感性诊断 |
| 内部矛盾 | LR schedule / 严格度倒置 / 长度依据缺失 | 已消解 |
| 前置阻塞 | 无 | 无；no-rerank 和长度统计进入正式 pipeline |

## 附录 B：rev1.2 未解决、需在实施计划中决定的问题

1. **NFCorpus 与 SciFact train 的实际可用 query 数**——影响是否能达到 §10 的 1,000 strong winner 硬门槛，正式数据准备时确认。
2. **HyDE 长度 ablation 的两个档位是否就是 128/256**——当前默认 128/256；若正式 V2 Prompt 分布显示该划分失去区分度，实施计划需版本化调整。
3. **Prompt 侧 150 词上限与 Contract 侧 256 token 的实际换算比**——正式生成时记录；若换算比明显偏高，后续版本下调 150 词上限，不放宽 Contract。
4. **tie ablation 与 HyDE 长度 ablation 是否需要交叉**——本稿默认不交叉（各自独立对比主版本），以控制训练次数在 3 次以内。
5. **Vec-count pilot 若各级比较全部落在 `frozen_under_noise`**——即 1/2/3 条 Vec 在 n≈300 上完全不可分辨时的处置。本稿默认按规则冻结为 1，但 §14 第 6 步允许上调；是否改为直接冻结为 3（保留最大信息量）需要决定。

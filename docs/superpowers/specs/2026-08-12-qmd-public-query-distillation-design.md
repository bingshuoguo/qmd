# QMD Public Query Retrieval Distillation 数据生产设计

状态：待审批

日期：2026-08-12

范围：只设计 Public Query Tasks 的数据生产流程；审批后另写实施计划

## 1. 背景与目标

QMD 的 expansion model 最终应服务于通用的英文检索场景，包括技术文档、代码知识、工作文档和个人笔记，而不应只适配 SciFact 的科学/医学问句。

第一阶段先使用公开检索数据验证一件事：能否用 teacher 生成多样 expansion，再通过 QMD 的真实检索结果和公开 qrels 自动选择训练目标，从而得到 1,000–2,000 条可用于 completion-only SFT 的数据。第一阶段不引入 LLM judge，也不构造本地 Synthetic Query Tasks。

本设计的交付物是可追溯的数据集，不是已经证明有效的模型。模型收益仍需在后续独立 benchmark 上验证。

## 2. 已锁定的决策

| 项目 | 决策 |
|---|---|
| 数据范围 | Public Query Tasks only |
| 输入池 | FiQA train 750；CQADupStack programmers 875；CQADupStack unix 875，共 2,500 条 |
| teacher | DeepSeek `deepseek-v4-flash`，每个 query 独立生成 4 个候选 |
| prompt | 固定使用 `finetune/benchmarks/distill/prompts/deepseek-json4-v2-system.txt`，不做 prompt 实验 |
| 自动判定 | 不使用 LLM judge |
| 检索排序 | `Recall@10 > nDCG@10 > MRR@10`，按字典序比较 |
| 接受标签 | `winner` 与 `qualified_tie` 都进入训练集，标签必须保留 |
| 最低门槛 | 通过 Contract v1.1；不能所有 expansion 都只是重复原 query |
| 目标规模 | 1,000–2,000 条 accepted records，指 winner 与 qualified tie 的总数 |
| 检索隔离 | 三个来源使用三个独立 collection，query 只能检索自身来源的 corpus |
| Synthetic 数据 | 本阶段不构建；Public-only 证明有收益后再讨论 |

## 3. 非目标

本阶段不做以下工作：

- 不调整或比较 teacher prompt。
- 不使用 LLM 判断语义、实体、错误码、版本或否定词是否保留。
- 不把词面重叠、embedding 相似度或人工规则作为新的语义门禁。
- 不从 FiQA test、CQADupStack holdout domains 或 SciFact test 选择训练目标。
- 不改变 QMD 线上 expansion/retrieval 行为。
- 不把 Synthetic Query Tasks、私有文档或用户数据混入数据集。
- 不以 winner 数量不足为理由临时更换数据源、放宽 qrels 或挑选 raw 表现较差的 query。

## 4. 数据源合同

### 4.1 来源定义

| `source_id` | 公开来源 | query/qrels | corpus | 正例定义 | 主池配额 |
|---|---|---|---|---|---:|
| `fiqa-train` | BEIR FiQA | `qrels/train.tsv` | FiQA corpus | 原始 relevance `>= 2`，在 QMD 评分时二值化为 1 | 750 |
| `cqadup-programmers` | BEIR CQADupStack programmers | domain 的 `qrels/test.tsv` | programmers corpus | 原始 binary relevance `> 0` | 875 |
| `cqadup-unix` | BEIR CQADupStack unix | domain 的 `qrels/test.tsv` | unix corpus | 原始 binary relevance `> 0` | 875 |

CQADupStack 官方只提供 test qrels；这里的 programmers/unix 是公开数据派生的训练来源，因此后续不能再把这两个 domain 当作独立测试集。

本阶段固定从 BEIR 官方公开归档获取数据：

| archive | URL | 官方 MD5 |
|---|---|---|
| `fiqa.zip` | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip` | `17918ed23cd04fb15047f73e6c3bd9d9` |
| `cqadupstack.zip` | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip` | `4e41456d7df8ee7760a7f866133bda78` |

数据准备必须记录：下载 URL、原始压缩包文件名、官方 MD5、实际 MD5、解压后 corpus/queries/qrels 的 SHA-256、来源 split/domain、原始 license metadata 和 BEIR license disclaimer。MD5 不匹配时立即失败，不得继续。

### 4.2 独立评估保留集

以下集合在本阶段保持 untouched，不能参与抽样、candidate selection、阈值调整或 prompt 调整：

- FiQA dev 和 test。
- CQADupStack android 和 webmasters。
- SciFact test。

正式 pool 冻结前，对训练 query 与上述保留集做 NFKC、casefold、首尾空白清理、连续空白折叠后的精确碰撞检查。碰撞 query 从训练候选中移除，并记录原因。v0 不用 embedding 或 LLM 做近似重复判定。

## 5. 核心数据与所有权

数据生产围绕三个不可变核心对象展开：`source manifest`、`input pool`、`distillation record`。

### 5.1 Source manifest

每个来源只有一个 manifest，拥有 corpus、query、qrels、collection 和 index provenance：

```json
{
  "source_id": "fiqa-train",
  "archive": {"url": "...", "md5": "...", "sha256": "..."},
  "source": {"dataset": "fiqa", "split": "train", "domain": null},
  "relevance_threshold": 2,
  "artifact_hashes": {"corpus": "...", "queries": "...", "qrels": "..."},
  "collection_name": "qmd-distill-public-v0-fiqa-train",
  "index_fingerprint": "..."
}
```

qrels 文件是 relevance 的唯一事实来源；pool 和 distillation record 不维护另一份可被独立修改的 qrels。

### 5.2 Input pool

每行只标识一个待蒸馏 query：

```json
{
  "input_id": "fiqa-train:<qid>",
  "source_id": "fiqa-train",
  "qid": "<original-qid>",
  "query": "<original-query>",
  "sample_key": "<sha256>"
}
```

`input_id` 必须 namespaced，避免不同数据集 qid 冲突。主池固定为 2,500 行；smoke pool 另存，不能进入正式 SFT。

### 5.3 Distillation record

每个 input 只有一条 record，内部包含 raw retrieval、4 个候选及最终选择：

```json
{
  "schema_version": "qmd-public-distill-v0",
  "input_id": "fiqa-train:<qid>",
  "source_id": "fiqa-train",
  "qid": "<qid>",
  "query": "<query>",
  "raw": {"status": "scored", "metrics": {"recall_at_10": 0, "ndcg_at_10": 0, "mrr_at_10": 0}},
  "candidates": [],
  "selected_candidate_index": 2,
  "selection_status": "winner"
}
```

`candidates` 保存 teacher 原始响应、解析后的 expansion、Contract 结果、repeat-only 结果、检索状态、top-30 doc IDs 和全部指标。原始响应和规范化输出都要保留，不能只留 winner。

## 6. 可复现抽样

全流程固定 `experiment_seed = 42`，抽样键为：

```text
sha256("qmd-public-v0\0seed=42\0" + source_id + "\0" + qid)
```

每个来源的 eligible query 必须同时满足：

1. query 非空；
2. 至少有一个达到该来源 relevance threshold 的 qrels doc；
3. 所有正例 doc ID 均能映射到对应 corpus；
4. 不与保留集 query 发生规范化后的精确碰撞。

先按规范化 query 在全来源去重；同一规范化 query 只保留 `sample_key` 最小的一条。随后各来源独立按 `sample_key, qid` 升序排序：

- 前 10 条进入 smoke pool；
- 排除 smoke 后，依次取各自的 750/875/875 条进入正式 input pool；
- 某个来源不足其固定配额时 preparation 失败，不从其他来源补齐。

因此 smoke 固定为 30 条，正式 input pool 固定为 2,500 条；两者的 `input_id` 必须零重叠。smoke 的 candidate、selected 和 materialized records 只用于流水线验证，永远不能合并进正式 SFT。正式运行不会因结果好坏重新抽样。

## 7. Teacher 候选生成

### 7.1 固定配置

正式运行固定：

```text
DISTILL_PROVIDER=deepseek
DISTILL_MODEL=deepseek-v4-flash
DISTILL_API_BASE_URL=https://api.deepseek.com
DISTILL_THINKING_MODE=disabled
DISTILL_PROMPT_VERSION=qmd-expansion-teacher-v2-deepseek-json4-semantic-safe
DISTILL_SYSTEM_PROMPT_FILE=finetune/benchmarks/distill/prompts/deepseek-json4-v2-system.txt
DISTILL_USER_PROMPT_TEMPLATE_FILE=finetune/benchmarks/distill/prompts/deepseek-json4-repro-user.txt
DISTILL_MAX_OUTPUT_TOKENS=4096
```

每个 query 发起 4 次独立 API 请求，每次请求产生一个完整 expansion set。每个 candidate 最多尝试 3 次；重试只处理请求失败、空响应或无法解析的 JSON，不因检索分数低而重试。

当前客户端没有显式指定 temperature/top_p，本阶段沿用 provider default，并在 run manifest 中明确记录 `temperature: provider_default`、`top_p: provider_default`。API key 不能写入任何 artifact。

候选生成完成后冻结 `candidates.jsonl`。后续验证、打分和选择必须读取冻结 artifact，不能重新调用 teacher 来“复现”文本。

### 7.2 Prompt 版本固定

正式运行前，system prompt、user prompt template 和环境模板必须进入版本控制，并把各文件 SHA-256 写入 manifest。prompt 内容不是本阶段变量；smoke 失败只能修复流程或解析问题，不能据此迭代 prompt。

## 8. 候选最低门槛

### 8.1 Contract v1.1

这里的 Contract v1.1 指仓库中的 `training-target-v1.1` 机械格式合同。正式运行按当时已提交文件的 SHA-256 固定版本，不在 public pipeline 中重新定义一套合同。

其核心硬规则包括：

- 输入 query 非空且为单行；
- output 由 `[type, text]` 对组成，type 只能是 `lex`、`vec`、`hyde`；
- `lex` 1–3 条、`vec` 1–3 条、`hyde` 0–1 条；
- expansion 文本非空、单行，不含控制字符或 chat control token；
- 同类型内的 expansion 规范化后不能重复；
- lex 引号平衡且含正向检索项；
- `/only:` 类控制指令不能成为训练目标。

长度异常、模板化或时效性提示如果在 Contract 中只是 warning，就只记录 warning，不作为本阶段拒绝条件。public pipeline 不升级 warning 为 error。

### 8.2 非重复规则

定义规范化函数：Unicode NFKC → casefold → trim → 连续空白折叠为一个空格。

如果 candidate 中每一条 expansion 规范化后都等于原 query，则标记 `repeat_only`，不进入检索选择。只要至少一条 expansion 与原 query 不同，就通过此门槛。v0 不计算编辑距离、词面覆盖率或 embedding similarity。

## 9. QMD 检索与打分

### 9.1 检索隔离与 profile

数据生产使用一个实验专属 SQLite 数据库，避免污染用户的全局 QMD 数据库。数据库中建立三个 collection；每个 collection 只包含一个 source 的 corpus。所有查询必须显式指定其 `collection_name`，禁止跨 collection 混检。

检索复用现有 `retrieveForBenchmark() -> executeHybridRetrieval()` 路径，只改变传入的 expansion，不复制或简化 QMD 检索算法。固定 profile：

- `result_limit = 30`
- `per_list_limit = 30`
- `candidate_limit = 40`
- `rerank = true`
- `auto_generate_expansions = false`

embedding model、reranker model、QMD commit、dirty state、collection-scoped index fingerprint、vector completeness 和 profile hash 都必须进入 run manifest。dirty run 只能作为诊断，不能产出正式 SFT 数据。

### 9.2 同 query 的公平比较

每个 query 分别运行：

1. raw：原 query，无 teacher expansion；
2. candidate 0–3：相同原 query，加对应 expansion set。

raw 与 4 个 candidate 必须共享同一 collection、index、retrieval profile、原 query 处理和 reranking 路径。检索异常必须标记为 error，不能用全零指标代替。

保存 top 30 排名，并计算 `Recall@10`、`nDCG@10`、`MRR@10`。另外计算 Recall@20/30 作为诊断指标，但它们不参与训练目标选择。

## 10. Winner 与 Qualified Tie 选择

### 10.1 指标顺序

定义三元组：

```text
score = (Recall@10, nDCG@10, MRR@10)
```

按字典序比较：先比较 Recall@10；相同才比较 nDCG@10；仍相同才比较 MRR@10。所有值使用同一次评分程序产生的原始数值，不做显示位数舍入后再比较。

在通过 Contract、非重复检查且检索成功的 candidates 中选 score 最大者：

- best score 大于 raw score：`winner`；
- best score 与 raw score 三项完全相同：`qualified_tie`；
- best score 小于 raw score：`no_winner`，不进入 SFT；
- 没有有效 candidate：`no_valid_candidate`，不进入 SFT；
- raw 或任一 candidate 检索发生异常：对应 record 标记 `retrieval_error`，正式 scoring run 视为未完成，修复后从冻结 candidate 恢复评分；它不等价于 `no_winner`，也不能通过忽略出错 candidate 来产生训练目标。

### 10.2 Candidate 完全平局

如果多个 candidates 的三项指标完全相同，先按 `candidate_index` 排序，再使用完整 SHA-256 digest 转为无符号整数：

```text
sha256("qmd-public-v0\0seed=42\0" + source_id + "\0" + qid) mod tied_candidate_count
```

确定唯一 candidate。这里不使用“最短文本优先”或“candidate 0 优先”，避免系统性压低 expansion 多样性。每个 query 最多产生一个 SFT target。

## 11. Accepted 数据集物化

`winner + qualified_tie` 构成 accepted set，两个标签必须在 provenance record 和 SFT record 中保留。

### 11.1 数量规则

- accepted `< 1,000`：保留本次结果并输出诊断报告，但不宣称达到训练规模；不得现场补样。是否创建新冻结 batch 另行审批。
- accepted 在 `1,000–2,000`：全部保留。
- accepted `> 2,000`：按 `source_id + selection_status` 分层裁剪到 2,000。

超过 2,000 时，各层配额使用 largest-remainder：先分配 `floor(2000 * stratum_count / accepted_count)`，剩余名额按小数余数降序分配；余数相同按 stratum name 字节序。层内按 `sample_key, qid` 排序取前 N 条。

### 11.2 Train/validation 切分

最终 accepted set 按 `source_id + selection_status` 分层切为 90% train、10% validation，同样使用 largest-remainder 确定 validation 配额，层内按独立 split hash 排序：

```text
sha256("qmd-public-v0-sft-split\0seed=42\0" + input_id)
```

规范化 query 已在 pool 阶段全局去重，因此同一精确 query 不会跨 train/validation。validation 只用于观察 SFT 训练过程，不参与 teacher candidate 选择，也不替代独立 retrieval benchmark。

SFT JSONL 每行至少保存：`query`、canonical `output`、`input_id`、`source_id`、`selection_status`、`selected_candidate_index` 和 `experiment_id`。训练 loader 可以忽略 provenance 字段，但不能从原始数据中删除它们。

## 12. Artifact 布局与可恢复执行

建议的本地 artifact 布局：

```text
finetune/data/public-distill-v0/
  archives/                         # 下载文件，不提交 Git
  prepared/
    source-manifest.json
    pool-smoke.jsonl
    pool-main.jsonl
    pool-manifest.json
  experiments/<experiment_id>/
    run-manifest.json
    candidates.jsonl
    scored.jsonl
    selected.jsonl
    sft-train.jsonl
    sft-validation.jsonl
    report.json
    errors.jsonl
    index.sqlite                    # 实验专属，不提交 Git
```

每一步读取上一步的 immutable artifact，并校验记录数与 SHA-256。相同 `experiment_id` 只能 resume 缺失工作，不能覆盖已经成功的 candidate 或 scoring 记录。若配置、prompt、pool、QMD commit 或 index fingerprint 改变，必须使用新的 `experiment_id`。

## 13. 执行阶段与门禁

审批后的实施和运行按以下顺序：

1. 实现并测试多来源 prepare、通用 distill record、Recall@10-first selection 和 artifact manifest。
2. 用户显式下载或授权下载 FiQA/CQADupStack archive；校验官方 MD5。
3. prepare 三个来源、建立 isolated collections/index、冻结 30 条 smoke pool 和 2,500 条 main pool。
4. smoke：FiQA 10 + programmers 10 + unix 10，共 30 queries；每条生成 4 个 candidates，共 120 次 teacher 请求。
5. smoke 必须顺序跑通下载校验、转换、索引、候选生成、Contract v1.1、非重复检查、QMD 检索、winner/qualified-tie/no-winner 分类和 SFT 格式物化。任何阶段失败都不得启动 main。
6. smoke 只判断流水线闭环，包括零未解释异常、artifact 可 resume、collection 无串库、指标可复算；不据此调整 prompt。smoke 的物化结果必须带 `smoke_only=true`，并存放在独立 experiment 目录，不得进入最终 SFT。
7. main：只有 smoke 门禁通过后才能运行 2,500 queries × 4 candidates，最多 10,000 个成功 candidate 请求；冻结 candidates 后完成评分和选择。
8. 生成 accepted set、train/validation、质量报告和完整 provenance。
9. 数据审计通过后，才进入 completion-only SFT 实验设计。

当前本地没有 FiQA/CQADupStack 数据不构成设计阻塞；下载、解压、索引都属于 Spec 获批后的执行阶段。本轮不得提前运行这些操作。

## 14. 失败语义

必须区分以下失败，不能统一记成 rejected 或零分：

| 层级 | 状态示例 | 处理 |
|---|---|---|
| 数据准备 | archive hash mismatch、missing qrels doc | 整个 prepare 失败 |
| teacher | request error、empty response、JSON parse error | candidate 最多重试 3 次，之后记录 error |
| 合同 | contract_invalid、repeat_only | candidate 不评分或不参与选择 |
| 检索 | raw_retrieval_error、candidate_retrieval_error | record 标记 retrieval_error，正式 scoring run 未完成；修复后 resume |
| 选择 | no_winner、no_valid_candidate | 保留审计记录，不进入 SFT |
| 物化 | hash/count mismatch、duplicate input_id | 整步失败，不输出部分正式 SFT |

## 15. 报告要求

`report.json` 至少包含：

- 各来源 eligible、smoke、main、accepted 数量；
- 4-candidate 生成成功率、重试次数与错误分布；
- Contract invalid、repeat-only、retrieval error 数量；
- winner、qualified_tie、no_winner、no_valid_candidate 数量；
- accepted 中各来源与标签占比；
- raw 与 selected 的 Recall@10、nDCG@10、MRR@10 聚合差异；
- Recall@20/30 诊断结果；
- expansion 类型数量、文本长度和 Contract warnings 分布；
- prompt/pool/source/index/QMD commit/artifact hashes；
- 保留集精确碰撞排除记录；
- 最终 SFT train/validation 行数和 SHA-256。

报告不得把 accepted set 上的 oracle selection 增益写成模型增益。它只说明 qrels 能否从 teacher 候选中产生训练标签。

## 16. 兼容性与实施边界

实施时应扩展现有 benchmark/distillation 基础设施，而不是复制检索路径：

- 保留 `retrieveForBenchmark() -> executeHybridRetrieval()` 作为唯一 QMD 检索执行链。
- 将现有 SciFact 专用 distill 数据模型和 CLI 泛化为可携带 `source_id` 的版本；已有 SciFact artifacts/commands 继续可读、可运行。
- BEIR prepare 增加可配置 split 和 CQADupStack domain root，不改变 SciFact converter 的既有输出。
- 当前 SciFact 的 `Recall@30 -> MRR@10 -> nDCG@10` winner 规则不被静默改写；Public v0 通过显式 policy version 使用新的 `Recall@10 -> nDCG@10 -> MRR@10` 规则。
- 不在业务检索代码里加入 FiQA/CQADupStack 特例；数据源差异只存在于 source config/manifest。

## 17. 验收标准

实现完成需要同时满足：

1. 单元测试覆盖抽样、阈值二值化、Contract/repeat-only、三指标字典序、qualified tie、hash tie-break、分层裁剪和错误状态。
2. 集成测试证明 query 不能检索其他 source collection。
3. 同一 archive、seed、code、prompt 和 profile 两次 prepare/score 的 artifact hash 一致。
4. smoke 30 条按 `FiQA 10 + programmers 10 + unix 10` 从下载校验到 SFT 格式物化走通完整链路，且能中断续跑。
5. main pool 精确为 2,500 条，smoke 30 条与 main 的 `input_id` 零重叠；smoke 物化记录带 `smoke_only=true`，最终 SFT 中该字段的记录数为零。
6. 每个 SFT target 都能反查 source qrels、4 个 frozen candidates、raw/selected 排名和选择原因。
7. 正式产物来自 clean commit，manifest 无密钥且所有关键输入有 hash。

当且仅当 Public-only SFT 相对 raw/base model 在 untouched benchmarks 上表现出可重复的检索收益，才讨论第二阶段 Synthetic Query Tasks；Synthetic 数据的构造方法不属于本 Spec。

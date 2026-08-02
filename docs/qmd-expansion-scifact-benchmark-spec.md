# QMD Expansion SciFact Benchmark Spec

状态：设计已确认，可实施

Benchmark ID：`qmd-expansion-scifact-v1`

## 1. 目标与边界

建立独立、无后训练数据泄漏、可重复运行的 qrels benchmark，通过真实 QMD 检索链路比较：

```text
raw
current
candidate
lex-only
vec-only
hyde-only
```

第一阶段只证明评测系统可信，不要求候选模型必须提升。

第一阶段不做在线模型生成、重新标注 SciFact、其他公开数据集导入、模型训练、generation latency、UI 或评测服务。

## 2. 数据源与 qrels

唯一数据源：

```yaml
url: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
archive_md5: 5f7d1de60b170fc8027bb7898e2efca1
split: test
```

下载后必须先验证 MD5，失败时终止。

qrels 直接继承 SciFact evidence-document 语义：

- `relevance = 1` 表示摘要包含支持或反驳 claim 的证据；
- 未标注文档计分 gain 为 `0`，结果中仍标记为 `unjudged`；
- qrels 保持二元，不转换为 0–3；
- 允许一个 query 对应多个 relevant documents；
- 零正例 query 不进入 Recall、MRR、nDCG 汇总。

`source-qrels.tsv` 逐字节保留 BEIR test qrels。`qrels.tsv` 是正式评测使用的派生冻结集：它只能按 `excluded-qids.json` 整体排除 qid，不得改变 relevance label。两者都保持 BEIR 原生三列格式和表头：

```text
query-id	corpus-id	score
1	123	1
```

## 3. 冻结 artifact

```text
finetune/benchmarks/qmd-expansion-scifact-v1/
├── benchmark.yaml
├── retrieval-profile.yaml
├── index-manifest.json
├── queries.jsonl
├── source-qrels.tsv
├── qrels.tsv
├── excluded-qids.json
├── documents.jsonl
├── leakage-report.json
├── corpus/<source_doc_id>.md
├── expansions/{current,candidate}.jsonl
└── runs/
    ├── {raw,current,candidate}.json
    └── results/*.jsonl
```

Markdown 转换固定为 UTF-8/LF：

```markdown
# <title>

<text>
```

不得摘要、改写或补充原文。`documents.jsonl` 保存 collection-relative path：

```json
{"doc_id":"123","path":"123.md"}
```

QMD filepath 必须通过该映射还原为 source `doc_id`；内部 chunk 命中按文件聚合回 source document。

`queries.jsonl` 只保留派生 `qrels.tsv` 涉及的 query：

```json
{"qid":"1","query":"<SciFact claim>"}
```

`excluded-qids.json` 即使没有排除项也必须存在并写为 `[]`。有排除项时使用：

```json
[
  {
    "qid": "1",
    "reason": "confirmed_training_leakage",
    "evidence_refs": ["leakage-report.json#matches/0"]
  }
]
```

预生成 expansion：

```json
{
  "qid": "1",
  "query": "<SciFact claim>",
  "status": "ok",
  "raw_output": "hyde: ...\nlex: ...\nvec: ...",
  "output": [
    ["hyde", "<hypothetical document>"],
    ["lex", "<lexical query>"],
    ["vec", "<semantic query>"]
  ],
  "fallback_used": false,
  "error": null
}
```

每个 benchmark `qid` 在 expansion 文件中必须且只能出现一次，不能缺失或额外增加。`query` 必须与同一 `qid` 的 benchmark query 完全一致；`output` 保留模型实际输出顺序，类型只能是 `lex`、`vec`、`hyde`。

`status` 只能是 `ok`、`format_error` 或 `generation_error`。非 `ok` 记录必须保留原始输出和错误原因，并使用空 `output`；受控检索按 raw-only 执行，同时单独计入 format/generation failure，不能丢弃该 query。

## 4. 最小配置

`benchmark.yaml`：

```yaml
benchmark_id: qmd-expansion-scifact-v1
source:
  url: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
  archive_md5: 5f7d1de60b170fc8027bb7898e2efca1
  split: test
source_qrels_sha256: <source-qrels.tsv sha256>
excluded_qids_sha256: <excluded-qids.json sha256>
leakage_report_sha256: <leakage-report.json sha256>
converted_data_sha256: <deterministic aggregate hash>
qrels:
  relevant_threshold: 1
  unjudged: nonrelevant
  graded: false
cutoffs: [1, 3, 5, 10, 20, 30]
metrics: [recall_at_cutoffs, mrr_at_10, ndcg_at_10]
```

聚合 hash 只覆盖 `corpus/`、`queries.jsonl`、`qrels.tsv`、`documents.jsonl` 的相对路径和内容，不包含 source qrels、排除记录、leakage report、index、expansion 或运行结果。其余冻结文件使用各自的 SHA256。

聚合 hash 算法固定如下：

1. 收集 `corpus/` 下全部普通文件以及 `queries.jsonl`、`qrels.tsv`、`documents.jsonl`；
2. 路径统一为相对 benchmark 根目录的 UTF-8 POSIX 路径并按字节序排序；
3. 对每个文件的原始字节计算 SHA256；
4. 将每条 `"<relative-path>\\0<file-sha256>\\n"` 的 UTF-8 字节按顺序输入最终 SHA256。

实现必须提供单独的 `verify` 命令重复计算该值，禁止依赖文件修改时间、目录遍历顺序或绝对路径。

`retrieval-profile.yaml`：

```yaml
profile_id: qmd-scifact-controlled-v1
collection_name: qmd-expansion-scifact-v1
collection_root: corpus
embedding_model: <model-id>
reranker_model: <model-id-or-null>
result_limit: 30
per_list_limit: 30
candidate_limit: 40
rerank: true
auto_expand: false
strong_signal_bypass: false
```

`collection_root` 相对 benchmark 根目录解析。profile 校验必须拒绝：

- `result_limit < max(cutoffs)`；
- `per_list_limit < result_limit`；
- `candidate_limit < result_limit`；
- `auto_expand != false` 或 `strong_signal_bypass != false`。

其他检索行为由 run 记录的 QMD commit、实际 profile/config hash 和模型 artifact hash 确定。

`index-manifest.json` 由只读的 index 校验命令生成，不负责创建 collection 或 embeddings：

```json
{
  "collection_name": "qmd-expansion-scifact-v1",
  "collection_root": "<resolved-absolute-path>",
  "documents_sha256": "<documents.jsonl sha256>",
  "embedding_model": "<model-id>",
  "embedding_fingerprint": "<qmd-embedding-fingerprint>",
  "document_count": 0,
  "vector_document_count": 0,
  "vector_chunk_count": 0,
  "pending_embedding_count": 0,
  "index_fingerprint": "<collection-scoped-semantic-sha256>"
}
```

`index_fingerprint` 必须基于该 collection 的稳定语义内容计算，不得使用整个 SQLite 文件 hash 代替：

1. 通过 `documents.jsonl` 将 collection filepath 还原为 source `doc_id`；
2. document 记录按 source `doc_id` 字节序排序，编码为 `D\0<doc_id>\0<path>\0<content-hash>\n`；
3. vector 记录按 `(doc_id, seq)` 排序，embedding 统一编码为 IEEE 754 float32 little-endian bytes；
4. vector 记录编码为 `V\0<doc_id>\0<seq>\0<pos>\0<total_chunks>\0<model>\0<embedding-fingerprint>\0<embedding-bytes-sha256>\n`；
5. 将全部 UTF-8 记录按上述顺序输入最终 SHA256。

启动 benchmark 前必须重新计算该值并与 manifest 完全一致。

`run.json`：

```json
{
  "run_id": "candidate-model-a",
  "benchmark_id": "qmd-expansion-scifact-v1",
  "benchmark_manifest_sha256": "<hash>",
  "retrieval_profile": "qmd-scifact-controlled-v1",
  "retrieval_profile_sha256": "<hash>",
  "qmd_commit": "<commit>",
  "qmd_dirty": false,
  "qmd_diff_sha256": null,
  "qmd_config_sha256": "<hash>",
  "collection_name": "qmd-expansion-scifact-v1",
  "collection_root": "<resolved-absolute-path>",
  "index_manifest_sha256": "<hash>",
  "index_fingerprint": "<hash>",
  "embedding_artifact_sha256": "<hash>",
  "reranker_artifact_sha256": "<hash-or-null>",
  "variant": "candidate",
  "expansion_model": "<model-id>",
  "expansions_sha256": "<hash>",
  "retrieval": {
    "result_limit": 30,
    "per_list_limit": 30,
    "candidate_limit": 40
  },
  "command": ["qmd", "bench", "..."],
  "runtime": {
    "qmd": "<version>",
    "bun_or_node": "<version>",
    "sqlite": "<version>",
    "sqlite_vec": "<version>",
    "platform": "<os-arch>"
  },
  "status": "completed",
  "results": "runs/results/candidate-model-a.jsonl",
  "metrics": {
    "recall_at_20": 0.0,
    "recall_at_30": 0.0,
    "mrr_at_10": 0.0,
    "ndcg_at_10": 0.0,
    "expansion_pass_rate": 1.0,
    "format_error_rate": 0.0,
    "generation_error_rate": 0.0,
    "fallback_rate": 0.0
  }
}
```

`raw` 的 `expansion_model`、`expansions_sha256` 和四个 expansion rate 为 `null`。run 中的 collection、index、模型 artifact 和 retrieval 参数必须记录实际运行值，不能只复制声明值。`qmd_dirty = true` 时还必须记录 `qmd_diff_sha256`；这种 run 只用于诊断，不得作为 readiness 或 model promotion 的正式结果。

`retrieval_profile_sha256` 是 profile 文件原始字节的 SHA256。`qmd_config_sha256` 覆盖本次进程实际生效的检索相关配置和环境覆盖值，并使用 key 字节序排序、无额外空白的 canonical JSON 计算 SHA256；不得只 hash 一个可能未被加载的配置文件。

## 5. 统一检索契约

```ts
retrieveForBenchmark({
  originalQuery,
  expansions,
  collection,
  resultLimit,
  perListLimit,
  candidateLimit,
  rerank,
})
```

不变量：

1. 始终执行 original-query BM25 和 vector retrieval；
2. 所有 variant 使用相同的 original-query RRF 权重；
3. 不在线生成 expansion，不执行 strong-signal shortcut；
4. `raw` 使用空 expansions；
5. current/candidate 只允许 expansions 不同；
6. `lex` 路由到 BM25，`vec`/`hyde` 路由到 vector；
7. corpus、index、embedding、fusion、reranker和候选数量完全一致；
8. runner 必须显式指定 profile 中的唯一 collection，禁止检索全部 collection；
9. original BM25/vector 和每条 expansion retrieval 都使用相同的 `perListLimit`；
10. RRF 后只保留 `candidateLimit`，最终只返回 `resultLimit`；
11. 不改变生产搜索入口的默认行为。

不得直接比较现有 raw `hybridQuery()` 和仅含 expansions 的 `structuredSearch()`；两者是否包含 original-query 检索及其 RRF 权重不同。

实现时不得在 `src/bench/` 复制一份 hybrid pipeline。`src/store.ts` 应抽出一个供生产和 benchmark 共用的内部执行核心：

```ts
executeHybridRetrieval(store, {
  originalQuery,
  originalFts,
  expansions,
  collection,
  resultLimit,
  perListLimit,
  candidateLimit,
  rerank,
})
```

- 现有 `hybridQuery()` 继续先执行 original FTS，负责 strong-signal 判断和在线 `expandQuery()`，再调用公共执行核心；
- 新增的 benchmark seam 先执行同样的 original FTS，再直接传入预生成 expansions；
- `originalFts` 显式传入公共执行核心，避免生产路径重复执行 BM25；
- 当 expansions 为空时，公共执行核心仍执行 original-query BM25/vector；
- 公共执行核心不得继续硬编码每路 top-20；生产入口传入现有默认值，benchmark seam 传入 profile 的冻结值；
- 原有 `hybridQuery()`、`structuredSearch()` 和 `QMDStore.search()` 的公开签名及默认行为保持不变。

benchmark seam 是内部能力，不新增面向普通用户的 SDK API。

## 6. 计分与结果

标准指标：

- `Recall@1/3/5/10/20/30`；
- `MRR@10`；
- binary `nDCG@10`。

设 query 的 relevant document 集合为 \(R\)，返回排名为 \(d_1 \ldots d_k\)：

```text
Recall@K = |{d1..dK} ∩ R| / |R|

MRR@10 =
  1 / first_relevant_rank，若首个 relevant rank <= 10
  0，否则

DCG@10  = Σ(rel_i / log2(i + 1)), i = 1..10
IDCG@10 = 同一 query 的 relevant labels 按理想顺序计算
nDCG@10 = DCG@10 / IDCG@10
```

SciFact 为 binary qrels，因此 `rel_i` 只能为 `0` 或 `1`。所有汇总指标采用 query-level macro average，不按 relevant document 数量加权。

每条 `results.jsonl` 至少保存：

```json
{
  "qid": "1",
  "variant": "candidate",
  "retrieval_status": "ok",
  "expansion_status": "ok",
  "expansions": [],
  "ranking": [
    {"rank": 1, "doc_id": "123", "relevance": 1},
    {"rank": 2, "doc_id": "456", "relevance": null}
  ],
  "latency_ms": 0,
  "metrics": {
    "recall_at_20": 1.0,
    "recall_at_30": 1.0,
    "mrr_at_10": 1.0,
    "ndcg_at_10": 1.0
  },
  "fallback_used": false,
  "expansion_error": null,
  "retrieval_error": null
}
```

每条正式 ranking 最多保存 `result_limit` 条。只要 corpus 中至少存在 `result_limit` 个可检索文档，ranking 少于 `result_limit` 就必须记录诊断信息；runner 不得用补零或重复文档凑满排名。

`relevance = null` 表示 unjudged；计分 gain 为 `0`，但结果中不得改写成已标注的 `0`。

`expansion_status`、`fallback_used` 和 `expansion_error` 必须从冻结 expansion 记录原样复制；`retrieval_status` 和 `retrieval_error` 只描述本次检索执行。

backend、索引或 structured query 执行失败必须记录 `retrieval_status = "error"` 和原因，不得转成零分进入平均值。任意 query 出现执行错误时：

- 允许输出 partial results 供调试；
- `run.json.status` 必须为 `failed`；
- 不得生成可用于模型发布判断的官方汇总指标。

模型预生成阶段的 `format_error` 或 `generation_error` 不属于检索执行错误：该 query 使用空 expansions 继续检索，并通过单独的失败率反映模型问题。

## 7. 校验与兼容

冻结前必须验证：

- qrels `qid` 全部存在于 queries；
- relevant `doc_id` 全部存在于 corpus；
- `doc_id -> filepath` 一一对应且覆盖完整；
- Markdown 标题和摘要与源数据一致；
- 没有零正例 query 混入普通指标；
- QMD 结果能聚合回 source `doc_id`；
- test query 与后训练数据不存在确认的 exact/near-duplicate 泄漏。

启动每次 benchmark run 前还必须强校验：

- profile 的 `collection_name` 存在，且解析后的 root 与 `collection_root` 一致；
- collection 中 active document 集合与 `documents.jsonl` 一一对应，不缺失且不包含额外文档；
- source `doc_id -> filepath -> content hash` 映射完整、唯一且可逆；
- embedding model 和 embedding fingerprint 与 profile/index manifest 一致；
- `pending_embedding_count = 0`；
- document、vector document、vector chunk 数量与 index manifest 一致；
- 重新计算的 `index_fingerprint` 与 index manifest 一致。

任一校验失败时必须在检索前终止 run，不得输出可计分结果。

`leakage-report.json` 记录 benchmark query hash、训练数据 hash、规范化规则、exact match、near-duplicate 候选及人工确认结果。近似分数只召回疑似样本，不自动判定泄漏。

第一版泄漏检查固定为：

1. 保存原始 query；
2. 对 query 执行 Unicode NFKC；
3. 使用大小写无关、行尾锚定的 `(?:^|\s)(?:/)?only\s*:\s*(lex|vec|hyde)\s*$` 移除训练专用 suffix，同时记录匹配到的 suffix 和类型；
4. 对剩余 query 执行 lowercase、首尾裁剪和连续空白合并，得到 normalized query；
5. 对 normalized query 做 exact match；
6. 使用 token Jaccard `>= 0.80` 或 character 3-gram Jaccard `>= 0.85` 召回 near-duplicate 候选；
7. near-duplicate 候选必须人工确认是否表达同一信息需求；
8. 被确认泄漏的 test qid 写入 `excluded-qids.json`；
9. `queries.jsonl` 和 `qrels.tsv` 按该文件整体排除 qid，并重新生成所有派生冻结 hash。

阈值只用于候选召回；不能把低于阈值解释为已证明不存在语义泄漏。

`leakage-report.json` 中每条训练 query 至少保存 `raw_query`、`normalized_query`、`removed_only_suffix` 和 `only_type`。`source-qrels.tsv` 始终保持不变。只要存在排除项，本 benchmark 就是 SciFact test 的泄漏过滤派生集，不得把指标表述为完整官方 SciFact test 得分。

v2 loader 还必须拒绝：

- 重复 query ID；
- 同一 `(qid, doc_id)` 的重复或冲突 qrels；
- 重复 doc ID 或重复 filepath；
- qrels 中不存在的 query/doc ID；
- 非 `0/1` relevance；
- expansion 缺失、额外或重复 qid；
- expansion query 与 benchmark query 不一致；
- 非法 expansion 类型、空文本或多行单项文本；
- manifest 中未知 metric、非法 cutoff 或 benchmark ID 不匹配；
- collection/index manifest 缺失、过期或与实际 index 不一致；
- `result_limit`、`per_list_limit`、`candidate_limit` 不满足 profile 约束。

兼容要求：

- 现有 `qmd bench <v1-fixture.json>` 继续运行并保留 legacy 语义；
- v2 使用独立 qrels 和标准指标；
- benchmark seam 仅在显式传入预生成 expansions 时启用；
- 生产 query expansion 和检索默认行为不变。

## 8. 验收与发布门槛

Benchmark readiness：

1. archive MD5、转换和 doc-id mapping 校验通过；
2. source qrels、排除记录、派生 qrels 和 leakage report hash 校验通过；
3. collection 内容、embedding 完整性和 index fingerprint 校验通过；
4. scorer 通过人工构造的手算测试；
5. 重复运行产生一致 ranking 和指标；
6. raw/current/candidate 使用同一受控检索入口和检索深度；
7. 错误明确可见，逐 query 结果可复核；
8. leakage report 与 v1 兼容测试通过。

候选模型是否提升不影响 benchmark readiness。

Model promotion gate 单独判断：

- candidate `Recall@20/30` 优于 current；
- `MRR@10`、binary `nDCG@10` 不明显退化；
- expansion pass rate 达标；
- fallback rate 不劣于 current。

定义正式 benchmark query 数为 `N`：

```text
expansion_pass_rate =
  count(expansion_status == "ok") / N

format_error_rate =
  count(expansion_status == "format_error") / N

generation_error_rate =
  count(expansion_status == "generation_error") / N

fallback_rate =
  count(expansion_status == "ok" and fallback_used == true)
  / count(expansion_status == "ok")
```

`expansion_pass_rate` 是正式名称，不使用含义模糊的 `format_pass_rate`。format 和 generation error 必须分别报告。若没有 `expansion_status == "ok"` 的 query，`fallback_rate = null`。

`fallback_used = true` 只表示生成成功且输出可以解析，但经过类型和内容校验后没有可用 expansion，因而使用预定义降级 expansion。`format_error` 和 `generation_error` 的 `fallback_used` 必须为 `false`；非 `ok` expansion 仍按第 3 节约定使用空 expansions 执行 raw-only retrieval。retrieval execution error 只改变 `retrieval_status`，不得改写已经冻结的 expansion 状态。raw run 的 `expansion_status` 和 `fallback_used` 均为 `null`。

本 benchmark 只测 retrieval latency，不声明 generation latency。

“优于”“不明显退化”的数值阈值不属于 benchmark readiness。首次 current baseline 产生后必须另行形成 promotion policy；在此之前只能报告绝对值和逐 query delta，不能自动判定模型可发布。

## 9. v1/v2 入口与命令

现有入口保持不变：

```bash
qmd bench <v1-fixture.json>
```

v2 使用 benchmark 目录和命名 run：

```bash
# 建立 collection/embeddings 后，只读检查实际索引并写出 index-manifest.json
qmd bench finetune/benchmarks/qmd-expansion-scifact-v1 --snapshot-index

# 原始 query，不加载 expansion 文件
qmd bench finetune/benchmarks/qmd-expansion-scifact-v1 --run raw

# 自动读取 expansions/current.jsonl
qmd bench finetune/benchmarks/qmd-expansion-scifact-v1 \
  --run current \
  --expansion-model-id <current-model-id>

# 自动读取 expansions/candidate.jsonl
qmd bench finetune/benchmarks/qmd-expansion-scifact-v1 \
  --run candidate \
  --expansion-model-id <candidate-model-id>

# 使用同一 candidate expansion 文件做类型消融
qmd bench finetune/benchmarks/qmd-expansion-scifact-v1 \
  --run candidate \
  --expansion-model-id <candidate-model-id> \
  --only lex
```

`--snapshot-index` 只读取 profile 指定的 collection，校验 doc mapping 和 embedding 完整性，然后显式写出 `index-manifest.json`；不得创建 collection、生成 embedding 或修改 QMD SQLite。普通 run 只验证该 manifest，不自动更新它。

`--expansion-model-id` 只声明预生成 expansion artifact 的来源并写入 run；它不得选择、加载或调用模型。runner 只读取 `expansions/<run>.jsonl`，以文件 SHA256 固定实际输入。第一阶段不声称根据 model ID 反向证明 expansion 文件来源；若后续增加 expansion manifest，再由该参数与 manifest 做一致性校验。

`--only` 只接受 `lex`、`vec` 或 `hyde`，生成的 run ID 分别追加 `-lex-only`、`-vec-only`、`-hyde-only`。runner 自动写入 `runs/<run-id>.json` 和 `runs/results/<run-id>.jsonl`；`--json` 只控制终端输出格式，不改变 artifact。

v2 runner 只读取已存在且通过 `index-manifest.json` 校验的 QMD collection/index，不负责执行 `qmd collection add`、`qmd embed` 或修改全局 SQLite。实际建 collection 和 embeddings 的命令由使用者显式执行。runner 必须把 profile 的 `collection_name` 传入每条 BM25/vector retrieval，禁止省略 collection。

## 10. 数据准备工具

新增一个显式运行的 SciFact 转换工具，建议位置：

```text
finetune/benchmarks/prepare-scifact.ts
```

职责仅包括：

1. 下载或读取指定的 `scifact.zip`；
2. 校验固定 MD5；
3. 安全解压，拒绝绝对路径和 `..` 路径穿越；
4. 读取 corpus、queries 和 test qrels；
5. 将原始 test qrels 逐字节写为 `source-qrels.tsv`；
6. 读取冻结的 `excluded-qids.json`，整体过滤对应 qid；
7. 只保留派生 qrels 涉及的 queries；
8. 按固定 Markdown 模板写入 corpus；
9. 生成派生的 `queries.jsonl`、`qrels.tsv`、`documents.jsonl`；
10. 执行完整性校验并计算各冻结 hash；
11. 提供 `prepare` 和 `verify` 两种模式。

转换工具不得：

- 创建 QMD collection；
- 生成 embeddings；
- 调用 expansion model；
- 修改 `source-qrels.tsv`、relevance label 或选择性编辑单条 qrels；
- 自动修改 benchmark manifest 中除转换 hash 以外的已确认字段。

相同源 archive 和相同代码必须逐字节生成相同的冻结数据。

## 11. 代码改造范围

| 文件 | 变更 |
|---|---|
| `src/bench/types.ts` | 增加 v2 manifest、qrels、mapping、run 和 canonical evaluation 类型；保留 v1 类型 |
| `src/bench/qrels.ts` | 新增 v2 loader、source/derived qrels、排除记录、映射和一致性校验 |
| `src/bench/score.ts` | 保留 legacy scorer；新增标准 Recall@K、MRR@10、binary nDCG@10 |
| `src/bench/index.ts` | 校验唯一 collection、doc mapping、embedding 完整性并生成 collection-scoped index fingerprint |
| `src/bench/bench.ts` | 区分 v1 fixture 与 v2 benchmark 目录，校验 profile/index，执行命名 run并写出 artifacts |
| `src/store.ts` | 抽取生产和 benchmark 共用的 hybrid retrieval core，显式接收 collection 和三层 retrieval limit |
| `src/cli/qmd.ts` | 解析 v2 的 `--snapshot-index`、`--run`、`--expansion-model-id`、`--only`，保持旧命令兼容 |
| `finetune/benchmarks/prepare-scifact.ts` | 下载、校验、转换和验证 SciFact |
| `test/bench-score.test.ts` | 保留 legacy 测试并增加标准指标手算用例 |
| `test/bench-qrels.test.ts` | 新增 loader、mapping 和错误输入测试 |
| `test/bench-v2.test.ts` | 新增受控检索、artifact 和 v1 兼容集成测试 |
| `README.md` | 记录 v2 最小使用流程，并明确 v1 legacy 指标 |

不修改：

- `src/llm.ts` 的模型生成契约；
- 训练数据 schema 和 SFT/GRPO 代码；
- 现有公开 `QMDStore` SDK 接口；
- 现有 v1 fixture 文件格式。

## 12. 测试计划

### 12.1 指标单元测试

至少覆盖：

- 单 relevant doc 在 rank 1、rank 3、cutoff 外；
- 多 relevant docs 部分召回；
- 无 relevant doc 输入被 loader 拒绝；
- binary nDCG 的理想排序和非理想排序；
- unjudged document gain 为 0 但状态保持 `null`；
- cutoff 大于返回结果数量；
- legacy scorer 结果不变。

所有标准指标测试必须使用可手工计算的小排名，禁止用另一套评测库生成 expected value。

### 12.2 数据校验测试

至少覆盖：

- archive MD5 不匹配；
- zip 路径穿越；
- 重复/缺失 qid 和 doc ID；
- 冲突 qrels；
- 不合法 relevance；
- mapping 不唯一；
- Markdown 转换与确定性 aggregate hash；
- source qrels 被修改或派生 qrels 未按完整 qid 过滤；
- `only: lex`、`/only:lex` 及大小写/空白变体的行尾归一化；
- expansion 覆盖不完整和 query 不一致；
- expansion/format/generation/fallback rate 的分子、分母和零分母；
- profile 三层 retrieval limit 非法。

### 12.3 检索集成测试

使用临时数据库和小型 corpus 验证：

- `raw` 仍执行 original BM25/vector；
- 传入 expansions 后 original retrieval 不消失；
- `lex`、`vec`、`hyde` 路由正确；
- raw/current/candidate 使用相同 original-query 权重；
- benchmark 只检索指定 collection，其他 collection 的高分文档不会混入；
- collection 文档缺失、额外、过期或 embedding 不完整时在检索前失败；
- document 或任一 vector bytes 改变时 index fingerprint 改变；
- `result_limit = 30` 时实际保留 top 30，且每路和 candidate 深度使用 profile 值；
- `--expansion-model-id` 不加载模型或触发在线生成；
- strong-signal shortcut 和在线 expansion 未触发；
- rerank on/off 只改变指定阶段；
- backend error 使 run 失败而不是产生零分；
- 生产 `hybridQuery()` 在未使用 benchmark seam 时行为不变。

### 12.4 验证命令

实现阶段至少运行：

```bash
bun test test/bench-score.test.ts
bun test test/bench-qrels.test.ts
bun test test/bench-v2.test.ts
bun test test/eval-bm25.test.ts
```

实际 SciFact collection 创建和 embedding 命令只写入操作说明，由使用者手动执行，不在自动测试中操作用户的全局 QMD index。

## 13. 实施顺序

### Step 1：标准数据与 scorer

- 定义 canonical qid/doc-id/qrels 类型；
- 实现 v2 loader、校验和标准指标；
- 完成手算测试；
- 保持 v1 scorer 不变。

退出条件：不依赖 QMD 模型或索引即可证明所有指标和错误语义正确。

### Step 2：SciFact 转换与冻结

- 实现 archive 校验和确定性转换；
- 保留 source qrels，按冻结 exclusions 生成派生 queries/qrels；
- 生成 documents/Markdown；
- 完成 mapping、零正例和 aggregate hash 校验；
- 生成 leakage report。

退出条件：同一 archive 连续转换两次得到完全相同的 hash。

### Step 3：统一检索 seam

- 抽取共享 hybrid retrieval core；
- 接入 original query + precomputed expansions；
- 显式传入 collection、result/per-list/candidate limit；
- 验证 production default path 不变。

退出条件：受控 raw/current/candidate 仅因 expansions 不同，且旧检索回归测试通过。

### Step 4：v2 runner 与 CLI

- 支持 benchmark 目录、命名 run 和类型消融；
- 支持显式 snapshot index，普通 run 不自动改写 manifest；
- 在检索前验证 profile、collection/index manifest 和 embedding 完整性；
- 写出逐 query results 和 run manifest；
- 错误 run 不产生官方汇总。

退出条件：小型 fixture 可端到端重复运行，v1 CLI 行为保持兼容。

### Step 5：SciFact readiness

- 使用冻结 SciFact 数据建立 collection/index；
- 生成并冻结 index manifest；
- 运行 raw/current/candidate；
- 复核逐 query ranking、错误、expansion/format/generation/fallback rate；
- 确认 readiness checklist 全部通过。

退出条件：benchmark 被标记 ready；候选模型是否提升另行按 promotion gate 判断。

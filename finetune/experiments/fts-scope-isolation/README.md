# FTS 统计范围隔离实验（方案 A）

分支：`experiment/fts-scope-isolation`（基于 `2ad5c78`）。
状态：**已完成（2026-11-08）**。完整报告：[`report.html`](report.html)。

**结论**：非劣效，通过。hybrid + rerank q8 宏平均 I − S：nDCG@10 +0.0000
[−0.0004, +0.0004]，R@30 −0.0001 [−0.0003, +0.0000]；所有有效性检查都通过。
限定范围的 BM25 查询快 4–25 倍。效果小的主要原因：FTS 查询词是 AND 关系，
51–94% 的原始 query 没有任何 FTS 结果。

## 1. 问题

`documents_fts` 是所有 collection 共用的一张 FTS5 表。查询时结果按 collection 精确过滤，
但 BM25 的 IDF、avgdl 按整张表计算。我们想把语义改成「查询范围就是语料」：
只用被查询 collection 的统计打分。

上线前要回答：

- **Q1（主问题）**：改用 collection 内统计后，端到端检索质量（hybrid + rerank）
  是否**不劣于**现状？
- **Q2**：BM25 单路的变化有多大？变化来自重复语料，还是跨领域语料？
- **Q3**：strong-signal 跳过扩展（阈值 `0.85` / 分差 `0.15`）的触发率和效果会怎么变？

可复现性本身就是收益（一个 collection 的结果不再受 index 里其他 collection 影响），
所以 Q1 的判定标准是**非劣效**，不要求显著提升。

## 2. 背景事实（来自 G0 index，只读查询）

G0 index `artifacts/sft-serving-baseline/g0-20261026/abc/index/index.sqlite`
有 11 个 collection，共 256,673 个 active 文档，全部在一张 FTS 表里：

| collection | 文档数 | 备注 |
|---|---:|---|
| qmd-eval-v1-fiqa-test | 57,638 | **评测目标** |
| qmd-eval-v1-fiqa-dev | 57,638 | 与 fiqa-test **完全相同**（57,638 个 hash 重合） |
| qmd-expansion-scifact-v1 | 5,183 | **评测目标** |
| qmd-distill-public-v0-scifact-train | 5,183 | 与 scifact-v1 重合 5,164 个 hash |
| qmd-eval-v1-cqadup-android | 22,998 | **评测目标**，无重复 |
| qmd-eval-v1-cqadup-webmasters | 17,405 | 同属 CQADupStack 的另一个子论坛 |
| qmd-distill-public-v0-cqadup-unix | 47,382 | 同上 |
| qmd-distill-public-v0-cqadup-programmers | 32,176 | 同上 |
| nfcorpus-train-dev / nf-freshstack nfcorpus-test | 3,633 × 2 | 两份互相重复（3,708 个 hash 重合） |
| nf-freshstack-v1-freshstack-test | 3,804 | |

可以看出，现有评测里 FiQA 和 SciFact 的每篇文档在 FTS 表里都存了两份，
领域词的 DF 被翻倍。Android 没有重复，但混入了另外三个 CQADupStack 子论坛的统计。

## 3. 设计

### 3.1 用「复制 + 删除 collection」构造隔离 index，不重新 embed

每个隔离 index 都是 G0 index 的一份副本，然后用 `qmd collection remove`
删掉目标以外的所有 collection。

`removeCollection`（`src/store.ts:4496`）会在一个事务里完成三件事：

- 删除被删 collection 的向量分区；
- 删除它的 documents，触发 `documents_ad` 删掉对应的 FTS 行；
- 清理不再被引用的 content。

**目标 collection 的 documents、content 和向量分区都不会被改动。** 所以：

- 不用重新 embed，向量逐字节相同；
- `inspectBenchmarkIndex` 只对目标 collection 的文档和向量计算指纹，
  隔离 index 的 `index_fingerprint` 应该和共享 index **完全一致**，
  bench 自带的 manifest 校验会直接验证这一点（门槛 G1）；
- 两种 index 唯一的区别是 FTS 表里还有哪些文档。hybrid 结果的任何差异，
  都只能来自 FTS 列表（原始 query 和 lex 扩展）的打分变化。

> **不要用改 `index.yml` 再触发 `syncConfigToDb` 的办法删 collection。**
> 它只删除 `store_collections` 里的行（`src/store.ts:1524`），documents 仍是 active，
> FTS 行全部保留，等于没有隔离。必须用 `qmd collection remove`。

### 3.2 index 副本

| 代号 | 内容 | 用于 |
|---|---|---|
| `S` | G0 index 完整副本（11 个 collection） | 对照组：所有路线 |
| `I-fiqa` / `I-scifact` / `I-android` | 只保留目标 collection | 处理组：所有路线 |
| `D-fiqa` | `S` 删掉 fiqa-dev | 分解：只去掉重复语料（仅 BM25 单路） |
| `D-scifact` | `S` 删掉 scifact-train | 同上 |

Android 没有重复语料，所以不需要 `D-android`。

### 3.3 实验路线

| 路线 | 内容 | index | 作用 | 预计耗时 |
|---|---|---|---|---|
| R1 BM25 单路 | 原始 query 调用 `searchFTS`，limit 30 | S、D、I | Q2：直接衡量统计的影响；S→D→I 分解出「去重」和「去跨领域」两部分 | 几分钟 |
| R2 向量单路 | 原始 query 调用 `searchVec`，limit 30 | S、I | **对照检查**：结果必须完全相同（门槛 G2） | 约 10 分钟 |
| R3 hybrid + rerank | `qmd bench --run raw` 和 `--run g0d-<tag>`（冻结的 Q8 扩展） | S、I | **Q1 主终点** | 约 10.5 小时（12 次运行） |
| R4 strong-signal | 不固定 `originalFts`，按生产阈值判断是否触发；触发的 query 改用 `expansions = []` 重跑 hybrid | S、I | Q3：触发率，以及「贴近生产」的 q8 结果 | 取决于触发率，预计少于 1 小时 |

R3 沿用 merge-eval 的配置：rank fusion + rerank，`result_limit 30`、
`per_list_limit 30`、`candidate_limit 40`、`strong_signal_bypass: false`。

R4 的「贴近生产」结果：每个 query 如果在该 index 上触发 strong-signal，
就用不带扩展的 hybrid 结果，否则用 R3 的 q8 结果。
三个数据集的 `queries.jsonl` 都没有 `intent`，生产环境对它们都会做 strong-signal 判断。

## 4. 有效性门槛（不通过就不看结果）

| 门槛 | 检查 | 通过标准 |
|---|---|---|
| G1 | bench 运行时的 manifest 校验（S 和 I 用**同一份** `index-manifest.json`） | 两组都通过，`index_fingerprint` 相同 |
| G2 | R2 向量单路结果 | S 与 I 每个 query 的 doc_id 列表完全相同 |
| G3 | 隔离后 FTS 行数 | `I-x` 的 `documents_fts` 行数 = 目标 collection 的 active 文档数 |
| G4 | 删除后 BM25 统计正确 | 对 `I-x` 抽 50 个 query，`searchFTS` 的分数和「从同一批文档新建的 FTS5 表」的 `bm25()` 分数一致（误差 < 1e-9） |
| G5 | 缓存状态一致 | 所有副本执行同样的 `qmd cleanup`（清空 `llm_cache`、FTS optimize、VACUUM），再开始运行 |

G4 用来排除 FTS5 删除标记（tombstone）没合并导致统计偏差的可能。
G5 是因为 G0 index 的 `llm_cache` 里还有 1,034 条旧的 rerank 缓存，只有 S 保留缓存会形成混杂因素。

## 5. 指标与统计

- 指标：nDCG@10、R@30 为主；MRR@10、R@10 为辅。和之前的评测一致。
- 方法：配对、按数据集分层的 bootstrap，20,000 次重采样，seed 42，
  计算三个数据集的宏平均和每个数据集的结果。复用 `merge-eval-20261106/bootstrap-analysis.py` 的写法。
- 比较：`I − S`，分别对 raw 和 q8 两组计算；R1 另外计算 `D − S` 和 `I − D`。

### 判定规则（运行前确定）

| 结果（R3，q8 组，宏平均） | 结论 |
|---|---|
| nDCG@10 的 95% CI 下界 > **−0.005**，且 R@30 下界 > **−0.01** | **非劣效**：采纳「collection 内统计」，进入方案 C 的设计 |
| 上述成立，并且某个指标 CI 整体 > 0 | 有显著提升，结论更强 |
| 任一下界低于阈值 | 不采纳；用 R1 的分解和每个数据集的结果找原因（比如小语料的 IDF 截断、重复语料） |

raw 组和 R4 只作为辅助证据，不参与判定。
R4 如果显示 strong-signal 触发率明显变化，方案 C 上线时需要重新标定这两个阈值。

## 6. 步骤

所有命令都同时设置 `INDEX_PATH` 和独立的 `QMD_CONFIG_DIR`。
从源码运行：`node_modules/.bin/tsx src/cli/qmd.ts`（本机没装 Bun）。

```sh
X=artifacts/fts-scope-isolation
SRC=artifacts/sft-serving-baseline/g0-20261026/abc/index/index.sqlite
CFG=artifacts/sft-serving-baseline/merge-eval-20261106/config/index.yml
QMD="node_modules/.bin/tsx src/cli/qmd.ts"

# 1) 副本和配置（每个副本 1.7 GB）
for v in S I-fiqa I-scifact I-android D-fiqa D-scifact; do
  mkdir -p $X/index/$v $X/config/$v
  cp $SRC $X/index/$v/index.sqlite
  cp $CFG $X/config/$v/index.yml
done

# 2) 删除 collection（以 I-fiqa 为例；I-* 保留目标，D-* 只删重复的那个）
for c in <除 qmd-eval-v1-fiqa-test 以外的 10 个 collection>; do
  INDEX_PATH=$X/index/I-fiqa/index.sqlite QMD_CONFIG_DIR=$X/config/I-fiqa $QMD collection remove $c
done

# 3) 所有副本（包括 S）执行同样的 cleanup（门槛 G5）
for v in S I-fiqa I-scifact I-android D-fiqa D-scifact; do
  INDEX_PATH=$X/index/$v/index.sqlite QMD_CONFIG_DIR=$X/config/$v $QMD cleanup
done

# 4) bench 目录：从 merge-eval 复制，不带 runs/，S 和 I 各一份
#    $X/bench/{S,I}/<dataset>/，包含同一份 index-manifest.json 和 expansions/g0d-<tag>.jsonl

# 5) R3：按数据集交错运行，S-raw → I-raw → S-q8 → I-q8，平衡运行期间的系统状态变化
INDEX_PATH=$X/index/S/index.sqlite QMD_CONFIG_DIR=$X/config/S \
  $QMD bench $X/bench/S/qmd-eval-v1-fiqa-test --run raw
INDEX_PATH=$X/index/S/index.sqlite QMD_CONFIG_DIR=$X/config/S \
  $QMD bench $X/bench/S/qmd-eval-v1-fiqa-test --run g0d-fiqa --model local-d-q8-nosys-greedy
# ……I-fiqa 用 $X/bench/I/…；scifact、android 同理
```

R1、R2、R4 和门槛检查需要一个一次性的 `probe.ts`，
用 `createInternalStore` 加上 `searchFTS`、`searchVec`、`executeHybridRetrieval`，
输出和 bench 相同的 per-query JSONL（qid、ranking[doc_id]、retrieval_status），
供 `analyze.py` 统一计算指标。两个脚本放在本目录。

## 7. 控制的混杂因素

- **代码**：所有运行用同一个 commit。S 组也重新跑，不复用 merge-eval（`e5e4527`）的结果。
- **缓存**：G5。同一 index 内 raw 先于 q8 运行，rerank 缓存最多 1,000 条，S 和 I 的运行顺序相同，影响是对称的。
- **GPU 不确定性**：llama.cpp 的 rerank 有轻微不确定性。交错运行，并把 S 组内的重复性当作噪声基线：可选地对 S-raw/scifact 跑两次，统计结果不一致的比例。
- **系统休眠**：bench 已经设置 `inactivityTimeoutMs: 0`（`f64c287`）。长时间运行时请接上电源。
- **配置污染**：每个副本有自己的 `QMD_CONFIG_DIR`，不碰全局的 `~/.config/qmd/index.yml`。

## 8. 资源

- 磁盘：6 个副本约 10 GB（cleanup 后 I/D 副本会缩小），本机剩余 88 GB。
- 时间：R3 共 12 次运行，按 merge-eval 的耗时（FiQA 60/82 分钟、SciFact 30/38 分钟、Android 45/62 分钟）估算，每组 index 约 5.3 小时，合计约 10.5 小时，适合过夜跑。R1、R2、R4 加起来不到 1.5 小时。

## 9. 产出

- `artifacts/fts-scope-isolation/`（不进 git）：index 副本、配置、bench 目录、per-query 结果、日志。
- 本目录：`README.md`（本文件，执行后补结果）、`probe.ts`、`analyze.py`。
- 结论写入记忆，并更新 `docs/vector-partitions.html` 中关于 FTS 的部分。

## 10. 待确认

1. 非劣效阈值：nDCG@10 −0.005 / R@30 −0.01 是否合适。
2. S 组是否重新跑（多花约 5.3 小时，推荐），还是复用 merge-eval 的结果，只做一次可复现性抽查。
3. 是否需要 R4（strong-signal）。不做的话可以省掉 `executeHybridRetrieval` 那部分脚本。

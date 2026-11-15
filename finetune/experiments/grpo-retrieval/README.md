# GRPO 奖励环境（Phase 1）Runbook

目标：在目标机（AutoDL 4090）上部署一个**正确 + 够快**的检索奖励服务，作为 GRPO 的奖励打分基础。本阶段不训练、不改 qmd 检索逻辑。

对应 spec：`docs/specs/2026-11-10-grpo-reward-environment.md`。

## 1. 冻结环境（与 spec 一致）

| 项 | 值 |
|---|---|
| FTS 语义 | AND（不设 `QMD_FTS_OPERATOR=OR`） |
| 扩展格式 | V-H：`vec:` + `hyde:`，无 `lex:` |
| 检索路径 | A 路径，`rerank=false` |
| limits | result 30 / per_list 30 / candidate 40 |
| 融合 | RRF + rank fusion（沿用 `retrieveForBenchmark`） |
| embedding | `embeddinggemma-300M-Q8_0`（fingerprint `c37385`） |

奖励服务只回 `metrics` / `raw_metrics` / `status`；reward 公式（减 raw 基线、格式/长度惩罚）在 Phase 2 的客户端 `reward_func`。

## 2. 语料映射结论（与 spec §3「共用语料，共用 index」一致）

对 G0 封存 index（`artifacts/sft-serving-baseline/g0-20261026/abc/index/index.sqlite`）做了逐 collection 指纹核对，发现：

1. **FiQA 训练/评测共语料成立**：`qmd-eval-v1-fiqa-test` 与 fiqa-train 的 `documents_sha256`、`index_fingerprint` 逐位一致（`c3bf1b24…` / `695d1ee3…`）。所以 fiqa 用一个 collection 即可，fiqa-train 的 benchmark 已统一 `collection_name` 到 `qmd-eval-v1-fiqa-test`（见 §5）。

2. **SciFact 训练/评测只有标题空白差异**：`qmd-expansion-scifact-v1`（评测）与 `qmd-distill-public-v0-scifact-train`（训练）doc_id 相同（5183），仅 **19/5183 篇文档的标题行空白不同**（评测侧标题保留行首/行尾空格，训练侧 `distill prepare` 做过空白归一化）。非空白字符逐位一致，FTS token 与 embedding 基本等价——不是数据损坏，只是指纹不同（`725dc71d…` vs `7b3c6de1…`）。所以 scifact **合并为一个 collection，统一用归一化后的内容**（`qmd-distill-public-v0-scifact-train`），scifact-test 的 benchmark 已统一 `collection_name` 过去（见 §5）。

**结论**：5 个语料 → **5 个单语料 index 文件**（fiqa / scifact / programmers / unix / android，每文件一个 collection）。fiqa-train+test 共用一个 fiqa index，scifact-train+test 共用一个 scifact index——这正是 spec §3「共用语料，共用 index」。7 个 source 按 `collection_name` 路由到这 5 个 index（§3）。

## 3. index / collection / source 映射

| source_id | benchmark 目录 | collection（index 内） | 语料 | docs |
|---|---|---|---|---|
| `fiqa-train` | `finetune/data/public-distill-v0/prepared/fiqa-train` | `qmd-eval-v1-fiqa-test` | FiQA | 57,638 |
| `fiqa-test` | `artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-eval-v1-fiqa-test` | `qmd-eval-v1-fiqa-test` | FiQA | 57,638 |
| `scifact-train` | `finetune/data/public-distill-v0/prepared/qmd-distill-public-v0-scifact-train` | `qmd-distill-public-v0-scifact-train` | SciFact | 5,183 |
| `scifact-test` | `artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-expansion-scifact-v1` | `qmd-distill-public-v0-scifact-train` | SciFact | 5,183 |
| `programmers` | `finetune/data/public-distill-v0/prepared/cqadup-programmers` | `qmd-distill-public-v0-cqadup-programmers` | CQADup | 32,176 |
| `unix` | `finetune/data/public-distill-v0/prepared/cqadup-unix` | `qmd-distill-public-v0-cqadup-unix` | CQADup | 47,382 |
| `android` | `artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-eval-v1-cqadup-android` | `qmd-eval-v1-cqadup-android` | CQADup | 22,998 |

G0 index 内 11 个 collection，另有 6 个本阶段不用：`qmd-distill-public-v0-nfcorpus-train-dev`、`qmd-eval-v1-cqadup-webmasters`、`qmd-eval-v1-fiqa-dev`（fiqa-test 的重复语料）、`qmd-nf-freshstack-v1-freshstack-test`、`qmd-nf-freshstack-v1-nfcorpus-test`、`qmd-expansion-scifact-v1`（scifact 评测侧原始语料，已并入归一化 collection）。

## 4. 组件与代码状态

| 文件 | 状态 |
|---|---|
| `reward-server.ts` | ✅ 已从 `archive/completion-only-sft-20261104` 取回，`tsc -p tsconfig.json --noEmit` 0 错误 |
| `reward_client.py` | ✅ 同上 |
| `verify_p0.py` | ✅ 同上（`--limit N` 支持采样 replay） |

安全要点（已核实）：`createStore` 不触发 `syncConfigToDb`（只有 CLI 的 `getStore` 会），所以奖励服务直接 `createStore(db)` 打开非全局 index **安全**。只有用 `qmd collection remove` 这类 CLI 命令时，才需要 `INDEX_PATH` + 隔离 `QMD_CONFIG_DIR`（见 §6）。

## 5. 已完成的本地改动

- `finetune/data/public-distill-v0/prepared/fiqa-train/retrieval-profile.yaml`：`collection_name` → `qmd-eval-v1-fiqa-test`
- `finetune/data/public-distill-v0/prepared/fiqa-train/index-manifest.json`：`collection_name` → `qmd-eval-v1-fiqa-test`
- `artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-expansion-scifact-v1/retrieval-profile.yaml`：`collection_name` → `qmd-distill-public-v0-scifact-train`、`collection_root` → 归一化 corpus 路径
- `artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-expansion-scifact-v1/index-manifest.json`：`collection_name`/`collection_root` 同上、`index_fingerprint` → `7b3c6de1…`（`documents_sha256` 保持 `32ca44b7…`，该 benchmark 目录自己的 `documents.jsonl` 未动）

（这些是 gitignored 数据。fiqa-train 改 2 个字符串、其余字段与评测侧逐位一致；scifact-test 的 `index_fingerprint` 改成归一化 collection 的指纹。`inspectBenchmarkIndex` 复核应通过；若本地不复核，§8 用 `--check-index` 重生成确认。）

## 6. 启动奖励服务（5 个单语料 index + 7 个 source）

```sh
cd qmd   # 仓库根
node_modules/.bin/tsx finetune/experiments/grpo-retrieval/reward-server.ts \
  --db qmd-eval-v1-fiqa-test=artifacts/grpo-retrieval/index/fiqa/index.sqlite \
  --db qmd-distill-public-v0-scifact-train=artifacts/grpo-retrieval/index/scifact/index.sqlite \
  --db qmd-distill-public-v0-cqadup-programmers=artifacts/grpo-retrieval/index/programmers/index.sqlite \
  --db qmd-distill-public-v0-cqadup-unix=artifacts/grpo-retrieval/index/unix/index.sqlite \
  --db qmd-eval-v1-cqadup-android=artifacts/grpo-retrieval/index/android/index.sqlite \
  --source fiqa-train=finetune/data/public-distill-v0/prepared/fiqa-train \
  --source fiqa-test=artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-eval-v1-fiqa-test \
  --source scifact-train=finetune/data/public-distill-v0/prepared/qmd-distill-public-v0-scifact-train \
  --source scifact-test=artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-expansion-scifact-v1 \
  --source programmers=finetune/data/public-distill-v0/prepared/cqadup-programmers \
  --source unix=finetune/data/public-distill-v0/prepared/cqadup-unix \
  --source android=artifacts/sft-serving-baseline/g0-20261026/abc/bench/qmd-eval-v1-cqadup-android \
  --port 8787
```

`--db` 是 `collection=<index.sqlite>`（每个 source 按其 `collection_name` 路由到对应 store）。`GET /health` 确认 7 个 source、`retrieval.rerank=false`、raw 缓存为空。

> ⚠️ 第一次 `createStore` 打开旧 vector 布局的 index 时会触发**一次向前迁移**（partitioned vector 布局升级 + vacuum，实测 G0 index 1.77GB→1.9GB、~68s），原地改 index 文件；指纹不变（`inspectBenchmarkIndex` 仍能 7/7 复核通过）。本地 smoke 已把 G0 index 迁移掉，之后 `setup-index.sh` 复制的就是已迁移版本，AutoDL 不会重复迁移。

## 7. index 构建（验收用，spec 第 3 步）

```sh
finetune/experiments/grpo-retrieval/setup-index.sh
```

产出 `artifacts/grpo-retrieval/index/<name>/index.sqlite`（5 个单语料 index：fiqa / scifact / programmers / unix / android，每文件一个 collection）。脚本内所有 `qmd collection remove` 都带 `INDEX_PATH` + **每个 index 独立的** `QMD_CONFIG_DIR`（`qmd collection remove` 会改写它读到的 config、`syncConfigToDb` 会删掉 config 里没有的 collection，所以每个 index 必须用独立 config 目录，从全量 11-collection 配置复制而来）。

## 8. 正确性验收（采样 replay）

奖励服务走 path A（`rerank=false`），所以 oracle 必须也是 path A，**不能**用 G0 历史 rerank 数字（spec §5.1 已声明）。

1. 造 path-A benchmark 目录：复制评测 benchmark 目录，把 `retrieval-profile.yaml` 的 `rerank` 改为 `false`（保留 `index-manifest.json`）。
2. 生成 path-A oracle：

   ```sh
   INDEX_PATH=<index> QMD_CONFIG_DIR=<config> \
     node_modules/.bin/tsx src/cli/qmd.ts bench <pathA-dir> --run raw
   INDEX_PATH=<index> QMD_CONFIG_DIR=<config> \
     node_modules/.bin/tsx src/cli/qmd.ts bench <pathA-dir> --run <tag> --model local-d-q8-nosys-greedy
   ```

   （`<tag>` 对应 frozen 扩展 `expansions/g0d-<dataset>.jsonl`。）

3. 采样 replay（先小样本，再按需放大）：

   ```sh
   python3 finetune/experiments/grpo-retrieval/verify_p0.py \
     --source fiqa-test \
     --expansions <pathA-dir>/expansions/g0d-fiqa.jsonl \
     --results  <pathA-dir>/runs/results/<fiqa-raw-run>.jsonl \
     --raw-results <pathA-dir>/runs/results/raw.jsonl \
     --limit 20 --batch 8
   ```

   对 scifact-test / android 同理（`--source scifact-test` / `--source android`）。

4. 训练集 smoke：`programmers` / `unix` / `scifact-train` 三个 source 能被 `/health` 报出、`/score` 对 raw query 打分不报错、`metrics` 与 path-A bench raw 一致。

5. format_error 注入：构造截断 / 非法 `vec:` 行，确认 `status=format_error` 且 `metrics` 回退到 `raw_metrics`。

## 9. 速度验收

- 用 SFT 采样出的真实 V-H 扩展，p50 ≤ 0.3 秒 / completion。
- 达标即停；不达标按 spec §5.2 阶梯追加（多进程 → 服务内缓存 → 最后手段改检索代码并重做全部验收）。

## 10. 部署到 AutoDL

- 传 `src/` + `finetune/experiments/grpo-retrieval/` + 5 个单语料 index（`artifacts/grpo-retrieval/index/<name>/index.sqlite`）+ 7 个 benchmark 目录 + 模型（embeddinggemma-300M-Q8_0）。
- AutoDL 装 CUDA 版 node-llama-cpp；服务以 `disposeModelsOnInactivity:false` 长时运行（代码已写死）。
- 上传通道（scp 或已有通道）待定，见 spec §8 待确认第 2 条。

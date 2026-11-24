# 官方查询扩展模型评测：tobil/qmd-query-expansion-1.7B vs 我们的 v2-VH SFT

| 项目 | 内容 |
|---|---|
| 日期 | 2026-11-23 → 11-24 |
| 分支 | `experiment/official-model-eval`（从 main `2ad5c78` 切出） |
| 被测模型 | `tobil/qmd-query-expansion-1.7B-gguf` @ `7816de0`（qmd 的默认 generate 模型） |
| 对照 | 我们的 v2-VH SFT final-adapter，Q8_0，G0 profile D（无 system prompt、greedy） |
| 代码 | 本目录脚本；qmd `src/` 与 main 一致 |
| 数据与结果 | `artifacts/official-model-eval/`（本机，gitignored）；分析输出在 `analysis/` |

## 结论

**官方模型的扩展让排序变差，我们的 SFT 在同一套评测上全面不低于它，多数指标显著更好。**

1. **线上路径（rank fusion + rerank），主评测集 1647 条**：官方模型比不扩展 nDCG@10 低 1.0pp（显著），只有 R@30 +0.9pp（显著）。我们的 SFT 比官方 nDCG@10 高 1.4pp、R@10 高 1.7pp（都显著），R@30 高 0.7pp（不显著）。
2. **两个模型都没训练过的集合（Webmasters + FreshStack，607 条）**：线上路径上 SFT 比官方 nDCG@10 +0.6pp、R@30 +2.4pp，都不显著（样本少，CI 宽）。不 rerank 的 path A 上两项都显著（+1.8pp、+3.1pp）。所以 SFT 的优势不只来自训练分布。
3. **官方模型的问题出在模型本身，不在部署配置。** 用 Q8、贴合训练格式的最好配置，和 qmd 线上默认配置（Q4、默认 system prompt、T=0.7）相比，所有指标都没有显著差别。

## 1. 设置

**评测集和检索**：与 SFT 历次评测相同。主集 FiQA test 648 / SciFact 300 / CQADup Android 699；分布外 CQADup Webmasters 506 / FreshStack test 101。使用 G0 index 副本（已迁移到分区向量布局），`INDEX_PATH` + 隔离的 `QMD_CONFIG_DIR`。两条检索路径：

- **rr**（线上路径）：rank fusion + Qwen3-Reranker-0.6B，candidate_limit 40。
- **norr**（path A）：同一路径但关闭 rerank。最初用作快速筛查，最后全部跑完。

**统计**：每条 query 的 nDCG@10 / R@10 / R@30 由 rankings + qrels 重新计算，按数据集分层做配对 bootstrap（20k 次，seed 42），宏平均为各数据集均值的平均。

**Arms**

| Arm | 生成方式 | 回答的问题 |
|---|---|---|
| raw | 不扩展 | 对照 |
| sft | 我们的 SFT Q8，profile D；主集复用冻结的 `g0d-*` 扩展，分布外集新生成（seed 42） | 我们的最好配置 |
| offq8 | 官方 Q8_0，训练格式 prompt、无 system turn，预填它自己会输出的空 think 块，greedy，lex/vec/hyde 行文法（末行换行可选） | 官方模型的最好配置 |
| offvh | offq8 去掉 lex 行（不重新生成） | 只比 vec+hyde 的质量；lex 有没有用 |
| offprod | 官方 Q4_K_M，走 qmd 线上默认路径（`finetune/benchmarks/expand.ts --model hf:...`） | 用户现在实际拿到的效果 |

**可复现性检查**：rr 路径的 raw 在 FiQA（648/648）和 SciFact（300/300）上 top-30 排名与 merge-eval 完全相同，SFT 在 FiQA 上也是 648/648 相同。因此 SciFact/Android 的 raw 和 sft 两个 rr arm 直接复用了 merge-eval 的结果（扩展文件逐字节相同）。path A 的 raw/sft 与 G0 step 3 数值一致（FiQA 0.3931 vs 0.3930，SciFact 完全相同）。

## 2. 结果

### 2.1 线上路径（rr）

各 arm 绝对值（nDCG@10 / R@30）：

| arm | FiQA | SciFact | Android | Webmasters | FreshStack* |
|---|---|---|---|---|---|
| raw | 0.4654 / 0.6242 | 0.7645 / 0.9356 | 0.5334 / 0.7569 | 0.4510 / 0.6729 | 0.2283 / 0.3418 |
| sft | **0.4681 / 0.6558** | **0.7692** / 0.9389 | **0.5366 / 0.7695** | **0.4520 / 0.6990** | 0.2521 / **0.3947** |
| offq8 | 0.4513 / 0.6317 | 0.7524 / **0.9489** | 0.5284 / 0.7632 | 0.4456 / 0.6771 | 0.2466 / 0.3695 |
| offvh | 0.4558 / 0.6469 | 0.7462 / 0.9389 | 0.5363 / 0.7689 | 0.4486 / 0.6832 | **0.2541** / 0.3845 |
| offprod | 0.4520 / 0.6363 | 0.7546 / **0.9489** | 0.5303 / 0.7591 | 0.4492 / 0.6804 | 0.2343 / 0.3384 |

\* FreshStack 的 query 000549 在所有 arm（含 raw）上都因 reranker 上下文溢出失败（见 §3.3），配对比较时剔除，n=100。

宏平均差值，95% CI（**粗体**为显著）：

| 比较 | 主集 nDCG@10 | 主集 R@10 | 主集 R@30 | 分布外 nDCG@10 | 分布外 R@30 |
|---|---|---|---|---|---|
| offq8 − raw | **−0.0104** [−0.017, −0.004] | **−0.0098** | **+0.0090** [+0.003, +0.016] | +0.0064 | +0.0159 |
| offprod − raw | **−0.0088** [−0.015, −0.002] | −0.0018 | **+0.0092** | +0.0021 | +0.0021 |
| sft − raw | +0.0036 | +0.0075 | **+0.0159** [+0.009, +0.023] | **+0.0124** | **+0.0395** |
| **sft − offq8** | **+0.0140** [+0.007, +0.021] | **+0.0173** | +0.0068 [−0.001, +0.015] | +0.0059 | +0.0235 [−0.002, +0.049] |
| sft − offvh | **+0.0119** | **+0.0156** | +0.0032 | +0.0007 | +0.0130 |
| offvh − offq8 | +0.0021 | +0.0017 | +0.0036 | **+0.0053** | **+0.0105** |
| offq8 − offprod | −0.0016 | −0.0080 | −0.0002 | +0.0043 | +0.0139 |

### 2.2 path A（不 rerank）

| 比较 | 主集 nDCG@10 | 主集 R@30 | 分布外 nDCG@10 | 分布外 R@30 |
|---|---|---|---|---|
| offq8 − raw | **−0.0217** | −0.0065 | −0.0101 | −0.0028 |
| offprod − raw | **−0.0263** | −0.0050 | −0.0149 | −0.0191 |
| sft − raw | **+0.0104** | **+0.0114** | +0.0078 | **+0.0280** |
| **sft − offq8** | **+0.0321** | **+0.0179** | **+0.0179** | **+0.0308** |
| offvh − offq8 | **+0.0143** | **+0.0085** | **+0.0092** | +0.0103 |

没有 reranker 兜底时，官方模型对 top-10 的伤害更大：SciFact nDCG@10 −4.6pp（offq8）/ −4.8pp（offprod）。完整输出见 `analysis/analysis-{rr,norr}.txt`。

## 3. 发现

### 3.1 官方模型的输出是退化的

2254 条 query 汇总（`analysis/expansion-stats.txt`）：

| arm | hyde 词数 p50 | 模板化 hyde | lex 行/query | lex 是自身某条 vec 的前缀 |
|---|---|---|---|---|
| sft | 109 | 0% | 0 | — |
| offq8 | 21 | 54% | 2.0 | **98%** |
| offprod | 25 | 47% | 2.0 | **99%** |

典型输出（FiQA，offq8）：

```
lex: How can individual
lex: Using google for
vec: How can individual investors get stock information from google?
vec: Using google for financial data inquiries by individual investors
hyde: The topic of Do individual investors use Google to obtain stock quotes? covers Using google for financial data inquiries by individual investors.
```

lex 行是 vec 行的前 2–3 个词（"What is the"、"How can i"），hyde 是"The topic of {query} covers {vec}"/"{query} is an important concept that relates to {vec}"之类的模板。即使输入官方训练集里的原句（"ancient Greece democracy Athens"），模型也不复现训练目标，而是输出同样的模板。HF 上的训练集（`tobil/qmd-query-expansion-train`）质量明显更高，所以发布的 GGUF 很可能不是用这份数据直接 SFT 出来的。这种"拼接 vec 前缀凑 lex、套模板凑 hyde"的模式符合基于规则奖励的 GRPO 出现 reward hacking 的特征，但**这一点没有验证**。

这也解释了 §2 的数字：lex 行在 FTS（AND）下匹配大量含停用词的文档，把噪声带进 RRF（去掉 lex 后 path A 主集 nDCG@10 +1.4pp）；短模板 hyde 几乎没有在 query 之外增加信息，而它的 vec 行（query 的同义改写）带来了一点深层召回，所以 rr 路径上 R@30 是正的。

### 3.2 qmd 默认文法会让官方模型泄漏 `</think>`

不加文法时，官方模型先输出空 think 块 `<think>\n\n</think>\n\n`，最后一行不带换行，直接结束。qmd 默认文法（`src/llm.ts` 的 `line ::= type ": " content "\n"`）要求**每行**都以换行结尾，模型无法在 hyde 后直接停止，就把 `</think>` token 当作正文输出。线上路径（offprod）中 **66% 的扩展**的 hyde 行以 `</think>` 结尾。

- 渲染的 prompt 与训练格式逐字一致（node-llama-cpp Qwen wrapper，已用 `probe-template.ts` 核对），问题只在文法。
- 把末行换行改为可选（`root ::= line ("\n" line)* "\n"?`）后泄漏为 0/20，输出与无文法时一致（`probe-nogrammar.ts`）。我们的 VH 文法本来就把末行换行设为可选，所以 SFT 不受影响。
- 这个泄漏对检索的单独影响没有拆出来：offprod 和 offq8 同时差在量化、system prompt、采样和泄漏几项上，合计差别不显著。

### 3.3 rerank 路径的上下文溢出

FreshStack query 000549（很长的 StackOverflow 式问题）在所有 arm 上都报 "input lengths of some of the given documents exceed the context size … at least 4981"。说明 rerank 截断预算对超长 query 不成立，与扩展无关。

## 4. 局限

- **主集对 SFT 有分布内优势**：SFT 的蒸馏数据包含 fiqa-train、scifact-train、cqadup programmers/unix。官方本地训练数据（`finetune/data/qmd_expansion_*.jsonl`，2718 条）与 1647 条评测 query 精确匹配 0 条。分布外集合用来补这一点，但只有 607 条，rr 路径上 CI 偏宽。
- **offprod 只有一次无种子采样**（线上路径 T=0.7、不传 seed），没有估计采样方差。
- **offq8 的"最好配置"是我们定义的**：预填空 think 块 + 末行换行可选，依据是模型不加文法时的自然输出。
- 没有评测 HF 上的 f16，以及 `-sft` / `-grpo` 中间 checkpoint。

## 5. 对 qmd 的含义

- 默认 generate 模型目前在这套评测上**对排序是负贡献**：线上 rerank 路径 nDCG@10 −0.9pp，FTS-only/vsearch 这类不 rerank 的路径伤害更大。
- 我们的 v2-VH SFT（Q8 + profile D）是现成的替代：同样 1.7B，主集 R@30 +1.6pp、分布外 R@30 +4.0pp（都显著），排序不退化。
- 可以独立修的小问题（未改代码）：§3.2 默认文法的末行换行；§3.3 超长 query 的 rerank 截断。

## 6. 复现

```sh
# 生成（官方最好配置 / 线上配置 / SFT 分布外），写入 bench-rr/*/expansions/
finetune/experiments/official-model-eval/gen-lane.sh
# 检索：rr 和 norr 各一组；多条 lane 共享同一列表，用 runs/.claims/ 原子认领
finetune/experiments/official-model-eval/retrieve-lane-claim.sh bench-rr  <tag:arm> ...
finetune/experiments/official-model-eval/retrieve-lane-claim.sh bench-norr <tag:arm> ...
# 分析
finetune/.venv/bin/python finetune/experiments/official-model-eval/analyze.py artifacts/official-model-eval/bench-rr
```

用时：生成约 2–3 s/query；rr 检索约 7–11 s/query（FreshStack 约 20 s），norr 约 1 s/query。三条 lane 并行，总计约 17 小时（Mac，接电源）。

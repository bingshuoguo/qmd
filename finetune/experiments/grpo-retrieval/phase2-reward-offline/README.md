# GRPO Phase 2 reward_func：草案与离线分布验证（2026-11-15）

目的：在训练前确认 Phase 2 的 reward 设计有可用的组内信号，并据此定 GRPO 配置。本实验不训练、不改 `src/`。

数据产物（gitignored）：`artifacts/grpo-retrieval/phase2-reward-offline/`。

## 1. reward 草案

```
status == format_error（含截断）:  R = −1
status == ok:                      R = ΔM − P_len

ΔM    = M(metrics) − M(raw_metrics)，M = 0.7·nDCG@10 + 0.3·Recall@30   ∈ [0,1]
P_len = 0.2 · clip((n_tokens − 320) / 320, 0, 1)
truncated = eos_token_id not in completion_ids
```

- **减 raw 不影响梯度**：GRPO 组内同 qid，`M_raw` 是常数，减均值后抵消。它只提供日志零点和格式惩罚的参照。
- **格式惩罚 −1**：format_error 会回退到 raw，所以 ΔM=0。不罚的话，故意写错格式就成了零风险选项。合法扩展一定有 ΔM ≥ −M_raw ≥ −1，因此 −1 能保证 format_error 不优于任何合法输出。
- **P_len** 只是护栏：320 约等于 SFT 长度 p97，正常输出基本碰不到。
- **0.7/0.3 混合**：R@30 用来覆盖线上 path B 的 rerank 候选池收益。和只用 nDCG 相比，零方差组少 3–6pp（FiQA/Android）。

## 2. 实验 A：SFT Q4 三个 seed（全量 1647 qid，离线，无检索）

脚本 `offline_reward.py`，输入为 `v2-vh-sft-serving-v1-exec-20261023` 的 Q4 seed42/43/44 扩展及其 bench 结果。采样用 serving profile：T=0.7、top-k 20、top-p 0.8、presence 0.5、grammar 约束。每个 qid 的 3 个 seed 视为一组 G=3。该 bench 跑在旧的混合 index 上。

| | ΔM 均值 | sd | ΔM>0 / <0 | G=3 零方差组 |
|---|---|---|---|---|
| FiQA | +0.003 | 0.127 | 28% / 30% | 37% |
| SciFact | −0.002 | 0.097 | 9% / 14% | 70% |
| Android | +0.007 | 0.126 | 23% / 20% | 49% |

- **格式错误**：3/4941，全部是 601 token 截断。合法扩展的最差 ΔM 为 −0.511，与 −1 之间有 0.49 的余量。
- **batch vs group 缩放**：组内极差 <0.02 的组，group 缩放后 max|A|=1.38，batch 缩放后为 0.07。group 缩放会把名次噪声放大成满幅 advantage，**用 `scale_rewards="batch"`**。
- **格式错误与 batch 缩放叠加**：batch reward sd 只有 0.124，一条 −1 对应的 advantage 达到 −5.4，约为正常值的 4 倍。预案是把 advantage 裁到 ±3，或把惩罚改为 −0.6。由于 SFT 格式错误率很低，暂不处理，留到短跑时观察。
- **长度惩罚**触发率 0–1.3%，只有 18/1647 组的组内排序因此改变。

## 3. 实验 B：bf16 merged SFT，按 GRPO rollout 条件采样（150 qid × G=8，真实检索）

- 采样（`sample_bf16.py`）：模型 `v2-vh-sft-serving-v1-exec-20261023/model/merged`（bf16，与 Q4 同一 adapter）。纯温度 T=0.7，不加 top-k/top-p/grammar，max_new_tokens 640，MPS 上运行。prompt 渲染已断言与 SFT 训练样本逐字节一致。每个数据集随机抽 50 个可打分 qid（seed 0）。耗时约 31 s/qid，共 79 分钟。
- 打分（`score_groups.py`）：用 `reward-server.ts` 加载 3 个单语料 index（fiqa/scifact/android，path A，rerank=false），启动时 index 校验 3/3 通过。raw 指标与 path-A oracle 一致，150/150。

| | ΔM 均值 | G=3 / 4 / **8** 零方差 | 组内极差 p50 / p90 | best-of-8 |
|---|---|---|---|---|
| FiQA | +0.030 | 35% / 28% / **20%** | 0.150 / 0.489 | +0.121 |
| SciFact | +0.020 | 56% / 50% / **40%** | 0.056 / 0.350 | +0.074 |
| Android | +0.012 | 51% / 46% / **40%** | 0.100 / 0.358 | +0.085 |

同一批 qid 在 G=3 下与 Q4 对比：零方差组 FiQA 36%→24%，SciFact 56%→58%，Android 46%→52%。组内 sd 0.044→0.056、0.042→0.025、0.033→0.036。

**结论**

1. **量化和采样参数不是信号弱的主因**。bf16 只在 FiQA 上明显更好，另外两个数据集持平，且每组 n=50，差异在噪声范围内。
2. **平均增益小是 SFT 起点决定的，与公式无关**。同一批 qid 上，bf16 采样、Q4 采样、Q8 greedy 的 ΔM 都在 +0.01~+0.03。GRPO 依赖的是组内差距：best-of-8 比组均值高 0.05~0.10，within-qid 方差占总方差的 43%，说明有信号可学。
3. **G=8 明显有效**。实验 A 根据 3 个样本外推 G=8 的上界为 38/71/50%，实测为 20/40/40%，外推过于保守。
4. **天花板 query 在 G=8 下仍没有信号**。raw nDCG@10=1 的 qid 中，零方差组占比：SciFact 15/21，Android 14/17。
5. **不加 grammar 时格式错误为 2/1200**：一条是 vec 行之间有空行，一条是 hyde 后面跟了 markdown 列表。−1 惩罚正好用来教格式。截断为 0，最长 622 token，接近 640。
6. **奖励服务本机延迟**（M1 Pro/Metal）：p50 205 ms，p95 577 ms。

## 4. 对 Phase 2 配置的建议

| 项 | 建议 | 依据 |
|---|---|---|
| reward 公式 | 按 §1，不改 | 实验 A、B |
| `max_completion_length` | 200 → **640** | SFT 输出中 31% 超过 200 token；实验 B 最长 622 |
| `num_generations` | **8** | 实验 B：零方差组 20/40/40% |
| `scale_rewards` | **`"batch"`** | 实验 A 的缩放对比 |
| `mask_truncated_completions` | `False` | 截断样本需要计入 loss，模型才能学到别写飞 |
| rollout 采样 | 纯 T=0.7 | 不需要模仿 serving 的 top-k/top-p |
| 训练集 | **过滤 raw nDCG@10=1 的 qid**，或对 SciFact 降采样 | SciFact 有 60% 的 qid 是天花板 |

## 4.1 实验 C：训练 source 天花板比例（2026-11-15）

脚本 `raw_ceiling.py`：用 reward-server（4 个单语料 index，path A，启动时 index 校验 4/4 通过）对全部可打分 qid 跑 raw 检索（发空 completion，服务回 `raw_metrics`）。共 8257 个 qid，耗时约 25 分钟，平均 150–213 ms/q。数据：`train-raw.jsonl`。

| source | n | nDCG@10=1（天花板） | R@30=0（召回不到） | 平均 nDCG@10 | 平均 R@30 | 过滤后保留 | 保留后占比 |
|---|---|---|---|---|---|---|---|
| scifact-train | 809 | **75.4%** | 2.0% | 0.876 | 0.980 | 199 | 3.1% |
| programmers | 876 | 23.1% | 24.2% | 0.421 | 0.692 | 674 | 10.5% |
| unix | 1072 | 24.2% | 28.8% | 0.407 | 0.654 | 813 | 12.6% |
| fiqa-train | 5500 | 13.6% | 19.4% | 0.402 | 0.636 | 4753 | **73.8%** |

对照实验 B（G=8）：天花板 qid 中 77% 的组零方差（43 组），非天花板 qid 只有 18%（107 组）。

**结论**

1. **天花板过滤收益很大，代价很小**。去掉 22% 的 qid（8257 → 6439），可以把零方差组从约 77% 的那一档整体移到约 18% 的那一档。
2. **SciFact-train 基本是"已解决"的数据集**：75% 是天花板，R@30 达到 0.98。过滤后只剩 199 条，占 3%，对训练几乎没有贡献，可以保留但不必刻意加权。
3. **过滤后 FiQA 占 74%**，分布偏向单一领域。如果希望 CQADup 的权重更高，需要按 source 重新加权采样，比如三个大 source 等权。这需要决定。
4. **R@30=0 的 qid 是另一类难题**：FiQA、CQADup 中有 19–29% 的 qid，raw 检索在前 30 条里一篇相关文档都找不到。这类 qid 的 ΔM 只可能 ≥ 0，属于"扩展唯一能帮上忙"的类型，但也可能几乎所有采样都是 0。实验 B 没有单独拆出这一类，是否保留有待观察。

## 4.2 实验 D：R@30=0 的成因与标注质量（2026-11-15）

**深度探针**（`r30zero-probe.ts`，只读）：R@30=0 和 R@30>0 各抽 100 个 qid（fiqa-train 40 / programmers 30 / unix 30），统计相关文档在 BM25 和向量检索前 1000 名中的位置。
- 相关文档全部存在于 corpus/index 中，没有缺失。两组的 query 长度、文档长度无差异。
- R@30=0 组中，相关文档的向量排名分布：≤30 占 4%，31–100 占 40%，101–300 占 29%，>300 占 27%。R@30>0 组 99% 在 ≤30。
- BM25 两组都有 55% 结果为空，不是区分两组的因素。
- R@30=0 组相关文档与 query 的词重合中位数为 0.29–0.33，对照组为 0.57–0.60。FiQA 中只有 1 个相关文档的 qid 占 57%，对照组为 32%。

**盲评**（LLM，单评审，文档截断到 500 字，标注文档和 top 未标注文档混在一起打乱顺序；数据在 `judge/`）：

| | 标注文档为强相关* | 标注文档无关 | top 未标注文档为强相关* |
|---|---|---|---|
| CQADup R@30>0 | 77% | 2% | 17% |
| CQADup R@30=0 | **15%** | 10% | 13% |
| FiQA R@30>0 | 82% | 0% | **42%** |
| FiQA R@30=0 | 55%（62% 需要问题正文才能判断） | 0% | 25% |

\*CQADup 指重复问题，FiQA 指直接回答。CQADup 的 R@30=0 组中，有 62% 的 qid 其 top 文档得分不低于标注文档。

结论：
- 标错（标注了无关文档）很少。主要问题有两个：一是 R@30=0 组的标注多为弱关联，CQADup 标注的是用户手标的重复链接，FiQA 的 query 只有标题、没有正文；二是**整个数据集普遍存在漏标**，即便是 R@30>0 的 FiQA qid，top 未标注答案也有 42% 是直接回答。
- 对评测的影响：相对比较仍可用，但漏标会压缩系统间差距，扩展的真实收益可能被低估。建议另加 LLM judge 指标。

**R@30=0 的 G=8 采样**（100 qid，bf16 T=0.7，真实检索打分；数据为 `r30zero-bf16-T0.7-G8.*`）：零方差组占 56%（非天花板评测 qid 为 18%）。只有 27% 的组中至少有一个样本把相关文档召回进 top 30，且这一比例随相关文档原始排名变化：向量排名 ≤100 时为 45%，101–300 为 14%，>300 为 11%。由于 raw 为 0，ΔM 只能 ≥0，这类 qid 只提供正向信号。**建议从训练集中排除 R@30=0 的 qid。**

## 4.3 实验 E：qrels reward 与 LLM judge 的一致性（2026-11-15）

`consistency.py`：从两批 G=8 数据中，取组内 reward 极差 ≥0.1 的组，选出 reward 最高和最低的两个 rollout，分别取 top-10（`reward-server.ts` 新增 `with_ranking` 字段），每篇文档截取 300 字。两个列表随机分配到 A/B 两侧，交给 LLM 做两两盲评（3 个 agent 分片并行）。共 101 对，平局 6 对。数据在 `consistency/`。

| 切片 | 一致率（不含平局） | 95% CI |
|---|---|---|
| 全部 | 66%（63/95） | [56%, 75%] |
| **中间类 qid（0<nDCG<1，R@30>0）** | **69%（41/59）** | [57%, 80%] |
| R@30=0 | 61%（19/31） | [44%, 76%] |
| 极差 [0.1, 0.2) | 53% | [36%, 70%] |
| 极差 [0.2, 0.4) | 65% | [51%, 77%] |
| **极差 ≥ 0.4** | **94%（15/16）** | [72%, 99%] |
| 中间类，按极差加权 | **76%** | [64%, 86%]（bootstrap） |

位置偏差检查：judge 选 A 53 次、选 B 42 次。高分 rollout 在 A 侧时一致率 70%，在 B 侧时 62%，存在轻微的 A 偏好，但随机分配两侧可以抵消。

结论：
1. **qrels reward 的方向是对的，强度和极差相关**。极差 ≥0.4 时几乎总与 judge 一致；极差 <0.2 时接近随机，这部分组内差异主要是标注噪声。
2. **对 GRPO 而言，按极差加权的一致率才是相关指标**，因为 advantage 与 reward 差值成正比，小差值本身贡献的梯度就小。中间类加权一致率为 76%，达到预设的 ≥~75% 门槛，**但 CI 下界为 64%，属于勉强达标**。
3. 判定：**方案 B（qrels，只用中间类 qid）可以作为起点**；方案 C（离线补标）作为改进项，不阻塞训练。小极差的噪声可以通过 `scale_rewards="batch"` 削弱，不按组标准差放大。
4. CQADup 训练 source 样本太少（programmers 7 对、unix 8 对，且全部是 R@30=0），无法单独下结论。中间类的证据主要来自评测侧 qid（fiqa-test、android、scifact-test）。

## 5. 未决与局限

- 实验 B 每个数据集只有 50 个 qid，且用的是评测集而非训练集，bf16 与 Q4 之间几个 pp 的差异不显著。
- 过滤后 FiQA 占 74%：是否按 source 重新加权（见 §4.1）。如果同时排除 R@30=0，训练集约为 0<nDCG<1 且 R@30>0 的 qid，规模需要重新统计。
- 一致性验证的中间类证据来自评测侧 qid，训练侧中间类 qid（尤其是 CQADup）还没直接验证。
- 实验 D/E 的评审都是 LLM 单评审，没有人工复核。
- 没测 T=1.0（看组内差距能否进一步拉大，以及格式错误率变化）。
- 0.7/0.3 的权重是否与 Phase 3 评测口径对齐，需要决定。
- 系数 0.7/0.3、320、0.2 是根据分布推算的，需要在第一轮短跑后按监控指标调整（format_error 率、零方差组比例、分数据集 ΔM、长度 p95）。

## 6. 复现

```sh
# 实验 A（离线，秒级）
python3 -I finetune/experiments/grpo-retrieval/phase2-reward-offline/offline_reward.py \
  artifacts/sft-serving-baseline/v2-vh-sft-serving-v1-exec-20261023T100921/benchmarks

# 实验 B：采样（约 80 分钟，MPS）
P=artifacts/grpo-retrieval/pathA
finetune/.venv/bin/python -I finetune/experiments/grpo-retrieval/phase2-reward-offline/sample_bf16.py \
  --model artifacts/sft-serving-baseline/v2-vh-sft-serving-v1-exec-20261023T100921/model/merged \
  --template finetune/experiments/grpo-retrieval/phase2-reward-offline/template.txt \
  --train-sample finetune/data/public-distill-v2-vh-prompt-v1/experiments/public-main-v2-vh-prompt-v1/sft-train.jsonl \
  --out artifacts/grpo-retrieval/phase2-reward-offline/bf16-T0.7-G8.jsonl --G 8 --temperature 0.7 \
  --source fiqa-test=$P/qmd-eval-v1-fiqa-test:50 \
  --source scifact-test=$P/qmd-distill-public-v0-scifact-train:50 \
  --source android=$P/qmd-eval-v1-cqadup-android:50

# 实验 B：打分（先启动 reward-server，3 个 --db + 3 个 --source，见 ../README.md §6）
python3 -I finetune/experiments/grpo-retrieval/phase2-reward-offline/score_groups.py \
  artifacts/grpo-retrieval/phase2-reward-offline/bf16-T0.7-G8.jsonl \
  --q4-bench artifacts/sft-serving-baseline/v2-vh-sft-serving-v1-exec-20261023T100921/benchmarks
```

`stats.py` 用于统计 path-A oracle 的 ΔM 和长度分布（§1 中系数的依据）。

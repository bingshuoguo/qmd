# SFT-VH V2-A Pareto Data Reselection Design

Date: 2026-08-29 (Asia/Shanghai)

Status: approved design

Scope: construct and evaluate the next QMD `hyde + vec` completion-only SFT release without changing the deployed expansion protocol or introducing new teacher data in the first experiment.

## 1. Decision Summary

SFT-VH V2-A is a controlled data-reselection experiment. It reuses the existing teacher candidate text, recomputes retrieval metrics under a clean frozen QMD environment, admits only strict Pareto winners, adds semantic quality as a release gate, and trains the same pinned Qwen3-1.7B LoRA configuration with a length-aware prompt.

The experiment optimizes general retrieval quality rather than a single benchmark or a single cutoff. The target is to improve equal-weight macro Recall@30, MRR@10, and nDCG@10 while preserving Recall@10 and Recall@20.

The approved sequence is:

1. V2-A: reuse and reselect existing candidates.
2. V2-B: add FreshStack, new NFCorpus candidates, or another teacher only if V2-A passes its promotion gates.

Webmasters remains a stress test. It is reported independently and does not enter the primary macro or independently veto a generally useful expansion model.

## 2. Why V2-A Is Necessary

The current `public-main-v2-vh-prompt-v1` adapter established that SFT can improve QMD retrieval:

- Recall@30 improved by 1.50 percentage points with a positive paired-bootstrap confidence interval.
- MRR@10 improved by 1.48 percentage points with a positive confidence interval.
- nDCG@10 improved by 0.64 percentage points with a positive confidence interval.
- Recall@10 and Recall@20 did not improve significantly.

Its supervision set nevertheless has four correctable weaknesses:

1. The winner rule optimizes `R@20 > raw`, permits nDCG@10 to fall by 0.02, and does not guard Recall@30 or MRR@10. Among the 1,017 SFT targets, 36 lower Recall@30, 47 lower MRR@10, and 65 lower nDCG@10 relative to raw retrieval.
2. A 200-record semantic audit reported 20 failures and 6 uncertain records. Those records were diagnostic findings rather than admission failures and remained in the SFT release.
3. The source distribution is dominated by NFCorpus: 728 of 1,017 records. SciFact contributes only 2 records.
4. The student generates longer output than its targets. Training HyDE averages about 82 words, while held-out SFT generation averages about 113 words and has a 608-word maximum.

The evidence supports improving supervision quality before switching to preference optimization or online reinforcement learning.

## 3. Goals and Non-Goals

### 3.1 Goals

- Produce 1,000 to 2,000 final training records plus a separate validation/retrieval-dev split of approximately 10% of the training count.
- Admit only candidates that do not regress any tracked per-query retrieval metric.
- Make entity drift, lost constraints, lost negation, unsupported facts, and ambiguous over-interpretation release-blocking failures.
- Preserve scarce eligible source data without duplicate sampling.
- Keep data, prompt, training, retrieval, and evaluation provenance reproducible.
- Separate prompt-only gains from gains caused by the new supervision release.
- Select checkpoints by retrieval quality, not validation loss.

### 3.2 Non-Goals

- Do not change the runtime `hyde + vec` output protocol.
- Do not teach a `no expansion` or routing action in V2-A.
- Do not introduce FreshStack or newly generated teacher candidates in V2-A.
- Do not use qualified ties, trade-offs, or failed candidates as SFT positives.
- Do not implement DPO, ORPO, GRPO, or PPO in this experiment.
- Do not claim multilingual quality; V2-A training and primary evidence are English-language.
- Do not make a production release decision from training loss or protocol validity alone.

## 4. Experiment Identity

The existing experiment already uses `v2-vh` identifiers. V2-A must use new immutable IDs rather than overwrite or ambiguously extend that release.

Approved identity pattern:

```text
source release:  public-distill-v3-vh-pareto
prompt release:  public-distill-v3-vh-pareto-prompt-v1
experiment:      public-main-v3-vh-pareto-prompt-v1
prompt version:  qmd-student-expansion-v3-vh-pareto-v1
```

Any change to candidate input hashes, retrieval profile, admission policy, semantic decisions, split membership, prompt bytes, or materialized JSONL requires a new release or experiment ID.

## 5. Candidate Inputs

V2-A reuses candidate text from the existing V2-VH projection and native-generation artifacts. Previous winner labels and previous retrieval metrics are diagnostic inputs only. Formal admission requires rescoring every candidate under the newly frozen clean environment.

The input manifest must record:

- source artifact paths, row counts, byte sizes, and SHA-256 hashes;
- unique query and candidate counts;
- candidate generation provider, model revision, prompt version, parameters, and provenance;
- source dataset, split, qid, input_id, sample_key, and family/group identifier when available;
- exact raw candidate output and parsed canonical output;
- duplicate and overlap reports against all primary held-out queries.

Gold documents and qrels must never enter teacher prompts. They are used only by the frozen retrieval scorer.

## 6. Output Contract

Each candidate and each student completion has this protocol:

```text
hyde: <one hypothetical relevant passage>
vec: <one semantic search query>
vec: <optional second complementary semantic search query>
```

Contract requirements:

- exactly one HyDE line;
- one or two Vec lines after HyDE;
- no Lex lines, think blocks, commentary, bullets, or Markdown wrappers;
- HyDE target length: 40 to 100 words in the output language;
- short or unambiguous queries may use fewer than 40 words;
- complex queries may exceed 100 words;
- HyDE hard maximum: 150 words and 256 pinned-Qwen tokens;
- each Vec hard maximum: 48 pinned-Qwen tokens;
- total completion hard maximum: 384 pinned-Qwen tokens;
- HyDE 5-gram repetition rate must not exceed 30%;
- Vec1 and Vec2 must not be near duplicates;
- a Vec must not merely copy the complete input query.

Length is measured and reported. It must not be enforced by silently truncating text. A hard-limit violation rejects the candidate.

## 7. Clean Retrieval Rescoring

All candidates must be scored through the real QMD retrieval path with `rerank=false` and a frozen no-auto-generation profile.

The formal rescoring manifest pins:

- QMD Git commit and `qmd_dirty=false`;
- collection name and root;
- collection-scoped index fingerprint;
- active document/source mapping;
- embedding model identity and SHA-256;
- vector completeness;
- retrieval cutoffs and limits;
- `result_limit=30`, `per_list_limit=30`, and `candidate_limit=40` or their explicitly versioned successors;
- query, document, qrels, and split hashes;
- expansion parser and contract versions;
- rerank disabled;
- zero retrieval errors.

For every query, score raw retrieval and every Contract-valid candidate for:

- Recall@10;
- Recall@20;
- Recall@30;
- MRR@10;
- nDCG@10.

## 8. Strict Pareto Admission

A candidate is retrieval-eligible only when all five per-query guardrails hold:

```text
candidate Recall@10  >= raw Recall@10
candidate Recall@20  >= raw Recall@20
candidate Recall@30  >= raw Recall@30
candidate MRR@10     >= raw MRR@10
candidate nDCG@10    >= raw nDCG@10
```

At least one metric must be strictly greater than raw. There is no epsilon regression allowance.

When a query has multiple eligible candidates, select in this order:

1. greater number of strictly improved metrics;
2. greater Recall@30 delta;
3. greater MRR@10 delta;
4. greater nDCG@10 delta;
5. greater Recall@20 delta;
6. fewer Vec lines;
7. fewer completion tokens;
8. lower stable candidate ID.

The final tie-breaker must be deterministic. A query without an eligible candidate is `no_pareto_winner` and does not enter V2-A SFT.

## 9. Semantic Admission

Retrieval benefit does not prove semantic fidelity. Every Pareto-selected candidate must pass all three semantic layers.

### 9.1 Deterministic Checks

Deterministic checks cover what can be established without a judge:

- quoted phrases, version identifiers, error codes, numbers, units, and explicit named entities are preserved in at least one output line;
- explicit negation and comparison direction are represented in HyDE and at least one Vec;
- output does not introduce a contradictory number, entity, polarity, or version;
- Vec items pass the duplication checks from the output Contract.

A deterministic failure rejects the candidate before judge review.

### 9.2 Full LLM Judge

An independently versioned judge reviews every remaining candidate and returns one of:

- `pass`;
- `fail`;
- `uncertain`.

The judge assigns normalized reason codes including:

- `entity_drift`;
- `lost_constraint`;
- `lost_negation`;
- `unsupported_fact`;
- `unsupported_causality`;
- `ambiguous_overinterpretation`;
- `unsupported_abbreviation_expansion`;
- `non_complementary_vec`;
- `intent_drift`.

The judge prompt, model revision, decoding parameters, request/response hashes, raw response, parsed status, and reason codes are retained.

### 9.3 Human Review

Human review covers:

- every judge `fail`;
- every judge `uncertain`;
- a deterministic, source-and-phenomenon-stratified 10% sample of judge `pass` records.

Only final `semantic_status=pass` records can enter the release. A human may override a judge decision only with a recorded decision and reason.

If the pass sample contains a critical error such as entity drift, lost negation, lost constraint, or fabricated fact, release sealing pauses. The corresponding source/phenomenon stratum is reviewed in full, the judge or deterministic rule is corrected, and semantic admission is rerun from the frozen candidate input.

The release gate is zero known critical semantic errors.

## 10. Repair and Failure Handling

The pipeline uses explicit non-overlapping terminal statuses:

- `contract_invalid`;
- `retrieval_error`;
- `metric_tradeoff`;
- `no_pareto_winner`;
- `semantic_fail`;
- `semantic_uncertain`;
- `eligible`;
- `selected`;
- `not_selected_capacity`.

Errors must not be collapsed into `no_pareto_winner`.

Candidate text is never silently repaired. If a rejected candidate is rewritten, the rewrite becomes a new candidate with a new ID and provenance and must repeat Contract validation, retrieval scoring, Pareto admission, and semantic admission. V2-A does not perform such regeneration; repair is part of the fallback/V2-B path.

Rejected candidates remain in the ledger for failure analysis and possible later preference-pair construction. They are not SFT positives.

## 11. Source Allocation and Dataset Size

The final training split contains 1,000 to 2,000 records. Validation/retrieval-dev is separate and approximately 10% of the training count, so the sealed release is expected to contain roughly 1,100 to 2,200 records.

Allocation rules:

1. Use every eligible record from a scarce source, subject to assigning some records to the family-disjoint validation split.
2. Do not duplicate or oversample a scarce source to meet a target ratio.
3. Do not enforce equal source proportions.
4. Treat a 40% NFCorpus share as a distribution warning, not a hard rejection rule.
5. If the eligible training set exceeds 2,000, reduce abundant, repetitive strata while preserving source and query-phenomenon coverage.
6. Record every source's `input_count`, `pareto_count`, `semantic_pass_count`, `selected_train_count`, `selected_validation_count`, and rejection reasons.

If semantic admission and family-safe splitting produce fewer than 1,000 training records, V2-A must not train. It exits with a supply report and proceeds to a separately approved fallback-generation design.

## 12. Query Phenomena and Splitting

Each query receives deterministic or reviewed phenomenon tags where applicable:

- short/ambiguous/entity-only;
- negation/exclusion;
- comparison/preference;
- number/unit/threshold;
- version/product/error-code;
- temporal/current-information;
- abbreviation/acronym;
- long natural-language question;
- code/error-message;
- finance;
- health/medical;
- science;
- software/technical;
- general knowledge.

Splitting requirements:

- seed 42;
- source-stratified where possible;
- family/group-disjoint;
- normalized-query and near-duplicate disjoint;
- validation size approximately 10% of train;
- no overlap with any primary held-out or stress-test query;
- split membership frozen before training and recorded by hash.

The validation split serves two purposes: completion-only loss diagnostics and retrieval-dev checkpoint selection. Validation loss never selects the adapter.

## 13. Student Prompt and Prompt-Only A/B

The V2 prompt preserves the existing output types and adds the approved length behavior. Its exact bytes are frozen before data materialization.

Required instructions include:

```text
Return exactly one hyde line followed by one or two vec lines.
Target 40-100 words for HyDE; use fewer for a simple query and never exceed 150 words.
Preserve entities, versions, numbers, constraints, negation, comparison, and intent.
Do not invent unsupported facts, causal claims, laws, statistics, or definitions.
Make Vec queries complementary rather than duplicates.
```

Before V2 training, run a prompt-only diagnostic on retrieval-dev:

1. current SFT-VH adapter with its original prompt;
2. current SFT-VH adapter with the V2 prompt.

Compare retrieval metrics, Contract validity, HyDE length distribution, repetition, semantic failures, generated tokens, and latency. This experiment determines the stronger SFT-VH prompt arm before the primary test is opened.

The stronger arm becomes the controlled SFT-VH baseline for V2 data-effect comparisons. The original deployed SFT-VH arm is retained for system-level comparison.

## 14. Training Configuration

V2-A keeps the SFT-VH model and optimization configuration fixed:

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

All train and validation sequences are preflighted before model loading. A sequence over 1,024 tokens is rejected; the trainer must not truncate completion-only supervision.

Save epoch 1, epoch 2, and epoch 3. Set `load_best_model_at_end=false`. Run retrieval-dev for every checkpoint and select the candidate checkpoint from retrieval evidence.

Training loss and validation loss are diagnostics only.

## 15. Seed Strategy

The first V2-A training run uses seed and data seed 42. If its selected checkpoint passes the candidate promotion gate, run a complete second training with a distinct frozen seed.

Final research promotion requires:

- the required point-estimate directions in both seeds;
- the average per-query metric across the two seeds to pass the paired-bootstrap gates;
- no seed-specific semantic, protocol, or truncation regression hidden by averaging.

The second seed is not used to choose a different prompt, data release, or checkpoint rule.

## 16. Evaluation Layers

### 16.1 Retrieval-Dev

Retrieval-dev is the frozen V2 validation split. It is used for:

- prompt-only A/B;
- epoch 1/2/3 checkpoint selection;
- development diagnostics and failure analysis.

It must not be added to formal held-out macro results.

### 16.2 Primary Held-Out Suite

The primary suite contains:

- SciFact test;
- FiQA test;
- CQADup Android;
- NFCorpus test;
- FreshStack test.

Each dataset contributes equal weight to the macro, regardless of query count. Primary test queries, qrels, documents, and results are not used to alter the data release, prompt, selection rule, checkpoint rule, or semantic policy.

The SFT-VH baseline must be evaluated on NFCorpus test and FreshStack test under the same clean profile before V2 test comparison.

### 16.3 Stress Tests

Stress tests are reported independently:

- CQADup Webmasters;
- short entity and ambiguous queries;
- negation, number, version, and abbreviation slices;
- long code and error-message queries;
- a human-reviewed semantic-drift set.

Webmasters does not enter the primary macro and cannot independently veto a generally useful expansion model. Material regressions remain visible and may motivate later routing work.

## 17. Formal Comparison Arms

The formal evaluation includes:

| Arm | Purpose |
| --- | --- |
| Raw query | Original retrieval baseline |
| SFT-VH + original prompt | Current deployed research baseline |
| SFT-VH + V2 prompt | Prompt-only and controlled data baseline |
| V2-A selected checkpoint + V2 prompt, seed 42 | Candidate data experiment |
| V2-A selected checkpoint + V2 prompt, second seed | Replication |

All non-raw arms use the same parser, retrieval profile, index, embedding model, cutoffs, limits, rerank setting, and latency measurement method.

## 18. Promotion Gates

Compare V2-A against the stronger SFT-VH prompt arm selected on retrieval-dev. Report comparison against raw and the original SFT-VH arm as additional system-level evidence.

On the five-dataset equal-weight primary macro, V2-A must satisfy:

```text
Recall@10:
  paired-bootstrap 95% CI lower bound >= -0.5 percentage points

Recall@20:
  point estimate >= 0

Recall@30, MRR@10, nDCG@10:
  all three point estimates > 0
  at least two paired-bootstrap 95% CI lower bounds > 0
```

Generation and semantic gates:

```text
Contract-valid rate >= 99.5%
format errors = 0
fallbacks = 0
generation errors = 0
known critical semantic errors = 0
```

Every dataset and phenomenon slice reports per-query win/tie/loss counts and top gains/losses. Equal-weight macro results must not hide a material per-dataset regression.

Latency reporting includes expansion generation and retrieval. V2-A remains a research candidate unless a later production decision explicitly approves its end-to-end cost and compatibility.

## 19. Required Artifacts

The V2-A package contains immutable manifests and hashes for:

- candidate input inventory;
- duplicate/leakage audit;
- frozen retrieval environment and index fingerprints;
- raw and per-candidate retrieval results;
- Pareto selection ledger;
- Contract/length/repetition audit;
- LLM judge raw and normalized results;
- human-review decisions and pass-sample selection;
- source and phenomenon allocation report;
- split manifest;
- prompt-only A/B results;
- sealed SFT train and validation JSONL;
- release manifest;
- training config and run manifests;
- all three checkpoint adapters per seed;
- generation manifests;
- formal retrieval runs;
- paired-bootstrap and per-query win/loss reports;
- end-to-end latency report.

Raw source and candidate artifacts remain immutable. Derived artifacts are written under new experiment IDs and are never used to overwrite SFT-VH evidence.

## 20. V2-B Entry Gate

V2-B may be designed only after one of these outcomes:

1. V2-A passes promotion gates, and additional domain coverage is the next explicit objective.
2. V2-A cannot reach 1,000 training records after semantic admission, and a supply report identifies which phenomena or sources need new candidates.
3. V2-A fails retrieval promotion, and per-query analysis identifies a correctable candidate-coverage gap rather than a selection or prompt problem.

V2-B may then introduce FreshStack train, new NFCorpus candidates, or another teacher. It must preserve family-disjoint held-out tests and use new release and experiment IDs.

## 21. Acceptance of This Design

The following decisions were approved interactively:

- comprehensive retrieval objective;
- strict five-metric per-query Pareto admission;
- two-stage V2-A then V2-B sequence;
- scarce eligible source records are retained rather than discarded for quotas;
- 1,000 to 2,000 final training records;
- full deterministic and LLM semantic review plus human fail/uncertain review and 10% pass sampling;
- 40 to 100 word HyDE target and 150 word hard maximum;
- explicit failure ledger with no silent repair;
- prompt-only pre-training A/B;
- seed 42 candidate run followed by second-seed replication after promotion;
- five-dataset primary held-out suite;
- Webmasters as a separately reported stress test.

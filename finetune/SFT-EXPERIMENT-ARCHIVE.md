# SFT Experiment Archive

Provenance index for the query-expansion SFT experiments (2026-08 → 2026-11).
Code lives in the main-line tree; experiment-era scripts that were culled are
archived on git tag `archive/completion-only-sft-20261104`. Result artifacts
and training data are **not in git** — they exist only on the local machine.

## Where things live

| Layer | Location | Notes |
| --- | --- | --- |
| Main-line code | this tree | Search, distillation, SFT, OPD code kept after review |
| Experiment-era code | tag `archive/completion-only-sft-20261104` | Historical prompts, legacy env profiles, SciFact pilot entry points, V2-A Pareto tooling, OpenAI teacher profile, GRPO-retrieval. `git checkout <tag> -- <path>` to recover |
| Result artifacts | `artifacts/` (gitignored) | `sft-serving-baseline/` 16G, `sft-runs/` 3.2G, `grpo-retrieval-20261102/` 1.6G, `grpo-retrieval/` 1.5G, `fusion-analysis/` |
| Training/distillation data | `finetune/data/` (gitignored) | `public-distill-v0` 4G, `public-distill-v1`, `-v2-vh`, `-v2-vh-prompt-v1`, `-v3-vh-pareto`, `nf-freshstack-v1`, `distillation/scifact-v1` (24 pilot runs) |
| Local profiles | `.env.distill.*.local` (gitignored) | Credentials; never commit |

**No off-machine backup exists** for `artifacts/` or `finetune/data/` as of
2026-11-04.

## Experiment timeline

### 1. SciFact teacher-selection pilot (Aug 2026)

Purpose: pick a teacher prompt for query distillation. 24 runs under
`finetune/data/distillation/scifact-v1/runs/` compared DeepSeek v1/v2/v3,
JSON2/3/4 shapes, and gpt-5.6-terra (OpenAI Responses API).

Outcome: DeepSeek `qmd-expansion-teacher-v2-deepseek-json4-semantic-safe`
(JSON4 V2) won; `public-distill/generate.ts` enforces it. The pilot entry
points (`benchmarks/distill/{split,generate,score,ablate,report}.ts`) and the
losing prompts/env profiles are on the archive tag only.

### 2. Public distillation releases → SFT runs 01–04

Each release freezes a distillation dataset; each run trains Qwen LoRA with
completion-only loss and evaluates on BEIR/FreshStack retrieval suites.
Full snapshots in `artifacts/sft-runs/<archive>/` (adapter weights, frozen
code, evaluation reports, `archive-manifest.json`).

| Run | Release | Model | Verdict |
| --- | --- | --- | --- |
| 01 `public-main-v1` | public-distill-v1 | Qwen3-1.7B ckpt-226 | Historical research baseline |
| 02 `public-main-v2-vh-prompt-v1` | public-distill-v2-vh-prompt-v1 | Qwen3-1.7B final-adapter | Research candidate; gains on R@30/MRR@10/nDCG@10; Webmasters ranking risk |
| 03 `public-main-v2-vh-prompt-v1-qwen3-8b` | same, FSDP BF16 | Qwen3-8B | 7 datasets complete (2026-09-13); research candidate; run manifest records a dirty tree |
| 04 `public-main-v3-vh-pareto-prompt-v1` | public-distill-v3-vh-pareto | Qwen3-1.7B ckpt-226 | See correction below |

**V2-A Pareto correction (read before using run 04 numbers)** — two errors
invalidated the original "vs raw" deltas: (a) the `raw` arm had the reranker
on while every expansion arm ran with it off; (b) the `sft-vh-v2a` arm
emitted capitalized `HyDE:` which the strict parser rejected, so 100% of its
queries were format errors — it was never actually evaluated. Corrected
reading: expansion helps ~1–2pp, `v2a` ≈ `sft-vh-orig` (macro nDCG@10 0.4480
vs 0.4478), no clear V2-A win. Details in
`artifacts/sft-runs/04-public-main-v3-vh-pareto-prompt-v1/evaluation/evaluation-report.md`
§0. Consequently the V2-A selection tooling (`v2a_canonicalize`,
`v2a_pareto_select`, `benchmarks/v2a-pareto/`, `sft-v2a-pareto.yaml`) is on
the archive tag; only the shared prompt-identity module
(`dataset/v2a_pareto_release.py`) remains in the main line.

### 3. G0 serving investigation (2026-10-26 → 10-30)

Why SFT gains were lost at serving time:
- node-llama-cpp's default system prompt (~0.01 nDCG@10) and Q4_K_M (~0.011)
  were the causes; Q8_0 + empty system prompt + greedy recovers them
  (macro nDCG@10 0.558 vs raw 0.548, no rerank). Hence VH inference profiles
  default `system_prompt: ""`.
- Reranker scores were double-sigmoided; fixed by node-llama-cpp 3.20.0.
- The position-aware RRF blend was non-monotonic and lost to no-rerank even
  with fixed scores. Rank fusion `1/(60+rrf_rank) + 1/(60+rerank_rank)` was
  best offline (0.588 raw / 0.592 Q8) and shipped in the main line.
- Expansions under rank fusion: +0.004 nDCG@10 (n.s.) but +1.7pp R@30 /
  +2.7pp pool recall (significant). Candidate pool stays at 40.

Evidence: `artifacts/sft-serving-baseline/g0-20261026/README.md`.

### 4. OPD teacher qualification

`finetune/experiments/opd/` — evaluates the sealed SFT-VH student against
`Qwen/Qwen3-8B` on the English four-source comparison (no training). The 8B
SFT adapter it compares is run 03 (`sft-teacher-8b.yaml`).

### 5. GRPO retrieval (experimental, 2026-11-02)

`finetune/experiments/grpo-retrieval/` (archive tag only) plus results in
`artifacts/grpo-retrieval-20261102/`. Judged not worth pursuing for now.
Superseded by §7 (reward environment rebuilt, learnability gate and a GRPO run).

### 6. Post-merge SFT validation (2026-11-06 → 11-07)

After merging 202 upstream commits (`e5e4527`, tobi/qmd e428df7..86d2365) and
the legacy→partitioned vector-layout migration, the v2-VH final-adapter effect
was re-validated against the G0 rank-fusion baseline (`4bf85e6`) with frozen
Q8_0 profile-D expansions on FiQA test / SciFact / CQADup Android.

Verdict: **effect fully preserved.** Merge effect on both arms n.s.
(|Δ| ≤ 0.0015); SFT R@30 +0.0159 [+0.0091, +0.0229] significant on the merged
stack ≈ pre-merge +0.0169; nDCG@10 +0.0036 n.s. ≈ +0.0040. End-to-end with
expansions regenerated on the merged code: R@30 +0.0315 significant;
609/648 generations byte-identical, the rest same-structure paraphrases
(llama.cpp GPU nondeterminism).

Two bench bugs surfaced by the migrated index were fixed in `f64c287`:
`inspectBenchmarkIndex` still queried the partitioned vec table with the
legacy `hash_seq` key, and `openBenchmarkStore` disposed the embed model
mid-run after a system sleep. Hazard: `syncConfigToDb` rewrites the
collection registry of any index opened by getStore()-style commands from the
global config — pair `INDEX_PATH` with an isolated `QMD_CONFIG_DIR` when
touching non-global indexes.

Evidence: `artifacts/sft-serving-baseline/merge-eval-20261106/` (local, not in
git; index copy deleted after evaluation).

### 7. GRPO retrieval reward, learnability gate and short run (2026-11-10 → 11-18)

Code on branch `experiment/grpo` (not merged to main); full write-up in
`finetune/experiments/grpo-retrieval/RESULTS.md`. Reward = ΔM of real path-A
retrieval (M = 0.7·nDCG@10 + 0.3·R@30) from a multi-index reward server,
verified bitwise against CUDA-regenerated oracles on an AutoDL 4090.

Verdict: **GRPO does not help this task; within-group quality differences are
not learnable.** Learnability gate (947 training-source queries × 8 samples):
a Qwen3-1.7B LoRA reward model scores 0.464 [0.415, 0.513] pairwise accuracy
on unseen queries (dev-selected lr 2e-5: 0.472), flat learning curve, RM-BoN-8
−0.8pp vs greedy. GRPO from the v2-VH SFT model (cleaned pool of 4669 queries,
300 steps, seed 42): −0.16pp ΔM vs SFT [−0.55, +0.23] on FiQA test / SciFact /
CQADup Android (1647 queries); on-policy reward on unseen queries flat for all
300 steps. Not done: best-of-N distillation, rerank path, second seed.

Evidence: `artifacts/grpo-retrieval/` (local, not in git), indexed with sha256
in `artifacts/grpo-retrieval/archive-manifest.json`:

| Path | Size | Content |
| --- | --- | --- |
| `grpo-runs/s42-300/` | 92M | GRPO policy LoRA `final-adapter/`, `trainer_state.json` (every step), run log |
| `grpo-runs/eval/` | 8.7M | SFT and GRPO greedy expansions on the 3 eval sets, scored |
| `rm-gate-v1/` | 160M | gate RM adapters (`main/best`, `hp/lr2e-5/best`) and results of all 6 RM runs |
| `gate-v1/` | 100M | gate samples (greedy + 8) scored, preference pairs, probes, embeddings |
| `grpo-pool-v1/`, `eval-v1/` | 2M | cleaned GRPO training pool; eval query list; run scripts |
| `autodl-acceptance-20261115/` | 17M | 4090 reward-server acceptance: CUDA oracles, sampled replay passes 1/2, scripts, server logs |
| `index/` | 1.1G | 5 single-corpus indexes (rebuild with `setup-index.sh`) |

Left only on the AutoDL data disk (released if the instance expires): GRPO
optimizer states and checkpoints 50–250, RM adapters for frac25 / frac50 /
gap02 / lr1e-5 (each retrains in 5–12 min).

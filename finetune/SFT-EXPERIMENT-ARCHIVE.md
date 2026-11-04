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
| Result artifacts | `artifacts/` (gitignored) | `sft-serving-baseline/` 16G, `sft-runs/` 3.2G, `grpo-retrieval-20261102/` 1.6G, `fusion-analysis/` |
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

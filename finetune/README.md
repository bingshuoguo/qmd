---
license: mit
language:
  - en
base_model: Qwen/Qwen3-1.7B
tags:
  - query-expansion
  - search
  - gguf
  - qwen3
pipeline_tag: text-generation
---

# QMD Query Expansion Fine-Tuning

Train small language models to expand search queries for [QMD](https://github.com/tobi/qmd)'s hybrid retrieval pipeline.

## What This Does

Given a raw search query like `"auth config"`, the trained model produces structured expansions:

```
hyde: Authentication can be configured by setting the AUTH_SECRET environment variable.
lex: authentication configuration
lex: auth settings setup
vec: how to configure authentication settings
vec: authentication configuration options
```

These feed into QMD's three search backends:
- **`lex:`** lines go to BM25 full-text search (short, keyword-focused)
- **`vec:`** lines go to vector similarity search (natural language phrases)
- **`hyde:`** is a hypothetical document passage for embedding-based retrieval ([HyDE](https://arxiv.org/abs/2212.10496) technique)

## Quick Start

### Cloud training via HuggingFace Jobs (no GPU needed)

```bash
# 1. SFT: teach the model the output format (~45 min on A10G, ~$1.50)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py

# 2. Evaluate against test queries (needs local GPU or use eval job)
uv run eval.py tobil/qmd-query-expansion-1.7B

# 3. Convert to GGUF for local deployment (Ollama, llama.cpp)
uv run convert_gguf.py --size 1.7B

# NOTE: GRPO is currently experimental and moved to finetune/experiments/grpo
# if you want to run it manually, use:
#   cd finetune && uv run python experiments/grpo/grpo.py
```

### Local training (if you have a GPU)

```bash
uv run train.py sft  --config configs/sft.yaml

# Experimental GRPO
cd finetune && uv run python experiments/grpo/grpo.py
```

### Monitoring HF Jobs

```bash
hf jobs ps                           # list running jobs
hf jobs inspect <job-id>             # check status
hf jobs logs <job-id>                # stream logs
hf jobs cancel <job-id>              # cancel a job
```

## Prompt Format

All tools use the same prompt — **Qwen3 chat template with `/no_think`**:

```
<|im_start|>user
/no_think Expand this search query: {query}<|im_end|>
<|im_start|>assistant
```

The `/no_think` directive suppresses Qwen3's chain-of-thought mode, producing
direct `lex:/vec:/hyde:` output without `<think>` blocks.

## File Structure

```
finetune/
├── reward.py          # Scoring/reward function (single source of truth)
├── train.py           # SFT training entrypoint
├── eval.py            # Generate expansions and score them
├── convert_gguf.py    # GGUF conversion for Ollama/llama.cpp
├── jobs/
│   ├── sft.py         # Self-contained SFT for HuggingFace Jobs
│   ├── eval.py        # Self-contained eval for HuggingFace Jobs
│   └── eval_common.py # Shared eval utilities
├── configs/
│   └── sft.yaml       # SFT hyperparameters for Qwen3-1.7B
├── evals/
│   └── queries.txt    # 31 test queries across 8 categories
├── experiments/
│   └── grpo/          # Experimental GRPO configuration and script (optional)
├── data/              # Training JSONL files (all concatenated for training)
├── dataset/
│   ├── contract.py           # Contract v1: validation rules (single source of truth)
│   ├── schema.py             # Typed model + loader + renderers (delegates to contract)
│   ├── prepare_data.py       # Format Qwen3 prompt/completion, dedup, split; --only-mode switch
│   ├── completion.py         # Exact tokenization + completion-only loss mask
│   ├── validate_schema.py    # Lightweight Contract v1 check (tokenizer-free)
│   ├── validate_contract.py  # Full Contract v1 audit (pinned tokenizer, writes report)
│   ├── build_conflict_ledger.py # Deterministic conflict ledger for human review
│   ├── score_data.py         # Score all examples using reward.py
│   └── analyze_data.py       # Analyze distribution and quality
├── tests/             # Unit tests (contract, completion, schema delegation)
├── fixtures/          # Frozen Contract v1 fixture cases
├── SCORING.md         # Detailed scoring rubric reference
└── README.md          # This file
```

## Training Pipeline

### Stage 1: SFT (Supervised Fine-Tuning)

Teaches the model the `lex:/vec:/hyde:` output format from labeled examples.
The full rendered chat prompt remains model input, but
`completion_only_loss=True` masks user and template tokens so only the
assistant expansion is supervised.

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3-1.7B` |
| Method | LoRA (rank 16, alpha 32) |
| Target modules | All projection layers (q/k/v/o/gate/up/down) |
| Dataset | ~2,290 examples (train split) |
| Effective batch size | 16 (4 x 4 gradient accumulation) |
| Epochs | 5 |
| Learning rate | 2e-4 (cosine schedule) |

```bash
uv run train.py sft --config configs/sft.yaml
uv run train.py sft --config configs/sft.yaml --dry-run  # preview config
```

### Stage 2: (Experimental) GRPO

GRPO is currently treated as experimental and kept under `experiments/grpo/`.
It is not part of the default production path for this repository.

```bash
# Optional experimental GRPO run
cd finetune && uv run python experiments/grpo/grpo.py
```

## Evaluation

`eval.py` generates expansions from a model and scores them against test queries:

```bash
# Evaluate a SFT model
uv run eval.py --model tobil/qmd-query-expansion-1.7B-sft

# Evaluate an SFT output dir
uv run eval.py outputs/sft

# Verbose output with deduction details
uv run eval.py tobil/qmd-query-expansion-1.7B -v

# Optional: evaluate GRPO experimental output (if run)
uv run eval.py outputs/grpo

# Save detailed scores to JSON
uv run eval.py tobil/qmd-query-expansion-1.7B -o scores.json
```

## Reward Function

`reward.py` is the single source of truth for scoring. It is used for evaluation
and (optionally) as the GRPO reward signal in the experimental path.

Five scoring dimensions (max 120 without hyde, 140 with):

| Dimension | Points | What It Measures |
|-----------|--------|------------------|
| **Format** | 0-30 | Has lex/vec lines, no invalid lines |
| **Diversity** | 0-30 | Multiple expansion types, diverse content, no query echoes |
| **HyDE** | 0-20 | Present, 50-200 chars, single line, not repetitive |
| **Quality** | 0-20 | Lex shorter than vec, natural language, preserves key terms |
| **Entity** | -45 to +20 | Named entities preserved in lex and vec lines |
| **Think bonus** | 0-20 | Reward for NOT using `<think>` mode |

**Hard failures** (instant 0.0):
- Chat template leakage (`<|im_start|>`, `<|im_end|>`, etc.)
- Any line without a valid `lex:`, `vec:`, or `hyde:` prefix

```bash
# Self-test the reward function
uv run reward.py
```

## GGUF Conversion

Merges base + SFT and (optionally) GRPO adapters into a single model, then
produces quantized GGUF files for deployment:

```bash
# Use preset for 1.7B
uv run convert_gguf.py --size 1.7B

# Custom models
uv run convert_gguf.py --base Qwen/Qwen3-1.7B \
                       --sft tobil/qmd-query-expansion-1.7B-sft \
                       --grpo tobil/qmd-query-expansion-1.7B-grpo \
                       --output tobil/qmd-query-expansion-1.7B-gguf
```

### Using with Ollama

```bash
huggingface-cli download tobil/qmd-query-expansion-1.7B-gguf \
    qmd-query-expansion-1.7B-q4_k_m.gguf --local-dir .

echo 'FROM ./qmd-query-expansion-1.7B-q4_k_m.gguf' > Modelfile
ollama create qmd-expand -f Modelfile
ollama run qmd-expand
```

## Data Pipeline

All JSONL files in `data/` are concatenated for training. Validation rules are
owned by **`dataset/contract.py`** (Contract v1) — the single source of truth.
`dataset/schema.py` provides the typed Pydantic model, the fail-fast
`load_examples()` loader (which delegates to Contract v1), and the output
renderers, so the training path and the offline audit can never drift apart.

```bash
# Format prompt/completion records, deduplicate, split train/val
uv run python -m dataset.prepare_data

# Validate data quality
just validate
```

### Only-mode records

Records whose query carries an `/only:lex|vec|hyde` directive (the
`data/*_only*.jsonl` files) are **quarantined by Contract v1** — valid, but
scoped out of the default training target. `prepare_data` exposes this as an
explicit switch:

```bash
uv run python -m dataset.prepare_data                        # default: exclude only-mode
uv run python -m dataset.prepare_data --only-mode include    # legacy: train on them too
```

### SciFact distill-only pilot

SciFact distillation is isolated under `data/distillation/scifact-v1/`; the
default `data/*.jsonl` SFT glob never reads it.  The frozen test benchmark and
its existing Markdown collection/index remain unchanged.

```bash
# From the repository root: freeze the 644/161 train/val membership.
npm run distill:split

# Generate exactly four teacher candidates per query. The command resumes from
# candidates.jsonl.partial when interrupted. Interactive terminals show
# candidate-level progress, elapsed time, ETA, and generation errors.
npm run distill:generate -- \
  --experiment distill-pilot-v1 \
  --model /absolute/path/to/teacher.gguf

# Or use the official OpenAI Responses API after setting OPENAI_API_KEY in the
# environment. The key is never written to candidates or the run manifest.
npm run distill:generate -- \
  --experiment distill-openai-pilot-v1 \
  --provider openai \
  --model <openai-model-id> \
  --reasoning-effort low

# Use that same experiment id in the validate, score, and materialize run paths below.

# From finetune/: apply the single-source Contract v1.1 validator.
uv run python -m dataset.scifact_distill validate \
  --run-dir data/distillation/scifact-v1/runs/distill-pilot-v1 \
  --tokenizer-revision <40-character-hugging-face-commit> \
  --local-files-only

# From the repository root: score raw + valid candidates through the frozen
# QMD retrieval profile, then apply the deterministic winner rule.
npm run distill:score -- \
  --run-dir finetune/data/distillation/scifact-v1/runs/distill-pilot-v1

# From finetune/: write completion-only SFT train.jsonl and val.jsonl.
uv run python -m dataset.scifact_distill materialize \
  --run-dir data/distillation/scifact-v1/runs/distill-pilot-v1 \
  --tokenizer-revision <40-character-hugging-face-commit> \
  --local-files-only
```

Responses-API teacher prompts are environment-configurable. For the checked-in
GPT V2 RAG prompt, copy `.env.distill.openai-v2.example` to the ignored local
file `.env.distill.local`, then add `DISTILL_API_KEY` and the compatible
`DISTILL_RESPONSES_ENDPOINT`. The older inline `.env.distill.v1.example` and
`.env.distill.v2.example` remain available for reproducing earlier profiles.
The generator reads:

- `DISTILL_PROMPT_VERSION`: semantic version/id recorded in the run manifest;
- `DISTILL_SYSTEM_PROMPT` / `DISTILL_USER_PROMPT_TEMPLATE`: inline Prompt text;
- `DISTILL_SYSTEM_PROMPT_FILE` / `DISTILL_USER_PROMPT_TEMPLATE_FILE`: preferred
  file-based Prompt configuration for long prompts and JSON examples;
- the user template must contain the literal
  placeholder `{{query}}` (all occurrences are replaced);
- `DISTILL_MAX_OUTPUT_TOKENS`: default output budget; CLI
  `--max-output-tokens` takes precedence.

The Prompt version and exactly one complete inline or file pair must be set
together. Inline and file configuration cannot be mixed. This prevents a v1
Prompt from being mislabeled with a v2 version and avoids `.env` quote
truncation. The output-budget variable is independent.

DeepSeek uses the OpenAI-compatible Chat Completions protocol, not the
Responses API. Copy `.env.distill.deepseek-v2.example` to `.env.distill.local`
and set the local key. V2 is the general-purpose, semantics-preserving teacher
Prompt; the V1 example and Prompt files remain available for reproducible A/B
comparison. The provider resolves
`DISTILL_API_BASE_URL=https://api.deepseek.com` to
`https://api.deepseek.com/chat/completions`, requests JSON Output, and records
the base URL, resolved endpoint, thinking mode, model, and Prompt hash in the
manifest. Distillation defaults to `DISTILL_THINKING_MODE=disabled`; when
enabled, DeepSeek accepts only `--reasoning-effort high|max`.

SciFact V3 is a separate experiment profile; it does not replace the
general-purpose V2 Prompt. Copy `.env.distill.deepseek-v3.example` for a
teacher that treats every SciFact input as an unverified scientific claim and
requests exactly 3 lex + 3 vec + 1 HyDE. Its post-generation gate is opt-in and
**observational only** — it records but never rejects:

```bash
npm run distill:generate -- \
  --experiment distill-deepseek-v4-flash-v3-pilot-100 \
  --provider deepseek \
  --model deepseek-v4-flash \
  --thinking-mode disabled \
  --max-queries 100

# From finetune/:
uv run python -m dataset.scifact_distill validate \
  --run-dir data/distillation/scifact-v1/runs/distill-deepseek-v4-flash-v3-pilot-100 \
  --tokenizer-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-files-only \
  --semantic-gate scifact-observational-v3
```

The gate is deliberately recorded separately from Contract v1.1. Older runs
remain reproducible and are not retroactively filtered.

### Why the semantic gate stopped rejecting anything

Gate `scifact-observational-v3` emits `valid: true` unconditionally. Every
blocking check it used to apply was a presence test over a fixed word list
standing in for a semantic property, and each was measured on
`distill-deepseek-v4-flash-v3-pilot-100` to reject faithful expansions often
enough that the benchmark scored the gate instead of the teacher:

| Removed check | Fired | False | What it actually caught |
| --- | --- | --- | --- |
| `unsupported_clinical_advice` | 17 | 17 (100%) | All 17 were `\bshould\b` inside the model's own description of the retrieval task ("evidence should assess …"), never clinical advice. The other seven patterns never fired. |
| `established_fact_risk` | 3 | 3 | All three were plain questions ("Do studies link …?"), the purest neutral framing; the hedge list has no interrogative form and no `studies`. |
| `unsupported_mechanism` | 77 | ~60% | The claim already carried the mechanism through wording the list cannot see, e.g. "R2D2 stops miRNA production **by increasing** Dcr2 selectivity". Driven by `\bvia\b` (31), `\bmechanism\b` (20), `\bdue to\b` (20). |
| `unsupported_comparison` | 37 | ~60% | Same failure: "Allogeneic MCS is **not as effective as** autologous" is a comparison the list misses, so the faithful "allogeneic versus autologous" was rejected. `\bthan\b` also matches "other than" and "rather than". |
| `fixed_profile_count` | 2 | — | Bound scoring eligibility to one prompt's 3/3/1 shape while Contract v1.1 already validates structure, and the V2 prompt legitimately asks for 1-3. |

They are removed rather than downgraded to advisories: a word list that cannot
see the concept it is checking produces noise, not a weaker signal.

`negation_lost` survives as the sole advisory, recorded under
`semantic_gate.advisories` and never affecting `valid`. It is kept because it
marks the one failure mode a human audit independently rated critical (qid
1004, where "no known association" became a positive-association query). It is
not trustworthy on its own: 63 of 171 flags (36.8%) fired on expansions that
preserved the negation with `absent`, `absence`, `devoid`, `deficient`, or
`unrelated to`. Nor does it protect retrieval here — across seven pilot runs,
candidates that dropped the negation scored MRR@10 0.9051 against 0.8562 for
those that kept it, and both trailed the unexpanded raw baseline of 0.9375 on
the same queries (permutation p = 0.093, suggestive rather than conclusive).
SciFact's gold abstract for a refuted claim states the positive proposition, so
dropping the negation aligns the query with the target document.

Negation preservation is therefore unmeasured by the winner rule rather than
enforced by it. That is a deliberate trade: general-purpose retrieval does need
it — "Version 3.2 does not support offline sync" must not retrieve documents
saying it does — but SciFact cannot supply that signal, so the guarantee has to
come from a dedicated adversarial set rather than from this gate.

Before a full V2 run, reuse the frozen eight-query V1 smoke set so Prompt
quality can be compared on identical inputs:

```bash
npm run distill:generate -- \
  --experiment distill-deepseek-v4-flash-smoke-v2 \
  --provider deepseek \
  --qids-file finetune/data/distillation/scifact-v1/deepseek-v1-smoke-qids.txt
```

This creates a new immutable run directory and makes four teacher calls per
query. Validate and inspect the smoke artifact before starting a larger pilot.

New Responses-API runs default to 4096 output tokens. When resuming a historical
run whose manifest lacks `max_output_tokens`, the generator preserves the old
1200-token interpretation. The manifest records the resolved prompts, prompt
hash, source, version, and output budget; resume refuses mismatched settings.

For Smoke and A/B runs, pass an arbitrary frozen query set as one qid per line:

```bash
npm run distill:generate -- \
  --experiment distill-packy-terra-headroom-v2 \
  --provider openai-compatible \
  --qids-file /absolute/path/to/headroom-qids.txt
```

`--qids-file` and `--max-queries` are mutually exclusive. The qid-file hash is
recorded in the manifest so v1/v2 cannot silently evaluate different queries.

Candidate selection compares `(Recall@30, MRR@10, nDCG@10)`
lexicographically. Exact metric ties choose the smallest zero-based candidate
index; a candidate must strictly beat the raw query or the query is recorded as
`no_winner` and omitted from SFT.

For a fixed 100-query gateway pilot, generate a separate run with
`--max-queries 100`, validate it with tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, and score it with `--min-winners`.
The score command writes `pilot_gate` into the manifest and returns exit code 2
when the gate fails. If it passes, seed the full run from the pilot so the
first 100 teacher outputs are reused rather than regenerated:

**Pick the threshold against headroom, not against the query count.** A winner
must beat raw retrieval, and canonical metrics are capped at 1, so no candidate
can outrank a query whose raw metrics are already `(1, 1, 1)`. Winners can only
come from queries with headroom — 24 of the 100 in this pilot. The historical
`--min-winners 30` was therefore unsatisfiable by any teacher, and the three
`pilot_gate.passed: false` results it produced said nothing about teacher
quality; `distill-packy-terra-pilot-v1` failed it while converting 12 of 24
(50%). `score.ts` now records `headroom_queries` in `pilot_gate` and raises an
explicit error when the threshold exceeds it, after persisting the manifest so
the scoring work is not lost.

```bash
npm run distill:generate -- \
  --experiment distill-packy-terra-pilot-v1 \
  --provider openai-compatible \
  --reasoning-effort low \
  --max-queries 100

# Retry only candidate slots recorded as generation_error. Successful teacher
# outputs are preserved. Re-run validation after this command.
npm run distill:generate -- \
  --experiment distill-packy-terra-pilot-v1 \
  --provider openai-compatible \
  --reasoning-effort low \
  --max-queries 100 \
  --retry-generation-errors

# Preserve the original run: derive a clean run and retry only failed slots.
# Successful candidates are copied while old score/selection data is invalidated.
npm run distill:generate -- \
  --experiment distill-packy-terra-pilot-v1-clean \
  --provider openai-compatible \
  --reasoning-effort low \
  --max-output-tokens 4096 \
  --max-queries 100 \
  --seed-run finetune/data/distillation/scifact-v1/runs/distill-packy-terra-pilot-v1 \
  --retry-generation-errors

# From finetune/:
uv run python -m dataset.scifact_distill validate \
  --run-dir data/distillation/scifact-v1/runs/distill-packy-terra-pilot-v1 \
  --tokenizer-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-files-only

# From the repository root:
# Interactive terminals show query progress, elapsed time, ETA, and live winners.
npm run distill:score -- \
  --run-dir finetune/data/distillation/scifact-v1/runs/distill-packy-terra-pilot-v1 \
  --min-winners 12

# Reuse the scored raw/all metrics and run the remaining controlled ablations.
# The command resumes from ablation.jsonl.partial when interrupted.
npm run distill:ablate -- \
  --run-dir finetune/data/distillation/scifact-v1/runs/distill-packy-terra-pilot-v1

# Summarize reliability, Contract, headroom conversion, candidate comparisons,
# retrieval metrics, and optional winner semantic admission. This also writes
# distill-report.json and headroom-qids.txt into the run directory.
npm run distill:report -- \
  --run-dir finetune/data/distillation/scifact-v1/runs/distill-packy-terra-pilot-v1 \
  --semantic-audit /absolute/path/to/semantic-audit.json

# Add --baseline-run-dir and --baseline-semantic-audit for the v2 release gate.
# The audit schema is shown in benchmarks/distill/semantic-audit.example.json.
# Run only after pilot_gate.passed is true.
npm run distill:generate -- \
  --experiment distill-packy-terra-v1 \
  --provider openai-compatible \
  --reasoning-effort low \
  --seed-run finetune/data/distillation/scifact-v1/runs/distill-packy-terra-pilot-v1
```

### What the release gate compares

`--baseline-run-dir` turns the report into an A/B against a prior run;
`report.ts` refuses the comparison unless both runs carry the same ordered
headroom qids. The checks are:

| Check | Meaning |
| --- | --- |
| `generation_error_lte_2_percent` | Teacher/API reliability, not expansion quality — a truncation storm fails here. |
| `headroom_coverage_100_percent` | Every headroom query produced at least one non-error candidate. |
| `contract_pass_100_percent` | Contract v1.1 clean across all candidates. |
| `winner_delta_at_least_4` | At least four more headroom queries converted than the baseline. |
| `candidate_worse_rate_within_3_points` | Did not trade wins for a wider tail of regressions. |
| `mrr_not_lower` / `ndcg_not_lower` | Headroom macro did not regress. |
| `critical_winners_zero` | Applied only when the candidate run has a complete semantic audit. |
| `semantic_admitted_winners_increased` | Applied only when both runs have one. |

Two earlier checks are gone. `conversion_delta_at_least_15_points` was provably
redundant: both runs share the same headroom `H`, so
`conversionDelta === winnerDelta / H`, and at `H = 24` the two thresholds agree
on every possible `winnerDelta` — it restated `winner_delta_at_least_4` and
cost it a second vote in the all-must-pass verdict. The two semantic checks
used to read `candidate.semantic?.…`, so a missing audit made them `false` and
failed every unaudited run for a reason unrelated to its quality; they are now
skipped unless the audits they read exist, and `semantic_checks_applied` in the
output records which way it went.

The thresholds themselves are point estimates over 24 headroom queries and
carry no tolerance band, while the underlying noise is large — a ±1 winner
difference is McNemar p = 1.000 and the conversion 95% CI spans roughly
[0.05, 0.37]. Read a single failed check as a prompt to look, not as a verdict.

## Architecture Notes

The production training approach is currently **SFT-only**:

1. **SFT** establishes format compliance and basic query understanding. It uses
   a large LoRA (rank 16, all projection layers) because it needs to learn a
   new output format from scratch.

2. **GRPO** exists as an optional experimental path under `experiments/grpo/`
   and is not in the production training pipeline.

The reward function is entirely rule-based (no LLM judge) which makes it fast,
deterministic, and suitable as an RL signal. See `SCORING.md` for the full rubric.

## Training Results (Qwen3-1.7B, v2)

### SFT

| Metric | Value |
|--------|-------|
| Final train loss | 0.472 |
| Final eval loss | 0.304 |
| Token accuracy (train) | 97.4% |
| Token accuracy (eval) | 93.8% |
| Epochs | 5 |
| Hardware | A10G (24 GB VRAM) |

### Evaluation Scores

| Model | Average Score | Excellent (30) |
|-------|--------------|-----------------|
| SFT | 92.0% | 30/30 |

> GRPO scores are not tracked in this branch; see `experiments/grpo/` for historical
> experimental results.

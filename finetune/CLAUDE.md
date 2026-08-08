# QMD Query Expansion Fine-Tuning

## Overview

Train Qwen3-1.7B to expand search queries into structured `hyde:/lex:/vec:` output for QMD's hybrid retrieval pipeline.

## Output Format

```
hyde: A hypothetical document passage that would answer the query.
lex: keyword1
lex: keyword2
vec: semantic query reformulation
vec: another semantic variation
```

- `hyde:` always comes FIRST (one line max)
- `lex:` lines for BM25 keyword search (1-3 lines, short keywords)
- `vec:` lines for vector similarity search (1-3 lines, natural language)

## Training Data Format

**There is exactly one JSONL format.** Every file in `data/*.jsonl` must match the strict Pydantic schema in `dataset/schema.py`:

```json
{"query": "auth config", "output": [["hyde", "..."], ["lex", "..."], ["vec", "..."]]}
```

- `query`: non-empty string
- `output`: list of `[type, text]` pairs where type is `"lex"`, `"vec"`, or `"hyde"`
- Extra metadata fields (`category`, `intent`, `is_short`) are allowed but ignored

The validation rules are owned by **`dataset/contract.py`** (Contract v1) — the single source of truth, a pure function with no file I/O or tokenizer dependency. `dataset/schema.py` provides the typed Pydantic view (`TrainingExample`), the fail-fast `load_examples()` loader, and the output renderers; `load_examples()` delegates every record to `contract.validate_training_target` so the training path and the offline audit never diverge. No format alternatives, no legacy fallbacks.

**Only-mode quarantine:** records whose query carries an `/only:lex|vec|hyde` directive (the `data/*_only*.jsonl` files) are valid but **quarantined by Contract v1** out of the default training target. `load_examples()` loads them with `quarantined=True` rather than rejecting them; `prepare_data` decides whether to train on them (see Stage 0).

**All `.jsonl` files in `data/` are concatenated and deduplicated for training runs.** The prepared train/val files in `data/train/` are ephemeral build artifacts. Prepared records contain `prompt` and `completion`: their concatenation is the exact rendered Qwen sequence, while SFT masks every prompt/template token from loss.

## HuggingFace Repositories

| Repository | Purpose |
|------------|---------|
| `tobil/qmd-query-expansion-1.7B` | Final merged model (SFT baseline) |
| `tobil/qmd-query-expansion-1.7B-gguf` | GGUF quantized versions for deployment |
| `tobil/qmd-query-expansion-1.7B-sft` | SFT adapter checkpoint (intermediate) |
| `tobil/qmd-query-expansion-train` | Prepared training dataset |
| `tobil/qmd-query-expansion-1.7B-grpo` | Experimental GRPO adapter (optional) |

**Rules:**
- No versioned repos (`-v1`, `-v2`, `-v4`, etc.) - update in place
- Only push when eval scores improve over current deployed model
- Always include eval results in model card when pushing

## Dataset Tools

| Script | Purpose |
|--------|---------|
| `dataset/contract.py` | Contract v1 validation rules — single source of truth (pure, no I/O) |
| `dataset/schema.py` | Pydantic `TrainingExample` model + `load_examples()` (delegates to contract) + renderers |
| `dataset/prepare_data.py` | Load via schema, apply Qwen3 chat template, split prompt/completion, dedup, split; `--only-mode` switch |
| `dataset/completion.py` | Preserve rendered tokens and build the completion-only loss mask |
| `dataset/validate_schema.py` | Lightweight Contract v1 check (tokenizer-free) |
| `dataset/validate_contract.py` | Full Contract v1 audit (pinned tokenizer, writes report dir) |
| `dataset/build_conflict_ledger.py` | Deterministic conflict ledger for human review |
| `dataset/score_data.py` | Score all examples using reward.py |
| `dataset/analyze_data.py` | Analyze distribution and quality |

## Training Pipeline

Always use **Qwen3-1.7B** as the base model unless explicitly stated otherwise.

### Stage 0: Prepare Data

```bash
uv run python -m dataset.prepare_data
# Creates: data/train/train.jsonl, data/train/val.jsonl (ephemeral)
# Default excludes Contract v1 only-mode (quarantined) records.
# To train on them as before: --only-mode include
```

### Stage 1: SFT

SFT uses `completion_only_loss=True`: the model sees the full chat prompt, but
only the assistant's `hyde:/lex:/vec:` completion receives loss labels.

```bash
# Local (requires CUDA)
uv run train.py sft --config configs/sft.yaml

# Cloud (HuggingFace Jobs)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py
```

### Stage 2: (Experimental) GRPO

```bash
# Experimental script
cd finetune && HF_TOKEN=${HF_TOKEN} uv run python experiments/grpo/grpo.py
```

### HuggingFace Jobs

```bash
hf jobs ps                    # List running jobs
hf jobs logs <job-id>         # Stream logs
hf jobs inspect <job-id>      # Check status
hf jobs cancel <job-id>       # Cancel a job
```

### Evaluation

```bash
uv run eval.py ./outputs/sft
uv run eval.py tobil/qmd-query-expansion-1.7B
uv run eval.py ./outputs/sft -o eval_results.json
```

## Quality Scoring

`reward.py` is the single source of truth for scoring:

```bash
uv run reward.py   # Self-test
```

See `SCORING.md` for the full rubric.

## Experiments

Experimental training configurations live in `experiments/`:

```
experiments/
├── lfm2/          # LiquidAI LFM2-1.2B (hybrid architecture, faster inference)
│   ├── sft_lfm2.yaml
│   └── sft_lfm2.py
├── grpo/          # Experimental GRPO recipe and config
│   ├── grpo.py
│   └── grpo.yaml
└── gepa/          # DSPy-based prompt optimization (GEPA)
    ├── dspy_gepa.py
    └── ...
```

These are not part of the main training pipeline.

## Key Files

```
finetune/
├── reward.py          # Scoring function (single source of truth)
├── train.py           # SFT training entrypoint
├── eval.py            # Generate and score expansions
├── convert_gguf.py    # GGUF conversion
├── SCORING.md         # Detailed scoring rubric
├── CLAUDE.md          # This file
├── Justfile           # Common commands
├── data/              # All training JSONL files (strict schema)
├── dataset/           # Schema + data tools (Pydantic-based)
├── jobs/              # Self-contained HuggingFace Jobs scripts
├── configs/           # Training configs (sft.yaml)
├── evals/             # Test queries
├── experiments/       # Experimental configs (LFM2, GEPA, GRPO)
└── outputs/           # Local training outputs (gitignored)
```
